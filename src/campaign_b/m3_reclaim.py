"""Safe reclaim of Campaign B screening M3 checkpoint bloat.

Shared by ``scripts/persist_reclaim_m3.py`` (CLI dry-run / execute) and the
notebook 97 drain path (``run_pipeline_to_m6`` / ``run_post_m2_pipeline``).

Fail-closed strip criterion (default)
-------------------------------------
``M3_COMPLETE`` + downstream ``M4_COMPLETE`` (or M5/M6 progress) + not
CERTIFIED / ONE_STEP_CERTIFIED lineage → delete ``runs/M3-*/checkpoints/``
(keep reports/acceptance/config/artifacts/work_items). After strip, that M3
is no longer a valid M4 parent resume source.

Never deletes selected package directories or reports.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


CERTIFIED_STATUSES = frozenset({'CERTIFIED', 'ONE_STEP_CERTIFIED'})
PROTECTED_NAME_MARKERS = ('CERTIFIED', 'm6_certified_catalog')

KEEP_RELATIVE_PREFIXES = (
    'reports/',
    'run_config.json',
    'run_manifest.json',
    'test_report.json',
    'artifacts/',
    'work_items/',
    'cache/',
    'logs/',
)

# Top-level pre_m6 / package statuses that imply M4 consumed M3 (safe to strip).
_DOWNSTREAM_READY_STATUSES = frozenset({
    'PRE_M6_READY',
    'M4_COMPLETE',
    'M6_COMPLETE',
})


def fmt_bytes(n: int) -> str:
    if n < 1024:
        return f'{n} B'
    for unit, scale in (('KiB', 1024), ('MiB', 1024**2), ('GiB', 1024**3), ('TiB', 1024**4)):
        if n < scale * 1024 or unit == 'TiB':
            return f'{n / scale:.3f} {unit}'
    return f'{n} B'


def dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    payload = _load_json(path)
    return payload if isinstance(payload, dict) else None


def discover_selected_packages(persistent_root: Path) -> list[Path]:
    root = Path(persistent_root) / 'campaign_b'
    if not root.is_dir():
        return []
    found: list[Path] = []
    for campaign in sorted(root.iterdir()):
        if not campaign.is_dir() or campaign.name.startswith('_'):
            continue
        selected = campaign / 'selected'
        if not selected.is_dir():
            continue
        for package in sorted(selected.iterdir()):
            if package.is_dir() and (package / 'candidate_manifest.json').is_file():
                found.append(package)
    return found


def m3_complete_on_disk(run_root: Path) -> bool:
    report = _load_json_dict(run_root / 'reports' / 'M3_report.json')
    acceptance = _load_json_dict(run_root / 'reports' / 'M3_acceptance.json')
    if report is None or acceptance is None:
        return False
    return (
        report.get('phase') == 'M3_COMPLETE'
        and acceptance.get('status') == 'PASS'
    )


def m4_complete_on_disk(persistent_root: Path, m4_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m4_run_id
    report = _load_json_dict(root / 'reports' / 'M4_report.json')
    acceptance = _load_json_dict(root / 'reports' / 'M4_acceptance.json')
    if report is None or acceptance is None:
        return False
    return report.get('phase') == 'M4_COMPLETE' and acceptance.get('status') == 'PASS'


def m5_progress_on_disk(persistent_root: Path, m5_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m5_run_id
    if (root / 'reports' / 'M5_obligation_report.json').is_file():
        return True
    if (root / 'reports' / 'M5_report.json').is_file():
        return True
    ckpt = root / 'checkpoints'
    if ckpt.is_dir():
        try:
            return any(p.is_dir() and p.name.startswith('ckpt_') for p in ckpt.iterdir())
        except OSError:
            return False
    return False


def m6_status_on_disk(persistent_root: Path, m6_run_id: str) -> tuple[bool, str | None]:
    """Return (exists_or_progress, certification_status_or_None)."""
    root = Path(persistent_root) / 'runs' / m6_run_id
    if not root.is_dir():
        return False, None
    acceptance = _load_json_dict(root / 'reports' / 'M6_acceptance.json')
    status: str | None = None
    if isinstance(acceptance, dict):
        raw = acceptance.get('certification_status')
        if isinstance(raw, str):
            status = raw
    report = _load_json_dict(root / 'reports' / 'M6_report.json')
    progress = bool(acceptance or report or (root / 'checkpoints').is_dir())
    return progress, status


def package_archived(package: Path) -> bool:
    for name in ('ADVANCE.json', 'package_audit.json', 'SCREENING_STATUS.json'):
        doc = _load_json_dict(package / name)
        if not isinstance(doc, dict):
            continue
        for key in ('archived', 'abandoned', 'package_state', 'status', 'state'):
            val = doc.get(key)
            if val is True:
                return True
            if isinstance(val, str) and val.upper() in {
                'ARCHIVED', 'ABANDONED', 'DROPPED', 'SUPERSEDED',
            }:
                return True
    return False


@dataclass
class PackageRef:
    package_rel: str
    m3_run_id: str | None
    gpu_status: str | None
    child_m4: str | None = None
    child_m5: str | None = None
    child_m6: str | None = None
    archived: bool = False
    m4_complete: bool = False
    m5_progress: bool = False
    m6_progress: bool = False
    m6_cert_status: str | None = None


@dataclass
class M3Classification:
    run_id: str
    run_rel: str
    size_bytes: int
    checkpoints_bytes: int
    tensors_bytes: int
    complete: bool
    referenced: bool
    package_refs: list[str] = field(default_factory=list)
    has_downstream_safe: bool = False
    certified_lineage: bool = False
    archived_only_refs: bool = False
    already_stripped: bool = False
    reclaimable_strip_bytes: int = 0
    reclaimable_keep_latest_bytes: int = 0
    skip_reasons: list[str] = field(default_factory=list)


def committed_ckpt_dirs(checkpoints: Path) -> list[Path]:
    if not checkpoints.is_dir():
        return []
    found: list[Path] = []
    try:
        children = list(checkpoints.iterdir())
    except OSError:
        return []
    for child in children:
        if not child.is_dir() or not child.name.startswith('ckpt_'):
            continue
        if (child / 'COMMITTED').is_file():
            found.append(child)
    found.sort(key=lambda p: p.name)
    return found


def tensors_bytes(checkpoints: Path) -> int:
    total = 0
    if not checkpoints.is_dir():
        return 0
    try:
        children = list(checkpoints.iterdir())
    except OSError:
        return 0
    for child in children:
        if not child.is_dir() or not child.name.startswith('ckpt_'):
            continue
        tensors = child / 'tensors'
        if tensors.is_dir():
            total += dir_size(tensors)
    return total


def keep_latest_reclaimable(checkpoints: Path) -> tuple[int, list[Path]]:
    """Bytes and dirs that would be removed if keeping only newest COMMITTED ckpt."""
    committed = committed_ckpt_dirs(checkpoints)
    if len(committed) <= 1:
        extra: list[Path] = []
        bytes_extra = 0
        if checkpoints.is_dir():
            committed_names = {p.name for p in committed}
            try:
                for child in checkpoints.iterdir():
                    if (
                        child.is_dir()
                        and child.name.startswith('ckpt_')
                        and child.name not in committed_names
                    ):
                        extra.append(child)
                        bytes_extra += dir_size(child)
            except OSError:
                pass
        return bytes_extra, extra
    remove = list(committed[:-1])
    if checkpoints.is_dir():
        committed_names = {p.name for p in committed}
        try:
            for child in checkpoints.iterdir():
                if (
                    child.is_dir()
                    and child.name.startswith('ckpt_')
                    and child.name not in committed_names
                ):
                    remove.append(child)
        except OSError:
            pass
    return sum(dir_size(p) for p in remove), remove


def build_package_index(persistent_root: Path) -> dict[str, list[PackageRef]]:
    """Map m3_run_id -> package references."""
    index: dict[str, list[PackageRef]] = {}
    for package in discover_selected_packages(persistent_root):
        try:
            rel = str(package.relative_to(persistent_root))
        except ValueError:
            rel = str(package)
        child = _load_json_dict(package / 'child_run_ids.json') or {}
        gpu = _load_json_dict(package / 'GPU_M3.json') or {}
        m3_from_child = child.get('M3') if isinstance(child.get('M3'), str) else None
        m3_from_gpu = gpu.get('m3_run_id') if isinstance(gpu.get('m3_run_id'), str) else None
        m3_id = m3_from_child or m3_from_gpu
        m4_id = child.get('M4') if isinstance(child.get('M4'), str) else None
        m5_id = child.get('M5') if isinstance(child.get('M5'), str) else None
        m6_id = child.get('M6') if isinstance(child.get('M6'), str) else None
        m4_ok = bool(m4_id and m4_complete_on_disk(persistent_root, m4_id))
        m5_ok = bool(m5_id and m5_progress_on_disk(persistent_root, m5_id))
        m6_ok, m6_cert = (False, None)
        if m6_id:
            m6_ok, m6_cert = m6_status_on_disk(persistent_root, m6_id)
        ref = PackageRef(
            package_rel=rel,
            m3_run_id=m3_id if isinstance(m3_id, str) else None,
            gpu_status=str(gpu['status']) if isinstance(gpu.get('status'), str) else None,
            child_m4=m4_id,
            child_m5=m5_id,
            child_m6=m6_id,
            archived=package_archived(package),
            m4_complete=m4_ok,
            m5_progress=m5_ok,
            m6_progress=m6_ok,
            m6_cert_status=m6_cert,
        )
        if isinstance(m3_id, str) and m3_id.startswith('M3-'):
            index.setdefault(m3_id, []).append(ref)
    return index


def classify_m3_runs(
    persistent_root: Path,
    *,
    include_certified_lineage: bool = False,
    only_run_ids: set[str] | frozenset[str] | None = None,
) -> list[M3Classification]:
    runs = Path(persistent_root) / 'runs'
    if not runs.is_dir():
        return []
    pkg_index = build_package_index(persistent_root)
    rows: list[M3Classification] = []
    try:
        m3_dirs = sorted(
            p for p in runs.iterdir()
            if p.is_dir() and p.name.startswith('M3-') and not p.is_symlink()
        )
    except OSError as exc:
        print(f'ERROR: cannot list runs/: {exc}', file=sys.stderr)
        return []

    for run_root in m3_dirs:
        run_id = run_root.name
        if only_run_ids is not None and run_id not in only_run_ids:
            continue
        if any(marker in run_id for marker in PROTECTED_NAME_MARKERS):
            continue
        checkpoints = run_root / 'checkpoints'
        size = dir_size(run_root)
        ckpt_bytes = dir_size(checkpoints) if checkpoints.is_dir() else 0
        tensor_bytes = tensors_bytes(checkpoints)
        complete = m3_complete_on_disk(run_root)
        refs = pkg_index.get(run_id, [])
        referenced = bool(refs)
        package_rels = [r.package_rel for r in refs]
        has_downstream = any(
            r.m4_complete or r.m5_progress or r.m6_progress for r in refs
        )
        certified_lineage = any(
            r.m6_cert_status in CERTIFIED_STATUSES for r in refs
        )
        archived_only = bool(refs) and all(r.archived for r in refs)
        already_stripped = (checkpoints / 'STRIPPED_FOR_RECLAIM.json').is_file()
        keep_latest_bytes, _ = keep_latest_reclaimable(checkpoints)

        skip: list[str] = []
        reclaim_strip = 0
        if already_stripped:
            skip.append('already_stripped')
        elif certified_lineage and not include_certified_lineage:
            skip.append('certified_m6_lineage')
        elif not complete:
            skip.append('m3_incomplete')
        elif not has_downstream:
            skip.append('no_downstream_m4_complete_or_later')
        elif ckpt_bytes <= 0:
            skip.append('no_checkpoints')
        else:
            reclaim_strip = ckpt_bytes

        rows.append(M3Classification(
            run_id=run_id,
            run_rel=f'runs/{run_id}',
            size_bytes=size,
            checkpoints_bytes=ckpt_bytes,
            tensors_bytes=tensor_bytes,
            complete=complete,
            referenced=referenced,
            package_refs=package_rels,
            has_downstream_safe=has_downstream,
            certified_lineage=certified_lineage,
            archived_only_refs=archived_only,
            already_stripped=already_stripped,
            reclaimable_strip_bytes=reclaim_strip,
            reclaimable_keep_latest_bytes=keep_latest_bytes,
            skip_reasons=skip,
        ))
    return rows


def write_stripped_marker(checkpoints: Path, *, mode: str, bytes_removed: int) -> None:
    checkpoints.mkdir(parents=True, exist_ok=True)
    marker = {
        'status': 'STRIPPED_FOR_RECLAIM',
        'mode': mode,
        'bytes_removed_approx': bytes_removed,
        'note': (
            'M3 checkpoint tensors removed after downstream M4+ progress. '
            'This run is no longer a valid M4 parent resume source.'
        ),
    }
    path = checkpoints / 'STRIPPED_FOR_RECLAIM.json'
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def strip_checkpoints_for_run(
    persistent_root: Path,
    row: M3Classification,
    *,
    execute: bool,
) -> tuple[int, str]:
    """Return (bytes_affected, action_label)."""
    run_root = Path(persistent_root) / 'runs' / row.run_id
    checkpoints = run_root / 'checkpoints'
    if not checkpoints.is_dir():
        return 0, 'SKIP_NO_CHECKPOINTS'
    if (checkpoints / 'STRIPPED_FOR_RECLAIM.json').is_file():
        return 0, 'SKIP_ALREADY_STRIPPED'
    bytes_to_free = dir_size(checkpoints)
    if not execute:
        return bytes_to_free, 'WOULD_STRIP_CHECKPOINTS'
    try:
        for child in list(checkpoints.iterdir()):
            if child.is_symlink():
                print(f'  FAIL_CLOSED symlink in checkpoints: {child}', file=sys.stderr)
                return 0, 'FAIL_SYMLINK'
            if child.is_dir():
                shutil.rmtree(child)
            elif child.is_file():
                child.unlink()
    except OSError as exc:
        print(f'  FAILED strip {row.run_id}: {exc}', file=sys.stderr)
        return 0, 'FAIL_OS'
    write_stripped_marker(checkpoints, mode='strip-checkpoints', bytes_removed=bytes_to_free)
    return bytes_to_free, 'STRIPPED_CHECKPOINTS'


def keep_latest_for_run(
    persistent_root: Path,
    row: M3Classification,
    *,
    execute: bool,
) -> tuple[int, str]:
    run_root = Path(persistent_root) / 'runs' / row.run_id
    checkpoints = run_root / 'checkpoints'
    reclaim_bytes, remove = keep_latest_reclaimable(checkpoints)
    if reclaim_bytes <= 0 or not remove:
        return 0, 'SKIP_NOTHING_TO_TRIM'
    if not execute:
        return reclaim_bytes, 'WOULD_KEEP_LATEST'
    freed = 0
    for path in remove:
        if path.is_symlink():
            print(f'  FAIL_CLOSED symlink: {path}', file=sys.stderr)
            continue
        size = dir_size(path)
        try:
            shutil.rmtree(path)
            freed += size
        except OSError as exc:
            print(f'  FAILED remove {path}: {exc}', file=sys.stderr)
    return freed, 'KEPT_LATEST_REMOVED_OLDER'


def delete_run(
    persistent_root: Path,
    row: M3Classification,
    *,
    execute: bool,
    allow_referenced_archived: bool,
) -> tuple[int, str]:
    if row.certified_lineage:
        return 0, 'SKIP_CERTIFIED_LINEAGE'
    if row.referenced and not (allow_referenced_archived and row.archived_only_refs):
        return 0, 'SKIP_REFERENCED'
    run_root = Path(persistent_root) / 'runs' / row.run_id
    if not run_root.is_dir():
        return 0, 'SKIP_MISSING'
    size = dir_size(run_root)
    if not execute:
        return size, 'WOULD_DELETE_RUN'
    try:
        shutil.rmtree(run_root)
    except OSError as exc:
        print(f'  FAILED delete {row.run_id}: {exc}', file=sys.stderr)
        return 0, 'FAIL_OS'
    return size, 'DELETED_RUN'


def clear_abandoned_package_pointers(
    persistent_root: Path,
    m3_run_id: str,
    *,
    execute: bool,
) -> int:
    """Clear GPU_M3 / child M3 pointer for archived packages pointing at deleted run."""
    cleared = 0
    for package in discover_selected_packages(persistent_root):
        if not package_archived(package):
            continue
        child = _load_json_dict(package / 'child_run_ids.json')
        gpu = _load_json_dict(package / 'GPU_M3.json')
        touched = False
        if isinstance(child, dict) and child.get('M3') == m3_run_id:
            if execute:
                child = dict(child)
                child['M3'] = None
                child['m3_cleared_by_reclaim'] = True
                (package / 'child_run_ids.json').write_text(
                    json.dumps(child, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8',
                )
            touched = True
        if isinstance(gpu, dict) and gpu.get('m3_run_id') == m3_run_id:
            if execute:
                gpu = dict(gpu)
                gpu['m3_run_id'] = None
                gpu['status'] = 'M3_POINTER_CLEARED_RECLAIM'
                gpu['cleared_by_reclaim'] = True
                (package / 'GPU_M3.json').write_text(
                    json.dumps(gpu, indent=2, sort_keys=True) + '\n',
                    encoding='utf-8',
                )
            touched = True
        if touched:
            cleared += 1
    return cleared


def m3_run_ids_from_stage_results(
    persistent_root: Path,
    *,
    pre_m6_summary: dict[str, Any] | None = None,
    m6_summary: dict[str, Any] | None = None,
) -> set[str]:
    """Collect M3 run ids implied by this round's pre_m6 / m6 results (incremental)."""
    found: set[str] = set()
    persistent_root = Path(persistent_root)

    def _add_from_package(package_path: object) -> None:
        if not isinstance(package_path, str) or not package_path:
            return
        pkg = Path(package_path)
        child = _load_json_dict(pkg / 'child_run_ids.json') or {}
        m3 = child.get('M3')
        if isinstance(m3, str) and m3.startswith('M3-'):
            found.add(m3)
        gpu = _load_json_dict(pkg / 'GPU_M3.json') or {}
        m3g = gpu.get('m3_run_id')
        if isinstance(m3g, str) and m3g.startswith('M3-'):
            found.add(m3g)

    for summary in (pre_m6_summary, m6_summary):
        if not isinstance(summary, dict):
            continue
        for row in summary.get('results') or []:
            if not isinstance(row, dict):
                continue
            status = row.get('status')
            if status in _DOWNSTREAM_READY_STATUSES or status == 'M6_COMPLETE':
                _add_from_package(row.get('package'))
            for session in row.get('sessions') or []:
                if not isinstance(session, dict):
                    continue
                m3 = session.get('m3_run_id')
                if isinstance(m3, str) and m3.startswith('M3-'):
                    if session.get('status') in _DOWNSTREAM_READY_STATUSES or (
                        status in _DOWNSTREAM_READY_STATUSES
                    ):
                        found.add(m3)
            m3_direct = row.get('m3_run_id')
            if (
                isinstance(m3_direct, str)
                and m3_direct.startswith('M3-')
                and status in _DOWNSTREAM_READY_STATUSES
            ):
                found.add(m3_direct)
    return found


@dataclass
class StripReclaimSummary:
    execute: bool
    scope: str  # 'incremental' | 'full_scan'
    candidates: int
    stripped: int
    skipped: int
    bytes_freed: int
    run_ids: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            'bytes_freed_human': fmt_bytes(self.bytes_freed),
        }


def strip_eligible_m3_checkpoints(
    persistent_root: Path,
    *,
    execute: bool = True,
    include_certified_lineage: bool = False,
    only_run_ids: set[str] | frozenset[str] | None = None,
) -> StripReclaimSummary:
    """Strip checkpoints for runs that pass fail-closed reclaim criteria.

    When ``only_run_ids`` is set, only those runs are classified (incremental).
    When None, full safe scan of all ``runs/M3-*``.
    """
    persistent_root = Path(persistent_root)
    rows = classify_m3_runs(
        persistent_root,
        include_certified_lineage=include_certified_lineage,
        only_run_ids=only_run_ids,
    )
    targets = [r for r in rows if r.reclaimable_strip_bytes > 0]
    scope = 'incremental' if only_run_ids is not None else 'full_scan'
    freed = 0
    stripped = 0
    actions: list[dict[str, Any]] = []
    stripped_ids: list[str] = []
    for row in targets:
        nbytes, label = strip_checkpoints_for_run(
            persistent_root, row, execute=execute,
        )
        actions.append({
            'run_id': row.run_id,
            'label': label,
            'bytes': nbytes,
        })
        if label in {'STRIPPED_CHECKPOINTS', 'WOULD_STRIP_CHECKPOINTS'}:
            freed += nbytes
            stripped += 1
            stripped_ids.append(row.run_id)
    return StripReclaimSummary(
        execute=execute,
        scope=scope,
        candidates=len(targets),
        stripped=stripped,
        skipped=len(rows) - len(targets),
        bytes_freed=freed,
        run_ids=stripped_ids,
        actions=actions,
    )


def auto_strip_after_pipeline_round(
    persistent_root: Path,
    *,
    pre_m6_summary: dict[str, Any] | None = None,
    m6_summary: dict[str, Any] | None = None,
    execute: bool = True,
) -> dict[str, Any]:
    """Incremental strip from round results; fall back to one full safe scan.

    Prefer run ids from this round's PRE_M6_READY / M6 results. If none are
    found, run a full fail-closed scan once (safe when the round made no
    strip-eligible progress, or when results omit package paths).
    """
    preferred = m3_run_ids_from_stage_results(
        persistent_root,
        pre_m6_summary=pre_m6_summary,
        m6_summary=m6_summary,
    )
    if preferred:
        summary = strip_eligible_m3_checkpoints(
            persistent_root,
            execute=execute,
            only_run_ids=preferred,
        )
        # If incremental found ids but none were strip-eligible yet (e.g. M4
        # not fully on disk), do not escalate — wait for later round.
        out = summary.as_dict()
        out['preferred_run_ids'] = sorted(preferred)
        return out

    summary = strip_eligible_m3_checkpoints(
        persistent_root,
        execute=execute,
        only_run_ids=None,
    )
    out = summary.as_dict()
    out['preferred_run_ids'] = []
    out['fallback_full_scan'] = True
    return out
