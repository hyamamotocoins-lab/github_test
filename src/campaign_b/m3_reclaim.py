"""Safe reclaim of Campaign B screening M3 checkpoint bloat.

Shared by ``scripts/persist_reclaim_m3.py`` (CLI dry-run / execute) and the
notebook 97 drain path (``run_pipeline_to_m6`` / ``run_post_m2_pipeline``).

Fail-closed strip criterion (default)
-------------------------------------
``M3_COMPLETE`` + downstream ``M4_COMPLETE`` (or M5/M6 progress) + not
CERTIFIED / ONE_STEP_CERTIFIED lineage → delete ``runs/M3-*/checkpoints/``
(keep reports/acceptance/config/artifacts/work_items). After strip, that M3
is no longer a valid M4 parent resume source.

``strip-tensors`` (weaker): same eligibility, but only deletes
``checkpoints/*/tensors/`` (and similar bulky tensor dirs); checkpoint
metadata / LATEST remain. Marker: ``STRIPPED_TENSORS_FOR_RECLAIM.json``.

``enforce_persist_m3_cap``: while ``runs/M3-*`` total size exceeds a GiB
cap, strip-checkpoints oldest eligible COMPLETE+downstream runs.

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

STRIPPED_CHECKPOINTS_MARKER = 'STRIPPED_FOR_RECLAIM.json'
STRIPPED_TENSORS_MARKER = 'STRIPPED_TENSORS_FOR_RECLAIM.json'
# Bulky tensor payload dirs under ``checkpoints/ckpt_*/``.
TENSOR_DIR_NAMES = frozenset({'tensors', 'tensor_cache', 'tensor_blobs'})

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

# Aggressive default: keep total screening M3 footprint small on Paperspace.
# Cap only strips COMPLETE+downstream-eligible runs (fail-closed).
DEFAULT_PERSIST_M3_CAP_GIB: float = 32.0


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
    already_stripped_tensors: bool = False
    reclaimable_strip_bytes: int = 0
    reclaimable_tensors_bytes: int = 0
    reclaimable_keep_latest_bytes: int = 0
    mtime: float = 0.0
    skip_reasons: list[str] = field(default_factory=list)
    skip_reasons_tensors: list[str] = field(default_factory=list)


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


def iter_tensor_dirs(checkpoints: Path) -> list[Path]:
    """Return bulky tensor payload directories under ``checkpoints/ckpt_*/``."""
    found: list[Path] = []
    if not checkpoints.is_dir():
        return found
    try:
        children = list(checkpoints.iterdir())
    except OSError:
        return found
    for child in children:
        if not child.is_dir() or not child.name.startswith('ckpt_'):
            continue
        for name in TENSOR_DIR_NAMES:
            tensors = child / name
            if tensors.is_dir():
                found.append(tensors)
    return found


def tensors_bytes(checkpoints: Path) -> int:
    return sum(dir_size(path) for path in iter_tensor_dirs(checkpoints))


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
        already_stripped = (checkpoints / STRIPPED_CHECKPOINTS_MARKER).is_file()
        already_stripped_tensors = (
            already_stripped
            or (checkpoints / STRIPPED_TENSORS_MARKER).is_file()
        )
        keep_latest_bytes, _ = keep_latest_reclaimable(checkpoints)
        try:
            mtime = run_root.stat().st_mtime
        except OSError:
            mtime = 0.0

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

        skip_tensors: list[str] = []
        reclaim_tensors = 0
        if already_stripped:
            skip_tensors.append('already_stripped')
        elif already_stripped_tensors:
            skip_tensors.append('already_stripped_tensors')
        elif certified_lineage and not include_certified_lineage:
            skip_tensors.append('certified_m6_lineage')
        elif not complete:
            skip_tensors.append('m3_incomplete')
        elif not has_downstream:
            skip_tensors.append('no_downstream_m4_complete_or_later')
        elif tensor_bytes <= 0:
            skip_tensors.append('no_tensors')
        else:
            reclaim_tensors = tensor_bytes

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
            already_stripped_tensors=already_stripped_tensors,
            reclaimable_strip_bytes=reclaim_strip,
            reclaimable_tensors_bytes=reclaim_tensors,
            reclaimable_keep_latest_bytes=keep_latest_bytes,
            mtime=mtime,
            skip_reasons=skip,
            skip_reasons_tensors=skip_tensors,
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
    path = checkpoints / STRIPPED_CHECKPOINTS_MARKER
    path.write_text(json.dumps(marker, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def write_stripped_tensors_marker(
    checkpoints: Path,
    *,
    mode: str,
    bytes_removed: int,
) -> None:
    checkpoints.mkdir(parents=True, exist_ok=True)
    marker = {
        'status': 'STRIPPED_TENSORS_FOR_RECLAIM',
        'mode': mode,
        'bytes_removed_approx': bytes_removed,
        'note': (
            'M3 checkpoint tensor payloads removed after downstream M4+ progress. '
            'Checkpoint metadata / LATEST kept; run is no longer a valid M4 parent '
            'resume source (tensors required for resume).'
        ),
    }
    path = checkpoints / STRIPPED_TENSORS_MARKER
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
    if (checkpoints / STRIPPED_CHECKPOINTS_MARKER).is_file():
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


def strip_tensors_for_run(
    persistent_root: Path,
    row: M3Classification,
    *,
    execute: bool,
) -> tuple[int, str]:
    """Delete only bulky tensor dirs; keep checkpoint metadata / LATEST.

    Same fail-closed eligibility as strip-checkpoints (caller must filter).
    """
    run_root = Path(persistent_root) / 'runs' / row.run_id
    checkpoints = run_root / 'checkpoints'
    if not checkpoints.is_dir():
        return 0, 'SKIP_NO_CHECKPOINTS'
    if (checkpoints / STRIPPED_CHECKPOINTS_MARKER).is_file():
        return 0, 'SKIP_ALREADY_STRIPPED'
    if (checkpoints / STRIPPED_TENSORS_MARKER).is_file():
        return 0, 'SKIP_ALREADY_STRIPPED_TENSORS'
    targets = iter_tensor_dirs(checkpoints)
    if not targets:
        return 0, 'SKIP_NO_TENSORS'
    bytes_to_free = sum(dir_size(path) for path in targets)
    if not execute:
        return bytes_to_free, 'WOULD_STRIP_TENSORS'
    freed = 0
    try:
        for path in targets:
            if path.is_symlink():
                print(f'  FAIL_CLOSED symlink tensor dir: {path}', file=sys.stderr)
                return 0, 'FAIL_SYMLINK'
            size = dir_size(path)
            shutil.rmtree(path)
            freed += size
    except OSError as exc:
        print(f'  FAILED strip-tensors {row.run_id}: {exc}', file=sys.stderr)
        return 0, 'FAIL_OS'
    write_stripped_tensors_marker(
        checkpoints, mode='strip-tensors', bytes_removed=freed,
    )
    return freed, 'STRIPPED_TENSORS'


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


def keep_latest_for_m3_run_id(
    persistent_root: Path,
    m3_run_id: str,
    *,
    execute: bool = True,
) -> dict[str, Any]:
    """Trim older ckpt_* for one M3 run; keep newest COMMITTED only.

    Used on the M3 hot path (notebook 97 / gpu_m3_batch) to stop mid-flight
    runs from accumulating ckpt_000001…ckpt_000014. Does not require
    downstream eligibility. Skips runs already fully stripped for reclaim.
    """
    persistent_root = Path(persistent_root)
    if not isinstance(m3_run_id, str) or not m3_run_id.startswith('M3-'):
        return {
            'run_id': m3_run_id,
            'bytes_freed': 0,
            'label': 'SKIP_BAD_RUN_ID',
            'execute': bool(execute),
        }
    run_root = persistent_root / 'runs' / m3_run_id
    checkpoints = run_root / 'checkpoints'
    if not checkpoints.is_dir():
        return {
            'run_id': m3_run_id,
            'bytes_freed': 0,
            'label': 'SKIP_NO_CHECKPOINTS',
            'execute': bool(execute),
        }
    if (checkpoints / STRIPPED_CHECKPOINTS_MARKER).is_file():
        return {
            'run_id': m3_run_id,
            'bytes_freed': 0,
            'label': 'SKIP_ALREADY_STRIPPED',
            'execute': bool(execute),
        }
    row = M3Classification(
        run_id=m3_run_id,
        run_rel=f'runs/{m3_run_id}',
        size_bytes=0,
        checkpoints_bytes=0,
        tensors_bytes=0,
        complete=False,
        referenced=False,
    )
    nbytes, label = keep_latest_for_run(persistent_root, row, execute=execute)
    return {
        'run_id': m3_run_id,
        'bytes_freed': nbytes,
        'bytes_freed_human': fmt_bytes(nbytes),
        'label': label,
        'execute': bool(execute),
    }


@dataclass
class KeepLatestReclaimSummary:
    execute: bool
    scope: str  # 'full_scan'
    candidates: int
    trimmed: int
    skipped: int
    bytes_freed: int
    run_ids: list[str] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            'bytes_freed_human': fmt_bytes(self.bytes_freed),
            # Alias for summary consumers that expect strip-like naming.
            'stripped': self.trimmed,
        }


def keep_latest_all_m3_runs(
    persistent_root: Path,
    *,
    execute: bool = True,
    include_certified_lineage: bool = False,
) -> KeepLatestReclaimSummary:
    """Full-scan keep-latest over all ``runs/M3-*`` (CLI keep-latest mode).

    Trims older / uncommitted ``ckpt_*`` on every M3 run that still has
    reclaimable keep-latest bytes — including incomplete / mid-flight runs
    that are not strip-eligible. Skips already-stripped runs and (by default)
    CERTIFIED M6 lineage. Used at notebook 97 session start so older M3
    checkpoints accumulate across sessions are reclaimed without a separate CLI.
    """
    persistent_root = Path(persistent_root)
    rows = classify_m3_runs(
        persistent_root,
        include_certified_lineage=include_certified_lineage,
    )
    targets: list[M3Classification] = []
    for row in rows:
        if row.already_stripped:
            continue
        if row.certified_lineage and not include_certified_lineage:
            continue
        if row.reclaimable_keep_latest_bytes <= 0:
            continue
        targets.append(row)

    freed = 0
    trimmed = 0
    actions: list[dict[str, Any]] = []
    trimmed_ids: list[str] = []
    for row in targets:
        nbytes, label = keep_latest_for_run(
            persistent_root, row, execute=execute,
        )
        actions.append({
            'run_id': row.run_id,
            'label': label,
            'bytes': nbytes,
        })
        if label in {'KEPT_LATEST_REMOVED_OLDER', 'WOULD_KEEP_LATEST'}:
            freed += nbytes
            trimmed += 1
            trimmed_ids.append(row.run_id)
    return KeepLatestReclaimSummary(
        execute=execute,
        scope='full_scan',
        candidates=len(targets),
        trimmed=trimmed,
        skipped=len(rows) - len(targets),
        bytes_freed=freed,
        run_ids=trimmed_ids,
        actions=actions,
    )


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


def strip_m3_after_m4_complete(
    persistent_root: Path,
    m3_run_id: str,
    *,
    execute: bool = True,
    include_certified_lineage: bool = False,
) -> dict[str, Any]:
    """Immediately strip one M3 run after its child M4 reaches M4_COMPLETE.

    Fail-closed: uses the same classify/eligibility path as batch reclaim.
    Safe because M4 has finished verifying/consuming the M3 parent checkpoint.
    """
    if not isinstance(m3_run_id, str) or not m3_run_id.startswith('M3-'):
        return {
            'run_id': m3_run_id,
            'stripped': 0,
            'bytes_freed': 0,
            'label': 'SKIP_BAD_M3_ID',
            'execute': bool(execute),
        }
    summary = strip_eligible_m3_checkpoints(
        Path(persistent_root),
        execute=execute,
        include_certified_lineage=include_certified_lineage,
        only_run_ids={m3_run_id},
    )
    out = summary.as_dict()
    out['trigger'] = 'm4_complete_immediate'
    out['requested_run_id'] = m3_run_id
    return out


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


def strip_eligible_m3_tensors(
    persistent_root: Path,
    *,
    execute: bool = True,
    include_certified_lineage: bool = False,
    only_run_ids: set[str] | frozenset[str] | None = None,
) -> StripReclaimSummary:
    """Strip tensor dirs only for fail-closed-eligible COMPLETE+downstream runs."""
    persistent_root = Path(persistent_root)
    rows = classify_m3_runs(
        persistent_root,
        include_certified_lineage=include_certified_lineage,
        only_run_ids=only_run_ids,
    )
    targets = [r for r in rows if r.reclaimable_tensors_bytes > 0]
    scope = 'incremental' if only_run_ids is not None else 'full_scan'
    freed = 0
    stripped = 0
    actions: list[dict[str, Any]] = []
    stripped_ids: list[str] = []
    for row in targets:
        nbytes, label = strip_tensors_for_run(
            persistent_root, row, execute=execute,
        )
        actions.append({
            'run_id': row.run_id,
            'label': label,
            'bytes': nbytes,
        })
        if label in {'STRIPPED_TENSORS', 'WOULD_STRIP_TENSORS'}:
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


def m3_runs_total_bytes(persistent_root: Path) -> int:
    """Sum size of all ``runs/M3-*`` directories (non-symlink)."""
    runs = Path(persistent_root) / 'runs'
    if not runs.is_dir():
        return 0
    total = 0
    try:
        children = list(runs.iterdir())
    except OSError:
        return 0
    for child in children:
        if not child.is_dir() or child.is_symlink():
            continue
        if not child.name.startswith('M3-'):
            continue
        if any(marker in child.name for marker in PROTECTED_NAME_MARKERS):
            continue
        total += dir_size(child)
    return total


def enforce_persist_m3_cap(
    persistent_root: Path,
    *,
    cap_gib: float,
    execute: bool = True,
    include_certified_lineage: bool = False,
) -> dict[str, Any]:
    """Strip oldest eligible COMPLETE+downstream M3 runs until under cap.

    Measures ``runs/M3-*`` total size. Uses fail-closed strip-checkpoints
    criteria (same as CLI). Stops when under cap or no eligible targets remain.
    """
    persistent_root = Path(persistent_root)
    if cap_gib is None or float(cap_gib) <= 0:
        raise ValueError('cap_gib must be a positive float (use None at call site to disable)')
    cap_bytes = int(float(cap_gib) * (1024 ** 3))
    before = m3_runs_total_bytes(persistent_root)
    out: dict[str, Any] = {
        'execute': bool(execute),
        'cap_gib': float(cap_gib),
        'cap_bytes': cap_bytes,
        'bytes_before': before,
        'bytes_before_human': fmt_bytes(before),
        'under_cap_before': before <= cap_bytes,
        'stripped': 0,
        'bytes_freed': 0,
        'run_ids': [],
        'actions': [],
        'bytes_after': before,
        'bytes_after_human': fmt_bytes(before),
        'under_cap_after': before <= cap_bytes,
        'stopped_reason': 'already_under_cap' if before <= cap_bytes else None,
    }
    if before <= cap_bytes:
        return out

    rows = classify_m3_runs(
        persistent_root,
        include_certified_lineage=include_certified_lineage,
    )
    eligible = [r for r in rows if r.reclaimable_strip_bytes > 0]
    eligible.sort(key=lambda r: (r.mtime, r.run_id))

    freed = 0
    stripped = 0
    actions: list[dict[str, Any]] = []
    stripped_ids: list[str] = []
    remaining = before
    for row in eligible:
        if remaining <= cap_bytes:
            break
        nbytes, label = strip_checkpoints_for_run(
            persistent_root, row, execute=execute,
        )
        actions.append({
            'run_id': row.run_id,
            'label': label,
            'bytes': nbytes,
            'mtime': row.mtime,
        })
        if label not in {'STRIPPED_CHECKPOINTS', 'WOULD_STRIP_CHECKPOINTS'}:
            continue
        freed += nbytes
        stripped += 1
        stripped_ids.append(row.run_id)
        remaining = max(0, remaining - nbytes)

    after = remaining if execute else max(0, before - freed)
    # On execute, remeasure for accuracy (markers / leftover metadata).
    if execute:
        after = m3_runs_total_bytes(persistent_root)

    stopped = 'under_cap' if after <= cap_bytes else 'no_more_eligible_targets'
    out.update({
        'stripped': stripped,
        'bytes_freed': freed,
        'bytes_freed_human': fmt_bytes(freed),
        'run_ids': stripped_ids,
        'actions': actions,
        'bytes_after': after,
        'bytes_after_human': fmt_bytes(after),
        'under_cap_after': after <= cap_bytes,
        'stopped_reason': stopped,
        'eligible_candidates': len(eligible),
    })
    return out


def auto_strip_after_pipeline_round(
    persistent_root: Path,
    *,
    pre_m6_summary: dict[str, Any] | None = None,
    m6_summary: dict[str, Any] | None = None,
    execute: bool = True,
    persist_m3_cap_gib: float | None = None,
    force_full_scan: bool = False,
) -> dict[str, Any]:
    """Incremental strip from round results; optionally force a full safe scan.

    Prefer run ids from this round's PRE_M6_READY / M6 results. If none are
    found, run a full fail-closed scan once. When ``force_full_scan`` is True
    (session start), always run a full scan — even if incremental ids exist —
    so backlog COMPLETE+downstream runs are stripped when this round made no
    new PRE_M6 progress.

    When ``persist_m3_cap_gib`` is not None, also enforce the size cap after
    the incremental/full strip (oldest eligible first).
    """
    preferred = m3_run_ids_from_stage_results(
        persistent_root,
        pre_m6_summary=pre_m6_summary,
        m6_summary=m6_summary,
    )
    out: dict[str, Any]
    if force_full_scan:
        summary = strip_eligible_m3_checkpoints(
            persistent_root,
            execute=execute,
            only_run_ids=None,
        )
        out = summary.as_dict()
        out['preferred_run_ids'] = sorted(preferred)
        out['force_full_scan'] = True
        out['scope'] = 'full_scan'
    elif preferred:
        summary = strip_eligible_m3_checkpoints(
            persistent_root,
            execute=execute,
            only_run_ids=preferred,
        )
        # If incremental found ids but none were strip-eligible yet (e.g. M4
        # not fully on disk), do not escalate — wait for later round / next
        # session-start full scan.
        out = summary.as_dict()
        out['preferred_run_ids'] = sorted(preferred)
    else:
        summary = strip_eligible_m3_checkpoints(
            persistent_root,
            execute=execute,
            only_run_ids=None,
        )
        out = summary.as_dict()
        out['preferred_run_ids'] = []
        out['fallback_full_scan'] = True

    if persist_m3_cap_gib is not None:
        cap = enforce_persist_m3_cap(
            persistent_root,
            cap_gib=float(persist_m3_cap_gib),
            execute=execute,
        )
        out['persist_cap'] = cap
        # Roll cap frees into session totals when execute stripped extra runs.
        if int(cap.get('stripped') or 0) > 0:
            out['stripped'] = int(out.get('stripped') or 0) + int(cap['stripped'])
            out['bytes_freed'] = int(out.get('bytes_freed') or 0) + int(
                cap.get('bytes_freed') or 0,
            )
            out['bytes_freed_human'] = fmt_bytes(int(out['bytes_freed']))
            extra_ids = [
                rid for rid in (cap.get('run_ids') or [])
                if rid not in (out.get('run_ids') or [])
            ]
            out['run_ids'] = list(out.get('run_ids') or []) + extra_ids
    return out
