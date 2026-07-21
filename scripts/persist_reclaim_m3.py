#!/usr/bin/env python3
"""Safe reclaim of Campaign B screening M3 checkpoint bloat on Paperspace.

Why unreferenced≈2 is not enough
---------------------------------
Almost every ``runs/M3-*`` dir is still pointed to by a selected package
(``GPU_M3.json`` / ``child_run_ids.json``). Deleting only unreferenced runs
frees ~1 GiB; the ~150 GiB bill is dominated by *referenced* screening M3
runs (~640 MiB each), mostly ``checkpoints/*/tensors/``.

Reclaim modes
-------------
``strip-checkpoints`` (Option A, default):
  For ``M3_COMPLETE`` runs that already have downstream ``M4_COMPLETE``
  (or M5/M6 progress), delete the M3 ``checkpoints/`` tree. Keeps
  reports/acceptance/config/artifacts/work_items. After this, that M3
  cannot be used as an M4 parent resume source (fail-closed by design).

``keep-latest-checkpoint`` (Option C):
  Keep only the newest ``COMMITTED`` ``ckpt_*``; delete older checkpoint
  dirs. Useful for incomplete / still-needed runs that may resume.

``delete-run`` (Option B):
  Delete an entire ``runs/M3-*`` directory. Only for unreferenced runs,
  or packages marked archived/abandoned, and only with
  ``--allow-delete-run``. Never deletes ``campaign_b/*/selected/*``
  package dirs. Optionally clears package M3 pointers when abandoned.

Safety
------
- Default is dry-run. Mutations require ``--execute``.
- Never touches ``m6_certified_catalog`` or runs whose M6 acceptance is
  ``CERTIFIED`` / ``ONE_STEP_CERTIFIED`` (skip lineage by default).
- Never deletes selected package directories.
- Fail closed: ambiguous / protected / incomplete-without-downstream
  targets are skipped with an explicit reason.

Paperspace (from repo root)::

  export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg

  # Dry-run (default): classify + reclaimable GiB
  python scripts/persist_reclaim_m3.py --mode strip-checkpoints

  # Execute safe strip
  python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute
"""

from __future__ import annotations

import argparse
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


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f'{n} B'
    for unit, scale in (('KiB', 1024), ('MiB', 1024**2), ('GiB', 1024**3), ('TiB', 1024**4)):
        if n < scale * 1024 or unit == 'TiB':
            return f'{n / scale:.3f} {unit}'
    return f'{n} B'


def _dir_size(path: Path) -> int:
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


def _m3_complete_on_disk(run_root: Path) -> bool:
    report = _load_json_dict(run_root / 'reports' / 'M3_report.json')
    acceptance = _load_json_dict(run_root / 'reports' / 'M3_acceptance.json')
    if report is None or acceptance is None:
        return False
    return (
        report.get('phase') == 'M3_COMPLETE'
        and acceptance.get('status') == 'PASS'
    )


def _m4_complete_on_disk(persistent_root: Path, m4_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m4_run_id
    report = _load_json_dict(root / 'reports' / 'M4_report.json')
    acceptance = _load_json_dict(root / 'reports' / 'M4_acceptance.json')
    if report is None or acceptance is None:
        return False
    return report.get('phase') == 'M4_COMPLETE' and acceptance.get('status') == 'PASS'


def _m5_progress_on_disk(persistent_root: Path, m5_run_id: str) -> bool:
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


def _m6_status_on_disk(persistent_root: Path, m6_run_id: str) -> tuple[bool, str | None]:
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


def _package_archived(package: Path) -> bool:
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


def _committed_ckpt_dirs(checkpoints: Path) -> list[Path]:
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


def _tensors_bytes(checkpoints: Path) -> int:
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
            total += _dir_size(tensors)
    return total

def _keep_latest_reclaimable(checkpoints: Path) -> tuple[int, list[Path]]:
    """Bytes and dirs that would be removed if keeping only newest COMMITTED ckpt."""
    committed = _committed_ckpt_dirs(checkpoints)
    if len(committed) <= 1:
        # Still reclaim non-COMMITTED tmp-like ckpt dirs / incomplete.
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
                        bytes_extra += _dir_size(child)
            except OSError:
                pass
        return bytes_extra, extra
    keep = committed[-1]
    remove = [p for p in committed[:-1]]
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
    return sum(_dir_size(p) for p in remove), remove


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
        m4_ok = bool(m4_id and _m4_complete_on_disk(persistent_root, m4_id))
        m5_ok = bool(m5_id and _m5_progress_on_disk(persistent_root, m5_id))
        m6_ok, m6_cert = (False, None)
        if m6_id:
            m6_ok, m6_cert = _m6_status_on_disk(persistent_root, m6_id)
        ref = PackageRef(
            package_rel=rel,
            m3_run_id=m3_id if isinstance(m3_id, str) else None,
            gpu_status=str(gpu['status']) if isinstance(gpu.get('status'), str) else None,
            child_m4=m4_id,
            child_m5=m5_id,
            child_m6=m6_id,
            archived=_package_archived(package),
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
        # Never operate under CERTIFIED-named run dirs.
        if any(marker in run_id for marker in PROTECTED_NAME_MARKERS):
            continue
        checkpoints = run_root / 'checkpoints'
        size = _dir_size(run_root)
        ckpt_bytes = _dir_size(checkpoints) if checkpoints.is_dir() else 0
        tensor_bytes = _tensors_bytes(checkpoints)
        complete = _m3_complete_on_disk(run_root)
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
        keep_latest_bytes, _ = _keep_latest_reclaimable(checkpoints)

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


def _write_stripped_marker(checkpoints: Path, *, mode: str, bytes_removed: int) -> None:
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
    bytes_to_free = _dir_size(checkpoints)
    if not execute:
        return bytes_to_free, 'WOULD_STRIP_CHECKPOINTS'
    # Remove each ckpt_* (and any other children except we recreate marker).
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
    _write_stripped_marker(checkpoints, mode='strip-checkpoints', bytes_removed=bytes_to_free)
    return bytes_to_free, 'STRIPPED_CHECKPOINTS'


def keep_latest_for_run(
    persistent_root: Path,
    row: M3Classification,
    *,
    execute: bool,
) -> tuple[int, str]:
    run_root = Path(persistent_root) / 'runs' / row.run_id
    checkpoints = run_root / 'checkpoints'
    reclaim_bytes, remove = _keep_latest_reclaimable(checkpoints)
    if reclaim_bytes <= 0 or not remove:
        return 0, 'SKIP_NOTHING_TO_TRIM'
    if not execute:
        return reclaim_bytes, 'WOULD_KEEP_LATEST'
    freed = 0
    for path in remove:
        if path.is_symlink():
            print(f'  FAIL_CLOSED symlink: {path}', file=sys.stderr)
            continue
        size = _dir_size(path)
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
    size = _dir_size(run_root)
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
        if not _package_archived(package):
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


def print_report(rows: list[M3Classification], *, mode: str) -> dict[str, Any]:
    total = len(rows)
    complete = sum(1 for r in rows if r.complete)
    incomplete = total - complete
    referenced = sum(1 for r in rows if r.referenced)
    unreferenced = total - referenced
    strip_targets = [r for r in rows if r.reclaimable_strip_bytes > 0]
    strip_bytes = sum(r.reclaimable_strip_bytes for r in strip_targets)
    keep_latest_bytes = sum(r.reclaimable_keep_latest_bytes for r in rows)
    total_size = sum(r.size_bytes for r in rows)
    ckpt_size = sum(r.checkpoints_bytes for r in rows)
    tensor_size = sum(r.tensors_bytes for r in rows)

    print('=== M3 classification ===')
    print(f'  total_m3:           {total}')
    print(f'  complete:           {complete}')
    print(f'  incomplete:         {incomplete}')
    print(f'  referenced:         {referenced}')
    print(f'  unreferenced:       {unreferenced}')
    print(f'  total_m3_size:      {_fmt_bytes(total_size)}')
    print(f'  checkpoints_size:   {_fmt_bytes(ckpt_size)}')
    print(f'  tensors_in_ckpts:   {_fmt_bytes(tensor_size)}')
    print()
    print(f'=== mode={mode} reclaimable (dry-run estimate) ===')
    if mode == 'strip-checkpoints':
        print(f'  safe_strip_targets: {len(strip_targets)}')
        print(f'  reclaimable:        {_fmt_bytes(strip_bytes)}')
        print(
            '  criterion: M3_COMPLETE + downstream M4_COMPLETE/M5/M6 '
            '+ not CERTIFIED lineage + checkpoints present'
        )
    elif mode == 'keep-latest-checkpoint':
        print(f'  reclaimable:        {_fmt_bytes(keep_latest_bytes)}')
        print('  criterion: delete older/incomplete ckpt_* ; keep newest COMMITTED')
    else:
        unref_bytes = sum(r.size_bytes for r in rows if not r.referenced)
        archived_bytes = sum(
            r.size_bytes for r in rows if r.archived_only_refs and r.referenced
        )
        print(f'  unreferenced_runs:  {_fmt_bytes(unref_bytes)} ({unreferenced} runs)')
        print(
            f'  archived_ref_runs:  {_fmt_bytes(archived_bytes)} '
            f'(needs --allow-delete-run + archived package)'
        )

    print()
    print('=== top reclaim candidates (strip-checkpoints) ===')
    for row in sorted(strip_targets, key=lambda r: r.reclaimable_strip_bytes, reverse=True)[:20]:
        print(
            f'  {_fmt_bytes(row.reclaimable_strip_bytes):>12}  {row.run_rel}  '
            f'refs={len(row.package_refs)}'
        )
    skipped = [r for r in rows if r.reclaimable_strip_bytes <= 0]
    reason_counts: dict[str, int] = {}
    for row in skipped:
        key = ','.join(row.skip_reasons) if row.skip_reasons else 'other'
        reason_counts[key] = reason_counts.get(key, 0) + 1
    print()
    print('=== strip skip reasons ===')
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {count:5d}  {reason}')

    return {
        'total_m3': total,
        'complete': complete,
        'incomplete': incomplete,
        'referenced': referenced,
        'unreferenced': unreferenced,
        'total_m3_size_bytes': total_size,
        'checkpoints_size_bytes': ckpt_size,
        'tensors_size_bytes': tensor_size,
        'strip_targets': len(strip_targets),
        'reclaimable_strip_bytes': strip_bytes,
        'reclaimable_keep_latest_bytes': keep_latest_bytes,
        'skip_reason_counts': reason_counts,
    }


def execute_mode(
    persistent_root: Path,
    rows: list[M3Classification],
    *,
    mode: str,
    execute: bool,
    allow_delete_run: bool,
    clear_pointers: bool,
    include_certified_lineage: bool,
) -> int:
    verb = 'EXECUTE' if execute else 'DRY-RUN'
    print()
    print(f'=== {verb} mode={mode} ===')
    freed = 0
    actions = 0
    if mode == 'strip-checkpoints':
        targets = [r for r in rows if r.reclaimable_strip_bytes > 0]
        if include_certified_lineage:
            # Recompute: allow certified by ignoring that skip — already in reclaimable
            # only when flag was set at classify time.
            pass
        for row in targets:
            nbytes, label = strip_checkpoints_for_run(
                persistent_root, row, execute=execute,
            )
            print(f'{label:28s} {_fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
    elif mode == 'keep-latest-checkpoint':
        for row in rows:
            if row.certified_lineage and not include_certified_lineage:
                continue
            if row.reclaimable_keep_latest_bytes <= 0:
                continue
            nbytes, label = keep_latest_for_run(
                persistent_root, row, execute=execute,
            )
            if nbytes <= 0 and label.startswith('SKIP'):
                continue
            print(f'{label:28s} {_fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
    elif mode == 'delete-run':
        if not allow_delete_run:
            print(
                'ERROR: delete-run requires --allow-delete-run '
                '(and never deletes selected package dirs).',
                file=sys.stderr,
            )
            return 2
        for row in rows:
            eligible = (not row.referenced) or row.archived_only_refs
            if not eligible:
                continue
            nbytes, label = delete_run(
                persistent_root,
                row,
                execute=execute,
                allow_referenced_archived=True,
            )
            if label.startswith('SKIP'):
                print(f'{label:28s} {_fmt_bytes(0):>12}  {row.run_rel}')
                continue
            print(f'{label:28s} {_fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
            if clear_pointers and (row.archived_only_refs or not row.referenced):
                n = clear_abandoned_package_pointers(
                    persistent_root, row.run_id, execute=execute,
                )
                if n:
                    print(f'  pointers_cleared={n} packages for {row.run_id}')
    else:
        print(f'ERROR: unknown mode {mode!r}', file=sys.stderr)
        return 2

    print()
    print(
        f'{"Would free" if not execute else "Freed"} ≈ {_fmt_bytes(freed)} '
        f'across {actions} actions.'
    )
    if not execute:
        print('Re-run with --execute to apply. Default remains dry-run.')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg',
        ),
        help='Persist root (default: $VALIDATED_RG_PERSIST_ROOT or /storage/...)',
    )
    parser.add_argument(
        '--mode',
        choices=('strip-checkpoints', 'keep-latest-checkpoint', 'delete-run'),
        default='strip-checkpoints',
        help='Reclaim strategy (default: strip-checkpoints)',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete files (default is dry-run)',
    )
    parser.add_argument(
        '--allow-delete-run',
        action='store_true',
        help='Required for --mode delete-run',
    )
    parser.add_argument(
        '--clear-pointers',
        action='store_true',
        help='With delete-run, clear archived package M3 pointers',
    )
    parser.add_argument(
        '--include-certified-lineage',
        action='store_true',
        help='Allow reclaim on M3 runs whose package has CERTIFIED M6 (dangerous)',
    )
    parser.add_argument(
        '--json-summary',
        type=Path,
        default=None,
        help='Optional path to write machine-readable summary JSON',
    )
    args = parser.parse_args(argv)

    root = Path(args.persistent_root).expanduser()
    print(f'persist_root: {root}')
    print(f'exists: {root.is_dir()}')
    print(f'mode: {args.mode}')
    print(f'execute: {bool(args.execute)}')
    if not root.is_dir():
        print('ERROR: persist root missing; fail closed.', file=sys.stderr)
        return 2

    # Hard refuse to treat selected/ as a run root.
    if (root / 'selected').is_dir() and not (root / 'runs').is_dir():
        print('ERROR: path looks like a package tree, not persist root.', file=sys.stderr)
        return 2

    rows = classify_m3_runs(
        root, include_certified_lineage=args.include_certified_lineage,
    )
    summary = print_report(rows, mode=args.mode)
    summary['mode'] = args.mode
    summary['execute'] = bool(args.execute)
    summary['persistent_root'] = str(root)

    code = execute_mode(
        root,
        rows,
        mode=args.mode,
        execute=bool(args.execute),
        allow_delete_run=bool(args.allow_delete_run),
        clear_pointers=bool(args.clear_pointers),
        include_certified_lineage=bool(args.include_certified_lineage),
    )

    if args.json_summary is not None:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        # Include per-run compact rows for automation.
        summary['runs'] = [asdict(r) for r in rows]
        args.json_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(f'wrote_json_summary: {args.json_summary}')

    # Unused but documents keep prefixes for operators.
    _ = KEEP_RELATIVE_PREFIXES
    return code


if __name__ == '__main__':
    raise SystemExit(main())
