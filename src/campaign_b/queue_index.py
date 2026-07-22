"""Lightweight on-disk indexes for Campaign B M3 / pre_m6 queue listing.

Indexes live under ``{PERSIST}/campaign_b/_indexes/`` (~50–100 bytes per
eligible package; typically <1 MiB total). They cache resume/stage tier only —
package JSON remains the source of truth. Listing validates entries until
``max_candidates`` matches are found (early exit).

Ordering (drain-oriented): resume / mid-pipeline first, then discovery path.
No ``q_upper`` ranking (screening priority is unused for backlog drain).

Set ``VALIDATED_RG_DISABLE_QUEUE_INDEX=1`` to force full scans (debug).
"""

from __future__ import annotations

import heapq
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now

# v2: resume/stage-tier only (no q_upper sort keys).
_INDEX_SCHEMA_VERSION = 2
_DISABLE_ENV = 'VALIDATED_RG_DISABLE_QUEUE_INDEX'
_GPU_M3_INDEX_NAME = 'gpu_m3_queue.json'
_PRE_M6_INDEX_NAME = 'pre_m6_queue.json'
_OBLIGATION_INDEX_NAME = 'obligation_queue.json'
_M6_INDEX_NAME = 'm6_queue.json'


def fetch_limit_for_batch(
    *,
    max_items: int,
    max_queue: int,
    oversample: int = 8,
    floor: int = 8,
) -> int:
    """How many index/scan rows to validate before running ``max_items``.

    Avoids validating ``max_queue`` (often 2000) when only 1 session is needed.
    """
    want = max(int(max_items) * int(oversample), int(max_items), int(floor))
    cap = max(1, int(max_queue))
    return min(want, cap)


def _indexes_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_indexes'


def _index_path(persistent_root: Path, name: str) -> Path:
    return _indexes_root(persistent_root) / name


def _index_disabled() -> bool:
    return os.environ.get(_DISABLE_ENV, '').strip() in {'1', 'true', 'yes'}


def _rel_package(persistent_root: Path, package: Path) -> str:
    return str(Path(package).resolve().relative_to(Path(persistent_root).resolve()))


def _load_index(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    doc = read_json(path)
    if not isinstance(doc, dict):
        return None
    if int(doc.get('schema_version') or 0) != _INDEX_SCHEMA_VERSION:
        return None
    entries = doc.get('entries')
    if not isinstance(entries, dict):
        return None
    return doc


def _save_index(path: Path, *, kind: str, entries: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, {
        'schema_version': _INDEX_SCHEMA_VERSION,
        'kind': kind,
        'updated_at': utc_now(),
        'entry_count': len(entries),
        'entries': entries,
        'note': (
            'Resume/stage-tier cache only (no q_upper ranking). '
            'Package ADVANCE/GPU_M3/PRE_M6 JSON is authoritative. '
            'Rebuild on miss or stale validation.'
        ),
    })


def _gpu_m3_sort_tuple(entry: dict[str, Any], rel: str) -> tuple[Any, ...]:
    """Resume tier first, then stable path order (no q_upper)."""
    return (int(entry['t']) if entry.get('t') is not None else 1, rel)


def _gpu_m3_row_for_package(
    package: Path,
    *,
    deprioritize_after: int,
) -> dict[str, Any] | None:
    from .gpu_m3_batch import (
        _consecutive_failures,
        _gpu_status,
        _is_ready_for_m3,
        _queue_tier,
    )

    package = Path(package)
    if not _is_ready_for_m3(package):
        return None
    status = _gpu_status(package)
    fail_count = _consecutive_failures(package)
    tier = _queue_tier(
        status,
        fail_count,
        deprioritize_after=int(deprioritize_after),
    )
    return {
        'package': str(package),
        'candidate_id': package.name,
        'q_upper': None,
        'sort_key': float(tier),
        'gpu_status': status,
        'consecutive_failures': fail_count,
        'queue_tier': tier,
        'deprioritized': tier >= 2,
    }


def _gpu_m3_compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        't': int(row['queue_tier']) if row.get('queue_tier') is not None else 1,
        'gs': row.get('gpu_status'),
        'cf': int(row.get('consecutive_failures') or 0),
    }


def _gpu_m3_eligible_row(
    row: dict[str, Any],
    *,
    include_complete: bool,
    include_errors: bool,
    deprioritize_after: int,
) -> bool:
    from .gpu_m3_batch import M3_FAIL_DEPRIORITIZE_AFTER

    status = row.get('gpu_status')
    blocked_excluded = {'M3_COMPLETE', 'M3_BLOCKED_BAD_M2', 'M3_BLOCKED_NONFINITE'}
    deprioritize_after = max(1, int(deprioritize_after))
    if status in blocked_excluded and not include_complete:
        if status == 'M3_COMPLETE' or not include_errors:
            return False
    if status == 'M3_ERROR' and not include_errors:
        return False
    fail_count = int(row.get('consecutive_failures') or 0)
    if (
        not include_errors
        and fail_count >= deprioritize_after
        and status in {'M3_RUNNING', 'M3_CHECKPOINT', 'M3_ERROR', None}
    ):
        return False
    return True


def rebuild_gpu_m3_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    deprioritize_after: int | None = None,
) -> dict[str, Any]:
    from .advance_selected import discover_selected_packages
    from .gpu_m3_batch import M3_FAIL_DEPRIORITIZE_AFTER

    persistent_root = Path(persistent_root)
    deprioritize_after = int(
        deprioritize_after if deprioritize_after is not None else M3_FAIL_DEPRIORITIZE_AFTER,
    )
    entries: dict[str, Any] = {}
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        row = _gpu_m3_row_for_package(
            package, deprioritize_after=deprioritize_after,
        )
        if row is None:
            continue
        entries[_rel_package(persistent_root, package)] = _gpu_m3_compact(row)
    _save_index(_index_path(persistent_root, _GPU_M3_INDEX_NAME), kind='gpu_m3', entries=entries)
    return {'entry_count': len(entries), 'rebuilt': True}


def sync_gpu_m3_index_entry(
    package: Path,
    persistent_root: Path,
    *,
    deprioritize_after: int | None = None,
) -> None:
    """Upsert or remove one package in the gpu_m3 index (best-effort)."""
    if _index_disabled():
        return
    from .gpu_m3_batch import M3_FAIL_DEPRIORITIZE_AFTER

    persistent_root = Path(persistent_root)
    package = Path(package)
    path = _index_path(persistent_root, _GPU_M3_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return
    entries = dict(doc.get('entries') or {})
    rel = _rel_package(persistent_root, package)
    deprioritize_after = int(
        deprioritize_after if deprioritize_after is not None else M3_FAIL_DEPRIORITIZE_AFTER,
    )
    row = _gpu_m3_row_for_package(package, deprioritize_after=deprioritize_after)
    if row is None or not _gpu_m3_eligible_row(
        row,
        include_complete=False,
        include_errors=False,
        deprioritize_after=deprioritize_after,
    ):
        entries.pop(rel, None)
    else:
        entries[rel] = _gpu_m3_compact(row)
    _save_index(path, kind='gpu_m3', entries=entries)


def _scan_gpu_m3_rows(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    include_errors: bool = False,
    fail_deprioritize_after: int,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """Full filesystem scan with optional heap early-exit (no index)."""
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    deprioritize_after = max(1, int(fail_deprioritize_after))
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [p for p in packages if only_campaign_run_id in p.parts]

    limit = int(max_candidates) if max_candidates is not None else None
    heap: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    oversample = 4

    for package in packages:
        row = _gpu_m3_row_for_package(package, deprioritize_after=deprioritize_after)
        if row is None:
            continue
        if not _gpu_m3_eligible_row(
            row,
            include_complete=include_complete,
            include_errors=include_errors,
            deprioritize_after=deprioritize_after,
        ):
            continue
        key = (int(row['queue_tier']), row['package'])
        if limit is None:
            heap.append((key, row))
            continue
        if len(heap) < limit * oversample:
            heapq.heappush(heap, (key, row))
        elif key < heap[-1][0]:
            heapq.heapreplace(heap, (key, row))

    if limit is None:
        return [r for _, r in sorted(heap, key=lambda x: x[0])]
    return [r for _, r in sorted(heap, key=lambda x: x[0])][:limit]


def list_gpu_m3_queue_indexed(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    include_errors: bool = False,
    fail_deprioritize_after: int,
) -> list[dict[str, Any]] | None:
    """Return rows from index with early-exit validation, or None to fall back."""
    if _index_disabled():
        return None
    persistent_root = Path(persistent_root)
    path = _index_path(persistent_root, _GPU_M3_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return None

    deprioritize_after = max(1, int(fail_deprioritize_after))
    entries: dict[str, Any] = dict(doc.get('entries') or {})
    ordered = sorted(
        entries.items(),
        key=lambda item: _gpu_m3_sort_tuple(item[1], item[0]),
    )

    results: list[dict[str, Any]] = []
    stale: list[str] = []
    need = int(max_candidates) if max_candidates is not None else None

    for rel, compact in ordered:
        if need is not None and len(results) >= need:
            break
        if only_campaign_run_id and only_campaign_run_id not in rel:
            continue
        package = persistent_root / rel
        row = _gpu_m3_row_for_package(package, deprioritize_after=deprioritize_after)
        if row is None:
            stale.append(rel)
            continue
        if not _gpu_m3_eligible_row(
            row,
            include_complete=include_complete,
            include_errors=include_errors,
            deprioritize_after=deprioritize_after,
        ):
            if row.get('gpu_status') in {
                'M3_COMPLETE', 'M3_BLOCKED_BAD_M2', 'M3_BLOCKED_NONFINITE',
            } or (
                row.get('gpu_status') == 'M3_ERROR' and not include_errors
            ):
                stale.append(rel)
            continue
        results.append(row)

    if stale:
        for rel in stale:
            entries.pop(rel, None)
        _save_index(path, kind='gpu_m3', entries=entries)

    if need is not None and not results and ordered:
        return None
    return results


def _pre_m6_row_for_package(
    package: Path,
    persistent_root: Path,
) -> dict[str, Any] | None:
    from .pre_m6_batch import (
        _child_ids,
        _gpu_m3_status,
        _m3_complete_on_disk,
        _m4_complete_on_disk,
        _m5_done_on_disk,
        _pre_m6_status,
    )

    package = Path(package)
    persistent_root = Path(persistent_root)
    status = _pre_m6_status(package)
    child = _child_ids(package)
    if not isinstance(child, dict):
        return None
    m3_id = child.get('M3')
    if not isinstance(m3_id, str) or not m3_id.startswith('M3-'):
        return None
    gpu = _gpu_m3_status(package)
    if gpu != 'M3_COMPLETE' and not _m3_complete_on_disk(persistent_root, m3_id):
        return None
    stage = 'NEED_M4'
    stage_rank = 1
    if isinstance(child.get('M4'), str) and _m4_complete_on_disk(
        persistent_root, str(child['M4']),
    ):
        stage = 'NEED_M5'
        stage_rank = 0
    if isinstance(child.get('M5'), str) and _m5_done_on_disk(
        persistent_root, str(child['M5']),
    ):
        stage = 'PRE_M6_READY'
        stage_rank = 2
    return {
        'package': str(package),
        'candidate_id': package.name,
        'q_upper': None,
        'stage': stage,
        'stage_rank': stage_rank,
        'm3_run_id': m3_id,
        'm4_run_id': child.get('M4'),
        'm5_run_id': child.get('M5'),
        'pre_m6_status': status,
    }


def _pre_m6_compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        'sr': int(row['stage_rank']) if row.get('stage_rank') is not None else 1,
        'st': row.get('stage'),
        'ps': row.get('pre_m6_status'),
    }


def _pre_m6_eligible_row(
    row: dict[str, Any],
    *,
    include_complete: bool,
    include_errors: bool,
) -> bool:
    from .pre_m6_batch import PRE_M6_BLOCKED_EXCLUDED

    status = row.get('pre_m6_status')
    if status == 'PRE_M6_READY' and not include_complete:
        return False
    if status in PRE_M6_BLOCKED_EXCLUDED and not include_errors:
        return False
    if row.get('stage') == 'PRE_M6_READY' and not include_complete:
        return False
    return True


def rebuild_pre_m6_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    entries: dict[str, Any] = {}
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        row = _pre_m6_row_for_package(package, persistent_root)
        if row is None:
            continue
        if row.get('stage') == 'PRE_M6_READY':
            continue
        entries[_rel_package(persistent_root, package)] = _pre_m6_compact(row)
    _save_index(_index_path(persistent_root, _PRE_M6_INDEX_NAME), kind='pre_m6', entries=entries)
    return {'entry_count': len(entries), 'rebuilt': True}


def sync_pre_m6_index_entry(
    package: Path,
    persistent_root: Path,
) -> None:
    if _index_disabled():
        return
    persistent_root = Path(persistent_root)
    package = Path(package)
    path = _index_path(persistent_root, _PRE_M6_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return
    entries = dict(doc.get('entries') or {})
    rel = _rel_package(persistent_root, package)
    row = _pre_m6_row_for_package(package, persistent_root)
    if row is None or not _pre_m6_eligible_row(
        row, include_complete=False, include_errors=False,
    ):
        entries.pop(rel, None)
    else:
        entries[rel] = _pre_m6_compact(row)
    _save_index(path, kind='pre_m6', entries=entries)


def _pre_m6_sort_tuple(entry: dict[str, Any], rel: str) -> tuple[Any, ...]:
    """NEED_M5 before NEED_M4, then stable path (no q_upper)."""
    return (int(entry['sr']) if entry.get('sr') is not None else 1, rel)


def _scan_pre_m6_rows(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    include_errors: bool = False,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [p for p in packages if only_campaign_run_id in p.parts]

    limit = int(max_candidates) if max_candidates is not None else None
    heap: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    oversample = 4

    for package in packages:
        row = _pre_m6_row_for_package(package, persistent_root)
        if row is None:
            continue
        if not _pre_m6_eligible_row(
            row,
            include_complete=include_complete,
            include_errors=include_errors,
        ):
            continue
        key = (int(row['stage_rank']), row['package'])
        if limit is None:
            heap.append((key, row))
            continue
        if len(heap) < limit * oversample:
            heapq.heappush(heap, (key, row))
        elif key < heap[-1][0]:
            heapq.heapreplace(heap, (key, row))

    if limit is None:
        return [r for _, r in sorted(heap, key=lambda x: x[0])]
    return [r for _, r in sorted(heap, key=lambda x: x[0])][:limit]


def list_pre_m6_queue_indexed(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    include_errors: bool = False,
) -> list[dict[str, Any]] | None:
    if _index_disabled():
        return None
    persistent_root = Path(persistent_root)
    path = _index_path(persistent_root, _PRE_M6_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return None

    entries: dict[str, Any] = dict(doc.get('entries') or {})
    ordered = sorted(
        entries.items(),
        key=lambda item: _pre_m6_sort_tuple(item[1], item[0]),
    )

    results: list[dict[str, Any]] = []
    stale: list[str] = []
    need = int(max_candidates) if max_candidates is not None else None

    for rel, _compact in ordered:
        if need is not None and len(results) >= need:
            break
        if only_campaign_run_id and only_campaign_run_id not in rel:
            continue
        package = persistent_root / rel
        row = _pre_m6_row_for_package(package, persistent_root)
        if row is None:
            stale.append(rel)
            continue
        if not _pre_m6_eligible_row(
            row,
            include_complete=include_complete,
            include_errors=include_errors,
        ):
            stale.append(rel)
            continue
        results.append(row)

    if stale:
        for rel in stale:
            entries.pop(rel, None)
        _save_index(path, kind='pre_m6', entries=entries)

    if need is not None and not results and ordered:
        return None
    return results


def ensure_gpu_m3_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    fail_deprioritize_after: int,
) -> None:
    """Create gpu_m3 index on first use if missing."""
    if _index_disabled():
        return
    path = _index_path(Path(persistent_root), _GPU_M3_INDEX_NAME)
    if _load_index(path) is not None:
        return
    rebuild_gpu_m3_index(
        persistent_root,
        only_campaign_run_id=only_campaign_run_id,
        deprioritize_after=fail_deprioritize_after,
    )


def ensure_pre_m6_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> None:
    if _index_disabled():
        return
    path = _index_path(Path(persistent_root), _PRE_M6_INDEX_NAME)
    if _load_index(path) is not None:
        return
    rebuild_pre_m6_index(persistent_root, only_campaign_run_id=only_campaign_run_id)


def _obligation_row_for_package(
    package: Path,
    persistent_root: Path,
) -> dict[str, Any] | None:
    from .pre_m6_batch import (
        _child_ids,
        _load,
        _m4_complete_on_disk,
        _pre_m6_status,
    )

    package = Path(package)
    persistent_root = Path(persistent_root)
    child = _child_ids(package)
    if not isinstance(child, dict):
        return None
    m4_id = str(child.get('M4') or '')
    m5_id = str(child.get('M5') or '')
    if not m4_id.startswith('M4-') or not m5_id.startswith('M5-'):
        return None
    if not _m4_complete_on_disk(persistent_root, m4_id):
        return None
    obl_path = (
        Path(persistent_root) / 'runs' / m5_id / 'reports' / 'M5_obligation_report.json'
    )
    open_ids: list[str] = []
    all_closed = False
    if obl_path.is_file():
        doc = _load(obl_path)
        if isinstance(doc, dict):
            open_ids = list(doc.get('open_obligations') or [])
            all_closed = bool(doc.get('all_closed'))
    if all_closed and not open_ids:
        return None
    return {
        'package': str(package),
        'candidate_id': package.name,
        'q_upper': None,
        'm4_run_id': m4_id,
        'm5_run_id': m5_id,
        'open_obligations': open_ids,
        'pre_m6_status': _pre_m6_status(package),
    }


def rebuild_obligation_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    entries: dict[str, Any] = {}
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        row = _obligation_row_for_package(package, persistent_root)
        if row is None:
            continue
        entries[_rel_package(persistent_root, package)] = {'p': 0}
    _save_index(
        _index_path(persistent_root, _OBLIGATION_INDEX_NAME),
        kind='obligations',
        entries=entries,
    )
    return {'entry_count': len(entries), 'rebuilt': True}


def sync_obligation_index_entry(
    package: Path,
    persistent_root: Path,
) -> None:
    if _index_disabled():
        return
    persistent_root = Path(persistent_root)
    package = Path(package)
    path = _index_path(persistent_root, _OBLIGATION_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return
    entries = dict(doc.get('entries') or {})
    rel = _rel_package(persistent_root, package)
    row = _obligation_row_for_package(package, persistent_root)
    if row is None:
        entries.pop(rel, None)
    else:
        entries[rel] = {'p': 0}
    _save_index(path, kind='obligations', entries=entries)


def _scan_obligation_rows(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [p for p in packages if only_campaign_run_id in p.parts]
    rows: list[dict[str, Any]] = []
    limit = int(max_candidates) if max_candidates is not None else None
    for package in packages:
        row = _obligation_row_for_package(package, persistent_root)
        if row is None:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def list_obligation_queue_indexed(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
) -> list[dict[str, Any]] | None:
    if _index_disabled():
        return None
    persistent_root = Path(persistent_root)
    path = _index_path(persistent_root, _OBLIGATION_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return None
    entries: dict[str, Any] = dict(doc.get('entries') or {})
    ordered = sorted(entries.keys())
    results: list[dict[str, Any]] = []
    stale: list[str] = []
    need = int(max_candidates) if max_candidates is not None else None
    for rel in ordered:
        if need is not None and len(results) >= need:
            break
        if only_campaign_run_id and only_campaign_run_id not in rel:
            continue
        package = persistent_root / rel
        row = _obligation_row_for_package(package, persistent_root)
        if row is None:
            stale.append(rel)
            continue
        results.append(row)
    if stale:
        for rel in stale:
            entries.pop(rel, None)
        _save_index(path, kind='obligations', entries=entries)
    if need is not None and not results and ordered:
        return None
    return results


def ensure_obligation_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> None:
    if _index_disabled():
        return
    path = _index_path(Path(persistent_root), _OBLIGATION_INDEX_NAME)
    if _load_index(path) is not None:
        return
    rebuild_obligation_index(
        persistent_root, only_campaign_run_id=only_campaign_run_id,
    )


def _m6_row_for_package(
    package: Path,
    persistent_root: Path,
    *,
    include_complete: bool = False,
) -> dict[str, Any] | None:
    from .m6_batch import _m5_ready_for_m6, _m6_done
    from .pre_m6_batch import _child_ids, _load

    package = Path(package)
    persistent_root = Path(persistent_root)
    child = _child_ids(package)
    if not isinstance(child, dict):
        return None
    m5_id = str(child.get('M5') or '')
    m6_id = str(child.get('M6') or '')
    if not m5_id.startswith('M5-') or not m6_id.startswith('M6-'):
        return None
    gate = _load(package / 'M6_GATE.json') or {}
    gate_status = str(gate.get('status') or '')
    ready = _m5_ready_for_m6(persistent_root, m5_id)
    if not ready['ok'] and gate_status != 'READY_FOR_STAGED_M6':
        return None
    if not ready['ok']:
        return None
    if _m6_done(persistent_root, m6_id) and not include_complete:
        return None
    return {
        'package': str(package),
        'candidate_id': package.name,
        'q_upper': None,
        'm5_run_id': m5_id,
        'm6_run_id': m6_id,
        'm5_certification_status': ready.get('certification_status'),
        'gate_status': gate_status or 'READY_FOR_STAGED_M6',
    }


def rebuild_m6_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    entries: dict[str, Any] = {}
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        row = _m6_row_for_package(package, persistent_root)
        if row is None:
            continue
        entries[_rel_package(persistent_root, package)] = {'p': 0}
    _save_index(
        _index_path(persistent_root, _M6_INDEX_NAME),
        kind='m6',
        entries=entries,
    )
    return {'entry_count': len(entries), 'rebuilt': True}


def sync_m6_index_entry(
    package: Path,
    persistent_root: Path,
) -> None:
    if _index_disabled():
        return
    persistent_root = Path(persistent_root)
    package = Path(package)
    path = _index_path(persistent_root, _M6_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return
    entries = dict(doc.get('entries') or {})
    rel = _rel_package(persistent_root, package)
    row = _m6_row_for_package(package, persistent_root)
    if row is None:
        entries.pop(rel, None)
    else:
        entries[rel] = {'p': 0}
    _save_index(path, kind='m6', entries=entries)


def _scan_m6_rows(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    max_candidates: int | None = None,
) -> list[dict[str, Any]]:
    from .advance_selected import discover_selected_packages

    persistent_root = Path(persistent_root)
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [p for p in packages if only_campaign_run_id in p.parts]
    rows: list[dict[str, Any]] = []
    limit = int(max_candidates) if max_candidates is not None else None
    for package in packages:
        row = _m6_row_for_package(
            package, persistent_root, include_complete=include_complete,
        )
        if row is None:
            continue
        rows.append(row)
        if limit is not None and len(rows) >= limit:
            break
    return rows


def list_m6_queue_indexed(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
) -> list[dict[str, Any]] | None:
    if _index_disabled():
        return None
    if include_complete:
        return None
    persistent_root = Path(persistent_root)
    path = _index_path(persistent_root, _M6_INDEX_NAME)
    doc = _load_index(path)
    if doc is None:
        return None
    entries: dict[str, Any] = dict(doc.get('entries') or {})
    ordered = sorted(entries.keys())
    results: list[dict[str, Any]] = []
    stale: list[str] = []
    need = int(max_candidates) if max_candidates is not None else None
    for rel in ordered:
        if need is not None and len(results) >= need:
            break
        if only_campaign_run_id and only_campaign_run_id not in rel:
            continue
        package = persistent_root / rel
        row = _m6_row_for_package(package, persistent_root)
        if row is None:
            stale.append(rel)
            continue
        results.append(row)
    if stale:
        for rel in stale:
            entries.pop(rel, None)
        _save_index(path, kind='m6', entries=entries)
    if need is not None and not results and ordered:
        return None
    return results


def ensure_m6_index(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None = None,
) -> None:
    if _index_disabled():
        return
    path = _index_path(Path(persistent_root), _M6_INDEX_NAME)
    if _load_index(path) is not None:
        return
    rebuild_m6_index(persistent_root, only_campaign_run_id=only_campaign_run_id)
