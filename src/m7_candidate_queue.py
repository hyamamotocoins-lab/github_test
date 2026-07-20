"""Campaign C candidate queue: list, materialize, pick next for S0 series."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json
from .cutoff_dims import resource_gate
from .m7_archive import is_archived, read_advance, read_archive
from .m7_auto_execute import materialize_s3_lineage_package
from .m7_staged_lineage import inspect_staged_m2_progress
from .m7_status import M6_PARENT_RUN_ID_FROZEN, M7_RUN_ID_CAMPAIGN_C


class M7CandidateQueueError(RuntimeError):
    """Raised when candidate queue operations fail closed."""


def search_root_for(persistent_root: Path, m7c_run_id: str) -> Path:
    root = Path(persistent_root) / 'searches' / m7c_run_id
    if not root.is_dir():
        raise M7CandidateQueueError(f'Search root missing: {root}')
    return root.resolve()


def load_ranking(search_root: Path) -> list[dict[str, Any]]:
    path = Path(search_root) / 'reports' / 'candidate_ranking.json'
    if not path.is_file():
        raise M7CandidateQueueError(f'Missing ranking: {path}')
    payload = read_json(path)
    if isinstance(payload, dict):
        rows = payload.get('ranking')
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = None
    if not isinstance(rows, list) or not rows:
        raise M7CandidateQueueError('candidate_ranking.json has no ranking rows.')
    return [row for row in rows if isinstance(row, dict)]


def package_root_for(search_root: Path, candidate_id: str) -> Path:
    return Path(search_root) / 'auto_execute' / candidate_id


def latest_sweep_summary(package_root: Path) -> dict[str, Any] | None:
    sweep_root = Path(package_root) / 'rank_sweep'
    if not sweep_root.is_dir():
        return None
    sweeps = sorted(
        (path for path in sweep_root.iterdir() if path.is_dir() and path.name.startswith('SWEEP-')),
        key=lambda path: path.name,
        reverse=True,
    )
    for sweep in sweeps:
        summary_path = sweep / 'rank_sweep_summary.json'
        if summary_path.is_file():
            payload = read_json(summary_path)
            if isinstance(payload, dict):
                payload = dict(payload)
                payload['_sweep_root'] = str(sweep)
                return payload
    return None


def m2_status(
    package_root: Path,
    persistent_root: Path,
) -> dict[str, Any]:
    from .m2_shared_registry import (
        BINDING_READY,
        canonical_m2_run_id_for_package,
        read_binding,
        verify_shared_m2,
    )

    binding = read_binding(package_root)
    m2_run_id = canonical_m2_run_id_for_package(package_root)
    if not m2_run_id:
        return {
            'ready': False,
            'm2_run_id': None,
            'reason': 'M2 binding / child_run_ids.M2 missing',
            'm2_binding': binding,
        }
    progress = inspect_staged_m2_progress(persistent_root, run_id=m2_run_id)
    acceptance = (
        Path(persistent_root) / 'runs' / m2_run_id / 'reports' / 'M2_acceptance.json'
    )
    ready = False
    if isinstance(binding, dict) and binding.get('state') == BINDING_READY:
        ready = acceptance.is_file()
        if ready and binding.get('structural_key') and binding.get('proof_key'):
            try:
                verify_shared_m2(
                    persistent_root,
                    str(binding['structural_key']),
                    str(binding['proof_key']),
                    require_source_match=False,
                )
            except Exception as exc:  # noqa: BLE001
                return {
                    'ready': False,
                    'm2_run_id': m2_run_id,
                    'progress': progress,
                    'm2_binding': binding,
                    'reason': f'shared M2 verify failed: {exc}',
                }
    else:
        # Legacy packages without READY_SHARED binding.
        ready = bool(progress.get('m2_complete')) and acceptance.is_file()
        if acceptance.is_file():
            acc = read_json(acceptance)
            if isinstance(acc, dict) and acc.get('status') not in {None, 'PASS'}:
                ready = False
    return {
        'ready': ready,
        'm2_run_id': m2_run_id,
        'progress': progress,
        'acceptance_path': str(acceptance) if acceptance.is_file() else None,
        'm2_binding': binding,
        'm2_state': (binding or {}).get('state') if isinstance(binding, dict) else None,
        'shared': (
            isinstance(binding, dict) and binding.get('state') == BINDING_READY
        ),
    }


def m2_ready(package_root: Path, persistent_root: Path) -> bool:
    return bool(m2_status(package_root, persistent_root).get('ready'))


def _est_q(row: dict[str, Any]) -> float:
    try:
        return float(row.get('q_cert_upper') or 1e9)
    except (TypeError, ValueError):
        return 1e9


def _j2(row: dict[str, Any]) -> int:
    scheme = row.get('scheme') or {}
    try:
        return int(scheme.get('j2_max', 1))
    except (TypeError, ValueError):
        return 1


def list_queue_rows(
    search_root: Path,
    *,
    persistent_root: Path,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
) -> list[dict[str, Any]]:
    rows = load_ranking(search_root)
    out: list[dict[str, Any]] = []
    for row in sorted(rows, key=_est_q):
        candidate_id = str(row.get('candidate_id') or '')
        if not candidate_id:
            continue
        gate = resource_gate(
            _j2(row),
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        package = package_root_for(search_root, candidate_id)
        materialized = (package / 'MANIFEST.json').is_file()
        archived = is_archived(package) if materialized else False
        archive = read_archive(package) if archived else None
        advance = read_advance(package) if materialized else None
        sweep = latest_sweep_summary(package) if materialized else None
        m2 = m2_status(package, persistent_root) if materialized else {
            'ready': False, 'm2_run_id': None, 'reason': 'not_materialized',
        }
        out.append({
            'candidate_id': candidate_id,
            'scheme_hash': row.get('scheme_hash'),
            'q_cert_upper': row.get('q_cert_upper'),
            'estimated_q': _est_q(row),
            'j2_max': _j2(row),
            'scheme': row.get('scheme') or {},
            'resource_gate': gate,
            'staged_executable': bool(gate.get('staged_executable')),
            'instant_executable': bool(gate.get('executable')),
            'materialized': materialized,
            'package_root': str(package) if materialized else None,
            'archived': archived,
            'archive_reason': (archive or {}).get('reason') if archive else None,
            'advance': advance,
            'm2_ready': bool(m2.get('ready')),
            'm2_run_id': m2.get('m2_run_id'),
            's0_selection_status': (sweep or {}).get('selection_status'),
            's0_selected_rank': (sweep or {}).get('selected_rank'),
            'latest_sweep_root': (sweep or {}).get('_sweep_root'),
            'ranking_row': row,
        })
    return out


def next_actionable_candidate(
    search_root: Path,
    *,
    persistent_root: Path,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    prefer_staged: bool = True,
    skip_advanced: bool = True,
) -> dict[str, Any] | None:
    """Return next non-archived, live-capable candidate for S0 series."""
    rows = list_queue_rows(
        search_root,
        persistent_root=persistent_root,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    candidates = []
    for row in rows:
        if row.get('archived'):
            continue
        if skip_advanced and row.get('advance'):
            continue
        if prefer_staged:
            if not (row.get('staged_executable') or row.get('instant_executable')):
                continue
        else:
            if not row.get('instant_executable'):
                continue
        candidates.append(row)
    if not candidates:
        return None
    # Prefer staged (j2>=2), then already M2-ready, then materialized, then lowest q.
    candidates.sort(key=lambda row: (
        0 if row.get('staged_executable') else 1,
        0 if row.get('m2_ready') else 1,
        0 if row.get('materialized') else 1,
        float(row.get('estimated_q') or 1e9),
        str(row.get('candidate_id') or ''),
    ))
    return candidates[0]


def resolve_parent_ids(search_root: Path) -> dict[str, str]:
    """Infer parent_m6 / search_run_id from LOCK or STATUS."""
    search_root = Path(search_root)
    lock = read_json(search_root / 'LOCK.json')
    status = read_json(search_root / 'auto_execute' / 'STATUS.json')
    parent_m6 = M6_PARENT_RUN_ID_FROZEN
    search_run_id = search_root.name
    if isinstance(lock, dict):
        parent_m6 = str(lock.get('parent_m6_run_id') or parent_m6)
        search_run_id = str(lock.get('run_id') or search_run_id)
    if isinstance(status, dict):
        parent_m6 = str(status.get('parent_m6_run_id') or parent_m6)
        search_run_id = str(status.get('search_run_id') or search_run_id)
    return {
        'parent_m6_run_id': parent_m6,
        'search_run_id': search_run_id,
    }


def ensure_materialized(
    search_root: Path,
    candidate_row: dict[str, Any],
    *,
    parent_m6_run_id: str | None = None,
    search_run_id: str | None = None,
    parent_j2_max: int = 1,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Materialize package if missing; return MANIFEST-like dict."""
    candidate_id = str(
        candidate_row.get('candidate_id')
        or (candidate_row.get('ranking_row') or {}).get('candidate_id')
        or ''
    )
    if not candidate_id:
        raise M7CandidateQueueError('candidate_id missing for materialize.')
    package = package_root_for(search_root, candidate_id)
    if (package / 'MANIFEST.json').is_file():
        # Refresh M2 binding against current registry.
        scheme = read_json(package / 'scheme.json')
        j2_max = int((scheme or {}).get('j2_max', 2)) if isinstance(scheme, dict) else 2
        from .m2_shared_registry import resolve_m2_binding
        persist = Path(search_root).resolve().parent.parent
        if project_root is not None:
            binding = resolve_m2_binding(
                persistent_root=persist,
                project_root=project_root,
                package_root=package,
                j2_max=j2_max,
            )
            child_ids = read_json(package / 'child_run_ids.json')
            if isinstance(child_ids, dict) and binding.get('canonical_run_id'):
                child_ids = dict(child_ids)
                child_ids['M2'] = str(binding['canonical_run_id'])
                from .common import atomic_write_json
                atomic_write_json(package / 'child_run_ids.json', child_ids)
        manifest = read_json(package / 'MANIFEST.json')
        if isinstance(manifest, dict):
            return manifest
    ids = resolve_parent_ids(search_root)
    ranking_row = candidate_row.get('ranking_row') or candidate_row
    if not isinstance(ranking_row, dict):
        raise M7CandidateQueueError('ranking_row required to materialize.')
    return materialize_s3_lineage_package(
        Path(search_root),
        ranking_row,
        parent_m6_run_id=parent_m6_run_id or ids['parent_m6_run_id'],
        search_run_id=search_run_id or ids['search_run_id'],
        parent_j2_max=parent_j2_max,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
        persistent_root=Path(search_root).resolve().parent.parent,
        project_root=project_root,
    )


def materialize_top_k(
    search_root: Path,
    *,
    persistent_root: Path,
    k: int = 3,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
) -> list[dict[str, Any]]:
    """Materialize up to k non-archived staged/instant candidates."""
    rows = list_queue_rows(
        search_root,
        persistent_root=persistent_root,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    created: list[dict[str, Any]] = []
    for row in rows:
        if len(created) >= k:
            break
        if row.get('archived'):
            continue
        if not (row.get('staged_executable') or row.get('instant_executable')):
            continue
        manifest = ensure_materialized(
            search_root,
            row,
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        created.append(manifest)
    return created


def default_campaign_c_ids() -> dict[str, str]:
    return {
        'm7c_run_id': M7_RUN_ID_CAMPAIGN_C,
        'parent_m6_run_id': M6_PARENT_RUN_ID_FROZEN,
    }
