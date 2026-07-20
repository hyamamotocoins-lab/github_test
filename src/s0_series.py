"""S0 rank-sweep series driver for Campaign C candidate queue.

Runs exploratory nested RSVD sweeps only. Never starts production M6.
Staged M2 is never launched here; missing M2 returns NEED_M2.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .common import read_json, utc_now
from .cutoff_dims import cutoff_dimension_payload
from .m3_config import M3Config
from .m5_closure_schema import (
    SELECTION_NO_SELECTION,
    SELECTION_REJECT,
    SELECTION_SELECTED,
)
from .m7_archive import (
    append_series_log,
    archive_from_sweep,
    write_advance,
)
from .m7_candidate_queue import (
    ensure_materialized,
    list_queue_rows,
    m2_ready,
    m2_status,
    next_actionable_candidate,
    package_root_for,
)
from .m7_staged_lineage import write_child_m2_acceptance_audit
from .rank_sweep import default_sweep_config, run_rank_sweep
from .rank_sweep_reporting import write_rank_sweep_report


class S0SeriesError(RuntimeError):
    """Raised when the S0 series cannot proceed safely."""


def build_m3_config_for_package(
    project_root: Path,
    package_root: Path,
    persistent_root: Path,
) -> M3Config:
    over = read_json(Path(package_root) / 'm3_config_overrides.json')
    child_ids = read_json(Path(package_root) / 'child_run_ids.json')
    if not isinstance(over, dict) or not isinstance(child_ids, dict):
        raise S0SeriesError('Package missing m3_config_overrides or child_run_ids.')
    from .m2_package_audit import (
        package_m2_audit_path,
        read_package_m2_audit,
        write_package_m2_shared_audit,
    )
    from .m2_shared_registry import BINDING_READY, canonical_m2_run_id_for_package, read_binding
    m2_run_id = canonical_m2_run_id_for_package(package_root) or str(
        child_ids.get('M2') or ''
    )
    if not m2_run_id:
        raise S0SeriesError('Shared/canonical M2 run id missing.')
    binding = read_binding(package_root)
    if isinstance(binding, dict) and binding.get('state') != BINDING_READY:
        # Allow legacy packages with acceptance on disk.
        pass
    if child_ids.get('M2') != m2_run_id:
        child_ids = dict(child_ids)
        child_ids['M2'] = m2_run_id
        from .common import atomic_write_json
        atomic_write_json(Path(package_root) / 'child_run_ids.json', child_ids)

    audit = read_package_m2_audit(package_root)
    if audit is None:
        # Prefer package-local audit; fall back to writing from shared run.
        sk = (binding or {}).get('structural_key') if isinstance(binding, dict) else None
        pk = (binding or {}).get('proof_key') if isinstance(binding, dict) else None
        if sk and pk:
            audit = write_package_m2_shared_audit(
                package_root,
                run_root=Path(persistent_root) / 'runs' / m2_run_id,
                structural_key=str(sk),
                proof_key=str(pk),
                registry_record_sha256=(
                    binding.get('registry_record_sha256')
                    if isinstance(binding, dict) else None
                ),
            )
        else:
            audit = write_child_m2_acceptance_audit(
                project_root,
                run_root=Path(persistent_root) / 'runs' / m2_run_id,
            )
    # Shared package audit must be verified in place (never rewrite global audit).
    if audit.get('shared_m2') is True or package_m2_audit_path(package_root).is_file():
        parent_audit_path = str(package_m2_audit_path(package_root).resolve())
    else:
        parent_audit_path = 'audit/m2_accepted_parent.json'
    base = asdict(M3Config())
    base.update({
        'parent_run_id': audit['accepted_run_id'],
        'parent_checkpoint': Path(audit['checkpoint_path']).name,
        'parent_checkpoint_path': audit['checkpoint_path'],
        'parent_report_path': audit['m2_report_path'],
        'parent_acceptance_path': audit['m2_acceptance_path'],
        'parent_audit_path': parent_audit_path,
        'j2_max': int(over['j2_max']),
        'sector_count': int(over['sector_count']),
        'operator_dimension': int(over['operator_dimension']),
        'target_rank': int(over['target_rank']),
        'require_cuda': True,
    })
    return M3Config(**base)


def run_s0_for_package(
    *,
    project_root: Path,
    persistent_root: Path,
    package_root: Path,
    candidate_id: str,
    sweep_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run one exploratory S0 sweep and write ARCHIVE or ADVANCE."""
    if not m2_ready(package_root, persistent_root):
        status = m2_status(package_root, persistent_root)
        return {
            'status': 'NEED_M2',
            'candidate_id': candidate_id,
            'package_root': str(package_root),
            'm2': status,
            'next_notebook': '73_m7_staged_s3_lineage.ipynb',
            'generated_at': utc_now(),
        }
    m3_config = build_m3_config_for_package(
        project_root, package_root, persistent_root,
    )
    config = default_sweep_config(**(sweep_config or {}))
    summary = run_rank_sweep(
        project_root=project_root,
        persistent_root=persistent_root,
        package_root=package_root,
        m3_config=m3_config,
        candidate_id=candidate_id,
        sweep_config=config,
    )
    paths = write_rank_sweep_report(Path(summary['sweep_root']))
    selection = str(summary.get('selection_status') or '')
    result: dict[str, Any] = {
        'candidate_id': candidate_id,
        'package_root': str(package_root),
        'sweep_root': summary.get('sweep_root'),
        'selection_status': selection,
        'selected_rank': summary.get('selected_rank'),
        'selection_reasons': summary.get('selection_reasons'),
        'paths': paths,
        'cutoff_dims': cutoff_dimension_payload(m3_config.j2_max),
        'status': 'EXPLORATORY_NOT_CERTIFIED',
        'generated_at': utc_now(),
    }
    if selection == SELECTION_SELECTED:
        advance = write_advance(
            package_root,
            selected_rank=int(summary['selected_rank']),
            sweep_root=summary['sweep_root'],
            selection_reasons=list(summary.get('selection_reasons') or []),
        )
        result['series_status'] = 'SELECTED'
        result['advance'] = advance
        result['next_notebook'] = '78_m3_rigorous_rank_candidate.ipynb'
        return result
    if selection in {SELECTION_NO_SELECTION, SELECTION_REJECT}:
        archive = archive_from_sweep(
            package_root,
            Path(summary['sweep_root']),
            selection_status=selection,
            selection_reasons=list(summary.get('selection_reasons') or []),
        )
        result['series_status'] = 'ARCHIVED'
        result['archive'] = archive
        result['next_notebook'] = '82_campaign_c_candidate_queue.ipynb'
        return result
    raise S0SeriesError(f'Unexpected selection_status={selection!r}')


def run_s0_series(
    *,
    project_root: Path,
    persistent_root: Path,
    search_root: Path,
    max_candidates: int = 1,
    sweep_config: dict[str, Any] | None = None,
    materialize_if_missing: bool = True,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
) -> dict[str, Any]:
    """Process up to max_candidates actionable packages.

    - NEED_M2: stop immediately (do not skip silently).
    - ARCHIVED: continue to next until budget exhausted.
    - SELECTED: stop for notebook 78.
    """
    if max_candidates < 1:
        raise S0SeriesError('max_candidates must be >= 1.')
    events: list[dict[str, Any]] = []
    processed = 0
    while processed < max_candidates:
        nxt = next_actionable_candidate(
            search_root,
            persistent_root=persistent_root,
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        if nxt is None:
            append_series_log(search_root, {
                'event': 'SERIES_EXHAUSTED',
                'processed': processed,
            })
            return {
                'series_status': 'EXHAUSTED',
                'processed': processed,
                'events': events,
                'queue': list_queue_rows(
                    search_root,
                    persistent_root=persistent_root,
                    max_executable_j2_max=max_executable_j2_max,
                    max_staged_j2_max=max_staged_j2_max,
                ),
                'generated_at': utc_now(),
            }
        candidate_id = str(nxt['candidate_id'])
        if materialize_if_missing:
            ensure_materialized(
                search_root,
                nxt,
                max_executable_j2_max=max_executable_j2_max,
                max_staged_j2_max=max_staged_j2_max,
            )
        package_root = package_root_for(search_root, candidate_id)
        outcome = run_s0_for_package(
            project_root=project_root,
            persistent_root=persistent_root,
            package_root=package_root,
            candidate_id=candidate_id,
            sweep_config=sweep_config,
        )
        events.append(outcome)
        processed += 1
        series_status = outcome.get('series_status') or outcome.get('status')
        if outcome.get('status') == 'NEED_M2':
            append_series_log(search_root, {
                'event': 'NEED_M2',
                'candidate_id': candidate_id,
            })
            return {
                'series_status': 'NEED_M2',
                'processed': processed,
                'candidate_id': candidate_id,
                'package_root': str(package_root),
                'events': events,
                'next_notebook': '73_m7_staged_s3_lineage.ipynb',
                'env_hint': {
                    'VALIDATED_RG_STAGED_CANDIDATE': candidate_id,
                    'VALIDATED_RG_STAGED_PACKAGE': str(package_root),
                },
                'generated_at': utc_now(),
            }
        if series_status == 'SELECTED':
            append_series_log(search_root, {
                'event': 'SERIES_SELECTED',
                'candidate_id': candidate_id,
                'selected_rank': outcome.get('selected_rank'),
            })
            return {
                'series_status': 'SELECTED',
                'processed': processed,
                'candidate_id': candidate_id,
                'selected_rank': outcome.get('selected_rank'),
                'events': events,
                'next_notebook': '78_m3_rigorous_rank_candidate.ipynb',
                'env_hint': {
                    'VALIDATED_RG_STAGED_CANDIDATE': candidate_id,
                    'VALIDATED_RG_STAGED_PACKAGE': str(package_root),
                },
                'note': 'PR-2 not implemented; do not claim residual/gap certificates.',
                'generated_at': utc_now(),
            }
        # ARCHIVED → continue
    append_series_log(search_root, {
        'event': 'SERIES_BUDGET_DONE',
        'processed': processed,
    })
    return {
        'series_status': 'BUDGET_DONE',
        'processed': processed,
        'events': events,
        'next_notebook': '82_campaign_c_candidate_queue.ipynb',
        'generated_at': utc_now(),
    }
