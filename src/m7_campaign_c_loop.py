"""Campaign C: queue prepare (82) + S0 series (83) until NEED_CANONICAL_M2.

Does not start GPU M2 or production M6. Stops when the next actionable
candidate requires a new canonical shared M2 (or SELECTED / exhausted).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import utc_now
from .m2_shared_registry import (
    BINDING_NEED,
    BINDING_READY,
    BINDING_WAITING,
    resolve_m2_binding,
)
from .m7_archive import append_series_log
from .m7_candidate_queue import (
    ensure_materialized,
    list_queue_rows,
    materialize_top_k,
    next_actionable_candidate,
    package_root_for,
)
from .m7_promotion import evaluate_promotion, ranking_snapshot_hashes
from .rank_sweep import default_sweep_config
from .s0_series import run_s0_for_package


class CampaignCLoopError(RuntimeError):
    """Raised when the Campaign C queue/S0 loop cannot proceed safely."""


_STOP_BINDING = {BINDING_NEED, BINDING_WAITING}


def prepare_candidate_binding(
    *,
    project_root: Path,
    persistent_root: Path,
    search_root: Path,
    campaign_run_id: str,
    row: dict[str, Any],
    rank_among_executable: int,
    ranking_snapshot_sha256: str,
    candidate_set_sha256: str,
    top_k: int = 3,
) -> dict[str, Any]:
    """Materialize (if needed), promote, and resolve shared M2 binding."""
    package_root = package_root_for(search_root, str(row['candidate_id']))
    ensure_materialized(search_root, row)
    evaluate_promotion(
        package_root=package_root,
        j2_max=int(row['j2_max']),
        estimated_q=float(row['estimated_q']),
        rank_among_executable=rank_among_executable,
        top_k=top_k,
        campaign_run_id=campaign_run_id,
        ranking_snapshot_sha256=ranking_snapshot_sha256,
        candidate_set_sha256=candidate_set_sha256,
    )
    binding = resolve_m2_binding(
        persistent_root=persistent_root,
        project_root=project_root,
        package_root=package_root,
        j2_max=int(row['j2_max']),
    )
    return {
        'candidate_id': row['candidate_id'],
        'package_root': str(package_root),
        'binding': binding,
        'state': binding.get('state'),
        'canonical_run_id': binding.get('canonical_run_id'),
        'lookup_hit': binding.get('lookup_hit'),
        'source_drift': binding.get('source_drift'),
    }


def run_campaign_c_queue_s0_loop(
    *,
    project_root: Path,
    persistent_root: Path,
    search_root: Path,
    campaign_run_id: str,
    max_candidates: int = 32,
    materialize_top_k_n: int = 3,
    promote_top_k: int = 3,
    sweep_config: dict[str, Any] | None = None,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
) -> dict[str, Any]:
    """Run 82-style prepare + 83-style S0 until NEED_CANONICAL_M2 (or terminal).

    Stop conditions:
    - NEED_CANONICAL_M2 / WAITING_FOR_CANONICAL_M2 (go to notebook 73)
    - NEED_M2 from S0 gate
    - SELECTED (exploratory; notebook 78 scaffold)
    - EXHAUSTED (no actionable left)
    - BUDGET_DONE (hit max_candidates without the above)

    Continues across ARCHIVED / NO_SELECTION.
    Never starts production M6.
    """
    if max_candidates < 1:
        raise CampaignCLoopError('max_candidates must be >= 1')

    materialize_top_k(
        search_root,
        persistent_root=persistent_root,
        k=materialize_top_k_n,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )

    events: list[dict[str, Any]] = []
    processed = 0
    config = default_sweep_config(**(sweep_config or {}))

    while processed < max_candidates:
        rows = list_queue_rows(
            search_root,
            persistent_root=persistent_root,
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        snap = ranking_snapshot_hashes([r['ranking_row'] for r in rows])
        live = [
            r for r in rows
            if (r.get('staged_executable') or r.get('instant_executable'))
            and not r.get('archived')
        ]
        nxt = next_actionable_candidate(
            search_root,
            persistent_root=persistent_root,
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        if nxt is None:
            append_series_log(search_root, {
                'event': 'CAMPAIGN_C_LOOP_EXHAUSTED',
                'processed': processed,
            })
            return {
                'series_status': 'EXHAUSTED',
                'processed': processed,
                'events': events,
                'queue': rows,
                'note': 'No actionable candidates left.',
                'generated_at': utc_now(),
            }

        candidate_id = str(nxt['candidate_id'])
        rank_idx = next(
            (i for i, r in enumerate(live) if r['candidate_id'] == candidate_id),
            0,
        )
        prepared = prepare_candidate_binding(
            project_root=project_root,
            persistent_root=persistent_root,
            search_root=search_root,
            campaign_run_id=campaign_run_id,
            row=nxt,
            rank_among_executable=rank_idx,
            ranking_snapshot_sha256=snap['ranking_snapshot_sha256'],
            candidate_set_sha256=snap['candidate_set_sha256'],
            top_k=promote_top_k,
        )
        state = prepared.get('state')
        package_root = Path(prepared['package_root'])
        events.append({'event': 'PREPARE', **prepared})

        if state in _STOP_BINDING:
            append_series_log(search_root, {
                'event': state,
                'candidate_id': candidate_id,
            })
            return {
                'series_status': state,
                'processed': processed,
                'candidate_id': candidate_id,
                'package_root': str(package_root),
                'binding': prepared.get('binding'),
                'events': events,
                'next_notebook': '73_m7_staged_s3_lineage.ipynb',
                'env_hint': {
                    'VALIDATED_RG_STAGED_CANDIDATE': candidate_id,
                    'VALIDATED_RG_STAGED_PACKAGE': str(package_root),
                },
                'note': (
                    'Stop: need canonical shared M2. '
                    'Do not start a second M2 if WAITING_FOR_CANONICAL_M2.'
                    if state == BINDING_WAITING else
                    'Stop: NEED_CANONICAL_M2. Run notebook 73 for this candidate only.'
                ),
                'generated_at': utc_now(),
            }

        if state != BINDING_READY:
            # Unexpected binding; fail closed rather than silent skip.
            return {
                'series_status': 'BINDING_BLOCKED',
                'processed': processed,
                'candidate_id': candidate_id,
                'package_root': str(package_root),
                'binding': prepared.get('binding'),
                'events': events,
                'note': f'Unexpected binding state={state!r}; inspect package.',
                'generated_at': utc_now(),
            }

        outcome = run_s0_for_package(
            project_root=project_root,
            persistent_root=persistent_root,
            package_root=package_root,
            candidate_id=candidate_id,
            sweep_config=config,
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
                'note': 'PR-2 not implemented; do not claim residual/gap certificates. Do NOT start production M6.',
                'generated_at': utc_now(),
            }
        # ARCHIVED / other non-terminal → continue

    append_series_log(search_root, {
        'event': 'CAMPAIGN_C_LOOP_BUDGET_DONE',
        'processed': processed,
    })
    return {
        'series_status': 'BUDGET_DONE',
        'processed': processed,
        'events': events,
        'next_notebook': '85_campaign_c_queue_s0_loop.ipynb',
        'note': 'Hit max_candidates without NEED_CANONICAL_M2; raise VALIDATED_RG_LOOP_MAX or re-run.',
        'generated_at': utc_now(),
    }
