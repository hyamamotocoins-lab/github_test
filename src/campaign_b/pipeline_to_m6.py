"""End-to-end Campaign B pipeline: SELECTED → M3 → M4/M5 → obligations → M6.

Consumes packages produced by notebook 89 mass explore (and any other B SELECTED)
and drives them through stages 90–94 without starting production paperspace gate 81.
May be re-run while 89 continues; each round picks up newly SELECTED work.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .schemas import screening_only_payload


def _ledger_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_pipeline_to_m6'


def run_pipeline_to_m6(
    *,
    persistent_root: Path,
    project_root: Path,
    max_rounds: int = 1,
    max_advance: int | None = None,
    max_m3_sessions: int = 8,
    max_pre_m6_packages: int = 8,
    max_stage_sessions: int = 6,
    max_obligation_packages: int = 8,
    max_m6_packages: int = 8,
    max_queue: int = 500,
    only_campaign_run_id: str | None = None,
    skip_advance: bool = False,
    skip_m3: bool = False,
    skip_pre_m6: bool = False,
    skip_obligations: bool = False,
    skip_m6: bool = False,
) -> dict[str, Any]:
    """Run stages 90→91→92→93→94 in order for up to max_rounds passes.

    Stops early when a full pass makes no forward progress (so concurrent
    notebook 89 SELECTED can be drained by raising max_rounds, without
    spinning forever on a stuck queue). Never calls production gate 81.
    Summary certification_status is always NOT_CERTIFIED / SCREENING_ONLY.
    """
    from .advance_selected import run_advance_selected
    from .close_obligations import run_close_obligations_batch
    from .gpu_m3_batch import run_gpu_m3_batch
    from .m6_batch import run_m6_batch
    from .pre_m6_batch import run_pre_m6_batch

    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    started = utc_now()
    rounds: list[dict[str, Any]] = []

    for round_index in range(1, int(max_rounds) + 1):
        round_doc: dict[str, Any] = {
            'round': round_index,
            'started_at': utc_now(),
            'stages': {},
        }
        progress = 0

        if not skip_advance:
            adv = run_advance_selected(
                persistent_root=persistent_root,
                max_candidates=max_advance,
                force=False,
                only_campaign_run_id=only_campaign_run_id,
            )
            round_doc['stages']['advance'] = {
                'discovered': adv.get('discovered'),
                'advanced': adv.get('advanced'),
                'ready_for_m3': adv.get('ready_for_m3'),
                'errors': adv.get('errors'),
            }
            progress += int(adv.get('advanced') or 0)

        if not skip_m3:
            m3 = run_gpu_m3_batch(
                persistent_root=persistent_root,
                project_root=project_root,
                max_sessions=max_m3_sessions,
                max_queue=max_queue,
                only_campaign_run_id=only_campaign_run_id,
            )
            round_doc['stages']['m3'] = {
                'queue_size': m3.get('queue_size'),
                'sessions_ok': m3.get('sessions_ok'),
                'm3_complete': m3.get('m3_complete'),
                'm3_checkpoint': m3.get('m3_checkpoint'),
                'sessions_error': m3.get('sessions_error'),
                'errors': m3.get('errors'),
            }
            # Count completions and checkpoints once (not sessions_ok, which
            # double-counts completions).
            progress += int(m3.get('m3_complete') or 0) + int(m3.get('m3_checkpoint') or 0)

        if not skip_pre_m6:
            pre = run_pre_m6_batch(
                persistent_root=persistent_root,
                project_root=project_root,
                max_packages=max_pre_m6_packages,
                max_stage_sessions=max_stage_sessions,
                max_queue=max_queue,
                only_campaign_run_id=only_campaign_run_id,
            )
            round_doc['stages']['pre_m6'] = {
                'queue_size': pre.get('queue_size'),
                'packages_attempted': pre.get('packages_attempted'),
                'pre_m6_ready': pre.get('pre_m6_ready'),
                'm4_checkpoint': pre.get('m4_checkpoint'),
                'errors': pre.get('errors'),
            }
            # Include m4_checkpoint so multi-session M4 resume continues.
            progress += int(pre.get('pre_m6_ready') or 0) + int(pre.get('m4_checkpoint') or 0)

        if not skip_obligations:
            obl = run_close_obligations_batch(
                persistent_root=persistent_root,
                project_root=project_root,
                max_packages=max_obligation_packages,
                max_queue=max_queue,
                only_campaign_run_id=only_campaign_run_id,
            )
            round_doc['stages']['obligations'] = {
                'queue_size': obl.get('queue_size'),
                'attempted': obl.get('attempted'),
                'all_closed_count': obl.get('all_closed_count'),
                'm5_complete_count': obl.get('m5_complete_count'),
                'still_open': obl.get('still_open'),
                'errors': obl.get('errors'),
            }
            progress += int(obl.get('all_closed_count') or 0)

        if not skip_m6:
            m6 = run_m6_batch(
                persistent_root=persistent_root,
                project_root=project_root,
                max_packages=max_m6_packages,
                max_queue=max_queue,
                only_campaign_run_id=only_campaign_run_id,
            )
            round_doc['stages']['m6'] = {
                'queue_size': m6.get('queue_size'),
                'attempted': m6.get('attempted'),
                'm6_complete': m6.get('m6_complete'),
                'm6_certified_count': m6.get('m6_certified_count'),
                'm6_not_certified_count': m6.get('m6_not_certified_count'),
                'errors': m6.get('errors'),
                'results': m6.get('results'),
            }
            progress += int(m6.get('m6_complete') or 0)

        round_doc['finished_at'] = utc_now()
        round_doc['progress'] = progress
        rounds.append(round_doc)
        if progress == 0:
            break

    summary = {
        'schema_version': 1,
        'session_id': f"PIPE-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'rounds_run': len(rounds),
        'max_rounds': max_rounds,
        'only_campaign_run_id': only_campaign_run_id,
        'totals': {
            'advanced': sum(
                int((r.get('stages') or {}).get('advance', {}).get('advanced') or 0)
                for r in rounds
            ),
            'm3_complete': sum(
                int((r.get('stages') or {}).get('m3', {}).get('m3_complete') or 0)
                for r in rounds
            ),
            'pre_m6_ready': sum(
                int((r.get('stages') or {}).get('pre_m6', {}).get('pre_m6_ready') or 0)
                for r in rounds
            ),
            'obligations_closed': sum(
                int(
                    (r.get('stages') or {}).get('obligations', {}).get('all_closed_count')
                    or 0
                )
                for r in rounds
            ),
            'm6_complete': sum(
                int((r.get('stages') or {}).get('m6', {}).get('m6_complete') or 0)
                for r in rounds
            ),
            'm6_certified': sum(
                int(
                    (r.get('stages') or {}).get('m6', {}).get('m6_certified_count')
                    or 0
                )
                for r in rounds
            ),
            'm6_not_certified': sum(
                int(
                    (r.get('stages') or {}).get('m6', {}).get('m6_not_certified_count')
                    or 0
                )
                for r in rounds
            ),
        },
        'rounds': rounds,
        'note': (
            'Pipeline 90→91→92→93→94 over Campaign B SELECTED from 89. '
            'Does not run production paperspace M6 gate 81. '
            'Re-run while 89 continues to pick up new SELECTED. '
            'NOT_CERTIFIED / SCREENING_ONLY.'
        ),
        **screening_only_payload(),
    }
    root = _ledger_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_PIPELINE_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description='Campaign B pipeline SELECTED → M6 (stages 90–94)',
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', '.'),
    )
    parser.add_argument('--max-rounds', type=int, default=1)
    parser.add_argument('--max-advance', type=int, default=None)
    parser.add_argument('--max-m3-sessions', type=int, default=8)
    parser.add_argument('--max-pre-m6-packages', type=int, default=8)
    parser.add_argument('--max-stage-sessions', type=int, default=6)
    parser.add_argument('--max-obligation-packages', type=int, default=8)
    parser.add_argument('--max-m6-packages', type=int, default=8)
    parser.add_argument('--max-queue', type=int, default=500)
    parser.add_argument('--campaign-run-id', default=None)
    args = parser.parse_args(argv)
    summary = run_pipeline_to_m6(
        persistent_root=Path(args.persistent_root),
        project_root=Path(args.project_root).resolve(),
        max_rounds=args.max_rounds,
        max_advance=args.max_advance,
        max_m3_sessions=args.max_m3_sessions,
        max_pre_m6_packages=args.max_pre_m6_packages,
        max_stage_sessions=args.max_stage_sessions,
        max_obligation_packages=args.max_obligation_packages,
        max_m6_packages=args.max_m6_packages,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
    )
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'rounds_run': summary.get('rounds_run'),
        'totals': summary.get('totals'),
        'certification_status': summary.get('certification_status'),
        'claim_scope': summary.get('claim_scope'),
        'note': summary.get('note'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
