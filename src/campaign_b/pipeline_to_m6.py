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


def _cheap_queue_presence(
    persistent_root: Path,
    *,
    only_campaign_run_id: str | None,
    skip_m3: bool,
    skip_pre_m6: bool,
    skip_obligations: bool,
    skip_m6: bool,
) -> dict[str, int]:
    """Cheap presence check (max_candidates=1) for remaining runnable work."""
    from .close_obligations import list_obligation_queue
    from .gpu_m3_batch import list_gpu_m3_queue
    from .m6_batch import list_m6_queue
    from .pre_m6_batch import list_pre_m6_queue

    out: dict[str, int] = {
        'gpu_m3': 0,
        'pre_m6': 0,
        'obligations': 0,
        'm6': 0,
    }
    if not skip_m3:
        out['gpu_m3'] = len(list_gpu_m3_queue(
            persistent_root,
            max_candidates=1,
            only_campaign_run_id=only_campaign_run_id,
        ))
    if not skip_pre_m6:
        out['pre_m6'] = len(list_pre_m6_queue(
            persistent_root,
            max_candidates=1,
            only_campaign_run_id=only_campaign_run_id,
        ))
    if not skip_obligations:
        out['obligations'] = len(list_obligation_queue(
            persistent_root,
            max_candidates=1,
            only_campaign_run_id=only_campaign_run_id,
        ))
    if not skip_m6:
        out['m6'] = len(list_m6_queue(
            persistent_root,
            max_candidates=1,
            only_campaign_run_id=only_campaign_run_id,
        ))
    return out


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
    auto_strip_m3_checkpoints: bool = True,
    persist_m3_cap_gib: float | None = 80.0,
    auto_keep_latest_m3_checkpoint: bool = True,
    max_idle_rounds: int = 2,
) -> dict[str, Any]:
    """Run stages 90→91→92→93→94 in order for up to max_rounds passes.

    Stop policy (``stop_reason`` in summary):
    - ``DRAINED_OR_IDLE``: progress==0 and no remaining runnable queues.
    - ``NO_ATTEMPTS_WITH_BACKLOG``: progress==0, runnable remains, but this
      round made no M3/pre_m6 attempts (misconfig / CUDA / empty session).
    - ``STUCK_BACKLOG``: progress==0 with runnable + attempts for
      ``max_idle_rounds`` consecutive idle rounds (all attempts failed).
    - ``MAX_ROUNDS``: hit ``max_rounds`` without an idle stop.
    Do **not** stop solely on progress==0 while runnable work remains and
    this round made attempts — retry up to ``max_idle_rounds`` (default 2).

    Never calls production gate 81. Summary certification_status is always
    NOT_CERTIFIED / SCREENING_ONLY.

    When ``auto_strip_m3_checkpoints`` is True (default for notebook 97):
    - once at session start: full fail-closed strip of COMPLETE+downstream
      M3 runs (so backlog ~GiB reclaim happens even if this session makes
      no new PRE_M6_READY);
    - after each round: incremental strip from this round's pre_m6/m6
      results (full-scan fallback when no preferred ids).

    ``persist_m3_cap_gib`` (default 80.0; None disables): after strip,
    enforce a ``runs/M3-*`` size cap by stripping oldest eligible runs.

    ``auto_keep_latest_m3_checkpoint`` (default True):
    - once at session start: full keep-latest over all ``runs/M3-*`` so older
      ckpts on M3s not touched this session are reclaimed (~GiB backlog);
    - during M3 sessions: per-run keep-latest so mid-flight runs do not pile
      ckpt_000001…N again.
    """
    from .advance_selected import run_advance_selected
    from .close_obligations import run_close_obligations_batch
    from .execution_keys import gpu_lane_lease, refresh_gpu_lane_heartbeat
    from .gpu_m3_batch import run_gpu_m3_batch
    from .m3_reclaim import (
        auto_strip_after_pipeline_round,
        fmt_bytes,
        keep_latest_all_m3_runs,
    )
    from .m6_batch import run_m6_batch
    from .pre_m6_batch import run_pre_m6_batch

    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    started = utc_now()
    rounds: list[dict[str, Any]] = []
    reclaim_totals = {
        'stripped': 0,
        'bytes_freed': 0,
        'rounds_with_reclaim': 0,
        'session_start_full_scan': None,
        'session_start_keep_latest': None,
        'keep_latest_bytes_freed': 0,
    }
    stop_reason = 'MAX_ROUNDS'
    idle_rounds = 0
    last_remaining: dict[str, int] = {
        'gpu_m3': 0, 'pre_m6': 0, 'obligations': 0, 'm6': 0,
    }
    max_idle = max(1, int(max_idle_rounds))

    with gpu_lane_lease(persistent_root, owner='pipeline_to_m6'):
        # Full keep-latest once per session across all M3 runs (not only the
        # ones this batch will touch) — clears accumulated older ckpts.
        if auto_keep_latest_m3_checkpoint:
            session_kl = keep_latest_all_m3_runs(
                persistent_root, execute=True,
            ).as_dict()
            reclaim_totals['session_start_keep_latest'] = session_kl
            reclaim_totals['keep_latest_bytes_freed'] += int(
                session_kl.get('bytes_freed') or 0,
            )

        # Always reclaim already-eligible COMPLETE+downstream backlog once
        # per session — do not wait for this round to produce PRE_M6_READY.
        if auto_strip_m3_checkpoints:
            session_start = auto_strip_after_pipeline_round(
                persistent_root,
                pre_m6_summary=None,
                m6_summary=None,
                execute=True,
                persist_m3_cap_gib=persist_m3_cap_gib,
                force_full_scan=True,
            )
            reclaim_totals['session_start_full_scan'] = session_start
            reclaim_totals['stripped'] += int(session_start.get('stripped') or 0)
            reclaim_totals['bytes_freed'] += int(session_start.get('bytes_freed') or 0)
            if int(session_start.get('stripped') or 0) > 0:
                reclaim_totals['rounds_with_reclaim'] += 1

        for round_index in range(1, int(max_rounds) + 1):
            # Keep foreign-host stale threshold from reclaiming a live job.
            refresh_gpu_lane_heartbeat(persistent_root)
            round_doc: dict[str, Any] = {
                'round': round_index,
                'started_at': utc_now(),
                'stages': {},
            }
            progress = 0
            pre: dict[str, Any] | None = None
            m6: dict[str, Any] | None = None
            sessions_attempted = 0
            packages_attempted = 0

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
                    auto_keep_latest_m3_checkpoint=auto_keep_latest_m3_checkpoint,
                )
                sessions_attempted = int(m3.get('sessions_attempted') or 0)
                round_doc['stages']['m3'] = {
                    'queue_size': m3.get('queue_size'),
                    'sessions_attempted': sessions_attempted,
                    'sessions_ok': m3.get('sessions_ok'),
                    'm3_complete': m3.get('m3_complete'),
                    'm3_checkpoint': m3.get('m3_checkpoint'),
                    'sessions_error': m3.get('sessions_error'),
                    'errors': m3.get('errors'),
                    'keep_latest_bytes_freed': m3.get('keep_latest_bytes_freed'),
                }
                reclaim_totals['keep_latest_bytes_freed'] += int(
                    m3.get('keep_latest_bytes_freed') or 0,
                )
                # Count completions and checkpoints once (not sessions_ok, which
                # double-counts completions).
                progress += (
                    int(m3.get('m3_complete') or 0) + int(m3.get('m3_checkpoint') or 0)
                )

            if not skip_pre_m6:
                pre = run_pre_m6_batch(
                    persistent_root=persistent_root,
                    project_root=project_root,
                    max_packages=max_pre_m6_packages,
                    max_stage_sessions=max_stage_sessions,
                    max_queue=max_queue,
                    only_campaign_run_id=only_campaign_run_id,
                )
                packages_attempted = int(pre.get('packages_attempted') or 0)
                round_doc['stages']['pre_m6'] = {
                    'queue_size': pre.get('queue_size'),
                    'packages_attempted': packages_attempted,
                    'pre_m6_ready': pre.get('pre_m6_ready'),
                    'm4_checkpoint': pre.get('m4_checkpoint'),
                    'errors': pre.get('errors'),
                }
                # Include m4_checkpoint so multi-session M4 resume continues.
                progress += (
                    int(pre.get('pre_m6_ready') or 0) + int(pre.get('m4_checkpoint') or 0)
                )

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

            if auto_strip_m3_checkpoints:
                reclaim = auto_strip_after_pipeline_round(
                    persistent_root,
                    pre_m6_summary=pre,
                    m6_summary=m6,
                    execute=True,
                    persist_m3_cap_gib=persist_m3_cap_gib,
                    force_full_scan=False,
                )
                round_doc['m3_reclaim'] = reclaim
                reclaim_totals['stripped'] += int(reclaim.get('stripped') or 0)
                reclaim_totals['bytes_freed'] += int(reclaim.get('bytes_freed') or 0)
                if int(reclaim.get('stripped') or 0) > 0:
                    reclaim_totals['rounds_with_reclaim'] += 1

            last_remaining = _cheap_queue_presence(
                persistent_root,
                only_campaign_run_id=only_campaign_run_id,
                skip_m3=skip_m3,
                skip_pre_m6=skip_pre_m6,
                skip_obligations=skip_obligations,
                skip_m6=skip_m6,
            )
            runnable = any(int(v) > 0 for v in last_remaining.values())
            attempts_made = sessions_attempted > 0 or packages_attempted > 0
            round_doc['finished_at'] = utc_now()
            round_doc['progress'] = progress
            round_doc['remaining_runnable'] = dict(last_remaining)
            round_doc['attempts_made'] = attempts_made
            rounds.append(round_doc)

            if progress > 0:
                idle_rounds = 0
                continue

            # progress == 0
            if not runnable:
                stop_reason = 'DRAINED_OR_IDLE'
                break
            if not attempts_made:
                stop_reason = 'NO_ATTEMPTS_WITH_BACKLOG'
                break
            idle_rounds += 1
            if idle_rounds >= max_idle:
                stop_reason = 'STUCK_BACKLOG'
                break
            # Runnable + attempts but all failed — retry until max_idle_rounds.

    reclaim_totals['bytes_freed_human'] = fmt_bytes(int(reclaim_totals['bytes_freed']))
    reclaim_totals['keep_latest_bytes_freed_human'] = fmt_bytes(
        int(reclaim_totals['keep_latest_bytes_freed']),
    )
    stuck_diagnostics: dict[str, Any] | None = None
    if stop_reason == 'STUCK_BACKLOG' and rounds:
        last = rounds[-1]
        stages = last.get('stages') or {}
        m3 = stages.get('m3') or {}
        pre = stages.get('pre_m6') or {}
        obl = stages.get('obligations') or {}
        stuck_diagnostics = {
            'round': last.get('round'),
            'remaining_runnable': dict(last_remaining),
            'sessions_error': m3.get('sessions_error'),
            'sessions_attempted': m3.get('sessions_attempted'),
            'sessions_ok': m3.get('sessions_ok'),
            'm3_errors': (m3.get('errors') or [])[:20],
            'pre_m6_errors': (pre.get('errors') or [])[:10],
            'obligation_errors': (obl.get('errors') or [])[:10],
            'hint': (
                'Inspect campaign_b/_gpu_m3/LATEST_GPU_M3_SESSION.json. '
                'JSON/NaN failures become M3_BLOCKED_NONFINITE and leave the '
                'default queue so fresh READY_FOR_M3 can schedule.'
            ),
        }
    summary = {
        'schema_version': 1,
        'session_id': f"PIPE-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'rounds_run': len(rounds),
        'max_rounds': max_rounds,
        'max_idle_rounds': max_idle,
        'stop_reason': stop_reason,
        'remaining_runnable': dict(last_remaining),
        'stuck_diagnostics': stuck_diagnostics,
        'only_campaign_run_id': only_campaign_run_id,
        'auto_strip_m3_checkpoints': bool(auto_strip_m3_checkpoints),
        'auto_keep_latest_m3_checkpoint': bool(auto_keep_latest_m3_checkpoint),
        'persist_m3_cap_gib': persist_m3_cap_gib,
        'm3_reclaim': reclaim_totals,
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
            'm3_checkpoints_stripped': reclaim_totals['stripped'],
            'm3_reclaim_bytes_freed': reclaim_totals['bytes_freed'],
            'm3_keep_latest_bytes_freed': reclaim_totals['keep_latest_bytes_freed'],
        },
        'rounds': rounds,
        'note': (
            'Pipeline 90→91→92→93→94 over Campaign B SELECTED from 89. '
            'Does not run production paperspace M6 gate 81. '
            f'stop_reason={stop_reason}. '
            'Re-run while 89 continues / after stop to pick up backlog. '
            'Holds exclusive GPU lane lease under campaign_b/_locks/gpu_lane.json. '
            + (
                'Session-start full strip + per-round auto-strip ON '
                f"(stripped={reclaim_totals['stripped']}, "
                f"freed≈{reclaim_totals['bytes_freed_human']}). "
                if auto_strip_m3_checkpoints
                else 'M3 auto-strip disabled for this session. '
            )
            + (
                'Session-start full keep-latest + per-session keep-latest ON '
                f"(freed≈{reclaim_totals['keep_latest_bytes_freed_human']}). "
                if auto_keep_latest_m3_checkpoint
                else 'Keep-latest OFF. '
            )
            + 'NOT_CERTIFIED / SCREENING_ONLY.'
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
        'stop_reason': summary.get('stop_reason'),
        'remaining_runnable': summary.get('remaining_runnable'),
        'totals': summary.get('totals'),
        'certification_status': summary.get('certification_status'),
        'claim_scope': summary.get('claim_scope'),
        'note': summary.get('note'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
