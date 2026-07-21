"""Post-M2 Campaign B pipeline (notebook 97) — consumer lane for M2-ready work.

Default mode matches notebook 95: drain packages that already have shared /
canonical M2 (READY_FOR_M3 or m2_binding READY) through advance → M3 → M6.

Optional screening (producer) remains available when skip_screening=False and
drain_existing_backlog=False (Phase-1 end-to-end backlog-aware loop).

WAITING_FOR_M2 reconciler is documented as TODO until driver support.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from .schemas import screening_only_payload

_DISABLE_WALLCLOCK_ENV = 'VALIDATED_RG_DISABLE_SESSION_WALLCLOCK'

# TODO(driver): promote NEED_CANONICAL_M2 → WAITING_FOR_M2 and reconcile when
# M2_READY.json appears. Until then we only *detect* M2 readiness / absence.
WAITING_FOR_M2_TODO = (
    'WAITING_FOR_M2 candidate state + reconciler not yet wired in driver.py; '
    'screening still uses mass_explore auto_generate / NEED_CANONICAL_M2 paths.'
)


@dataclass(slots=True)
class PostM2Config:
    persistent_root: Path
    project_root: Path
    selected_backlog_target: int = 8
    screening_chunk_size: int = 32
    max_rounds: int = 50
    max_m3_sessions: int = 8
    max_pre_m6_packages: int = 8
    max_stage_sessions: int = 6
    max_obligation_packages: int = 8
    max_m6_packages: int = 8
    max_queue: int = 500
    max_advance: int | None = None
    only_campaign_run_id: str | None = None
    gpu_workers: int = 1  # enforced as 1
    disable_session_wallclock: bool = True
    # Defaults: 95-equivalent consumer (no new screening).
    skip_screening: bool = True
    drain_existing_backlog: bool = True
    # After M4+ consumes M3, strip M3 checkpoints (fail-closed criteria).
    auto_strip_m3_checkpoints: bool = True
    # Cap total runs/M3-* size (GiB); None disables. Default matches notebook 97.
    persist_m3_cap_gib: float | None = 80.0
    # During active M3: keep only latest COMMITTED ckpt (quota crisis default ON).
    auto_keep_latest_m3_checkpoint: bool = True


def _ledger_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_post_m2'


def find_m2_ready_markers(persistent_root: Path) -> list[dict[str, Any]]:
    """Scan runs/*/M2_READY.json without mutating anything."""
    runs = Path(persistent_root) / 'runs'
    found: list[dict[str, Any]] = []
    if not runs.is_dir():
        return found
    for run_dir in sorted(runs.iterdir()):
        if not run_dir.is_dir() or not run_dir.name.startswith('M2-'):
            continue
        marker = run_dir / 'M2_READY.json'
        if not marker.is_file():
            continue
        doc = read_json(marker)
        found.append({
            'run_id': run_dir.name,
            'path': str(marker),
            'ready': True if not isinstance(doc, dict) else bool(doc.get('ready', True)),
            'payload_keys': sorted(doc.keys()) if isinstance(doc, dict) else [],
        })
    return found


def run_post_m2_pipeline(
    *,
    persistent_root: Path,
    project_root: Path,
    selected_backlog_target: int = 8,
    screening_chunk_size: int = 32,
    max_rounds: int = 50,
    max_m3_sessions: int = 8,
    max_pre_m6_packages: int = 8,
    max_stage_sessions: int = 6,
    max_obligation_packages: int = 8,
    max_m6_packages: int = 8,
    max_queue: int = 500,
    max_advance: int | None = None,
    only_campaign_run_id: str | None = None,
    skip_screening: bool = True,
    drain_existing_backlog: bool = True,
    disable_session_wallclock: bool = True,
    auto_strip_m3_checkpoints: bool = True,
    persist_m3_cap_gib: float | None = 80.0,
    auto_keep_latest_m3_checkpoint: bool = True,
) -> dict[str, Any]:
    """Drain M2-ready backlog (default) or run backlog-aware end-to-end.

    Default (``drain_existing_backlog=True``, ``skip_screening=True``):
      Same stage chain as notebook 95 — ``advance → M3 → pre_m6 → obligations → M6``
      over packages that are SELECTED with READY_FOR_M3 / m2_binding READY.
      Does not wait on ``M2_READY.json`` markers (those are informational only).

    When ``drain_existing_backlog=False``: Phase-1 ``run_end_to_end`` loop
    (M3-first, optional screening when backlog thin).

    ``auto_strip_m3_checkpoints`` (default True): session-start full strip of
    COMPLETE+downstream M3, plus per-round incremental strip. Ledger records
    stripped count / GiB. See ``docs/campaign_b_m3_storage_reclaim.md``.

    ``persist_m3_cap_gib`` (default 80.0; None disables): after each strip,
    enforce a ``runs/M3-*`` size cap (oldest eligible first).

    ``auto_keep_latest_m3_checkpoint`` (default True): during M3 sessions,
    trim older ``ckpt_*`` so mid-flight runs do not accumulate many checkpoints.

    Recommended ops: notebook 89 (producer) ∥ this consumer. Backlog growth is OK.
    Exclusive GPU lane lease under campaign_b/_locks/gpu_lane.json — do not run
    concurrently with notebook 96 (second consumer fails closed).
    """
    from .execution_keys import gpu_lane_lease

    cfg = PostM2Config(
        persistent_root=Path(persistent_root),
        project_root=Path(project_root),
        selected_backlog_target=int(selected_backlog_target),
        screening_chunk_size=int(screening_chunk_size),
        max_rounds=int(max_rounds),
        max_m3_sessions=int(max_m3_sessions),
        max_pre_m6_packages=int(max_pre_m6_packages),
        max_stage_sessions=int(max_stage_sessions),
        max_obligation_packages=int(max_obligation_packages),
        max_m6_packages=int(max_m6_packages),
        max_queue=int(max_queue),
        max_advance=max_advance,
        only_campaign_run_id=only_campaign_run_id,
        gpu_workers=1,
        disable_session_wallclock=disable_session_wallclock,
        skip_screening=skip_screening,
        drain_existing_backlog=drain_existing_backlog,
        auto_strip_m3_checkpoints=bool(auto_strip_m3_checkpoints),
        persist_m3_cap_gib=persist_m3_cap_gib,
        auto_keep_latest_m3_checkpoint=bool(auto_keep_latest_m3_checkpoint),
    )
    if cfg.disable_session_wallclock:
        os.environ[_DISABLE_WALLCLOCK_ENV] = '1'

    m2_ready = find_m2_ready_markers(cfg.persistent_root)

    with gpu_lane_lease(cfg.persistent_root, owner='notebook_97_post_m2'):
        if cfg.drain_existing_backlog:
            from .pipeline_to_m6 import run_pipeline_to_m6

            # 95-equivalent consumer: advance first, then GPU M3→M6.
            # Nested lease with pipeline_to_m6 is intentional.
            inner = run_pipeline_to_m6(
                persistent_root=cfg.persistent_root,
                project_root=cfg.project_root,
                max_rounds=cfg.max_rounds,
                max_advance=cfg.max_advance,
                max_m3_sessions=cfg.max_m3_sessions,
                max_pre_m6_packages=cfg.max_pre_m6_packages,
                max_stage_sessions=cfg.max_stage_sessions,
                max_obligation_packages=cfg.max_obligation_packages,
                max_m6_packages=cfg.max_m6_packages,
                max_queue=cfg.max_queue,
                only_campaign_run_id=cfg.only_campaign_run_id,
                auto_strip_m3_checkpoints=cfg.auto_strip_m3_checkpoints,
                persist_m3_cap_gib=cfg.persist_m3_cap_gib,
                auto_keep_latest_m3_checkpoint=cfg.auto_keep_latest_m3_checkpoint,
            )
            mode = 'drain_existing_backlog'
            inner_key = 'pipeline_to_m6'
            m3_reclaim = inner.get('m3_reclaim') or {}
            stop_reason = inner.get('stop_reason')
            remaining = inner.get('remaining_runnable') or {}
            inner_summary = {
                'session_id': inner.get('session_id'),
                'rounds_run': inner.get('rounds_run'),
                'stop_reason': stop_reason,
                'remaining_runnable': remaining,
                'totals': inner.get('totals'),
                'auto_strip_m3_checkpoints': inner.get('auto_strip_m3_checkpoints'),
                'auto_keep_latest_m3_checkpoint': inner.get(
                    'auto_keep_latest_m3_checkpoint',
                ),
                'persist_m3_cap_gib': inner.get('persist_m3_cap_gib'),
                'm3_reclaim': m3_reclaim,
            }
            note = (
                'Notebook 97 post-M2 consumer (95-equivalent). '
                'Run alongside notebook 89 (producer); backlog growth is OK. '
                'Drains SELECTED / READY_FOR_M3 / m2_binding-READY through '
                'advance → M3 → M6. Screening off by default. '
                'M2_READY markers are informational only (not a wait gate). '
                f'stop_reason={stop_reason}; re-run cell 3 after stop to resume. '
                'GPU lane lease held; do not run concurrently with notebook 96. '
                + (
                    f"Auto-strip M3 checkpoints ON "
                    f"(stripped={m3_reclaim.get('stripped', 0)}, "
                    f"freed≈{m3_reclaim.get('bytes_freed_human', '0 B')}). "
                    if cfg.auto_strip_m3_checkpoints
                    else 'Auto-strip M3 checkpoints OFF. '
                )
                + (
                    f"Keep-latest ON "
                    f"(freed≈{m3_reclaim.get('keep_latest_bytes_freed_human', '0 B')}). "
                    if cfg.auto_keep_latest_m3_checkpoint
                    else 'Keep-latest OFF. '
                )
                + 'NOT_CERTIFIED / SCREENING_ONLY.'
            )
        else:
            from .end_to_end import EndToEndConfig, run_end_to_end
            from .m3_reclaim import (
                enforce_persist_m3_cap,
                strip_eligible_m3_checkpoints,
            )

            e2e = EndToEndConfig(
                persistent_root=cfg.persistent_root,
                project_root=cfg.project_root,
                selected_backlog_target=cfg.selected_backlog_target,
                screening_chunk_size=cfg.screening_chunk_size,
                max_rounds=cfg.max_rounds,
                max_m3_sessions=cfg.max_m3_sessions,
                max_pre_m6_packages=cfg.max_pre_m6_packages,
                max_stage_sessions=cfg.max_stage_sessions,
                max_obligation_packages=cfg.max_obligation_packages,
                max_m6_packages=cfg.max_m6_packages,
                max_queue=cfg.max_queue,
                max_advance=cfg.max_advance,
                only_campaign_run_id=cfg.only_campaign_run_id,
                skip_screening=cfg.skip_screening,
                disable_session_wallclock=cfg.disable_session_wallclock,
            )
            # Nested lease with end_to_end acquire_gpu_lock is intentional.
            # Session-start full strip before e2e so backlog reclaim is not
            # gated on this loop producing PRE_M6_READY.
            m3_reclaim: dict[str, Any] = {}
            if cfg.auto_strip_m3_checkpoints:
                m3_reclaim = strip_eligible_m3_checkpoints(
                    cfg.persistent_root, execute=True,
                ).as_dict()
                m3_reclaim['force_full_scan'] = True
                if cfg.persist_m3_cap_gib is not None:
                    cap = enforce_persist_m3_cap(
                        cfg.persistent_root,
                        cap_gib=float(cfg.persist_m3_cap_gib),
                        execute=True,
                    )
                    m3_reclaim['persist_cap'] = cap
            inner = run_end_to_end(e2e)
            mode = 'end_to_end'
            inner_key = 'end_to_end'
            inner_summary = {
                'session_id': inner.get('session_id'),
                'rounds_run': inner.get('rounds_run'),
                'totals': inner.get('totals'),
                'm3_reclaim': m3_reclaim,
            }
            note = (
                'Notebook 97 post-M2 pipeline (opt-in end_to_end path). '
                'Prefer 89∥97 producer-consumer; backlog growth is OK. '
                'GPU lane lease held; do not run concurrently with notebook 96. '
                + (
                    f"Auto-strip M3 checkpoints ON "
                    f"(stripped={m3_reclaim.get('stripped', 0)}, "
                    f"freed≈{m3_reclaim.get('bytes_freed_human', '0 B')}). "
                    if cfg.auto_strip_m3_checkpoints
                    else 'Auto-strip M3 checkpoints OFF. '
                )
                + 'NOT_CERTIFIED / SCREENING_ONLY.'
            )

    summary = {
        'schema_version': 1,
        'session_id': f"P2M-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'notebook': 97,
        'mode': mode,
        'drain_existing_backlog': cfg.drain_existing_backlog,
        'skip_screening': cfg.skip_screening,
        'auto_strip_m3_checkpoints': cfg.auto_strip_m3_checkpoints,
        'auto_keep_latest_m3_checkpoint': cfg.auto_keep_latest_m3_checkpoint,
        'persist_m3_cap_gib': cfg.persist_m3_cap_gib,
        'stop_reason': (
            inner.get('stop_reason')
            if isinstance(inner.get('stop_reason'), str)
            else None
        ),
        'remaining_runnable': (
            inner.get('remaining_runnable')
            if isinstance(inner.get('remaining_runnable'), dict)
            else None
        ),
        'm3_reclaim': (
            (inner.get('m3_reclaim') if isinstance(inner.get('m3_reclaim'), dict) else None)
            or (inner_summary.get('m3_reclaim') if isinstance(inner_summary, dict) else None)
            or {}
        ),
        'started_at': inner.get('started_at'),
        'finished_at': utc_now(),
        'gpu_workers': 1,
        'm2_ready_markers': m2_ready,
        'm2_ready_count': len(m2_ready),
        'waiting_for_m2_todo': WAITING_FOR_M2_TODO,
        inner_key: inner_summary,
        'note': note,
        **screening_only_payload(),
    }
    root = _ledger_root(cfg.persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_POST_M2_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary
