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
) -> dict[str, Any]:
    """Drain M2-ready backlog (default) or run backlog-aware end-to-end.

    Default (``drain_existing_backlog=True``, ``skip_screening=True``):
      Same stage chain as notebook 95 — ``advance → M3 → pre_m6 → obligations → M6``
      over packages that are SELECTED with READY_FOR_M3 / m2_binding READY.
      Does not wait on ``M2_READY.json`` markers (those are informational only).

    When ``drain_existing_backlog=False``: Phase-1 ``run_end_to_end`` loop
    (M3-first, optional screening when backlog thin).

    Single GPU: do not start parallel M3 workers. Do not run concurrently with 96.
    """
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
    )
    if cfg.disable_session_wallclock:
        os.environ[_DISABLE_WALLCLOCK_ENV] = '1'

    m2_ready = find_m2_ready_markers(cfg.persistent_root)

    if cfg.drain_existing_backlog:
        from .pipeline_to_m6 import run_pipeline_to_m6

        # 95-equivalent consumer: advance first, then GPU M3→M6.
        # Screening stays off unless the caller opts into the e2e path.
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
        )
        mode = 'drain_existing_backlog'
        inner_key = 'pipeline_to_m6'
        inner_summary = {
            'session_id': inner.get('session_id'),
            'rounds_run': inner.get('rounds_run'),
            'totals': inner.get('totals'),
        }
        note = (
            'Notebook 97 post-M2 consumer (95-equivalent). '
            'Drains SELECTED / READY_FOR_M3 / m2_binding-READY through '
            'advance → M3 → M6. Screening off by default. '
            'M2_READY markers are informational only (not a wait gate). '
            'Do not run concurrently with notebook 96. '
            'NOT_CERTIFIED / SCREENING_ONLY.'
        )
    else:
        from .end_to_end import EndToEndConfig, run_end_to_end

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
        inner = run_end_to_end(e2e)
        mode = 'end_to_end'
        inner_key = 'end_to_end'
        inner_summary = {
            'session_id': inner.get('session_id'),
            'rounds_run': inner.get('rounds_run'),
            'totals': inner.get('totals'),
        }
        note = (
            'Notebook 97 post-M2 pipeline (parallel-split Lane B–D via end_to_end). '
            'Single GPU. Reuses Phase-1 backlog-aware loop. '
            'Do not run concurrently with notebook 96. '
            'NOT_CERTIFIED / SCREENING_ONLY.'
        )

    summary = {
        'schema_version': 1,
        'session_id': f"P2M-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'notebook': 97,
        'mode': mode,
        'drain_existing_backlog': cfg.drain_existing_backlog,
        'skip_screening': cfg.skip_screening,
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
