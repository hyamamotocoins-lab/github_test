"""Backlog-aware Campaign B end-to-end scheduler (notebook 96, Phase 1).

Replaces the 89/95 dual loop with a single loop:
  1. M3 + downstream first
  2. screen + advance only when GPU M3 queue length < selected_backlog_target
Progress counts completions only (not m3_checkpoint alone).
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .schemas import screening_only_payload

_DISABLE_WALLCLOCK_ENV = 'VALIDATED_RG_DISABLE_SESSION_WALLCLOCK'


@dataclass(slots=True)
class EndToEndConfig:
    """Phase-1 end-to-end knobs (YAML + overrides)."""

    persistent_root: Path
    project_root: Path
    selected_backlog_target: int = 8
    screening_chunk_size: int = 32
    max_rounds: int = 100
    max_m3_sessions: int = 8
    max_pre_m6_packages: int = 8
    max_stage_sessions: int = 6
    max_obligation_packages: int = 8
    max_m6_packages: int = 8
    max_queue: int = 500
    max_advance: int | None = None
    only_campaign_run_id: str | None = None
    mass_explore_config: str = 'campaign_b_mass_explore.yaml'
    m3_backend: str = 'legacy_rsvd'  # ignored until Phase 2
    disable_session_wallclock: bool = True
    skip_screening: bool = False
    skip_advance: bool = False
    skip_m3: bool = False
    skip_pre_m6: bool = False
    skip_obligations: bool = False
    skip_m6: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


def _ledger_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_end_to_end'


def load_end_to_end_config(
    config_path: Path | str | None = None,
    *,
    persistent_root: Path | None = None,
    project_root: Path | None = None,
    **overrides: Any,
) -> EndToEndConfig:
    """Load YAML (optional) and apply overrides."""
    raw: dict[str, Any] = {}
    project = Path(project_root or overrides.pop('project_root', '.') or '.').resolve()
    if config_path is not None:
        import yaml

        path = Path(config_path)
        if not path.is_file():
            # Allow bare filename relative to project configs/
            alt = project / 'configs' / path.name
            path = alt if alt.is_file() else path
        text = Path(path).read_text(encoding='utf-8')
        loaded = yaml.safe_load(text) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f'invalid end-to-end config: {path}')
        raw = loaded

    persist = Path(
        persistent_root
        or raw.get('persistent_root')
        or os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg')
    )
    if project_root is None and raw.get('project_root'):
        project = Path(str(raw['project_root'])).resolve()

    def _get(key: str, default: Any) -> Any:
        if key in overrides and overrides[key] is not None:
            return overrides[key]
        return raw.get(key, default)

    return EndToEndConfig(
        persistent_root=persist,
        project_root=project,
        selected_backlog_target=int(_get('selected_backlog_target', 8)),
        screening_chunk_size=int(_get('screening_chunk_size', 32)),
        max_rounds=int(_get('max_rounds', 100)),
        max_m3_sessions=int(_get('max_m3_sessions', 8)),
        max_pre_m6_packages=int(_get('max_pre_m6_packages', 8)),
        max_stage_sessions=int(_get('max_stage_sessions', 6)),
        max_obligation_packages=int(_get('max_obligation_packages', 8)),
        max_m6_packages=int(_get('max_m6_packages', 8)),
        max_queue=int(_get('max_queue', 500)),
        max_advance=_get('max_advance', None),
        only_campaign_run_id=_get('only_campaign_run_id', None),
        mass_explore_config=str(_get('mass_explore_config', 'campaign_b_mass_explore.yaml')),
        m3_backend=str(_get('m3_backend', 'legacy_rsvd')),
        disable_session_wallclock=bool(_get('disable_session_wallclock', True)),
        skip_screening=bool(_get('skip_screening', False)),
        skip_advance=bool(_get('skip_advance', False)),
        skip_m3=bool(_get('skip_m3', False)),
        skip_pre_m6=bool(_get('skip_pre_m6', False)),
        skip_obligations=bool(_get('skip_obligations', False)),
        skip_m6=bool(_get('skip_m6', False)),
        extra={k: v for k, v in raw.items() if k not in {
            'persistent_root', 'project_root', 'selected_backlog_target',
            'screening_chunk_size', 'max_rounds', 'max_m3_sessions',
            'max_pre_m6_packages', 'max_stage_sessions', 'max_obligation_packages',
            'max_m6_packages', 'max_queue', 'max_advance', 'only_campaign_run_id',
            'mass_explore_config', 'm3_backend', 'disable_session_wallclock',
            'skip_screening', 'skip_advance', 'skip_m3', 'skip_pre_m6',
            'skip_obligations', 'skip_m6', 'schema_version', 'campaign',
        }},
    )


def _write_screening_chunk_config(cfg: EndToEndConfig) -> Path:
    """Write a one-wave mass-explore YAML with limited candidates."""
    project = cfg.project_root
    base_name = cfg.mass_explore_config
    base = project / 'configs' / base_name
    if not base.is_file():
        base = Path(base_name)
    if not base.is_file():
        raise FileNotFoundError(f'mass explore config not found: {base_name}')

    runtime = _ledger_root(cfg.persistent_root) / 'runtime'
    runtime.mkdir(parents=True, exist_ok=True)
    # Space YAMLs next to runtime config for mass_explore resolution.
    for name in ('campaign_b_s2_space_v1.yaml', 'campaign_b_s2_space_expanded_v1.yaml'):
        src = project / 'configs' / name
        if src.is_file():
            shutil.copy2(src, runtime / name)

    chunk = int(cfg.screening_chunk_size)
    lines: list[str] = []
    skipping_mass = False
    for line in base.read_text(encoding='utf-8').splitlines():
        if line.startswith('mass_explore:'):
            skipping_mass = True
            continue
        if skipping_mass:
            if line.strip() and not line.startswith((' ', '\t')):
                skipping_mass = False
            else:
                continue
        if line.startswith('persistent_root:'):
            lines.append(f'persistent_root: {cfg.persistent_root}')
        elif line.startswith('candidate_limit:'):
            lines.append(f'candidate_limit: {chunk}')
            continue
        else:
            lines.append(line)
    # Force single limited wave (replace mass_explore block).
    if not any(line.startswith('candidate_limit:') for line in lines):
        lines.append(f'candidate_limit: {chunk}')
    lines.append('mass_explore:')
    lines.append('  max_waves: 1')
    lines.append(f'  candidates_per_wave: {chunk}')
    lines.append('  skip_seen_schemes: true')
    lines.append('  space_paths:')
    lines.append('    - campaign_b_s2_space_v1.yaml')
    lines.append('    - campaign_b_s2_space_expanded_v1.yaml')

    out = runtime / 'screening_chunk.yaml'
    out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return out


def _run_screening_chunk(cfg: EndToEndConfig) -> dict[str, Any]:
    from .mass_explore import run_mass_explore

    chunk_cfg = _write_screening_chunk_config(cfg)
    return run_mass_explore(chunk_cfg)


def _m3_queue_len(cfg: EndToEndConfig) -> int:
    from .gpu_m3_batch import list_gpu_m3_queue

    queue = list_gpu_m3_queue(
        cfg.persistent_root,
        max_candidates=cfg.max_queue,
        only_campaign_run_id=cfg.only_campaign_run_id,
    )
    return len(queue)


def run_end_to_end(
    config: EndToEndConfig | Path | str | None = None,
    **overrides: Any,
) -> dict[str, Any]:
    """Run backlog-aware end-to-end rounds until idle or max_rounds."""
    if isinstance(config, EndToEndConfig):
        cfg = config
        for key, value in overrides.items():
            if hasattr(cfg, key) and value is not None:
                setattr(cfg, key, value)
    else:
        cfg = load_end_to_end_config(config, **overrides)

    if cfg.disable_session_wallclock:
        os.environ[_DISABLE_WALLCLOCK_ENV] = '1'

    from .advance_selected import run_advance_selected
    from .close_obligations import run_close_obligations_batch
    from .execution_keys import acquire_gpu_lock, release_gpu_lock
    from .gpu_m3_batch import run_gpu_m3_batch
    from .m6_batch import run_m6_batch
    from .pipeline_recovery import recover_interrupted_work
    from .pre_m6_batch import run_pre_m6_batch

    started = utc_now()
    recovery = recover_interrupted_work(cfg.persistent_root)
    acquire_gpu_lock(cfg.persistent_root, owner='notebook_96_end_to_end')
    rounds: list[dict[str, Any]] = []

    try:
        for round_index in range(1, int(cfg.max_rounds) + 1):
            round_doc: dict[str, Any] = {
                'round': round_index,
                'started_at': utc_now(),
                'stages': {},
                'm3_backend': cfg.m3_backend,  # recorded only; unused in Phase 1
            }
            progress = 0

            # --- 1) M3 / downstream first ---
            if not cfg.skip_m3:
                m3 = run_gpu_m3_batch(
                    persistent_root=cfg.persistent_root,
                    project_root=cfg.project_root,
                    max_sessions=cfg.max_m3_sessions,
                    max_queue=cfg.max_queue,
                    only_campaign_run_id=cfg.only_campaign_run_id,
                )
                round_doc['stages']['m3'] = {
                    'queue_size': m3.get('queue_size'),
                    'm3_complete': m3.get('m3_complete'),
                    'm3_checkpoint': m3.get('m3_checkpoint'),
                    'sessions_error': m3.get('sessions_error'),
                    'errors': m3.get('errors'),
                }
                # Completions only — do NOT count m3_checkpoint alone.
                progress += int(m3.get('m3_complete') or 0)

            if not cfg.skip_pre_m6:
                pre = run_pre_m6_batch(
                    persistent_root=cfg.persistent_root,
                    project_root=cfg.project_root,
                    max_packages=cfg.max_pre_m6_packages,
                    max_stage_sessions=cfg.max_stage_sessions,
                    max_queue=cfg.max_queue,
                    only_campaign_run_id=cfg.only_campaign_run_id,
                )
                round_doc['stages']['pre_m6'] = {
                    'queue_size': pre.get('queue_size'),
                    'pre_m6_ready': pre.get('pre_m6_ready'),
                    'm4_checkpoint': pre.get('m4_checkpoint'),
                    'errors': pre.get('errors'),
                }
                progress += int(pre.get('pre_m6_ready') or 0)

            if not cfg.skip_obligations:
                obl = run_close_obligations_batch(
                    persistent_root=cfg.persistent_root,
                    project_root=cfg.project_root,
                    max_packages=cfg.max_obligation_packages,
                    max_queue=cfg.max_queue,
                    only_campaign_run_id=cfg.only_campaign_run_id,
                )
                round_doc['stages']['obligations'] = {
                    'queue_size': obl.get('queue_size'),
                    'all_closed_count': obl.get('all_closed_count'),
                    'still_open': obl.get('still_open'),
                    'errors': obl.get('errors'),
                }
                progress += int(obl.get('all_closed_count') or 0)

            if not cfg.skip_m6:
                m6 = run_m6_batch(
                    persistent_root=cfg.persistent_root,
                    project_root=cfg.project_root,
                    max_packages=cfg.max_m6_packages,
                    max_queue=cfg.max_queue,
                    only_campaign_run_id=cfg.only_campaign_run_id,
                )
                round_doc['stages']['m6'] = {
                    'queue_size': m6.get('queue_size'),
                    'm6_complete': m6.get('m6_complete'),
                    'm6_certified_count': m6.get('m6_certified_count'),
                    'm6_not_certified_count': m6.get('m6_not_certified_count'),
                    'errors': m6.get('errors'),
                }
                progress += int(m6.get('m6_complete') or 0)

            # --- 2) screen + advance only if backlog thin ---
            queue_len = _m3_queue_len(cfg)
            round_doc['m3_queue_len'] = queue_len
            round_doc['backlog_gate_open'] = queue_len < int(cfg.selected_backlog_target)

            if round_doc['backlog_gate_open']:
                if not cfg.skip_screening:
                    screen = _run_screening_chunk(cfg)
                    selected = int(screen.get('selected_total') or 0)
                    round_doc['stages']['screening'] = {
                        'session_id': screen.get('session_id'),
                        'selected_total': selected,
                        'archived_total': screen.get('archived_total'),
                        'waves': len(screen.get('waves') or []),
                    }
                    progress += selected
                if not cfg.skip_advance:
                    adv = run_advance_selected(
                        persistent_root=cfg.persistent_root,
                        max_candidates=cfg.max_advance,
                        force=False,
                        only_campaign_run_id=cfg.only_campaign_run_id,
                    )
                    round_doc['stages']['advance'] = {
                        'discovered': adv.get('discovered'),
                        'advanced': adv.get('advanced'),
                        'ready_for_m3': adv.get('ready_for_m3'),
                        'errors': adv.get('errors'),
                    }
                    progress += int(adv.get('advanced') or 0)
            else:
                round_doc['stages']['screening'] = {'skipped': True, 'reason': 'backlog_full'}
                round_doc['stages']['advance'] = {'skipped': True, 'reason': 'backlog_full'}

            round_doc['finished_at'] = utc_now()
            round_doc['progress'] = progress
            rounds.append(round_doc)
            if progress == 0:
                break
    finally:
        release_gpu_lock(cfg.persistent_root, owner='notebook_96_end_to_end')

    summary = {
        'schema_version': 1,
        'session_id': f"E2E-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'rounds_run': len(rounds),
        'max_rounds': cfg.max_rounds,
        'selected_backlog_target': cfg.selected_backlog_target,
        'm3_backend': cfg.m3_backend,
        'wallclock_disabled': os.environ.get(_DISABLE_WALLCLOCK_ENV),
        'recovery': {
            'removed_tmp_count': recovery.get('removed_tmp_count'),
        },
        'totals': {
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
            'advanced': sum(
                int((r.get('stages') or {}).get('advance', {}).get('advanced') or 0)
                for r in rounds
            ),
            'selected_from_screening': sum(
                int((r.get('stages') or {}).get('screening', {}).get('selected_total') or 0)
                for r in rounds
            ),
        },
        'rounds': rounds,
        'note': (
            'Notebook 96 Phase-1 backlog-aware scheduler. '
            'Screen+advance only when M3 queue < selected_backlog_target. '
            'Progress excludes m3_checkpoint-only. '
            'Does not run production gate 81. '
            'NOT_CERTIFIED / SCREENING_ONLY.'
        ),
        **screening_only_payload(),
    }
    root = _ledger_root(cfg.persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_END_TO_END_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary
