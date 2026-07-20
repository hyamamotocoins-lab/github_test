"""Multi-wave Campaign B mass exploration — maximize screened candidates.

Runs successive Campaign B waves across versioned S2 search-space YAMLs,
skipping already-seen normalized schemes. Never invents Campaign C params.
All outputs remain NOT_CERTIFIED / SCREENING_ONLY.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from .config import load_campaign_b_config, mint_campaign_run_id
from .driver import run_campaign_b
from .resume_pointer import write_resume_pointer
from .schemas import screening_only_payload


def _mass_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_mass_explore'


def _seen_path(persistent_root: Path) -> Path:
    return _mass_root(persistent_root) / 'seen_normalized_schemes.json'


def load_seen_schemes(persistent_root: Path) -> set[str]:
    path = _seen_path(persistent_root)
    if not path.is_file():
        return set()
    payload = read_json(path)
    if isinstance(payload, dict):
        keys = payload.get('normalized_scheme_keys') or []
        return {str(k) for k in keys}
    if isinstance(payload, list):
        return {str(k) for k in payload}
    return set()


def save_seen_schemes(persistent_root: Path, keys: set[str]) -> None:
    root = _mass_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(_seen_path(persistent_root), {
        'schema_version': 1,
        'updated_at': utc_now(),
        'count': len(keys),
        'normalized_scheme_keys': sorted(keys),
        **screening_only_payload(),
    })


def harvest_seen_from_campaign(campaign_root: Path) -> set[str]:
    """Collect normalized keys from a finished campaign queue."""
    queue_path = Path(campaign_root) / 'queue.json'
    if not queue_path.is_file():
        return set()
    queue = read_json(queue_path)
    if not isinstance(queue, dict):
        return set()
    out: set[str] = set()
    for cand in queue.get('candidates') or []:
        key = cand.get('normalized_scheme_key')
        if key:
            out.add(str(key))
    return out


def _write_wave_config(
    *,
    base_config_path: Path,
    runtime_dir: Path,
    space_path: Path,
    persistent_root: Path,
    wave_index: int,
    candidate_limit: int | None,
) -> Path:
    runtime_dir.mkdir(parents=True, exist_ok=True)
    space_copy = runtime_dir / f'wave_{wave_index:02d}_space.yaml'
    shutil.copy2(space_path, space_copy)
    lines = base_config_path.read_text(encoding='utf-8').splitlines()
    out: list[str] = []
    for line in lines:
        if line.startswith('persistent_root:'):
            out.append(f'persistent_root: {persistent_root}')
        elif line.startswith('search_space_path:'):
            out.append(f'search_space_path: {space_copy.name}')
        elif line.startswith('campaign_run_id:') or line.startswith('resume_campaign_run_id:'):
            continue
        else:
            out.append(line)
    # Force mass-explore defaults
    out.append('never_stop: true')
    out.append('stop_after_first_verified_q_lt_1: false')
    out.append('inherit_deadline: false')
    if candidate_limit is not None:
        out.append(f'candidate_limit: {int(candidate_limit)}')
    run_id = mint_campaign_run_id()
    # Distinct run id per wave
    out.append(f'campaign_run_id: {run_id}')
    cfg_path = runtime_dir / f'wave_{wave_index:02d}_config.yaml'
    cfg_path.write_text('\n'.join(out) + '\n', encoding='utf-8')
    return cfg_path


def run_mass_explore(config_path: Path | str) -> dict[str, Any]:
    """Run multiple Campaign B waves to maximize candidate coverage."""
    base_path = Path(config_path).resolve()
    base_cfg = load_campaign_b_config(base_path)
    persistent_root = Path(base_cfg.persistent_root)
    mass = dict(base_cfg.raw.get('mass_explore') or {})
    max_waves = int(mass.get('max_waves', 4))
    per_wave = mass.get('candidates_per_wave')
    candidate_limit = int(per_wave) if per_wave is not None else base_cfg.candidate_limit
    skip_seen = bool(mass.get('skip_seen_schemes', True))

    space_names = list(mass.get('space_paths') or [base_cfg.search_space_path and base_cfg.search_space_path.name])
    space_names = [str(n) for n in space_names if n]
    if not space_names:
        space_names = ['campaign_b_s2_space_expanded_v1.yaml']

    runtime_dir = _mass_root(persistent_root) / 'runtime'
    runtime_dir.mkdir(parents=True, exist_ok=True)
    seen = load_seen_schemes(persistent_root) if skip_seen else set()

    session_id = f"MASS-{utc_now().replace(':', '').replace('-', '')[:15]}Z"
    session = {
        'schema_version': 1,
        'session_id': session_id,
        'started_at': utc_now(),
        'waves': [],
        'selected_total': 0,
        'archived_total': 0,
        'processed_scheme_keys': 0,
        **screening_only_payload(),
    }
    atomic_write_json(_mass_root(persistent_root) / 'LATEST_MASS_SESSION.json', session)

    # Patch generate to honor exclude set via env file consumed by driver:
    # We inject exclude keys into a side-car that driver reads if present.
    exclude_path = runtime_dir / 'exclude_normalized_keys.json'

    from . import driver as driver_mod
    from .candidate_generator import generate_campaign_b_queue_candidates as _orig_gen

    def _gen_with_exclude(**kwargs: Any) -> list[dict[str, Any]]:
        if skip_seen:
            kwargs = dict(kwargs)
            kwargs['exclude_normalized_keys'] = set(seen)
        return _orig_gen(**kwargs)

    driver_mod.generate_campaign_b_queue_candidates = _gen_with_exclude  # type: ignore[attr-defined]

    try:
        wave_index = 0
        for space_name in space_names:
            if wave_index >= max_waves:
                break
            space_path = (base_path.parent / space_name).resolve()
            if not space_path.is_file():
                session['waves'].append({
                    'wave': wave_index,
                    'space': space_name,
                    'error': f'missing space file: {space_path}',
                })
                continue

            cfg_path = _write_wave_config(
                base_config_path=base_path,
                runtime_dir=runtime_dir,
                space_path=space_path,
                persistent_root=persistent_root,
                wave_index=wave_index,
                candidate_limit=candidate_limit,
            )
            atomic_write_json(exclude_path, {
                'normalized_scheme_keys': sorted(seen),
                'count': len(seen),
            })

            summary = run_campaign_b(cfg_path)
            run_id = str(summary.get('campaign_run_id') or '')
            campaign_root = persistent_root / 'campaign_b' / run_id
            new_keys = harvest_seen_from_campaign(campaign_root)
            before = len(seen)
            seen |= new_keys
            save_seen_schemes(persistent_root, seen)

            wave_rec = {
                'wave': wave_index,
                'space': space_name,
                'campaign_run_id': run_id,
                'terminal_reason': summary.get('terminal_reason'),
                'selected_count': summary.get('selected_count'),
                'archived_count': summary.get('archived_count'),
                'pending_count': summary.get('pending_count'),
                'new_schemes': len(seen) - before,
                'seen_total': len(seen),
                'campaign_root': str(campaign_root),
                **screening_only_payload(),
            }
            session['waves'].append(wave_rec)
            session['selected_total'] += int(summary.get('selected_count') or 0)
            session['archived_total'] += int(summary.get('archived_count') or 0)
            session['processed_scheme_keys'] = len(seen)
            session['updated_at'] = utc_now()
            atomic_write_json(_mass_root(persistent_root) / 'LATEST_MASS_SESSION.json', session)
            write_resume_pointer(
                persistent_root,
                campaign_run_id=run_id or session_id,
                terminal_reason=str(summary.get('terminal_reason')),
                campaign_root=campaign_root if run_id else None,
                extra={'mass_session_id': session_id, 'wave': wave_index},
            )

            # If this wave added no new schemes, skip remaining duplicates of same space
            if skip_seen and (len(seen) - before) == 0 and int(summary.get('selected_count') or 0) == 0:
                # Still advance wave_index; try next space file
                pass
            wave_index += 1

        # Extra waves: re-run expanded space until max_waves if still producing novelty
        expanded = (base_path.parent / 'campaign_b_s2_space_expanded_v1.yaml').resolve()
        while wave_index < max_waves and expanded.is_file():
            before = len(seen)
            cfg_path = _write_wave_config(
                base_config_path=base_path,
                runtime_dir=runtime_dir,
                space_path=expanded,
                persistent_root=persistent_root,
                wave_index=wave_index,
                candidate_limit=candidate_limit,
            )
            summary = run_campaign_b(cfg_path)
            run_id = str(summary.get('campaign_run_id') or '')
            campaign_root = persistent_root / 'campaign_b' / run_id
            new_keys = harvest_seen_from_campaign(campaign_root)
            seen |= new_keys
            save_seen_schemes(persistent_root, seen)
            added = len(seen) - before
            session['waves'].append({
                'wave': wave_index,
                'space': expanded.name,
                'campaign_run_id': run_id,
                'terminal_reason': summary.get('terminal_reason'),
                'selected_count': summary.get('selected_count'),
                'archived_count': summary.get('archived_count'),
                'new_schemes': added,
                'seen_total': len(seen),
                **screening_only_payload(),
            })
            session['selected_total'] += int(summary.get('selected_count') or 0)
            session['archived_total'] += int(summary.get('archived_count') or 0)
            session['processed_scheme_keys'] = len(seen)
            session['updated_at'] = utc_now()
            atomic_write_json(_mass_root(persistent_root) / 'LATEST_MASS_SESSION.json', session)
            wave_index += 1
            if added == 0:
                break
    finally:
        driver_mod.generate_campaign_b_queue_candidates = _orig_gen  # type: ignore[attr-defined]

    session['finished_at'] = utc_now()
    session['status'] = 'MASS_EXPLORE_COMPLETE'
    atomic_write_json(_mass_root(persistent_root) / 'LATEST_MASS_SESSION.json', session)
    atomic_write_json(
        _mass_root(persistent_root) / f'{session_id}_summary.json',
        session,
    )
    return session


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print(
            'usage: python -m src.campaign_b.mass_explore <campaign_b_mass_explore.yaml>',
            file=sys.stderr,
        )
        return 2
    summary = run_mass_explore(Path(args[0]))
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'selected_total': summary.get('selected_total'),
        'archived_total': summary.get('archived_total'),
        'processed_scheme_keys': summary.get('processed_scheme_keys'),
        'waves': len(summary.get('waves') or []),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
