#!/usr/bin/env python3
"""Estimate remaining work for notebooks 89 (mass explore) and 95 (pipeline to M6).

Relies on ledgers / queues / campaign state — not process lists (ipykernel often
invisible to pgrep). Pair with scripts/check_89_95_status.py for RUNNING/STALE.

Paperspace one-liner (from repo root):
  python scripts/check_89_95_remaining.py

Optional:
  VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg \\
    python scripts/check_89_95_remaining.py --repo-root /notebooks/github_test_work
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Known finite B/S2 space cardinalities (validated against candidate_generator).
SPACE_SIZE_KNOWN: dict[str, int] = {
    'campaign_b_s2_space_v1.yaml': 405,
    'campaign_b_s2_space_expanded_v1.yaml': 45360,
}

DEFAULT_MAX_WAVES = 8
EXPANDED_SPACE_NAME = 'campaign_b_s2_space_expanded_v1.yaml'
DEFAULT_SPACE_PATHS: tuple[str, ...] = (
    'campaign_b_s2_space_v1.yaml',
    EXPANDED_SPACE_NAME,
)
ADVANCED_STATUSES = frozenset({
    'LINEAGE_PLANNED',
    'FIXTURE_RESIDUAL_DONE',
    'READY_FOR_M3',
})
STAGE_LEDGERS: tuple[tuple[str, str], ...] = (
    ('advance', 'campaign_b/_advance/LATEST_ADVANCE_SESSION.json'),
    ('gpu_m3', 'campaign_b/_gpu_m3/LATEST_GPU_M3_SESSION.json'),
    ('pre_m6', 'campaign_b/_pre_m6/LATEST_PRE_M6_SESSION.json'),
    ('obligations', 'campaign_b/_obligations/LATEST_OBLIGATION_SESSION.json'),
    ('m6', 'campaign_b/_m6/LATEST_M6_SESSION.json'),
)


def _ensure_repo_on_path(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def _age_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    try:
        return max(0.0, time.time() - path.stat().st_mtime)
    except OSError:
        return None


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return 'n/a'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.2f}h'


def _load_json(path: Path) -> dict[str, Any] | list[Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    if isinstance(raw, (dict, list)):
        return raw
    return None


def _load_json_dict(path: Path) -> dict[str, Any] | None:
    raw = _load_json(path)
    return raw if isinstance(raw, dict) else None


def _print_block(title: str, rows: list[tuple[str, Any]]) -> None:
    print(f'=== {title} ===')
    for key, val in rows:
        print(f'  {key}: {val}')
    print()


def _list_len(raw: Any) -> int | None:
    if isinstance(raw, list):
        return len(raw)
    return None


def _space_size_from_yaml(path: Path) -> int | None:
    """Cartesian product of B/S2 axes (matches generator when no collisions)."""
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore
    except ImportError:
        return None
    try:
        space = yaml.safe_load(path.read_text(encoding='utf-8'))
    except (OSError, UnicodeError, Exception):  # noqa: BLE001 — fail soft
        return None
    if not isinstance(space, dict):
        return None
    rank = list((space.get('rank') or {}).get('values') or [])
    rsvd = space.get('rsvd') or {}
    layers = space.get('layers') or {}
    residual = space.get('residual') or {}
    staging = space.get('staging') or {}
    dims = [
        rank,
        list(rsvd.get('oversampling') or layers.get('oversampling') or []),
        list(rsvd.get('power_iterations') or layers.get('power_iterations') or []),
        list(rsvd.get('seeds') or layers.get('seed') or [20260720]),
        list(layers.get('perron_weight_strategy') or ['all_ones']),
        list(layers.get('coupling_policy') or ['uniform_full']),
        list(residual.get('tolerances') or [None]),
        list(residual.get('norm_models') or [None]),
        list(staging.get('j2_values') or [2]),
    ]
    if not dims[0] or not dims[1] or not dims[2]:
        return None
    total = 1
    for axis in dims:
        total *= max(1, len(axis))
    return int(total)


def resolve_space_size(name: str, configs_dir: Path) -> tuple[int | None, str]:
    known = SPACE_SIZE_KNOWN.get(name)
    computed = _space_size_from_yaml(configs_dir / name)
    if computed is not None:
        return computed, 'yaml_product'
    if known is not None:
        return known, 'known_constant'
    return None, 'unknown'


def _seen_count(payload: dict[str, Any] | list[Any] | None) -> int | None:
    if payload is None:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        if payload.get('count') is not None:
            try:
                return int(payload['count'])
            except (TypeError, ValueError):
                pass
        keys = payload.get('normalized_scheme_keys')
        if isinstance(keys, list):
            return len(keys)
    return None


def _queue_state_counts(queue: dict[str, Any] | None) -> dict[str, int]:
    counts = {
        'pending': 0,
        'reserved': 0,
        'running': 0,
        'selected': 0,
        'archived': 0,
        'other': 0,
        'total': 0,
    }
    if not isinstance(queue, dict):
        return counts
    for cand in queue.get('candidates') or []:
        if not isinstance(cand, dict):
            continue
        counts['total'] += 1
        state = str(cand.get('state') or '').upper()
        if state == 'PENDING':
            counts['pending'] += 1
        elif state == 'RESERVED':
            counts['reserved'] += 1
        elif state in {'RUNNING', 'IN_PROGRESS', 'SCREENING'}:
            counts['running'] += 1
        elif state == 'SELECTED':
            counts['selected'] += 1
        elif state == 'ARCHIVED':
            counts['archived'] += 1
        else:
            counts['other'] += 1
    return counts


def _campaign_counts(campaign_root: Path) -> dict[str, Any]:
    queue = _load_json_dict(campaign_root / 'queue.json')
    ledger = _load_json_dict(campaign_root / 'ledger.json')
    summary = _load_json_dict(campaign_root / 'campaign_summary.json')
    qcounts = _queue_state_counts(queue)
    selected = _list_len((ledger or {}).get('selected')) if ledger else None
    archived = _list_len((ledger or {}).get('archived_ids')) if ledger else None
    if selected is None and summary is not None:
        try:
            selected = int(summary.get('selected_count'))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            selected = None
    if archived is None and summary is not None:
        try:
            archived = int(summary.get('archived_count'))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            archived = None
    pending = qcounts['pending']
    if summary is not None and summary.get('pending_count') is not None:
        try:
            pending = int(summary['pending_count'])
        except (TypeError, ValueError):
            pass
    return {
        'campaign_root': str(campaign_root),
        'campaign_state': (ledger or {}).get('campaign_state') if ledger else None,
        'terminal_reason': (
            (summary or {}).get('terminal_reason')
            or ((ledger or {}).get('terminal_reason') if ledger else None)
        ),
        'queue_pending': pending,
        'queue_reserved': qcounts['reserved'],
        'queue_running': qcounts['running'],
        'queue_total': qcounts['total'],
        'selected': selected if selected is not None else qcounts['selected'],
        'archived': archived if archived is not None else qcounts['archived'],
        'has_queue': queue is not None,
        'has_ledger': ledger is not None,
        'has_summary': summary is not None,
    }


def _newest_m7_campaign(persist: Path) -> Path | None:
    root = persist / 'campaign_b'
    if not root.is_dir():
        return None
    best: Path | None = None
    best_mtime = -1.0
    try:
        entries = list(root.iterdir())
    except OSError:
        return None
    for path in entries:
        if not path.is_dir() or path.name.startswith('_'):
            continue
        if not path.name.startswith('M7-'):
            continue
        marker = path / 'queue.json'
        if not marker.is_file():
            marker = path / 'ledger.json'
        if not marker.is_file():
            continue
        try:
            mtime = marker.stat().st_mtime
        except OSError:
            continue
        if mtime > best_mtime:
            best_mtime = mtime
            best = path
    return best


def _load_mass_config_raw(config_path: Path) -> dict[str, Any] | None:
    if not config_path.is_file():
        return None
    try:
        import yaml  # type: ignore
        raw = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    except Exception:  # noqa: BLE001
        return None
    return raw if isinstance(raw, dict) else None


def _max_waves_from_config(config_path: Path) -> int:
    raw = _load_mass_config_raw(config_path)
    if raw is None:
        return DEFAULT_MAX_WAVES
    mass = raw.get('mass_explore') or {}
    if isinstance(mass, dict) and mass.get('max_waves') is not None:
        try:
            return int(mass['max_waves'])
        except (TypeError, ValueError):
            return DEFAULT_MAX_WAVES
    return DEFAULT_MAX_WAVES


def _space_paths_from_config(config_path: Path) -> list[str]:
    raw = _load_mass_config_raw(config_path)
    if raw is None:
        return list(DEFAULT_SPACE_PATHS)
    mass = raw.get('mass_explore') or {}
    paths: list[str] = []
    if isinstance(mass, dict):
        for name in mass.get('space_paths') or []:
            text = str(name or '').strip()
            if text:
                paths.append(Path(text).name)
    if paths:
        return paths
    search = raw.get('search_space_path')
    if search:
        return [Path(str(search)).name]
    return list(DEFAULT_SPACE_PATHS)


def _expected_space_for_wave(wave_index: int, space_paths: list[str]) -> str:
    """Match mass_explore wave→space mapping (named spaces then expanded extras)."""
    if wave_index < 0:
        wave_index = 0
    if wave_index < len(space_paths):
        return space_paths[wave_index]
    if EXPANDED_SPACE_NAME in space_paths:
        return EXPANDED_SPACE_NAME
    if space_paths:
        return space_paths[-1]
    return EXPANDED_SPACE_NAME


def _read_yaml_scalars(path: Path) -> dict[str, str]:
    """Best-effort top-level scalar parse (no PyYAML required)."""
    out: dict[str, str] = {}
    try:
        text = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError):
        return out
    for line in text.splitlines():
        if not line or line[0] in {' ', '\t', '#', '-'}:
            continue
        if ':' not in line:
            continue
        key, _, val = line.partition(':')
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not key or not val:
            continue
        if val.startswith(('{', '[', '|', '>')):
            continue
        out[key] = val
    return out


def _logical_space_name(raw_name: str | None, *, wave_index: int | None, space_paths: list[str]) -> str:
    name = Path(str(raw_name or '')).name
    if name.startswith('wave_') and name.endswith('_space.yaml'):
        if wave_index is not None:
            return _expected_space_for_wave(wave_index, space_paths)
        return EXPANDED_SPACE_NAME
    if name:
        return name
    if wave_index is not None:
        return _expected_space_for_wave(wave_index, space_paths)
    return EXPANDED_SPACE_NAME


def _space_size_for_name(
    space_name: str,
    *,
    configs_dir: Path,
    v1_size: int | None,
    exp_size: int | None,
) -> int | None:
    basename = Path(space_name).name
    if 'expanded' in basename:
        resolved, _ = resolve_space_size(basename, configs_dir)
        return resolved if resolved is not None else exp_size
    if basename in SPACE_SIZE_KNOWN or (
        basename.endswith('v1.yaml') and 'expanded' not in basename
    ):
        resolved, _ = resolve_space_size(basename, configs_dir)
        if resolved is not None:
            return resolved
        if basename in SPACE_SIZE_KNOWN:
            return SPACE_SIZE_KNOWN[basename]
        return v1_size
    resolved, _ = resolve_space_size(basename, configs_dir)
    if resolved is not None:
        return resolved
    return exp_size


def _session_recorded_run_ids(waves_list: list[Any]) -> set[str]:
    out: set[str] = set()
    for wave in waves_list:
        if not isinstance(wave, dict):
            continue
        run_id = str(wave.get('campaign_run_id') or '').strip()
        if run_id:
            out.add(run_id)
    return out


def _list_runtime_wave_configs(runtime_dir: Path) -> list[dict[str, Any]]:
    """Parse `_mass_explore/runtime/wave_XX_config.yaml` written before each wave runs."""
    if not runtime_dir.is_dir():
        return []
    found: list[dict[str, Any]] = []
    try:
        entries = list(runtime_dir.iterdir())
    except OSError:
        return []
    for path in entries:
        if not path.is_file():
            continue
        name = path.name
        if not (name.startswith('wave_') and name.endswith('_config.yaml')):
            continue
        mid = name[len('wave_'):-len('_config.yaml')]
        try:
            wave_index = int(mid)
        except ValueError:
            continue
        scalars = _read_yaml_scalars(path)
        run_id = str(scalars.get('campaign_run_id') or '').strip() or None
        space = str(scalars.get('search_space_path') or '').strip() or None
        found.append({
            'wave': wave_index,
            'path': str(path),
            'campaign_run_id': run_id,
            'search_space_path': space,
        })
    found.sort(key=lambda row: int(row['wave']))
    return found


def _campaign_looks_active(campaign: dict[str, Any] | None) -> bool:
    if not campaign:
        return False
    in_flight = (
        int(campaign.get('queue_pending') or 0)
        + int(campaign.get('queue_reserved') or 0)
        + int(campaign.get('queue_running') or 0)
    )
    state = str(campaign.get('campaign_state') or '').upper()
    if in_flight > 0:
        return True
    return state in {
        'RUNNING', 'SCREENING', 'CREATED', 'RESUMED', 'IN_PROGRESS',
    }


def _detect_inflight_wave(
    persist: Path,
    *,
    waves_list: list[Any],
    waves_done: int,
    space_paths: list[str],
) -> dict[str, Any] | None:
    """Find a wave started after the last session append (session updates on return only).

    Priority:
      1. runtime/wave_NN_config.yaml with NN >= waves_done
      2. LATEST_CAMPAIGN_B_RESUME pointing at an unrecorded active M7 campaign
      3. newest M7 campaign dir not listed in the mass session waves
    """
    recorded = _session_recorded_run_ids(waves_list)
    runtime_dir = persist / 'campaign_b' / '_mass_explore' / 'runtime'
    runtime_waves = _list_runtime_wave_configs(runtime_dir)
    for row in reversed(runtime_waves):
        wave_index = int(row['wave'])
        if wave_index < waves_done:
            continue
        run_id = row.get('campaign_run_id')
        if not run_id or run_id in recorded:
            continue
        campaign_root = persist / 'campaign_b' / str(run_id)
        space = _logical_space_name(
            row.get('search_space_path'),
            wave_index=wave_index,
            space_paths=space_paths,
        )
        return {
            'wave': wave_index,
            'campaign_run_id': str(run_id),
            'space': space,
            'source': 'runtime_wave_config',
            'campaign_root': campaign_root if campaign_root.is_dir() else None,
        }

    resume = _load_json_dict(
        persist / 'campaign_b' / 'LATEST_CAMPAIGN_B_RESUME.json',
    )
    if resume:
        run_id = str(
            resume.get('campaign_run_id')
            or resume.get('resume_campaign_run_id')
            or '',
        ).strip()
        if run_id and run_id not in recorded and run_id.startswith('M7-'):
            campaign_root = persist / 'campaign_b' / run_id
            if campaign_root.is_dir():
                counts = _campaign_counts(campaign_root)
                if _campaign_looks_active(counts):
                    wave_hint = resume.get('wave')
                    try:
                        wave_index = (
                            int(wave_hint) if wave_hint is not None else waves_done
                        )
                    except (TypeError, ValueError):
                        wave_index = waves_done
                    if wave_index < waves_done:
                        wave_index = waves_done
                    return {
                        'wave': wave_index,
                        'campaign_run_id': run_id,
                        'space': _expected_space_for_wave(wave_index, space_paths),
                        'source': 'resume_pointer',
                        'campaign_root': campaign_root,
                    }

    newest = _newest_m7_campaign(persist)
    if newest is not None and newest.name not in recorded:
        counts = _campaign_counts(newest)
        if _campaign_looks_active(counts):
            wave_index = waves_done
            return {
                'wave': wave_index,
                'campaign_run_id': newest.name,
                'space': _expected_space_for_wave(wave_index, space_paths),
                'source': 'newest_m7_campaign',
                'campaign_root': newest,
            }
    return None


def _label_89(
    *,
    session: dict[str, Any] | None,
    waves_done: int,
    max_waves: int,
    campaign: dict[str, Any] | None,
) -> str:
    if session and (
        session.get('finished_at')
        or session.get('status') == 'MASS_EXPLORE_COMPLETE'
    ):
        return 'COMPLETE'
    if session is None and campaign is None:
        return 'UNKNOWN'
    if campaign is not None:
        state = str(campaign.get('campaign_state') or '').upper()
        terminal = campaign.get('terminal_reason')
        if _campaign_looks_active(campaign):
            return 'WAVE_IN_PROGRESS'
        if state in {'EXHAUSTED', 'FINALIZED', 'COMPLETE', 'DONE'} or terminal:
            if waves_done >= max_waves:
                return 'COMPLETE'
            return 'BETWEEN_WAVES'
    if waves_done <= 0:
        return 'UNKNOWN' if session is None else 'BETWEEN_WAVES'
    if waves_done >= max_waves:
        return 'COMPLETE'
    return 'BETWEEN_WAVES'


def estimate_89(
    persist: Path,
    *,
    configs_dir: Path,
    mass_config: Path,
) -> dict[str, Any]:
    mass_root = persist / 'campaign_b' / '_mass_explore'
    session_path = mass_root / 'LATEST_MASS_SESSION.json'
    seen_path = mass_root / 'seen_normalized_schemes.json'
    session = _load_json_dict(session_path)
    seen_payload = _load_json(seen_path)
    seen = _seen_count(seen_payload)
    max_waves = _max_waves_from_config(mass_config)
    space_paths = _space_paths_from_config(mass_config)
    waves = (session or {}).get('waves') if session else None
    waves_list = waves if isinstance(waves, list) else []
    waves_done = len(waves_list)

    v1_size, v1_src = resolve_space_size('campaign_b_s2_space_v1.yaml', configs_dir)
    exp_size, exp_src = resolve_space_size(EXPANDED_SPACE_NAME, configs_dir)

    current_wave_index: int | None = None
    current_space: str | None = None
    current_run_id: str | None = None
    recorded_run_id: str | None = None
    if waves_list:
        last = waves_list[-1]
        if isinstance(last, dict):
            try:
                current_wave_index = int(last.get('wave'))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                current_wave_index = waves_done - 1
            current_space = str(last.get('space') or '') or None
            recorded_run_id = str(last.get('campaign_run_id') or '') or None
            current_run_id = recorded_run_id

    resume = _load_json_dict(
        persist / 'campaign_b' / 'LATEST_CAMPAIGN_B_RESUME.json',
    )
    inflight = None
    session_finished = bool(
        session
        and (session.get('finished_at') or session.get('status') == 'MASS_EXPLORE_COMPLETE')
    )
    if not session_finished:
        inflight = _detect_inflight_wave(
            persist,
            waves_list=waves_list,
            waves_done=waves_done,
            space_paths=space_paths,
        )

    detection_source = 'session_wave'
    if inflight is not None:
        current_wave_index = int(inflight['wave'])
        current_space = str(inflight.get('space') or '') or current_space
        current_run_id = str(inflight['campaign_run_id'])
        detection_source = str(inflight.get('source') or 'inflight')
    elif current_run_id is None and resume:
        current_run_id = (
            str(resume.get('campaign_run_id') or resume.get('resume_campaign_run_id') or '')
            or None
        )
        detection_source = 'resume_pointer'

    # When the last recorded wave is done but the next wave has not started yet,
    # remaining-scheme math must use the *next* wave's space (usually expanded).
    next_wave_index = waves_done
    next_space = _expected_space_for_wave(next_wave_index, space_paths)
    if inflight is None and not session_finished and waves_done < max_waves:
        # Prefer next/current space over the exhausted previous wave space.
        if current_space is None or (
            waves_done > 0
            and 'expanded' not in Path(str(current_space)).name
            and 'expanded' in Path(next_space).name
        ):
            current_space = next_space
            if current_wave_index is None or current_wave_index < next_wave_index:
                current_wave_index = next_wave_index
            detection_source = (
                detection_source
                if detection_source != 'session_wave'
                else 'next_wave_space'
            )

    campaign_root: Path | None = None
    if inflight is not None and inflight.get('campaign_root') is not None:
        campaign_root = Path(inflight['campaign_root'])
    elif current_run_id:
        candidate = persist / 'campaign_b' / current_run_id
        if candidate.is_dir():
            campaign_root = candidate
    if campaign_root is None:
        campaign_root = _newest_m7_campaign(persist)
        if campaign_root is not None:
            # Avoid attributing an unrelated newer campaign when session already
            # points at a recorded exhausted wave and nothing is in flight.
            if inflight is not None or recorded_run_id is None:
                current_run_id = campaign_root.name
                detection_source = 'newest_m7_campaign'
            elif campaign_root.name == recorded_run_id:
                current_run_id = recorded_run_id
            else:
                # Keep the session campaign unless newest looks active & unrecorded.
                newest_counts = _campaign_counts(campaign_root)
                if (
                    campaign_root.name not in _session_recorded_run_ids(waves_list)
                    and _campaign_looks_active(newest_counts)
                ):
                    current_run_id = campaign_root.name
                    current_wave_index = waves_done
                    current_space = _expected_space_for_wave(waves_done, space_paths)
                    detection_source = 'newest_m7_campaign'
                else:
                    campaign_root = persist / 'campaign_b' / recorded_run_id
                    current_run_id = recorded_run_id

    campaign = _campaign_counts(campaign_root) if campaign_root else None
    label = _label_89(
        session=session,
        waves_done=waves_done,
        max_waves=max_waves,
        campaign=campaign,
    )

    active_space_name = _logical_space_name(
        current_space,
        wave_index=current_wave_index,
        space_paths=space_paths,
    )
    # Fail closed toward expanded when mid-session after v1 is exhausted.
    if (
        not session_finished
        and waves_done >= 1
        and label != 'COMPLETE'
        and 'expanded' not in Path(active_space_name).name
    ):
        active_space_name = next_space if 'expanded' in Path(next_space).name else EXPANDED_SPACE_NAME

    active_space_size = _space_size_for_name(
        active_space_name,
        configs_dir=configs_dir,
        v1_size=v1_size,
        exp_size=exp_size,
    )

    unseen_approx: int | None = None
    if active_space_size is not None and seen is not None:
        unseen_approx = max(0, int(active_space_size) - int(seen))

    waves_remaining = max(0, max_waves - waves_done)
    if label == 'WAVE_IN_PROGRESS' and waves_remaining == 0:
        waves_remaining = 1
    queue_pending = int((campaign or {}).get('queue_pending') or 0)
    remaining_estimate: dict[str, Any] = {
        'current_wave_queue_pending': queue_pending,
        'schemes_unseen_in_active_space_approx': unseen_approx,
        'waves_remaining_incl_current': waves_remaining if label != 'COMPLETE' else 0,
        'active_space_name': active_space_name,
        'active_space_size': active_space_size,
        'detection_source': detection_source,
        'note': (
            'unseen ≈ active/next wave space_size − seen_count (skip_seen); '
            'in-flight waves detected via runtime config / resume / newer M7; '
            'wave queue pending is the immediate backlog'
        ),
    }

    return {
        'label': label,
        'session_path': str(session_path),
        'session_age': _fmt_age(_age_seconds(session_path)),
        'seen_path': str(seen_path),
        'seen_age': _fmt_age(_age_seconds(seen_path)),
        'session_id': (session or {}).get('session_id'),
        'finished_at': (session or {}).get('finished_at'),
        'waves_done': waves_done,
        'max_waves': max_waves,
        'current_wave_index': current_wave_index,
        'current_space': current_space,
        'current_campaign_run_id': current_run_id,
        'space_v1_size': v1_size,
        'space_v1_source': v1_src,
        'space_expanded_size': exp_size,
        'space_expanded_source': exp_src,
        'seen_normalized_schemes': seen,
        'selected_total_session': (session or {}).get('selected_total'),
        'archived_total_session': (session or {}).get('archived_total'),
        'campaign': campaign,
        'remaining': remaining_estimate,
        'inflight_detection': inflight,
    }


def _already_advanced(package: Path) -> bool:
    doc = _load_json_dict(package / 'ADVANCE.json')
    if not isinstance(doc, dict):
        return False
    return str(doc.get('status') or '') in ADVANCED_STATUSES


def _discover_selected_fallback(persist: Path) -> list[Path]:
    """Mirror advance_selected.discover_selected_packages without importing src."""
    root = Path(persist) / 'campaign_b'
    if not root.is_dir():
        return []
    found: list[Path] = []
    try:
        campaigns = sorted(root.iterdir())
    except OSError:
        return []
    for campaign in campaigns:
        if not campaign.is_dir() or campaign.name.startswith('_'):
            continue
        selected = campaign / 'selected'
        if not selected.is_dir():
            continue
        try:
            packages = sorted(selected.iterdir())
        except OSError:
            continue
        for package in packages:
            if package.is_dir() and (package / 'candidate_manifest.json').is_file():
                found.append(package)
    return found


def _count_m6_complete(persist: Path, packages: list[Path]) -> int | None:
    try:
        from src.campaign_b.m6_batch import _child_ids, _m6_done
    except Exception:  # noqa: BLE001
        return None
    n = 0
    for package in packages:
        try:
            child = _child_ids(package)
            if not isinstance(child, dict):
                continue
            m6_id = str(child.get('M6') or '')
            if m6_id.startswith('M6-') and _m6_done(persist, m6_id):
                n += 1
        except Exception:  # noqa: BLE001
            continue
    return n


def _safe_queue_len(fn: Any, persist: Path) -> int | None:
    try:
        rows = fn(persist)
    except Exception:  # noqa: BLE001 — partial / corrupt trees
        return None
    if not isinstance(rows, list):
        return None
    return len(rows)


def _fallback_ready_for_m3(packages: list[Path]) -> int:
    """READY_FOR_M3 / binding ready and not yet M3_COMPLETE (no src import)."""
    n = 0
    for package in packages:
        advance = _load_json_dict(package / 'ADVANCE.json')
        binding = _load_json_dict(package / 'm2_binding.json')
        ready = False
        if isinstance(advance, dict) and advance.get('status') == 'READY_FOR_M3':
            ready = True
        elif isinstance(binding, dict):
            status = binding.get('status') or binding.get('binding_status')
            if status in {'READY_SHARED', 'READY', 'READY_BINDING'}:
                ready = True
        if not ready:
            continue
        gpu = _load_json_dict(package / 'GPU_M3.json')
        if isinstance(gpu, dict) and gpu.get('status') == 'M3_COMPLETE':
            continue
        n += 1
    return n


def estimate_95(persist: Path, *, stale_s: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        'label': 'UNKNOWN',
        'active_stage': None,
        'stage_ages': {},
        'counts': {},
        'errors': [],
    }
    discover_selected_packages = None
    list_obligation_queue = None
    list_gpu_m3_queue = None
    list_m6_queue = None
    list_pre_m6_queue = None
    try:
        from src.campaign_b.advance_selected import (
            discover_selected_packages as _disc,
        )
        from src.campaign_b.close_obligations import list_obligation_queue as _obl
        from src.campaign_b.gpu_m3_batch import list_gpu_m3_queue as _m3
        from src.campaign_b.m6_batch import list_m6_queue as _m6
        from src.campaign_b.pre_m6_batch import list_pre_m6_queue as _pre
        discover_selected_packages = _disc
        list_obligation_queue = _obl
        list_gpu_m3_queue = _m3
        list_m6_queue = _m6
        list_pre_m6_queue = _pre
    except Exception as exc:  # noqa: BLE001
        out['errors'].append(
            f'import_helpers_failed: {type(exc).__name__}: {exc} '
            '(using filesystem fallbacks where possible)',
        )

    packages: list[Path] = []
    if discover_selected_packages is not None:
        try:
            packages = list(discover_selected_packages(persist))
        except Exception as exc:  # noqa: BLE001
            out['errors'].append(f'discover_selected: {type(exc).__name__}: {exc}')
            packages = _discover_selected_fallback(persist)
    else:
        packages = _discover_selected_fallback(persist)

    selected_total = len(packages)
    need_advance = 0
    for package in packages:
        try:
            if not _already_advanced(package):
                need_advance += 1
        except Exception:  # noqa: BLE001
            continue

    ready_for_m3 = (
        _safe_queue_len(list_gpu_m3_queue, persist)
        if list_gpu_m3_queue is not None else None
    )
    if ready_for_m3 is None:
        ready_for_m3 = _fallback_ready_for_m3(packages)

    awaiting_pre_m6 = (
        _safe_queue_len(list_pre_m6_queue, persist)
        if list_pre_m6_queue is not None else None
    )
    open_obligations = (
        _safe_queue_len(list_obligation_queue, persist)
        if list_obligation_queue is not None else None
    )
    ready_for_m6 = (
        _safe_queue_len(list_m6_queue, persist)
        if list_m6_queue is not None else None
    )
    m6_complete = _count_m6_complete(persist, packages)

    out['counts'] = {
        'selected_packages': selected_total,
        'selected_not_advanced': need_advance,
        'ready_for_m3_not_m3_complete': ready_for_m3,
        'm3_complete_awaiting_pre_m6': awaiting_pre_m6,
        'pre_m6_open_obligations': open_obligations,
        'ready_for_m6': ready_for_m6,
        'm6_complete': m6_complete,
    }

    stage_ages: dict[str, float | None] = {}
    freshest_name: str | None = None
    freshest_age: float | None = None
    for name, rel in STAGE_LEDGERS:
        path = persist / rel
        age = _age_seconds(path)
        stage_ages[name] = age
        payload = _load_json_dict(path)
        finished = bool(payload and payload.get('finished_at'))
        # Prefer in-flight (no finished_at) or simply freshest ledger.
        if age is None:
            continue
        score = age
        if finished:
            score = age + 1e-3  # slight penalty vs unfinished twin
        if freshest_age is None or score < freshest_age:
            freshest_age = score
            freshest_name = name

    out['stage_ages'] = {k: _fmt_age(v) for k, v in stage_ages.items()}
    if freshest_name is not None and freshest_age is not None and freshest_age <= stale_s:
        out['active_stage'] = freshest_name
        out['label'] = f'ACTIVE_{freshest_name.upper()}'
    else:
        backlog = [
            need_advance,
            ready_for_m3 or 0,
            awaiting_pre_m6 or 0,
            open_obligations or 0,
            ready_for_m6 or 0,
        ]
        if selected_total == 0 and not any(v is not None for v in stage_ages.values()):
            out['label'] = 'UNKNOWN'
        elif sum(backlog) == 0:
            if (m6_complete or 0) > 0 or selected_total == 0:
                out['label'] = 'DRAINED'
            else:
                out['label'] = 'IDLE'
        else:
            out['label'] = 'BACKLOG'
        out['active_stage'] = None

    pipeline = _load_json_dict(
        persist / 'campaign_b' / '_pipeline_to_m6' / 'LATEST_PIPELINE_SESSION.json',
    )
    out['pipeline_session_id'] = (pipeline or {}).get('session_id')
    out['pipeline_finished_at'] = (pipeline or {}).get('finished_at')
    out['pipeline_ledger_age'] = _fmt_age(
        _age_seconds(
            persist / 'campaign_b' / '_pipeline_to_m6' / 'LATEST_PIPELINE_SESSION.json',
        ),
    )
    out['freshest_stage_age'] = _fmt_age(freshest_age)
    return out


def check_89(
    persist: Path,
    *,
    configs_dir: Path,
    mass_config: Path,
) -> str:
    info = estimate_89(persist, configs_dir=configs_dir, mass_config=mass_config)
    camp = info.get('campaign') or {}
    rem = info.get('remaining') or {}
    rows: list[tuple[str, Any]] = [
        ('label', info['label']),
        ('session_id', info.get('session_id')),
        ('session_age', info.get('session_age')),
        ('finished_at', info.get('finished_at')),
        ('waves_done', f"{info.get('waves_done')} / {info.get('max_waves')}"),
        ('current_wave_index', info.get('current_wave_index')),
        ('current_space', info.get('current_space')),
        ('current_campaign_run_id', info.get('current_campaign_run_id')),
        ('detection_source', rem.get('detection_source')),
        ('active_space_name', rem.get('active_space_name')),
        ('active_space_size', rem.get('active_space_size')),
        ('space_v1_size', f"{info.get('space_v1_size')} ({info.get('space_v1_source')})"),
        (
            'space_expanded_size',
            f"{info.get('space_expanded_size')} ({info.get('space_expanded_source')})",
        ),
        ('seen_normalized_schemes', info.get('seen_normalized_schemes')),
        ('seen_file_age', info.get('seen_age')),
        ('session_selected_total', info.get('selected_total_session')),
        ('session_archived_total', info.get('archived_total_session')),
        ('campaign_state', camp.get('campaign_state')),
        ('terminal_reason', camp.get('terminal_reason')),
        ('queue_pending', camp.get('queue_pending')),
        ('queue_reserved', camp.get('queue_reserved')),
        ('queue_running', camp.get('queue_running')),
        ('queue_total', camp.get('queue_total')),
        ('selected', camp.get('selected')),
        ('archived', camp.get('archived')),
        ('remaining.queue_pending', rem.get('current_wave_queue_pending')),
        (
            'remaining.schemes_unseen_approx',
            rem.get('schemes_unseen_in_active_space_approx'),
        ),
        (
            'remaining.waves_remaining',
            rem.get('waves_remaining_incl_current'),
        ),
    ]
    _print_block('Notebook 89 — mass explore remaining', rows)
    return str(info['label'])


def check_95(persist: Path, *, stale_s: float) -> str:
    info = estimate_95(persist, stale_s=stale_s)
    counts = info.get('counts') or {}
    rows: list[tuple[str, Any]] = [
        ('label', info.get('label')),
        ('active_stage', info.get('active_stage')),
        ('freshest_stage_age', info.get('freshest_stage_age')),
        ('pipeline_session_id', info.get('pipeline_session_id')),
        ('pipeline_ledger_age', info.get('pipeline_ledger_age')),
        ('pipeline_finished_at', info.get('pipeline_finished_at')),
        ('selected_packages', counts.get('selected_packages')),
        ('selected_not_advanced', counts.get('selected_not_advanced')),
        ('ready_for_m3_not_m3_complete', counts.get('ready_for_m3_not_m3_complete')),
        ('m3_complete_awaiting_pre_m6', counts.get('m3_complete_awaiting_pre_m6')),
        ('pre_m6_open_obligations', counts.get('pre_m6_open_obligations')),
        ('ready_for_m6', counts.get('ready_for_m6')),
        ('m6_complete', counts.get('m6_complete')),
    ]
    for name, age in (info.get('stage_ages') or {}).items():
        rows.append((f'stage_age.{name}', age))
    for err in info.get('errors') or []:
        rows.append(('error', err))
    _print_block('Notebook 95 — pipeline remaining', rows)
    return str(info.get('label') or 'UNKNOWN')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Remaining-work estimate for Campaign B notebooks 89 and 95',
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT',
            '/storage/validated_4d_su2_rg',
        ),
    )
    parser.add_argument(
        '--repo-root',
        default=str(Path(__file__).resolve().parents[1]),
        help='Repo root (for configs + importing src.campaign_b helpers)',
    )
    parser.add_argument(
        '--mass-config',
        default=None,
        help='Path to campaign_b_mass_explore.yaml (default: <repo>/configs/...)',
    )
    parser.add_argument(
        '--stale-minutes',
        type=float,
        default=90.0,
        help='Stage ledger fresher than this → treat as active_stage',
    )
    args = parser.parse_args(argv)

    persist = Path(args.persistent_root)
    repo_root = Path(args.repo_root)
    _ensure_repo_on_path(repo_root)
    configs_dir = repo_root / 'configs'
    mass_config = Path(args.mass_config) if args.mass_config else (
        configs_dir / 'campaign_b_mass_explore.yaml'
    )
    stale_s = float(args.stale_minutes) * 60.0

    print(f'persist_root: {persist}')
    print(f'repo_root: {repo_root}')
    print(f'stale_minutes: {args.stale_minutes}')
    print(f'checked_at: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    print()
    print(
        '89 labels: WAVE_IN_PROGRESS | BETWEEN_WAVES | COMPLETE | UNKNOWN '
        '(from mass session + campaign queue/ledger, not processes).'
    )
    print(
        '95 labels: ACTIVE_<stage> | BACKLOG | DRAINED | IDLE | UNKNOWN '
        '(stage queues via advance/gpu_m3/pre_m6/obligations/m6 discovery helpers).'
    )
    print(
        'Note: 95 LATEST_PIPELINE_SESSION is end-of-session only; '
        'stage LATEST_* ages show mid-run activity.'
    )
    print()

    s89 = check_89(persist, configs_dir=configs_dir, mass_config=mass_config)
    s95 = check_95(persist, stale_s=stale_s)
    print(f'SUMMARY  89={s89}  95={s95}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
