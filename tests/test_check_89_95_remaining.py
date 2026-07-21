"""CPU unit tests for scripts/check_89_95_remaining.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'scripts' / 'check_89_95_remaining.py'


def _load_mod():
    spec = importlib.util.spec_from_file_location('check_89_95_remaining', SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['check_89_95_remaining'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def rem():
    return _load_mod()


def test_space_size_known_and_yaml(rem) -> None:
    configs = REPO_ROOT / 'configs'
    v1, src = rem.resolve_space_size('campaign_b_s2_space_v1.yaml', configs)
    exp, esrc = rem.resolve_space_size('campaign_b_s2_space_expanded_v1.yaml', configs)
    assert v1 == 405
    assert exp == 45360
    assert src in {'yaml_product', 'known_constant'}
    assert esrc in {'yaml_product', 'known_constant'}


def test_estimate_89_wave_in_progress(rem, tmp_path: Path) -> None:
    mass = tmp_path / 'campaign_b' / '_mass_explore'
    mass.mkdir(parents=True)
    run_id = 'M7-20260721T115724Z-b-7696b9087a66'
    camp = tmp_path / 'campaign_b' / run_id
    camp.mkdir(parents=True)
    (mass / 'LATEST_MASS_SESSION.json').write_text(
        json.dumps({
            'session_id': 'MASS-TEST',
            'waves': [{
                'wave': 1,
                'space': 'campaign_b_s2_space_expanded_v1.yaml',
                'campaign_run_id': run_id,
            }],
            'selected_total': 2,
            'archived_total': 10,
        }),
        encoding='utf-8',
    )
    (mass / 'seen_normalized_schemes.json').write_text(
        json.dumps({
            'count': 405,
            'normalized_scheme_keys': [f'k{i}' for i in range(405)],
        }),
        encoding='utf-8',
    )
    (camp / 'queue.json').write_text(
        json.dumps({
            'candidates': [
                {'candidate_id': 'a', 'state': 'PENDING'},
                {'candidate_id': 'b', 'state': 'PENDING'},
                {'candidate_id': 'c', 'state': 'SELECTED'},
            ],
        }),
        encoding='utf-8',
    )
    (camp / 'ledger.json').write_text(
        json.dumps({
            'campaign_state': 'RUNNING',
            'selected': [{'candidate_id': 'c'}],
            'archived_ids': ['x'],
        }),
        encoding='utf-8',
    )
    configs = REPO_ROOT / 'configs'
    info = rem.estimate_89(
        tmp_path,
        configs_dir=configs,
        mass_config=configs / 'campaign_b_mass_explore.yaml',
    )
    assert info['label'] == 'WAVE_IN_PROGRESS'
    assert info['waves_done'] == 1
    assert info['max_waves'] == 8
    assert info['seen_normalized_schemes'] == 405
    assert info['campaign']['queue_pending'] == 2
    assert info['remaining']['current_wave_queue_pending'] == 2
    assert info['remaining']['schemes_unseen_in_active_space_approx'] == 45360 - 405


def test_estimate_89_complete(rem, tmp_path: Path) -> None:
    mass = tmp_path / 'campaign_b' / '_mass_explore'
    mass.mkdir(parents=True)
    (mass / 'LATEST_MASS_SESSION.json').write_text(
        json.dumps({
            'session_id': 'MASS-DONE',
            'status': 'MASS_EXPLORE_COMPLETE',
            'finished_at': '2026-07-21T00:00:00Z',
            'waves': [{'wave': i} for i in range(8)],
        }),
        encoding='utf-8',
    )
    configs = REPO_ROOT / 'configs'
    info = rem.estimate_89(
        tmp_path,
        configs_dir=configs,
        mass_config=configs / 'campaign_b_mass_explore.yaml',
    )
    assert info['label'] == 'COMPLETE'


def test_estimate_95_backlog_counts(rem, tmp_path: Path) -> None:
    pkg = tmp_path / 'campaign_b' / 'M7-TEST-b-aaa' / 'selected' / 'CAND-1'
    pkg.mkdir(parents=True)
    (pkg / 'candidate_manifest.json').write_text(
        json.dumps({
            'candidate_id': 'CAND-1',
            'scheme': {'change_class': 'S2', 'target_rank': 16},
        }),
        encoding='utf-8',
    )
    # Not advanced → selected_not_advanced == 1
    rem._ensure_repo_on_path(REPO_ROOT)
    info = rem.estimate_95(tmp_path, stale_s=90 * 60)
    assert info['counts']['selected_packages'] == 1
    assert info['counts']['selected_not_advanced'] == 1
    assert info['label'] in {'BACKLOG', 'UNKNOWN', 'IDLE', 'DRAINED'} or str(
        info['label'],
    ).startswith('ACTIVE_')

    # Mark advanced + READY_FOR_M3 without GPU_M3 → ready_for_m3 queue
    (pkg / 'ADVANCE.json').write_text(
        json.dumps({'status': 'READY_FOR_M3'}),
        encoding='utf-8',
    )
    (pkg / 'm2_binding.json').write_text(
        json.dumps({'status': 'READY_SHARED'}),
        encoding='utf-8',
    )
    info2 = rem.estimate_95(tmp_path, stale_s=90 * 60)
    assert info2['counts']['selected_not_advanced'] == 0
    assert info2['counts']['ready_for_m3_not_m3_complete'] == 1
