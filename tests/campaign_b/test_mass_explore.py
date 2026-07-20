from __future__ import annotations

from pathlib import Path

from src.campaign_b.candidate_generator import generate_campaign_b_queue_candidates
from src.campaign_b.mass_explore import harvest_seen_from_campaign, load_seen_schemes, save_seen_schemes


def _tiny_space() -> dict:
    return {
        'schema_version': 1,
        'campaign': 'B_S2',
        'rank': {'values': [64]},
        'rsvd': {
            'oversampling': [24],
            'power_iterations': [3],
            'seeds': [1, 2],
        },
        'residual': {'tolerances': [0.0], 'norm_models': ['frobenius']},
        'staging': {'j2_values': [2], 'forbid_j2_1': True},
        'layers': {
            'perron_weight_strategy': ['all_ones'],
            'coupling_policy': ['uniform_full'],
        },
    }


def test_exclude_normalized_keys() -> None:
    cands = generate_campaign_b_queue_candidates(
        campaign_run_id='M7-test-b',
        search_space=_tiny_space(),
        structural_key='sk',
        proof_key='pk',
        source_tree_hash='abc',
        parent_m6_run_id='p',
        parent_scheme_hash='0' * 64,
    )
    assert len(cands) == 2
    exclude = {cands[0]['normalized_scheme_key']}
    filtered = generate_campaign_b_queue_candidates(
        campaign_run_id='M7-test-b2',
        search_space=_tiny_space(),
        structural_key='sk',
        proof_key='pk',
        source_tree_hash='abc',
        parent_m6_run_id='p',
        parent_scheme_hash='0' * 64,
        exclude_normalized_keys=exclude,
    )
    assert len(filtered) == 1
    assert filtered[0]['normalized_scheme_key'] == cands[1]['normalized_scheme_key']


def test_seen_scheme_persistence(tmp_path: Path) -> None:
    save_seen_schemes(tmp_path, {'a', 'b'})
    assert load_seen_schemes(tmp_path) == {'a', 'b'}
    camp = tmp_path / 'campaign_b' / 'run1'
    camp.mkdir(parents=True)
    from src.common import atomic_write_json
    atomic_write_json(camp / 'queue.json', {
        'candidates': [
            {'normalized_scheme_key': 'x'},
            {'normalized_scheme_key': 'y'},
        ],
    })
    assert harvest_seen_from_campaign(camp) == {'x', 'y'}
