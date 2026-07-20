from __future__ import annotations

from pathlib import Path

from src.campaign_b.candidate_generator import generate_campaign_b_queue_candidates
from src.campaign_b.errors import InvariantViolation
from src.campaign_b.queue_store import QueueStore
from src.campaign_b.schemas import assert_staged_candidate


def _space() -> dict:
    return {
        'schema_version': 1,
        'campaign': 'B_S2',
        'rank': {'values': [32, 64]},
        'rsvd': {
            'oversampling': [16],
            'power_iterations': [2],
            'seeds': [1],
        },
        'residual': {'tolerances': [0.0], 'norm_models': ['frobenius']},
        'staging': {'j2_values': [2, 3], 'forbid_j2_1': True},
        'layers': {
            'perron_weight_strategy': ['all_ones'],
            'coupling_policy': ['uniform_full'],
        },
    }


def test_all_candidates_staged() -> None:
    cands = generate_campaign_b_queue_candidates(
        campaign_run_id='M7-test-b',
        search_space=_space(),
        structural_key='sk',
        proof_key='pk',
        source_tree_hash='abc',
        parent_m6_run_id='p',
        parent_scheme_hash='0' * 64,
    )
    assert cands
    for c in cands:
        assert_staged_candidate(c)
        assert c['j2'] >= 2
        assert c['execution_mode'] == 'staged'
        assert c['certification_status'] == 'NOT_CERTIFIED'


def test_reject_j2_1_generation() -> None:
    space = _space()
    space['staging']['j2_values'] = [1]
    try:
        generate_campaign_b_queue_candidates(
            campaign_run_id='M7-test-b',
            search_space=space,
            structural_key='sk',
            proof_key='pk',
            source_tree_hash='abc',
            parent_m6_run_id='p',
            parent_scheme_hash='0' * 64,
        )
        raised = False
    except InvariantViolation:
        raised = True
    assert raised


def test_queue_resume(tmp_path: Path) -> None:
    store = QueueStore(tmp_path)
    cands = generate_campaign_b_queue_candidates(
        campaign_run_id='M7-test-b',
        search_space=_space(),
        structural_key='sk',
        proof_key='pk',
        source_tree_hash='abc',
        parent_m6_run_id='p',
        parent_scheme_hash='0' * 64,
        limit=2,
    )
    q1 = store.load_or_init(cands, campaign_run_id='M7-test-b')
    store.update_candidate(q1, cands[0]['candidate_id'], state='ARCHIVED')
    q2 = store.load_or_init([], campaign_run_id='M7-test-b')
    states = {c['candidate_id']: c['state'] for c in q2['candidates']}
    assert states[cands[0]['candidate_id']] == 'ARCHIVED'
    assert states[cands[1]['candidate_id']] == 'PENDING'
