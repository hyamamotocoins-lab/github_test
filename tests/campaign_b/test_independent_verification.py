from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json, read_json
from src.campaign_b.screening import run_primary_screening
from src.campaign_b.independent_verifier import run_independent_verifier

from .conftest import REPO


def test_atomic_write(tmp_path: Path) -> None:
    path = tmp_path / 'x.json'
    atomic_write_json(path, {'a': 1})
    assert read_json(path) == {'a': 1}


def test_independent_verification_agrees() -> None:
    candidate = {
        'candidate_id': 'B-test',
        'j2': 2,
        'execution_mode': 'staged',
        'scheme_hash': 'abc',
        'scheme': {
            'change_class': 'S2',
            'target_rank': 64,
            'oversampling': 24,
            'power_iterations': 3,
            'perron_weight_strategy': 'all_ones',
            'coupling_policy': 'uniform_full',
            'j2': 2,
            'execution_mode': 'staged',
        },
        'certification_status': 'NOT_CERTIFIED',
        'claim_scope': 'SCREENING_ONLY',
    }
    primary = run_primary_screening(
        candidate,
        parent_q_upper=1.011,
        parent_rank=16,
        screening_margin=1e-6,
    )
    verify = run_independent_verifier(
        candidate=candidate,
        primary_result=primary,
        parent_q_upper=1.011,
        parent_rank=16,
        screening_margin=1e-6,
        q_atol=1e-9,
        q_rtol=1e-6,
        repo_root=REPO,
    )
    assert verify['accepted'] is True
    assert verify['both_q_lt_1'] is True
