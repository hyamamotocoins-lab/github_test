"""Tests for Campaign B M6 queue discovery (no full M6 run)."""

from __future__ import annotations

from pathlib import Path

from src.campaign_b.m6_batch import list_m6_queue
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def test_list_m6_queue_requires_m5_package(tmp_path: Path) -> None:
    pkg = tmp_path / 'campaign_b' / 'M7-A' / 'selected' / 'B-aaa'
    pkg.mkdir(parents=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': 'B-aaa',
        'scheme': {'change_class': 'S2', 'target_rank': 16, 'num_steps': 3},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': 0.81})
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M5': 'M5-X', 'M6': 'M6-X',
    })
    assert list_m6_queue(tmp_path) == []

    m5 = tmp_path / 'runs' / 'M5-X'
    (m5 / 'reports').mkdir(parents=True)
    (m5 / 'artifacts' / 'one_step_certificate').mkdir(parents=True)
    atomic_write_json(m5 / 'reports' / 'M5_acceptance.json', {
        'milestone': 'M5',
        'phase': 'M5_COMPLETE',
        'status': 'PASS',
        'accepted_for_next_milestone': 'M6',
        'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(m5 / 'reports' / 'M5_obligation_report.json', {
        'all_closed': True,
        'open_obligations': [],
    })
    atomic_write_json(m5 / 'artifacts' / 'one_step_certificate' / 'verdict.json', {
        'independent_verifier': 'PASS',
    })
    atomic_write_json(pkg / 'M6_GATE.json', {
        'status': 'READY_FOR_STAGED_M6',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    queue = list_m6_queue(tmp_path)
    assert len(queue) == 1
    assert queue[0]['m6_run_id'] == 'M6-X'
