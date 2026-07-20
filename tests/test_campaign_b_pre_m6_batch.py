"""Tests for pre-M6 queue discovery (no CUDA)."""

from __future__ import annotations

from pathlib import Path

from src.campaign_b.pre_m6_batch import list_pre_m6_queue
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def test_list_pre_m6_requires_m3_complete(tmp_path: Path) -> None:
    pkg = tmp_path / 'campaign_b' / 'M7-A' / 'selected' / 'B-aaa'
    pkg.mkdir(parents=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': 'B-aaa',
        'scheme': {'change_class': 'S2', 'target_rank': 16},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': 0.81})
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M2': 'M2-X', 'M3': 'M3-X', 'M4': 'M4-X', 'M5': 'M5-X', 'M6': 'M6-X',
    })
    # Not complete yet
    assert list_pre_m6_queue(tmp_path) == []

    atomic_write_json(pkg / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    # Still need M3 on disk unless GPU says complete — GPU_M3 alone is enough
    queue = list_pre_m6_queue(tmp_path)
    assert len(queue) == 1
    assert queue[0]['stage'] == 'NEED_M4'
    assert queue[0]['candidate_id'] == 'B-aaa'
