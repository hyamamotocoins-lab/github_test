"""CPU tests for notebook 98 read-only status and 97 post-M2 thin wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.campaign_b.pipeline_status import collect_pipeline_status
from src.campaign_b.post_m2_pipeline import find_m2_ready_markers, run_post_m2_pipeline
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def test_status_read_only_and_counts_queues(tmp_path: Path) -> None:
    camp = tmp_path / 'campaign_b' / 'M7-x'
    camp.mkdir(parents=True)
    atomic_write_json(camp / 'queue.json', {
        'candidates': [
            {'candidate_id': 'a', 'state': 'SELECTED'},
            {'candidate_id': 'b', 'state': 'NEED_CANONICAL_M2'},
            {'candidate_id': 'c', 'state': 'WAITING_FOR_M2'},
        ],
    })
    ready = tmp_path / 'runs' / 'M2-abc' / 'M2_READY.json'
    ready.parent.mkdir(parents=True)
    atomic_write_json(ready, {'ready': True})

    with (
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[{}, {}]),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', return_value=[{}]),
        patch('src.campaign_b.close_obligations.list_obligation_queue', return_value=[]),
        patch('src.campaign_b.m6_batch.list_m6_queue', return_value=[{}, {}, {}]),
        patch(
            'src.campaign_b.advance_selected.discover_selected_packages',
            return_value=[],
        ),
    ):
        status = collect_pipeline_status(tmp_path)

    assert status['read_only'] is True
    assert status['queues']['gpu_m3'] == 2
    assert status['queues']['m6'] == 3
    assert status['candidate_states']['SELECTED'] == 1
    assert status['candidate_states']['WAITING_FOR_M2'] == 1
    assert status['m2']['m2_ready_count'] == 1
    assert status['certification_status'] == CERTIFICATION_STATUS
    # Ensure queue.json not mutated
    q = (camp / 'queue.json').read_text(encoding='utf-8')
    assert 'SELECTED' in q


def test_find_m2_ready_markers(tmp_path: Path) -> None:
    assert find_m2_ready_markers(tmp_path) == []
    p = tmp_path / 'runs' / 'M2-1' / 'M2_READY.json'
    p.parent.mkdir(parents=True)
    atomic_write_json(p, {'ready': True, 'sectors_done': 3})
    markers = find_m2_ready_markers(tmp_path)
    assert len(markers) == 1
    assert markers[0]['run_id'] == 'M2-1'


def test_post_m2_delegates_to_end_to_end(tmp_path: Path) -> None:
    fake = {
        'session_id': 'E2E-mock',
        'started_at': 't0',
        'rounds_run': 1,
        'totals': {'m3_complete': 0},
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    }
    with patch(
        'src.campaign_b.end_to_end.run_end_to_end',
        return_value=fake,
    ) as mocked:
        summary = run_post_m2_pipeline(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=2,
            skip_screening=True,
        )
    assert mocked.called
    assert summary['notebook'] == 97
    assert summary['gpu_workers'] == 1
    assert summary['end_to_end']['session_id'] == 'E2E-mock'
    assert 'WAITING_FOR_M2' in summary['waiting_for_m2_todo']
    assert summary['certification_status'] == CERTIFICATION_STATUS
    ledger = tmp_path / 'campaign_b' / '_post_m2' / 'LATEST_POST_M2_SESSION.json'
    assert ledger.is_file()
