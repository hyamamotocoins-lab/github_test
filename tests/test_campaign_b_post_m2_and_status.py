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
    assert status['write_status_snapshot'] is False
    assert 'status_snapshot_path' not in status
    assert status['queues']['gpu_m3'] == 2
    assert status['queues']['m6'] == 3
    assert status['candidate_states']['SELECTED'] == 1
    assert status['candidate_states']['WAITING_FOR_M2'] == 1
    assert status['m2']['m2_ready_count'] == 1
    assert status['certification_status'] == CERTIFICATION_STATUS
    # Default path must not write under persist (quota-safe / truly read-only).
    assert not (tmp_path / 'campaign_b' / '_status_dashboard').exists()
    # Ensure queue.json not mutated
    q = (camp / 'queue.json').read_text(encoding='utf-8')
    assert 'SELECTED' in q


def test_status_optional_snapshot_write(tmp_path: Path) -> None:
    with (
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', return_value=[]),
        patch('src.campaign_b.close_obligations.list_obligation_queue', return_value=[]),
        patch('src.campaign_b.m6_batch.list_m6_queue', return_value=[]),
        patch(
            'src.campaign_b.advance_selected.discover_selected_packages',
            return_value=[],
        ),
    ):
        status = collect_pipeline_status(tmp_path, write_status_snapshot=True)
    dash = tmp_path / 'campaign_b' / '_status_dashboard'
    assert dash.is_dir()
    assert (dash / 'LATEST_STATUS.json').is_file()
    assert status['write_status_snapshot'] is True
    assert status['status_snapshot_path']
    assert Path(status['status_snapshot_path']).is_file()


def test_find_m2_ready_markers(tmp_path: Path) -> None:
    assert find_m2_ready_markers(tmp_path) == []
    p = tmp_path / 'runs' / 'M2-1' / 'M2_READY.json'
    p.parent.mkdir(parents=True)
    atomic_write_json(p, {'ready': True, 'sectors_done': 3})
    markers = find_m2_ready_markers(tmp_path)
    assert len(markers) == 1
    assert markers[0]['run_id'] == 'M2-1'


def test_post_m2_defaults_drain_via_pipeline_to_m6(tmp_path: Path) -> None:
    """Default 97 path = 95-equivalent drain (no screening)."""
    fake = {
        'session_id': 'PIPE-mock',
        'started_at': 't0',
        'rounds_run': 2,
        'totals': {'m3_complete': 3, 'advanced': 1},
        'auto_strip_m3_checkpoints': True,
        'auto_keep_latest_m3_checkpoint': True,
        'persist_m3_cap_gib': 80.0,
        'm3_reclaim': {
            'stripped': 2,
            'bytes_freed': 1000,
            'bytes_freed_human': '1000 B',
            'keep_latest_bytes_freed_human': '0 B',
        },
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    }
    with (
        patch(
            'src.campaign_b.pipeline_to_m6.run_pipeline_to_m6',
            return_value=fake,
        ) as pipe,
        patch('src.campaign_b.end_to_end.run_end_to_end') as e2e,
    ):
        summary = run_post_m2_pipeline(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=5,
            max_m3_sessions=16,
        )
    assert pipe.called
    assert not e2e.called
    kwargs = pipe.call_args.kwargs
    assert kwargs['max_rounds'] == 5
    assert kwargs['max_m3_sessions'] == 16
    assert kwargs['auto_strip_m3_checkpoints'] is True
    assert kwargs['auto_keep_latest_m3_checkpoint'] is True
    assert kwargs['persist_m3_cap_gib'] == 80.0
    assert summary['notebook'] == 97
    assert summary['mode'] == 'drain_existing_backlog'
    assert summary['drain_existing_backlog'] is True
    assert summary['skip_screening'] is True
    assert summary['auto_strip_m3_checkpoints'] is True
    assert summary['auto_keep_latest_m3_checkpoint'] is True
    assert summary['m3_reclaim']['stripped'] == 2
    assert summary['gpu_workers'] == 1
    assert summary['pipeline_to_m6']['session_id'] == 'PIPE-mock'
    assert summary['pipeline_to_m6']['totals']['m3_complete'] == 3
    assert 'end_to_end' not in summary
    assert '95-equivalent' in summary['note']
    assert 'backlog growth is OK' in summary['note']
    assert 'Auto-strip M3 checkpoints ON' in summary['note']
    assert summary['certification_status'] == CERTIFICATION_STATUS
    ledger = tmp_path / 'campaign_b' / '_post_m2' / 'LATEST_POST_M2_SESSION.json'
    assert ledger.is_file()


def test_post_m2_can_disable_auto_strip(tmp_path: Path) -> None:
    fake = {
        'session_id': 'PIPE-mock',
        'started_at': 't0',
        'rounds_run': 1,
        'totals': {},
        'auto_strip_m3_checkpoints': False,
        'm3_reclaim': {'stripped': 0, 'bytes_freed': 0, 'bytes_freed_human': '0 B'},
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    }
    with patch(
        'src.campaign_b.pipeline_to_m6.run_pipeline_to_m6',
        return_value=fake,
    ) as pipe:
        summary = run_post_m2_pipeline(
            persistent_root=tmp_path,
            project_root=tmp_path,
            auto_strip_m3_checkpoints=False,
        )
    assert pipe.call_args.kwargs['auto_strip_m3_checkpoints'] is False
    assert summary['auto_strip_m3_checkpoints'] is False
    assert 'Auto-strip M3 checkpoints OFF' in summary['note']


def test_post_m2_opt_in_end_to_end_screening_path(tmp_path: Path) -> None:
    fake = {
        'session_id': 'E2E-mock',
        'started_at': 't0',
        'rounds_run': 1,
        'totals': {'m3_complete': 0},
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    }
    with (
        patch(
            'src.campaign_b.end_to_end.run_end_to_end',
            return_value=fake,
        ) as mocked,
        patch('src.campaign_b.pipeline_to_m6.run_pipeline_to_m6') as pipe,
    ):
        summary = run_post_m2_pipeline(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=2,
            drain_existing_backlog=False,
            skip_screening=True,
        )
    assert mocked.called
    assert not pipe.called
    e2e_cfg = mocked.call_args.args[0]
    assert e2e_cfg.skip_screening is True
    assert summary['mode'] == 'end_to_end'
    assert summary['drain_existing_backlog'] is False
    assert summary['end_to_end']['session_id'] == 'E2E-mock'
    assert 'WAITING_FOR_M2' in summary['waiting_for_m2_todo']
    assert summary['certification_status'] == CERTIFICATION_STATUS
