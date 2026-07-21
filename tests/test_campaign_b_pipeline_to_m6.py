"""CPU unit tests for Campaign B pipeline 90→94 orchestrator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.campaign_b.pipeline_to_m6 import run_pipeline_to_m6
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE


def _stage(
    *,
    advanced: int = 0,
    m3_complete: int = 0,
    m3_checkpoint: int = 0,
    sessions_ok: int = 0,
    pre_m6_ready: int = 0,
    m4_checkpoint: int = 0,
    all_closed_count: int = 0,
    m6_complete: int = 0,
    m6_certified_count: int = 0,
    m6_not_certified_count: int = 0,
) -> dict[str, Any]:
    return {
        'advanced': advanced,
        'discovered': advanced,
        'ready_for_m3': advanced,
        'errors': 0,
        'queue_size': 0,
        'sessions_ok': sessions_ok,
        'm3_complete': m3_complete,
        'm3_checkpoint': m3_checkpoint,
        'sessions_error': 0,
        'packages_attempted': pre_m6_ready + m4_checkpoint,
        'pre_m6_ready': pre_m6_ready,
        'm4_checkpoint': m4_checkpoint,
        'attempted': all_closed_count + m6_complete,
        'all_closed_count': all_closed_count,
        'm5_complete_count': all_closed_count,
        'still_open': [],
        'm6_complete': m6_complete,
        'm6_certified_count': m6_certified_count,
        'm6_not_certified_count': m6_not_certified_count,
        'results': [],
    }


def test_pipeline_stops_when_no_progress(tmp_path: Path) -> None:
    empty = _stage()
    call_counts = {'n': 0}

    def _advance(**_kwargs: Any) -> dict[str, Any]:
        call_counts['n'] += 1
        return empty

    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', side_effect=_advance),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=5,
        )

    assert summary['rounds_run'] == 1
    assert call_counts['n'] == 1
    assert summary['certification_status'] == CERTIFICATION_STATUS
    assert summary['claim_scope'] == CLAIM_SCOPE
    assert '81' in summary['note']
    ledger = tmp_path / 'campaign_b' / '_pipeline_to_m6' / 'LATEST_PIPELINE_SESSION.json'
    assert ledger.is_file()
    written = json.loads(ledger.read_text(encoding='utf-8'))
    assert written['certification_status'] == CERTIFICATION_STATUS


def test_pipeline_multi_round_then_idle(tmp_path: Path) -> None:
    # Per-stage queues: round1 advances+completes M3; round2 closes+M6; round3 idle.
    adv_q = [_stage(advanced=2), _stage(), _stage()]
    m3_q = [_stage(m3_complete=1), _stage(), _stage()]
    pre_q = [_stage(), _stage(pre_m6_ready=1), _stage()]
    obl_q = [_stage(), _stage(all_closed_count=1), _stage()]
    m6_q = [
        _stage(),
        _stage(m6_complete=1, m6_not_certified_count=1),
        _stage(),
    ]

    def _pop(queue: list[dict[str, Any]]):
        def _inner(**_kwargs: Any) -> dict[str, Any]:
            if not queue:
                return _stage()
            return queue.pop(0)
        return _inner

    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', side_effect=_pop(adv_q)),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', side_effect=_pop(m3_q)),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', side_effect=_pop(pre_q)),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            side_effect=_pop(obl_q),
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', side_effect=_pop(m6_q)),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=10,
        )

    assert summary['rounds_run'] == 3
    assert summary['totals']['advanced'] == 2
    assert summary['totals']['m3_complete'] == 1
    assert summary['totals']['pre_m6_ready'] == 1
    assert summary['totals']['obligations_closed'] == 1
    assert summary['totals']['m6_complete'] == 1
    assert summary['totals']['m6_certified'] == 0
    assert summary['totals']['m6_not_certified'] == 1
    assert summary['certification_status'] == CERTIFICATION_STATUS


def test_pipeline_passes_stage_kwargs(tmp_path: Path) -> None:
    captured: dict[str, Any] = {}
    empty = _stage()

    def _adv(**kwargs: Any) -> dict[str, Any]:
        captured['advance'] = kwargs
        return empty

    def _m3(**kwargs: Any) -> dict[str, Any]:
        captured['m3'] = kwargs
        return empty

    def _pre(**kwargs: Any) -> dict[str, Any]:
        captured['pre'] = kwargs
        return empty

    def _obl(**kwargs: Any) -> dict[str, Any]:
        captured['obl'] = kwargs
        return empty

    def _m6(**kwargs: Any) -> dict[str, Any]:
        captured['m6'] = kwargs
        return empty

    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', side_effect=_adv),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', side_effect=_m3),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', side_effect=_pre),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            side_effect=_obl,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', side_effect=_m6),
    ):
        run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path / 'proj',
            max_rounds=1,
            max_advance=3,
            max_m3_sessions=4,
            max_pre_m6_packages=5,
            max_stage_sessions=7,
            max_obligation_packages=8,
            max_m6_packages=9,
            max_queue=100,
            only_campaign_run_id='M7-TEST',
        )

    assert captured['advance']['max_candidates'] == 3
    assert captured['advance']['only_campaign_run_id'] == 'M7-TEST'
    assert captured['m3']['max_sessions'] == 4
    assert captured['pre']['max_packages'] == 5
    assert captured['pre']['max_stage_sessions'] == 7
    assert captured['obl']['max_packages'] == 8
    assert captured['m6']['max_packages'] == 9
    assert captured['m6']['max_queue'] == 100


def test_pipeline_counts_m4_checkpoint_as_progress(tmp_path: Path) -> None:
    """M4 checkpoint alone must keep multi-round looping (resume path)."""
    responses = [_stage(m4_checkpoint=1), _stage()]
    idx = {'i': 0}

    def _pre(**_kwargs: Any) -> dict[str, Any]:
        i = min(idx['i'], len(responses) - 1)
        idx['i'] += 1
        return responses[i]

    empty = _stage()
    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', side_effect=_pre),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=5,
            skip_advance=True,
            skip_m3=True,
            skip_obligations=True,
            skip_m6=True,
        )

    assert summary['rounds_run'] == 2
    assert summary['rounds'][0]['progress'] == 1
