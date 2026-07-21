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
    sessions_attempted: int | None = None,
    sessions_error: int = 0,
    pre_m6_ready: int = 0,
    m4_checkpoint: int = 0,
    packages_attempted: int | None = None,
    all_closed_count: int = 0,
    m6_complete: int = 0,
    m6_certified_count: int = 0,
    m6_not_certified_count: int = 0,
) -> dict[str, Any]:
    if packages_attempted is None:
        packages_attempted = pre_m6_ready + m4_checkpoint
    if sessions_attempted is None:
        sessions_attempted = sessions_ok or (
            m3_complete + m3_checkpoint + sessions_error
        )
    return {
        'advanced': advanced,
        'discovered': advanced,
        'ready_for_m3': advanced,
        'errors': 0,
        'queue_size': 0,
        'sessions_ok': sessions_ok,
        'sessions_attempted': sessions_attempted,
        'm3_complete': m3_complete,
        'm3_checkpoint': m3_checkpoint,
        'sessions_error': sessions_error,
        'packages_attempted': packages_attempted,
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


def _empty_queues(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    return []


def _one_m3(*_args: Any, **_kwargs: Any) -> list[dict[str, Any]]:
    return [{'package': '/fake/pkg'}]


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
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=5,
            auto_strip_m3_checkpoints=False,
        )

    assert summary['rounds_run'] == 1
    assert call_counts['n'] == 1
    assert summary['stop_reason'] == 'DRAINED_OR_IDLE'
    assert summary['certification_status'] == CERTIFICATION_STATUS
    assert summary['claim_scope'] == CLAIM_SCOPE
    assert '81' in summary['note']
    ledger = tmp_path / 'campaign_b' / '_pipeline_to_m6' / 'LATEST_PIPELINE_SESSION.json'
    assert ledger.is_file()
    written = json.loads(ledger.read_text(encoding='utf-8'))
    assert written['certification_status'] == CERTIFICATION_STATUS
    assert written['stop_reason'] == 'DRAINED_OR_IDLE'


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
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=10,
            auto_strip_m3_checkpoints=False,
        )

    assert summary['rounds_run'] == 3
    assert summary['stop_reason'] == 'DRAINED_OR_IDLE'
    assert summary['totals']['advanced'] == 2
    assert summary['totals']['m3_complete'] == 1
    assert summary['totals']['pre_m6_ready'] == 1
    assert summary['totals']['obligations_closed'] == 1
    assert summary['totals']['m6_complete'] == 1
    assert summary['totals']['m6_certified'] == 0
    assert summary['totals']['m6_not_certified'] == 1
    assert summary['certification_status'] == CERTIFICATION_STATUS


def test_pipeline_retries_failed_attempts_then_stuck(tmp_path: Path) -> None:
    """progress==0 with runnable + attempts continues until max_idle_rounds."""
    failed = _stage(sessions_attempted=2, sessions_error=2)
    failed['errors'] = [
        {
            'package': '/fake/pkg',
            'error': 'ValueError: Out of range float values are not JSON compliant',
            'status': 'M3_BLOCKED_NONFINITE',
        },
    ]
    empty = _stage()
    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=failed),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_one_m3),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=10,
            max_idle_rounds=2,
            auto_strip_m3_checkpoints=False,
        )

    assert summary['rounds_run'] == 2
    assert summary['stop_reason'] == 'STUCK_BACKLOG'
    assert summary['remaining_runnable']['gpu_m3'] == 1
    diag = summary.get('stuck_diagnostics')
    assert isinstance(diag, dict)
    assert diag.get('sessions_error') == 2
    assert diag['m3_errors'][0]['error'].startswith('ValueError:')


def test_pipeline_no_attempts_with_backlog(tmp_path: Path) -> None:
    """progress==0 + runnable + zero attempts → misconfig stop immediately."""
    empty = _stage(sessions_attempted=0, packages_attempted=0)
    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_one_m3),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=10,
            auto_strip_m3_checkpoints=False,
        )

    assert summary['rounds_run'] == 1
    assert summary['stop_reason'] == 'NO_ATTEMPTS_WITH_BACKLOG'


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
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
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
            auto_strip_m3_checkpoints=False,
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
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=5,
            skip_advance=True,
            skip_m3=True,
            skip_obligations=True,
            skip_m6=True,
            auto_strip_m3_checkpoints=False,
        )

    assert summary['rounds_run'] == 2
    assert summary['rounds'][0]['progress'] == 1
    assert summary['stop_reason'] == 'DRAINED_OR_IDLE'


def test_pipeline_auto_strips_when_flag_set(tmp_path: Path) -> None:
    empty = _stage()
    reclaim_calls: list[dict[str, Any]] = []

    def _reclaim(root: Path, **kwargs: Any) -> dict[str, Any]:
        reclaim_calls.append({'root': root, **kwargs})
        return {
            'execute': True,
            'scope': 'full_scan' if kwargs.get('force_full_scan') else 'incremental',
            'candidates': 0,
            'stripped': 1,
            'skipped': 0,
            'bytes_freed': 42,
            'bytes_freed_human': '42 B',
            'run_ids': ['M3-x'],
            'actions': [],
            'preferred_run_ids': [],
            'fallback_full_scan': True,
            'force_full_scan': bool(kwargs.get('force_full_scan')),
        }

    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.m3_reclaim.auto_strip_after_pipeline_round',
            side_effect=_reclaim,
        ),
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=1,
            auto_strip_m3_checkpoints=True,
        )

    # Session-start full scan + per-round strip.
    assert len(reclaim_calls) == 2
    assert reclaim_calls[0]['force_full_scan'] is True
    assert reclaim_calls[1]['force_full_scan'] is False
    assert summary['auto_strip_m3_checkpoints'] is True
    assert summary['m3_reclaim']['stripped'] == 2
    assert summary['m3_reclaim']['bytes_freed'] == 84
    assert summary['m3_reclaim']['session_start_full_scan'] is not None
    assert summary['totals']['m3_checkpoints_stripped'] == 2
    assert summary['rounds'][0]['m3_reclaim']['stripped'] == 1
    assert summary['stop_reason'] == 'DRAINED_OR_IDLE'


def test_pipeline_skips_auto_strip_when_disabled(tmp_path: Path) -> None:
    empty = _stage()
    with (
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', side_effect=_empty_queues),
        patch('src.campaign_b.pre_m6_batch.list_pre_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.close_obligations.list_obligation_queue',
            side_effect=_empty_queues,
        ),
        patch('src.campaign_b.m6_batch.list_m6_queue', side_effect=_empty_queues),
        patch(
            'src.campaign_b.m3_reclaim.auto_strip_after_pipeline_round',
        ) as reclaim,
    ):
        summary = run_pipeline_to_m6(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_rounds=1,
            auto_strip_m3_checkpoints=False,
        )

    assert not reclaim.called
    assert summary['auto_strip_m3_checkpoints'] is False
    assert summary['m3_reclaim']['stripped'] == 0
    assert 'm3_reclaim' not in summary['rounds'][0]
    assert summary['stop_reason'] == 'DRAINED_OR_IDLE'
