"""CPU mock tests for notebook 96 backlog-aware end-to-end scheduler."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.campaign_b.end_to_end import EndToEndConfig, run_end_to_end
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE


def _stage(**kwargs: Any) -> dict[str, Any]:
    base = {
        'advanced': 0,
        'discovered': 0,
        'ready_for_m3': 0,
        'errors': 0,
        'queue_size': 0,
        'sessions_ok': 0,
        'm3_complete': 0,
        'm3_checkpoint': 0,
        'sessions_error': 0,
        'packages_attempted': 0,
        'pre_m6_ready': 0,
        'm4_checkpoint': 0,
        'attempted': 0,
        'all_closed_count': 0,
        'm5_complete_count': 0,
        'still_open': [],
        'm6_complete': 0,
        'm6_certified_count': 0,
        'm6_not_certified_count': 0,
        'results': [],
        'selected_total': 0,
        'archived_total': 0,
        'waves': [],
        'session_id': 'MOCK',
    }
    base.update(kwargs)
    return base


def _cfg(tmp_path: Path, **kwargs: Any) -> EndToEndConfig:
    defaults = dict(
        persistent_root=tmp_path,
        project_root=tmp_path,
        selected_backlog_target=8,
        max_rounds=5,
        skip_screening=True,
        screening_chunk_size=32,
        # Keep unit tests free of real reclaim I/O unless a test opts in.
        auto_strip_m3_checkpoints=False,
        auto_keep_latest_m3_checkpoint=False,
        persist_m3_cap_gib=None,
    )
    defaults.update(kwargs)
    return EndToEndConfig(**defaults)


def test_idle_stop_when_no_completions(tmp_path: Path) -> None:
    empty = _stage()
    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
    ):
        summary = run_end_to_end(_cfg(tmp_path, skip_screening=True))

    assert summary['rounds_run'] == 1
    assert summary['rounds'][0]['progress'] == 0
    assert summary['certification_status'] == CERTIFICATION_STATUS
    assert summary['claim_scope'] == CLAIM_SCOPE
    ledger = tmp_path / 'campaign_b' / '_end_to_end' / 'LATEST_END_TO_END_SESSION.json'
    assert ledger.is_file()


def test_m3_checkpoint_alone_not_progress(tmp_path: Path) -> None:
    """m3_checkpoint without m3_complete must not keep the loop alive."""
    responses = [_stage(m3_checkpoint=3), _stage()]
    idx = {'i': 0}

    def _m3(**_kwargs: Any) -> dict[str, Any]:
        i = min(idx['i'], len(responses) - 1)
        idx['i'] += 1
        return responses[i]

    empty = _stage()
    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', side_effect=_m3),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
    ):
        summary = run_end_to_end(_cfg(tmp_path, skip_screening=True, max_rounds=5))

    assert summary['rounds_run'] == 1
    assert summary['rounds'][0]['progress'] == 0


def test_backlog_gate_skips_screen_and_advance_when_full(tmp_path: Path) -> None:
    empty = _stage()
    # Queue length >= target → gate closed.
    full_queue = [{'package': f'p{i}'} for i in range(8)]
    adv_calls = {'n': 0}
    screen_calls = {'n': 0}

    def _adv(**_kwargs: Any) -> dict[str, Any]:
        adv_calls['n'] += 1
        return empty

    def _screen(_cfg: Any) -> dict[str, Any]:
        screen_calls['n'] += 1
        return _stage(selected_total=1)

    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=full_queue),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', side_effect=_adv),
        patch('src.campaign_b.end_to_end._run_screening_chunk', side_effect=_screen),
    ):
        summary = run_end_to_end(
            _cfg(tmp_path, skip_screening=False, selected_backlog_target=8),
        )

    assert summary['rounds'][0]['backlog_gate_open'] is False
    assert summary['rounds'][0]['m3_queue_len'] == 8
    assert adv_calls['n'] == 0
    assert screen_calls['n'] == 0
    assert summary['rounds'][0]['stages']['screening'].get('skipped') is True


def test_backlog_gate_opens_screen_when_thin(tmp_path: Path) -> None:
    empty = _stage()
    thin_queue = [{'package': 'p0'}]  # len 1 < 8
    screen_calls = {'n': 0}

    def _screen(_cfg: Any) -> dict[str, Any]:
        screen_calls['n'] += 1
        # Only first wave yields SELECTED; later idle.
        if screen_calls['n'] == 1:
            return _stage(selected_total=2)
        return _stage(selected_total=0)

    # Round1: m3_complete + screening selected → progress; later idle.
    m3_q = [_stage(m3_complete=1), _stage()]
    adv_q = [_stage(advanced=1), _stage()]

    def _pop(queue: list[dict[str, Any]]):
        def _inner(**_kwargs: Any) -> dict[str, Any]:
            return queue.pop(0) if queue else empty
        return _inner

    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=thin_queue),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', side_effect=_pop(m3_q)),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.advance_selected.run_advance_selected',
            side_effect=_pop(adv_q),
        ),
        patch('src.campaign_b.end_to_end._run_screening_chunk', side_effect=_screen),
    ):
        summary = run_end_to_end(
            _cfg(tmp_path, skip_screening=False, selected_backlog_target=8, max_rounds=5),
        )

    assert summary['rounds'][0]['backlog_gate_open'] is True
    assert screen_calls['n'] >= 1
    assert summary['rounds'][0]['progress'] == 1 + 2 + 1  # m3 + selected + advanced
    assert summary['totals']['selected_from_screening'] == 2
    assert summary['rounds_run'] == 2


def test_sets_disable_wallclock_env(tmp_path: Path, monkeypatch: Any) -> None:
    import os

    monkeypatch.delenv('VALIDATED_RG_DISABLE_SESSION_WALLCLOCK', raising=False)
    empty = _stage()
    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
    ):
        run_end_to_end(_cfg(tmp_path))
    assert os.environ.get('VALIDATED_RG_DISABLE_SESSION_WALLCLOCK') == '1'


def test_m3_reclaim_session_start_and_per_round(tmp_path: Path) -> None:
    """Mirror notebook 97: session-start keep-latest + strip, per-round strip."""
    empty = _stage()
    reclaim_calls: list[dict[str, Any]] = []

    def _reclaim(*_a: Any, **kwargs: Any) -> dict[str, Any]:
        reclaim_calls.append(dict(kwargs))
        if kwargs.get('force_full_scan'):
            return {
                'stripped': 2,
                'bytes_freed': 50,
                'force_full_scan': True,
            }
        return {
            'stripped': 1,
            'bytes_freed': 34,
            'force_full_scan': False,
        }

    m3_q = [_stage(m3_complete=1, keep_latest_bytes_freed=7), _stage()]

    def _m3(**kwargs: Any) -> dict[str, Any]:
        assert kwargs.get('auto_keep_latest_m3_checkpoint') is True
        return m3_q.pop(0) if m3_q else empty

    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', side_effect=_m3),
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch(
            'src.campaign_b.m3_reclaim.auto_strip_after_pipeline_round',
            side_effect=_reclaim,
        ),
        patch('src.campaign_b.m3_reclaim.keep_latest_all_m3_runs') as keep_latest,
    ):
        keep_latest.return_value.as_dict.return_value = {
            'execute': True,
            'scope': 'full_scan',
            'candidates': 3,
            'trimmed': 2,
            'stripped': 2,
            'skipped': 1,
            'bytes_freed': 99,
            'bytes_freed_human': '99 B',
            'run_ids': ['M3-a', 'M3-b'],
            'actions': [],
        }
        summary = run_end_to_end(
            _cfg(
                tmp_path,
                max_rounds=2,
                auto_strip_m3_checkpoints=True,
                auto_keep_latest_m3_checkpoint=True,
                persist_m3_cap_gib=80.0,
            ),
        )

    assert len(reclaim_calls) == 3  # session-start + round1 + round2 (idle stop)
    # First call is session-start full scan; later calls are per-round.
    assert reclaim_calls[0]['force_full_scan'] is True
    assert reclaim_calls[0]['persist_m3_cap_gib'] == 80.0
    assert reclaim_calls[1]['force_full_scan'] is False
    assert keep_latest.called
    assert summary['auto_strip_m3_checkpoints'] is True
    assert summary['auto_keep_latest_m3_checkpoint'] is True
    assert summary['persist_m3_cap_gib'] == 80.0
    assert summary['m3_reclaim']['session_start_full_scan'] is not None
    assert summary['m3_reclaim']['session_start_keep_latest']['trimmed'] == 2
    assert summary['m3_reclaim']['stripped'] == 2 + 1 + 1  # session + 2 rounds
    assert summary['m3_reclaim']['keep_latest_bytes_freed'] == 99 + 7
    assert 'm3_reclaim' in summary['rounds'][0]
    assert 'Auto-strip M3 checkpoints ON' in summary['note']
    assert 'Keep-latest ON' in summary['note']


def test_m3_reclaim_off_skips_hooks(tmp_path: Path) -> None:
    empty = _stage()
    with (
        patch('src.campaign_b.pipeline_recovery.recover_interrupted_work', return_value={}),
        patch('src.campaign_b.execution_keys.acquire_gpu_lock', return_value={}),
        patch('src.campaign_b.execution_keys.release_gpu_lock', return_value=True),
        patch('src.campaign_b.gpu_m3_batch.list_gpu_m3_queue', return_value=[]),
        patch('src.campaign_b.gpu_m3_batch.run_gpu_m3_batch', return_value=empty) as m3,
        patch('src.campaign_b.pre_m6_batch.run_pre_m6_batch', return_value=empty),
        patch(
            'src.campaign_b.close_obligations.run_close_obligations_batch',
            return_value=empty,
        ),
        patch('src.campaign_b.m6_batch.run_m6_batch', return_value=empty),
        patch('src.campaign_b.advance_selected.run_advance_selected', return_value=empty),
        patch('src.campaign_b.m3_reclaim.auto_strip_after_pipeline_round') as strip,
        patch('src.campaign_b.m3_reclaim.keep_latest_all_m3_runs') as keep_latest,
    ):
        summary = run_end_to_end(
            _cfg(
                tmp_path,
                auto_strip_m3_checkpoints=False,
                auto_keep_latest_m3_checkpoint=False,
            ),
        )

    assert not strip.called
    assert not keep_latest.called
    assert m3.call_args.kwargs.get('auto_keep_latest_m3_checkpoint') is False
    assert summary['auto_strip_m3_checkpoints'] is False
    assert summary['auto_keep_latest_m3_checkpoint'] is False
    assert summary['m3_reclaim']['session_start_keep_latest'] is None
    assert summary['m3_reclaim']['stripped'] == 0
    assert 'm3_reclaim' not in summary['rounds'][0]
