from __future__ import annotations

from pathlib import Path

import pytest

from src.checkpoint import CheckpointError, RunState
from src.common import utc_now
from src.m1_config import M1Config
from src.m1_reporting import validate_m1_acceptance
from src.work_queue import WorkQueue


def _state() -> RunState:
    config = M1Config()
    return RunState('M1-fail-closed', config.config_hash(), utc_now(), utc_now(), milestone='M1', phase='M1_RUNNING')


def test_m1_config_and_state_forbid_certified() -> None:
    with pytest.raises(ValueError):
        M1Config(certification_status='CERTIFIED')
    state = _state(); state.certification_status = 'CERTIFIED'
    with pytest.raises(CheckpointError):
        state.assert_m1_safe()


def test_missing_tail_artifact_prevents_m1_complete() -> None:
    with pytest.raises(RuntimeError, match='missing phase artifacts'):
        validate_m1_acceptance(_state(), WorkQueue(), {}, {})


def test_failed_independent_verifier_prevents_m1_complete() -> None:
    phases = (
        'M1_COEFFICIENT_BATCH', 'M1_VALUE_TAIL', 'M1_GRADIENT_TAIL',
        'M1_RG_TRAJECTORY', 'M1_INDEPENDENT_VERIFY', 'M1_REPORT',
    )
    queue = WorkQueue(); results = {}
    for phase in phases:
        item_id = queue.add(phase, 'input', {'phase': phase}, 1.0)
        item = queue.items[item_id]; item.status = 'done'
        item.result_relpath = f'{phase}.json'; item.result_sha256 = '0' * 64
        result = {}
        if phase == 'M1_COEFFICIENT_BATCH':
            result['rigor'] = 'RIGOROUS_RATIONAL_POSITIVE_SERIES'
        if phase in {'M1_VALUE_TAIL', 'M1_GRADIENT_TAIL'}:
            result['rigor'] = 'RIGOROUS_RATIONAL_ANALYTIC_BOUND'
        if phase == 'M1_RG_TRAJECTORY':
            result['rigor'] = 'RIGOROUS_RATIONAL_INTERVAL_RECURRENCE'
        if phase == 'M1_INDEPENDENT_VERIFY':
            result['status'] = 'FAIL'
        if phase == 'M1_REPORT':
            result['status'] = 'READY'
        results[phase] = {'phase': phase, 'result': result}
    tests = {
        'm0_regression_cpu_suite': 'PASS',
        'm1_required_cpu_suite': 'PASS', 'optional_gpu_suite': 'NOT_RUN_NO_CUDA',
        'm1_fresh_process_resume': 'PASS',
    }
    with pytest.raises(RuntimeError, match='independent_verifier'):
        validate_m1_acceptance(_state(), queue, results, tests)
