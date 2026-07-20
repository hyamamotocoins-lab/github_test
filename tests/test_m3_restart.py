from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import CheckpointManager, RunState
from src.common import utc_now
from src.m3_config import M3Config
from src.m3_orchestrator import create_or_resume_m3
from src.m3_parent import M3ParentError, verify_accepted_m2_parent
from src.session_guard import SessionGuard
from src.work_queue import WorkQueue
from tests.m3_helpers import make_synthetic_accepted_m2, passing_m3_test_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def _cpu_config(**updates: object) -> M3Config:
    return M3Config(require_cuda=False, **updates)


def test_accepted_m2_parent_tensors_are_pinned_and_tamper_fails(
    tmp_path: Path,
) -> None:
    config, project = make_synthetic_accepted_m2(
        tmp_path, _cpu_config(),
    )
    evidence = verify_accepted_m2_parent(project, config)
    assert len(evidence.projector_tensors) == 64
    Path(config.parent_report_path).write_text('{}', encoding='utf-8')
    with pytest.raises(M3ParentError, match='report'):
        verify_accepted_m2_parent(project, config)


def test_m3_checkpoint_restores_same_basis_candidate(tmp_path: Path) -> None:
    config = _cpu_config()
    run_root = tmp_path / 'runs/M3-basis-check'
    manager = CheckpointManager(
        run_root, config, source_hash='3' * 64, notebook_hash='4' * 64,
    )
    state = RunState(
        'M3-basis-check', config.config_hash(), utc_now(), utc_now(),
        milestone='M3', phase='M3_RUNNING',
    )
    queue = WorkQueue()
    rng = np.random.default_rng(31)
    tensors = {
        'rsvd_left': rng.standard_normal((729, 16)),
        'rsvd_singular_values': np.linspace(1.0, 0.1, 16),
        'rsvd_right_t': rng.standard_normal((16, 729)),
    }
    saved = manager.save(state, queue, tensors)
    loaded = manager.load_latest(restore_rng=False)
    assert loaded is not None and loaded.path == saved.path
    assert set(loaded.tensors) == set(tensors)
    for name in tensors:
        np.testing.assert_array_equal(loaded.tensors[name], tensors[name])


def test_m3_checkpoint_resume_and_fresh_process(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m2(
        tmp_path, _cpu_config(),
    )
    persistent = tmp_path / 'persist'
    first = create_or_resume_m3(
        persistent, config, project, run_id='M3-restart-test',
        test_report=passing_m3_test_report(),
    )
    assert first.run_one_item_for_test() == 'M3_BACKEND_DIAGNOSTIC'
    resumed = create_or_resume_m3(
        persistent, config, project, run_id='M3-restart-test',
        test_report=passing_m3_test_report(),
    )
    assert sum(item.status == 'done' for item in resumed.queue.items.values()) == 1

    code = (
        'from pathlib import Path; '
        'from src.m3_config import M3Config; '
        'from src.m3_orchestrator import create_or_resume_m3; '
        f'c=M3Config(require_cuda=False,parent_run_id={config.parent_run_id!r},'
        f'parent_checkpoint_path={config.parent_checkpoint_path!r},'
        f'parent_report_path={config.parent_report_path!r},'
        f'parent_acceptance_path={config.parent_acceptance_path!r}); '
        f'o=create_or_resume_m3(Path({str(persistent)!r}),c,'
        f'Path({str(project)!r}),run_id="M3-restart-test"); '
        'assert sum(i.status=="done" for i in o.queue.items.values())==1; '
        'print(o.state.phase)'
    )
    environment = os.environ.copy()
    environment['PYTHONPATH'] = (
        str(PROJECT_ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    )
    completed = subprocess.run(
        [sys.executable, '-c', code], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, timeout=180, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'M3_BOOTSTRAP' in completed.stdout


def test_m3_session_drain_checkpoints_and_resumes(tmp_path: Path) -> None:
    base = _cpu_config(
        checkpoint_interval_s=1.0, max_work_item_s=2.0,
        short_task_limit_s=0.5, checkpoint_reserve_s=0.1,
        no_long_task_after_s=3.0, drain_after_s=4.0,
        final_save_after_s=5.0, hard_return_s=6.0,
    )
    config, project = make_synthetic_accepted_m2(tmp_path, base)
    orchestrator = create_or_resume_m3(
        tmp_path / 'persist', config, project,
        run_id='M3-drain-test', test_report=passing_m3_test_report(),
    )
    clock = FakeClock()
    orchestrator.guard = SessionGuard(config, clock=clock)
    clock.value = base.drain_after_s
    summary = orchestrator.run_until_checkpoint()
    assert 'drain checkpoint complete' in summary['stop_reason']
    resumed = create_or_resume_m3(
        tmp_path / 'persist', config, project,
        run_id='M3-drain-test', test_report=passing_m3_test_report(),
    )
    assert resumed.state.phase == 'M3_RUNNING'


@pytest.mark.gpu
def test_full_m3_gpu_run_writes_core_reproduced_report(tmp_path: Path) -> None:
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    config, project = make_synthetic_accepted_m2(tmp_path)
    orchestrator = create_or_resume_m3(
        tmp_path / 'persist', config, project,
        run_id='M3-full-gpu-test', test_report=passing_m3_test_report(),
    )
    summary = orchestrator.run_until_checkpoint()
    assert summary['phase'] == 'M3_COMPLETE'
    assert summary['milestone_status'] == 'CORE_REPRODUCED'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert (orchestrator.run_root / 'reports/M3_report.json').is_file()
    assert (orchestrator.run_root / 'reports/M3_acceptance.json').is_file()
    loaded = orchestrator.checkpoints.load_latest(restore_rng=False)
    assert loaded is not None
    assert {
        'rsvd_left', 'rsvd_singular_values', 'rsvd_right_t',
        'triad_left', 'triad_core', 'triad_right',
    }.issubset(loaded.tensors)
