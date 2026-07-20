from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import CheckpointManager, ConfigMismatchError, RunState
from src.common import sha256_file, utc_now
from src.m2_config import M2Config
from src.m2_orchestrator import create_or_resume_m2
from src.m2_parent import M2ParentError, verify_accepted_m1_parent
from src.session_guard import SessionGuard
from src.work_queue import WorkQueue
from tests.m2_helpers import make_synthetic_accepted_m1, passing_m2_test_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_accepted_m1_parent_is_content_pinned_and_tamper_fails(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m1(tmp_path)
    verified = verify_accepted_m1_parent(project, config)
    assert verified['m1_report_sha256'] == sha256_file(
        Path(config.parent_report_path),
    )
    Path(config.parent_report_path).write_text('{}', encoding='utf-8')
    with pytest.raises(M2ParentError, match='report'):
        verify_accepted_m1_parent(project, config)


def test_m2_tensor_checkpoint_roundtrip_fallback_and_identity(tmp_path: Path) -> None:
    config = M2Config()
    run_root = tmp_path / 'runs' / 'M2-checkpoint-test'
    manager = CheckpointManager(
        run_root, config, source_hash='a' * 64, notebook_hash='b' * 64,
    )
    state = RunState(
        'M2-checkpoint-test', config.config_hash(), utc_now(), utc_now(),
        milestone='M2', phase='M2_RUNNING',
    )
    queue = WorkQueue()
    first_tensors = {
        'projector_111111': np.arange(4096, dtype=np.float64).reshape(64, 64),
    }
    first = manager.save(state, queue, first_tensors)
    second_tensors = {
        'projector_111111': np.eye(64, dtype=np.float64),
    }
    second = manager.save(state, queue, second_tensors)
    loaded = manager.load_latest(restore_rng=False)
    assert loaded is not None
    assert loaded.path == second.path
    np.testing.assert_array_equal(
        loaded.tensors['projector_111111'], second_tensors['projector_111111'],
    )

    shard = next(
        path for path in (second.path / 'tensors').rglob('*')
        if path.is_file() and path.name != 'index.json'
    )
    shard.write_bytes(b'corrupt')
    fallback = manager.load_latest(restore_rng=False)
    assert fallback is not None
    assert fallback.path == first.path
    assert fallback.skipped_invalid
    np.testing.assert_array_equal(
        fallback.tensors['projector_111111'], first_tensors['projector_111111'],
    )

    wrong_identity = CheckpointManager(
        run_root, config, source_hash='a' * 64, notebook_hash='c' * 64,
    )
    with pytest.raises(ConfigMismatchError, match='notebook'):
        wrong_identity.load_latest(restore_rng=False)


def test_m2_checkpoint_resume_and_fresh_process(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m1(tmp_path)
    persistent = tmp_path / 'persist'
    first = create_or_resume_m2(
        persistent, config, project, run_id='M2-restart-test',
        test_report=passing_m2_test_report(),
    )
    assert first.run_one_item_for_test() == 'M2_WIGNER_CACHE'
    resumed = create_or_resume_m2(
        persistent, config, project, run_id='M2-restart-test',
        test_report=passing_m2_test_report(),
    )
    assert sum(item.status == 'done' for item in resumed.queue.items.values()) == 1

    code = (
        'from pathlib import Path; '
        'from src.m2_config import M2Config; '
        'from src.m2_orchestrator import create_or_resume_m2; '
        f'c=M2Config(parent_run_id={config.parent_run_id!r},'
        f'parent_checkpoint_path={config.parent_checkpoint_path!r},'
        f'parent_report_path={config.parent_report_path!r},'
        f'parent_acceptance_path={config.parent_acceptance_path!r}); '
        f'o=create_or_resume_m2(Path({str(persistent)!r}),c,'
        f'Path({str(project)!r}),run_id="M2-restart-test"); '
        'assert sum(i.status=="done" for i in o.queue.items.values())==1; '
        'print(o.state.phase)'
    )
    environment = os.environ.copy()
    environment['PYTHONPATH'] = (
        str(PROJECT_ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    )
    completed = subprocess.run(
        [sys.executable, '-c', code], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, timeout=120, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'M2_BOOTSTRAP' in completed.stdout


def test_m2_session_drain_checkpoints_and_resumes(tmp_path: Path) -> None:
    base = M2Config(
        checkpoint_interval_s=1.0, max_work_item_s=2.0,
        short_task_limit_s=0.5, checkpoint_reserve_s=0.1,
        no_long_task_after_s=3.0, drain_after_s=4.0,
        final_save_after_s=5.0, hard_return_s=6.0,
    )
    config, project = make_synthetic_accepted_m1(tmp_path, base)
    orchestrator = create_or_resume_m2(
        tmp_path / 'persist', config, project,
        run_id='M2-drain-test', test_report=passing_m2_test_report(),
    )
    clock = FakeClock()
    orchestrator.guard = SessionGuard(config, clock=clock)
    clock.value = base.drain_after_s
    summary = orchestrator.run_until_checkpoint()
    assert 'drain checkpoint complete' in summary['stop_reason']
    resumed = create_or_resume_m2(
        tmp_path / 'persist', config, project,
        run_id='M2-drain-test', test_report=passing_m2_test_report(),
    )
    assert resumed.state.phase == 'M2_RUNNING'


def test_full_m2_run_writes_acceptance_and_stays_not_certified(
    tmp_path: Path,
) -> None:
    config, project = make_synthetic_accepted_m1(tmp_path)
    orchestrator = create_or_resume_m2(
        tmp_path / 'persist', config, project,
        run_id='M2-full-test', test_report=passing_m2_test_report(),
    )
    summary = orchestrator.run_until_checkpoint()
    assert summary['phase'] == 'M2_COMPLETE'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert (orchestrator.run_root / 'reports' / 'M2_report.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'M2_report.md').is_file()
    assert (orchestrator.run_root / 'reports' / 'M2_acceptance.json').is_file()
    assert all(item.status == 'done' for item in orchestrator.queue.items.values())


@pytest.mark.gpu
def test_m2_optional_gpu_smoke_keeps_exact_proof_path_on_cpu() -> None:
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    from src.runtime import configure_numerics

    configure_numerics()
    tensor = torch.eye(2, dtype=torch.float64, device='cuda')
    assert float(torch.linalg.det(tensor).cpu()) == 1.0
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert M2Config().certification_status == 'NOT_CERTIFIED'
