from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import CheckpointManager, RunState
from src.common import atomic_write_json, hash_tree, sha256_file, utc_now
from src.config import ConfigError, RunConfig
from src.orchestrator import RunCompatibilityError, create_or_resume
from src.runtime import PERSIST_ACK_TOKEN, PersistentRootError, validate_persistent_root
from src.session_guard import SessionGuard, SessionState
from src.work_queue import WorkItem, WorkQueue

try:
    import torch
except ImportError:
    torch = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_HASH = hash_tree(PROJECT_ROOT / 'src')


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def manager_for(run_root: Path, config: RunConfig) -> CheckpointManager:
    run_root.mkdir(parents=True, exist_ok=True)
    return CheckpointManager(run_root, config, SOURCE_HASH, None)


def new_state(config: RunConfig, run_id: str = 'run') -> RunState:
    return RunState(run_id, config.config_hash(), utc_now(), utc_now())


def test_config_is_immutable_and_fail_closed() -> None:
    config = RunConfig()
    assert config.certification_status == 'NOT_CERTIFIED'
    with pytest.raises(ConfigError):
        RunConfig(certification_status='CERTIFIED')
    with pytest.raises(ConfigError):
        RunConfig(hard_return_s=5.5 * 3600 + 1)
    with pytest.raises(ConfigError):
        RunConfig(dummy_size=0)
    with pytest.raises(ConfigError):
        RunConfig(dummy_predicted_s=21 * 60.0)
    with pytest.raises(ConfigError):
        RunConfig(prefer_cuda='yes')
    with pytest.raises(ConfigError):
        RunConfig(dummy_items=1.5)


def test_persistent_root_is_explicit_and_rejects_ephemeral(tmp_path: Path) -> None:
    with pytest.raises(PersistentRootError):
        validate_persistent_root(str(tmp_path), acknowledgement=None)
    with pytest.raises(PersistentRootError):
        validate_persistent_root('/tmp/validated-rg', acknowledgement=PERSIST_ACK_TOKEN)
    with pytest.raises(PersistentRootError):
        validate_persistent_root('/private/var/folders/validated-rg', acknowledgement=PERSIST_ACK_TOKEN)


def test_session_guard_boundaries_and_prediction() -> None:
    config = RunConfig(
        checkpoint_interval_s=1.0, max_work_item_s=2.0, short_task_limit_s=0.5,
        checkpoint_reserve_s=0.1, no_long_task_after_s=3.0, drain_after_s=4.0,
        final_save_after_s=5.0, hard_return_s=6.0, dummy_predicted_s=1.0,
    )
    clock = FakeClock()
    guard = SessionGuard(config, clock=clock)
    assert guard.state() is SessionState.RUN
    assert guard.may_start(1.0)
    assert not guard.checkpoint_due()
    clock.value = 1.0
    assert guard.checkpoint_due()
    guard.mark_checkpoint()
    assert not guard.checkpoint_due()
    clock.value = 3.0
    assert guard.state() is SessionState.NO_LONG_TASK
    assert not guard.may_start(1.0)
    clock.value = 4.0
    assert guard.state() is SessionState.DRAIN
    clock.value = 5.0
    assert guard.state() is SessionState.FINAL_SAVE
    clock.value = 6.0
    assert guard.state() is SessionState.RETURN


def test_work_queue_rejects_content_hash_drift() -> None:
    queue = WorkQueue()
    item_id = queue.add('DUMMY', 'input', {'index': 0}, 1.0)
    payload = queue.to_payload()
    payload['items'][item_id]['parameters']['index'] = 1
    with pytest.raises(ValueError, match='content hash mismatch'):
        WorkQueue.from_payload(payload)


def test_atomic_checkpoint_and_sharded_round_trip(tmp_path: Path) -> None:
    config = RunConfig(tensor_shard_bytes=64)
    manager = manager_for(tmp_path / 'run', config)
    state = new_state(config)
    queue = WorkQueue()
    queue.add('DUMMY', config.config_hash(), {'index': 0}, 1.0)
    array = np.arange(256, dtype=np.float64).reshape(32, 8)
    empty = np.empty((0, 8), dtype=np.float64)
    scalar = np.asarray(7.0, dtype=np.float64)
    saved = manager.save(state, queue, {'array': array, 'empty': empty, 'scalar': scalar})
    assert (saved.path / 'COMMITTED').is_file()
    manager.verify(saved.path)
    index = json.loads((saved.path / 'tensors' / 'index.json').read_text())
    assert len(index['array']['files']) > 1
    for filename in index['array']['files']:
        assert np.load(saved.path / 'tensors' / filename, allow_pickle=False).nbytes <= 64
    loaded = manager.load_latest(restore_rng=False)
    assert loaded is not None and loaded.path == saved.path
    np.testing.assert_array_equal(loaded.tensors['array'], array)
    restored = manager.load_tensors(saved.path)
    np.testing.assert_array_equal(restored['array'], array)
    np.testing.assert_array_equal(restored['empty'], empty)
    np.testing.assert_array_equal(restored['scalar'], scalar)


def test_corrupt_latest_falls_back(tmp_path: Path) -> None:
    config = RunConfig()
    manager = manager_for(tmp_path / 'run', config)
    state = new_state(config)
    queue = WorkQueue()
    first = manager.save(state, queue)
    state.notes.append('second')
    second = manager.save(state, queue)
    (second.path / 'state.json').write_text('corrupt', encoding='utf-8')
    loaded = manager.load_latest(restore_rng=False)
    assert loaded is not None
    assert loaded.path == first.path
    assert loaded.skipped_invalid and second.path.name in loaded.skipped_invalid[0]


def test_rng_restoration(tmp_path: Path) -> None:
    config = RunConfig()
    manager = manager_for(tmp_path / 'run', config)
    random.seed(17)
    np.random.seed(19)
    if torch is not None:
        torch.manual_seed(23)
    state = new_state(config)
    saved = manager.save(state, WorkQueue())
    expected_python = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = float(torch.rand(())) if torch is not None else None
    for _ in range(5):
        random.random(); np.random.random()
        if torch is not None:
            torch.rand(())
    loaded = manager.load_latest(restore_rng=True)
    assert loaded is not None and loaded.path == saved.path
    assert random.random() == expected_python
    assert float(np.random.random()) == expected_numpy
    if torch is not None:
        assert float(torch.rand(())) == expected_torch


def test_interrupted_item_recovery_requires_done_marker(tmp_path: Path) -> None:
    run_root = tmp_path / 'run'
    run_root.mkdir()
    queue = WorkQueue()
    item_id = queue.add('DUMMY', 'input', {'index': 0}, 1.0)
    item = queue.items[item_id]
    item.status = 'running'
    assert queue.recover_interrupted(run_root) == [item_id]
    assert item.status == 'pending'
    result = run_root / 'artifacts' / 'result.bin'
    result.parent.mkdir(parents=True)
    result.write_bytes(b'ok')
    digest = sha256_file(result)
    marker = run_root / 'work_items' / f'{item_id}.done'
    atomic_write_json(marker, {'item_id': item_id, 'result_relpath': 'artifacts/result.bin', 'result_sha256': digest})
    item.status = 'running'
    queue.recover_interrupted(run_root)
    assert item.status == 'done' and item.result_sha256 == digest
    result.unlink()
    queue.recover_interrupted(run_root)
    assert item.status == 'pending' and item.result_sha256 is None
    outside = tmp_path / 'outside.bin'
    outside.write_bytes(b'outside')
    atomic_write_json(marker, {
        'item_id': item_id, 'result_relpath': '../outside.bin',
        'result_sha256': sha256_file(outside),
    })
    item.status = 'running'
    queue.recover_interrupted(run_root)
    assert item.status == 'pending'


def test_resume_restores_checkpointed_tensor_state(tmp_path: Path) -> None:
    config = RunConfig(dummy_items=0, tensor_shard_bytes=64, prefer_cuda=False)
    first = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='tensor-resume', prefer_cuda=False)
    expected = np.arange(24, dtype=np.float64).reshape(3, 8)
    first.tensors['resume_array'] = expected
    first.checkpoint('tensor resume test')
    resumed = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='tensor-resume', prefer_cuda=False)
    np.testing.assert_array_equal(resumed.tensors['resume_array'], expected)


def test_incomplete_existing_run_fails_closed(tmp_path: Path) -> None:
    (tmp_path / 'runs' / 'partial-run').mkdir(parents=True)
    with pytest.raises(RunCompatibilityError, match='incomplete'):
        create_or_resume(tmp_path, RunConfig(prefer_cuda=False), PROJECT_ROOT, run_id='partial-run', prefer_cuda=False)


def test_resume_rejects_runtime_signature_drift(tmp_path: Path) -> None:
    config = RunConfig(dummy_items=0, prefer_cuda=False)
    created = create_or_resume(
        tmp_path, config, PROJECT_ROOT, run_id='runtime-drift', prefer_cuda=False,
    )
    manifest_path = created.run_root / 'run_manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['runtime_compatibility']['numpy'] = 'incompatible-test-version'
    atomic_write_json(manifest_path, manifest)
    with pytest.raises(RunCompatibilityError, match='runtime changed'):
        create_or_resume(
            tmp_path, config, PROJECT_ROOT, run_id='runtime-drift', prefer_cuda=False,
        )


def test_failed_item_cannot_be_reported_as_m0_complete(tmp_path: Path) -> None:
    config = RunConfig(dummy_items=1, dummy_size=8, dummy_steps=1, prefer_cuda=False)
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='failed-run', prefer_cuda=False)
    item = next(iter(orchestrator.queue.items.values()))
    item.status = 'failed'
    item.last_error = 'synthetic failure'
    with pytest.raises(RuntimeError, match='cannot complete'):
        orchestrator.run_until_checkpoint()
    assert orchestrator.state.phase != 'M0_COMPLETE'


def test_cpu_only_dummy_orchestrator(tmp_path: Path) -> None:
    config = RunConfig(dummy_items=2, dummy_size=8, dummy_steps=1, tensor_shard_bytes=64, prefer_cuda=False)
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='cpu-run', prefer_cuda=False, test_report={'fresh_process_resume': 'PASS'})
    summary = orchestrator.run_until_checkpoint()
    assert summary['phase'] == 'M0_COMPLETE'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert all(item.status == 'done' for item in orchestrator.queue.items.values())
    assert (orchestrator.run_root / 'reports' / 'M0_report.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'session_summary.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'latest_metrics.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'next_session_plan.md').is_file()


def test_short_simulated_session_returns_after_drain_checkpoint(tmp_path: Path) -> None:
    config = RunConfig(
        dummy_items=1, dummy_size=8, dummy_steps=1, checkpoint_interval_s=1.0,
        max_work_item_s=2.0, short_task_limit_s=0.5, checkpoint_reserve_s=0.1,
        no_long_task_after_s=3.0, drain_after_s=4.0, final_save_after_s=5.0,
        hard_return_s=6.0, dummy_predicted_s=1.0, prefer_cuda=False,
    )
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='timer-run', prefer_cuda=False)
    clock = FakeClock(0.0)
    orchestrator.guard = SessionGuard(config, clock=clock)
    clock.value = 4.0
    summary = orchestrator.run_until_checkpoint()
    assert summary['session_state'] == 'DRAIN'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert orchestrator.state.checkpoint_index >= 2


def test_hard_return_uses_done_marker_instead_of_late_checkpoint(tmp_path: Path) -> None:
    config = RunConfig(
        dummy_items=1, dummy_size=8, dummy_steps=1, dummy_predicted_s=0.1,
        checkpoint_interval_s=1.0, max_work_item_s=2.0, short_task_limit_s=0.5,
        checkpoint_reserve_s=0.1, no_long_task_after_s=3.0, drain_after_s=4.0,
        final_save_after_s=5.0, hard_return_s=6.0, prefer_cuda=False,
    )
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='hard-return', prefer_cuda=False)
    clock = FakeClock(0.0)
    orchestrator.guard = SessionGuard(config, clock=clock)
    original_execute = orchestrator._execute_dummy
    def finish_at_deadline(item: WorkItem) -> tuple[str, str]:
        artifact = original_execute(item)
        clock.value = config.hard_return_s
        return artifact
    orchestrator._execute_dummy = finish_at_deadline
    summary = orchestrator.run_until_checkpoint()
    assert summary['session_state'] == 'RETURN'
    assert 'done marker' in summary['stop_reason']
    resumed = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='hard-return', prefer_cuda=False)
    assert all(item.status == 'done' for item in resumed.queue.items.values())


def test_fresh_process_resume(tmp_path: Path) -> None:
    config = RunConfig(dummy_items=2, dummy_size=8, dummy_steps=1, prefer_cuda=False)
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='fresh-process', prefer_cuda=False)
    first = orchestrator.queue.next_pending()
    assert first is not None
    first.attempts += 1
    first.status = 'running'
    orchestrator.checkpoint('fresh-process test before first item')
    relative, digest = orchestrator._execute_dummy(first)
    first.result_relpath = relative
    first.result_sha256 = digest
    first.status = 'done'
    orchestrator.checkpoint('fresh-process test after first item')
    code = (
        'from pathlib import Path; '
        'from src.config import RunConfig; '
        'from src.orchestrator import create_or_resume; '
        f'c=RunConfig(dummy_items=2, dummy_size=8, dummy_steps=1, prefer_cuda=False); '
        f'o=create_or_resume(Path({str(tmp_path)!r}), c, Path({str(PROJECT_ROOT)!r}), run_id="fresh-process", prefer_cuda=False); '
        's=o.run_until_checkpoint(); assert s["phase"]=="M0_COMPLETE"; '
        'assert all(i.status=="done" for i in o.queue.items.values()); print(s["phase"])'
    )
    env = os.environ.copy()
    env['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + env.get('PYTHONPATH', '')
    completed = subprocess.run([sys.executable, '-c', code], cwd=PROJECT_ROOT, env=env, capture_output=True, text=True, timeout=60, check=False)
    assert completed.returncode == 0, completed.stderr
    assert 'M0_COMPLETE' in completed.stdout


@pytest.mark.gpu
def test_optional_cuda_dummy(tmp_path: Path) -> None:
    if torch is None or not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    rng_config = RunConfig()
    rng_manager = manager_for(tmp_path / 'cuda-rng', rng_config)
    torch.cuda.manual_seed_all(29)
    rng_manager.save(new_state(rng_config, run_id='cuda-rng'), WorkQueue())
    expected_cuda = float(torch.rand((), device='cuda').cpu())
    for _ in range(5):
        torch.rand((), device='cuda')
    assert rng_manager.load_latest(restore_rng=True) is not None
    assert float(torch.rand((), device='cuda').cpu()) == expected_cuda
    config = RunConfig(dummy_items=1, dummy_size=8, dummy_steps=1)
    orchestrator = create_or_resume(tmp_path, config, PROJECT_ROOT, run_id='gpu-run', prefer_cuda=True)
    summary = orchestrator.run_until_checkpoint()
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
