from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.checkpoint import CheckpointManager, RunState
from src.common import utc_now
from src.m4_config import M4Config
from src.m4_orchestrator import create_or_resume_m4
from src.m4_parent import M4ParentError, verify_accepted_m3_parent
from src.source_channels import SOURCE_CLASSES
from src.work_queue import WorkQueue
from tests.m4_helpers import make_synthetic_accepted_m3, passing_m4_test_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_accepted_m3_parent_is_pinned_and_tamper_fails(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m3(tmp_path)
    evidence = verify_accepted_m3_parent(project, config)
    assert len(evidence.tensors) == 6
    Path(config.parent_report_path).write_text('{}', encoding='utf-8')
    with pytest.raises(M4ParentError, match='report'):
        verify_accepted_m3_parent(project, config)


def test_m4_checkpoint_restores_all_derivative_channels(tmp_path: Path) -> None:
    config, _ = make_synthetic_accepted_m3(tmp_path)
    run_root = tmp_path / 'm4-checkpoint'
    manager = CheckpointManager(
        run_root, config, source_hash='4' * 64, notebook_hash='5' * 64,
    )
    state = RunState(
        run_root.name, config.config_hash(), utc_now(), utc_now(),
        milestone='M4', phase='M4_RUNNING',
    )
    queue = WorkQueue()
    tensors = {'normalized_primal': np.eye(16)}
    tensors.update({
        f'normalized_tangent_{source.value}': np.full((16, 16), index)
        for index, source in enumerate(SOURCE_CLASSES)
    })
    saved = manager.save(state, queue, tensors)
    loaded = manager.load_latest(restore_rng=False)
    assert loaded is not None and loaded.path == saved.path
    for name, value in tensors.items():
        np.testing.assert_array_equal(loaded.tensors[name], value)


def test_initial_two_hour_policy_then_standard_resume(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m3(tmp_path)
    persistent = tmp_path / 'persist'
    first = create_or_resume_m4(
        persistent, config, project, run_id='M4-policy-test',
        test_report=passing_m4_test_report(),
    )
    assert first.session_policy == 'INITIAL_TWO_HOUR_LIMIT'
    assert first.guard.config.hard_return_s == 2 * 3600
    resumed = create_or_resume_m4(
        persistent, config, project, run_id='M4-policy-test',
        test_report=passing_m4_test_report(),
    )
    assert resumed.session_policy == 'RESUMED_STANDARD_FIVE_HOUR_THIRTY_LIMIT'
    assert resumed.guard.config.hard_return_s == 5.5 * 3600


def test_m4_checkpoint_resume_and_fresh_process(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m3(tmp_path)
    persistent = tmp_path / 'persist'
    first = create_or_resume_m4(
        persistent, config, project, run_id='M4-restart-test',
        test_report=passing_m4_test_report(),
    )
    assert first.run_one_item_for_test() == 'M4_SOURCE_CHANNELS'
    resumed = create_or_resume_m4(
        persistent, config, project, run_id='M4-restart-test',
        test_report=passing_m4_test_report(),
    )
    assert sum(item.status == 'done' for item in resumed.queue.items.values()) == 1
    code = (
        'from pathlib import Path; '
        'from src.m4_config import M4Config; '
        'from src.m4_orchestrator import create_or_resume_m4; '
        f'c=M4Config(require_cuda=False,parent_run_id={config.parent_run_id!r},'
        f'parent_checkpoint_path={config.parent_checkpoint_path!r},'
        f'parent_report_path={config.parent_report_path!r},'
        f'parent_acceptance_path={config.parent_acceptance_path!r}); '
        f'o=create_or_resume_m4(Path({str(persistent)!r}),c,'
        f'Path({str(project)!r}),run_id="M4-restart-test"); '
        'assert sum(i.status=="done" for i in o.queue.items.values())==1; '
        'assert o.session_policy.startswith("RESUMED"); print(o.state.phase)'
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
    assert 'M4_BOOTSTRAP' in completed.stdout


def test_full_m4_cpu_run_is_complete_but_blocked_math(tmp_path: Path) -> None:
    config, project = make_synthetic_accepted_m3(tmp_path)
    orchestrator = create_or_resume_m4(
        tmp_path / 'persist', config, project, run_id='M4-full-test',
        test_report=passing_m4_test_report(),
    )
    summary = orchestrator.run_until_checkpoint()
    assert summary['phase'] == 'M4_COMPLETE'
    assert summary['milestone_status'] == 'BLOCKED_MATH'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert (orchestrator.run_root / 'reports/M4_report.json').is_file()
    assert (orchestrator.run_root / 'reports/M4_report.md').is_file()
    loaded = orchestrator.checkpoints.load_latest(restore_rng=False)
    assert loaded is not None
    assert {
        'normalized_primal',
        *(f'normalized_tangent_{source.value}' for source in SOURCE_CLASSES),
    } <= set(loaded.tensors)


def test_child_m3_audit_rewrite_accepts_nondefault_checkpoint_index(
    tmp_path: Path,
) -> None:
    from dataclasses import asdict

    from src.common import atomic_write_json, atomic_write_text, read_json, sha256_file
    from src.m7_staged_lineage import write_child_m3_acceptance_audit

    config, project = make_synthetic_accepted_m3(tmp_path)
    parent_run = Path(config.parent_checkpoint_path).parents[1]
    ckpt = Path(config.parent_checkpoint_path)
    state = read_json(ckpt / 'state.json')
    assert isinstance(state, dict)
    state['checkpoint_index'] = 16
    atomic_write_json(ckpt / 'state.json', state)
    hashes = {
        path.relative_to(ckpt).as_posix(): sha256_file(path)
        for path in ckpt.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    atomic_write_json(ckpt / 'hashes.json', hashes)
    atomic_write_text(ckpt / 'COMMITTED', utc_now())
    audit = write_child_m3_acceptance_audit(project, run_root=parent_run)
    assert audit['checkpoint_index'] == 16
    assert audit['accepted_run_id'] == config.parent_run_id
    base = asdict(config)
    base.update({
        'parent_checkpoint': Path(audit['checkpoint_path']).name,
        'parent_checkpoint_path': audit['checkpoint_path'],
        'parent_report_path': audit['m3_report_path'],
        'parent_acceptance_path': audit['m3_acceptance_path'],
    })
    evidence = verify_accepted_m3_parent(project, M4Config(**base))
    assert evidence.hashes['m3_audit_sha256']
