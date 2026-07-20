from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.m1_config import M1Config
from src.m1_orchestrator import _notebook_hash, create_or_resume_m1
from src.session_guard import SessionGuard, SessionState
from tests.m1_helpers import make_synthetic_accepted_parent, passing_test_report

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class FakeClock:
    def __init__(self, value: float = 0.0) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


def test_m1_notebook_identity_ignores_saved_outputs_but_tracks_source(tmp_path: Path) -> None:
    baseline = _notebook_hash(PROJECT_ROOT)
    assert baseline is not None
    source_path = PROJECT_ROOT / 'notebooks' / '20_m1_exact_2d.ipynb'
    payload = json.loads(source_path.read_text(encoding='utf-8'))
    for index, cell in enumerate(payload['cells']):
        cell['id'] = f'saved-cell-{index}'
        if cell['cell_type'] == 'code':
            cell['execution_count'] = index + 1
            cell['outputs'] = [{
                'name': 'stdout', 'output_type': 'stream', 'text': ['saved output\n'],
            }]
    project = tmp_path / 'project'
    notebook_path = project / 'notebooks' / '20_m1_exact_2d.ipynb'
    notebook_path.parent.mkdir(parents=True)
    notebook_path.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    assert _notebook_hash(project) == baseline

    first_source = payload['cells'][0]['source']
    if isinstance(first_source, list):
        first_source.append('\nsource identity change')
    else:
        payload['cells'][0]['source'] = first_source + '\nsource identity change'
    notebook_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    assert _notebook_hash(project) != baseline


def test_m1_checkpoint_resume_and_fresh_process(tmp_path: Path) -> None:
    config = make_synthetic_accepted_parent(tmp_path)
    persistent = tmp_path / 'persist'
    first = create_or_resume_m1(
        persistent, config, PROJECT_ROOT, run_id='M1-restart-test', test_report=passing_test_report(),
    )
    assert first.run_one_item_for_test() == 'M1_COEFFICIENT_BATCH'
    resumed = create_or_resume_m1(
        persistent, config, PROJECT_ROOT, run_id='M1-restart-test', test_report=passing_test_report(),
    )
    assert sum(item.status == 'done' for item in resumed.queue.items.values()) == 1
    code = (
        'from pathlib import Path; '
        'from src.m1_config import M1Config; '
        'from src.m1_orchestrator import create_or_resume_m1; '
        f'c=M1Config(parent_checkpoint_path={config.parent_checkpoint_path!r}); '
        f'o=create_or_resume_m1(Path({str(persistent)!r}),c,Path({str(PROJECT_ROOT)!r}),run_id="M1-restart-test"); '
        'assert sum(i.status=="done" for i in o.queue.items.values())==1; print(o.state.phase)'
    )
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(PROJECT_ROOT) + os.pathsep + environment.get('PYTHONPATH', '')
    completed = subprocess.run(
        [sys.executable, '-c', code], cwd=PROJECT_ROOT, env=environment,
        capture_output=True, text=True, timeout=90, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert 'M1_BOOTSTRAP' in completed.stdout


def test_m1_session_drain_checkpoint_and_resume(tmp_path: Path) -> None:
    base = M1Config(
        checkpoint_interval_s=1.0, max_work_item_s=2.0, short_task_limit_s=0.5,
        checkpoint_reserve_s=0.1, no_long_task_after_s=3.0, drain_after_s=4.0,
        final_save_after_s=5.0, hard_return_s=6.0,
    )
    config = make_synthetic_accepted_parent(tmp_path, base)
    persistent = tmp_path / 'persist'
    orchestrator = create_or_resume_m1(persistent, config, PROJECT_ROOT, run_id='M1-drain-test')
    clock = FakeClock(); orchestrator.guard = SessionGuard(config, clock=clock)
    clock.value = base.drain_after_s
    assert orchestrator.guard.elapsed_s() == base.drain_after_s
    assert orchestrator.guard.state() is SessionState.DRAIN
    summary = orchestrator.run_until_checkpoint()
    assert 'drain checkpoint complete' in summary['stop_reason']
    resumed = create_or_resume_m1(persistent, config, PROJECT_ROOT, run_id='M1-drain-test')
    assert resumed.state.phase == 'M1_RUNNING'


def test_full_m1_run_writes_acceptance_and_remains_not_certified(tmp_path: Path) -> None:
    config = make_synthetic_accepted_parent(tmp_path)
    orchestrator = create_or_resume_m1(
        tmp_path / 'persist', config, PROJECT_ROOT, run_id='M1-full-test',
        test_report=passing_test_report(),
    )
    summary = orchestrator.run_until_checkpoint()
    assert summary['phase'] == 'M1_COMPLETE'
    assert summary['certification_status'] == 'NOT_CERTIFIED'
    assert (orchestrator.run_root / 'reports' / 'M1_report.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'M1_report.md').is_file()
    assert (orchestrator.run_root / 'reports' / 'M1_acceptance.json').is_file()
    assert all(item.status == 'done' for item in orchestrator.queue.items.values())


@pytest.mark.gpu
def test_m1_optional_gpu_smoke_keeps_exact_path_on_cpu() -> None:
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    from src.runtime import configure_numerics, environment_info
    configure_numerics(); tensor = torch.eye(2, dtype=torch.float64, device='cuda')
    assert float(torch.linalg.det(tensor).cpu()) == 1.0
    info = environment_info()
    assert info['cuda_available'] is True
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert M1Config().certification_status == 'NOT_CERTIFIED'
