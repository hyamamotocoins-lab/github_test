from __future__ import annotations

import json
import resource
import shlex
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointSaveResult, RunState
from .common import atomic_write_json, atomic_write_text, fsync_descriptor, fsync_directory, utc_now
from .runtime import PERSIST_ACK_ENV, PERSIST_ACK_TOKEN, PERSIST_ROOT_ENV
from .work_queue import WorkQueue

try:
    import torch
except ImportError:
    torch = None


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        record = {'timestamp': utc_now(), 'event': event, **payload}
        line = json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False) + '\n'
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(line)
            handle.flush()
            fsync_descriptor(handle.fileno())
        fsync_directory(self.path.parent)


def peak_memory_report() -> dict[str, int | None]:
    cpu_raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    gpu_peak = None
    if torch is not None and torch.cuda.is_available():
        gpu_peak = int(torch.cuda.max_memory_allocated())
    return {'cpu_ru_maxrss_raw': cpu_raw, 'gpu_peak_allocated_bytes': gpu_peak}


def next_session_instructions(persistent_root: Path, run_id: str, project_root: Path) -> str:
    return '\n'.join((
        'NEXT SESSION (fresh kernel):',
        f'1. Mount the same durable storage containing {persistent_root}.',
        f'2. Set VALIDATED_RG_PROJECT_ROOT={shlex.quote(str(project_root))}.',
        f'3. Set {PERSIST_ROOT_ENV}={shlex.quote(str(persistent_root))}.',
        f'4. Set {PERSIST_ACK_ENV}={PERSIST_ACK_TOKEN}.',
        f'5. Set VALIDATED_RG_RUN_ID={run_id}.',
        '6. Rerun this notebook from the first cell in a fresh kernel.',
        '7. Confirm the M0 tests pass, then call orchestrator.run_until_checkpoint().',
    ))


def write_session_artifacts(
    run_root: Path,
    state: RunState,
    queue: WorkQueue,
    *,
    stop_reason: str,
    elapsed_s: float,
    remaining_s: float,
    persistent_root: Path,
    project_root: Path,
) -> dict[str, str]:
    state.assert_m0_safe()
    queue.validate()
    reports = run_root / 'reports'
    unfinished = [
        {
            'item_id': item.item_id, 'phase': item.phase, 'status': item.status,
            'attempts': item.attempts, 'last_error': item.last_error,
        }
        for item in queue.items.values() if item.status != 'done'
    ]
    counts = {
        status: sum(item.status == status for item in queue.items.values())
        for status in ('pending', 'running', 'done', 'failed', 'blocked_resource')
    }
    summary_path = reports / 'session_summary.json'
    metrics_path = reports / 'latest_metrics.json'
    next_path = reports / 'next_session_plan.md'
    atomic_write_json(summary_path, {
        'schema_version': 1, 'generated_at': utc_now(), 'run_id': state.run_id,
        'phase': state.phase, 'checkpoint_index': state.checkpoint_index,
        'stop_reason': stop_reason, 'elapsed_s': elapsed_s, 'remaining_s': remaining_s,
        'certification_status': state.certification_status, 'queue_counts': counts,
        'unfinished_work_items': unfinished,
        'mathematical_status': 'M0_ONLY_NOT_CERTIFIED',
        'unproved_or_unimplemented_scope': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'P1-P13'],
    })
    atomic_write_json(metrics_path, {
        'generated_at': utc_now(), 'elapsed_s': elapsed_s, 'remaining_s': remaining_s,
        'phase': state.phase, 'checkpoint_index': state.checkpoint_index,
        'queue_counts': counts, 'memory': peak_memory_report(),
        'rigorous_error_bounds': {}, 'approximate_spectral_radius': None,
        'certification_status': state.certification_status,
    })
    if state.phase == 'M0_COMPLETE':
        plan = (
            '# Next action\n\nM0 completed. Review `M0_report.json` and its tests. '
            'Do not start M1 until the M0 acceptance result is confirmed in the execution environment.\n'
        )
    else:
        plan = '# Next session plan\n\n```text\n' + next_session_instructions(
            persistent_root, state.run_id, project_root
        ) + '\n```\n'
    atomic_write_text(next_path, plan)
    return {
        'session_summary': str(summary_path), 'latest_metrics': str(metrics_path),
        'next_session_plan': str(next_path),
    }


def write_m0_report(
    run_root: Path,
    state: RunState,
    queue: WorkQueue,
    test_report: dict[str, Any],
    checkpoint: CheckpointSaveResult | None,
    generated_files: list[str],
) -> Path:
    state.assert_m0_safe()
    report = {
        'milestone': 'M0',
        'generated_at': utc_now(),
        'run_id': state.run_id,
        'phase': state.phase,
        'certification_status': state.certification_status,
        'files_changed': generated_files,
        'tests': test_report,
        'restart_test_status': test_report.get('fresh_process_resume', 'UNKNOWN'),
        'remaining_todos': ['M1 exact 2D benchmark', 'M2 low-cutoff armillary', 'M3 GPU Triad-ATRG', 'M4 forward AD', 'M5 one-step validation', 'M6 multi-step certificate'],
        'heuristic_bounds': ['All RG, residual, tail, influence, and certificate bounds are absent in M0; none is treated as zero.'],
        'memory': peak_memory_report(),
        'checkpoint': None if checkpoint is None else {
            'path': str(checkpoint.path),
            'size_bytes': checkpoint.size_bytes,
            'save_s': checkpoint.save_s,
            'verify_s': checkpoint.verify_s,
        },
        'queue': {
            'done': sum(item.status == 'done' for item in queue.items.values()),
            'pending': sum(item.status == 'pending' for item in queue.items.values()),
            'running': sum(item.status == 'running' for item in queue.items.values()),
            'failed': sum(item.status == 'failed' for item in queue.items.values()),
        },
    }
    path = run_root / 'reports' / 'M0_report.json'
    atomic_write_json(path, report)
    return path
