"""Notebook-friendly background control for staged M2 lineage.

Jupyter kernels on Paperspace die when heavy SymPy work runs in-process.
These helpers launch a detached worker and let notebook cells only poll.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, utc_now
from .m7_staged_lineage import inspect_staged_m2_progress


class M7StagedNotebookError(RuntimeError):
    """Raised when notebook background control fails closed."""


def _worker_dir(package_root: Path) -> Path:
    path = package_root / 'notebook_worker'
    path.mkdir(parents=True, exist_ok=True)
    return path


def worker_paths(package_root: Path) -> dict[str, Path]:
    root = _worker_dir(package_root)
    return {
        'dir': root,
        'pid': root / 'worker.pid',
        'log': root / 'worker.log',
        'status': root / 'worker_status.json',
        'stop': root / 'STOP',
    }


def is_pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def read_worker_status(package_root: Path) -> dict[str, Any]:
    paths = worker_paths(package_root)
    status: dict[str, Any] = {
        'running': False,
        'pid': None,
        'log_path': str(paths['log']),
        'status_path': str(paths['status']),
    }
    if paths['pid'].is_file():
        try:
            pid = int(paths['pid'].read_text(encoding='utf-8').strip())
        except ValueError:
            pid = None
        status['pid'] = pid
        status['running'] = bool(pid and is_pid_alive(pid))
    if paths['status'].is_file():
        doc = read_json(paths['status'])
        if isinstance(doc, dict):
            status['last_status'] = doc
    if paths['log'].is_file():
        status['log_bytes'] = paths['log'].stat().st_size
        try:
            tail = paths['log'].read_text(encoding='utf-8', errors='replace').splitlines()[-20:]
        except OSError:
            tail = []
        status['log_tail'] = tail
    return status


def start_staged_background_worker(
    package_root: Path,
    *,
    project_root: Path,
    persistent_root: Path,
    test_report: dict[str, Any] | None = None,
    split_batch_to: int = 1,
    checkpoint_keep: int = 5,
) -> dict[str, Any]:
    """Start detached staged M2 worker; returns immediately for notebook use."""
    package_root = package_root.resolve()
    project_root = project_root.resolve()
    persistent_root = persistent_root.resolve()
    paths = worker_paths(package_root)
    current = read_worker_status(package_root)
    if current.get('running'):
        return {
            'started': False,
            'reason': 'already_running',
            **current,
        }

    child_ids = read_json(package_root / 'child_run_ids.json')
    if not isinstance(child_ids, dict) or not isinstance(child_ids.get('M2'), str):
        raise M7StagedNotebookError('Package child_run_ids.M2 missing.')
    m2_id = child_ids['M2']

    if test_report is not None:
        atomic_write_json(paths['dir'] / 'test_report.json', test_report)

    if paths['stop'].exists():
        paths['stop'].unlink()

    worker_py = paths['dir'] / 'run_worker.py'
    worker_py.write_text(
        f'''#!/usr/bin/env python3
from __future__ import annotations
import json, os, sys, traceback
from pathlib import Path

PACKAGE = Path({str(package_root)!r})
PROJECT = Path({str(project_root)!r})
PERSIST = Path({str(persistent_root)!r})
STATUS = Path({str(paths['status'])!r})
sys.path.insert(0, str(PROJECT))
os.environ['VALIDATED_RG_PROJECT_ROOT'] = str(PROJECT)
os.environ['VALIDATED_RG_PERSIST_ROOT'] = str(PERSIST)
os.environ['VALIDATED_RG_M2_SPLIT_BATCH_TO'] = {str(split_batch_to)!r}
os.environ['VALIDATED_RG_CHECKPOINT_KEEP'] = {str(checkpoint_keep)!r}

def write_status(payload):
    STATUS.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding='utf-8')

write_status({{'state': 'starting', 'pid': os.getpid()}})
try:
    from src.common import read_json, utc_now
    from src.m7_staged_lineage import run_staged_lineage_from_package
    report_path = PACKAGE / 'notebook_worker' / 'test_report.json'
    test_report = read_json(report_path) if report_path.is_file() else None
    result = run_staged_lineage_from_package(
        PACKAGE,
        persistent_root=PERSIST,
        project_root=PROJECT,
        rewrite_m2_audit=True,
        test_report=test_report if isinstance(test_report, dict) else None,
    )
    write_status({{
        'state': 'finished',
        'pid': os.getpid(),
        'finished_at': utc_now(),
        'result': {{
            'status': result.get('status'),
            'audit_rewritten': result.get('audit_rewritten'),
            'm2_complete': (result.get('m2_session') or {{}}).get('m2_complete'),
            'stop_reason': (result.get('m2_session') or {{}}).get('stop_reason'),
            'checkpoint_index': (result.get('m2_session') or {{}}).get('checkpoint_index'),
        }},
    }})
except Exception as exc:
    write_status({{
        'state': 'failed',
        'pid': os.getpid(),
        'error': f'{{type(exc).__name__}}: {{exc}}',
        'traceback': traceback.format_exc(),
    }})
    raise
''',
        encoding='utf-8',
    )

    log_handle = paths['log'].open('a', encoding='utf-8')
    log_handle.write(f'\n===== worker start {utc_now()} =====\n')
    log_handle.flush()
    proc = subprocess.Popen(
        [os.environ.get('PYTHON', 'python'), '-u', str(worker_py)],
        cwd=str(project_root),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env={
            **os.environ,
            'PYTHONPATH': str(project_root),
            'VALIDATED_RG_PROJECT_ROOT': str(project_root),
            'VALIDATED_RG_PERSIST_ROOT': str(persistent_root),
            'VALIDATED_RG_M2_SPLIT_BATCH_TO': str(split_batch_to),
            'VALIDATED_RG_CHECKPOINT_KEEP': str(checkpoint_keep),
            'VALIDATED_RG_M2_RUN_ID': m2_id,
        },
    )
    paths['pid'].write_text(str(proc.pid), encoding='utf-8')
    atomic_write_json(paths['status'], {
        'state': 'launched',
        'pid': proc.pid,
        'm2_run_id': m2_id,
        'launched_at': utc_now(),
        'log_path': str(paths['log']),
    })
    time.sleep(0.5)
    return {
        'started': True,
        'pid': proc.pid,
        'm2_run_id': m2_id,
        'log_path': str(paths['log']),
        'status_path': str(paths['status']),
        'alive': is_pid_alive(proc.pid),
        'progress': inspect_staged_m2_progress(persistent_root, run_id=m2_id),
    }


def poll_staged_background_worker(
    package_root: Path,
    *,
    persistent_root: Path,
) -> dict[str, Any]:
    child_ids = read_json(package_root / 'child_run_ids.json')
    m2_id = str((child_ids or {}).get('M2') or '')
    status = read_worker_status(package_root)
    progress = (
        inspect_staged_m2_progress(persistent_root, run_id=m2_id)
        if m2_id.startswith('M2-') else {'exists': False}
    )
    return {
        **status,
        'm2_run_id': m2_id,
        'progress': progress,
        'polled_at': utc_now(),
    }


def stop_staged_background_worker(package_root: Path) -> dict[str, Any]:
    paths = worker_paths(package_root)
    status = read_worker_status(package_root)
    pid = status.get('pid')
    if not status.get('running') or not isinstance(pid, int):
        return {'stopped': False, 'reason': 'not_running', **status}
    paths['stop'].write_text(utc_now(), encoding='utf-8')
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {'stopped': True, 'reason': 'already_dead', 'pid': pid}
    return {'stopped': True, 'pid': pid, 'signal': 'SIGTERM'}
