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


def kill_stray_staged_processes(
    *,
    package_root: Path | None = None,
    include_execute_lineage: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Kill leftover staged workers / execute_lineage processes."""
    killed: list[dict[str, Any]] = []
    sig = signal.SIGKILL if force else signal.SIGTERM

    if package_root is not None:
        stop = stop_staged_background_worker(package_root)
        if stop.get('stopped'):
            killed.append({'source': 'notebook_worker', **stop})

    patterns = ['m7_staged', 'run_worker.py']
    if include_execute_lineage:
        patterns.append('execute_lineage.py')

    try:
        out = subprocess.check_output(['ps', 'ax', '-o', 'pid=,command='], text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        return {'killed': killed, 'error': str(exc)}

    self_pid = os.getpid()
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1]
        if pid == self_pid:
            continue
        if not any(pat in cmd for pat in patterns):
            continue
        if 'grep' in cmd:
            continue
        try:
            os.kill(pid, sig)
            killed.append({'pid': pid, 'cmd': cmd, 'signal': sig.name})
        except ProcessLookupError:
            killed.append({'pid': pid, 'cmd': cmd, 'already_dead': True})

    # Clear stale pid file if process is gone.
    if package_root is not None:
        paths = worker_paths(package_root)
        if paths['pid'].is_file():
            try:
                old = int(paths['pid'].read_text(encoding='utf-8').strip())
            except ValueError:
                old = None
            if old is None or not is_pid_alive(old):
                paths['pid'].unlink(missing_ok=True)

    return {'killed': killed, 'force': force, 'at': utc_now()}


def watch_staged_background_worker(
    package_root: Path,
    *,
    persistent_root: Path,
    project_root: Path | None = None,
    test_report: dict[str, Any] | None = None,
    poll_s: float = 30.0,
    max_hours: float = 12.0,
    auto_restart: bool = True,
    split_batch_to: int = 1,
    checkpoint_keep: int = 5,
) -> dict[str, Any]:
    """Poll until M2 complete, worker finished, or deadline.

    If the worker dies without completion and auto_restart=True, relaunch.
    Safe for notebook: only sleeps/polls in-kernel; heavy work stays detached.
    """
    if poll_s < 5.0:
        raise M7StagedNotebookError('poll_s must be >= 5 seconds.')
    deadline = time.monotonic() + max_hours * 3600.0
    history: list[dict[str, Any]] = []
    restarts = 0

    while time.monotonic() < deadline:
        snap = poll_staged_background_worker(
            package_root, persistent_root=persistent_root,
        )
        prog = snap.get('progress') or {}
        last = snap.get('last_status') or {}
        summary = {
            'at': snap.get('polled_at'),
            'running': snap.get('running'),
            'pid': snap.get('pid'),
            'm2_complete': prog.get('m2_complete'),
            'fraction_done': prog.get('fraction_done'),
            'queue_counts': prog.get('queue_counts'),
            'worker_state': last.get('state'),
            'checkpoint_index': prog.get('checkpoint_index'),
        }
        history.append(summary)
        print(json.dumps(summary, ensure_ascii=False, default=str), flush=True)

        if prog.get('m2_complete'):
            return {
                'outcome': 'M2_COMPLETE',
                'restarts': restarts,
                'final': snap,
                'history_tail': history[-20:],
            }
        if last.get('state') == 'finished' and prog.get('m2_complete'):
            return {
                'outcome': 'WORKER_FINISHED',
                'restarts': restarts,
                'final': snap,
                'history_tail': history[-20:],
            }
        if last.get('state') == 'failed' and not snap.get('running'):
            if auto_restart and project_root is not None:
                restarts += 1
                print(f'worker failed; auto_restart #{restarts}', flush=True)
                start_staged_background_worker(
                    package_root,
                    project_root=project_root,
                    persistent_root=persistent_root,
                    test_report=test_report,
                    split_batch_to=split_batch_to,
                    checkpoint_keep=checkpoint_keep,
                )
            else:
                return {
                    'outcome': 'WORKER_FAILED',
                    'restarts': restarts,
                    'final': snap,
                    'history_tail': history[-20:],
                }
        elif (
            not snap.get('running')
            and not prog.get('m2_complete')
            and auto_restart
            and project_root is not None
        ):
            # Dead worker, no finished status (killed externally).
            restarts += 1
            print(f'worker dead; auto_restart #{restarts}', flush=True)
            start_staged_background_worker(
                package_root,
                project_root=project_root,
                persistent_root=persistent_root,
                test_report=test_report,
                split_batch_to=split_batch_to,
                checkpoint_keep=checkpoint_keep,
            )

        time.sleep(poll_s)

    return {
        'outcome': 'TIMEOUT',
        'restarts': restarts,
        'final': poll_staged_background_worker(
            package_root, persistent_root=persistent_root,
        ),
        'history_tail': history[-20:],
    }
