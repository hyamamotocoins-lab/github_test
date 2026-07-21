"""Exclusive GPU lane lease for Campaign B consumers (96 / 97 / pipeline_to_m6).

File: ``{PERSIST}/campaign_b/_locks/gpu_lane.json``

Acquire fails closed when another live process holds a fresh lease. Stale leases
(dead PID or expired heartbeat) are reclaimed on start. Same-process nested
acquire (e.g. 97 → pipeline_to_m6) refreshes the heartbeat and increments depth.
"""

from __future__ import annotations

import os
import socket
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from ..common import atomic_write_json, read_json, utc_now
from .errors import GpuLaneHeldError
from .schemas import screening_only_payload

__all__ = [
    'DEFAULT_STALE_HEARTBEAT_SEC',
    'ExecutionKey',
    'GPU_LOCK_NAME',
    'GpuLaneHeldError',
    'M3_EXECUTION_KEY',
    'acquire_gpu_lock',
    'gpu_lane_lease',
    'gpu_lane_path',
    'gpu_lock_path',
    'm3_execution_key',
    'read_gpu_lane_lease',
    'release_gpu_lock',
]

GPU_LOCK_NAME = 'gpu_lane'
M3_EXECUTION_KEY = 'shared_m3_batch'
# Match the six-hour GPU session budget; dead-PID reclaim is the primary path.
DEFAULT_STALE_HEARTBEAT_SEC = 6 * 3600

# Nested hold depth per resolved persistent_root (same process only).
_hold_depth: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class ExecutionKey:
    """Logical lock identity for shared M3 batch scheduling."""

    name: str
    owner: str = 'campaign_b'


def gpu_lane_path(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_locks' / 'gpu_lane.json'


def gpu_lock_path(persistent_root: Path) -> Path:
    """Alias for ``gpu_lane_path`` (historical name)."""
    return gpu_lane_path(persistent_root)


def _root_key(persistent_root: Path) -> str:
    return str(Path(persistent_root).resolve())


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it — treat as live.
        return True
    except OSError:
        return False


def _parse_iso(ts: object) -> datetime | None:
    if not isinstance(ts, str) or not ts.strip():
        return None
    try:
        dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _lease_stale_reason(
    doc: dict[str, Any],
    *,
    stale_heartbeat_sec: int,
    now: datetime,
) -> str | None:
    """Return a reclaim reason, or None if the lease is still held."""
    try:
        pid = int(doc.get('pid') or 0)
    except (TypeError, ValueError):
        return 'invalid_pid'

    host = str(doc.get('hostname') or '')
    local_host = socket.gethostname()

    if host and host == local_host and not _pid_alive(pid):
        return f'dead_pid:{pid}'

    if host and host != local_host:
        # Different machine: only reclaim on heartbeat expiry (cannot probe PID).
        heartbeat = _parse_iso(doc.get('heartbeat_at') or doc.get('acquired_at'))
        if heartbeat is None:
            return 'missing_heartbeat_foreign_host'
        age = (now - heartbeat).total_seconds()
        if age > float(stale_heartbeat_sec):
            return f'stale_heartbeat_foreign_host:{int(age)}s'
        return None

    heartbeat = _parse_iso(doc.get('heartbeat_at') or doc.get('acquired_at'))
    if heartbeat is None:
        return 'missing_heartbeat'
    age = (now - heartbeat).total_seconds()
    if age > float(stale_heartbeat_sec):
        return f'stale_heartbeat:{int(age)}s'
    if not _pid_alive(pid):
        return f'dead_pid:{pid}'
    return None


def _write_lease(
    path: Path,
    *,
    owner: str,
    pid: int,
    hostname: str,
    acquired_at: str | None = None,
    depth: int = 1,
    reclaimed_from: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    payload: dict[str, Any] = {
        'schema_version': 1,
        'key': GPU_LOCK_NAME,
        'owner': owner,
        'pid': pid,
        'hostname': hostname,
        'acquired_at': acquired_at or now,
        'heartbeat_at': now,
        'depth': int(depth),
        'enforced': True,
        'note': (
            'Exclusive GPU lane lease. Fail closed if held by another live process. '
            'Stale (dead pid / old heartbeat) may be reclaimed on acquire.'
        ),
        **screening_only_payload(),
    }
    if reclaimed_from is not None:
        payload['reclaimed_from'] = reclaimed_from
        payload['reclaimed_at'] = now
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)
    return payload


def read_gpu_lane_lease(persistent_root: Path) -> dict[str, Any] | None:
    path = gpu_lane_path(persistent_root)
    if not path.is_file():
        return None
    doc = read_json(path)
    return doc if isinstance(doc, dict) else None


def acquire_gpu_lock(
    persistent_root: Path,
    *,
    owner: str = 'campaign_b',
    stale_heartbeat_sec: int = DEFAULT_STALE_HEARTBEAT_SEC,
) -> dict[str, Any]:
    """Acquire exclusive GPU lane lease or fail closed.

    Same-process nested calls increment depth and refresh heartbeat.
    """
    persistent_root = Path(persistent_root)
    key = _root_key(persistent_root)
    path = gpu_lane_path(persistent_root)
    pid = os.getpid()
    hostname = socket.gethostname()
    now = datetime.now(timezone.utc)

    depth = _hold_depth.get(key, 0)
    if depth > 0:
        new_depth = depth + 1
        _hold_depth[key] = new_depth
        existing = read_gpu_lane_lease(persistent_root) or {}
        return _write_lease(
            path,
            owner=str(owner),
            pid=pid,
            hostname=hostname,
            acquired_at=str(existing.get('acquired_at') or utc_now()),
            depth=new_depth,
        )

    reclaimed_from: str | None = None
    if path.is_file():
        doc = read_gpu_lane_lease(persistent_root) or {}
        same_process = (
            int(doc.get('pid') or 0) == pid
            and str(doc.get('hostname') or '') == hostname
        )
        if same_process:
            _hold_depth[key] = int(doc.get('depth') or 1) + 1
            return _write_lease(
                path,
                owner=str(owner),
                pid=pid,
                hostname=hostname,
                acquired_at=str(doc.get('acquired_at') or utc_now()),
                depth=_hold_depth[key],
            )
        stale_reason = _lease_stale_reason(
            doc,
            stale_heartbeat_sec=int(stale_heartbeat_sec),
            now=now,
        )
        if stale_reason is None:
            raise GpuLaneHeldError(
                'GPU lane lease held by another live process: '
                f"owner={doc.get('owner')!r} pid={doc.get('pid')!r} "
                f"hostname={doc.get('hostname')!r} "
                f"heartbeat_at={doc.get('heartbeat_at')!r} "
                f'path={path}. '
                'Do not run notebook 96 and 97 (or two GPU consumers) together. '
                'Wait for the holder to finish, or reclaim only after the process is dead.'
            )
        reclaimed_from = (
            f"owner={doc.get('owner')} pid={doc.get('pid')} "
            f"host={doc.get('hostname')} reason={stale_reason}"
        )

    _hold_depth[key] = 1
    return _write_lease(
        path,
        owner=str(owner),
        pid=pid,
        hostname=hostname,
        depth=1,
        reclaimed_from=reclaimed_from,
    )


def release_gpu_lock(
    persistent_root: Path,
    *,
    owner: str = 'campaign_b',
) -> bool:
    """Release lease if this process holds it. Nested releases decrement depth."""
    persistent_root = Path(persistent_root)
    key = _root_key(persistent_root)
    path = gpu_lane_path(persistent_root)
    depth = _hold_depth.get(key, 0)

    if depth > 1:
        _hold_depth[key] = depth - 1
        doc = read_gpu_lane_lease(persistent_root) or {}
        if path.is_file():
            _write_lease(
                path,
                owner=str(doc.get('owner') or owner),
                pid=os.getpid(),
                hostname=socket.gethostname(),
                acquired_at=str(doc.get('acquired_at') or utc_now()),
                depth=_hold_depth[key],
            )
        return True

    if depth == 1:
        _hold_depth.pop(key, None)
    elif depth == 0:
        # Best-effort: only unlink if we still own the file.
        pass

    if not path.is_file():
        _hold_depth.pop(key, None)
        return False

    doc = read_gpu_lane_lease(persistent_root) or {}
    same_process = (
        int(doc.get('pid') or 0) == os.getpid()
        and str(doc.get('hostname') or '') == socket.gethostname()
    )
    if not same_process:
        return False
    try:
        path.unlink()
    except OSError:
        return False
    _hold_depth.pop(key, None)
    return True


@contextmanager
def gpu_lane_lease(
    persistent_root: Path,
    *,
    owner: str = 'campaign_b',
    stale_heartbeat_sec: int = DEFAULT_STALE_HEARTBEAT_SEC,
) -> Iterator[dict[str, Any]]:
    """Context manager around ``acquire_gpu_lock`` / ``release_gpu_lock``."""
    payload = acquire_gpu_lock(
        persistent_root,
        owner=owner,
        stale_heartbeat_sec=stale_heartbeat_sec,
    )
    try:
        yield payload
    finally:
        release_gpu_lock(persistent_root, owner=owner)


def m3_execution_key() -> ExecutionKey:
    return ExecutionKey(name=M3_EXECUTION_KEY)
