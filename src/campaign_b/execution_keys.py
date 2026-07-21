"""Exclusive GPU lane lease for Campaign B consumers (96 / 97 / pipeline_to_m6).

File: ``{PERSIST}/campaign_b/_locks/gpu_lane.json``

Acquire fails closed when another live process holds a fresh lease. Stale leases
(dead PID or expired heartbeat) are reclaimed on start. Same-process nested
acquire (e.g. 97 → pipeline_to_m6) refreshes the heartbeat and increments depth.

Foreign-host leases (Paperspace machine switch) cannot probe the remote PID.
They use a shorter stale threshold (default 15 min via
``FOREIGN_HOST_STALE_HEARTBEAT_SEC`` / env ``VALIDATED_RG_GPU_LANE_FOREIGN_STALE_SEC``)
so a dead lock from a previous hostname is reclaimable without a manual ``rm``.

Live multi-machine GPU consumers must call ``refresh_gpu_lane_heartbeat`` during
long work; 15 minutes without refresh ⇒ reclaimable after host change or crash.
Same-host dead PID remains immediately reclaimable.
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
    'FOREIGN_HOST_STALE_HEARTBEAT_SEC',
    'ExecutionKey',
    'GPU_LOCK_NAME',
    'GpuLaneHeldError',
    'M3_EXECUTION_KEY',
    'acquire_gpu_lock',
    'force_release_gpu_lane_lease',
    'foreign_stale_heartbeat_sec',
    'gpu_lane_lease',
    'gpu_lane_path',
    'gpu_lock_path',
    'm3_execution_key',
    'read_gpu_lane_lease',
    'refresh_gpu_lane_heartbeat',
    'release_gpu_lock',
    'try_reclaim_gpu_lane_lease',
]

GPU_LOCK_NAME = 'gpu_lane'
M3_EXECUTION_KEY = 'shared_m3_batch'
# Match the six-hour GPU session budget; dead-PID reclaim is the primary path
# for same-host crashes.
DEFAULT_STALE_HEARTBEAT_SEC = 6 * 3600
# Paperspace host change: cannot probe remote PID; reclaim after short silence.
FOREIGN_HOST_STALE_HEARTBEAT_SEC = 15 * 60
_FOREIGN_STALE_ENV = 'VALIDATED_RG_GPU_LANE_FOREIGN_STALE_SEC'

# Nested hold depth per resolved persistent_root (same process only).
_hold_depth: dict[str, int] = {}


@dataclass(frozen=True, slots=True)
class ExecutionKey:
    """Logical lock identity for shared M3 batch scheduling."""

    name: str
    owner: str = 'campaign_b'


def foreign_stale_heartbeat_sec(
    explicit: int | None = None,
) -> int:
    """Resolve foreign-host stale threshold (arg > env > default 15 min)."""
    if explicit is not None:
        return int(explicit)
    raw = os.environ.get(_FOREIGN_STALE_ENV)
    if raw is not None and str(raw).strip():
        return int(raw)
    return FOREIGN_HOST_STALE_HEARTBEAT_SEC


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
    foreign_stale_sec: int,
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
        # Different machine: only reclaim on (shorter) heartbeat expiry.
        heartbeat = _parse_iso(doc.get('heartbeat_at') or doc.get('acquired_at'))
        if heartbeat is None:
            return 'missing_heartbeat_foreign_host'
        age = (now - heartbeat).total_seconds()
        if age > float(foreign_stale_sec):
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
            'Same-host dead pid reclaims immediately; foreign-host reclaim uses the '
            'shorter FOREIGN_HOST_STALE_HEARTBEAT_SEC threshold. Live holders must '
            'refresh heartbeats during long work.'
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


def refresh_gpu_lane_heartbeat(persistent_root: Path) -> dict[str, Any] | None:
    """Update ``heartbeat_at`` if this process holds the lease.

    Preserves owner / pid / hostname / depth / acquired_at. Returns the updated
    lease dict, or None if this process does not hold the lease (best-effort).
    """
    persistent_root = Path(persistent_root)
    path = gpu_lane_path(persistent_root)
    if not path.is_file():
        return None
    doc = read_gpu_lane_lease(persistent_root)
    if not isinstance(doc, dict):
        return None
    same_process = (
        int(doc.get('pid') or 0) == os.getpid()
        and str(doc.get('hostname') or '') == socket.gethostname()
    )
    if not same_process:
        return None
    now = utc_now()
    updated = dict(doc)
    updated['heartbeat_at'] = now
    atomic_write_json(path, updated)
    return updated


def acquire_gpu_lock(
    persistent_root: Path,
    *,
    owner: str = 'campaign_b',
    stale_heartbeat_sec: int = DEFAULT_STALE_HEARTBEAT_SEC,
    foreign_stale_sec: int | None = None,
) -> dict[str, Any]:
    """Acquire exclusive GPU lane lease or fail closed.

    Same-process nested calls increment depth and refresh heartbeat.
    Foreign-host stale threshold defaults to 15 min (see
    ``foreign_stale_heartbeat_sec``).
    """
    persistent_root = Path(persistent_root)
    key = _root_key(persistent_root)
    path = gpu_lane_path(persistent_root)
    pid = os.getpid()
    hostname = socket.gethostname()
    now = datetime.now(timezone.utc)
    foreign_sec = foreign_stale_heartbeat_sec(foreign_stale_sec)

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
            foreign_stale_sec=int(foreign_sec),
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
                'Wait for the holder to finish, or reclaim only after the process is dead '
                f'(foreign host: after {foreign_sec}s without heartbeat).'
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


def try_reclaim_gpu_lane_lease(
    persistent_root: Path,
    *,
    stale_heartbeat_sec: int = DEFAULT_STALE_HEARTBEAT_SEC,
    foreign_stale_sec: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Reclaim GPU lane lease using the same stale rules as acquire.

    Without ``force``, deletes only when same-host dead PID or
    foreign/same-host stale heartbeat. With ``force``, deletes even if the
    lease looks live.

    Returns a result dict with ``action``, ``path``, and optional ``reason``.
    """
    persistent_root = Path(persistent_root)
    path = gpu_lane_path(persistent_root)
    key = _root_key(persistent_root)
    if not path.is_file():
        return {
            'action': 'noop',
            'path': str(path),
            'reason': 'no_lease_file',
        }

    doc = read_gpu_lane_lease(persistent_root) or {}
    now = datetime.now(timezone.utc)
    foreign_sec = foreign_stale_heartbeat_sec(foreign_stale_sec)
    stale_reason = _lease_stale_reason(
        doc,
        stale_heartbeat_sec=int(stale_heartbeat_sec),
        foreign_stale_sec=int(foreign_sec),
        now=now,
    )

    if stale_reason is None and not force:
        return {
            'action': 'refused',
            'path': str(path),
            'reason': 'lease_looks_live',
            'lease': {
                'owner': doc.get('owner'),
                'pid': doc.get('pid'),
                'hostname': doc.get('hostname'),
                'heartbeat_at': doc.get('heartbeat_at'),
            },
            'hint': (
                'Pass --force to delete a lease that still looks live '
                '(only if you are sure no GPU consumer is running).'
            ),
        }

    reason = stale_reason if stale_reason is not None else 'force'
    try:
        path.unlink()
    except OSError as exc:
        return {
            'action': 'error',
            'path': str(path),
            'reason': f'unlink_failed:{exc}',
        }
    _hold_depth.pop(key, None)
    return {
        'action': 'reclaimed',
        'path': str(path),
        'reason': reason,
        'previous': {
            'owner': doc.get('owner'),
            'pid': doc.get('pid'),
            'hostname': doc.get('hostname'),
            'heartbeat_at': doc.get('heartbeat_at'),
        },
    }


def force_release_gpu_lane_lease(persistent_root: Path) -> dict[str, Any]:
    """Delete the GPU lane lease file even if it looks live."""
    return try_reclaim_gpu_lane_lease(persistent_root, force=True)


@contextmanager
def gpu_lane_lease(
    persistent_root: Path,
    *,
    owner: str = 'campaign_b',
    stale_heartbeat_sec: int = DEFAULT_STALE_HEARTBEAT_SEC,
    foreign_stale_sec: int | None = None,
) -> Iterator[dict[str, Any]]:
    """Context manager around ``acquire_gpu_lock`` / ``release_gpu_lock``."""
    payload = acquire_gpu_lock(
        persistent_root,
        owner=owner,
        stale_heartbeat_sec=stale_heartbeat_sec,
        foreign_stale_sec=foreign_stale_sec,
    )
    try:
        yield payload
    finally:
        release_gpu_lock(persistent_root, owner=owner)


def m3_execution_key() -> ExecutionKey:
    return ExecutionKey(name=M3_EXECUTION_KEY)
