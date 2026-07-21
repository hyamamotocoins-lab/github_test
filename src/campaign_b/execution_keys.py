"""Execution-key stubs for Campaign B multi-lane scheduling (Phase 2+).

Phase 1 callers may import these symbols; they record intent only and do not
serialize GPU access yet.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .schemas import screening_only_payload

GPU_LOCK_NAME = 'gpu_lane_0'
M3_EXECUTION_KEY = 'shared_m3_batch'


@dataclass(frozen=True, slots=True)
class ExecutionKey:
    """Logical lock identity (not yet enforced on disk in Phase 1)."""

    name: str
    owner: str = 'phase1_stub'


def gpu_lock_path(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_execution_keys' / f'{GPU_LOCK_NAME}.json'


def acquire_gpu_lock(
    persistent_root: Path,
    *,
    owner: str = 'phase1_stub',
) -> dict[str, Any]:
    """Stub acquire: writes a marker file; does not block or fail on conflict."""
    path = gpu_lock_path(persistent_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema_version': 1,
        'key': GPU_LOCK_NAME,
        'owner': owner,
        'acquired_at': utc_now(),
        'enforced': False,
        'note': 'Phase-1 stub — single-notebook discipline replaces real lock.',
        **screening_only_payload(),
    }
    atomic_write_json(path, payload)
    return payload


def release_gpu_lock(persistent_root: Path, *, owner: str = 'phase1_stub') -> bool:
    """Stub release: removes marker if present. Returns True if removed."""
    path = gpu_lock_path(persistent_root)
    if not path.is_file():
        return False
    try:
        path.unlink()
        return True
    except OSError:
        return False


def m3_execution_key() -> ExecutionKey:
    return ExecutionKey(name=M3_EXECUTION_KEY)
