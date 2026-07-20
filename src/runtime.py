from __future__ import annotations

import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .common import atomic_write_json, fsync_descriptor, fsync_directory, read_json, utc_now

try:
    import torch
except ImportError:
    torch = None

PERSIST_ROOT_ENV = 'VALIDATED_RG_PERSIST_ROOT'
PERSIST_ACK_ENV = 'VALIDATED_RG_PERSIST_ACK'
PERSIST_ACK_TOKEN = 'I_CONFIRM_THIS_PATH_IS_PERSISTENT'


class PersistentRootError(RuntimeError):
    '''Raised when durable storage has not been explicitly established.'''


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def validate_persistent_root(raw: str | None = None, acknowledgement: str | None = None) -> Path:
    raw = raw if raw is not None else os.environ.get(PERSIST_ROOT_ENV)
    acknowledgement = acknowledgement if acknowledgement is not None else os.environ.get(PERSIST_ACK_ENV)
    if not raw:
        raise PersistentRootError(f'{PERSIST_ROOT_ENV} is required; no computation was started.')
    if acknowledgement != PERSIST_ACK_TOKEN:
        raise PersistentRootError(
            f'Set {PERSIST_ACK_ENV}={PERSIST_ACK_TOKEN} only after confirming shutdown persistence.'
        )
    root = Path(raw).expanduser().resolve()
    forbidden = (
        Path('/tmp'), Path('/var/tmp'), Path('/dev/shm'), Path('/private/tmp'),
        Path('/private/var/folders'),
    )
    if root == Path('/') or any(root == item or _is_relative_to(root, item) for item in forbidden):
        raise PersistentRootError(f'Ephemeral or unsafe checkpoint root rejected: {root}')
    if root == Path('/content') or (_is_relative_to(root, Path('/content')) and not (
        _is_relative_to(root, Path('/content/drive')) or _is_relative_to(root, Path('/content/gdrive'))
    )):
        raise PersistentRootError('/content local storage is ephemeral; use a mounted Drive path.')
    root.mkdir(parents=True, exist_ok=True)
    probe = root / f'.write-probe-{uuid.uuid4().hex}'
    payload = utc_now().encode('utf-8')
    with probe.open('wb') as handle:
        handle.write(payload)
        handle.flush()
        fsync_descriptor(handle.fileno())
    if probe.read_bytes() != payload:
        raise PersistentRootError('Persistent-root write/read probe failed.')
    probe.unlink()
    fsync_directory(root)
    marker = root / '.validated_rg_persistent_root.json'
    if marker.exists():
        stored = read_json(marker)
        if stored.get('resolved_root') != str(root):
            raise PersistentRootError('Persistent-root marker does not match the resolved path.')
    else:
        atomic_write_json(marker, {'schema_version': 1, 'resolved_root': str(root), 'acknowledged_at': utc_now()})
    for relative in (
        'project', 'cache/wigner', 'cache/fusion', 'cache/armillary',
        'cache/contraction_paths',
    ):
        (root / relative).mkdir(parents=True, exist_ok=True)
    fsync_directory(root)
    return root


def configure_numerics() -> None:
    if torch is not None and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def environment_info() -> dict[str, Any]:
    configure_numerics()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    info: dict[str, Any] = {
        'captured_at': utc_now(),
        'python': sys.version,
        'platform': platform.platform(),
        'numpy': np.__version__,
        'torch': getattr(torch, '__version__', None),
        'cuda_available': cuda_available,
        'cuda_runtime': getattr(getattr(torch, 'version', None), 'cuda', None) if torch is not None else None,
        'cudnn': torch.backends.cudnn.version() if cuda_available else None,
        'tf32_matmul': torch.backends.cuda.matmul.allow_tf32 if cuda_available else None,
        'tf32_cudnn': torch.backends.cudnn.allow_tf32 if cuda_available else None,
        'execution_environment': (
            'paperspace_gradient'
            if Path('/notebooks').is_dir() and Path('/storage').is_dir()
            else 'generic'
        ),
        'paperspace_notebooks_mount': Path('/notebooks').is_dir(),
        'paperspace_storage_mount': Path('/storage').is_dir(),
        'configured_persistent_root': os.environ.get(PERSIST_ROOT_ENV),
    }
    if cuda_available:
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update({
            'gpu_index': index,
            'gpu_name': properties.name,
            'gpu_total_memory': properties.total_memory,
            'gpu_capability': list(torch.cuda.get_device_capability(index)),
            'gpu_count': torch.cuda.device_count(),
        })
    return info


RUNTIME_COMPATIBILITY_KEYS = (
    'python', 'platform', 'numpy', 'torch', 'cuda_available', 'cuda_runtime', 'cudnn',
    'tf32_matmul', 'tf32_cudnn', 'execution_environment', 'gpu_name',
    'gpu_capability', 'gpu_count',
)


def runtime_compatibility_signature(info: dict[str, Any] | None = None) -> dict[str, Any]:
    captured = environment_info() if info is None else info
    return {key: captured.get(key) for key in RUNTIME_COMPATIBILITY_KEYS}
