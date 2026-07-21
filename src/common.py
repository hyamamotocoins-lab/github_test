from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_for_json(obj: Any) -> tuple[Any, bool]:
    """Replace non-finite floats with ``None`` for JSON serialization.

    Returns ``(cleaned, nonfinite_values_present)``. Callers MUST record the
    flag and must NOT emit ``CERTIFIED`` from sanitized exploratory payloads —
    nullified floats are diagnostic only, never a rigorous bound.
    """
    found = False

    def walk(value: Any) -> Any:
        nonlocal found
        if value is None or isinstance(value, (str, bytes, bool)):
            return value
        if isinstance(value, int) and not isinstance(value, bool):
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                found = True
                return None
            return value
        if isinstance(value, complex):
            if not (math.isfinite(value.real) and math.isfinite(value.imag)):
                found = True
                return None
            return {'real': value.real, 'imag': value.imag}
        if isinstance(value, dict):
            return {key: walk(item) for key, item in value.items()}
        if isinstance(value, tuple):
            return tuple(walk(item) for item in value)
        if isinstance(value, list):
            return [walk(item) for item in value]
        # NumPy scalars / 0-d arrays often expose ``.item()``.
        item = getattr(value, 'item', None)
        shape = getattr(value, 'shape', None)
        if callable(item) and shape == ():
            try:
                return walk(item())
            except Exception:  # noqa: BLE001 — leave non-JSON types to dumps default
                return value
        tolist = getattr(value, 'tolist', None)
        if callable(tolist) and type(value).__name__ == 'ndarray':
            return walk(tolist())
        return value

    return walk(obj), found


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def fsync_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        not_supported = getattr(errno, 'ENOTSUP', errno.EINVAL)
        unsupported = {errno.EINVAL, not_supported, getattr(errno, 'EOPNOTSUPP', not_supported)}
        if exc.errno not in unsupported:
            raise


def fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        fsync_descriptor(descriptor)
    finally:
        os.close(descriptor)


def fsync_file(path: Path) -> None:
    with path.open('rb') as handle:
        fsync_descriptor(handle.fileno())


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.tmp-{uuid.uuid4().hex}')
    with temporary.open('wb') as handle:
        handle.write(payload)
        handle.flush()
        fsync_descriptor(handle.fileno())
    os.replace(temporary, path)
    fsync_directory(path.parent)


def atomic_write_text(path: Path, payload: str) -> None:
    atomic_write_bytes(path, payload.encode('utf-8'))


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_bytes(path, canonical_json_bytes(payload))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def safe_component(value: str) -> str:
    if not value or value in {'.', '..'} or '/' in value or '\\' in value:
        raise ValueError(f'Unsafe path component: {value!r}')
    return value


def hash_tree(root: Path, suffixes: Iterable[str] = ('.py',)) -> str:
    allowed = set(suffixes)
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts):
        relative = path.relative_to(root).as_posix().encode('utf-8')
        digest.update(len(relative).to_bytes(8, 'big'))
        digest.update(relative)
        file_hash = sha256_file(path).encode('ascii')
        digest.update(file_hash)
    return digest.hexdigest()


def directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob('*') if item.is_file())
