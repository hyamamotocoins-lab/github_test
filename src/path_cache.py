from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .common import atomic_write_json, canonical_json_bytes, read_json


@dataclass(frozen=True, slots=True)
class PathKey:
    graph_hash: str
    block_shapes: tuple[tuple[int, ...], ...]
    dtype: str
    device: str
    memory_limit_bytes: int
    algorithm_version: str = 'm3-path-v1'

    def payload(self) -> dict[str, Any]:
        return {
            'graph_hash': self.graph_hash,
            'block_shapes': [list(shape) for shape in self.block_shapes],
            'dtype': self.dtype, 'device': self.device,
            'memory_limit_bytes': self.memory_limit_bytes,
            'algorithm_version': self.algorithm_version,
        }

    def digest(self) -> str:
        return hashlib.sha256(canonical_json_bytes(self.payload())).hexdigest()


class ContractionPathCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.hits = 0
        self.misses = 0
        if path.is_file():
            payload = read_json(path)
            if (
                not isinstance(payload, dict)
                or payload.get('schema_version') != 1
                or not isinstance(payload.get('entries'), dict)
            ):
                raise ValueError('Malformed contraction path cache.')
            self.entries: dict[str, dict[str, Any]] = payload['entries']
        else:
            self.entries = {}

    def _save(self) -> None:
        atomic_write_json(self.path, {
            'schema_version': 1,
            'entries': {key: self.entries[key] for key in sorted(self.entries)},
        })

    def get_or_create(
        self, key: PathKey, builder: Callable[[], dict[str, Any]],
    ) -> tuple[dict[str, Any], bool]:
        digest = key.digest()
        existing = self.entries.get(digest)
        if existing is not None:
            if existing.get('key') != key.payload():
                raise RuntimeError('Contraction path cache digest collision.')
            self.hits += 1
            return existing['path'], True
        path = builder()
        if not isinstance(path, dict):
            raise TypeError('Contraction path builder must return metadata.')
        self.entries[digest] = {'key': key.payload(), 'path': path}
        self.misses += 1
        self._save()
        return path, False

    def stats(self) -> dict[str, int | str]:
        return {
            'entries': len(self.entries), 'hits': self.hits,
            'misses': self.misses,
            'cache_sha256': hashlib.sha256(
                canonical_json_bytes({
                    key: self.entries[key] for key in sorted(self.entries)
                })
            ).hexdigest(),
        }


def graph_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
