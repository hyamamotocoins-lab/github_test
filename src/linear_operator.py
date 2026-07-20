from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .armillary import all_link_star_keys
from .common import canonical_json_bytes
from .contraction_backend import ContractionBackend
from .gpu_sharding import ShardPlan, make_shard_plan, shard_order_hash
from .path_cache import ContractionPathCache, PathKey, graph_hash


@dataclass(frozen=True, slots=True)
class SectorBlock:
    label: str
    representations: tuple[int, ...]
    projector: np.ndarray
    weight: float
    offset: int

    @property
    def dimension(self) -> int:
        return int(self.projector.shape[0])

    def payload(self) -> dict[str, Any]:
        contiguous = np.asarray(self.projector, dtype='<f8', order='C')
        return {
            'label': self.label,
            'representations': list(self.representations),
            'shape': list(contiguous.shape), 'weight': self.weight,
            'offset': self.offset,
            'projector_sha256': hashlib.sha256(contiguous.tobytes()).hexdigest(),
        }


class ArmillaryLinearOperator:
    def __init__(
        self, blocks: tuple[SectorBlock, ...], backend: ContractionBackend,
        path_cache: ContractionPathCache, sectors_per_shard: int,
    ) -> None:
        if len(blocks) < 1:
            raise ValueError('M3 operator requires a nonempty sector block set.')
        self.blocks = blocks
        self.backend = backend
        self.path_cache = path_cache
        self.shard_plan = make_shard_plan(len(blocks), sectors_per_shard)
        self.shape = (
            sum(block.dimension for block in blocks),
            sum(block.dimension for block in blocks),
        )
        if self.shape[0] != self.shape[1] or self.shape[0] < 1:
            raise ValueError(f'M3 operator dimension invalid: {self.shape}')
        for expected_offset, block in zip(
            np.cumsum([0, *(item.dimension for item in blocks[:-1])]),
            blocks, strict=True,
        ):
            if block.offset != int(expected_offset):
                raise ValueError('M3 sector block offsets are nondeterministic.')
        self._graph_hash = graph_hash({
            'kind': 'weighted_block_armillary_projector',
            'blocks': [block.payload() for block in blocks],
            'shard_order_hash': shard_order_hash(self.shard_plan.shards),
        })

    @property
    def dimension(self) -> int:
        return self.shape[0]

    @property
    def graph_hash(self) -> str:
        return self._graph_hash

    def metadata(self) -> dict[str, Any]:
        payload = {
            'shape': list(self.shape), 'graph_hash': self.graph_hash,
            'backend': self.backend.name,
            'sector_count': len(self.blocks),
            'shard_plan': self.shard_plan.payload(),
            'shard_order_hash': shard_order_hash(self.shard_plan.shards),
            'blocks': [block.payload() for block in self.blocks],
        }
        payload['metadata_sha256'] = hashlib.sha256(
            canonical_json_bytes(payload)
        ).hexdigest()
        return payload

    def _apply_tensor(
        self, value: torch.Tensor, *, adjoint: bool,
    ) -> torch.Tensor:
        original_vector = value.ndim == 1
        if original_vector:
            value = value[:, None]
        if value.ndim != 2 or value.shape[0] != self.dimension:
            raise ValueError(
                f'M3 operator expected ({self.dimension}, k), got {tuple(value.shape)}'
            )
        value = value.to(device=self.backend.device, dtype=self.backend.dtype)
        result = torch.zeros_like(value)
        memory = self.backend.memory_snapshot()
        memory_limit = int(memory.get('free_bytes', 0))
        for shard in self.shard_plan.shards:
            for block_index in shard:
                block = self.blocks[block_index]
                start = block.offset
                stop = start + block.dimension
                matrix_np = block.projector.T if adjoint else block.projector
                matrix = self.backend.tensor(matrix_np)
                key = PathKey(
                    graph_hash=self.graph_hash,
                    block_shapes=(
                        tuple(matrix.shape), tuple(value[start:stop].shape),
                    ),
                    dtype='float64', device=str(self.backend.device),
                    memory_limit_bytes=memory_limit,
                )
                path, _ = self.path_cache.get_or_create(
                    key,
                    lambda: {
                        'expression': 'ij,jk->ik',
                        'strategy': 'sector-local matrix-free shard',
                    },
                )
                result[start:stop] = block.weight * self.backend.contract(
                    'ij,jk->ik', (matrix, value[start:stop]), path=path,
                )
                del matrix
        return result[:, 0] if original_vector else result

    def matmat_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return self._apply_tensor(value, adjoint=False)

    def rmatmat_tensor(self, value: torch.Tensor) -> torch.Tensor:
        return self._apply_tensor(value, adjoint=True)

    def matvec(self, value: np.ndarray) -> np.ndarray:
        tensor = self.backend.tensor(np.asarray(value, dtype=np.float64))
        return self.backend.to_numpy(self._apply_tensor(tensor, adjoint=False))

    def rmatvec(self, value: np.ndarray) -> np.ndarray:
        tensor = self.backend.tensor(np.asarray(value, dtype=np.float64))
        return self.backend.to_numpy(self._apply_tensor(tensor, adjoint=True))

    def matmat(self, value: np.ndarray) -> np.ndarray:
        tensor = self.backend.tensor(np.asarray(value, dtype=np.float64))
        return self.backend.to_numpy(self._apply_tensor(tensor, adjoint=False))

    def rmatmat(self, value: np.ndarray) -> np.ndarray:
        tensor = self.backend.tensor(np.asarray(value, dtype=np.float64))
        return self.backend.to_numpy(self._apply_tensor(tensor, adjoint=True))

    def explicit_matrix(self) -> np.ndarray:
        result = np.zeros(self.shape, dtype=np.float64)
        for block in self.blocks:
            start = block.offset
            stop = start + block.dimension
            result[start:stop, start:stop] = block.weight * block.projector
        return result

    def frobenius_norm(self) -> float:
        squared = sum(
            float(np.linalg.norm(block.weight * block.projector, 'fro') ** 2)
            for block in self.blocks
        )
        return float(np.sqrt(squared))

    def factor_residual_frobenius(
        self, left: np.ndarray, singular_values: np.ndarray,
        right_t: np.ndarray,
    ) -> float:
        left = np.asarray(left, dtype=np.float64)
        singular_values = np.asarray(singular_values, dtype=np.float64)
        right_t = np.asarray(right_t, dtype=np.float64)
        if (
            left.shape[0] != self.dimension
            or right_t.shape[1] != self.dimension
            or left.shape[1] != singular_values.size
            or right_t.shape[0] != singular_values.size
        ):
            raise ValueError('Low-rank factor shapes do not match the M3 operator.')
        squared = 0.0
        for block in self.blocks:
            start = block.offset
            stop = start + block.dimension
            approximation = (
                left[start:stop] * singular_values[None, :]
            ) @ right_t[:, start:stop]
            difference = block.weight * block.projector - approximation
            squared += float(np.linalg.norm(difference, 'fro') ** 2)
        return float(np.sqrt(squared))


def build_armillary_operator(
    projector_tensors: Mapping[str, np.ndarray],
    backend: ContractionBackend, path_cache_path: Path,
    *, sectors_per_shard: int, weight_base: float = 0.5,
) -> ArmillaryLinearOperator:
    if not 0.0 < weight_base <= 1.0:
        raise ValueError('M3 sector weight base must lie in (0,1].')
    blocks: list[SectorBlock] = []
    offset = 0
    expected_names: set[str] = set()
    for key in all_link_star_keys():
        label = ''.join(str(value) for value in key.representations)
        name = f'projector_{label}'
        expected_names.add(name)
        if name not in projector_tensors:
            raise ValueError(f'M2 parent tensor shard is missing: {name}')
        projector = np.asarray(projector_tensors[name], dtype=np.float64)
        expected_dimension = int(np.prod([value + 1 for value in key.representations]))
        if projector.shape != (expected_dimension, expected_dimension):
            raise ValueError(f'M2 parent tensor shard shape changed: {name}')
        if not np.isfinite(projector).all():
            raise ValueError(f'M2 parent tensor shard is nonfinite: {name}')
        weight = float(weight_base ** sum(key.representations))
        blocks.append(SectorBlock(
            label, key.representations, projector.copy(), weight, offset,
        ))
        offset += expected_dimension
    if set(projector_tensors) != expected_names:
        raise ValueError('M2 parent tensor shard set changed.')
    return ArmillaryLinearOperator(
        tuple(blocks), backend, ContractionPathCache(path_cache_path),
        sectors_per_shard,
    )
