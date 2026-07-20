from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Iterable, TypeVar

import torch

T = TypeVar('T')


class BlockedResourceError(RuntimeError):
    '''Raised after the fixed M3 OOM retry budget is exhausted.'''


@dataclass(frozen=True, slots=True)
class ShardPlan:
    shards: tuple[tuple[int, ...], ...]
    sectors_per_shard: int
    sector_count: int

    def payload(self) -> dict[str, object]:
        return {
            'sector_count': self.sector_count,
            'sectors_per_shard': self.sectors_per_shard,
            'shards': [list(shard) for shard in self.shards],
        }


@dataclass(frozen=True, slots=True)
class OOMRecoveryResult(Generic[T]):
    value: T
    final_shard_size: int
    oom_retries: int
    attempted_shard_sizes: tuple[int, ...]


def make_shard_plan(sector_count: int, sectors_per_shard: int) -> ShardPlan:
    if sector_count < 1 or sectors_per_shard < 1:
        raise ValueError('Shard counts must be positive.')
    shards = tuple(
        tuple(range(start, min(start + sectors_per_shard, sector_count)))
        for start in range(0, sector_count, sectors_per_shard)
    )
    return ShardPlan(shards, sectors_per_shard, sector_count)


def is_cuda_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or (
        isinstance(exc, RuntimeError)
        and 'out of memory' in str(exc).lower()
    )


def run_with_oom_recovery(
    operation: Callable[[int], T], initial_shard_size: int,
    *, min_shard_size: int = 1, max_oom_retries: int = 3,
) -> OOMRecoveryResult[T]:
    if (
        initial_shard_size < min_shard_size
        or min_shard_size < 1
        or max_oom_retries < 1
    ):
        raise ValueError('Invalid OOM recovery policy.')
    shard_size = initial_shard_size
    retries = 0
    attempted: list[int] = []
    while True:
        attempted.append(shard_size)
        try:
            return OOMRecoveryResult(
                operation(shard_size), shard_size, retries, tuple(attempted),
            )
        except BaseException as exc:
            if not is_cuda_oom(exc):
                raise
            retries += 1
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if retries >= max_oom_retries or shard_size <= min_shard_size:
                raise BlockedResourceError(
                    f'GPU OOM after {retries} retries; '
                    f'attempted shard sizes {attempted}.'
                ) from exc
            shard_size = max(min_shard_size, shard_size // 2)


def require_memory_headroom(
    free_bytes: int, total_bytes: int, required_fraction: float,
) -> None:
    if total_bytes <= 0 or free_bytes < 0:
        raise ValueError('GPU memory snapshot is invalid.')
    if not 0.0 < required_fraction < 1.0:
        raise ValueError('Required memory headroom must lie in (0,1).')
    fraction = free_bytes / total_bytes
    if fraction < required_fraction:
        raise BlockedResourceError(
            f'GPU free-memory fraction {fraction:.6f} is below '
            f'the required {required_fraction:.6f}.'
        )


def move_tensors_to_cpu(
    tensors: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for name, tensor in tensors.items():
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise TypeError('GPU drain expects a string-to-tensor mapping.')
        result[name] = tensor.detach().to(device='cpu').contiguous()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    return result


def shard_order_hash(shards: Iterable[Iterable[int]]) -> str:
    import hashlib
    from .common import canonical_json_bytes

    payload = [list(shard) for shard in shards]
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
