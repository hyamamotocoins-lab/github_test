from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from src.armillary import (
    all_link_star_keys, build_armillary_sector, checkpoint_tensor_shards,
)
from src.contraction_backend import TorchBackend
from src.gpu_sharding import (
    BlockedResourceError, make_shard_plan, run_with_oom_recovery,
)
from src.linear_operator import build_armillary_operator
from src.rsvd import randomized_svd
from src.triad_atrg import triad_from_rsvd


@lru_cache(maxsize=1)
def _projectors() -> dict[str, np.ndarray]:
    return checkpoint_tensor_shards(
        build_armillary_sector(key) for key in all_link_star_keys()
    )


def _operator(tmp_path: Path, *, cuda: bool = False):
    return build_armillary_operator(
        _projectors(), TorchBackend(use_cuda=cuda),
        tmp_path / ('cuda_paths.json' if cuda else 'cpu_paths.json'),
        sectors_per_shard=16,
    )


def test_matrix_free_matvec_matches_explicit_and_reuses_path_cache(
    tmp_path: Path,
) -> None:
    operator = _operator(tmp_path)
    rng = np.random.default_rng(20260720)
    vector = rng.standard_normal(operator.dimension)
    explicit = operator.explicit_matrix()
    actual = operator.matvec(vector)
    np.testing.assert_allclose(actual, explicit @ vector, rtol=1e-13, atol=1e-13)
    first_stats = operator.path_cache.stats()
    assert first_stats['entries'] > 0
    operator.matvec(vector)
    second_stats = operator.path_cache.stats()
    assert second_stats['entries'] == first_stats['entries']
    assert second_stats['hits'] > first_stats['hits']

    reconstructed = build_armillary_operator(
        _projectors(), TorchBackend(use_cuda=False),
        tmp_path / 'cpu_paths.json', sectors_per_shard=16,
    )
    assert reconstructed.graph_hash == operator.graph_hash
    assert reconstructed.path_cache.stats()['entries'] == first_stats['entries']


def test_matrix_free_adjoint_consistency(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    rng = np.random.default_rng(17)
    x = rng.standard_normal(operator.dimension)
    y = rng.standard_normal(operator.dimension)
    left = float(np.vdot(operator.matvec(x), y))
    right = float(np.vdot(x, operator.rmatvec(y)))
    scale = max(1.0, abs(left), abs(right))
    assert abs(left - right) / scale < 1e-13


def test_fixed_seed_rsvd_matches_blockwise_explicit_svd(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    result = randomized_svd(
        operator, target_rank=16, oversampling=16,
        power_iterations=2, seed=20260720,
    )
    explicit_values = np.sort(np.concatenate([
        np.linalg.svd(block.weight * block.projector, compute_uv=False)
        for block in operator.blocks
    ]))[::-1]
    np.testing.assert_allclose(
        result.singular_values, explicit_values[:16],
        rtol=2e-6, atol=1e-10,
    )
    optimal_residual = float(np.linalg.norm(explicit_values[16:]))
    assert result.residual_frobenius <= optimal_residual * (1.0 + 1e-7) + 1e-11
    assert result.orthogonality_residual < 1e-10

    repeated = randomized_svd(
        operator, target_rank=16, oversampling=16,
        power_iterations=2, seed=20260720,
    )
    np.testing.assert_array_equal(result.left, repeated.left)
    np.testing.assert_array_equal(
        result.singular_values, repeated.singular_values,
    )
    np.testing.assert_array_equal(result.right_t, repeated.right_t)


def test_triad_factorization_uses_three_small_factors(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    result = randomized_svd(
        operator, target_rank=8, oversampling=8,
        power_iterations=2, seed=7,
    )
    triad = triad_from_rsvd(operator, result)
    rng = np.random.default_rng(9)
    vector = rng.standard_normal(operator.dimension)
    expected = result.left @ (
        result.singular_values * (result.right_t @ vector)
    )
    np.testing.assert_allclose(triad.apply(vector), expected, rtol=1e-13, atol=1e-13)
    assert triad.rank == 8
    assert triad.left.shape == (729, 8)
    assert triad.core.shape == (8, 8)
    assert triad.right.shape == (8, 729)


def test_shard_order_and_oom_recovery_are_deterministic() -> None:
    plan = make_shard_plan(64, 16)
    assert [len(shard) for shard in plan.shards] == [16, 16, 16, 16]
    attempts: list[int] = []

    def succeeds_after_two_ooms(size: int) -> int:
        attempts.append(size)
        if len(attempts) <= 2:
            raise RuntimeError('CUDA out of memory synthetic test')
        return size

    recovered = run_with_oom_recovery(
        succeeds_after_two_ooms, 16, max_oom_retries=3,
    )
    assert recovered.value == recovered.final_shard_size == 4
    assert recovered.oom_retries == 2
    assert recovered.attempted_shard_sizes == (16, 8, 4)

    with pytest.raises(BlockedResourceError):
        run_with_oom_recovery(
            lambda size: (_ for _ in ()).throw(
                RuntimeError(f'out of memory at shard {size}')
            ),
            16, max_oom_retries=3,
        )


@pytest.mark.gpu
def test_cuda_matrix_free_matches_cpu_and_disables_tf32(tmp_path: Path) -> None:
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    cpu = _operator(tmp_path, cuda=False)
    gpu = _operator(tmp_path, cuda=True)
    rng = np.random.default_rng(20260720)
    value = rng.standard_normal((cpu.dimension, 3))
    np.testing.assert_allclose(
        gpu.matmat(value), cpu.matmat(value), rtol=1e-12, atol=1e-12,
    )
    assert torch.backends.cuda.matmul.allow_tf32 is False
    assert torch.backends.cudnn.allow_tf32 is False
