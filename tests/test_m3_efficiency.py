"""CPU unit tests for M3 efficiency helpers (storage + blockwise reference)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from src.armillary import (
    all_link_star_keys, build_armillary_sector, checkpoint_tensor_shards,
)
from src.contraction_backend import TorchBackend
from src.linear_operator import SectorBlock, build_armillary_operator
from src.m3_orchestrator import (
    M3CompatibilityError,
    _blockwise_reference_matvec,
    _checkpoint_keep_count,
    _prune_committed_checkpoints,
    _reference_singular_values,
)
from src.path_cache import ContractionPathCache


@lru_cache(maxsize=1)
def _projectors() -> dict[str, np.ndarray]:
    return checkpoint_tensor_shards(
        build_armillary_sector(key) for key in all_link_star_keys()
    )


def _operator(tmp_path: Path):
    return build_armillary_operator(
        _projectors(), TorchBackend(use_cuda=False),
        tmp_path / 'paths.json', sectors_per_shard=16,
    )


def test_blockwise_reference_matches_dense_without_global_alloc(
    tmp_path: Path,
) -> None:
    operator = _operator(tmp_path)
    rng = np.random.default_rng(20260722)
    vector = rng.standard_normal(operator.dimension)
    reference = _blockwise_reference_matvec(operator, vector)
    dense = operator.explicit_matrix() @ vector
    np.testing.assert_allclose(reference, dense, rtol=0.0, atol=1e-14)
    adjoint = _blockwise_reference_matvec(operator, vector, adjoint=True)
    np.testing.assert_allclose(
        adjoint, operator.explicit_matrix().T @ vector, rtol=0.0, atol=1e-14,
    )


def test_reference_singular_values_match_block_svd(tmp_path: Path) -> None:
    operator = _operator(tmp_path)
    fast, metadata = _reference_singular_values(operator)
    exact = np.sort(np.concatenate([
        np.linalg.svd(block.weight * block.projector, compute_uv=False)
        for block in operator.blocks
    ]))[::-1]
    # Projector-rank mode omits exact zeros; tops and tail residuals must agree.
    nonzero = exact[exact > 1e-14]
    np.testing.assert_allclose(fast, nonzero, rtol=1e-12, atol=1e-12)
    target_rank = 16
    assert float(np.linalg.norm(fast[target_rank:])) == pytest.approx(
        float(np.linalg.norm(exact[target_rank:])), rel=0.0, abs=1e-12,
    )
    assert metadata['svd_fallback_blocks'] == 0
    assert metadata['reference_spectrum_mode'] == 'projector_rank_exact'
    assert metadata['projector_fast_blocks'] == len(operator.blocks)


def test_reference_singular_values_svd_fallback_for_unsafe_block(
    tmp_path: Path,
) -> None:
    from src.linear_operator import ArmillaryLinearOperator

    bad = np.array([[1.0, 0.5], [0.0, 1.0]], dtype=np.float64)
    block = SectorBlock('bad', (0, 0, 0, 0), bad, 2.0, 0)
    cache = ContractionPathCache(tmp_path / 'fallback_paths.json')
    operator = ArmillaryLinearOperator(
        (block,), TorchBackend(use_cuda=False), cache, sectors_per_shard=1,
    )
    values, metadata = _reference_singular_values(operator)
    expected = np.linalg.svd(2.0 * bad, compute_uv=False)
    np.testing.assert_allclose(values, expected, rtol=0.0, atol=0.0)
    assert metadata['svd_fallback_blocks'] == 1
    assert metadata['reference_spectrum_mode'] == 'projector_rank_with_svd_fallback'


def test_prune_committed_checkpoints_keeps_newest_and_skips_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / 'checkpoints'
    root.mkdir()
    for index in (1, 2, 3, 4):
        path = root / f'ckpt_{index:06d}'
        path.mkdir()
        (path / 'COMMITTED').write_text('ok\n', encoding='utf-8')
        (path / 'payload.bin').write_bytes(b'x' * (10 * index))
    # Uncommitted and symlink must be ignored for removal candidates / follow.
    pending = root / 'ckpt_000099'
    pending.mkdir()
    (pending / 'payload.bin').write_bytes(b'pending')
    link = root / 'ckpt_000050'
    link.symlink_to(root / 'ckpt_000004')

    result = _prune_committed_checkpoints(root, keep=2)
    assert result['removed'] == 2
    assert set(result['removed_names']) == {'ckpt_000001', 'ckpt_000002'}
    assert result['kept'] == ['ckpt_000003', 'ckpt_000004']
    assert result['skipped_symlinks'] == ['ckpt_000050']
    assert (root / 'ckpt_000003').is_dir()
    assert (root / 'ckpt_000004').is_dir()
    assert not (root / 'ckpt_000001').exists()
    assert pending.is_dir()
    assert link.is_symlink()


def test_checkpoint_keep_count_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv('VALIDATED_RG_M3_CHECKPOINT_KEEP', raising=False)
    assert _checkpoint_keep_count() == 1
    monkeypatch.setenv('VALIDATED_RG_M3_CHECKPOINT_KEEP', '3')
    assert _checkpoint_keep_count() == 3
    monkeypatch.setenv('VALIDATED_RG_M3_CHECKPOINT_KEEP', '0')
    with pytest.raises(M3CompatibilityError):
        _checkpoint_keep_count()
    monkeypatch.setenv('VALIDATED_RG_M3_CHECKPOINT_KEEP', 'nope')
    with pytest.raises(M3CompatibilityError):
        _checkpoint_keep_count()
