from __future__ import annotations

from fractions import Fraction

from src.interval_kernel import construct
from src.m4_config import M4Config
from src.m7_lineage import (
    apply_s2_residual_model,
    apply_s3_cutoff_model,
    effective_projected_rank,
    is_perfect_square,
)


def test_effective_projected_rank_maps_to_square() -> None:
    assert effective_projected_rank(16) == 16
    assert effective_projected_rank(24) == 25
    assert effective_projected_rank(32) == 36
    assert effective_projected_rank(48) == 49
    assert effective_projected_rank(64) == 64
    assert is_perfect_square(25)


def test_m4_config_allows_square_ranks_only() -> None:
    from dataclasses import asdict
    base = asdict(M4Config())
    assert M4Config(**{**base, 'projected_rank': 64}).projected_rank == 64
    try:
        M4Config(**{**base, 'projected_rank': 24})
    except ValueError:
        pass
    else:
        raise AssertionError('non-square projected_rank must be rejected')


def test_residual_model_shrinks_high_rank() -> None:
    entries = [[construct('2'), construct('0')], [construct('0'), construct('2')]]
    shrunk = apply_s2_residual_model(
        entries,
        parent_rank=16,
        target_rank=64,
        oversampling=24,
        power_iterations=3,
        residual_fraction=Fraction(3, 5),
    )
    assert shrunk[0][0].hi < Fraction(2)
    assert shrunk[0][0].hi < 1  # core 2/5 after residual wipe is 0.8


def test_s3_cutoff_model_can_shrink_below_one() -> None:
    entries = [[construct('2'), construct('0')], [construct('0'), construct('2')]]
    shrunk = apply_s3_cutoff_model(
        entries,
        parent_j2_max=1,
        j2_max=4,
        channel_policy='certified_pruned',
        block_geometry='approved_geometry_B',
        truncation_fraction=Fraction(7, 10),
    )
    assert shrunk[0][0].hi < 1
