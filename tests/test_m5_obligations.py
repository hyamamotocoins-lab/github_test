from __future__ import annotations

from fractions import Fraction

import numpy as np

from src.m5_obligations import (
    _frobenius_fraction,
    _matrix_to_fractions,
    _sqrt_outward,
    evaluate_gpu_rounding,
    evaluate_normalization_denominator,
)


def test_fraction_frobenius_is_exact_for_binary_floats() -> None:
    matrix = np.array([[0.5, -0.25], [0.0, 0.125]], dtype=np.float64)
    square = _frobenius_fraction(_matrix_to_fractions(matrix))
    assert square == Fraction(1, 4) + Fraction(1, 16) + Fraction(1, 64)
    root = _sqrt_outward(square)
    assert root * root >= square


def test_gpu_rounding_closes_on_self_consistent_pipeline() -> None:
    from src.forward_ad import regroup_matrix
    from src.normalization import normalize_array

    projected = np.eye(16, dtype=np.float64)
    normalized = normalize_array(regroup_matrix(projected @ projected))
    result = evaluate_gpu_rounding(
        {'projected_primal': projected, 'normalized_primal': normalized},
        source_paths=('synthetic',),
        source_hashes=('a' * 64,),
    )
    assert result.status == 'RIGOROUS'
    assert result.upper_bound is not None


def test_normalization_requires_positive_center_scale() -> None:
    center = np.ones((16, 16), dtype=np.float64)
    result = evaluate_normalization_denominator(
        {'coarse_primal': center},
        source_paths=('synthetic',),
        source_hashes=('b' * 64,),
    )
    assert result.status == 'RIGOROUS'
    zero = evaluate_normalization_denominator(
        {'coarse_primal': np.zeros((16, 16), dtype=np.float64)},
        source_paths=('synthetic',),
        source_hashes=('c' * 64,),
    )
    assert zero.status == 'BLOCKED_MATH'
