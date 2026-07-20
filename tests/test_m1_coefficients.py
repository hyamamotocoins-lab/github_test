from __future__ import annotations

from fractions import Fraction

from src.su2_special_functions import (
    besseli_integer_positive_series_bounds, normalized_wilson_coefficient_bounds,
    wilson_character_coefficient_bounds,
)

BETA = Fraction(11, 5)


def test_bessel_positive_series_contains_higher_precision_rational_reference() -> None:
    for n in range(0, 17):
        coarse = besseli_integer_positive_series_bounds(n, BETA, 48)
        reference = besseli_integer_positive_series_bounds(n, BETA, 112)
        assert reference.subset_of(coarse)
        assert reference.lo >= 0


def test_coefficient_and_normalized_coefficient_are_positive_and_nested() -> None:
    for n in range(1, 17):
        coarse = wilson_character_coefficient_bounds(n, BETA, 56)
        fine = wilson_character_coefficient_bounds(n, BETA, 96)
        assert fine.subset_of(coarse) and fine.lo > 0
        normalized = normalized_wilson_coefficient_bounds(n, BETA, 96, 120)
        assert normalized.lo > 0 and normalized.hi <= 1
