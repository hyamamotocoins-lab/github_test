from __future__ import annotations

from fractions import Fraction

import pytest

from src.exact_arithmetic import ExactArithmeticError, RationalInterval
from src.su2_representations import Irrep


def test_interval_exact_operations_and_round_trip() -> None:
    left = RationalInterval(Fraction(1, 3), Fraction(1, 2))
    right = RationalInterval(Fraction(2, 5), Fraction(3, 5))
    assert left * right == RationalInterval(Fraction(2, 15), Fraction(3, 10))
    assert left.positive_power(4) == RationalInterval(Fraction(1, 81), Fraction(1, 16))
    assert RationalInterval.from_payload(left.to_payload()) == left
    assert left.subset_of(RationalInterval(Fraction(0), Fraction(1)))


def test_interval_rejects_float_nan_inf_and_negative_radius() -> None:
    with pytest.raises(TypeError):
        RationalInterval(0.0, 1.0)
    with pytest.raises(TypeError):
        RationalInterval(float('nan'), float('inf'))
    with pytest.raises(ExactArithmeticError):
        RationalInterval(Fraction(2), Fraction(1))
    with pytest.raises(ExactArithmeticError):
        RationalInterval(Fraction(-1), Fraction(1)).positive_power(4)


def test_irrep_integer_convention_casimir_and_tensor_product() -> None:
    half = Irrep(1); one = Irrep(2)
    assert half.spin == Fraction(1, 2) and half.dimension == 2
    assert half.casimir == Fraction(3, 4) and one.casimir == Fraction(2)
    assert half.dual() == half and half.reverse_orientation() == half
    assert half.tensor_product(half) == (Irrep(0), Irrep(2))
    with pytest.raises((TypeError, ValueError)):
        Irrep(0.5)
