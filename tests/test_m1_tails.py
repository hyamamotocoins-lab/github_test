from __future__ import annotations

from fractions import Fraction

from src.tail_bounds import gradient_tail_enclosure, value_tail_enclosure

BETA = Fraction(11, 5)
CUTOFFS = (6, 8, 10, 12, 14, 16)
PROTOTYPE_VALUE = {
    6: ('0.002677892949651685', '0.002677892949651686'),
    8: ('0.000068908927820552', '0.000068908927820553'),
    10: ('0.000001078814460529', '0.000001078814460530'),
    12: ('0.000000011303940764', '0.000000011303940765'),
    14: ('0.000000000084628276', '0.000000000084628277'),
    16: ('0.000000000000474661', '0.000000000000474662'),
}
PROTOTYPE_GRADIENT_UPPER = {
    6: '0.009776809377036234', 8: '0.000317113428056905',
    10: '0.000006015764950566', 12: '0.000000074161320899',
    14: '0.000000000638962130', 16: '0.000000000004054931',
}


def test_value_tail_nonnegative_monotone_precision_nested_and_regression() -> None:
    previous = None
    for cutoff in CUTOFFS:
        coarse = value_tail_enclosure(BETA, cutoff, 64, 80)
        fine = value_tail_enclosure(BETA, cutoff, 96, 120)
        assert fine.subset_of(coarse) and fine.lo >= 0
        if previous is not None:
            assert fine.hi <= previous.hi
        previous = fine
        fixture = RationalFixture(*PROTOTYPE_VALUE[cutoff])
        assert fine.subset_of(fixture.interval)
        assert fine.scale(Fraction(216)).hi == 216 * fine.hi


def test_gradient_tail_nonnegative_monotone_precision_nested_and_factors() -> None:
    previous = None
    for cutoff in CUTOFFS:
        coarse = gradient_tail_enclosure(BETA, cutoff, 80)
        fine = gradient_tail_enclosure(BETA, cutoff, 120)
        assert fine.subset_of(coarse) and fine.lo == 0
        if previous is not None:
            assert fine.hi <= previous.hi
        previous = fine
        assert fine.hi <= Fraction(PROTOTYPE_GRADIENT_UPPER[cutoff])
        assert fine.scale(Fraction(6)).hi == 6 * fine.hi
        assert fine.scale(Fraction(216)).hi == 216 * fine.hi


class RationalFixture:
    def __init__(self, lo: str, hi: str) -> None:
        from src.exact_arithmetic import RationalInterval
        self.interval = RationalInterval(Fraction(lo), Fraction(hi))
