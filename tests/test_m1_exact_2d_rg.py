from __future__ import annotations

from fractions import Fraction

from src.exact_2d_rg import exact_2d_rg_trajectory, normalized_fourier_ratio
from src.exact_arithmetic import RationalInterval

BETA = Fraction(11, 5)
PROTOTYPE_STEP_ZERO = {
    2: ('0.464479025270420310654070', '0.464479025270420310654071'),
    3: ('0.155492681326508526083508', '0.155492681326508526083509'),
    4: ('0.040408076198124330426319', '0.040408076198124330426320'),
}


def test_normalized_ratios_precision_nesting_and_prototype_regression() -> None:
    for n in (2, 3, 4):
        coarse = normalized_fourier_ratio(BETA, n, 64)
        fine = normalized_fourier_ratio(BETA, n, 96)
        fixture = RationalInterval(Fraction(PROTOTYPE_STEP_ZERO[n][0]), Fraction(PROTOTYPE_STEP_ZERO[n][1]))
        assert fine.subset_of(coarse) and fine.subset_of(fixture)


def test_exact_2d_interval_recurrence_is_fourth_power() -> None:
    trajectories = exact_2d_rg_trajectory(BETA, (2, 3, 4), 3, 96)
    for trajectory in trajectories.values():
        assert len(trajectory) == 4
        for before, after in zip(trajectory, trajectory[1:]):
            assert after == before.positive_power(4)
            assert after.lo >= 0 and after.hi <= before.hi
