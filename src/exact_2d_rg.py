from __future__ import annotations

from fractions import Fraction
from typing import Any, Iterable

from .exact_arithmetic import ExactArithmeticError, RationalInterval
from .su2_special_functions import wilson_character_coefficient_bounds


def normalized_fourier_ratio(
    beta: Fraction, dimension_n: int, series_terms: int = 96,
) -> RationalInterval:
    if not isinstance(dimension_n, int) or isinstance(dimension_n, bool) or dimension_n < 2:
        raise ExactArithmeticError('Nontrivial irrep dimension n must be at least two.')
    trivial = wilson_character_coefficient_bounds(1, beta, series_terms)
    coefficient = wilson_character_coefficient_bounds(dimension_n, beta, series_terms)
    denominator = trivial.scale(Fraction(dimension_n))
    ratio = coefficient.divide_positive(denominator)
    if ratio.lo < 0 or ratio.hi > 1:
        raise ExactArithmeticError('Normalized Fourier ratio left the expected [0,1] range.')
    return ratio


def exact_2d_rg_trajectory(
    beta: Fraction, dimensions: Iterable[int], steps: int, series_terms: int = 96,
) -> dict[int, tuple[RationalInterval, ...]]:
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ExactArithmeticError('RG steps must be a nonnegative integer.')
    result: dict[int, tuple[RationalInterval, ...]] = {}
    for dimension in dimensions:
        current = normalized_fourier_ratio(beta, dimension, series_terms)
        trajectory = [current]
        for _ in range(steps):
            current = current.positive_power(4)
            trajectory.append(current)
        result[dimension] = tuple(trajectory)
    return result


def trajectory_payload(
    beta: Fraction, dimensions: tuple[int, ...], steps: int, series_terms: int, decimal_places: int,
) -> dict[str, Any]:
    trajectories = exact_2d_rg_trajectory(beta, dimensions, steps, series_terms)
    return {
        'rigor': 'RIGOROUS_RATIONAL_INTERVAL_RECURRENCE',
        'theorem': (
            'For normalized central Fourier ratios r_n=a_n/(n*a_1), '
            'fourfold Haar convolution for a 2x2 block gives r_n_next=r_n^4.'
        ),
        'beta': {'numerator': beta.numerator, 'denominator': beta.denominator},
        'steps': steps,
        'trajectories': {
            str(n): [interval.to_payload(decimal_places) for interval in trajectory]
            for n, trajectory in trajectories.items()
        },
    }
