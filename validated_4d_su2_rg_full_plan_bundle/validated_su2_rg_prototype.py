from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, getcontext
from fractions import Fraction
from math import factorial
from pathlib import Path
from typing import Iterable

getcontext().prec = 60


@dataclass(frozen=True)
class RationalInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError("Invalid interval")

    def positive_power(self, exponent: int) -> "RationalInterval":
        if exponent < 0 or self.lo < 0:
            raise ValueError("Only nonnegative intervals and powers are supported.")
        return RationalInterval(self.lo**exponent, self.hi**exponent)


def frac_to_decimal(x: Fraction, places: int, rounding: str) -> Decimal:
    q = Decimal(x.numerator) / Decimal(x.denominator)
    unit = Decimal(1).scaleb(-places)
    return q.quantize(unit, rounding=rounding)


def interval_string(interval: RationalInterval, places: int = 18) -> str:
    lo = frac_to_decimal(interval.lo, places, ROUND_FLOOR)
    hi = frac_to_decimal(interval.hi, places, ROUND_CEILING)
    return f"[{lo}, {hi}]"


def exp_positive_series_bounds(x: Fraction, terms: int = 140) -> RationalInterval:
    if x < 0:
        raise ValueError("x must be nonnegative")
    total = Fraction(1)
    term = Fraction(1)
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    ratio = x / (terms + 1)
    if ratio >= 1:
        raise ValueError("Increase terms.")
    return RationalInterval(total, total + term * ratio / (1 - ratio))


def besseli_integer_positive_series_bounds(
    n: int, x: Fraction, terms: int = 100
) -> RationalInterval:
    if n < 0 or x < 0:
        raise ValueError("Require n >= 0 and x >= 0.")
    y = x / 2
    term = y**n / factorial(n)
    total = term
    for k in range(1, terms + 1):
        term *= y * y / (k * (n + k))
        total += term
    next_ratio = y * y / ((terms + 1) * (n + terms + 1))
    if next_ratio >= 1:
        raise ValueError("Increase terms.")
    return RationalInterval(
        total, total + term * next_ratio / (1 - next_ratio)
    )


def su2_wilson_character_coefficient_bounds(
    dimension_n: int, beta: Fraction
) -> RationalInterval:
    bessel = besseli_integer_positive_series_bounds(dimension_n, beta)
    scale = Fraction(2 * dimension_n, 1) / beta
    return RationalInterval(scale * bessel.lo, scale * bessel.hi)


def normalized_character_tail(beta: Fraction, cutoff_n: int) -> RationalInterval:
    exp_beta = exp_positive_series_bounds(beta)
    partial_lo = Fraction(0)
    partial_hi = Fraction(0)
    for n in range(1, cutoff_n + 1):
        a_n = su2_wilson_character_coefficient_bounds(n, beta)
        partial_lo += n * a_n.lo
        partial_hi += n * a_n.hi
    return RationalInterval(
        max(Fraction(0), exp_beta.lo - partial_hi) / exp_beta.hi,
        (exp_beta.hi - partial_lo) / exp_beta.lo,
    )


def normalized_fourier_ratio(beta: Fraction, dimension_n: int) -> RationalInterval:
    a_trivial = su2_wilson_character_coefficient_bounds(1, beta)
    a_n = su2_wilson_character_coefficient_bounds(dimension_n, beta)
    return RationalInterval(
        a_n.lo / (dimension_n * a_trivial.hi),
        a_n.hi / (dimension_n * a_trivial.lo),
    )


def two_dimensional_exact_rg(
    beta: Fraction, dimensions: Iterable[int], steps: int
) -> dict[int, list[RationalInterval]]:
    result = {}
    for n in dimensions:
        current = normalized_fourier_ratio(beta, n)
        trajectory = [current]
        for _ in range(steps):
            current = current.positive_power(4)
            trajectory.append(current)
        result[n] = trajectory
    return result



def normalized_character_derivative_tail_upper(
    beta: Fraction, cutoff_n: int
) -> RationalInterval:
    """
    Casimir-gradient tail for the normalized Wilson plaquette weight.

    We use ||grad chi_j||_infty <= (2j+1)^2/2 and
    I_n(beta) <= (beta/2)^n/n! exp(beta^2/(4(n+1))).
    """
    if beta <= 0 or cutoff_n < 0:
        raise ValueError("Require beta > 0 and cutoff_n >= 0.")
    x = beta / 2
    n = cutoff_n + 1
    first = Fraction(n**3, 1) * x**n / factorial(n)
    ratio = x * Fraction((n + 1) ** 2, n**3)
    if ratio >= 1:
        raise ValueError("Cutoff too small.")
    tail_sum = first / (1 - ratio)
    exp_beta = exp_positive_series_bounds(beta)
    correction = exp_positive_series_bounds(
        beta * beta / Fraction(4 * (cutoff_n + 2))
    )
    return RationalInterval(
        Fraction(0),
        correction.hi * tail_sum / (beta * exp_beta.lo),
    )

def main() -> None:
    beta = Fraction(11, 5)
    cutoffs = [6, 8, 10, 12, 14, 16]
    print("beta =", beta)
    print("Initial 4D-cell character tail:")
    for cutoff in cutoffs:
        tail = normalized_character_tail(beta, cutoff)
        cell = RationalInterval(216 * tail.lo, 216 * tail.hi)
        print(cutoff, interval_string(tail), interval_string(cell))

    print("\nCasimir-gradient tails:")
    for cutoff in cutoffs:
        tail = normalized_character_derivative_tail_upper(beta, cutoff)
        print(cutoff, interval_string(tail), interval_string(RationalInterval(6*tail.lo, 6*tail.hi)))
    print("\nExact 2D RG:")
    for n, trajectory in two_dimensional_exact_rg(
        beta, dimensions=[2, 3, 4], steps=3
    ).items():
        print("n =", n)
        for step, interval in enumerate(trajectory):
            print(" ", step, interval_string(interval, 24))


if __name__ == "__main__":
    main()
