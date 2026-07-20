from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from fractions import Fraction
from math import factorial
from typing import Any


class IndependentVerificationError(RuntimeError):
    '''Raised when the independent M1 convolution path does not enclose the primary result.'''


@dataclass(frozen=True, slots=True)
class VerifierInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        if not isinstance(self.lo, Fraction) or not isinstance(self.hi, Fraction) or self.lo > self.hi:
            raise IndependentVerificationError('Invalid independent-verifier interval.')

    def multiply(self, other: 'VerifierInterval') -> 'VerifierInterval':
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return VerifierInterval(min(products), max(products))

    def fourth_power(self) -> 'VerifierInterval':
        if self.lo < 0:
            raise IndependentVerificationError('Verifier fourth power requires nonnegative input.')
        square = self.multiply(self)
        return square.multiply(square)

    def divide_positive(self, other: 'VerifierInterval') -> 'VerifierInterval':
        if other.lo <= 0:
            raise IndependentVerificationError('Verifier denominator is not strictly positive.')
        return self.multiply(VerifierInterval(Fraction(1, other.hi), Fraction(1, other.lo)))

    def scale_positive(self, value: Fraction) -> 'VerifierInterval':
        if value < 0:
            raise IndependentVerificationError('Verifier positive scaling received a negative value.')
        return VerifierInterval(self.lo * value, self.hi * value)


def _direct_bessel_interval(n: int, beta: Fraction, terms: int) -> VerifierInterval:
    half = beta / 2
    total = Fraction(0)
    for k in range(terms + 1):
        total += half ** (2 * k + n) / (factorial(k) * factorial(n + k))
    next_term = half ** (2 * (terms + 1) + n) / (factorial(terms + 1) * factorial(n + terms + 1))
    next_ratio = half * half / ((terms + 2) * (n + terms + 2))
    if next_ratio >= 1:
        raise IndependentVerificationError('Independent Bessel remainder is not contractive.')
    return VerifierInterval(total, total + next_term / (1 - next_ratio))


def _direct_coefficient_interval(n: int, beta: Fraction, terms: int) -> VerifierInterval:
    bessel = _direct_bessel_interval(n, beta, terms)
    return bessel.scale_positive(Fraction(2 * n, 1) / beta)


@dataclass(frozen=True, slots=True)
class DiagonalConvolutionOperator:
    coefficients: dict[int, VerifierInterval]

    def fourfold_block(self) -> 'DiagonalConvolutionOperator':
        blocked: dict[int, VerifierInterval] = {}
        for n, coefficient in self.coefficients.items():
            blocked[n] = coefficient.fourth_power().scale_positive(Fraction(1, n**3))
        return DiagonalConvolutionOperator(blocked)

    def normalized_ratio(self, n: int) -> VerifierInterval:
        if 1 not in self.coefficients or n not in self.coefficients:
            raise IndependentVerificationError('Requested convolution sector is absent.')
        denominator = self.coefficients[1].scale_positive(Fraction(n))
        return self.coefficients[n].divide_positive(denominator)


def _fraction_from_hex(payload: dict[str, Any]) -> Fraction:
    return Fraction(int(payload['numerator_hex'], 16), int(payload['denominator_hex'], 16))


def _primary_interval(payload: dict[str, Any]) -> VerifierInterval:
    return VerifierInterval(_fraction_from_hex(payload['lo']), _fraction_from_hex(payload['hi']))


def _verifier_interval_payload(interval: VerifierInterval) -> dict[str, Any]:
    return {
        'lo': {
            'numerator_hex': format(interval.lo.numerator, 'x'),
            'denominator_hex': format(interval.lo.denominator, 'x'),
        },
        'hi': {
            'numerator_hex': format(interval.hi.numerator, 'x'),
            'denominator_hex': format(interval.hi.denominator, 'x'),
        },
    }


def independent_convolution_verify(
    primary_payload: dict[str, Any], beta: Fraction, dimensions: tuple[int, ...],
    steps: int, verifier_terms: int,
) -> dict[str, Any]:
    coefficients = {
        n: _direct_coefficient_interval(n, beta, verifier_terms)
        for n in (1,) + dimensions
    }
    operator = DiagonalConvolutionOperator(coefficients)
    verifier_trajectories: dict[int, list[VerifierInterval]] = {
        n: [operator.normalized_ratio(n)] for n in dimensions
    }
    for _ in range(steps):
        operator = operator.fourfold_block()
        for n in dimensions:
            verifier_trajectories[n].append(operator.normalized_ratio(n))
    checks: list[dict[str, Any]] = []
    for n in dimensions:
        primary_steps = primary_payload['trajectories'][str(n)]
        if len(primary_steps) != steps + 1:
            raise IndependentVerificationError('Primary trajectory length mismatch.')
        for step, primary_step in enumerate(primary_steps):
            primary = _primary_interval(primary_step)
            independent = verifier_trajectories[n][step]
            overlaps = max(primary.lo, independent.lo) <= min(primary.hi, independent.hi)
            primary_inside = independent.lo <= primary.lo and primary.hi <= independent.hi
            if not overlaps or not primary_inside:
                raise IndependentVerificationError(
                    f'Independent convolution failed containment for n={n}, step={step}.'
                )
            checks.append({
                'dimension': n, 'step': step, 'overlaps': overlaps,
                'primary_inside_independent': primary_inside,
            })
    return {
        'status': 'PASS',
        'method': (
            'Independent direct-factorial Bessel sums plus a finite diagonal Peter-Weyl '
            'convolution operator using chi_n*chi_m=delta_nm*chi_n/n.'
        ),
        'does_not_call_primary_recurrence': True,
        'verifier_series_terms': verifier_terms,
        'arb_status': 'AVAILABLE_NOT_USED' if importlib.util.find_spec('flint') else 'NOT_AVAILABLE_NOT_REQUIRED',
        'independent_trajectories': {
            str(n): [_verifier_interval_payload(interval) for interval in trajectory]
            for n, trajectory in verifier_trajectories.items()
        },
        'checks': checks,
    }
