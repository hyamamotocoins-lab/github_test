from __future__ import annotations

from fractions import Fraction
from functools import lru_cache
from math import factorial

from .exact_arithmetic import ExactArithmeticError, RationalInterval


def _validate_terms(terms: int) -> None:
    if not isinstance(terms, int) or isinstance(terms, bool) or terms < 1:
        raise ExactArithmeticError('Series terms must be a positive integer.')


@lru_cache(maxsize=4096)
def exp_positive_series_bounds(x: Fraction, terms: int = 120) -> RationalInterval:
    if not isinstance(x, Fraction) or x < 0:
        raise ExactArithmeticError('Exponential proof path requires a nonnegative Fraction.')
    _validate_terms(terms)
    total = Fraction(1)
    term = Fraction(1)
    for k in range(1, terms + 1):
        term *= x / k
        total += term
    next_ratio = x / (terms + 1)
    if next_ratio >= 1:
        raise ExactArithmeticError('Exponential remainder ratio is not contractive; increase terms.')
    remainder_upper = term * next_ratio / (1 - next_ratio)
    return RationalInterval(total, total + remainder_upper)


@lru_cache(maxsize=16384)
def besseli_integer_positive_series_bounds(
    n: int, beta: Fraction, terms: int = 96,
) -> RationalInterval:
    if not isinstance(n, int) or isinstance(n, bool) or n < 0:
        raise ExactArithmeticError('Bessel order must be a nonnegative integer.')
    if not isinstance(beta, Fraction) or beta < 0:
        raise ExactArithmeticError('beta must be a nonnegative Fraction.')
    _validate_terms(terms)
    half_beta = beta / 2
    term = half_beta**n / factorial(n)
    total = term
    for k in range(1, terms + 1):
        term *= half_beta * half_beta / (k * (n + k))
        total += term
    next_ratio = half_beta * half_beta / ((terms + 1) * (n + terms + 1))
    if next_ratio >= 1:
        raise ExactArithmeticError('Bessel remainder ratio is not contractive; increase terms.')
    remainder_upper = term * next_ratio / (1 - next_ratio)
    return RationalInterval(total, total + remainder_upper)


def wilson_character_coefficient_bounds(
    dimension_n: int, beta: Fraction, terms: int = 96,
) -> RationalInterval:
    if not isinstance(dimension_n, int) or isinstance(dimension_n, bool) or dimension_n < 1:
        raise ExactArithmeticError('Irrep dimension n must be a positive integer.')
    if not isinstance(beta, Fraction) or beta <= 0:
        raise ExactArithmeticError('Wilson beta must be a positive Fraction.')
    bessel = besseli_integer_positive_series_bounds(dimension_n, beta, terms)
    return bessel.scale(Fraction(2 * dimension_n, 1) / beta)


def normalized_wilson_coefficient_bounds(
    dimension_n: int, beta: Fraction, terms: int = 96, exp_terms: int = 120,
) -> RationalInterval:
    raw = wilson_character_coefficient_bounds(dimension_n, beta, terms)
    exp_beta = exp_positive_series_bounds(beta, exp_terms)
    return raw.divide_positive(exp_beta)
