from __future__ import annotations

from fractions import Fraction
from math import factorial
from typing import Any

from .exact_arithmetic import ExactArithmeticError, RationalInterval
from .su2_special_functions import (
    exp_positive_series_bounds, normalized_wilson_coefficient_bounds,
    wilson_character_coefficient_bounds,
)

VALUE_TAIL_PROOF = (
    'At theta=0, chi_n(I)=n and exp(beta)=sum_{n>=1} n*a_n. '
    'Positivity and |chi_n|<=n give ||tail||_infty <= exp(-beta)*sum_{n>N} n*a_n.'
)
GRADIENT_TAIL_PROOF = (
    'Use the explicit weight-sum bound ||grad chi_j||_infty <= n^2/2, '
    'I_n(beta) <= (beta/2)^n/n!*exp(beta^2/(4(n+1))), and a decreasing term-ratio majorant.'
)


def coefficient_enclosures(
    beta: Fraction, max_dimension: int, series_terms: int, exp_terms: int, decimal_places: int,
) -> dict[str, Any]:
    if max_dimension < 1:
        raise ExactArithmeticError('max_dimension must be positive.')
    coefficients: dict[str, Any] = {}
    for n in range(1, max_dimension + 1):
        raw = wilson_character_coefficient_bounds(n, beta, series_terms)
        normalized = normalized_wilson_coefficient_bounds(n, beta, series_terms, exp_terms)
        coefficients[str(n)] = {
            'j2': n - 1, 'dimension': n,
            'raw_a_n': raw.to_payload(decimal_places),
            'normalized_exp_minus_beta_a_n': normalized.to_payload(decimal_places),
        }
    return {
        'rigor': 'RIGOROUS_RATIONAL_POSITIVE_SERIES',
        'formula': 'a_n(beta)=2*n*I_n(beta)/beta',
        'normalization': 'w_bar=exp(-beta)*sum_n a_n chi_n',
        'beta': {'numerator': beta.numerator, 'denominator': beta.denominator},
        'series_terms': series_terms, 'exp_terms': exp_terms,
        'coefficients': coefficients,
    }


def value_tail_enclosure(
    beta: Fraction, cutoff_n: int, series_terms: int = 96, exp_terms: int = 120,
) -> RationalInterval:
    if not isinstance(cutoff_n, int) or isinstance(cutoff_n, bool) or cutoff_n < 0:
        raise ExactArithmeticError('cutoff_n must be a nonnegative integer.')
    exp_beta = exp_positive_series_bounds(beta, exp_terms)
    partial_lo = Fraction(0)
    partial_hi = Fraction(0)
    for n in range(1, cutoff_n + 1):
        coefficient = wilson_character_coefficient_bounds(n, beta, series_terms)
        partial_lo += n * coefficient.lo
        partial_hi += n * coefficient.hi
    lower = max(Fraction(0), Fraction(1) - partial_hi / exp_beta.lo)
    upper = Fraction(1) - partial_lo / exp_beta.hi
    if upper < 0:
        raise ExactArithmeticError('Value-tail upper bound became negative; precision is inconsistent.')
    result = RationalInterval(lower, upper)
    result.assert_nonnegative()
    return result


def gradient_tail_enclosure(
    beta: Fraction, cutoff_n: int, exp_terms: int = 120,
) -> RationalInterval:
    if not isinstance(beta, Fraction) or beta <= 0:
        raise ExactArithmeticError('beta must be a positive Fraction.')
    if not isinstance(cutoff_n, int) or isinstance(cutoff_n, bool) or cutoff_n < 0:
        raise ExactArithmeticError('cutoff_n must be a nonnegative integer.')
    first_n = cutoff_n + 1
    half_beta = beta / 2
    first = Fraction(first_n**3, 1) * half_beta**first_n / factorial(first_n)
    ratio = half_beta * Fraction((first_n + 1) ** 2, first_n**3)
    if ratio >= 1:
        raise ExactArithmeticError('Gradient-tail n-series ratio is not contractive at this cutoff.')
    n_series_upper = first / (1 - ratio)
    correction = exp_positive_series_bounds(beta * beta / Fraction(4 * (cutoff_n + 2)), exp_terms)
    exp_beta = exp_positive_series_bounds(beta, exp_terms)
    upper = correction.hi * n_series_upper / (beta * exp_beta.lo)
    result = RationalInterval(Fraction(0), upper)
    result.assert_nonnegative()
    return result


def tail_table(
    beta: Fraction, cutoffs: tuple[int, ...], series_terms: int, exp_terms: int,
    decimal_places: int, kind: str,
) -> dict[str, Any]:
    if kind not in {'value', 'gradient'}:
        raise ExactArithmeticError(f'Unknown tail kind: {kind!r}')
    entries: dict[str, Any] = {}
    previous: RationalInterval | None = None
    for cutoff in cutoffs:
        interval = (
            value_tail_enclosure(beta, cutoff, series_terms, exp_terms)
            if kind == 'value'
            else gradient_tail_enclosure(beta, cutoff, exp_terms)
        )
        if previous is not None and interval.hi > previous.hi:
            raise ExactArithmeticError(f'{kind} tail upper bound is not monotone at cutoff {cutoff}.')
        previous = interval
        entry: dict[str, Any] = {'tail': interval.to_payload(decimal_places)}
        if kind == 'value':
            entry['cell_216'] = interval.scale(Fraction(216)).to_payload(decimal_places)
        else:
            entry['fine_link_6'] = interval.scale(Fraction(6)).to_payload(decimal_places)
            entry['coarse_cell_216'] = interval.scale(Fraction(216)).to_payload(decimal_places)
        entries[str(cutoff)] = entry
    return {
        'rigor': 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
        'kind': kind, 'proof': VALUE_TAIL_PROOF if kind == 'value' else GRADIENT_TAIL_PROOF,
        'metric_normalization': 'C2(j)=j(j+1)',
        'entries': entries,
    }
