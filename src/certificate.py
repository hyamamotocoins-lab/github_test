"""Finite Perron / Collatz–Wielandt certificate for M5."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Sequence

from .interval_kernel import (
    IntervalKernelError,
    ProofInterval,
    construct,
    divide,
    multiply,
    sum_outward,
)


class CertificateError(RuntimeError):
    """Raised when a Collatz–Wielandt certificate cannot be formed."""


@dataclass(frozen=True, slots=True)
class PositiveRationalVector:
    components: tuple[Fraction, ...]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.components or len(self.components) != len(self.labels):
            raise CertificateError('Perron vector dimension/label mismatch.')
        if len(self.labels) != len(set(self.labels)):
            raise CertificateError('Perron vector labels must be unique.')
        for value in self.components:
            if not isinstance(value, Fraction) or value <= 0:
                raise CertificateError('Perron vector components must be strictly positive rationals.')

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'representation': 'positive_rational',
            'labels': list(self.labels),
            'components': [
                {
                    'numerator_hex': format(value.numerator, 'x'),
                    'denominator_hex': format(value.denominator, 'x'),
                    'decimal': format(value, 'f'),
                }
                for value in self.components
            ],
        }


@dataclass(frozen=True, slots=True)
class NonnegativeIntervalMatrix:
    labels: tuple[str, ...]
    entries: tuple[tuple[ProofInterval, ...], ...]

    def __post_init__(self) -> None:
        n = len(self.labels)
        if n == 0 or len(self.entries) != n:
            raise CertificateError('Matrix dimension is invalid.')
        if len(self.labels) != len(set(self.labels)):
            raise CertificateError('Matrix labels must be unique.')
        for row in self.entries:
            if len(row) != n:
                raise CertificateError('Matrix is not square.')
            for entry in row:
                if not isinstance(entry, ProofInterval):
                    raise CertificateError('Matrix entries must be ProofInterval values.')
                try:
                    entry.assert_nonnegative()
                except IntervalKernelError as exc:
                    raise CertificateError('Matrix entries must be nonnegative.') from exc

    @property
    def dimension(self) -> int:
        return len(self.labels)


@dataclass(frozen=True, slots=True)
class CollatzBound:
    row_quotients: tuple[ProofInterval, ...]
    q_collatz: ProofInterval
    outside_matrix_tail: ProofInterval
    q_cert: ProofInterval
    margin: ProofInterval
    verdict: str

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'row_quotients': [item.serialize() for item in self.row_quotients],
            'q_collatz': self.q_collatz.serialize(),
            'outside_matrix_tail': self.outside_matrix_tail.serialize(),
            'q_cert': self.q_cert.serialize(),
            'margin': self.margin.serialize(),
            'q_collatz_upper': format(self.q_collatz.hi, 'f'),
            'outside_matrix_tail_upper': format(self.outside_matrix_tail.hi, 'f'),
            'q_cert_upper': format(self.q_cert.hi, 'f'),
            'q_cert_lower': format(self.q_cert.lo, 'f'),
            'margin_lower': format(self.margin.lo, 'f'),
            'verdict': self.verdict,
        }


def positive_rational_vector(
    values: Sequence[Any],
    labels: Sequence[str],
) -> PositiveRationalVector:
    components: list[Fraction] = []
    for value in values:
        if isinstance(value, Fraction):
            components.append(value)
        elif isinstance(value, int) and not isinstance(value, bool):
            components.append(Fraction(value))
        elif isinstance(value, str):
            components.append(Fraction(value))
        else:
            raise CertificateError('Perron vector entries must be rational-compatible.')
    return PositiveRationalVector(tuple(components), tuple(labels))


def nonnegative_interval_matrix(
    entries: Sequence[Sequence[Any]],
    labels: Sequence[str],
) -> NonnegativeIntervalMatrix:
    converted: list[tuple[ProofInterval, ...]] = []
    for row in entries:
        converted.append(
            tuple(
                item if isinstance(item, ProofInterval) else construct(item)
                for item in row
            )
        )
    return NonnegativeIntervalMatrix(tuple(labels), tuple(converted))


def matrix_vector_outward(
    matrix: NonnegativeIntervalMatrix,
    vector: PositiveRationalVector,
) -> tuple[ProofInterval, ...]:
    if matrix.labels != vector.labels:
        raise CertificateError('Matrix/vector label ordering mismatch.')
    result: list[ProofInterval] = []
    for row in matrix.entries:
        terms = [
            multiply(row[index], construct(vector.components[index]))
            for index in range(matrix.dimension)
        ]
        result.append(sum_outward(terms))
    return tuple(result)


def collatz_certificate(
    matrix: NonnegativeIntervalMatrix,
    vector: PositiveRationalVector,
    *,
    outside_matrix_tail: ProofInterval | Any = 0,
) -> CollatzBound:
    if any(component <= 0 for component in vector.components):
        raise CertificateError('w_a <= 0 is forbidden.')
    product = matrix_vector_outward(matrix, vector)
    quotients: list[ProofInterval] = []
    for index, component in enumerate(vector.components):
        quotients.append(divide(product[index], construct(component)))

    q_cw = quotients[0]
    for quotient in quotients[1:]:
        q_cw = construct(min(q_cw.lo, quotient.lo), max(q_cw.hi, quotient.hi))

    tail = (
        outside_matrix_tail
        if isinstance(outside_matrix_tail, ProofInterval)
        else construct(outside_matrix_tail)
    )
    try:
        tail.assert_nonnegative()
    except IntervalKernelError as exc:
        raise CertificateError('outside-matrix tail must be nonnegative.') from exc

    q_cert = q_cw.add(tail)
    one = construct(1)
    margin = one.subtract(q_cert)

    if q_cert.hi < 1:
        verdict = 'ONE_STEP_CERTIFIED'
    elif q_cert.lo >= 1:
        verdict = 'NOT_CERTIFIED'
    else:
        verdict = 'BLOCKED_MATH'

    return CollatzBound(
        row_quotients=tuple(quotients),
        q_collatz=q_cw,
        outside_matrix_tail=tail,
        q_cert=q_cert,
        margin=margin,
        verdict=verdict,
    )


def fail_closed_verdict(bound: CollatzBound) -> str:
    """Return a fail-closed certification status from a Collatz bound."""
    if bound.verdict == 'ONE_STEP_CERTIFIED' and bound.q_cert.hi < 1:
        return 'ONE_STEP_CERTIFIED'
    if bound.verdict == 'NOT_CERTIFIED' and bound.q_cert.lo >= 1:
        return 'NOT_CERTIFIED'
    return 'BLOCKED_MATH'
