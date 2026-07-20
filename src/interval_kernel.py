"""Proof-critical outward interval arithmetic for M5 one-step certification.

Uses exact rational endpoints (fractions.Fraction). Silent binary64 fallback is
forbidden. Denominator intervals that contain zero are rejected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .exact_arithmetic import (
    ExactArithmeticError,
    RationalInterval,
    fraction_decimal_text,
    fraction_from_payload,
    fraction_payload,
)


class IntervalKernelError(ExactArithmeticError):
    """Raised when a proof-critical interval operation is invalid."""


DEFAULT_BACKEND = 'rational_fraction'
DEFAULT_PRECISION_BITS = 256
DEFAULT_ROUNDING = 'outward'


def _require_finite_fraction(value: Fraction, label: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f'{label} must be fractions.Fraction, not {type(value)!r}.')
    # Fraction is always exact/finite; reject pathological zero denominators.
    if value.denominator <= 0:
        raise IntervalKernelError(f'{label} has a nonpositive denominator.')


def _parse_scalar(value: Any) -> Fraction:
    if isinstance(value, Fraction):
        _require_finite_fraction(value, 'value')
        return value
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal)):
        raise IntervalKernelError(
            'Interval endpoints must be Fraction, int, Decimal, or decimal string.'
        )
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        return Fraction(value)
    text = value.strip()
    if not text or text.lower() in {'nan', 'inf', '+inf', '-inf'}:
        raise IntervalKernelError(f'Refused non-finite decimal endpoint: {value!r}')
    # Prefer exact rational parsing for strings like "1/10"; fall back to decimal.
    try:
        if '/' in text:
            return Fraction(text)
        return Fraction(Decimal(text))
    except (InvalidOperation, ValueError, ZeroDivisionError) as exc:
        raise IntervalKernelError(f'Malformed decimal endpoint: {value!r}') from exc


@dataclass(frozen=True, slots=True)
class ProofInterval:
    """Closed scalar interval with provenance for proof-critical arithmetic."""

    lo: Fraction
    hi: Fraction
    arithmetic_backend: str = DEFAULT_BACKEND
    precision_bits: int = DEFAULT_PRECISION_BITS
    rounding_policy: str = DEFAULT_ROUNDING

    def __post_init__(self) -> None:
        _require_finite_fraction(self.lo, 'lo')
        _require_finite_fraction(self.hi, 'hi')
        if self.lo > self.hi:
            raise IntervalKernelError('Interval lower endpoint exceeds upper endpoint.')
        if self.arithmetic_backend != DEFAULT_BACKEND:
            raise IntervalKernelError(
                f'Unsupported arithmetic backend: {self.arithmetic_backend!r}'
            )
        if (
            not isinstance(self.precision_bits, int)
            or isinstance(self.precision_bits, bool)
            or self.precision_bits < 64
        ):
            raise IntervalKernelError('precision_bits must be an integer >= 64.')
        if self.rounding_policy != DEFAULT_ROUNDING:
            raise IntervalKernelError(
                f'Unsupported rounding policy: {self.rounding_policy!r}'
            )

    @classmethod
    def construct(
        cls,
        lower: Any,
        upper: Any | None = None,
        *,
        arithmetic_backend: str = DEFAULT_BACKEND,
        precision_bits: int = DEFAULT_PRECISION_BITS,
        rounding_policy: str = DEFAULT_ROUNDING,
    ) -> 'ProofInterval':
        lo = _parse_scalar(lower)
        hi = lo if upper is None else _parse_scalar(upper)
        return cls(
            lo=lo,
            hi=hi,
            arithmetic_backend=arithmetic_backend,
            precision_bits=precision_bits,
            rounding_policy=rounding_policy,
        )

    @classmethod
    def from_rational_interval(
        cls,
        interval: RationalInterval,
        *,
        arithmetic_backend: str = DEFAULT_BACKEND,
        precision_bits: int = DEFAULT_PRECISION_BITS,
        rounding_policy: str = DEFAULT_ROUNDING,
    ) -> 'ProofInterval':
        if not isinstance(interval, RationalInterval):
            raise TypeError('from_rational_interval requires RationalInterval.')
        return cls(
            lo=interval.lo,
            hi=interval.hi,
            arithmetic_backend=arithmetic_backend,
            precision_bits=precision_bits,
            rounding_policy=rounding_policy,
        )

    def to_rational_interval(self) -> RationalInterval:
        return RationalInterval(self.lo, self.hi)

    def _same_policy(self, other: 'ProofInterval') -> None:
        if not isinstance(other, ProofInterval):
            raise TypeError('ProofInterval arithmetic requires another ProofInterval.')
        if (
            self.arithmetic_backend != other.arithmetic_backend
            or self.precision_bits != other.precision_bits
            or self.rounding_policy != other.rounding_policy
        ):
            raise IntervalKernelError('Mismatched interval arithmetic policy.')

    def add(self, other: 'ProofInterval') -> 'ProofInterval':
        self._same_policy(other)
        return ProofInterval(
            self.lo + other.lo,
            self.hi + other.hi,
            self.arithmetic_backend,
            self.precision_bits,
            self.rounding_policy,
        )

    def subtract(self, other: 'ProofInterval') -> 'ProofInterval':
        self._same_policy(other)
        return ProofInterval(
            self.lo - other.hi,
            self.hi - other.lo,
            self.arithmetic_backend,
            self.precision_bits,
            self.rounding_policy,
        )

    def multiply(self, other: 'ProofInterval') -> 'ProofInterval':
        self._same_policy(other)
        products = (
            self.lo * other.lo,
            self.lo * other.hi,
            self.hi * other.lo,
            self.hi * other.hi,
        )
        return ProofInterval(
            min(products),
            max(products),
            self.arithmetic_backend,
            self.precision_bits,
            self.rounding_policy,
        )

    def divide(self, other: 'ProofInterval') -> 'ProofInterval':
        self._same_policy(other)
        if other.lo <= 0 <= other.hi:
            raise IntervalKernelError(
                'Division by a denominator interval that contains zero is forbidden.'
            )
        if other.lo == 0 or other.hi == 0:
            raise IntervalKernelError(
                'Division by a denominator interval with a zero endpoint is forbidden.'
            )
        reciprocals = (Fraction(1, other.lo), Fraction(1, other.hi))
        recip = ProofInterval(
            min(reciprocals),
            max(reciprocals),
            self.arithmetic_backend,
            self.precision_bits,
            self.rounding_policy,
        )
        return self.multiply(recip)

    def square(self) -> 'ProofInterval':
        if self.lo >= 0:
            return ProofInterval(
                self.lo * self.lo,
                self.hi * self.hi,
                self.arithmetic_backend,
                self.precision_bits,
                self.rounding_policy,
            )
        if self.hi <= 0:
            return ProofInterval(
                self.hi * self.hi,
                self.lo * self.lo,
                self.arithmetic_backend,
                self.precision_bits,
                self.rounding_policy,
            )
        # Interval contains zero: min square is 0, max is max of endpoint squares.
        return ProofInterval(
            Fraction(0),
            max(self.lo * self.lo, self.hi * self.hi),
            self.arithmetic_backend,
            self.precision_bits,
            self.rounding_policy,
        )

    def absolute_upper(self) -> Fraction:
        return max(abs(self.lo), abs(self.hi))

    def positive_lower_assertion(self) -> Fraction:
        if self.lo <= 0:
            raise IntervalKernelError(
                'positive_lower_assertion failed: lower endpoint is not strictly positive.'
            )
        return self.lo

    def assert_nonnegative(self) -> None:
        if self.lo < 0:
            raise IntervalKernelError(
                'Nonnegative enclosure has a negative lower endpoint.'
            )

    def serialize(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'lo': fraction_payload(self.lo),
            'hi': fraction_payload(self.hi),
            'arithmetic_backend': self.arithmetic_backend,
            'precision_bits': self.precision_bits,
            'rounding_policy': self.rounding_policy,
            'decimal_lo': fraction_decimal_text(self.lo),
            'decimal_hi': fraction_decimal_text(self.hi),
        }

    @classmethod
    def deserialize(cls, payload: dict[str, Any]) -> 'ProofInterval':
        if not isinstance(payload, dict):
            raise IntervalKernelError('Serialized interval must be a mapping.')
        if payload.get('schema_version') != 1:
            raise IntervalKernelError('Unsupported interval schema version.')
        return cls(
            lo=fraction_from_payload(payload['lo']),
            hi=fraction_from_payload(payload['hi']),
            arithmetic_backend=str(payload.get('arithmetic_backend', DEFAULT_BACKEND)),
            precision_bits=int(payload.get('precision_bits', DEFAULT_PRECISION_BITS)),
            rounding_policy=str(payload.get('rounding_policy', DEFAULT_ROUNDING)),
        )


def construct(lower: Any, upper: Any | None = None, **kwargs: Any) -> ProofInterval:
    return ProofInterval.construct(lower, upper, **kwargs)


def add(left: ProofInterval, right: ProofInterval) -> ProofInterval:
    return left.add(right)


def subtract(left: ProofInterval, right: ProofInterval) -> ProofInterval:
    return left.subtract(right)


def multiply(left: ProofInterval, right: ProofInterval) -> ProofInterval:
    return left.multiply(right)


def divide(left: ProofInterval, right: ProofInterval) -> ProofInterval:
    return left.divide(right)


def square(value: ProofInterval) -> ProofInterval:
    return value.square()


def absolute_upper(value: ProofInterval) -> Fraction:
    return value.absolute_upper()


def positive_lower_assertion(value: ProofInterval) -> Fraction:
    return value.positive_lower_assertion()


def sum_outward(values: Iterable[ProofInterval]) -> ProofInterval:
    iterator = iter(values)
    try:
        total = next(iterator)
    except StopIteration as exc:
        raise IntervalKernelError('sum_outward requires at least one interval.') from exc
    if not isinstance(total, ProofInterval):
        raise TypeError('sum_outward requires ProofInterval values.')
    for item in iterator:
        total = total.add(item)
    return total


def dot_outward(
    left: Sequence[ProofInterval], right: Sequence[ProofInterval],
) -> ProofInterval:
    if len(left) != len(right):
        raise IntervalKernelError('dot_outward dimension mismatch.')
    if not left:
        raise IntervalKernelError('dot_outward requires a positive dimension.')
    products = [left[index].multiply(right[index]) for index in range(len(left))]
    return sum_outward(products)


def serialize(value: ProofInterval) -> dict[str, Any]:
    return value.serialize()


def deserialize(payload: dict[str, Any]) -> ProofInterval:
    return ProofInterval.deserialize(payload)


def is_finite_nonnegative_interval(value: ProofInterval) -> bool:
    return (
        isinstance(value, ProofInterval)
        and value.lo >= 0
        and math.isfinite(float(value.lo)) is not False
    )
