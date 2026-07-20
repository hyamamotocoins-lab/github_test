from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from typing import Any


class ExactArithmeticError(ValueError):
    '''Raised when a proof-path interval operation is invalid.'''


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f'{name} must be fractions.Fraction, not {type(value)!r}.')


def fraction_payload(value: Fraction) -> dict[str, str]:
    _require_fraction(value, 'value')
    return {
        'numerator_hex': format(value.numerator, 'x'),
        'denominator_hex': format(value.denominator, 'x'),
    }


def fraction_from_payload(payload: dict[str, Any]) -> Fraction:
    if not isinstance(payload, dict):
        raise ExactArithmeticError('Fraction payload must be a mapping.')
    try:
        numerator = int(payload['numerator_hex'], 16)
        denominator = int(payload['denominator_hex'], 16)
    except (KeyError, TypeError, ValueError) as exc:
        raise ExactArithmeticError('Malformed hexadecimal fraction payload.') from exc
    if denominator <= 0:
        raise ExactArithmeticError('Fraction denominator must be positive.')
    return Fraction(numerator, denominator)


def outward_decimal(value: Fraction, places: int, upper: bool) -> str:
    _require_fraction(value, 'value')
    if not isinstance(places, int) or isinstance(places, bool) or places < 1:
        raise ExactArithmeticError('places must be a positive integer.')
    rounding = ROUND_CEILING if upper else ROUND_FLOOR
    with localcontext() as context:
        context.prec = places + 100
        context.rounding = rounding
        quotient = Decimal(value.numerator) / Decimal(value.denominator)
        unit = Decimal(1).scaleb(-places)
        return str(quotient.quantize(unit, rounding=rounding))


def fraction_decimal_text(value: Fraction) -> str:
    """Render a Fraction as a plain decimal string without Fraction.__format__.

    Python 3.11 and earlier reject format(Fraction, 'f'); Paperspace uses 3.11.
    """
    _require_fraction(value, 'value')
    with localcontext() as context:
        context.prec = max(50, value.denominator.bit_length() + 16)
        return format(Decimal(value.numerator) / Decimal(value.denominator), 'f')


@dataclass(frozen=True, slots=True)
class RationalInterval:
    lo: Fraction
    hi: Fraction

    def __post_init__(self) -> None:
        _require_fraction(self.lo, 'lo')
        _require_fraction(self.hi, 'hi')
        if self.lo > self.hi:
            raise ExactArithmeticError('Interval lower endpoint exceeds upper endpoint.')

    @classmethod
    def point(cls, value: Fraction) -> 'RationalInterval':
        _require_fraction(value, 'value')
        return cls(value, value)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'RationalInterval':
        if not isinstance(payload, dict):
            raise ExactArithmeticError('Interval payload must be a mapping.')
        return cls(fraction_from_payload(payload['lo']), fraction_from_payload(payload['hi']))

    def to_payload(self, decimal_places: int = 36) -> dict[str, Any]:
        return {
            'lo': fraction_payload(self.lo),
            'hi': fraction_payload(self.hi),
            'decimal_lo': outward_decimal(self.lo, decimal_places, upper=False),
            'decimal_hi': outward_decimal(self.hi, decimal_places, upper=True),
        }

    def __add__(self, other: 'RationalInterval') -> 'RationalInterval':
        if not isinstance(other, RationalInterval):
            raise TypeError('RationalInterval addition requires another RationalInterval.')
        return RationalInterval(self.lo + other.lo, self.hi + other.hi)

    def __sub__(self, other: 'RationalInterval') -> 'RationalInterval':
        if not isinstance(other, RationalInterval):
            raise TypeError('RationalInterval subtraction requires another RationalInterval.')
        return RationalInterval(self.lo - other.hi, self.hi - other.lo)

    def __mul__(self, other: 'RationalInterval') -> 'RationalInterval':
        if not isinstance(other, RationalInterval):
            raise TypeError('RationalInterval multiplication requires another RationalInterval.')
        products = (self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi)
        return RationalInterval(min(products), max(products))

    def scale(self, factor: Fraction) -> 'RationalInterval':
        _require_fraction(factor, 'factor')
        return (
            RationalInterval(self.lo * factor, self.hi * factor)
            if factor >= 0
            else RationalInterval(self.hi * factor, self.lo * factor)
        )

    def divide_positive(self, denominator: 'RationalInterval') -> 'RationalInterval':
        if not isinstance(denominator, RationalInterval) or denominator.lo <= 0:
            raise ExactArithmeticError('Denominator interval must be strictly positive.')
        return self * RationalInterval(Fraction(1, denominator.hi), Fraction(1, denominator.lo))

    def positive_power(self, exponent: int) -> 'RationalInterval':
        if not isinstance(exponent, int) or isinstance(exponent, bool) or exponent < 0:
            raise ExactArithmeticError('Exponent must be a nonnegative integer.')
        if self.lo < 0:
            raise ExactArithmeticError('Positive-power proof path requires a nonnegative interval.')
        return RationalInterval(self.lo**exponent, self.hi**exponent)

    def subset_of(self, outer: 'RationalInterval') -> bool:
        return isinstance(outer, RationalInterval) and outer.lo <= self.lo and self.hi <= outer.hi

    def overlaps(self, other: 'RationalInterval') -> bool:
        return isinstance(other, RationalInterval) and max(self.lo, other.lo) <= min(self.hi, other.hi)

    def assert_nonnegative(self) -> None:
        if self.lo < 0:
            raise ExactArithmeticError('A rigorous nonnegative enclosure has a negative lower endpoint.')


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode('utf-8')
