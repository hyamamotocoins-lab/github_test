from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sympy import Expr, Rational, simplify, srepr
from sympy.physics.wigner import wigner_3j, wigner_6j

from .common import atomic_write_json, read_json, sha256_file
from .fusion import (
    cg_coefficient, convention_hash, convention_payload, coupling_outputs, magnetic_values,
)


def _entry_key(kind: str, values: tuple[int, ...]) -> str:
    return kind + ':' + ','.join(str(value) for value in values)


def _exact_payload(value: Expr) -> dict[str, str]:
    exact = simplify(value)
    return {'str': str(exact), 'srepr': srepr(exact)}


@dataclass(slots=True)
class WignerCache:
    entries: dict[str, dict[str, str]] = field(default_factory=dict)

    def cg(
        self, left_j2: int, left_m2: int, right_j2: int, right_m2: int,
        total_j2: int, total_m2: int,
    ) -> Expr:
        values = (left_j2, left_m2, right_j2, right_m2, total_j2, total_m2)
        value = cg_coefficient(*values)
        self.entries[_entry_key('cg', values)] = _exact_payload(value)
        return value

    def three_j(self, values: tuple[int, int, int, int, int, int]) -> Expr:
        j1, j2, j3, m1, m2, m3 = values
        value = simplify(wigner_3j(
            Rational(j1, 2), Rational(j2, 2), Rational(j3, 2),
            Rational(m1, 2), Rational(m2, 2), Rational(m3, 2),
        ))
        self.entries[_entry_key('3j', values)] = _exact_payload(value)
        return value

    def six_j(self, values: tuple[int, int, int, int, int, int]) -> Expr:
        value = simplify(wigner_6j(*(Rational(item, 2) for item in values)))
        self.entries[_entry_key('6j', values)] = _exact_payload(value)
        return value

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'library': 'sympy exact',
            'convention': convention_payload(),
            'convention_hash': convention_hash(),
            'entry_count': len(self.entries),
            'entries': {key: self.entries[key] for key in sorted(self.entries)},
        }

    def save(self, path: Path) -> str:
        atomic_write_json(path, self.payload())
        return sha256_file(path)


def generate_low_cutoff_cache(path: Path, j2_max: int = 1, leg_count: int = 6) -> str:
    if (
        not isinstance(j2_max, int) or isinstance(j2_max, bool) or j2_max < 0
        or leg_count != 6
    ):
        raise ValueError('M2 cache generation requires nonnegative j2_max and six legs.')
    cache = WignerCache()
    reachable = set(range(j2_max + 1))
    for _ in range(leg_count - 1):
        next_reachable: set[int] = set()
        for left_j2 in sorted(reachable):
            for right_j2 in range(j2_max + 1):
                for total_j2 in coupling_outputs(left_j2, right_j2):
                    next_reachable.add(total_j2)
                    for left_m2 in magnetic_values(left_j2):
                        for right_m2 in magnetic_values(right_j2):
                            total_m2 = left_m2 + right_m2
                            if total_m2 in magnetic_values(total_j2):
                                cache.cg(
                                    left_j2, left_m2, right_j2, right_m2,
                                    total_j2, total_m2,
                                )
        reachable = next_reachable
    if j2_max >= 1:
        cache.three_j((1, 1, 0, 1, -1, 0))
        cache.three_j((1, 1, 0, -1, 1, 0))
        cache.six_j((1, 1, 0, 1, 1, 0))
    return cache.save(path)


def validate_cache(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict) or payload.get('schema_version') != 1:
        raise ValueError('Malformed Wigner cache.')
    if payload.get('convention_hash') != convention_hash():
        raise ValueError('Wigner cache convention hash mismatch.')
    entries = payload.get('entries')
    if not isinstance(entries, dict) or payload.get('entry_count') != len(entries) or not entries:
        raise ValueError('Wigner cache entry set is invalid.')
    for key, value in entries.items():
        if not isinstance(key, str) or not isinstance(value, dict):
            raise ValueError('Wigner cache entry is malformed.')
        if set(value) != {'str', 'srepr'} or not all(isinstance(item, str) for item in value.values()):
            raise ValueError('Wigner cache exact expression is malformed.')
    return payload
