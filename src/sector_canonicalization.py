from __future__ import annotations

import hashlib
from itertools import permutations, product
from typing import Iterable

from .common import canonical_json_bytes


def transverse_cubic_actions() -> tuple[tuple[int, ...], ...]:
    actions: set[tuple[int, ...]] = set()
    for axis_permutation in permutations(range(3)):
        for flips in product((False, True), repeat=3):
            order: list[int] = []
            for new_axis, old_axis in enumerate(axis_permutation):
                pair = [2 * old_axis, 2 * old_axis + 1]
                if flips[new_axis]:
                    pair.reverse()
                order.extend(pair)
            actions.add(tuple(order))
    result = tuple(sorted(actions))
    if len(result) != 48:
        raise ArithmeticError('Transverse cubic action must have exactly 48 elements.')
    return result


def apply_action(values: Iterable[int], action: tuple[int, ...]) -> tuple[int, ...]:
    items = tuple(values)
    if len(items) != 6 or sorted(action) != list(range(6)):
        raise ValueError('Cubic action must permute six link-star legs.')
    return tuple(items[index] for index in action)


def canonicalize_sector(
    representations: Iterable[int], orientations: Iterable[int],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    reps = tuple(representations)
    signs = tuple(orientations)
    if len(reps) != 6 or len(signs) != 6:
        raise ValueError('Sector canonicalization expects six legs.')
    orbit = (
        (apply_action(reps, action), apply_action(signs, action))
        for action in transverse_cubic_actions()
    )
    return min(orbit)


def sector_orbit(
    representations: Iterable[int], orientations: Iterable[int],
) -> tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]:
    reps = tuple(representations)
    signs = tuple(orientations)
    return tuple(sorted({
        (apply_action(reps, action), apply_action(signs, action))
        for action in transverse_cubic_actions()
    }))


def action_table_hash() -> str:
    return hashlib.sha256(canonical_json_bytes(transverse_cubic_actions())).hexdigest()
