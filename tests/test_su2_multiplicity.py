from __future__ import annotations

from src.armillary import all_link_star_keys
from src.dense_reference import build_dense_reference
from src.su2_multiplicity import MULTIPLICITY_METHOD, singlet_multiplicity


def test_singlet_multiplicity_matches_dense_rank_at_j2_max_1() -> None:
    for key in all_link_star_keys(1):
        mu = singlet_multiplicity(key.representations)
        dense = build_dense_reference(key.representations, key.orientations)
        assert mu == dense.singlet_rank


def test_max_sector_and_odd_sectors() -> None:
    assert singlet_multiplicity((2, 2, 2, 2, 2, 2)) == 15
    assert singlet_multiplicity((1, 0, 0, 0, 0, 0)) == 0
    assert MULTIPLICITY_METHOD == 'weight_count_w0_minus_w2_v1'


def test_j2_max_2_odd_and_even_zero_counts() -> None:
    odd_zero = 0
    even_zero = 0
    for labels in __import__('itertools').product(range(3), repeat=6):
        mu = singlet_multiplicity(labels)
        if sum(labels) % 2:
            assert mu == 0
            odd_zero += 1
        elif mu == 0:
            even_zero += 1
    assert odd_zero == 364
    assert even_zero == 6
