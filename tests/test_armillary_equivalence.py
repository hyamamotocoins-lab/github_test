from __future__ import annotations

from src.armillary import (
    SectorKey, all_link_star_keys, build_armillary_sector, fixed_link_orientations,
)
from src.dense_reference import build_dense_reference, exact_matrix_difference_zero
from src.sector_canonicalization import (
    action_table_hash, apply_action, canonicalize_sector, sector_orbit,
    transverse_cubic_actions,
)


def test_all_64_low_cutoff_sectors_match_independent_dense_reference_exactly() -> None:
    keys = all_link_star_keys()
    assert len(keys) == 64
    ranks: dict[tuple[int, ...], int] = {}
    for key in keys:
        armillary = build_armillary_sector(key)
        dense = build_dense_reference(key.representations, key.orientations)
        assert armillary.isometry_exact
        assert dense.generator_residual_zero
        assert armillary.singlet_rank == dense.singlet_rank
        assert exact_matrix_difference_zero(armillary.reconstructed_dense, dense.projector)
        ranks[key.representations] = dense.singlet_rank
    assert ranks[(0, 0, 0, 0, 0, 0)] == 1
    assert ranks[(1, 1, 1, 1, 1, 1)] == 5
    assert all(
        rank == 0
        for representations, rank in ranks.items()
        if sum(representations) % 2
    )


def test_gauge_noninvariant_odd_half_spin_sectors_vanish() -> None:
    orientations = fixed_link_orientations()
    for index in range(6):
        reps = tuple(1 if leg == index else 0 for leg in range(6))
        key = SectorKey(reps, orientations)
        armillary = build_armillary_sector(key)
        dense = build_dense_reference(reps, orientations)
        assert armillary.singlet_rank == dense.singlet_rank == 0
        assert not any(armillary.reconstructed_dense)
        assert not any(dense.projector)


def test_transverse_cubic_symmetry_is_deterministic_and_complete() -> None:
    actions = transverse_cubic_actions()
    assert len(actions) == len(set(actions)) == 48
    assert len(action_table_hash()) == 64
    reps = (1, 0, 1, 1, 0, 0)
    signs = fixed_link_orientations()
    canonical = canonicalize_sector(reps, signs)
    assert canonicalize_sector(*canonical) == canonical
    orbit = sector_orbit(reps, signs)
    assert canonical == min(orbit)
    assert all(
        (apply_action(reps, action), apply_action(signs, action)) in orbit
        for action in actions
    )


def test_canonicalization_reduces_the_fixed_orientation_sector_set() -> None:
    keys = all_link_star_keys()
    canonical = {
        canonicalize_sector(key.representations, key.orientations)
        for key in keys
    }
    assert 1 < len(canonical) < len(keys)
