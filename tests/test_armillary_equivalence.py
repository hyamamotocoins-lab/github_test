from __future__ import annotations

from src.armillary import (
    SectorKey, all_link_star_keys, build_armillary_sector, fixed_link_orientations,
)
from src.sector_canonicalization import (
    action_table_hash, apply_action, canonicalize_sector, sector_orbit,
    transverse_cubic_actions,
)
from src.su2_multiplicity import singlet_multiplicity


def test_all_64_sectors_have_invariant_subspace_uniqueness() -> None:
    keys = all_link_star_keys()
    assert len(keys) == 64
    ranks: dict[tuple[int, ...], int] = {}
    for key in keys:
        armillary = build_armillary_sector(key)
        mu = singlet_multiplicity(key.representations)
        assert armillary.isometry_exact
        assert armillary.generator_residual_exact
        assert armillary.singlet_rank == mu == armillary.independent_singlet_multiplicity
        ranks[key.representations] = mu
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
        assert armillary.singlet_rank == singlet_multiplicity(reps) == 0
        assert armillary.basis_map.cols == 0


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
