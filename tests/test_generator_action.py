from __future__ import annotations

from sympy import eye, simplify, zeros

from src.armillary import SectorKey, build_armillary_sector, fixed_link_orientations
from src.fusion import fusion_basis_matrix
from src.generator_action import (
    apply_total_j_to_basis, exact_generator_annihilation, matrix_exact_zero,
)


def test_generator_annihilation_on_half_spin_singlet_basis() -> None:
    reps = (1, 1, 1, 1, 1, 1)
    paths, basis = fusion_basis_matrix(reps)
    assert len(paths) == 5
    assert exact_generator_annihilation(basis, reps)
    jz_b, jp_b, jm_b = apply_total_j_to_basis(basis, reps)
    assert matrix_exact_zero(jz_b)
    assert matrix_exact_zero(jp_b)
    assert matrix_exact_zero(jm_b)


def test_empty_basis_is_annihilated() -> None:
    reps = (1, 0, 0, 0, 0, 0)
    paths, basis = fusion_basis_matrix(reps)
    assert len(paths) == 0
    assert basis.shape[1] == 0
    assert exact_generator_annihilation(basis, reps)


def test_armillary_sector_records_generator_residual() -> None:
    key = SectorKey((1, 1, 1, 1, 1, 1), fixed_link_orientations())
    sector = build_armillary_sector(key)
    assert sector.generator_residual_exact
    assert sector.isometry_exact
    assert (sector.basis_map.T * sector.basis_map).applyfunc(simplify) == eye(5)
    assert sector.basis_map.shape == (64, 5)


def test_identity_operator_on_non_singlet_fails_annihilation() -> None:
    # A single computational-basis vector is generally not a singlet.
    reps = (1, 1)
    fake = zeros(4, 1)
    fake[0, 0] = 1
    assert not exact_generator_annihilation(fake, reps)
