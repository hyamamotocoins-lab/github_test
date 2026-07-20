"""Armillary CG fusion basis with invariant-subspace uniqueness certificates."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np
from sympy import Matrix, eye, simplify, zeros

from .dense_reference import matrix_hash, matrix_to_float64
from .fusion import (
    fusion_basis_matrix, orientation_map, representation_dimension,
)
from .generator_action import exact_generator_annihilation
from .su2_multiplicity import singlet_multiplicity


@dataclass(frozen=True, order=True, slots=True)
class SectorKey:
    representations: tuple[int, ...]
    orientations: tuple[int, ...]
    fusion_tree: str = 'left-associated'

    def __post_init__(self) -> None:
        if len(self.representations) != 6 or len(self.orientations) != 6:
            raise ValueError('M2 SectorKey must describe a six-leg 4D link star.')
        if any(
            not isinstance(j2, int) or isinstance(j2, bool) or j2 < 0
            for j2 in self.representations
        ):
            raise ValueError('Sector representations must be nonnegative integers.')
        if any(sign not in {-1, 1} for sign in self.orientations):
            raise ValueError('Sector orientations must be +1 or -1.')
        if self.fusion_tree != 'left-associated':
            raise ValueError('M2 silently changed the fixed fusion tree.')


@dataclass(frozen=True, slots=True)
class ArmillarySector:
    key: SectorKey
    fusion_paths: tuple[tuple[int, ...], ...]
    basis_map: Matrix
    armillary_tensor: Matrix
    isometry_exact: bool
    generator_residual_exact: bool
    independent_singlet_multiplicity: int

    @property
    def singlet_rank(self) -> int:
        return len(self.fusion_paths)


def build_armillary_sector(key: SectorKey) -> ArmillarySector:
    paths, outgoing_basis = fusion_basis_matrix(key.representations)
    mu = singlet_multiplicity(key.representations)
    rank = len(paths)
    if rank != mu:
        raise ArithmeticError(
            f'Armillary column count {rank} != independent multiplicity {mu} '
            f'for reps={list(key.representations)}.',
        )
    # Generator annihilation in the outgoing magnetic basis is equivalent to
    # physical-basis annihilation because the orientation/duality map is orthogonal.
    residual_exact = exact_generator_annihilation(outgoing_basis, key.representations)
    if not residual_exact:
        raise ArithmeticError(
            f'Armillary basis fails exact generator annihilation for reps={list(key.representations)}.',
        )
    dual_map = orientation_map(key.representations, key.orientations)
    if rank:
        basis_map = (dual_map.T * outgoing_basis).applyfunc(simplify)
    else:
        dimension = representation_dimension(key.representations)
        basis_map = zeros(dimension, 0)
    armillary_tensor = eye(rank)
    isometry = (basis_map.T * basis_map).applyfunc(simplify) == eye(rank)
    if not isometry:
        raise ArithmeticError('Armillary basis map is not exactly isometric.')
    return ArmillarySector(
        key, paths, basis_map, armillary_tensor, isometry, residual_exact, mu,
    )


def fixed_link_orientations() -> tuple[int, ...]:
    return (1, -1, 1, -1, 1, -1)


def all_link_star_keys(j2_max: int = 1) -> tuple[SectorKey, ...]:
    if not isinstance(j2_max, int) or isinstance(j2_max, bool) or j2_max < 0:
        raise ValueError('j2_max must be a nonnegative integer.')
    orientations = fixed_link_orientations()
    return tuple(
        SectorKey(tuple(labels), orientations)
        for labels in product(range(j2_max + 1), repeat=6)
    )


def sector_summary(sector: ArmillarySector) -> dict[str, object]:
    return {
        'reps': list(sector.key.representations),
        'orientations': list(sector.key.orientations),
        'fusion_tree': sector.key.fusion_tree,
        'fusion_paths': [list(path) for path in sector.fusion_paths],
        'dense_dimension': representation_dimension(sector.key.representations),
        'singlet_rank': sector.singlet_rank,
        'independent_singlet_multiplicity': sector.independent_singlet_multiplicity,
        'basis_map_hash': matrix_hash(sector.basis_map),
        'isometry_exact': sector.isometry_exact,
        'generator_residual_exact': sector.generator_residual_exact,
    }


def checkpoint_tensor_shards(
    sectors: Iterable[ArmillarySector],
) -> dict[str, np.ndarray]:
    """Diagnostic float64 Haar projectors P ≈ B Bᵀ (not a certification bound)."""
    tensors: dict[str, np.ndarray] = {}
    for sector in sectors:
        label = ''.join(str(value) for value in sector.key.representations)
        if sector.singlet_rank == 0:
            dim = representation_dimension(sector.key.representations)
            tensors[f'projector_{label}'] = np.zeros((dim, dim), dtype=np.float64)
            continue
        basis = matrix_to_float64(sector.basis_map)
        tensors[f'projector_{label}'] = basis @ basis.T
    return tensors
