from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Mapping

import numpy as np

from .cutoff_dims import leg_hilbert_sum, operator_dimension as cutoff_operator_dimension

if TYPE_CHECKING:
    from .forward_ad import DualTensor


class SourceClass(str, Enum):
    TEMPORAL_LINK = 'temporal_link'
    SPATIAL_LINK = 'spatial_link'
    ELECTRIC_LIKE = 'electric_like'
    MAGNETIC_LIKE = 'magnetic_like'
    LOW_REPRESENTATION = 'low_representation'


SOURCE_CLASSES: tuple[SourceClass, ...] = tuple(SourceClass)


def j2_max_from_operator_dimension(dimension: int) -> int:
    if not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError('operator_dimension must be a positive integer.')
    for j2_max in range(0, 8):
        if cutoff_operator_dimension(j2_max) == dimension:
            return j2_max
    raise ValueError(
        f'operator_dimension={dimension} is not a supported cutoff L^6.'
    )


def leg_dimension_from_operator_dimension(dimension: int) -> int:
    return leg_hilbert_sum(j2_max_from_operator_dimension(dimension))


def source_generators(
    *,
    j2_max: int | None = None,
    operator_dimension: int | None = None,
) -> dict[SourceClass, np.ndarray]:
    if j2_max is None and operator_dimension is None:
        j2_max = 1
    elif j2_max is None:
        assert operator_dimension is not None
        j2_max = j2_max_from_operator_dimension(int(operator_dimension))
    elif operator_dimension is not None:
        expected = cutoff_operator_dimension(int(j2_max))
        if int(operator_dimension) != expected:
            raise ValueError(
                'j2_max/operator_dimension disagree for M4 source generators.'
            )
    if not isinstance(j2_max, int) or isinstance(j2_max, bool) or j2_max < 0:
        raise ValueError('j2_max must be a nonnegative integer.')
    leg = leg_hilbert_sum(j2_max)
    coordinates = np.indices((leg,) * 6, dtype=np.int16).reshape(6, -1).T
    occupied = (coordinates > 0).astype(np.float64)
    temporal = occupied[:, :2]
    spatial = occupied[:, 2:]
    spatial_pairs = sum(
        spatial[:, first] * spatial[:, second]
        for first in range(4)
        for second in range(first + 1, 4)
    )
    generators = {
        SourceClass.TEMPORAL_LINK: temporal.mean(axis=1),
        SourceClass.SPATIAL_LINK: spatial.mean(axis=1),
        SourceClass.ELECTRIC_LIKE: (temporal[:, 0] - temporal[:, 1]) ** 2,
        SourceClass.MAGNETIC_LIKE: spatial_pairs / 6.0,
        SourceClass.LOW_REPRESENTATION: (
            occupied.sum(axis=1) == 1
        ).astype(np.float64),
    }
    expected = cutoff_operator_dimension(j2_max)
    for source, generator in generators.items():
        if generator.shape != (expected,) or not np.isfinite(generator).all():
            raise ValueError(f'Invalid M4 source generator: {source.value}')
        if np.any(generator < 0.0) or not np.any(generator > 0.0):
            raise ValueError(f'Degenerate M4 source generator: {source.value}')
    return generators


def generator_symmetry_residuals(
    generators: Mapping[SourceClass, np.ndarray],
) -> dict[str, float]:
    temporal_swap = (1, 0, 2, 3, 4, 5)
    spatial_cycle = (0, 1, 3, 4, 5, 2)
    residuals: dict[str, float] = {}
    probe = np.asarray(next(iter(generators.values())), dtype=np.float64)
    leg = leg_dimension_from_operator_dimension(int(probe.size))
    for source in SOURCE_CLASSES:
        value = np.asarray(generators[source], dtype=np.float64).reshape((leg,) * 6)
        for label, axes in (
            ('temporal_swap', temporal_swap),
            ('spatial_cycle', spatial_cycle),
        ):
            residuals[f'{source.value}:{label}'] = float(
                np.max(np.abs(value - value.transpose(axes)))
            )
    return residuals


def _validate_triad_shapes(
    left: np.ndarray,
    core: np.ndarray,
    right: np.ndarray,
    basis: np.ndarray,
) -> tuple[int, int]:
    if left.ndim != 2 or core.ndim != 2 or right.ndim != 2 or basis.ndim != 2:
        raise ValueError('M4 Triad factors must be rank-2 arrays.')
    dim, rank = left.shape
    if (
        core.shape != (rank, rank)
        or right.shape != (rank, dim)
        or basis.shape != (dim, rank)
    ):
        raise ValueError(
            'M4 Triad/basis shapes must be (D,R), (R,R), (R,D), (D,R); '
            f'got left={left.shape}, core={core.shape}, right={right.shape}, '
            f'basis={basis.shape}.'
        )
    if dim < 1 or rank < 1:
        raise ValueError('M4 Triad dimensions must be positive.')
    # projected_rank must remain a perfect square for regroup_matrix.
    leg = int(round(rank ** 0.5))
    if leg * leg != rank:
        raise ValueError(
            f'M4 projected rank {rank} must be a perfect square for regrouping.'
        )
    return dim, rank


def projected_parent_dual(
    left: np.ndarray,
    core: np.ndarray,
    right: np.ndarray,
    basis: np.ndarray,
    generators: Mapping[SourceClass, np.ndarray],
    *,
    source_scale: float = 1.0,
) -> DualTensor:
    from .forward_ad import DualTensor

    left = np.asarray(left, dtype=np.float64)
    core = np.asarray(core, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    dim, _rank = _validate_triad_shapes(left, core, right, basis)
    if not np.isfinite(source_scale):
        raise ValueError('M4 source scale must be finite.')
    left_projected = basis.T @ left
    right_projected = right @ basis
    primal = left_projected @ core @ right_projected
    tangent: dict[SourceClass, np.ndarray] = {}
    for source in SOURCE_CLASSES:
        generator = np.asarray(generators[source], dtype=np.float64)
        if generator.shape != (dim,):
            raise ValueError(f'M4 generator shape changed: {source.value}')
        left_derivative = basis.T @ (generator[:, None] * left)
        right_derivative = (right * generator[None, :]) @ basis
        tangent[source] = source_scale * (
            left_derivative @ core @ right_projected
            + left_projected @ core @ right_derivative
        )
    return DualTensor(primal, tangent)


def deformed_projected_parent(
    left: np.ndarray,
    core: np.ndarray,
    right: np.ndarray,
    basis: np.ndarray,
    generator: np.ndarray,
    parameter: float,
) -> np.ndarray:
    arrays = tuple(
        np.asarray(value, dtype=np.float64)
        for value in (left, core, right, basis)
    )
    left_value, core_value, right_value, basis_value = arrays
    dim, _rank = _validate_triad_shapes(
        left_value, core_value, right_value, basis_value,
    )
    generator = np.asarray(generator, dtype=np.float64)
    if generator.shape != (dim,) or not np.isfinite(parameter):
        raise ValueError('Invalid M4 finite-difference source deformation.')
    weight = np.exp(parameter * generator)
    if not np.isfinite(weight).all():
        raise FloatingPointError('M4 source exponential is nonfinite.')
    return (
        basis_value.T
        @ (weight[:, None] * left_value)
        @ core_value
        @ (right_value * weight[None, :])
        @ basis_value
    )
