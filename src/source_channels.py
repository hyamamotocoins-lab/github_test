from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Mapping

import numpy as np

if TYPE_CHECKING:
    from .forward_ad import DualTensor


class SourceClass(str, Enum):
    TEMPORAL_LINK = 'temporal_link'
    SPATIAL_LINK = 'spatial_link'
    ELECTRIC_LIKE = 'electric_like'
    MAGNETIC_LIKE = 'magnetic_like'
    LOW_REPRESENTATION = 'low_representation'


SOURCE_CLASSES: tuple[SourceClass, ...] = tuple(SourceClass)


def source_generators() -> dict[SourceClass, np.ndarray]:
    coordinates = np.indices((3,) * 6, dtype=np.int8).reshape(6, -1).T
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
    for source, generator in generators.items():
        if generator.shape != (729,) or not np.isfinite(generator).all():
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
    for source in SOURCE_CLASSES:
        value = np.asarray(generators[source], dtype=np.float64).reshape((3,) * 6)
        for label, axes in (
            ('temporal_swap', temporal_swap),
            ('spatial_cycle', spatial_cycle),
        ):
            residuals[f'{source.value}:{label}'] = float(
                np.max(np.abs(value - value.transpose(axes)))
            )
    return residuals


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
    if (
        left.shape != (729, 16)
        or core.shape != (16, 16)
        or right.shape != (16, 729)
        or basis.shape != (729, 16)
    ):
        raise ValueError('M4 requires the accepted M3 rank-16 Triad shapes.')
    if not np.isfinite(source_scale):
        raise ValueError('M4 source scale must be finite.')
    left_projected = basis.T @ left
    right_projected = right @ basis
    primal = left_projected @ core @ right_projected
    tangent: dict[SourceClass, np.ndarray] = {}
    for source in SOURCE_CLASSES:
        generator = np.asarray(generators[source], dtype=np.float64)
        if generator.shape != (729,):
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
    generator = np.asarray(generator, dtype=np.float64)
    if generator.shape != (729,) or not np.isfinite(parameter):
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
