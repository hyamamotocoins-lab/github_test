from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .source_channels import SOURCE_CLASSES, SourceClass


def _finite_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f'{label} must be a finite rank-2 FP64 array.')
    return array


@dataclass(frozen=True, slots=True)
class DualTensor:
    primal: np.ndarray
    tangent: Mapping[SourceClass, np.ndarray]

    def __post_init__(self) -> None:
        primal = _finite_array(self.primal, 'DualTensor primal')
        if set(self.tangent) != set(SOURCE_CLASSES):
            raise ValueError('DualTensor must contain every symmetry-reduced channel.')
        normalized: dict[SourceClass, np.ndarray] = {}
        for source in SOURCE_CLASSES:
            tangent = _finite_array(
                self.tangent[source], f'DualTensor tangent {source.value}',
            )
            if tangent.shape != primal.shape:
                raise ValueError('DualTensor primal/tangent shapes differ.')
            normalized[source] = tangent
        object.__setattr__(self, 'primal', primal)
        object.__setattr__(self, 'tangent', normalized)

    @property
    def shape(self) -> tuple[int, int]:
        return self.primal.shape

    def tensor_payload(self, prefix: str) -> dict[str, np.ndarray]:
        result = {f'{prefix}_primal': self.primal.copy()}
        result.update({
            f'{prefix}_tangent_{source.value}': self.tangent[source].copy()
            for source in SOURCE_CLASSES
        })
        return result


def zero_source_dual(primal: np.ndarray) -> DualTensor:
    primal = _finite_array(primal, 'zero-source primal')
    return DualTensor(
        primal,
        {source: np.zeros_like(primal) for source in SOURCE_CLASSES},
    )


def dual_matmul(left: DualTensor, right: DualTensor) -> DualTensor:
    if left.shape[1] != right.shape[0]:
        raise ValueError('DualTensor contraction dimensions do not match.')
    return DualTensor(
        left.primal @ right.primal,
        {
            source: (
                left.tangent[source] @ right.primal
                + left.primal @ right.tangent[source]
            )
            for source in SOURCE_CLASSES
        },
    )


def fixed_basis_project(
    value: DualTensor, left_basis: np.ndarray, right_basis: np.ndarray,
) -> DualTensor:
    left_basis = _finite_array(left_basis, 'left projection basis')
    right_basis = _finite_array(right_basis, 'right projection basis')
    if (
        left_basis.shape[0] != value.shape[0]
        or right_basis.shape[0] != value.shape[1]
    ):
        raise ValueError('Fixed projection basis is incompatible with DualTensor.')
    return DualTensor(
        left_basis.T @ value.primal @ right_basis,
        {
            source: left_basis.T @ value.tangent[source] @ right_basis
            for source in SOURCE_CLASSES
        },
    )


def regroup_matrix(value: np.ndarray) -> np.ndarray:
    value = _finite_array(value, 'regroup input')
    if value.shape[0] != value.shape[1]:
        raise ValueError('M4 regrouping requires a square matrix.')
    leg = int(round(np.sqrt(value.shape[0])))
    if leg * leg != value.shape[0]:
        raise ValueError('M4 regrouping dimension must be a perfect square.')
    return value.reshape(leg, leg, leg, leg).transpose(0, 2, 1, 3).reshape(
        value.shape
    )


def dual_regroup(value: DualTensor) -> DualTensor:
    return DualTensor(
        regroup_matrix(value.primal),
        {
            source: regroup_matrix(value.tangent[source])
            for source in SOURCE_CLASSES
        },
    )
