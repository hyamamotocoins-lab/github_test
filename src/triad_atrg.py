from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .linear_operator import ArmillaryLinearOperator
from .rsvd import RSVDResult, array_sha256


@dataclass(frozen=True, slots=True)
class TriadFactorization:
    left: np.ndarray
    core: np.ndarray
    right: np.ndarray
    residual_frobenius: float
    relative_residual_frobenius: float

    @property
    def rank(self) -> int:
        return int(self.core.shape[0])

    def apply(self, value: np.ndarray) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        return self.left @ (self.core @ (self.right @ vector))

    def reconstruct(self) -> np.ndarray:
        return self.left @ self.core @ self.right

    def tensors(self) -> dict[str, np.ndarray]:
        return {
            'triad_left': self.left,
            'triad_core': self.core,
            'triad_right': self.right,
        }

    def summary(self) -> dict[str, Any]:
        return {
            'rank': self.rank,
            'factor_shapes': {
                'left': list(self.left.shape),
                'core': list(self.core.shape),
                'right': list(self.right.shape),
            },
            'intermediate_ranks': [self.rank, self.rank],
            'factor_bytes': int(
                self.left.nbytes + self.core.nbytes + self.right.nbytes
            ),
            'residual_frobenius': self.residual_frobenius,
            'relative_residual_frobenius': self.relative_residual_frobenius,
            'left_sha256': array_sha256(self.left),
            'core_sha256': array_sha256(self.core),
            'right_sha256': array_sha256(self.right),
            'rigor': 'EXPLORATORY_TRIAD_FACTORIZATION_NOT_A_CERTIFICATE',
        }


def triad_from_rsvd(
    operator: ArmillaryLinearOperator, result: RSVDResult,
) -> TriadFactorization:
    left = np.asarray(result.left, dtype=np.float64, order='C')
    core = np.diag(np.asarray(result.singular_values, dtype=np.float64))
    right = np.asarray(result.right_t, dtype=np.float64, order='C')
    if (
        left.shape[0] != operator.dimension
        or right.shape[1] != operator.dimension
        or left.shape[1] != core.shape[0]
        or core.shape[1] != right.shape[0]
    ):
        raise ValueError('RSVD factors do not define an M3 triad.')
    residual = operator.factor_residual_frobenius(
        left, np.diag(core), right,
    )
    norm = operator.frobenius_norm()
    if not np.isfinite(residual) or not norm > 0.0:
        raise ArithmeticError('Triad residual is invalid.')
    return TriadFactorization(
        left.copy(), core.copy(), right.copy(), residual, residual / norm,
    )
