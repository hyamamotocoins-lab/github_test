from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .linear_operator import ArmillaryLinearOperator


@dataclass(frozen=True, slots=True)
class RSVDResult:
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray
    seed: int
    target_rank: int
    oversampling: int
    power_iterations: int
    elapsed_s: float
    orthogonality_residual: float
    residual_frobenius: float
    relative_residual_frobenius: float

    def tensors(self) -> dict[str, np.ndarray]:
        return {
            'rsvd_left': self.left,
            'rsvd_singular_values': self.singular_values,
            'rsvd_right_t': self.right_t,
        }

    def summary(self) -> dict[str, Any]:
        return {
            'seed': self.seed, 'target_rank': self.target_rank,
            'oversampling': self.oversampling,
            'power_iterations': self.power_iterations,
            'elapsed_s': self.elapsed_s,
            'singular_values': self.singular_values.tolist(),
            'orthogonality_residual': self.orthogonality_residual,
            'residual_frobenius': self.residual_frobenius,
            'relative_residual_frobenius': self.relative_residual_frobenius,
            'basis_sha256': array_sha256(self.left),
            'singular_values_sha256': array_sha256(self.singular_values),
            'right_sha256': array_sha256(self.right_t),
            'rigor': 'EXPLORATORY_FIXED_SEED_NOT_A_CERTIFICATE',
        }


def array_sha256(value: np.ndarray) -> str:
    array = np.asarray(value, dtype='<f8', order='C')
    payload = (
        str(array.shape).encode('ascii')
        + b'\0' + array.tobytes(order='C')
    )
    return hashlib.sha256(payload).hexdigest()


def randomized_svd(
    operator: ArmillaryLinearOperator, *, target_rank: int,
    oversampling: int, power_iterations: int, seed: int,
) -> RSVDResult:
    n_rows, n_cols = operator.shape
    if not 1 <= target_rank < min(n_rows, n_cols):
        raise ValueError('RSVD target rank is invalid.')
    if oversampling < 1 or power_iterations < 0:
        raise ValueError('RSVD oversampling or power iteration count is invalid.')
    internal_rank = min(n_cols, target_rank + oversampling)
    generator = torch.Generator(device=operator.backend.device)
    generator.manual_seed(seed)
    started = time.monotonic()
    omega = torch.randn(
        (n_cols, internal_rank), generator=generator,
        device=operator.backend.device, dtype=operator.backend.dtype,
    )
    sample = operator.matmat_tensor(omega)
    for _ in range(power_iterations):
        left_q, _ = torch.linalg.qr(sample, mode='reduced')
        adjoint_sample = operator.rmatmat_tensor(left_q)
        right_q, _ = torch.linalg.qr(adjoint_sample, mode='reduced')
        sample = operator.matmat_tensor(right_q)
    q, _ = torch.linalg.qr(sample, mode='reduced')
    projected = operator.rmatmat_tensor(q).T
    u_hat, singular_values, right_t = torch.linalg.svd(
        projected, full_matrices=False,
    )
    left = q @ u_hat[:, :target_rank]
    singular_values = singular_values[:target_rank]
    right_t = right_t[:target_rank]
    operator.backend.synchronize()
    elapsed = time.monotonic() - started
    left_np = operator.backend.to_numpy(left)
    singular_np = operator.backend.to_numpy(singular_values)
    right_np = operator.backend.to_numpy(right_t)
    if not (
        np.isfinite(left_np).all()
        and np.isfinite(singular_np).all()
        and np.isfinite(right_np).all()
        and np.all(singular_np >= 0.0)
    ):
        raise ArithmeticError('RSVD produced nonfinite or negative singular data.')
    orthogonality = float(np.linalg.norm(
        left_np.T @ left_np - np.eye(target_rank), ord='fro',
    ))
    residual = operator.factor_residual_frobenius(
        left_np, singular_np, right_np,
    )
    norm = operator.frobenius_norm()
    if not np.isfinite(residual) or not norm > 0.0:
        raise ArithmeticError('RSVD residual or operator norm is invalid.')
    return RSVDResult(
        left_np, singular_np, right_np, seed, target_rank,
        oversampling, power_iterations, elapsed, orthogonality,
        residual, residual / norm,
    )


def influence_proxy(result: RSVDResult) -> dict[str, Any]:
    if result.singular_values.size == 0:
        raise ValueError('Influence proxy requires singular values.')
    proxy = float(result.singular_values[0])
    if not np.isfinite(proxy) or proxy < 0.0:
        raise ArithmeticError('Influence proxy is nonfinite or negative.')
    if proxy > 1.2:
        screening = 'EARLY_TERMINATE_CURRENT_SCHEME'
    elif proxy < 0.8:
        screening = 'PRIORITIZE_M4_M5'
    else:
        screening = 'INVESTIGATE_CUTOFF_AND_RANK'
    return {
        'value': proxy, 'screening': screening,
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }
