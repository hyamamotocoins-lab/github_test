"""Derived finite-cutoff dimensions for SU(2) six-leg link star."""

from __future__ import annotations

from typing import Any


def leg_hilbert_sum(j2_max: int) -> int:
    """Σ_{j2=0}^{J} (j2+1) = (J+1)(J+2)/2."""
    if j2_max < 0:
        raise ValueError('j2_max must be nonnegative.')
    return (j2_max + 1) * (j2_max + 2) // 2


def sector_count(j2_max: int) -> int:
    """Number of six-leg representation sectors: (j2_max+1)^6."""
    if j2_max < 0:
        raise ValueError('j2_max must be nonnegative.')
    return (j2_max + 1) ** 6


def operator_dimension(j2_max: int) -> int:
    """Dense operator dimension: [Σ (j2+1)]^6."""
    return leg_hilbert_sum(j2_max) ** 6


def cutoff_dimension_payload(j2_max: int) -> dict[str, Any]:
    return {
        'j2_max': int(j2_max),
        'sector_count': sector_count(j2_max),
        'leg_hilbert_sum': leg_hilbert_sum(j2_max),
        'operator_dimension': operator_dimension(j2_max),
        'dense_fp64_bytes_estimate': operator_dimension(j2_max) ** 2 * 8,
    }


def resource_gate(
    j2_max: int,
    *,
    max_executable_j2_max: int = 2,
    max_dense_fp64_gb: float = 32.0,
) -> dict[str, Any]:
    """Decide whether a live lineage execute is resource-safe."""
    payload = cutoff_dimension_payload(j2_max)
    dense_gb = payload['dense_fp64_bytes_estimate'] / (1024 ** 3)
    blocked_reasons: list[str] = []
    if j2_max > max_executable_j2_max:
        blocked_reasons.append(
            f'j2_max={j2_max} exceeds max_executable_j2_max={max_executable_j2_max}'
        )
    if dense_gb > max_dense_fp64_gb:
        blocked_reasons.append(
            f'dense FP64 estimate {dense_gb:.2f} GiB exceeds budget '
            f'{max_dense_fp64_gb:.2f} GiB (matrix-free still required)'
        )
    # Exact M2 SymPy over many sectors is not auto-executable beyond j2=1.
    if j2_max > 1:
        blocked_reasons.append(
            'exact M2 SymPy armillary auto-execute is limited to j2_max=1; '
            'higher cutoffs require staged/manual M2 acceptance first'
        )
    return {
        **payload,
        'dense_fp64_gib_estimate': dense_gb,
        'executable': not blocked_reasons,
        'blocked_reasons': blocked_reasons,
        'automation_policy': (
            'materialize+dry_run always; live execute only when executable'
        ),
    }
