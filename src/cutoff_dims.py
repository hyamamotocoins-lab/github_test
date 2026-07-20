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


def odd_sum_sector_count(j2_max: int) -> int:
    """Number of six-leg sectors whose representation sum is odd."""
    from itertools import product
    if j2_max < 0:
        raise ValueError('j2_max must be nonnegative.')
    return sum(
        1
        for labels in product(range(j2_max + 1), repeat=6)
        if sum(labels) % 2 == 1
    )


def expected_m2_gate_counts(j2_max: int) -> dict[str, int]:
    n = sector_count(j2_max)
    odd = odd_sum_sector_count(j2_max)
    return {
        'sector_count': n,
        'generator_residual_zero_count': n,
        'odd_half_zero_count': odd,
        'isometry_exact_count': n,
        'exact_match_count': n,
    }


def resource_gate(
    j2_max: int,
    *,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    max_dense_fp64_gb: float = 32.0,
) -> dict[str, Any]:
    """Decide whether a live lineage execute is resource-safe."""
    payload = cutoff_dimension_payload(j2_max)
    dense_gb = payload['dense_fp64_bytes_estimate'] / (1024 ** 3)
    blocked_reasons: list[str] = []
    staged_blocked: list[str] = []
    if j2_max > max_executable_j2_max:
        blocked_reasons.append(
            f'j2_max={j2_max} exceeds max_executable_j2_max={max_executable_j2_max}'
        )
    if dense_gb > max_dense_fp64_gb:
        blocked_reasons.append(
            f'dense FP64 estimate {dense_gb:.2f} GiB exceeds budget '
            f'{max_dense_fp64_gb:.2f} GiB (matrix-free still required)'
        )
    # Instant auto-execute (single session, no sector batching): j2_max=1 only.
    instant_ok = j2_max == 1 and not blocked_reasons
    if j2_max > 1:
        blocked_reasons.append(
            'instant exact-M2 auto-execute is limited to j2_max=1; '
            'use staged sector-batched M2 for j2_max>=2'
        )
    if j2_max > max_staged_j2_max:
        staged_blocked.append(
            f'j2_max={j2_max} exceeds max_staged_j2_max={max_staged_j2_max}'
        )
    return {
        **payload,
        'dense_fp64_gib_estimate': dense_gb,
        'executable': instant_ok,  # j2=1 instant path
        'blocked_reasons': blocked_reasons if not instant_ok else [],
        'staged_executable': j2_max >= 2 and not staged_blocked,
        'staged_blocked_reasons': staged_blocked,
        'default_sector_batch_size': 16 if j2_max >= 2 else 0,
        'automation_policy': (
            'j2=1: instant live; j2=2: staged sector-batched M2→M6; '
            'j2>max_staged: archive/screening only'
        ),
    }
