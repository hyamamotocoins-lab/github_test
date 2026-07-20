"""Approximate singular-value cluster detection for exploratory rank sweeps.

All outputs are HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class SpectralClusterError(RuntimeError):
    """Raised when approximate cluster parsing cannot proceed."""


def approximate_gaps(singular_values: np.ndarray) -> np.ndarray:
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    if values.size < 2:
        raise SpectralClusterError('Need at least two singular values for gaps.')
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise SpectralClusterError('Singular values must be finite and nonnegative.')
    return values[:-1] - values[1:]


def detect_approximate_clusters(
    singular_values: np.ndarray,
    *,
    relative_gap_threshold: float = 0.05,
    absolute_gap_floor: float = 1e-12,
) -> list[dict[str, Any]]:
    """Partition singular values into approximate contiguous clusters.

    A boundary after index i (1-based rank i) is declared when
    gap_i / max(sigma_i, eps) >= relative_gap_threshold and gap_i >= floor.
    """
    values = np.asarray(singular_values, dtype=np.float64).reshape(-1)
    gaps = approximate_gaps(values)
    boundaries: list[int] = []
    for index, gap in enumerate(gaps, start=1):
        scale = max(float(values[index - 1]), absolute_gap_floor)
        if float(gap) >= absolute_gap_floor and float(gap) / scale >= relative_gap_threshold:
            boundaries.append(index)
    boundaries.append(int(values.size))
    clusters: list[dict[str, Any]] = []
    start = 1
    for end in boundaries:
        segment = values[start - 1:end]
        gap_after = (
            float(gaps[end - 1]) if end < values.size else None
        )
        clusters.append({
            'cluster_start': start,
            'cluster_end': end,
            'size': end - start + 1,
            'sigma_max': float(segment[0]),
            'sigma_min': float(segment[-1]),
            'gap_after': gap_after,
            'is_cluster_terminus': True,
            'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        })
        start = end + 1
    return clusters


def cluster_terminus_ranks(clusters: list[dict[str, Any]]) -> list[int]:
    return [int(row['cluster_end']) for row in clusters if row.get('is_cluster_terminus')]


def rank_is_mid_cluster(rank: int, clusters: list[dict[str, Any]]) -> bool:
    for row in clusters:
        start = int(row['cluster_start'])
        end = int(row['cluster_end'])
        if start <= rank < end:
            return True
        if rank == end:
            return False
    return False
