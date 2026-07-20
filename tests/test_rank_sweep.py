from __future__ import annotations

import numpy as np

from src.partial_error_budget import provisional_budget, select_rank_from_budgets
from src.spectral_cluster import (
    detect_approximate_clusters, rank_is_mid_cluster,
)


def test_cluster_terminus_and_midpoints() -> None:
    # Two clear clusters: 1.0,0.99 then gap, then 0.2,0.19
    values = np.array([1.0, 0.99, 0.2, 0.19, 0.05], dtype=np.float64)
    clusters = detect_approximate_clusters(
        values, relative_gap_threshold=0.2, absolute_gap_floor=1e-9,
    )
    ends = [row['cluster_end'] for row in clusters]
    assert 2 in ends
    assert rank_is_mid_cluster(1, clusters) is True
    assert rank_is_mid_cluster(2, clusters) is False


def test_provisional_budget_gates() -> None:
    good = provisional_budget(
        rank=16,
        effective_projected_rank=16,
        relative_residual=0.01,
        approximate_gap=0.2,
        influence_proxy_value=0.5,
        engineering_margin=0.05,
    )
    assert good['passes_optimistic_gate'] is True
    assert good['passes_provisional_gate'] is True
    assert good['certificate_usable'] is False

    bad = provisional_budget(
        rank=16,
        effective_projected_rank=16,
        relative_residual=0.5,
        approximate_gap=0.01,
        influence_proxy_value=0.99,
        engineering_margin=0.05,
    )
    assert bad['passes_provisional_gate'] is False


def test_select_rank_prefers_gap_and_headroom() -> None:
    rows = [
        {
            'rank': 16,
            'effective_projected_rank': 16,
            'approximate_gap': 0.01,
            'is_cluster_terminus': True,
            'resource_ok': True,
            'budget': provisional_budget(
                rank=16, effective_projected_rank=16,
                relative_residual=0.02, approximate_gap=0.01,
                influence_proxy_value=0.4, engineering_margin=0.05,
            ),
        },
        {
            'rank': 36,
            'effective_projected_rank': 36,
            'approximate_gap': 0.2,
            'is_cluster_terminus': True,
            'resource_ok': True,
            'budget': provisional_budget(
                rank=36, effective_projected_rank=36,
                relative_residual=0.01, approximate_gap=0.2,
                influence_proxy_value=0.4, engineering_margin=0.05,
            ),
        },
        {
            'rank': 25,
            'effective_projected_rank': 25,
            'approximate_gap': 0.3,
            'is_cluster_terminus': False,
            'resource_ok': True,
            'budget': provisional_budget(
                rank=25, effective_projected_rank=25,
                relative_residual=0.005, approximate_gap=0.3,
                influence_proxy_value=0.4, engineering_margin=0.05,
            ),
        },
    ]
    selection = select_rank_from_budgets(rows)
    assert selection['selection_status'] == 'SELECTED'
    assert selection['selected_rank'] == 36
