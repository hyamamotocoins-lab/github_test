"""Exploratory partial error budget for Campaign C rank selection.

Never writes certificate-usable bounds. Values are provisional proxies only.
"""

from __future__ import annotations

from typing import Any, Mapping

from .common import sha256_bytes, canonical_json_bytes


class PartialErrorBudgetError(RuntimeError):
    """Raised when a provisional budget cannot be assembled safely."""


def provisional_budget(
    *,
    rank: int,
    effective_projected_rank: int,
    relative_residual: float,
    approximate_gap: float | None,
    influence_proxy_value: float,
    engineering_margin: float,
    representation_tail_proxy: float = 0.0,
    channel_tail_proxy: float = 0.0,
    rounding_proxy: float = 0.0,
    input_radius_proxy: float = 0.0,
    normalization_proxy: float = 0.0,
    basis_variation_proxy: float | None = None,
) -> dict[str, Any]:
    if engineering_margin < 0.0:
        raise PartialErrorBudgetError('engineering_margin must be nonnegative.')
    if relative_residual < 0.0 or influence_proxy_value < 0.0:
        raise PartialErrorBudgetError('Residual/proxy values must be nonnegative.')

    # Heuristic: basis variation blows up when gap is tiny.
    if basis_variation_proxy is None:
        if approximate_gap is None or approximate_gap <= 0.0:
            basis_variation_proxy = float('inf')
        else:
            basis_variation_proxy = relative_residual / max(approximate_gap, 1e-300)

    terms = {
        'core': {
            'proxy': float(influence_proxy_value),
            'provenance': 'M3_influence_proxy_sigma1',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'rounding': {
            'proxy': float(rounding_proxy),
            'provenance': 'placeholder_zero_unless_configured',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'input_radius': {
            'proxy': float(input_radius_proxy),
            'provenance': 'placeholder_zero_unless_configured',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'normalization': {
            'proxy': float(normalization_proxy),
            'provenance': 'placeholder_zero_unless_configured',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'rsvd': {
            'proxy': float(relative_residual),
            'provenance': 'nested_rsvd_relative_frobenius',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'basis_variation': {
            'proxy': float(basis_variation_proxy),
            'provenance': 'residual_over_approximate_gap',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'representation_tail': {
            'proxy': float(representation_tail_proxy),
            'provenance': 'configured_proxy_or_zero',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
        'channel_tail': {
            'proxy': float(channel_tail_proxy),
            'provenance': 'configured_proxy_or_zero',
            'rigor': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
    }

    finite_terms = []
    for name, payload in terms.items():
        value = float(payload['proxy'])
        if value == float('inf'):
            q_prov = float('inf')
            break
        finite_terms.append(value)
    else:
        q_prov = float(sum(finite_terms))

    # Optimistic majorant zeros positive unknown tails that are not yet modeled.
    q_optimistic = float(
        terms['core']['proxy'] + terms['rsvd']['proxy']
    )
    if terms['basis_variation']['proxy'] != float('inf'):
        q_optimistic += float(terms['basis_variation']['proxy'])

    certificate_usable = False
    payload = {
        'schema_version': 1,
        'status': 'EXPLORATORY',
        'rank': int(rank),
        'effective_projected_rank': int(effective_projected_rank),
        'q_optimistic': q_optimistic,
        'q_provisional': q_prov,
        'engineering_margin': float(engineering_margin),
        'passes_optimistic_gate': bool(q_optimistic < 1.0),
        'passes_provisional_gate': bool(
            q_prov < 1.0 - float(engineering_margin)
        ),
        'terms': terms,
        'certificate_usable': certificate_usable,
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }
    payload['budget_hash'] = sha256_bytes(canonical_json_bytes(payload))
    return payload


def select_rank_from_budgets(
    rows: list[Mapping[str, Any]],
    *,
    require_cluster_terminus: bool = True,
) -> dict[str, Any]:
    """Apply design §8 selection rules to exploratory budget rows."""
    feasible: list[dict[str, Any]] = []
    reasons_reject: list[str] = []
    for row in rows:
        rank = int(row['rank'])
        budget = row.get('budget') or {}
        if require_cluster_terminus and not row.get('is_cluster_terminus'):
            reasons_reject.append(f'rank={rank}: mid-cluster')
            continue
        gap = row.get('approximate_gap')
        if gap is None or float(gap) <= 0.0:
            reasons_reject.append(f'rank={rank}: nonpositive approximate gap')
            continue
        if not budget.get('passes_optimistic_gate'):
            reasons_reject.append(f'rank={rank}: q_optimistic>=1')
            continue
        if not budget.get('passes_provisional_gate'):
            reasons_reject.append(f'rank={rank}: q_prov fails engineering margin')
            continue
        if row.get('resource_ok') is False:
            reasons_reject.append(f'rank={rank}: resource gate failed')
            continue
        feasible.append(dict(row))

    if not feasible:
        return {
            'selection_status': 'NO_SELECTION',
            'selected_rank': None,
            'selection_reasons': reasons_reject or ['no feasible ranks'],
            'feasible_ranks': [],
        }

    # Prefer larger approximate gap, then larger headroom, then smaller rank.
    def key(row: Mapping[str, Any]) -> tuple[float, float, int]:
        budget = row.get('budget') or {}
        headroom = 1.0 - float(budget.get('q_provisional', 1.0))
        return (-float(row.get('approximate_gap') or 0.0), -headroom, int(row['rank']))

    feasible.sort(key=key)
    chosen = feasible[0]
    return {
        'selection_status': 'SELECTED',
        'selected_rank': int(chosen['rank']),
        'effective_projected_rank': int(chosen.get('effective_projected_rank') or chosen['rank']),
        'selection_reasons': [
            'cluster terminus',
            'positive approximate gap',
            'q_optimistic<1',
            'q_prov within engineering margin',
            f"preferred gap/headroom/rank among {len(feasible)} feasible",
        ],
        'feasible_ranks': [int(row['rank']) for row in feasible],
        'chosen_row': chosen,
        'rejected': reasons_reject,
    }
