"""M7 search configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .m7_status import (
    M6_PARENT_RUN_ID_FROZEN,
    M7_RUN_ID_CAMPAIGN_B,
    M7_RUN_ID_FROZEN,
)


@dataclass(frozen=True, slots=True)
class M7Config:
    parent_m6_run_id: str = M6_PARENT_RUN_ID_FROZEN
    run_id: str = M7_RUN_ID_FROZEN
    mode: str = 'paperspace'
    # paperspace | cpu_fixture_cert | cpu_fixture_search | cpu_fixture_campaign_b
    max_candidates_total: int = 64
    max_rigorous_replays: int = 16
    stop_on_first_certified: bool = True
    required_q_cert_upper: str = '1'
    # Operational margin target (still require independent q_cert_upper < 1).
    promotion_estimated_q: str = '9/10'
    campaign: str = 'A'
    num_steps: int = 3
    # Campaign B lineage: plan_only | fixture_residual | execute
    lineage_mode: str = 'plan_only'
    max_lineage_replays: int = 2
    parent_rank: int = 16

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def default_m7_config(**overrides: Any) -> M7Config:
    base = M7Config()
    if not overrides:
        return base
    payload = asdict(base)
    payload.update(overrides)
    if payload.get('campaign') == 'B' and 'run_id' not in overrides:
        payload['run_id'] = M7_RUN_ID_CAMPAIGN_B
    if payload.get('campaign') == 'B' and 'lineage_mode' not in overrides:
        if str(payload.get('mode', '')).startswith('cpu_fixture'):
            payload['lineage_mode'] = 'fixture_residual'
        else:
            payload['lineage_mode'] = 'plan_only'
    return M7Config(**payload)


def campaign_a_search_space() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'campaign': 'A',
        'layers': {
            'majorant_policy': [
                'DIRECT_MULTI_STEP_PRODUCT',
                'STAGE_DEPENDENT_WEIGHTED_PRODUCT',
            ],
            'perron_weight_strategy': [
                'all_ones',
                'interval_power',
                'collatz_lp_heuristic',
                'inverse_row_sum',
            ],
            'source_partition': [
                'current',
                'symmetry_blocks',
            ],
            'input_subdivision': [1, 2, 4],
            'coupling_policy': [
                'uniform_full',
                'diagonal_plus_l1_tail',
            ],
        },
        'excluded_until_improvement': ['S2', 'S3', 'S4'],
    }


def campaign_b_search_space() -> dict[str, Any]:
    """Finite-approximation layer (S2): rank / RSVD quality / residual tightening."""
    return {
        'schema_version': 1,
        'campaign': 'B',
        'layers': {
            # Design listed 24/32/48; M4 regroup needs perfect squares, so those
            # requests are lifted to 25/36/49 via effective_projected_rank.
            'target_rank': [16, 24, 32, 48, 64],
            'oversampling': [8, 16, 24],
            'power_iterations': [1, 2, 3],
            'perron_weight_strategy': [
                'all_ones',
                'interval_power',
                'inverse_row_sum',
            ],
            'coupling_policy': [
                'uniform_full',
            ],
        },
        'excluded_until_improvement': ['S3', 'S4'],
        'geometry_note': (
            'M4 dual_regroup requires projected_rank to be a perfect square. '
            'Non-square target_rank values are mapped upward before lineage.'
        ),
    }


def search_space_for_campaign(campaign: str) -> dict[str, Any]:
    if campaign == 'B':
        return campaign_b_search_space()
    return campaign_a_search_space()
