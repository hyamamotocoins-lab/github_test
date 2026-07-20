"""M7 search configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .m7_status import (
    M6_PARENT_RUN_ID_FROZEN,
    M7_RUN_ID_CAMPAIGN_B,
    M7_RUN_ID_CAMPAIGN_C,
    M7_RUN_ID_FROZEN,
)


@dataclass(frozen=True, slots=True)
class M7Config:
    parent_m6_run_id: str = M6_PARENT_RUN_ID_FROZEN
    run_id: str = M7_RUN_ID_FROZEN
    mode: str = 'paperspace'
    # paperspace | cpu_fixture_cert | cpu_fixture_search
    # | cpu_fixture_campaign_b | cpu_fixture_campaign_c
    max_candidates_total: int = 64
    max_rigorous_replays: int = 16
    stop_on_first_certified: bool = True
    required_q_cert_upper: str = '1'
    # Operational margin target (still require independent q_cert_upper < 1).
    promotion_estimated_q: str = '9/10'
    campaign: str = 'A'
    num_steps: int = 3
    # Campaign B/C lineage: plan_only | fixture_residual | execute | auto
    lineage_mode: str = 'plan_only'
    max_lineage_replays: int = 2
    parent_rank: int = 16
    parent_j2_max: int = 1
    # Design: Campaign C requires human review before lineage execution.
    human_review_approved: bool = False
    # Auto pipeline: after plan_only, materialize+dry_run best candidate.
    auto_approve_for_materialize: bool = False
    max_executable_j2_max: int = 2

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def default_m7_config(**overrides: Any) -> M7Config:
    base = M7Config()
    if not overrides:
        return base
    payload = asdict(base)
    payload.update(overrides)
    campaign = payload.get('campaign')
    if campaign == 'B' and 'run_id' not in overrides:
        payload['run_id'] = M7_RUN_ID_CAMPAIGN_B
    if campaign == 'C' and 'run_id' not in overrides:
        payload['run_id'] = M7_RUN_ID_CAMPAIGN_C
    if campaign in {'B', 'C'} and 'lineage_mode' not in overrides:
        if str(payload.get('mode', '')).startswith('cpu_fixture'):
            payload['lineage_mode'] = 'fixture_residual'
        else:
            payload['lineage_mode'] = 'plan_only'
    if campaign == 'C' and payload.get('lineage_mode') == 'auto':
        # Auto implies materialize path; still respects human_review unless
        # auto_approve_for_materialize is set.
        pass
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


def campaign_c_search_space() -> dict[str, Any]:
    """Algebraic / geometry layer (S3): cutoff, channels, block geometry."""
    return {
        'schema_version': 1,
        'campaign': 'C',
        'requires_human_review': True,
        'layers': {
            'j2_max': [1, 2, 3, 4],
            'channel_policy': [
                'complete_at_cutoff',
                'certified_pruned',
            ],
            'block_geometry': [
                'current',
                'approved_geometry_B',
            ],
            'perron_weight_strategy': [
                'all_ones',
                'interval_power',
            ],
            'coupling_policy': [
                'uniform_full',
            ],
        },
        'excluded_until_improvement': ['S4'],
        'math_locks': {
            'resource_gate': (
                'Configs accept j2_max in [1,4] with derived dims. '
                'Live exact-M2 auto-execute remains gated to j2_max=1; '
                'higher cutoffs materialize+dry_run and need staged M2 acceptance.'
            ),
        },
        'notes': (
            'Campaign C invalidates M2–M6. Use lineage_mode=auto to '
            'auto-select best plan, stamp/await review, materialize package, '
            'and dry-run config construction.'
        ),
    }


def search_space_for_campaign(campaign: str) -> dict[str, Any]:
    if campaign == 'B':
        return campaign_b_search_space()
    if campaign == 'C':
        return campaign_c_search_space()
    return campaign_a_search_space()
