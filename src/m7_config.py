"""M7 search configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .m7_status import M6_PARENT_RUN_ID_FROZEN, M7_RUN_ID_FROZEN


@dataclass(frozen=True, slots=True)
class M7Config:
    parent_m6_run_id: str = M6_PARENT_RUN_ID_FROZEN
    run_id: str = M7_RUN_ID_FROZEN
    mode: str = 'paperspace'
    # paperspace | cpu_fixture_cert | cpu_fixture_search
    max_candidates_total: int = 64
    max_rigorous_replays: int = 16
    stop_on_first_certified: bool = True
    required_q_cert_upper: str = '1'
    # Operational margin target (still require independent q_cert_upper < 1).
    promotion_estimated_q: str = '9/10'
    campaign: str = 'A'
    num_steps: int = 3

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def default_m7_config(**overrides: Any) -> M7Config:
    base = M7Config()
    if not overrides:
        return base
    payload = asdict(base)
    payload.update(overrides)
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
