"""Shared helpers for Campaign B tests."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def write_tiny_config(tmp: Path, *, hard_limit_sec: float = 30.0) -> Path:
    space = tmp / 'space.yaml'
    space.write_text(
        """
schema_version: 1
campaign: B_S2
rank:
  values: [64]
rsvd:
  oversampling: [24]
  power_iterations: [3]
  seeds: [20260720]
residual:
  tolerances: [0.0]
  norm_models: [frobenius]
staging:
  j2_values: [2]
  forbid_j2_1: true
layers:
  perron_weight_strategy: [all_ones]
  coupling_policy: [uniform_full]
""".strip(),
        encoding='utf-8',
    )
    cfg = tmp / 'campaign.yaml'
    cfg.write_text(
        f"""
schema_version: 1
campaign: B_S2
persistent_root: {tmp / 'persist'}
search_space_path: space.yaml
time_budget_sec: {hard_limit_sec}
time_budget:
  hard_limit_sec: {hard_limit_sec}
  admission_close_sec: {max(1.0, hard_limit_sec * 0.9)}
  finalization_start_sec: {max(1.0, hard_limit_sec * 0.95)}
  emergency_flush_sec: 1
  enforce_wall_clock: false
inherit_deadline: false
screening_margin: 0.000001
parent_q_upper: 1.011045
parent_rank: 16
shared_m2:
  allow_generate_canonical: false
  on_missing: continue_archive
never_stop: true
stop_after_first_verified_q_lt_1: false
execution_policy:
  staged_only: true
  minimum_j2: 2
  allow_campaign_c: false
  allow_production_m6: false
  certification_status: NOT_CERTIFIED
  claim_scope: SCREENING_ONLY
parent_evidence:
  campaign_c_status: CAMPAIGN_C_S3_EXHAUSTED_GOTO_B
  campaign_c_best_q: 1.011
source_tree_roots:
  - {(REPO / 'src').as_posix()}
""".strip(),
        encoding='utf-8',
    )
    return cfg
