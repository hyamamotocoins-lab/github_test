"""M7 status vocabulary for certified scheme search."""

from __future__ import annotations

from typing import Final

from .m6_status import M6_RUN_ID_FROZEN

M7_RUN_ID_FROZEN: Final = 'M7-20260720T074400Z-9f2a18c6d401'
M7_RUN_ID_CAMPAIGN_B: Final = 'M7-20260720T080000Z-b7c4e91a2b85'
M7_RUN_ID_CAMPAIGN_C: Final = 'M7-20260720T081500Z-c8d5f02b3c96'
M6_PARENT_RUN_ID_FROZEN: Final = M6_RUN_ID_FROZEN

# Paperspace Campaign C may continue under q<1 hunt run ids minted by m7_q_lt1_hunt.
_M7C_QLT1_TAG: Final = 'qlt1c'


def is_allowed_campaign_c_run_id(run_id: str) -> bool:
    """Accept the frozen Campaign C id or minted q<1-hunt continuation ids."""
    if run_id == M7_RUN_ID_CAMPAIGN_C:
        return True
    # M7-YYYYMMDDTHHMMSSZ-qlt1c... (see mint_m7c_qlt1_run_id)
    if not isinstance(run_id, str) or not run_id.startswith('M7-'):
        return False
    parts = run_id.split('-')
    if len(parts) < 3:
        return False
    stamp, tag = parts[1], parts[2]
    if len(stamp) != 16 or stamp[8] != 'T' or not stamp.endswith('Z'):
        return False
    return tag.startswith(_M7C_QLT1_TAG)

M7_INITIALIZED: Final = 'M7_INITIALIZED'
M7_DIAGNOSIS_COMPLETE: Final = 'M7_DIAGNOSIS_COMPLETE'
M7_SEARCHING: Final = 'M7_SEARCHING'
M7_CANDIDATE_RUNNING: Final = 'M7_CANDIDATE_RUNNING'
M7_CERTIFIED_SCHEME_FOUND: Final = 'M7_CERTIFIED_SCHEME_FOUND'
M7_SEARCH_SPACE_EXHAUSTED: Final = 'M7_SEARCH_SPACE_EXHAUSTED'
M7_RESOURCE_LIMIT_REACHED: Final = 'M7_RESOURCE_LIMIT_REACHED'
M7_LINEAGE_PLANNED: Final = 'M7_LINEAGE_PLANNED'
M7_BLOCKED_MATH: Final = 'M7_BLOCKED_MATH'
M7_BLOCKED_POLICY: Final = 'M7_BLOCKED_POLICY'
M7_HUMAN_REVIEW_REQUIRED: Final = 'M7_HUMAN_REVIEW_REQUIRED'
M7_COMPLETE: Final = 'M7_COMPLETE'

CERTIFIED_SCHEME_FOUND: Final = 'CERTIFIED_SCHEME_FOUND'
SCHEME_REJECTED: Final = 'SCHEME_REJECTED'

CHANGE_S0: Final = 'S0'
CHANGE_S1: Final = 'S1'
CHANGE_S2: Final = 'S2'
CHANGE_S3: Final = 'S3'
CHANGE_S4: Final = 'S4'

DIAG_D0: Final = 'D0_inherited_majorant_failure'
DIAG_D1: Final = 'D1_core_expansion'
DIAG_D2: Final = 'D2_truncation_dominated'
DIAG_D3: Final = 'D3_interval_dependency'
DIAG_D4: Final = 'D4_normalization'
DIAG_D5: Final = 'D5_arithmetic'
DIAG_D6: Final = 'D6_unresolved_analytic_leaf'

POLICY_PARENT_INHERIT: Final = 'PARENT_ONE_STEP_INHERITANCE'
POLICY_DIRECT_PRODUCT: Final = 'DIRECT_MULTI_STEP_PRODUCT'
POLICY_STAGE_WEIGHTED: Final = 'STAGE_DEPENDENT_WEIGHTED_PRODUCT'
POLICY_RECENTERED: Final = 'RECENTERED_CELLWISE_PRODUCT'
