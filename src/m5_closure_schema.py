"""Schema / status vocabulary for Campaign C M5 obligation-closure program."""

from __future__ import annotations

from typing import Final

CLOSURE_PROGRAM_VERSION: Final = 1

EXPLORATORY: Final = 'EXPLORATORY_NOT_CERTIFIED'
RIGOROUS: Final = 'RIGOROUS'
OPEN: Final = 'OPEN'
FAILED: Final = 'FAILED'

SELECTION_NO_SELECTION: Final = 'NO_SELECTION'
SELECTION_SELECTED: Final = 'SELECTED'
SELECTION_REJECT: Final = 'REJECT_CANDIDATE'

M5_OBLIGATIONS_OPEN: Final = 'M5_PROOF_OBLIGATIONS_OPEN'
M5_OBLIGATIONS_CLOSED: Final = 'M5_PROOF_OBLIGATIONS_CLOSED'
M5_MAJORANT_NONCONTRACTIVE: Final = 'M5_MAJORANT_NONCONTRACTIVE'
ONE_STEP_CERTIFIED: Final = 'ONE_STEP_CERTIFIED'
NOT_CERTIFIED: Final = 'NOT_CERTIFIED'

OPEN_OBLIGATION_IDS: Final[tuple[str, ...]] = (
    'M3_RSVD_PROJECTION_RESIDUAL',
    'BASIS_VARIATION_RESIDUAL',
    'INITIAL_REPRESENTATION_TAIL',
    'OMITTED_FUSION_AND_CHANNEL_TAIL',
)

NOTEBOOKS: Final[dict[str, str]] = {
    'S0': '77_rank_gap_budget_sweep.ipynb',
    'S2': '78_m3_rigorous_rank_candidate.ipynb',
    'S3_S4': '79_m4_basis_variation_and_tails.ipynb',
    'S5_S6': '80_m5_close_and_verify.ipynb',
    'M6': '81_m6_production_gate.ipynb',
    'QUEUE': '82_campaign_c_candidate_queue.ipynb',
    'S0_SERIES': '83_s0_rank_sweep_series.ipynb',
    'SHARED_M2': '84_shared_m2_registry.ipynb',
}
