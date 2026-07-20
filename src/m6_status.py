"""Machine-readable M6 status vocabulary."""

from __future__ import annotations

from typing import Final

from .m5_status import M5_RUN_ID_FROZEN, NOT_CERTIFIED, ONE_STEP_CERTIFIED

M6_RUN_ID_FROZEN: Final = 'M6-20260720T061700Z-7c4e91a2b850'
M5_PARENT_RUN_ID_FROZEN: Final = M5_RUN_ID_FROZEN

M6_IMPLEMENTATION_IN_PROGRESS: Final = 'IMPLEMENTATION_IN_PROGRESS'
M6_BLOCKED_IMPLEMENTATION: Final = 'BLOCKED_IMPLEMENTATION'
M6_BLOCKED_MATH: Final = 'BLOCKED_MATH'
M6_VERIFICATION_FAILED: Final = 'VERIFICATION_FAILED'
M6_COMPLETE: Final = 'M6_COMPLETE'

STEP_ENCLOSED: Final = 'STEP_ENCLOSED'
CERTIFIED: Final = 'CERTIFIED'

# Re-export parent certification vocabulary used at the M5→M6 gate.
ALLOWED_M5_CERTIFICATION: Final[frozenset[str]] = frozenset({
    NOT_CERTIFIED,
    ONE_STEP_CERTIFIED,
})

ERROR_LEDGER_IDS: Final[tuple[str, ...]] = tuple(f'E{index}' for index in range(1, 13))
