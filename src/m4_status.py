from __future__ import annotations

from typing import Final


M4_IMPLEMENTATION_COMPLETE: Final = 'M4_IMPLEMENTATION_COMPLETE'
M4_DERIVATIVE_ACCEPTED: Final = 'DERIVATIVE_ACCEPTED'
M4_IMPLEMENTATION_IN_PROGRESS: Final = 'IMPLEMENTATION_IN_PROGRESS'
M4_ENCLOSURE_BLOCKED: Final = 'BLOCKED_MATH'
MIN_CENTERED_FD_ACCEPTANCE_ORDER: Final = 1.8

M4_CLOSED_OBLIGATIONS: Final[tuple[str, ...]] = (
    'forward tangent construction',
    'tangent checkpoint restore',
    'zero tangent consistency',
    'symmetry consistency',
    'finite-difference regression',
)

M5_OPEN_PROOF_OBLIGATIONS: Final[tuple[str, ...]] = (
    'GPU rounding and backward error',
    'M3 RSVD projection residual',
    'cutoff and rank dependence',
    'initial representation tail',
    'input radius propagation',
    'normalization and denominator error',
    'omitted fusion and channel tail',
    'basis variation residual',
)


def m4_bound_handoff() -> dict[str, list[str]]:
    """Return a fresh machine-readable M4-to-M5 proof-obligation ledger."""
    return {
        'closed_in_M4': list(M4_CLOSED_OBLIGATIONS),
        'open_for_M5': list(M5_OPEN_PROOF_OBLIGATIONS),
    }


def m4_milestone_status(phase: str) -> str:
    """Separate derivative acceptance from the still-blocked enclosure status."""
    return (
        M4_DERIVATIVE_ACCEPTED
        if phase == 'M4_COMPLETE'
        else M4_IMPLEMENTATION_IN_PROGRESS
    )
