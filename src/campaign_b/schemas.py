"""Campaign B status / terminal reason constants and light validation."""

from __future__ import annotations

from typing import Any, Final

from .errors import InvariantViolation

CERTIFICATION_STATUS: Final = 'NOT_CERTIFIED'
CLAIM_SCOPE: Final = 'SCREENING_ONLY'

TERMINAL_Q_LT_1: Final = 'B_Q_LT_1_LINEAGE_READY'
TERMINAL_EXHAUSTED: Final = 'B_SCREENING_EXHAUSTED'
TERMINAL_TIME: Final = 'B_TIME_BUDGET_EXHAUSTED'
TERMINAL_NEED_M2: Final = 'B_BLOCKED_NEED_CANONICAL_M2'
TERMINAL_FAIL: Final = 'B_FAIL_CLOSED'

CAMPAIGN_STATES: Final = frozenset({
    'CREATED',
    'PREFLIGHT',
    'RUNNING',
    'ADMISSION_CLOSED',
    'FINALIZING',
    'COMPLETE',
    'BLOCKED_NEED_CANONICAL_M2',
    'FAIL_CLOSED',
    'TIME_BUDGET_EXHAUSTED',
})

CANDIDATE_STATES: Final = frozenset({
    'PENDING',
    'RESERVED',
    'SCREENING',
    'SCREENED_Q_GE_1',
    'SCREENED_Q_LT_1',
    'BORDERLINE_Q',
    'M2_RESOLVE',
    'NEED_CANONICAL_M2',
    'WAITING_FOR_M2',  # parallel-split design; reconciler TODO in driver
    'READY_SHARED',
    'S0',
    'INDEPENDENT_VERIFY',
    'VERIFY_REJECTED',
    'PACKAGE_AUDIT',
    'AUDIT_REJECTED',
    'SELECTED',
    'ARCHIVED',
})

ALLOWED_PHASES: Final = frozenset({
    'B_QUEUE',
    'B_SCREEN',
    'M2_BIND',
    'S0',
    'INDEPENDENT_SCREEN_VERIFY',
    'PACKAGE_AUDIT',
    'ARCHIVE',
    'SELECTED',
    'FINALIZE',
})

FORBIDDEN_PHASE_TOKENS: Final = frozenset({
    'M6',
    'PRODUCTION_M6',
    'M6_COMPLETE',
    'MASS_GAP',
    'CERTIFIED',
})

ARCHIVE_REASONS: Final = frozenset({
    'Q_GE_1',
    'BORDERLINE_Q',
    'INDEPENDENT_VERIFY_MISMATCH',
    'DUPLICATE_NORMALIZED_SCHEME',
    'INSUFFICIENT_TIME_BUDGET',
    'M2_NOT_AVAILABLE',
    'NUMERICAL_INSTABILITY',
})


def screening_only_payload() -> dict[str, str]:
    return {
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    }


def assert_not_certified(payload: dict[str, Any], *, context: str) -> None:
    status = payload.get('certification_status')
    if status is None:
        raise InvariantViolation(f'{context}: missing certification_status')
    if status != CERTIFICATION_STATUS:
        raise InvariantViolation(
            f'{context}: certification_status must be {CERTIFICATION_STATUS}, '
            f'got {status!r}'
        )
    if 'CERTIFIED' in str(payload.get('status', '')).upper() and status != CERTIFICATION_STATUS:
        raise InvariantViolation(f'{context}: forbidden CERTIFIED status')
    claim = payload.get('claim_scope')
    if claim is not None and claim != CLAIM_SCOPE:
        raise InvariantViolation(
            f'{context}: claim_scope must be {CLAIM_SCOPE}, got {claim!r}'
        )


def assert_staged_candidate(candidate: dict[str, Any]) -> None:
    j2 = int(candidate.get('j2') or candidate.get('j2_max') or 0)
    mode = str(candidate.get('execution_mode') or '')
    if j2 < 2:
        raise InvariantViolation(f'staged-only violated: j2={j2}')
    if mode != 'staged':
        raise InvariantViolation(
            f'staged-only violated: execution_mode={mode!r}'
        )


def assert_phase_allowed(phase: str) -> None:
    token = phase.upper().replace('-', '_')
    for forbidden in FORBIDDEN_PHASE_TOKENS:
        if forbidden in token and phase not in ALLOWED_PHASES:
            # Allow S0 etc.; block anything with M6 in the name.
            if 'M6' in token:
                raise InvariantViolation(f'forbidden phase: {phase}')
    if 'M6' in token:
        raise InvariantViolation(f'forbidden production phase: {phase}')
