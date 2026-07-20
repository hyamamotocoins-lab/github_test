"""Campaign / candidate state machine transitions."""

from __future__ import annotations

from typing import Any

from .errors import CampaignFatalError
from .schemas import CAMPAIGN_STATES, CANDIDATE_STATES


CAMPAIGN_TRANSITIONS: dict[str, frozenset[str]] = {
    'CREATED': frozenset({'PREFLIGHT', 'FAIL_CLOSED'}),
    'PREFLIGHT': frozenset({'RUNNING', 'FAIL_CLOSED'}),
    'RUNNING': frozenset({
        'ADMISSION_CLOSED',
        'FINALIZING',
        'COMPLETE',
        'BLOCKED_NEED_CANONICAL_M2',
        'FAIL_CLOSED',
        'TIME_BUDGET_EXHAUSTED',
    }),
    'ADMISSION_CLOSED': frozenset({
        'FINALIZING',
        'COMPLETE',
        'FAIL_CLOSED',
        'TIME_BUDGET_EXHAUSTED',
    }),
    'FINALIZING': frozenset({
        'COMPLETE',
        'FAIL_CLOSED',
        'TIME_BUDGET_EXHAUSTED',
        'BLOCKED_NEED_CANONICAL_M2',
    }),
    'COMPLETE': frozenset(),
    'BLOCKED_NEED_CANONICAL_M2': frozenset(),
    'FAIL_CLOSED': frozenset(),
    'TIME_BUDGET_EXHAUSTED': frozenset(),
}


CANDIDATE_TRANSITIONS: dict[str, frozenset[str]] = {
    'PENDING': frozenset({'RESERVED', 'ARCHIVED'}),
    'RESERVED': frozenset({'SCREENING', 'PENDING', 'ARCHIVED'}),
    'SCREENING': frozenset({
        'SCREENED_Q_GE_1',
        'SCREENED_Q_LT_1',
        'BORDERLINE_Q',
        'ARCHIVED',
    }),
    'SCREENED_Q_GE_1': frozenset({'ARCHIVED'}),
    'BORDERLINE_Q': frozenset({'ARCHIVED'}),
    'SCREENED_Q_LT_1': frozenset({'M2_RESOLVE', 'ARCHIVED'}),
    'M2_RESOLVE': frozenset({'READY_SHARED', 'NEED_CANONICAL_M2', 'ARCHIVED'}),
    'NEED_CANONICAL_M2': frozenset({'ARCHIVED'}),
    'READY_SHARED': frozenset({'S0', 'ARCHIVED'}),
    'S0': frozenset({'INDEPENDENT_VERIFY', 'ARCHIVED'}),
    'INDEPENDENT_VERIFY': frozenset({
        'PACKAGE_AUDIT', 'VERIFY_REJECTED', 'ARCHIVED',
    }),
    'VERIFY_REJECTED': frozenset({'ARCHIVED'}),
    'PACKAGE_AUDIT': frozenset({'SELECTED', 'AUDIT_REJECTED', 'ARCHIVED'}),
    'AUDIT_REJECTED': frozenset({'ARCHIVED'}),
    'SELECTED': frozenset(),
    'ARCHIVED': frozenset(),
}


def transition_campaign(state: str, new_state: str) -> str:
    if state not in CAMPAIGN_STATES:
        raise CampaignFatalError(f'unknown campaign state: {state}')
    if new_state not in CAMPAIGN_STATES:
        raise CampaignFatalError(f'unknown campaign state: {new_state}')
    allowed = CAMPAIGN_TRANSITIONS.get(state, frozenset())
    if new_state not in allowed:
        raise CampaignFatalError(
            f'illegal campaign transition {state} -> {new_state}'
        )
    return new_state


def transition_candidate(state: str, new_state: str) -> str:
    if state not in CANDIDATE_STATES:
        raise CampaignFatalError(f'unknown candidate state: {state}')
    if new_state not in CANDIDATE_STATES:
        raise CampaignFatalError(f'unknown candidate state: {new_state}')
    allowed = CANDIDATE_TRANSITIONS.get(state, frozenset())
    if new_state not in allowed:
        raise CampaignFatalError(
            f'illegal candidate transition {state} -> {new_state}'
        )
    return new_state


def classify_q(
    q_upper: float,
    *,
    screening_margin: float,
) -> str:
    """Return SCREENED_Q_LT_1 / BORDERLINE_Q / SCREENED_Q_GE_1."""
    margin = float(screening_margin)
    if abs(float(q_upper) - 1.0) <= margin:
        return 'BORDERLINE_Q'
    if float(q_upper) < 1.0 - margin:
        return 'SCREENED_Q_LT_1'
    return 'SCREENED_Q_GE_1'


def apply_candidate_state(record: dict[str, Any], new_state: str) -> dict[str, Any]:
    current = str(record.get('state') or 'PENDING')
    record['state'] = transition_candidate(current, new_state)
    return record
