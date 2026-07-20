from __future__ import annotations

from src.campaign_b.state_machine import classify_q, transition_candidate, transition_campaign
from src.campaign_b.errors import CampaignFatalError


def test_campaign_transitions() -> None:
    assert transition_campaign('CREATED', 'PREFLIGHT') == 'PREFLIGHT'
    assert transition_campaign('PREFLIGHT', 'RUNNING') == 'RUNNING'


def test_illegal_campaign_transition() -> None:
    try:
        transition_campaign('CREATED', 'COMPLETE')
        ok = False
    except CampaignFatalError:
        ok = True
    assert ok


def test_classify_q() -> None:
    assert classify_q(0.9, screening_margin=1e-6) == 'SCREENED_Q_LT_1'
    assert classify_q(1.0, screening_margin=1e-6) == 'BORDERLINE_Q'
    assert classify_q(1.1, screening_margin=1e-6) == 'SCREENED_Q_GE_1'


def test_candidate_path_to_selected() -> None:
    state = 'PENDING'
    for nxt in (
        'RESERVED', 'SCREENING', 'SCREENED_Q_LT_1', 'M2_RESOLVE',
        'READY_SHARED', 'S0', 'INDEPENDENT_VERIFY', 'PACKAGE_AUDIT', 'SELECTED',
    ):
        state = transition_candidate(state, nxt)
    assert state == 'SELECTED'
