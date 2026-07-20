"""Primary Campaign B screening (floating-point / enclosure; NOT_CERTIFIED)."""

from __future__ import annotations

from typing import Any

from ..m7_lineage import screen_s2_candidate
from .errors import InvariantViolation
from .schemas import assert_staged_candidate, screening_only_payload
from .state_machine import classify_q


def run_primary_screening(
    candidate: dict[str, Any],
    *,
    parent_q_upper: float,
    parent_rank: int = 16,
    screening_margin: float = 1e-6,
) -> dict[str, Any]:
    assert_staged_candidate(candidate)
    screen = screen_s2_candidate(
        candidate,
        parent_q_upper=float(parent_q_upper),
        parent_rank=int(parent_rank),
    )
    if screen.get('certified') is True:
        raise InvariantViolation('screening must not emit certified=True')
    estimated = float(screen['estimated_q'])
    q_class = classify_q(estimated, screening_margin=screening_margin)
    result = {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'screen_status': screen.get('screen_status'),
        'estimated_q': screen.get('estimated_q'),
        'q_upper': estimated,
        'q_class': q_class,
        'is_q_lt_1': q_class == 'SCREENED_Q_LT_1',
        'is_borderline': q_class == 'BORDERLINE_Q',
        'parent_q_upper': format(float(parent_q_upper), '.17g'),
        'parent_rank': int(parent_rank),
        'effective_projected_rank': screen.get('effective_projected_rank'),
        'm4_geometry_compatible': screen.get('m4_geometry_compatible'),
        'certified': False,
        'j2': int(candidate.get('j2') or 0),
        'execution_mode': candidate.get('execution_mode'),
        'scheme_hash': candidate.get('scheme_hash'),
        'notes': screen.get('notes'),
        **screening_only_payload(),
    }
    return result


def is_q_lt_1(result: dict[str, Any], screening_margin: float) -> bool:
    q_upper = float(result.get('q_upper') if result.get('q_upper') is not None
                    else result.get('estimated_q'))
    return classify_q(q_upper, screening_margin=screening_margin) == 'SCREENED_Q_LT_1'
