from __future__ import annotations

from fractions import Fraction

import pytest

from src.exact_2d_rg import trajectory_payload
from src.m1_verifier import IndependentVerificationError, independent_convolution_verify

BETA = Fraction(11, 5)


def test_independent_diagonal_convolution_contains_primary() -> None:
    primary = trajectory_payload(BETA, (2, 3, 4), 3, 96, 36)
    result = independent_convolution_verify(primary, BETA, (2, 3, 4), 3, 72)
    assert result['status'] == 'PASS'
    assert result['does_not_call_primary_recurrence'] is True
    assert set(result['independent_trajectories']) == {'2', '3', '4'}
    assert len(result['checks']) == 12
    assert all(check['overlaps'] and check['primary_inside_independent'] for check in result['checks'])


def test_independent_verifier_fails_closed_on_corrupt_primary_interval() -> None:
    primary = trajectory_payload(BETA, (2, 3, 4), 3, 96, 36)
    corrupt = primary['trajectories']['2'][0]
    corrupt['lo'] = {'numerator_hex': '2', 'denominator_hex': '1'}
    corrupt['hi'] = {'numerator_hex': '2', 'denominator_hex': '1'}
    with pytest.raises(IndependentVerificationError):
        independent_convolution_verify(primary, BETA, (2, 3, 4), 3, 72)
