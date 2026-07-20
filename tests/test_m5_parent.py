from __future__ import annotations

from pathlib import Path

import pytest

from src.common import atomic_write_json, read_json, sha256_file
from src.m5_parent import M5ParentError, verify_accepted_m4_parent
from tests.m5_helpers import make_synthetic_accepted_m4


def test_m5_parent_accepts_derivative_and_retains_open_bounds(
    tmp_path: Path,
) -> None:
    project, persistent, run_id = make_synthetic_accepted_m4(tmp_path)
    evidence = verify_accepted_m4_parent(project, persistent, run_id)
    assert len(evidence.tensors) == 6
    assert evidence.regression['minimum_observed_centered_fd_order'] >= 1.8
    assert evidence.regression['zero_tangent_residual'] == 0.0
    assert evidence.regression['symmetry_residual'] == 0.0
    assert evidence.bound_ledger['closed_in_M4'][-1] == (
        'finite-difference regression'
    )
    assert evidence.bound_ledger['open_for_M5'][-1] == (
        'basis variation residual'
    )
    assert 'forward tangent regression residual' not in (
        evidence.bound_ledger['open_for_M5']
    )


def test_m5_parent_fails_closed_for_tamper_or_incomplete_handoff(
    tmp_path: Path,
) -> None:
    project, persistent, run_id = make_synthetic_accepted_m4(tmp_path)
    audit_path = project / 'audit/m4_accepted_parent.json'
    audit = read_json(audit_path)
    audit['bound_ledger']['open_for_M5'].pop()
    atomic_write_json(audit_path, audit)
    with pytest.raises(M5ParentError, match='proof-obligation ledger'):
        verify_accepted_m4_parent(project, persistent, run_id)

    audit = read_json(audit_path)
    audit['bound_ledger']['open_for_M5'].append('basis variation residual')
    audit['derivative_regression']['max_final_relative_error'] = 1.0
    atomic_write_json(audit_path, audit)
    with pytest.raises(M5ParentError, match='regression audit'):
        verify_accepted_m4_parent(project, persistent, run_id)

    project, persistent, run_id = make_synthetic_accepted_m4(
        tmp_path / 'regression'
    )
    audit_path = project / 'audit/m4_accepted_parent.json'
    audit = read_json(audit_path)
    report_path = Path(audit['m4_report_path'])
    report = read_json(report_path)
    channel = report['results']['M4_FINITE_DIFFERENCE']['result'][
        'channels'
    ]['temporal_link']
    channel['steps'][-1]['relative_error_frobenius'] = (
        channel['steps'][-2]['relative_error_frobenius'] * 2.0
    )
    channel['final_relative_error'] = channel['steps'][-1][
        'relative_error_frobenius'
    ]
    atomic_write_json(report_path, report)
    audit['m4_report_sha256'] = sha256_file(report_path)
    atomic_write_json(audit_path, audit)
    with pytest.raises(
        M5ParentError,
        match=r'does not converge|maximum regression residual changed',
    ):
        verify_accepted_m4_parent(project, persistent, run_id)
