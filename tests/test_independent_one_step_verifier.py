from __future__ import annotations

from pathlib import Path

import pytest

from src.independent_one_step_verifier import (
    IndependentVerifierError,
    verify_one_step_package,
)
from src.interval_kernel import construct
from src.m5_package import assemble_one_step_package, make_contractive_fixture_inputs
from src.common import read_json, atomic_write_json


def _build_package(tmp_path: Path) -> Path:
    fixture = make_contractive_fixture_inputs()
    package = tmp_path / 'one_step_certificate'
    assemble_one_step_package(
        package,
        run_id='M5-independent-fixture',
        parent_run_id='M4-fixture',
        config={'cutoff': 2, 'bond_dimension': 16, 'weight_m': '0'},
        conventions={'metric_unit': 'lattice', 'source_speed_unit': 'lattice'},
        initial_tail={'status': 'PASS'},
        basis_equivalence={'status': 'PASS'},
        contraction_residuals={'status': 'PASS'},
        derivative_residuals={'status': 'PASS'},
        normalization_bounds={
            'status': 'PASS',
            'z_min_interval': construct('1').serialize(),
        },
        influence_entries=fixture['entries'],
        row_order=fixture['labels'],
        column_order=fixture['labels'],
        weighted_matrix_entries=fixture['weighted_matrix'],
        weighted_labels=fixture['labels'],
        perron_values=fixture['perron'],
        outside_matrix_tail=fixture['outside_tail'],
    )
    return package


def test_independent_verifier_agrees_with_fixture(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    report = verify_one_step_package(package)
    assert report.agreement is True
    assert report.independent_verdict == 'ONE_STEP_CERTIFIED'
    from fractions import Fraction
    assert Fraction(report.recomputed_q_cert['hi']) == Fraction(3, 20)


def test_independent_verifier_detects_influence_tamper(tmp_path: Path) -> None:
    package = _build_package(tmp_path)
    path = package / 'influence_matrix_intervals.json'
    payload = read_json(path)
    payload['entries'][0]['influence_upper_interval'] = construct('9').serialize()
    atomic_write_json(path, payload)
    # hashes no longer match exact package expectations via recompute
    with pytest.raises(IndependentVerifierError, match='Influence entry formula'):
        verify_one_step_package(package)


def test_independent_verifier_does_not_import_certificate_module() -> None:
    import src.independent_one_step_verifier as module
    source = Path(module.__file__).read_text(encoding='utf-8')
    assert 'from .certificate' not in source
    assert 'from src.certificate' not in source
    assert 'from .influence' not in source
    assert 'from src.influence' not in source
