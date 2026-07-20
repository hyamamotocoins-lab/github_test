from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from src.certificate import (
    CertificateError,
    collatz_certificate,
    nonnegative_interval_matrix,
    positive_rational_vector,
)
from src.influence import InfluenceError, enclose_influence_entry
from src.interval_kernel import (
    IntervalKernelError,
    construct,
    deserialize,
    divide,
    serialize,
    sum_outward,
)
from src.m5_package import (
    assemble_one_step_package,
    make_contractive_fixture_inputs,
    make_noncontractive_fixture_inputs,
)
from src.proof_manifest import (
    ONE_STEP_CERTIFICATE_FILES,
    ProofDependency,
    ProofManifestError,
    exact_file_set,
    verify_immutable_package,
)
from src.residual_validation import (
    ProofMethod,
    ResidualDependency,
    ResidualLedger,
    ResidualValidationError,
    RigourStatus,
)


def test_interval_basic_arithmetic() -> None:
    a = construct('1/2', '2/3')
    b = construct('1/3', '1/2')
    assert a.add(b).lo == Fraction(5, 6)
    assert a.multiply(b).hi == Fraction(1, 3)
    assert a.square().lo == Fraction(1, 4)


def test_decimal_enclosure() -> None:
    value = construct('0.1', '0.2')
    assert value.lo == Fraction('0.1')
    assert value.hi == Fraction('0.2')


def test_zero_containing_denominator_rejection() -> None:
    numerator = construct('1')
    denominator = construct('-1', '1')
    with pytest.raises(IntervalKernelError, match='zero'):
        divide(numerator, denominator)


def test_nonpositive_normalization_failure() -> None:
    with pytest.raises(InfluenceError, match='z_min'):
        enclose_influence_entry(
            row_type='a',
            column_type='b',
            displacement=(0,),
            diameter='1',
            derivative_core_l1='1',
            derivative_error='0',
            normalization_lower='0',
            dependencies=('normalization_bounds.json',),
            metric_unit='lattice',
            source_speed_unit='lattice',
        )


def test_deterministic_residual_aggregate() -> None:
    ledger = ResidualLedger()
    digest = 'a' * 64
    ledger.add_term(
        term_id='t1',
        category='proj',
        mathematical_meaning='projection residual',
        norm_id='frobenius',
        lower='1/10',
        upper='1/5',
        source_artifacts=('contraction_residuals.json',),
        source_hashes=(digest,),
        rigour_status=RigourStatus.RIGOROUS,
        formula_id='F-proj',
        sector_scope='all',
        cutoff_scope='N=2',
        rank_scope='chi=16',
        proof_method=ProofMethod.EXPLICIT_FROBENIUS,
        contribution_key='proj:all',
    )
    ledger.add_term(
        term_id='t2',
        category='round',
        mathematical_meaning='rounding residual',
        norm_id='frobenius',
        lower='0',
        upper='1/10',
        source_artifacts=('contraction_residuals.json',),
        source_hashes=(digest,),
        dependencies=(ResidualDependency('t1', 'uses'),),
        rigour_status=RigourStatus.RIGOROUS,
        formula_id='F-round',
        sector_scope='all',
        cutoff_scope='N=2',
        rank_scope='chi=16',
        proof_method=ProofMethod.CPU_MULTIPRECISION,
        contribution_key='round:all',
    )
    aggregate = ledger.aggregate_rigorous(
        aggregate_id='out',
        term_ids=('t1', 't2'),
        formula_id='F-sum',
    )
    assert aggregate.upper == Fraction(3, 10)


def test_heuristic_residual_rejection() -> None:
    ledger = ResidualLedger()
    digest = 'b' * 64
    ledger.add_term(
        term_id='h1',
        category='rsvd',
        mathematical_meaning='randomized failure probability',
        norm_id='frobenius',
        lower='0',
        upper='1',
        source_artifacts=('x.json',),
        source_hashes=(digest,),
        rigour_status=RigourStatus.HEURISTIC,
        formula_id='F-h',
        sector_scope='all',
        cutoff_scope='N=2',
        rank_scope='chi=16',
        proof_method=ProofMethod.BLOCK_NORM,
        contribution_key='h:all',
    )
    with pytest.raises(ResidualValidationError, match='Heuristic'):
        ledger.aggregate_rigorous(
            aggregate_id='bad',
            term_ids=('h1',),
            formula_id='F-sum',
        )


def test_missing_provenance_rejection() -> None:
    ledger = ResidualLedger()
    with pytest.raises(ResidualValidationError, match='provenance'):
        ledger.add_term(
            term_id='t',
            category='x',
            mathematical_meaning='m',
            norm_id='n',
            lower='0',
            upper='0',
            source_artifacts=(),
            source_hashes=(),
            rigour_status=RigourStatus.RIGOROUS,
            formula_id='F',
            sector_scope='s',
            cutoff_scope='c',
            rank_scope='r',
            proof_method=ProofMethod.ANALYTIC_TAIL,
            contribution_key='k',
        )


def test_duplicate_contribution_rejection() -> None:
    ledger = ResidualLedger()
    digest = 'c' * 64
    kwargs = dict(
        category='x',
        mathematical_meaning='m',
        norm_id='n',
        lower='0',
        upper='1',
        source_artifacts=('a.json',),
        source_hashes=(digest,),
        rigour_status=RigourStatus.RIGOROUS,
        formula_id='F',
        sector_scope='s',
        cutoff_scope='c',
        rank_scope='r',
        proof_method=ProofMethod.ANALYTIC_TAIL,
        contribution_key='same',
    )
    ledger.add_term(term_id='t1', **kwargs)
    with pytest.raises(ResidualValidationError, match='Duplicate residual contribution'):
        ledger.add_term(term_id='t2', **kwargs)


def test_influence_formula() -> None:
    entry = enclose_influence_entry(
        row_type='a',
        column_type='b',
        displacement=(1, 0, 0, 0),
        diameter='2',
        derivative_core_l1='1/4',
        derivative_error='1/4',
        normalization_lower='1',
        dependencies=('normalization_bounds.json',),
        metric_unit='lattice',
        source_speed_unit='lattice',
    )
    assert entry.influence_upper.lo == Fraction(1)
    assert entry.influence_upper.hi == Fraction(1)


def test_invalid_dimensions_rejection() -> None:
    with pytest.raises(CertificateError, match='dimension'):
        nonnegative_interval_matrix([['1']], labels=('a', 'b'))


def test_positive_rational_perron_vector() -> None:
    vector = positive_rational_vector(('1', '2/3'), ('a', 'b'))
    assert vector.components[1] == Fraction(2, 3)
    with pytest.raises(CertificateError, match='positive'):
        positive_rational_vector(('0', '1'), ('a', 'b'))


def test_exact_small_rational_collatz_bound() -> None:
    matrix = nonnegative_interval_matrix(
        (('1/10', '1/20'), ('1/20', '1/10')),
        ('a', 'b'),
    )
    vector = positive_rational_vector(('1', '1'), ('a', 'b'))
    bound = collatz_certificate(matrix, vector, outside_matrix_tail='0')
    assert bound.q_cert.hi == Fraction(3, 20)


def test_q_less_than_one_success() -> None:
    matrix = nonnegative_interval_matrix((('1/2', '0'), ('0', '1/2')), ('a', 'b'))
    vector = positive_rational_vector(('1', '1'), ('a', 'b'))
    bound = collatz_certificate(matrix, vector)
    assert bound.verdict == 'ONE_STEP_CERTIFIED'
    assert bound.q_cert.hi < 1


def test_q_equal_one_failure() -> None:
    matrix = nonnegative_interval_matrix((('1', '0'), ('0', '1')), ('a', 'b'))
    vector = positive_rational_vector(('1', '1'), ('a', 'b'))
    bound = collatz_certificate(matrix, vector)
    assert bound.verdict == 'NOT_CERTIFIED'
    assert bound.q_cert.lo >= 1


def test_q_greater_than_one_failure() -> None:
    matrix = nonnegative_interval_matrix((('2', '0'), ('0', '2')), ('a', 'b'))
    vector = positive_rational_vector(('1', '1'), ('a', 'b'))
    bound = collatz_certificate(matrix, vector)
    assert bound.verdict == 'NOT_CERTIFIED'
    assert bound.q_cert.lo > 1


def test_missing_q_failure_via_blocked_crossing() -> None:
    matrix = nonnegative_interval_matrix(
        ((construct('1/2', '3/2'), construct('0')), (construct('0'), construct('1/2', '3/2'))),
        ('a', 'b'),
    )
    vector = positive_rational_vector(('1', '1'), ('a', 'b'))
    bound = collatz_certificate(matrix, vector)
    assert bound.verdict == 'BLOCKED_MATH'


def test_nan_inf_rejection() -> None:
    with pytest.raises(IntervalKernelError):
        construct('nan')
    with pytest.raises(IntervalKernelError):
        construct('inf')


def test_package_missing_and_extra_and_symlink(
    tmp_path: Path,
) -> None:
    package = tmp_path / 'one_step_certificate'
    package.mkdir()
    for name in ONE_STEP_CERTIFICATE_FILES:
        (package / name).write_text('{}', encoding='utf-8')
    exact_file_set(package)
    (package / 'extra.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ProofManifestError, match='extra'):
        exact_file_set(package)
    (package / 'extra.json').unlink()
    (package / 'verdict.json').unlink()
    with pytest.raises(ProofManifestError, match='missing'):
        exact_file_set(package)
    (package / 'verdict.json').write_text('{}', encoding='utf-8')
    link = package / 'link.json'
    try:
        link.symlink_to(package / 'config.json')
    except OSError:
        pytest.skip('symlink unavailable')
    with pytest.raises(ProofManifestError, match='Symlink'):
        exact_file_set(package)


def test_hash_tampering_detection(tmp_path: Path) -> None:
    package = tmp_path / 'pkg'
    package.mkdir()
    for name in ONE_STEP_CERTIFICATE_FILES:
        (package / name).write_text('{"ok":true}', encoding='utf-8')
    manifest = verify_immutable_package(package)
    (package / 'config.json').write_text('{"ok":false}', encoding='utf-8')
    with pytest.raises(ProofManifestError, match='tampering'):
        verify_immutable_package(package, expected_hashes=manifest['file_hashes'])


def test_dependency_cycle_rejection() -> None:
    from src.proof_manifest import validate_dependency_dag

    with pytest.raises(ProofManifestError, match='cycle'):
        validate_dependency_dag([
            ProofDependency('a', ('b',), 'config.json'),
            ProofDependency('b', ('a',), 'verdict.json'),
        ])


def test_serialization_restart_exactness() -> None:
    original = construct('1/7', '2/7')
    restored = deserialize(serialize(original))
    assert restored == original


def test_missing_residual_is_not_zero() -> None:
    ledger = ResidualLedger()
    with pytest.raises(ResidualValidationError, match='not treated as zero'):
        ledger.require_term('absent')


def test_main_and_independent_fixture_agreement(tmp_path: Path) -> None:
    fixture = make_contractive_fixture_inputs()
    package = tmp_path / 'one_step_certificate'
    result = assemble_one_step_package(
        package,
        run_id='M5-fixture-contractive',
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
    assert result['verdict']['certification_status'] == 'ONE_STEP_CERTIFIED'
    assert result['independent_report']['agreement'] is True

    package_fail = tmp_path / 'one_step_certificate_fail'
    fixture_fail = make_noncontractive_fixture_inputs()
    result_fail = assemble_one_step_package(
        package_fail,
        run_id='M5-fixture-noncontractive',
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
        influence_entries=fixture_fail['entries'],
        row_order=fixture_fail['labels'],
        column_order=fixture_fail['labels'],
        weighted_matrix_entries=fixture_fail['weighted_matrix'],
        weighted_labels=fixture_fail['labels'],
        perron_values=fixture_fail['perron'],
        outside_matrix_tail=fixture_fail['outside_tail'],
    )
    assert result_fail['verdict']['certification_status'] == 'NOT_CERTIFIED'
    assert result_fail['independent_report']['agreement'] is True


def test_sum_outward_and_package_json_allow_nan_false(tmp_path: Path) -> None:
    total = sum_outward((construct('1/2'), construct('1/3')))
    assert total.hi == Fraction(5, 6)
    path = tmp_path / 'x.json'
    path.write_text(json.dumps({'v': 1}, allow_nan=False), encoding='utf-8')
    assert '"v": 1' in path.read_text(encoding='utf-8')
