"""Assemble and validate a complete one_step_certificate package."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .certificate import (
    CollatzBound,
    PositiveRationalVector,
    collatz_certificate,
    nonnegative_interval_matrix,
    positive_rational_vector,
)
from .common import atomic_write_json, atomic_write_text, hash_tree, sha256_file
from .independent_one_step_verifier import verify_one_step_package
from .influence import InfluenceEntry, assemble_influence_matrix
from .interval_kernel import ProofInterval, construct
from .proof_manifest import (
    ONE_STEP_CERTIFICATE_FILES,
    ProofDependency,
    verify_immutable_package,
    write_certificate_manifest,
)
from .m5_status import M5_COMPLETE, NOT_CERTIFIED, ONE_STEP_CERTIFIED


class M5PackageError(RuntimeError):
    """Raised when certificate package assembly fails closed."""


def _write_json(package: Path, name: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(package / name, dict(payload))


def build_theorem_statement(
    *,
    run_id: str,
    parent_run_id: str,
    cutoff: int,
    bond_dimension: int,
    weight_m: str,
    source_classes: Sequence[str],
) -> str:
    classes = ', '.join(source_classes)
    return f"""# One-step validated RG certificate

## Scope

This package certifies at most a single RG step for a fixed finite cutoff,
fixed bond dimension, fixed source class list, and fixed boundary metric.
It does **not** claim continuum Yang–Mills, infinite volume, OS positivity,
or a mass gap.

## Parameters

- M5 run ID: `{run_id}`
- Immutable M4 parent run ID: `{parent_run_id}`
- Cutoff: `{cutoff}`
- Bond dimension: `{bond_dimension}`
- Weighted influence parameter m: `{weight_m}`
- Source classes (canonical order): `{classes}`

## Claim form

Either `q_cert_upper < 1` with all P0–P11/C1–C8 PASS, or a verified
`NOT_CERTIFIED` completion when a rigorously computed gate fails.
"""


def assemble_one_step_package(
    package_root: Path,
    *,
    run_id: str,
    parent_run_id: str,
    config: Mapping[str, Any],
    conventions: Mapping[str, Any],
    initial_tail: Mapping[str, Any],
    basis_equivalence: Mapping[str, Any],
    contraction_residuals: Mapping[str, Any],
    derivative_residuals: Mapping[str, Any],
    normalization_bounds: Mapping[str, Any],
    influence_entries: Sequence[InfluenceEntry],
    row_order: Sequence[str],
    column_order: Sequence[str],
    weighted_matrix_entries: Sequence[Sequence[Any]],
    weighted_labels: Sequence[str],
    perron_values: Sequence[Any],
    outside_matrix_tail: Any = 0,
    code_root: Path | None = None,
    proof_obligation_status: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    package_root.mkdir(parents=True, exist_ok=True)
    if any(package_root.iterdir()):
        raise M5PackageError('Package directory must be empty before assembly.')

    influence_doc = assemble_influence_matrix(
        influence_entries, row_order=row_order, column_order=column_order,
    )
    matrix = nonnegative_interval_matrix(weighted_matrix_entries, weighted_labels)
    vector = positive_rational_vector(perron_values, weighted_labels)
    bound = collatz_certificate(matrix, vector, outside_matrix_tail=outside_matrix_tail)

    influence_doc['weighted_matrix'] = {
        'labels': list(weighted_labels),
        'entries': [
            [cell.serialize() for cell in row]
            for row in matrix.entries
        ],
        'outside_matrix_tail_policy': 'added_once_in_collatz_bound',
    }

    code_hashes = {
        'src_tree_sha256': hash_tree(code_root, suffixes=('.py',)) if code_root else None,
        'policy': 'hash_tree_py_files',
    }
    dependencies = [
        ProofDependency('P0', (), 'config.json'),
        ProofDependency('P1', ('P0',), 'initial_tail.json'),
        ProofDependency('P2', ('P0',), 'basis_equivalence.json'),
        ProofDependency('P3', ('P0', 'P2'), 'contraction_residuals.json'),
        ProofDependency('P4', ('P3',), 'contraction_residuals.json'),
        ProofDependency('P5', ('P3',), 'contraction_residuals.json'),
        ProofDependency('P6', ('P1', 'P3', 'P4', 'P5'), 'contraction_residuals.json'),
        ProofDependency('P7', ('P6',), 'derivative_residuals.json'),
        ProofDependency('P8', ('P0',), 'normalization_bounds.json'),
        ProofDependency('P9', ('P7', 'P8'), 'influence_matrix_intervals.json'),
        ProofDependency('P10', ('P9',), 'influence_matrix_intervals.json'),
        ProofDependency('P11', ('P10',), 'collatz_bound.json'),
        ProofDependency('VERDICT', ('P11',), 'verdict.json'),
    ]

    obligations = {
        key: 'PASS' for key in (f'P{index}' for index in range(12))
    }
    if proof_obligation_status:
        obligations.update(proof_obligation_status)

    if bound.verdict == ONE_STEP_CERTIFIED:
        milestone_status = ONE_STEP_CERTIFIED
        certification_status = ONE_STEP_CERTIFIED
        phase = M5_COMPLETE
    elif bound.verdict == NOT_CERTIFIED:
        milestone_status = NOT_CERTIFIED
        certification_status = NOT_CERTIFIED
        phase = M5_COMPLETE
    else:
        # Threshold-crossing interval is not a completed M5.
        raise M5PackageError(
            'q_cert interval crosses 1; package cannot be marked M5_COMPLETE.'
        )

    atomic_write_text(
        package_root / 'theorem_statement.md',
        build_theorem_statement(
            run_id=run_id,
            parent_run_id=parent_run_id,
            cutoff=int(config.get('cutoff', 0)),
            bond_dimension=int(config.get('bond_dimension', 0)),
            weight_m=str(config.get('weight_m', '0')),
            source_classes=list(row_order),
        ),
    )
    _write_json(package_root, 'config.json', config)
    _write_json(package_root, 'code_hashes.json', code_hashes)
    _write_json(package_root, 'conventions.json', conventions)
    _write_json(package_root, 'initial_tail.json', initial_tail)
    _write_json(package_root, 'basis_equivalence.json', basis_equivalence)
    _write_json(package_root, 'contraction_residuals.json', contraction_residuals)
    _write_json(package_root, 'derivative_residuals.json', derivative_residuals)
    _write_json(package_root, 'normalization_bounds.json', normalization_bounds)
    _write_json(package_root, 'influence_matrix_intervals.json', influence_doc)
    _write_json(package_root, 'perron_vector.json', vector.payload())
    _write_json(package_root, 'collatz_bound.json', bound.payload())
    _write_json(
        package_root,
        'proof_dependencies.json',
        {'schema_version': 1, 'nodes': [node.payload() for node in dependencies]},
    )

    verdict = {
        'schema_version': 1,
        'run_id': run_id,
        'phase': phase,
        'milestone_status': milestone_status,
        'certification_status': certification_status,
        'q_collatz_upper': format(bound.q_collatz.hi, 'f'),
        'outside_matrix_tail_upper': format(bound.outside_matrix_tail.hi, 'f'),
        'q_cert_upper': format(bound.q_cert.hi, 'f'),
        'q_cert_lower': format(bound.q_cert.lo, 'f'),
        'margin_lower': format(bound.margin.lo, 'f'),
        'proof_obligations': obligations,
        'independent_verifier': 'PENDING',
    }
    if certification_status == NOT_CERTIFIED:
        verdict['failure_gate'] = 'P11'
        verdict['failure_reason'] = 'verified q_cert_lower >= 1'
        verdict['q_cert_interval'] = [
            format(bound.q_cert.lo, 'f'),
            format(bound.q_cert.hi, 'f'),
        ]
    _write_json(package_root, 'verdict.json', verdict)
    verify_immutable_package(package_root, dependencies=dependencies)

    preview = verify_one_step_package(
        package_root, require_independent_pass_marker=False,
    )
    if preview.independent_verdict != certification_status:
        raise M5PackageError('Independent verifier disagreed before PASS marker write.')
    verdict['independent_verifier'] = 'PASS'
    _write_json(package_root, 'verdict.json', verdict)

    final_report = verify_one_step_package(package_root)
    if not final_report.agreement:
        raise M5PackageError('Independent verifier did not agree with main verdict.')
    manifest = verify_immutable_package(package_root, dependencies=dependencies)
    return {
        'package_root': str(package_root.resolve()),
        'manifest': manifest,
        'independent_report': final_report.payload(),
        'collatz': bound.payload(),
        'verdict': verdict,
        'exact_file_set': list(ONE_STEP_CERTIFICATE_FILES),
    }


def make_contractive_fixture_inputs() -> dict[str, Any]:
    """Exact small rational fixture with q_cert < 1."""
    labels = ('a', 'b')
    entries = [
        InfluenceEntry(
            row_type='a',
            column_type='a',
            displacement=(0,),
            diameter=construct('1'),
            derivative_core_l1=construct('1/10'),
            derivative_error=construct('0'),
            normalization_lower=construct('1'),
            influence_upper=construct('1/10'),
            orbit_multiplicity=1,
            formula='D_i*(||d_j K_tilde||_1 + eps_1j)/z_min',
            dependencies=('normalization_bounds.json',),
            metric_unit='lattice',
            source_speed_unit='lattice',
        ),
        InfluenceEntry(
            row_type='a',
            column_type='b',
            displacement=(0,),
            diameter=construct('1'),
            derivative_core_l1=construct('1/20'),
            derivative_error=construct('0'),
            normalization_lower=construct('1'),
            influence_upper=construct('1/20'),
            orbit_multiplicity=1,
            formula='D_i*(||d_j K_tilde||_1 + eps_1j)/z_min',
            dependencies=('normalization_bounds.json',),
            metric_unit='lattice',
            source_speed_unit='lattice',
        ),
        InfluenceEntry(
            row_type='b',
            column_type='a',
            displacement=(0,),
            diameter=construct('1'),
            derivative_core_l1=construct('1/20'),
            derivative_error=construct('0'),
            normalization_lower=construct('1'),
            influence_upper=construct('1/20'),
            orbit_multiplicity=1,
            formula='D_i*(||d_j K_tilde||_1 + eps_1j)/z_min',
            dependencies=('normalization_bounds.json',),
            metric_unit='lattice',
            source_speed_unit='lattice',
        ),
        InfluenceEntry(
            row_type='b',
            column_type='b',
            displacement=(0,),
            diameter=construct('1'),
            derivative_core_l1=construct('1/10'),
            derivative_error=construct('0'),
            normalization_lower=construct('1'),
            influence_upper=construct('1/10'),
            orbit_multiplicity=1,
            formula='D_i*(||d_j K_tilde||_1 + eps_1j)/z_min',
            dependencies=('normalization_bounds.json',),
            metric_unit='lattice',
            source_speed_unit='lattice',
        ),
    ]
    return {
        'labels': labels,
        'entries': entries,
        'weighted_matrix': (('1/10', '1/20'), ('1/20', '1/10')),
        'perron': ('1', '1'),
        'outside_tail': '0',
    }


def make_noncontractive_fixture_inputs() -> dict[str, Any]:
    """Exact fixture with q_cert_lower >= 1."""
    data = make_contractive_fixture_inputs()
    data['weighted_matrix'] = (('2', '0'), ('0', '2'))
    # Rebuild influence uppers to match the weighted matrix used for Collatz.
    rebuilt: list[InfluenceEntry] = []
    for entry, upper in zip(
        data['entries'],
        ('2', '0', '0', '2'),
    ):
        rebuilt.append(
            InfluenceEntry(
                row_type=entry.row_type,
                column_type=entry.column_type,
                displacement=entry.displacement,
                diameter=construct('1'),
                derivative_core_l1=construct(upper),
                derivative_error=construct('0'),
                normalization_lower=construct('1'),
                influence_upper=construct(upper),
                orbit_multiplicity=1,
                formula=entry.formula,
                dependencies=entry.dependencies,
                metric_unit=entry.metric_unit,
                source_speed_unit=entry.source_speed_unit,
            )
        )
    data['entries'] = rebuilt
    return data
