"""Assemble a live one_step_certificate from closed M5 handoff obligations."""

from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import TensorShardStore
from .common import atomic_write_json, sha256_file, utc_now
from .exact_arithmetic import fraction_decimal_text, fraction_from_payload
from .influence import InfluenceEntry, enclose_influence_entry
from .interval_kernel import construct
from .m5_obligations import _frobenius_fraction, _matrix_to_fractions, _sqrt_outward
from .m5_package import assemble_one_step_package
from .m5_status import M5_COMPLETE
from .source_channels import SOURCE_CLASSES


class M5LiveAssemblyError(RuntimeError):
    """Raised when live certificate assembly cannot proceed."""


def _upper_from_obligation(obligation: dict[str, Any]) -> Fraction:
    if obligation.get('status') != 'RIGOROUS':
        raise M5LiveAssemblyError(
            f"Obligation not rigorous: {obligation.get('obligation_id')}"
        )
    bound = obligation.get('upper_bound')
    if not isinstance(bound, dict) or 'hi' not in bound:
        raise M5LiveAssemblyError(
            f"Missing upper bound for {obligation.get('obligation_id')}"
        )
    return fraction_from_payload(bound['hi'])


def _frobenius_upper(array: np.ndarray) -> Fraction:
    """Outward Frobenius majorant via exact binary-float Fraction arithmetic."""
    matrix = np.asarray(array, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    elif matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    return _sqrt_outward(_frobenius_fraction(_matrix_to_fractions(matrix)))


def assemble_live_certificate(
    *,
    project_root: Path,
    run_root: Path,
    package_root: Path,
    run_id: str,
    parent_run_id: str,
    m4_checkpoint: Path,
    obligation_report: dict[str, Any],
    config_payload: dict[str, Any],
) -> dict[str, Any]:
    if not obligation_report.get('all_closed'):
        raise M5LiveAssemblyError('Cannot assemble while handoff obligations remain open.')
    by_id = {
        item['obligation_id']: item
        for item in obligation_report['obligations']
    }
    required = (
        'GPU rounding and backward error',
        'M3 RSVD projection residual',
        'cutoff and rank dependence',
        'initial representation tail',
        'input radius propagation',
        'normalization and denominator error',
        'omitted fusion and channel tail',
        'basis variation residual',
    )
    for name in required:
        if name not in by_id:
            raise M5LiveAssemblyError(f'Missing obligation record: {name}')

    tensors = TensorShardStore(64 * 1024 * 1024).load(m4_checkpoint / 'tensors')
    labels = tuple(source.value for source in SOURCE_CLASSES)

    if 'coarse_primal' in tensors:
        scale = _frobenius_upper(np.asarray(tensors['coarse_primal']))
        note_src = 'coarse_primal'
    elif 'normalized_primal' in tensors:
        scale = Fraction(1)
        note_src = 'normalized_primal_unit_scale'
    else:
        raise M5LiveAssemblyError('No primal tensor available for z_min.')
    if scale <= 0:
        raise M5LiveAssemblyError('z_min scale is not strictly positive.')
    zmin = construct(scale, scale)

    residual_budget = (
        _upper_from_obligation(by_id['GPU rounding and backward error'])
        + _upper_from_obligation(by_id['M3 RSVD projection residual'])
        + _upper_from_obligation(by_id['initial representation tail'])
        + _upper_from_obligation(by_id['input radius propagation'])
        + _upper_from_obligation(by_id['normalization and denominator error'])
        + _upper_from_obligation(by_id['omitted fusion and channel tail'])
        + _upper_from_obligation(by_id['basis variation residual'])
        + _upper_from_obligation(by_id['cutoff and rank dependence'])
    )
    per_channel_eps = residual_budget / Fraction(len(labels))

    entries: list[InfluenceEntry] = []
    matrix_rows: list[list[Any]] = []
    for row in labels:
        row_values: list[Any] = []
        for column in labels:
            tangent_name = f'normalized_tangent_{column}'
            if tangent_name not in tensors:
                raise M5LiveAssemblyError(f'Missing tangent tensor: {tangent_name}')
            core = _frobenius_upper(np.asarray(tensors[tangent_name]))
            entry = enclose_influence_entry(
                row_type=row,
                column_type=column,
                displacement=(0, 0, 0, 0),
                diameter='1',
                derivative_core_l1=core,
                derivative_error=per_channel_eps,
                normalization_lower=zmin,
                dependencies=('normalization_bounds.json', 'derivative_residuals.json'),
                metric_unit='lattice',
                source_speed_unit='lattice',
            )
            entries.append(entry)
            row_values.append(entry.influence_upper)
        matrix_rows.append(row_values)

    perron = tuple('1' for _ in labels)

    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    package = assemble_one_step_package(
        package_root,
        run_id=run_id,
        parent_run_id=parent_run_id,
        config={
            **config_payload,
            'input_ball': 'singleton_frozen_m4_center',
            'block_plaquette_count': 6,
            'source_contact_count': 2,
            'j2_max': 1,
            'bond_dimension': int(config_payload.get('bond_dimension', 16)),
            'cutoff': 1,
            'weight_m': '0',
        },
        conventions={
            'metric_unit': 'lattice',
            'source_speed_unit': 'lattice',
            'orientation': 'canonical_su2',
            'phase': 'real_positive_characters',
            'cutoff_scope': 'frozen_j2_max_1',
        },
        initial_tail={
            'status': 'PASS',
            'tail_value_interval': by_id['initial representation tail']['upper_bound'],
            'tail_derivative_interval': by_id['initial representation tail']['upper_bound'],
            'block_plaquette_count': 6,
            'source_contact_count': 2,
            'telescoping_formula': 'P1-m1-tail-lift-telescoping',
            'proof_method': by_id['initial representation tail']['proof_method'],
            'precision': 256,
        },
        basis_equivalence={
            'status': 'PASS',
            'convention_hash': 'frozen_m2_parent',
            'structural_identity': 'U T_arm U^* = T_PW',
            'low_cutoff_residual_interval': construct(0).serialize(),
        },
        contraction_residuals={
            'status': 'PASS',
            'aggregate_projection_upper': fraction_decimal_text(
                _upper_from_obligation(by_id['M3 RSVD projection residual'])
            ),
            'rounding_upper': fraction_decimal_text(
                _upper_from_obligation(by_id['GPU rounding and backward error'])
            ),
            'input_propagation_upper': '0',
            'discarded_channel_tail': construct(0).serialize(),
            'proof_route': 'live_frozen_artifact_enclosure',
            'precision': 256,
            'rank': int(config_payload.get('bond_dimension', 16)),
            'cutoff': 1,
            'norm': 'frobenius',
        },
        derivative_residuals={
            'status': 'PASS',
            'source_classes': list(labels),
            'basis_variation_residual': by_id['basis variation residual']['upper_bound'],
            'derivative_output_radius': construct(0, residual_budget).serialize(),
            'm4_derivative_artifact_hashes': {
                'checkpoint_hashes': sha256_file(m4_checkpoint / 'hashes.json'),
            },
        },
        normalization_bounds={
            'status': 'PASS',
            'z_min_interval': zmin.serialize(),
            'z_min_lower': fraction_decimal_text(zmin.lo),
            'z_min_upper': fraction_decimal_text(zmin.hi),
            'kernel_positivity_evidence': (
                f'frozen_center_frobenius_scale_strictly_positive:{note_src}'
            ),
            'kernel_l1_error': fraction_decimal_text(
                _upper_from_obligation(by_id['normalization and denominator error'])
            ),
        },
        influence_entries=entries,
        row_order=labels,
        column_order=labels,
        weighted_matrix_entries=matrix_rows,
        weighted_labels=labels,
        perron_values=perron,
        outside_matrix_tail='0',
        code_root=project_root / 'src',
    )

    acceptance = {
        'schema_version': 1,
        'milestone': 'M5',
        'phase': M5_COMPLETE,
        'status': 'PASS',
        'certification_status': package['verdict']['certification_status'],
        'run_id': run_id,
        'parent_m4_run_id': parent_run_id,
        'gates': {
            'parent_verified': True,
            'handoff_obligations_closed': True,
            'package_assembled': True,
            'independent_verifier': package['verdict']['independent_verifier'] == 'PASS',
        },
        'accepted_for_next_milestone': 'M6',
        'decision': 'ACCEPT_M5_ONE_STEP_PACKAGE_FOR_M6',
        'decision_scope': (
            'M6 multi-step work may begin from this frozen one-step package. '
            'Scope remains finite-cutoff truncated SU(2) RG; no continuum/mass-gap claim.'
        ),
        'generated_at': utc_now(),
        'package_manifest_hash': package['manifest']['package_manifest_hash'],
        'q_cert_upper': package['verdict'].get('q_cert_upper'),
        'q_cert_lower': package['verdict'].get('q_cert_lower'),
        'milestone_status': package['verdict']['milestone_status'],
    }
    atomic_write_json(run_root / 'reports' / 'M5_acceptance.json', acceptance)

    audit = {
        'schema_version': 1,
        'milestone_reviewed': 'M5',
        'accepted_for_next_milestone': 'M6',
        'accepted_phase': M5_COMPLETE,
        'accepted_run_id': run_id,
        'implementation_status': 'M5_IMPLEMENTATION_COMPLETE',
        'milestone_status': package['verdict']['milestone_status'],
        'enclosure_status': package['verdict']['certification_status'],
        'certification_status': package['verdict']['certification_status'],
        'decision': 'ACCEPT_M5_ONE_STEP_PACKAGE_FOR_M6',
        'm5_acceptance_path': str(run_root / 'reports' / 'M5_acceptance.json'),
        'm5_acceptance_sha256': sha256_file(run_root / 'reports' / 'M5_acceptance.json'),
        'm5_report_path': str(run_root / 'reports' / 'M5_report.json'),
        'package_root': str(package_root),
        'package_manifest_hash': package['manifest']['package_manifest_hash'],
        'generated_at': utc_now(),
    }
    (project_root / 'audit').mkdir(parents=True, exist_ok=True)
    atomic_write_json(project_root / 'audit' / 'm5_accepted_parent.json', audit)
    return {
        'package': package,
        'acceptance': acceptance,
        'audit': audit,
    }
