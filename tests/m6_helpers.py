"""Helpers for synthetic accepted M5 parents used by M6 tests."""

from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json, atomic_write_text, sha256_file
from src.interval_kernel import construct
from src.m5_package import (
    assemble_one_step_package,
    make_contractive_fixture_inputs,
    make_noncontractive_fixture_inputs,
)
from src.m5_status import M5_COMPLETE
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS


def seed_project_docs(project: Path) -> None:
    (project / 'src').mkdir(parents=True, exist_ok=True)
    (project / 'audit').mkdir(parents=True, exist_ok=True)
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_write_text(path, f'# fixture\n{relative}\n')


def make_synthetic_accepted_m5(
    tmp_path: Path,
    *,
    run_id: str = 'M5-synthetic-for-m6',
    contractive: bool = False,
) -> tuple[Path, Path, str]:
    """Create a minimal accepted M5 parent with one_step_certificate."""
    project = tmp_path / 'project'
    persistent = tmp_path / 'persist'
    seed_project_docs(project)
    run_root = persistent / 'runs' / run_id
    package_root = run_root / 'artifacts' / 'one_step_certificate'
    package_root.mkdir(parents=True)

    fixture = (
        make_contractive_fixture_inputs()
        if contractive
        else make_noncontractive_fixture_inputs()
    )
    zmin = construct('1')
    package = assemble_one_step_package(
        package_root,
        run_id=run_id,
        parent_run_id='M4-synthetic',
        config={'cutoff': 1, 'bond_dimension': 16, 'weight_m': '0'},
        conventions={
            'metric_unit': 'lattice',
            'source_speed_unit': 'lattice',
            'orientation': 'canonical_su2',
            'phase': 'real_positive_characters',
        },
        initial_tail={'status': 'PASS', 'tail_value_interval': construct(0).serialize()},
        basis_equivalence={'status': 'PASS', 'convention_hash': 'fixture'},
        contraction_residuals={
            'status': 'PASS',
            'aggregate_projection_upper': '0',
            'rounding_upper': '0',
            'input_propagation_upper': '0',
            'discarded_channel_tail': construct(0).serialize(),
            'proof_route': 'fixture',
            'precision': 64,
            'rank': 16,
            'cutoff': 1,
            'norm': 'frobenius',
        },
        derivative_residuals={
            'status': 'PASS',
            'source_classes': list(fixture['labels']),
            'basis_variation_residual': construct(0).serialize(),
            'derivative_output_radius': construct(0).serialize(),
            'm4_derivative_artifact_hashes': {},
        },
        normalization_bounds={
            'status': 'PASS',
            'z_min_interval': zmin.serialize(),
            'z_min_lower': '1',
            'z_min_upper': '1',
            'kernel_positivity_evidence': 'fixture',
            'kernel_l1_error': '0',
        },
        influence_entries=fixture['entries'],
        row_order=fixture['labels'],
        column_order=fixture['labels'],
        weighted_matrix_entries=fixture['weighted_matrix'],
        weighted_labels=fixture['labels'],
        perron_values=fixture['perron'],
        outside_matrix_tail=fixture['outside_tail'],
        code_root=project / 'src',
    )
    acceptance = {
        'schema_version': 1,
        'milestone': 'M5',
        'phase': M5_COMPLETE,
        'status': 'PASS',
        'certification_status': package['verdict']['certification_status'],
        'run_id': run_id,
        'accepted_for_next_milestone': 'M6',
        'decision': 'ACCEPT_M5_ONE_STEP_PACKAGE_FOR_M6',
        'package_manifest_hash': package['manifest']['package_manifest_hash'],
    }
    reports = run_root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    acceptance_path = reports / 'M5_acceptance.json'
    atomic_write_json(acceptance_path, acceptance)
    atomic_write_json(project / 'audit' / 'm5_accepted_parent.json', {
        'schema_version': 1,
        'milestone_reviewed': 'M5',
        'accepted_for_next_milestone': 'M6',
        'accepted_run_id': run_id,
        'accepted_phase': M5_COMPLETE,
        'certification_status': acceptance['certification_status'],
        'm5_acceptance_sha256': sha256_file(acceptance_path),
    })
    return project, persistent, run_id
