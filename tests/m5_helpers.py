from __future__ import annotations

import math
from pathlib import Path

from src.common import atomic_write_json, read_json, sha256_file
from src.m4_orchestrator import create_or_resume_m4
from src.m4_status import m4_bound_handoff
from tests.m4_helpers import make_synthetic_accepted_m3, passing_m4_test_report


def _minimum_centered_fd_order(difference: dict) -> float:
    channels = difference['channels']
    minimum_order = math.inf
    for channel in channels.values():
        steps = channel['steps']
        for earlier, later in zip(steps, steps[1:]):
            earlier_h = float(earlier['step'])
            later_h = float(later['step'])
            earlier_e = float(earlier['relative_error_frobenius'])
            later_e = float(later['relative_error_frobenius'])
            order = math.log(later_e / earlier_e) / math.log(later_h / earlier_h)
            minimum_order = min(minimum_order, order)
    return float(minimum_order)


def make_synthetic_accepted_m4(
    tmp_path: Path,
) -> tuple[Path, Path, str]:
    config, project = make_synthetic_accepted_m3(tmp_path)
    persistent = tmp_path / 'persist'
    run_id = 'M4-synthetic-derivative-accepted'
    orchestrator = create_or_resume_m4(
        persistent, config, project, run_id=run_id,
        test_report=passing_m4_test_report(),
    )
    summary = orchestrator.run_until_checkpoint()
    # M4 runtime summary keeps enclosure milestone_status=BLOCKED_MATH while the
    # derivative acceptance decision for M5 is recorded separately in the audit.
    if summary.get('phase') != 'M4_COMPLETE' or summary.get('certification_status') != 'NOT_CERTIFIED':
        raise RuntimeError('Synthetic M4 derivative did not pass its gates.')

    run_root = persistent / 'runs' / run_id
    report_path = run_root / 'reports/M4_report.json'
    acceptance_path = run_root / 'reports/M4_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    checkpoint = run_root / 'checkpoints/ckpt_000014'
    report = read_json(report_path)
    difference = report['results']['M4_FINITE_DIFFERENCE']['result']
    sources = report['results']['M4_SOURCE_CHANNELS']['result']
    atomic_write_json(project / 'audit/m4_accepted_parent.json', {
        'schema_version': 1,
        'milestone_reviewed': 'M4',
        'accepted_for_next_milestone': 'M5',
        'accepted_phase': 'M4_COMPLETE',
        'accepted_run_id': run_id,
        'checkpoint_index': 14,
        'implementation_status': 'M4_IMPLEMENTATION_COMPLETE',
        'milestone_status': 'DERIVATIVE_ACCEPTED',
        'enclosure_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
        'decision': 'ACCEPT_M4_DERIVATIVE_FOR_M5_ONE_STEP_VALIDATION',
        'independent_artifact_reload_performed': True,
        'm4_report_path': str(report_path),
        'm4_report_sha256': sha256_file(report_path),
        'm4_acceptance_path': str(acceptance_path),
        'm4_acceptance_sha256': sha256_file(acceptance_path),
        'manifest_path': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'checkpoint_path': str(checkpoint),
        'checkpoint_hash_manifest_sha256': sha256_file(
            checkpoint / 'hashes.json'
        ),
        'proof_artifact_hashes': report['proof_artifact_hashes'],
        'bound_ledger': m4_bound_handoff(),
        'derivative_regression': {
            'classification': (
                'REPRODUCIBLE_REGRESSION_ACCEPTANCE_NOT_A_DETERMINISTIC_PROOF_BOUND'
            ),
            'all_channels_converged': True,
            'minimum_observed_centered_fd_order': _minimum_centered_fd_order(
                difference
            ),
            'max_final_relative_error': difference['max_final_relative_error'],
            'zero_tangent_residual': sources['zero_source_max_abs'],
            'symmetry_residual': sources['max_symmetry_residual'],
            'finite_difference_is_proof_bound': False,
        },
    })
    return project, persistent, run_id
