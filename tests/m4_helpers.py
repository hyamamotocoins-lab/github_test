from __future__ import annotations

from pathlib import Path

import numpy as np

from src.checkpoint import TensorShardStore
from src.common import atomic_write_json, atomic_write_text, sha256_file, utc_now
from src.m4_config import M4Config
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS
from src.rsvd import array_sha256
from src.work_queue import WorkQueue


def passing_m4_test_report() -> dict[str, object]:
    return {
        'accepted_m3_parent': 'PASS',
        'm0_m1_m2_m3_regression_cpu_suite': 'PASS',
        'm4_required_cpu_suite': 'PASS',
        'm4_required_gpu_suite': 'PASS',
        'm4_fresh_process_resume': 'PASS',
        'm4_derivative_checkpoint_restore': 'PASS',
        'elapsed_s': 1.0,
    }


def make_synthetic_accepted_m3(
    tmp_path: Path, base: M4Config | None = None,
) -> tuple[M4Config, Path]:
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'audit').mkdir(parents=True)
    (project / 'notebooks').mkdir(parents=True)
    atomic_write_json(project / 'notebooks/50_m4_derivatives.ipynb', {
        'cells': [{
            'cell_type': 'markdown', 'metadata': {},
            'source': ['# synthetic M4 notebook\n'],
        }],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
    })
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, f'synthetic fixture: {relative}\n')

    rng = np.random.default_rng(20260720)
    left, _ = np.linalg.qr(rng.standard_normal((729, 16)))
    right_basis, _ = np.linalg.qr(rng.standard_normal((729, 16)))
    singular_values = np.linspace(1.0, 0.25, 16)
    right = right_basis.T
    core = np.diag(singular_values)
    tensors = {
        'rsvd_left': left, 'rsvd_singular_values': singular_values,
        'rsvd_right_t': right, 'triad_left': left.copy(),
        'triad_core': core, 'triad_right': right.copy(),
    }

    parent_run_id = 'M3-synthetic-accepted'
    parent_run = tmp_path / 'parent' / 'runs' / parent_run_id
    checkpoint = parent_run / 'checkpoints/ckpt_000014'
    (parent_run / 'artifacts').mkdir(parents=True)
    (parent_run / 'work_items').mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    queue = WorkQueue()
    phase_hashes: dict[str, str] = {}
    phases = (
        'M3_BACKEND_DIAGNOSTIC', 'M3_OPERATOR_BUILD',
        'M3_MATRIX_FREE_VALIDATE', 'M3_RSVD', 'M3_TRIAD', 'M3_REPORT',
    )
    for phase in phases:
        item_id = queue.add(
            phase, '3' * 64, {'milestone': 'M3', 'phase': phase}, 1.0,
        )
        item = queue.items[item_id]
        result_path = parent_run / 'artifacts' / f'{item_id}.json'
        atomic_write_json(result_path, {
            'milestone': 'M3', 'phase': phase,
            'certification_status': 'NOT_CERTIFIED',
        })
        digest = sha256_file(result_path)
        relative = result_path.relative_to(parent_run).as_posix()
        item.status = 'done'; item.attempts = 1
        item.result_relpath = relative; item.result_sha256 = digest
        phase_hashes[phase] = digest
        atomic_write_json(parent_run / 'work_items' / f'{item_id}.done', {
            'item_id': item_id, 'result_relpath': relative,
            'result_sha256': digest,
        })

    reports = parent_run / 'reports'; reports.mkdir()
    report_path = reports / 'M3_report.json'
    acceptance_path = reports / 'M3_acceptance.json'
    manifest_path = parent_run / 'run_manifest.json'
    rsvd = {
        'basis_sha256': array_sha256(left),
        'singular_values_sha256': array_sha256(singular_values),
        'right_sha256': array_sha256(right),
        'residual_frobenius': 0.125,
        'relative_residual_frobenius': 0.1,
        'influence_proxy': {
            'value': 1.0, 'screening': 'INVESTIGATE_CUTOFF_AND_RANK',
            'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        },
    }
    triad = {
        'left_sha256': array_sha256(left),
        'core_sha256': array_sha256(core),
        'right_sha256': array_sha256(right),
    }
    atomic_write_json(report_path, {
        'milestone': 'M3', 'run_id': parent_run_id,
        'phase': 'M3_COMPLETE', 'milestone_status': 'CORE_REPRODUCED',
        'certification_status': 'NOT_CERTIFIED',
        'acceptance_gates': {'synthetic_fixture': True},
        'rigorous_bounds': [], 'proof_artifact_hashes': phase_hashes,
        'results': {
            'M3_RSVD': {'result': rsvd},
            'M3_TRIAD': {'result': triad},
        },
        'memory': {'gpu_peak_allocated_bytes': 0},
    })
    atomic_write_json(acceptance_path, {
        'milestone': 'M3', 'phase': 'M3_COMPLETE', 'status': 'PASS',
        'milestone_status': 'CORE_REPRODUCED',
        'certification_status': 'NOT_CERTIFIED',
        'gates': {'synthetic_fixture': True},
    })
    atomic_write_json(manifest_path, {
        'milestone': 'M3', 'run_id': parent_run_id,
        'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(checkpoint / 'state.json', {
        'run_id': parent_run_id, 'phase': 'M3_COMPLETE',
        'checkpoint_index': 14, 'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(checkpoint / 'bounds.json', {})
    atomic_write_json(checkpoint / 'work_queue.json', queue.to_payload())
    atomic_write_json(checkpoint / 'meta.json', {'fixture': True})
    TensorShardStore(64 * 1024 * 1024).save(checkpoint / 'tensors', tensors)
    hashes = {
        path.relative_to(checkpoint).as_posix(): sha256_file(path)
        for path in checkpoint.rglob('*') if path.is_file()
    }
    atomic_write_json(checkpoint / 'hashes.json', hashes)
    atomic_write_text(checkpoint / 'COMMITTED', utc_now())

    overrides = {
        'parent_run_id': parent_run_id,
        'parent_checkpoint_path': str(checkpoint),
        'parent_report_path': str(report_path),
        'parent_acceptance_path': str(acceptance_path),
        'require_cuda': False,
    }
    config = M4Config(**(
        overrides if base is None
        else {**base.canonical_payload(), **overrides}
    ))
    atomic_write_json(project / config.parent_audit_path, {
        'schema_version': 1, 'milestone_reviewed': 'M3',
        'accepted_for_next_milestone': 'M4',
        'accepted_phase': 'M3_COMPLETE', 'accepted_run_id': parent_run_id,
        'checkpoint_index': 14,
        'decision': 'ACCEPT_M3_FOR_M4_FORWARD_DERIVATIVE_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm3_report_path': str(report_path),
        'm3_report_sha256': sha256_file(report_path),
        'm3_acceptance_path': str(acceptance_path),
        'm3_acceptance_sha256': sha256_file(acceptance_path),
        'checkpoint_path': str(checkpoint),
        'checkpoint_hash_manifest_sha256': sha256_file(checkpoint / 'hashes.json'),
        'manifest_path': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'proof_artifact_hashes': phase_hashes,
    })
    return config, project
