from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from src.armillary import (
    all_link_star_keys, build_armillary_sector, checkpoint_tensor_shards,
)
from src.checkpoint import TensorShardStore
from src.common import atomic_write_json, atomic_write_text, sha256_file, utc_now
from src.m3_config import M3Config
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS
from src.work_queue import WorkQueue


@lru_cache(maxsize=1)
def m2_projector_fixture() -> dict[str, np.ndarray]:
    return checkpoint_tensor_shards(
        build_armillary_sector(key) for key in all_link_star_keys()
    )


def passing_m3_test_report() -> dict[str, object]:
    return {
        'accepted_m2_parent': 'PASS',
        'm0_m1_m2_regression_cpu_suite': 'PASS',
        'm3_required_cpu_suite': 'PASS',
        'm3_required_gpu_suite': 'PASS',
        'm3_fresh_process_resume': 'PASS',
        'm3_checkpoint_basis_restore': 'PASS',
        'm3_oom_recovery': 'PASS',
        'elapsed_s': 1.0,
    }


def make_synthetic_accepted_m2(
    tmp_path: Path, base: M3Config | None = None,
) -> tuple[M3Config, Path]:
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'audit').mkdir(parents=True)
    (project / 'notebooks').mkdir(parents=True)
    atomic_write_json(project / 'notebooks/40_m3_gpu_triad_atrg.ipynb', {
        'cells': [{
            'cell_type': 'markdown', 'metadata': {},
            'source': ['# synthetic M3 notebook\n'],
        }],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
    })
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, f'synthetic fixture: {relative}\n')

    parent_run_id = 'M2-synthetic-accepted'
    parent_run = tmp_path / 'parent' / 'runs' / parent_run_id
    checkpoint = parent_run / 'checkpoints/ckpt_000014'
    (parent_run / 'artifacts').mkdir(parents=True)
    (parent_run / 'work_items').mkdir(parents=True)
    checkpoint.mkdir(parents=True)
    queue = WorkQueue()
    phase_hashes: dict[str, str] = {}
    phases = (
        'M2_WIGNER_CACHE', 'M2_DENSE_REFERENCE', 'M2_ARMILLARY',
        'M2_EQUIVALENCE', 'M2_SYMMETRY', 'M2_REPORT',
    )
    for phase in phases:
        item_id = queue.add(
            phase, '2' * 64, {'milestone': 'M2', 'phase': phase}, 1.0,
        )
        item = queue.items[item_id]
        result_path = parent_run / 'artifacts' / f'{item_id}.json'
        atomic_write_json(result_path, {
            'milestone': 'M2', 'phase': phase,
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
    report_path = reports / 'M2_report.json'
    acceptance_path = reports / 'M2_acceptance.json'
    manifest_path = parent_run / 'run_manifest.json'
    atomic_write_json(report_path, {
        'milestone': 'M2', 'run_id': parent_run_id,
        'phase': 'M2_COMPLETE', 'certification_status': 'NOT_CERTIFIED',
        'acceptance_gates': {'synthetic_fixture': True},
        'heuristic_results': [], 'proof_artifact_hashes': phase_hashes,
        'results': {
            'M2_EQUIVALENCE': {
                'result': {'exact_match_count': 64, 'mismatches': []},
            },
        },
    })
    atomic_write_json(acceptance_path, {
        'milestone': 'M2', 'phase': 'M2_COMPLETE', 'status': 'PASS',
        'certification_status': 'NOT_CERTIFIED',
        'gates': {'synthetic_fixture': True},
    })
    atomic_write_json(manifest_path, {
        'milestone': 'M2', 'run_id': parent_run_id,
        'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(checkpoint / 'state.json', {
        'run_id': parent_run_id, 'phase': 'M2_COMPLETE',
        'checkpoint_index': 14, 'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(checkpoint / 'bounds.json', {})
    atomic_write_json(checkpoint / 'work_queue.json', queue.to_payload())
    atomic_write_json(checkpoint / 'meta.json', {'fixture': True})
    TensorShardStore(16 * 1024 * 1024).save(
        checkpoint / 'tensors', m2_projector_fixture(),
    )
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
    }
    if base is None:
        config = M3Config(**overrides)
    else:
        config = M3Config(**{**base.canonical_payload(), **overrides})
    atomic_write_json(project / config.parent_audit_path, {
        'schema_version': 1, 'milestone_reviewed': 'M2',
        'accepted_for_next_milestone': 'M3',
        'accepted_phase': 'M2_COMPLETE', 'accepted_run_id': parent_run_id,
        'checkpoint_index': 14,
        'decision': 'ACCEPT_M2_FOR_M3_EXPLORATORY_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm2_report_path': str(report_path),
        'm2_report_sha256': sha256_file(report_path),
        'm2_acceptance_path': str(acceptance_path),
        'm2_acceptance_sha256': sha256_file(acceptance_path),
        'checkpoint_path': str(checkpoint),
        'checkpoint_hash_manifest_sha256': sha256_file(checkpoint / 'hashes.json'),
        'manifest_path': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'proof_artifact_hashes': phase_hashes,
    })
    return config, project
