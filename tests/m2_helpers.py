from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json, atomic_write_text, sha256_file, utc_now
from src.m2_config import M2Config
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS
from src.work_queue import WorkQueue


def passing_m2_test_report() -> dict[str, object]:
    return {
        'accepted_m1_parent': 'PASS',
        'm0_m1_regression_cpu_suite': 'PASS',
        'm2_required_cpu_suite': 'PASS',
        'm2_fresh_process_resume': 'PASS',
        'optional_gpu_suite': 'NOT_RUN_NO_CUDA',
        'elapsed_s': 1.0,
    }


def make_synthetic_accepted_m1(
    tmp_path: Path, base: M2Config | None = None,
) -> tuple[M2Config, Path]:
    project = tmp_path / 'project'
    (project / 'src').mkdir(parents=True)
    (project / 'audit').mkdir(parents=True)
    (project / 'notebooks').mkdir(parents=True)
    atomic_write_json(project / 'notebooks' / '30_m2_armillary.ipynb', {
        'cells': [{
            'cell_type': 'markdown', 'metadata': {},
            'source': ['# synthetic M2 notebook\n'],
        }],
        'metadata': {}, 'nbformat': 4, 'nbformat_minor': 5,
    })
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(path, f'synthetic fixture: {relative}\n')

    parent_run_id = 'M1-synthetic-accepted'
    parent_run = tmp_path / 'parent' / 'runs' / parent_run_id
    checkpoint = parent_run / 'checkpoints' / 'ckpt_000014'
    (parent_run / 'artifacts').mkdir(parents=True)
    (parent_run / 'work_items').mkdir(parents=True)
    checkpoint.mkdir(parents=True)

    queue = WorkQueue()
    phase_hashes: dict[str, str] = {}
    for phase in (
        'M1_COEFFICIENT_BATCH', 'M1_VALUE_TAIL', 'M1_GRADIENT_TAIL',
        'M1_RG_TRAJECTORY', 'M1_INDEPENDENT_VERIFY', 'M1_REPORT',
    ):
        item_id = queue.add(
            phase, '1' * 64, {'milestone': 'M1', 'phase': phase}, 1.0,
        )
        item = queue.items[item_id]
        result_path = parent_run / 'artifacts' / f'{item_id}.json'
        atomic_write_json(result_path, {
            'milestone': 'M1', 'phase': phase, 'status': 'PASS',
        })
        digest = sha256_file(result_path)
        relative = result_path.relative_to(parent_run).as_posix()
        item.status = 'done'
        item.attempts = 1
        item.result_relpath = relative
        item.result_sha256 = digest
        phase_hashes[phase] = digest
        atomic_write_json(parent_run / 'work_items' / f'{item_id}.done', {
            'item_id': item_id, 'result_relpath': relative,
            'result_sha256': digest,
        })

    reports = parent_run / 'reports'
    reports.mkdir()
    report_path = reports / 'M1_report.json'
    acceptance_path = reports / 'M1_acceptance.json'
    manifest_path = parent_run / 'run_manifest.json'
    atomic_write_json(report_path, {
        'milestone': 'M1', 'run_id': parent_run_id,
        'phase': 'M1_COMPLETE', 'certification_status': 'NOT_CERTIFIED',
        'acceptance_gates': {'synthetic_fixture': True},
        'heuristic_results': [], 'proof_artifact_hashes': phase_hashes,
    })
    atomic_write_json(acceptance_path, {
        'milestone': 'M1', 'phase': 'M1_COMPLETE', 'status': 'PASS',
        'certification_status': 'NOT_CERTIFIED',
        'gates': {'synthetic_fixture': True},
    })
    atomic_write_json(manifest_path, {
        'milestone': 'M1', 'run_id': parent_run_id,
        'certification_status': 'NOT_CERTIFIED',
    })

    atomic_write_json(checkpoint / 'state.json', {
        'run_id': parent_run_id, 'phase': 'M1_COMPLETE',
        'checkpoint_index': 14, 'certification_status': 'NOT_CERTIFIED',
    })
    atomic_write_json(checkpoint / 'bounds.json', {})
    atomic_write_json(checkpoint / 'work_queue.json', queue.to_payload())
    atomic_write_json(checkpoint / 'meta.json', {'fixture': True})
    atomic_write_json(checkpoint / 'tensors.json', {})
    hashes = {
        path.relative_to(checkpoint).as_posix(): sha256_file(path)
        for path in checkpoint.rglob('*') if path.is_file()
    }
    atomic_write_json(checkpoint / 'hashes.json', hashes)
    atomic_write_text(checkpoint / 'COMMITTED', utc_now())

    config = M2Config(
        parent_run_id=parent_run_id,
        parent_checkpoint_path=str(checkpoint),
        parent_report_path=str(report_path),
        parent_acceptance_path=str(acceptance_path),
    ) if base is None else M2Config(
        **{
            **base.canonical_payload(),
            'orientations': base.orientations,
            'parent_run_id': parent_run_id,
            'parent_checkpoint_path': str(checkpoint),
            'parent_report_path': str(report_path),
            'parent_acceptance_path': str(acceptance_path),
        }
    )
    audit_path = project / config.parent_audit_path
    atomic_write_json(audit_path, {
        'schema_version': 1, 'milestone_reviewed': 'M1',
        'accepted_for_next_milestone': 'M2',
        'accepted_phase': 'M1_COMPLETE', 'accepted_run_id': parent_run_id,
        'checkpoint_index': 14,
        'decision': 'ACCEPT_M1_FOR_M2_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm1_report_path': str(report_path),
        'm1_report_sha256': sha256_file(report_path),
        'm1_acceptance_path': str(acceptance_path),
        'm1_acceptance_sha256': sha256_file(acceptance_path),
        'checkpoint_path': str(checkpoint),
        'checkpoint_hash_manifest_sha256': sha256_file(checkpoint / 'hashes.json'),
        'manifest_path': str(manifest_path),
        'manifest_sha256': sha256_file(manifest_path),
        'proof_artifact_hashes': phase_hashes,
    })
    return config, project
