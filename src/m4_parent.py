from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import TensorShardStore
from .common import read_json, sha256_file
from .m4_config import M4Config
from .work_queue import WorkQueue


class M4ParentError(RuntimeError):
    '''Raised when the reviewed M3 parent is missing or changed.'''


@dataclass(frozen=True, slots=True)
class M4ParentEvidence:
    hashes: dict[str, str]
    tensors: dict[str, np.ndarray]
    metrics: dict[str, Any]


def _digest(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in '0123456789abcdef' for character in value
    ):
        raise M4ParentError(f'{label} is not a SHA-256 digest.')
    return value


def _verify_file(path: Path, expected: object, label: str) -> str:
    digest = _digest(expected, f'{label} expected digest')
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise M4ParentError(f'{label} is missing, unsafe, or changed: {path}')
    return digest


def _all_true(value: object) -> bool:
    return (
        isinstance(value, dict) and bool(value)
        and all(item is True for item in value.values())
    )


def _verify_checkpoint(checkpoint: Path) -> str:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise M4ParentError('Accepted M3 checkpoint is missing or unsafe.')
    if not (checkpoint / 'COMMITTED').is_file():
        raise M4ParentError('Accepted M3 checkpoint is not committed.')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise M4ParentError('Accepted M3 checkpoint contains a symlink.')
    hashes_path = checkpoint / 'hashes.json'
    hashes = read_json(hashes_path) if hashes_path.is_file() else None
    if not isinstance(hashes, dict):
        raise M4ParentError('Accepted M3 checkpoint manifest is malformed.')
    normalized = {
        relative: _digest(digest, f'checkpoint entry {relative}')
        for relative, digest in hashes.items()
        if isinstance(relative, str)
    }
    if len(normalized) != len(hashes):
        raise M4ParentError('Accepted M3 checkpoint path is malformed.')
    actual = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    if actual != set(normalized):
        raise M4ParentError('Accepted M3 checkpoint file set changed.')
    for relative, digest in normalized.items():
        if sha256_file(checkpoint / relative) != digest:
            raise M4ParentError(f'Accepted M3 checkpoint hash mismatch: {relative}')
    return sha256_file(hashes_path)


def _verify_queue(
    checkpoint: Path, config: M4Config, *, checkpoint_index: int,
) -> WorkQueue:
    state = read_json(checkpoint / 'state.json')
    if not isinstance(state, dict) or any((
        state.get('run_id') != config.parent_run_id,
        state.get('phase') != 'M3_COMPLETE',
        state.get('checkpoint_index') != checkpoint_index,
        state.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M4ParentError('Accepted M3 checkpoint state is invalid.')
    queue = WorkQueue.from_payload(read_json(checkpoint / 'work_queue.json'))
    if len(queue.items) != 6 or any(item.status != 'done' for item in queue.items.values()):
        raise M4ParentError('Accepted M3 work queue is not complete.')
    run_root = checkpoint.parents[1]
    for item in queue.items.values():
        if not item.result_relpath or not item.result_sha256:
            raise M4ParentError(f'Accepted M3 item lacks metadata: {item.phase}')
        result = (run_root / item.result_relpath).resolve()
        try:
            result.relative_to(run_root.resolve())
        except ValueError as exc:
            raise M4ParentError('Accepted M3 result escapes its run root.') from exc
        marker_path = run_root / 'work_items' / f'{item.item_id}.done'
        marker = read_json(marker_path) if marker_path.is_file() else None
        if not result.is_file() or sha256_file(result) != item.result_sha256:
            raise M4ParentError(f'Accepted M3 result changed: {item.phase}')
        if not isinstance(marker, dict) or any((
            marker.get('item_id') != item.item_id,
            marker.get('result_relpath') != item.result_relpath,
            marker.get('result_sha256') != item.result_sha256,
        )):
            raise M4ParentError(f'Accepted M3 marker changed: {item.phase}')
    return queue


def _array_sha256(value: np.ndarray) -> str:
    contiguous = np.asarray(value, dtype='<f8', order='C')
    payload = (
        str(contiguous.shape).encode('ascii')
        + b'\0' + contiguous.tobytes(order='C')
    )
    return hashlib.sha256(payload).hexdigest()


def _load_tensors(
    checkpoint: Path,
    report: dict[str, Any],
    *,
    projected_rank: int,
    operator_dimension: int = 729,
) -> dict[str, np.ndarray]:
    loaded = TensorShardStore(64 * 1024 * 1024).load(checkpoint / 'tensors')
    rank = int(projected_rank)
    dim = int(operator_dimension)
    expected_shapes = {
        'rsvd_left': (dim, rank),
        'rsvd_singular_values': (rank,),
        'rsvd_right_t': (rank, dim),
        'triad_left': (dim, rank),
        'triad_core': (rank, rank),
        'triad_right': (rank, dim),
    }
    if set(loaded) != set(expected_shapes):
        raise M4ParentError('Accepted M3 tensor set changed.')
    tensors: dict[str, np.ndarray] = {}
    for name, shape in expected_shapes.items():
        value = np.asarray(loaded[name], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise M4ParentError(f'Accepted M3 tensor is invalid: {name}')
        tensors[name] = value.copy()
    if (
        not np.array_equal(tensors['triad_left'], tensors['rsvd_left'])
        or not np.array_equal(tensors['triad_right'], tensors['rsvd_right_t'])
        or not np.array_equal(
            np.diag(tensors['rsvd_singular_values']), tensors['triad_core']
        )
    ):
        raise M4ParentError('Accepted M3 Triad/RSVD factors disagree.')
    rsvd = report['results']['M3_RSVD']['result']
    triad = report['results']['M3_TRIAD']['result']
    expected_hashes = {
        'rsvd_left': rsvd['basis_sha256'],
        'rsvd_singular_values': rsvd['singular_values_sha256'],
        'rsvd_right_t': rsvd['right_sha256'],
        'triad_left': triad['left_sha256'],
        'triad_core': triad['core_sha256'],
        'triad_right': triad['right_sha256'],
    }
    for name, expected in expected_hashes.items():
        if _array_sha256(tensors[name]) != expected:
            raise M4ParentError(f'Accepted M3 tensor/report hash differs: {name}')
    return tensors


def verify_accepted_m3_parent(
    project_root: Path, config: M4Config,
) -> M4ParentEvidence:
    audit_path = project_root / config.parent_audit_path
    if audit_path.is_symlink() or not audit_path.is_file():
        raise M4ParentError('M3 acceptance audit is missing or unsafe.')
    audit = read_json(audit_path)
    expected_audit: dict[str, Any] = {
        'milestone_reviewed': 'M3',
        'accepted_for_next_milestone': 'M4',
        'accepted_phase': 'M3_COMPLETE',
        'accepted_run_id': config.parent_run_id,
        'decision': 'ACCEPT_M3_FOR_M4_FORWARD_DERIVATIVE_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
    }
    if not isinstance(audit, dict) or any(
        audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise M4ParentError('M3 acceptance audit identity or decision is invalid.')
    if (
        not isinstance(audit.get('checkpoint_index'), int)
        or isinstance(audit.get('checkpoint_index'), bool)
        or audit['checkpoint_index'] < 1
    ):
        raise M4ParentError('M3 acceptance audit checkpoint index is invalid.')
    report_path = Path(config.parent_report_path).resolve()
    acceptance_path = Path(config.parent_acceptance_path).resolve()
    checkpoint = Path(config.parent_checkpoint_path).resolve()
    manifest_path = checkpoint.parents[1] / 'run_manifest.json'
    path_keys = {
        'm3_report_path': report_path,
        'm3_acceptance_path': acceptance_path,
        'checkpoint_path': checkpoint,
    }
    if 'manifest_path' in audit:
        path_keys['manifest_path'] = manifest_path
    for key, path in path_keys.items():
        audited = audit.get(key)
        if not isinstance(audited, str) or Path(audited).resolve() != path:
            raise M4ParentError(f'Accepted M3 path changed: {key}')
    if checkpoint.name != config.parent_checkpoint:
        raise M4ParentError('Accepted M3 checkpoint name changed.')
    report_hash = _verify_file(
        report_path, audit.get('m3_report_sha256'), 'accepted M3 report',
    )
    acceptance_hash = _verify_file(
        acceptance_path, audit.get('m3_acceptance_sha256'),
        'accepted M3 acceptance',
    )
    manifest_hash = _verify_file(
        manifest_path, audit.get('manifest_sha256'), 'accepted M3 manifest',
    )
    checkpoint_hash = _verify_checkpoint(checkpoint)
    if checkpoint_hash != audit.get('checkpoint_hash_manifest_sha256'):
        raise M4ParentError('Accepted M3 checkpoint manifest changed.')
    queue = _verify_queue(
        checkpoint, config, checkpoint_index=int(audit['checkpoint_index']),
    )
    report = read_json(report_path)
    if not isinstance(report, dict) or any((
        report.get('run_id') != config.parent_run_id,
        report.get('phase') != 'M3_COMPLETE',
        report.get('milestone_status') != 'CORE_REPRODUCED',
        report.get('certification_status') != 'NOT_CERTIFIED',
        report.get('rigorous_bounds') != [],
        not _all_true(report.get('acceptance_gates')),
        report.get('proof_artifact_hashes') != audit.get('proof_artifact_hashes'),
    )):
        raise M4ParentError('Accepted M3 report no longer satisfies every gate.')
    if report.get('proof_artifact_hashes') != {
        item.phase: item.result_sha256 for item in queue.items.values()
    }:
        raise M4ParentError('Accepted M3 report hashes differ from its queue.')
    acceptance = read_json(acceptance_path)
    if not isinstance(acceptance, dict) or any((
        acceptance.get('milestone') != 'M3',
        acceptance.get('phase') != 'M3_COMPLETE',
        acceptance.get('milestone_status') != 'CORE_REPRODUCED',
        acceptance.get('status') != 'PASS',
        acceptance.get('certification_status') != 'NOT_CERTIFIED',
        not _all_true(acceptance.get('gates')),
    )):
        raise M4ParentError('Accepted M3 acceptance gates changed or failed.')
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or any((
        manifest.get('milestone') != 'M3',
        manifest.get('run_id') != config.parent_run_id,
        manifest.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M4ParentError('Accepted M3 manifest identity changed.')
    tensors = _load_tensors(
        checkpoint, report,
        projected_rank=config.projected_rank,
        operator_dimension=config.operator_dimension,
    )
    rsvd = report['results']['M3_RSVD']['result']
    return M4ParentEvidence(
        {
            'm3_audit_sha256': sha256_file(audit_path),
            'm3_report_sha256': report_hash,
            'm3_acceptance_sha256': acceptance_hash,
            'm3_manifest_sha256': manifest_hash,
            'parent_checkpoint_hash_manifest_sha256': checkpoint_hash,
        },
        tensors,
        {
            'rsvd_residual_frobenius': rsvd['residual_frobenius'],
            'rsvd_relative_residual_frobenius': (
                rsvd['relative_residual_frobenius']
            ),
            'influence_proxy': rsvd['influence_proxy'],
            'screening': rsvd['influence_proxy']['screening'],
            'parent_gpu_peak_allocated_bytes': (
                report['memory']['gpu_peak_allocated_bytes']
            ),
        },
    )
