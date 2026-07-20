from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .armillary import (
    all_link_star_keys, build_armillary_sector, checkpoint_tensor_shards,
)
from .checkpoint import TensorShardStore
from .common import read_json, sha256_file
from .m3_config import M3Config
from .work_queue import WorkQueue


class M3ParentError(RuntimeError):
    '''Raised when the accepted M2 parent changes or cannot seed M3.'''


@dataclass(frozen=True, slots=True)
class M3ParentEvidence:
    hashes: dict[str, str]
    projector_tensors: dict[str, np.ndarray]


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in '0123456789abcdef' for character in value
    ):
        raise M3ParentError(f'{label} is not a SHA-256 digest.')
    return value


def _verify_file(path: Path, digest: object, label: str) -> str:
    expected = _sha(digest, f'{label} expected digest')
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise M3ParentError(f'{label} is missing, unsafe, or changed: {path}')
    return expected


def _all_true(value: object) -> bool:
    return (
        isinstance(value, dict) and bool(value)
        and all(item is True for item in value.values())
    )


def _verify_checkpoint(checkpoint: Path) -> str:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise M3ParentError('Accepted M2 checkpoint is missing or unsafe.')
    if not (checkpoint / 'COMMITTED').is_file():
        raise M3ParentError('Accepted M2 checkpoint is not committed.')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise M3ParentError('Accepted M2 checkpoint contains a symlink.')
    hashes_path = checkpoint / 'hashes.json'
    expected = read_json(hashes_path) if hashes_path.is_file() else None
    if not isinstance(expected, dict):
        raise M3ParentError('Accepted M2 checkpoint hash manifest is malformed.')
    normalized: dict[str, str] = {}
    for relative, digest in expected.items():
        if not isinstance(relative, str):
            raise M3ParentError('Accepted M2 checkpoint path is not a string.')
        normalized[relative] = _sha(digest, f'checkpoint entry {relative}')
    actual = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    if actual != set(normalized):
        raise M3ParentError('Accepted M2 checkpoint file set changed.')
    for relative, digest in normalized.items():
        if sha256_file(checkpoint / relative) != digest:
            raise M3ParentError(f'Accepted M2 checkpoint hash mismatch: {relative}')
    return sha256_file(hashes_path)


def _verify_queue(checkpoint: Path, config: M3Config) -> WorkQueue:
    state = read_json(checkpoint / 'state.json')
    if not isinstance(state, dict) or any((
        state.get('run_id') != config.parent_run_id,
        state.get('phase') != 'M2_COMPLETE',
        state.get('checkpoint_index') != 14,
        state.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M3ParentError('Accepted M2 checkpoint state is invalid.')
    queue = WorkQueue.from_payload(read_json(checkpoint / 'work_queue.json'))
    if len(queue.items) != 6 or any(item.status != 'done' for item in queue.items.values()):
        raise M3ParentError('Accepted M2 work queue is not complete.')
    run_root = checkpoint.parents[1]
    for item in queue.items.values():
        if not item.result_relpath or not item.result_sha256:
            raise M3ParentError(f'Accepted M2 item lacks metadata: {item.phase}')
        result = (run_root / item.result_relpath).resolve()
        try:
            result.relative_to(run_root.resolve())
        except ValueError as exc:
            raise M3ParentError('Accepted M2 result escapes its run root.') from exc
        marker_path = run_root / 'work_items' / f'{item.item_id}.done'
        marker = read_json(marker_path) if marker_path.is_file() else None
        if not result.is_file() or sha256_file(result) != item.result_sha256:
            raise M3ParentError(f'Accepted M2 proof artifact changed: {item.phase}')
        if not isinstance(marker, dict) or any((
            marker.get('item_id') != item.item_id,
            marker.get('result_relpath') != item.result_relpath,
            marker.get('result_sha256') != item.result_sha256,
        )):
            raise M3ParentError(f'Accepted M2 done marker changed: {item.phase}')
    return queue


def _load_and_crosscheck_tensors(checkpoint: Path) -> dict[str, np.ndarray]:
    loaded = TensorShardStore(16 * 1024 * 1024).load(checkpoint / 'tensors')
    expected = checkpoint_tensor_shards(
        build_armillary_sector(key) for key in all_link_star_keys()
    )
    if set(loaded) != set(expected) or len(loaded) != 64:
        raise M3ParentError('Accepted M2 projector tensor set changed.')
    result: dict[str, np.ndarray] = {}
    for name in sorted(expected):
        actual = np.asarray(loaded[name], dtype=np.float64)
        if not np.array_equal(actual, expected[name]):
            raise M3ParentError(
                f'Accepted M2 float tensor disagrees with exact reconstruction: {name}'
            )
        result[name] = actual.copy()
    return result


def verify_accepted_m2_parent(
    project_root: Path, config: M3Config,
) -> M3ParentEvidence:
    audit_path = project_root / config.parent_audit_path
    if audit_path.is_symlink() or not audit_path.is_file():
        raise M3ParentError('M2 acceptance audit is missing or unsafe.')
    audit = read_json(audit_path)
    expected_audit: dict[str, Any] = {
        'milestone_reviewed': 'M2',
        'accepted_for_next_milestone': 'M3',
        'accepted_phase': 'M2_COMPLETE',
        'accepted_run_id': config.parent_run_id,
        'checkpoint_index': 14,
        'decision': 'ACCEPT_M2_FOR_M3_EXPLORATORY_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
    }
    if not isinstance(audit, dict) or any(
        audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise M3ParentError('M2 acceptance audit identity or decision is invalid.')

    report_path = Path(config.parent_report_path).resolve()
    acceptance_path = Path(config.parent_acceptance_path).resolve()
    checkpoint = Path(config.parent_checkpoint_path).resolve()
    manifest_path = checkpoint.parents[1] / 'run_manifest.json'
    for key, path in {
        'm2_report_path': report_path,
        'm2_acceptance_path': acceptance_path,
        'checkpoint_path': checkpoint,
        'manifest_path': manifest_path,
    }.items():
        audited = audit.get(key)
        if not isinstance(audited, str) or Path(audited).resolve() != path:
            raise M3ParentError(f'Accepted M2 path changed: {key}')
    if checkpoint.name != config.parent_checkpoint:
        raise M3ParentError('Accepted M2 checkpoint name changed.')

    report_hash = _verify_file(
        report_path, audit.get('m2_report_sha256'), 'accepted M2 report',
    )
    acceptance_hash = _verify_file(
        acceptance_path, audit.get('m2_acceptance_sha256'),
        'accepted M2 acceptance',
    )
    manifest_hash = _verify_file(
        manifest_path, audit.get('manifest_sha256'), 'accepted M2 manifest',
    )
    checkpoint_hash = _verify_checkpoint(checkpoint)
    if checkpoint_hash != audit.get('checkpoint_hash_manifest_sha256'):
        raise M3ParentError('Accepted M2 checkpoint hash manifest changed.')
    queue = _verify_queue(checkpoint, config)

    report = read_json(report_path)
    if not isinstance(report, dict) or any((
        report.get('run_id') != config.parent_run_id,
        report.get('phase') != 'M2_COMPLETE',
        report.get('certification_status') != 'NOT_CERTIFIED',
        report.get('heuristic_results') != [],
        not _all_true(report.get('acceptance_gates')),
        report.get('proof_artifact_hashes') != audit.get('proof_artifact_hashes'),
    )):
        raise M3ParentError('Accepted M2 report no longer satisfies every gate.')
    if report.get('proof_artifact_hashes') != {
        item.phase: item.result_sha256 for item in queue.items.values()
    }:
        raise M3ParentError('Accepted M2 report hashes differ from its queue.')
    equivalence = (
        report.get('results', {}).get('M2_EQUIVALENCE', {}).get('result', {})
    )
    if (
        equivalence.get('exact_match_count') != 64
        or equivalence.get('mismatches') != []
    ):
        raise M3ParentError('Accepted M2 exact equivalence result changed.')

    acceptance = read_json(acceptance_path)
    if not isinstance(acceptance, dict) or any((
        acceptance.get('milestone') != 'M2',
        acceptance.get('phase') != 'M2_COMPLETE',
        acceptance.get('status') != 'PASS',
        acceptance.get('certification_status') != 'NOT_CERTIFIED',
        not _all_true(acceptance.get('gates')),
    )):
        raise M3ParentError('Accepted M2 acceptance gates changed or failed.')
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or any((
        manifest.get('milestone') != 'M2',
        manifest.get('run_id') != config.parent_run_id,
        manifest.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M3ParentError('Accepted M2 manifest identity changed.')

    tensors = _load_and_crosscheck_tensors(checkpoint)
    return M3ParentEvidence({
        'm2_audit_sha256': sha256_file(audit_path),
        'm2_report_sha256': report_hash,
        'm2_acceptance_sha256': acceptance_hash,
        'm2_manifest_sha256': manifest_hash,
        'parent_checkpoint_hash_manifest_sha256': checkpoint_hash,
    }, tensors)
