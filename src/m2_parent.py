from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import read_json, sha256_file
from .m2_config import M2Config
from .work_queue import WorkQueue


class M2ParentError(RuntimeError):
    '''Raised when the independently accepted M1 parent changes or is incomplete.'''


def _require_sha256(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in '0123456789abcdef' for character in value
    ):
        raise M2ParentError(f'{label} is not a SHA-256 digest.')
    return value


def _verify_file(path: Path, expected_digest: object, label: str) -> str:
    digest = _require_sha256(expected_digest, f'{label} expected digest')
    if path.is_symlink() or not path.is_file() or sha256_file(path) != digest:
        raise M2ParentError(f'{label} is missing, unsafe, or changed: {path}')
    return digest


def _verify_checkpoint(checkpoint: Path) -> str:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise M2ParentError(f'Accepted M1 checkpoint is unavailable or unsafe: {checkpoint}')
    if not (checkpoint / 'COMMITTED').is_file() or not (checkpoint / 'hashes.json').is_file():
        raise M2ParentError('Accepted M1 checkpoint is not committed or lacks hashes.json.')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise M2ParentError('Accepted M1 checkpoint contains a symlink.')
    expected = read_json(checkpoint / 'hashes.json')
    if not isinstance(expected, dict):
        raise M2ParentError('Accepted M1 checkpoint hash manifest is malformed.')
    normalized: dict[str, str] = {}
    for relative, digest in expected.items():
        if not isinstance(relative, str):
            raise M2ParentError('Accepted M1 checkpoint contains a non-string path.')
        normalized[relative] = _require_sha256(digest, f'checkpoint entry {relative}')
    actual_files = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    if actual_files != set(normalized):
        raise M2ParentError('Accepted M1 checkpoint file set differs from hashes.json.')
    for relative, digest in normalized.items():
        if sha256_file(checkpoint / relative) != digest:
            raise M2ParentError(f'Accepted M1 checkpoint hash mismatch: {relative}')
    return sha256_file(checkpoint / 'hashes.json')


def _verify_work_items(checkpoint: Path, config: M2Config) -> WorkQueue:
    state = read_json(checkpoint / 'state.json')
    if not isinstance(state, dict) or any((
        state.get('run_id') != config.parent_run_id,
        state.get('phase') != 'M1_COMPLETE',
        state.get('checkpoint_index') != 14,
        state.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M2ParentError('Accepted M1 checkpoint state is not complete and fail-closed.')
    queue = WorkQueue.from_payload(read_json(checkpoint / 'work_queue.json'))
    if len(queue.items) != 6 or any(item.status != 'done' for item in queue.items.values()):
        raise M2ParentError('Accepted M1 work queue is not complete.')
    run_root = checkpoint.parents[1]
    for item in queue.items.values():
        if not item.result_relpath or not item.result_sha256:
            raise M2ParentError(f'Accepted M1 done item lacks metadata: {item.phase}')
        result = (run_root / item.result_relpath).resolve()
        try:
            result.relative_to(run_root.resolve())
        except ValueError as exc:
            raise M2ParentError('Accepted M1 result escapes its run root.') from exc
        marker_path = run_root / 'work_items' / f'{item.item_id}.done'
        marker = read_json(marker_path) if marker_path.is_file() else None
        if not result.is_file() or sha256_file(result) != item.result_sha256:
            raise M2ParentError(f'Accepted M1 proof artifact changed: {item.phase}')
        if not isinstance(marker, dict) or any((
            marker.get('item_id') != item.item_id,
            marker.get('result_relpath') != item.result_relpath,
            marker.get('result_sha256') != item.result_sha256,
        )):
            raise M2ParentError(f'Accepted M1 done marker changed: {item.phase}')
    return queue


def _all_true(mapping: object) -> bool:
    return (
        isinstance(mapping, dict) and bool(mapping)
        and all(value is True for value in mapping.values())
    )


def verify_accepted_m1_parent(
    project_root: Path, config: M2Config,
) -> dict[str, str]:
    audit_path = project_root / config.parent_audit_path
    if audit_path.is_symlink() or not audit_path.is_file():
        raise M2ParentError('M1 acceptance audit is missing or unsafe.')
    audit = read_json(audit_path)
    expected_audit: dict[str, Any] = {
        'milestone_reviewed': 'M1',
        'accepted_for_next_milestone': 'M2',
        'accepted_phase': 'M1_COMPLETE',
        'accepted_run_id': config.parent_run_id,
        'checkpoint_index': 14,
        'decision': 'ACCEPT_M1_FOR_M2_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
    }
    if not isinstance(audit, dict) or any(
        audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise M2ParentError('M1 acceptance audit identity or decision is invalid.')

    report_path = Path(config.parent_report_path).resolve()
    acceptance_path = Path(config.parent_acceptance_path).resolve()
    checkpoint = Path(config.parent_checkpoint_path).resolve()
    manifest_path = checkpoint.parents[1] / 'run_manifest.json'
    audited_paths = {
        'm1_report_path': report_path,
        'm1_acceptance_path': acceptance_path,
        'checkpoint_path': checkpoint,
        'manifest_path': manifest_path,
    }
    for key, path in audited_paths.items():
        audited = audit.get(key)
        if not isinstance(audited, str) or Path(audited).resolve() != path:
            raise M2ParentError(f'Accepted M1 path changed: {key}')
    if checkpoint.name != config.parent_checkpoint:
        raise M2ParentError('Accepted M1 checkpoint name changed.')

    report_hash = _verify_file(
        report_path, audit.get('m1_report_sha256'), 'accepted M1 report',
    )
    acceptance_hash = _verify_file(
        acceptance_path, audit.get('m1_acceptance_sha256'), 'accepted M1 acceptance',
    )
    manifest_hash = _verify_file(
        manifest_path, audit.get('manifest_sha256'), 'accepted M1 manifest',
    )
    checkpoint_hash = _verify_checkpoint(checkpoint)
    if checkpoint_hash != audit.get('checkpoint_hash_manifest_sha256'):
        raise M2ParentError('Accepted M1 checkpoint hash manifest changed.')
    queue = _verify_work_items(checkpoint, config)

    report = read_json(report_path)
    if not isinstance(report, dict) or any((
        report.get('run_id') != config.parent_run_id,
        report.get('phase') != 'M1_COMPLETE',
        report.get('certification_status') != 'NOT_CERTIFIED',
        report.get('heuristic_results') != [],
        not _all_true(report.get('acceptance_gates')),
        report.get('proof_artifact_hashes') != audit.get('proof_artifact_hashes'),
    )):
        raise M2ParentError('Accepted M1 report no longer satisfies every gate.')
    if report.get('proof_artifact_hashes') != {
        item.phase: item.result_sha256 for item in queue.items.values()
    }:
        raise M2ParentError('Accepted M1 report hashes differ from its completed queue.')

    acceptance = read_json(acceptance_path)
    if not isinstance(acceptance, dict) or any((
        acceptance.get('milestone') != 'M1',
        acceptance.get('phase') != 'M1_COMPLETE',
        acceptance.get('status') != 'PASS',
        acceptance.get('certification_status') != 'NOT_CERTIFIED',
        not _all_true(acceptance.get('gates')),
    )):
        raise M2ParentError('Accepted M1 acceptance gates changed or failed.')

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or any((
        manifest.get('milestone') != 'M1',
        manifest.get('run_id') != config.parent_run_id,
        manifest.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M2ParentError('Accepted M1 manifest identity changed.')

    return {
        'm1_audit_sha256': sha256_file(audit_path),
        'm1_report_sha256': report_hash,
        'm1_acceptance_sha256': acceptance_hash,
        'm1_manifest_sha256': manifest_hash,
        'parent_checkpoint_hash_manifest_sha256': checkpoint_hash,
    }
