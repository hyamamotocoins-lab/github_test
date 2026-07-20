"""Package-local immutable M2 parent audits for shared M2 reuse."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, sha256_file, utc_now
from .m2_batching import proof_artifact_hash_map
from .work_queue import WorkQueue


class M2PackageAuditError(RuntimeError):
    """Raised when package-local M2 audit cannot be written safely."""


def package_m2_audit_path(package_root: Path) -> Path:
    return Path(package_root) / 'audits' / 'm2_shared_parent.json'


def write_package_m2_shared_audit(
    package_root: Path,
    *,
    run_root: Path,
    structural_key: str,
    proof_key: str,
    registry_record_sha256: str | None,
) -> dict[str, Any]:
    """Write immutable package-local audit pointing at shared canonical M2."""
    run_root = Path(run_root).resolve()
    report_path = run_root / 'reports' / 'M2_report.json'
    acceptance_path = run_root / 'reports' / 'M2_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    if not report_path.is_file() or not acceptance_path.is_file():
        raise M2PackageAuditError('Shared M2 report/acceptance missing.')
    report = read_json(report_path)
    acceptance = read_json(acceptance_path)
    if not isinstance(report, dict) or not isinstance(acceptance, dict):
        raise M2PackageAuditError('Shared M2 artifacts malformed.')
    if report.get('phase') != 'M2_COMPLETE' or acceptance.get('status') != 'PASS':
        raise M2PackageAuditError('Shared M2 is not acceptance-complete.')

    checkpoint_meta = report.get('checkpoint') or {}
    raw_ckpt = str(checkpoint_meta.get('path') or '').strip()
    checkpoint_path = Path(raw_ckpt) if raw_ckpt else Path()
    if not raw_ckpt or not checkpoint_path.is_dir():
        ckpt_root = run_root / 'checkpoints'
        candidates = sorted(
            path for path in ckpt_root.glob('ckpt_*')
            if (path / 'COMMITTED').is_file()
        )
        if not candidates:
            raise M2PackageAuditError('No committed M2 checkpoint.')
        checkpoint_path = candidates[-1]

    hashes_path = checkpoint_path / 'hashes.json'
    queue = read_json(checkpoint_path / 'work_queue.json')
    state = read_json(checkpoint_path / 'state.json')
    if not isinstance(queue, dict) or not isinstance(state, dict):
        raise M2PackageAuditError('Checkpoint queue/state malformed.')
    work = WorkQueue.from_payload(queue)

    out = package_m2_audit_path(package_root)
    if out.is_file():
        existing = read_json(out)
        if isinstance(existing, dict) and existing.get('m2_proof_key') == proof_key:
            # Immutable: same proof key ⇒ return existing.
            return existing
        if isinstance(existing, dict):
            raise M2PackageAuditError(
                'Package M2 audit already exists for a different proof key.'
            )

    audit = {
        'milestone_reviewed': 'M2',
        'accepted_for_next_milestone': 'M3',
        'accepted_phase': 'M2_COMPLETE',
        'accepted_run_id': report.get('run_id'),
        'checkpoint_index': int(state.get('checkpoint_index', checkpoint_meta.get('index', 0))),
        'decision': 'ACCEPT_SHARED_M2_FOR_CANDIDATE_M3',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm2_report_path': str(report_path),
        'm2_acceptance_path': str(acceptance_path),
        'checkpoint_path': str(checkpoint_path),
        'manifest_path': str(manifest_path),
        'm2_report_sha256': sha256_file(report_path),
        'm2_acceptance_sha256': sha256_file(acceptance_path),
        'manifest_sha256': sha256_file(manifest_path),
        'checkpoint_hash_manifest_sha256': sha256_file(hashes_path),
        'proof_artifact_hashes': proof_artifact_hash_map(work.items.values()),
        'm2_structural_key': structural_key,
        'm2_proof_key': proof_key,
        'm2_registry_record_sha256': registry_record_sha256,
        'm2_parent_audit_path': str(out),
        'shared_m2': True,
        'staged_child_lineage': True,
        'generated_at': utc_now(),
        'scope_limitation': (
            'Package-local shared M2 parent audit; does not overwrite global '
            'audit/m2_accepted_parent.json; not a continuum or mass-gap claim.'
        ),
    }
    atomic_write_json(out, audit)
    audit['m2_parent_audit_sha256'] = sha256_file(out)
    atomic_write_json(out, audit)
    return audit


def read_package_m2_audit(package_root: Path) -> dict[str, Any] | None:
    path = package_m2_audit_path(package_root)
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None
