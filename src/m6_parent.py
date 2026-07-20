"""Verify accepted M5 parent for M6."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file
from .m6_status import ALLOWED_M5_CERTIFICATION, M5_PARENT_RUN_ID_FROZEN


class M6ParentError(RuntimeError):
    """Raised when the accepted M5 parent cannot be verified."""


@dataclass(frozen=True, slots=True)
class M6ParentEvidence:
    run_id: str
    acceptance: dict[str, Any]
    acceptance_path: Path
    package_root: Path
    hashes: dict[str, str]


def verify_accepted_m5_parent(
    project_root: Path,
    persistent_root: Path,
    run_id: str,
    *,
    require_frozen_id: bool = False,
) -> M6ParentEvidence:
    if require_frozen_id and run_id != M5_PARENT_RUN_ID_FROZEN:
        raise M6ParentError(
            f'Paperspace M6 requires frozen M5 parent {M5_PARENT_RUN_ID_FROZEN}.'
        )
    run_root = persistent_root / 'runs' / run_id
    acceptance_path = run_root / 'reports' / 'M5_acceptance.json'
    if not acceptance_path.is_file():
        raise M6ParentError(f'Missing M5 acceptance: {acceptance_path}')
    acceptance = read_json(acceptance_path)
    if not isinstance(acceptance, dict):
        raise M6ParentError('M5 acceptance is malformed.')
    if any((
        acceptance.get('milestone') != 'M5',
        acceptance.get('phase') != 'M5_COMPLETE',
        acceptance.get('status') != 'PASS',
        acceptance.get('accepted_for_next_milestone') != 'M6',
        acceptance.get('certification_status') not in ALLOWED_M5_CERTIFICATION,
    )):
        raise M6ParentError('M5 acceptance identity/status rejected.')

    package_root = run_root / 'artifacts' / 'one_step_certificate'
    if not package_root.is_dir():
        raise M6ParentError(f'Missing M5 one_step_certificate: {package_root}')
    verdict_path = package_root / 'verdict.json'
    if not verdict_path.is_file():
        raise M6ParentError('M5 package verdict.json is missing.')
    verdict = read_json(verdict_path)
    if not isinstance(verdict, dict) or verdict.get('independent_verifier') != 'PASS':
        raise M6ParentError('M5 package independent verifier is not PASS.')

    audit_path = project_root / 'audit' / 'm5_accepted_parent.json'
    hashes = {
        'm5_acceptance_sha256': sha256_file(acceptance_path),
        'm5_package_verdict_sha256': sha256_file(verdict_path),
    }
    if audit_path.is_file():
        hashes['m5_audit_sha256'] = sha256_file(audit_path)

    return M6ParentEvidence(
        run_id=run_id,
        acceptance=acceptance,
        acceptance_path=acceptance_path,
        package_root=package_root,
        hashes=hashes,
    )
