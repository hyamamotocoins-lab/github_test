"""Package audit and archive for Campaign B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, sha256_file, utc_now
from .errors import CampaignFatalError, InvariantViolation
from .schemas import CERTIFICATION_STATUS, CLAIM_SCOPE, screening_only_payload


REQUIRED_SELECTED_FILES = (
    'candidate_manifest.json',
    'campaign_manifest.json',
    'scheme.json',
    'structural_key.json',
    'proof_key.json',
    'm2_binding.json',
    'shared_m2_audit.json',
    's0_result.json',
    'independent_verification.json',
    'package_audit.json',
    'source_tree_manifest.json',
    'environment_manifest.json',
    'hashes.sha256',
    'README.md',
)


def audit_lineage_package(package_dir: Path) -> dict[str, Any]:
    root = Path(package_dir)
    missing = [name for name in REQUIRED_SELECTED_FILES if not (root / name).is_file()]
    if missing:
        return {
            'accepted': False,
            'reason': 'MISSING_ARTIFACTS',
            'missing': missing,
            **screening_only_payload(),
        }

    readme = (root / 'README.md').read_text(encoding='utf-8')
    if 'STATUS: NOT_CERTIFIED' not in readme:
        raise InvariantViolation('selected README missing NOT_CERTIFIED banner')

    for name in (
        'candidate_manifest.json',
        's0_result.json',
        'independent_verification.json',
        'm2_binding.json',
    ):
        payload = read_json(root / name)
        if not isinstance(payload, dict):
            raise CampaignFatalError(f'{name} corrupt')
        if payload.get('certification_status') not in {None, CERTIFICATION_STATUS}:
            raise InvariantViolation(f'{name} certification_status invalid')
        if payload.get('claim_scope') not in {None, CLAIM_SCOPE}:
            raise InvariantViolation(f'{name} claim_scope invalid')

    verify = read_json(root / 'independent_verification.json')
    if not isinstance(verify, dict) or not verify.get('accepted'):
        return {
            'accepted': False,
            'reason': 'VERIFY_NOT_ACCEPTED',
            **screening_only_payload(),
        }

    # Re-check hashes
    expected: dict[str, str] = {}
    for line in (root / 'hashes.sha256').read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition('  ')
        expected[name] = digest
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            return {
                'accepted': False,
                'reason': 'HASH_MISSING_FILE',
                'file': name,
                **screening_only_payload(),
            }
        if sha256_file(path) != digest:
            raise InvariantViolation(f'hash mismatch for {name}')

    return {
        'accepted': True,
        'reason': None,
        'audited_at': utc_now(),
        **screening_only_payload(),
    }


def archive_candidate(
    archive_root: Path,
    *,
    candidate: dict[str, Any],
    screening_result: dict[str, Any] | None,
    reason_code: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    cand_id = str(candidate.get('candidate_id'))
    dest = Path(archive_root) / cand_id
    dest.mkdir(parents=True, exist_ok=True)
    atomic_write_json(dest / 'candidate_manifest.json', {
        **candidate,
        **screening_only_payload(),
    })
    if screening_result is not None:
        atomic_write_json(dest / 'screening_result.json', screening_result)
    atomic_write_json(dest / 'reason.json', {
        'code': reason_code,
        'at': utc_now(),
        **(extra or {}),
        **screening_only_payload(),
    })
    hashes = {}
    for path in sorted(dest.iterdir()):
        if path.is_file():
            hashes[path.name] = sha256_file(path)
    lines = [f'{digest}  {name}' for name, digest in sorted(hashes.items())]
    (dest / 'hashes.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    atomic_write_json(dest / 'COMPLETED.json', {
        'completed_at': utc_now(),
        **screening_only_payload(),
    })
    return dest
