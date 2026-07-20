"""Shared M2 resolve + S0 + lineage package for Campaign B."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import atomic_write_json, hash_tree, read_json, sha256_file, utc_now
from ..m2_shared_registry import (
    BINDING_NEED,
    BINDING_READY,
    STATE_COMPLETE,
    lookup_shared_m2_reusable,
)
from .errors import CampaignFatalError, NeedCanonicalM2
from .schemas import assert_phase_allowed, assert_staged_candidate, screening_only_payload


def resolve_shared_m2(
    *,
    candidate: dict[str, Any],
    persistent_root: Path,
    source_tree_hash: str,
    allow_generate_canonical: bool,
    structural_key: str | None = None,
    proof_key: str | None = None,
) -> dict[str, Any]:
    assert_phase_allowed('M2_BIND')
    assert_staged_candidate(candidate)
    sk = structural_key or candidate.get('structural_key')
    pk = proof_key or candidate.get('proof_key')
    if not sk or not pk:
        raise NeedCanonicalM2('structural_key/proof_key required for M2 resolve')

    record, how = lookup_shared_m2_reusable(
        Path(persistent_root),
        str(sk),
        str(pk),
        source_hash=source_tree_hash,
    )
    if record is None or record.get('registry_state') != STATE_COMPLETE:
        if allow_generate_canonical:
            # Generation is intentionally not implemented in autonomous B driver.
            raise NeedCanonicalM2(
                f'canonical M2 missing for {sk}/{pk}; generation not wired'
            )
        return {
            'status': BINDING_NEED,
            'structural_key': sk,
            'proof_key': pk,
            'reason': 'NEED_CANONICAL_M2',
            **screening_only_payload(),
        }

    package_dir = record.get('canonical_package_dir') or record.get('package_dir')
    source_match = record.get('source_hash') == source_tree_hash
    reuse_class = (
        'EXACT_SOURCE_MATCH' if source_match else 'AUDITED_SOURCE_DRIFT_REUSE'
    )
    if how == 'structural_source_fallback' or (
        how and how != 'exact' and source_match
    ):
        reuse_class = 'EXACT_SOURCE_MATCH' if source_match else reuse_class

    binding = {
        'schema_version': 1,
        'status': BINDING_READY,
        'binding_status': BINDING_READY,
        'structural_key': sk,
        'proof_key': pk,
        'canonical_run_id': record.get('canonical_run_id') or record.get('run_id'),
        'canonical_package_dir': package_dir,
        'registry_state': record.get('registry_state'),
        'source_hash': record.get('source_hash'),
        'requested_source_hash': source_tree_hash,
        'reuse_class': reuse_class,
        'lookup_how': how,
        'registry_record_sha256': record.get('registry_record_sha256'),
        'resolved_at': utc_now(),
        **screening_only_payload(),
    }
    return binding


def run_s0_screening_record(
    *,
    candidate: dict[str, Any],
    m2_binding: dict[str, Any],
    primary_screen: dict[str, Any],
) -> dict[str, Any]:
    """Record S0-stage screening outcome bound to shared M2 (no production M6)."""
    assert_phase_allowed('S0')
    if m2_binding.get('status') not in {BINDING_READY, 'READY_SHARED'}:
        raise CampaignFatalError('S0 requires READY shared M2 binding')
    return {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'stage': 'S0',
        'm2_binding_status': m2_binding.get('status'),
        'canonical_run_id': m2_binding.get('canonical_run_id'),
        'q_upper': primary_screen.get('q_upper'),
        'estimated_q': primary_screen.get('estimated_q'),
        'screen_status': primary_screen.get('screen_status'),
        'j2': candidate.get('j2'),
        'execution_mode': candidate.get('execution_mode'),
        'scheme': candidate.get('scheme'),
        'scheme_hash': candidate.get('scheme_hash'),
        'notes': (
            'Campaign B S0 screening record; NOT_CERTIFIED; no production M6.'
        ),
        'completed_at': utc_now(),
        **screening_only_payload(),
    }


def build_lineage_package(
    *,
    package_root: Path,
    candidate: dict[str, Any],
    campaign_manifest: dict[str, Any],
    m2_binding: dict[str, Any],
    s0_result: dict[str, Any],
    verification: dict[str, Any],
    package_audit: dict[str, Any],
    source_tree_hash: str,
    environment_manifest: dict[str, Any],
) -> Path:
    assert_phase_allowed('SELECTED')
    root = Path(package_root)
    root.mkdir(parents=True, exist_ok=True)
    files = {
        'candidate_manifest.json': {
            **candidate,
            **screening_only_payload(),
        },
        'campaign_manifest.json': campaign_manifest,
        'scheme.json': candidate.get('scheme'),
        'structural_key.json': {
            'structural_key': candidate.get('structural_key'),
        },
        'proof_key.json': {
            'proof_key': candidate.get('proof_key'),
        },
        'm2_binding.json': m2_binding,
        'shared_m2_audit.json': {
            'reuse_class': m2_binding.get('reuse_class'),
            'registry_record_sha256': m2_binding.get('registry_record_sha256'),
            'canonical_run_id': m2_binding.get('canonical_run_id'),
            **screening_only_payload(),
        },
        's0_result.json': s0_result,
        'independent_verification.json': verification,
        'package_audit.json': package_audit,
        'source_tree_manifest.json': {
            'source_tree_hash': source_tree_hash,
        },
        'environment_manifest.json': environment_manifest,
    }
    for name, payload in files.items():
        atomic_write_json(root / name, payload)

    readme = (
        'STATUS: NOT_CERTIFIED\n'
        'SCOPE: SCREENING-ONLY CANDIDATE\n'
        'PROHIBITED INTERPRETATION:\n'
        '  - no production M6 claim\n'
        '  - no continuum claim\n'
        '  - no mass-gap claim\n'
    )
    (root / 'README.md').write_text(readme, encoding='utf-8')

    hashes: dict[str, str] = {}
    for path in sorted(root.iterdir()):
        if path.name == 'hashes.sha256' or not path.is_file():
            continue
        hashes[path.name] = sha256_file(path)
    lines = [f'{digest}  {name}' for name, digest in sorted(hashes.items())]
    (root / 'hashes.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    # COMPLETED marker only after all artifacts exist.
    atomic_write_json(root / 'COMPLETED.json', {
        'completed_at': utc_now(),
        **screening_only_payload(),
    })
    return root
