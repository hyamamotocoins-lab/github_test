"""Two-level M2 compatibility keys: structural vs proof.

Structural key: same mathematical M2 problem across candidates.
Proof key: same acceptance artifact (schema + source + shared notebook token).

Campaign C shared M2 uses SHARED_M2_NOTEBOOK_TOKEN in the proof key so that
candidates resolve to the same registry slot. The run's actual notebook hash is
provenance only and must not fragment the shared registry.

Do NOT mix source_hash into the structural key. Do NOT reuse across
proof keys with allow_code_drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import canonical_json_bytes, read_json, sha256_bytes, sha256_file
from .fusion import convention_hash
from .m2_config import M2Config
from .sector_canonicalization import action_table_hash


class M2CompatibilityError(RuntimeError):
    """Raised when M2 compatibility keys cannot be formed safely."""


SECTOR_ENUMERATION_SCHEMA = 'M2_LINK_STAR_SECTORS_V1'
DEFAULT_PROOF_SCHEMA = 'M2_PROOF_SCHEMA_V2'
DEFAULT_PROOF_METHOD = 'invariant_subspace_uniqueness_v1'
# Fixed token for shared-registry proof keys (not a per-notebook digest).
SHARED_M2_NOTEBOOK_TOKEN = 'shared-m2-v1'


def accepted_m1_identity_sha256(
    *,
    m1_report_sha256: str,
    m1_acceptance_sha256: str,
    checkpoint_hash_manifest_sha256: str,
    m1_manifest_sha256: str | None = None,
) -> str:
    payload = {
        'm1_report_sha256': str(m1_report_sha256),
        'm1_acceptance_sha256': str(m1_acceptance_sha256),
        'checkpoint_hash_manifest_sha256': str(checkpoint_hash_manifest_sha256),
        'm1_manifest_sha256': str(m1_manifest_sha256 or ''),
    }
    return sha256_bytes(canonical_json_bytes(payload))


def accepted_m1_identity_from_audit(project_root: Path | None) -> dict[str, Any]:
    """Load accepted M1 digests; include run_id as provenance only."""
    if project_root is None:
        raise M2CompatibilityError('project_root required for M1 identity')
    audit_path = Path(project_root) / 'audit' / 'm1_accepted_parent.json'
    if not audit_path.is_file():
        raise M2CompatibilityError(f'M1 audit missing: {audit_path}')
    audit = read_json(audit_path)
    if not isinstance(audit, dict):
        raise M2CompatibilityError('M1 audit malformed')
    required = (
        'm1_report_sha256', 'm1_acceptance_sha256',
        'checkpoint_hash_manifest_sha256',
    )
    for field in required:
        if not audit.get(field):
            raise M2CompatibilityError(f'M1 audit missing {field}')
    identity = accepted_m1_identity_sha256(
        m1_report_sha256=str(audit['m1_report_sha256']),
        m1_acceptance_sha256=str(audit['m1_acceptance_sha256']),
        checkpoint_hash_manifest_sha256=str(audit['checkpoint_hash_manifest_sha256']),
        m1_manifest_sha256=(
            str(audit['manifest_sha256']) if audit.get('manifest_sha256') else None
        ),
    )
    return {
        'accepted_m1_identity_sha256': identity,
        'm1_parent_run_id': audit.get('accepted_run_id'),
        'm1_report_sha256': audit['m1_report_sha256'],
        'm1_acceptance_sha256': audit['m1_acceptance_sha256'],
        'checkpoint_hash_manifest_sha256': audit['checkpoint_hash_manifest_sha256'],
        'm1_manifest_sha256': audit.get('manifest_sha256'),
    }


def structural_payload(
    *,
    config: M2Config | None = None,
    accepted_m1_identity: str,
    m1_parent_run_id_provenance: str | None = None,
) -> dict[str, Any]:
    cfg = config or M2Config()
    payload = {
        'accepted_m1_identity_sha256': str(accepted_m1_identity),
        **cfg.semantic_compatibility_payload(),
        'fusion_convention_hash': convention_hash(),
        'sector_enumeration_schema': SECTOR_ENUMERATION_SCHEMA,
        'symmetry_action_table_hash': action_table_hash(),
    }
    # Provenance only — not used for equality of mathematical identity.
    if m1_parent_run_id_provenance:
        payload['m1_parent_run_id_provenance'] = str(m1_parent_run_id_provenance)
    return payload


def compute_structural_key(
    *,
    accepted_m1_identity: str,
    config: M2Config | None = None,
    m1_parent_run_id_provenance: str | None = None,
) -> str:
    # Exclude provenance from the hashed payload.
    payload = structural_payload(
        config=config,
        accepted_m1_identity=accepted_m1_identity,
        m1_parent_run_id_provenance=None,
    )
    return sha256_bytes(canonical_json_bytes(payload))


def proof_payload(
    *,
    structural_key: str,
    source_hash: str,
    notebook_hash: str,
    proof_schema: str = DEFAULT_PROOF_SCHEMA,
    proof_method: str = DEFAULT_PROOF_METHOD,
) -> dict[str, Any]:
    return {
        'structural_key': str(structural_key),
        'proof_schema': str(proof_schema),
        'proof_method': str(proof_method),
        'source_hash': str(source_hash),
        'notebook_hash': str(notebook_hash),
    }


def compute_proof_key(
    *,
    structural_key: str,
    source_hash: str,
    notebook_hash: str,
    proof_schema: str = DEFAULT_PROOF_SCHEMA,
    proof_method: str = DEFAULT_PROOF_METHOD,
) -> str:
    return sha256_bytes(canonical_json_bytes(proof_payload(
        structural_key=structural_key,
        source_hash=source_hash,
        notebook_hash=notebook_hash,
        proof_schema=proof_schema,
        proof_method=proof_method,
    )))


def shared_run_id_for_keys(structural_key: str, proof_key: str) -> str:
    if len(structural_key) < 8 or len(proof_key) < 12:
        raise M2CompatibilityError('keys too short for shared run id')
    return f'M2-SHARED-{structural_key[:8]}-{proof_key[:12]}'


def keys_from_project(
    project_root: Path,
    *,
    j2_max: int,
    source_hash: str,
    notebook_hash: str,
    proof_schema: str = DEFAULT_PROOF_SCHEMA,
    proof_method: str = DEFAULT_PROOF_METHOD,
) -> dict[str, Any]:
    m1 = accepted_m1_identity_from_audit(project_root)
    # j2_max>1 requires sector_batch_size>=1 for M2Config validation.
    batch = 0 if j2_max <= 1 else 16
    config = M2Config(j2_max=j2_max, sector_batch_size=batch)
    structural_key = compute_structural_key(
        accepted_m1_identity=m1['accepted_m1_identity_sha256'],
        config=config,
        m1_parent_run_id_provenance=m1.get('m1_parent_run_id'),
    )
    proof_key = compute_proof_key(
        structural_key=structural_key,
        source_hash=source_hash,
        notebook_hash=notebook_hash,
        proof_schema=proof_schema,
        proof_method=proof_method,
    )
    return {
        'structural_key': structural_key,
        'proof_key': proof_key,
        'accepted_m1': m1,
        'shared_run_id': shared_run_id_for_keys(structural_key, proof_key),
        'structural_payload': structural_payload(
            config=config,
            accepted_m1_identity=m1['accepted_m1_identity_sha256'],
            m1_parent_run_id_provenance=m1.get('m1_parent_run_id'),
        ),
        'proof_payload': proof_payload(
            structural_key=structural_key,
            source_hash=source_hash,
            notebook_hash=notebook_hash,
            proof_schema=proof_schema,
            proof_method=proof_method,
        ),
    }


def keys_from_run_artifacts(
    run_root: Path,
    *,
    project_root: Path | None = None,
    shared_registry: bool = True,
) -> dict[str, Any]:
    """Derive structural+proof keys from a completed M2 run.

    When shared_registry=True (default), proof_key uses SHARED_M2_NOTEBOOK_TOKEN
    so Campaign C candidates share one registry slot for the same source tree.
    The run's actual notebook_hash is returned as run_notebook_hash (provenance).
    """
    run_root = Path(run_root)
    config = read_json(run_root / 'run_config.json')
    manifest = read_json(run_root / 'run_manifest.json')
    if not isinstance(config, dict) or not isinstance(manifest, dict):
        raise M2CompatibilityError('run_config/manifest missing')
    source_hash = str(manifest.get('source_hash') or '')
    run_notebook_hash = str(manifest.get('notebook_hash') or '')
    if not source_hash or not run_notebook_hash:
        raise M2CompatibilityError('manifest missing source_hash/notebook_hash')
    notebook_hash = (
        SHARED_M2_NOTEBOOK_TOKEN if shared_registry else run_notebook_hash
    )

    if project_root is not None:
        m1 = accepted_m1_identity_from_audit(project_root)
        accepted = m1['accepted_m1_identity_sha256']
        provenance = m1.get('m1_parent_run_id')
    else:
        # Fall back to digests recorded on the M2 manifest/parent block.
        parent = manifest.get('parent') if isinstance(manifest.get('parent'), dict) else {}
        accepted = accepted_m1_identity_sha256(
            m1_report_sha256=str(parent.get('m1_report_sha256') or manifest.get('m1_report_sha256') or 'MISSING'),
            m1_acceptance_sha256=str(parent.get('m1_acceptance_sha256') or manifest.get('m1_acceptance_sha256') or 'MISSING'),
            checkpoint_hash_manifest_sha256=str(
                parent.get('parent_checkpoint_hash_manifest_sha256')
                or manifest.get('parent_checkpoint_hash_manifest_sha256')
                or 'MISSING'
            ),
            m1_manifest_sha256=str(parent.get('m1_manifest_sha256') or ''),
        )
        provenance = str(config.get('parent_run_id') or '')

    j2_max = int(config['j2_max'])
    batch = 0 if j2_max <= 1 else max(1, int(config.get('sector_batch_size') or 16))
    m2_config = M2Config(j2_max=j2_max, sector_batch_size=batch)
    structural_key = compute_structural_key(
        accepted_m1_identity=accepted,
        config=m2_config,
        m1_parent_run_id_provenance=provenance,
    )
    proof_key = compute_proof_key(
        structural_key=structural_key,
        source_hash=source_hash,
        notebook_hash=notebook_hash,
        proof_schema=str(config.get('proof_schema') or DEFAULT_PROOF_SCHEMA),
        proof_method=str(config.get('proof_method') or DEFAULT_PROOF_METHOD),
    )
    return {
        'structural_key': structural_key,
        'proof_key': proof_key,
        'source_hash': source_hash,
        'notebook_hash': notebook_hash,
        'run_notebook_hash': run_notebook_hash,
        'shared_run_id': shared_run_id_for_keys(structural_key, proof_key),
        'j2_max': j2_max,
    }
