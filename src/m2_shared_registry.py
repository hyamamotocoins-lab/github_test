"""Shared M2 registry with structural/proof keys, reservation, and quarantine.

Canonical run directory is source of truth. Registry stores paths + hashes only.
"""

from __future__ import annotations

import os
import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json, hash_tree, read_json, sha256_bytes, sha256_file,
    canonical_json_bytes, utc_now,
)
from .m2_batching import proof_artifact_hash_map
from .m2_compatibility import (
    SHARED_M2_NOTEBOOK_TOKEN,
    keys_from_project,
    keys_from_run_artifacts,
    shared_run_id_for_keys,
)
from .work_queue import WorkQueue


class M2SharedRegistryError(RuntimeError):
    """Raised when shared M2 registry operations fail closed."""


STATE_EMPTY = 'EMPTY'
STATE_RESERVED = 'RESERVED'
STATE_RUNNING = 'RUNNING'
STATE_COMPLETE = 'COMPLETE'
STATE_QUARANTINED = 'QUARANTINED'
STATE_LEGACY_PENDING = 'LEGACY_PENDING_REVERIFY'

MODE_STRICT = 'STRICT_PRODUCTION'
MODE_LEGACY = 'LEGACY_QUARANTINE'

BINDING_UNRESOLVED = 'UNRESOLVED'
BINDING_REJECTED = 'REJECTED_SCREENING'
BINDING_NEED = 'NEED_CANONICAL_M2'
BINDING_WAITING = 'WAITING_FOR_CANONICAL_M2'
BINDING_READY = 'READY_SHARED'


def registry_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'shared_m2_registry'


def proof_entry_dir(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
) -> Path:
    return registry_root(persistent_root) / structural_key / 'proofs' / proof_key


def lookup_shared_m2(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
) -> dict[str, Any] | None:
    path = proof_entry_dir(persistent_root, structural_key, proof_key) / 'canonical_run.json'
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def list_structural_proof_records(
    persistent_root: Path,
    structural_key: str,
) -> list[dict[str, Any]]:
    root = registry_root(persistent_root) / structural_key / 'proofs'
    if not root.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for entry in sorted(root.iterdir()):
        path = entry / 'canonical_run.json'
        if not path.is_file():
            continue
        payload = read_json(path)
        if isinstance(payload, dict):
            records.append(payload)
    return records


def lookup_shared_m2_reusable(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
    *,
    source_hash: str | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Exact proof-key hit, else COMPLETE under same structural+source_hash.

    Fallback recovers Campaign C runs registered under a pre-token notebook hash.
    """
    exact = lookup_shared_m2(persistent_root, structural_key, proof_key)
    if exact is not None:
        return exact, 'exact'
    if not source_hash:
        return None, None
    matches = [
        record for record in list_structural_proof_records(
            persistent_root, structural_key,
        )
        if record.get('registry_state') == STATE_COMPLETE
        and record.get('source_hash') == source_hash
    ]
    if not matches:
        return None, None
    matches.sort(key=lambda record: str(record.get('registered_at') or ''), reverse=True)
    return matches[0], 'structural_source_fallback'


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace('Z', '+00:00'))
    except ValueError:
        return None


def reserve_shared_m2(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
    *,
    owner_id: str,
    lease_seconds: int = 3600,
    canonical_run_id: str | None = None,
) -> dict[str, Any]:
    """Atomically reserve a proof-key slot (mkdir exclusive)."""
    existing = lookup_shared_m2(persistent_root, structural_key, proof_key)
    if existing and existing.get('registry_state') == STATE_COMPLETE:
        raise M2SharedRegistryError('Already COMPLETE; reuse instead of reserve.')
    if existing and existing.get('registry_state') == STATE_QUARANTINED:
        raise M2SharedRegistryError('Entry QUARANTINED; cannot reserve.')

    entry = proof_entry_dir(persistent_root, structural_key, proof_key)
    entry.parent.mkdir(parents=True, exist_ok=True)
    reserved_marker = entry
    try:
        os.mkdir(reserved_marker)
        created = True
    except FileExistsError:
        created = False

    reservation_path = entry / 'RESERVATION.json'
    if not created:
        # Recover stale lease if possible.
        current = read_json(reservation_path) if reservation_path.is_file() else None
        if isinstance(current, dict):
            expires = _parse_iso(str(current.get('lease_expires_at') or ''))
            if expires and expires > _now():
                if current.get('owner_id') == owner_id:
                    return current
                raise M2SharedRegistryError(
                    f'Reserved by {current.get("owner_id")} until {current.get("lease_expires_at")}'
                )
            # Stale: take over.
        else:
            # Directory exists without reservation — treat as conflict unless COMPLETE.
            record = lookup_shared_m2(persistent_root, structural_key, proof_key)
            if record and record.get('registry_state') == STATE_COMPLETE:
                raise M2SharedRegistryError('Already COMPLETE.')
            raise M2SharedRegistryError('Proof entry exists without usable reservation.')

    run_id = canonical_run_id or shared_run_id_for_keys(structural_key, proof_key)
    now = _now()
    payload = {
        'schema_version': 1,
        'owner_id': owner_id,
        'host': socket.gethostname(),
        'pid': os.getpid(),
        'reserved_at': now.isoformat(),
        'heartbeat_at': now.isoformat(),
        'lease_expires_at': (now + timedelta(seconds=int(lease_seconds))).isoformat(),
        'canonical_run_id': run_id,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'registry_state': STATE_RESERVED,
    }
    atomic_write_json(reservation_path, payload)
    atomic_write_json(entry / 'canonical_run.json', {
        'schema_version': 1,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'canonical_run_id': run_id,
        'registry_state': STATE_RESERVED,
        'registered_at': utc_now(),
    })
    return payload


def heartbeat_reservation(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
    *,
    owner_id: str,
    lease_seconds: int = 3600,
) -> dict[str, Any]:
    entry = proof_entry_dir(persistent_root, structural_key, proof_key)
    path = entry / 'RESERVATION.json'
    current = read_json(path) if path.is_file() else None
    if not isinstance(current, dict) or current.get('owner_id') != owner_id:
        raise M2SharedRegistryError('Cannot heartbeat: not owner')
    now = _now()
    current['heartbeat_at'] = now.isoformat()
    current['lease_expires_at'] = (now + timedelta(seconds=int(lease_seconds))).isoformat()
    current['registry_state'] = STATE_RUNNING
    atomic_write_json(path, current)
    return current


def register_shared_m2_from_run(
    persistent_root: Path,
    run_root: Path,
    *,
    project_root: Path | None = None,
    registration_mode: str = MODE_STRICT,
    allow_overwrite: bool = False,
) -> dict[str, Any]:
    """Register completed M2. STRICT_PRODUCTION or LEGACY_QUARANTINE only."""
    if registration_mode not in {MODE_STRICT, MODE_LEGACY}:
        raise M2SharedRegistryError(f'Unknown registration_mode={registration_mode}')
    run_root = Path(run_root).resolve()
    report_path = run_root / 'reports' / 'M2_report.json'
    acceptance_path = run_root / 'reports' / 'M2_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    config_path = run_root / 'run_config.json'
    for path in (report_path, acceptance_path, manifest_path, config_path):
        if not path.is_file():
            raise M2SharedRegistryError(f'Missing: {path}')

    report = read_json(report_path)
    acceptance = read_json(acceptance_path)
    manifest = read_json(manifest_path)
    if not all(isinstance(doc, dict) for doc in (report, acceptance, manifest)):
        raise M2SharedRegistryError('Artifacts malformed')
    if report.get('phase') != 'M2_COMPLETE' or acceptance.get('status') != 'PASS':
        raise M2SharedRegistryError('M2 not acceptance-complete')

    # Queue completeness from latest committed checkpoint.
    ckpt_root = run_root / 'checkpoints'
    committed = sorted(
        path for path in ckpt_root.glob('ckpt_*')
        if (path / 'COMMITTED').is_file()
    ) if ckpt_root.is_dir() else []
    if not committed:
        raise M2SharedRegistryError('No committed checkpoint')
    latest = committed[-1]
    queue_payload = read_json(latest / 'work_queue.json')
    state = read_json(latest / 'state.json')
    if not isinstance(queue_payload, dict) or not isinstance(state, dict):
        raise M2SharedRegistryError('Checkpoint queue/state malformed')
    work = WorkQueue.from_payload(queue_payload)
    counts = {'pending': 0, 'running': 0, 'done': 0, 'failed': 0}
    for item in work.items.values():
        status = str(getattr(item, 'status', 'pending'))
        counts[status] = counts.get(status, 0) + 1
    if counts.get('failed', 0) or counts.get('pending', 0) or counts.get('running', 0):
        raise M2SharedRegistryError(f'Queue not fully done: {counts}')
    if state.get('phase') != 'M2_COMPLETE':
        raise M2SharedRegistryError('state.phase != M2_COMPLETE')
    if state.get('certification_status') not in {None, 'NOT_CERTIFIED'}:
        raise M2SharedRegistryError('Unexpected certification_status')

    key_info = keys_from_run_artifacts(run_root, project_root=project_root)
    structural_key = key_info['structural_key']
    proof_key = key_info['proof_key']

    # Detect code drift signals for quarantine.
    drift = bool(manifest.get('code_drift') or manifest.get('allow_code_drift'))
    if registration_mode == MODE_STRICT and drift:
        raise M2SharedRegistryError(
            'STRICT_PRODUCTION refuses runs with code_drift; use LEGACY_QUARANTINE'
        )

    registry_state = (
        STATE_LEGACY_PENDING if registration_mode == MODE_LEGACY else STATE_COMPLETE
    )
    if registration_mode == MODE_LEGACY:
        registry_state = STATE_QUARANTINED if drift else STATE_LEGACY_PENDING

    entry = proof_entry_dir(persistent_root, structural_key, proof_key)
    if entry.is_dir() and (entry / 'canonical_run.json').is_file() and not allow_overwrite:
        existing = lookup_shared_m2(persistent_root, structural_key, proof_key)
        if isinstance(existing, dict) and existing.get('registry_state') == STATE_COMPLETE:
            return existing

    entry.mkdir(parents=True, exist_ok=True)
    run_id = str(report.get('run_id') or run_root.name)
    preferred = shared_run_id_for_keys(structural_key, proof_key)
    record = {
        'schema_version': 2,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'canonical_run_id': run_id,
        'canonical_run_id_is_legacy': not run_id.startswith('M2-SHARED-'),
        'preferred_new_run_id': preferred,
        'canonical_run_root': str(run_root),
        'acceptance_relpath': 'reports/M2_acceptance.json',
        'acceptance_sha256': sha256_file(acceptance_path),
        'report_relpath': 'reports/M2_report.json',
        'report_sha256': sha256_file(report_path),
        'manifest_relpath': 'run_manifest.json',
        'manifest_sha256': sha256_file(manifest_path),
        'checkpoint_relpath': str(latest.relative_to(run_root)),
        'checkpoint_hash_manifest_sha256': sha256_file(latest / 'hashes.json'),
        'source_hash': key_info['source_hash'],
        'notebook_hash': key_info['notebook_hash'],
        'run_notebook_hash': key_info.get('run_notebook_hash'),
        'proof_artifact_hashes': proof_artifact_hash_map(work.items.values()),
        'queue_counts': counts,
        'registry_state': registry_state,
        'registration_mode': registration_mode,
        'certification_status': 'NOT_CERTIFIED',
        'registered_at': utc_now(),
        'interpretation': 'SHARED_STRUCTURAL_M2_NOT_A_CANDIDATE_CONTRACTIVITY_CLAIM',
    }
    record['registry_record_sha256'] = sha256_bytes(canonical_json_bytes({
        k: v for k, v in record.items() if k != 'registry_record_sha256'
    }))
    atomic_write_json(entry / 'canonical_run.json', record)
    if registry_state == STATE_COMPLETE:
        verify_shared_m2(
            persistent_root, structural_key, proof_key,
            require_source_match=True,
            current_source_hash=key_info['source_hash'],
            current_notebook_hash=key_info['notebook_hash'],
        )
    return record


def verify_shared_m2(
    persistent_root: Path,
    structural_key: str,
    proof_key: str,
    *,
    require_source_match: bool = True,
    current_source_hash: str | None = None,
    current_notebook_hash: str | None = None,
) -> dict[str, Any]:
    record = lookup_shared_m2(persistent_root, structural_key, proof_key)
    if record is None:
        raise M2SharedRegistryError('No registry record')
    if record.get('registry_state') in {STATE_QUARANTINED, STATE_LEGACY_PENDING}:
        raise M2SharedRegistryError(
            f"Registry state {record.get('registry_state')} is not production-reusable"
        )
    if record.get('registry_state') != STATE_COMPLETE:
        raise M2SharedRegistryError(
            f"Registry state {record.get('registry_state')} not COMPLETE"
        )

    run_root = Path(str(record['canonical_run_root']))
    reasons: list[str] = []
    if not run_root.is_dir():
        reasons.append('run root missing')

    def check(rel: str, expected: str | None, label: str) -> None:
        path = run_root / rel
        if not path.is_file():
            reasons.append(f'{label} missing')
            return
        if expected and sha256_file(path) != expected:
            reasons.append(f'{label} hash mismatch')

    check(str(record.get('acceptance_relpath')), record.get('acceptance_sha256'), 'acceptance')
    check(str(record.get('report_relpath')), record.get('report_sha256'), 'report')
    check(str(record.get('manifest_relpath')), record.get('manifest_sha256'), 'manifest')
    ckpt_rel = str(record.get('checkpoint_relpath') or '')
    if ckpt_rel:
        hashes = run_root / ckpt_rel / 'hashes.json'
        if not hashes.is_file():
            reasons.append('checkpoint hashes missing')
        elif record.get('checkpoint_hash_manifest_sha256') and (
            sha256_file(hashes) != record.get('checkpoint_hash_manifest_sha256')
        ):
            reasons.append('checkpoint hash mismatch')
        if not (run_root / ckpt_rel / 'COMMITTED').is_file():
            reasons.append('checkpoint not COMMITTED')

    acceptance = read_json(run_root / str(record.get('acceptance_relpath')))
    report = read_json(run_root / str(record.get('report_relpath')))
    if not isinstance(acceptance, dict) or acceptance.get('status') != 'PASS':
        reasons.append('acceptance not PASS')
    if not isinstance(report, dict) or report.get('phase') != 'M2_COMPLETE':
        reasons.append('report not M2_COMPLETE')
    if isinstance(report, dict) and report.get('certification_status') not in {
        None, 'NOT_CERTIFIED',
    }:
        reasons.append('bad certification_status')

    if require_source_match:
        if current_source_hash and current_source_hash != record.get('source_hash'):
            reasons.append('source_hash mismatch')
        if current_notebook_hash and current_notebook_hash != record.get('notebook_hash'):
            # Shared-registry token may differ from older run-notebook provenance.
            if (
                current_notebook_hash != SHARED_M2_NOTEBOOK_TOKEN
                and record.get('notebook_hash') != SHARED_M2_NOTEBOOK_TOKEN
            ):
                reasons.append('notebook_hash mismatch')

    status = 'PASS' if not reasons else 'FAIL'
    payload = {
        'schema_version': 1,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'status': status,
        'reasons': reasons,
        'canonical_run_id': record.get('canonical_run_id'),
        'registry_record_sha256': record.get('registry_record_sha256'),
        'verified_at': utc_now(),
    }
    entry = proof_entry_dir(persistent_root, structural_key, proof_key)
    atomic_write_json(entry / 'VERIFY.json', payload)
    if status != 'PASS':
        raise M2SharedRegistryError('verify failed: ' + '; '.join(reasons))
    return payload


def sync_child_run_ids_m2(package_root: Path, canonical_run_id: str) -> dict[str, Any]:
    """Ensure package child_run_ids.json includes M2 = canonical shared run id."""
    if not isinstance(canonical_run_id, str) or not canonical_run_id.strip():
        raise M2SharedRegistryError('canonical_run_id required to sync child_run_ids.M2')
    path = Path(package_root) / 'child_run_ids.json'
    child_ids = read_json(path) if path.is_file() else {}
    if not isinstance(child_ids, dict):
        child_ids = {}
    child_ids = dict(child_ids)
    child_ids['M2'] = canonical_run_id.strip()
    atomic_write_json(path, child_ids)
    return child_ids


def ensure_package_m2_run_id(package_root: Path) -> str:
    """Return M2 run id from child_run_ids or m2_binding; sync child_run_ids if needed."""
    package_root = Path(package_root)
    child_ids = read_json(package_root / 'child_run_ids.json') if (package_root / 'child_run_ids.json').is_file() else {}
    if isinstance(child_ids, dict) and isinstance(child_ids.get('M2'), str) and child_ids['M2'].strip():
        return child_ids['M2'].strip()
    binding = read_binding(package_root)
    if isinstance(binding, dict) and isinstance(binding.get('canonical_run_id'), str) and binding['canonical_run_id'].strip():
        sync_child_run_ids_m2(package_root, binding['canonical_run_id'])
        return binding['canonical_run_id'].strip()
    raise M2SharedRegistryError(
        'Package child_run_ids.M2 missing and m2_binding.canonical_run_id unset. '
        'Run resolve_m2_binding / notebook 82 promotion cell first.'
    )


def write_binding(package_root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    # READY_SHARED bindings are immutable.
    path = Path(package_root) / 'm2_binding.json'
    existing = read_json(path) if path.is_file() else None
    if (
        isinstance(existing, dict)
        and existing.get('state') == BINDING_READY
        and binding.get('state') != BINDING_READY
    ):
        raise M2SharedRegistryError('READY_SHARED binding is immutable')
    if (
        isinstance(existing, dict)
        and existing.get('state') == BINDING_READY
        and binding.get('state') == BINDING_READY
        and existing.get('registry_record_sha256')
        and binding.get('registry_record_sha256')
        and existing.get('registry_record_sha256') != binding.get('registry_record_sha256')
    ):
        raise M2SharedRegistryError('READY_SHARED registry record hash changed')
    atomic_write_json(path, binding)
    canonical = binding.get('canonical_run_id')
    if isinstance(canonical, str) and canonical.strip():
        sync_child_run_ids_m2(package_root, canonical)
    return binding


def read_binding(package_root: Path) -> dict[str, Any] | None:
    path = Path(package_root) / 'm2_binding.json'
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def resolve_m2_binding(
    *,
    persistent_root: Path,
    project_root: Path,
    package_root: Path,
    j2_max: int,
    notebook_hash: str = SHARED_M2_NOTEBOOK_TOKEN,
    owner_id: str | None = None,
) -> dict[str, Any]:
    """Resolve package binding against two-level registry."""
    source_hash = hash_tree(Path(project_root) / 'src')
    keys = keys_from_project(
        project_root,
        j2_max=j2_max,
        source_hash=source_hash,
        notebook_hash=notebook_hash,
    )
    structural_key = keys['structural_key']
    proof_key = keys['proof_key']
    record, hit = lookup_shared_m2_reusable(
        persistent_root,
        structural_key,
        proof_key,
        source_hash=source_hash,
    )
    owner = owner_id or Path(package_root).name

    if (
        record
        and record.get('registry_state') == STATE_COMPLETE
        and hit == 'structural_source_fallback'
    ):
        # Alias pre-token registrations under the shared notebook token proof key.
        record = register_shared_m2_from_run(
            persistent_root,
            Path(str(record['canonical_run_root'])),
            project_root=project_root,
            registration_mode=MODE_STRICT,
            allow_overwrite=True,
        )
        hit = 'exact'

    if record and record.get('registry_state') == STATE_COMPLETE:
        verify_shared_m2(
            persistent_root, structural_key, record.get('proof_key') or proof_key,
            require_source_match=True,
            current_source_hash=source_hash,
            current_notebook_hash=notebook_hash,
        )
        binding = {
            'schema_version': 2,
            'structural_key': structural_key,
            'proof_key': proof_key,
            'state': BINDING_READY,
            'mode': 'REUSE_SHARED',
            'canonical_run_id': record['canonical_run_id'],
            'registry_record_sha256': record.get('registry_record_sha256'),
            'acceptance_sha256': record.get('acceptance_sha256'),
            'verified_at': utc_now(),
            'lookup_hit': hit,
            'certification_status': 'NOT_CERTIFIED',
        }
        return write_binding(package_root, binding)

    if record and record.get('registry_state') in {
        STATE_RESERVED, STATE_RUNNING, STATE_LEGACY_PENDING, STATE_QUARANTINED,
    }:
        binding = {
            'schema_version': 2,
            'structural_key': structural_key,
            'proof_key': proof_key,
            'state': BINDING_WAITING,
            'mode': 'WAIT',
            'canonical_run_id': record.get('canonical_run_id') or keys['shared_run_id'],
            'registry_record_sha256': record.get('registry_record_sha256'),
            'acceptance_sha256': None,
            'verified_at': None,
            'registry_state': record.get('registry_state'),
            'lookup_hit': hit,
            'certification_status': 'NOT_CERTIFIED',
        }
        return write_binding(package_root, binding)

    binding = {
        'schema_version': 2,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'state': BINDING_NEED,
        'mode': 'NEED_CANONICAL_M2',
        'canonical_run_id': keys['shared_run_id'],
        'registry_record_sha256': None,
        'acceptance_sha256': None,
        'verified_at': None,
        'owner_hint': owner,
        'certification_status': 'NOT_CERTIFIED',
    }
    return write_binding(package_root, binding)


def backfill_shared_m2_from_package(
    persistent_root: Path,
    package_root: Path,
    *,
    project_root: Path | None = None,
    registration_mode: str = MODE_LEGACY,
) -> dict[str, Any]:
    child_ids = read_json(Path(package_root) / 'child_run_ids.json')
    if not isinstance(child_ids, dict) or not child_ids.get('M2'):
        raise M2SharedRegistryError('Package has no child_run_ids.M2')
    run_root = Path(persistent_root) / 'runs' / str(child_ids['M2'])
    record = register_shared_m2_from_run(
        persistent_root,
        run_root,
        project_root=project_root,
        registration_mode=registration_mode,
    )
    binding = {
        'schema_version': 2,
        'structural_key': record['structural_key'],
        'proof_key': record['proof_key'],
        'state': (
            BINDING_READY
            if record.get('registry_state') == STATE_COMPLETE
            else BINDING_WAITING
        ),
        'mode': (
            'REUSE_SHARED'
            if record.get('registry_state') == STATE_COMPLETE
            else 'LEGACY_BACKFILL'
        ),
        'canonical_run_id': record['canonical_run_id'],
        'registry_record_sha256': record.get('registry_record_sha256'),
        'acceptance_sha256': record.get('acceptance_sha256'),
        'verified_at': utc_now() if record.get('registry_state') == STATE_COMPLETE else None,
        'registry_state': record.get('registry_state'),
        'certification_status': 'NOT_CERTIFIED',
        'note': 'Legacy backfill; STRICT_PRODUCTION reuse requires reverify or new canonical run.',
    }
    write_binding(package_root, binding)
    return {'record': record, 'binding': binding}


def canonical_m2_run_id_for_package(package_root: Path) -> str | None:
    binding = read_binding(package_root)
    if isinstance(binding, dict) and binding.get('canonical_run_id'):
        return str(binding['canonical_run_id'])
    child_ids = read_json(Path(package_root) / 'child_run_ids.json')
    if isinstance(child_ids, dict) and child_ids.get('M2'):
        return str(child_ids['M2'])
    return None


# Back-compat aliases used by earlier wiring.
MODE_REUSE_SHARED = 'REUSE_SHARED'
MODE_NEED_CANONICAL = 'NEED_CANONICAL_M2'
read_m2_binding = read_binding


def resolve_m2_for_package(
    *,
    persistent_root: Path,
    j2_max: int,
    project_root: Path | None = None,
    package_root: Path | None = None,
) -> dict[str, Any]:
    if project_root is None or package_root is None:
        raise M2SharedRegistryError('project_root and package_root required')
    return resolve_m2_binding(
        persistent_root=persistent_root,
        project_root=project_root,
        package_root=package_root,
        j2_max=j2_max,
    )
