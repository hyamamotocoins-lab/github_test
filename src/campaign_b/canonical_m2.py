"""Auto-prepare canonical shared M2 (j2>=2 staged) for Campaign B.

Mirrors the operator path notebook 84 resolve + notebook 73 staged generation,
without requiring a Campaign C package: the shared M2 is keyed by structural /
proof keys from M1 + j2_max=2, same as Campaign C staged shared M2.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, hash_tree, utc_now
from ..m2_compatibility import SHARED_M2_NOTEBOOK_TOKEN, keys_from_project
from ..m2_shared_registry import (
    BINDING_READY,
    MODE_STRICT,
    STATE_COMPLETE,
    alias_shared_m2_under_proof_key,
    find_complete_shared_run_on_disk,
    heartbeat_reservation,
    lookup_shared_m2_reusable,
    register_shared_m2_from_run,
    reserve_shared_m2,
)
from ..m7_staged_lineage import run_staged_m2_session
from .errors import NeedCanonicalM2
from .schemas import screening_only_payload


def _binding_from_record(
    record: dict[str, Any],
    *,
    structural_key: str,
    proof_key: str,
    source_tree_hash: str,
    how: str | None,
) -> dict[str, Any]:
    source_match = record.get('source_hash') == source_tree_hash
    return {
        'schema_version': 1,
        'status': BINDING_READY,
        'binding_status': BINDING_READY,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'canonical_run_id': record.get('canonical_run_id') or record.get('run_id'),
        'canonical_package_dir': (
            record.get('canonical_package_dir') or record.get('package_dir')
        ),
        'registry_state': record.get('registry_state'),
        'source_hash': record.get('source_hash'),
        'requested_source_hash': source_tree_hash,
        'reuse_class': (
            'EXACT_SOURCE_MATCH' if source_match else 'AUDITED_SOURCE_DRIFT_REUSE'
        ),
        'lookup_how': how,
        'registry_record_sha256': record.get('registry_record_sha256'),
        'resolved_at': utc_now(),
        **screening_only_payload(),
    }


def ensure_canonical_shared_m2(
    *,
    persistent_root: Path,
    project_root: Path,
    source_tree_hash: str | None = None,
    j2_max: int = 2,
    owner_id: str = 'campaign_b_auto',
    max_sessions: int = 10_000,
    progress_dir: Path | None = None,
    sector_batch_size: int | None = None,
) -> dict[str, Any]:
    """Lookup or generate+register canonical shared M2 for staged j2>=2.

    Returns a READY binding dict, or raises NeedCanonicalM2 if generation cannot
    proceed (e.g. j2=1, resource gate blocked).
    """
    if int(j2_max) < 2:
        raise NeedCanonicalM2('auto canonical shared M2 requires j2_max>=2')

    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    source_hash = source_tree_hash or hash_tree(project_root / 'src')
    keys = keys_from_project(
        project_root,
        j2_max=int(j2_max),
        source_hash=source_hash,
        notebook_hash=SHARED_M2_NOTEBOOK_TOKEN,
    )
    structural_key = str(keys['structural_key'])
    proof_key = str(keys['proof_key'])
    run_id = str(keys['shared_run_id'])

    progress_root = Path(progress_dir or (persistent_root / 'campaign_b' / '_auto_m2'))
    progress_root.mkdir(parents=True, exist_ok=True)

    def _lookup() -> tuple[dict[str, Any] | None, str | None]:
        record, how = lookup_shared_m2_reusable(
            persistent_root,
            structural_key,
            proof_key,
            source_hash=source_hash,
        )
        if record and record.get('registry_state') == STATE_COMPLETE:
            return record, how
        on_disk = find_complete_shared_run_on_disk(persistent_root, structural_key)
        if on_disk is not None:
            record = register_shared_m2_from_run(
                persistent_root,
                on_disk,
                project_root=project_root,
                registration_mode=MODE_STRICT,
                allow_overwrite=True,
            )
            record = alias_shared_m2_under_proof_key(
                persistent_root,
                record,
                structural_key=structural_key,
                proof_key=proof_key,
            )
            return record, 'disk_complete_adopt'
        return None, None

    record, how = _lookup()
    if record is not None:
        return _binding_from_record(
            record,
            structural_key=structural_key,
            proof_key=proof_key,
            source_tree_hash=source_hash,
            how=how,
        )

    # Reserve then drive staged sessions until COMPLETE (same as 73 loop).
    try:
        reserve_shared_m2(
            persistent_root,
            structural_key,
            proof_key,
            owner_id=owner_id,
            canonical_run_id=run_id,
        )
    except Exception as exc:
        # Another owner may hold the lease — wait and re-lookup.
        atomic_write_json(progress_root / 'waiting.json', {
            'at': utc_now(),
            'error': str(exc),
            'run_id': run_id,
            **screening_only_payload(),
        })
        for _ in range(60):
            time.sleep(30)
            record, how = _lookup()
            if record is not None:
                return _binding_from_record(
                    record,
                    structural_key=structural_key,
                    proof_key=proof_key,
                    source_tree_hash=source_hash,
                    how=how or 'waited_for_other_owner',
                )
        raise NeedCanonicalM2(
            f'could not reserve shared M2 {run_id}; other owner still holding'
        ) from exc

    last_summary: dict[str, Any] | None = None
    for session_index in range(int(max_sessions)):
        heartbeat_reservation(
            persistent_root, structural_key, proof_key, owner_id=owner_id,
        )
        summary = run_staged_m2_session(
            persistent_root=persistent_root,
            project_root=project_root,
            j2_max=int(j2_max),
            run_id=run_id,
            sector_batch_size=sector_batch_size,
        )
        last_summary = summary
        atomic_write_json(progress_root / 'session_progress.json', {
            'session_index': session_index,
            'run_id': run_id,
            'm2_complete': bool(summary.get('m2_complete')),
            'at': utc_now(),
            'progress': summary.get('progress'),
            **screening_only_payload(),
        })
        if summary.get('m2_complete'):
            record = register_shared_m2_from_run(
                persistent_root,
                Path(summary['run_root']),
                project_root=project_root,
                registration_mode=MODE_STRICT,
                allow_overwrite=True,
            )
            record = alias_shared_m2_under_proof_key(
                persistent_root,
                record,
                structural_key=structural_key,
                proof_key=proof_key,
            )
            binding = _binding_from_record(
                record,
                structural_key=structural_key,
                proof_key=proof_key,
                source_tree_hash=source_hash,
                how='auto_generated_staged',
            )
            binding['generated'] = True
            binding['sessions'] = session_index + 1
            atomic_write_json(progress_root / 'complete.json', binding)
            return binding

    raise NeedCanonicalM2(
        f'staged M2 {run_id} not complete after {max_sessions} sessions; '
        f'last={last_summary}'
    )
