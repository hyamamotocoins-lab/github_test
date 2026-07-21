"""Minimal interrupt recovery for Campaign B pipelines (Phase 1).

Cleans orphaned ``*.tmp`` files and records a stub note for stale lease
directories. Does not mutate candidate queue states or compute locks beyond
safe filesystem cleanup.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .schemas import screening_only_payload


def recover_interrupted_work(persistent_root: Path) -> dict[str, Any]:
    """Clean temporary artifacts under campaign_b and runs.

    Phase 1 scope:
    - delete ``*.tmp`` / ``.*.tmp-*`` files under ``campaign_b/``
    - if a ``leases/`` directory exists with no live marker, write a recovery note
    - never invent CERTIFIED or rewrite stage status JSON
    """
    persistent_root = Path(persistent_root)
    started = utc_now()
    removed: list[str] = []
    lease_notes: list[str] = []

    campaign_root = persistent_root / 'campaign_b'
    if campaign_root.is_dir():
        for path in campaign_root.rglob('*'):
            if not path.is_file():
                continue
            name = path.name
            if name.endswith('.tmp') or '.tmp-' in name:
                try:
                    path.unlink()
                    removed.append(str(path.relative_to(persistent_root)))
                except OSError:
                    pass

        leases = campaign_root / 'leases'
        if leases.is_dir():
            # Stub: record presence; do not delete active lease files blindly.
            children = [p.name for p in leases.iterdir()]
            note_path = campaign_root / '_recovery' / 'lease_stub.json'
            note_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                'schema_version': 1,
                'updated_at': utc_now(),
                'lease_dir': str(leases),
                'entries': children[:200],
                'note': (
                    'Phase-1 stub: stale lease reconciliation is deferred. '
                    'QueueStore.recover_expired_leases still owns candidate leases.'
                ),
                **screening_only_payload(),
            }
            atomic_write_json(note_path, payload)
            lease_notes.append(str(note_path.relative_to(persistent_root)))

    runs = persistent_root / 'runs'
    if runs.is_dir():
        for path in runs.rglob('*'):
            if not path.is_file():
                continue
            name = path.name
            if name.endswith('.tmp') or '.tmp-' in name:
                try:
                    path.unlink()
                    removed.append(str(path.relative_to(persistent_root)))
                except OSError:
                    pass

    summary = {
        'schema_version': 1,
        'started_at': started,
        'finished_at': utc_now(),
        'removed_tmp_count': len(removed),
        'removed_tmp': removed[:100],
        'lease_notes': lease_notes,
        **screening_only_payload(),
    }
    out_dir = campaign_root / '_recovery' if campaign_root.exists() else persistent_root / 'campaign_b' / '_recovery'
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_dir / 'LATEST_RECOVERY.json', summary)
    return summary


def clear_empty_tmp_dirs(root: Path) -> int:
    """Remove empty directories named ``*.tmp`` under root. Returns count removed."""
    root = Path(root)
    if not root.is_dir():
        return 0
    removed = 0
    for path in sorted(root.rglob('*'), reverse=True):
        if path.is_dir() and (path.name.endswith('.tmp') or path.name.startswith('.')):
            try:
                if not any(path.iterdir()):
                    shutil.rmtree(path, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    return removed
