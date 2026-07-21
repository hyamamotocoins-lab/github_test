"""Read-only Campaign B pipeline status (notebook 98).

Never mutates compute state, locks, leases, or queue candidate statuses.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..common import read_json, utc_now
from .schemas import screening_only_payload


def _safe_list_len(fn: Any, *args: Any, **kwargs: Any) -> int | None:
    try:
        return len(fn(*args, **kwargs))
    except Exception:  # noqa: BLE001 — status must never fail closed on probe errors
        return None


def _count_candidate_states(persistent_root: Path) -> dict[str, int]:
    root = Path(persistent_root) / 'campaign_b'
    counts: Counter[str] = Counter()
    if not root.is_dir():
        return {}
    for campaign in root.iterdir():
        if not campaign.is_dir() or campaign.name.startswith('_'):
            continue
        queue_path = campaign / 'queue.json'
        if not queue_path.is_file():
            continue
        try:
            queue = read_json(queue_path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(queue, dict):
            continue
        for cand in queue.get('candidates') or []:
            if isinstance(cand, dict):
                counts[str(cand.get('state') or 'UNKNOWN')] += 1
    return dict(sorted(counts.items()))


def _stale_lease_probe(persistent_root: Path) -> dict[str, Any]:
    """Count expired candidate leases without recovering them."""
    root = Path(persistent_root) / 'campaign_b'
    stale = 0
    active = 0
    samples: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc)
    if not root.is_dir():
        return {'stale': 0, 'active': 0, 'samples': []}
    for campaign in root.iterdir():
        if not campaign.is_dir() or campaign.name.startswith('_'):
            continue
        queue_path = campaign / 'queue.json'
        if not queue_path.is_file():
            continue
        try:
            queue = read_json(queue_path)
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(queue, dict):
            continue
        for cand in queue.get('candidates') or []:
            if not isinstance(cand, dict):
                continue
            lease = cand.get('lease')
            if not isinstance(lease, dict):
                continue
            expires = lease.get('lease_expires_at')
            active += 1
            expired = False
            if isinstance(expires, str):
                try:
                    exp = datetime.fromisoformat(expires.replace('Z', '+00:00'))
                    expired = exp <= now
                except ValueError:
                    expired = False
            if expired:
                stale += 1
                if len(samples) < 20:
                    samples.append({
                        'campaign': campaign.name,
                        'candidate_id': cand.get('candidate_id') or cand.get('id'),
                        'state': cand.get('state'),
                        'lease_expires_at': expires,
                    })
    return {'stale': stale, 'active': active, 'samples': samples}


def _m2_summary(persistent_root: Path) -> dict[str, Any]:
    from .post_m2_pipeline import find_m2_ready_markers

    ready = find_m2_ready_markers(persistent_root)
    runs = Path(persistent_root) / 'runs'
    m2_dirs = 0
    if runs.is_dir():
        m2_dirs = sum(
            1 for p in runs.iterdir()
            if p.is_dir() and p.name.startswith('M2-')
        )
    return {
        'm2_run_dirs': m2_dirs,
        'm2_ready_count': len(ready),
        'm2_ready': ready[:20],
    }


def _selected_package_counts(persistent_root: Path) -> dict[str, Any]:
    from .advance_selected import discover_selected_packages
    from .gpu_m3_batch import _gpu_status

    packages = discover_selected_packages(persistent_root)
    gpu_counts: Counter[str] = Counter()
    for package in packages:
        status = _gpu_status(package) or 'NO_GPU_M3'
        gpu_counts[status] += 1
        m6 = package / 'M6_STATUS.json'
        if m6.is_file():
            try:
                doc = read_json(m6)
                if isinstance(doc, dict):
                    gpu_counts[f"M6:{doc.get('status') or 'UNKNOWN'}"] += 1
                    cert = doc.get('certification_status_m6')
                    if cert:
                        gpu_counts[f'M6_CERT:{cert}'] += 1
            except Exception:  # noqa: BLE001
                pass
    return {
        'selected_packages': len(packages),
        'gpu_m3_and_m6_status_counts': dict(sorted(gpu_counts.items())),
    }


def _maybe_write_status_snapshot(
    persistent_root: Path,
    status: dict[str, Any],
    *,
    write_status_snapshot: bool,
) -> str | None:
    """Optionally persist a status JSON under campaign_b/_status_dashboard/.

    Default is off so notebook 98 stays usable when disk quota blocks writes.
    """
    if not write_status_snapshot:
        return None
    from ..common import atomic_write_json

    out_dir = Path(persistent_root) / 'campaign_b' / '_status_dashboard'
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = str(status.get('scanned_at') or utc_now()).replace(':', '').replace('+', '_')
    out_path = out_dir / f'status_{stamp}.json'
    atomic_write_json(out_path, status)
    atomic_write_json(out_dir / 'LATEST_STATUS.json', status)
    return str(out_path)


def collect_pipeline_status(
    persistent_root: Path,
    *,
    write_status_snapshot: bool = False,
) -> dict[str, Any]:
    """Assemble a read-only status snapshot.

    Never mutates compute state, locks, leases, or queues. By default performs
    no writes under ``persistent_root`` (safe under disk quota). Set
    ``write_status_snapshot=True`` only to optionally dump under
    ``campaign_b/_status_dashboard/``.
    """
    persistent_root = Path(persistent_root)
    from .gpu_m3_batch import list_gpu_m3_queue
    from .m6_batch import list_m6_queue
    from .pre_m6_batch import list_pre_m6_queue
    from .close_obligations import list_obligation_queue

    queues = {
        'gpu_m3': _safe_list_len(list_gpu_m3_queue, persistent_root, max_candidates=5000),
        'pre_m6': _safe_list_len(list_pre_m6_queue, persistent_root, max_candidates=5000),
        'obligations': _safe_list_len(
            list_obligation_queue, persistent_root, max_candidates=5000,
        ),
        'm6': _safe_list_len(list_m6_queue, persistent_root, max_candidates=5000),
    }

    # Latest session pointers (read only).
    pointers: dict[str, Any] = {}
    for rel in (
        'campaign_b/_end_to_end/LATEST_END_TO_END_SESSION.json',
        'campaign_b/_post_m2/LATEST_POST_M2_SESSION.json',
        'campaign_b/_pipeline_to_m6/LATEST_PIPELINE_SESSION.json',
        'campaign_b/_mass_explore/LATEST_MASS_SESSION.json',
        'campaign_b/_m6/LATEST_M6_SESSION.json',
        'campaign_b/_m6_certified_catalog/CATALOG.json',
    ):
        path = persistent_root / rel
        if path.is_file():
            try:
                pointers[rel] = read_json(path)
            except Exception as exc:  # noqa: BLE001
                pointers[rel] = {'error': str(exc)}

    status = {
        'schema_version': 1,
        'scanned_at': utc_now(),
        'persistent_root': str(persistent_root),
        'read_only': True,
        'write_status_snapshot': bool(write_status_snapshot),
        'm2': _m2_summary(persistent_root),
        'candidate_states': _count_candidate_states(persistent_root),
        'queues': queues,
        'selected': _selected_package_counts(persistent_root),
        'leases': _stale_lease_probe(persistent_root),
        'session_pointers': {
            k: {
                'session_id': (v.get('session_id') if isinstance(v, dict) else None),
                'finished_at': (v.get('finished_at') if isinstance(v, dict) else None),
                'totals': (v.get('totals') if isinstance(v, dict) else None),
                'total': (v.get('total') if isinstance(v, dict) else None),
                'error': (v.get('error') if isinstance(v, dict) else None),
            }
            for k, v in pointers.items()
        },
        'waiting_for_m2_note': (
            'Driver WAITING_FOR_M2 reconciler is TODO; counts may appear under '
            'NEED_CANONICAL_M2 instead.'
        ),
        'note': (
            'Notebook 98 read-only dashboard. No locks, leases, or statuses mutated. '
            'No validate_persistent_root write probe; optional _status_dashboard '
            'snapshot is off by default.'
        ),
        **screening_only_payload(),
    }
    snapshot_path = _maybe_write_status_snapshot(
        persistent_root,
        status,
        write_status_snapshot=write_status_snapshot,
    )
    if snapshot_path is not None:
        status['status_snapshot_path'] = snapshot_path
    return status


def format_status_text(status: dict[str, Any]) -> str:
    """Human-readable text block for notebook display (no filesystem writes)."""
    return json.dumps(status, indent=2, ensure_ascii=False, default=str)
