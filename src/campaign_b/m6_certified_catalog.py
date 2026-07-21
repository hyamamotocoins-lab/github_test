"""Persistent catalog of M6 CERTIFIED results (notebook 99).

Never invents CERTIFIED — only records what on-disk reports already claim.
Campaign B claim_scope may remain SCREENING_ONLY even when an orchestrator
report says CERTIFIED.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..common import (
    atomic_write_json,
    canonical_json_bytes,
    read_json,
    sha256_bytes,
    utc_now,
)
from .schemas import screening_only_payload

CATALOG_REL = Path('campaign_b') / '_m6_certified_catalog'
CATALOG_NAME = 'CATALOG.json'
NEW_SINCE_NAME = 'NEW_SINCE_LAST_SCAN.json'


def catalog_dir(persistent_root: Path) -> Path:
    return Path(persistent_root) / CATALOG_REL


def catalog_path(persistent_root: Path) -> Path:
    return catalog_dir(persistent_root) / CATALOG_NAME


def entry_id(*, run_id: str, package_path: str, report_path: str) -> str:
    payload = {
        'run_id': run_id,
        'package_path': package_path,
        'report_path': report_path,
    }
    return sha256_bytes(canonical_json_bytes(payload))[:32]


def _is_certified_value(value: Any) -> bool:
    return str(value or '').strip().upper() == 'CERTIFIED'


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        doc = read_json(path)
    except Exception:  # noqa: BLE001
        return None
    return doc if isinstance(doc, dict) else None


def _maybe_entry_from_report(
    *,
    persistent_root: Path,
    report_path: Path,
    status_key: str,
    source: str,
    package_path: str = '',
    run_id: str = '',
) -> dict[str, Any] | None:
    doc = _load_json(report_path)
    if doc is None:
        return None
    status = doc.get(status_key)
    if status is None and status_key == 'certification_status':
        # Some summaries nest under verdict.
        verdict = doc.get('verdict')
        if isinstance(verdict, dict):
            status = verdict.get('certification_status')
    if not _is_certified_value(status):
        return None
    rid = run_id or str(doc.get('run_id') or doc.get('m6_run_id') or report_path.parent.parent.name)
    pkg = package_path
    rel_report = str(report_path)
    try:
        rel_report = str(report_path.relative_to(persistent_root))
    except ValueError:
        pass
    eid = entry_id(run_id=str(rid), package_path=pkg, report_path=rel_report)
    return {
        'entry_id': eid,
        'run_id': str(rid),
        'package_path': pkg,
        'report_path': rel_report,
        'source': source,
        'certification_status': 'CERTIFIED',
        'claim_scope_note': (
            'Orchestrator/report claimed CERTIFIED; Campaign B claim_scope may '
            'still be SCREENING_ONLY until production gate 81 / human review.'
        ),
        'q_cert_upper': doc.get('q_cert_upper') or (
            (doc.get('verdict') or {}).get('q_cert_upper')
            if isinstance(doc.get('verdict'), dict) else None
        ),
        'phase': doc.get('phase') or doc.get('status'),
        'found_at': utc_now(),
    }


def scan_m6_certified_sources(persistent_root: Path) -> list[dict[str, Any]]:
    """Full rescan of persist for claimed CERTIFIED M6 results."""
    persistent_root = Path(persistent_root)
    found: dict[str, dict[str, Any]] = {}

    def _add(entry: dict[str, Any] | None) -> None:
        if entry is None:
            return
        found[entry['entry_id']] = entry

    runs = persistent_root / 'runs'
    if runs.is_dir():
        for run_dir in runs.iterdir():
            if not run_dir.is_dir() or not run_dir.name.startswith('M6-'):
                continue
            reports = run_dir / 'reports'
            if not reports.is_dir():
                continue
            for name in (
                'M6_report.json',
                'M6_acceptance.json',
                'session_summary.json',
            ):
                path = reports / name
                _add(_maybe_entry_from_report(
                    persistent_root=persistent_root,
                    report_path=path,
                    status_key='certification_status',
                    source=f'runs/{name}',
                    run_id=run_dir.name,
                ))

    # Campaign B selected packages + GPU/M6 markers.
    campaign = persistent_root / 'campaign_b'
    if campaign.is_dir():
        for path in campaign.rglob('M6_STATUS.json'):
            if 'selected' not in path.parts and '_m6' not in path.parts:
                # Still accept any campaign_b **/M6_STATUS.json
                pass
            doc = _load_json(path)
            if doc is None:
                continue
            status = doc.get('certification_status_m6') or doc.get('certification_status')
            if not _is_certified_value(status):
                continue
            try:
                rel = str(path.relative_to(persistent_root))
            except ValueError:
                rel = str(path)
            pkg = str(path.parent)
            try:
                pkg = str(path.parent.relative_to(persistent_root))
            except ValueError:
                pass
            rid = str(doc.get('m6_run_id') or doc.get('run_id') or '')
            eid = entry_id(run_id=rid, package_path=pkg, report_path=rel)
            found[eid] = {
                'entry_id': eid,
                'run_id': rid,
                'package_path': pkg,
                'report_path': rel,
                'source': 'M6_STATUS.json',
                'certification_status': 'CERTIFIED',
                'claim_scope_note': (
                    'Package M6_STATUS claimed CERTIFIED; Campaign B claim_scope '
                    'may still be SCREENING_ONLY.'
                ),
                'q_cert_upper': doc.get('q_cert_upper'),
                'phase': doc.get('phase') or doc.get('status'),
                'candidate_id': doc.get('candidate_id') or path.parent.name,
                'found_at': utc_now(),
            }

        # Also GPU_M3 sibling dirs sometimes store M6_GATE.
        for path in campaign.rglob('M6_GATE.json'):
            doc = _load_json(path)
            if doc is None:
                continue
            status = doc.get('certification_status_m6') or doc.get('certification_status')
            if not _is_certified_value(status):
                continue
            try:
                rel = str(path.relative_to(persistent_root))
                pkg = str(path.parent.relative_to(persistent_root))
            except ValueError:
                rel = str(path)
                pkg = str(path.parent)
            rid = str(doc.get('m6_run_id') or '')
            eid = entry_id(run_id=rid, package_path=pkg, report_path=rel)
            if eid not in found:
                found[eid] = {
                    'entry_id': eid,
                    'run_id': rid,
                    'package_path': pkg,
                    'report_path': rel,
                    'source': 'M6_GATE.json',
                    'certification_status': 'CERTIFIED',
                    'claim_scope_note': (
                        'M6_GATE claimed CERTIFIED; Campaign B claim_scope may '
                        'still be SCREENING_ONLY.'
                    ),
                    'q_cert_upper': doc.get('q_cert_upper'),
                    'phase': doc.get('phase') or doc.get('status'),
                    'found_at': utc_now(),
                }

        latest = campaign / '_m6' / 'LATEST_M6_SESSION.json'
        session = _load_json(latest)
        if isinstance(session, dict):
            for row in session.get('results') or []:
                if not isinstance(row, dict):
                    continue
                if not _is_certified_value(row.get('certification_status_m6')):
                    continue
                rid = str(row.get('m6_run_id') or '')
                pkg = str(row.get('candidate_id') or '')
                rel = f'campaign_b/_m6/LATEST_M6_SESSION.json#{rid}'
                eid = entry_id(run_id=rid, package_path=pkg, report_path=rel)
                if eid not in found:
                    found[eid] = {
                        'entry_id': eid,
                        'run_id': rid,
                        'package_path': pkg,
                        'report_path': rel,
                        'source': 'LATEST_M6_SESSION.json',
                        'certification_status': 'CERTIFIED',
                        'claim_scope_note': (
                            'Batch session list claimed CERTIFIED; verify against '
                            'runs/M6-*/reports before continuum claims. '
                            'Campaign B claim_scope may still be SCREENING_ONLY.'
                        ),
                        'q_cert_upper': row.get('q_cert_upper'),
                        'phase': row.get('phase') or row.get('status'),
                        'candidate_id': row.get('candidate_id'),
                        'found_at': utc_now(),
                    }

    return sorted(found.values(), key=lambda e: (e.get('run_id') or '', e['entry_id']))


def load_catalog(persistent_root: Path) -> dict[str, Any]:
    path = catalog_path(persistent_root)
    if not path.is_file():
        return {
            'schema_version': 1,
            'entries': [],
            'entry_ids': [],
            **screening_only_payload(),
        }
    doc = read_json(path)
    if not isinstance(doc, dict):
        return {
            'schema_version': 1,
            'entries': [],
            'entry_ids': [],
            **screening_only_payload(),
        }
    return doc


def scan_and_update_catalog(persistent_root: Path) -> dict[str, Any]:
    """Full rescan; merge into durable catalog; return all + newly_found."""
    persistent_root = Path(persistent_root)
    scanned_at = utc_now()
    discovered = scan_m6_certified_sources(persistent_root)
    previous = load_catalog(persistent_root)
    prev_ids = {
        str(e.get('entry_id'))
        for e in (previous.get('entries') or [])
        if isinstance(e, dict) and e.get('entry_id')
    }
    # Prefer previous entry metadata when id already known; add new ones.
    by_id: dict[str, dict[str, Any]] = {}
    for entry in previous.get('entries') or []:
        if isinstance(entry, dict) and entry.get('entry_id'):
            by_id[str(entry['entry_id'])] = entry
    newly: list[dict[str, Any]] = []
    for entry in discovered:
        eid = str(entry['entry_id'])
        if eid not in by_id:
            newly.append(entry)
            by_id[eid] = entry
        else:
            # Refresh report fields but keep original found_at if present.
            merged = dict(by_id[eid])
            orig_found = merged.get('found_at')
            merged.update(entry)
            if orig_found:
                merged['found_at'] = orig_found
                merged['last_seen_at'] = scanned_at
            by_id[eid] = merged

    all_certified = sorted(
        by_id.values(),
        key=lambda e: (e.get('run_id') or '', e.get('entry_id') or ''),
    )
    catalog = {
        'schema_version': 1,
        'updated_at': scanned_at,
        'total': len(all_certified),
        'entry_ids': [e['entry_id'] for e in all_certified],
        'entries': all_certified,
        'note': (
            'Durable merge of on-disk M6 CERTIFIED claims only. '
            'Never invents CERTIFIED. Campaign B claim_scope may remain '
            'SCREENING_ONLY even when certification_status is CERTIFIED.'
        ),
        **screening_only_payload(),
    }
    out_dir = catalog_dir(persistent_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(catalog_path(persistent_root), catalog)
    new_doc = {
        'schema_version': 1,
        'scanned_at': scanned_at,
        'newly_found_count': len(newly),
        'newly_found': newly,
        **screening_only_payload(),
    }
    atomic_write_json(out_dir / NEW_SINCE_NAME, new_doc)

    return {
        'all_certified': all_certified,
        'newly_found': newly,
        'total': len(all_certified),
        'scanned_at': scanned_at,
        'catalog_path': str(catalog_path(persistent_root)),
        **screening_only_payload(),
    }
