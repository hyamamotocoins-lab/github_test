"""CPU unit tests for notebook 99 M6 CERTIFIED durable catalog."""

from __future__ import annotations

import json
from pathlib import Path

from src.campaign_b.m6_certified_catalog import (
    catalog_path,
    scan_and_update_catalog,
)
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, payload)


def test_catalog_finds_certified_ignores_not_certified(tmp_path: Path) -> None:
    # CERTIFIED run report
    _write(tmp_path / 'runs' / 'M6-CERT-1' / 'reports' / 'M6_report.json', {
        'run_id': 'M6-CERT-1',
        'phase': 'M6_COMPLETE',
        'certification_status': 'CERTIFIED',
        'verdict': {'q_cert_upper': '0.9'},
    })
    # NOT_CERTIFIED must not appear
    _write(tmp_path / 'runs' / 'M6-FAIL-1' / 'reports' / 'M6_report.json', {
        'run_id': 'M6-FAIL-1',
        'phase': 'M6_COMPLETE',
        'certification_status': 'NOT_CERTIFIED',
    })
    # Package M6_STATUS CERTIFIED
    pkg = (
        tmp_path / 'campaign_b' / 'M7-demo' / 'selected' / 'B-aaa'
    )
    _write(pkg / 'M6_STATUS.json', {
        'status': 'M6_COMPLETE',
        'm6_run_id': 'M6-CERT-PKG',
        'certification_status_m6': 'CERTIFIED',
        'q_cert_upper': '0.5',
        'candidate_id': 'B-aaa',
    })
    # Package NOT_CERTIFIED
    pkg2 = tmp_path / 'campaign_b' / 'M7-demo' / 'selected' / 'B-bbb'
    _write(pkg2 / 'M6_STATUS.json', {
        'status': 'M6_COMPLETE',
        'm6_run_id': 'M6-FAIL-PKG',
        'certification_status_m6': 'NOT_CERTIFIED',
    })
    # Session list with one CERTIFIED
    _write(tmp_path / 'campaign_b' / '_m6' / 'LATEST_M6_SESSION.json', {
        'results': [
            {
                'candidate_id': 'B-ccc',
                'm6_run_id': 'M6-CERT-SESS',
                'certification_status_m6': 'CERTIFIED',
                'status': 'M6_COMPLETE',
            },
            {
                'candidate_id': 'B-ddd',
                'm6_run_id': 'M6-FAIL-SESS',
                'certification_status_m6': 'NOT_CERTIFIED',
            },
        ],
    })

    result = scan_and_update_catalog(tmp_path)
    assert result['total'] >= 3
    assert result['certification_status'] == CERTIFICATION_STATUS
    assert result['claim_scope'] == CLAIM_SCOPE
    run_ids = {e['run_id'] for e in result['all_certified']}
    assert 'M6-CERT-1' in run_ids
    assert 'M6-CERT-PKG' in run_ids
    assert 'M6-CERT-SESS' in run_ids
    assert 'M6-FAIL-1' not in run_ids
    assert 'M6-FAIL-PKG' not in run_ids
    assert 'M6-FAIL-SESS' not in run_ids
    assert catalog_path(tmp_path).is_file()

    # Second scan: no new if unchanged; total stable
    result2 = scan_and_update_catalog(tmp_path)
    assert result2['total'] == result['total']
    assert result2['newly_found'] == []

    # New CERTIFIED appears → newly_found
    _write(tmp_path / 'runs' / 'M6-CERT-2' / 'reports' / 'M6_acceptance.json', {
        'run_id': 'M6-CERT-2',
        'certification_status': 'CERTIFIED',
        'phase': 'M6_COMPLETE',
    })
    result3 = scan_and_update_catalog(tmp_path)
    assert result3['total'] == result['total'] + 1
    assert any(e['run_id'] == 'M6-CERT-2' for e in result3['newly_found'])
    # Durable catalog on disk
    catalog = json.loads(catalog_path(tmp_path).read_text(encoding='utf-8'))
    assert catalog['total'] == result3['total']
    assert all(e['certification_status'] == 'CERTIFIED' for e in catalog['entries'])


def test_never_invents_certified_from_empty(tmp_path: Path) -> None:
    result = scan_and_update_catalog(tmp_path)
    assert result['total'] == 0
    assert result['all_certified'] == []
    assert result['newly_found'] == []
