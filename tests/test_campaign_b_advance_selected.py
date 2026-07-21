"""Tests for Campaign B SELECTED advancement (CPU lineage / fixture path)."""

from __future__ import annotations

import json
from pathlib import Path

from src.campaign_b.advance_selected import (
    discover_selected_packages,
    run_advance_selected,
)
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def _make_selected(root: Path, campaign: str, cand: str, *, q: float) -> Path:
    pkg = root / 'campaign_b' / campaign / 'selected' / cand
    pkg.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': cand,
        'scheme_hash': f'hash-{cand}',
        'scheme': {
            'change_class': 'S2',
            'target_rank': 16,
            'oversampling': 16,
            'power_iterations': 2,
            'seed': 1,
            'perron_weight_strategy': 'all_ones',
        },
    })
    atomic_write_json(pkg / 's0_result.json', {
        'q_upper': q,
        'status': 'SELECTED',
    })
    atomic_write_json(pkg / 'm2_binding.json', {
        'status': 'READY_SHARED',
        'mode': 'SHARED',
    })
    return pkg


def test_discover_and_advance_without_parent(tmp_path: Path) -> None:
    _make_selected(tmp_path, 'M7-TEST-b-aaa', 'CAND-low', q=0.91)
    _make_selected(tmp_path, 'M7-TEST-b-aaa', 'CAND-high', q=0.99)
    _make_selected(tmp_path, 'M7-TEST-b-bbb', 'CAND-other', q=0.95)

    found = discover_selected_packages(tmp_path)
    assert len(found) == 3

    session = run_advance_selected(
        persistent_root=tmp_path,
        only_campaign_run_id='M7-TEST-b-aaa',
        force=True,
    )
    assert session['certification_status'] == CERTIFICATION_STATUS
    assert session['claim_scope'] == CLAIM_SCOPE
    # discovered is post-filter when only_campaign_run_id is set
    assert session['discovered'] == 2
    assert session['attempted'] == 2
    assert session['advanced'] == 2
    assert session['parent_m6_package'] is None

    low = tmp_path / 'campaign_b' / 'M7-TEST-b-aaa' / 'selected' / 'CAND-low'
    advance = json.loads((low / 'ADVANCE.json').read_text(encoding='utf-8'))
    assert advance['status'] == 'READY_FOR_M3'
    assert advance['certification_status'] == CERTIFICATION_STATUS
    assert (low / 'lineage_plan.json').is_file()
    plan = json.loads((low / 'lineage_plan.json').read_text(encoding='utf-8'))
    assert 'M3' in plan['child_run_ids']
    assert 'production M6' in ' '.join(advance['prohibited'])

    ledger = tmp_path / 'campaign_b' / '_advance' / 'LATEST_ADVANCE_SESSION.json'
    assert ledger.is_file()


def test_skip_already_advanced(tmp_path: Path) -> None:
    pkg = _make_selected(tmp_path, 'M7-TEST-b-ccc', 'CAND-x', q=0.9)
    atomic_write_json(pkg / 'ADVANCE.json', {
        'status': 'LINEAGE_PLANNED',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    session = run_advance_selected(persistent_root=tmp_path, force=False)
    assert session['skipped'] == 1
    assert session['advanced'] == 0
