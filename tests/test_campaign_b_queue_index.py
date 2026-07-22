"""Tests for Campaign B queue indexes (CPU only)."""

from __future__ import annotations

import os
from pathlib import Path

from src.campaign_b.gpu_m3_batch import list_gpu_m3_queue
from src.campaign_b.pre_m6_batch import list_pre_m6_queue
from src.campaign_b.queue_index import (
    _index_path,
    rebuild_gpu_m3_index,
    rebuild_pre_m6_index,
)
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def _pkg(root: Path, campaign: str, cand: str, *, q: float) -> Path:
    pkg = root / 'campaign_b' / campaign / 'selected' / cand
    pkg.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': cand,
        'scheme': {'change_class': 'S2'},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': q})
    atomic_write_json(pkg / 'm2_binding.json', {'status': 'READY_SHARED'})
    atomic_write_json(pkg / 'ADVANCE.json', {
        'status': 'READY_FOR_M3',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    return pkg


def test_gpu_m3_index_early_exit_max_one(tmp_path: Path) -> None:
    for i in range(30):
        _pkg(tmp_path, 'M7-X', f'CAND-{i:02d}', q=0.5 + i * 0.01)
    rebuild_gpu_m3_index(tmp_path)
    idx = _index_path(tmp_path, 'gpu_m3_queue.json')
    assert idx.is_file()
    assert idx.stat().st_size < 50_000

    queue = list_gpu_m3_queue(tmp_path, max_candidates=1)
    assert len(queue) == 1
    # Path order among fresh READY (no q_upper ranking).
    assert queue[0]['candidate_id'] == 'CAND-00'
    assert queue[0]['q_upper'] is None


def test_gpu_m3_resume_outranks_fresh(tmp_path: Path) -> None:
    _pkg(tmp_path, 'M7-R', 'CAND-fresh', q=0.1)
    resume = _pkg(tmp_path, 'M7-R', 'CAND-resume', q=0.9)
    atomic_write_json(resume / 'GPU_M3.json', {
        'status': 'M3_CHECKPOINT',
        'consecutive_failures': 0,
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    rebuild_gpu_m3_index(tmp_path)
    queue = list_gpu_m3_queue(tmp_path, max_candidates=2)
    assert queue[0]['candidate_id'] == 'CAND-resume'
    assert queue[1]['candidate_id'] == 'CAND-fresh'


def test_gpu_m3_index_updates_on_status_write(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, 'M7-Y', 'CAND-a', q=0.9)
    _pkg(tmp_path, 'M7-Y', 'CAND-b', q=0.8)
    rebuild_gpu_m3_index(tmp_path)

    atomic_write_json(pkg / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    from src.campaign_b.gpu_m3_batch import _write_gpu_status

    _write_gpu_status(pkg, {'status': 'M3_COMPLETE'}, persistent_root=tmp_path)

    queue = list_gpu_m3_queue(tmp_path, max_candidates=5)
    ids = [r['candidate_id'] for r in queue]
    assert 'CAND-a' not in ids
    assert 'CAND-b' in ids


def test_pre_m6_index_lists_m3_complete(tmp_path: Path) -> None:
    pkg = _pkg(tmp_path, 'M7-Z', 'CAND-m3', q=0.7)
    atomic_write_json(pkg / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M3': 'M3-TEST-001',
        'M4': 'M4-TEST-001',
        'M5': 'M5-TEST-001',
    })
    runs = tmp_path / 'runs' / 'M3-TEST-001' / 'reports'
    runs.mkdir(parents=True)
    atomic_write_json(runs / 'M3_report.json', {'phase': 'M3_COMPLETE'})

    rebuild_pre_m6_index(tmp_path)
    queue = list_pre_m6_queue(tmp_path, max_candidates=1)
    assert len(queue) == 1
    assert queue[0]['candidate_id'] == 'CAND-m3'
    assert queue[0]['stage'] == 'NEED_M4'


def test_disable_queue_index_falls_back_to_scan(tmp_path: Path, monkeypatch) -> None:
    _pkg(tmp_path, 'M7-D', 'CAND-one', q=0.6)
    rebuild_gpu_m3_index(tmp_path)
    monkeypatch.setenv('VALIDATED_RG_DISABLE_QUEUE_INDEX', '1')
    queue = list_gpu_m3_queue(tmp_path, max_candidates=1)
    assert len(queue) == 1
    monkeypatch.delenv('VALIDATED_RG_DISABLE_QUEUE_INDEX', raising=False)
