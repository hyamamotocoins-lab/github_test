"""Tests for Campaign B GPU M3 queue ranking (no CUDA required)."""

from __future__ import annotations

from pathlib import Path

from src.campaign_b.gpu_m3_batch import list_gpu_m3_queue
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json


def _pkg(root: Path, campaign: str, cand: str, *, q: float, ready: bool = True) -> Path:
    pkg = root / 'campaign_b' / campaign / 'selected' / cand
    pkg.mkdir(parents=True, exist_ok=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': cand,
        'j2': 2,
        'execution_mode': 'staged',
        'structural_key': 'sk',
        'proof_key': 'pk',
        'scheme': {
            'change_class': 'S2',
            'target_rank': 16,
            'oversampling': 16,
            'power_iterations': 2,
            'seed': 1,
        },
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': q})
    if ready:
        atomic_write_json(pkg / 'm2_binding.json', {
            'status': 'READY_SHARED',
            'canonical_run_id': 'M2-TEST',
            'structural_key': 'sk',
            'proof_key': 'pk',
        })
        atomic_write_json(pkg / 'ADVANCE.json', {
            'status': 'READY_FOR_M3',
            'certification_status': CERTIFICATION_STATUS,
            'claim_scope': CLAIM_SCOPE,
        })
    return pkg


def test_list_gpu_m3_queue_ranks_by_q(tmp_path: Path) -> None:
    _pkg(tmp_path, 'M7-A', 'CAND-hi', q=0.95)
    _pkg(tmp_path, 'M7-A', 'CAND-lo', q=0.81)
    _pkg(tmp_path, 'M7-A', 'CAND-skip', q=0.70, ready=False)
    done = _pkg(tmp_path, 'M7-A', 'CAND-done', q=0.75)
    atomic_write_json(done / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })

    queue = list_gpu_m3_queue(tmp_path, max_candidates=10)
    assert [r['candidate_id'] for r in queue] == ['CAND-lo', 'CAND-hi']
    assert queue[0]['q_upper'] == 0.81


def test_list_gpu_m3_queue_excludes_error_and_nonfinite(tmp_path: Path) -> None:
    _pkg(tmp_path, 'M7-B', 'CAND-fresh', q=0.90)
    err = _pkg(tmp_path, 'M7-B', 'CAND-err', q=0.50)
    nf = _pkg(tmp_path, 'M7-B', 'CAND-nf', q=0.40)
    resume = _pkg(tmp_path, 'M7-B', 'CAND-resume', q=0.70)
    atomic_write_json(err / 'GPU_M3.json', {
        'status': 'M3_ERROR',
        'error': 'ValueError: Out of range float values are not JSON compliant',
        'consecutive_failures': 1,
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    atomic_write_json(nf / 'GPU_M3.json', {
        'status': 'M3_BLOCKED_NONFINITE',
        'nonfinite_values_present': True,
        'consecutive_failures': 1,
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    atomic_write_json(resume / 'GPU_M3.json', {
        'status': 'M3_CHECKPOINT',
        'consecutive_failures': 0,
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })

    queue = list_gpu_m3_queue(tmp_path, max_candidates=20)
    ids = [r['candidate_id'] for r in queue]
    assert 'CAND-err' not in ids
    assert 'CAND-nf' not in ids
    assert ids[0] == 'CAND-resume'
    assert 'CAND-fresh' in ids

    with_err = list_gpu_m3_queue(
        tmp_path, max_candidates=20, include_errors=True,
    )
    with_ids = [r['candidate_id'] for r in with_err]
    assert 'CAND-err' in with_ids
    assert 'CAND-nf' in with_ids
    assert with_ids.index('CAND-resume') < with_ids.index('CAND-err')
