from __future__ import annotations

from pathlib import Path

from src.cutoff_dims import expected_m2_gate_counts, odd_sum_sector_count
from src.m2_batching import m2_batch_plan, merge_m2_batch_payloads, predicted_batch_s
from src.m2_config import M2Config
from src.m7_staged_lineage import inspect_staged_m2_progress


def test_inspect_staged_m2_missing_run(tmp_path: Path) -> None:
    info = inspect_staged_m2_progress(tmp_path, run_id='M2-missing')
    assert info['exists'] is False
    assert info['run_id'] == 'M2-missing'


def test_odd_sum_and_expected_gates_j2_1() -> None:
    assert odd_sum_sector_count(1) == 32
    gates = expected_m2_gate_counts(1)
    assert gates['sector_count'] == 64
    assert gates['odd_half_zero_count'] == 32


def test_odd_sum_j2_2_is_half_of_sectors() -> None:
    # (j2_max+1)^6 = 729 is odd, so odd/even partition is not equal;
    # still must be deterministic and positive.
    odd = odd_sum_sector_count(2)
    assert odd == 364  # verified enumeration for j2_max=2
    gates = expected_m2_gate_counts(2)
    assert gates['sector_count'] == 729
    assert gates['odd_half_zero_count'] == odd


def test_m2_config_requires_batch_for_j2_gt_1() -> None:
    try:
        M2Config(j2_max=2, sector_batch_size=0)
        raised = False
    except ValueError:
        raised = True
    assert raised
    cfg = M2Config(j2_max=2, sector_batch_size=16)
    assert cfg.sector_batch_size == 16


def test_batch_plan_and_merge_dense() -> None:
    plan = m2_batch_plan(2, 200)
    assert plan['batched'] is True
    assert plan['n_batches'] == 4  # ceil(729/200)
    assert predicted_batch_s(16) <= 15 * 60

    payloads = []
    remaining = 729
    for batch_index in range(plan['n_batches']):
        size = min(200, remaining)
        remaining -= size
        # Fabricate passing partial dense batch (zeros filled at merge check).
        # Use exact residual/odd counts only on final merge via controlled totals.
        payloads.append({
            'schema_version': 1,
            'milestone': 'M2',
            'phase': 'M2_DENSE_REFERENCE',
            'item_id': f'b{batch_index}',
            'config_hash': 'a' * 64,
            'certification_status': 'NOT_CERTIFIED',
            'generated_at': '2026-07-20T00:00:00Z',
            'result': {
                'status': 'PASS',
                'sector_count': size,
                'generator_residual_zero_count': size,
                'odd_half_zero_count': (
                    364 if batch_index == 0 else 0
                ) if False else 0,
                'batch_index': batch_index,
                'sectors': [{'reps': [0] * 6}] * size,
                'partial': True,
            },
        })
    # Fix odd_half totals to exactly 364 across batches.
    payloads[0]['result']['odd_half_zero_count'] = 364
    for payload in payloads[1:]:
        payload['result']['odd_half_zero_count'] = 0
    merged = merge_m2_batch_payloads(
        'M2_DENSE_REFERENCE', payloads, j2_max=2,
    )
    assert merged['result']['sector_count'] == 729
    assert merged['result']['generator_residual_zero_count'] == 729
    assert merged['result']['odd_half_zero_count'] == 364
