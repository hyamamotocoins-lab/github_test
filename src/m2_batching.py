"""Sector-batched M2 phase helpers for j2_max>=2 staged execution."""

from __future__ import annotations

import math
from typing import Any

from .cutoff_dims import expected_m2_gate_counts, sector_count

BATCHED_M2_PHASES = (
    'M2_DENSE_REFERENCE',
    'M2_ARMILLARY',
    'M2_EQUIVALENCE',
)
NON_BATCHED_M2_PHASES = (
    'M2_WIGNER_CACHE',
    'M2_SYMMETRY',
    'M2_REPORT',
)


def m2_batch_plan(j2_max: int, sector_batch_size: int) -> dict[str, Any]:
    total = sector_count(j2_max)
    if sector_batch_size <= 0:
        return {
            'batched': False,
            'total_sectors': total,
            'n_batches': 1,
            'sector_batch_size': total,
        }
    n_batches = int(math.ceil(total / sector_batch_size))
    return {
        'batched': True,
        'total_sectors': total,
        'n_batches': n_batches,
        'sector_batch_size': sector_batch_size,
    }


def predicted_batch_s(batch_size: int, *, per_sector_s: float = 45.0) -> float:
    """Conservative per-batch wall-time prediction under the 20-minute item cap."""
    raw = max(60.0, float(batch_size) * per_sector_s)
    return min(15.0 * 60.0, raw)


def proof_artifact_hash_map(queue_items: Any) -> dict[str, str]:
    """Stable proof-artifact map; batched phases use phase#bNNNN keys."""
    mapping: dict[str, str] = {}
    for item in queue_items:
        if item.status != 'done' or not item.result_sha256:
            continue
        batch = item.parameters.get('batch_index')
        if batch is None:
            mapping[item.phase] = item.result_sha256
        else:
            mapping[f'{item.phase}#b{int(batch):04d}'] = item.result_sha256
    return mapping


def merge_m2_batch_payloads(
    phase: str,
    batch_payloads: list[dict[str, Any]],
    *,
    j2_max: int,
) -> dict[str, Any]:
    """Merge per-batch phase artifacts into one acceptance-shaped payload."""
    if not batch_payloads:
        raise RuntimeError(f'No batch payloads to merge for {phase}')
    expected = expected_m2_gate_counts(j2_max)
    ordered = sorted(
        batch_payloads,
        key=lambda payload: int(
            (payload.get('result') or {}).get('batch_index', 0),
        ),
    )
    sectors: list[Any] = []
    residual = 0
    odd_zero = 0
    isometry = 0
    matches = 0
    mismatches: list[Any] = []
    max_dimension = 0
    tensor_count = 0
    for payload in ordered:
        if payload.get('phase') != phase:
            raise RuntimeError(f'Batch phase mismatch for {phase}')
        if payload.get('certification_status') != 'NOT_CERTIFIED':
            raise RuntimeError(f'Batch artifact changed certification: {phase}')
        result = payload.get('result')
        if not isinstance(result, dict) or result.get('status') != 'PASS':
            raise RuntimeError(f'Malformed batch result for {phase}')
        batch_sectors = result.get('sectors') or []
        if not isinstance(batch_sectors, list):
            raise RuntimeError(f'Batch sectors malformed for {phase}')
        sectors.extend(batch_sectors)
        residual += int(result.get('generator_residual_zero_count') or 0)
        odd_zero += int(result.get('odd_half_zero_count') or 0)
        isometry += int(result.get('isometry_exact_count') or 0)
        matches += int(result.get('exact_match_count') or 0)
        mismatches.extend(result.get('mismatches') or [])
        max_dimension = max(
            max_dimension, int(result.get('max_dense_dimension') or 0),
        )
        tensor_count += int(result.get('checkpoint_tensor_count') or 0)

    if len(sectors) != expected['sector_count']:
        raise RuntimeError(
            f'{phase} merged sector_count {len(sectors)} != '
            f"{expected['sector_count']}",
        )
    merged_result: dict[str, Any] = {
        'status': 'PASS',
        'sector_count': len(sectors),
        'sectors': sectors,
        'batched': True,
        'n_batches': len(ordered),
        'j2_max': j2_max,
    }
    if phase == 'M2_DENSE_REFERENCE':
        if residual != expected['generator_residual_zero_count']:
            raise RuntimeError('Merged dense residual gate failed closed.')
        if odd_zero != expected['odd_half_zero_count']:
            raise RuntimeError('Merged odd-half zero gate failed closed.')
        merged_result.update({
            'generator_residual_zero_count': residual,
            'odd_half_zero_count': odd_zero,
        })
    elif phase == 'M2_ARMILLARY':
        if isometry != expected['isometry_exact_count']:
            raise RuntimeError('Merged armillary isometry gate failed closed.')
        merged_result.update({
            'isometry_exact_count': isometry,
            'checkpoint_tensor_count': tensor_count,
        })
    elif phase == 'M2_EQUIVALENCE':
        if matches != expected['exact_match_count'] or mismatches:
            raise RuntimeError('Merged dense/armillary equivalence failed closed.')
        merged_result.update({
            'exact_match_count': matches,
            'mismatches': mismatches,
            'max_dense_dimension': max_dimension,
            'comparison': 'exact symbolic matrix equality',
        })
    else:
        raise RuntimeError(f'Unsupported batched M2 phase: {phase}')

    template = ordered[0]
    return {
        'schema_version': template.get('schema_version', 1),
        'milestone': 'M2',
        'phase': phase,
        'item_id': f'merged:{phase}',
        'config_hash': template.get('config_hash'),
        'certification_status': 'NOT_CERTIFIED',
        'generated_at': template.get('generated_at'),
        'result': merged_result,
        'merged_from_batches': [
            {
                'item_id': payload.get('item_id'),
                'batch_index': (payload.get('result') or {}).get('batch_index'),
            }
            for payload in ordered
        ],
    }
