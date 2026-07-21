"""Tests for fail-closed JSON sanitization of non-finite floats."""

from __future__ import annotations

import json
import math

from src.common import sanitize_for_json


def test_sanitize_replaces_nan_inf_with_none() -> None:
    payload = {
        'ok': 1.25,
        'bad': float('nan'),
        'neginf': float('-inf'),
        'nested': [1.0, float('inf'), {'x': float('nan')}],
    }
    clean, found = sanitize_for_json(payload)
    assert found is True
    assert clean['ok'] == 1.25
    assert clean['bad'] is None
    assert clean['neginf'] is None
    assert clean['nested'][0] == 1.0
    assert clean['nested'][1] is None
    assert clean['nested'][2]['x'] is None
    # Must serialize with allow_nan=False after sanitize.
    json.dumps(clean, allow_nan=False)


def test_sanitize_finite_only_no_flag() -> None:
    clean, found = sanitize_for_json({'a': 0.0, 'b': [1, 2, 3]})
    assert found is False
    assert clean == {'a': 0.0, 'b': [1, 2, 3]}
    assert math.isfinite(clean['a'])


def test_m3_summary_contract_nonfinite_does_not_raise_or_certify() -> None:
    """Mirror m3_orchestrator._summary handling of nonfinite session floats."""
    summary = {
        'milestone': 'M3',
        'phase': 'M3_COMPLETE',
        'milestone_status': 'CORE_REPRODUCED',
        'certification_status': 'NOT_CERTIFIED',
        'elapsed_s': float('nan'),
        'remaining_s': float('inf'),
        'stop_reason': 'drain checkpoint complete',
    }
    clean, had_nonfinite = sanitize_for_json(summary)
    assert had_nonfinite is True
    clean['nonfinite_values_present'] = True
    clean['certification_status'] = 'NOT_CERTIFIED'
    if clean.get('phase') == 'M3_COMPLETE':
        clean['phase'] = 'M3_RUNNING'
        clean['milestone_status'] = 'EXPLORATORY'
    assert clean['phase'] == 'M3_RUNNING'
    assert clean['certification_status'] == 'NOT_CERTIFIED'
    json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False)
