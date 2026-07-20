"""Rigorous evaluation of M7 scheme candidates against an M6 influence matrix."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import read_json
from .exact_arithmetic import fraction_from_payload
from .interval_kernel import construct
from .m5_package import make_contractive_fixture_inputs
from .m6_status import CERTIFIED
from .m7_collatz_search import (
    coerce_interval,
    diagonal_plus_l1_tail,
    evaluate_collatz,
    inverse_row_sum_weights,
    matrix_midpoints,
    power_iteration_perron,
)
from .m7_status import SCHEME_REJECTED


class M7ReplayError(RuntimeError):
    """Raised when a candidate cannot be rigorously evaluated."""


def _load_matrix(package_root: Path) -> tuple[list[str], list[list[Any]], Any]:
    influence = read_json(package_root / 'final_influence_matrix.json')
    bound = read_json(package_root / 'final_bound.json')
    if not isinstance(influence, dict):
        raise M7ReplayError('final_influence_matrix.json malformed.')
    labels = list(influence.get('labels') or [])
    entries = influence.get('entries')
    if not isinstance(entries, list) or not labels:
        raise M7ReplayError('Influence matrix missing labels/entries.')
    matrix = [list(row) for row in entries]
    outside = construct(0)
    if isinstance(bound, dict) and isinstance(bound.get('outside_matrix_tail'), dict):
        outside = bound['outside_matrix_tail']
    return labels, matrix, outside


def perron_for_strategy(
    strategy: str,
    labels: list[str],
    entries: list[list[Any]],
) -> list[Any]:
    mid = matrix_midpoints(entries)
    if strategy == 'all_ones':
        return ['1'] * len(labels)
    if strategy in {'interval_power', 'collatz_lp_heuristic'}:
        iters = 128 if strategy == 'collatz_lp_heuristic' else 64
        return [str(value) for value in power_iteration_perron(mid, iterations=iters)]
    if strategy == 'inverse_row_sum':
        return [str(value) for value in inverse_row_sum_weights(mid)]
    raise M7ReplayError(f'Unknown perron strategy: {strategy}')


def _perron_for_strategy(
    strategy: str,
    labels: list[str],
    entries: list[list[Any]],
) -> list[Any]:
    return perron_for_strategy(strategy, labels, entries)


def interval_product_power(
    entries: list[list[Any]],
    power: int,
) -> list[list[Any]]:
    """Outward interval matrix power for DIRECT_MULTI_STEP_PRODUCT."""
    if power < 1:
        raise M7ReplayError('Product power must be >= 1.')
    n = len(entries)
    current = [[coerce_interval(cell) for cell in row] for row in entries]
    if power == 1:
        return current
    result = current
    for _ in range(power - 1):
        nxt = [[construct(0) for _ in range(n)] for _ in range(n)]
        for i in range(n):
            for j in range(n):
                acc = construct(0)
                for k in range(n):
                    acc = acc.add(result[i][k].multiply(current[k][j]))
                nxt[i][j] = acc
        result = nxt
    return result


def evaluate_candidate_rigorous(
    package_root: Path,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    scheme = candidate.get('scheme')
    if not isinstance(scheme, dict):
        raise M7ReplayError('Candidate scheme missing.')

    if scheme.get('majorant_policy') == 'FIXTURE_CONTRACTIVE_REFERENCE':
        fixture = make_contractive_fixture_inputs()
        result = evaluate_collatz(
            [list(row) for row in fixture['weighted_matrix']],
            list(fixture['labels']),
            list(fixture['perron']),
            outside_tail=fixture['outside_tail'],
        )
        return _pack(candidate, result, notes='fixture_contractive_reference')

    labels, entries, outside = _load_matrix(package_root)
    coupling = scheme.get('coupling_policy', 'uniform_full')
    working = entries
    tail: Any = outside
    if coupling == 'diagonal_plus_l1_tail':
        working, extra = diagonal_plus_l1_tail(entries)
        if isinstance(tail, dict) and isinstance(tail.get('hi'), dict):
            base = fraction_from_payload(tail['hi'])
        else:
            base = construct(tail).hi
        tail = construct(0, base + extra)

    policy = scheme.get('majorant_policy', 'PARENT_MATRIX_REWEIGHT_ONLY')
    num_steps = int(scheme.get('num_steps', 3))
    if policy in {
        'DIRECT_MULTI_STEP_PRODUCT',
        'STAGE_DEPENDENT_WEIGHTED_PRODUCT',
    }:
        working = interval_product_power(working, num_steps)
        notes = f'{policy}_power_{num_steps}_of_inherited_step_majorant'
    else:
        notes = 'parent_matrix_reweight_only'

    perron = _perron_for_strategy(
        str(scheme.get('perron_weight_strategy', 'all_ones')),
        labels,
        working,
    )
    result = evaluate_collatz(working, labels, perron, outside_tail=tail)
    return _pack(candidate, result, notes=notes)


def _pack(candidate: dict[str, Any], result: dict[str, Any], *, notes: str) -> dict[str, Any]:
    certified = bool(result['certified'])
    q_lo = result['q_cert_lo']
    q_hi = result['q_cert_hi']
    return {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'scheme_hash': candidate.get('scheme_hash'),
        'change_class': candidate.get('change_class'),
        'scheme': candidate.get('scheme'),
        'notes': notes,
        # Exact endpoints as hex rationals (safe to round-trip on Py3.11).
        'q_cert_lower_rational': {
            'numerator_hex': format(q_lo.numerator, 'x'),
            'denominator_hex': format(q_lo.denominator, 'x'),
        },
        'q_cert_upper_rational': {
            'numerator_hex': format(q_hi.numerator, 'x'),
            'denominator_hex': format(q_hi.denominator, 'x'),
        },
        # Short display only — do not parse these back with Fraction().
        'q_cert_lower': format(float(q_lo), '.17g'),
        'q_cert_upper': format(float(q_hi), '.17g'),
        'q_collatz_upper': format(float(result['q_collatz_hi']), '.17g'),
        'collatz_verdict': result['verdict'],
        'status': 'M6_CERTIFIED' if certified else 'M6_NOT_CERTIFIED',
        'scheme_result': CERTIFIED if certified else SCHEME_REJECTED,
        'certified': certified,
        'mathematical_interpretation': {
            'proved_if_certified': 'Declared majorant satisfies q_cert_upper < 1.',
            'proved_if_rejected': (
                'Declared majorant does not prove contraction.'
            ),
            'not_proved': 'True-map expansion/non-contraction.',
        },
        'bound_payload': result['payload'],
    }
