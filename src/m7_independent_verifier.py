"""Independent verifier for an accepted M7 certified scheme package.

Must not import producer Collatz helpers from m7_replay/m7_collatz_search.
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import read_json, utc_now
from .exact_arithmetic import fraction_decimal_text, fraction_from_payload


class M7IndependentVerifierError(RuntimeError):
    """Raised when M7 independent verification fails closed."""


def _interval(payload: dict[str, Any]) -> tuple[Fraction, Fraction]:
    return fraction_from_payload(payload['lo']), fraction_from_payload(payload['hi'])


def verify_accepted_scheme(final_package_root: Path) -> dict[str, Any]:
    root = final_package_root.resolve()
    acceptance = read_json(root / 'M7_acceptance.json')
    scheme = read_json(root / 'accepted_scheme.json')
    bound = read_json(root / 'final_bound.json')
    influence = read_json(root / 'final_influence_matrix.json')
    perron = read_json(root / 'perron_vector.json')
    if not all(isinstance(doc, dict) for doc in (acceptance, scheme, bound, influence, perron)):
        raise M7IndependentVerifierError('Accepted package artifacts malformed.')

    labels = influence.get('labels')
    entries = influence.get('entries')
    components = perron.get('components')
    if not isinstance(labels, list) or not isinstance(entries, list):
        raise M7IndependentVerifierError('Influence malformed.')
    if not isinstance(components, list) or len(components) != len(labels):
        raise M7IndependentVerifierError('Perron malformed.')

    weights = [
        Fraction(int(item['numerator_hex'], 16), int(item['denominator_hex'], 16))
        for item in components
        if isinstance(item, dict)
    ]
    if any(weight <= 0 for weight in weights):
        raise M7IndependentVerifierError('Perron not strictly positive.')

    matrix = []
    for row in entries:
        matrix.append([_interval(cell) for cell in row])

    quotients = []
    for i, row in enumerate(matrix):
        acc_lo = Fraction(0)
        acc_hi = Fraction(0)
        for j, cell in enumerate(row):
            acc_lo += cell[0] * weights[j]
            acc_hi += cell[1] * weights[j]
        quotients.append((acc_lo / weights[i], acc_hi / weights[i]))
    q_lo = min(q[0] for q in quotients)
    q_hi = max(q[1] for q in quotients)
    tail = bound.get('outside_matrix_tail')
    if not isinstance(tail, dict):
        raise M7IndependentVerifierError('Missing outside_matrix_tail.')
    t_lo, t_hi = _interval(tail)
    q_cert_lo, q_cert_hi = q_lo + t_lo, q_hi + t_hi

    if q_cert_hi >= 1:
        raise M7IndependentVerifierError(
            f'Independent q_cert_upper={q_cert_hi} does not prove contraction.'
        )
    recorded = bound.get('q_cert')
    if isinstance(recorded, dict):
        r_lo, r_hi = _interval(recorded)
        if (r_lo, r_hi) != (q_cert_lo, q_cert_hi):
            raise M7IndependentVerifierError('Bound mismatch under independent recomputation.')

    if acceptance.get('status') != 'CERTIFIED_SCHEME_FOUND':
        raise M7IndependentVerifierError('Acceptance status is not CERTIFIED_SCHEME_FOUND.')

    return {
        'schema_version': 1,
        'status': 'PASS',
        'independent_verifier': 'PASS',
        'q_cert_lower': fraction_decimal_text(q_cert_lo),
        'q_cert_upper': fraction_decimal_text(q_cert_hi),
        'scheme_hash': scheme.get('scheme_hash'),
        'candidate_id': scheme.get('candidate_id'),
        'generated_at': utc_now(),
    }
