"""M7 failure diagnosis against a frozen M6 parent package."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import read_json
from .exact_arithmetic import fraction_from_payload
from .m7_status import (
    DIAG_D0,
    DIAG_D1,
    DIAG_D6,
    POLICY_PARENT_INHERIT,
)


class M7DiagnosisError(RuntimeError):
    """Raised when diagnosis cannot proceed."""


def _interval_hi(payload: Any) -> Fraction | None:
    if not isinstance(payload, dict):
        return None
    if 'hi' in payload and isinstance(payload['hi'], dict):
        return fraction_from_payload(payload['hi'])
    return None


def diagnose_m6_package(
    package_root: Path,
    *,
    parent_m5_q: Fraction | None = None,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    verdict = read_json(package_root / 'verdict.json')
    influence = read_json(package_root / 'final_influence_matrix.json')
    final_bound = read_json(package_root / 'final_bound.json')
    ledger = read_json(package_root / 'error_ledger.json')
    run_config = read_json(package_root / 'run_config.json')
    if not all(isinstance(doc, dict) for doc in (verdict, influence, final_bound, ledger)):
        raise M7DiagnosisError('M6 package artifacts malformed.')

    q_hi = None
    q_payload = final_bound.get('q_cert')
    if isinstance(q_payload, dict) and 'hi' in q_payload:
        q_hi = _interval_hi(q_payload)
    if q_hi is None and verdict.get('q_cert_upper'):
        raw = str(verdict['q_cert_upper'])
        # Avoid Fraction(long decimal) on Python 3.11 digit-limit.
        if len(raw) <= 200:
            try:
                q_hi = Fraction(raw)
            except (ValueError, ZeroDivisionError):
                q_hi = Fraction.from_float(float(raw))
        else:
            q_hi = Fraction.from_float(float(raw))

    policy = None
    if isinstance(run_config, dict):
        policy = run_config.get('composition_policy') or run_config.get('majorant_policy')
    if isinstance(policy, str) and 'inherits_m5' in policy:
        policy = POLICY_PARENT_INHERIT

    open_leaves = []
    leaves = ledger.get('leaves') if isinstance(ledger, dict) else None
    if isinstance(leaves, dict):
        open_leaves = [
            key for key, value in leaves.items()
            if not (isinstance(value, dict) and value.get('status') == 'RIGOROUS')
        ]

    codes: list[str] = []
    if open_leaves:
        codes.append(DIAG_D6)
    inherited = False
    if parent_m5_q is not None and q_hi is not None and parent_m5_q == q_hi:
        inherited = True
    if policy == POLICY_PARENT_INHERIT or (
        isinstance(policy, str) and 'inherit' in policy.lower()
    ):
        inherited = True
        codes.append(DIAG_D0)
    elif inherited:
        codes.append(DIAG_D0)
    else:
        codes.append(DIAG_D1)

    # Crude core estimate: Collatz with all-ones equals max row-sum upper.
    labels = influence.get('labels') if isinstance(influence, dict) else None
    entries = influence.get('entries') if isinstance(influence, dict) else None
    row_sum_uppers: list[str] = []
    max_row = Fraction(0)
    if isinstance(labels, list) and isinstance(entries, list):
        for row in entries:
            if not isinstance(row, list):
                continue
            total = Fraction(0)
            for cell in row:
                hi = _interval_hi(cell)
                if hi is not None:
                    total += hi
            row_sum_uppers.append(format(float(total), '.17g'))
            max_row = max(max_row, total)

    return {
        'schema_version': 1,
        'diagnosis_codes': codes,
        'primary_code': codes[0] if codes else DIAG_D1,
        'q_cert_upper': format(float(q_hi), '.17g') if q_hi is not None else None,
        'q_cert_upper_rational': (
            {
                'numerator_hex': format(q_hi.numerator, 'x'),
                'denominator_hex': format(q_hi.denominator, 'x'),
            }
            if q_hi is not None else None
        ),
        'parent_m5_q_cert_upper': (
            format(float(parent_m5_q), '.17g') if parent_m5_q is not None else None
        ),
        'parent_m5_q_cert_upper_rational': (
            {
                'numerator_hex': format(parent_m5_q.numerator, 'x'),
                'denominator_hex': format(parent_m5_q.denominator, 'x'),
            }
            if parent_m5_q is not None else None
        ),
        'inherited_majorant': inherited,
        'composition_policy': policy,
        'open_ledger_leaves': open_leaves,
        'max_row_sum_upper': format(float(max_row), '.17g'),
        'max_row_sum_upper_rational': {
            'numerator_hex': format(max_row.numerator, 'x'),
            'denominator_hex': format(max_row.denominator, 'x'),
        },
        'row_sum_uppers': row_sum_uppers,
        'recommended_campaign': 'B' if DIAG_D0 in codes else 'A',
        'notes': (
            'q_cert >= 1 is a certificate failure of the declared majorant, '
            'not a proof that the true RG map is expansive. '
            'Campaign A (S0 reweight / identical-step product) cannot beat '
            'spectral radius of an inherited expanding majorant.'
        ),
        'next_actions': [
            'Campaign A exhausted for inherited majorant with q≈2.5: S0 cannot '
            'push Collatz below rho.',
            'Promote Campaign B (S2): increase target_rank / tighten residuals '
            'and rebuild M3→M6 lineage under LOCK.',
            'In parallel, implement true stage-dependent B_r (not B^K of the '
            'same inherited matrix).',
        ],
    }
