"""Independent read-only verifier for M6 final_certificate packages.

Must not import `src.certificate` or `src.influence`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file, utc_now
from .exact_arithmetic import fraction_decimal_text
from .m6_ledger import all_leaves_closed, open_leaves
from .m6_package import M6_CERTIFICATE_ROOT_FILES, STEP_FILES
from .m6_status import CERTIFIED, M6_COMPLETE, NOT_CERTIFIED, STEP_ENCLOSED
from .proof_manifest import ProofManifestError, reject_symlinks


class M6IndependentVerifierError(RuntimeError):
    """Raised when M6 independent verification fails closed."""


def _fraction_from_payload(payload: dict[str, Any]) -> Fraction:
    try:
        return Fraction(
            int(payload['numerator_hex'], 16),
            int(payload['denominator_hex'], 16),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise M6IndependentVerifierError('Malformed fraction payload.') from exc


def _interval_from_payload(payload: dict[str, Any]) -> tuple[Fraction, Fraction]:
    if not isinstance(payload, dict):
        raise M6IndependentVerifierError('Interval payload must be a mapping.')
    lo = _fraction_from_payload(payload['lo'])
    hi = _fraction_from_payload(payload['hi'])
    if lo > hi:
        raise M6IndependentVerifierError('Interval endpoints unordered.')
    if not all(math.isfinite(float(x)) for x in (lo, hi)):
        raise M6IndependentVerifierError('Nonfinite interval endpoint.')
    return lo, hi


def _require_nonnegative(lo: Fraction, hi: Fraction, label: str) -> None:
    if lo < 0 or hi < 0:
        raise M6IndependentVerifierError(f'{label} is negative.')


@dataclass(frozen=True, slots=True)
class M6IndependentReport:
    agreement: bool
    independent_verdict: str
    q_cert_lower: str
    q_cert_upper: str
    main_verdict: dict[str, Any]
    notes: str

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'status': 'PASS' if self.agreement else 'FAIL',
            'agreement': self.agreement,
            'independent_verdict': self.independent_verdict,
            'q_cert_lower': self.q_cert_lower,
            'q_cert_upper': self.q_cert_upper,
            'main_certification_status': self.main_verdict.get('certification_status'),
            'notes': self.notes,
            'generated_at': utc_now(),
        }


def _exact_root_files(package_root: Path) -> None:
    reject_symlinks(package_root)
    present = {path.name for path in package_root.iterdir() if path.is_file()}
    missing = set(M6_CERTIFICATE_ROOT_FILES) - present
    if missing:
        raise M6IndependentVerifierError(f'Missing root files: {sorted(missing)}')


def _verify_steps(package_root: Path, num_steps: int) -> None:
    for index in range(num_steps):
        step_dir = package_root / f'rg_step_{index:02d}'
        if not step_dir.is_dir():
            raise M6IndependentVerifierError(f'Missing step directory: {step_dir.name}')
        for name in STEP_FILES:
            if not (step_dir / name).is_file():
                raise M6IndependentVerifierError(f'Missing {step_dir.name}/{name}')
        verdict = read_json(step_dir / 'step_verdict.json')
        if not isinstance(verdict, dict) or verdict.get('status') != STEP_ENCLOSED:
            raise M6IndependentVerifierError(f'Step {index} is not STEP_ENCLOSED.')


def verify_final_certificate(
    package_root: Path,
    *,
    require_independent_pass_marker: bool = True,
) -> M6IndependentReport:
    package_root = package_root.resolve()
    if not package_root.is_dir():
        raise M6IndependentVerifierError(f'Package missing: {package_root}')
    try:
        _exact_root_files(package_root)
    except ProofManifestError as exc:
        raise M6IndependentVerifierError(str(exc)) from exc

    verdict = read_json(package_root / 'verdict.json')
    if not isinstance(verdict, dict):
        raise M6IndependentVerifierError('verdict.json malformed.')
    if verdict.get('milestone') != 'M6' or verdict.get('phase') != M6_COMPLETE:
        raise M6IndependentVerifierError('Verdict milestone/phase mismatch.')

    num_steps = int(verdict.get('num_steps', 0))
    _verify_steps(package_root, num_steps)

    ledger = read_json(package_root / 'error_ledger.json')
    if not all_leaves_closed(ledger):
        raise M6IndependentVerifierError(
            'Open ledger leaves: ' + ', '.join(open_leaves(ledger))
        )

    influence = read_json(package_root / 'final_influence_matrix.json')
    perron = read_json(package_root / 'perron_vector.json')
    final_bound = read_json(package_root / 'final_bound.json')
    if not all(isinstance(doc, dict) for doc in (influence, perron, final_bound)):
        raise M6IndependentVerifierError('Core bound artifacts malformed.')

    labels = influence.get('labels')
    entries = influence.get('entries')
    components = perron.get('components')
    if not isinstance(labels, list) or not isinstance(entries, list):
        raise M6IndependentVerifierError('Influence matrix malformed.')
    if not isinstance(components, list) or len(components) != len(labels):
        raise M6IndependentVerifierError('Perron vector malformed.')

    matrix: list[list[tuple[Fraction, Fraction]]] = []
    for row in entries:
        if not isinstance(row, list) or len(row) != len(labels):
            raise M6IndependentVerifierError('Influence row malformed.')
        parsed_row = []
        for cell in row:
            lo, hi = _interval_from_payload(cell)
            _require_nonnegative(lo, hi, 'influence entry')
            parsed_row.append((lo, hi))
        matrix.append(parsed_row)

    weights: list[Fraction] = []
    for item in components:
        if not isinstance(item, dict):
            raise M6IndependentVerifierError('Perron component malformed.')
        value = Fraction(
            int(item['numerator_hex'], 16),
            int(item['denominator_hex'], 16),
        )
        if value <= 0:
            raise M6IndependentVerifierError('Perron component not positive.')
        weights.append(value)

    quotients: list[tuple[Fraction, Fraction]] = []
    for row_index, row in enumerate(matrix):
        acc_lo = Fraction(0)
        acc_hi = Fraction(0)
        for column_index, cell in enumerate(row):
            acc_lo += cell[0] * weights[column_index]
            acc_hi += cell[1] * weights[column_index]
        denom = weights[row_index]
        quotients.append((acc_lo / denom, acc_hi / denom))

    q_lo = min(item[0] for item in quotients)
    q_hi = max(item[1] for item in quotients)

    tail_raw = final_bound.get('outside_matrix_tail')
    if not isinstance(tail_raw, dict) or 'lo' not in tail_raw or 'hi' not in tail_raw:
        raise M6IndependentVerifierError('final_bound missing outside_matrix_tail.')
    tail_lo, tail_hi = _interval_from_payload(tail_raw)
    _require_nonnegative(tail_lo, tail_hi, 'outside_matrix_tail')

    q_cert_lo = q_lo + tail_lo
    q_cert_hi = q_hi + tail_hi

    if q_cert_hi < 1:
        independent_verdict = CERTIFIED
    elif q_cert_lo >= 1:
        independent_verdict = NOT_CERTIFIED
    else:
        raise M6IndependentVerifierError('Independent q_cert crosses 1.')

    main_status = verdict.get('certification_status')
    if main_status != independent_verdict:
        raise M6IndependentVerifierError(
            f'Main/independent disagreement: {main_status} vs {independent_verdict}'
        )

    recorded = final_bound.get('q_cert')
    if isinstance(recorded, dict) and 'lo' in recorded and 'hi' in recorded:
        rec_lo, rec_hi = _interval_from_payload(recorded)
        if (rec_lo, rec_hi) != (q_cert_lo, q_cert_hi):
            raise M6IndependentVerifierError('Recomputed q_cert mismatches final_bound.')

    marker = verdict.get('independent_verifier')
    if require_independent_pass_marker:
        if marker not in {'PASS', 'PENDING'}:
            raise M6IndependentVerifierError(
                f'Unexpected independent_verifier marker: {marker!r}'
            )
        agreement = main_status == independent_verdict
        if marker == 'PASS' and not agreement:
            agreement = False
    else:
        agreement = main_status == independent_verdict

    return M6IndependentReport(
        agreement=agreement,
        independent_verdict=independent_verdict,
        q_cert_lower=fraction_decimal_text(q_cert_lo),
        q_cert_upper=fraction_decimal_text(q_cert_hi),
        main_verdict=verdict,
        notes=(
            f'Recomputed Collatz over {len(labels)} channels and {num_steps} steps.'
        ),
    )


def hash_package_files(package_root: Path) -> dict[str, str]:
    _exact_root_files(package_root)
    hashes = {
        name: sha256_file(package_root / name)
        for name in M6_CERTIFICATE_ROOT_FILES
    }
    verdict = read_json(package_root / 'verdict.json')
    num_steps = int(verdict.get('num_steps', 0))
    for index in range(num_steps):
        step_dir = package_root / f'rg_step_{index:02d}'
        for name in STEP_FILES:
            relative = f'{step_dir.name}/{name}'
            hashes[relative] = sha256_file(step_dir / name)
    return hashes
