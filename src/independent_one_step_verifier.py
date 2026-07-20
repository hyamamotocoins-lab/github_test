"""Independent read-only verifier for one-step certificate packages.

Final arithmetic is reimplemented here with fractions.Fraction and must not import
`src.certificate` or `src.influence`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import canonical_json_bytes, read_json, sha256_file
from .exact_arithmetic import fraction_decimal_text
from .proof_manifest import (
    ONE_STEP_CERTIFICATE_FILES,
    ProofManifestError,
    load_proof_dependencies,
    package_manifest_hash,
    reject_symlinks,
    verify_immutable_package,
)


class IndependentVerifierError(RuntimeError):
    """Raised when independent verification fails closed."""


def _fraction_from_payload(payload: dict[str, Any]) -> Fraction:
    try:
        return Fraction(
            int(payload['numerator_hex'], 16),
            int(payload['denominator_hex'], 16),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise IndependentVerifierError('Malformed fraction payload.') from exc


def _interval_from_payload(payload: dict[str, Any]) -> tuple[Fraction, Fraction]:
    if not isinstance(payload, dict):
        raise IndependentVerifierError('Interval payload must be a mapping.')
    lo = _fraction_from_payload(payload['lo'])
    hi = _fraction_from_payload(payload['hi'])
    if lo > hi:
        raise IndependentVerifierError('Interval endpoints are unordered.')
    if not all(math.isfinite(float(x)) for x in (lo, hi)):
        raise IndependentVerifierError('Interval endpoints must be finite.')
    return lo, hi


def _require_nonnegative(lo: Fraction, hi: Fraction, label: str) -> None:
    if lo < 0 or hi < 0:
        raise IndependentVerifierError(f'{label} is negative.')


def _add(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    return a[0] + b[0], a[1] + b[1]


def _mul(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    products = (a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1])
    return min(products), max(products)


def _div(a: tuple[Fraction, Fraction], b: tuple[Fraction, Fraction]) -> tuple[Fraction, Fraction]:
    if b[0] <= 0 <= b[1] or b[0] == 0 or b[1] == 0:
        raise IndependentVerifierError('Denominator interval contains zero.')
    recip = (Fraction(1, b[1]), Fraction(1, b[0])) if b[0] > 0 else (Fraction(1, b[0]), Fraction(1, b[1]))
    # For strictly positive denominators, reciprocal endpoints swap.
    if b[0] > 0:
        recip = (Fraction(1, b[1]), Fraction(1, b[0]))
    return _mul(a, recip)


def _recompute_influence_entry(entry: dict[str, Any]) -> tuple[Fraction, Fraction]:
    diameter = _interval_from_payload(entry['diameter_interval'])
    core = _interval_from_payload(entry['derivative_core_l1_interval'])
    error = _interval_from_payload(entry['derivative_error_interval'])
    zmin = _interval_from_payload(entry['normalization_lower_interval'])
    _require_nonnegative(*diameter, 'diameter')
    _require_nonnegative(*core, 'derivative core')
    _require_nonnegative(*error, 'derivative error')
    if zmin[0] <= 0:
        raise IndependentVerifierError('Independent z_min is not strictly positive.')
    multiplicity = entry.get('orbit_multiplicity', 1)
    if not isinstance(multiplicity, int) or multiplicity < 1:
        raise IndependentVerifierError('Invalid orbit multiplicity.')
    if entry.get('metric_unit') != entry.get('source_speed_unit'):
        raise IndependentVerifierError('Metric/source unit incompatibility.')
    numerator = _add(core, error)
    ratio = _div(numerator, zmin)
    influence = _mul(diameter, ratio)
    if multiplicity != 1:
        influence = _mul(influence, (Fraction(multiplicity), Fraction(multiplicity)))
    recorded = _interval_from_payload(entry['influence_upper_interval'])
    if influence != recorded:
        raise IndependentVerifierError('Influence entry formula mismatch.')
    return influence


def _recompute_collatz(package: Path) -> dict[str, Any]:
    influence = read_json(package / 'influence_matrix_intervals.json')
    perron = read_json(package / 'perron_vector.json')
    collatz = read_json(package / 'collatz_bound.json')
    if not isinstance(influence, dict) or not isinstance(perron, dict) or not isinstance(collatz, dict):
        raise IndependentVerifierError('Collatz reconstruction inputs are malformed.')

    weighted = influence.get('weighted_matrix')
    if not isinstance(weighted, dict):
        raise IndependentVerifierError('weighted_matrix is required for independent Collatz check.')
    labels = weighted.get('labels')
    entries = weighted.get('entries')
    if not isinstance(labels, list) or not isinstance(entries, list):
        raise IndependentVerifierError('weighted_matrix shape is invalid.')
    n = len(labels)
    matrix: list[list[tuple[Fraction, Fraction]]] = []
    for row in entries:
        if not isinstance(row, list) or len(row) != n:
            raise IndependentVerifierError('weighted_matrix row is invalid.')
        converted = [_interval_from_payload(cell) for cell in row]
        for cell in converted:
            _require_nonnegative(*cell, 'weighted matrix entry')
        matrix.append(converted)

    components_payload = perron.get('components')
    vector_labels = perron.get('labels')
    if vector_labels != labels or not isinstance(components_payload, list):
        raise IndependentVerifierError('Perron vector labels/order mismatch.')
    vector = [_fraction_from_payload(item) for item in components_payload]
    if any(value <= 0 for value in vector):
        raise IndependentVerifierError('Nonpositive Perron component.')

    quotients: list[tuple[Fraction, Fraction]] = []
    for i in range(n):
        total_lo = Fraction(0)
        total_hi = Fraction(0)
        for j in range(n):
            term = _mul(matrix[i][j], (vector[j], vector[j]))
            total_lo += term[0]
            total_hi += term[1]
        quotients.append(_div((total_lo, total_hi), (vector[i], vector[i])))

    q_cw = (
        min(q[0] for q in quotients),
        max(q[1] for q in quotients),
    )
    outside = _interval_from_payload(collatz['outside_matrix_tail'])
    _require_nonnegative(*outside, 'outside-matrix tail')
    q_cert = _add(q_cw, outside)
    recorded = _interval_from_payload(collatz['q_cert'])
    if q_cert != recorded:
        raise IndependentVerifierError('Independent q_cert disagrees with package.')
    return {
        'recomputed_collatz_rows': [
            {
                'lo': fraction_decimal_text(q[0]),
                'hi': fraction_decimal_text(q[1]),
            }
            for q in quotients
        ],
        'recomputed_q_cert': {
            'lo': fraction_decimal_text(q_cert[0]),
            'hi': fraction_decimal_text(q_cert[1]),
            'lower_fraction': {
                'numerator_hex': format(q_cert[0].numerator, 'x'),
                'denominator_hex': format(q_cert[0].denominator, 'x'),
            },
            'upper_fraction': {
                'numerator_hex': format(q_cert[1].numerator, 'x'),
                'denominator_hex': format(q_cert[1].denominator, 'x'),
            },
        },
    }


@dataclass(frozen=True, slots=True)
class IndependentVerifierReport:
    package_manifest_hash: str
    recomputed_artifact_hashes: dict[str, str]
    recomputed_normalization: dict[str, Any]
    recomputed_influence_entries: list[dict[str, Any]]
    recomputed_weighted_matrix: dict[str, Any]
    recomputed_collatz_rows: list[dict[str, str]]
    recomputed_q_cert: dict[str, Any]
    main_verdict: dict[str, Any]
    independent_verdict: str
    agreement: bool

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'package_manifest_hash': self.package_manifest_hash,
            'recomputed_artifact_hashes': self.recomputed_artifact_hashes,
            'recomputed_normalization': self.recomputed_normalization,
            'recomputed_influence_entries': self.recomputed_influence_entries,
            'recomputed_weighted_matrix': self.recomputed_weighted_matrix,
            'recomputed_collatz_rows': self.recomputed_collatz_rows,
            'recomputed_q_cert': self.recomputed_q_cert,
            'main_verdict': self.main_verdict,
            'independent_verdict': self.independent_verdict,
            'agreement': self.agreement,
        }


def verify_one_step_package(
    package_root: Path,
    *,
    require_independent_pass_marker: bool = True,
) -> IndependentVerifierReport:
    """Read-only independent verification of a complete one_step_certificate package."""
    try:
        reject_symlinks(package_root)
        manifest = verify_immutable_package(package_root)
    except ProofManifestError as exc:
        raise IndependentVerifierError(str(exc)) from exc

    dependencies = load_proof_dependencies(package_root / 'proof_dependencies.json')
    dependency_artifacts = {node.artifact for node in dependencies}
    if not dependency_artifacts <= set(ONE_STEP_CERTIFICATE_FILES):
        raise IndependentVerifierError('Dependency references unknown package artifact.')

    for relative in ONE_STEP_CERTIFICATE_FILES:
        if not relative.endswith('.json'):
            continue
        path = package_root / relative
        payload = read_json(path)
        rewritten = canonical_json_bytes(payload)
        if json.loads(rewritten.decode('utf-8')) != payload:
            raise IndependentVerifierError(f'Non-canonical or invalid JSON: {relative}')

    normalization = read_json(package_root / 'normalization_bounds.json')
    if not isinstance(normalization, dict):
        raise IndependentVerifierError('normalization_bounds.json is malformed.')
    zmin = _interval_from_payload(normalization['z_min_interval'])
    if zmin[0] <= 0:
        raise IndependentVerifierError('Independent normalization lower bound is not positive.')

    influence_doc = read_json(package_root / 'influence_matrix_intervals.json')
    if not isinstance(influence_doc, dict) or not isinstance(influence_doc.get('entries'), list):
        raise IndependentVerifierError('influence_matrix_intervals.json is malformed.')
    recomputed_entries: list[dict[str, Any]] = []
    for entry in influence_doc['entries']:
        if not isinstance(entry, dict):
            raise IndependentVerifierError('Influence entry is malformed.')
        lo, hi = _recompute_influence_entry(entry)
        recomputed_entries.append({
            'row_type': entry['row_type'],
            'column_type': entry['column_type'],
            'displacement': entry['displacement'],
            'influence_upper': {
                'lo': fraction_decimal_text(lo),
                'hi': fraction_decimal_text(hi),
            },
        })

    weighted = influence_doc.get('weighted_matrix')
    if not isinstance(weighted, dict):
        raise IndependentVerifierError('weighted_matrix missing from influence package.')

    collatz_part = _recompute_collatz(package_root)
    verdict = read_json(package_root / 'verdict.json')
    if not isinstance(verdict, dict):
        raise IndependentVerifierError('verdict.json is malformed.')

    q_cert = collatz_part['recomputed_q_cert']
    q_hi = Fraction(
        int(q_cert['upper_fraction']['numerator_hex'], 16),
        int(q_cert['upper_fraction']['denominator_hex'], 16),
    )
    q_lo = Fraction(
        int(q_cert['lower_fraction']['numerator_hex'], 16),
        int(q_cert['lower_fraction']['denominator_hex'], 16),
    )
    if q_hi < 1:
        independent_verdict = 'ONE_STEP_CERTIFIED'
    elif q_lo >= 1:
        independent_verdict = 'NOT_CERTIFIED'
    else:
        independent_verdict = 'BLOCKED_MATH'

    main_status = verdict.get('certification_status')
    if main_status != independent_verdict:
        raise IndependentVerifierError(
            f'Main/independent verdict disagreement: {main_status} vs {independent_verdict}'
        )
    marker = verdict.get('independent_verifier')
    if require_independent_pass_marker and marker != 'PASS':
        raise IndependentVerifierError(
            'Package independent_verifier marker is not PASS.'
        )
    agreement = main_status == independent_verdict and (
        marker == 'PASS' or not require_independent_pass_marker
    )

    return IndependentVerifierReport(
        package_manifest_hash=manifest['package_manifest_hash'],
        recomputed_artifact_hashes=manifest['file_hashes'],
        recomputed_normalization={
            'z_min_lower': fraction_decimal_text(zmin[0]),
            'z_min_upper': fraction_decimal_text(zmin[1]),
        },
        recomputed_influence_entries=recomputed_entries,
        recomputed_weighted_matrix={
            'labels': weighted.get('labels'),
            'dimension': len(weighted.get('labels', [])),
        },
        recomputed_collatz_rows=collatz_part['recomputed_collatz_rows'],
        recomputed_q_cert=q_cert,
        main_verdict=verdict,
        independent_verdict=independent_verdict,
        agreement=agreement,
    )


def package_hash_fingerprint(package_root: Path) -> str:
    hashes = {
        relative: sha256_file(package_root / relative)
        for relative in ONE_STEP_CERTIFICATE_FILES
    }
    return package_manifest_hash(hashes)
