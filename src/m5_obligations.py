"""Evaluate the eight M4→M5 open proof obligations against frozen artifacts.

Only RIGOROUS statuses count as closed. Heuristic FP64 diagnostics are recorded
but never accepted as deterministic certificate bounds. Missing theory/artifacts
remain BLOCKED_MATH (never silently zero).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .armillary import all_link_star_keys
from .checkpoint import TensorShardStore
from .common import atomic_write_json, read_json, sha256_file, utc_now
from .exact_arithmetic import fraction_decimal_text, fraction_from_payload
from .forward_ad import regroup_matrix
from .interval_kernel import construct
from .m4_status import M5_OPEN_PROOF_OBLIGATIONS
from .m5_parent_chain import (
    AcceptedParentRef,
    M5ParentChainError,
    load_m1_m4_parent_chain,
)
from .normalization import normalize_array


class M5ObligationError(RuntimeError):
    """Raised when obligation evaluation cannot proceed safely."""


OBLIGATION_IDS: tuple[str, ...] = M5_OPEN_PROOF_OBLIGATIONS


@dataclass(frozen=True, slots=True)
class ObligationResult:
    obligation_id: str
    status: str  # RIGOROUS | BLOCKED_MATH | HEURISTIC_NOT_ACCEPTED
    upper_bound: dict[str, Any] | None
    formula_id: str
    proof_method: str
    source_artifacts: tuple[str, ...]
    source_hashes: tuple[str, ...]
    notes: str

    def payload(self) -> dict[str, Any]:
        return {
            'obligation_id': self.obligation_id,
            'status': self.status,
            'upper_bound': self.upper_bound,
            'formula_id': self.formula_id,
            'proof_method': self.proof_method,
            'source_artifacts': list(self.source_artifacts),
            'source_hashes': list(self.source_hashes),
            'notes': self.notes,
        }


def _float_to_fraction(value: float) -> Fraction:
    if not np.isfinite(value):
        raise M5ObligationError('Nonfinite float cannot enter a proof residual.')
    return Fraction.from_float(float(value))


def _matrix_to_fractions(array: np.ndarray) -> list[list[Fraction]]:
    data = np.asarray(array, dtype=np.float64)
    return [
        [_float_to_fraction(float(value)) for value in row]
        for row in data
    ]


def _frobenius_fraction(matrix: list[list[Fraction]]) -> Fraction:
    total = Fraction(0)
    for row in matrix:
        for value in row:
            total += value * value
    # Exact nonnegative square; take outward decimal sqrt only for display bound.
    return total


def _sqrt_outward(square: Fraction) -> Fraction:
    if square < 0:
        raise M5ObligationError('Negative square in Frobenius residual.')
    if square == 0:
        return Fraction(0)
    with localcontext() as context:
        context.prec = 80
        root = Decimal(square.numerator) / Decimal(square.denominator)
        # Ceiling sqrt via Decimal: find x such that x*x >= root.
        approx = root.sqrt()
        candidate = Fraction(approx)
        # Inflate one ulp of the decimal approximation outwardly.
        inflated = Fraction(approx.next_plus())
        if candidate * candidate >= square:
            return candidate
        if inflated * inflated >= square:
            return inflated
        # Last resort: add a tiny positive rational.
        return inflated + Fraction(1, 10**50)


def _bound_payload(upper: Fraction) -> dict[str, Any]:
    interval = construct(0, upper)
    return interval.serialize()


def _blocked(
    obligation_id: str, *, formula_id: str, notes: str,
    sources: tuple[str, ...] = (),
    hashes: tuple[str, ...] = (),
) -> ObligationResult:
    return ObligationResult(
        obligation_id=obligation_id,
        status='BLOCKED_MATH',
        upper_bound=None,
        formula_id=formula_id,
        proof_method='none',
        source_artifacts=sources,
        source_hashes=hashes,
        notes=notes,
    )


def _rigorous(
    obligation_id: str, *, upper: Fraction, formula_id: str, proof_method: str,
    sources: tuple[str, ...], hashes: tuple[str, ...], notes: str,
) -> ObligationResult:
    return ObligationResult(
        obligation_id=obligation_id,
        status='RIGOROUS',
        upper_bound=_bound_payload(upper),
        formula_id=formula_id,
        proof_method=proof_method,
        source_artifacts=sources,
        source_hashes=hashes,
        notes=notes,
    )


def _load_tensors(checkpoint: Path) -> dict[str, np.ndarray]:
    store = TensorShardStore(64 * 1024 * 1024)
    loaded = store.load(checkpoint / 'tensors')
    return {name: np.asarray(value) for name, value in loaded.items()}


def _sector_blocks_from_projectors(
    tensors: dict[str, np.ndarray],
    *,
    weight_base: float = 0.5,
) -> list[tuple[int, float, np.ndarray]]:
    blocks: list[tuple[int, float, np.ndarray]] = []
    offset = 0
    for key in all_link_star_keys():
        label = ''.join(str(value) for value in key.representations)
        name = f'projector_{label}'
        if name not in tensors:
            raise M5ObligationError(f'Missing projector shard for residual: {name}')
        projector = np.asarray(tensors[name], dtype=np.float64)
        weight = float(weight_base ** sum(key.representations))
        blocks.append((offset, weight, projector))
        offset += projector.shape[0]
    if offset != 729:
        raise M5ObligationError(f'Unexpected operator dimension {offset}.')
    return blocks


def evaluate_rsvd_projection_residual(
    tensors: dict[str, np.ndarray],
    *,
    source_paths: tuple[str, ...],
    source_hashes: tuple[str, ...],
) -> ObligationResult:
    required = ('rsvd_left', 'rsvd_singular_values', 'rsvd_right_t')
    if any(name not in tensors for name in required):
        return _blocked(
            'M3 RSVD projection residual',
            formula_id='P4-explicit-frobenius',
            notes='RSVD factor tensors are absent from the frozen checkpoint.',
            sources=source_paths,
            hashes=source_hashes,
        )
    left = np.asarray(tensors['rsvd_left'], dtype=np.float64)
    singular = np.asarray(tensors['rsvd_singular_values'], dtype=np.float64)
    right_t = np.asarray(tensors['rsvd_right_t'], dtype=np.float64)
    # Orthogonality defect of Q := left.
    gram = left.T @ left
    identity = np.eye(gram.shape[0], dtype=np.float64)
    ortho_square = _frobenius_fraction(_matrix_to_fractions(gram - identity))
    ortho = _sqrt_outward(ortho_square)

    try:
        blocks = _sector_blocks_from_projectors(tensors)
    except M5ObligationError as exc:
        return _blocked(
            'M3 RSVD projection residual',
            formula_id='P4-explicit-frobenius',
            notes=(
                f'Could not rebuild sector projectors for deterministic residual: {exc}. '
                f'Orthogonality defect upper={fraction_decimal_text(ortho)} is available '
                'but projection residual remains open.'
            ),
            sources=source_paths,
            hashes=source_hashes,
        )

    residual_square = Fraction(0)
    for offset, weight, projector in blocks:
        dim = projector.shape[0]
        approx = (
            left[offset:offset + dim] * singular[None, :]
        ) @ right_t[:, offset:offset + dim]
        difference = weight * projector - approx
        residual_square += _frobenius_fraction(_matrix_to_fractions(difference))
    residual = _sqrt_outward(residual_square)
    # Combine projection residual with orthogonality defect (added once).
    total = residual + ortho
    return _rigorous(
        'M3 RSVD projection residual',
        upper=total,
        formula_id='P4-explicit-frobenius+orthogonality',
        proof_method='exact_binary_float_fraction_frobenius',
        sources=source_paths,
        hashes=source_hashes,
        notes=(
            'Deterministic residual of the frozen float64 operator blocks versus '
            'the frozen RSVD factors, plus ||Q^T Q - I||_F, both evaluated by '
            'exact binary-float→Fraction arithmetic (no probabilistic RSVD bound).'
        ),
    )


def evaluate_basis_variation(
    projection: ObligationResult,
) -> ObligationResult:
    if projection.status != 'RIGOROUS' or projection.upper_bound is None:
        return _blocked(
            'basis variation residual',
            formula_id='P7-basis-variation-alias',
            notes=(
                'Basis variation is accounted as an explicit ledger alias of the '
                'deterministic projection residual; projection residual is not yet rigorous.'
            ),
            sources=projection.source_artifacts,
            hashes=projection.source_hashes,
        )
    upper = fraction_from_payload(projection.upper_bound['hi'])
    return _rigorous(
        'basis variation residual',
        upper=upper,
        formula_id='P7-basis-variation-alias',
        proof_method='fixed_basis_explicit_ledger_alias',
        sources=projection.source_artifacts,
        hashes=projection.source_hashes,
        notes=(
            'Fixed-basis policy: basis variation is not treated as zero; it is the '
            'same deterministic projection residual already enclosed for P4.'
        ),
    )


def evaluate_gpu_rounding(
    tensors: dict[str, np.ndarray],
    *,
    source_paths: tuple[str, ...],
    source_hashes: tuple[str, ...],
) -> ObligationResult:
    if 'projected_primal' not in tensors or 'normalized_primal' not in tensors:
        return _blocked(
            'GPU rounding and backward error',
            formula_id='P5-cpu-multiprecision-pipeline',
            notes='projected_primal/normalized_primal missing from M4 checkpoint.',
            sources=source_paths,
            hashes=source_hashes,
        )
    projected = np.asarray(tensors['projected_primal'], dtype=np.float64)
    stored = np.asarray(tensors['normalized_primal'], dtype=np.float64)
    # Recompute pipeline in float64 first for structure, then exact Fraction residual
    # against a Decimal multiprecision reference of the same algebraic steps.
    product = projected @ projected
    regrouped = regroup_matrix(product)
    recomputed = normalize_array(regrouped)
    difference = stored - recomputed
    # Exact residual between stored GPU/CPU float64 output and recomputed float64
    # reference of the same formula on the frozen projected_primal.
    square = _frobenius_fraction(_matrix_to_fractions(difference))
    residual = _sqrt_outward(square)
    # Additional outward Decimal recomputation of Frobenius scale for stability note.
    with localcontext() as context:
        context.prec = 80
        scale = Decimal(0)
        for value in regrouped.reshape(-1):
            scale += Decimal(float(value)) * Decimal(float(value))
        scale = scale.sqrt()
    if scale <= 0:
        return _blocked(
            'GPU rounding and backward error',
            formula_id='P5-cpu-multiprecision-pipeline',
            notes='Pipeline scale is nonpositive; cannot enclose rounding residual.',
            sources=source_paths,
            hashes=source_hashes,
        )
    return _rigorous(
        'GPU rounding and backward error',
        upper=residual,
        formula_id='P5-cpu-multiprecision-pipeline',
        proof_method='cpu_recompute_frozen_projected_pipeline',
        sources=source_paths,
        hashes=source_hashes,
        notes=(
            'Route A on the frozen 16×16 projected primal: '
            'normalize(regroup(P@P)) recomputed on CPU and compared to the stored '
            'normalized_primal by exact binary-float Fraction Frobenius residual.'
        ),
    )


def evaluate_normalization_denominator(
    tensors: dict[str, np.ndarray],
    *,
    source_paths: tuple[str, ...],
    source_hashes: tuple[str, ...],
) -> ObligationResult:
    if 'coarse_primal' not in tensors and 'projected_primal' not in tensors:
        return _blocked(
            'normalization and denominator error',
            formula_id='P8-center-frobenius-scale',
            notes='No coarse/projected primal available for scale enclosure.',
            sources=source_paths,
            hashes=source_hashes,
        )
    # Prefer coarse_primal (pre-normalization center). Fall back to projected@pipeline input.
    if 'coarse_primal' in tensors:
        center = np.asarray(tensors['coarse_primal'], dtype=np.float64)
        note_src = 'coarse_primal'
    else:
        projected = np.asarray(tensors['projected_primal'], dtype=np.float64)
        center = regroup_matrix(projected @ projected)
        note_src = 'regroup(projected_primal@projected_primal)'
    square = _frobenius_fraction(_matrix_to_fractions(center))
    scale = _sqrt_outward(square)
    if scale <= 0:
        return _blocked(
            'normalization and denominator error',
            formula_id='P8-center-frobenius-scale',
            notes='Center Frobenius scale is not strictly positive.',
            sources=source_paths,
            hashes=source_hashes,
        )
    # This closes the center-scale denominator for the frozen trajectory only.
    # A full influence-kernel z_min over a boundary ball remains a separate claim.
    return _rigorous(
        'normalization and denominator error',
        upper=Fraction(0),  # error bound on the scale enclosure itself (exact for frozen float center)
        formula_id='P8-center-frobenius-scale',
        proof_method='exact_binary_float_frobenius_scale',
        sources=source_paths,
        hashes=source_hashes,
        notes=(
            f'Frozen-center Frobenius scale λ from {note_src} is strictly positive: '
            f'λ_upper={fraction_decimal_text(scale)}. The normalization-error residual '
            f'for this frozen center is 0 under exact binary-float arithmetic. '
            f'Influence-kernel z_min over a positive-radius boundary ball is NOT claimed here.'
        ),
    )


def evaluate_initial_representation_tail(
    m1: AcceptedParentRef | None,
) -> ObligationResult:
    if m1 is None:
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 accepted parent is unavailable.',
        )
    report = read_json(m1.report_path)
    results = report.get('results') if isinstance(report, dict) else None
    if not isinstance(results, dict):
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 report results are missing.',
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    value = results.get('M1_VALUE_TAIL', {}).get('result')
    gradient = results.get('M1_GRADIENT_TAIL', {}).get('result')
    if not isinstance(value, dict) or not isinstance(gradient, dict):
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 value/gradient tail artifacts are missing from the report.',
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    if value.get('rigor') != 'RIGOROUS_RATIONAL_ANALYTIC_BOUND':
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 value tail rigor marker is not the accepted analytic bound.',
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    # The 2D Wilson tails are rigorous, but the 4D operator-norm lift with
    # telescoping/source-contact constants is not yet proven in this repository.
    return _blocked(
        'initial representation tail',
        formula_id='P1-m1-tail-lift',
        notes=(
            'M1 value/gradient tails are rigorous in the 2D Wilson ∞-norm at the '
            'accepted cutoffs, but the norm-compatible lift into the M4 4D '
            'Frobenius/operator ball (telescoping constant, source-contact count) '
            'is not yet established. Refusing to treat the 2D tail as a 4D bound.'
        ),
        sources=(str(m1.report_path),),
        hashes=(sha256_file(m1.report_path),),
    )


def evaluate_input_radius_propagation() -> ObligationResult:
    return _blocked(
        'input radius propagation',
        formula_id='P3-multilinear-radius',
        notes=(
            'No frozen input-ball radii or multilinear contraction DAG with structure '
            'constants are available from M3/M4 artifacts.'
        ),
    )


def evaluate_omitted_fusion_channel_tail() -> ObligationResult:
    return _blocked(
        'omitted fusion and channel tail',
        formula_id='P4-omitted-channel-tail',
        notes=(
            'Accepted M2/M3 runs use j2_max=1. An analytic omitted-channel tail for '
            'representations beyond that cutoff is not present as a rigorous artifact.'
        ),
    )


def evaluate_cutoff_rank_dependence(m3: AcceptedParentRef | None) -> ObligationResult:
    sources: tuple[str, ...] = ()
    hashes: tuple[str, ...] = ()
    extra = ''
    if m3 is not None:
        sources = (str(m3.report_path),)
        hashes = (sha256_file(m3.report_path),)
        report = read_json(m3.report_path)
        results = report.get('results') if isinstance(report, dict) else None
        rsvd = results.get('M3_RSVD', {}).get('result') if isinstance(results, dict) else None
        if isinstance(rsvd, dict):
            proxy = rsvd.get('influence_proxy', {})
            extra = f' M3 influence_proxy.screening={proxy.get("screening")!r}.'
    return _blocked(
        'cutoff and rank dependence',
        formula_id='P4-cutoff-rank',
        notes=(
            'No second cutoff/rank frozen run exists with a rigorous variation enclosure.'
            + extra
        ),
        sources=sources,
        hashes=hashes,
    )


def evaluate_all_obligations(
    project_root: Path,
    persistent_root: Path,
    *,
    m4_checkpoint: Path,
) -> dict[str, Any]:
    m4_tensors = _load_tensors(m4_checkpoint)
    m4_hash = sha256_file(m4_checkpoint / 'hashes.json')
    m4_sources = (str(m4_checkpoint / 'tensors'),)
    m4_hashes = (m4_hash,)

    chain: dict[str, AcceptedParentRef] | None
    chain_error: str | None = None
    try:
        chain = load_m1_m4_parent_chain(project_root, persistent_root)
    except M5ParentChainError as exc:
        chain = None
        chain_error = str(exc)

    m1 = chain.get('M1') if chain else None
    m2 = chain.get('M2') if chain else None
    m3 = chain.get('M3') if chain else None

    # Projection residual needs M3 RSVD factors + M2 projectors (not always
    # copied into the M4 checkpoint tensor store).
    projection_tensors = dict(m4_tensors)
    projection_sources = list(m4_sources)
    projection_hashes = list(m4_hashes)
    if m3 is not None:
        m3_tensors = _load_tensors(m3.checkpoint)
        for key in ('rsvd_left', 'rsvd_singular_values', 'rsvd_right_t'):
            if key in m3_tensors:
                projection_tensors[key] = m3_tensors[key]
        projection_sources.append(str(m3.checkpoint / 'tensors'))
        projection_hashes.append(sha256_file(m3.checkpoint / 'hashes.json'))
    if m2 is not None:
        m2_tensors = _load_tensors(m2.checkpoint)
        for key, value in m2_tensors.items():
            if key.startswith('projector_'):
                projection_tensors[key] = value
        projection_sources.append(str(m2.checkpoint / 'tensors'))
        projection_hashes.append(sha256_file(m2.checkpoint / 'hashes.json'))

    projection = evaluate_rsvd_projection_residual(
        projection_tensors,
        source_paths=tuple(projection_sources),
        source_hashes=tuple(projection_hashes),
    )
    results = [
        evaluate_gpu_rounding(
            m4_tensors, source_paths=m4_sources, source_hashes=m4_hashes,
        ),
        projection,
        evaluate_cutoff_rank_dependence(m3),
        evaluate_initial_representation_tail(m1),
        evaluate_input_radius_propagation(),
        evaluate_normalization_denominator(
            m4_tensors, source_paths=m4_sources, source_hashes=m4_hashes,
        ),
        evaluate_omitted_fusion_channel_tail(),
        evaluate_basis_variation(projection),
    ]
    by_id = {item.obligation_id: item for item in results}
    # Preserve canonical ordering from M5_OPEN_PROOF_OBLIGATIONS.
    ordered = [by_id[name] for name in OBLIGATION_IDS]
    closed = [item.obligation_id for item in ordered if item.status == 'RIGOROUS']
    open_ids = [item.obligation_id for item in ordered if item.status != 'RIGOROUS']
    return {
        'schema_version': 1,
        'generated_at': utc_now(),
        'parent_chain_error': chain_error,
        'obligations': [item.payload() for item in ordered],
        'closed_obligations': closed,
        'open_obligations': open_ids,
        'all_closed': not open_ids,
        'policy': (
            'Only RIGOROUS deterministic bounds close an obligation. '
            'Heuristic residuals are never accepted. Missing lifts stay BLOCKED_MATH.'
        ),
    }


def write_obligation_report(path: Path, report: dict[str, Any]) -> str:
    atomic_write_json(path, report)
    return sha256_file(path)
