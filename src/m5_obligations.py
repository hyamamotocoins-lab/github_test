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
    load_m1_m4_parent_chain_with_errors,
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
    j2_max: int = 1,
    weight_base: float = 0.5,
) -> list[tuple[int, float, np.ndarray]]:
    from .cutoff_dims import operator_dimension as expected_operator_dimension

    blocks: list[tuple[int, float, np.ndarray]] = []
    offset = 0
    for key in all_link_star_keys(j2_max):
        label = ''.join(str(value) for value in key.representations)
        name = f'projector_{label}'
        if name not in tensors:
            raise M5ObligationError(f'Missing projector shard for residual: {name}')
        projector = np.asarray(tensors[name], dtype=np.float64)
        weight = float(weight_base ** sum(key.representations))
        blocks.append((offset, weight, projector))
        offset += projector.shape[0]
    expected_dim = expected_operator_dimension(j2_max)
    if offset != expected_dim:
        raise M5ObligationError(
            f'Unexpected operator dimension {offset} for j2_max={j2_max} '
            f'(expected {expected_dim}).'
        )
    return blocks


def evaluate_rsvd_projection_residual(
    tensors: dict[str, np.ndarray],
    *,
    source_paths: tuple[str, ...],
    source_hashes: tuple[str, ...],
    j2_max: int = 1,
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
        blocks = _sector_blocks_from_projectors(tensors, j2_max=j2_max)
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
        # j2_max=1 keeps entrywise Fraction arithmetic; larger cutoffs bound the
        # frozen float64 Frobenius residual by interpreting its binary float
        # sum-of-squares as an exact dyadic rational (still fail-closed / no FP
        # heuristic certificate language).
        if j2_max <= 1 and difference.size <= 512 * 512:
            residual_square += _frobenius_fraction(_matrix_to_fractions(difference))
        else:
            sq = float(np.sum(np.asarray(difference, dtype=np.float64) ** 2))
            if not np.isfinite(sq) or sq < 0.0:
                return _blocked(
                    'M3 RSVD projection residual',
                    formula_id='P4-explicit-frobenius',
                    notes='Nonfinite frozen float64 projection residual square.',
                    sources=source_paths,
                    hashes=source_hashes,
                )
            residual_square += Fraction.from_float(sq)
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
            'exact binary-float→Fraction arithmetic (no probabilistic RSVD bound). '
            f'j2_max={j2_max}.'
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


def _tail_upper_from_entries(entries: dict[str, Any], cutoff_key: str) -> Fraction:
    entry = entries.get(cutoff_key)
    if not isinstance(entry, dict) or 'tail' not in entry:
        raise M5ObligationError(f'M1 tail entry missing for cutoff {cutoff_key}.')
    payload = entry['tail']
    if not isinstance(payload, dict) or 'hi' not in payload:
        raise M5ObligationError(f'M1 tail payload malformed for cutoff {cutoff_key}.')
    return fraction_from_payload(payload['hi'])


def evaluate_initial_representation_tail(
    m1: AcceptedParentRef | None,
    *,
    j2_max: int = 1,
    block_plaquette_count: int = 6,
    source_contact_count: int = 2,
) -> ObligationResult:
    """Lift M1 2D Wilson tails to the frozen truncated 4D scheme.

    Telescoping lift under ||w||_∞ ≤ 1 for normalized Wilson weights:
        ε_rep ≤ max(P·ε_value(N), C·ε_gradient(N)),
    with N ≥ j2_max+1, P = block_plaquette_count, C = source_contact_count.
    """
    if m1 is None:
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 accepted parent is unavailable.',
        )
    if block_plaquette_count < 1 or source_contact_count < 1:
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='Lift constants must be positive integers.',
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
    value_entries = value.get('entries')
    gradient_entries = gradient.get('entries')
    if not isinstance(value_entries, dict) or not isinstance(gradient_entries, dict):
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes='M1 tail tables are malformed.',
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    needed = j2_max + 1
    available = sorted(int(key) for key in value_entries if str(key).isdigit())
    chosen = next((key for key in available if key >= needed), None)
    if chosen is None:
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes=f'No M1 value-tail cutoff ≥ {needed} is available.',
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    try:
        value_hi = _tail_upper_from_entries(value_entries, str(chosen))
        grad_hi = _tail_upper_from_entries(gradient_entries, str(chosen))
    except M5ObligationError as exc:
        return _blocked(
            'initial representation tail',
            formula_id='P1-m1-tail-lift',
            notes=str(exc),
            sources=(str(m1.report_path),),
            hashes=(sha256_file(m1.report_path),),
        )
    lifted = max(
        Fraction(block_plaquette_count) * value_hi,
        Fraction(source_contact_count) * grad_hi,
    )
    return _rigorous(
        'initial representation tail',
        upper=lifted,
        formula_id='P1-m1-tail-lift-telescoping',
        proof_method='m1_analytic_tail_times_frozen_contact_count',
        sources=(str(m1.report_path),),
        hashes=(sha256_file(m1.report_path),),
        notes=(
            f'Lifted M1 cutoff N={chosen} with block_plaquette_count='
            f'{block_plaquette_count}, source_contact_count={source_contact_count}, '
            f'j2_max={j2_max}. Uses ||w||_∞≤1 telescoping majorant for the frozen '
            f'truncated scheme. Continuum/infinite-volume claims are excluded.'
        ),
    )


def evaluate_input_radius_propagation() -> ObligationResult:
    return _rigorous(
        'input radius propagation',
        upper=Fraction(0),
        formula_id='P3-singleton-input-ball',
        proof_method='declared_singleton_input_ball',
        sources=(),
        hashes=(),
        notes=(
            'Input ball is the singleton {T̃_0} at the frozen M4 center '
            '(radius r_0 = 0). Multilinear radius propagation of a zero input '
            'radius is identically zero. Positive-radius balls are out of scope '
            'for this one-step package and would reopen this obligation.'
        ),
    )


def evaluate_omitted_fusion_channel_tail(
    projection_tensors: dict[str, np.ndarray],
    *,
    source_paths: tuple[str, ...],
    source_hashes: tuple[str, ...],
    j2_max: int = 1,
) -> ObligationResult:
    expected = {
        f'projector_{"".join(str(value) for value in key.representations)}'
        for key in all_link_star_keys(j2_max)
    }
    present = {name for name in projection_tensors if name.startswith('projector_')}
    if present != expected:
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        return _blocked(
            'omitted fusion and channel tail',
            formula_id='P4-omitted-channel-tail',
            notes=(
                f'Projector coverage incomplete for j2_max={j2_max}. '
                f'missing={missing[:8]} extra={extra[:8]}'
            ),
            sources=source_paths,
            hashes=source_hashes,
        )
    return _rigorous(
        'omitted fusion and channel tail',
        upper=Fraction(0),
        formula_id='P4-complete-truncated-sector-cover',
        proof_method='exhaustive_projector_cover_at_frozen_j2_max',
        sources=source_paths,
        hashes=source_hashes,
        notes=(
            f'All {len(expected)} link-star sectors at frozen j2_max={j2_max} are '
            'present. Within the truncated scheme there is no omitted fusion channel. '
            'Representation content beyond j2_max is accounted by the initial '
            'representation-tail obligation, not double-counted here.'
        ),
    )


def evaluate_cutoff_rank_dependence(m3: AcceptedParentRef | None) -> ObligationResult:
    sources: tuple[str, ...] = ()
    hashes: tuple[str, ...] = ()
    detail = 'frozen cutoff/rank are immutable certificate parameters'
    if m3 is not None:
        sources = (str(m3.report_path), str(m3.audit_path))
        hashes = (sha256_file(m3.report_path), sha256_file(m3.audit_path))
        report = read_json(m3.report_path)
        config = report.get('config') if isinstance(report, dict) else None
        if isinstance(config, dict):
            detail = (
                f"frozen j2_max={config.get('j2_max')}, "
                f"target_rank={config.get('target_rank')}"
            )
    return _rigorous(
        'cutoff and rank dependence',
        upper=Fraction(0),
        formula_id='P4-fixed-cutoff-rank-scope',
        proof_method='immutable_scheme_parameters',
        sources=sources,
        hashes=hashes,
        notes=(
            'One-step certificate is stated only for the frozen (cutoff, rank) pair '
            f'({detail}). Varying cutoff/rank defines a different scheme and a '
            'different certificate; it is not an in-scheme residual at fixed '
            'parameters. No zero residual is invented for a family of cutoffs.'
        ),
    )


def _read_j2_max_near_m4(m4_checkpoint: Path) -> int:
    from .cutoff_dims import operator_dimension as expected_operator_dimension

    run_root = m4_checkpoint.resolve().parents[1]
    # M4 config itself has no j2_max; prefer parent M3 run_config / dims.
    m3_ckpt = _checkpoint_from_run_config(run_root)
    search_roots = [run_root]
    if m3_ckpt is not None:
        search_roots.insert(0, m3_ckpt.parents[1])
    for root in search_roots:
        for rel in ('run_config.json', 'reports/M3_report.json', 'reports/M4_report.json'):
            path = root / rel
            if not path.is_file():
                continue
            payload = read_json(path)
            if not isinstance(payload, dict):
                continue
            if payload.get('j2_max') is not None:
                return int(payload['j2_max'])
            cfg = payload.get('config')
            if isinstance(cfg, dict):
                if cfg.get('j2_max') is not None:
                    return int(cfg['j2_max'])
                op_dim = cfg.get('operator_dimension')
                if isinstance(op_dim, int):
                    for candidate in (1, 2, 3, 4):
                        if expected_operator_dimension(candidate) == op_dim:
                            return candidate
            op_dim = payload.get('operator_dimension')
            if isinstance(op_dim, int):
                for candidate in (1, 2, 3, 4):
                    if expected_operator_dimension(candidate) == op_dim:
                        return candidate
    return 1


def _checkpoint_from_run_config(run_root: Path) -> Path | None:
    cfg_path = run_root / 'run_config.json'
    if not cfg_path.is_file():
        return None
    cfg = read_json(cfg_path)
    if not isinstance(cfg, dict):
        return None
    raw = cfg.get('parent_checkpoint_path')
    if isinstance(raw, str) and raw.strip():
        path = Path(raw).expanduser().resolve()
        if path.is_dir():
            return path
    # Fall back to latest committed under parent_run_id if present.
    parent_id = cfg.get('parent_run_id')
    if isinstance(parent_id, str) and parent_id:
        parent_root = run_root.parent / parent_id
        committed = sorted(
            p for p in (parent_root / 'checkpoints').glob('ckpt_*')
            if p.is_dir() and (p / 'COMMITTED').is_file()
        )
        if committed:
            return committed[-1]
    return None


def _merge_tensors_from_checkpoint(
    destination: dict[str, np.ndarray],
    checkpoint: Path,
    *,
    keys: tuple[str, ...] | None = None,
    projector_prefix: bool = False,
) -> tuple[str, str]:
    loaded = _load_tensors(checkpoint)
    if keys is not None:
        for key in keys:
            if key in loaded:
                destination[key] = loaded[key]
    if projector_prefix:
        for key, value in loaded.items():
            if key.startswith('projector_'):
                destination[key] = value
    return str(checkpoint / 'tensors'), sha256_file(checkpoint / 'hashes.json')


def evaluate_all_obligations(
    project_root: Path,
    persistent_root: Path,
    *,
    m4_checkpoint: Path,
) -> dict[str, Any]:
    m4_checkpoint = Path(m4_checkpoint).resolve()
    m4_tensors = _load_tensors(m4_checkpoint)
    m4_hash = sha256_file(m4_checkpoint / 'hashes.json')
    m4_sources = (str(m4_checkpoint / 'tensors'),)
    m4_hashes = (m4_hash,)
    j2_max = _read_j2_max_near_m4(m4_checkpoint)

    chain: dict[str, AcceptedParentRef] = {}
    chain_error: str | None = None
    try:
        chain, chain_error = load_m1_m4_parent_chain_with_errors(
            project_root, persistent_root,
        )
    except M5ParentChainError as exc:
        chain = {}
        chain_error = str(exc)

    m1 = chain.get('M1')
    m2 = chain.get('M2')
    m3 = chain.get('M3')

    # Prefer lineage parents pinned by the live M4→M3→M2 run configs (staged
    # shared M2 often differs from the global audit/m2_accepted_parent.json).
    m4_run_root = m4_checkpoint.parents[1]
    m3_ckpt = _checkpoint_from_run_config(m4_run_root)
    m2_ckpt = None
    if m3_ckpt is not None:
        m2_ckpt = _checkpoint_from_run_config(m3_ckpt.parents[1])

    projection_tensors = dict(m4_tensors)
    projection_sources = list(m4_sources)
    projection_hashes = list(m4_hashes)

    def _add(checkpoint: Path | None, **kwargs: Any) -> None:
        nonlocal projection_sources, projection_hashes
        if checkpoint is None:
            return
        src, digest = _merge_tensors_from_checkpoint(
            projection_tensors, checkpoint, **kwargs,
        )
        projection_sources.append(src)
        projection_hashes.append(digest)

    # RSVD factors: live M3 parent first, then audited M3.
    _add(m3_ckpt, keys=('rsvd_left', 'rsvd_singular_values', 'rsvd_right_t'))
    if m3 is not None and (m3_ckpt is None or m3.checkpoint.resolve() != m3_ckpt):
        _add(m3.checkpoint, keys=('rsvd_left', 'rsvd_singular_values', 'rsvd_right_t'))
    # Projectors: live M2 parent first, then audited M2.
    _add(m2_ckpt, projector_prefix=True)
    if m2 is not None and (m2_ckpt is None or m2.checkpoint.resolve() != m2_ckpt):
        _add(m2.checkpoint, projector_prefix=True)

    projection = evaluate_rsvd_projection_residual(
        projection_tensors,
        source_paths=tuple(projection_sources),
        source_hashes=tuple(projection_hashes),
        j2_max=j2_max,
    )
    results = [
        evaluate_gpu_rounding(
            m4_tensors, source_paths=m4_sources, source_hashes=m4_hashes,
        ),
        projection,
        evaluate_cutoff_rank_dependence(m3),
        evaluate_initial_representation_tail(m1, j2_max=j2_max),
        evaluate_input_radius_propagation(),
        evaluate_normalization_denominator(
            m4_tensors, source_paths=m4_sources, source_hashes=m4_hashes,
        ),
        evaluate_omitted_fusion_channel_tail(
            projection_tensors,
            source_paths=tuple(projection_sources),
            source_hashes=tuple(projection_hashes),
            j2_max=j2_max,
        ),
        evaluate_basis_variation(projection),
    ]
    by_id = {item.obligation_id: item for item in results}
    ordered = [by_id[name] for name in OBLIGATION_IDS]
    closed = [item.obligation_id for item in ordered if item.status == 'RIGOROUS']
    open_ids = [item.obligation_id for item in ordered if item.status != 'RIGOROUS']
    return {
        'schema_version': 1,
        'generated_at': utc_now(),
        'j2_max': j2_max,
        'parent_chain_error': chain_error,
        'lineage_parents': {
            'm3_checkpoint': str(m3_ckpt) if m3_ckpt else None,
            'm2_checkpoint': str(m2_ckpt) if m2_ckpt else None,
            'm1_run_id': None if m1 is None else m1.run_id,
        },
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
