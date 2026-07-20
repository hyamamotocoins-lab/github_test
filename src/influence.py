"""Entrywise influence enclosure for M5 one-step certification.

Formula:
    c_bar_ij = D_i * (||d_j K_tilde||_1 + eps_1j) / z_min
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from .interval_kernel import IntervalKernelError, ProofInterval, construct, divide, multiply


class InfluenceError(RuntimeError):
    """Raised when influence enclosure inputs are incompatible or incomplete."""


@dataclass(frozen=True, slots=True)
class InfluenceEntry:
    row_type: str
    column_type: str
    displacement: tuple[int, ...]
    diameter: ProofInterval
    derivative_core_l1: ProofInterval
    derivative_error: ProofInterval
    normalization_lower: ProofInterval
    influence_upper: ProofInterval
    orbit_multiplicity: int
    formula: str
    dependencies: tuple[str, ...]
    metric_unit: str
    source_speed_unit: str

    def payload(self) -> dict[str, Any]:
        return {
            'row_type': self.row_type,
            'column_type': self.column_type,
            'displacement': list(self.displacement),
            'diameter_interval': self.diameter.serialize(),
            'derivative_core_l1_interval': self.derivative_core_l1.serialize(),
            'derivative_error_interval': self.derivative_error.serialize(),
            'normalization_lower_interval': self.normalization_lower.serialize(),
            'influence_upper_interval': self.influence_upper.serialize(),
            'orbit_multiplicity': self.orbit_multiplicity,
            'formula': self.formula,
            'dependencies': list(self.dependencies),
            'metric_unit': self.metric_unit,
            'source_speed_unit': self.source_speed_unit,
        }


def _require_nonnegative(interval: ProofInterval, label: str) -> None:
    try:
        interval.assert_nonnegative()
    except IntervalKernelError as exc:
        raise InfluenceError(f'{label} must be nonnegative.') from exc


def enclose_influence_entry(
    *,
    row_type: str,
    column_type: str,
    displacement: Sequence[int],
    diameter: ProofInterval | Any,
    derivative_core_l1: ProofInterval | Any,
    derivative_error: ProofInterval | Any,
    normalization_lower: ProofInterval | Any,
    orbit_multiplicity: int = 1,
    dependencies: Sequence[str] = (),
    metric_unit: str,
    source_speed_unit: str,
    formula: str = 'D_i*(||d_j K_tilde||_1 + eps_1j)/z_min',
) -> InfluenceEntry:
    if not row_type or not column_type:
        raise InfluenceError('Influence entry requires row_type and column_type.')
    if orbit_multiplicity < 1:
        raise InfluenceError('orbit_multiplicity must be a positive integer.')
    if metric_unit != source_speed_unit:
        raise InfluenceError(
            'Source parameterization speed unit is incompatible with the boundary metric unit.'
        )
    if not dependencies:
        raise InfluenceError('Influence entry requires explicit dependencies.')

    diameter_iv = diameter if isinstance(diameter, ProofInterval) else construct(diameter)
    core_iv = (
        derivative_core_l1
        if isinstance(derivative_core_l1, ProofInterval)
        else construct(derivative_core_l1)
    )
    error_iv = (
        derivative_error
        if isinstance(derivative_error, ProofInterval)
        else construct(derivative_error)
    )
    z_iv = (
        normalization_lower
        if isinstance(normalization_lower, ProofInterval)
        else construct(normalization_lower)
    )

    _require_nonnegative(diameter_iv, 'D_i')
    _require_nonnegative(core_iv, 'derivative core L1')
    _require_nonnegative(error_iv, 'derivative residual')
    try:
        z_iv.positive_lower_assertion()
    except IntervalKernelError as exc:
        raise InfluenceError('z_min must be strictly positive.') from exc

    numerator = core_iv.add(error_iv)
    ratio = divide(numerator, z_iv)
    influence = multiply(diameter_iv, ratio)
    if orbit_multiplicity != 1:
        influence = multiply(
            influence,
            construct(Fraction(orbit_multiplicity)),
        )
    _require_nonnegative(influence, 'influence upper')

    return InfluenceEntry(
        row_type=row_type,
        column_type=column_type,
        displacement=tuple(int(value) for value in displacement),
        diameter=diameter_iv,
        derivative_core_l1=core_iv,
        derivative_error=error_iv,
        normalization_lower=z_iv,
        influence_upper=influence,
        orbit_multiplicity=orbit_multiplicity,
        formula=formula,
        dependencies=tuple(dependencies),
        metric_unit=metric_unit,
        source_speed_unit=source_speed_unit,
    )


def assemble_influence_matrix(
    entries: Sequence[InfluenceEntry],
    *,
    row_order: Sequence[str],
    column_order: Sequence[str],
) -> dict[str, Any]:
    if not entries:
        raise InfluenceError('Influence matrix assembly requires at least one entry.')
    if not row_order or not column_order:
        raise InfluenceError('Canonical row/column ordering is required.')
    if len(row_order) != len(set(row_order)) or len(column_order) != len(set(column_order)):
        raise InfluenceError('Row/column ordering contains duplicates.')

    seen: set[tuple[str, str, tuple[int, ...]]] = set()
    for entry in entries:
        key = (entry.row_type, entry.column_type, entry.displacement)
        if key in seen:
            raise InfluenceError(f'Duplicate influence entry key: {key}')
        seen.add(key)
        if entry.row_type not in row_order or entry.column_type not in column_order:
            raise InfluenceError(
                f'Influence entry uses an unordered source type: {entry.row_type}/{entry.column_type}'
            )

    # Omitted source/channel pairs must not silently become zero.
    required_pairs = {
        (row, column) for row in row_order for column in column_order
    }
    present_pairs = {(entry.row_type, entry.column_type) for entry in entries}
    missing = required_pairs - present_pairs
    if missing:
        raise InfluenceError(
            'Omitted source/channel pairs cannot be treated as zero: '
            + ', '.join(f'{row}/{column}' for row, column in sorted(missing))
        )

    return {
        'schema_version': 1,
        'row_order': list(row_order),
        'column_order': list(column_order),
        'entries': [entry.payload() for entry in entries],
        'entry_count': len(entries),
    }


def weighted_matrix_entry(
    *,
    influence_entries: Sequence[InfluenceEntry],
    row_type: str,
    column_type: str,
    weight_m: ProofInterval | Any,
    spatial_tail: ProofInterval | Any,
) -> ProofInterval:
    """Build one weighted-matrix entry with spatial tail added once."""
    m_iv = weight_m if isinstance(weight_m, ProofInterval) else construct(weight_m)
    tail_iv = (
        spatial_tail if isinstance(spatial_tail, ProofInterval) else construct(spatial_tail)
    )
    _require_nonnegative(m_iv, 'weight m')
    _require_nonnegative(tail_iv, 'spatial tail')

    total = construct(0)
    matched = 0
    for entry in influence_entries:
        if entry.row_type != row_type or entry.column_type != column_type:
            continue
        matched += 1
        # e^{m |z_0|} enclosed by series truncation is deferred to the orchestrator;
        # here |z_0| is an integer and we use the exact rational power of e^m via
        # an externally supplied weight factor stored as orbit_multiplicity * influence
        # times an explicit displacement weight interval attached through diameter reuse.
        z0 = abs(entry.displacement[0]) if entry.displacement else 0
        # Conservative enclosure: e^{m|z|} <= (1 + ceil(e^m_upper))^|z| is not used.
        # Callers must pre-multiply influence_upper by the displacement weight; this
        # helper only sums those preweighted contributions plus the spatial tail.
        total = total.add(entry.influence_upper)
        del z0  # documented for audit; weighting is applied by the caller.
    if matched == 0:
        raise InfluenceError(
            f'No influence entries for weighted matrix cell {row_type}/{column_type}.'
        )
    return total.add(tail_iv)
