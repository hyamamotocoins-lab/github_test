"""Campaign B/C lineage planning and residual/cutoff model evaluation.

Full GPU rebuilds are gated by lineage_mode. Fixture modes exist only for
controller tests and must never be treated as continuum claims.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import atomic_write_json, utc_now
from .interval_kernel import ProofInterval, construct
from .m7_collatz_search import coerce_interval, evaluate_collatz
from .m7_replay import _pack, perron_for_strategy
from .m7_status import CHANGE_S2, CHANGE_S3


class M7LineageError(RuntimeError):
    """Raised when a lineage plan or residual/cutoff model cannot proceed."""


def is_perfect_square(value: int) -> bool:
    if value < 1:
        return False
    root = int(round(math.sqrt(value)))
    return root * root == value


def effective_projected_rank(target_rank: int) -> int:
    """Map a requested RSVD rank to an M4-compatible perfect square."""
    if is_perfect_square(target_rank):
        return int(target_rank)
    root = int(math.ceil(math.sqrt(target_rank)))
    return root * root


def build_s2_lineage_plan(
    candidate: dict[str, Any],
    *,
    parent_m6_run_id: str,
    search_run_id: str,
) -> dict[str, Any]:
    scheme = candidate.get('scheme') or {}
    target_rank = int(scheme.get('target_rank', 16))
    projected = effective_projected_rank(target_rank)
    oversampling = int(scheme.get('oversampling', 16))
    power_iterations = int(scheme.get('power_iterations', 2))
    digest = str(candidate.get('candidate_id', 'CAND')).replace('CAND-', '')[:12]
    m3_id = f'M3-{search_run_id[3:11]}S2-{digest}'
    m4_id = f'M4-{search_run_id[3:11]}S2-{digest}'
    m5_id = f'M5-{search_run_id[3:11]}S2-{digest}'
    m6_id = f'M6-{search_run_id[3:11]}S2-{digest}'
    geometry_ok = is_perfect_square(projected)
    return {
        'schema_version': 1,
        'change_class': CHANGE_S2,
        'candidate_id': candidate.get('candidate_id'),
        'scheme_hash': candidate.get('scheme_hash'),
        'parent_m6_run_id': parent_m6_run_id,
        'search_run_id': search_run_id,
        'requested_target_rank': target_rank,
        'effective_projected_rank': projected,
        'm4_geometry_compatible': geometry_ok,
        'parameters': {
            'target_rank': target_rank,
            'effective_projected_rank': projected,
            'oversampling': oversampling,
            'power_iterations': power_iterations,
            'seed': int(scheme.get('seed', 20260720)),
        },
        'child_run_ids': {
            'M3': m3_id,
            'M4': m4_id,
            'M5': m5_id,
            'M6': m6_id,
        },
        'invalidated_nodes': ['M3', 'M4', 'M5', 'M6'],
        'reused_artifacts': ['M2', 'M1', 'M0'],
        'execution_steps': [
            'create_or_resume_m3 with target_rank/oversampling/power_iterations',
            'ACCEPT M3 → rewrite audit/m3_accepted_parent.json for child M4',
            'create_or_resume_m4 with projected_rank=effective_projected_rank',
            'ACCEPT M4 → rewrite audit/m4_accepted_parent.json for child M5',
            'create_or_resume_m5(mode!=paperspace) with bond_dimension=rank',
            'create_or_resume_m6(mode!=paperspace) with bond_dimension=rank',
            'Feed child final_certificate into M7 independent verifier',
        ],
        'notes': (
            'S2 requires a new M3→M6 lineage under LOCK. '
            'M4 regroup demands a perfect-square projected_rank; non-square '
            'requests are lifted to the next square. '
            'q_cert>=1 on the child remains certificate failure only.'
        ),
        'generated_at': utc_now(),
    }


def write_lineage_plan(path: Path, plan: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, plan)
    return path


def _split_core_residual(cell: Any, residual_fraction: Fraction) -> tuple[Any, Any]:
    interval = coerce_interval(cell)
    width = interval.hi  # nonnegative majorant cell
    residual = width * residual_fraction
    core = width - residual
    if core < 0:
        core = Fraction(0)
        residual = width
    return construct(0, core), construct(0, residual)


def apply_s2_residual_model(
    entries: list[list[Any]],
    *,
    parent_rank: int,
    target_rank: int,
    oversampling: int,
    power_iterations: int,
    parent_oversampling: int = 16,
    parent_power_iterations: int = 2,
    residual_fraction: Fraction = Fraction(3, 5),
) -> list[list[ProofInterval]]:
    """Optimistic residual shrink for fixture/controller tests only.

    Models truncation residual ~ 1/rank^2 with mild oversampling/power gains.
    Does not replace a live M3 RSVD residual certificate.
    """
    if target_rank < 1 or parent_rank < 1:
        raise M7LineageError('Ranks must be positive.')
    rank_factor = Fraction(parent_rank * parent_rank, target_rank * target_rank)
    over_factor = Fraction(parent_oversampling, max(oversampling, 1))
    # More power iterations reduce residual; never amplify above 1.
    power_factor = Fraction(parent_power_iterations + 1, power_iterations + 1)
    shrink = rank_factor * over_factor * power_factor
    if shrink > 1:
        shrink = Fraction(1)
    rebuilt: list[list[ProofInterval]] = []
    for row in entries:
        out_row: list[ProofInterval] = []
        for cell in row:
            core, residual = _split_core_residual(cell, residual_fraction)
            new_hi = core.hi + residual.hi * shrink
            out_row.append(construct(0, new_hi))
        rebuilt.append(out_row)
    return rebuilt


def evaluate_s2_fixture_residual(
    package_root: Path,
    candidate: dict[str, Any],
    *,
    parent_rank: int = 16,
) -> dict[str, Any]:
    """Rigorous Collatz on a residual-shrunk majorant (fixture/controller only)."""
    from .common import read_json

    scheme = candidate.get('scheme') or {}
    if scheme.get('change_class') != CHANGE_S2:
        raise M7LineageError('S2 residual model requires change_class=S2.')

    influence = read_json(package_root / 'final_influence_matrix.json')
    bound = read_json(package_root / 'final_bound.json')
    if not isinstance(influence, dict):
        raise M7LineageError('Missing influence matrix for S2 residual model.')
    labels = list(influence.get('labels') or [])
    entries = influence.get('entries')
    if not isinstance(entries, list) or not labels:
        raise M7LineageError('Influence matrix malformed.')
    outside = construct(0)
    if isinstance(bound, dict) and isinstance(bound.get('outside_matrix_tail'), dict):
        outside = bound['outside_matrix_tail']

    target_rank = int(scheme.get('target_rank', parent_rank))
    working = apply_s2_residual_model(
        [list(row) for row in entries],
        parent_rank=parent_rank,
        target_rank=target_rank,
        oversampling=int(scheme.get('oversampling', 16)),
        power_iterations=int(scheme.get('power_iterations', 2)),
    )
    perron = perron_for_strategy(
        str(scheme.get('perron_weight_strategy', 'all_ones')),
        labels,
        working,
    )
    result = evaluate_collatz(working, labels, perron, outside_tail=outside)
    notes = (
        f'S2_FIXTURE_RESIDUAL_MODEL rank {parent_rank}->{target_rank}; '
        'not a live M3→M6 lineage certificate.'
    )
    packed = _pack(candidate, result, notes=notes)
    packed['lineage_mode'] = 'fixture_residual'
    packed['effective_projected_rank'] = effective_projected_rank(target_rank)
    return packed


def screen_s2_candidate(
    candidate: dict[str, Any],
    *,
    parent_q_upper: float,
    parent_rank: int = 16,
) -> dict[str, Any]:
    """Floating-point screening only — never emits CERTIFIED."""
    scheme = candidate.get('scheme') or {}
    target_rank = int(scheme.get('target_rank', parent_rank))
    projected = effective_projected_rank(target_rank)
    oversampling = int(scheme.get('oversampling', 16))
    power_iterations = int(scheme.get('power_iterations', 2))
    rank_factor = (parent_rank / max(target_rank, 1)) ** 2
    over_factor = 16 / max(oversampling, 1)
    power_factor = (2 + 1) / (power_iterations + 1)
    # Assume ~20% of q is residual-dominated (screening heuristic).
    residual_share = 0.20
    core_share = 1.0 - residual_share
    estimated_q = parent_q_upper * (
        core_share + residual_share * rank_factor * over_factor * power_factor
    )
    if estimated_q < 0.90 and is_perfect_square(projected):
        status = 'SCREEN_PROMISING'
    elif estimated_q < parent_q_upper * 0.98:
        status = 'SCREEN_INCONCLUSIVE'
    else:
        status = 'SCREEN_REJECTED'
    return {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'screen_status': status,
        'estimated_q': format(estimated_q, '.17g'),
        'parent_q_upper': format(parent_q_upper, '.17g'),
        'effective_projected_rank': projected,
        'm4_geometry_compatible': is_perfect_square(projected),
        'certified': False,
        'notes': 'Screening only; CERTIFIED is forbidden from screening.',
    }


def build_s3_lineage_plan(
    candidate: dict[str, Any],
    *,
    parent_m6_run_id: str,
    search_run_id: str,
    parent_j2_max: int = 1,
) -> dict[str, Any]:
    scheme = candidate.get('scheme') or {}
    j2_max = int(scheme.get('j2_max', parent_j2_max))
    channel_policy = str(scheme.get('channel_policy', 'complete_at_cutoff'))
    block_geometry = str(scheme.get('block_geometry', 'current'))
    digest = str(candidate.get('candidate_id', 'CAND')).replace('CAND-', '')[:12]
    tag = search_run_id[3:11]
    m2_id = f'M2-{tag}S3-{digest}'
    m3_id = f'M3-{tag}S3-{digest}'
    m4_id = f'M4-{tag}S3-{digest}'
    m5_id = f'M5-{tag}S3-{digest}'
    m6_id = f'M6-{tag}S3-{digest}'
    execution_blocked = j2_max != 1
    return {
        'schema_version': 1,
        'change_class': CHANGE_S3,
        'candidate_id': candidate.get('candidate_id'),
        'scheme_hash': candidate.get('scheme_hash'),
        'parent_m6_run_id': parent_m6_run_id,
        'search_run_id': search_run_id,
        'parameters': {
            'j2_max': j2_max,
            'parent_j2_max': parent_j2_max,
            'channel_policy': channel_policy,
            'block_geometry': block_geometry,
            'seed': int(scheme.get('seed', 20260720)),
        },
        'child_run_ids': {
            'M2': m2_id,
            'M3': m3_id,
            'M4': m4_id,
            'M5': m5_id,
            'M6': m6_id,
        },
        'invalidated_nodes': ['M2', 'M3', 'M4', 'M5', 'M6'],
        'reused_artifacts': ['M1', 'M0'],
        'execution_blocked_by_math_lock': execution_blocked,
        'execution_steps': [
            'HUMAN REVIEW: approve Campaign C scheme + governing-doc delta',
            'Unlock M2Config/M3Config for j2_max (and derived sector dims) if >1',
            'create_or_resume_m2 with j2_max / block_geometry / channel_policy',
            'ACCEPT M2 → rewrite audit/m2_accepted_parent.json',
            'create_or_resume_m3 on new M2 (sector_count/operator_dimension must match)',
            'ACCEPT M3 → M4 → M5 → M6 child lineage (non-paperspace run IDs)',
            'Feed child final_certificate into M7 independent verifier',
        ],
        'notes': (
            'S3 requires a new M2→M6 lineage under LOCK and human review. '
            'Current M2/M3 pilots are fail-closed at j2_max=1; higher cutoffs '
            'need dimension unlock before execute. '
            'q_cert>=1 on the child remains certificate failure only.'
        ),
        'generated_at': utc_now(),
    }


def apply_s3_cutoff_model(
    entries: list[list[Any]],
    *,
    parent_j2_max: int,
    j2_max: int,
    channel_policy: str,
    block_geometry: str,
    truncation_fraction: Fraction = Fraction(7, 10),
) -> list[list[ProofInterval]]:
    """Optimistic cutoff/geometry shrink for fixture/controller tests only."""
    if j2_max < 1 or parent_j2_max < 1:
        raise M7LineageError('j2_max values must be positive.')
    # Higher cutoff reduces representation truncation; geometry_B / pruned
    # channels add mild extra shrink (screening-style, not a proof).
    cutoff_factor = Fraction(parent_j2_max, j2_max) ** 2
    channel_factor = (
        Fraction(4, 5) if channel_policy == 'certified_pruned' else Fraction(1)
    )
    geometry_factor = (
        Fraction(3, 4) if block_geometry == 'approved_geometry_B' else Fraction(1)
    )
    shrink = cutoff_factor * channel_factor * geometry_factor
    if shrink > 1:
        shrink = Fraction(1)
    rebuilt: list[list[ProofInterval]] = []
    for row in entries:
        out_row: list[ProofInterval] = []
        for cell in row:
            core, truncation = _split_core_residual(cell, truncation_fraction)
            new_hi = core.hi + truncation.hi * shrink
            out_row.append(construct(0, new_hi))
        rebuilt.append(out_row)
    return rebuilt


def evaluate_s3_fixture_cutoff(
    package_root: Path,
    candidate: dict[str, Any],
    *,
    parent_j2_max: int = 1,
) -> dict[str, Any]:
    """Rigorous Collatz on a cutoff-shrunk majorant (fixture/controller only)."""
    from .common import read_json

    scheme = candidate.get('scheme') or {}
    if scheme.get('change_class') != CHANGE_S3:
        raise M7LineageError('S3 cutoff model requires change_class=S3.')

    influence = read_json(package_root / 'final_influence_matrix.json')
    bound = read_json(package_root / 'final_bound.json')
    if not isinstance(influence, dict):
        raise M7LineageError('Missing influence matrix for S3 cutoff model.')
    labels = list(influence.get('labels') or [])
    entries = influence.get('entries')
    if not isinstance(entries, list) or not labels:
        raise M7LineageError('Influence matrix malformed.')
    outside = construct(0)
    if isinstance(bound, dict) and isinstance(bound.get('outside_matrix_tail'), dict):
        outside = bound['outside_matrix_tail']

    j2_max = int(scheme.get('j2_max', parent_j2_max))
    working = apply_s3_cutoff_model(
        [list(row) for row in entries],
        parent_j2_max=parent_j2_max,
        j2_max=j2_max,
        channel_policy=str(scheme.get('channel_policy', 'complete_at_cutoff')),
        block_geometry=str(scheme.get('block_geometry', 'current')),
    )
    perron = perron_for_strategy(
        str(scheme.get('perron_weight_strategy', 'all_ones')),
        labels,
        working,
    )
    result = evaluate_collatz(working, labels, perron, outside_tail=outside)
    notes = (
        f'S3_FIXTURE_CUTOFF_MODEL j2_max {parent_j2_max}->{j2_max}; '
        'not a live M2→M6 lineage certificate.'
    )
    packed = _pack(candidate, result, notes=notes)
    packed['lineage_mode'] = 'fixture_residual'
    packed['j2_max'] = j2_max
    return packed


def screen_s3_candidate(
    candidate: dict[str, Any],
    *,
    parent_q_upper: float,
    parent_j2_max: int = 1,
) -> dict[str, Any]:
    """Floating-point S3 screening only — never emits CERTIFIED."""
    scheme = candidate.get('scheme') or {}
    j2_max = int(scheme.get('j2_max', parent_j2_max))
    channel_policy = str(scheme.get('channel_policy', 'complete_at_cutoff'))
    block_geometry = str(scheme.get('block_geometry', 'current'))
    # Assume a large share of the failed majorant is cutoff/truncation dominated
    # when changing algebraic scheme (more optimistic than S2 residual share).
    truncation_share = 0.70
    core_share = 1.0 - truncation_share
    cutoff_factor = (parent_j2_max / max(j2_max, 1)) ** 2
    channel_factor = 0.80 if channel_policy == 'certified_pruned' else 1.0
    geometry_factor = 0.75 if block_geometry == 'approved_geometry_B' else 1.0
    estimated_q = parent_q_upper * (
        core_share
        + truncation_share * cutoff_factor * channel_factor * geometry_factor
    )
    execution_blocked = j2_max != 1
    if estimated_q < 0.90 and not execution_blocked:
        status = 'SCREEN_PROMISING'
    elif estimated_q < 0.90 and execution_blocked:
        status = 'SCREEN_INCONCLUSIVE'  # promising numerics, math lock remains
    elif estimated_q < parent_q_upper * 0.98:
        status = 'SCREEN_INCONCLUSIVE'
    else:
        status = 'SCREEN_REJECTED'
    return {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'screen_status': status,
        'estimated_q': format(estimated_q, '.17g'),
        'parent_q_upper': format(parent_q_upper, '.17g'),
        'j2_max': j2_max,
        'channel_policy': channel_policy,
        'block_geometry': block_geometry,
        'execution_blocked_by_math_lock': execution_blocked,
        'certified': False,
        'notes': (
            'Screening only; CERTIFIED is forbidden from screening. '
            'j2_max>1 requires M2/M3 unlock before execute.'
        ),
    }
