"""Campaign B candidate generation from versioned search space only.

Does not import Campaign C generators (invariant I3).
"""

from __future__ import annotations

from typing import Any

from ..common import canonical_json_bytes, sha256_bytes
from ..m7_status import CHANGE_S2
from .errors import CampaignFatalError, InvariantViolation
from .schemas import assert_staged_candidate, screening_only_payload


def _effective_projected_rank(target_rank: int) -> int:
    """M4-compatible perfect-square lift (same rule as m7_lineage)."""
    import math
    root = int(math.isqrt(int(target_rank)))
    if root * root == int(target_rank):
        return int(target_rank)
    root = int(math.ceil(math.sqrt(target_rank)))
    return root * root


def _short_hash(payload: Any) -> str:
    return sha256_bytes(canonical_json_bytes(payload))[:16]


def scheme_hash(scheme: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(scheme))


def normalized_scheme_key(scheme: dict[str, Any]) -> str:
    """Drop cosmetic fields; keep numerically operative parameters."""
    keys = (
        'change_class',
        'majorant_policy',
        'target_rank',
        'effective_projected_rank',
        'oversampling',
        'power_iterations',
        'perron_weight_strategy',
        'coupling_policy',
        'seed',
        'num_steps',
        'residual_tolerance',
        'residual_norm_model',
        'j2',
        'execution_mode',
    )
    normalized = {k: scheme[k] for k in keys if k in scheme}
    return sha256_bytes(canonical_json_bytes(normalized))


def _allowed_set(space: dict[str, Any], *path: str) -> set[Any] | None:
    cursor: Any = space
    for part in path:
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    if isinstance(cursor, list):
        return set(cursor)
    return None


def _assert_allowed(value: Any, allowed: set[Any] | None, label: str) -> None:
    if allowed is None:
        return
    if value not in allowed:
        raise InvariantViolation(f'{label}={value!r} not in search space')


def build_candidate_id(
    *,
    campaign_run_id: str,
    scheme: dict[str, Any],
    structural_key: str,
    proof_key: str,
    j2: int,
    source_tree_hash: str,
) -> str:
    digest = _short_hash({
        'campaign_run_id': campaign_run_id,
        'scheme': scheme,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'j2': int(j2),
        'source_tree_hash': source_tree_hash,
    })
    return f'B-{digest}'


def generate_campaign_b_queue_candidates(
    *,
    campaign_run_id: str,
    search_space: dict[str, Any],
    structural_key: str,
    proof_key: str,
    source_tree_hash: str,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
    limit: int | None = None,
    exclude_normalized_keys: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate only values present in the versioned search space."""
    if search_space.get('campaign') not in {'B_S2', 'B'}:
        raise InvariantViolation('refusing non-B search space')

    excluded = set(exclude_normalized_keys or ())

    rank_values = list(
        (search_space.get('rank') or {}).get('values')
        or (search_space.get('layers') or {}).get('target_rank')
        or []
    )
    rsvd = search_space.get('rsvd') or {}
    layers = search_space.get('layers') or {}
    oversampling = list(rsvd.get('oversampling') or layers.get('oversampling') or [])
    power_iterations = list(
        rsvd.get('power_iterations') or layers.get('power_iterations') or []
    )
    seeds = list(rsvd.get('seeds') or layers.get('seed') or [20260720])
    strategies = list(layers.get('perron_weight_strategy') or ['all_ones'])
    couplings = list(layers.get('coupling_policy') or ['uniform_full'])
    residual = search_space.get('residual') or {}
    tolerances = list(residual.get('tolerances') or [None])
    norm_models = list(residual.get('norm_models') or [None])
    staging = search_space.get('staging') or {}
    j2_values = [int(v) for v in staging.get('j2_values') or [2]]
    if any(j < 2 for j in j2_values):
        raise InvariantViolation('j2_values must all be >= 2')
    if staging.get('forbid_j2_1') is False:
        raise InvariantViolation('forbid_j2_1 must be true')

    if not rank_values or not oversampling or not power_iterations:
        raise CampaignFatalError('incomplete Campaign B search space')

    # Prefer higher ranks first (residual tightening).
    ranks = sorted((int(r) for r in rank_values), reverse=True)
    candidates: list[dict[str, Any]] = []
    seen_exact: set[str] = set()
    seen_normalized: set[str] = set()
    index = 0

    for j2 in sorted(j2_values):
        for target_rank in ranks:
            for over in oversampling:
                for power in power_iterations:
                    for seed in seeds:
                        for strategy in strategies:
                            for coupling in couplings:
                                for tol in tolerances:
                                    for norm in norm_models:
                                        index += 1
                                        projected = _effective_projected_rank(int(target_rank))
                                        scheme: dict[str, Any] = {
                                            'change_class': CHANGE_S2,
                                            'majorant_policy': 'S2_RANK_RESIDUAL_LINEAGE',
                                            'target_rank': int(target_rank),
                                            'effective_projected_rank': projected,
                                            'oversampling': int(over),
                                            'power_iterations': int(power),
                                            'perron_weight_strategy': strategy,
                                            'coupling_policy': coupling,
                                            'seed': int(seed),
                                            'num_steps': 3,
                                            'j2': int(j2),
                                            'execution_mode': 'staged',
                                        }
                                        if tol is not None:
                                            scheme['residual_tolerance'] = tol
                                        if norm is not None:
                                            scheme['residual_norm_model'] = norm

                                        cand_id = build_candidate_id(
                                            campaign_run_id=campaign_run_id,
                                            scheme=scheme,
                                            structural_key=structural_key,
                                            proof_key=proof_key,
                                            j2=j2,
                                            source_tree_hash=source_tree_hash,
                                        )
                                        exact_key = sha256_bytes(canonical_json_bytes({
                                            'candidate_id': cand_id,
                                            'scheme': scheme,
                                        }))
                                        norm_key = normalized_scheme_key(scheme)
                                        if (
                                            exact_key in seen_exact
                                            or norm_key in seen_normalized
                                            or norm_key in excluded
                                        ):
                                            continue
                                        seen_exact.add(exact_key)
                                        seen_normalized.add(norm_key)

                                        record = {
                                            'schema_version': 1,
                                            'candidate_id': cand_id,
                                            'index': index,
                                            'campaign_kind': 'B_S2',
                                            'change_class': CHANGE_S2,
                                            'scheme': scheme,
                                            'scheme_hash': scheme_hash(scheme),
                                            'normalized_scheme_key': norm_key,
                                            'j2': int(j2),
                                            'execution_mode': 'staged',
                                            'parent_m6_run_id': parent_m6_run_id,
                                            'parent_scheme_hash': parent_scheme_hash,
                                            'structural_key': structural_key,
                                            'proof_key': proof_key,
                                            'source_tree_hash': source_tree_hash,
                                            'state': 'PENDING',
                                            'priority_score': 0.0,
                                            **screening_only_payload(),
                                        }
                                        assert_staged_candidate(record)
                                        candidates.append(record)
                                        if limit is not None and len(candidates) >= limit:
                                            return candidates
    return candidates


def score_candidate(
    candidate: dict[str, Any],
    *,
    parent_q_upper: float,
    novelty_factor: float = 1.0,
    lineage_reuse_factor: float = 1.0,
    stability_factor: float = 1.0,
    predicted_runtime_sec: float = 10.0,
    runtime_floor: float = 1.0,
) -> float:
    """expected_q_improvement / runtime * modifiers (screening heuristic)."""
    from ..m7_lineage import screen_s2_candidate

    screen = screen_s2_candidate(
        candidate,
        parent_q_upper=parent_q_upper,
        parent_rank=16,
    )
    estimated = float(screen['estimated_q'])
    improvement = max(0.0, float(parent_q_upper) - estimated)
    denom = max(float(predicted_runtime_sec), float(runtime_floor))
    return (
        improvement / denom
        * float(novelty_factor)
        * float(lineage_reuse_factor)
        * float(stability_factor)
    )


def assign_priorities(
    candidates: list[dict[str, Any]],
    *,
    parent_q_upper: float,
) -> list[dict[str, Any]]:
    for candidate in candidates:
        candidate['priority_score'] = score_candidate(
            candidate,
            parent_q_upper=parent_q_upper,
        )
    candidates.sort(
        key=lambda c: (-float(c.get('priority_score') or 0.0), c['candidate_id']),
    )
    return candidates

