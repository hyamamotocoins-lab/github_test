"""M7 candidate generation for Campaign A/B/C."""

from __future__ import annotations

import hashlib
from typing import Any

from .common import canonical_json_bytes
from .m7_config import (
    campaign_a_search_space,
    campaign_b_search_space,
    campaign_c_search_space,
)
from .m7_lineage import effective_projected_rank
from .m7_status import CHANGE_S0, CHANGE_S1, CHANGE_S2, CHANGE_S3


def scheme_hash(scheme: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(scheme)).hexdigest()


def generate_campaign_a_candidates(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
    limit: int = 64,
) -> list[dict[str, Any]]:
    space = campaign_a_search_space()
    layers = space['layers']
    candidates: list[dict[str, Any]] = []
    index = 0

    # S0 first: perron strategies on inherited matrix.
    for strategy in layers['perron_weight_strategy']:
        for coupling in layers['coupling_policy']:
            index += 1
            scheme = {
                'change_class': CHANGE_S0,
                'majorant_policy': 'PARENT_MATRIX_REWEIGHT_ONLY',
                'perron_weight_strategy': strategy,
                'coupling_policy': coupling,
                'num_steps': 3,
            }
            candidates.append(_candidate(index, scheme, parent_m6_run_id, parent_scheme_hash))
            if len(candidates) >= limit:
                return candidates

    # S1: composition policy changes.
    for policy in layers['majorant_policy']:
        for strategy in layers['perron_weight_strategy']:
            for coupling in layers['coupling_policy']:
                for subdiv in layers['input_subdivision']:
                    index += 1
                    scheme = {
                        'change_class': CHANGE_S1,
                        'majorant_policy': policy,
                        'perron_weight_strategy': strategy,
                        'coupling_policy': coupling,
                        'input_subdivision': subdiv,
                        'source_partition': 'current',
                        'num_steps': 3,
                    }
                    candidates.append(
                        _candidate(index, scheme, parent_m6_run_id, parent_scheme_hash)
                    )
                    if len(candidates) >= limit:
                        return candidates
    return candidates


def generate_campaign_b_candidates(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """S2 numerical-representation candidates (rank / RSVD quality)."""
    space = campaign_b_search_space()
    layers = space['layers']
    candidates: list[dict[str, Any]] = []
    index = 0
    # Prefer higher ranks first (Campaign B goal is residual tightening).
    ranks = sorted((int(r) for r in layers['target_rank']), reverse=True)
    for target_rank in ranks:
        for oversampling in layers['oversampling']:
            for power_iterations in layers['power_iterations']:
                for strategy in layers['perron_weight_strategy']:
                    for coupling in layers['coupling_policy']:
                        index += 1
                        projected = effective_projected_rank(target_rank)
                        scheme = {
                            'change_class': CHANGE_S2,
                            'majorant_policy': 'S2_RANK_RESIDUAL_LINEAGE',
                            'target_rank': int(target_rank),
                            'effective_projected_rank': projected,
                            'oversampling': int(oversampling),
                            'power_iterations': int(power_iterations),
                            'perron_weight_strategy': strategy,
                            'coupling_policy': coupling,
                            'seed': 20260720,
                            'num_steps': 3,
                        }
                        candidates.append(
                            _candidate(
                                index, scheme, parent_m6_run_id, parent_scheme_hash,
                            )
                        )
                        if len(candidates) >= limit:
                            return candidates
    return candidates


def generate_fixture_contractive_candidate(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
) -> dict[str, Any]:
    scheme = {
        'change_class': CHANGE_S0,
        'majorant_policy': 'FIXTURE_CONTRACTIVE_REFERENCE',
        'perron_weight_strategy': 'all_ones',
        'coupling_policy': 'uniform_full',
        'num_steps': 3,
        'fixture': 'make_contractive_fixture_inputs',
    }
    return _candidate(0, scheme, parent_m6_run_id, parent_scheme_hash)


def generate_fixture_s2_cert_candidate(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
) -> dict[str, Any]:
    """High-rank S2 fixture intended to pass residual-model Collatz (<1)."""
    scheme = {
        'change_class': CHANGE_S2,
        'majorant_policy': 'S2_RANK_RESIDUAL_LINEAGE',
        'target_rank': 64,
        'effective_projected_rank': 64,
        'oversampling': 24,
        'power_iterations': 3,
        'perron_weight_strategy': 'all_ones',
        'coupling_policy': 'uniform_full',
        'seed': 20260720,
        'num_steps': 3,
        'fixture': 's2_residual_model',
    }
    return _candidate(0, scheme, parent_m6_run_id, parent_scheme_hash)


def generate_campaign_c_candidates(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
    limit: int = 64,
) -> list[dict[str, Any]]:
    """S3 algebraic/geometry candidates (cutoff / channels / block geometry)."""
    space = campaign_c_search_space()
    layers = space['layers']
    candidates: list[dict[str, Any]] = []
    index = 0
    # Prefer staged j2=2 (q<1 hunt) and instant j2=1 before higher gated cutoffs.
    j2_all = sorted({int(v) for v in layers['j2_max']})
    j2_values = (
        [j for j in j2_all if j == 2]
        + [j for j in j2_all if j == 1]
        + [j for j in reversed(j2_all) if j not in {1, 2}]
    )
    seeds = [int(s) for s in layers.get('seed', [20260720])]
    for j2_max in j2_values:
        for channel_policy in layers['channel_policy']:
            for block_geometry in layers['block_geometry']:
                for strategy in layers['perron_weight_strategy']:
                    for coupling in layers['coupling_policy']:
                        for seed in seeds:
                            index += 1
                            scheme = {
                                'change_class': CHANGE_S3,
                                'majorant_policy': 'S3_GEOMETRY_CUTOFF_LINEAGE',
                                'j2_max': int(j2_max),
                                'channel_policy': channel_policy,
                                'block_geometry': block_geometry,
                                'perron_weight_strategy': strategy,
                                'coupling_policy': coupling,
                                'seed': seed,
                                'num_steps': 3,
                            }
                            candidates.append(
                                _candidate(
                                    index, scheme, parent_m6_run_id, parent_scheme_hash,
                                )
                            )
                            if len(candidates) >= limit:
                                return candidates
    return candidates


def generate_fixture_s3_cert_candidate(
    *,
    parent_m6_run_id: str,
    parent_scheme_hash: str,
) -> dict[str, Any]:
    """High-cutoff S3 fixture intended to pass cutoff-model Collatz (<1)."""
    scheme = {
        'change_class': CHANGE_S3,
        'majorant_policy': 'S3_GEOMETRY_CUTOFF_LINEAGE',
        'j2_max': 4,
        'channel_policy': 'certified_pruned',
        'block_geometry': 'approved_geometry_B',
        'perron_weight_strategy': 'all_ones',
        'coupling_policy': 'uniform_full',
        'seed': 20260720,
        'num_steps': 3,
        'fixture': 's3_cutoff_model',
    }
    return _candidate(0, scheme, parent_m6_run_id, parent_scheme_hash)


def _candidate(
    index: int,
    scheme: dict[str, Any],
    parent_m6_run_id: str,
    parent_scheme_hash: str,
) -> dict[str, Any]:
    digest = scheme_hash(scheme)[:12]
    candidate_id = f'CAND-{index:06d}-{digest}'
    return {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'scheme_hash': 'sha256:' + scheme_hash(scheme),
        'parent_scheme_hash': parent_scheme_hash,
        'parent_m6_run_id': parent_m6_run_id,
        'change_class': scheme['change_class'],
        'changed_parameters': sorted(scheme.keys()),
        'scheme': scheme,
        'status': 'PROPOSED',
    }
