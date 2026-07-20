"""M7 candidate generation for Campaign A."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterator

from .common import canonical_json_bytes
from .m7_config import campaign_a_search_space
from .m7_status import CHANGE_S0, CHANGE_S1


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
