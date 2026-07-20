"""Frozen M6 LOCK conventions (pre-implementation freeze)."""

from __future__ import annotations

from typing import Any

from .m6_status import ERROR_LEDGER_IDS
from .source_channels import SOURCE_CLASSES


def m6_lock_payload(*, num_steps: int, j2_max: int, bond_dimension: int) -> dict[str, Any]:
    labels = [source.value for source in SOURCE_CLASSES]
    return {
        'schema_version': 1,
        'lock_id': 'M6-LOCK-20260720',
        'scope': (
            'finite-cutoff, finite-step 4D SU(2) truncated RG influence certificate; '
            'no continuum/mass-gap claim'
        ),
        'tensor_norm': {
            'name': 'frobenius',
            'arithmetic': 'exact_binary_float_to_fraction_outward',
            'metric_unit': 'lattice',
            'source_speed_unit': 'lattice',
        },
        'representation_cutoff': {
            'j2_max': j2_max,
            'adaptive_default': False,
            'adaptive_rule': (
                'tail > 25% margin → increase cutoff; '
                'total residual > 50% margin → forbid next step'
            ),
        },
        'retained_channels': {
            'geometry': '4d_link_star_six_leg',
            'fusion_tree': 'left-associated',
            'orientations': [1, -1, 1, -1, 1, -1],
            'bond_dimension': bond_dimension,
            'policy': (
                'exhaustive projector cover at frozen j2_max; '
                'beyond-cutoff content owned by representation-tail leaf'
            ),
        },
        'conventions': {
            'orientation': 'canonical_su2',
            'phase': 'real_positive_characters',
            'basis_equivalence': 'U T_arm U^* = T_PW',
            'rsvd_basis': 'frozen_m3_parent',
        },
        'source_classes': labels,
        'source_class_order_hash_policy': 'canonical_json_of_labels',
        'error_ledger_ids': list(ERROR_LEDGER_IDS),
        'error_ledger_names': {
            'E1': 'GPU / rounding residual',
            'E2': 'RSVD / low-rank projection residual',
            'E3': 'cutoff / rank in-scheme variation',
            'E4': 'representation tail',
            'E5': 'input radius propagation',
            'E6': 'normalization / denominator error',
            'E7': 'omitted fusion / channel tail',
            'E8': 'basis variation residual',
            'E9': 'multi-step derivative / chain-rule residual',
            'E10': 'outside-matrix / displacement tail',
            'E11': 'Collatz / weighting residual',
            'E12': 'independent-verifier disagreement',
        },
        'num_steps_default': num_steps,
        'zero_fill_policy': 'forbidden',
        'certification_vocabulary': [
            'STEP_ENCLOSED', 'BLOCKED_MATH', 'NOT_CERTIFIED', 'CERTIFIED',
        ],
    }
