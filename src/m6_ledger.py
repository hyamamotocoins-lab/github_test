"""M6 error ledger helpers."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Mapping

from .exact_arithmetic import fraction_decimal_text
from .interval_kernel import construct
from .m6_status import ERROR_LEDGER_IDS


def empty_ledger_template() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'policy': 'Missing leaf stays BLOCKED_MATH; zero-fill forbidden.',
        'leaves': {
            leaf_id: {
                'status': 'OPEN',
                'upper_bound': None,
                'notes': 'unset',
            }
            for leaf_id in ERROR_LEDGER_IDS
        },
    }


def close_leaf(
    ledger: dict[str, Any],
    leaf_id: str,
    *,
    upper: Fraction,
    notes: str,
    proof_method: str,
) -> None:
    if leaf_id not in ERROR_LEDGER_IDS:
        raise ValueError(f'Unknown ledger leaf: {leaf_id}')
    ledger['leaves'][leaf_id] = {
        'status': 'RIGOROUS',
        'upper_bound': construct(0, upper).serialize(),
        'upper_decimal': fraction_decimal_text(upper),
        'notes': notes,
        'proof_method': proof_method,
    }


def all_leaves_closed(ledger: Mapping[str, Any]) -> bool:
    leaves = ledger.get('leaves')
    if not isinstance(leaves, dict):
        return False
    return all(
        isinstance(leaves.get(leaf_id), dict)
        and leaves[leaf_id].get('status') == 'RIGOROUS'
        for leaf_id in ERROR_LEDGER_IDS
    )


def open_leaves(ledger: Mapping[str, Any]) -> list[str]:
    leaves = ledger.get('leaves')
    if not isinstance(leaves, dict):
        return list(ERROR_LEDGER_IDS)
    return [
        leaf_id
        for leaf_id in ERROR_LEDGER_IDS
        if not (
            isinstance(leaves.get(leaf_id), dict)
            and leaves[leaf_id].get('status') == 'RIGOROUS'
        )
    ]
