"""Campaign C q<1 hunt: diagnose ranking and route to B when S3 is exhausted.

Screening q_cert_upper < 1 is NOT a certificate. This module only steers
operators toward staged (j2>=2) candidates that screening marks as q<1, or
toward Campaign B after expanded Campaign C searches still stall above 1.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import utc_now
from .cutoff_dims import resource_gate
from .m7_candidate_queue import list_queue_rows, search_root_for
from .m7_config import campaign_c_search_space
from .m7_status import M7_RUN_ID_CAMPAIGN_B, M7_RUN_ID_CAMPAIGN_C


class M7QLt1HuntError(RuntimeError):
    """Raised when q<1 hunt diagnosis cannot proceed."""


def mint_m7c_qlt1_run_id(*, tag: str = 'qlt1c') -> str:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    safe = ''.join(ch for ch in tag if ch.isalnum())[:12] or 'qlt1c'
    return f'M7-{stamp}-{safe}'


def _est_q(row: dict[str, Any]) -> float:
    try:
        return float(row.get('estimated_q') if 'estimated_q' in row else row.get('q_cert_upper') or 1e9)
    except (TypeError, ValueError):
        return 1e9


def _is_qlt1_hunt_run(search_run_id: str) -> bool:
    parts = str(search_run_id).split('-')
    return len(parts) >= 3 and parts[2].startswith('qlt1c')


def diagnose_q_lt1_hunt(
    *,
    persistent_root: Path,
    search_run_id: str,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
) -> dict[str, Any]:
    """Summarize whether the current Campaign C ranking has staged q<1."""
    search_root = search_root_for(persistent_root, search_run_id)
    if not search_root.is_dir():
        raise M7QLt1HuntError(f'Search root missing: {search_root}')
    rows = list_queue_rows(
        search_root,
        persistent_root=persistent_root,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    staged = [r for r in rows if r.get('staged_executable')]
    staged_live = [r for r in staged if not r.get('archived')]
    staged_lt1 = [r for r in staged if _est_q(r) < 1.0]
    staged_lt1_live = [r for r in staged_lt1 if not r.get('archived')]
    any_lt1 = [r for r in rows if _est_q(r) < 1.0]
    gated_lt1 = []
    for row in rows:
        if _est_q(row) >= 1.0:
            continue
        gate = resource_gate(
            int(row.get('j2_max') or 1),
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )
        if not gate.get('staged_executable') and not gate.get('executable'):
            gated_lt1.append(row)

    best_staged = min(staged, key=_est_q) if staged else None
    best_any = min(rows, key=_est_q) if rows else None
    best_q = _est_q(best_staged) if best_staged is not None else None

    if staged_lt1_live:
        recommendation = 'RUN_85_ON_STAGED_Q_LT1'
        note = (
            'Live staged candidates with screening q<1 exist. '
            'Set VALIDATED_RG_M7C_RUN_ID to this search and run notebook 85 '
            '(then 73 only if NEED_CANONICAL_M2 for j2>=2).'
        )
    elif staged_lt1:
        recommendation = 'STAGED_Q_LT1_ALL_ARCHIVED'
        note = (
            'Staged q<1 candidates exist but are archived. '
            'Inspect ARCHIVE reasons; do not mint identical Campaign C searches.'
        )
    elif gated_lt1:
        recommendation = 'GATED_Q_LT1_ONLY'
        note = (
            'Screening q<1 appears only on j2>max_staged (resource-gated). '
            'Do not silently raise max_staged. Prefer Campaign B (S2).'
        )
    elif best_staged is not None and best_q is not None and best_q >= 1.0:
        if _is_qlt1_hunt_run(search_run_id) or search_run_id != M7_RUN_ID_CAMPAIGN_C:
            recommendation = 'CAMPAIGN_C_S3_EXHAUSTED_GOTO_B'
            note = (
                f'Campaign C S3 screening remains q≥1 '
                f'(best staged q≈{best_q:.6g} in {search_run_id}). '
                'Do not mint another identical C search. '
                f'Next: Campaign B (S2 rank/residual) under {M7_RUN_ID_CAMPAIGN_B} '
                '(notebook 86 / 72). Screening q is still NOT_CERTIFIED.'
            )
        else:
            recommendation = 'NEW_EXPANDED_CAMPAIGN_C_SEARCH'
            note = (
                f'No staged screening q<1 in {search_run_id} '
                f'(best staged q≈{best_q:.6g}). '
                'Mint one expanded M7C qlt1c search; if still q≥1, move to Campaign B.'
            )
    else:
        recommendation = 'NO_STAGED_CANDIDATES'
        note = 'No staged candidates in ranking; check search outputs.'

    def _brief(row: dict[str, Any] | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            'candidate_id': row.get('candidate_id'),
            'j2_max': row.get('j2_max'),
            'estimated_q': _est_q(row),
            'archived': bool(row.get('archived')),
            'staged_executable': bool(row.get('staged_executable')),
        }

    space = campaign_c_search_space()
    return {
        'schema_version': 1,
        'search_run_id': search_run_id,
        'search_root': str(search_root),
        'counts': {
            'rows': len(rows),
            'staged': len(staged),
            'staged_live': len(staged_live),
            'staged_q_lt1': len(staged_lt1),
            'staged_q_lt1_live': len(staged_lt1_live),
            'any_q_lt1': len(any_lt1),
            'gated_q_lt1': len(gated_lt1),
        },
        'best_staged': _brief(best_staged),
        'best_any': _brief(best_any),
        'staged_q_lt1_live': [_brief(r) for r in sorted(staged_lt1_live, key=_est_q)[:10]],
        'recommendation': recommendation,
        'note': note,
        'expanded_search_space_layers': space.get('layers'),
        'certification_status': 'NOT_CERTIFIED',
        'generated_at': utc_now(),
    }


def next_q_lt1_actions(diagnosis: dict[str, Any]) -> dict[str, Any]:
    """Operator-facing next steps from diagnose_q_lt1_hunt."""
    rec = diagnosis.get('recommendation')
    new_id = mint_m7c_qlt1_run_id()
    if rec == 'RUN_85_ON_STAGED_Q_LT1':
        return {
            'action': 'notebook_85',
            'env': {
                'VALIDATED_RG_M7C_RUN_ID': diagnosis.get('search_run_id'),
            },
            'note': diagnosis.get('note'),
        }
    if rec == 'NEW_EXPANDED_CAMPAIGN_C_SEARCH':
        return {
            'action': 'notebook_86_new_search',
            'env': {
                'VALIDATED_RG_M7C_RUN_ID': new_id,
            },
            'suggested_run_id': new_id,
            'note': diagnosis.get('note'),
            'hints': [
                'Pull latest main (expanded Campaign C seeds/coupling).',
                f"os.environ['VALIDATED_RG_M7C_RUN_ID'] = '{new_id}'",
                'Run Campaign C search cell once.',
                'If still best staged q≥1 → Campaign B (do not mint another C).',
            ],
        }
    if rec == 'CAMPAIGN_C_S3_EXHAUSTED_GOTO_B':
        return {
            'action': 'notebook_86_campaign_b',
            'env': {
                'VALIDATED_RG_M7B_RUN_ID': M7_RUN_ID_CAMPAIGN_B,
            },
            'note': diagnosis.get('note'),
            'hints': [
                f"os.environ['VALIDATED_RG_M7B_RUN_ID'] = '{M7_RUN_ID_CAMPAIGN_B}'",
                'Run Campaign B cell in notebook 86 (S2 rank/residual).',
                'Do not start production M6 from screening alone.',
            ],
        }
    return {
        'action': 'inspect',
        'suggested_run_id': new_id,
        'note': diagnosis.get('note'),
        'recommendation': rec,
    }
