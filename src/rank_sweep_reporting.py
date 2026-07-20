"""Reporting helpers for Campaign C exploratory rank sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_text, read_json


def rank_sweep_markdown(summary: dict[str, Any]) -> str:
    rows = summary.get('rank_rows') or []
    lines = [
        '# Campaign C exploratory rank / gap / budget sweep',
        '',
        f"- sweep_id: `{summary.get('sweep_id')}`",
        f"- candidate: `{summary.get('candidate_id')}`",
        f"- parent M2: `{summary.get('parent_m2_run_id')}`",
        f"- status: `{summary.get('status')}`",
        f"- selection: `{summary.get('selection_status')}` "
        f"(selected_rank={summary.get('selected_rank')})",
        '',
        'All values are `HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND`.',
        'Do not copy these numbers into M5 certificate ledgers.',
        '',
        '| rank | eff | rel residual | gap | terminus | q_opt | q_prov | gates |',
        '|---:|---:|---:|---:|:---:|---:|---:|:---|',
    ]
    for row in rows:
        budget = row.get('budget') or {}
        lines.append(
            '| {rank} | {eff} | {rel:.6e} | {gap} | {term} | {qopt} | {qprov} | {gates} |'.format(
                rank=row.get('rank'),
                eff=row.get('effective_projected_rank'),
                rel=float(row.get('relative_residual_frobenius') or 0.0),
                gap=(
                    'n/a' if row.get('approximate_gap') is None
                    else f"{float(row['approximate_gap']):.6e}"
                ),
                term='yes' if row.get('is_cluster_terminus') else 'no',
                qopt=(
                    'inf' if budget.get('q_optimistic') == float('inf')
                    else f"{float(budget.get('q_optimistic') or 0.0):.6f}"
                ),
                qprov=(
                    'inf' if budget.get('q_provisional') == float('inf')
                    else f"{float(budget.get('q_provisional') or 0.0):.6f}"
                ),
                gates=(
                    ('O' if budget.get('passes_optimistic_gate') else '-')
                    + ('P' if budget.get('passes_provisional_gate') else '-')
                ),
            )
        )
    lines.extend([
        '',
        '## Selection reasons',
        '',
    ])
    for reason in summary.get('selection_reasons') or []:
        lines.append(f'- {reason}')
    lines.extend([
        '',
        '## Nonclaims',
        '',
        '- Not a deterministic RSVD residual certificate.',
        '- Not a verified spectral gap.',
        '- Not a one-step q_cert bound.',
        '- Not a continuum / mass-gap claim.',
        '',
    ])
    return '\n'.join(lines)


def write_rank_sweep_report(sweep_root: Path) -> dict[str, str]:
    summary = read_json(sweep_root / 'rank_sweep_summary.json')
    if not isinstance(summary, dict):
        raise ValueError('rank_sweep_summary.json missing or malformed.')
    markdown_path = sweep_root / 'rank_sweep_summary.md'
    atomic_write_text(markdown_path, rank_sweep_markdown(summary))
    return {
        'summary_json': str(sweep_root / 'rank_sweep_summary.json'),
        'summary_markdown': str(markdown_path),
        'selection_json': str(sweep_root / 'rank_selection.json'),
    }
