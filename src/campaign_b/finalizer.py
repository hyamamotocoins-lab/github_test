"""Finalize Campaign B run: summary, hashes, restart hints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import atomic_write_json, sha256_file, utc_now
from .budget import BudgetManager
from .schemas import screening_only_payload


def finalize_campaign(
    *,
    campaign_root: Path,
    manifest: dict[str, Any],
    queue: dict[str, Any],
    ledger: dict[str, Any],
    budget: BudgetManager,
    terminal_reason: str | None,
) -> dict[str, Any]:
    root = Path(campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    summary = {
        'schema_version': 1,
        'campaign_run_id': manifest.get('campaign_run_id'),
        'terminal_reason': terminal_reason or ledger.get('terminal_reason'),
        'campaign_state': ledger.get('campaign_state'),
        'selected_count': len(ledger.get('selected') or []),
        'archived_count': len(ledger.get('archived_ids') or []),
        'pending_count': sum(
            1 for c in (queue.get('candidates') or []) if c.get('state') == 'PENDING'
        ),
        'budget': budget.snapshot(),
        'selected': ledger.get('selected') or [],
        'finalized_at': utc_now(),
        'restart_hint': {
            'resume_campaign_run_id': manifest.get('campaign_run_id'),
            'queue_path': str(root / 'queue.json'),
            # Fresh wall-clock window; unfinished candidates resume from queue.
            'inherit_deadline': False,
            'enforce_wall_clock': False,
            'note': (
                'Re-run with resume_campaign_run_id set to continue PENDING '
                'candidates. Six-hour cutoff is optional (enforce_wall_clock).'
            ),
        },
        **screening_only_payload(),
    }
    atomic_write_json(root / 'campaign_summary.json', summary)
    atomic_write_json(root / 'final_manifest.json', {
        **manifest,
        'summary_terminal_reason': summary['terminal_reason'],
        'finalized_at': summary['finalized_at'],
    })

    # Hash key artifacts
    hash_lines = []
    for name in (
        'campaign_manifest.json',
        'queue.json',
        'ledger.json',
        'campaign_summary.json',
        'final_manifest.json',
    ):
        path = root / name
        if path.is_file():
            hash_lines.append(f'{sha256_file(path)}  {name}')
    (root / 'campaign_hashes.sha256').write_text(
        '\n'.join(hash_lines) + ('\n' if hash_lines else ''),
        encoding='utf-8',
    )
    return summary
