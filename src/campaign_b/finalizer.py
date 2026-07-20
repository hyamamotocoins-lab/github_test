"""Finalize Campaign B run: summary, hashes, restart hints."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import atomic_write_json, sha256_file, utc_now
from .budget import BudgetManager
from .resume_pointer import write_resume_pointer
from .schemas import screening_only_payload


def finalize_campaign(
    *,
    campaign_root: Path,
    manifest: dict[str, Any],
    queue: dict[str, Any],
    ledger: dict[str, Any],
    budget: BudgetManager,
    terminal_reason: str | None,
    persistent_root: Path | None = None,
) -> dict[str, Any]:
    root = Path(campaign_root)
    root.mkdir(parents=True, exist_ok=True)
    run_id = str(manifest.get('campaign_run_id') or '')
    persist = Path(
        persistent_root
        or manifest.get('persistent_root')
        or root.parent.parent
    )
    pointer = None
    if run_id:
        pointer = write_resume_pointer(
            persist,
            campaign_run_id=run_id,
            terminal_reason=terminal_reason or ledger.get('terminal_reason'),
            campaign_root=root,
        )
    summary = {
        'schema_version': 1,
        'campaign_run_id': run_id or manifest.get('campaign_run_id'),
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
        'resume_pointer': pointer,
        'restart_hint': {
            'resume_campaign_run_id': run_id or manifest.get('campaign_run_id'),
            'env_key': 'VALIDATED_RG_M7B_RESUME_ID',
            'persist_pointer': str(persist / 'campaign_b' / 'LATEST_CAMPAIGN_B_RESUME.json'),
            'persist_export_sh': str(
                persist / 'campaign_b' / 'export_VALIDATED_RG_M7B_RESUME_ID.sh'
            ),
            'queue_path': str(root / 'queue.json'),
            # Fresh wall-clock window; unfinished candidates resume from queue.
            'inherit_deadline': False,
            'enforce_wall_clock': False,
            'note': (
                'Resume id is stored under persistent_root/campaign_b/. '
                'Notebook 87 loads it automatically; or: '
                f'source {persist / "campaign_b" / "export_VALIDATED_RG_M7B_RESUME_ID.sh"}'
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
