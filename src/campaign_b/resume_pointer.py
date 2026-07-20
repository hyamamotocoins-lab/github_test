"""Persist Campaign B resume id on the durable storage root."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..common import atomic_write_json, atomic_write_text, read_json, utc_now
from .schemas import screening_only_payload

RESUME_ENV_KEY = 'VALIDATED_RG_M7B_RESUME_ID'
POINTER_NAME = 'LATEST_CAMPAIGN_B_RESUME.json'
ENV_EXPORT_NAME = 'export_VALIDATED_RG_M7B_RESUME_ID.sh'
PLAIN_ID_NAME = 'VALIDATED_RG_M7B_RESUME_ID.txt'


def campaign_b_persist_dir(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b'


def resume_pointer_path(persistent_root: Path) -> Path:
    return campaign_b_persist_dir(persistent_root) / POINTER_NAME


def write_resume_pointer(
    persistent_root: Path,
    *,
    campaign_run_id: str,
    terminal_reason: str | None = None,
    campaign_root: Path | str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically save resume id for notebook / shell restarts."""
    root = campaign_b_persist_dir(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'schema_version': 1,
        'env_key': RESUME_ENV_KEY,
        RESUME_ENV_KEY: campaign_run_id,
        'campaign_run_id': campaign_run_id,
        'resume_campaign_run_id': campaign_run_id,
        'terminal_reason': terminal_reason,
        'campaign_root': str(campaign_root) if campaign_root else None,
        'updated_at': utc_now(),
        'export_hint': f'export {RESUME_ENV_KEY}={campaign_run_id}',
        **(extra or {}),
        **screening_only_payload(),
    }
    atomic_write_json(root / POINTER_NAME, payload)
    atomic_write_text(root / PLAIN_ID_NAME, f'{campaign_run_id}\n')
    atomic_write_text(
        root / ENV_EXPORT_NAME,
        (
            '#!/usr/bin/env bash\n'
            f'# Auto-written by Campaign B driver; source this file to resume.\n'
            f'export {RESUME_ENV_KEY}={campaign_run_id}\n'
        ),
    )
    return payload


def read_resume_id(persistent_root: Path) -> str | None:
    """Load resume id: JSON pointer first, then plain text file."""
    pointer = resume_pointer_path(persistent_root)
    if pointer.is_file():
        payload = read_json(pointer)
        if isinstance(payload, dict):
            for key in (RESUME_ENV_KEY, 'resume_campaign_run_id', 'campaign_run_id'):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    plain = campaign_b_persist_dir(persistent_root) / PLAIN_ID_NAME
    if plain.is_file():
        text = plain.read_text(encoding='utf-8').strip()
        if text:
            return text.splitlines()[0].strip()
    return None
