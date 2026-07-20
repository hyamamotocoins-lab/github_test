"""Archive helpers (re-export + ledger helpers)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .audit import archive_candidate as write_archive


def archive_and_note(
    archive_root: Path,
    ledger_archived_ids: list[str],
    *,
    candidate: dict[str, Any],
    screening_result: dict[str, Any] | None,
    reason_code: str,
    extra: dict[str, Any] | None = None,
) -> Path:
    path = write_archive(
        archive_root,
        candidate=candidate,
        screening_result=screening_result,
        reason_code=reason_code,
        extra=extra,
    )
    cand_id = str(candidate.get('candidate_id'))
    if cand_id not in ledger_archived_ids:
        ledger_archived_ids.append(cand_id)
    return path
