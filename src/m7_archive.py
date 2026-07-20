"""Archive / advance ledger for Campaign C S0 candidate series.

Does not claim certificates. Archive means "do not continue this candidate
toward production M6 from the current exploratory S0 outcome."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, utc_now

REASON_S0_NO_SELECTION = 'S0_NO_SELECTION'
REASON_S0_REJECT = 'S0_REJECT_CANDIDATE'
REASON_RESOURCE_GATED = 'RESOURCE_GATED'
REASON_OPERATOR_SKIP = 'OPERATOR_SKIP'

ARCHIVE_REASONS = frozenset({
    REASON_S0_NO_SELECTION,
    REASON_S0_REJECT,
    REASON_RESOURCE_GATED,
    REASON_OPERATOR_SKIP,
})


class M7ArchiveError(RuntimeError):
    """Raised when archive/advance bookkeeping fails closed."""


def package_candidate_id(package_root: Path) -> str:
    root = Path(package_root)
    manifest = read_json(root / 'MANIFEST.json')
    if isinstance(manifest, dict) and manifest.get('candidate_id'):
        return str(manifest['candidate_id'])
    return root.name


def search_root_from_package(package_root: Path) -> Path:
    """auto_execute/CAND-... → search_root."""
    root = Path(package_root).resolve()
    if root.parent.name != 'auto_execute':
        raise M7ArchiveError(
            f'package_root must live under auto_execute/: {root}'
        )
    return root.parent.parent


def is_archived(package_root: Path) -> bool:
    return (Path(package_root) / 'ARCHIVE.json').is_file()


def read_archive(package_root: Path) -> dict[str, Any] | None:
    path = Path(package_root) / 'ARCHIVE.json'
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def read_advance(package_root: Path) -> dict[str, Any] | None:
    path = Path(package_root) / 'ADVANCE.json'
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def append_series_log(search_root: Path, event: dict[str, Any]) -> Path:
    reports = Path(search_root) / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    path = reports / 's0_series_log.jsonl'
    line = {
        'schema_version': 1,
        'logged_at': utc_now(),
        **event,
    }
    # Append-only; not atomic across writers, but durable enough for operator ledger.
    with path.open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(line, ensure_ascii=False, sort_keys=True) + '\n')
    return path


def write_archive(
    package_root: Path,
    *,
    reason: str,
    details: dict[str, Any] | None = None,
    sweep_root: str | Path | None = None,
    selection_reasons: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    if reason not in ARCHIVE_REASONS:
        raise M7ArchiveError(f'Unknown archive reason: {reason}')
    root = Path(package_root)
    path = root / 'ARCHIVE.json'
    if path.is_file() and not overwrite:
        existing = read_archive(root)
        if isinstance(existing, dict):
            return existing
    candidate_id = package_candidate_id(root)
    payload = {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'package_root': str(root.resolve()),
        'reason': reason,
        'details': details or {},
        'sweep_root': str(sweep_root) if sweep_root else None,
        'selection_reasons': list(selection_reasons or []),
        'certificate_usable': False,
        'production_m6_blocked': True,
        'archived_at': utc_now(),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }
    atomic_write_json(path, payload)
    try:
        search_root = search_root_from_package(root)
        append_series_log(search_root, {
            'event': 'ARCHIVE',
            'candidate_id': candidate_id,
            'reason': reason,
            'sweep_root': payload['sweep_root'],
        })
    except M7ArchiveError:
        pass
    return payload


def write_advance(
    package_root: Path,
    *,
    selected_rank: int,
    sweep_root: str | Path,
    selection_reasons: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(package_root)
    path = root / 'ADVANCE.json'
    if path.is_file() and not overwrite:
        existing = read_advance(root)
        if isinstance(existing, dict):
            return existing
    candidate_id = package_candidate_id(root)
    payload = {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'package_root': str(root.resolve()),
        'status': 'SELECTED',
        'selected_rank': int(selected_rank),
        'sweep_root': str(sweep_root),
        'selection_reasons': list(selection_reasons or []),
        'next_notebook': '78_m3_rigorous_rank_candidate.ipynb',
        'certificate_usable': False,
        'advanced_at': utc_now(),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        'note': 'PR-2 rigorous residual/gap not implemented; ADVANCE is exploratory freeze only.',
    }
    atomic_write_json(path, payload)
    try:
        search_root = search_root_from_package(root)
        append_series_log(search_root, {
            'event': 'ADVANCE',
            'candidate_id': candidate_id,
            'selected_rank': int(selected_rank),
            'sweep_root': str(sweep_root),
        })
    except M7ArchiveError:
        pass
    return payload


def archive_from_sweep(
    package_root: Path,
    sweep_root: Path,
    *,
    selection_status: str,
    selection_reasons: list[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Backfill ARCHIVE.json from an existing S0 sweep outcome."""
    status = str(selection_status)
    if status == 'NO_SELECTION':
        reason = REASON_S0_NO_SELECTION
    elif status == 'REJECT_CANDIDATE':
        reason = REASON_S0_REJECT
    else:
        raise M7ArchiveError(
            f'Cannot archive from selection_status={selection_status!r}; '
            'expected NO_SELECTION or REJECT_CANDIDATE.'
        )
    return write_archive(
        package_root,
        reason=reason,
        details={'selection_status': status},
        sweep_root=sweep_root,
        selection_reasons=selection_reasons,
        overwrite=overwrite,
    )
