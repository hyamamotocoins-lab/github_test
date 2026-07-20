"""Resolve and verify the accepted M1–M4 parent chain for M5 obligation work."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .common import read_json, sha256_file


class M5ParentChainError(RuntimeError):
    """Raised when an accepted parent audit/run is missing or inconsistent."""


@dataclass(frozen=True, slots=True)
class AcceptedParentRef:
    milestone: str
    run_id: str
    audit_path: Path
    audit: dict[str, Any]
    run_root: Path
    checkpoint: Path
    report_path: Path


def _require_file(path: Path, label: str) -> None:
    if path.is_symlink() or not path.is_file():
        raise M5ParentChainError(f'{label} is missing or unsafe: {path}')


def _load_audit(project_root: Path, relative: str, milestone: str) -> dict[str, Any]:
    path = project_root / relative
    _require_file(path, f'{milestone} acceptance audit')
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise M5ParentChainError(f'{milestone} acceptance audit is malformed.')
    return payload


def load_accepted_parent(
    project_root: Path,
    persistent_root: Path,
    *,
    audit_relative: str,
    milestone: str,
    run_id_key: str = 'accepted_run_id',
) -> AcceptedParentRef:
    audit = _load_audit(project_root, audit_relative, milestone)
    run_id = audit.get(run_id_key)
    if not isinstance(run_id, str) or not run_id.startswith(f'{milestone}-'):
        raise M5ParentChainError(f'{milestone} accepted_run_id is invalid.')
    run_root = persistent_root.resolve() / 'runs' / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise M5ParentChainError(f'{milestone} run root is missing: {run_root}')
    checkpoint = run_root / 'checkpoints' / 'ckpt_000014'
    report_name = f'{milestone}_report.json'
    report_path = run_root / 'reports' / report_name
    _require_file(report_path, f'{milestone} report')
    if not checkpoint.is_dir() or checkpoint.is_symlink():
        raise M5ParentChainError(f'{milestone} checkpoint is missing: {checkpoint}')
    expected_report = audit.get(f'{milestone.lower()}_report_path')
    if isinstance(expected_report, str):
        if Path(expected_report).resolve() != report_path.resolve():
            raise M5ParentChainError(f'{milestone} report path drifted from audit.')
    expected_hash = audit.get(f'{milestone.lower()}_report_sha256')
    if isinstance(expected_hash, str) and sha256_file(report_path) != expected_hash:
        raise M5ParentChainError(f'{milestone} report hash mismatch.')
    return AcceptedParentRef(
        milestone=milestone,
        run_id=run_id,
        audit_path=project_root / audit_relative,
        audit=audit,
        run_root=run_root,
        checkpoint=checkpoint,
        report_path=report_path,
    )


def load_m1_m4_parent_chain(
    project_root: Path,
    persistent_root: Path,
) -> dict[str, AcceptedParentRef]:
    """Load M1–M4 accepted parents. Missing optional ancestors raise clearly."""
    return {
        'M1': load_accepted_parent(
            project_root, persistent_root,
            audit_relative='audit/m1_accepted_parent.json',
            milestone='M1',
        ),
        'M2': load_accepted_parent(
            project_root, persistent_root,
            audit_relative='audit/m2_accepted_parent.json',
            milestone='M2',
        ),
        'M3': load_accepted_parent(
            project_root, persistent_root,
            audit_relative='audit/m3_accepted_parent.json',
            milestone='M3',
        ),
        'M4': load_accepted_parent(
            project_root, persistent_root,
            audit_relative='audit/m4_accepted_parent.json',
            milestone='M4',
        ),
    }
