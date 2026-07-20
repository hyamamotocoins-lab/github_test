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
    report_name = f'{milestone}_report.json'
    report_path = run_root / 'reports' / report_name
    _require_file(report_path, f'{milestone} report')

    checkpoint: Path | None = None
    audited_ckpt = audit.get('checkpoint_path')
    if isinstance(audited_ckpt, str) and audited_ckpt.strip():
        candidate = Path(audited_ckpt).expanduser().resolve()
        try:
            candidate.relative_to(run_root.resolve())
        except ValueError as exc:
            raise M5ParentChainError(
                f'{milestone} audit checkpoint escapes run root.'
            ) from exc
        if candidate.is_dir() and not candidate.is_symlink():
            checkpoint = candidate
    if checkpoint is None:
        # Prefer latest committed checkpoint (staged M4 often exceeds ckpt_000014).
        committed = sorted(
            path for path in (run_root / 'checkpoints').glob('ckpt_*')
            if path.is_dir() and (path / 'COMMITTED').is_file()
        )
        if committed:
            checkpoint = committed[-1]
        else:
            fallback = run_root / 'checkpoints' / 'ckpt_000014'
            if fallback.is_dir() and not fallback.is_symlink():
                checkpoint = fallback
    if checkpoint is None:
        raise M5ParentChainError(f'{milestone} checkpoint is missing under {run_root}')

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


def load_m1_m4_parent_chain_with_errors(
    project_root: Path,
    persistent_root: Path,
) -> tuple[dict[str, AcceptedParentRef], str | None]:
    """Load M1–M4 parents independently; return soft error notes for missing ones."""
    chain: dict[str, AcceptedParentRef] = {}
    errors: list[str] = []
    for milestone, relative in (
        ('M1', 'audit/m1_accepted_parent.json'),
        ('M2', 'audit/m2_accepted_parent.json'),
        ('M3', 'audit/m3_accepted_parent.json'),
        ('M4', 'audit/m4_accepted_parent.json'),
    ):
        try:
            chain[milestone] = load_accepted_parent(
                project_root, persistent_root,
                audit_relative=relative,
                milestone=milestone,
            )
        except M5ParentChainError as exc:
            errors.append(f'{milestone}: {exc}')
    if not chain and errors:
        raise M5ParentChainError(
            'No accepted M1–M4 parents could be loaded: ' + '; '.join(errors)
        )
    return chain, ('; '.join(errors) if errors else None)


def load_m1_m4_parent_chain(
    project_root: Path,
    persistent_root: Path,
) -> dict[str, AcceptedParentRef]:
    """Load M1–M4 accepted parents (partial chain allowed)."""
    chain, _errors = load_m1_m4_parent_chain_with_errors(
        project_root, persistent_root,
    )
    return chain
