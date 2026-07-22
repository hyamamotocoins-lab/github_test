"""Pause / export Paperspace persist root to a downloadable archive.

Workflow (fail-closed)
----------------------
1. Plan: size the tree, require free space, refuse if archive would sit
   *inside* the persist root, warn on a live GPU lane lease.
2. Archive: write a zip (default STORE — tensors are already dense) plus
   sidecar ``.sha256`` and ``.manifest.json`` *outside* the persist root.
3. Verify: open the zip, check CRCs / member count, re-hash the archive.
4. Purge (optional, separate flag): only after a successful verify, delete
   the persist root contents so Paperspace disk is freed. The archive and
   sidecars are never deleted.

Default is dry-run. Mutations require ``execute=True``. Purge additionally
requires ``confirm_purge='PURGE_PERSIST_ROOT'``.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from ..common import atomic_write_json, sha256_file, utc_now
from .m3_reclaim import dir_size, fmt_bytes
from .schemas import screening_only_payload

PURGE_CONFIRM = 'PURGE_PERSIST_ROOT'
DEFAULT_EXPORT_DIR_NAME = 'exports'
DEFAULT_MARGIN_BYTES = 2 * 1024**3  # 2 GiB headroom beyond source size


class PersistExportError(RuntimeError):
    """Fail-closed export / purge error."""


def _sidecar_paths(archive: Path) -> tuple[Path, Path]:
    return (
        Path(str(archive) + '.manifest.json'),
        Path(str(archive) + '.sha256'),
    )


@dataclass(frozen=True)
class ExportPlan:
    persistent_root: Path
    archive_path: Path
    manifest_path: Path
    sha256_path: Path
    source_bytes: int
    free_bytes: int
    required_bytes: int
    file_count: int
    dir_count: int
    gpu_lease: dict[str, Any] | None
    warnings: tuple[str, ...] = ()
    ok: bool = True
    block_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            'persistent_root': str(self.persistent_root),
            'archive_path': str(self.archive_path),
            'manifest_path': str(self.manifest_path),
            'sha256_path': str(self.sha256_path),
            'source_bytes_human': fmt_bytes(self.source_bytes),
            'free_bytes_human': fmt_bytes(self.free_bytes),
            'required_bytes_human': fmt_bytes(self.required_bytes),
            **screening_only_payload(),
        }


@dataclass
class ArchiveResult:
    archive_path: Path
    manifest_path: Path
    sha256_path: Path
    sha256: str
    archive_bytes: int
    member_count: int
    source_bytes: int
    compression: str
    created_at: str
    dry_run: bool
    verified: bool = False
    verify_report: dict[str, Any] = field(default_factory=dict)
    purged: bool = False
    purge_report: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            'archive_path': str(self.archive_path),
            'manifest_path': str(self.manifest_path),
            'sha256_path': str(self.sha256_path),
            'sha256': self.sha256,
            'archive_bytes': self.archive_bytes,
            'archive_bytes_human': fmt_bytes(self.archive_bytes),
            'member_count': self.member_count,
            'source_bytes': self.source_bytes,
            'source_bytes_human': fmt_bytes(self.source_bytes),
            'compression': self.compression,
            'created_at': self.created_at,
            'dry_run': self.dry_run,
            'verified': self.verified,
            'verify_report': self.verify_report,
            'purged': self.purged,
            'purge_report': self.purge_report,
            **screening_only_payload(),
        }


def _resolve(path: Path | str) -> Path:
    return Path(path).expanduser().resolve()


def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _disk_free_bytes(path: Path) -> int:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    return int(usage.free)


def _count_tree(root: Path) -> tuple[int, int, int]:
    """Return (file_count, dir_count, total_bytes). Does not follow symlinks."""
    files = 0
    dirs = 0
    total = 0
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirs += 1
        # Do not descend into symlink dirs (os.walk already skips with followlinks=False
        # for files; prune symlink directories explicitly).
        keep: list[str] = []
        for name in dirnames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            files += 1
            try:
                total += p.stat().st_size
            except OSError:
                continue
    return files, dirs, total


def _read_gpu_lease(persistent_root: Path) -> dict[str, Any] | None:
    lock = Path(persistent_root) / 'campaign_b' / '_locks' / 'gpu_lane.json'
    if not lock.is_file():
        return None
    try:
        payload = json.loads(lock.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return {'status': 'UNREADABLE', 'path': str(lock)}
    return payload if isinstance(payload, dict) else {'status': 'INVALID', 'path': str(lock)}


def _lease_looks_live(lease: dict[str, Any] | None) -> bool:
    if not isinstance(lease, dict):
        return False
    if lease.get('status') in {'UNREADABLE', 'INVALID'}:
        return True
    pid = lease.get('pid')
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return bool(lease.get('owner'))
    if pid_i <= 0:
        return False
    try:
        os.kill(pid_i, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def default_archive_path(
    persistent_root: Path,
    *,
    export_dir: Path | None = None,
    stamp: str | None = None,
) -> Path:
    persistent_root = _resolve(persistent_root)
    if export_dir is None:
        export_dir = persistent_root.parent / DEFAULT_EXPORT_DIR_NAME
    else:
        export_dir = _resolve(export_dir)
    stamp = stamp or utc_now().replace(':', '').replace('-', '')[:15] + 'Z'
    name = f'{persistent_root.name}_{stamp}.zip'
    return export_dir / name


def plan_export(
    persistent_root: Path | str,
    *,
    archive_path: Path | str | None = None,
    export_dir: Path | str | None = None,
    margin_bytes: int = DEFAULT_MARGIN_BYTES,
    allow_live_gpu_lease: bool = False,
) -> ExportPlan:
    root = _resolve(persistent_root)
    warnings: list[str] = []
    if not root.is_dir():
        return ExportPlan(
            persistent_root=root,
            archive_path=Path(''),
            manifest_path=Path(''),
            sha256_path=Path(''),
            source_bytes=0,
            free_bytes=0,
            required_bytes=0,
            file_count=0,
            dir_count=0,
            gpu_lease=None,
            warnings=(),
            ok=False,
            block_reason=f'persistent_root missing or not a directory: {root}',
        )

    archive = (
        _resolve(archive_path)
        if archive_path is not None
        else default_archive_path(
            root,
            export_dir=_resolve(export_dir) if export_dir is not None else None,
        )
    )
    manifest_path, sha_path = _sidecar_paths(archive)
    if _is_under(archive, root):
        return ExportPlan(
            persistent_root=root,
            archive_path=archive,
            manifest_path=manifest_path,
            sha256_path=sha_path,
            source_bytes=0,
            free_bytes=0,
            required_bytes=0,
            file_count=0,
            dir_count=0,
            gpu_lease=_read_gpu_lease(root),
            warnings=(),
            ok=False,
            block_reason=(
                f'archive_path must be outside persistent_root '
                f'(got {archive} under {root})'
            ),
        )

    file_count, dir_count, source_bytes = _count_tree(root)
    free_bytes = _disk_free_bytes(archive.parent)
    required = int(source_bytes) + int(margin_bytes)
    lease = _read_gpu_lease(root)
    if _lease_looks_live(lease):
        msg = (
            'GPU lane lease looks live; stop notebook 96/97 (or release the lease) '
            'before export/purge to avoid racing a running session.'
        )
        if allow_live_gpu_lease:
            warnings.append(msg)
        else:
            return ExportPlan(
                persistent_root=root,
                archive_path=archive,
                manifest_path=manifest_path,
                sha256_path=sha_path,
                source_bytes=source_bytes,
                free_bytes=free_bytes,
                required_bytes=required,
                file_count=file_count,
                dir_count=dir_count,
                gpu_lease=lease,
                warnings=tuple(warnings),
                ok=False,
                block_reason=msg,
            )

    ok = True
    block: str | None = None
    if free_bytes < required:
        ok = False
        block = (
            f'insufficient free space: need ≈{fmt_bytes(required)} '
            f'(source {fmt_bytes(source_bytes)} + margin {fmt_bytes(margin_bytes)}), '
            f'have {fmt_bytes(free_bytes)} on {archive.parent}'
        )
    elif source_bytes <= 0 and file_count == 0:
        warnings.append('persistent_root is empty; archive will still be created')

    return ExportPlan(
        persistent_root=root,
        archive_path=archive,
        manifest_path=manifest_path,
        sha256_path=sha_path,
        source_bytes=source_bytes,
        free_bytes=free_bytes,
        required_bytes=required,
        file_count=file_count,
        dir_count=dir_count,
        gpu_lease=lease,
        warnings=tuple(warnings),
        ok=ok,
        block_reason=block,
    )


def _iter_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        keep: list[str] = []
        for name in dirnames:
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            yield p


def create_archive(
    plan: ExportPlan,
    *,
    execute: bool = False,
    compress: bool = False,
    progress_every: int = 500,
    progress_cb: Callable[[str], None] | None = None,
) -> ArchiveResult:
    if not plan.ok:
        raise PersistExportError(plan.block_reason or 'export plan is blocked')

    compression = zipfile.ZIP_DEFLATED if compress else zipfile.ZIP_STORED
    compression_name = 'DEFLATED' if compress else 'STORED'
    created = utc_now()
    if not execute:
        return ArchiveResult(
            archive_path=plan.archive_path,
            manifest_path=plan.manifest_path,
            sha256_path=plan.sha256_path,
            sha256='',
            archive_bytes=0,
            member_count=plan.file_count,
            source_bytes=plan.source_bytes,
            compression=compression_name,
            created_at=created,
            dry_run=True,
        )

    archive = plan.archive_path
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(archive.suffix + '.partial')
    if tmp.exists():
        tmp.unlink()

    root = plan.persistent_root
    member_count = 0
    log = progress_cb or (lambda _m: None)

    try:
        with zipfile.ZipFile(
            tmp,
            mode='w',
            compression=compression,
            allowZip64=True,
        ) as zf:
            for path in _iter_files(root):
                rel = path.relative_to(root).as_posix()
                zf.write(path, arcname=rel)
                member_count += 1
                if progress_every > 0 and member_count % progress_every == 0:
                    log(f'archived {member_count} files…')
        tmp.replace(archive)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    digest = sha256_file(archive)
    archive_bytes = archive.stat().st_size
    plan.sha256_path.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    manifest = {
        'schema_version': 1,
        'created_at': created,
        'persistent_root': str(root),
        'archive_path': str(archive),
        'sha256': digest,
        'archive_bytes': archive_bytes,
        'source_bytes': plan.source_bytes,
        'member_count': member_count,
        'planned_file_count': plan.file_count,
        'compression': compression_name,
        'note': (
            'Pause export of Paperspace persist root. '
            'Restore by unzipping into an empty VALIDATED_RG_PERSIST_ROOT.'
        ),
        **screening_only_payload(),
    }
    atomic_write_json(plan.manifest_path, manifest)
    log(
        f'archive ready: {archive} ({fmt_bytes(archive_bytes)}, '
        f'{member_count} members, sha256={digest[:12]}…)'
    )
    return ArchiveResult(
        archive_path=archive,
        manifest_path=plan.manifest_path,
        sha256_path=plan.sha256_path,
        sha256=digest,
        archive_bytes=archive_bytes,
        member_count=member_count,
        source_bytes=plan.source_bytes,
        compression=compression_name,
        created_at=created,
        dry_run=False,
    )


def verify_archive(
    archive_path: Path | str,
    *,
    expected_sha256: str | None = None,
    expected_member_count: int | None = None,
) -> dict[str, Any]:
    archive = _resolve(archive_path)
    if not archive.is_file():
        raise PersistExportError(f'archive missing: {archive}')
    digest = sha256_file(archive)
    if expected_sha256 and digest != expected_sha256:
        raise PersistExportError(
            f'sha256 mismatch: expected {expected_sha256}, got {digest}'
        )
    bad: list[str] = []
    with zipfile.ZipFile(archive, mode='r') as zf:
        bad = list(zf.testzip() or [])
        names = zf.namelist()
    if bad:
        raise PersistExportError(f'zip CRC failure on members: {bad[:10]}')
    if expected_member_count is not None and len(names) != int(expected_member_count):
        raise PersistExportError(
            f'member count mismatch: expected {expected_member_count}, got {len(names)}'
        )
    return {
        'archive_path': str(archive),
        'sha256': digest,
        'member_count': len(names),
        'archive_bytes': archive.stat().st_size,
        'archive_bytes_human': fmt_bytes(archive.stat().st_size),
        'ok': True,
        **screening_only_payload(),
    }


def purge_persistent_root(
    persistent_root: Path | str,
    *,
    archive_path: Path | str,
    confirm_purge: str,
    execute: bool = False,
    require_verified: bool = True,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Delete persist root contents after a verified archive exists outside it."""
    if confirm_purge != PURGE_CONFIRM:
        raise PersistExportError(
            f'purge refused: pass confirm_purge={PURGE_CONFIRM!r}'
        )
    root = _resolve(persistent_root)
    archive = _resolve(archive_path)
    if not root.is_dir():
        raise PersistExportError(f'persistent_root missing: {root}')
    if not archive.is_file():
        raise PersistExportError(f'archive missing; refuse purge: {archive}')
    if _is_under(archive, root):
        raise PersistExportError(
            'refuse purge: archive is inside persistent_root '
            '(would destroy the downloadable zip)'
        )

    verify_report: dict[str, Any] | None = None
    if require_verified:
        verify_report = verify_archive(archive, expected_sha256=expected_sha256)

    children = list(root.iterdir())
    report = {
        'persistent_root': str(root),
        'archive_path': str(archive),
        'child_count': len(children),
        'dry_run': not execute,
        'deleted': [],
        'errors': [],
        'verify_report': verify_report,
        **screening_only_payload(),
    }
    if not execute:
        report['deleted'] = [p.name for p in children]
        return report

    for child in children:
        try:
            if child.is_symlink() or child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
            report['deleted'].append(child.name)
        except OSError as exc:
            report['errors'].append({'path': str(child), 'error': str(exc)})
    if report['errors']:
        raise PersistExportError(
            f'purge incomplete: {len(report["errors"])} errors; '
            f'see report.errors'
        )
    # Leave a pointer outside the emptied root so operators know where the zip is.
    marker_dir = root.parent / DEFAULT_EXPORT_DIR_NAME
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f'{root.name}_PURGED.json'
    atomic_write_json(marker, {
        'status': 'PURGED_AFTER_EXPORT',
        'purged_at': utc_now(),
        'persistent_root': str(root),
        'archive_path': str(archive),
        'sha256': (verify_report or {}).get('sha256') or expected_sha256,
        **screening_only_payload(),
    })
    report['purge_marker'] = str(marker)
    report['dry_run'] = False
    return report


def export_and_optional_purge(
    persistent_root: Path | str,
    *,
    archive_path: Path | str | None = None,
    export_dir: Path | str | None = None,
    execute: bool = False,
    compress: bool = False,
    purge: bool = False,
    confirm_purge: str | None = None,
    allow_live_gpu_lease: bool = False,
    margin_bytes: int = DEFAULT_MARGIN_BYTES,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """One-shot pause helper: plan → archive → verify → optional purge."""
    plan = plan_export(
        persistent_root,
        archive_path=archive_path,
        export_dir=export_dir,
        margin_bytes=margin_bytes,
        allow_live_gpu_lease=allow_live_gpu_lease,
    )
    out: dict[str, Any] = {
        'plan': plan.to_dict(),
        'started_at': utc_now(),
        **screening_only_payload(),
    }
    if not plan.ok:
        out['status'] = 'BLOCKED'
        out['finished_at'] = utc_now()
        return out

    result = create_archive(
        plan,
        execute=execute,
        compress=compress,
        progress_cb=progress_cb,
    )
    if execute and not result.dry_run:
        verify_report = verify_archive(
            result.archive_path,
            expected_sha256=result.sha256,
            expected_member_count=result.member_count,
        )
        result.verified = True
        result.verify_report = verify_report
        if purge:
            if confirm_purge != PURGE_CONFIRM:
                raise PersistExportError(
                    f'purge requested but confirm_purge must be {PURGE_CONFIRM!r}'
                )
            result.purge_report = purge_persistent_root(
                plan.persistent_root,
                archive_path=result.archive_path,
                confirm_purge=confirm_purge,
                execute=True,
                require_verified=True,
                expected_sha256=result.sha256,
            )
            result.purged = True
    elif purge and not execute:
        children = (
            list(plan.persistent_root.iterdir())
            if plan.persistent_root.is_dir()
            else []
        )
        result.purge_report = {
            'dry_run': True,
            'would_delete': [p.name for p in children],
            'child_count': len(children),
            'note': (
                'Purge preview only; archive not written yet. '
                f'Real purge requires --execute --purge '
                f'--i-understand-purge {PURGE_CONFIRM}'
            ),
            **screening_only_payload(),
        }

    out['status'] = (
        'DRY_RUN' if not execute
        else ('PURGED' if result.purged else 'ARCHIVED')
    )
    out['result'] = result.to_dict()
    out['finished_at'] = utc_now()
    return out


def write_sha256_text(path: Path, digest: str, archive_name: str) -> None:
    path.write_text(f'{digest}  {archive_name}\n', encoding='utf-8')


# Re-export helpers used by tests / CLI.
__all__ = [
    'PURGE_CONFIRM',
    'ArchiveResult',
    'ExportPlan',
    'PersistExportError',
    'create_archive',
    'default_archive_path',
    'dir_size',
    'export_and_optional_purge',
    'fmt_bytes',
    'plan_export',
    'purge_persistent_root',
    'verify_archive',
]
