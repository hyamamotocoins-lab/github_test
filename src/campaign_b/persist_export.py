"""Pause / export Paperspace persist root to a downloadable archive.

Workflow (fail-closed)
----------------------
1. Plan: size selected trees, require free space, refuse if archive would sit
   *inside* the persist root, warn on a live GPU lane lease.
2. Archive: write a zip (default STORE) plus ``.sha256`` / ``.manifest.json``
   *outside* the persist root.
3. Verify: CRC + SHA-256.
4. Purge (optional): delete **persist root contents only** after verify.
   ``/storage/ssh`` is archived but **never purged** by this tool.

Tiers
-----
``tier_a`` (default): hard-to-restore state —

  - ``campaign_b/``
  - ``runs/M2-*``, ``M4-*``, ``M5-*``, ``M6-*``
  - ``runs/M3-*`` excluding ``checkpoints/``
  - top-level misc under persist
  - plus ``/storage/ssh`` (default extra root)

``full``: entire persist root (includes M3 checkpoints, ~55 GiB typical) + ssh.

Zip layout (restore with ``cd /storage && unzip archive.zip``)::

  validated_4d_su2_rg/...
  storage/ssh/...

Purge after ``tier_a`` also requires
``confirm_discard_m3_checkpoints='DISCARD_M3_CHECKPOINTS'``.
"""

from __future__ import annotations

import json
import os
import shutil
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Literal, Sequence

from ..common import atomic_write_json, sha256_file, utc_now
from .m3_reclaim import dir_size, fmt_bytes
from .schemas import screening_only_payload

PURGE_CONFIRM = 'PURGE_PERSIST_ROOT'
DISCARD_M3_CHECKPOINTS_CONFIRM = 'DISCARD_M3_CHECKPOINTS'
DEFAULT_EXPORT_DIR_NAME = 'exports'
DEFAULT_MARGIN_BYTES = 2 * 1024**3
DEFAULT_SSH_DIRNAME = 'ssh'

ExportTier = Literal['full', 'tier_a']
DEFAULT_EXPORT_TIER: ExportTier = 'tier_a'


class PersistExportError(RuntimeError):
    """Fail-closed export / purge error."""


def _sidecar_paths(archive: Path) -> tuple[Path, Path]:
    return (
        Path(str(archive) + '.manifest.json'),
        Path(str(archive) + '.sha256'),
    )


def _normalize_tier(tier: str) -> ExportTier:
    t = str(tier or DEFAULT_EXPORT_TIER).strip().lower().replace('-', '_')
    if t in {'a', 'hard', 'hard_to_restore', 'tier_a', 'tiera'}:
        return 'tier_a'
    if t in {'full', 'all', 'everything'}:
        return 'full'
    raise PersistExportError(
        f'unknown export tier {tier!r}; use tier_a or full'
    )


def _rel_included_in_tier(rel: Path, tier: ExportTier) -> bool:
    if tier == 'full':
        return True
    parts = rel.parts
    if not parts:
        return True
    top = parts[0]
    if top == 'campaign_b':
        return True
    if top != 'runs':
        return True
    if len(parts) < 2:
        return True
    run_id = parts[1]
    if run_id.startswith(('M2-', 'M4-', 'M5-', 'M6-')):
        return True
    if run_id.startswith('M3-'):
        return 'checkpoints' not in parts
    return 'checkpoints' not in parts


def default_ssh_root(persistent_root: Path) -> Path:
    """Paperspace SSH key dir sibling of persist: ``/storage/ssh``."""
    return _resolve(persistent_root).parent / DEFAULT_SSH_DIRNAME


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
    tier: ExportTier = DEFAULT_EXPORT_TIER
    extra_roots: tuple[str, ...] = ()
    breakdown: dict[str, Any] = field(default_factory=dict)
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
    tier: ExportTier = DEFAULT_EXPORT_TIER
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
            'tier': self.tier,
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
    return int(shutil.disk_usage(path).free)


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
    try:
        pid_i = int(lease.get('pid'))
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


def _walk_files(
    root: Path,
    *,
    include_rel: Callable[[Path], bool] | None = None,
) -> Iterable[tuple[Path, Path]]:
    """Yield (absolute_path, path_relative_to_root) for regular files."""
    root = Path(root)
    if not root.is_dir():
        return
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        pruned: list[str] = []
        for name in list(dirnames):
            p = Path(dirpath) / name
            if p.is_symlink():
                continue
            rel_dir = p.relative_to(root)
            # Prune checkpoint trees when the filter would reject their contents.
            if include_rel is not None and name == 'checkpoints':
                if not include_rel(rel_dir / '__file__'):
                    continue
            pruned.append(name)
        dirnames[:] = pruned
        for name in filenames:
            p = Path(dirpath) / name
            if p.is_symlink() or not p.is_file():
                continue
            rel = p.relative_to(root)
            if include_rel is not None and not include_rel(rel):
                continue
            yield p, rel


def _count_selected(
    root: Path,
    *,
    include_rel: Callable[[Path], bool] | None = None,
) -> tuple[int, int, int]:
    files = 0
    dirs = 0
    total = 0
    seen_dirs: set[Path] = set()
    for path, _rel in _walk_files(root, include_rel=include_rel):
        files += 1
        try:
            total += path.stat().st_size
        except OSError:
            continue
        parent = path.parent
        while True:
            if parent in seen_dirs:
                break
            seen_dirs.add(parent)
            dirs += 1
            if parent == root:
                break
            parent = parent.parent
    return files, dirs, total


def analyze_persist_tiers(persistent_root: Path | str) -> dict[str, Any]:
    """Size breakdown for full vs tier_a (persist only; no extras)."""
    root = _resolve(persistent_root)
    full_f, full_d, full_b = _count_selected(root)
    tier_f, tier_d, tier_b = _count_selected(
        root,
        include_rel=lambda rel: _rel_included_in_tier(rel, 'tier_a'),
    )
    return {
        'persistent_root': str(root),
        'full': {
            'bytes': full_b,
            'bytes_human': fmt_bytes(full_b),
            'files': full_f,
            'dirs': full_d,
        },
        'tier_a': {
            'bytes': tier_b,
            'bytes_human': fmt_bytes(tier_b),
            'files': tier_f,
            'dirs': tier_d,
        },
        'omitted_approx_bytes': max(0, full_b - tier_b),
        'omitted_approx_human': fmt_bytes(max(0, full_b - tier_b)),
        **screening_only_payload(),
    }


def default_archive_path(
    persistent_root: Path,
    *,
    export_dir: Path | None = None,
    stamp: str | None = None,
    tier: ExportTier = DEFAULT_EXPORT_TIER,
) -> Path:
    persistent_root = _resolve(persistent_root)
    if export_dir is None:
        export_dir = persistent_root.parent / DEFAULT_EXPORT_DIR_NAME
    else:
        export_dir = _resolve(export_dir)
    stamp = stamp or utc_now().replace(':', '').replace('-', '')[:15] + 'Z'
    name = f'{persistent_root.name}_{tier}_{stamp}.zip'
    return export_dir / name


def _normalize_extra_roots(
    persistent_root: Path,
    extra_roots: Sequence[Path | str] | None,
    *,
    include_ssh: bool,
) -> list[Path]:
    roots: list[Path] = []
    if extra_roots is not None:
        roots.extend(_resolve(p) for p in extra_roots)
    elif include_ssh:
        roots.append(default_ssh_root(persistent_root))
    out: list[Path] = []
    seen: set[Path] = set()
    for r in roots:
        if r in seen:
            continue
        seen.add(r)
        if _is_under(r, persistent_root):
            raise PersistExportError(
                f'extra root must be outside persistent_root: {r}'
            )
        out.append(r)
    return out


def _arcname_for_persist(persistent_root: Path, rel: Path) -> str:
    return f'{persistent_root.name}/{rel.as_posix()}'


def _arcname_for_extra(extra_root: Path, storage_parent: Path, rel: Path) -> str:
    """Map ``/storage/ssh/foo`` → ``storage/ssh/foo`` when under storage."""
    try:
        under = extra_root.relative_to(storage_parent)
        prefix = f'storage/{under.as_posix()}'
    except ValueError:
        prefix = f'extra/{extra_root.name}'
    if not rel.parts:
        return prefix
    return f'{prefix}/{rel.as_posix()}'


def plan_export(
    persistent_root: Path | str,
    *,
    archive_path: Path | str | None = None,
    export_dir: Path | str | None = None,
    margin_bytes: int = DEFAULT_MARGIN_BYTES,
    allow_live_gpu_lease: bool = False,
    tier: str = DEFAULT_EXPORT_TIER,
    include_ssh: bool = True,
    extra_roots: Sequence[Path | str] | None = None,
) -> ExportPlan:
    root = _resolve(persistent_root)
    tier_n = _normalize_tier(tier)
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
            tier=tier_n,
            warnings=(),
            ok=False,
            block_reason=f'persistent_root missing or not a directory: {root}',
        )

    try:
        extras = _normalize_extra_roots(
            root, extra_roots, include_ssh=include_ssh,
        )
    except PersistExportError as exc:
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
            tier=tier_n,
            warnings=(),
            ok=False,
            block_reason=str(exc),
        )

    archive = (
        _resolve(archive_path)
        if archive_path is not None
        else default_archive_path(
            root,
            export_dir=_resolve(export_dir) if export_dir is not None else None,
            tier=tier_n,
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
            tier=tier_n,
            extra_roots=tuple(str(p) for p in extras),
            warnings=(),
            ok=False,
            block_reason=(
                f'archive_path must be outside persistent_root '
                f'(got {archive} under {root})'
            ),
        )
    for extra in extras:
        if extra.exists() and _is_under(archive, extra):
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
                tier=tier_n,
                extra_roots=tuple(str(p) for p in extras),
                ok=False,
                block_reason=f'archive_path must be outside extra root {extra}',
            )

    include = (
        None if tier_n == 'full'
        else (lambda rel: _rel_included_in_tier(rel, tier_n))
    )
    file_count, dir_count, source_bytes = _count_selected(root, include_rel=include)
    breakdown: dict[str, Any] = {
        'persist': {
            'bytes': source_bytes,
            'bytes_human': fmt_bytes(source_bytes),
            'files': file_count,
            'tier': tier_n,
        },
        'extras': {},
    }
    extra_present: list[Path] = []
    for extra in extras:
        if not extra.is_dir():
            warnings.append(f'extra root missing (skipped): {extra}')
            breakdown['extras'][str(extra)] = {
                'present': False, 'bytes': 0, 'files': 0,
            }
            continue
        ef, ed, eb = _count_selected(extra)
        source_bytes += eb
        file_count += ef
        dir_count += ed
        extra_present.append(extra)
        breakdown['extras'][str(extra)] = {
            'present': True,
            'bytes': eb,
            'bytes_human': fmt_bytes(eb),
            'files': ef,
            'zip_prefix': _arcname_for_extra(extra, root.parent, Path()),
        }

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
                tier=tier_n,
                extra_roots=tuple(str(p) for p in extra_present),
                breakdown=breakdown,
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
        warnings.append('nothing to archive (empty persist + missing extras)')

    if tier_n == 'tier_a':
        warnings.append(
            'tier_a omits runs/M3-*/checkpoints/ (regenerable with GPU). '
            'Purge after this export discards those checkpoints forever.'
        )

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
        tier=tier_n,
        extra_roots=tuple(str(p) for p in extra_present),
        breakdown=breakdown,
        warnings=tuple(warnings),
        ok=ok,
        block_reason=block,
    )


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
            tier=plan.tier,
        )

    archive = plan.archive_path
    archive.parent.mkdir(parents=True, exist_ok=True)
    tmp = archive.with_suffix(archive.suffix + '.partial')
    if tmp.exists():
        tmp.unlink()

    root = plan.persistent_root
    storage_parent = root.parent
    include = (
        None if plan.tier == 'full'
        else (lambda rel: _rel_included_in_tier(rel, plan.tier))
    )
    member_count = 0
    log = progress_cb or (lambda _m: None)

    try:
        with zipfile.ZipFile(
            tmp, mode='w', compression=compression, allowZip64=True,
        ) as zf:
            # Restore instructions at zip root.
            restore_txt = (
                'Restore on Paperspace:\n'
                '  cd /storage && unzip -o THIS.zip\n'
                f'  → /storage/{root.name}/...\n'
                '  → /storage/ssh/...\n'
                f'Tier: {plan.tier}\n'
                'Purge of persist does NOT delete /storage/ssh.\n'
            )
            zf.writestr('RESTORE.txt', restore_txt)

            for path, rel in _walk_files(root, include_rel=include):
                zf.write(path, arcname=_arcname_for_persist(root, rel))
                member_count += 1
                if progress_every > 0 and member_count % progress_every == 0:
                    log(f'archived {member_count} files…')

            for extra_s in plan.extra_roots:
                extra = Path(extra_s)
                if not extra.is_dir():
                    continue
                for path, rel in _walk_files(extra):
                    zf.write(
                        path,
                        arcname=_arcname_for_extra(extra, storage_parent, rel),
                    )
                    member_count += 1
                    if progress_every > 0 and member_count % progress_every == 0:
                        log(f'archived {member_count} files…')
        tmp.replace(archive)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise

    # RESTORE.txt is an extra member beyond plan.file_count.
    digest = sha256_file(archive)
    archive_bytes = archive.stat().st_size
    plan.sha256_path.write_text(f'{digest}  {archive.name}\n', encoding='utf-8')
    with zipfile.ZipFile(archive, 'r') as zf:
        actual_members = len(zf.namelist())
    manifest = {
        'schema_version': 2,
        'created_at': created,
        'persistent_root': str(root),
        'tier': plan.tier,
        'extra_roots': list(plan.extra_roots),
        'breakdown': plan.breakdown,
        'archive_path': str(archive),
        'sha256': digest,
        'archive_bytes': archive_bytes,
        'source_bytes': plan.source_bytes,
        'member_count': actual_members,
        'planned_file_count': plan.file_count,
        'compression': compression_name,
        'zip_layout': {
            'persist_prefix': f'{root.name}/',
            'ssh_prefix': 'storage/ssh/',
            'restore': 'cd /storage && unzip -o ARCHIVE.zip',
        },
        'note': (
            'Pause export. tier_a omits M3 checkpoints. '
            '/storage/ssh is included but never auto-purged.'
        ),
        **screening_only_payload(),
    }
    atomic_write_json(plan.manifest_path, manifest)
    log(
        f'archive ready: {archive} ({fmt_bytes(archive_bytes)}, '
        f'{actual_members} members, tier={plan.tier}, sha256={digest[:12]}…)'
    )
    return ArchiveResult(
        archive_path=archive,
        manifest_path=plan.manifest_path,
        sha256_path=plan.sha256_path,
        sha256=digest,
        archive_bytes=archive_bytes,
        member_count=actual_members,
        source_bytes=plan.source_bytes,
        compression=compression_name,
        created_at=created,
        dry_run=False,
        tier=plan.tier,
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
    tier: str | None = None,
    confirm_discard_m3_checkpoints: str | None = None,
) -> dict[str, Any]:
    """Delete persist root contents after a verified archive. Never deletes ssh."""
    if confirm_purge != PURGE_CONFIRM:
        raise PersistExportError(
            f'purge refused: pass confirm_purge={PURGE_CONFIRM!r}'
        )
    tier_n = _normalize_tier(tier) if tier else None
    if tier_n == 'tier_a':
        if confirm_discard_m3_checkpoints != DISCARD_M3_CHECKPOINTS_CONFIRM:
            raise PersistExportError(
                'tier_a archive omits M3 checkpoints; purge requires '
                f'confirm_discard_m3_checkpoints={DISCARD_M3_CHECKPOINTS_CONFIRM!r}'
            )
    root = _resolve(persistent_root)
    archive = _resolve(archive_path)
    if not root.is_dir():
        raise PersistExportError(f'persistent_root missing: {root}')
    if not archive.is_file():
        raise PersistExportError(f'archive missing; refuse purge: {archive}')
    if _is_under(archive, root):
        raise PersistExportError(
            'refuse purge: archive is inside persistent_root'
        )

    verify_report: dict[str, Any] | None = None
    if require_verified:
        verify_report = verify_archive(archive, expected_sha256=expected_sha256)

    children = list(root.iterdir())
    report: dict[str, Any] = {
        'persistent_root': str(root),
        'archive_path': str(archive),
        'child_count': len(children),
        'dry_run': not execute,
        'deleted': [],
        'errors': [],
        'verify_report': verify_report,
        'ssh_purged': False,
        'note': '/storage/ssh is never deleted by purge_persistent_root',
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
            f'purge incomplete: {len(report["errors"])} errors'
        )
    marker_dir = root.parent / DEFAULT_EXPORT_DIR_NAME
    marker_dir.mkdir(parents=True, exist_ok=True)
    marker = marker_dir / f'{root.name}_PURGED.json'
    atomic_write_json(marker, {
        'status': 'PURGED_AFTER_EXPORT',
        'purged_at': utc_now(),
        'persistent_root': str(root),
        'archive_path': str(archive),
        'tier': tier_n,
        'sha256': (verify_report or {}).get('sha256') or expected_sha256,
        'ssh_purged': False,
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
    confirm_discard_m3_checkpoints: str | None = None,
    allow_live_gpu_lease: bool = False,
    margin_bytes: int = DEFAULT_MARGIN_BYTES,
    tier: str = DEFAULT_EXPORT_TIER,
    include_ssh: bool = True,
    extra_roots: Sequence[Path | str] | None = None,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    plan = plan_export(
        persistent_root,
        archive_path=archive_path,
        export_dir=export_dir,
        margin_bytes=margin_bytes,
        allow_live_gpu_lease=allow_live_gpu_lease,
        tier=tier,
        include_ssh=include_ssh,
        extra_roots=extra_roots,
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
                tier=plan.tier,
                confirm_discard_m3_checkpoints=confirm_discard_m3_checkpoints,
            )
            result.purged = True
    elif purge and not execute:
        children = (
            list(plan.persistent_root.iterdir())
            if plan.persistent_root.is_dir() else []
        )
        result.purge_report = {
            'dry_run': True,
            'would_delete': [p.name for p in children],
            'child_count': len(children),
            'ssh_purged': False,
            'note': (
                'Purge preview only; /storage/ssh is never deleted. '
                f'tier={plan.tier}'
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


__all__ = [
    'DEFAULT_EXPORT_TIER',
    'DISCARD_M3_CHECKPOINTS_CONFIRM',
    'PURGE_CONFIRM',
    'ArchiveResult',
    'ExportPlan',
    'PersistExportError',
    'analyze_persist_tiers',
    'create_archive',
    'default_archive_path',
    'default_ssh_root',
    'dir_size',
    'export_and_optional_purge',
    'fmt_bytes',
    'plan_export',
    'purge_persistent_root',
    'verify_archive',
]
