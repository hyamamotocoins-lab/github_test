#!/usr/bin/env python3
"""Read-only disk usage report for VALIDATED_RG_PERSIST_ROOT (Paperspace).

Paperspace one-liner (from repo root):
  python scripts/persist_disk_report.py

Optional destructive cleanup (temp patterns only):
  python scripts/persist_disk_report.py --delete-tmp

Safe-to-delete patterns with --delete-tmp:
  - .write-probe-* files
  - *.tmp files / .*.tmp-* atomic-write leftovers
  - .tmp-attempt-* and .tmp-* incomplete checkpoint dirs (no COMMITTED)
  - __pycache__ directories

Never deletes: selected packages, CERTIFIED catalogs, COMMITTED checkpoints,
committed run reports, or seen_normalized_schemes.json.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path


TEMP_FILE_PREFIXES = ('.write-probe-',)
TEMP_DIR_PREFIXES = ('.tmp-attempt-', '.tmp-')


def _fmt_bytes(n: int) -> str:
    if n < 1024:
        return f'{n} B'
    for unit, scale in (('KiB', 1024), ('MiB', 1024**2), ('GiB', 1024**3), ('TiB', 1024**4)):
        if n < scale * 1024 or unit == 'TiB':
            return f'{n / scale:.2f} {unit}'
    return f'{n} B'


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for root, _dirs, files in os.walk(path, followlinks=False):
            for name in files:
                try:
                    total += (Path(root) / name).stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def _is_temp_file(path: Path) -> bool:
    name = path.name
    if name.startswith(TEMP_FILE_PREFIXES):
        return True
    if name.endswith('.tmp'):
        return True
    if name.startswith('.') and '.tmp-' in name:
        return True
    return False


def _is_temp_dir(path: Path) -> bool:
    name = path.name
    if name == '__pycache__':
        return True
    if not name.startswith(TEMP_DIR_PREFIXES):
        return False
    # Incomplete atomic checkpoint dirs: never touch if COMMITTED exists.
    if (path / 'COMMITTED').is_file():
        return False
    return True


def _scan_temp_candidates(root: Path) -> list[tuple[Path, str, int]]:
    """Return (path, kind, size_bytes) for clearly temporary artifacts."""
    found: list[tuple[Path, str, int]] = []
    if not root.is_dir():
        return found
    for dirpath, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        base = Path(dirpath)
        # Prune selected packages and certified catalog from temp deletion walk.
        pruned: list[str] = []
        for d in list(dirnames):
            child = base / d
            if d == 'selected' or d == 'm6_certified_catalog' or 'CERTIFIED' in d:
                pruned.append(d)
                continue
            if _is_temp_dir(child):
                size = _dir_size(child)
                found.append((child, 'dir', size))
                pruned.append(d)
        for d in pruned:
            if d in dirnames:
                dirnames.remove(d)
        for name in filenames:
            path = base / name
            if _is_temp_file(path):
                try:
                    size = path.stat().st_size
                except OSError:
                    size = 0
                found.append((path, 'file', size))
    return found


def _top_children(root: Path, depth: int = 1) -> list[tuple[int, Path]]:
    """Sizes of immediate children (depth=1) or one more level of runs/campaign_b."""
    if not root.is_dir():
        return []
    rows: list[tuple[int, Path]] = []
    try:
        children = sorted(root.iterdir(), key=lambda p: p.name)
    except OSError:
        return []
    for child in children:
        if child.is_symlink():
            continue
        if child.is_dir():
            rows.append((_dir_size(child), child))
        elif child.is_file():
            try:
                rows.append((child.stat().st_size, child))
            except OSError:
                pass
    rows.sort(key=lambda item: item[0], reverse=True)

    # Extra detail for large buckets.
    if depth >= 2:
        extra: list[tuple[int, Path]] = []
        for size, path in rows[:8]:
            if path.is_dir() and path.name in {'runs', 'campaign_b', 'cache', 'project'}:
                try:
                    for sub in path.iterdir():
                        if sub.is_symlink():
                            continue
                        if sub.is_dir():
                            extra.append((_dir_size(sub), sub))
                        elif sub.is_file():
                            try:
                                extra.append((sub.stat().st_size, sub))
                            except OSError:
                                pass
                except OSError:
                    pass
        extra.sort(key=lambda item: item[0], reverse=True)
        return extra[:40] if extra else rows
    return rows


def _bucket_summary(root: Path) -> dict[str, int]:
    buckets: dict[str, int] = defaultdict(int)
    runs = root / 'runs'
    if runs.is_dir():
        try:
            for child in runs.iterdir():
                if not child.is_dir():
                    continue
                size = _dir_size(child)
                prefix = child.name.split('-', 1)[0]
                if prefix in {'M0', 'M1', 'M2', 'M3', 'M4', 'M5', 'M6', 'M7'}:
                    buckets[f'runs/{prefix}-*'] += size
                else:
                    buckets['runs/other'] += size
        except OSError:
            pass
    camp = root / 'campaign_b'
    if camp.is_dir():
        buckets['campaign_b'] = _dir_size(camp)
    cache = root / 'cache'
    if cache.is_dir():
        buckets['cache'] = _dir_size(cache)
    return dict(buckets)


def report(root: Path, *, top_n: int, depth: int) -> int:
    usage = shutil.disk_usage(str(root if root.exists() else '/'))
    print(f'persist_root: {root}')
    print(f'exists: {root.is_dir()}')
    print(
        'filesystem: '
        f'total={_fmt_bytes(usage.total)}  '
        f'used={_fmt_bytes(usage.used)}  '
        f'free={_fmt_bytes(usage.free)}  '
        f'used_pct={100.0 * usage.used / usage.total:.1f}%'
    )
    if not root.is_dir():
        print('ERROR: persist root missing; nothing to scan.')
        return 2

    root_size = _dir_size(root)
    print(f'persist_tree_size: {_fmt_bytes(root_size)}')
    print()
    print('=== bucket summary ===')
    buckets = _bucket_summary(root)
    for key, size in sorted(buckets.items(), key=lambda kv: kv[1], reverse=True):
        print(f'  {_fmt_bytes(size):>12}  {key}')

    print()
    print(f'=== top {top_n} under persist root (depth={depth}) ===')
    rows = _top_children(root, depth=depth)
    for size, path in rows[:top_n]:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        print(f'  {_fmt_bytes(size):>12}  {rel}')

    temps = _scan_temp_candidates(root)
    temp_bytes = sum(s for _p, _k, s in temps)
    print()
    print(f'=== temp candidates (safe with --delete-tmp): {len(temps)} paths, {_fmt_bytes(temp_bytes)} ===')
    for path, kind, size in sorted(temps, key=lambda t: t[2], reverse=True)[:25]:
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        print(f'  {_fmt_bytes(size):>12}  [{kind}]  {rel}')
    if len(temps) > 25:
        print(f'  ... and {len(temps) - 25} more')

    print()
    print('Do NOT delete without confirm: campaign_b/*/selected/*, CERTIFIED catalogs, COMMITTED ckpts.')
    print('Stop notebooks 89/97 (and 95/96) until free space recovers — validate_persistent_root needs write room.')
    return 0


def delete_tmp(root: Path, *, dry_run: bool) -> int:
    temps = _scan_temp_candidates(root)
    if not temps:
        print('No temp candidates found.')
        return 0
    freed = 0
    removed = 0
    for path, kind, size in sorted(temps, key=lambda t: len(str(t[0])), reverse=True):
        try:
            rel = path.relative_to(root)
        except ValueError:
            rel = path
        action = 'WOULD_DELETE' if dry_run else 'DELETE'
        print(f'{action} [{kind}] {_fmt_bytes(size):>10}  {rel}')
        if dry_run:
            freed += size
            removed += 1
            continue
        try:
            if kind == 'dir':
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            freed += size
            removed += 1
        except OSError as exc:
            print(f'  FAILED: {exc}', file=sys.stderr)
    verb = 'Would free' if dry_run else 'Freed'
    print(f'{verb} ≈ {_fmt_bytes(freed)} across {removed} paths.')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
        help='Persist root (default: $VALIDATED_RG_PERSIST_ROOT or /storage/validated_4d_su2_rg)',
    )
    parser.add_argument('--top', type=int, default=30, help='How many large paths to print')
    parser.add_argument(
        '--depth',
        type=int,
        default=2,
        choices=(1, 2),
        help='1=immediate children only; 2=also expand runs/campaign_b/cache',
    )
    parser.add_argument(
        '--delete-tmp',
        action='store_true',
        help='Delete only clearly temporary artifacts (see module docstring)',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='With --delete-tmp, list targets without deleting',
    )
    args = parser.parse_args()
    root = Path(args.persistent_root).expanduser()
    code = report(root, top_n=max(1, args.top), depth=args.depth)
    if args.delete_tmp:
        print()
        print('=== --delete-tmp ===')
        delete_tmp(root, dry_run=args.dry_run)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
