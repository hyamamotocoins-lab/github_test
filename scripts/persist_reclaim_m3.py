#!/usr/bin/env python3
"""Safe reclaim of Campaign B screening M3 checkpoint bloat on Paperspace.

Why unreferenced≈2 is not enough
---------------------------------
Almost every ``runs/M3-*`` dir is still pointed to by a selected package
(``GPU_M3.json`` / ``child_run_ids.json``). Deleting only unreferenced runs
frees ~1 GiB; the ~150 GiB bill is dominated by *referenced* screening M3
runs (~640 MiB each), mostly ``checkpoints/*/tensors/``.

Reclaim modes
-------------
``strip-checkpoints`` (Option A, default):
  For ``M3_COMPLETE`` runs that already have downstream ``M4_COMPLETE``
  (or M5/M6 progress), delete the M3 ``checkpoints/`` tree. Keeps
  reports/acceptance/config/artifacts/work_items. After this, that M3
  cannot be used as an M4 parent resume source (fail-closed by design).

``strip-tensors``:
  Same eligibility as strip-checkpoints, but only deletes
  ``checkpoints/*/tensors/`` (and similar bulky tensor dirs). Keeps
  checkpoint metadata / LATEST. Marker:
  ``STRIPPED_TENSORS_FOR_RECLAIM.json``.

``keep-latest-checkpoint`` (Option C):
  Keep only the newest ``COMMITTED`` ``ckpt_*``; delete older checkpoint
  dirs. Useful for incomplete / still-needed runs that may resume.

``delete-run`` (Option B):
  Delete an entire ``runs/M3-*`` directory. Only for unreferenced runs,
  or packages marked archived/abandoned, and only with
  ``--allow-delete-run``. Never deletes ``campaign_b/*/selected/*``
  package dirs. Optionally clears package M3 pointers when abandoned.

Safety
------
- Default is dry-run. Mutations require ``--execute``.
- Never touches ``m6_certified_catalog`` or runs whose M6 acceptance is
  ``CERTIFIED`` / ``ONE_STEP_CERTIFIED`` (skip lineage by default).
- Never deletes selected package directories.
- Fail closed: ambiguous / protected / incomplete-without-downstream
  targets are skipped with an explicit reason.

Core logic lives in ``src.campaign_b.m3_reclaim`` (shared with notebook 97
auto-strip). See also ``docs/campaign_b_m3_storage_reclaim.md``.

Paperspace (from repo root)::

  export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg

  # Dry-run (default): classify + reclaimable GiB
  python scripts/persist_reclaim_m3.py --mode strip-checkpoints

  # Execute safe strip
  python scripts/persist_reclaim_m3.py --mode strip-checkpoints --execute

  # Tensors-only (keeps ckpt metadata / LATEST)
  python scripts/persist_reclaim_m3.py --mode strip-tensors --execute
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

# Allow running as `python scripts/persist_reclaim_m3.py` without install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.campaign_b.m3_reclaim import (  # noqa: E402
    KEEP_RELATIVE_PREFIXES,
    M3Classification,
    classify_m3_runs,
    clear_abandoned_package_pointers,
    delete_run,
    fmt_bytes,
    keep_latest_for_run,
    strip_checkpoints_for_run,
    strip_tensors_for_run,
)


def print_report(rows: list[M3Classification], *, mode: str) -> dict[str, Any]:
    total = len(rows)
    complete = sum(1 for r in rows if r.complete)
    incomplete = total - complete
    referenced = sum(1 for r in rows if r.referenced)
    unreferenced = total - referenced
    strip_targets = [r for r in rows if r.reclaimable_strip_bytes > 0]
    strip_bytes = sum(r.reclaimable_strip_bytes for r in strip_targets)
    tensor_targets = [r for r in rows if r.reclaimable_tensors_bytes > 0]
    tensor_reclaim = sum(r.reclaimable_tensors_bytes for r in tensor_targets)
    keep_latest_bytes = sum(r.reclaimable_keep_latest_bytes for r in rows)
    total_size = sum(r.size_bytes for r in rows)
    ckpt_size = sum(r.checkpoints_bytes for r in rows)
    tensor_size = sum(r.tensors_bytes for r in rows)

    print('=== M3 classification ===')
    print(f'  total_m3:           {total}')
    print(f'  complete:           {complete}')
    print(f'  incomplete:         {incomplete}')
    print(f'  referenced:         {referenced}')
    print(f'  unreferenced:       {unreferenced}')
    print(f'  total_m3_size:      {fmt_bytes(total_size)}')
    print(f'  checkpoints_size:   {fmt_bytes(ckpt_size)}')
    print(f'  tensors_in_ckpts:   {fmt_bytes(tensor_size)}')
    print()
    print(f'=== mode={mode} reclaimable (dry-run estimate) ===')
    if mode == 'strip-checkpoints':
        print(f'  safe_strip_targets: {len(strip_targets)}')
        print(f'  reclaimable:        {fmt_bytes(strip_bytes)}')
        print(
            '  criterion: M3_COMPLETE + downstream M4_COMPLETE/M5/M6 '
            '+ not CERTIFIED lineage + checkpoints present'
        )
    elif mode == 'strip-tensors':
        print(f'  safe_tensor_targets: {len(tensor_targets)}')
        print(f'  reclaimable:         {fmt_bytes(tensor_reclaim)}')
        print(
            '  criterion: same as strip-checkpoints; delete only '
            'checkpoints/*/tensors (keep metadata / LATEST)'
        )
    elif mode == 'keep-latest-checkpoint':
        print(f'  reclaimable:        {fmt_bytes(keep_latest_bytes)}')
        print('  criterion: delete older/incomplete ckpt_* ; keep newest COMMITTED')
    else:
        unref_bytes = sum(r.size_bytes for r in rows if not r.referenced)
        archived_bytes = sum(
            r.size_bytes for r in rows if r.archived_only_refs and r.referenced
        )
        print(f'  unreferenced_runs:  {fmt_bytes(unref_bytes)} ({unreferenced} runs)')
        print(
            f'  archived_ref_runs:  {fmt_bytes(archived_bytes)} '
            f'(needs --allow-delete-run + archived package)'
        )

    print()
    print('=== top reclaim candidates (strip-checkpoints) ===')
    for row in sorted(strip_targets, key=lambda r: r.reclaimable_strip_bytes, reverse=True)[:20]:
        print(
            f'  {fmt_bytes(row.reclaimable_strip_bytes):>12}  {row.run_rel}  '
            f'refs={len(row.package_refs)}'
        )
    if mode == 'strip-tensors':
        print()
        print('=== top reclaim candidates (strip-tensors) ===')
        for row in sorted(
            tensor_targets, key=lambda r: r.reclaimable_tensors_bytes, reverse=True,
        )[:20]:
            print(
                f'  {fmt_bytes(row.reclaimable_tensors_bytes):>12}  {row.run_rel}  '
                f'refs={len(row.package_refs)}'
            )
    skipped = [r for r in rows if r.reclaimable_strip_bytes <= 0]
    reason_counts: dict[str, int] = {}
    for row in skipped:
        key = ','.join(row.skip_reasons) if row.skip_reasons else 'other'
        reason_counts[key] = reason_counts.get(key, 0) + 1
    print()
    print('=== strip skip reasons ===')
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        print(f'  {count:5d}  {reason}')

    return {
        'total_m3': total,
        'complete': complete,
        'incomplete': incomplete,
        'referenced': referenced,
        'unreferenced': unreferenced,
        'total_m3_size_bytes': total_size,
        'checkpoints_size_bytes': ckpt_size,
        'tensors_size_bytes': tensor_size,
        'strip_targets': len(strip_targets),
        'reclaimable_strip_bytes': strip_bytes,
        'tensor_strip_targets': len(tensor_targets),
        'reclaimable_tensors_bytes': tensor_reclaim,
        'reclaimable_keep_latest_bytes': keep_latest_bytes,
        'skip_reason_counts': reason_counts,
    }


def execute_mode(
    persistent_root: Path,
    rows: list[M3Classification],
    *,
    mode: str,
    execute: bool,
    allow_delete_run: bool,
    clear_pointers: bool,
    include_certified_lineage: bool,
) -> int:
    verb = 'EXECUTE' if execute else 'DRY-RUN'
    print()
    print(f'=== {verb} mode={mode} ===')
    freed = 0
    actions = 0
    if mode == 'strip-checkpoints':
        targets = [r for r in rows if r.reclaimable_strip_bytes > 0]
        for row in targets:
            nbytes, label = strip_checkpoints_for_run(
                persistent_root, row, execute=execute,
            )
            print(f'{label:28s} {fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
    elif mode == 'strip-tensors':
        targets = [r for r in rows if r.reclaimable_tensors_bytes > 0]
        for row in targets:
            nbytes, label = strip_tensors_for_run(
                persistent_root, row, execute=execute,
            )
            print(f'{label:28s} {fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
    elif mode == 'keep-latest-checkpoint':
        for row in rows:
            if row.certified_lineage and not include_certified_lineage:
                continue
            if row.reclaimable_keep_latest_bytes <= 0:
                continue
            nbytes, label = keep_latest_for_run(
                persistent_root, row, execute=execute,
            )
            if nbytes <= 0 and label.startswith('SKIP'):
                continue
            print(f'{label:28s} {fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
    elif mode == 'delete-run':
        if not allow_delete_run:
            print(
                'ERROR: delete-run requires --allow-delete-run '
                '(and never deletes selected package dirs).',
                file=sys.stderr,
            )
            return 2
        for row in rows:
            eligible = (not row.referenced) or row.archived_only_refs
            if not eligible:
                continue
            nbytes, label = delete_run(
                persistent_root,
                row,
                execute=execute,
                allow_referenced_archived=True,
            )
            if label.startswith('SKIP'):
                print(f'{label:28s} {fmt_bytes(0):>12}  {row.run_rel}')
                continue
            print(f'{label:28s} {fmt_bytes(nbytes):>12}  {row.run_rel}')
            freed += nbytes
            actions += 1
            if clear_pointers and (row.archived_only_refs or not row.referenced):
                n = clear_abandoned_package_pointers(
                    persistent_root, row.run_id, execute=execute,
                )
                if n:
                    print(f'  pointers_cleared={n} packages for {row.run_id}')
    else:
        print(f'ERROR: unknown mode {mode!r}', file=sys.stderr)
        return 2

    print()
    print(
        f'{"Would free" if not execute else "Freed"} ≈ {fmt_bytes(freed)} '
        f'across {actions} actions.'
    )
    if not execute:
        print('Re-run with --execute to apply. Default remains dry-run.')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg',
        ),
        help='Persist root (default: $VALIDATED_RG_PERSIST_ROOT or /storage/...)',
    )
    parser.add_argument(
        '--mode',
        choices=(
            'strip-checkpoints',
            'strip-tensors',
            'keep-latest-checkpoint',
            'delete-run',
        ),
        default='strip-checkpoints',
        help='Reclaim strategy (default: strip-checkpoints)',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Actually delete files (default is dry-run)',
    )
    parser.add_argument(
        '--allow-delete-run',
        action='store_true',
        help='Required for --mode delete-run',
    )
    parser.add_argument(
        '--clear-pointers',
        action='store_true',
        help='With delete-run, clear archived package M3 pointers',
    )
    parser.add_argument(
        '--include-certified-lineage',
        action='store_true',
        help='Allow reclaim on M3 runs whose package has CERTIFIED M6 (dangerous)',
    )
    parser.add_argument(
        '--json-summary',
        type=Path,
        default=None,
        help='Optional path to write machine-readable summary JSON',
    )
    args = parser.parse_args(argv)

    root = Path(args.persistent_root).expanduser()
    print(f'persist_root: {root}')
    print(f'exists: {root.is_dir()}')
    print(f'mode: {args.mode}')
    print(f'execute: {bool(args.execute)}')
    if not root.is_dir():
        print('ERROR: persist root missing; fail closed.', file=sys.stderr)
        return 2

    if (root / 'selected').is_dir() and not (root / 'runs').is_dir():
        print('ERROR: path looks like a package tree, not persist root.', file=sys.stderr)
        return 2

    rows = classify_m3_runs(
        root, include_certified_lineage=args.include_certified_lineage,
    )
    summary = print_report(rows, mode=args.mode)
    summary['mode'] = args.mode
    summary['execute'] = bool(args.execute)
    summary['persistent_root'] = str(root)

    code = execute_mode(
        root,
        rows,
        mode=args.mode,
        execute=bool(args.execute),
        allow_delete_run=bool(args.allow_delete_run),
        clear_pointers=bool(args.clear_pointers),
        include_certified_lineage=bool(args.include_certified_lineage),
    )

    if args.json_summary is not None:
        args.json_summary.parent.mkdir(parents=True, exist_ok=True)
        summary['runs'] = [asdict(r) for r in rows]
        args.json_summary.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + '\n',
            encoding='utf-8',
        )
        print(f'wrote_json_summary: {args.json_summary}')

    _ = KEEP_RELATIVE_PREFIXES
    return code


if __name__ == '__main__':
    raise SystemExit(main())
