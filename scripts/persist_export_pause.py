#!/usr/bin/env python3
"""Pause Paperspace: zip persist root for download, optionally purge it.

Safety
------
- Default is dry-run (plan only).
- ``--execute`` writes the zip + ``.sha256`` + ``.manifest.json`` *outside*
  the persist root (default: ``/storage/exports/``).
- ``--purge`` deletes persist contents only after CRC + sha256 verify, and
  only with ``--i-understand-purge PURGE_PERSIST_ROOT``.
- Refuses a live GPU lane lease unless ``--allow-live-gpu-lease``.
- Refuses if free disk < source size + 2 GiB margin (override with
  ``--margin-gib``).

Paperspace (from repo root)::

  export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg

  # 1) Plan / free-space check
  python scripts/persist_export_pause.py

  # 2) Create downloadable zip (keeps persist)
  python scripts/persist_export_pause.py --execute

  # 3) After downloading the zip locally, free Paperspace disk:
  python scripts/persist_export_pause.py --execute --purge-only \\
      --archive-path /storage/exports/validated_4d_su2_rg_....zip \\
      --i-understand-purge PURGE_PERSIST_ROOT

  # Jupyter UI download: put the zip under /notebooks
  python scripts/persist_export_pause.py --execute \\
      --export-dir /notebooks/persist_exports
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.campaign_b.persist_export import (  # noqa: E402
    PURGE_CONFIRM,
    PersistExportError,
    export_and_optional_purge,
    plan_export,
    purge_persistent_root,
    verify_archive,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Zip VALIDATED_RG_PERSIST_ROOT for download; optional purge after verify.'
        ),
    )
    parser.add_argument(
        '--persistent-root',
        '--persist-root',
        dest='persistent_root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT',
            '/storage/validated_4d_su2_rg',
        ),
    )
    parser.add_argument(
        '--export-dir',
        default=None,
        help='Directory for the zip (default: <persist.parent>/exports)',
    )
    parser.add_argument(
        '--archive-path',
        default=None,
        help='Explicit zip path (must be outside persist root)',
    )
    parser.add_argument(
        '--execute',
        action='store_true',
        help='Write the archive (default: dry-run plan only)',
    )
    parser.add_argument(
        '--compress',
        action='store_true',
        help='Use DEFLATE (slower; tensors rarely shrink much)',
    )
    parser.add_argument(
        '--purge',
        action='store_true',
        help='After verified archive, delete persist root contents',
    )
    parser.add_argument(
        '--purge-only',
        action='store_true',
        help=(
            'Skip zip; verify --archive-path and purge persist '
            f'(requires --execute and --i-understand-purge {PURGE_CONFIRM})'
        ),
    )
    parser.add_argument(
        '--i-understand-purge',
        dest='confirm_purge',
        default=None,
        help=f'Must equal {PURGE_CONFIRM} when --purge / --purge-only is set',
    )
    parser.add_argument(
        '--allow-live-gpu-lease',
        action='store_true',
        help='Do not block when gpu_lane.json PID looks alive',
    )
    parser.add_argument(
        '--margin-gib',
        type=float,
        default=2.0,
        help='Extra free-space margin beyond source size (default 2 GiB)',
    )
    parser.add_argument(
        '--plan-only',
        action='store_true',
        help='Print plan JSON and exit (same as default without --execute)',
    )
    args = parser.parse_args(argv)

    margin = int(float(args.margin_gib) * (1024**3))
    persist = Path(args.persistent_root)

    if args.purge_only:
        if not args.archive_path:
            print('ERROR: --purge-only requires --archive-path', file=sys.stderr)
            return 2
        try:
            v = verify_archive(args.archive_path)
        except PersistExportError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1
        if not args.execute:
            print(
                'Dry-run purge-only. Re-run with --execute '
                f'--i-understand-purge {PURGE_CONFIRM}',
                file=sys.stderr,
            )
            preview = purge_persistent_root(
                persist,
                archive_path=args.archive_path,
                confirm_purge=args.confirm_purge or PURGE_CONFIRM,
                execute=False,
                require_verified=True,
            )
            print(json.dumps(
                {'verify': v, 'purge_preview': preview},
                indent=2, ensure_ascii=False, default=str,
            ))
            return 0
        try:
            report = purge_persistent_root(
                persist,
                archive_path=args.archive_path,
                confirm_purge=args.confirm_purge or '',
                execute=True,
                expected_sha256=v['sha256'],
            )
        except PersistExportError as exc:
            print(f'ERROR: {exc}', file=sys.stderr)
            return 1
        print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.plan_only or not args.execute:
        plan = plan_export(
            persist,
            archive_path=args.archive_path,
            export_dir=args.export_dir,
            margin_bytes=margin,
            allow_live_gpu_lease=args.allow_live_gpu_lease,
        )
        print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False, default=str))
        if not plan.ok:
            print(f'BLOCKED: {plan.block_reason}', file=sys.stderr)
            return 2
        if not args.execute:
            print(
                '\nDry-run only. Re-run with --execute to write the zip. '
                'Add --purge --i-understand-purge PURGE_PERSIST_ROOT to delete '
                'persist after verify.',
                file=sys.stderr,
            )
        return 0

    try:
        summary = export_and_optional_purge(
            persist,
            archive_path=args.archive_path,
            export_dir=args.export_dir,
            execute=True,
            compress=args.compress,
            purge=args.purge,
            confirm_purge=args.confirm_purge,
            allow_live_gpu_lease=args.allow_live_gpu_lease,
            margin_bytes=margin,
            progress_cb=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    except PersistExportError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if summary.get('status') == 'BLOCKED':
        return 2
    result = summary.get('result') or {}
    archive = result.get('archive_path')
    if archive:
        print(f'\nDownload: {archive}', file=sys.stderr)
        print(f"SHA256:   {result.get('sha256')}", file=sys.stderr)
        if result.get('purged'):
            print('Persist root purged after verify.', file=sys.stderr)
        else:
            print(
                'Persist root kept. After local download, re-run with '
                '--execute --purge --i-understand-purge PURGE_PERSIST_ROOT',
                file=sys.stderr,
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
