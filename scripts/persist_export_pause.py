#!/usr/bin/env python3
"""Pause Paperspace: zip persist (+ /storage/ssh) for download, optional purge.

Default tier is ``tier_a`` (~2 GiB typical): campaign_b + M2/M4/M5/M6 +
M3 without checkpoints + ``/storage/ssh``. Use ``--tier full`` for the
entire persist tree (~57 GiB).

Zip layout (restore: ``cd /storage && unzip -o ARCHIVE.zip``)::

  validated_4d_su2_rg/...
  storage/ssh/...

Purge deletes persist only — **never** ``/storage/ssh``.
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
    DEFAULT_EXPORT_TIER,
    DISCARD_M3_CHECKPOINTS_CONFIRM,
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
            'Zip persist (+ /storage/ssh) for download; optional purge after verify.'
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
        '--tier',
        default=DEFAULT_EXPORT_TIER,
        choices=['tier_a', 'full'],
        help='tier_a=hard-to-restore (~2GiB); full=entire persist',
    )
    parser.add_argument(
        '--include-ssh',
        dest='include_ssh',
        action='store_true',
        default=True,
        help='Include /storage/ssh (default: on)',
    )
    parser.add_argument(
        '--no-include-ssh',
        dest='include_ssh',
        action='store_false',
        help='Omit /storage/ssh from the archive',
    )
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--compress', action='store_true')
    parser.add_argument(
        '--purge',
        action='store_true',
        help='After verified archive, delete persist root (not ssh)',
    )
    parser.add_argument(
        '--purge-only',
        action='store_true',
        help='Skip zip; verify --archive-path and purge persist',
    )
    parser.add_argument(
        '--i-understand-purge',
        dest='confirm_purge',
        default=None,
        help=f'Must equal {PURGE_CONFIRM}',
    )
    parser.add_argument(
        '--i-understand-discard-m3-checkpoints',
        dest='confirm_discard_m3',
        default=None,
        help=(
            f'Required for tier_a purge; must equal '
            f'{DISCARD_M3_CHECKPOINTS_CONFIRM}'
        ),
    )
    parser.add_argument('--allow-live-gpu-lease', action='store_true')
    parser.add_argument('--margin-gib', type=float, default=2.0)
    parser.add_argument('--plan-only', action='store_true')
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
            preview = purge_persistent_root(
                persist,
                archive_path=args.archive_path,
                confirm_purge=args.confirm_purge or PURGE_CONFIRM,
                execute=False,
                require_verified=True,
                tier=args.tier,
                confirm_discard_m3_checkpoints=(
                    args.confirm_discard_m3 or DISCARD_M3_CHECKPOINTS_CONFIRM
                ),
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
                tier=args.tier,
                confirm_discard_m3_checkpoints=args.confirm_discard_m3,
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
            tier=args.tier,
            include_ssh=args.include_ssh,
        )
        print(json.dumps(plan.to_dict(), indent=2, ensure_ascii=False, default=str))
        if not plan.ok:
            print(f'BLOCKED: {plan.block_reason}', file=sys.stderr)
            return 2
        if not args.execute:
            print(
                f'\nDry-run (tier={args.tier}, include_ssh={args.include_ssh}). '
                'Re-run with --execute to write the zip.',
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
            confirm_discard_m3_checkpoints=args.confirm_discard_m3,
            allow_live_gpu_lease=args.allow_live_gpu_lease,
            margin_bytes=margin,
            tier=args.tier,
            include_ssh=args.include_ssh,
            progress_cb=lambda msg: print(msg, file=sys.stderr, flush=True),
        )
    except PersistExportError as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1

    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if summary.get('status') == 'BLOCKED':
        return 2
    result = summary.get('result') or {}
    if result.get('archive_path'):
        print(f"\nDownload: {result['archive_path']}", file=sys.stderr)
        print(f"SHA256:   {result.get('sha256')}", file=sys.stderr)
        print(f"tier:     {result.get('tier')}", file=sys.stderr)
        print('Restore:  cd /storage && unzip -o ARCHIVE.zip', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
