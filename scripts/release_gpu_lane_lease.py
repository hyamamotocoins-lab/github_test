#!/usr/bin/env python3
"""Reclaim or force-release the Campaign B GPU lane lease.

Paperspace one-liner (from repo root)::

  python scripts/release_gpu_lane_lease.py

Default: reclaim if same-host dead PID, or foreign/same-host stale by the
same rules as ``acquire_gpu_lock`` (foreign host default 15 min).

Use ``--force`` only when you are sure no GPU consumer holds the lease::

  python scripts/release_gpu_lane_lease.py --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Reclaim GPU lane lease (dead PID / stale heartbeat), '
            'or --force delete even if it looks live.'
        ),
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT',
            '/storage/validated_4d_su2_rg',
        ),
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Delete even if the lease still looks live (explicit opt-in).',
    )
    parser.add_argument(
        '--stale-heartbeat-sec',
        type=int,
        default=None,
        help='Same-host stale heartbeat seconds (default: 6h).',
    )
    parser.add_argument(
        '--foreign-stale-sec',
        type=int,
        default=None,
        help=(
            'Foreign-host stale heartbeat seconds '
            '(default: env VALIDATED_RG_GPU_LANE_FOREIGN_STALE_SEC or 900).'
        ),
    )
    args = parser.parse_args(argv)

    # Allow running from repo root without installing the package.
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from src.campaign_b.execution_keys import (  # noqa: E402
        DEFAULT_STALE_HEARTBEAT_SEC,
        try_reclaim_gpu_lane_lease,
    )

    kwargs: dict = {
        'persistent_root': Path(args.persistent_root),
        'force': bool(args.force),
    }
    if args.stale_heartbeat_sec is not None:
        kwargs['stale_heartbeat_sec'] = int(args.stale_heartbeat_sec)
    else:
        kwargs['stale_heartbeat_sec'] = DEFAULT_STALE_HEARTBEAT_SEC
    if args.foreign_stale_sec is not None:
        kwargs['foreign_stale_sec'] = int(args.foreign_stale_sec)

    result = try_reclaim_gpu_lane_lease(**kwargs)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))

    action = result.get('action')
    if action == 'reclaimed':
        print(
            f"Reclaimed GPU lane lease ({result.get('reason')}): "
            f"{result.get('path')}",
            file=sys.stderr,
        )
        return 0
    if action == 'noop':
        print('No GPU lane lease file present.', file=sys.stderr)
        return 0
    if action == 'refused':
        print(
            'Lease looks live; refused without --force. '
            f"owner={((result.get('lease') or {}).get('owner'))!r} "
            f"pid={((result.get('lease') or {}).get('pid'))!r} "
            f"hostname={((result.get('lease') or {}).get('hostname'))!r}",
            file=sys.stderr,
        )
        return 2
    print(f"Release failed: {result.get('reason')}", file=sys.stderr)
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
