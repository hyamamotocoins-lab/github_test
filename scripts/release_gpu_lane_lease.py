#!/usr/bin/env python3
"""Inspect / reclaim the Campaign B GPU lane lease.

Lock file::

  {PERSIST}/campaign_b/_locks/gpu_lane.json

Paperspace (from repo root)::

  export VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg

  # Inspect only
  python scripts/release_gpu_lane_lease.py --status

  # Safe reclaim (default): dead PID / stale heartbeat
  python scripts/release_gpu_lane_lease.py --force-if-dead
  # equivalent: bare invoke also runs safe reclaim
  python scripts/release_gpu_lane_lease.py

  # DANGEROUS: unlink even if holder looks live
  python scripts/release_gpu_lane_lease.py --force --i-understand

Manual PID check (example 13712)::

  cat /storage/validated_4d_su2_rg/campaign_b/_locks/gpu_lane.json
  ps -p 13712 -o pid,ppid,etime,cmd || echo 'PID dead'
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
from pathlib import Path
from typing import Any


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            'Inspect or reclaim campaign_b/_locks/gpu_lane.json. '
            'Default reclaim is --force-if-dead (safe).'
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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--status',
        action='store_true',
        help='Inspect only; do not unlink',
    )
    mode.add_argument(
        '--force-if-dead',
        action='store_true',
        help=(
            'Unlink only if holder PID is dead or heartbeat is stale '
            '(safe; also the default when no mode flag is given)'
        ),
    )
    mode.add_argument(
        '--force',
        action='store_true',
        help='DANGEROUS: unlink even if holder looks live (requires --i-understand)',
    )
    parser.add_argument(
        '--i-understand',
        action='store_true',
        help='Required acknowledgement for --force',
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
        gpu_lane_path,
        read_gpu_lane_lease,
        try_reclaim_gpu_lane_lease,
    )

    persist = Path(args.persistent_root).expanduser().resolve()

    if args.status:
        path = gpu_lane_path(persist)
        doc = read_gpu_lane_lease(persist)
        payload: dict[str, Any]
        if doc is None:
            payload = {
                'present': False,
                'path': str(path),
                'local_hostname': socket.gethostname(),
            }
        else:
            try:
                pid = int(doc.get('pid') or 0)
            except (TypeError, ValueError):
                pid = 0
            host = str(doc.get('hostname') or '')
            local = socket.gethostname()
            same_host = bool(host) and host == local
            payload = {
                'present': True,
                'path': str(path),
                'local_hostname': local,
                'owner': doc.get('owner'),
                'pid': pid,
                'hostname': host,
                'same_host': same_host,
                'pid_alive_local': _pid_alive(pid) if same_host else None,
                'heartbeat_at': doc.get('heartbeat_at'),
                'acquired_at': doc.get('acquired_at'),
                'depth': doc.get('depth'),
                'lease': doc,
            }
        print(json.dumps(payload, indent=2, ensure_ascii=False, default=str))
        return 0

    if args.force and not args.i_understand:
        print(
            'Refusing --force without --i-understand. '
            'This can interrupt a live GPU consumer.',
            file=sys.stderr,
        )
        return 2

    # Bare invoke or --force-if-dead → safe reclaim; --force → force.
    kwargs: dict[str, Any] = {
        'persistent_root': persist,
        'force': bool(args.force),
        'stale_heartbeat_sec': (
            int(args.stale_heartbeat_sec)
            if args.stale_heartbeat_sec is not None
            else DEFAULT_STALE_HEARTBEAT_SEC
        ),
    }
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
            'Lease looks live; refused without --force --i-understand. '
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
