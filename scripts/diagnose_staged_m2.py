#!/usr/bin/env python3
"""Diagnose staged M2 silent deaths (disk / checkpoint storm / progress)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


def main() -> None:
    persist = Path(
        os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg')
    )
    run_id = os.environ.get('VALIDATED_RG_M2_RUN_ID', 'M2-20260720S3-000005-ac85c')
    run_root = persist / 'runs' / run_id
    ckpt_root = run_root / 'checkpoints'
    print('run_root', run_root)
    print('exists', run_root.is_dir())
    usage = shutil.disk_usage(str(persist if persist.exists() else '/'))
    print('disk', {
        'total_gi': round(usage.total / 1024**3, 2),
        'used_gi': round(usage.used / 1024**3, 2),
        'free_gi': round(usage.free / 1024**3, 2),
    })
    if not ckpt_root.is_dir():
        print('no checkpoints dir')
        return
    committed = sorted(
        p for p in ckpt_root.glob('ckpt_*') if (p / 'COMMITTED').is_file()
    )
    print('committed_checkpoints', len(committed))
    sizes = []
    for path in committed[-5:]:
        size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        sizes.append((path.name, round(size / 1024**2, 2)))
    print('recent_ckpt_mib', sizes)
    total_ckpt = sum(
        f.stat().st_size for f in ckpt_root.rglob('*') if f.is_file()
    )
    print('checkpoints_total_gi', round(total_ckpt / 1024**3, 3))
    if committed:
        state = json.loads((committed[-1] / 'state.json').read_text())
        queue = json.loads((committed[-1] / 'work_queue.json').read_text())
        items = queue.get('items') or {}
        counts: dict[str, int] = {}
        for item in items.values():
            counts[item.get('status', '?')] = counts.get(item.get('status', '?'), 0) + 1
        print('phase', state.get('phase'), 'ckpt_index', state.get('checkpoint_index'))
        print('queue_counts', counts)
        print('notes_tail', (state.get('notes') or [])[-5:])
        pending = [
            item for item in items.values()
            if item.get('status') == 'pending'
            and item.get('phase') == 'M2_DENSE_REFERENCE'
        ]
        if pending:
            pending.sort(key=lambda item: int((item.get('parameters') or {}).get('batch_start', 0)))
            p0 = pending[0]['parameters']
            print('next_dense_batch', p0)
    print(
        'hypothesis: before+after checkpoint on every tiny batch floods I/O; '
        'Jupyter kernels die without Python traceback. Use nohup + pull fix '
        'that skips before-checkpoints and prunes old ckpts.'
    )


if __name__ == '__main__':
    main()
