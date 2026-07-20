from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from src.common import atomic_write_json, atomic_write_text, sha256_file
from src.m1_config import M1Config
from src.work_queue import WorkQueue


def make_synthetic_accepted_parent(tmp_path: Path, base: M1Config | None = None) -> M1Config:
    config = base or M1Config()
    parent_run_root = tmp_path / 'accepted-storage' / 'runs' / config.parent_run_id
    checkpoint = parent_run_root / 'checkpoints' / 'ckpt_000014'
    checkpoint.mkdir(parents=True, exist_ok=False)
    state = {
        'run_id': config.parent_run_id, 'config_hash': '0' * 64,
        'created_at': '2026-07-19T12:04:06+00:00', 'updated_at': '2026-07-19T12:04:08+00:00',
        'phase': 'M0_COMPLETE', 'checkpoint_index': 14,
        'certification_status': 'NOT_CERTIFIED',
    }
    queue = WorkQueue()
    for index in range(6):
        item_id = queue.add('DUMMY', 'synthetic-parent', {'index': index}, predicted_s=1.0)
        item = queue.items[item_id]; item.status = 'done'
        result = parent_run_root / 'artifacts' / str(index) / 'result.bin'
        result.parent.mkdir(parents=True, exist_ok=True); result.write_bytes(f'result-{index}'.encode('ascii'))
        item.result_relpath = result.relative_to(parent_run_root).as_posix()
        item.result_sha256 = sha256_file(result)
        marker = parent_run_root / 'work_items' / f'{item_id}.done'
        marker.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(marker, {
            'item_id': item_id, 'result_relpath': item.result_relpath,
            'result_sha256': item.result_sha256,
        })
    atomic_write_json(checkpoint / 'state.json', state)
    atomic_write_json(checkpoint / 'work_queue.json', queue.to_payload())
    hashes = {name: sha256_file(checkpoint / name) for name in ('state.json', 'work_queue.json')}
    atomic_write_json(checkpoint / 'hashes.json', hashes)
    atomic_write_text(checkpoint / 'COMMITTED', 'committed\n')
    return replace(config, parent_checkpoint_path=str(checkpoint))


def passing_test_report() -> dict[str, Any]:
    return {
        'm0_regression_cpu_suite': 'PASS', 'm1_required_cpu_suite': 'PASS',
        'optional_gpu_suite': 'NOT_RUN_NO_CUDA', 'm1_fresh_process_resume': 'PASS',
    }
