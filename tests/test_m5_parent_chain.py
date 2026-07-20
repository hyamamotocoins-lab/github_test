from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json
from src.m5_parent_chain import load_accepted_parent


def test_load_accepted_parent_prefers_latest_committed(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    run_id = 'M4-test-latest-ckpt'
    run_root = persist / 'runs' / run_id
    for index in (14, 27):
        ckpt = run_root / 'checkpoints' / f'ckpt_{index:06d}'
        ckpt.mkdir(parents=True)
        (ckpt / 'COMMITTED').write_text('ok', encoding='utf-8')
    report = run_root / 'reports' / 'M4_report.json'
    report.parent.mkdir(parents=True)
    atomic_write_json(report, {'phase': 'M4_COMPLETE'})
    audit_dir = project / 'audit'
    audit_dir.mkdir(parents=True)
    atomic_write_json(audit_dir / 'm4_accepted_parent.json', {
        'accepted_run_id': run_id,
        # Intentionally omit checkpoint_path so latest committed wins.
    })
    ref = load_accepted_parent(
        project, persist,
        audit_relative='audit/m4_accepted_parent.json',
        milestone='M4',
    )
    assert ref.checkpoint.name == 'ckpt_000027'


def test_load_accepted_parent_uses_audit_checkpoint_when_present(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    run_id = 'M3-test-audit-ckpt'
    run_root = persist / 'runs' / run_id
    for index in (14, 20):
        ckpt = run_root / 'checkpoints' / f'ckpt_{index:06d}'
        ckpt.mkdir(parents=True)
        (ckpt / 'COMMITTED').write_text('ok', encoding='utf-8')
    report = run_root / 'reports' / 'M3_report.json'
    report.parent.mkdir(parents=True)
    atomic_write_json(report, {'phase': 'M3_COMPLETE'})
    pinned = run_root / 'checkpoints' / 'ckpt_000014'
    audit_dir = project / 'audit'
    audit_dir.mkdir(parents=True)
    atomic_write_json(audit_dir / 'm3_accepted_parent.json', {
        'accepted_run_id': run_id,
        'checkpoint_path': str(pinned),
    })
    ref = load_accepted_parent(
        project, persist,
        audit_relative='audit/m3_accepted_parent.json',
        milestone='M3',
    )
    assert ref.checkpoint.name == 'ckpt_000014'
