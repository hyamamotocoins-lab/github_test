"""CPU tests for persist pause export / purge."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.campaign_b.persist_export import (
    DISCARD_M3_CHECKPOINTS_CONFIRM,
    PURGE_CONFIRM,
    PersistExportError,
    create_archive,
    export_and_optional_purge,
    plan_export,
    purge_persistent_root,
    verify_archive,
)


def _fill_persist(root: Path) -> None:
    (root / 'campaign_b' / 'M7-A' / 'selected' / 'B-x').mkdir(parents=True)
    (root / 'campaign_b' / 'M7-A' / 'selected' / 'B-x' / 'PRE_M6.json').write_text(
        '{"status":"PRE_M6_READY"}', encoding='utf-8',
    )
    (root / 'runs' / 'M3-TEST' / 'checkpoints' / 'ckpt_1').mkdir(parents=True)
    (root / 'runs' / 'M3-TEST' / 'checkpoints' / 'ckpt_1' / 'blob.bin').write_bytes(
        b'\x00' * 4096,
    )
    (root / 'runs' / 'M3-TEST' / 'reports').mkdir(parents=True)
    (root / 'runs' / 'M3-TEST' / 'reports' / 'M3_report.json').write_text(
        '{"phase":"M3_COMPLETE"}', encoding='utf-8',
    )
    (root / 'runs' / 'M4-TEST' / 'reports').mkdir(parents=True)
    (root / 'runs' / 'M4-TEST' / 'reports' / 'M4_report.json').write_text(
        '{"phase":"M4_COMPLETE"}', encoding='utf-8',
    )


def _fill_ssh(ssh: Path) -> None:
    ssh.mkdir(parents=True)
    (ssh / 'id_ed25519').write_text('PRIVATE', encoding='utf-8')
    (ssh / 'id_ed25519.pub').write_text('PUBLIC', encoding='utf-8')


def test_tier_a_omits_m3_checkpoints_includes_ssh(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    ssh = tmp_path / 'ssh'
    _fill_ssh(ssh)

    plan = plan_export(
        root,
        export_dir=tmp_path / 'exports',
        margin_bytes=0,
        tier='tier_a',
        include_ssh=True,
    )
    assert plan.ok, plan.block_reason
    assert plan.tier == 'tier_a'
    assert str(ssh) in plan.extra_roots

    result = create_archive(plan, execute=True)
    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert 'RESTORE.txt' in names
    assert 'validated_4d_su2_rg/campaign_b/M7-A/selected/B-x/PRE_M6.json' in names
    assert 'validated_4d_su2_rg/runs/M3-TEST/reports/M3_report.json' in names
    assert 'validated_4d_su2_rg/runs/M4-TEST/reports/M4_report.json' in names
    assert 'storage/ssh/id_ed25519' in names
    assert 'storage/ssh/id_ed25519.pub' in names
    assert not any('checkpoints' in n for n in names if n.startswith('validated_'))


def test_full_includes_checkpoints(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    _fill_ssh(tmp_path / 'ssh')
    plan = plan_export(
        root, export_dir=tmp_path / 'exports', margin_bytes=0, tier='full',
    )
    result = create_archive(plan, execute=True)
    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert any('checkpoints' in n for n in names)


def test_purge_tier_a_requires_discard_confirm(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    _fill_ssh(tmp_path / 'ssh')
    plan = plan_export(
        root, export_dir=tmp_path / 'exports', margin_bytes=0, tier='tier_a',
    )
    result = create_archive(plan, execute=True)
    with pytest.raises(PersistExportError, match='DISCARD_M3'):
        purge_persistent_root(
            root,
            archive_path=result.archive_path,
            confirm_purge=PURGE_CONFIRM,
            execute=True,
            tier='tier_a',
            expected_sha256=result.sha256,
        )
    purged = purge_persistent_root(
        root,
        archive_path=result.archive_path,
        confirm_purge=PURGE_CONFIRM,
        execute=True,
        tier='tier_a',
        confirm_discard_m3_checkpoints=DISCARD_M3_CHECKPOINTS_CONFIRM,
        expected_sha256=result.sha256,
    )
    assert purged['errors'] == []
    assert list(root.iterdir()) == []
    assert (tmp_path / 'ssh' / 'id_ed25519').is_file()  # ssh never purged
    assert result.archive_path.is_file()


def test_plan_refuses_archive_inside_persist(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    plan = plan_export(
        root,
        archive_path=root / 'inside.zip',
        margin_bytes=0,
        include_ssh=False,
    )
    assert not plan.ok
    assert 'outside' in (plan.block_reason or '')


def test_one_shot_export_without_purge(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    _fill_ssh(tmp_path / 'ssh')
    summary = export_and_optional_purge(
        root,
        export_dir=tmp_path / 'exports',
        execute=True,
        purge=False,
        margin_bytes=0,
        tier='tier_a',
    )
    assert summary['status'] == 'ARCHIVED'
    assert (root / 'runs' / 'M3-TEST' / 'checkpoints').is_dir()
    assert Path(summary['result']['archive_path']).is_file()
