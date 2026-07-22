"""CPU tests for persist pause export / purge."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pytest

from src.campaign_b.persist_export import (
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
    (root / 'runs' / 'M3-TEST').mkdir(parents=True)
    (root / 'runs' / 'M3-TEST' / 'blob.bin').write_bytes(b'\x00' * 4096)


def test_plan_refuses_archive_inside_persist(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    plan = plan_export(
        root,
        archive_path=root / 'inside.zip',
        margin_bytes=0,
    )
    assert not plan.ok
    assert 'outside' in (plan.block_reason or '')


def test_archive_verify_and_purge(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    export_dir = tmp_path / 'exports'
    plan = plan_export(root, export_dir=export_dir, margin_bytes=0)
    assert plan.ok, plan.block_reason

    dry = create_archive(plan, execute=False)
    assert dry.dry_run
    assert not plan.archive_path.exists()

    result = create_archive(plan, execute=True)
    assert result.archive_path.is_file()
    assert result.sha256_path.is_file()
    assert result.manifest_path.is_file()
    assert result.member_count >= 2

    report = verify_archive(
        result.archive_path,
        expected_sha256=result.sha256,
        expected_member_count=result.member_count,
    )
    assert report['ok']

    with zipfile.ZipFile(result.archive_path) as zf:
        names = set(zf.namelist())
    assert 'runs/M3-TEST/blob.bin' in names

    with pytest.raises(PersistExportError):
        purge_persistent_root(
            root,
            archive_path=result.archive_path,
            confirm_purge='nope',
            execute=True,
        )

    purged = purge_persistent_root(
        root,
        archive_path=result.archive_path,
        confirm_purge=PURGE_CONFIRM,
        execute=True,
        expected_sha256=result.sha256,
    )
    assert purged['errors'] == []
    assert list(root.iterdir()) == []
    assert result.archive_path.is_file()
    marker = export_dir / f'{root.name}_PURGED.json'
    assert marker.is_file()
    doc = json.loads(marker.read_text(encoding='utf-8'))
    assert doc['status'] == 'PURGED_AFTER_EXPORT'


def test_one_shot_export_without_purge(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    summary = export_and_optional_purge(
        root,
        export_dir=tmp_path / 'exports',
        execute=True,
        purge=False,
        margin_bytes=0,
    )
    assert summary['status'] == 'ARCHIVED'
    assert (root / 'runs' / 'M3-TEST' / 'blob.bin').is_file()
    assert Path(summary['result']['archive_path']).is_file()


def test_one_shot_blocks_without_purge_confirm(tmp_path: Path) -> None:
    root = tmp_path / 'validated_4d_su2_rg'
    root.mkdir()
    _fill_persist(root)
    with pytest.raises(PersistExportError, match='confirm_purge'):
        export_and_optional_purge(
            root,
            export_dir=tmp_path / 'exports',
            execute=True,
            purge=True,
            confirm_purge=None,
            margin_bytes=0,
        )
