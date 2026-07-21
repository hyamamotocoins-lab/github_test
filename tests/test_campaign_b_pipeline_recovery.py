"""Minimal recovery helper tests."""

from __future__ import annotations

from pathlib import Path

from src.campaign_b.pipeline_recovery import recover_interrupted_work


def test_recover_removes_tmp_files(tmp_path: Path) -> None:
    camp = tmp_path / 'campaign_b' / '_end_to_end'
    camp.mkdir(parents=True)
    junk = camp / 'foo.json.tmp'
    junk.write_text('x', encoding='utf-8')
    nested = camp / 'runtime' / '.bar.yaml.tmp-abc'
    nested.parent.mkdir(parents=True)
    nested.write_text('y', encoding='utf-8')
    keep = camp / 'keep.json'
    keep.write_text('{}', encoding='utf-8')

    summary = recover_interrupted_work(tmp_path)
    assert summary['removed_tmp_count'] >= 2
    assert not junk.exists()
    assert not nested.exists()
    assert keep.is_file()
