"""CPU unit tests for scripts/persist_reclaim_m3.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / 'scripts' / 'persist_reclaim_m3.py'


def _load_mod():
    spec = importlib.util.spec_from_file_location('persist_reclaim_m3', SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules['persist_reclaim_m3'] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope='module')
def reclaim():
    return _load_mod()


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


def _make_blob(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'x' * size)


def _m3_complete_tree(run_root: Path, *, ckpt_blob: int = 1000, n_ckpts: int = 2) -> None:
    _write_json(run_root / 'reports' / 'M3_report.json', {
        'phase': 'M3_COMPLETE',
        'milestone_status': 'CORE_REPRODUCED',
    })
    _write_json(run_root / 'reports' / 'M3_acceptance.json', {'status': 'PASS'})
    _write_json(run_root / 'run_config.json', {'milestone': 'M3'})
    for i in range(1, n_ckpts + 1):
        ckpt = run_root / 'checkpoints' / f'ckpt_{i:06d}'
        (ckpt).mkdir(parents=True)
        (ckpt / 'COMMITTED').write_text('ok\n', encoding='utf-8')
        _write_json(ckpt / 'state.json', {'phase': 'M3_COMPLETE', 'checkpoint_index': i})
        _make_blob(ckpt / 'tensors' / 'triad_left.shard-000000.npy', ckpt_blob)


def _m4_complete(persistent_root: Path, m4_id: str) -> None:
    root = persistent_root / 'runs' / m4_id
    _write_json(root / 'reports' / 'M4_report.json', {'phase': 'M4_COMPLETE'})
    _write_json(root / 'reports' / 'M4_acceptance.json', {'status': 'PASS'})


def _package(
    persistent_root: Path,
    *,
    campaign: str,
    name: str,
    m3: str,
    m4: str | None = None,
    m5: str | None = None,
    m6: str | None = None,
    gpu_status: str = 'M3_COMPLETE',
    archived: bool = False,
    m6_cert: str | None = None,
) -> Path:
    pkg = persistent_root / 'campaign_b' / campaign / 'selected' / name
    pkg.mkdir(parents=True)
    _write_json(pkg / 'candidate_manifest.json', {'candidate_id': name})
    child: dict[str, object] = {'M3': m3}
    if m4:
        child['M4'] = m4
    if m5:
        child['M5'] = m5
    if m6:
        child['M6'] = m6
    _write_json(pkg / 'child_run_ids.json', child)
    _write_json(pkg / 'GPU_M3.json', {'status': gpu_status, 'm3_run_id': m3})
    if archived:
        _write_json(pkg / 'ADVANCE.json', {'package_state': 'ARCHIVED'})
    if m6 and m6_cert:
        _write_json(
            persistent_root / 'runs' / m6 / 'reports' / 'M6_acceptance.json',
            {'certification_status': m6_cert},
        )
    return pkg


def test_classify_and_strip_safe_complete_downstream(reclaim, tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3_safe = 'M3-safe-aaaa'
    m3_need = 'M3-need-bbbb'
    m3_unref = 'M3-unref-cccc'
    m4 = 'M4-safe-aaaa'

    _m3_complete_tree(root / 'runs' / m3_safe, ckpt_blob=2000, n_ckpts=2)
    _m3_complete_tree(root / 'runs' / m3_need, ckpt_blob=1500, n_ckpts=2)
    _m3_complete_tree(root / 'runs' / m3_unref, ckpt_blob=800, n_ckpts=1)
    _m4_complete(root, m4)

    _package(
        root, campaign='camp1', name='pkg-safe',
        m3=m3_safe, m4=m4,
    )
    _package(
        root, campaign='camp1', name='pkg-need',
        m3=m3_need, m4='M4-missing',
    )

    rows = reclaim.classify_m3_runs(root)
    by_id = {r.run_id: r for r in rows}
    assert by_id[m3_safe].complete is True
    assert by_id[m3_safe].referenced is True
    assert by_id[m3_safe].has_downstream_safe is True
    assert by_id[m3_safe].reclaimable_strip_bytes > 0

    assert by_id[m3_need].complete is True
    assert by_id[m3_need].reclaimable_strip_bytes == 0
    assert 'no_downstream_m4_complete_or_later' in by_id[m3_need].skip_reasons

    assert by_id[m3_unref].referenced is False
    assert by_id[m3_unref].reclaimable_strip_bytes == 0

    # Dry-run does not delete.
    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'strip-checkpoints',
    ])
    assert code == 0
    assert (root / 'runs' / m3_safe / 'checkpoints' / 'ckpt_000001' / 'tensors').is_dir()

    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'strip-checkpoints',
        '--execute',
    ])
    assert code == 0
    assert not (root / 'runs' / m3_safe / 'checkpoints' / 'ckpt_000001').exists()
    assert (root / 'runs' / m3_safe / 'checkpoints' / 'STRIPPED_FOR_RECLAIM.json').is_file()
    assert (root / 'runs' / m3_safe / 'reports' / 'M3_report.json').is_file()
    assert (root / 'runs' / m3_safe / 'run_config.json').is_file()
    # Incomplete-downstream untouched.
    assert (root / 'runs' / m3_need / 'checkpoints' / 'ckpt_000001' / 'tensors').is_dir()


def test_skip_certified_lineage_unless_flag(reclaim, tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-cert-lineage'
    m4 = 'M4-cert'
    m6 = 'M6-cert'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=500)
    _m4_complete(root, m4)
    _package(
        root, campaign='camp1', name='pkg-cert',
        m3=m3, m4=m4, m6=m6, m6_cert='CERTIFIED',
    )
    rows = reclaim.classify_m3_runs(root, include_certified_lineage=False)
    assert rows[0].reclaimable_strip_bytes == 0
    assert 'certified_m6_lineage' in rows[0].skip_reasons

    rows2 = reclaim.classify_m3_runs(root, include_certified_lineage=True)
    assert rows2[0].reclaimable_strip_bytes > 0


def test_keep_latest_checkpoint(reclaim, tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-keep-latest'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=400, n_ckpts=3)
    # Incomplete: no package downstream — still eligible for keep-latest.
    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'keep-latest-checkpoint',
        '--execute',
    ])
    assert code == 0
    ckpts = sorted((root / 'runs' / m3 / 'checkpoints').glob('ckpt_*'))
    assert [p.name for p in ckpts] == ['ckpt_000003']


def test_delete_run_requires_flag_and_skips_referenced(reclaim, tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3_ref = 'M3-ref-del'
    m3_unref = 'M3-unref-del'
    _m3_complete_tree(root / 'runs' / m3_ref, ckpt_blob=100)
    _m3_complete_tree(root / 'runs' / m3_unref, ckpt_blob=100)
    _package(root, campaign='camp1', name='pkg', m3=m3_ref)

    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'delete-run',
        '--execute',
    ])
    assert code == 2
    assert (root / 'runs' / m3_ref).is_dir()

    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'delete-run',
        '--allow-delete-run',
        '--execute',
    ])
    assert code == 0
    assert (root / 'runs' / m3_ref).is_dir()
    assert not (root / 'runs' / m3_unref).exists()


def test_fail_closed_missing_root(reclaim, tmp_path: Path) -> None:
    code = reclaim.main(['--persistent-root', str(tmp_path / 'missing')])
    assert code == 2
