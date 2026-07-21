"""CPU unit tests for scripts/persist_reclaim_m3.py and src.campaign_b.m3_reclaim."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.campaign_b import m3_reclaim as lib

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


def test_library_strip_eligible_matches_cli(tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-lib-safe'
    m4 = 'M4-lib-safe'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=900)
    _m4_complete(root, m4)
    _package(root, campaign='camp1', name='pkg', m3=m3, m4=m4)

    summary = lib.strip_eligible_m3_checkpoints(root, execute=True)
    assert summary.stripped == 1
    assert summary.bytes_freed > 0
    assert (root / 'runs' / m3 / 'checkpoints' / 'STRIPPED_FOR_RECLAIM.json').is_file()
    assert (root / 'runs' / m3 / 'reports' / 'M3_report.json').is_file()


def test_incremental_only_run_ids(tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3_a = 'M3-inc-a'
    m3_b = 'M3-inc-b'
    m4_a = 'M4-inc-a'
    m4_b = 'M4-inc-b'
    _m3_complete_tree(root / 'runs' / m3_a, ckpt_blob=500)
    _m3_complete_tree(root / 'runs' / m3_b, ckpt_blob=500)
    _m4_complete(root, m4_a)
    _m4_complete(root, m4_b)
    _package(root, campaign='camp1', name='pkg-a', m3=m3_a, m4=m4_a)
    _package(root, campaign='camp1', name='pkg-b', m3=m3_b, m4=m4_b)

    summary = lib.strip_eligible_m3_checkpoints(
        root, execute=True, only_run_ids={m3_a},
    )
    assert summary.scope == 'incremental'
    assert summary.run_ids == [m3_a]
    assert (root / 'runs' / m3_a / 'checkpoints' / 'STRIPPED_FOR_RECLAIM.json').is_file()
    assert (root / 'runs' / m3_b / 'checkpoints' / 'ckpt_000001').is_dir()


def test_auto_strip_after_round_uses_pre_m6_package(tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-round-a'
    m4 = 'M4-round-a'
    pkg = _package(root, campaign='camp1', name='pkg-round', m3=m3, m4=m4)
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=400)
    _m4_complete(root, m4)

    out = lib.auto_strip_after_pipeline_round(
        root,
        pre_m6_summary={
            'results': [{'package': str(pkg), 'status': 'PRE_M6_READY'}],
        },
        execute=True,
    )
    assert out['scope'] == 'incremental'
    assert out['stripped'] == 1
    assert m3 in out['preferred_run_ids']


def test_force_full_scan_strips_backlog_without_pre_m6(tmp_path: Path) -> None:
    """Session-start full scan must reclaim eligible runs even with no PRE_M6."""
    root = tmp_path / 'persist'
    m3 = 'M3-backlog'
    m4 = 'M4-backlog'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=800)
    _m4_complete(root, m4)
    _package(root, campaign='camp1', name='pkg', m3=m3, m4=m4)

    out = lib.auto_strip_after_pipeline_round(
        root,
        pre_m6_summary={'results': []},
        m6_summary={'results': []},
        execute=True,
        force_full_scan=True,
    )
    assert out['force_full_scan'] is True
    assert out['scope'] == 'full_scan'
    assert out['stripped'] == 1
    assert (root / 'runs' / m3 / 'checkpoints' / 'STRIPPED_FOR_RECLAIM.json').is_file()


def test_strip_tensors_keeps_ckpt_metadata(reclaim, tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-tensors-only'
    m4 = 'M4-tensors-only'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=1200, n_ckpts=2)
    _m4_complete(root, m4)
    _package(root, campaign='camp1', name='pkg', m3=m3, m4=m4)

    code = reclaim.main([
        '--persistent-root', str(root),
        '--mode', 'strip-tensors',
        '--execute',
    ])
    assert code == 0
    ckpt = root / 'runs' / m3 / 'checkpoints' / 'ckpt_000001'
    assert ckpt.is_dir()
    assert (ckpt / 'COMMITTED').is_file()
    assert (ckpt / 'state.json').is_file()
    assert not (ckpt / 'tensors').exists()
    assert (
        root / 'runs' / m3 / 'checkpoints' / 'STRIPPED_TENSORS_FOR_RECLAIM.json'
    ).is_file()
    assert (root / 'runs' / m3 / 'reports' / 'M3_report.json').is_file()


def test_enforce_persist_m3_cap_strips_oldest(tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    # Two eligible runs; tiny cap forces at least one strip.
    for suffix, blob in (('old', 5000), ('new', 5000)):
        m3 = f'M3-cap-{suffix}'
        m4 = f'M4-cap-{suffix}'
        run = root / 'runs' / m3
        _m3_complete_tree(run, ckpt_blob=blob)
        _m4_complete(root, m4)
        _package(root, campaign='camp1', name=f'pkg-{suffix}', m3=m3, m4=m4)
        # Ensure deterministic mtime order.
        import time
        time.sleep(0.02)
        run.touch()

    # Cap far below total so both may be needed; at least oldest should go.
    out = lib.enforce_persist_m3_cap(root, cap_gib=0.000001, execute=True)
    assert out['stripped'] >= 1
    assert out['bytes_freed'] > 0
    assert (root / 'runs' / 'M3-cap-old' / 'checkpoints' / 'STRIPPED_FOR_RECLAIM.json').is_file()


def test_keep_latest_for_m3_run_id_trims_midflight(tmp_path: Path) -> None:
    root = tmp_path / 'persist'
    m3 = 'M3-midflight'
    _m3_complete_tree(root / 'runs' / m3, ckpt_blob=300, n_ckpts=4)
    out = lib.keep_latest_for_m3_run_id(root, m3, execute=True)
    assert out['label'] == 'KEPT_LATEST_REMOVED_OLDER'
    assert out['bytes_freed'] > 0
    ckpts = sorted((root / 'runs' / m3 / 'checkpoints').glob('ckpt_*'))
    assert [p.name for p in ckpts] == ['ckpt_000004']


def test_write_m3_recipe_stub(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from src.campaign_b.gpu_m3_batch import write_m3_recipe_stub
    from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE

    run_root = tmp_path / 'runs' / 'M3-recipe'
    run_root.mkdir(parents=True)
    pkg = tmp_path / 'campaign_b' / 'c1' / 'selected' / 'cand'
    pkg.mkdir(parents=True)
    _write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': 'cand',
        'scheme': {
            'target_rank': 16,
            'perron_weight_strategy': 'all_ones',
            'seed': 7,
        },
    })
    config = SimpleNamespace(target_rank=16, seed=7, config_hash='abc')
    recipe = write_m3_recipe_stub(
        run_root=run_root,
        package=pkg,
        m3_run_id='M3-recipe',
        m2_run_id='M2-parent',
        config=config,
    )
    assert (run_root / 'reports' / 'M3_RECIPE.json').is_file()
    assert (pkg / 'm3_recipe.json').is_file()
    assert recipe['backend'] == 'legacy_rsvd'
    assert recipe['certification_status'] == CERTIFICATION_STATUS
    assert recipe['claim_scope'] == CLAIM_SCOPE
    assert recipe['m2_run_id'] == 'M2-parent'
    assert recipe['target_rank'] == 16


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
