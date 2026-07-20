from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.common import atomic_write_json, sha256_file
from src.m2_compatibility import (
    accepted_m1_identity_sha256,
    compute_proof_key,
    compute_structural_key,
    shared_run_id_for_keys,
)
from src.m2_config import M2Config
from src.m2_package_audit import write_package_m2_shared_audit
from src.m2_shared_registry import (
    BINDING_READY,
    MODE_LEGACY,
    MODE_STRICT,
    M2SharedRegistryError,
    ensure_package_m2_run_id,
    lookup_shared_m2,
    register_shared_m2_from_run,
    reserve_shared_m2,
    resolve_m2_binding,
    write_binding,
)
from src.m7_lineage import build_s3_lineage_plan
from src.m7_promotion import (
    STATUS_PROMOTE,
    evaluate_promotion,
    is_promoted,
    require_promote_for_canonical_m2,
)


def test_semantic_payload_ignores_batch_and_schedule() -> None:
    a = M2Config(j2_max=1, sector_batch_size=0, seed=1)
    b = M2Config(j2_max=1, sector_batch_size=0, seed=99, checkpoint_interval_s=60.0)
    assert a.semantic_compatibility_payload() == b.semantic_compatibility_payload()


def test_structural_key_ignores_run_id_uses_identity() -> None:
    identity = accepted_m1_identity_sha256(
        m1_report_sha256='aa',
        m1_acceptance_sha256='bb',
        checkpoint_hash_manifest_sha256='cc',
    )
    k1 = compute_structural_key(
        accepted_m1_identity=identity,
        config=M2Config(j2_max=2, sector_batch_size=16),
        m1_parent_run_id_provenance='M1-AAA',
    )
    k2 = compute_structural_key(
        accepted_m1_identity=identity,
        config=M2Config(j2_max=2, sector_batch_size=16),
        m1_parent_run_id_provenance='M1-BBB',
    )
    assert k1 == k2


def test_proof_key_differs_on_source_hash() -> None:
    identity = accepted_m1_identity_sha256(
        m1_report_sha256='aa',
        m1_acceptance_sha256='bb',
        checkpoint_hash_manifest_sha256='cc',
    )
    structural = compute_structural_key(
        accepted_m1_identity=identity,
        config=M2Config(j2_max=2, sector_batch_size=16),
    )
    p1 = compute_proof_key(
        structural_key=structural, source_hash='s1', notebook_hash='n1',
    )
    p2 = compute_proof_key(
        structural_key=structural, source_hash='s2', notebook_hash='n1',
    )
    assert p1 != p2
    assert shared_run_id_for_keys(structural, p1) != shared_run_id_for_keys(structural, p2)


def test_channel_policy_not_in_structural_key() -> None:
    identity = accepted_m1_identity_sha256(
        m1_report_sha256='aa', m1_acceptance_sha256='bb',
        checkpoint_hash_manifest_sha256='cc',
    )
    k = compute_structural_key(
        accepted_m1_identity=identity,
        config=M2Config(j2_max=2, sector_batch_size=16),
    )
    plan_a = build_s3_lineage_plan(
        {
            'candidate_id': 'CAND-000001-aaaaaaaaaaaa',
            'scheme_hash': 'h1',
            'scheme': {'change_class': 'S3', 'j2_max': 2, 'channel_policy': 'A'},
        },
        parent_m6_run_id='M6-x',
        search_run_id='M7-20260720T081500Z-c8d5f02b3c96',
        m2_binding={
            'state': 'NEED_CANONICAL_M2',
            'mode': 'NEED_CANONICAL_M2',
            'structural_key': k,
            'proof_key': 'p' * 64,
            'canonical_run_id': shared_run_id_for_keys(k, 'p' * 64),
        },
    )
    plan_b = build_s3_lineage_plan(
        {
            'candidate_id': 'CAND-000002-bbbbbbbbbbbb',
            'scheme_hash': 'h2',
            'scheme': {'change_class': 'S3', 'j2_max': 2, 'channel_policy': 'B'},
        },
        parent_m6_run_id='M6-x',
        search_run_id='M7-20260720T081500Z-c8d5f02b3c96',
        m2_binding={
            'state': 'NEED_CANONICAL_M2',
            'mode': 'NEED_CANONICAL_M2',
            'structural_key': k,
            'proof_key': 'p' * 64,
            'canonical_run_id': shared_run_id_for_keys(k, 'p' * 64),
        },
    )
    assert plan_a['M2_structural_key'] == plan_b['M2_structural_key']
    assert plan_a['child_run_ids']['M2'] == plan_b['child_run_ids']['M2']
    assert plan_a['child_run_ids']['M3'] != plan_b['child_run_ids']['M3']
    assert 'M2-' in plan_a['child_run_ids']['M2']
    assert 'S3-000001' not in plan_a['child_run_ids'].get('M2', '')


def test_reservation_exclusive(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    sk, pk = 's' * 64, 'p' * 64
    first = reserve_shared_m2(persist, sk, pk, owner_id='A', lease_seconds=3600)
    assert first['owner_id'] == 'A'
    with pytest.raises(M2SharedRegistryError):
        reserve_shared_m2(persist, sk, pk, owner_id='B', lease_seconds=3600)


def test_stale_lease_takeover(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    sk, pk = 's' * 64, 'p' * 64
    reserve_shared_m2(persist, sk, pk, owner_id='A', lease_seconds=-1)
    # Negative lease already expired.
    second = reserve_shared_m2(persist, sk, pk, owner_id='B', lease_seconds=3600)
    assert second['owner_id'] == 'B'


def _fake_complete_m2(run_root: Path, *, source_hash: str = 'src1') -> None:
    from src.work_queue import WorkQueue
    run_root.mkdir(parents=True)
    (run_root / 'reports').mkdir()
    (run_root / 'checkpoints' / 'ckpt_000001').mkdir(parents=True)
    atomic_write_json(run_root / 'run_config.json', {
        'j2_max': 2,
        'leg_count': 6,
        'orientations': [1, -1, 1, -1, 1, -1],
        'exact_decimal_digits': 80,
        'proof_schema': 'M2_PROOF_SCHEMA_V2',
        'proof_method': 'invariant_subspace_uniqueness_v1',
        'parent_run_id': 'M1-x',
        'parent_checkpoint': 'ckpt_000014',
        'sector_batch_size': 16,
    })
    atomic_write_json(run_root / 'run_manifest.json', {
        'source_hash': source_hash,
        'notebook_hash': 'nb1',
        'milestone': 'M2',
        'run_id': run_root.name,
        'certification_status': 'NOT_CERTIFIED',
        'parent': {
            'm1_report_sha256': 'aa',
            'm1_acceptance_sha256': 'bb',
            'parent_checkpoint_hash_manifest_sha256': 'cc',
        },
    })
    atomic_write_json(run_root / 'reports' / 'M2_report.json', {
        'run_id': run_root.name,
        'phase': 'M2_COMPLETE',
        'certification_status': 'NOT_CERTIFIED',
        'checkpoint': {'path': str(run_root / 'checkpoints' / 'ckpt_000001'), 'index': 1},
        'proof_artifact_hashes': {},
    })
    atomic_write_json(run_root / 'reports' / 'M2_acceptance.json', {
        'milestone': 'M2',
        'phase': 'M2_COMPLETE',
        'status': 'PASS',
        'certification_status': 'NOT_CERTIFIED',
        'gates': {'ok': True},
    })
    atomic_write_json(run_root / 'checkpoints' / 'ckpt_000001' / 'hashes.json', {'a': 1})
    atomic_write_json(run_root / 'checkpoints' / 'ckpt_000001' / 'state.json', {
        'phase': 'M2_COMPLETE',
        'certification_status': 'NOT_CERTIFIED',
        'checkpoint_index': 1,
    })
    item_id = WorkQueue.make_id('REPORT', 'i' * 64, {})
    atomic_write_json(run_root / 'checkpoints' / 'ckpt_000001' / 'work_queue.json', {
        'items': {
            item_id: {
                'item_id': item_id,
                'phase': 'REPORT',
                'status': 'done',
                'input_hash': 'i' * 64,
                'parameters': {},
                'result_sha256': 'd' * 64,
                'result_relpath': 'artifacts/x.json',
            }
        }
    })
    (run_root / 'checkpoints' / 'ckpt_000001' / 'COMMITTED').write_text('ok\n')


def test_strict_refuses_code_drift(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    run = persist / 'runs' / 'M2-legacy'
    _fake_complete_m2(run)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    manifest['code_drift'] = True
    atomic_write_json(run / 'run_manifest.json', manifest)
    with pytest.raises(M2SharedRegistryError, match='code_drift'):
        register_shared_m2_from_run(
            persist, run, registration_mode=MODE_STRICT,
        )


def test_legacy_quarantine_registers(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    run = persist / 'runs' / 'M2-legacy'
    _fake_complete_m2(run)
    manifest = json.loads((run / 'run_manifest.json').read_text())
    manifest['code_drift'] = True
    atomic_write_json(run / 'run_manifest.json', manifest)
    record = register_shared_m2_from_run(
        persist, run, registration_mode=MODE_LEGACY,
    )
    assert record['registry_state'] == 'QUARANTINED'
    assert lookup_shared_m2(
        persist, record['structural_key'], record['proof_key'],
    ) is not None


def test_promotion_ignores_s0_advance(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    atomic_write_json(pkg / 'ADVANCE.json', {
        'status': 'SELECTED', 'selected_rank': 36,
    })
    result = evaluate_promotion(
        package_root=pkg,
        j2_max=2,
        estimated_q=0.9,
        rank_among_executable=10,
        top_k=3,
    )
    assert result['status'] != STATUS_PROMOTE
    assert is_promoted(pkg) is False


def test_require_promote_fail_closed(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    with pytest.raises(Exception):
        require_promote_for_canonical_m2(pkg)


def test_ready_binding_immutable(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    write_binding(pkg, {
        'schema_version': 2,
        'state': BINDING_READY,
        'structural_key': 's' * 64,
        'proof_key': 'p' * 64,
        'canonical_run_id': 'M2-SHARED-x',
        'registry_record_sha256': 'r1',
        'mode': 'REUSE_SHARED',
    })
    child = json.loads((pkg / 'child_run_ids.json').read_text(encoding='utf-8'))
    assert child['M2'] == 'M2-SHARED-x'
    with pytest.raises(M2SharedRegistryError):
        write_binding(pkg, {
            'schema_version': 2,
            'state': 'NEED_CANONICAL_M2',
            'structural_key': 's' * 64,
            'proof_key': 'p' * 64,
            'canonical_run_id': 'M2-SHARED-x',
            'mode': 'NEED_CANONICAL_M2',
        })


def test_ensure_package_m2_run_id_syncs_from_binding(tmp_path: Path) -> None:
    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    atomic_write_json(pkg / 'm2_binding.json', {
        'schema_version': 2,
        'state': 'NEED_CANONICAL_M2',
        'canonical_run_id': 'M2-SHARED-ed77fc1e-207ed187722f',
        'mode': 'NEED_CANONICAL_M2',
    })
    # Simulate pre-fix package: binding present, child_run_ids.M2 absent.
    atomic_write_json(pkg / 'child_run_ids.json', {'M3': 'M3-x', 'M4': 'M4-x'})
    m2_id = ensure_package_m2_run_id(pkg)
    assert m2_id == 'M2-SHARED-ed77fc1e-207ed187722f'
    child = json.loads((pkg / 'child_run_ids.json').read_text(encoding='utf-8'))
    assert child['M2'] == m2_id
    assert child['M3'] == 'M3-x'


def test_shared_package_audit_verifies_as_m3_parent(tmp_path: Path) -> None:
    from src.m3_config import M3Config
    from src.m3_parent import verify_accepted_m2_parent
    from src.s0_series import build_m3_config_for_package
    from tests.m3_helpers import make_synthetic_accepted_m2

    base_config, project = make_synthetic_accepted_m2(
        tmp_path, M3Config(require_cuda=False, j2_max=1),
    )
    persist = tmp_path / 'persist'
    run_root = Path(base_config.parent_report_path).parents[1]
    shared_id = 'M2-SHARED-test'
    dest = persist / 'runs' / shared_id
    dest.parent.mkdir(parents=True)
    run_root.rename(dest)

    report_path = dest / 'reports' / 'M2_report.json'
    report = json.loads(report_path.read_text(encoding='utf-8'))
    report['run_id'] = shared_id
    # Force package audit writer to discover checkpoint under dest.
    report.pop('checkpoint', None)
    atomic_write_json(report_path, report)
    manifest = json.loads((dest / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest['run_id'] = shared_id
    atomic_write_json(dest / 'run_manifest.json', manifest)
    state = json.loads(
        (dest / 'checkpoints' / 'ckpt_000014' / 'state.json').read_text(encoding='utf-8')
    )
    state['run_id'] = shared_id
    ckpt = dest / 'checkpoints' / 'ckpt_000014'
    atomic_write_json(ckpt / 'state.json', state)
    from src.common import sha256_file
    hashes = {
        path.relative_to(ckpt).as_posix(): sha256_file(path)
        for path in ckpt.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    atomic_write_json(ckpt / 'hashes.json', hashes)

    pkg = tmp_path / 'pkg'
    pkg.mkdir()
    atomic_write_json(pkg / 'child_run_ids.json', {'M2': shared_id})
    atomic_write_json(pkg / 'm3_config_overrides.json', {
        'j2_max': 1,
        'sector_count': 64,
        'operator_dimension': 729,
        'target_rank': 16,
    })
    atomic_write_json(pkg / 'm2_binding.json', {
        'schema_version': 2,
        'state': 'READY_SHARED',
        'structural_key': 's' * 64,
        'proof_key': 'p' * 64,
        'canonical_run_id': shared_id,
        'registry_record_sha256': 'r' * 64,
        'mode': 'REUSE_SHARED',
    })
    audit = write_package_m2_shared_audit(
        pkg,
        run_root=dest,
        structural_key='s' * 64,
        proof_key='p' * 64,
        registry_record_sha256='r' * 64,
    )
    assert audit['decision'] == 'ACCEPT_SHARED_M2_FOR_CANDIDATE_M3'
    assert audit['shared_m2'] is True

    cfg = build_m3_config_for_package(project, pkg, persist)
    assert Path(cfg.parent_audit_path).is_absolute()
    assert Path(cfg.parent_audit_path).name == 'm2_shared_parent.json'
    assert cfg.parent_run_id == shared_id
    evidence = verify_accepted_m2_parent(project, cfg)
    assert len(evidence.projector_tensors) == 64


def test_package_audits_do_not_clobber(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    run = persist / 'runs' / 'M2-shared'
    _fake_complete_m2(run)
    pkg_a = tmp_path / 'A'
    pkg_b = tmp_path / 'B'
    pkg_a.mkdir()
    pkg_b.mkdir()
    a = write_package_m2_shared_audit(
        pkg_a, run_root=run,
        structural_key='s' * 64, proof_key='p' * 64,
        registry_record_sha256='r',
    )
    b = write_package_m2_shared_audit(
        pkg_b, run_root=run,
        structural_key='s' * 64, proof_key='p' * 64,
        registry_record_sha256='r',
    )
    assert a['accepted_run_id'] == b['accepted_run_id']
    assert (pkg_a / 'audits' / 'm2_shared_parent.json').is_file()
    assert (pkg_b / 'audits' / 'm2_shared_parent.json').is_file()
    # Global audit path untouched.
    assert not (tmp_path / 'audit' / 'm2_accepted_parent.json').exists()
