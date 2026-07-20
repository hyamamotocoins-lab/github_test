from __future__ import annotations

from pathlib import Path

from src.campaign_b.lineage import resolve_shared_m2
from src.m2_shared_registry import BINDING_NEED, STATE_COMPLETE
from src.common import atomic_write_json


def test_shared_m2_missing_stops() -> None:
    candidate = {
        'candidate_id': 'B-x',
        'j2': 2,
        'execution_mode': 'staged',
        'structural_key': 'missing-sk',
        'proof_key': 'missing-pk',
        'certification_status': 'NOT_CERTIFIED',
        'claim_scope': 'SCREENING_ONLY',
    }
    binding = resolve_shared_m2(
        candidate=candidate,
        persistent_root=Path('/tmp/campaign_b_no_registry'),
        source_tree_hash='abc',
        allow_generate_canonical=False,
    )
    assert binding['status'] == BINDING_NEED


def test_shared_m2_ready(tmp_path: Path) -> None:
    sk, pk = 'sk-test', 'pk-test'
    entry = (
        tmp_path / 'shared_m2_registry' / sk / 'proofs' / pk / 'canonical_run.json'
    )
    entry.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(entry, {
        'registry_state': STATE_COMPLETE,
        'canonical_run_id': 'M2-SHARED-test',
        'canonical_package_dir': str(tmp_path / 'pkg'),
        'source_hash': 'abc',
        'registry_record_sha256': 'deadbeef',
    })
    candidate = {
        'candidate_id': 'B-x',
        'j2': 2,
        'execution_mode': 'staged',
        'structural_key': sk,
        'proof_key': pk,
        'certification_status': 'NOT_CERTIFIED',
        'claim_scope': 'SCREENING_ONLY',
    }
    binding = resolve_shared_m2(
        candidate=candidate,
        persistent_root=tmp_path,
        source_tree_hash='abc',
        allow_generate_canonical=False,
    )
    assert binding['status'] == 'READY_SHARED'
    assert binding['reuse_class'] == 'EXACT_SOURCE_MATCH'


def test_source_drift_reuse_class(tmp_path: Path) -> None:
    sk, pk = 'sk-test', 'pk-test'
    entry = (
        tmp_path / 'shared_m2_registry' / sk / 'proofs' / pk / 'canonical_run.json'
    )
    entry.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(entry, {
        'registry_state': STATE_COMPLETE,
        'canonical_run_id': 'M2-SHARED-test',
        'canonical_package_dir': str(tmp_path / 'pkg'),
        'source_hash': 'old-hash',
        'registry_record_sha256': 'deadbeef',
    })
    candidate = {
        'candidate_id': 'B-x',
        'j2': 2,
        'execution_mode': 'staged',
        'structural_key': sk,
        'proof_key': pk,
        'certification_status': 'NOT_CERTIFIED',
        'claim_scope': 'SCREENING_ONLY',
    }
    binding = resolve_shared_m2(
        candidate=candidate,
        persistent_root=tmp_path,
        source_tree_hash='new-hash',
        allow_generate_canonical=False,
    )
    assert binding['reuse_class'] == 'AUDITED_SOURCE_DRIFT_REUSE'
