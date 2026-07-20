from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.campaign_b.canonical_m2 import ensure_canonical_shared_m2
from src.m2_shared_registry import BINDING_READY, STATE_COMPLETE


def test_ensure_reuses_complete_registry(tmp_path: Path) -> None:
    record = {
        'registry_state': STATE_COMPLETE,
        'canonical_run_id': 'M2-SHARED-test',
        'canonical_package_dir': str(tmp_path / 'pkg'),
        'source_hash': 'abc',
        'registry_record_sha256': 'deadbeef',
    }
    with patch(
        'src.campaign_b.canonical_m2.keys_from_project',
        return_value={
            'structural_key': 'sk',
            'proof_key': 'pk',
            'shared_run_id': 'M2-SHARED-test',
        },
    ), patch(
        'src.campaign_b.canonical_m2.lookup_shared_m2_reusable',
        return_value=(record, 'exact'),
    ):
        binding = ensure_canonical_shared_m2(
            persistent_root=tmp_path,
            project_root=tmp_path,
            source_tree_hash='abc',
            j2_max=2,
        )
    assert binding['status'] == BINDING_READY
    assert binding['canonical_run_id'] == 'M2-SHARED-test'
    assert binding.get('generated') is not True
