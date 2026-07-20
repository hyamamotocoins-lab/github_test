from __future__ import annotations

from pathlib import Path

from src.campaign_b.resume_pointer import (
    RESUME_ENV_KEY,
    read_resume_id,
    write_resume_pointer,
)
from src.common import read_json


def test_resume_pointer_roundtrip(tmp_path: Path) -> None:
    payload = write_resume_pointer(
        tmp_path,
        campaign_run_id='M7-test-b-resume',
        terminal_reason='B_BLOCKED_NEED_CANONICAL_M2',
        campaign_root=tmp_path / 'campaign_b' / 'M7-test-b-resume',
    )
    assert payload[RESUME_ENV_KEY] == 'M7-test-b-resume'
    assert read_resume_id(tmp_path) == 'M7-test-b-resume'
    stored = read_json(tmp_path / 'campaign_b' / 'LATEST_CAMPAIGN_B_RESUME.json')
    assert stored['resume_campaign_run_id'] == 'M7-test-b-resume'
    export = (tmp_path / 'campaign_b' / 'export_VALIDATED_RG_M7B_RESUME_ID.sh').read_text()
    assert 'export VALIDATED_RG_M7B_RESUME_ID=M7-test-b-resume' in export
    plain = (tmp_path / 'campaign_b' / 'VALIDATED_RG_M7B_RESUME_ID.txt').read_text().strip()
    assert plain == 'M7-test-b-resume'
