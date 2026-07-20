from __future__ import annotations

from pathlib import Path

from src.campaign_b.config import load_campaign_b_config, validate_search_space
from src.campaign_b.schemas import CERTIFICATION_STATUS

from .conftest import write_tiny_config


def test_load_tiny_config(tmp_path: Path) -> None:
    cfg_path = write_tiny_config(tmp_path)
    cfg = load_campaign_b_config(cfg_path)
    assert cfg.execution_policy()['certification_status'] == CERTIFICATION_STATUS
    assert cfg.allow_generate_canonical_m2 is False
    assert cfg.search_space['staging']['forbid_j2_1'] is True
    validate_search_space(cfg.search_space)


def test_reject_j2_1_in_space(tmp_path: Path) -> None:
    space = {
        'campaign': 'B_S2',
        'staging': {'j2_values': [1, 2], 'forbid_j2_1': True},
        'rank': {'values': [16]},
    }
    try:
        validate_search_space(space)
        raised = False
    except Exception:
        raised = True
    assert raised
