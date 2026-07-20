from __future__ import annotations

from pathlib import Path

from src.campaign_b.driver import run_campaign_b
from src.campaign_b.schemas import TERMINAL_NEED_M2, TERMINAL_TIME

from .conftest import write_tiny_config


def test_six_hour_cutoff_tiny_budget(tmp_path: Path) -> None:
    # Soft budget by default: tiny limit must not block; resume is the safety net.
    cfg = write_tiny_config(tmp_path, hard_limit_sec=0.05)
    text = cfg.read_text(encoding='utf-8')
    cfg.write_text(
        text.replace('enforce_wall_clock: false', 'enforce_wall_clock: true'),
        encoding='utf-8',
    )
    summary = run_campaign_b(cfg)
    assert summary['terminal_reason'] in {
        TERMINAL_TIME,
        TERMINAL_NEED_M2,
        'B_SCREENING_EXHAUSTED',
        'B_FAIL_CLOSED',
        'B_Q_LT_1_LINEAGE_READY',
    }
    assert summary['certification_status'] == 'NOT_CERTIFIED'


def test_driver_stops_for_missing_m2(tmp_path: Path) -> None:
    cfg = write_tiny_config(tmp_path, hard_limit_sec=120)
    summary = run_campaign_b(cfg)
    # Tiny high-rank space screens q<1 then needs canonical M2.
    assert summary['terminal_reason'] in {TERMINAL_NEED_M2, TERMINAL_TIME}
    assert summary['selected_count'] == 0
