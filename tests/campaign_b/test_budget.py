from __future__ import annotations

import time

from src.campaign_b.budget import BudgetManager


def test_budget_soft_by_default() -> None:
    budget = BudgetManager(
        hard_limit_sec=1,
        admission_close_sec=0.1,
        finalization_start_sec=0.2,
        emergency_flush_sec=0.05,
        enforce_wall_clock=False,
    )
    budget.start()
    time.sleep(0.3)
    assert budget.may_start('SCREENING', 1.0)
    assert not budget.admission_closed()
    assert not budget.must_finalize()


def test_budget_admission_and_finalize_when_enforced() -> None:
    budget = BudgetManager(
        hard_limit_sec=10,
        admission_close_sec=3,
        finalization_start_sec=6,
        emergency_flush_sec=1,
        enforce_wall_clock=True,
    )
    budget.start()
    assert budget.may_start('SCREENING', 1.0)
    time.sleep(3.2)
    assert budget.admission_closed()
    assert not budget.may_start('SCREENING', 1.0)
    time.sleep(3.0)
    assert budget.must_finalize()
