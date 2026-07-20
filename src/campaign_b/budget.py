"""Wall-clock budget for Campaign B (optional soft limit; resume-first)."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .errors import TimeBudgetClosed

HARD_LIMIT_SEC = 6 * 60 * 60
ADMISSION_CLOSE_SEC = 5 * 60 * 60 + 25 * 60
FINALIZATION_START_SEC = 5 * 60 * 60 + 35 * 60
EMERGENCY_FLUSH_SEC = 5 * 60


@dataclass
class BudgetManager:
    """Optional wall-clock budget.

    Default policy is resume-first: ``enforce_wall_clock=False`` means the
    driver does not stop mid-campaign for the six-hour window. Persistence and
    ``resume_campaign_run_id`` are the safety net if a process dies.
    """

    hard_limit_sec: float = HARD_LIMIT_SEC
    admission_close_sec: float = ADMISSION_CLOSE_SEC
    finalization_start_sec: float = FINALIZATION_START_SEC
    emergency_flush_sec: float = EMERGENCY_FLUSH_SEC
    enforce_wall_clock: bool = False
    started_at: float | None = None
    deadline_at: float | None = None

    def start(self, *, resume_deadline_at: float | None = None) -> None:
        now = time.monotonic()
        self.started_at = now
        if resume_deadline_at is not None:
            self.deadline_at = resume_deadline_at
        else:
            self.deadline_at = now + float(self.hard_limit_sec)

    def elapsed_sec(self) -> float:
        if self.started_at is None:
            return 0.0
        return max(0.0, time.monotonic() - self.started_at)

    def remaining_sec(self) -> float:
        if not self.enforce_wall_clock:
            return float('inf')
        if self.deadline_at is None:
            return float(self.hard_limit_sec)
        return max(0.0, self.deadline_at - time.monotonic())

    def must_finalize(self) -> bool:
        if not self.enforce_wall_clock:
            return False
        if self.started_at is None:
            return False
        return self.elapsed_sec() >= float(self.finalization_start_sec) or (
            self.remaining_sec() <= float(self.emergency_flush_sec)
        )

    def admission_closed(self) -> bool:
        if not self.enforce_wall_clock:
            return False
        if self.started_at is None:
            return False
        return self.elapsed_sec() >= float(self.admission_close_sec) or self.must_finalize()

    def required_finalize_reserve_sec(self, stage: str) -> float:
        if not self.enforce_wall_clock:
            return 0.0
        heavy = stage.upper() in {'SCREENING', 'S0', 'M2_RESOLVE', 'INDEPENDENT_VERIFY'}
        fraction = 0.20 if heavy else 0.05
        scaled_cap = max(1.0, float(self.hard_limit_sec) * fraction)
        if heavy:
            desired = max(float(self.emergency_flush_sec), 120.0)
        else:
            desired = max(float(self.emergency_flush_sec) * 0.1, 30.0)
        return min(desired, scaled_cap)

    def may_start(self, stage: str, predicted_runtime_sec: float) -> bool:
        if not self.enforce_wall_clock:
            return True
        if self.must_finalize():
            return False
        if self.admission_closed() and stage.upper() in {
            'SCREENING', 'S0', 'M2_RESOLVE', 'INDEPENDENT_VERIFY',
        }:
            return False
        remaining = self.remaining_sec()
        reserve = self.required_finalize_reserve_sec(stage)
        return remaining >= float(predicted_runtime_sec) + reserve

    def assert_may_start(self, stage: str, predicted_runtime_sec: float) -> None:
        if not self.may_start(stage, predicted_runtime_sec):
            raise TimeBudgetClosed(
                f'cannot start {stage}: remaining={self.remaining_sec():.1f}s '
                f'predicted={predicted_runtime_sec:.1f}s'
            )

    def snapshot(self) -> dict[str, float | bool | None]:
        remaining = self.remaining_sec()
        return {
            'enforce_wall_clock': self.enforce_wall_clock,
            'hard_limit_sec': float(self.hard_limit_sec),
            'admission_close_sec': float(self.admission_close_sec),
            'finalization_start_sec': float(self.finalization_start_sec),
            'emergency_flush_sec': float(self.emergency_flush_sec),
            'elapsed_sec': self.elapsed_sec(),
            'remaining_sec': (
                None if remaining == float('inf') else float(remaining)
            ),
            'admission_closed': self.admission_closed(),
            'must_finalize': self.must_finalize(),
            'deadline_at_monotonic': self.deadline_at,
        }
