from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from .config import RunConfig

# Notebook 96 / end-to-end mode: disable 5h20/5h30 session wall-clock stops.
# Item-level checkpoints remain; fail-closed certification is unchanged.
_DISABLE_WALLCLOCK_ENV = 'VALIDATED_RG_DISABLE_SESSION_WALLCLOCK'


def session_wallclock_disabled() -> bool:
    return os.environ.get(_DISABLE_WALLCLOCK_ENV, '').strip() in {'1', 'true', 'TRUE', 'yes', 'YES'}


class SessionState(str, Enum):
    RUN = 'RUN'
    NO_LONG_TASK = 'NO_LONG_TASK'
    DRAIN = 'DRAIN'
    FINAL_SAVE = 'FINAL_SAVE'
    RETURN = 'RETURN'


@dataclass(slots=True)
class SessionGuard:
    config: RunConfig
    clock: Callable[[], float] = time.monotonic
    started_monotonic: float = field(init=False)
    last_checkpoint_monotonic: float = field(init=False)

    def __post_init__(self) -> None:
        now = self.clock()
        self.started_monotonic = now
        self.last_checkpoint_monotonic = now

    def elapsed_s(self) -> float:
        return max(0.0, self.clock() - self.started_monotonic)

    def remaining_s(self) -> float | None:
        """Seconds until hard return, or ``None`` when session wall-clock is off.

        Returning ``None`` (not ``inf``) keeps session summaries JSON-safe so
        ``sanitize_for_json`` does not falsely flag scheduling as numerical
        nonfinite / demote ``M3_COMPLETE``.
        """
        if session_wallclock_disabled():
            return None
        return max(0.0, self.config.hard_return_s - self.elapsed_s())

    def state(self) -> SessionState:
        if session_wallclock_disabled():
            return SessionState.RUN
        elapsed = self.elapsed_s()
        if elapsed >= self.config.hard_return_s:
            return SessionState.RETURN
        if elapsed >= self.config.final_save_after_s:
            return SessionState.FINAL_SAVE
        if elapsed >= self.config.drain_after_s:
            return SessionState.DRAIN
        if elapsed >= self.config.no_long_task_after_s:
            return SessionState.NO_LONG_TASK
        return SessionState.RUN

    def checkpoint_due(self) -> bool:
        return self.clock() - self.last_checkpoint_monotonic >= self.config.checkpoint_interval_s

    def mark_checkpoint(self) -> None:
        self.last_checkpoint_monotonic = self.clock()

    def may_start(self, predicted_s: float) -> bool:
        if not math.isfinite(predicted_s) or predicted_s <= 0.0:
            return False
        if session_wallclock_disabled():
            if predicted_s > self.config.max_work_item_s:
                return False
            return True
        state = self.state()
        if state not in {SessionState.RUN, SessionState.NO_LONG_TASK}:
            return False
        if predicted_s > self.config.max_work_item_s:
            return False
        if state is SessionState.NO_LONG_TASK and predicted_s > self.config.short_task_limit_s:
            return False
        remaining = self.remaining_s()
        if remaining is None:
            # Wallclock-disabled path returns earlier; treat as unlimited budget.
            return True
        return remaining >= 1.3 * predicted_s + self.config.checkpoint_reserve_s
