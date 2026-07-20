"""Runtime estimators for admission control (upper-quantile style)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any


DEFAULT_RUNTIME = {
    'SCREENING': 8.0,
    'M2_RESOLVE': 5.0,
    'S0': 20.0,
    'INDEPENDENT_VERIFY': 15.0,
    'PACKAGE_AUDIT': 5.0,
    'ARCHIVE': 2.0,
}


@dataclass
class RuntimeEstimator:
    history: dict[str, list[float]] = field(default_factory=lambda: defaultdict(list))
    safety_factor: float = 1.5
    floor_sec: float = 1.0

    def observe(self, stage: str, runtime_sec: float) -> None:
        self.history[stage.upper()].append(float(runtime_sec))

    def upper_runtime_sec(self, stage: str, candidate: dict[str, Any] | None = None) -> float:
        key = stage.upper()
        samples = self.history.get(key) or []
        base = DEFAULT_RUNTIME.get(key, 10.0)
        if samples:
            ordered = sorted(samples)
            # 90th percentile or max for small n
            index = max(0, int(0.9 * (len(ordered) - 1)))
            base = ordered[index]
        # Mild rank scaling for screening / S0
        if candidate and key in {'SCREENING', 'S0', 'INDEPENDENT_VERIFY'}:
            scheme = candidate.get('scheme') or {}
            rank = int(scheme.get('target_rank') or 16)
            base *= max(1.0, rank / 16.0) ** 0.5
        return max(self.floor_sec, float(base) * self.safety_factor)

    def payload(self) -> dict[str, Any]:
        return {
            'safety_factor': self.safety_factor,
            'floor_sec': self.floor_sec,
            'history_counts': {k: len(v) for k, v in self.history.items()},
            'defaults': dict(DEFAULT_RUNTIME),
        }
