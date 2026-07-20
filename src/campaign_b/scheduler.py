"""Admission / scheduling helpers for Campaign B."""

from __future__ import annotations

from typing import Any

from .budget import BudgetManager
from .estimators import RuntimeEstimator


def may_start(
    stage: str,
    candidate: dict[str, Any],
    budget: BudgetManager,
    estimator: RuntimeEstimator,
) -> bool:
    predicted = estimator.upper_runtime_sec(stage, candidate)
    return budget.may_start(stage, predicted)
