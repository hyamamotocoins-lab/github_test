"""M6 run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .m6_status import M5_PARENT_RUN_ID_FROZEN, M6_RUN_ID_FROZEN


@dataclass(frozen=True, slots=True)
class M6Config:
    parent_m5_run_id: str = M5_PARENT_RUN_ID_FROZEN
    run_id: str = M6_RUN_ID_FROZEN
    num_steps: int = 3
    j2_max: int = 1
    bond_dimension: int = 16
    weight_m: str = '0'
    precision_bits: int = 256
    arithmetic_backend: str = 'rational_fraction'
    rounding_policy: str = 'outward'
    metric_unit: str = 'lattice'
    source_speed_unit: str = 'lattice'
    norm: str = 'frobenius'
    mode: str = 'paperspace'
    # paperspace | cpu_fixture_cert | cpu_fixture_not_certified

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def default_m6_config(**overrides: Any) -> M6Config:
    base = M6Config()
    if not overrides:
        return base
    payload = asdict(base)
    payload.update(overrides)
    return M6Config(**payload)
