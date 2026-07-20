"""M5 run configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .m5_status import M4_PARENT_RUN_ID_FROZEN, M5_RUN_ID_FROZEN


@dataclass(frozen=True, slots=True)
class M5Config:
    parent_m4_run_id: str = M4_PARENT_RUN_ID_FROZEN
    run_id: str = M5_RUN_ID_FROZEN
    cutoff: int = 2
    bond_dimension: int = 16
    weight_m: str = '0'
    precision_bits: int = 256
    arithmetic_backend: str = 'rational_fraction'
    rounding_policy: str = 'outward'
    metric_unit: str = 'lattice'
    source_speed_unit: str = 'lattice'
    mode: str = 'paperspace'
    # paperspace | cpu_fixture_cert | cpu_fixture_not_certified

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def default_m5_config(**overrides: Any) -> M5Config:
    base = M5Config()
    if not overrides:
        return base
    payload = asdict(base)
    payload.update(overrides)
    return M5Config(**payload)
