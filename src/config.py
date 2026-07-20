from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any


class ConfigError(ValueError):
    '''Raised when an immutable run configuration violates M0 safety rules.'''


@dataclass(frozen=True, slots=True)
class RunConfig:
    schema_version: int = 1
    project_name: str = 'validated_4d_su2_rg'
    milestone: str = 'M0'
    seed: int = 20260719
    checkpoint_interval_s: float = 15.0 * 60.0
    max_work_item_s: float = 20.0 * 60.0
    short_task_limit_s: float = 5.0 * 60.0
    checkpoint_reserve_s: float = 10.0 * 60.0
    no_long_task_after_s: float = 5.0 * 3600.0
    drain_after_s: float = 5.0 * 3600.0 + 15.0 * 60.0
    final_save_after_s: float = 5.0 * 3600.0 + 20.0 * 60.0
    hard_return_s: float = 5.0 * 3600.0 + 30.0 * 60.0
    tensor_shard_bytes: int = 256 * 1024 * 1024
    dummy_items: int = 6
    dummy_size: int = 128
    dummy_steps: int = 2
    dummy_predicted_s: float = 30.0
    max_item_attempts: int = 3
    prefer_cuda: bool = True
    certification_status: str = 'NOT_CERTIFIED'

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.project_name != 'validated_4d_su2_rg':
            raise ConfigError('Unsupported M0 schema or project identity.')
        if self.milestone != 'M0':
            raise ConfigError('This notebook implements M0 only.')
        if self.certification_status != 'NOT_CERTIFIED':
            raise ConfigError('M0 certification_status is immutable and must be NOT_CERTIFIED.')
        if not isinstance(self.prefer_cuda, bool):
            raise ConfigError('prefer_cuda must be a boolean execution policy.')
        integer_fields = (
            self.seed, self.tensor_shard_bytes, self.dummy_items, self.dummy_size,
            self.dummy_steps, self.max_item_attempts,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_fields):
            raise ConfigError('Seed, shard, dummy, and attempt fields must be integers.')
        schedule = (
            0.0,
            self.no_long_task_after_s,
            self.drain_after_s,
            self.final_save_after_s,
            self.hard_return_s,
        )
        duration_fields = schedule[1:] + (
            self.checkpoint_interval_s, self.max_work_item_s, self.short_task_limit_s,
            self.checkpoint_reserve_s, self.dummy_predicted_s,
        )
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) for value in duration_fields):
            raise ConfigError('Session and prediction durations must be real numbers.')
        if any(not math.isfinite(value) for value in schedule):
            raise ConfigError('Session schedule must be finite.')
        if tuple(sorted(schedule)) != schedule or len(set(schedule)) != len(schedule):
            raise ConfigError('Session thresholds must be strictly increasing.')
        if self.hard_return_s > 5.5 * 3600.0:
            raise ConfigError('Hard return may not exceed 5 h 30 min.')
        if self.final_save_after_s > 5.0 * 3600.0 + 20.0 * 60.0:
            raise ConfigError('Final checkpointing must begin no later than 5 h 20 min.')
        if not 0.0 < self.max_work_item_s <= 20.0 * 60.0:
            raise ConfigError('A work item may not exceed 20 minutes.')
        if self.checkpoint_interval_s <= 0.0 or self.checkpoint_interval_s > 15.0 * 60.0:
            raise ConfigError('Checkpoint interval may not exceed 15 minutes.')
        positive_durations = (
            self.max_work_item_s, self.short_task_limit_s, self.checkpoint_reserve_s,
            self.checkpoint_interval_s, self.dummy_predicted_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive_durations):
            raise ConfigError('Durations must be positive and finite.')
        if self.short_task_limit_s > self.max_work_item_s:
            raise ConfigError('Short-task limit may not exceed the work-item limit.')
        if self.dummy_predicted_s > self.max_work_item_s:
            raise ConfigError('Dummy prediction may not exceed the work-item limit.')
        if self.tensor_shard_bytes <= 0 or self.dummy_items < 0:
            raise ConfigError('Shard size and dummy item count are invalid.')
        if self.dummy_size <= 0 or self.dummy_steps <= 0 or self.max_item_attempts <= 0:
            raise ConfigError('Dummy dimensions, steps, and attempt count must be positive.')
        if self.dummy_size > 512 or self.dummy_steps > 8 or self.dummy_items > 10_000:
            raise ConfigError('M0 dummy workload exceeds the bounded safety caps.')
        if not 0 <= self.seed < 2**32:
            raise ConfigError('Seed must be accepted by NumPy RandomState (0 <= seed < 2**32).')

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()
