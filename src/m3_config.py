from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class M3Config:
    schema_version: int = 1
    project_name: str = 'validated_4d_su2_rg'
    milestone: str = 'M3'
    parent_milestone: str = 'M2'
    parent_run_id: str = 'M2-20260720T005145Z-dd3e385d0a61'
    parent_checkpoint: str = 'ckpt_000014'
    parent_checkpoint_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M2-20260720T005145Z-dd3e385d0a61/checkpoints/ckpt_000014'
    )
    parent_report_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M2-20260720T005145Z-dd3e385d0a61/reports/M2_report.json'
    )
    parent_acceptance_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M2-20260720T005145Z-dd3e385d0a61/reports/M2_acceptance.json'
    )
    parent_audit_path: str = 'audit/m2_accepted_parent.json'
    j2_max: int = 1
    sector_count: int = 64
    operator_dimension: int = 729
    target_rank: int = 16
    oversampling: int = 16
    power_iterations: int = 2
    seed: int = 20260720
    dtype: str = 'float64'
    require_cuda: bool = True
    initial_sector_shard_size: int = 16
    min_sector_shard_size: int = 1
    max_oom_retries: int = 3
    normal_memory_headroom: float = 0.25
    checkpoint_memory_headroom: float = 0.35
    checkpoint_interval_s: float = 15.0 * 60.0
    max_work_item_s: float = 20.0 * 60.0
    short_task_limit_s: float = 5.0 * 60.0
    checkpoint_reserve_s: float = 10.0 * 60.0
    no_long_task_after_s: float = 5.0 * 3600.0
    drain_after_s: float = 5.0 * 3600.0 + 15.0 * 60.0
    final_save_after_s: float = 5.0 * 3600.0 + 20.0 * 60.0
    hard_return_s: float = 5.0 * 3600.0 + 30.0 * 60.0
    tensor_shard_bytes: int = 64 * 1024 * 1024
    max_item_attempts: int = 3
    exploration_status: str = 'EXPLORATORY'
    certification_status: str = 'NOT_CERTIFIED'

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.project_name != 'validated_4d_su2_rg'
            or self.milestone != 'M3'
        ):
            raise ValueError('Unsupported M3 project/config schema.')
        if (
            self.parent_milestone != 'M2'
            or self.exploration_status != 'EXPLORATORY'
            or self.certification_status != 'NOT_CERTIFIED'
        ):
            raise ValueError('M3 parent or status invariant failed.')
        if not self.parent_run_id:
            raise ValueError('Accepted M2 parent identity is missing.')
        if (
            not isinstance(self.parent_checkpoint, str)
            or not self.parent_checkpoint.startswith('ckpt_')
        ):
            raise ValueError('Accepted M2 parent checkpoint name is invalid.')
        if Path(self.parent_checkpoint_path).name != self.parent_checkpoint:
            raise ValueError('M3 parent checkpoint path/name mismatch.')
        if Path(self.parent_report_path).name != 'M2_report.json':
            raise ValueError('M3 parent report path is invalid.')
        if Path(self.parent_acceptance_path).name != 'M2_acceptance.json':
            raise ValueError('M3 parent acceptance path is invalid.')
        audit_posix = Path(self.parent_audit_path).as_posix()
        audit_name = Path(self.parent_audit_path).name
        # Global project audit, or absolute package-local shared M2 audit.
        if audit_posix != 'audit/m2_accepted_parent.json' and not (
            Path(self.parent_audit_path).is_absolute()
            and audit_name == 'm2_shared_parent.json'
        ):
            raise ValueError(
                'M3 audit path must be audit/m2_accepted_parent.json or an '
                'absolute path ending in audits/m2_shared_parent.json.'
            )
        integers = (
            self.j2_max, self.sector_count, self.operator_dimension,
            self.target_rank, self.oversampling, self.power_iterations,
            self.seed, self.initial_sector_shard_size,
            self.min_sector_shard_size, self.max_oom_retries,
            self.tensor_shard_bytes, self.max_item_attempts,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integers
        ):
            raise ValueError('M3 discrete configuration fields must be integers.')
        from .cutoff_dims import operator_dimension as expected_operator_dimension
        from .cutoff_dims import sector_count as expected_sector_count

        if not 1 <= self.j2_max <= 4:
            raise ValueError('M3 j2_max must lie in [1, 4].')
        if (
            self.sector_count != expected_sector_count(self.j2_max)
            or self.operator_dimension != expected_operator_dimension(self.j2_max)
        ):
            raise ValueError(
                'M3 sector_count/operator_dimension must match derived '
                f'cutoff dims for j2_max={self.j2_max}.'
            )
        if not 1 <= self.target_rank < self.operator_dimension:
            raise ValueError('M3 target rank is invalid.')
        if self.oversampling < 1 or self.power_iterations < 0:
            raise ValueError('M3 RSVD parameters are invalid.')
        if not 0 <= self.seed < 2**32:
            raise ValueError('M3 seed must satisfy 0 <= seed < 2**32.')
        if self.dtype != 'float64':
            raise ValueError('M3 serious exploratory runs require FP64.')
        if not isinstance(self.require_cuda, bool):
            raise ValueError('M3 require_cuda must be a bool.')
        if (
            self.min_sector_shard_size < 1
            or self.initial_sector_shard_size < self.min_sector_shard_size
            or self.initial_sector_shard_size > self.sector_count
            or self.max_oom_retries != 3
        ):
            raise ValueError('M3 sharding/OOM policy is invalid.')
        for value, label in (
            (self.normal_memory_headroom, 'normal'),
            (self.checkpoint_memory_headroom, 'checkpoint'),
        ):
            if (
                not isinstance(value, (int, float)) or isinstance(value, bool)
                or not math.isfinite(value) or not 0.0 < value < 1.0
            ):
                raise ValueError(f'M3 {label} memory headroom is invalid.')
        if self.normal_memory_headroom < 0.25:
            raise ValueError('M3 normal GPU headroom must be at least 25%.')
        if self.checkpoint_memory_headroom < 0.35:
            raise ValueError('M3 checkpoint GPU headroom must be at least 35%.')
        schedule = (
            0.0, self.no_long_task_after_s, self.drain_after_s,
            self.final_save_after_s, self.hard_return_s,
        )
        if tuple(sorted(schedule)) != schedule or len(set(schedule)) != len(schedule):
            raise ValueError('M3 session thresholds must be strictly increasing.')
        durations = schedule[1:] + (
            self.checkpoint_interval_s, self.max_work_item_s,
            self.short_task_limit_s, self.checkpoint_reserve_s,
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0
            for value in durations
        ):
            raise ValueError('M3 duration fields must be positive finite reals.')
        if self.checkpoint_interval_s > 15 * 60 or self.max_work_item_s > 20 * 60:
            raise ValueError('M3 checkpoint/work-item limits exceed the roadmap.')
        if self.final_save_after_s > 5 * 3600 + 20 * 60:
            raise ValueError('M3 final checkpoint begins too late.')
        if self.hard_return_s > 5.5 * 3600:
            raise ValueError('M3 session return exceeds the six-hour safety margin.')

    def canonical_payload(self) -> dict[str, Any]:
        return asdict(self)

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(',', ':'),
            allow_nan=False,
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()
