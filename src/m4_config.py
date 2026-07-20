from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class M4Config:
    schema_version: int = 1
    project_name: str = 'validated_4d_su2_rg'
    milestone: str = 'M4'
    parent_milestone: str = 'M3'
    parent_run_id: str = 'M3-20260720T013551Z-ae995e91e861'
    parent_checkpoint: str = 'ckpt_000014'
    parent_checkpoint_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M3-20260720T013551Z-ae995e91e861/checkpoints/ckpt_000014'
    )
    parent_report_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M3-20260720T013551Z-ae995e91e861/reports/M3_report.json'
    )
    parent_acceptance_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M3-20260720T013551Z-ae995e91e861/reports/M3_acceptance.json'
    )
    parent_audit_path: str = 'audit/m3_accepted_parent.json'
    operator_dimension: int = 729
    projected_rank: int = 16
    source_channel_count: int = 5
    finite_difference_steps: tuple[float, ...] = (1e-3, 5e-4, 2.5e-4)
    finite_difference_relative_tolerance: float = 2e-6
    symmetry_tolerance: float = 0.0
    require_cuda: bool = True
    dtype: str = 'float64'
    normal_memory_headroom: float = 0.25
    checkpoint_memory_headroom: float = 0.35
    checkpoint_interval_s: float = 15.0 * 60.0
    max_work_item_s: float = 20.0 * 60.0
    short_task_limit_s: float = 5.0 * 60.0
    checkpoint_reserve_s: float = 10.0 * 60.0
    initial_no_long_task_after_s: float = 1.5 * 3600.0
    initial_drain_after_s: float = 1.0 * 3600.0 + 45.0 * 60.0
    initial_final_save_after_s: float = 1.0 * 3600.0 + 50.0 * 60.0
    initial_hard_return_s: float = 2.0 * 3600.0
    no_long_task_after_s: float = 5.0 * 3600.0
    drain_after_s: float = 5.0 * 3600.0 + 15.0 * 60.0
    final_save_after_s: float = 5.0 * 3600.0 + 20.0 * 60.0
    hard_return_s: float = 5.0 * 3600.0 + 30.0 * 60.0
    tensor_shard_bytes: int = 64 * 1024 * 1024
    max_item_attempts: int = 3
    milestone_status: str = 'BLOCKED_MATH'
    certification_status: str = 'NOT_CERTIFIED'

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.project_name != 'validated_4d_su2_rg'
            or self.milestone != 'M4'
            or self.parent_milestone != 'M3'
        ):
            raise ValueError('Unsupported M4 project/config schema.')
        if (
            self.milestone_status != 'BLOCKED_MATH'
            or self.certification_status != 'NOT_CERTIFIED'
        ):
            raise ValueError('M4 must remain fail-closed.')
        if Path(self.parent_checkpoint_path).name != self.parent_checkpoint:
            raise ValueError('M4 parent checkpoint path/name mismatch.')
        if Path(self.parent_report_path).name != 'M3_report.json':
            raise ValueError('M4 parent report path is invalid.')
        if Path(self.parent_acceptance_path).name != 'M3_acceptance.json':
            raise ValueError('M4 parent acceptance path is invalid.')
        if Path(self.parent_audit_path).as_posix() != 'audit/m3_accepted_parent.json':
            raise ValueError('M4 audit path is fixed.')
        integers = (
            self.operator_dimension, self.projected_rank,
            self.source_channel_count, self.tensor_shard_bytes,
            self.max_item_attempts,
        )
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integers
        ):
            raise ValueError('M4 discrete configuration fields must be integers.')
        if self.operator_dimension != 729 or self.source_channel_count != 5:
            raise ValueError('M4 is fixed to the accepted M3 finite core.')
        # projected_rank must be a perfect square so regroup_matrix (leg^4) works.
        leg = int(round(self.projected_rank ** 0.5))
        if (
            not 1 <= self.projected_rank < self.operator_dimension
            or leg * leg != self.projected_rank
        ):
            raise ValueError(
                'M4 projected_rank must be a perfect square in '
                f'[1, {self.operator_dimension}).'
            )
        if self.dtype != 'float64' or not isinstance(self.require_cuda, bool):
            raise ValueError('M4 requires an explicit FP64 backend policy.')
        if (
            len(self.finite_difference_steps) < 3
            or any(
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value <= 0.0
                for value in self.finite_difference_steps
            )
            or any(
                later >= earlier for earlier, later in zip(
                    self.finite_difference_steps,
                    self.finite_difference_steps[1:],
                )
            )
        ):
            raise ValueError('M4 finite-difference steps must strictly decrease.')
        for value, label in (
            (self.finite_difference_relative_tolerance, 'FD tolerance'),
            (self.symmetry_tolerance, 'symmetry tolerance'),
            (self.normal_memory_headroom, 'normal memory headroom'),
            (self.checkpoint_memory_headroom, 'checkpoint memory headroom'),
        ):
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise ValueError(f'M4 {label} is invalid.')
        if (
            self.normal_memory_headroom < 0.25
            or self.checkpoint_memory_headroom < 0.35
            or self.normal_memory_headroom >= 1.0
            or self.checkpoint_memory_headroom >= 1.0
        ):
            raise ValueError('M4 GPU headroom policy is invalid.')
        schedule = (
            0.0, self.no_long_task_after_s, self.drain_after_s,
            self.final_save_after_s, self.hard_return_s,
        )
        if tuple(sorted(schedule)) != schedule or len(set(schedule)) != len(schedule):
            raise ValueError('M4 session thresholds must strictly increase.')
        initial_schedule = (
            0.0, self.initial_no_long_task_after_s,
            self.initial_drain_after_s, self.initial_final_save_after_s,
            self.initial_hard_return_s,
        )
        if (
            tuple(sorted(initial_schedule)) != initial_schedule
            or len(set(initial_schedule)) != len(initial_schedule)
            or self.initial_hard_return_s > 2.0 * 3600.0
        ):
            raise ValueError('M4 initial two-hour session policy is invalid.')
        durations = schedule[1:] + (
            *initial_schedule[1:],
            self.checkpoint_interval_s, self.max_work_item_s,
            self.short_task_limit_s, self.checkpoint_reserve_s,
        )
        if any(
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value <= 0.0
            for value in durations
        ):
            raise ValueError('M4 duration fields must be positive finite reals.')
        if self.checkpoint_interval_s > 15 * 60:
            raise ValueError('M4 checkpoint interval exceeds 15 minutes.')
        if self.final_save_after_s > 5 * 3600 + 20 * 60:
            raise ValueError('M4 final checkpoint begins too late.')
        if self.hard_return_s > 5.5 * 3600:
            raise ValueError('M4 session return exceeds the safety margin.')

    def canonical_payload(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(',', ':'),
            allow_nan=False,
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()
