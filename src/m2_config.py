from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class M2Config:
    schema_version: int = 1
    project_name: str = 'validated_4d_su2_rg'
    milestone: str = 'M2'
    parent_milestone: str = 'M1'
    parent_run_id: str = 'M1-20260719T235423Z-a7cacde2ead9'
    parent_checkpoint: str = 'ckpt_000014'
    parent_checkpoint_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M1-20260719T235423Z-a7cacde2ead9/checkpoints/ckpt_000014'
    )
    parent_report_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M1-20260719T235423Z-a7cacde2ead9/reports/M1_report.json'
    )
    parent_acceptance_path: str = (
        '/storage/validated_4d_su2_rg/runs/'
        'M1-20260719T235423Z-a7cacde2ead9/reports/M1_acceptance.json'
    )
    parent_audit_path: str = 'audit/m1_accepted_parent.json'
    j2_max: int = 1
    leg_count: int = 6
    sector_batch_size: int = 0  # 0 → all sectors in one item (j2_max=1 default)
    orientations: tuple[int, ...] = (1, -1, 1, -1, 1, -1)
    seed: int = 20260720
    exact_decimal_digits: int = 80
    checkpoint_interval_s: float = 15.0 * 60.0
    max_work_item_s: float = 20.0 * 60.0
    short_task_limit_s: float = 5.0 * 60.0
    checkpoint_reserve_s: float = 10.0 * 60.0
    no_long_task_after_s: float = 5.0 * 3600.0
    drain_after_s: float = 5.0 * 3600.0 + 15.0 * 60.0
    final_save_after_s: float = 5.0 * 3600.0 + 20.0 * 60.0
    hard_return_s: float = 5.0 * 3600.0 + 30.0 * 60.0
    tensor_shard_bytes: int = 16 * 1024 * 1024
    max_item_attempts: int = 3
    certification_status: str = 'NOT_CERTIFIED'

    def __post_init__(self) -> None:
        if (
            self.schema_version != 1
            or self.project_name != 'validated_4d_su2_rg'
            or self.milestone != 'M2'
        ):
            raise ValueError('Unsupported M2 project/config schema.')
        if self.parent_milestone != 'M1' or self.certification_status != 'NOT_CERTIFIED':
            raise ValueError('M2 parent/status invariant failed.')
        if not self.parent_run_id or self.parent_checkpoint != 'ckpt_000014':
            raise ValueError('Accepted M1 parent identity is missing or changed.')
        if Path(self.parent_checkpoint_path).name != self.parent_checkpoint:
            raise ValueError('Parent checkpoint path/name mismatch.')
        if Path(self.parent_report_path).name != 'M1_report.json':
            raise ValueError('Parent report path is invalid.')
        if Path(self.parent_acceptance_path).name != 'M1_acceptance.json':
            raise ValueError('Parent acceptance path is invalid.')
        if Path(self.parent_audit_path).as_posix() != 'audit/m1_accepted_parent.json':
            raise ValueError('M2 audit path is fixed and may not be silently changed.')
        integer_fields = (
            self.j2_max, self.leg_count, self.sector_batch_size, self.seed,
            self.exact_decimal_digits,
            self.tensor_shard_bytes, self.max_item_attempts,
        ) + self.orientations
        if any(
            not isinstance(value, int) or isinstance(value, bool)
            for value in integer_fields
        ):
            raise ValueError('M2 discrete configuration fields must be integers.')
        if not 1 <= self.j2_max <= 4 or self.leg_count != 6:
            raise ValueError(
                'M2 requires leg_count=6 and j2_max in [1, 4] '
                '(higher cutoffs need a new governing-document revision).'
            )
        if self.sector_batch_size < 0:
            raise ValueError('M2 sector_batch_size must be nonnegative.')
        if self.j2_max > 1 and self.sector_batch_size == 0:
            # Fail closed: higher cutoffs must opt into sector batching.
            raise ValueError(
                'M2 j2_max>1 requires sector_batch_size>=1 for staged execution.'
            )
        if self.orientations != (1, -1, 1, -1, 1, -1):
            raise ValueError('M2 fixed link orientation convention changed.')
        if not 0 <= self.seed < 2**32:
            raise ValueError('M2 seed must satisfy 0 <= seed < 2**32.')
        if self.exact_decimal_digits < 64:
            raise ValueError('M2 diagnostic decimal precision is below the safe floor.')
        if self.tensor_shard_bytes <= 0 or self.max_item_attempts < 1:
            raise ValueError('M2 checkpoint or retry policy is invalid.')
        schedule = (
            0.0, self.no_long_task_after_s, self.drain_after_s,
            self.final_save_after_s, self.hard_return_s,
        )
        if tuple(sorted(schedule)) != schedule or len(set(schedule)) != len(schedule):
            raise ValueError('M2 session thresholds must be strictly increasing.')
        durations = schedule[1:] + (
            self.checkpoint_interval_s, self.max_work_item_s,
            self.short_task_limit_s, self.checkpoint_reserve_s,
        )
        if any(
            not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value <= 0
            for value in durations
        ):
            raise ValueError('M2 duration fields must be positive finite reals.')
        if self.checkpoint_interval_s > 15 * 60 or self.max_work_item_s > 20 * 60:
            raise ValueError('M2 checkpoint/work-item limits exceed the roadmap.')
        if self.final_save_after_s > 5 * 3600 + 20 * 60:
            raise ValueError('M2 final checkpoint begins too late.')
        if self.hard_return_s > 5.5 * 3600:
            raise ValueError('M2 session return exceeds the six-hour safety margin.')

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['orientations'] = list(self.orientations)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(
            self.canonical_payload(), sort_keys=True, separators=(',', ':'),
            allow_nan=False,
        )

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()
