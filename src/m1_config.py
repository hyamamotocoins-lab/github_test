from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class M1Config:
    schema_version: int = 1
    project_name: str = 'validated_4d_su2_rg'
    milestone: str = 'M1'
    parent_milestone: str = 'M0'
    parent_run_id: str = '20260719T120406Z-731966c8fd28'
    parent_checkpoint: str = 'ckpt_000014'
    parent_checkpoint_path: str = '/storage/validated_4d_su2_rg/runs/20260719T120406Z-731966c8fd28/checkpoints/ckpt_000014'
    beta_numerator: int = 11
    beta_denominator: int = 5
    seed: int = 20260719
    cutoffs: tuple[int, ...] = (6, 8, 10, 12, 14, 16)
    dimensions: tuple[int, ...] = (2, 3, 4)
    rg_steps: int = 3
    coefficient_series_terms: int = 96
    exp_series_terms: int = 120
    verifier_series_terms: int = 72
    decimal_places: int = 36
    checkpoint_interval_s: float = 15.0 * 60.0
    max_work_item_s: float = 20.0 * 60.0
    short_task_limit_s: float = 5.0 * 60.0
    checkpoint_reserve_s: float = 10.0 * 60.0
    no_long_task_after_s: float = 5.0 * 3600.0
    drain_after_s: float = 5.0 * 3600.0 + 15.0 * 60.0
    final_save_after_s: float = 5.0 * 3600.0 + 20.0 * 60.0
    hard_return_s: float = 5.0 * 3600.0 + 30.0 * 60.0
    tensor_shard_bytes: int = 256 * 1024 * 1024
    max_item_attempts: int = 3
    certification_status: str = 'NOT_CERTIFIED'

    def __post_init__(self) -> None:
        if self.schema_version != 1 or self.project_name != 'validated_4d_su2_rg' or self.milestone != 'M1':
            raise ValueError('Unsupported M1 project/config schema.')
        if self.parent_milestone != 'M0' or self.certification_status != 'NOT_CERTIFIED':
            raise ValueError('M1 parent/status invariant failed.')
        if not self.parent_run_id or self.parent_checkpoint != 'ckpt_000014':
            raise ValueError('Accepted M0 parent identity is missing or changed.')
        if Path(self.parent_checkpoint_path).name != self.parent_checkpoint:
            raise ValueError('Parent checkpoint path/name mismatch.')
        integers = (
            self.beta_numerator, self.beta_denominator, self.seed, self.rg_steps,
            self.coefficient_series_terms, self.exp_series_terms, self.verifier_series_terms,
            self.decimal_places, self.tensor_shard_bytes, self.max_item_attempts,
        ) + self.cutoffs + self.dimensions
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integers):
            raise ValueError('M1 discrete configuration fields must be integers.')
        if self.beta_numerator <= 0 or self.beta_denominator <= 0 or self.rg_steps < 0:
            raise ValueError('M1 beta and RG step count are invalid.')
        if not 0 <= self.seed < 2**32:
            raise ValueError('M1 seed must satisfy 0 <= seed < 2**32.')
        if tuple(sorted(set(self.cutoffs))) != self.cutoffs or min(self.cutoffs) < 1:
            raise ValueError('M1 cutoffs must be strictly increasing positive integers.')
        if tuple(sorted(set(self.dimensions))) != self.dimensions or min(self.dimensions) < 2:
            raise ValueError('M1 dimensions must be strictly increasing and start above the trivial irrep.')
        if min(self.coefficient_series_terms, self.exp_series_terms, self.verifier_series_terms, self.decimal_places) < 16:
            raise ValueError('M1 rational precision policy is below the fail-closed floor.')
        schedule = (0.0, self.no_long_task_after_s, self.drain_after_s, self.final_save_after_s, self.hard_return_s)
        if tuple(sorted(schedule)) != schedule or len(set(schedule)) != len(schedule):
            raise ValueError('M1 session thresholds must be strictly increasing.')
        durations = schedule[1:] + (self.checkpoint_interval_s, self.max_work_item_s, self.short_task_limit_s, self.checkpoint_reserve_s)
        if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value <= 0 for value in durations):
            raise ValueError('M1 duration fields must be positive finite reals.')
        if self.checkpoint_interval_s > 15 * 60 or self.max_work_item_s > 20 * 60:
            raise ValueError('M1 checkpoint/work-item limits exceed the roadmap.')
        if self.final_save_after_s > 5 * 3600 + 20 * 60 or self.hard_return_s > 5.5 * 3600:
            raise ValueError('M1 session finalization exceeds the hard limits.')

    def canonical_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload['cutoffs'] = list(self.cutoffs)
        payload['dimensions'] = list(self.dimensions)
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_payload(), sort_keys=True, separators=(',', ':'), allow_nan=False)

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode('utf-8')).hexdigest()
