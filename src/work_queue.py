from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import quantiles
from typing import Any, Literal

from .common import canonical_json_bytes, read_json, sha256_bytes, sha256_file

WorkStatus = Literal['pending', 'running', 'done', 'failed', 'blocked_resource']


@dataclass(slots=True)
class WorkItem:
    item_id: str
    phase: str
    input_hash: str
    parameters: dict[str, Any]
    status: WorkStatus = 'pending'
    attempts: int = 0
    predicted_s: float = 60.0
    result_relpath: str | None = None
    result_sha256: str | None = None
    last_error: str | None = None


@dataclass(slots=True)
class WorkQueue:
    items: dict[str, WorkItem] = field(default_factory=dict)
    timings_by_phase: dict[str, list[float]] = field(default_factory=dict)

    @staticmethod
    def make_id(phase: str, input_hash: str, parameters: dict[str, Any]) -> str:
        payload = {'phase': phase, 'input_hash': input_hash, 'parameters': parameters}
        return sha256_bytes(canonical_json_bytes(payload))

    def add(self, phase: str, input_hash: str, parameters: dict[str, Any], predicted_s: float) -> str:
        if not math.isfinite(predicted_s) or predicted_s <= 0.0:
            raise ValueError('predicted_s must be positive and finite.')
        item_id = self.make_id(phase, input_hash, parameters)
        candidate = WorkItem(item_id, phase, input_hash, parameters, predicted_s=predicted_s)
        existing = self.items.setdefault(item_id, candidate)
        if existing.phase != phase or existing.input_hash != input_hash or existing.parameters != parameters:
            raise RuntimeError('Content-address collision with unequal work specification.')
        return item_id

    def next_pending(self) -> WorkItem | None:
        for item_id in sorted(self.items):
            item = self.items[item_id]
            if item.status == 'pending':
                return item
        return None

    def predicted_duration(self, item: WorkItem) -> float:
        samples = self.timings_by_phase.get(item.phase, [])
        if len(samples) < 2:
            return item.predicted_s
        if len(samples) < 20:
            return max(item.predicted_s, max(samples))
        p95 = quantiles(samples, n=20, method='inclusive')[18]
        return max(item.predicted_s, p95)

    def record_timing(self, phase: str, elapsed_s: float) -> None:
        if math.isfinite(elapsed_s) and elapsed_s >= 0.0:
            values = self.timings_by_phase.setdefault(phase, [])
            values.append(elapsed_s)
            del values[:-100]

    def validate(self) -> None:
        allowed = {'pending', 'running', 'done', 'failed', 'blocked_resource'}
        for key, item in self.items.items():
            if not all(isinstance(value, str) for value in (key, item.item_id, item.phase, item.input_hash)):
                raise ValueError('Work item identity fields must be strings.')
            if not isinstance(item.parameters, dict):
                raise ValueError(f'Work item parameters must be a mapping: {key}')
            if key != item.item_id or key != self.make_id(item.phase, item.input_hash, item.parameters):
                raise ValueError(f'Work item content hash mismatch: {key}')
            if item.status not in allowed:
                raise ValueError(f'Unknown work item status: {item.status!r}')
            if not isinstance(item.attempts, int) or isinstance(item.attempts, bool) or item.attempts < 0:
                raise ValueError(f'Invalid work item attempt count: {key}')
            if not isinstance(item.predicted_s, (int, float)) or not math.isfinite(item.predicted_s) or item.predicted_s <= 0.0:
                raise ValueError(f'Invalid work item counters or prediction: {key}')
            if item.status == 'done' and (not item.result_relpath or not item.result_sha256):
                raise ValueError(f'Done work item lacks result metadata: {key}')
            if item.result_relpath is not None and not isinstance(item.result_relpath, str):
                raise ValueError(f'Invalid work item result path: {key}')
            if item.last_error is not None and not isinstance(item.last_error, str):
                raise ValueError(f'Invalid work item error field: {key}')
            if item.result_sha256 is not None and (
                not isinstance(item.result_sha256, str) or len(item.result_sha256) != 64
                or any(character not in '0123456789abcdef' for character in item.result_sha256)
            ):
                raise ValueError(f'Invalid work item result hash: {key}')
        for phase, samples in self.timings_by_phase.items():
            if not isinstance(phase, str) or any(not math.isfinite(value) or value < 0.0 for value in samples):
                raise ValueError(f'Invalid timing history for phase {phase!r}.')

    def recover_interrupted(self, run_root: Path) -> list[str]:
        repaired: list[str] = []
        marker_root = run_root / 'work_items'
        for item in self.items.values():
            if item.status not in {'running', 'done'}:
                continue
            previous_status = item.status
            marker = marker_root / f'{item.item_id}.done'
            valid = False
            if marker.is_file():
                try:
                    payload = read_json(marker)
                    relative = payload['result_relpath']
                    digest = payload['result_sha256']
                    if payload.get('item_id') != item.item_id:
                        raise ValueError('Done marker item_id mismatch.')
                    if not isinstance(relative, str) or not isinstance(digest, str):
                        raise ValueError('Done marker has invalid field types.')
                    result = (run_root / relative).resolve()
                    result.relative_to(run_root.resolve())
                    valid = result.is_file() and sha256_file(result) == digest
                    if previous_status == 'done' and (
                        item.result_relpath != relative or item.result_sha256 != digest
                    ):
                        valid = False
                    if valid:
                        item.result_relpath = relative
                        item.result_sha256 = digest
                except (KeyError, OSError, TypeError, ValueError):
                    valid = False
            item.status = 'done' if valid else 'pending'
            if not valid:
                item.result_relpath = None
                item.result_sha256 = None
                # Process death leaves status=running after attempts was already
                # incremented; do not spend the attempt budget on infrastructure kills.
                if previous_status == 'running' and item.attempts > 0:
                    item.attempts -= 1
            if previous_status == 'running' or item.status != previous_status:
                repaired.append(item.item_id)
        return repaired

    def reset_transient_attempt_budget(
        self,
        *,
        max_item_attempts: int,
        reasons: tuple[str, ...] = (
            'Maximum M2 attempt count exceeded.',
            'KeyboardInterrupt',
        ),
    ) -> list[str]:
        """Re-open items exhausted only by interrupts / session kills."""
        repaired: list[str] = []
        for item in self.items.values():
            error = item.last_error or ''
            transient = any(reason in error for reason in reasons)
            over_budget = item.attempts >= max_item_attempts
            if item.status == 'failed' and (transient or over_budget):
                item.status = 'pending'
                item.attempts = 0
                item.last_error = None
                repaired.append(item.item_id)
            elif item.status == 'pending' and over_budget:
                item.attempts = 0
                if transient:
                    item.last_error = None
                repaired.append(item.item_id)
        return repaired

    def reopen_phases(self, phases: tuple[str, ...] | list[str]) -> list[str]:
        """Force listed phases back to pending (e.g. after controller code drift)."""
        wanted = set(phases)
        repaired: list[str] = []
        for item in self.items.values():
            if item.phase not in wanted:
                continue
            if item.status == 'pending' and item.attempts == 0 and not item.result_relpath:
                continue
            item.status = 'pending'
            item.attempts = 0
            item.result_relpath = None
            item.result_sha256 = None
            item.last_error = None
            repaired.append(item.item_id)
        return repaired

    def to_payload(self) -> dict[str, Any]:
        return {
            'items': {key: asdict(value) for key, value in sorted(self.items.items())},
            'timings_by_phase': self.timings_by_phase,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> 'WorkQueue':
        queue = cls(
            items={key: WorkItem(**value) for key, value in payload['items'].items()},
            timings_by_phase={key: [float(v) for v in values] for key, values in payload.get('timings_by_phase', {}).items()},
        )
        queue.validate()
        return queue
