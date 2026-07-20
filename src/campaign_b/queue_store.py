"""Atomic queue / ledger persistence for Campaign B."""

from __future__ import annotations

import socket
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from .budget import BudgetManager
from .errors import CampaignFatalError
from .estimators import RuntimeEstimator
from .schemas import screening_only_payload
from .state_machine import apply_candidate_state


def _owner() -> str:
    return f'{socket.gethostname()}:{os_getpid()}'


def os_getpid() -> int:
    import os
    return os.getpid()


class QueueStore:
    def __init__(self, root: Path, *, lease_sec: int = 1800) -> None:
        self.root = Path(root)
        self.queue_path = self.root / 'queue.json'
        self.ledger_path = self.root / 'ledger.json'
        self.lease_sec = int(lease_sec)
        self.root.mkdir(parents=True, exist_ok=True)

    def load_or_init(
        self,
        candidates: list[dict[str, Any]],
        *,
        campaign_run_id: str,
    ) -> dict[str, Any]:
        if self.queue_path.is_file():
            payload = read_json(self.queue_path)
            if not isinstance(payload, dict):
                raise CampaignFatalError('corrupt queue.json')
            return payload
        payload = {
            'schema_version': 1,
            'campaign_run_id': campaign_run_id,
            'updated_at': utc_now(),
            'candidates': candidates,
            **screening_only_payload(),
        }
        self.save_queue(payload)
        return payload

    def save_queue(self, payload: dict[str, Any]) -> None:
        payload = dict(payload)
        payload['updated_at'] = utc_now()
        payload.update(screening_only_payload())
        atomic_write_json(self.queue_path, payload)

    def load_ledger(self) -> dict[str, Any]:
        if not self.ledger_path.is_file():
            ledger = {
                'schema_version': 1,
                'campaign_state': 'CREATED',
                'terminal_reason': None,
                'events': [],
                'selected': [],
                'archived_ids': [],
                'exceptions': [],
                'runtime_estimator': RuntimeEstimator().payload(),
                **screening_only_payload(),
            }
            atomic_write_json(self.ledger_path, ledger)
            return ledger
        payload = read_json(self.ledger_path)
        if not isinstance(payload, dict):
            raise CampaignFatalError('corrupt ledger.json')
        return payload

    def save_ledger(self, ledger: dict[str, Any]) -> None:
        ledger = dict(ledger)
        ledger['updated_at'] = utc_now()
        ledger.update(screening_only_payload())
        atomic_write_json(self.ledger_path, ledger)

    def set_campaign_state(self, state: str) -> None:
        ledger = self.load_ledger()
        ledger['campaign_state'] = state
        self.save_ledger(ledger)

    def set_terminal_reason(self, reason: str) -> None:
        ledger = self.load_ledger()
        ledger['terminal_reason'] = reason
        self.save_ledger(ledger)

    def record_event(self, event: dict[str, Any]) -> None:
        ledger = self.load_ledger()
        events = list(ledger.get('events') or [])
        events.append({**event, 'at': utc_now()})
        ledger['events'] = events
        self.save_ledger(ledger)

    def record_exception(self, exc: BaseException) -> None:
        ledger = self.load_ledger()
        exceptions = list(ledger.get('exceptions') or [])
        exceptions.append({
            'type': type(exc).__name__,
            'message': str(exc),
            'at': utc_now(),
        })
        ledger['exceptions'] = exceptions
        self.save_ledger(ledger)

    def record_selected(self, candidate_id: str, package_dir: str) -> None:
        ledger = self.load_ledger()
        selected = list(ledger.get('selected') or [])
        selected.append({
            'candidate_id': candidate_id,
            'package_dir': package_dir,
            'at': utc_now(),
            **screening_only_payload(),
        })
        ledger['selected'] = selected
        self.save_ledger(ledger)

    def recover_expired_leases(self, queue: dict[str, Any]) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        changed = False
        for cand in queue.get('candidates') or []:
            lease = cand.get('lease')
            if not lease:
                continue
            expires = lease.get('lease_expires_at')
            if not expires:
                continue
            try:
                exp_dt = datetime.fromisoformat(expires)
            except ValueError:
                continue
            if exp_dt.tzinfo is None:
                exp_dt = exp_dt.replace(tzinfo=timezone.utc)
            if exp_dt <= now and cand.get('state') == 'RESERVED':
                cand.pop('lease', None)
                apply_candidate_state(cand, 'PENDING')
                changed = True
        if changed:
            self.save_queue(queue)
        return queue

    def next_admissible(
        self,
        queue: dict[str, Any],
        *,
        budget: BudgetManager,
        estimator: RuntimeEstimator,
        archived_ids: set[str],
    ) -> dict[str, Any] | None:
        pending = [
            c for c in (queue.get('candidates') or [])
            if c.get('state') == 'PENDING'
            and c.get('candidate_id') not in archived_ids
        ]
        pending.sort(
            key=lambda c: (-float(c.get('priority_score') or 0.0), c['candidate_id']),
        )
        for cand in pending:
            predicted = estimator.upper_runtime_sec('SCREENING', cand)
            if budget.may_start('SCREENING', predicted):
                return cand
        return None

    def reserve(self, queue: dict[str, Any], candidate_id: str) -> dict[str, Any]:
        found = None
        for cand in queue.get('candidates') or []:
            if cand.get('candidate_id') == candidate_id:
                found = cand
                break
        if found is None:
            raise CampaignFatalError(f'candidate missing: {candidate_id}')
        apply_candidate_state(found, 'RESERVED')
        expires = datetime.now(timezone.utc) + timedelta(seconds=self.lease_sec)
        found['lease'] = {
            'candidate_id': candidate_id,
            'owner': _owner(),
            'reserved_at': utc_now(),
            'lease_expires_at': expires.isoformat(),
            'stage': 'SCREENING',
        }
        self.save_queue(queue)
        return found

    def update_candidate(
        self,
        queue: dict[str, Any],
        candidate_id: str,
        **fields: Any,
    ) -> dict[str, Any]:
        for cand in queue.get('candidates') or []:
            if cand.get('candidate_id') == candidate_id:
                if 'state' in fields:
                    apply_candidate_state(cand, str(fields.pop('state')))
                cand.update(fields)
                self.save_queue(queue)
                return cand
        raise CampaignFatalError(f'candidate missing: {candidate_id}')
