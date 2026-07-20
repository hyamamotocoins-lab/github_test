from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Iterable


class ErrorLedgerError(RuntimeError):
    '''Raised when error provenance is incomplete or double counted.'''


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')


@dataclass(frozen=True, slots=True)
class ErrorTerm:
    term_id: str
    name: str
    category: str
    applies_to: str
    source_checkpoint: str
    formula: str
    parents: tuple[str, ...]
    estimate: float | None
    deterministic_upper_bound: float | None
    rigor: str
    note: str

    def identity_payload(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop('term_id')
        payload['parents'] = list(self.parents)
        return payload

    def payload(self) -> dict[str, Any]:
        return {'term_id': self.term_id, **self.identity_payload()}


class ErrorLedger:
    def __init__(self) -> None:
        self.terms: dict[str, ErrorTerm] = {}

    @staticmethod
    def _validate_number(value: float | None, label: str) -> None:
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or value < 0.0
        ):
            raise ErrorLedgerError(f'{label} must be nonnegative finite or null.')

    def _add(
        self, *, name: str, category: str, applies_to: str,
        source_checkpoint: str, formula: str, parents: tuple[str, ...],
        estimate: float | None, deterministic_upper_bound: float | None,
        rigor: str, note: str,
    ) -> str:
        if rigor not in {'RIGOROUS', 'HEURISTIC', 'MISSING'}:
            raise ErrorLedgerError(f'Invalid rigor classification: {rigor}')
        if applies_to not in {'primal', 'tangent', 'both'}:
            raise ErrorLedgerError(f'Invalid error target: {applies_to}')
        if not all((name, category, source_checkpoint, formula, note)):
            raise ErrorLedgerError('Every error term requires complete provenance.')
        self._validate_number(estimate, 'estimate')
        self._validate_number(
            deterministic_upper_bound, 'deterministic upper bound',
        )
        if rigor == 'RIGOROUS' and deterministic_upper_bound is None:
            raise ErrorLedgerError('A rigorous term requires an upper bound.')
        if rigor == 'MISSING' and deterministic_upper_bound is not None:
            raise ErrorLedgerError('A missing bound cannot be marked deterministic.')
        if any(parent not in self.terms for parent in parents):
            raise ErrorLedgerError('Error DAG references an unknown parent.')
        identity = {
            'name': name, 'category': category, 'applies_to': applies_to,
            'source_checkpoint': source_checkpoint, 'formula': formula,
            'parents': list(parents), 'estimate': estimate,
            'deterministic_upper_bound': deterministic_upper_bound,
            'rigor': rigor, 'note': note,
        }
        term_id = hashlib.sha256(_canonical(identity)).hexdigest()
        term = ErrorTerm(term_id, name, category, applies_to, source_checkpoint,
                         formula, parents, estimate, deterministic_upper_bound,
                         rigor, note)
        existing = self.terms.get(term_id)
        if existing is not None and existing != term:
            raise ErrorLedgerError('Content-addressed error term collision.')
        self.terms[term_id] = term
        return term_id

    def add_leaf(
        self, *, name: str, category: str, applies_to: str,
        source_checkpoint: str, formula: str, estimate: float | None,
        deterministic_upper_bound: float | None, rigor: str, note: str,
    ) -> str:
        return self._add(
            name=name, category=category, applies_to=applies_to,
            source_checkpoint=source_checkpoint, formula=formula, parents=(),
            estimate=estimate,
            deterministic_upper_bound=deterministic_upper_bound,
            rigor=rigor, note=note,
        )

    def leaf_ids(self, term_id: str) -> set[str]:
        if term_id not in self.terms:
            raise ErrorLedgerError('Unknown error term.')
        term = self.terms[term_id]
        if not term.parents:
            return {term_id}
        result: set[str] = set()
        for parent in term.parents:
            leaves = self.leaf_ids(parent)
            if result & leaves:
                raise ErrorLedgerError('Error DAG double counts a leaf term.')
            result.update(leaves)
        return result

    def add_alias(
        self, *, name: str, category: str, applies_to: str,
        parent: str, source_checkpoint: str, formula: str, note: str,
    ) -> str:
        term = self.terms[parent]
        return self._add(
            name=name, category=category, applies_to=applies_to,
            source_checkpoint=source_checkpoint, formula=formula,
            parents=(parent,), estimate=term.estimate,
            deterministic_upper_bound=term.deterministic_upper_bound,
            rigor=term.rigor, note=note,
        )

    def add_sum(
        self, *, name: str, category: str, applies_to: str,
        parents: Iterable[str], source_checkpoint: str, formula: str,
        note: str,
    ) -> str:
        parent_tuple = tuple(parents)
        if not parent_tuple:
            raise ErrorLedgerError('An aggregate error requires parents.')
        leaves: set[str] = set()
        for parent in parent_tuple:
            current = self.leaf_ids(parent)
            if leaves & current:
                raise ErrorLedgerError('Aggregate error would double count a leaf.')
            leaves.update(current)
        leaf_terms = [self.terms[item] for item in sorted(leaves)]
        estimate = float(sum(
            item.estimate for item in leaf_terms if item.estimate is not None
        ))
        enclosed = all(
            item.rigor == 'RIGOROUS'
            and item.deterministic_upper_bound is not None
            for item in leaf_terms
        )
        bound = (
            float(sum(item.deterministic_upper_bound or 0.0 for item in leaf_terms))
            if enclosed else None
        )
        rigor = (
            'RIGOROUS' if enclosed
            else 'MISSING' if any(item.rigor == 'MISSING' for item in leaf_terms)
            else 'HEURISTIC'
        )
        return self._add(
            name=name, category=category, applies_to=applies_to,
            source_checkpoint=source_checkpoint, formula=formula,
            parents=parent_tuple, estimate=estimate,
            deterministic_upper_bound=bound, rigor=rigor, note=note,
        )

    def validate(self) -> None:
        for term_id, term in self.terms.items():
            if term.term_id != term_id:
                raise ErrorLedgerError('Error ledger key/id mismatch.')
            expected = hashlib.sha256(_canonical(term.identity_payload())).hexdigest()
            if expected != term_id:
                raise ErrorLedgerError('Error term content hash changed.')
            self.leaf_ids(term_id)

    def summary(self) -> dict[str, Any]:
        self.validate()
        leaves = [term for term in self.terms.values() if not term.parents]
        missing = [
            term.name for term in leaves
            if term.deterministic_upper_bound is None
        ]
        return {
            'term_count': len(self.terms),
            'leaf_count': len(leaves),
            'rigorous_leaf_count': sum(
                term.rigor == 'RIGOROUS' for term in leaves
            ),
            'heuristic_leaf_count': sum(
                term.rigor == 'HEURISTIC' for term in leaves
            ),
            'missing_leaf_count': sum(
                term.rigor == 'MISSING' for term in leaves
            ),
            'missing_deterministic_bound_terms': sorted(missing),
            'enclosure_ready': not missing,
            'double_counting_check': 'PASS',
        }

    def payload(self) -> dict[str, Any]:
        self.validate()
        return {
            'schema_version': 1,
            'terms': [
                self.terms[term_id].payload() for term_id in sorted(self.terms)
            ],
            'summary': self.summary(),
        }

    @classmethod
    def from_payload(cls, payload: object) -> ErrorLedger:
        if not isinstance(payload, dict) or payload.get('schema_version') != 1:
            raise ErrorLedgerError('Unsupported error ledger payload.')
        entries = payload.get('terms')
        if not isinstance(entries, list):
            raise ErrorLedgerError('Error ledger terms are missing.')
        ledger = cls()
        remaining = [entry for entry in entries]
        while remaining:
            progressed = False
            for entry in tuple(remaining):
                if not isinstance(entry, dict):
                    raise ErrorLedgerError('Malformed error ledger term.')
                parents = tuple(entry.get('parents', ()))
                if all(parent in ledger.terms for parent in parents):
                    term_id = ledger._add(
                        name=entry['name'], category=entry['category'],
                        applies_to=entry['applies_to'],
                        source_checkpoint=entry['source_checkpoint'],
                        formula=entry['formula'], parents=parents,
                        estimate=entry.get('estimate'),
                        deterministic_upper_bound=entry.get(
                            'deterministic_upper_bound'
                        ),
                        rigor=entry['rigor'], note=entry['note'],
                    )
                    if term_id != entry.get('term_id'):
                        raise ErrorLedgerError('Serialized error term hash changed.')
                    remaining.remove(entry)
                    progressed = True
            if not progressed:
                raise ErrorLedgerError('Error ledger contains a cycle.')
        ledger.validate()
        return ledger
