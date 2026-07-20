"""Deterministic residual ledger for M5 one-step certification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from fractions import Fraction
from typing import Any, Iterable, Mapping, Sequence

from .exact_arithmetic import fraction_decimal_text
from .interval_kernel import ProofInterval, construct


class ResidualValidationError(RuntimeError):
    """Raised when residual provenance or aggregation is invalid."""


class ProofMethod(str, Enum):
    ANALYTIC_TAIL = 'analytic_tail'
    EXPLICIT_FROBENIUS = 'explicit_frobenius'
    BLOCK_NORM = 'block_norm'
    CPU_MULTIPRECISION = 'cpu_multiprecision'
    STRUCTURAL_IDENTITY = 'structural_identity'
    OUTWARD_INTERVAL = 'outward_interval'
    RATIONAL_COLLATZ = 'rational_collatz'


class RigourStatus(str, Enum):
    RIGOROUS = 'RIGOROUS'
    HEURISTIC = 'HEURISTIC'
    MISSING = 'MISSING'


@dataclass(frozen=True, slots=True)
class ResidualDependency:
    term_id: str
    relation: str

    def payload(self) -> dict[str, str]:
        return {'term_id': self.term_id, 'relation': self.relation}


@dataclass(frozen=True, slots=True)
class ResidualTerm:
    term_id: str
    category: str
    mathematical_meaning: str
    norm_id: str
    lower: Fraction
    upper: Fraction
    arithmetic_backend: str
    precision_bits: int
    rounding_policy: str
    source_artifacts: tuple[str, ...]
    source_hashes: tuple[str, ...]
    dependencies: tuple[ResidualDependency, ...]
    rigour_status: RigourStatus
    formula_id: str
    sector_scope: str
    cutoff_scope: str
    rank_scope: str
    proof_method: ProofMethod
    contribution_key: str

    def __post_init__(self) -> None:
        if self.lower < 0 or self.upper < 0 or self.lower > self.upper:
            raise ResidualValidationError(
                f'Residual endpoints must satisfy 0 <= lower <= upper for {self.term_id}.'
            )
        if not all((
            self.term_id,
            self.category,
            self.mathematical_meaning,
            self.norm_id,
            self.formula_id,
            self.sector_scope,
            self.cutoff_scope,
            self.rank_scope,
            self.contribution_key,
        )):
            raise ResidualValidationError('Residual term is missing required provenance.')
        if not self.source_artifacts or not self.source_hashes:
            raise ResidualValidationError(
                f'Residual {self.term_id} has empty provenance.'
            )
        if len(self.source_artifacts) != len(self.source_hashes):
            raise ResidualValidationError(
                f'Residual {self.term_id} has mismatched artifact/hash lists.'
            )
        for digest in self.source_hashes:
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(ch not in '0123456789abcdef' for ch in digest)
            ):
                raise ResidualValidationError(
                    f'Residual {self.term_id} has a malformed source hash.'
                )

    @property
    def interval(self) -> ProofInterval:
        return construct(
            self.lower,
            self.upper,
            arithmetic_backend=self.arithmetic_backend,
            precision_bits=self.precision_bits,
            rounding_policy=self.rounding_policy,
        )

    def payload(self) -> dict[str, Any]:
        data = asdict(self)
        data['lower'] = fraction_decimal_text(self.lower)
        data['upper'] = fraction_decimal_text(self.upper)
        data['lower_fraction'] = {
            'numerator_hex': format(self.lower.numerator, 'x'),
            'denominator_hex': format(self.lower.denominator, 'x'),
        }
        data['upper_fraction'] = {
            'numerator_hex': format(self.upper.numerator, 'x'),
            'denominator_hex': format(self.upper.denominator, 'x'),
        }
        data['rigour_status'] = self.rigour_status.value
        data['proof_method'] = self.proof_method.value
        data['dependencies'] = [dep.payload() for dep in self.dependencies]
        data['source_artifacts'] = list(self.source_artifacts)
        data['source_hashes'] = list(self.source_hashes)
        return data


@dataclass(frozen=True, slots=True)
class ResidualAggregate:
    aggregate_id: str
    term_ids: tuple[str, ...]
    lower: Fraction
    upper: Fraction
    rigour_status: RigourStatus
    formula_id: str

    def payload(self) -> dict[str, Any]:
        return {
            'aggregate_id': self.aggregate_id,
            'term_ids': list(self.term_ids),
            'lower': fraction_decimal_text(self.lower),
            'upper': fraction_decimal_text(self.upper),
            'rigour_status': self.rigour_status.value,
            'formula_id': self.formula_id,
        }


@dataclass
class ResidualLedger:
    terms: dict[str, ResidualTerm] = field(default_factory=dict)
    contribution_keys: set[str] = field(default_factory=set)

    def add_term(
        self,
        *,
        term_id: str,
        category: str,
        mathematical_meaning: str,
        norm_id: str,
        lower: Any,
        upper: Any,
        source_artifacts: Sequence[str],
        source_hashes: Sequence[str],
        dependencies: Iterable[ResidualDependency] = (),
        rigour_status: RigourStatus,
        formula_id: str,
        sector_scope: str,
        cutoff_scope: str,
        rank_scope: str,
        proof_method: ProofMethod,
        contribution_key: str,
        arithmetic_backend: str = 'rational_fraction',
        precision_bits: int = 256,
        rounding_policy: str = 'outward',
    ) -> ResidualTerm:
        if term_id in self.terms:
            raise ResidualValidationError(f'Duplicate residual term_id: {term_id}')
        if contribution_key in self.contribution_keys:
            raise ResidualValidationError(
                f'Duplicate residual contribution key: {contribution_key}'
            )
        interval = construct(
            lower,
            upper,
            arithmetic_backend=arithmetic_backend,
            precision_bits=precision_bits,
            rounding_policy=rounding_policy,
        )
        if interval.lo < 0:
            raise ResidualValidationError('Residual lower endpoint must be nonnegative.')
        deps = tuple(dependencies)
        for dependency in deps:
            if dependency.term_id not in self.terms:
                raise ResidualValidationError(
                    f'Unknown residual dependency: {dependency.term_id}'
                )
        if self._would_create_cycle(term_id, deps):
            raise ResidualValidationError('Residual dependency DAG contains a cycle.')
        if rigour_status is RigourStatus.MISSING:
            raise ResidualValidationError(
                'Missing residual terms must not be recorded as zero; omit them and fail closed.'
            )
        term = ResidualTerm(
            term_id=term_id,
            category=category,
            mathematical_meaning=mathematical_meaning,
            norm_id=norm_id,
            lower=interval.lo,
            upper=interval.hi,
            arithmetic_backend=arithmetic_backend,
            precision_bits=precision_bits,
            rounding_policy=rounding_policy,
            source_artifacts=tuple(source_artifacts),
            source_hashes=tuple(source_hashes),
            dependencies=deps,
            rigour_status=rigour_status,
            formula_id=formula_id,
            sector_scope=sector_scope,
            cutoff_scope=cutoff_scope,
            rank_scope=rank_scope,
            proof_method=proof_method,
            contribution_key=contribution_key,
        )
        self.terms[term_id] = term
        self.contribution_keys.add(contribution_key)
        return term

    def _would_create_cycle(
        self, term_id: str, dependencies: tuple[ResidualDependency, ...],
    ) -> bool:
        graph: dict[str, set[str]] = {
            existing: {dep.term_id for dep in term.dependencies}
            for existing, term in self.terms.items()
        }
        graph[term_id] = {dep.term_id for dep in dependencies}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            visiting.add(node)
            for child in graph.get(node, ()):
                if visit(child):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False

        return visit(term_id)

    def require_term(self, term_id: str) -> ResidualTerm:
        if term_id not in self.terms:
            raise ResidualValidationError(
                f'Missing residual term is not treated as zero: {term_id}'
            )
        return self.terms[term_id]

    def aggregate_rigorous(
        self,
        *,
        aggregate_id: str,
        term_ids: Iterable[str],
        formula_id: str,
    ) -> ResidualAggregate:
        ids = tuple(term_ids)
        if not ids:
            raise ResidualValidationError('Aggregate requires at least one term.')
        if len(ids) != len(set(ids)):
            raise ResidualValidationError('Aggregate term_ids must be unique.')
        lower = Fraction(0)
        upper = Fraction(0)
        for term_id in ids:
            term = self.require_term(term_id)
            if term.rigour_status is not RigourStatus.RIGOROUS:
                raise ResidualValidationError(
                    f'Heuristic/missing residual cannot enter a rigorous aggregate: {term_id}'
                )
            lower += term.lower
            upper += term.upper
        return ResidualAggregate(
            aggregate_id=aggregate_id,
            term_ids=ids,
            lower=lower,
            upper=upper,
            rigour_status=RigourStatus.RIGOROUS,
            formula_id=formula_id,
        )

    def payload(self) -> dict[str, Any]:
        return {
            'schema_version': 1,
            'terms': {
                term_id: term.payload()
                for term_id, term in sorted(self.terms.items())
            },
        }


def content_addressed_term_id(identity: Mapping[str, Any]) -> str:
    payload = json.dumps(
        identity, sort_keys=True, separators=(',', ':'), allow_nan=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()
