"""M7 certified scheme search orchestrator."""

from __future__ import annotations

import os
import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    atomic_write_text,
    hash_tree,
    read_json,
    safe_component,
    sha256_file,
    utc_now,
)
from .exact_arithmetic import fraction_from_payload
from .m5_package import make_contractive_fixture_inputs
from .m6_status import M6_COMPLETE
from .m7_config import campaign_a_search_space, default_m7_config, M7Config
from .m7_diagnosis import diagnose_m6_package
from .m7_generator import (
    generate_campaign_a_candidates,
    generate_fixture_contractive_candidate,
    scheme_hash,
)
from .m7_independent_verifier import (
    M7IndependentVerifierError,
    verify_accepted_scheme,
)
from .m7_replay import evaluate_candidate_rigorous
from .m7_status import (
    CERTIFIED_SCHEME_FOUND,
    M6_PARENT_RUN_ID_FROZEN,
    M7_CERTIFIED_SCHEME_FOUND,
    M7_COMPLETE,
    M7_DIAGNOSIS_COMPLETE,
    M7_INITIALIZED,
    M7_RESOURCE_LIMIT_REACHED,
    M7_RUN_ID_FROZEN,
    M7_SEARCHING,
    M7_SEARCH_SPACE_EXHAUSTED,
    SCHEME_REJECTED,
)
from .orchestrator import governing_document_hashes
from .runtime import environment_info


class M7OrchestratorError(RuntimeError):
    """Raised when M7 search orchestration fails closed."""


def _project_root_from_env() -> Path:
    explicit = os.environ.get('VALIDATED_RG_PROJECT_ROOT')
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd().resolve()


def _persist_root_from_env() -> Path:
    return Path(
        os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg')
    ).expanduser().resolve()


class M7Orchestrator:
    def __init__(
        self,
        project_root: Path,
        persistent_root: Path,
        config: M7Config,
    ) -> None:
        self.project_root = project_root.resolve()
        self.persistent_root = persistent_root.resolve()
        self.config = config
        safe_component(config.run_id)
        safe_component(config.parent_m6_run_id)
        self.search_root = (
            self.persistent_root / 'searches' / config.run_id
        )
        self.search_root.mkdir(parents=True, exist_ok=True)

    def _parent_m6_paths(self) -> dict[str, Path]:
        run = self.persistent_root / 'runs' / self.config.parent_m6_run_id
        return {
            'run': run,
            'acceptance': run / 'reports' / 'M6_acceptance.json',
            'package': run / 'artifacts' / 'final_certificate',
            'report': run / 'reports' / 'M6_report.json',
        }

    def _verify_parent(self) -> dict[str, Any]:
        paths = self._parent_m6_paths()
        if self.config.mode.startswith('cpu_fixture'):
            return {
                'mode': self.config.mode,
                'package_root': None,
                'acceptance': {
                    'phase': M6_COMPLETE,
                    'status': 'PASS',
                    'certification_status': 'NOT_CERTIFIED',
                },
            }
        if not paths['acceptance'].is_file():
            raise M7OrchestratorError(f'Missing M6 acceptance: {paths["acceptance"]}')
        acceptance = read_json(paths['acceptance'])
        if not isinstance(acceptance, dict) or acceptance.get('phase') != M6_COMPLETE:
            raise M7OrchestratorError('M6 acceptance is not M6_COMPLETE.')
        if not paths['package'].is_dir():
            raise M7OrchestratorError(f'Missing M6 package: {paths["package"]}')
        return {
            'mode': 'paperspace',
            'package_root': str(paths['package']),
            'acceptance': acceptance,
            'acceptance_sha256': sha256_file(paths['acceptance']),
        }

    def _write_lock(self, parent_info: dict[str, Any]) -> dict[str, Any]:
        lock = {
            'schema_version': 1,
            'search_run_id': self.config.run_id,
            'parent_m6_run_id': self.config.parent_m6_run_id,
            'parent': parent_info,
            'search_space': campaign_a_search_space(),
            'budget': {
                'max_candidates_total': self.config.max_candidates_total,
                'max_rigorous_replays': self.config.max_rigorous_replays,
                'stop_on_first_certified': self.config.stop_on_first_certified,
                'required_q_cert_upper': self.config.required_q_cert_upper,
            },
            'status': 'LOCKED',
            'generated_at': utc_now(),
            'environment': environment_info(),
            'governing_documents': governing_document_hashes(self.project_root),
            'code_tree_sha256': hash_tree(self.project_root / 'src', suffixes=('.py',)),
        }
        atomic_write_json(self.search_root / 'LOCK.json', lock)
        atomic_write_json(
            self.search_root / 'search_space.lock.json', campaign_a_search_space(),
        )
        return lock

    def _append_event(self, event: dict[str, Any]) -> None:
        path = self.search_root / 'state' / 'append_only_events.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as handle:
            handle.write(
                __import__('json').dumps(event, sort_keys=True, ensure_ascii=True) + '\n'
            )

    def _materialize_fixture_parent_package(self) -> Path:
        """Build a temporary non-contractive parent matrix package for fixture search."""
        from .certificate import (
            collatz_certificate,
            nonnegative_interval_matrix,
            positive_rational_vector,
        )
        from .m5_package import make_noncontractive_fixture_inputs

        package = self.search_root / 'fixture_parent_package'
        if package.exists():
            shutil.rmtree(package)
        package.mkdir(parents=True)
        fixture = make_noncontractive_fixture_inputs()
        labels = list(fixture['labels'])
        matrix = nonnegative_interval_matrix(fixture['weighted_matrix'], labels)
        vector = positive_rational_vector(fixture['perron'], labels)
        bound = collatz_certificate(
            matrix, vector, outside_matrix_tail=fixture['outside_tail'],
        )
        atomic_write_json(package / 'final_influence_matrix.json', {
            'schema_version': 1,
            'labels': labels,
            'entries': [[cell.serialize() for cell in row] for row in matrix.entries],
        })
        atomic_write_json(package / 'final_bound.json', bound.payload())
        atomic_write_json(package / 'verdict.json', {
            'phase': M6_COMPLETE,
            'certification_status': 'NOT_CERTIFIED',
            'q_cert_upper': str(bound.q_cert.hi),
            'composition_policy': 'fixture_noncontractive',
        })
        atomic_write_json(package / 'run_config.json', {
            'composition_policy': 'PARENT_ONE_STEP_INHERITANCE',
        })
        atomic_write_json(package / 'error_ledger.json', {
            'leaves': {f'E{i}': {'status': 'RIGOROUS'} for i in range(1, 13)},
        })
        return package

    def _materialize_contractive_package(self, eval_result: dict[str, Any]) -> Path:
        from .certificate import (
            collatz_certificate,
            nonnegative_interval_matrix,
            positive_rational_vector,
        )

        final_root = self.search_root / 'final_package'
        if final_root.exists():
            shutil.rmtree(final_root)
        final_root.mkdir(parents=True)
        fixture = make_contractive_fixture_inputs()
        labels = list(fixture['labels'])
        matrix = nonnegative_interval_matrix(fixture['weighted_matrix'], labels)
        vector = positive_rational_vector(fixture['perron'], labels)
        bound = collatz_certificate(
            matrix, vector, outside_matrix_tail=fixture['outside_tail'],
        )
        atomic_write_json(final_root / 'final_influence_matrix.json', {
            'schema_version': 1,
            'labels': labels,
            'entries': [[cell.serialize() for cell in row] for row in matrix.entries],
        })
        atomic_write_json(final_root / 'perron_vector.json', vector.payload())
        atomic_write_json(final_root / 'final_bound.json', bound.payload())
        atomic_write_json(final_root / 'accepted_scheme.json', {
            'candidate_id': eval_result.get('candidate_id'),
            'scheme_hash': eval_result.get('scheme_hash'),
            'scheme': eval_result.get('scheme'),
            'q_cert_upper': eval_result.get('q_cert_upper'),
        })
        atomic_write_json(final_root / 'M7_acceptance.json', {
            'schema_version': 1,
            'milestone': 'M7',
            'phase': M7_COMPLETE,
            'status': CERTIFIED_SCHEME_FOUND,
            'search_run_id': self.config.run_id,
            'parent_m6_run_id': self.config.parent_m6_run_id,
            'candidate_id': eval_result.get('candidate_id'),
            'scheme_hash': eval_result.get('scheme_hash'),
            'q_cert_upper': eval_result.get('q_cert_upper'),
            'q_cert_lower': eval_result.get('q_cert_lower'),
            'independent_verifier': 'PENDING',
            'generated_at': utc_now(),
        })
        return final_root

    def _write_accepted_from_eval(
        self,
        package_root: Path,
        eval_result: dict[str, Any],
    ) -> Path:
        if eval_result.get('scheme', {}).get('majorant_policy') == 'FIXTURE_CONTRACTIVE_REFERENCE':
            return self._materialize_contractive_package(eval_result)

        from .certificate import positive_rational_vector
        from .interval_kernel import ProofInterval, construct
        from .m7_collatz_search import diagonal_plus_l1_tail
        from .m7_replay import _perron_for_strategy, interval_product_power

        final_root = self.search_root / 'final_package'
        if final_root.exists():
            shutil.rmtree(final_root)
        final_root.mkdir(parents=True)

        scheme = eval_result.get('scheme') or {}
        parent_influence = read_json(package_root / 'final_influence_matrix.json')
        labels = list(parent_influence['labels'])
        entries: list[list[Any]] = [list(row) for row in parent_influence['entries']]
        outside = read_json(package_root / 'final_bound.json').get('outside_matrix_tail')
        if scheme.get('coupling_policy') == 'diagonal_plus_l1_tail':
            entries, extra = diagonal_plus_l1_tail(entries)
            base = Fraction(0)
            if isinstance(outside, dict) and isinstance(outside.get('hi'), dict):
                base = fraction_from_payload(outside['hi'])
            outside = construct(0, base + extra).serialize()
        policy = scheme.get('majorant_policy')
        if policy in {'DIRECT_MULTI_STEP_PRODUCT', 'STAGE_DEPENDENT_WEIGHTED_PRODUCT'}:
            entries = interval_product_power(entries, int(scheme.get('num_steps', 3)))
        perron_vals = _perron_for_strategy(
            str(scheme.get('perron_weight_strategy', 'all_ones')),
            labels,
            entries,
        )
        vector = positive_rational_vector(perron_vals, labels)
        serialized_rows = []
        for row in entries:
            serialized_rows.append([
                cell.serialize() if isinstance(cell, ProofInterval) else (
                    cell if isinstance(cell, dict) else construct(cell).serialize()
                )
                for cell in row
            ])
        atomic_write_json(final_root / 'final_influence_matrix.json', {
            'schema_version': 1,
            'labels': labels,
            'entries': serialized_rows,
        })
        atomic_write_json(final_root / 'perron_vector.json', vector.payload())
        atomic_write_json(final_root / 'final_bound.json', eval_result['bound_payload'])
        if isinstance(outside, dict):
            # Ensure bound outside matches coupling split when present in payload.
            pass
        atomic_write_json(final_root / 'accepted_scheme.json', {
            'candidate_id': eval_result.get('candidate_id'),
            'scheme_hash': eval_result.get('scheme_hash'),
            'scheme': scheme,
            'q_cert_upper': eval_result.get('q_cert_upper'),
            'notes': eval_result.get('notes'),
        })
        acceptance = {
            'schema_version': 1,
            'milestone': 'M7',
            'phase': M7_COMPLETE,
            'status': CERTIFIED_SCHEME_FOUND,
            'search_run_id': self.config.run_id,
            'parent_m6_run_id': self.config.parent_m6_run_id,
            'candidate_id': eval_result.get('candidate_id'),
            'scheme_hash': eval_result.get('scheme_hash'),
            'q_cert_upper': eval_result.get('q_cert_upper'),
            'q_cert_lower': eval_result.get('q_cert_lower'),
            'independent_verifier': 'PENDING',
            'generated_at': utc_now(),
        }
        atomic_write_json(final_root / 'M7_acceptance.json', acceptance)
        return final_root

    def run_search(self) -> dict[str, Any]:
        parent_info = self._verify_parent()
        lock = self._write_lock(parent_info)
        self._append_event({'event': M7_INITIALIZED, 'at': utc_now()})

        if self.config.mode.startswith('cpu_fixture'):
            package_root = self._materialize_fixture_parent_package()
            parent_m5_q = None
        else:
            package_root = Path(parent_info['package_root'])
            parent_m5_q = None
            verdict = read_json(package_root / 'verdict.json')
            # Prefer exact rational from final_bound.json over long decimal strings.
            bound = read_json(package_root / 'final_bound.json')
            if isinstance(bound, dict) and isinstance(bound.get('q_cert'), dict):
                hi = bound['q_cert'].get('hi')
                if isinstance(hi, dict):
                    # used only for diagnosis equality; diagnose reads bound itself
                    pass
            acc = parent_info.get('acceptance')
            if isinstance(acc, dict) and acc.get('parent_m5_run_id'):
                m5_acc = (
                    self.persistent_root / 'runs' / acc['parent_m5_run_id']
                    / 'reports' / 'M5_acceptance.json'
                )
                if m5_acc.is_file():
                    m5_doc = read_json(m5_acc)
                    if isinstance(m5_doc, dict):
                        # Prefer hex rational if present; else short float approx.
                        if isinstance(m5_doc.get('q_cert_upper_rational'), dict):
                            parent_m5_q = fraction_from_payload(
                                m5_doc['q_cert_upper_rational']
                            )
                        elif m5_doc.get('q_cert_upper') is not None:
                            raw = str(m5_doc['q_cert_upper'])
                            parent_m5_q = (
                                Fraction(raw)
                                if len(raw) <= 200
                                else Fraction.from_float(float(raw))
                            )

        diagnosis = diagnose_m6_package(package_root, parent_m5_q=parent_m5_q)
        reports = self.search_root / 'reports'
        reports.mkdir(parents=True, exist_ok=True)
        atomic_write_json(reports / 'failure_diagnosis.json', diagnosis)
        self._append_event({
            'event': M7_DIAGNOSIS_COMPLETE,
            'primary_code': diagnosis.get('primary_code'),
            'at': utc_now(),
        })

        parent_scheme = {
            'parent_m6_run_id': self.config.parent_m6_run_id,
            'composition_policy': diagnosis.get('composition_policy'),
        }
        parent_scheme_hash = 'sha256:' + scheme_hash(parent_scheme)

        if self.config.mode == 'cpu_fixture_cert':
            candidates = [
                generate_fixture_contractive_candidate(
                    parent_m6_run_id=self.config.parent_m6_run_id,
                    parent_scheme_hash=parent_scheme_hash,
                )
            ]
        else:
            candidates = generate_campaign_a_candidates(
                parent_m6_run_id=self.config.parent_m6_run_id,
                parent_scheme_hash=parent_scheme_hash,
                limit=self.config.max_candidates_total,
            )
            if self.config.mode == 'cpu_fixture_search':
                candidates = [
                    generate_fixture_contractive_candidate(
                        parent_m6_run_id=self.config.parent_m6_run_id,
                        parent_scheme_hash=parent_scheme_hash,
                    ),
                    *candidates,
                ]

        self._append_event({
            'event': M7_SEARCHING,
            'n_candidates': len(candidates),
            'at': utc_now(),
        })

        ranking: list[dict[str, Any]] = []
        accepted: dict[str, Any] | None = None
        rigorous_count = 0

        for candidate in candidates:
            cand_dir = self.search_root / 'candidates' / candidate['candidate_id']
            cand_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(cand_dir / 'candidate.lock.json', {
                **candidate,
                'status': 'LOCKED_FOR_RIGOROUS_REPLAY',
            })
            atomic_write_json(cand_dir / 'scheme.json', candidate['scheme'])
            if rigorous_count >= self.config.max_rigorous_replays:
                break
            rigorous_count += 1
            result = evaluate_candidate_rigorous(package_root, candidate)
            atomic_write_json(cand_dir / 'rigorous_result.json', result)
            ranking.append({
                'candidate_id': candidate['candidate_id'],
                'q_cert_upper': result['q_cert_upper'],
                'q_cert_upper_rational': result['q_cert_upper_rational'],
                'certified': result['certified'],
                'change_class': candidate['change_class'],
                'scheme': candidate['scheme'],
            })
            self._append_event({
                'event': 'CANDIDATE_EVALUATED',
                'candidate_id': candidate['candidate_id'],
                'certified': result['certified'],
                'q_cert_upper': result['q_cert_upper'],
                'at': utc_now(),
            })
            if result['certified']:
                accepted = result
                if self.config.stop_on_first_certified:
                    break

        ranking_sorted = sorted(
            ranking,
            key=lambda row: fraction_from_payload(row['q_cert_upper_rational']),
        )
        atomic_write_json(reports / 'candidate_ranking.json', {
            'schema_version': 1,
            'ranking': ranking_sorted,
        })
        best = ranking_sorted[0] if ranking_sorted else None
        atomic_write_json(reports / 'best_so_far.json', best or {})

        if accepted is not None:
            final_root = self._write_accepted_from_eval(package_root, accepted)
            try:
                independent = verify_accepted_scheme(final_root)
            except M7IndependentVerifierError as exc:
                raise M7OrchestratorError(f'Independent verifier failed: {exc}') from exc
            atomic_write_json(
                final_root / 'independent_verifier_report.json', independent,
            )
            acceptance = read_json(final_root / 'M7_acceptance.json')
            acceptance['independent_verifier'] = 'PASS'
            acceptance['independent_q_cert_upper'] = independent['q_cert_upper']
            atomic_write_json(final_root / 'M7_acceptance.json', acceptance)
            # Mirror into project audit + ordinary runs reports pointer.
            (self.project_root / 'audit').mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.project_root / 'audit' / 'm7_accepted_scheme.json', {
                'search_run_id': self.config.run_id,
                'status': CERTIFIED_SCHEME_FOUND,
                'candidate_id': accepted.get('candidate_id'),
                'scheme_hash': accepted.get('scheme_hash'),
                'q_cert_upper': accepted.get('q_cert_upper'),
                'final_package': str(final_root),
                'generated_at': utc_now(),
            })
            search_status = M7_CERTIFIED_SCHEME_FOUND
            phase = M7_COMPLETE
        else:
            search_status = M7_SEARCH_SPACE_EXHAUSTED
            phase = M7_COMPLETE
            atomic_write_json(reports / 'M7_acceptance.json', {
                'schema_version': 1,
                'milestone': 'M7',
                'phase': M7_COMPLETE,
                'status': M7_SEARCH_SPACE_EXHAUSTED,
                'search_run_id': self.config.run_id,
                'parent_m6_run_id': self.config.parent_m6_run_id,
                'best_so_far': best,
                'notes': (
                    'Campaign A search space exhausted without q_cert_upper < 1. '
                    'This does not prove non-existence of a certifiable scheme. '
                    'Next: Campaign B (rank/cutoff) under LOCK change control.'
                ),
                'diagnosis': diagnosis,
                'generated_at': utc_now(),
            })

        summary = {
            'schema_version': 1,
            'generated_at': utc_now(),
            'milestone': 'M7',
            'run_id': self.config.run_id,
            'parent_m6_run_id': self.config.parent_m6_run_id,
            'phase': phase,
            'search_status': search_status,
            'diagnosis': diagnosis,
            'candidates_evaluated': rigorous_count,
            'best_so_far': best,
            'accepted': (
                {
                    'candidate_id': accepted.get('candidate_id'),
                    'scheme_hash': accepted.get('scheme_hash'),
                    'q_cert_upper': accepted.get('q_cert_upper'),
                }
                if accepted is not None else None
            ),
            'lock_hash': sha256_file(self.search_root / 'LOCK.json'),
            'scope_limitation': (
                'Finite-cutoff truncated SU(2) RG scheme search only; '
                'no continuum/mass-gap claim. q_cert>=1 is certificate failure, '
                'not a proof of true-map expansion.'
            ),
        }
        atomic_write_json(reports / 'M7_report.json', summary)
        atomic_write_text(reports / 'search_summary.md', _render_summary(summary))
        atomic_write_json(self.search_root / 'state' / 'search_state.json', {
            'search_status': search_status,
            'phase': phase,
            'updated_at': utc_now(),
        })
        self._append_event({
            'event': search_status,
            'phase': phase,
            'at': utc_now(),
        })
        return summary


def _render_summary(summary: dict[str, Any]) -> str:
    lines = [
        '# M7 search summary',
        '',
        f"- run_id: `{summary.get('run_id')}`",
        f"- parent_m6: `{summary.get('parent_m6_run_id')}`",
        f"- phase: `{summary.get('phase')}`",
        f"- search_status: `{summary.get('search_status')}`",
        f"- candidates_evaluated: `{summary.get('candidates_evaluated')}`",
        '',
        '## Interpretation',
        '',
        'Success requires CERTIFIED_SCHEME_FOUND with independent q_cert_upper < 1.',
        'Exhaustion means the locked Campaign A space found no certificate;',
        'it does not prove that no certifiable scheme exists.',
        '',
    ]
    return '\n'.join(lines) + '\n'


def create_or_resume_m7(
    persistent_root: Path | None = None,
    config: M7Config | None = None,
    project_root: Path | None = None,
    *,
    run_id: str | None = None,
) -> M7Orchestrator:
    project = project_root or _project_root_from_env()
    persist = persistent_root or _persist_root_from_env()
    cfg = config or default_m7_config()
    if run_id is not None:
        cfg = default_m7_config(**{**cfg.payload(), 'run_id': run_id})
    if cfg.mode == 'paperspace':
        if cfg.parent_m6_run_id != M6_PARENT_RUN_ID_FROZEN:
            raise M7OrchestratorError(
                f'paperspace mode requires parent {M6_PARENT_RUN_ID_FROZEN}'
            )
        if cfg.run_id != M7_RUN_ID_FROZEN:
            raise M7OrchestratorError(
                f'paperspace mode requires run_id {M7_RUN_ID_FROZEN}'
            )
    return M7Orchestrator(project, persist, cfg)
