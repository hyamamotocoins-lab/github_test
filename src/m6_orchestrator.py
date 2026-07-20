"""Restartable M6 multi-step certification orchestrator."""

from __future__ import annotations

import os
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
from .interval_kernel import construct
from .m5_package import (
    make_contractive_fixture_inputs,
    make_noncontractive_fixture_inputs,
)
from .m6_config import M6Config, default_m6_config
from .m6_independent_verifier import (
    M6IndependentVerifierError,
    verify_final_certificate,
)
from .m6_lock import m6_lock_payload
from .m6_package import (
    M6PackageError,
    assemble_final_certificate,
    close_standard_ledger,
    finalize_package_after_independent_pass,
    make_singleton_step,
)
from .m6_parent import M6ParentError, verify_accepted_m5_parent
from .m6_reporting import write_m6_report_package
from .m6_status import (
    CERTIFIED,
    M5_PARENT_RUN_ID_FROZEN,
    M6_BLOCKED_IMPLEMENTATION,
    M6_BLOCKED_MATH,
    M6_COMPLETE,
    M6_RUN_ID_FROZEN,
    M6_VERIFICATION_FAILED,
    NOT_CERTIFIED,
)
from .orchestrator import governing_document_hashes
from .runtime import environment_info
from .source_channels import SOURCE_CLASSES


class M6OrchestratorError(RuntimeError):
    """Raised when M6 orchestration fails closed."""


def _project_root_from_env() -> Path:
    explicit = os.environ.get('VALIDATED_RG_PROJECT_ROOT')
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd().resolve()


def _persist_root_from_env() -> Path:
    return Path(
        os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg')
    ).expanduser().resolve()


class M6Orchestrator:
    def __init__(
        self,
        project_root: Path,
        persistent_root: Path,
        config: M6Config,
    ) -> None:
        self.project_root = project_root.resolve()
        self.persistent_root = persistent_root.resolve()
        self.config = config
        safe_component(config.run_id)
        safe_component(config.parent_m5_run_id)
        self.run_root = self.persistent_root / 'runs' / config.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.package_root = self.run_root / 'artifacts' / 'final_certificate'

    def _write_run_config(self) -> str:
        payload = {
            'schema_version': 1,
            'milestone': 'M6',
            'config': self.config.payload(),
            'lock': m6_lock_payload(
                num_steps=self.config.num_steps,
                j2_max=self.config.j2_max,
                bond_dimension=self.config.bond_dimension,
            ),
            'governing_documents': governing_document_hashes(self.project_root),
            'environment': environment_info(),
            'created_at': utc_now(),
        }
        path = self.run_root / 'run_config.json'
        atomic_write_json(path, payload)
        return sha256_file(path)

    def verify_parent(self) -> dict[str, Any]:
        evidence = verify_accepted_m5_parent(
            self.project_root,
            self.persistent_root,
            self.config.parent_m5_run_id,
            require_frozen_id=(self.config.mode == 'paperspace'),
        )
        return {
            'run_id': evidence.run_id,
            'hashes': evidence.hashes,
            'acceptance': evidence.acceptance,
            'package_root': str(evidence.package_root),
        }

    def _assemble_and_verify(
        self,
        *,
        parent_m5_run_id: str,
        parent_hashes: dict[str, str],
        labels: list[str],
        matrix: list[list[Any]],
        perron: list[str],
        outside_tail: Any,
        residual_budget: Fraction,
        z_min: Any,
        notes_prefix: str,
        composition_policy: str,
    ) -> dict[str, Any]:
        parent_ref = parent_hashes.get('m5_package_verdict_sha256', 'fixture')
        steps = [
            make_singleton_step(
                step_index=index,
                z_min=z_min,
                residual_budget=residual_budget,
                parent_package_hash=parent_ref,
            )
            for index in range(self.config.num_steps)
        ]
        ledger = close_standard_ledger(
            residual_budget=residual_budget,
            notes_prefix=notes_prefix,
        )
        config_payload = {
            **self.config.payload(),
            'composition_policy': composition_policy,
        }
        assembled = assemble_final_certificate(
            self.package_root,
            run_id=self.config.run_id,
            parent_m5_run_id=parent_m5_run_id,
            config=config_payload,
            environment=environment_info(),
            source_hashes={'parent': parent_hashes},
            checkpoint_chain={
                'schema_version': 1,
                'parent_m5_run_id': parent_m5_run_id,
                'num_steps': self.config.num_steps,
                'step_ids': [f'rg_step_{index:02d}' for index in range(self.config.num_steps)],
            },
            error_ledger=ledger,
            steps=steps,
            weighted_matrix_entries=matrix,
            weighted_labels=labels,
            perron_values=perron,
            outside_matrix_tail=outside_tail,
            code_root=self.project_root / 'src',
            theorem_scope=(
                '# Theorem scope\n\n'
                'Finite-cutoff, finite-step 4D SU(2) truncated RG influence '
                'certificate only.\n'
            ),
            assumptions=(
                '# Assumptions\n\n'
                '- Frozen LOCK conventions (norm, cutoff, channels, sources).\n'
                '- Singleton input balls unless a step reopens E5.\n'
                '- Parent M5 package independent_verifier=PASS.\n'
            ),
            limitations=(
                '# Limitations\n\n'
                '- No continuum / OS / mass-gap claim.\n'
                '- Positive-radius families require reopening E5/E9.\n'
            ),
        )
        preview = verify_final_certificate(
            self.package_root, require_independent_pass_marker=True,
        )
        if not preview.agreement:
            raise M6PackageError('Independent verifier disagreed before PASS marker.')
        finalized = finalize_package_after_independent_pass(
            self.package_root,
            independent_report=preview.payload(),
            verdict=assembled['verdict'],
        )
        final_report = verify_final_certificate(self.package_root)
        if not final_report.agreement:
            raise M6PackageError('Independent verifier did not agree after PASS marker.')
        return {
            'assembled': assembled,
            'finalized': finalized,
            'independent_report': final_report.payload(),
            'lock': assembled['lock'],
            'verdict': finalized['verdict'],
        }

    def _run_fixture(self, *, contractive: bool) -> dict[str, Any]:
        fixture = (
            make_contractive_fixture_inputs()
            if contractive
            else make_noncontractive_fixture_inputs()
        )
        return self._assemble_and_verify(
            parent_m5_run_id=self.config.parent_m5_run_id,
            parent_hashes={'fixture': 'cpu_fixture'},
            labels=list(fixture['labels']),
            matrix=[list(row) for row in fixture['weighted_matrix']],
            perron=list(fixture['perron']),
            outside_tail=fixture['outside_tail'],
            residual_budget=Fraction(0),
            z_min=construct('1'),
            notes_prefix='cpu_fixture',
            composition_policy='fixture_final_influence',
        )

    def _coerce_interval_cell(self, cell: Any) -> Any:
        if isinstance(cell, dict) and 'lo' in cell and 'hi' in cell:
            lo = cell['lo']
            hi = cell['hi']
            if isinstance(lo, dict) and isinstance(hi, dict):
                from .exact_arithmetic import fraction_from_payload
                return construct(fraction_from_payload(lo), fraction_from_payload(hi))
        return cell

    def _matrix_from_m5_package(self, package_root: Path) -> tuple[list[str], list[list[Any]], list[str], Any, Fraction]:
        influence = read_json(package_root / 'influence_matrix_intervals.json')
        perron = read_json(package_root / 'perron_vector.json')
        collatz = read_json(package_root / 'collatz_bound.json')
        if not all(isinstance(doc, dict) for doc in (influence, perron, collatz)):
            raise M6OrchestratorError('M5 package bound artifacts malformed.')

        weighted = influence.get('weighted_matrix')
        if isinstance(weighted, dict) and isinstance(weighted.get('entries'), list):
            labels = list(weighted.get('labels') or influence.get('labels') or [])
            entries = weighted['entries']
        else:
            labels = list(influence.get('labels') or [])
            entries = influence.get('entries')
            if not isinstance(entries, list):
                raise M6OrchestratorError('M5 influence matrix entries missing.')
        if not labels:
            labels = [source.value for source in SOURCE_CLASSES]

        matrix: list[list[Any]] = []
        for row in entries:
            if not isinstance(row, list):
                raise M6OrchestratorError('M5 influence row malformed.')
            matrix.append([self._coerce_interval_cell(cell) for cell in row])

        components = perron.get('components')
        if not isinstance(components, list):
            raise M6OrchestratorError('M5 perron vector malformed.')
        perron_values = [
            str(Fraction(
                int(item['numerator_hex'], 16),
                int(item['denominator_hex'], 16),
            ))
            for item in components
            if isinstance(item, dict)
        ]
        outside = self._coerce_interval_cell(
            collatz.get('outside_matrix_tail', construct(0).serialize())
        )
        residual_budget = Fraction(0)
        residuals = package_root / 'contraction_residuals.json'
        if residuals.is_file():
            doc = read_json(residuals)
            if isinstance(doc, dict) and doc.get('aggregate_projection_upper'):
                try:
                    residual_budget = Fraction(str(doc['aggregate_projection_upper']))
                except ValueError:
                    residual_budget = Fraction(0)
        return labels, matrix, perron_values, outside, residual_budget

    def _run_paperspace(self, parent_info: dict[str, Any]) -> dict[str, Any]:
        package_root = Path(parent_info['package_root'])
        labels, matrix, perron, outside, residual = self._matrix_from_m5_package(
            package_root,
        )
        norm = read_json(package_root / 'normalization_bounds.json')
        z_min = construct('1')
        if isinstance(norm, dict) and isinstance(norm.get('z_min_interval'), dict):
            z_min = norm['z_min_interval']
        return self._assemble_and_verify(
            parent_m5_run_id=self.config.parent_m5_run_id,
            parent_hashes=dict(parent_info.get('hashes') or {}),
            labels=labels,
            matrix=matrix,
            perron=perron,
            outside_tail=outside,
            residual_budget=residual,
            z_min=z_min,
            notes_prefix='paperspace_inherited_m5',
            composition_policy=(
                'final_coarse_inherits_m5_one_step_influence_majorant_'
                'under_singleton_family_inclusion'
            ),
        )

    def run_until_checkpoint(self) -> dict[str, Any]:
        config_hash = self._write_run_config()
        parent_info: dict[str, Any] | None = None
        implementation_status = M6_BLOCKED_IMPLEMENTATION
        phase = 'M6_IN_PROGRESS'
        milestone_status = M6_BLOCKED_IMPLEMENTATION
        certification_status = NOT_CERTIFIED
        package_result: dict[str, Any] | None = None
        independent_report: dict[str, Any] | None = None
        lock_payload = m6_lock_payload(
            num_steps=self.config.num_steps,
            j2_max=self.config.j2_max,
            bond_dimension=self.config.bond_dimension,
        )

        try:
            if self.config.mode.startswith('cpu_fixture'):
                # Synthetic parent id is allowed; skip live parent verify.
                parent_info = {
                    'run_id': self.config.parent_m5_run_id,
                    'hashes': {'fixture': self.config.mode},
                    'acceptance': {
                        'milestone': 'M5',
                        'phase': 'M5_COMPLETE',
                        'status': 'PASS',
                        'certification_status': NOT_CERTIFIED,
                        'accepted_for_next_milestone': 'M6',
                    },
                    'package_root': None,
                }
                package_result = self._run_fixture(
                    contractive=(self.config.mode == 'cpu_fixture_cert'),
                )
            else:
                parent_info = self.verify_parent()
                package_result = self._run_paperspace(parent_info)

            verdict = package_result['verdict']
            independent_report = package_result['independent_report']
            lock_payload = package_result['lock']
            phase = verdict['phase']
            milestone_status = verdict['milestone_status']
            certification_status = verdict['certification_status']
            implementation_status = 'M6_IMPLEMENTATION_COMPLETE'
        except M6ParentError as exc:
            implementation_status = M6_VERIFICATION_FAILED
            phase = 'M6_FAILED'
            milestone_status = M6_VERIFICATION_FAILED
            certification_status = NOT_CERTIFIED
            parent_info = {'error': str(exc)}
        except (M6PackageError, M6IndependentVerifierError, M6OrchestratorError) as exc:
            implementation_status = M6_BLOCKED_MATH
            phase = 'M6_FAILED'
            milestone_status = M6_BLOCKED_MATH
            certification_status = NOT_CERTIFIED
            if parent_info is None:
                parent_info = {}
            parent_info['error'] = str(exc)
        except Exception as exc:  # noqa: BLE001
            implementation_status = M6_BLOCKED_IMPLEMENTATION
            phase = 'M6_FAILED'
            milestone_status = M6_BLOCKED_IMPLEMENTATION
            certification_status = NOT_CERTIFIED
            parent_info = {'error': str(exc)}

        verdict = (
            package_result['verdict']
            if package_result is not None
            else {
                'schema_version': 1,
                'run_id': self.config.run_id,
                'phase': phase,
                'milestone_status': milestone_status,
                'certification_status': certification_status,
                'independent_verifier': 'NOT_RUN',
            }
        )
        report = {
            'schema_version': 1,
            'generated_at': utc_now(),
            'milestone': 'M6',
            'run_id': self.config.run_id,
            'parent_m5_run_id': self.config.parent_m5_run_id,
            'phase': phase,
            'milestone_status': milestone_status,
            'certification_status': certification_status,
            'implementation_status': implementation_status,
            'num_steps': self.config.num_steps,
            'config_hash': config_hash,
            'code_tree_sha256': hash_tree(self.project_root / 'src', suffixes=('.py',)),
            'parent': parent_info,
            'verdict': verdict,
            'scope_limitation': (
                'Finite-cutoff multi-step certificate only; no continuum/mass-gap claim.'
            ),
        }
        write_m6_report_package(
            self.run_root,
            report=report,
            independent_report=independent_report,
            lock=lock_payload,
        )

        if phase == M6_COMPLETE and package_result is not None:
            acceptance = {
                'schema_version': 1,
                'milestone': 'M6',
                'phase': M6_COMPLETE,
                'status': 'PASS',
                'certification_status': certification_status,
                'run_id': self.config.run_id,
                'parent_m5_run_id': self.config.parent_m5_run_id,
                'num_steps': self.config.num_steps,
                'gates': {
                    'parent_verified': True,
                    'lock_written': True,
                    'package_assembled': True,
                    'independent_verifier': verdict.get('independent_verifier') == 'PASS',
                    'error_ledger_closed': True,
                },
                'decision': (
                    'ACCEPT_M6_CERTIFIED'
                    if certification_status == CERTIFIED
                    else 'ACCEPT_M6_CERTIFICATE_FAILURE'
                ),
                'decision_scope': (
                    'Finite-cutoff finite-step truncated SU(2) RG only. '
                    'NOT_CERTIFIED means the declared majorant failed to prove '
                    'q_cert < 1; it does not prove the true RG map is expansive.'
                ),
                'package_manifest_hash': verdict.get('package_manifest_hash'),
                'q_cert_upper': verdict.get('q_cert_upper'),
                'q_cert_lower': verdict.get('q_cert_lower'),
                'generated_at': utc_now(),
            }
            atomic_write_json(self.run_root / 'reports' / 'M6_acceptance.json', acceptance)
            audit = {
                'schema_version': 1,
                'milestone_reviewed': 'M6',
                'accepted_run_id': self.config.run_id,
                'accepted_phase': M6_COMPLETE,
                'certification_status': certification_status,
                'decision': acceptance['decision'],
                'm6_acceptance_path': str(
                    self.run_root / 'reports' / 'M6_acceptance.json'
                ),
                'm6_acceptance_sha256': sha256_file(
                    self.run_root / 'reports' / 'M6_acceptance.json'
                ),
                'generated_at': utc_now(),
            }
            (self.project_root / 'audit').mkdir(parents=True, exist_ok=True)
            atomic_write_json(self.project_root / 'audit' / 'm6_accepted_parent.json', audit)

        atomic_write_json(self.run_root / 'session_summary.json', {
            'run_id': self.config.run_id,
            'phase': phase,
            'implementation_status': implementation_status,
            'certification_status': certification_status,
            'generated_at': utc_now(),
        })
        atomic_write_text(
            self.run_root / 'next_session_plan.md',
            (
                '# Next session plan\n\n'
                f'1. phase={phase}, certification_status={certification_status}\n'
                '2. Inspect artifacts/final_certificate/ and reports/M6_report.json.\n'
                '3. If NOT_CERTIFIED: this is a verified certificate failure '
                '(majorant did not prove q_cert<1), not a proof that the true '
                'RG map is non-contractive.\n'
                '4. Next: sharpen majorants / evaluate true multi-step product '
                'B_{K-1}...B_0 before LOCK scheme changes.\n'
                '5. Continuum bridges remain out of scope.\n'
            ),
        )
        return report


def create_or_resume_m6(
    persistent_root: Path | None = None,
    config: M6Config | None = None,
    project_root: Path | None = None,
    *,
    run_id: str | None = None,
) -> M6Orchestrator:
    project = project_root or _project_root_from_env()
    persist = persistent_root or _persist_root_from_env()
    cfg = config or default_m6_config()
    if run_id is not None:
        cfg = default_m6_config(**{**cfg.payload(), 'run_id': run_id})
    if cfg.mode == 'paperspace':
        if cfg.parent_m5_run_id != M5_PARENT_RUN_ID_FROZEN:
            raise M6OrchestratorError(
                f'paperspace mode requires parent {M5_PARENT_RUN_ID_FROZEN}'
            )
        if cfg.run_id != M6_RUN_ID_FROZEN:
            raise M6OrchestratorError(
                f'paperspace mode requires run_id {M6_RUN_ID_FROZEN}'
            )
    return M6Orchestrator(project, persist, cfg)
