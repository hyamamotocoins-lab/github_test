"""Restartable M5 one-step certification orchestrator."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .common import (
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    hash_tree,
    safe_component,
    sha256_bytes,
    sha256_file,
    utc_now,
)
from .interval_kernel import construct
from .m5_config import M5Config, default_m5_config
from .m5_package import (
    assemble_one_step_package,
    make_contractive_fixture_inputs,
    make_noncontractive_fixture_inputs,
)
from .m5_parent import M5ParentError, verify_accepted_m4_parent
from .m5_reporting import write_m5_report_package
from .m5_status import (
    M4_PARENT_RUN_ID_FROZEN,
    M5_BLOCKED_IMPLEMENTATION,
    M5_BLOCKED_MATH,
    M5_RUN_ID_FROZEN,
    M5_VERIFICATION_FAILED,
    NOT_CERTIFIED,
    ONE_STEP_CERTIFIED,
    PROOF_OBLIGATIONS_OPEN,
)
from .orchestrator import governing_document_hashes
from .proof_manifest import write_certificate_manifest
from .runtime import environment_info


class M5OrchestratorError(RuntimeError):
    """Raised when M5 orchestration fails closed."""


def _project_root_from_env() -> Path:
    explicit = os.environ.get('VALIDATED_RG_PROJECT_ROOT')
    if explicit:
        return Path(explicit).expanduser().resolve()
    return Path.cwd().resolve()


def _persist_root_from_env() -> Path:
    return Path(
        os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg')
    ).expanduser().resolve()


def build_parent_artifact_inventory(
    project_root: Path,
    persistent_root: Path,
    parent_run_id: str,
    parent_hashes: dict[str, str],
) -> dict[str, Any]:
    run_root = persistent_root / 'runs' / parent_run_id
    records: list[dict[str, Any]] = []
    candidates = [
        ('reports/M4_report.json', 'M4', 'parent_report'),
        ('reports/M4_acceptance.json', 'M4', 'parent_acceptance'),
        ('run_manifest.json', 'M4', 'parent_manifest'),
        ('checkpoints/ckpt_000014/hashes.json', 'M4', 'parent_checkpoint_hashes'),
    ]
    for relative, milestone, role in candidates:
        path = run_root / relative
        if not path.is_file() or path.is_symlink():
            raise M5OrchestratorError(f'Parent inventory source missing: {relative}')
        records.append({
            'relative_path': relative,
            'sha256': sha256_file(path),
            'schema_version': 1,
            'producer_milestone': milestone,
            'producer_run_id': parent_run_id,
            'mathematical_role': role,
            'rigour_status': 'PARENT_ACCEPTED_NOT_ONE_STEP_BOUND',
            'precision': 'parent_native',
            'sector_ordering_hash': None,
            'convention_hash': None,
            'consumed_by': ['M5'],
        })
    audit = project_root / 'audit/m4_accepted_parent.json'
    records.append({
        'relative_path': 'audit/m4_accepted_parent.json',
        'sha256': sha256_file(audit),
        'schema_version': 1,
        'producer_milestone': 'M4',
        'producer_run_id': parent_run_id,
        'mathematical_role': 'acceptance_audit',
        'rigour_status': 'PARENT_ACCEPTED_NOT_ONE_STEP_BOUND',
        'precision': 'n/a',
        'sector_ordering_hash': None,
        'convention_hash': None,
        'consumed_by': ['M5'],
    })
    return {
        'schema_version': 1,
        'parent_run_id': parent_run_id,
        'parent_hashes': parent_hashes,
        'artifacts': records,
        'generated_at': utc_now(),
    }


def build_schema_mapping() -> dict[str, Any]:
    return {
        'schema_version': 1,
        'mapping': {
            'P1': 'M1 tail coefficient / tail theorem artifacts',
            'P2': 'M2 basis map / convention hash / equivalence tests',
            'P3': 'contraction graph / norms / structure constants',
            'P4': 'M3 fixed RSVD basis Q / sector shards',
            'P5': 'M3 contraction output / path / summation metadata',
            'P7': 'M4 primal/tangent tensors / basis-dependence ledger',
            'P8-P10': 'M5 kernel / normalization / source norms',
        },
        'policy': 'Do not invent values; stop if schema/version/hash mismatch.',
    }


class M5Orchestrator:
    def __init__(
        self,
        project_root: Path,
        persistent_root: Path,
        config: M5Config,
    ) -> None:
        self.project_root = project_root.resolve()
        self.persistent_root = persistent_root.resolve()
        self.config = config
        safe_component(config.run_id)
        safe_component(config.parent_m4_run_id)
        self.run_root = self.persistent_root / 'runs' / config.run_id
        self.run_root.mkdir(parents=True, exist_ok=True)
        self.package_root = self.run_root / 'artifacts' / 'one_step_certificate'

    def _write_run_config(self) -> str:
        payload = {
            'schema_version': 1,
            'milestone': 'M5',
            'config': self.config.payload(),
            'governing_documents': governing_document_hashes(self.project_root),
            'environment': environment_info(),
            'created_at': utc_now(),
        }
        path = self.run_root / 'run_config.json'
        atomic_write_json(path, payload)
        return sha256_file(path)

    def verify_parent(self) -> dict[str, Any]:
        evidence = verify_accepted_m4_parent(
            self.project_root,
            self.persistent_root,
            self.config.parent_m4_run_id,
        )
        return {
            'hashes': evidence.hashes,
            'bound_ledger': evidence.bound_ledger,
            'regression': evidence.regression,
            'tensor_keys': sorted(evidence.tensors),
        }

    def _assemble_from_fixture(self, contractive: bool) -> dict[str, Any]:
        if self.package_root.exists():
            shutil.rmtree(self.package_root)
        self.package_root.mkdir(parents=True, exist_ok=True)
        fixture = (
            make_contractive_fixture_inputs()
            if contractive
            else make_noncontractive_fixture_inputs()
        )
        zmin = construct('1')
        return assemble_one_step_package(
            self.package_root,
            run_id=self.config.run_id,
            parent_run_id=self.config.parent_m4_run_id,
            config=self.config.payload(),
            conventions={
                'metric_unit': self.config.metric_unit,
                'source_speed_unit': self.config.source_speed_unit,
                'orientation': 'canonical_su2',
                'phase': 'real_positive_characters',
            },
            initial_tail={
                'status': 'PASS',
                'beta_interval': ['1', '1'],
                'cutoff': self.config.cutoff,
                'norm': 'frobenius',
                'metric_normalization': 'haar_class_angle',
                'tail_value_interval': ['0', '0'],
                'tail_derivative_interval': ['0', '0'],
                'block_plaquette_count': 1,
                'source_contact_count': 1,
                'telescoping_formula': 'fixture_identity',
                'coefficient_artifact_hashes': [],
                'precision': self.config.precision_bits,
                'proof_method': 'fixture',
            },
            basis_equivalence={
                'status': 'PASS',
                'convention_hash': 'fixture',
                'structural_identity': 'U T_arm U^* = T_PW',
                'low_cutoff_residual_interval': ['0', '0'],
            },
            contraction_residuals={
                'status': 'PASS',
                'basis_hash': 'fixture',
                'orthogonality_defect': ['0', '0'],
                'projection_residual_by_sector': {},
                'discarded_channel_tail': ['0', '0'],
                'rank': self.config.bond_dimension,
                'cutoff': self.config.cutoff,
                'norm': 'frobenius',
                'proof_route': 'fixture',
                'precision': self.config.precision_bits,
                'aggregate_projection_upper': '0',
                'input_propagation_upper': '0',
                'rounding_upper': '0',
            },
            derivative_residuals={
                'status': 'PASS',
                'source_classes': list(fixture['labels']),
                'source_ordering_hash': sha256_bytes(
                    canonical_json_bytes(list(fixture['labels']))
                ),
                'tangent_center_norms': {},
                'tangent_input_radii': {},
                'basis_variation_residual': ['0', '0'],
                'normalization_derivative_terms': {},
                'source_contact_geometry': {'contact_count': 1},
                'derivative_output_radius': ['0', '0'],
                'symmetry_checks': {'status': 'PASS'},
                'zero_tangent_checks': {'status': 'PASS'},
                'm4_derivative_artifact_hashes': {},
            },
            normalization_bounds={
                'status': 'PASS',
                'z_min_interval': zmin.serialize(),
                'z_min_lower': format(zmin.lo, 'f'),
                'z_min_upper': format(zmin.hi, 'f'),
                'kernel_positivity_evidence': 'fixture_nonnegative_kernel',
                'kernel_l1_error': '0',
            },
            influence_entries=fixture['entries'],
            row_order=fixture['labels'],
            column_order=fixture['labels'],
            weighted_matrix_entries=fixture['weighted_matrix'],
            weighted_labels=fixture['labels'],
            perron_values=fixture['perron'],
            outside_matrix_tail=fixture['outside_tail'],
            code_root=self.project_root / 'src',
        )

    def run_until_checkpoint(self) -> dict[str, Any]:
        config_hash = self._write_run_config()
        parent_info: dict[str, Any] | None = None
        implementation_status = M5_BLOCKED_IMPLEMENTATION
        enclosure_status = PROOF_OBLIGATIONS_OPEN
        phase = 'M5_IN_PROGRESS'
        milestone_status = PROOF_OBLIGATIONS_OPEN
        certification_status = NOT_CERTIFIED
        package_result: dict[str, Any] | None = None
        independent_report: dict[str, Any] | None = None
        certificate_manifest: dict[str, Any] | None = None

        try:
            parent_info = self.verify_parent()
            if self.config.mode.startswith('cpu_fixture'):
                contractive = self.config.mode == 'cpu_fixture_cert'
                package_result = self._assemble_from_fixture(contractive=contractive)
                independent_report = package_result['independent_report']
                certificate_manifest = package_result['manifest']
                (self.run_root / 'reports').mkdir(parents=True, exist_ok=True)
                write_certificate_manifest(
                    self.run_root / 'reports' / 'M5_certificate_manifest.json',
                    self.package_root,
                )
                verdict = package_result['verdict']
                phase = verdict['phase']
                milestone_status = verdict['milestone_status']
                certification_status = verdict['certification_status']
                implementation_status = 'M5_IMPLEMENTATION_COMPLETE'
                enclosure_status = (
                    ONE_STEP_CERTIFIED
                    if certification_status == ONE_STEP_CERTIFIED
                    else NOT_CERTIFIED
                )
            else:
                inventory = build_parent_artifact_inventory(
                    self.project_root,
                    self.persistent_root,
                    self.config.parent_m4_run_id,
                    parent_info['hashes'],
                )
                mapping = build_schema_mapping()
                reports = self.run_root / 'reports'
                reports.mkdir(parents=True, exist_ok=True)
                atomic_write_json(reports / 'M5_parent_artifact_inventory.json', inventory)
                atomic_write_json(reports / 'M5_schema_mapping.json', mapping)
                open_bounds = parent_info['bound_ledger']['open_for_M5']
                # Proof primitives are present. Live storage still has open M4 handoff
                # obligations; do not invent zero residuals or mark M5_COMPLETE.
                if open_bounds:
                    implementation_status = 'M5_PROOF_PRIMITIVES_READY'
                    enclosure_status = PROOF_OBLIGATIONS_OPEN
                    phase = 'M5_IN_PROGRESS'
                    milestone_status = PROOF_OBLIGATIONS_OPEN
                    certification_status = NOT_CERTIFIED
                else:
                    implementation_status = M5_BLOCKED_MATH
                    enclosure_status = M5_BLOCKED_MATH
                    phase = 'M5_IN_PROGRESS'
                    milestone_status = M5_BLOCKED_MATH
                    certification_status = NOT_CERTIFIED
        except M5ParentError as exc:
            implementation_status = M5_VERIFICATION_FAILED
            enclosure_status = M5_VERIFICATION_FAILED
            phase = 'M5_FAILED'
            milestone_status = M5_VERIFICATION_FAILED
            certification_status = NOT_CERTIFIED
            parent_info = {'error': str(exc)}
        except Exception as exc:  # noqa: BLE001 - fail closed
            implementation_status = M5_BLOCKED_IMPLEMENTATION
            enclosure_status = M5_BLOCKED_IMPLEMENTATION
            phase = 'M5_FAILED'
            milestone_status = M5_BLOCKED_IMPLEMENTATION
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
                'open_for_M5': (
                    parent_info.get('bound_ledger', {}).get('open_for_M5')
                    if isinstance(parent_info, dict)
                    else None
                ),
            }
        )
        report = {
            'schema_version': 1,
            'generated_at': utc_now(),
            'milestone': 'M5',
            'run_id': self.config.run_id,
            'parent_m4_run_id': self.config.parent_m4_run_id,
            'phase': phase,
            'milestone_status': milestone_status,
            'certification_status': certification_status,
            'implementation_status': implementation_status,
            'enclosure_status': enclosure_status,
            'config_hash': config_hash,
            'code_tree_sha256': hash_tree(self.project_root / 'src', suffixes=('.py',)),
            'parent': parent_info,
            'verdict': verdict,
            'tests': {},
            'scope_limitation': (
                'Finite-cutoff one-step certificate only; no continuum/mass-gap claim.'
            ),
        }
        write_m5_report_package(
            self.run_root,
            report=report,
            independent_report=independent_report,
            certificate_manifest=certificate_manifest,
        )
        atomic_write_json(self.run_root / 'session_summary.json', {
            'run_id': self.config.run_id,
            'phase': phase,
            'implementation_status': implementation_status,
            'certification_status': certification_status,
            'generated_at': utc_now(),
        })
        atomic_write_json(self.run_root / 'latest_metrics.json', {
            'environment': environment_info(),
            'generated_at': utc_now(),
        })
        atomic_write_text(
            self.run_root / 'next_session_plan.md',
            (
                '# Next session plan\n\n'
                '1. Keep M4 parent immutable.\n'
                '2. Close open M5 proof obligations with deterministic residuals.\n'
                '3. Rebuild one_step_certificate and rerun independent verifier.\n'
                '4. Freeze only after independent_verifier=PASS.\n'
            ),
        )
        return report


def create_or_resume_m5(
    persistent_root: Path | None = None,
    config: M5Config | None = None,
    project_root: Path | None = None,
    *,
    run_id: str | None = None,
) -> M5Orchestrator:
    project = project_root or _project_root_from_env()
    persist = persistent_root or _persist_root_from_env()
    cfg = config or default_m5_config()
    if run_id is not None:
        cfg = default_m5_config(**{**cfg.payload(), 'run_id': run_id})
    if cfg.mode == 'paperspace':
        if cfg.parent_m4_run_id != M4_PARENT_RUN_ID_FROZEN:
            raise M5OrchestratorError(
                f'paperspace mode requires parent {M4_PARENT_RUN_ID_FROZEN}'
            )
        if cfg.run_id != M5_RUN_ID_FROZEN:
            raise M5OrchestratorError(
                f'paperspace mode requires run_id {M5_RUN_ID_FROZEN}'
            )
    return M5Orchestrator(project, persist, cfg)
