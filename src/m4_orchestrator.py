from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .checkpoint import CheckpointManager, CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, canonical_json_bytes, fsync_directory, hash_tree, read_json,
    safe_component, sha256_bytes, sha256_file, utc_now,
)
from .contraction_backend import (
    BackendUnavailableError, ContractionBackend, backend_selection, select_backend,
)
from .error_ledger import ErrorLedger
from .forward_ad import DualTensor, dual_matmul, dual_regroup, zero_source_dual
from .gpu_sharding import require_memory_headroom
from .m4_config import M4Config
from .m4_parent import M4ParentError, verify_accepted_m3_parent
from .m4_reporting import (
    M4_PHASES, load_m4_phase_results, validate_m4_acceptance,
    write_m4_report_package, write_m4_session_artifacts,
)
from .normalization import normalize_array, normalize_dual
from .orchestrator import governing_document_hashes, reference_artifact_hashes
from .reporting import JsonlLogger
from .runtime import environment_info, runtime_compatibility_signature
from .session_guard import SessionGuard, SessionState
from .source_channels import (
    SOURCE_CLASSES, SourceClass, deformed_projected_parent,
    generator_symmetry_residuals, projected_parent_dual, source_generators,
)
from .work_queue import WorkItem, WorkQueue

M4_NOTEBOOK_HASH_POLICY = 'canonical_nbformat4_cell_type_source_tags_v1'


class M4CompatibilityError(RuntimeError):
    '''Raised when an M4 run cannot be safely created or resumed.'''


def _seed_everything(seed: int = 20260720) -> None:
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _notebook_hash(project_root: Path) -> str:
    path = project_root / 'notebooks/50_m4_derivatives.ipynb'
    if not path.is_file():
        raise M4CompatibilityError('M4 user-facing notebook is missing.')
    payload = read_json(path)
    cells = payload.get('cells') if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get('nbformat') != 4 or not isinstance(cells, list):
        raise M4CompatibilityError('M4 notebook must be a valid nbformat 4 document.')
    identity_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise M4CompatibilityError(f'M4 notebook cell {index} is invalid.')
        cell_type = cell.get('cell_type')
        source = cell.get('source')
        metadata = cell.get('metadata', {})
        if cell_type not in {'code', 'markdown', 'raw'}:
            raise M4CompatibilityError(f'M4 notebook cell {index} type is invalid.')
        if isinstance(source, str):
            normalized = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            normalized = ''.join(source)
        else:
            raise M4CompatibilityError(f'M4 notebook cell {index} source is invalid.')
        if not isinstance(metadata, dict):
            raise M4CompatibilityError(f'M4 notebook cell {index} metadata is invalid.')
        tags = metadata.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise M4CompatibilityError(f'M4 notebook cell {index} tags are invalid.')
        identity_cells.append({'cell_type': cell_type, 'source': normalized, 'tags': tags})
    return sha256_bytes(canonical_json_bytes({
        'policy': M4_NOTEBOOK_HASH_POLICY, 'nbformat': 4, 'cells': identity_cells,
    }))


def _dual_from_tensors(tensors: dict[str, Any], prefix: str) -> DualTensor:
    primal_name = f'{prefix}_primal'
    tangent_names = {
        source: f'{prefix}_tangent_{source.value}' for source in SOURCE_CLASSES
    }
    if primal_name not in tensors or any(
        name not in tensors for name in tangent_names.values()
    ):
        raise RuntimeError(f'M4 checkpoint lacks complete {prefix} derivative state.')
    return DualTensor(
        np.asarray(tensors[primal_name], dtype=np.float64),
        {
            source: np.asarray(tensors[name], dtype=np.float64)
            for source, name in tangent_names.items()
        },
    )


def _array_pipeline(projected: np.ndarray) -> np.ndarray:
    from .forward_ad import regroup_matrix
    return normalize_array(regroup_matrix(projected @ projected))


class M4Orchestrator:
    def __init__(
        self, persistent_root: Path, run_root: Path, project_root: Path,
        config: M4Config, state: RunState, queue: WorkQueue,
        checkpoints: CheckpointManager, test_report: dict[str, Any],
        manifest: dict[str, Any], parent_tensors: dict[str, np.ndarray],
        parent_metrics: dict[str, Any], *, initial_session: bool,
        tensors: dict[str, Any] | None = None,
    ) -> None:
        self.persistent_root = persistent_root
        self.run_root = run_root
        self.project_root = project_root
        self.config = config
        self.state = state
        self.queue = queue
        self.checkpoints = checkpoints
        self.test_report = test_report
        self.manifest = manifest
        self.parent_tensors = parent_tensors
        self.parent_metrics = parent_metrics
        self.tensors = tensors or {}
        self.initial_session = initial_session
        if initial_session:
            guard_config = replace(
                config,
                no_long_task_after_s=config.initial_no_long_task_after_s,
                drain_after_s=config.initial_drain_after_s,
                final_save_after_s=config.initial_final_save_after_s,
                hard_return_s=config.initial_hard_return_s,
            )
            self.session_policy = 'INITIAL_TWO_HOUR_LIMIT'
        else:
            guard_config = config
            self.session_policy = 'RESUMED_STANDARD_FIVE_HOUR_THIRTY_LIMIT'
        self.guard = SessionGuard(guard_config)
        self.logger = JsonlLogger(run_root / 'logs' / 'events.jsonl')
        self.last_checkpoint: CheckpointSaveResult | None = None
        self.backend = select_backend(require_cuda=config.require_cuda)

    def _memory_headroom(self, *, checkpoint: bool) -> dict[str, Any]:
        if self.backend.is_cuda:
            self.backend.synchronize()
            if checkpoint:
                torch.cuda.empty_cache()
            snapshot = self.backend.memory_snapshot()
            require_memory_headroom(
                int(snapshot['free_bytes']), int(snapshot['total_bytes']),
                self.config.checkpoint_memory_headroom
                if checkpoint else self.config.normal_memory_headroom,
            )
            return snapshot
        if self.config.require_cuda:
            raise RuntimeError('M4 real run lost its required CUDA backend.')
        return self.backend.memory_snapshot()

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m4_safe()
        memory = self._memory_headroom(checkpoint=True)
        self.state.notes.append(
            f'{utc_now()} checkpoint: {reason}; session_policy={self.session_policy}; '
            f'gpu_free_fraction={memory["free_fraction"]}'
        )
        result = self.checkpoints.save(self.state, self.queue, self.tensors)
        self.guard.mark_checkpoint()
        self.last_checkpoint = result
        self.logger.emit(
            'm4_checkpoint_committed', run_id=self.state.run_id,
            checkpoint=result.index, reason=reason, size_bytes=result.size_bytes,
            session_policy=self.session_policy,
        )
        print(f'M4 checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _next_pending(self) -> WorkItem | None:
        for phase in M4_PHASES:
            candidates = sorted(
                (
                    item for item in self.queue.items.values()
                    if item.phase == phase and item.status == 'pending'
                ),
                key=lambda item: item.item_id,
            )
            if candidates:
                return candidates[0]
        return None

    def _source_result(self) -> dict[str, Any]:
        generators = source_generators()
        residuals = generator_symmetry_residuals(generators)
        maximum = max(residuals.values())
        if maximum > self.config.symmetry_tolerance:
            raise ArithmeticError('M4 source symmetry relation failed.')
        left = self.parent_tensors['triad_left']
        core = self.parent_tensors['triad_core']
        right = self.parent_tensors['triad_right']
        dual = projected_parent_dual(left, core, right, left, generators)
        zero = projected_parent_dual(
            left, core, right, left, generators, source_scale=0.0,
        )
        zero_max = max(
            float(np.max(np.abs(zero.tangent[source])))
            for source in SOURCE_CLASSES
        )
        self.tensors.update(dual.tensor_payload('projected'))
        return {
            'status': 'PASS', 'source_count': len(generators),
            'channels': [source.value for source in SOURCE_CLASSES],
            'generator_norms': {
                source.value: float(np.linalg.norm(generators[source]))
                for source in SOURCE_CLASSES
            },
            'symmetry_residuals': residuals,
            'max_symmetry_residual': maximum,
            'zero_source_max_abs': zero_max,
            'projected_shape': list(dual.shape),
            'tangent_norms': {
                source.value: float(np.linalg.norm(dual.tangent[source], 'fro'))
                for source in SOURCE_CLASSES
            },
        }

    @staticmethod
    def _torch_regroup(value: torch.Tensor) -> torch.Tensor:
        if value.shape[0] != value.shape[1]:
            raise ValueError('M4 GPU regrouping requires a square matrix.')
        leg = int(round(float(value.shape[0]) ** 0.5))
        if leg * leg != int(value.shape[0]):
            raise ValueError('M4 GPU regrouping dimension must be a perfect square.')
        return value.reshape(leg, leg, leg, leg).permute(0, 2, 1, 3).reshape(
            value.shape
        )

    def _pipeline_result(self) -> dict[str, Any]:
        projected = _dual_from_tensors(self.tensors, 'projected')
        coarse = dual_regroup(dual_matmul(projected, projected))
        self.tensors.update(coarse.tensor_payload('coarse'))
        self._memory_headroom(checkpoint=False)
        primal_gpu = self.backend.tensor(projected.primal)
        coarse_gpu = self._torch_regroup(self.backend.matmul(primal_gpu, primal_gpu))
        errors = [
            float(np.max(np.abs(
                self.backend.to_numpy(coarse_gpu) - coarse.primal
            )))
        ]
        for source in SOURCE_CLASSES:
            tangent_gpu = self.backend.tensor(projected.tangent[source])
            value = self._torch_regroup(
                self.backend.matmul(tangent_gpu, primal_gpu)
                + self.backend.matmul(primal_gpu, tangent_gpu)
            )
            errors.append(float(np.max(np.abs(
                self.backend.to_numpy(value) - coarse.tangent[source]
            ))))
        maximum = max(errors)
        if maximum > 1e-12:
            raise ArithmeticError('M4 CPU/GPU forward-AD parity failed.')
        memory = self.backend.memory_snapshot()
        return {
            'status': 'PASS', 'fixed_basis_projection': True,
            'contraction_rule': 'PRODUCT_RULE_COMPLETE',
            'regrouping': True,
            'basis_variation_policy': (
                'FIXED_BASIS_WITH_EXPLICIT_LEDGER_TERM'
            ),
            'projected_shape': list(projected.shape),
            'coarse_shape': list(coarse.shape),
            'cpu_gpu_max_abs_error': maximum,
            'coarse_primal_frobenius_norm': float(
                np.linalg.norm(coarse.primal, 'fro')
            ),
            'coarse_tangent_frobenius_norms': {
                source.value: float(np.linalg.norm(coarse.tangent[source], 'fro'))
                for source in SOURCE_CLASSES
            },
            'backend': backend_selection(self.backend).payload(),
            'tf32_disabled': (
                not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32
            ),
            'gpu_memory': memory,
        }

    def _normalization_result(self) -> dict[str, Any]:
        coarse = _dual_from_tensors(self.tensors, 'coarse')
        normalized, info = normalize_dual(coarse)
        self.tensors.update(normalized.tensor_payload('normalized'))
        all_finite = (
            np.isfinite(normalized.primal).all()
            and all(
                np.isfinite(normalized.tangent[source]).all()
                for source in SOURCE_CLASSES
            )
        )
        if not all_finite:
            raise FloatingPointError('M4 normalized derivative state is nonfinite.')
        return {
            'status': 'PASS', **info.payload(),
            'normalized_frobenius_norm': float(
                np.linalg.norm(normalized.primal, 'fro')
            ),
            'normalized_tangent_frobenius_norms': {
                source.value: float(
                    np.linalg.norm(normalized.tangent[source], 'fro')
                )
                for source in SOURCE_CLASSES
            },
            'all_outputs_finite': bool(all_finite),
            'normalization_lower_bound_rigorous': False,
        }

    def _finite_difference_result(self) -> dict[str, Any]:
        normalized = _dual_from_tensors(self.tensors, 'normalized')
        generators = source_generators()
        left = self.parent_tensors['triad_left']
        core = self.parent_tensors['triad_core']
        right = self.parent_tensors['triad_right']
        channels: dict[str, Any] = {}
        all_converged = True
        final_relative: list[float] = []
        for source in SOURCE_CLASSES:
            errors: list[dict[str, float]] = []
            analytic = normalized.tangent[source]
            scale = max(float(np.linalg.norm(analytic, 'fro')), 1e-300)
            for step in self.config.finite_difference_steps:
                plus = _array_pipeline(deformed_projected_parent(
                    left, core, right, left, generators[source], step,
                ))
                minus = _array_pipeline(deformed_projected_parent(
                    left, core, right, left, generators[source], -step,
                ))
                finite_difference = (plus - minus) / (2.0 * step)
                absolute = float(np.linalg.norm(finite_difference - analytic, 'fro'))
                relative = absolute / scale
                errors.append({
                    'step': step, 'absolute_error_frobenius': absolute,
                    'relative_error_frobenius': relative,
                })
            relative_errors = [
                item['relative_error_frobenius'] for item in errors
            ]
            converged = (
                all(
                    later <= earlier * 1.05
                    for earlier, later in zip(
                        relative_errors, relative_errors[1:]
                    )
                )
                and relative_errors[-1]
                <= self.config.finite_difference_relative_tolerance
            )
            all_converged = all_converged and converged
            final_relative.append(relative_errors[-1])
            channels[source.value] = {
                'converged': converged, 'steps': errors,
                'final_relative_error': relative_errors[-1],
                'final_absolute_error': errors[-1]['absolute_error_frobenius'],
            }
        if not all_converged:
            raise ArithmeticError('M4 finite-difference regression failed.')
        return {
            'status': 'PASS', 'channels': channels,
            'all_channels_converged': all_converged,
            'max_final_relative_error': max(final_relative),
            'finite_difference_is_proof_bound': False,
            'interpretation': 'REGRESSION_ONLY_NOT_A_DETERMINISTIC_BOUND',
        }

    def _ledger_result(self) -> dict[str, Any]:
        phase_results = load_m4_phase_results(self.run_root, self.queue)
        difference = phase_results['M4_FINITE_DIFFERENCE']['result']
        fd_estimate = max(
            value['final_absolute_error']
            for value in difference['channels'].values()
        )
        parent = self.config.parent_checkpoint_path
        current = str(
            self.last_checkpoint.path if self.last_checkpoint else self.run_root
        )
        ledger = ErrorLedger()
        representation = ledger.add_leaf(
            name='initial representation tail',
            category='initial_representation_tail', applies_to='both',
            source_checkpoint=parent,
            formula='M4 j2_max=1 tail must be transferred into the chosen operator norm',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='M1 tail bounds are not silently reused in an incompatible M4 norm.',
        )
        equivalence = ledger.add_leaf(
            name='M2 dense-armillary basis equivalence',
            category='basis_equivalence_error', applies_to='both',
            source_checkpoint=parent, formula='exact symbolic equality => radius 0',
            estimate=0.0, deterministic_upper_bound=0.0, rigor='RIGOROUS',
            note='The accepted low-cutoff M2 equality is exact.',
        )
        input_radius = ledger.add_leaf(
            name='input radius propagation',
            category='input_radius_propagation', applies_to='both',
            source_checkpoint=parent,
            formula='multilinear product of input norms plus input radii',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='M3 provides no validated input ball.',
        )
        rounding = ledger.add_leaf(
            name='GPU rounding and backward error',
            category='gpu_rounding_backward', applies_to='both',
            source_checkpoint=current,
            formula='deterministic FP64 contraction backward-error bound',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='CPU/GPU agreement is regression evidence, not an enclosure.',
        )
        rsvd = ledger.add_leaf(
            name='M3 RSVD projection residual',
            category='rsvd_projection_residual', applies_to='both',
            source_checkpoint=parent,
            formula='||(I-QQ*)A||_F diagnostic',
            estimate=float(self.parent_metrics['rsvd_residual_frobenius']),
            deterministic_upper_bound=None, rigor='HEURISTIC',
            note='The fixed-seed residual is not a deterministic upper bound.',
        )
        basis_variation = ledger.add_alias(
            name='fixed-basis variation residual',
            category='basis_variation', applies_to='tangent', parent=rsvd,
            source_checkpoint=current,
            formula='basis variation is charged to the projection-residual branch',
            note='Basis variation is explicit and is never set to zero.',
        )
        omitted = ledger.add_leaf(
            name='omitted fusion and channel tail',
            category='omitted_fusion_channel_tail', applies_to='both',
            source_checkpoint=parent,
            formula='sum of all sectors/channels omitted beyond j2_max=1',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='No four-dimensional omitted-channel bound exists yet.',
        )
        normalization = ledger.add_leaf(
            name='normalization and denominator error',
            category='normalization_error', applies_to='both',
            source_checkpoint=current,
            formula='requires a rigorous positive lower bound for lambda',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='The computed positive FP64 norm is not a rigorous lower bound.',
        )
        tangent = ledger.add_leaf(
            name='forward tangent regression residual',
            category='tangent_error', applies_to='tangent',
            source_checkpoint=current,
            formula='max centered finite-difference discrepancy at smallest h',
            estimate=float(fd_estimate), deterministic_upper_bound=None,
            rigor='HEURISTIC',
            note='Finite difference is used only for regression.',
        )
        cutoff = ledger.add_leaf(
            name='cutoff and rank dependence',
            category='cutoff_rank_dependence', applies_to='both',
            source_checkpoint=parent,
            formula='variation under larger j2_max and target rank',
            estimate=None, deterministic_upper_bound=None, rigor='MISSING',
            note='Required because the M3 influence proxy is near one.',
        )
        primal = ledger.add_sum(
            name='primal output partial radius', category='output_radius',
            applies_to='primal',
            parents=(representation, equivalence, input_radius, rounding, rsvd,
                     omitted, normalization, cutoff),
            source_checkpoint=current, formula='sum of unique primal error leaves',
            note='Finite partial estimate only; not an enclosure while bounds are missing.',
        )
        tangent_output = ledger.add_sum(
            name='tangent output partial radius', category='output_radius',
            applies_to='tangent',
            parents=(representation, equivalence, input_radius, rounding,
                     basis_variation, omitted, normalization, tangent, cutoff),
            source_checkpoint=current, formula='sum of unique tangent error leaves',
            note='Finite partial estimate only; not an enclosure while bounds are missing.',
        )
        ledger_payload = ledger.payload()
        summary = ledger_payload['summary']
        categories = {
            term['category'] for term in ledger_payload['terms']
        }
        required = {
            'initial_representation_tail', 'basis_equivalence_error',
            'input_radius_propagation', 'gpu_rounding_backward',
            'rsvd_projection_residual', 'basis_variation',
            'omitted_fusion_channel_tail', 'normalization_error',
            'tangent_error', 'cutoff_rank_dependence', 'output_radius',
        }
        aggregates = {
            'primal': {
                'term_id': primal,
                'partial_estimate': ledger.terms[primal].estimate,
                'deterministic_upper_bound': (
                    ledger.terms[primal].deterministic_upper_bound
                ),
                'rigor': ledger.terms[primal].rigor,
            },
            'tangent': {
                'term_id': tangent_output,
                'partial_estimate': ledger.terms[tangent_output].estimate,
                'deterministic_upper_bound': (
                    ledger.terms[tangent_output].deterministic_upper_bound
                ),
                'rigor': ledger.terms[tangent_output].rigor,
            },
        }
        return {
            'status': 'PASS', 'milestone_status': 'BLOCKED_MATH',
            'enclosure_status': 'BLOCKED_MATH',
            'ledger': ledger_payload, 'summary': summary,
            'required_categories_complete': required <= categories,
            'basis_variation_accounted': (
                ledger.terms[basis_variation].estimate
                == ledger.terms[rsvd].estimate
            ),
            'aggregates': aggregates,
        }

    def _report_result(self) -> dict[str, Any]:
        results = load_m4_phase_results(self.run_root, self.queue)
        required = set(M4_PHASES) - {'M4_REPORT'}
        if not required <= set(results):
            raise RuntimeError('M4 report phase lacks prerequisite artifacts.')
        if results['M4_ERROR_LEDGER']['result']['enclosure_status'] != 'BLOCKED_MATH':
            raise RuntimeError('M4 report attempted to bypass missing error bounds.')
        return {'status': 'READY', 'input_phases': sorted(required)}

    def _phase_result(self, phase: str) -> dict[str, Any]:
        functions = {
            'M4_SOURCE_CHANNELS': self._source_result,
            'M4_DUAL_PIPELINE': self._pipeline_result,
            'M4_NORMALIZATION': self._normalization_result,
            'M4_FINITE_DIFFERENCE': self._finite_difference_result,
            'M4_ERROR_LEDGER': self._ledger_result,
            'M4_REPORT': self._report_result,
        }
        return functions[phase]()

    def _execute_item(self, item: WorkItem) -> tuple[str, str]:
        result = self._phase_result(item.phase)
        parent = self.run_root / 'artifacts' / item.item_id
        final = parent / 'committed'
        temporary = parent / f'.tmp-{uuid.uuid4().hex}'
        parent.mkdir(parents=True, exist_ok=True)
        if final.exists():
            shutil.rmtree(final)
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            atomic_write_json(temporary / 'result.json', {
                'schema_version': 1, 'milestone': 'M4',
                'phase': item.phase, 'item_id': item.item_id,
                'config_hash': self.config.config_hash(),
                'milestone_status': 'BLOCKED_MATH',
                'certification_status': 'NOT_CERTIFIED',
                'generated_at': utc_now(), 'result': result,
            })
            fsync_directory(temporary)
            os.replace(temporary, final)
            fsync_directory(parent)
            committed = final / 'result.json'
            relative = committed.relative_to(self.run_root).as_posix()
            digest = sha256_file(committed)
            atomic_write_json(
                self.run_root / 'work_items' / f'{item.item_id}.done',
                {
                    'item_id': item.item_id, 'result_relpath': relative,
                    'result_sha256': digest, 'completed_at': utc_now(),
                },
            )
            if sha256_file(committed) != digest:
                raise RuntimeError('M4 result verification failed after commit.')
            return relative, digest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def run_one_item_for_test(self) -> str | None:
        item = self._next_pending()
        if item is None:
            return None
        item.attempts += 1
        item.status = 'running'
        self.checkpoint(f'before test item {item.phase}')
        relative, digest = self._execute_item(item)
        item.result_relpath = relative
        item.result_sha256 = digest
        item.status = 'done'
        self.checkpoint(f'after test item {item.phase}')
        return item.phase

    def _summary(self, reason: str) -> dict[str, Any]:
        artifacts = write_m4_session_artifacts(
            self.run_root, self.state, self.queue, reason,
            self.guard.elapsed_s(), self.guard.remaining_s(),
            self.persistent_root, self.project_root, self.session_policy,
        )
        summary = {
            'milestone': 'M4', 'run_id': self.state.run_id,
            'phase': self.state.phase, 'milestone_status': 'BLOCKED_MATH',
            'certification_status': 'NOT_CERTIFIED',
            'checkpoint_index': self.state.checkpoint_index,
            'session_policy': self.session_policy,
            'stop_reason': reason, 'elapsed_s': self.guard.elapsed_s(),
            'remaining_s': self.guard.remaining_s(),
            'session_artifacts': artifacts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m4_safe()
        acceptance = self.run_root / 'reports' / 'M4_acceptance.json'
        if self.state.phase == 'M4_COMPLETE' and acceptance.is_file():
            return self._summary('M4 already complete; no work was started')
        self.state.phase = 'M4_RUNNING'
        self.logger.emit(
            'm4_session_started', run_id=self.state.run_id,
            session_policy=self.session_policy,
        )
        while True:
            session_state = self.guard.state()
            if session_state is SessionState.RETURN:
                return self._summary('hard return threshold reached')
            if session_state in {SessionState.DRAIN, SessionState.FINAL_SAVE}:
                self.checkpoint(f'M4 session state {session_state.value}')
                return self._summary(f'{session_state.value.lower()} checkpoint complete')
            if self.guard.checkpoint_due():
                self.checkpoint('periodic 15-minute M4 checkpoint')
            item = self._next_pending()
            if item is None:
                if any(
                    queued.status != 'done' for queued in self.queue.items.values()
                ):
                    self.checkpoint('M4 queue incomplete with no runnable item')
                    raise RuntimeError('M4 queue cannot complete.')
                results = load_m4_phase_results(self.run_root, self.queue)
                validate_m4_acceptance(
                    self.state, self.queue, results, self.test_report,
                )
                self.state.bounds = {
                    'source_derivative': 'FP64_REGRESSION_ONLY',
                    'basis_variation': 'MISSING_DETERMINISTIC_BOUND',
                    'error_ledger': 'COMPLETE_PROVENANCE_BLOCKED_MATH',
                    'normalization': 'MISSING_RIGOROUS_LOWER_BOUND',
                    'cutoff_rank': 'INVESTIGATE_CUTOFF_AND_RANK',
                }
                self.state.phase = 'M4_COMPLETE'
                final_checkpoint = self.checkpoint('M4 implementation gates complete')
                paths = write_m4_report_package(
                    self.run_root, self.config, self.state, self.queue,
                    self.test_report, final_checkpoint, self.manifest,
                )
                self.logger.emit(
                    'm4_milestone_complete', run_id=self.state.run_id,
                    milestone_status='BLOCKED_MATH', reports=paths,
                )
                return self._summary(
                    f'M4 complete but BLOCKED_MATH; report at {paths["json"]}'
                )
            if not self.guard.may_start(self.queue.predicted_duration(item)):
                self.checkpoint('insufficient safe time for next M4 item')
                return self._summary('next M4 item deferred to a fresh session')
            item.attempts += 1
            if item.attempts > self.config.max_item_attempts:
                item.status = 'failed'
                item.last_error = 'Maximum M4 attempt count exceeded.'
                self.checkpoint('M4 item exceeded attempt limit')
                raise RuntimeError(item.last_error)
            item.status = 'running'
            self.checkpoint(f'before M4 item {item.phase}')
            started = time.monotonic()
            try:
                relative, digest = self._execute_item(item)
                item.result_relpath = relative
                item.result_sha256 = digest
                item.status = 'done'
                item.last_error = None
                self.queue.record_timing(item.phase, time.monotonic() - started)
                if self.guard.state() is SessionState.RETURN:
                    return self._summary(
                        'hard return after atomic marker; resume repairs queue'
                    )
                self.checkpoint(f'after M4 item {item.phase}')
            except KeyboardInterrupt:
                item.status = 'pending'
                item.last_error = 'KeyboardInterrupt'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'interrupted M4 item {item.phase}')
                self._summary(f'KeyboardInterrupt in {item.phase}')
                raise
            except Exception as exc:
                item.status = (
                    'failed' if item.attempts >= self.config.max_item_attempts
                    else 'pending'
                )
                item.last_error = f'{type(exc).__name__}: {exc}'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'exception in M4 item {item.phase}')
                self._summary(
                    f'exception in {item.phase}: {type(exc).__name__}'
                )
                raise


def create_or_resume_m4(
    persistent_root: Path, config: M4Config, project_root: Path,
    run_id: str | None = None, test_report: dict[str, Any] | None = None,
) -> M4Orchestrator:
    try:
        evidence = verify_accepted_m3_parent(project_root, config)
    except M4ParentError as exc:
        raise M4CompatibilityError(str(exc)) from exc
    try:
        probe_backend = select_backend(require_cuda=config.require_cuda)
    except BackendUnavailableError as exc:
        raise M4CompatibilityError(str(exc)) from exc
    config_hash = config.config_hash()
    source_hash = hash_tree(project_root / 'src')
    notebook_hash = _notebook_hash(project_root)
    environment = environment_info()
    runtime_signature = runtime_compatibility_signature(environment)
    selection = backend_selection(probe_backend).payload()
    del probe_backend
    runs_root = persistent_root / 'runs'; runs_root.mkdir(parents=True, exist_ok=True)
    requested = run_id or os.environ.get('VALIDATED_RG_M4_RUN_ID')
    latest_pointer = persistent_root / 'LATEST_M4_RUN.json'
    if requested is None and latest_pointer.is_file():
        pointer = read_json(latest_pointer)
        if isinstance(pointer, dict) and pointer.get('config_hash') == config_hash:
            requested = pointer.get('run_id')
    if requested is None:
        requested = (
            'M4-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            + '-' + config_hash[:12]
        )
    if not isinstance(requested, str):
        raise M4CompatibilityError('M4 run ID must be a string.')
    safe_component(requested)
    if not requested.startswith('M4-'):
        raise M4CompatibilityError('M4 run ID must use the M4 namespace.')
    run_root = runs_root / requested
    manifest_path = run_root / 'run_manifest.json'
    immutable = {
        'milestone': 'M4', 'parent_milestone': config.parent_milestone,
        'parent_run_id': config.parent_run_id,
        'parent_checkpoint': config.parent_checkpoint,
        'parent_checkpoint_path': config.parent_checkpoint_path,
        **evidence.hashes, 'config_hash': config_hash,
        'source_hash': source_hash, 'notebook_hash': notebook_hash,
        'notebook_hash_policy': M4_NOTEBOOK_HASH_POLICY,
        'governing_document_hashes': governing_document_hashes(project_root),
        'reference_artifact_hashes': reference_artifact_hashes(project_root),
        'runtime_compatibility': runtime_signature,
        'backend_selection': selection,
        'initial_session_policy': {
            'final_save_after_s': config.initial_final_save_after_s,
            'hard_return_s': config.initial_hard_return_s,
        },
        'resumed_session_policy': {
            'final_save_after_s': config.final_save_after_s,
            'hard_return_s': config.hard_return_s,
        },
        'milestone_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
    }
    if run_root.exists() and (run_root / 'run_config.json').is_file():
        if read_json(run_root / 'run_config.json') != config.canonical_payload():
            raise M4CompatibilityError('Immutable M4 config changed.')
        manifest = read_json(manifest_path) if manifest_path.is_file() else None
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in immutable.items()
        ):
            raise M4CompatibilityError('M4 manifest/source/parent/runtime changed.')
        report_path = run_root / 'test_report.json'
        if test_report is None:
            effective_report = read_json(report_path) if report_path.is_file() else {}
        else:
            effective_report = test_report
            atomic_write_json(report_path, test_report)
        manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise M4CompatibilityError('Existing M4 run has no valid checkpoint.')
        repaired = loaded.queue.recover_interrupted(run_root)
        orchestrator = M4Orchestrator(
            persistent_root, run_root, project_root, config,
            loaded.state, loaded.queue, manager, effective_report,
            manifest, evidence.tensors, evidence.metrics,
            initial_session=False, tensors=loaded.tensors,
        )
        if repaired:
            orchestrator.checkpoint(
                f'recovered {len(repaired)} interrupted M4 item(s)'
            )
        print('Resumed M4 from:', loaded.path)
        return orchestrator
    if run_root.exists():
        raise M4CompatibilityError('Incomplete M4 run directory exists.')
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ('logs', 'reports', 'artifacts', 'work_items', 'checkpoints'):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'run_id': requested, 'created_at': utc_now(),
        'environment': environment,
        'source_channels': [source.value for source in SOURCE_CLASSES],
        'rounding_policy': 'FP64 exploratory derivative; no float is a proof bound',
        **immutable,
    }
    atomic_write_json(run_root / 'run_config.json', config.canonical_payload())
    atomic_write_json(run_root / 'test_report.json', test_report or {})
    atomic_write_json(manifest_path, manifest)
    _seed_everything()
    state = RunState(
        requested, config_hash, utc_now(), utc_now(),
        milestone='M4', phase='M4_BOOTSTRAP',
    )
    state.notes.append(
        'Initial M4 session uses the user-requested two-hour hard-return policy.'
    )
    queue = WorkQueue()
    for phase in M4_PHASES:
        queue.add(
            phase, config_hash, {'milestone': 'M4', 'phase': phase},
            predicted_s=5.0 * 60.0,
        )
    manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
    orchestrator = M4Orchestrator(
        persistent_root, run_root, project_root, config,
        state, queue, manager, test_report or {}, manifest,
        evidence.tensors, evidence.metrics, initial_session=True,
    )
    orchestrator.checkpoint('initial M4 run state')
    atomic_write_json(latest_pointer, {
        'milestone': 'M4', 'run_id': requested,
        'config_hash': config_hash, 'updated_at': utc_now(),
    })
    return orchestrator
