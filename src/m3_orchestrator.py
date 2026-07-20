from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
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
from .gpu_sharding import (
    BlockedResourceError, require_memory_headroom, run_with_oom_recovery,
)
from .linear_operator import ArmillaryLinearOperator, build_armillary_operator
from .m3_config import M3Config
from .m3_parent import M3ParentError, verify_accepted_m2_parent
from .m3_reporting import (
    M3_PHASES, load_m3_phase_results, validate_m3_acceptance,
    write_m3_report_package, write_m3_session_artifacts,
)
from .orchestrator import governing_document_hashes, reference_artifact_hashes
from .reporting import JsonlLogger
from .rsvd import RSVDResult, influence_proxy, randomized_svd
from .runtime import environment_info, runtime_compatibility_signature
from .session_guard import SessionGuard, SessionState
from .triad_atrg import triad_from_rsvd
from .work_queue import WorkItem, WorkQueue

M3_NOTEBOOK_HASH_POLICY = 'canonical_nbformat4_cell_type_source_tags_v1'


class M3CompatibilityError(RuntimeError):
    '''Raised when an M3 run cannot be created or resumed exactly.'''


def _seed_everything(seed: int) -> None:
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
    path = project_root / 'notebooks/40_m3_gpu_triad_atrg.ipynb'
    if not path.is_file():
        raise M3CompatibilityError('M3 user-facing notebook is missing.')
    payload = read_json(path)
    cells = payload.get('cells') if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get('nbformat') != 4 or not isinstance(cells, list):
        raise M3CompatibilityError('M3 notebook must be a valid nbformat 4 document.')
    identity_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise M3CompatibilityError(f'M3 notebook cell {index} is not a mapping.')
        cell_type = cell.get('cell_type')
        source = cell.get('source')
        metadata = cell.get('metadata', {})
        if cell_type not in {'code', 'markdown', 'raw'}:
            raise M3CompatibilityError(f'M3 notebook cell {index} has an invalid type.')
        if isinstance(source, str):
            normalized = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            normalized = ''.join(source)
        else:
            raise M3CompatibilityError(f'M3 notebook cell {index} has invalid source.')
        if not isinstance(metadata, dict):
            raise M3CompatibilityError(f'M3 notebook cell {index} has invalid metadata.')
        tags = metadata.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise M3CompatibilityError(f'M3 notebook cell {index} has invalid tags.')
        identity_cells.append({
            'cell_type': cell_type, 'source': normalized, 'tags': tags,
        })
    return sha256_bytes(canonical_json_bytes({
        'policy': M3_NOTEBOOK_HASH_POLICY, 'nbformat': 4,
        'cells': identity_cells,
    }))


class M3Orchestrator:
    def __init__(
        self, persistent_root: Path, run_root: Path, project_root: Path,
        config: M3Config, state: RunState, queue: WorkQueue,
        checkpoints: CheckpointManager, test_report: dict[str, Any],
        manifest: dict[str, Any], parent_tensors: dict[str, np.ndarray],
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
        self.tensors = tensors or {}
        self.guard = SessionGuard(config)
        self.logger = JsonlLogger(run_root / 'logs' / 'events.jsonl')
        self.last_checkpoint: CheckpointSaveResult | None = None
        self.backend: ContractionBackend = select_backend(
            require_cuda=config.require_cuda,
        )
        (run_root / 'cache').mkdir(parents=True, exist_ok=True)
        self.path_cache_path = run_root / 'cache' / 'contraction_paths.json'

    def _memory_headroom(self, *, checkpoint: bool) -> dict[str, Any]:
        if self.backend.is_cuda:
            torch.cuda.synchronize(self.backend.device)
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
            raise BlockedResourceError('M3 real run lost its required CUDA backend.')
        return self.backend.memory_snapshot()

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m3_safe()
        memory = self._memory_headroom(checkpoint=True)
        self.state.notes.append(
            f'{utc_now()} checkpoint: {reason}; gpu_free_fraction={memory["free_fraction"]}'
        )
        result = self.checkpoints.save(self.state, self.queue, self.tensors)
        self.guard.mark_checkpoint()
        self.last_checkpoint = result
        self.logger.emit(
            'm3_checkpoint_committed', run_id=self.state.run_id,
            checkpoint=result.index, reason=reason, size_bytes=result.size_bytes,
        )
        print(f'M3 checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _operator(self, sectors_per_shard: int | None = None) -> ArmillaryLinearOperator:
        return build_armillary_operator(
            self.parent_tensors, self.backend, self.path_cache_path,
            sectors_per_shard=(
                sectors_per_shard or self.config.initial_sector_shard_size
            ),
        )

    def _next_pending(self) -> WorkItem | None:
        for phase in M3_PHASES:
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

    def _backend_result(self) -> dict[str, Any]:
        before = self.backend.memory_snapshot()
        self._memory_headroom(checkpoint=False)
        probe = self.backend.tensor(np.eye(4, dtype=np.float64))
        product = self.backend.matmul(probe, probe)
        self.backend.synchronize()
        if not np.array_equal(
            self.backend.to_numpy(product), np.eye(4, dtype=np.float64),
        ):
            raise ArithmeticError('M3 backend FP64 identity probe failed.')
        del probe, product
        after = self.backend.memory_snapshot()
        return {
            'status': 'PASS', 'selection': backend_selection(self.backend).payload(),
            'memory_before': before, 'memory_after': after,
            'tf32_disabled': (
                not torch.backends.cuda.matmul.allow_tf32
                and not torch.backends.cudnn.allow_tf32
            ),
        }

    def _operator_result(self) -> dict[str, Any]:
        operator = self._operator()
        metadata = operator.metadata()
        return {
            'status': 'PASS', 'dimension': operator.dimension,
            'sector_count': len(operator.blocks),
            'parent_tensor_count': len(self.parent_tensors),
            'operator_frobenius_norm': operator.frobenius_norm(),
            'metadata': metadata,
        }

    def _validation_result(self) -> dict[str, Any]:
        operator = self._operator()
        rng = np.random.default_rng(self.config.seed)
        x = rng.standard_normal(operator.dimension)
        y = rng.standard_normal(operator.dimension)
        explicit = operator.explicit_matrix()
        started = time.monotonic()
        matrix_free = operator.matvec(x)
        self.backend.synchronize()
        matvec_s = time.monotonic() - started
        error = float(np.max(np.abs(matrix_free - explicit @ x)))
        left = float(np.vdot(matrix_free, y))
        right = float(np.vdot(x, operator.rmatvec(y)))
        adjoint_error = abs(left - right) / max(1.0, abs(left), abs(right))
        stats_before = operator.path_cache.stats()
        operator.matvec(x)
        stats_after = operator.path_cache.stats()
        reused = (
            stats_after['entries'] == stats_before['entries']
            and stats_after['hits'] > stats_before['hits']
        )
        if (
            not np.isfinite(error) or error > 1e-12
            or not np.isfinite(adjoint_error) or adjoint_error > 1e-12
            or not reused
        ):
            raise ArithmeticError('M3 matrix-free validation failed closed.')
        return {
            'status': 'PASS', 'dimension': operator.dimension,
            'matvec_max_abs_error': error,
            'adjoint_relative_error': float(adjoint_error),
            'matvec_elapsed_s': matvec_s,
            'path_cache_reused': reused,
            'path_cache_before': stats_before,
            'path_cache_after': stats_after,
            'explicit_matrix_bytes': explicit.nbytes,
            'gpu_memory_after': self.backend.memory_snapshot(),
        }

    def _rsvd_result(self) -> dict[str, Any]:
        selected_operator: ArmillaryLinearOperator | None = None

        def compute(shard_size: int) -> RSVDResult:
            nonlocal selected_operator
            self._memory_headroom(checkpoint=False)
            selected_operator = self._operator(shard_size)
            return randomized_svd(
                selected_operator, target_rank=self.config.target_rank,
                oversampling=self.config.oversampling,
                power_iterations=self.config.power_iterations,
                seed=self.config.seed,
            )

        recovered = run_with_oom_recovery(
            compute, self.config.initial_sector_shard_size,
            min_shard_size=self.config.min_sector_shard_size,
            max_oom_retries=self.config.max_oom_retries,
        )
        result = recovered.value
        if selected_operator is None:
            raise RuntimeError('M3 RSVD failed to retain its operator.')
        explicit_values = np.sort(np.concatenate([
            np.linalg.svd(
                block.weight * block.projector, compute_uv=False,
            )
            for block in selected_operator.blocks
        ]))[::-1]
        explicit_error = float(np.max(np.abs(
            result.singular_values - explicit_values[:self.config.target_rank]
        )))
        optimal_residual = float(np.linalg.norm(
            explicit_values[self.config.target_rank:]
        ))
        ratio = (
            result.residual_frobenius / optimal_residual
            if optimal_residual > 0.0 else 1.0
        )
        if (
            not np.isfinite(explicit_error) or explicit_error > 1e-5
            or not np.isfinite(ratio) or ratio > 1.00001
        ):
            raise ArithmeticError('M3 RSVD explicit comparison failed closed.')
        self.tensors.update(result.tensors())
        summary = result.summary()
        return {
            'status': 'PASS', 'milestone_status': 'CORE_REPRODUCED',
            **summary,
            'explicit_top_singular_max_abs_error': explicit_error,
            'explicit_optimal_residual_frobenius': optimal_residual,
            'residual_to_explicit_optimal_ratio': ratio,
            'influence_proxy': influence_proxy(result),
            'final_sector_shard_size': recovered.final_shard_size,
            'oom_retries': recovered.oom_retries,
            'attempted_shard_sizes': list(recovered.attempted_shard_sizes),
            'gpu_memory_after': self.backend.memory_snapshot(),
        }

    def _triad_result(self) -> dict[str, Any]:
        required = {'rsvd_left', 'rsvd_singular_values', 'rsvd_right_t'}
        if not required.issubset(self.tensors):
            raise RuntimeError('M3 Triad phase is missing checkpointed RSVD tensors.')
        rsvd_phase = load_m3_phase_results(
            self.run_root, self.queue,
        ).get('M3_RSVD', {}).get('result', {})
        result = RSVDResult(
            np.asarray(self.tensors['rsvd_left']),
            np.asarray(self.tensors['rsvd_singular_values']),
            np.asarray(self.tensors['rsvd_right_t']),
            self.config.seed, self.config.target_rank,
            self.config.oversampling, self.config.power_iterations,
            float(rsvd_phase.get('elapsed_s', 0.0)),
            float(rsvd_phase.get('orthogonality_residual', 0.0)),
            float(rsvd_phase.get('residual_frobenius', 0.0)),
            float(rsvd_phase.get('relative_residual_frobenius', 0.0)),
        )
        triad = triad_from_rsvd(self._operator(), result)
        self.tensors.update(triad.tensors())
        return {'status': 'PASS', **triad.summary()}

    def _compute_phase(self, item: WorkItem) -> dict[str, Any]:
        if item.phase == 'M3_BACKEND_DIAGNOSTIC':
            return self._backend_result()
        if item.phase == 'M3_OPERATOR_BUILD':
            return self._operator_result()
        if item.phase == 'M3_MATRIX_FREE_VALIDATE':
            return self._validation_result()
        if item.phase == 'M3_RSVD':
            return self._rsvd_result()
        if item.phase == 'M3_TRIAD':
            return self._triad_result()
        if item.phase == 'M3_REPORT':
            required = M3_PHASES[:-1]
            results = load_m3_phase_results(self.run_root, self.queue)
            missing = [phase for phase in required if phase not in results]
            if missing:
                raise RuntimeError(f'M3 report work item is missing inputs: {missing}')
            return {'status': 'READY', 'input_phases': list(required)}
        raise RuntimeError(f'Unknown M3 phase: {item.phase}')

    def _execute_item(self, item: WorkItem) -> tuple[str, str]:
        parent = self.run_root / 'artifacts' / item.item_id
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f'.tmp-attempt-{item.attempts:03d}-{uuid.uuid4().hex}'
        final = parent / f'attempt_{item.attempts:03d}'
        if final.exists():
            raise RuntimeError(f'M3 attempt output exists: {final}')
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            result = self._compute_phase(item)
            result_file = temporary / 'result.json'
            atomic_write_json(result_file, {
                'schema_version': 1, 'milestone': 'M3',
                'phase': item.phase, 'item_id': item.item_id,
                'config_hash': self.config.config_hash(),
                'milestone_status': (
                    'CORE_REPRODUCED'
                    if item.phase in {'M3_RSVD', 'M3_TRIAD', 'M3_REPORT'}
                    else 'EXPLORATORY'
                ),
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
                raise RuntimeError('M3 result verification failed after commit.')
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
        elapsed = self.guard.elapsed_s()
        remaining = self.guard.remaining_s()
        artifacts = write_m3_session_artifacts(
            self.run_root, self.state, self.queue, reason,
            elapsed, remaining, self.persistent_root, self.project_root,
        )
        summary = {
            'milestone': 'M3', 'run_id': self.state.run_id,
            'phase': self.state.phase,
            'milestone_status': (
                'CORE_REPRODUCED'
                if self.state.phase == 'M3_COMPLETE' else 'EXPLORATORY'
            ),
            'certification_status': 'NOT_CERTIFIED',
            'checkpoint_index': self.state.checkpoint_index,
            'stop_reason': reason, 'elapsed_s': elapsed,
            'remaining_s': remaining, 'session_artifacts': artifacts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m3_safe()
        acceptance = self.run_root / 'reports' / 'M3_acceptance.json'
        if self.state.phase == 'M3_COMPLETE' and acceptance.is_file():
            return self._summary('M3 already complete; no work was started')
        self.state.phase = 'M3_RUNNING'
        self.logger.emit('m3_session_started', run_id=self.state.run_id)
        while True:
            session_state = self.guard.state()
            if session_state is SessionState.RETURN:
                return self._summary('hard return threshold reached')
            if session_state in {SessionState.DRAIN, SessionState.FINAL_SAVE}:
                self.checkpoint(f'M3 session state {session_state.value}')
                return self._summary(f'{session_state.value.lower()} checkpoint complete')
            if self.guard.checkpoint_due():
                self.checkpoint('periodic 15-minute M3 checkpoint')
            item = self._next_pending()
            if item is None:
                incomplete = [
                    queued for queued in self.queue.items.values()
                    if queued.status != 'done'
                ]
                if incomplete:
                    self.checkpoint('M3 queue is incomplete with no runnable item')
                    raise RuntimeError('M3 queue cannot complete.')
                results = load_m3_phase_results(self.run_root, self.queue)
                validate_m3_acceptance(
                    self.state, self.queue, results, self.test_report,
                )
                self.state.bounds = {
                    'matrix_free_validation': 'FP64_CORE_REPRODUCED',
                    'adjoint_consistency': 'FP64_CORE_REPRODUCED',
                    'rsvd': 'EXPLORATORY_NOT_A_BOUND',
                    'triad_residual': 'EXPLORATORY_NOT_A_BOUND',
                    'influence_proxy': 'HEURISTIC_NOT_A_BOUND',
                }
                self.state.phase = 'M3_COMPLETE'
                final_checkpoint = self.checkpoint('M3 acceptance gates complete')
                paths = write_m3_report_package(
                    self.run_root, self.config, self.state, self.queue,
                    self.test_report, final_checkpoint, self.manifest,
                )
                self.logger.emit(
                    'm3_milestone_complete', run_id=self.state.run_id,
                    reports=paths,
                )
                return self._summary(
                    f'M3 complete; report written to {paths["json"]}',
                )
            predicted = self.queue.predicted_duration(item)
            if not self.guard.may_start(predicted):
                self.checkpoint('insufficient safe time for next M3 item')
                return self._summary('next M3 item deferred to a fresh session')
            item.attempts += 1
            if item.attempts > self.config.max_item_attempts:
                item.status = 'failed'
                item.last_error = 'Maximum M3 attempt count exceeded.'
                self.checkpoint('M3 item exceeded attempt limit')
                raise RuntimeError(item.last_error)
            item.status = 'running'
            self.checkpoint(f'before M3 item {item.phase}')
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
                        'hard return after atomic M3 marker; resume will repair queue',
                    )
                self.checkpoint(f'after M3 item {item.phase}')
            except BlockedResourceError as exc:
                item.status = 'blocked_resource'
                item.last_error = f'{type(exc).__name__}: {exc}'
                self.checkpoint(f'M3 resource blocked in {item.phase}')
                return self._summary(item.last_error)
            except KeyboardInterrupt:
                item.status = 'pending'
                item.last_error = 'KeyboardInterrupt'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'interrupted M3 item {item.phase}')
                self._summary(f'KeyboardInterrupt in {item.phase}')
                raise
            except Exception as exc:
                item.status = (
                    'failed'
                    if item.attempts >= self.config.max_item_attempts
                    else 'pending'
                )
                item.last_error = f'{type(exc).__name__}: {exc}'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'exception in M3 item {item.phase}')
                self._summary(
                    f'exception in M3 item {item.phase}: {type(exc).__name__}',
                )
                raise


def create_or_resume_m3(
    persistent_root: Path, config: M3Config, project_root: Path,
    run_id: str | None = None, test_report: dict[str, Any] | None = None,
) -> M3Orchestrator:
    try:
        evidence = verify_accepted_m2_parent(project_root, config)
    except M3ParentError as exc:
        raise M3CompatibilityError(str(exc)) from exc
    try:
        probe_backend = select_backend(require_cuda=config.require_cuda)
    except BackendUnavailableError as exc:
        raise M3CompatibilityError(str(exc)) from exc
    config_hash = config.config_hash()
    source_hash = hash_tree(project_root / 'src')
    notebook_hash = _notebook_hash(project_root)
    document_hashes = governing_document_hashes(project_root)
    reference_hashes = reference_artifact_hashes(project_root)
    environment = environment_info()
    runtime_signature = runtime_compatibility_signature(environment)
    selection = backend_selection(probe_backend).payload()
    del probe_backend
    runs_root = persistent_root / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)
    requested = run_id or os.environ.get('VALIDATED_RG_M3_RUN_ID')
    latest_pointer = persistent_root / 'LATEST_M3_RUN.json'
    if requested is None and latest_pointer.is_file():
        pointer = read_json(latest_pointer)
        if isinstance(pointer, dict) and pointer.get('config_hash') == config_hash:
            requested = pointer.get('run_id')
    if requested is None:
        requested = (
            'M3-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            + '-' + config_hash[:12]
        )
    if not isinstance(requested, str):
        raise M3CompatibilityError('M3 run ID must be a string.')
    safe_component(requested)
    if not requested.startswith('M3-'):
        raise M3CompatibilityError('M3 run ID must use the M3 namespace.')
    run_root = runs_root / requested
    manifest_path = run_root / 'run_manifest.json'
    immutable = {
        'milestone': 'M3', 'parent_milestone': config.parent_milestone,
        'parent_run_id': config.parent_run_id,
        'parent_checkpoint': config.parent_checkpoint,
        'parent_checkpoint_path': config.parent_checkpoint_path,
        **evidence.hashes, 'config_hash': config_hash,
        'source_hash': source_hash, 'notebook_hash': notebook_hash,
        'notebook_hash_policy': M3_NOTEBOOK_HASH_POLICY,
        'governing_document_hashes': document_hashes,
        'reference_artifact_hashes': reference_hashes,
        'runtime_compatibility': runtime_signature,
        'backend_selection': selection,
        'exploration_status': 'EXPLORATORY',
        'certification_status': 'NOT_CERTIFIED',
    }
    if run_root.exists() and (run_root / 'run_config.json').is_file():
        if read_json(run_root / 'run_config.json') != config.canonical_payload():
            raise M3CompatibilityError('Immutable M3 config changed.')
        manifest = read_json(manifest_path) if manifest_path.is_file() else None
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value for key, value in immutable.items()
        ):
            raise M3CompatibilityError('M3 manifest/source/parent/runtime changed.')
        report_path = run_root / 'test_report.json'
        if test_report is None:
            effective_report = read_json(report_path) if report_path.is_file() else {}
        else:
            effective_report = test_report
            atomic_write_json(report_path, test_report)
        manager = CheckpointManager(
            run_root, config, source_hash, notebook_hash,
        )
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise M3CompatibilityError('Existing M3 run has no valid checkpoint.')
        repaired = loaded.queue.recover_interrupted(run_root)
        orchestrator = M3Orchestrator(
            persistent_root, run_root, project_root, config,
            loaded.state, loaded.queue, manager, effective_report,
            manifest, evidence.projector_tensors, loaded.tensors,
        )
        if repaired:
            orchestrator.checkpoint(
                f'recovered {len(repaired)} interrupted M3 item(s)',
            )
        print('Resumed M3 from:', loaded.path)
        return orchestrator
    if run_root.exists():
        raise M3CompatibilityError('Incomplete M3 run directory exists.')
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        'logs', 'reports', 'artifacts', 'work_items', 'checkpoints', 'cache',
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'run_id': requested, 'created_at': utc_now(),
        'environment': environment,
        'sector_ordering': 'lexicographic M2 projector block ordering',
        'rounding_policy': 'FP64 exploratory; no GPU value is a rigorous bound',
        **immutable,
    }
    atomic_write_json(run_root / 'run_config.json', config.canonical_payload())
    atomic_write_json(run_root / 'test_report.json', test_report or {})
    atomic_write_json(manifest_path, manifest)
    _seed_everything(config.seed)
    state = RunState(
        requested, config_hash, utc_now(), utc_now(),
        milestone='M3', phase='M3_BOOTSTRAP',
    )
    queue = WorkQueue()
    for phase in M3_PHASES:
        queue.add(
            phase, config_hash, {'milestone': 'M3', 'phase': phase},
            predicted_s=5.0 * 60.0,
        )
    manager = CheckpointManager(
        run_root, config, source_hash, notebook_hash,
    )
    orchestrator = M3Orchestrator(
        persistent_root, run_root, project_root, config,
        state, queue, manager, test_report or {},
        manifest, evidence.projector_tensors,
    )
    orchestrator.checkpoint('initial M3 run state')
    atomic_write_json(latest_pointer, {
        'milestone': 'M3', 'run_id': requested,
        'config_hash': config_hash, 'updated_at': utc_now(),
    })
    return orchestrator
