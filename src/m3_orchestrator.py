from __future__ import annotations

import gc
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
    safe_component, sanitize_for_json, sha256_bytes, sha256_file, utc_now,
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


def _directory_size(path: Path) -> int:
    total = 0
    if not path.exists() or path.is_symlink():
        return 0
    for root, directories, files in os.walk(path, followlinks=False):
        directories[:] = [
            name for name in directories
            if not (Path(root) / name).is_symlink()
        ]
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def _checkpoint_keep_count() -> int:
    raw = os.environ.get('VALIDATED_RG_M3_CHECKPOINT_KEEP', '1').strip()
    try:
        keep = int(raw)
    except ValueError as exc:
        raise M3CompatibilityError(
            'VALIDATED_RG_M3_CHECKPOINT_KEEP must be an integer.'
        ) from exc
    if not 1 <= keep <= 8:
        raise M3CompatibilityError(
            'VALIDATED_RG_M3_CHECKPOINT_KEEP must lie in [1, 8].'
        )
    return keep


def _prune_committed_checkpoints(
    checkpoint_root: Path, *, keep: int,
) -> dict[str, Any]:
    '''Keep only the newest committed checkpoints; never follow symlinks.'''
    candidates: list[tuple[int, Path]] = []
    skipped_symlinks: list[str] = []
    if not checkpoint_root.is_dir():
        return {
            'removed': 0, 'bytes_freed': 0, 'kept': [],
            'skipped_symlinks': skipped_symlinks,
        }
    for path in checkpoint_root.iterdir():
        if not path.name.startswith('ckpt_'):
            continue
        if path.is_symlink():
            skipped_symlinks.append(path.name)
            continue
        if not path.is_dir() or not (path / 'COMMITTED').is_file():
            continue
        try:
            index = int(path.name.removeprefix('ckpt_'))
        except ValueError:
            continue
        candidates.append((index, path))
    candidates.sort()
    remove = candidates[:-keep] if len(candidates) > keep else []
    freed = 0
    removed: list[str] = []
    for _index, path in remove:
        size = _directory_size(path)
        shutil.rmtree(path)
        freed += size
        removed.append(path.name)
    return {
        'removed': len(removed), 'removed_names': removed,
        'bytes_freed': freed,
        'kept': [path.name for _, path in candidates[-keep:]],
        'skipped_symlinks': skipped_symlinks,
    }


def _blockwise_reference_matvec(
    operator: ArmillaryLinearOperator, vector: np.ndarray, *, adjoint: bool = False,
) -> np.ndarray:
    '''Exact CPU block reference without allocating the global dense matrix.'''
    source = np.asarray(vector, dtype=np.float64)
    if source.ndim != 1 or source.shape[0] != operator.dimension:
        raise ValueError('M3 block reference vector has the wrong shape.')
    output = np.empty_like(source)
    offset = 0
    for block in operator.blocks:
        projector = np.asarray(block.projector, dtype=np.float64)
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ArithmeticError('M3 operator block projector is not square.')
        size = projector.shape[0]
        stop = offset + size
        if stop > operator.dimension:
            raise ArithmeticError('M3 operator block layout exceeds its dimension.')
        matrix = projector.T if adjoint else projector
        output[offset:stop] = float(block.weight) * (
            matrix @ source[offset:stop]
        )
        offset = stop
    if offset != operator.dimension:
        raise ArithmeticError('M3 operator block layout does not fill its dimension.')
    return output


def _reference_singular_values(
    operator: ArmillaryLinearOperator,
) -> tuple[np.ndarray, dict[str, Any]]:
    '''Use projector rank spectra, with exact-SVD fallback for unsafe blocks.'''
    values: list[np.ndarray] = []
    fast_blocks = 0
    fallback_blocks = 0
    max_symmetry_error = 0.0
    max_trace_error = 0.0
    max_frobenius_rank_error = 0.0
    for block in operator.blocks:
        projector = np.asarray(block.projector, dtype=np.float64)
        if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
            raise ArithmeticError('M3 RSVD reference projector is not square.')
        size = projector.shape[0]
        symmetry_error = float(np.max(np.abs(projector - projector.T)))
        trace = float(np.trace(projector))
        rank = int(round(trace))
        trace_error = abs(trace - rank)
        frobenius_sq = float(np.vdot(projector, projector).real)
        frobenius_rank_error = abs(frobenius_sq - rank)
        max_symmetry_error = max(max_symmetry_error, symmetry_error)
        max_trace_error = max(max_trace_error, trace_error)
        max_frobenius_rank_error = max(
            max_frobenius_rank_error, frobenius_rank_error,
        )
        scale = max(1, size)
        projector_safe = (
            0 <= rank <= size
            and symmetry_error <= 1e-12 * scale
            and trace_error <= 1e-10 * scale
            and frobenius_rank_error <= 1e-10 * scale
        )
        if projector_safe:
            if rank:
                values.append(np.full(rank, abs(float(block.weight))))
            fast_blocks += 1
        else:
            values.append(np.linalg.svd(
                float(block.weight) * projector, compute_uv=False,
            ))
            fallback_blocks += 1
    concatenated = (
        np.concatenate(values) if values else np.empty(0, dtype=np.float64)
    )
    concatenated.sort()
    concatenated = concatenated[::-1]
    return concatenated, {
        'reference_spectrum_mode': (
            'projector_rank_with_svd_fallback'
            if fallback_blocks else 'projector_rank_exact'
        ),
        'projector_fast_blocks': fast_blocks,
        'svd_fallback_blocks': fallback_blocks,
        'max_projector_symmetry_error': max_symmetry_error,
        'max_projector_trace_rank_error': max_trace_error,
        'max_projector_frobenius_rank_error': max_frobenius_rank_error,
    }


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
        self._operator_cache: dict[int, ArmillaryLinearOperator] = {}
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

    def _record_storage_cleanup(
        self, event: str, payload: dict[str, Any],
    ) -> None:
        path = self.run_root / 'reports' / 'M3_storage_cleanup.json'
        current = read_json(path) if path.is_file() else {}
        if not isinstance(current, dict):
            current = {}
        events = current.get('events')
        if not isinstance(events, list):
            events = []
        entry = {'event': event, 'recorded_at': utc_now(), **payload}
        events.append(entry)
        events = events[-100:]
        total = int(current.get('total_bytes_freed', 0)) + int(
            payload.get('bytes_freed', 0)
        )
        atomic_write_json(path, {
            'schema_version': 1, 'run_id': self.state.run_id,
            'updated_at': utc_now(), 'total_bytes_freed': total,
            'events': events,
        })

    def _prune_old_checkpoints(self) -> dict[str, Any]:
        result = _prune_committed_checkpoints(
            self.run_root / 'checkpoints', keep=_checkpoint_keep_count(),
        )
        if result['removed'] or result['skipped_symlinks']:
            self._record_storage_cleanup('keep_latest_checkpoints', result)
            self.logger.emit(
                'm3_checkpoints_pruned', run_id=self.state.run_id, **result,
            )
        return result

    def _cleanup_completed_run(self) -> dict[str, Any]:
        '''Remove regenerable cache and obsolete attempts after final checkpoint.'''
        freed = 0
        removed_attempts = 0
        removed_temporary = 0
        referenced: set[Path] = set()
        for item in self.queue.items.values():
            if not item.result_relpath:
                continue
            try:
                referenced.add((self.run_root / item.result_relpath).resolve().parent)
            except OSError:
                continue
        artifacts = self.run_root / 'artifacts'
        if artifacts.is_dir():
            for item_root in artifacts.iterdir():
                if item_root.is_symlink() or not item_root.is_dir():
                    continue
                for attempt in item_root.iterdir():
                    if attempt.is_symlink() or not attempt.is_dir():
                        continue
                    if attempt.name.startswith('.tmp-attempt-'):
                        size = _directory_size(attempt)
                        shutil.rmtree(attempt)
                        freed += size
                        removed_temporary += 1
                    elif (
                        attempt.name.startswith('attempt_')
                        and attempt.resolve() not in referenced
                    ):
                        size = _directory_size(attempt)
                        shutil.rmtree(attempt)
                        freed += size
                        removed_attempts += 1
        cache = self.run_root / 'cache'
        removed_cache_files = 0
        if cache.is_dir() and not cache.is_symlink():
            for child in list(cache.iterdir()):
                if child.is_symlink():
                    continue
                size = _directory_size(child) if child.is_dir() else child.stat().st_size
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                freed += size
                removed_cache_files += 1
        payload = {
            'removed_attempts': removed_attempts,
            'removed_temporary_attempts': removed_temporary,
            'removed_cache_entries': removed_cache_files,
            'bytes_freed': freed,
            'final_checkpoint_retained_for_m4': True,
            'note': (
                'The newest M3 checkpoint is retained because M4 consumes the '
                'RSVD/Triad tensors. Full checkpoint stripping is safe only after '
                'downstream M4/M5/M6 progress.'
            ),
        }
        self._record_storage_cleanup('m3_complete_cleanup', payload)
        return payload

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m3_safe()
        memory = self._memory_headroom(checkpoint=True)
        self.state.notes.append(
            f'{utc_now()} checkpoint: {reason}; gpu_free_fraction={memory["free_fraction"]}'
        )
        result = self.checkpoints.save(self.state, self.queue, self.tensors)
        self.guard.mark_checkpoint()
        self.last_checkpoint = result
        cleanup = self._prune_old_checkpoints()
        self.logger.emit(
            'm3_checkpoint_committed', run_id=self.state.run_id,
            checkpoint=result.index, reason=reason, size_bytes=result.size_bytes,
            old_checkpoints_removed=cleanup['removed'],
            bytes_freed=cleanup['bytes_freed'],
        )
        print(f'M3 checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _operator(self, sectors_per_shard: int | None = None) -> ArmillaryLinearOperator:
        shard_size = int(
            sectors_per_shard or self.config.initial_sector_shard_size
        )
        cached = self._operator_cache.get(shard_size)
        if cached is None:
            cached = build_armillary_operator(
                self.parent_tensors, self.backend, self.path_cache_path,
                sectors_per_shard=shard_size, j2_max=self.config.j2_max,
            )
            self._operator_cache[shard_size] = cached
        return cached

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
        reference = _blockwise_reference_matvec(operator, x)
        started = time.monotonic()
        matrix_free = operator.matvec(x)
        self.backend.synchronize()
        matvec_s = time.monotonic() - started
        error = float(np.max(np.abs(matrix_free - reference)))
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
        dense_bytes = int(operator.dimension) ** 2 * np.dtype(np.float64).itemsize
        return {
            'status': 'PASS', 'dimension': operator.dimension,
            'matvec_max_abs_error': error,
            'adjoint_relative_error': float(adjoint_error),
            'matvec_elapsed_s': matvec_s,
            'path_cache_reused': reused,
            'path_cache_before': stats_before,
            'path_cache_after': stats_after,
            'reference_mode': 'block_explicit_no_global_dense',
            'explicit_matrix_bytes': 0,
            'avoided_dense_matrix_bytes': dense_bytes,
            'gpu_memory_after': self.backend.memory_snapshot(),
        }

    def _rsvd_result(self) -> dict[str, Any]:
        selected_operator: ArmillaryLinearOperator | None = None

        def compute(shard_size: int) -> RSVDResult:
            nonlocal selected_operator
            self._memory_headroom(checkpoint=False)
            selected_operator = self._operator(shard_size)
            try:
                return randomized_svd(
                    selected_operator, target_rank=self.config.target_rank,
                    oversampling=self.config.oversampling,
                    power_iterations=self.config.power_iterations,
                    seed=self.config.seed,
                )
            except Exception:
                # Do not retain a failed/OOM operator instance in the cache.
                self._operator_cache.pop(int(shard_size), None)
                selected_operator = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                raise

        recovered = run_with_oom_recovery(
            compute, self.config.initial_sector_shard_size,
            min_shard_size=self.config.min_sector_shard_size,
            max_oom_retries=self.config.max_oom_retries,
        )
        result = recovered.value
        if selected_operator is None:
            raise RuntimeError('M3 RSVD failed to retain its operator.')
        explicit_values, reference_metadata = _reference_singular_values(
            selected_operator,
        )
        reference_top = np.zeros(self.config.target_rank, dtype=np.float64)
        copied = min(self.config.target_rank, explicit_values.size)
        reference_top[:copied] = explicit_values[:copied]
        explicit_error = float(np.max(np.abs(
            result.singular_values - reference_top
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
            **reference_metadata,
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
            payload = {
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
            }
            clean, had_nonfinite = sanitize_for_json(payload)
            if had_nonfinite:
                # Diagnostic write only — never treat sanitized floats as PASS.
                clean['nonfinite_values_present'] = True
                clean['certification_status'] = 'NOT_CERTIFIED'
                if isinstance(clean.get('result'), dict):
                    clean['result'] = {
                        **clean['result'],
                        'status': 'FAIL_NONFINITE',
                        'nonfinite_values_present': True,
                    }
                result_file = temporary / 'result.json'
                atomic_write_json(result_file, clean)
                fsync_directory(temporary)
                os.replace(temporary, final)
                fsync_directory(parent)
                # Keep diagnostic under final/; do not delete on this path.
                temporary = None  # type: ignore[assignment]
                raise ArithmeticError(
                    f'M3 phase {item.phase} produced non-finite floats; '
                    'fail closed (NOT_CERTIFIED / nonfinite_values_present).'
                )
            result_file = temporary / 'result.json'
            atomic_write_json(result_file, clean)
            fsync_directory(temporary)
            os.replace(temporary, final)
            fsync_directory(parent)
            temporary = None  # type: ignore[assignment]
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
            if temporary is not None and temporary.exists():
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
        clean, had_nonfinite = sanitize_for_json(summary)
        if had_nonfinite:
            clean['nonfinite_values_present'] = True
            clean['certification_status'] = 'NOT_CERTIFIED'
            # Exploratory path must not look complete when metrics are nonfinite.
            if clean.get('phase') == 'M3_COMPLETE':
                clean['phase'] = 'M3_RUNNING'
                clean['milestone_status'] = 'EXPLORATORY'
                clean['stop_reason'] = (
                    f'{reason}; nonfinite_values_present (fail closed)'
                )
        print(json.dumps(clean, ensure_ascii=False, indent=2, allow_nan=False))
        return clean

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m3_safe()
        acceptance = self.run_root / 'reports' / 'M3_acceptance.json'
        if self.state.phase == 'M3_COMPLETE' and acceptance.is_file():
            self._prune_old_checkpoints()
            self._cleanup_completed_run()
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
                    sector_count=self.config.sector_count,
                    operator_dimension=self.config.operator_dimension,
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
                self._cleanup_completed_run()
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
    *,
    allow_code_drift: bool = False,
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
    env_drift = os.environ.get('VALIDATED_RG_M3_ALLOW_CODE_DRIFT', '').strip().lower()
    allow_code_drift = bool(
        allow_code_drift or env_drift in {'1', 'true', 'yes', 'on'}
    )
    # m2_audit_sha256: staged notebooks may rewrite audit/m2_accepted_parent.json
    # (e.g. generated_at) while parent_run_id / report / checkpoint stay pinned.
    # Live verify_accepted_m2_parent already re-checks audit content vs M2 artifacts.
    relax_fields = {
        'source_hash', 'notebook_hash', 'runtime_compatibility', 'backend_selection',
        'm2_audit_sha256',
    } if allow_code_drift else set()
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
        if not isinstance(manifest, dict):
            raise M3CompatibilityError('Existing M3 run lacks a valid manifest.')
        mismatches = {
            key: {'expected': value, 'found': manifest.get(key)}
            for key, value in immutable.items()
            if key not in relax_fields and manifest.get(key) != value
        }
        if mismatches:
            raise M3CompatibilityError(
                'M3 manifest/source/parent/runtime changed: '
                + ', '.join(sorted(mismatches))
            )
        drifted = {
            key: {
                'manifest': manifest.get(key),
                'current': immutable.get(key),
            }
            for key in relax_fields
            if manifest.get(key) != immutable.get(key)
        }
        if drifted:
            reports_dir = run_root / 'reports'
            reports_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(reports_dir / 'code_drift.json', {
                'schema_version': 1,
                'allow_code_drift': True,
                'recorded_at': utc_now(),
                'run_id': requested,
                'drifted_fields': drifted,
                'note': (
                    'Controller source/runtime and/or M2 audit file bytes drifted '
                    'since M3 run creation; config_hash and M2 parent run/report/'
                    'checkpoint identity remain pinned and re-verified.'
                ),
            })
            print('WARNING: resuming M3 with code drift:', ', '.join(sorted(drifted)))
        report_path = run_root / 'test_report.json'
        if test_report is None:
            effective_report = read_json(report_path) if report_path.is_file() else {}
        else:
            effective_report = test_report
            atomic_write_json(report_path, test_report)
        manager = CheckpointManager(
            run_root, config, source_hash, notebook_hash,
            require_source_match=not allow_code_drift,
        )
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise M3CompatibilityError('Existing M3 run has no valid checkpoint.')
        repaired = loaded.queue.recover_interrupted(run_root)
        if allow_code_drift:
            repaired.extend(
                loaded.queue.reset_transient_attempt_budget(
                    max_item_attempts=config.max_item_attempts,
                )
            )
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
    if run_root.exists() and not (run_root / 'run_config.json').is_file():
        # Notebooks may pre-seed only test_report.json before create_or_resume.
        unexpected = sorted(
            path.name for path in run_root.iterdir()
            if path.name not in {'test_report.json'} and not path.name.startswith('.')
        )
        if unexpected:
            raise M3CompatibilityError(
                'Incomplete M3 run directory exists with unexpected entries: '
                + ', '.join(unexpected)
                + f'. Rename or remove {run_root} then retry.'
            )
    elif run_root.exists():
        raise M3CompatibilityError('Incomplete M3 run directory exists.')
    run_root.mkdir(parents=True, exist_ok=True)
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
