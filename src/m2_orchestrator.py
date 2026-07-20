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

from .armillary import (
    all_link_star_keys, build_armillary_sector, checkpoint_tensor_shards,
    sector_summary,
)
from .checkpoint import CheckpointManager, CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, canonical_json_bytes, fsync_directory, hash_tree, read_json,
    safe_component, sha256_bytes, sha256_file, utc_now,
)
from .dense_reference import (
    build_dense_reference, exact_matrix_difference_zero, matrix_hash,
)
from .fusion import convention_hash, convention_payload, representation_dimension
from .m2_config import M2Config
from .m2_parent import M2ParentError, verify_accepted_m1_parent
from .m2_reporting import (
    load_m2_phase_results, validate_m2_acceptance, write_m2_report_package,
    write_m2_session_artifacts,
)
from .orchestrator import governing_document_hashes, reference_artifact_hashes
from .reporting import JsonlLogger
from .runtime import environment_info, runtime_compatibility_signature
from .sector_canonicalization import (
    action_table_hash, canonicalize_sector, transverse_cubic_actions,
)
from .session_guard import SessionGuard, SessionState
from .wigner_cache import generate_low_cutoff_cache, validate_cache
from .work_queue import WorkItem, WorkQueue

try:
    import torch
except ImportError:
    torch = None

M2_PHASE_ORDER = (
    'M2_WIGNER_CACHE', 'M2_DENSE_REFERENCE', 'M2_ARMILLARY',
    'M2_EQUIVALENCE', 'M2_SYMMETRY', 'M2_REPORT',
)
M2_NOTEBOOK_HASH_POLICY = 'canonical_nbformat4_cell_type_source_tags_v1'


class M2CompatibilityError(RuntimeError):
    '''Raised when an M2 run cannot be created or resumed exactly.'''


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()


def _notebook_hash(project_root: Path) -> str:
    path = project_root / 'notebooks/30_m2_armillary.ipynb'
    if not path.is_file():
        raise M2CompatibilityError('M2 user-facing notebook is missing.')
    payload = read_json(path)
    cells = payload.get('cells') if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get('nbformat') != 4 or not isinstance(cells, list):
        raise M2CompatibilityError('M2 notebook must be a valid nbformat 4 document.')
    identity_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise M2CompatibilityError(f'M2 notebook cell {index} is not a mapping.')
        cell_type = cell.get('cell_type')
        source = cell.get('source')
        metadata = cell.get('metadata', {})
        if cell_type not in {'code', 'markdown', 'raw'}:
            raise M2CompatibilityError(f'M2 notebook cell {index} has an invalid type.')
        if isinstance(source, str):
            normalized_source = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            normalized_source = ''.join(source)
        else:
            raise M2CompatibilityError(f'M2 notebook cell {index} has invalid source text.')
        if not isinstance(metadata, dict):
            raise M2CompatibilityError(f'M2 notebook cell {index} has invalid metadata.')
        tags = metadata.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise M2CompatibilityError(f'M2 notebook cell {index} has invalid execution tags.')
        identity_cells.append({
            'cell_type': cell_type, 'source': normalized_source, 'tags': tags,
        })
    identity = {
        'policy': M2_NOTEBOOK_HASH_POLICY, 'nbformat': 4, 'cells': identity_cells,
    }
    return sha256_bytes(canonical_json_bytes(identity))


class M2Orchestrator:
    def __init__(
        self, persistent_root: Path, run_root: Path, project_root: Path,
        config: M2Config, state: RunState, queue: WorkQueue,
        checkpoints: CheckpointManager, test_report: dict[str, Any],
        manifest: dict[str, Any], tensors: dict[str, Any] | None = None,
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
        self.tensors = tensors or {}
        self.guard = SessionGuard(config)
        self.logger = JsonlLogger(run_root / 'logs' / 'events.jsonl')
        self.last_checkpoint: CheckpointSaveResult | None = None

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m2_safe()
        self.state.notes.append(f'{utc_now()} checkpoint: {reason}')
        result = self.checkpoints.save(self.state, self.queue, self.tensors)
        self.guard.mark_checkpoint()
        self.last_checkpoint = result
        self.logger.emit(
            'm2_checkpoint_committed', run_id=self.state.run_id,
            checkpoint=result.index, reason=reason, size_bytes=result.size_bytes,
        )
        print(f'M2 checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _next_pending(self) -> WorkItem | None:
        for phase in M2_PHASE_ORDER:
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

    def _compute_wigner(self, temporary: Path) -> dict[str, Any]:
        first = temporary / 'wigner_cache.json'
        second = temporary / 'wigner_cache_regenerated.json'
        first_digest = generate_low_cutoff_cache(
            first, self.config.j2_max, self.config.leg_count,
        )
        second_digest = generate_low_cutoff_cache(
            second, self.config.j2_max, self.config.leg_count,
        )
        payload = validate_cache(first)
        validate_cache(second)
        if first_digest != second_digest:
            raise ArithmeticError('Exact Wigner cache regeneration is nondeterministic.')
        return {
            'status': 'PASS', 'entry_count': payload['entry_count'],
            'convention_hash': payload['convention_hash'],
            'cache_filename': first.name, 'cache_sha256': first_digest,
            'regenerated_filename': second.name,
            'regenerated_sha256': second_digest,
            'regeneration_sha256_match': True,
        }

    def _compute_dense(self) -> dict[str, Any]:
        sectors: list[dict[str, Any]] = []
        zero_count = 0
        residual_count = 0
        for key in all_link_star_keys(self.config.j2_max):
            dense = build_dense_reference(key.representations, key.orientations)
            residual_count += int(dense.generator_residual_zero)
            is_zero = not any(dense.projector)
            if sum(key.representations) % 2 and is_zero:
                zero_count += 1
            sectors.append({
                'representations': list(key.representations),
                'orientations': list(key.orientations),
                'dense_dimension': representation_dimension(key.representations),
                'singlet_rank': dense.singlet_rank,
                'projector_hash': matrix_hash(dense.projector),
                'generator_residual_zero': dense.generator_residual_zero,
            })
        if residual_count != 64 or zero_count != 32:
            raise ArithmeticError('Dense reference exact gauge checks failed closed.')
        return {
            'status': 'PASS', 'sector_count': len(sectors),
            'generator_residual_zero_count': residual_count,
            'odd_half_zero_count': zero_count, 'sectors': sectors,
        }

    def _compute_armillary(self) -> dict[str, Any]:
        built = [
            build_armillary_sector(key)
            for key in all_link_star_keys(self.config.j2_max)
        ]
        isometry_count = sum(sector.isometry_exact for sector in built)
        if isometry_count != 64:
            raise ArithmeticError('Armillary exact isometry checks failed closed.')
        self.tensors = checkpoint_tensor_shards(built)
        return {
            'status': 'PASS', 'sector_count': len(built),
            'isometry_exact_count': isometry_count,
            'checkpoint_tensor_count': len(self.tensors),
            'sectors': [sector_summary(sector) for sector in built],
        }

    def _compute_equivalence(self) -> dict[str, Any]:
        mismatches: list[list[int]] = []
        matches = 0
        max_dimension = 0
        keys = all_link_star_keys(self.config.j2_max)
        for key in keys:
            dense = build_dense_reference(key.representations, key.orientations)
            armillary = build_armillary_sector(key)
            max_dimension = max(
                max_dimension, representation_dimension(key.representations),
            )
            same = (
                dense.singlet_rank == armillary.singlet_rank
                and dense.generator_residual_zero
                and armillary.isometry_exact
                and exact_matrix_difference_zero(
                    dense.projector, armillary.reconstructed_dense,
                )
            )
            if same:
                matches += 1
            else:
                mismatches.append(list(key.representations))
        if mismatches or matches != 64:
            raise ArithmeticError(f'Dense/armillary exact mismatches: {mismatches}')
        return {
            'status': 'PASS', 'sector_count': len(keys),
            'exact_match_count': matches, 'mismatches': mismatches,
            'max_dense_dimension': max_dimension,
            'comparison': 'exact symbolic matrix equality',
        }

    def _compute_symmetry(self) -> dict[str, Any]:
        keys = all_link_star_keys(self.config.j2_max)
        actions = transverse_cubic_actions()
        canonical = {
            canonicalize_sector(key.representations, key.orientations)
            for key in keys
        }
        repeated = {
            canonicalize_sector(key.representations, key.orientations)
            for key in keys
        }
        if canonical != repeated or not 1 < len(canonical) < len(keys):
            raise ArithmeticError('M2 sector canonicalization is invalid or nondeterministic.')
        return {
            'status': 'PASS', 'group_order': len(actions),
            'action_table_hash': action_table_hash(),
            'canonical_sector_count': len(canonical), 'deterministic': True,
        }

    def _compute_phase(self, item: WorkItem, temporary: Path) -> dict[str, Any]:
        if item.phase == 'M2_WIGNER_CACHE':
            return self._compute_wigner(temporary)
        if item.phase == 'M2_DENSE_REFERENCE':
            return self._compute_dense()
        if item.phase == 'M2_ARMILLARY':
            return self._compute_armillary()
        if item.phase == 'M2_EQUIVALENCE':
            return self._compute_equivalence()
        if item.phase == 'M2_SYMMETRY':
            return self._compute_symmetry()
        if item.phase == 'M2_REPORT':
            required = M2_PHASE_ORDER[:-1]
            results = load_m2_phase_results(self.run_root, self.queue)
            missing = [phase for phase in required if phase not in results]
            if missing:
                raise RuntimeError(f'M2 report work item is missing inputs: {missing}')
            return {'status': 'READY', 'input_phases': list(required)}
        raise RuntimeError(f'Unknown M2 work phase: {item.phase}')

    def _execute_item(self, item: WorkItem) -> tuple[str, str]:
        parent = self.run_root / 'artifacts' / item.item_id
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f'.tmp-attempt-{item.attempts:03d}-{uuid.uuid4().hex}'
        final = parent / f'attempt_{item.attempts:03d}'
        if final.exists():
            raise RuntimeError(f'M2 attempt output already exists: {final}')
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            result = self._compute_phase(item, temporary)
            result_file = temporary / 'result.json'
            atomic_write_json(result_file, {
                'schema_version': 1, 'milestone': 'M2', 'phase': item.phase,
                'item_id': item.item_id, 'config_hash': self.config.config_hash(),
                'certification_status': 'NOT_CERTIFIED', 'generated_at': utc_now(),
                'result': result,
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
                raise RuntimeError('M2 result verification failed after commit.')
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

    def _summary(self, stop_reason: str) -> dict[str, Any]:
        elapsed = self.guard.elapsed_s()
        remaining = self.guard.remaining_s()
        artifacts = write_m2_session_artifacts(
            self.run_root, self.state, self.queue, stop_reason,
            elapsed, remaining, self.persistent_root, self.project_root,
        )
        summary = {
            'milestone': 'M2', 'run_id': self.state.run_id,
            'phase': self.state.phase,
            'checkpoint_index': self.state.checkpoint_index,
            'certification_status': 'NOT_CERTIFIED', 'stop_reason': stop_reason,
            'elapsed_s': elapsed, 'remaining_s': remaining,
            'session_artifacts': artifacts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m2_safe()
        acceptance_path = self.run_root / 'reports' / 'M2_acceptance.json'
        if self.state.phase == 'M2_COMPLETE' and acceptance_path.is_file():
            return self._summary('M2 already complete; no work or checkpoint was started')
        self.state.phase = 'M2_RUNNING'
        self.logger.emit('m2_session_started', run_id=self.state.run_id)
        while True:
            session_state = self.guard.state()
            if session_state is SessionState.RETURN:
                return self._summary(
                    'hard return threshold reached; using last committed checkpoint',
                )
            if session_state in {SessionState.DRAIN, SessionState.FINAL_SAVE}:
                self.checkpoint(f'M2 session state {session_state.value}')
                return self._summary(f'{session_state.value.lower()} checkpoint complete')
            if self.guard.checkpoint_due():
                self.checkpoint('periodic 15-minute M2 checkpoint')
            item = self._next_pending()
            if item is None:
                incomplete = [
                    queued for queued in self.queue.items.values()
                    if queued.status != 'done'
                ]
                if incomplete:
                    self.checkpoint('M2 queue has no runnable item but is incomplete')
                    raise RuntimeError(
                        'M2 cannot complete with failed/blocked/running work items.',
                    )
                results = load_m2_phase_results(self.run_root, self.queue)
                validate_m2_acceptance(
                    self.state, self.queue, results, self.test_report,
                )
                self.state.bounds = {
                    'dense_total_generator_residuals': 'EXACT_SYMBOLIC_ZERO',
                    'armillary_isometries': 'EXACT_SYMBOLIC_IDENTITY',
                    'dense_armillary_difference': 'EXACT_SYMBOLIC_ZERO',
                    'odd_half_spin_sectors': 'EXACT_ZERO_PROJECTOR',
                    'float64_checkpoint_tensors': 'DIAGNOSTIC_ONLY_NOT_A_BOUND',
                }
                self.state.phase = 'M2_COMPLETE'
                final_checkpoint = self.checkpoint('M2 acceptance gates complete')
                paths = write_m2_report_package(
                    self.run_root, self.config, self.state, self.queue,
                    self.test_report, final_checkpoint, self.manifest,
                )
                self.logger.emit(
                    'm2_milestone_complete', run_id=self.state.run_id,
                    reports=paths,
                )
                return self._summary(
                    f'M2 complete; report written to {paths["json"]}',
                )
            predicted = self.queue.predicted_duration(item)
            if not self.guard.may_start(predicted):
                self.checkpoint('insufficient safe time for next M2 item')
                return self._summary(
                    'next M2 work item deferred to a fresh session',
                )
            item.attempts += 1
            if item.attempts > self.config.max_item_attempts:
                item.status = 'failed'
                item.last_error = 'Maximum M2 attempt count exceeded.'
                self.checkpoint('M2 work item exceeded attempt limit')
                raise RuntimeError(item.last_error)
            item.status = 'running'
            self.checkpoint(f'before M2 item {item.phase}')
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
                        'hard return reached after atomic M2 done marker; '
                        'resume will repair queue',
                    )
                self.checkpoint(f'after M2 item {item.phase}')
            except KeyboardInterrupt:
                item.status = 'pending'
                item.last_error = 'KeyboardInterrupt'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'interrupted M2 item {item.phase}')
                self._summary(f'KeyboardInterrupt in M2 item {item.phase}')
                raise
            except Exception as exc:
                item.status = (
                    'failed'
                    if item.attempts >= self.config.max_item_attempts
                    else 'pending'
                )
                item.last_error = f'{type(exc).__name__}: {exc}'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'exception in M2 item {item.phase}')
                self._summary(
                    f'exception in M2 item {item.phase}: {type(exc).__name__}',
                )
                raise


def create_or_resume_m2(
    persistent_root: Path, config: M2Config, project_root: Path,
    run_id: str | None = None, test_report: dict[str, Any] | None = None,
) -> M2Orchestrator:
    try:
        parent_hashes = verify_accepted_m1_parent(project_root, config)
    except M2ParentError as exc:
        raise M2CompatibilityError(str(exc)) from exc
    config_hash = config.config_hash()
    source_hash = hash_tree(project_root / 'src')
    notebook_hash = _notebook_hash(project_root)
    document_hashes = governing_document_hashes(project_root)
    reference_hashes = reference_artifact_hashes(project_root)
    environment = environment_info()
    runtime_signature = runtime_compatibility_signature(environment)
    runs_root = persistent_root / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)
    requested = run_id or os.environ.get('VALIDATED_RG_M2_RUN_ID')
    latest_pointer = persistent_root / 'LATEST_M2_RUN.json'
    if requested is None and latest_pointer.is_file():
        pointer = read_json(latest_pointer)
        if isinstance(pointer, dict) and pointer.get('config_hash') == config_hash:
            requested = pointer.get('run_id')
    if requested is None:
        requested = (
            'M2-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
            + '-' + config_hash[:12]
        )
    if not isinstance(requested, str):
        raise M2CompatibilityError('M2 run ID must be a string.')
    safe_component(requested)
    if not requested.startswith('M2-'):
        raise M2CompatibilityError('M2 run ID must use the M2- namespace.')

    run_root = runs_root / requested
    manifest_path = run_root / 'run_manifest.json'
    immutable_manifest_fields = {
        'milestone': 'M2', 'parent_milestone': config.parent_milestone,
        'parent_run_id': config.parent_run_id,
        'parent_checkpoint': config.parent_checkpoint,
        'parent_checkpoint_path': config.parent_checkpoint_path,
        **parent_hashes, 'config_hash': config_hash, 'source_hash': source_hash,
        'notebook_hash': notebook_hash,
        'notebook_hash_policy': M2_NOTEBOOK_HASH_POLICY,
        'convention_hash': convention_hash(),
        'governing_document_hashes': document_hashes,
        'reference_artifact_hashes': reference_hashes,
        'runtime_compatibility': runtime_signature,
        'certification_status': 'NOT_CERTIFIED',
    }
    if run_root.exists() and (run_root / 'run_config.json').is_file():
        if read_json(run_root / 'run_config.json') != config.canonical_payload():
            raise M2CompatibilityError('Immutable M2 run configuration differs.')
        if not manifest_path.is_file():
            raise M2CompatibilityError('Existing M2 run lacks run_manifest.json.')
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or any(
            manifest.get(key) != value
            for key, value in immutable_manifest_fields.items()
        ):
            raise M2CompatibilityError(
                'M2 manifest/source/parent/runtime identity changed.',
            )
        saved_test_report_path = run_root / 'test_report.json'
        if test_report is None:
            effective_test_report = (
                read_json(saved_test_report_path)
                if saved_test_report_path.is_file()
                else {}
            )
        else:
            effective_test_report = test_report
            atomic_write_json(saved_test_report_path, test_report)
        manager = CheckpointManager(
            run_root, config, source_hash, notebook_hash,
        )
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise M2CompatibilityError(
                'Existing M2 run has no valid checkpoint.',
            )
        repaired = loaded.queue.recover_interrupted(run_root)
        orchestrator = M2Orchestrator(
            persistent_root, run_root, project_root, config,
            loaded.state, loaded.queue, manager, effective_test_report,
            manifest, loaded.tensors,
        )
        if repaired:
            orchestrator.checkpoint(
                f'recovered {len(repaired)} interrupted M2 item(s)',
            )
        print('Resumed M2 from:', loaded.path)
        return orchestrator

    if run_root.exists():
        raise M2CompatibilityError(
            'M2 run directory exists but is incomplete; refusing overwrite.',
        )
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in (
        'logs', 'reports', 'artifacts', 'work_items', 'checkpoints',
    ):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'run_id': requested, 'created_at': utc_now(),
        'environment': environment, 'convention': convention_payload(),
        'sector_ordering': (
            'lexicographic six-tuple j2 with fixed (+,-,+,-,+,-) orientations'
        ),
        'rounding_policy': (
            'exact SymPy proof path; float64 checkpoint shards diagnostic only'
        ),
        **immutable_manifest_fields,
    }
    atomic_write_json(
        run_root / 'run_config.json', config.canonical_payload(),
    )
    atomic_write_json(run_root / 'test_report.json', test_report or {})
    atomic_write_json(manifest_path, manifest)
    _seed_everything(config.seed)
    state = RunState(
        requested, config_hash, utc_now(), utc_now(),
        milestone='M2', phase='M2_BOOTSTRAP',
    )
    queue = WorkQueue()
    for phase in M2_PHASE_ORDER:
        queue.add(
            phase, config_hash, {'milestone': 'M2', 'phase': phase},
            predicted_s=5.0 * 60.0,
        )
    manager = CheckpointManager(
        run_root, config, source_hash, notebook_hash,
    )
    orchestrator = M2Orchestrator(
        persistent_root, run_root, project_root, config, state,
        queue, manager, test_report or {}, manifest,
    )
    orchestrator.checkpoint('initial M2 run state')
    atomic_write_json(latest_pointer, {
        'milestone': 'M2', 'run_id': requested,
        'config_hash': config_hash, 'updated_at': utc_now(),
    })
    return orchestrator
