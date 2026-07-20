from __future__ import annotations

import json
import os
import random
import shutil
import time
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import CheckpointManager, CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, canonical_json_bytes, fsync_directory, hash_tree, read_json,
    safe_component, sha256_bytes, sha256_file, utc_now,
)
from .exact_2d_rg import trajectory_payload
from .m1_config import M1Config
from .m1_reporting import (
    load_phase_results, validate_m1_acceptance, write_m1_report_package,
    write_m1_session_artifacts,
)
from .m1_verifier import independent_convolution_verify
from .orchestrator import governing_document_hashes, reference_artifact_hashes
from .reporting import JsonlLogger
from .runtime import environment_info, runtime_compatibility_signature
from .session_guard import SessionGuard, SessionState
from .su2_representations import CONVENTION
from .tail_bounds import coefficient_enclosures, tail_table
from .work_queue import WorkItem, WorkQueue

try:
    import torch
except ImportError:
    torch = None

M1_PHASE_ORDER = (
    'M1_COEFFICIENT_BATCH', 'M1_VALUE_TAIL', 'M1_GRADIENT_TAIL',
    'M1_RG_TRAJECTORY', 'M1_INDEPENDENT_VERIFY', 'M1_REPORT',
)
M0_ACCEPTANCE_RECORD = 'audit/m0_accepted_parent.json'
M1_NOTEBOOK_HASH_POLICY = 'canonical_nbformat4_cell_type_source_tags_v1'


class M1CompatibilityError(RuntimeError):
    '''Raised when an M1 run or its accepted M0 parent is incompatible.'''


def _seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()


def _notebook_hash(project_root: Path) -> str | None:
    path = project_root / 'notebooks/20_m1_exact_2d.ipynb'
    if not path.is_file():
        return None
    payload = read_json(path)
    cells = payload.get('cells') if isinstance(payload, dict) else None
    if not isinstance(payload, dict) or payload.get('nbformat') != 4 or not isinstance(cells, list):
        raise M1CompatibilityError('M1 notebook must be a valid nbformat 4 document.')
    identity_cells: list[dict[str, Any]] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            raise M1CompatibilityError(f'M1 notebook cell {index} is not a mapping.')
        cell_type = cell.get('cell_type')
        source = cell.get('source')
        metadata = cell.get('metadata', {})
        if cell_type not in {'code', 'markdown', 'raw'}:
            raise M1CompatibilityError(f'M1 notebook cell {index} has an invalid type.')
        if isinstance(source, str):
            normalized_source = source
        elif isinstance(source, list) and all(isinstance(line, str) for line in source):
            normalized_source = ''.join(source)
        else:
            raise M1CompatibilityError(f'M1 notebook cell {index} has invalid source text.')
        if not isinstance(metadata, dict):
            raise M1CompatibilityError(f'M1 notebook cell {index} has invalid metadata.')
        tags = metadata.get('tags', [])
        if not isinstance(tags, list) or any(not isinstance(tag, str) for tag in tags):
            raise M1CompatibilityError(f'M1 notebook cell {index} has invalid execution tags.')
        identity_cells.append({
            'cell_type': cell_type, 'source': normalized_source, 'tags': tags,
        })
    identity = {
        'policy': M1_NOTEBOOK_HASH_POLICY, 'nbformat': 4, 'cells': identity_cells,
    }
    return sha256_bytes(canonical_json_bytes(identity))


def _validate_acceptance_record(project_root: Path, config: M1Config) -> str:
    path = project_root / M0_ACCEPTANCE_RECORD
    if not path.is_file():
        raise M1CompatibilityError('M0 acceptance record is missing.')
    payload = read_json(path)
    expected = {
        'parent_milestone': config.parent_milestone,
        'parent_run_id': config.parent_run_id,
        'parent_checkpoint': config.parent_checkpoint,
        'accepted_phase': 'M0_COMPLETE',
        'certification_status': 'NOT_CERTIFIED',
        'restart_test_status': 'PASS',
    }
    if not isinstance(payload, dict) or any(payload.get(key) != value for key, value in expected.items()):
        raise M1CompatibilityError('M0 acceptance record identity/status is invalid.')
    if payload.get('independent_checkpoint_audit_performed') is not False:
        raise M1CompatibilityError('M0 acceptance audit scope was silently changed.')
    return sha256_file(path)


def verify_accepted_m0_checkpoint(config: M1Config) -> str:
    checkpoint = Path(config.parent_checkpoint_path).expanduser().resolve()
    if checkpoint.name != config.parent_checkpoint or checkpoint.is_symlink() or not checkpoint.is_dir():
        raise M1CompatibilityError(f'Accepted M0 checkpoint path is unavailable or unsafe: {checkpoint}')
    if not (checkpoint / 'COMMITTED').is_file() or not (checkpoint / 'hashes.json').is_file():
        raise M1CompatibilityError('Accepted M0 checkpoint is not committed or lacks hashes.json.')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise M1CompatibilityError('Accepted M0 checkpoint contains a symlink.')
    expected = read_json(checkpoint / 'hashes.json')
    if not isinstance(expected, dict) or any(
        not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
        for name, digest in expected.items()
    ):
        raise M1CompatibilityError('Accepted M0 checkpoint hash manifest is malformed.')
    actual_files = {
        path.relative_to(checkpoint).as_posix() for path in checkpoint.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    if actual_files != set(expected):
        raise M1CompatibilityError('Accepted M0 checkpoint file set differs from hashes.json.')
    for relative, digest in expected.items():
        if sha256_file(checkpoint / relative) != digest:
            raise M1CompatibilityError(f'Accepted M0 checkpoint hash mismatch: {relative}')
    state = read_json(checkpoint / 'state.json')
    if not isinstance(state, dict) or (
        state.get('run_id') != config.parent_run_id
        or state.get('phase') != 'M0_COMPLETE'
        or state.get('certification_status') != 'NOT_CERTIFIED'
        or state.get('checkpoint_index') != 14
    ):
        raise M1CompatibilityError('Accepted M0 checkpoint state does not match the acceptance record.')
    queue = WorkQueue.from_payload(read_json(checkpoint / 'work_queue.json'))
    counts = {status: sum(item.status == status for item in queue.items.values()) for status in ('done', 'pending', 'running', 'failed')}
    if counts != {'done': 6, 'pending': 0, 'running': 0, 'failed': 0}:
        raise M1CompatibilityError(f'Accepted M0 queue is not complete: {counts}')
    parent_run_root = checkpoint.parents[1]
    for item in queue.items.values():
        if not item.result_relpath or not item.result_sha256:
            raise M1CompatibilityError(f'Accepted M0 done item lacks result metadata: {item.item_id}')
        result_path = (parent_run_root / item.result_relpath).resolve()
        try:
            result_path.relative_to(parent_run_root.resolve())
        except ValueError as exc:
            raise M1CompatibilityError('Accepted M0 result path escapes its run root.') from exc
        marker_path = parent_run_root / 'work_items' / f'{item.item_id}.done'
        if not result_path.is_file() or sha256_file(result_path) != item.result_sha256:
            raise M1CompatibilityError(f'Accepted M0 result artifact is missing or corrupt: {item.item_id}')
        marker = read_json(marker_path) if marker_path.is_file() else None
        if not isinstance(marker, dict) or (
            marker.get('item_id') != item.item_id
            or marker.get('result_relpath') != item.result_relpath
            or marker.get('result_sha256') != item.result_sha256
        ):
            raise M1CompatibilityError(f'Accepted M0 done marker is missing or inconsistent: {item.item_id}')
    return sha256_file(checkpoint / 'hashes.json')


class M1Orchestrator:
    def __init__(
        self, persistent_root: Path, run_root: Path, project_root: Path, config: M1Config,
        state: RunState, queue: WorkQueue, checkpoints: CheckpointManager,
        test_report: dict[str, Any], manifest: dict[str, Any],
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
        self.guard = SessionGuard(config)
        self.logger = JsonlLogger(run_root / 'logs' / 'events.jsonl')
        self.last_checkpoint: CheckpointSaveResult | None = None

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m1_safe()
        self.state.notes.append(f'{utc_now()} checkpoint: {reason}')
        result = self.checkpoints.save(self.state, self.queue, {})
        self.guard.mark_checkpoint(); self.last_checkpoint = result
        self.logger.emit('m1_checkpoint_committed', run_id=self.state.run_id, checkpoint=result.index, reason=reason)
        print(f'M1 checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _next_pending(self) -> WorkItem | None:
        for phase in M1_PHASE_ORDER:
            candidates = sorted(
                (item for item in self.queue.items.values() if item.phase == phase and item.status == 'pending'),
                key=lambda item: item.item_id,
            )
            if candidates:
                return candidates[0]
        return None

    def _phase_result(self, phase: str) -> dict[str, Any]:
        results = load_phase_results(self.run_root, self.queue)
        if phase not in results:
            raise RuntimeError(f'Required prior M1 phase is incomplete: {phase}')
        return results[phase]['result']

    def _compute_phase(self, item: WorkItem) -> dict[str, Any]:
        beta = Fraction(self.config.beta_numerator, self.config.beta_denominator)
        if item.phase == 'M1_COEFFICIENT_BATCH':
            return coefficient_enclosures(
                beta, max(self.config.cutoffs), self.config.coefficient_series_terms,
                self.config.exp_series_terms, self.config.decimal_places,
            )
        if item.phase == 'M1_VALUE_TAIL':
            return tail_table(
                beta, self.config.cutoffs, self.config.coefficient_series_terms,
                self.config.exp_series_terms, self.config.decimal_places, 'value',
            )
        if item.phase == 'M1_GRADIENT_TAIL':
            return tail_table(
                beta, self.config.cutoffs, self.config.coefficient_series_terms,
                self.config.exp_series_terms, self.config.decimal_places, 'gradient',
            )
        if item.phase == 'M1_RG_TRAJECTORY':
            return trajectory_payload(
                beta, self.config.dimensions, self.config.rg_steps,
                self.config.coefficient_series_terms, self.config.decimal_places,
            )
        if item.phase == 'M1_INDEPENDENT_VERIFY':
            return independent_convolution_verify(
                self._phase_result('M1_RG_TRAJECTORY'), beta, self.config.dimensions,
                self.config.rg_steps, self.config.verifier_series_terms,
            )
        if item.phase == 'M1_REPORT':
            required = M1_PHASE_ORDER[:-1]
            results = load_phase_results(self.run_root, self.queue)
            missing = [phase for phase in required if phase not in results]
            if missing:
                raise RuntimeError(f'M1 report work item is missing inputs: {missing}')
            return {'status': 'READY', 'input_phases': list(required)}
        raise RuntimeError(f'Unknown M1 work phase: {item.phase}')

    def _execute_item(self, item: WorkItem) -> tuple[str, str]:
        parent = self.run_root / 'artifacts' / item.item_id
        parent.mkdir(parents=True, exist_ok=True)
        temporary = parent / f'.tmp-attempt-{item.attempts:03d}-{uuid.uuid4().hex}'
        final = parent / f'attempt_{item.attempts:03d}'
        if final.exists():
            raise RuntimeError(f'M1 attempt output already exists: {final}')
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            result_file = temporary / 'result.json'
            atomic_write_json(result_file, {
                'schema_version': 1, 'milestone': 'M1', 'phase': item.phase,
                'item_id': item.item_id, 'config_hash': self.config.config_hash(),
                'certification_status': 'NOT_CERTIFIED', 'generated_at': utc_now(),
                'result': self._compute_phase(item),
            })
            fsync_directory(temporary)
            os.replace(temporary, final); fsync_directory(parent)
            committed = final / 'result.json'
            relative = committed.relative_to(self.run_root).as_posix(); digest = sha256_file(committed)
            atomic_write_json(self.run_root / 'work_items' / f'{item.item_id}.done', {
                'item_id': item.item_id, 'result_relpath': relative,
                'result_sha256': digest, 'completed_at': utc_now(),
            })
            if sha256_file(committed) != digest:
                raise RuntimeError('M1 result verification failed after commit.')
            return relative, digest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def run_one_item_for_test(self) -> str | None:
        item = self._next_pending()
        if item is None:
            return None
        item.attempts += 1; item.status = 'running'
        self.checkpoint(f'before test item {item.phase}')
        relative, digest = self._execute_item(item)
        item.result_relpath = relative; item.result_sha256 = digest; item.status = 'done'
        self.checkpoint(f'after test item {item.phase}')
        return item.phase

    def _summary(self, stop_reason: str) -> dict[str, Any]:
        elapsed = self.guard.elapsed_s(); remaining = self.guard.remaining_s()
        artifacts = write_m1_session_artifacts(
            self.run_root, self.state, self.queue, stop_reason, elapsed, remaining,
            self.persistent_root, self.project_root,
        )
        summary = {
            'milestone': 'M1', 'run_id': self.state.run_id, 'phase': self.state.phase,
            'checkpoint_index': self.state.checkpoint_index,
            'certification_status': 'NOT_CERTIFIED', 'stop_reason': stop_reason,
            'elapsed_s': elapsed, 'remaining_s': remaining, 'session_artifacts': artifacts,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        return summary

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m1_safe()
        if self.state.phase == 'M1_COMPLETE' and (self.run_root / 'reports' / 'M1_acceptance.json').is_file():
            return self._summary('M1 already complete; no work or checkpoint was started')
        self.state.phase = 'M1_RUNNING'
        self.logger.emit('m1_session_started', run_id=self.state.run_id)
        while True:
            session_state = self.guard.state()
            if session_state is SessionState.RETURN:
                return self._summary('hard return threshold reached; using last committed checkpoint')
            if session_state in {SessionState.DRAIN, SessionState.FINAL_SAVE}:
                self.checkpoint(f'M1 session state {session_state.value}')
                return self._summary(f'{session_state.value.lower()} checkpoint complete')
            if self.guard.checkpoint_due():
                self.checkpoint('periodic 15-minute M1 checkpoint')
            item = self._next_pending()
            if item is None:
                incomplete = [queued for queued in self.queue.items.values() if queued.status != 'done']
                if incomplete:
                    self.checkpoint('M1 queue has no runnable item but is incomplete')
                    raise RuntimeError('M1 cannot complete with failed/blocked/running work items.')
                results = load_phase_results(self.run_root, self.queue)
                validate_m1_acceptance(self.state, self.queue, results, self.test_report)
                self.state.bounds = {
                    'coefficient_enclosures': 'RIGOROUS_RATIONAL_POSITIVE_SERIES',
                    'value_tail': 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
                    'gradient_tail': 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
                    'exact_2d_rg': 'RIGOROUS_RATIONAL_INTERVAL_RECURRENCE',
                    'independent_verifier': 'PASS',
                }
                self.state.phase = 'M1_COMPLETE'
                final_checkpoint = self.checkpoint('M1 acceptance gates complete')
                paths = write_m1_report_package(
                    self.run_root, self.config, self.state, self.queue, self.test_report,
                    final_checkpoint, self.manifest,
                )
                self.logger.emit('m1_milestone_complete', run_id=self.state.run_id, reports=paths)
                return self._summary(f'M1 complete; report written to {paths["json"]}')
            predicted = self.queue.predicted_duration(item)
            if not self.guard.may_start(predicted):
                self.checkpoint('insufficient safe time for next M1 item')
                return self._summary('next M1 work item deferred to a fresh session')
            item.attempts += 1
            if item.attempts > self.config.max_item_attempts:
                item.status = 'failed'; item.last_error = 'Maximum M1 attempt count exceeded.'
                self.checkpoint('M1 work item exceeded attempt limit')
                raise RuntimeError(item.last_error)
            item.status = 'running'; self.checkpoint(f'before M1 item {item.phase}')
            started = time.monotonic()
            try:
                relative, digest = self._execute_item(item)
                item.result_relpath = relative; item.result_sha256 = digest; item.status = 'done'
                item.last_error = None; self.queue.record_timing(item.phase, time.monotonic() - started)
                if self.guard.state() is SessionState.RETURN:
                    return self._summary('hard return reached after atomic M1 done marker; resume will repair queue')
                self.checkpoint(f'after M1 item {item.phase}')
            except KeyboardInterrupt:
                item.status = 'pending'; item.last_error = 'KeyboardInterrupt'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'interrupted M1 item {item.phase}')
                self._summary(f'KeyboardInterrupt in M1 item {item.phase}')
                raise
            except Exception as exc:
                item.status = 'failed' if item.attempts >= self.config.max_item_attempts else 'pending'
                item.last_error = f'{type(exc).__name__}: {exc}'
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint(f'exception in M1 item {item.phase}')
                self._summary(f'exception in M1 item {item.phase}: {type(exc).__name__}')
                raise


def create_or_resume_m1(
    persistent_root: Path, config: M1Config, project_root: Path,
    run_id: str | None = None, test_report: dict[str, Any] | None = None,
) -> M1Orchestrator:
    acceptance_hash = _validate_acceptance_record(project_root, config)
    parent_hash = verify_accepted_m0_checkpoint(config)
    config_hash = config.config_hash(); source_hash = hash_tree(project_root / 'src')
    notebook_hash = _notebook_hash(project_root)
    document_hashes = governing_document_hashes(project_root)
    reference_hashes = reference_artifact_hashes(project_root)
    environment = environment_info(); runtime_signature = runtime_compatibility_signature(environment)
    convention_hash = sha256_bytes(canonical_json_bytes(CONVENTION))
    runs_root = persistent_root / 'runs'; runs_root.mkdir(parents=True, exist_ok=True)
    requested = run_id or os.environ.get('VALIDATED_RG_M1_RUN_ID')
    latest_pointer = persistent_root / 'LATEST_M1_RUN.json'
    if requested is None and latest_pointer.is_file():
        pointer = read_json(latest_pointer)
        if isinstance(pointer, dict) and pointer.get('config_hash') == config_hash:
            requested = pointer.get('run_id')
    if requested is None:
        requested = 'M1-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + config_hash[:12]
    if not isinstance(requested, str):
        raise M1CompatibilityError('M1 run ID must be a string.')
    safe_component(requested)
    if not requested.startswith('M1-'):
        raise M1CompatibilityError('M1 run ID must use the M1- namespace and cannot resume M0.')
    run_root = runs_root / requested; manifest_path = run_root / 'run_manifest.json'
    immutable_manifest_fields = {
        'milestone': 'M1', 'parent_milestone': config.parent_milestone,
        'parent_run_id': config.parent_run_id, 'parent_checkpoint': config.parent_checkpoint,
        'parent_checkpoint_path': config.parent_checkpoint_path,
        'parent_checkpoint_hash_manifest_sha256': parent_hash,
        'm0_acceptance_record_sha256': acceptance_hash, 'config_hash': config_hash,
        'source_hash': source_hash, 'convention_hash': convention_hash,
        'governing_document_hashes': document_hashes,
        'notebook_hash_policy': M1_NOTEBOOK_HASH_POLICY,
        'reference_artifact_hashes': reference_hashes,
        'runtime_compatibility': runtime_signature, 'certification_status': 'NOT_CERTIFIED',
    }
    if run_root.exists() and (run_root / 'run_config.json').is_file():
        if read_json(run_root / 'run_config.json') != config.canonical_payload():
            raise M1CompatibilityError('Immutable M1 run configuration differs.')
        if not manifest_path.is_file():
            raise M1CompatibilityError('Existing M1 run lacks run_manifest.json.')
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in immutable_manifest_fields.items()):
            raise M1CompatibilityError('M1 manifest/source/parent/runtime identity changed.')
        manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise M1CompatibilityError('Existing M1 run has no valid checkpoint.')
        repaired = loaded.queue.recover_interrupted(run_root)
        orchestrator = M1Orchestrator(
            persistent_root, run_root, project_root, config, loaded.state, loaded.queue,
            manager, test_report or {}, manifest,
        )
        if repaired:
            orchestrator.checkpoint(f'recovered {len(repaired)} interrupted M1 item(s)')
        print('Resumed M1 from:', loaded.path)
        return orchestrator
    if run_root.exists():
        raise M1CompatibilityError('M1 run directory exists but is incomplete; refusing overwrite.')
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ('logs', 'reports', 'artifacts', 'work_items', 'checkpoints'):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': 1, 'run_id': requested, 'created_at': utc_now(),
        'notebook_hash': notebook_hash, 'environment': environment,
        'sector_ordering': 'ascending irrep dimension n',
        'rounding_policy': 'exact Fraction endpoints; outward Decimal display only',
        **immutable_manifest_fields,
    }
    atomic_write_json(run_root / 'run_config.json', config.canonical_payload())
    atomic_write_json(manifest_path, manifest)
    _seed_everything(config.seed)
    state = RunState(
        requested, config_hash, utc_now(), utc_now(), milestone='M1', phase='M1_BOOTSTRAP',
    )
    queue = WorkQueue()
    for phase in M1_PHASE_ORDER:
        queue.add(phase, config_hash, {'milestone': 'M1', 'phase': phase}, predicted_s=60.0)
    manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
    orchestrator = M1Orchestrator(
        persistent_root, run_root, project_root, config, state, queue, manager,
        test_report or {}, manifest,
    )
    orchestrator.checkpoint('initial M1 run state')
    atomic_write_json(latest_pointer, {
        'milestone': 'M1', 'run_id': requested, 'config_hash': config_hash, 'updated_at': utc_now(),
    })
    return orchestrator
