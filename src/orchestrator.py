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

from .checkpoint import CheckpointManager, CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, fsync_descriptor, fsync_directory, fsync_file, hash_tree, read_json,
    sha256_file, utc_now,
)
from .config import RunConfig
from .reporting import (
    JsonlLogger, next_session_instructions, write_m0_report, write_session_artifacts,
)
from .runtime import environment_info, runtime_compatibility_signature
from .session_guard import SessionGuard, SessionState
from .work_queue import WorkItem, WorkQueue

try:
    import torch
except ImportError:
    torch = None

GENERATED_FILES = [
    'notebooks/20_m1_exact_2d.ipynb', 'README.md', 'requirements/paperspace.txt',
    'src/__init__.py', 'src/config.py', 'src/common.py', 'src/runtime.py',
    'src/session_guard.py', 'src/work_queue.py', 'src/checkpoint.py',
    'src/reporting.py', 'src/orchestrator.py', 'tests/test_m0.py', 'pytest.ini',
]

GOVERNING_DOCUMENTS = [
    'validated_4d_su2_rg_codex_design.md', 'AGENTS.md',
    'validated_4d_su2_rg_full_plan_bundle/validated_4d_su2_rg_codex_design_v0_2.md',
    'validated_4d_su2_rg_full_plan_bundle/M1_M6_VALIDATED_RG_ROADMAP.md',
    'validated_4d_su2_rg_full_plan_bundle/MATHEMATICAL_CERTIFICATION_SPEC.md',
    'validated_4d_su2_rg_full_plan_bundle/CODEX_PROMPTS_M1_M6.md',
    'validated_4d_su2_rg_full_plan_bundle/AGENTS_validated_4d_su2_rg_v0_2.md',
]

REFERENCE_ARTIFACTS = [
    'validated_4d_su2_rg_full_plan_bundle/validated_su2_rg_prototype.py',
    'validated_4d_su2_rg_full_plan_bundle/validated_su2_rg_report.md',
    'validated_4d_su2_rg_full_plan_bundle/validated_4d_su2_rg_M1_M6_tracker.ipynb',
]


def governing_document_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in GOVERNING_DOCUMENTS:
        path = project_root / relative
        if not path.is_file():
            raise RunCompatibilityError(f'Missing governing document: {relative}')
        hashes[relative] = sha256_file(path)
    return hashes


def reference_artifact_hashes(project_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REFERENCE_ARTIFACTS:
        path = project_root / relative
        if not path.is_file():
            raise RunCompatibilityError(f'Missing full-plan reference artifact: {relative}')
        hashes[relative] = sha256_file(path)
    return hashes


class RunCompatibilityError(RuntimeError):
    '''Raised when an existing durable run cannot be resumed exactly.'''


class Orchestrator:
    def __init__(
        self,
        persistent_root: Path,
        run_root: Path,
        project_root: Path,
        config: RunConfig,
        state: RunState,
        queue: WorkQueue,
        checkpoints: CheckpointManager,
        prefer_cuda: bool,
        test_report: dict[str, Any],
    ) -> None:
        self.persistent_root = persistent_root
        self.run_root = run_root
        self.project_root = project_root
        self.config = config
        self.state = state
        self.queue = queue
        self.checkpoints = checkpoints
        self.prefer_cuda = prefer_cuda
        self.test_report = test_report
        self.guard = SessionGuard(config)
        self.logger = JsonlLogger(run_root / 'logs' / 'events.jsonl')
        self.tensors: dict[str, Any] = {}
        self.last_checkpoint: CheckpointSaveResult | None = None

    def checkpoint(self, reason: str) -> CheckpointSaveResult:
        self.state.assert_m0_safe()
        self.state.notes.append(f'{utc_now()} checkpoint: {reason}')
        result = self.checkpoints.save(self.state, self.queue, self.tensors)
        self.guard.mark_checkpoint()
        self.last_checkpoint = result
        self.logger.emit(
            'checkpoint_committed', run_id=self.state.run_id, phase=self.state.phase,
            checkpoint=result.index, reason=reason, size_bytes=result.size_bytes,
            save_s=result.save_s, verify_s=result.verify_s,
        )
        print(f'Checkpoint {result.index:06d} committed and verified: {result.path}')
        return result

    def _execute_dummy(self, item: WorkItem) -> tuple[str, str]:
        size = int(item.parameters.get('size', self.config.dummy_size))
        steps = int(item.parameters.get('steps', self.config.dummy_steps))
        if size <= 0 or steps <= 0:
            raise ValueError('Dummy dimensions and steps must be positive.')
        artifact_parent = self.run_root / 'artifacts' / item.item_id
        artifact_parent.mkdir(parents=True, exist_ok=True)
        temporary = artifact_parent / f'.tmp-attempt-{item.attempts:03d}-{uuid.uuid4().hex}'
        final = artifact_parent / f'attempt_{item.attempts:03d}'
        if final.exists():
            raise RuntimeError(f'Attempt output already exists: {final}')
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            if torch is not None:
                device = 'cuda' if self.prefer_cuda and torch.cuda.is_available() else 'cpu'
                value = torch.eye(size, dtype=torch.float64, device=device)
                for _ in range(steps):
                    value = value @ value.mT
                    norm = torch.linalg.vector_norm(value)
                    if not bool(torch.isfinite(norm)) or float(norm) <= 0.0:
                        raise FloatingPointError('Dummy torch workload produced invalid normalization.')
                    value = value / norm
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError('Dummy torch workload produced NaN or Inf.')
                result_file = temporary / 'result.pt'
                torch.save(value[:8, :8].cpu(), result_file)
                fsync_file(result_file)
                device_name = device
            else:
                value = np.eye(size, dtype=np.float64)
                for _ in range(steps):
                    value = value @ value.T
                    norm = np.linalg.norm(value)
                    if not np.isfinite(norm) or norm <= 0.0:
                        raise FloatingPointError('Dummy NumPy workload produced invalid normalization.')
                    value = value / norm
                if not np.isfinite(value).all():
                    raise FloatingPointError('Dummy NumPy workload produced NaN or Inf.')
                result_file = temporary / 'result.npy'
                with result_file.open('wb') as handle:
                    np.save(handle, value[:8, :8], allow_pickle=False)
                    handle.flush()
                    fsync_descriptor(handle.fileno())
                device_name = 'cpu-numpy'
            atomic_write_json(temporary / 'metadata.json', {
                'item_id': item.item_id, 'attempt': item.attempts, 'device': device_name,
                'certification_status': 'NOT_CERTIFIED', 'created_at': utc_now(),
            })
            fsync_directory(temporary)
            os.replace(temporary, final)
            fsync_directory(artifact_parent)
            committed_result = final / result_file.name
            relative = committed_result.relative_to(self.run_root).as_posix()
            digest = sha256_file(committed_result)
            marker = self.run_root / 'work_items' / f'{item.item_id}.done'
            atomic_write_json(marker, {
                'item_id': item.item_id, 'result_relpath': relative,
                'result_sha256': digest, 'completed_at': utc_now(),
            })
            if sha256_file(committed_result) != digest:
                raise RuntimeError('Result verification failed immediately after commit.')
            return relative, digest
        except Exception:
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _summary(self, stop_reason: str) -> dict[str, Any]:
        self.state.assert_m0_safe()
        elapsed_s = self.guard.elapsed_s()
        remaining_s = self.guard.remaining_s()
        summary = {
            'run_id': self.state.run_id,
            'phase': self.state.phase,
            'checkpoint_index': self.state.checkpoint_index,
            'certification_status': self.state.certification_status,
            'elapsed_s': elapsed_s,
            'remaining_s': remaining_s,
            'session_state': self.guard.state().value,
            'stop_reason': stop_reason,
        }
        summary['session_artifacts'] = write_session_artifacts(
            self.run_root, self.state, self.queue, stop_reason=stop_reason,
            elapsed_s=elapsed_s, remaining_s=remaining_s,
            persistent_root=self.persistent_root, project_root=self.project_root,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        if self.state.phase == 'M0_COMPLETE':
            print('NEXT ACTION: inspect reports/M0_report.json; M0 is complete and M1 is not started automatically.')
        else:
            print(next_session_instructions(self.persistent_root, self.state.run_id, self.project_root))
        return summary

    def run_until_checkpoint(self) -> dict[str, Any]:
        self.state.assert_m0_safe()
        self.logger.emit('session_started', run_id=self.state.run_id, phase=self.state.phase)
        stop_reason = 'unknown'
        while True:
            session_state = self.guard.state()
            if session_state is SessionState.RETURN:
                stop_reason = 'hard return threshold reached; returning with last verified checkpoint'
                break
            if session_state in {SessionState.DRAIN, SessionState.FINAL_SAVE}:
                self.checkpoint(f'session state {session_state.value}')
                stop_reason = f'{session_state.value.lower()} checkpoint complete'
                break
            if self.guard.checkpoint_due():
                self.checkpoint('periodic 15-minute checkpoint')
            item = self.queue.next_pending()
            if item is None:
                incomplete = [
                    queued for queued in self.queue.items.values() if queued.status != 'done'
                ]
                if incomplete:
                    self.checkpoint('queue has no runnable item but contains incomplete work')
                    summary = ', '.join(f'{queued.item_id[:12]}:{queued.status}' for queued in incomplete)
                    raise RuntimeError(f'M0 cannot complete with incomplete work items: {summary}')
                self.state.phase = 'M0_COMPLETE'
                self.checkpoint('M0 work queue complete')
                report = write_m0_report(
                    self.run_root, self.state, self.queue, self.test_report,
                    self.last_checkpoint, GENERATED_FILES,
                )
                self.logger.emit('milestone_complete', run_id=self.state.run_id, phase=self.state.phase, report=str(report))
                stop_reason = f'M0 complete; report written to {report}'
                break
            predicted_s = self.queue.predicted_duration(item)
            if not self.guard.may_start(predicted_s):
                self.checkpoint('insufficient safe time for next bounded item')
                stop_reason = 'next work item deferred to a fresh session'
                break
            item.attempts += 1
            if item.attempts > self.config.max_item_attempts:
                item.status = 'failed'
                item.last_error = 'Maximum attempt count exceeded.'
                self.checkpoint('item exceeded maximum attempts')
                raise RuntimeError(f'Work item permanently failed: {item.item_id}')
            item.status = 'running'
            self.state.phase = item.phase
            self.checkpoint(f'before work item {item.item_id[:12]}')
            started = time.monotonic()
            try:
                if item.phase != 'DUMMY':
                    raise RunCompatibilityError(f'Queue phase {item.phase!r} is outside the approved M0 scope.')
                relative, digest = self._execute_dummy(item)
                elapsed = time.monotonic() - started
                item.result_relpath = relative
                item.result_sha256 = digest
                item.status = 'done'
                item.last_error = None
                self.queue.record_timing(item.phase, elapsed)
                self.logger.emit(
                    'item_completed', run_id=self.state.run_id, phase=item.phase,
                    item_id=item.item_id, elapsed_s=elapsed, result_sha256=digest,
                )
                if self.guard.state() is SessionState.RETURN:
                    stop_reason = (
                        'hard return reached after artifact commit; the prior running-item checkpoint '
                        'will repair from the verified done marker on resume'
                    )
                    break
                self.checkpoint(f'after work item {item.item_id[:12]}')
            except KeyboardInterrupt:
                item.status = 'pending'
                item.last_error = 'KeyboardInterrupt'
                item.result_relpath = None
                item.result_sha256 = None
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint('KeyboardInterrupt recovery')
                print(next_session_instructions(self.persistent_root, self.state.run_id, self.project_root))
                raise
            except Exception as exc:
                item.last_error = repr(exc)
                item.status = 'failed' if item.attempts >= self.config.max_item_attempts else 'pending'
                item.result_relpath = None
                item.result_sha256 = None
                if self.guard.state() is not SessionState.RETURN:
                    self.checkpoint('work item exception')
                print(next_session_instructions(self.persistent_root, self.state.run_id, self.project_root))
                raise
        self.logger.emit('session_stopped', run_id=self.state.run_id, reason=stop_reason)
        return self._summary(stop_reason)


def _notebook_hash(project_root: Path) -> str | None:
    notebook = project_root / 'notebooks/20_m1_exact_2d.ipynb'
    return sha256_file(notebook) if notebook.is_file() else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()


def create_or_resume(
    persistent_root: Path,
    config: RunConfig,
    project_root: Path,
    run_id: str | None = None,
    prefer_cuda: bool = True,
    test_report: dict[str, Any] | None = None,
) -> Orchestrator:
    if prefer_cuda != config.prefer_cuda:
        raise RunCompatibilityError('prefer_cuda differs from the immutable run configuration.')
    config_hash = config.config_hash()
    source_hash = hash_tree(project_root / 'src')
    notebook_hash = _notebook_hash(project_root)
    document_hashes = governing_document_hashes(project_root)
    reference_hashes = reference_artifact_hashes(project_root)
    current_environment = environment_info()
    current_runtime_signature = runtime_compatibility_signature(current_environment)
    runs_root = persistent_root / 'runs'
    runs_root.mkdir(parents=True, exist_ok=True)
    requested = run_id or os.environ.get('VALIDATED_RG_RUN_ID')
    latest_pointer = persistent_root / 'LATEST_RUN.json'
    if requested is None and latest_pointer.is_file():
        pointer = read_json(latest_pointer)
        if not isinstance(pointer, dict):
            raise RunCompatibilityError('LATEST_RUN.json is malformed; choose a run ID explicitly.')
        if pointer.get('config_hash') == config_hash:
            requested = pointer.get('run_id')
            if not isinstance(requested, str):
                raise RunCompatibilityError('LATEST_RUN.json has no valid run_id; choose one explicitly.')
    if requested is None:
        requested = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + config_hash[:12]
    if not isinstance(requested, str) or not requested or '/' in requested or '\\' in requested or requested in {'.', '..'}:
        raise ValueError(f'Unsafe run_id: {requested!r}')
    run_root = runs_root / requested
    if run_root.exists() and (run_root / 'run_config.json').is_file():
        stored_config = read_json(run_root / 'run_config.json')
        if stored_config != config.canonical_payload():
            raise RunCompatibilityError('Immutable run configuration differs from the stored run.')
        manifest_path = run_root / 'run_manifest.json'
        if not manifest_path.is_file():
            raise RunCompatibilityError('Run directory is incomplete (run_manifest.json is missing).')
        manifest = read_json(manifest_path)
        if not isinstance(manifest, dict) or (
            manifest.get('run_id') != requested
            or manifest.get('config_hash') != config_hash
            or manifest.get('certification_status') != 'NOT_CERTIFIED'
            or manifest.get('sector_ordering') != 'M0_NONE'
            or manifest.get('normalization_convention') != 'M0_NONE'
            or manifest.get('wigner_convention') != 'M0_NONE'
        ):
            raise RunCompatibilityError('Run manifest identity or certification invariant is invalid.')
        if manifest.get('source_hash') != source_hash:
            raise RunCompatibilityError('Generated source changed; start a new run or perform an explicit migration.')
        if manifest.get('governing_document_hashes') != document_hashes:
            raise RunCompatibilityError('A governing M0/M1-M6 document changed; start a new run or audit explicitly.')
        if manifest.get('reference_artifact_hashes') != reference_hashes:
            raise RunCompatibilityError('A full-plan prototype/report/tracker changed; start a new run or audit explicitly.')
        if manifest.get('runtime_compatibility') != current_runtime_signature:
            raise RunCompatibilityError(
                'Python/NumPy/PyTorch/CUDA/GPU runtime changed; resume with the original Paperspace runtime.'
            )
        required_directories = ('logs', 'reports', 'artifacts', 'work_items', 'checkpoints')
        missing_directories = [name for name in required_directories if not (run_root / name).is_dir()]
        if missing_directories:
            raise RunCompatibilityError(f'Run directory is incomplete; missing directories: {missing_directories}')
        manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
        loaded = manager.load_latest(restore_rng=True)
        if loaded is None:
            raise RunCompatibilityError('Existing run has no valid checkpoint.')
        repaired = loaded.queue.recover_interrupted(run_root)
        orchestrator = Orchestrator(
            persistent_root, run_root, project_root, config, loaded.state, loaded.queue,
            manager, prefer_cuda, test_report or {},
        )
        orchestrator.tensors = loaded.tensors
        print('Resumed from:', loaded.path)
        for warning in loaded.skipped_invalid:
            print('Fallback skipped invalid checkpoint:', warning)
        if repaired:
            orchestrator.checkpoint(f'recovered {len(repaired)} interrupted work item(s)')
        return orchestrator
    if run_root.exists():
        raise RunCompatibilityError(
            'Run directory exists but is incomplete (run_config.json is missing); refusing to overwrite it.'
        )
    run_root.mkdir(parents=True, exist_ok=False)
    for relative in ('logs', 'reports', 'artifacts', 'work_items', 'checkpoints'):
        (run_root / relative).mkdir(parents=True, exist_ok=True)
    manager = CheckpointManager(run_root, config, source_hash, notebook_hash)
    atomic_write_json(run_root / 'run_config.json', config.canonical_payload())
    atomic_write_json(run_root / 'run_manifest.json', {
        'schema_version': 1, 'run_id': requested, 'created_at': utc_now(),
        'config_hash': config_hash, 'source_hash': source_hash,
        'notebook_hash': notebook_hash, 'environment': current_environment,
        'runtime_compatibility': current_runtime_signature,
        'governing_document_hashes': document_hashes,
        'reference_artifact_hashes': reference_hashes,
        'sector_ordering': 'M0_NONE', 'normalization_convention': 'M0_NONE',
        'wigner_convention': 'M0_NONE', 'certification_status': 'NOT_CERTIFIED',
    })
    _seed_everything(config.seed)
    state = RunState(requested, config_hash, utc_now(), utc_now())
    queue = WorkQueue()
    input_hash = config_hash
    for index in range(config.dummy_items):
        queue.add(
            'DUMMY', input_hash, {'index': index, 'size': config.dummy_size, 'steps': config.dummy_steps},
            predicted_s=config.dummy_predicted_s,
        )
    orchestrator = Orchestrator(
        persistent_root, run_root, project_root, config, state, queue, manager,
        prefer_cuda, test_report or {},
    )
    orchestrator.checkpoint('initial run state')
    atomic_write_json(latest_pointer, {'run_id': requested, 'config_hash': config_hash, 'updated_at': utc_now()})
    return orchestrator
