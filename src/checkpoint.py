from __future__ import annotations

import os
import pickle
import random
import shutil
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .common import (
    atomic_write_json, atomic_write_text, directory_size, fsync_descriptor, fsync_directory,
    fsync_file, read_json,
    safe_component, sha256_file, utc_now,
)
from .runtime import environment_info
from .work_queue import WorkQueue

try:
    import torch
except ImportError:
    torch = None


class CheckpointError(RuntimeError):
    '''Raised when checkpoint creation, validation, or restoration fails.'''


class ConfigMismatchError(CheckpointError):
    '''Raised when a checkpoint belongs to incompatible immutable inputs.'''


class CheckpointConfig(Protocol):
    tensor_shard_bytes: int

    def config_hash(self) -> str: ...

    def canonical_payload(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class RunState:
    run_id: str
    config_hash: str
    created_at: str
    updated_at: str
    milestone: str = 'M0'
    phase: str = 'BOOTSTRAP'
    subphase: str = ''
    rg_step: int = 0
    direction: str = ''
    checkpoint_index: int = 0
    certification_status: str = 'NOT_CERTIFIED'
    bounds: dict[str, str] = field(default_factory=dict)
    normalization_log: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def assert_safe(self) -> None:
        if self.milestone == 'M0':
            self.assert_m0_safe()
        elif self.milestone == 'M1':
            self.assert_m1_safe()
        elif self.milestone == 'M2':
            self.assert_m2_safe()
        elif self.milestone == 'M3':
            self.assert_m3_safe()
        elif self.milestone == 'M4':
            self.assert_m4_safe()
        else:
            raise CheckpointError(f'Unsupported milestone in checkpoint state: {self.milestone!r}')

    def _assert_common_safe(self) -> None:
        if self.certification_status != 'NOT_CERTIFIED':
            raise CheckpointError('Checkpoint state attempted to leave NOT_CERTIFIED.')
        safe_component(self.run_id)
        if not isinstance(self.checkpoint_index, int) or self.checkpoint_index < 0:
            raise CheckpointError('Checkpoint index must be a nonnegative integer.')
        if not all(isinstance(value, str) for value in (self.config_hash, self.created_at, self.updated_at)):
            raise CheckpointError('State identity/timestamp fields must be strings.')
        if len(self.config_hash) != 64 or any(character not in '0123456789abcdef' for character in self.config_hash):
            raise CheckpointError('State config hash is malformed.')
        if not all(isinstance(value, str) for value in self.notes + self.normalization_log):
            raise CheckpointError('State notes and normalization log must contain strings only.')

    def assert_m0_safe(self) -> None:
        self._assert_common_safe()
        if self.milestone != 'M0':
            raise CheckpointError('M0 assertion received a non-M0 state.')
        if self.phase not in {'BOOTSTRAP', 'DUMMY', 'M0_COMPLETE'}:
            raise CheckpointError(f'Phase {self.phase!r} is outside M0.')
        if self.rg_step != 0 or self.direction or self.subphase:
            raise CheckpointError('M0 state may not contain RG progression fields.')
        if self.bounds:
            raise CheckpointError('M0 constructs no rigorous bounds; bounds must remain absent.')

    def assert_m1_safe(self) -> None:
        self._assert_common_safe()
        if self.milestone != 'M1':
            raise CheckpointError('M1 assertion received a non-M1 state.')
        if self.phase not in {'M1_BOOTSTRAP', 'M1_RUNNING', 'M1_COMPLETE'}:
            raise CheckpointError(f'Phase {self.phase!r} is outside M1.')
        if self.rg_step != 0 or self.direction or self.subphase:
            raise CheckpointError('M1 state may not contain 4D RG progression fields.')
        self._assert_bound_mapping('M1')

    def assert_m2_safe(self) -> None:
        self._assert_common_safe()
        if self.milestone != 'M2':
            raise CheckpointError('M2 assertion received a non-M2 state.')
        if self.phase not in {'M2_BOOTSTRAP', 'M2_RUNNING', 'M2_COMPLETE'}:
            raise CheckpointError(f'Phase {self.phase!r} is outside M2.')
        if self.rg_step != 0 or self.direction or self.subphase:
            raise CheckpointError(
                'M2 establishes a local basis identity and may not contain RG progression.'
            )
        self._assert_bound_mapping('M2')

    def assert_m3_safe(self) -> None:
        self._assert_common_safe()
        if self.milestone != 'M3':
            raise CheckpointError('M3 assertion received a non-M3 state.')
        if self.phase not in {'M3_BOOTSTRAP', 'M3_RUNNING', 'M3_COMPLETE'}:
            raise CheckpointError(f'Phase {self.phase!r} is outside M3.')
        if self.rg_step != 0 or self.direction or self.subphase:
            raise CheckpointError(
                'M3 is an exploratory local pilot and may not claim RG progression.'
            )
        self._assert_bound_mapping('M3')

    def assert_m4_safe(self) -> None:
        self._assert_common_safe()
        if self.milestone != 'M4':
            raise CheckpointError('M4 assertion received a non-M4 state.')
        if self.phase not in {'M4_BOOTSTRAP', 'M4_RUNNING', 'M4_COMPLETE'}:
            raise CheckpointError(f'Phase {self.phase!r} is outside M4.')
        if self.rg_step != 0 or self.direction or self.subphase:
            raise CheckpointError(
                'M4 defines derivative/error state but may not claim RG progression.'
            )
        self._assert_bound_mapping('M4')

    def _assert_bound_mapping(self, milestone: str) -> None:
        if not isinstance(self.bounds, dict) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in self.bounds.items()
        ):
            raise CheckpointError(f'{milestone} bound provenance must be a string mapping.')


@dataclass(frozen=True, slots=True)
class CheckpointSaveResult:
    path: Path
    index: int
    size_bytes: int
    save_s: float
    verify_s: float


@dataclass(frozen=True, slots=True)
class LoadedCheckpoint:
    state: RunState
    queue: WorkQueue
    path: Path
    tensors: dict[str, Any]
    skipped_invalid: tuple[str, ...]


def _rng_runtime_signature() -> dict[str, Any]:
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    return {
        'python': sys.version,
        'numpy': np.__version__,
        'torch': getattr(torch, '__version__', None),
        'cuda_runtime': getattr(getattr(torch, 'version', None), 'cuda', None) if torch is not None else None,
        'cuda_available': cuda_available,
        'cuda_device_count': torch.cuda.device_count() if cuda_available else 0,
    }


def capture_rng_state() -> dict[str, Any]:
    payload: dict[str, Any] = {
        'python': random.getstate(), 'numpy': np.random.get_state(),
        'runtime_signature': _rng_runtime_signature(),
    }
    if torch is not None:
        payload['torch_cpu'] = torch.get_rng_state().cpu()
        if torch.cuda.is_available():
            payload['torch_cuda'] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return payload


def restore_rng_state(payload: dict[str, Any]) -> None:
    if not isinstance(payload, dict) or 'python' not in payload or 'numpy' not in payload:
        raise CheckpointError('RNG checkpoint is missing Python or NumPy state.')
    if payload.get('runtime_signature') != _rng_runtime_signature():
        raise CheckpointError('RNG runtime signature changed; exact restoration is impossible.')
    saved_with_torch = 'torch_cpu' in payload
    if saved_with_torch != (torch is not None):
        raise CheckpointError('PyTorch availability changed; exact RNG restoration is impossible.')
    saved_with_cuda = 'torch_cuda' in payload
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    if saved_with_cuda != cuda_available:
        raise CheckpointError('CUDA availability changed; exact RNG restoration is impossible.')
    random.setstate(payload['python'])
    np.random.set_state(payload['numpy'])
    if torch is not None:
        torch.set_rng_state(payload['torch_cpu'])
        if cuda_available:
            if len(payload['torch_cuda']) != torch.cuda.device_count():
                raise CheckpointError('CUDA RNG device count changed; refusing partial restoration.')
            torch.cuda.set_rng_state_all(payload['torch_cuda'])


def _write_pickle(path: Path, payload: Any) -> None:
    with path.open('wb') as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        fsync_descriptor(handle.fileno())


class TensorShardStore:
    def __init__(self, shard_bytes: int) -> None:
        if shard_bytes <= 0:
            raise ValueError('shard_bytes must be positive.')
        self.shard_bytes = shard_bytes

    def save(self, root: Path, tensors: dict[str, Any]) -> dict[str, Any]:
        root.mkdir(parents=True, exist_ok=False)
        index: dict[str, Any] = {}
        for name, value in sorted(tensors.items()):
            safe_component(name)
            is_torch = torch is not None and isinstance(value, torch.Tensor)
            if is_torch:
                array = value.detach().cpu().contiguous()
                shape = tuple(array.shape)
                item_size = array.element_size()
                kind = 'torch'
                dtype = str(array.dtype)
            elif isinstance(value, np.ndarray):
                array = np.array(value, copy=True, order='C', subok=False)
                shape = tuple(array.shape)
                item_size = array.dtype.itemsize
                kind = 'numpy'
                dtype = str(array.dtype)
            else:
                raise TypeError(f'Unsupported tensor type for {name}: {type(value)!r}')
            flat = array.reshape(-1)
            element_count = int(flat.numel()) if kind == 'torch' else int(flat.size)
            elements_per_shard = max(1, self.shard_bytes // max(1, item_size))
            files: list[str] = []
            starts = (0,) if element_count == 0 else range(0, element_count, elements_per_shard)
            for shard_index, start in enumerate(starts):
                stop = min(element_count, start + elements_per_shard)
                shard = flat[start:stop]
                suffix = '.pt' if kind == 'torch' else '.npy'
                filename = f'{name}.shard-{shard_index:06d}{suffix}'
                path = root / filename
                if kind == 'torch':
                    torch.save(shard, path)
                    fsync_file(path)
                else:
                    with path.open('wb') as handle:
                        np.save(handle, shard, allow_pickle=False)
                        handle.flush()
                        fsync_descriptor(handle.fileno())
                files.append(filename)
            index[name] = {'kind': kind, 'dtype': dtype, 'shape': list(shape), 'files': files}
        atomic_write_json(root / 'index.json', index)
        return index

    def load(self, root: Path) -> dict[str, Any]:
        index = read_json(root / 'index.json')
        if not isinstance(index, dict):
            raise CheckpointError('Tensor index must be a mapping.')
        result: dict[str, Any] = {}
        for name, metadata in index.items():
            safe_component(name)
            if not isinstance(metadata, dict):
                raise CheckpointError(f'Tensor metadata must be a mapping: {name}')
            shape = metadata.get('shape')
            if not isinstance(shape, list) or any(
                not isinstance(size, int) or isinstance(size, bool) or size < 0 for size in shape
            ):
                raise CheckpointError(f'Invalid tensor shape metadata: {name}')
            kind = metadata.get('kind')
            if kind not in {'torch', 'numpy'} or not isinstance(metadata.get('dtype'), str):
                raise CheckpointError(f'Invalid tensor kind or dtype metadata: {name}')
            raw_files = metadata['files']
            if not isinstance(raw_files, list) or not raw_files:
                raise CheckpointError(f'Tensor {name} has no shard files.')
            if any(not isinstance(filename, str) for filename in raw_files):
                raise CheckpointError(f'Tensor {name} has a non-string shard name.')
            if len(set(raw_files)) != len(raw_files) or (shape == [] and len(raw_files) != 1):
                raise CheckpointError(f'Tensor {name} has invalid or duplicate shard entries.')
            files: list[Path] = []
            for shard_index, filename in enumerate(raw_files):
                safe_component(filename)
                suffix = '.pt' if kind == 'torch' else '.npy'
                expected_filename = f'{name}.shard-{shard_index:06d}{suffix}'
                if filename != expected_filename:
                    raise CheckpointError(f'Tensor {name} has a mismatched shard filename.')
                files.append(root / filename)
            if kind == 'torch':
                if torch is None:
                    raise CheckpointError(f'PyTorch is required to load tensor {name}.')
                try:
                    chunks = [torch.load(path, map_location='cpu', weights_only=True) for path in files]
                except TypeError:  # PyTorch versions predating weights_only
                    chunks = [torch.load(path, map_location='cpu') for path in files]
                if any(not isinstance(chunk, torch.Tensor) for chunk in chunks):
                    raise CheckpointError(f'Non-tensor object found in shards: {name}')
                flat = torch.cat(chunks, dim=0)
                value = flat.reshape(tuple(shape))
                if list(value.shape) != shape:
                    raise CheckpointError(f'Sharded tensor shape mismatch: {name}')
                if str(value.dtype) != metadata['dtype']:
                    raise CheckpointError(f'Sharded tensor dtype mismatch: {name}')
            elif kind == 'numpy':
                chunks = [np.load(path, allow_pickle=False) for path in files]
                flat = np.concatenate(chunks, axis=0)
                value = flat.reshape(tuple(shape))
                if list(value.shape) != shape:
                    raise CheckpointError(f'Sharded array shape mismatch: {name}')
                if str(value.dtype) != metadata['dtype']:
                    raise CheckpointError(f'Sharded array dtype mismatch: {name}')
            result[name] = value
        return result


class CheckpointManager:
    def __init__(
        self,
        run_root: Path,
        config: CheckpointConfig,
        source_hash: str,
        notebook_hash: str | None,
        *,
        require_source_match: bool = True,
    ) -> None:
        self.run_root = run_root
        self.config = config
        self.source_hash = source_hash
        self.notebook_hash = notebook_hash
        self.require_source_match = require_source_match
        self.checkpoint_root = run_root / 'checkpoints'
        self.checkpoint_root.mkdir(parents=True, exist_ok=True)
        self.tensor_store = TensorShardStore(config.tensor_shard_bytes)

    def _candidate_paths(self) -> list[Path]:
        candidates: list[tuple[int, Path]] = []
        for path in self.checkpoint_root.glob('ckpt_*'):
            if not path.is_dir():
                continue
            try:
                index = int(path.name.removeprefix('ckpt_'))
            except ValueError:
                continue
            candidates.append((index, path))
        return [path for _, path in sorted(candidates, reverse=True)]

    def _next_index(self, state: RunState) -> int:
        existing = [int(path.name.removeprefix('ckpt_')) for path in self._candidate_paths()]
        return max([state.checkpoint_index, *existing], default=0) + 1

    def save(self, state: RunState, queue: WorkQueue, tensors: dict[str, Any] | None = None) -> CheckpointSaveResult:
        started = time.monotonic()
        state.assert_safe()
        if state.run_id != self.run_root.name:
            raise CheckpointError('State run_id does not match the managed run directory.')
        queue.validate()
        previous_index = state.checkpoint_index
        next_index = self._next_index(state)
        state.checkpoint_index = next_index
        state.updated_at = utc_now()
        temporary = self.checkpoint_root / f'.tmp-{uuid.uuid4().hex}'
        final = self.checkpoint_root / f'ckpt_{next_index:06d}'
        temporary.mkdir(parents=False, exist_ok=False)
        try:
            atomic_write_json(temporary / 'state.json', asdict(state))
            atomic_write_json(temporary / 'bounds.json', state.bounds)
            atomic_write_json(temporary / 'work_queue.json', queue.to_payload())
            atomic_write_json(temporary / 'meta.json', {
                'schema_version': 1,
                'saved_at': utc_now(),
                'config': self.config.canonical_payload(),
                'config_hash': self.config.config_hash(),
                'source_hash': self.source_hash,
                'notebook_hash': self.notebook_hash,
                'environment': environment_info(),
            })
            _write_pickle(temporary / 'rng_state.pkl', capture_rng_state())
            self.tensor_store.save(temporary / 'tensors', tensors or {})
            hashes: dict[str, str] = {}
            for path in sorted(temporary.rglob('*')):
                if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}:
                    hashes[path.relative_to(temporary).as_posix()] = sha256_file(path)
            atomic_write_json(temporary / 'hashes.json', hashes)
            fsync_directory(temporary)
            os.replace(temporary, final)
            fsync_directory(self.checkpoint_root)
            atomic_write_text(final / 'COMMITTED', utc_now())
            atomic_write_json(self.checkpoint_root / 'LATEST.json', {'checkpoint': final.name, 'saved_at': utc_now()})
            save_elapsed = time.monotonic() - started
            verify_started = time.monotonic()
            self.verify(final)
            verify_elapsed = time.monotonic() - verify_started
            return CheckpointSaveResult(final, next_index, directory_size(final), save_elapsed, verify_elapsed)
        except Exception:
            state.checkpoint_index = previous_index
            if temporary.exists():
                shutil.rmtree(temporary, ignore_errors=True)
            raise

    def verify(self, path: Path) -> None:
        if path.is_symlink() or path.parent.resolve() != self.checkpoint_root.resolve():
            raise CheckpointError(f'Checkpoint path is outside the managed root: {path}')
        if not path.is_dir() or not (path / 'COMMITTED').is_file():
            raise CheckpointError(f'Uncommitted checkpoint: {path}')
        if any(item.is_symlink() for item in path.rglob('*')):
            raise CheckpointError(f'Symlink found inside checkpoint: {path}')
        hashes_path = path / 'hashes.json'
        if not hashes_path.is_file():
            raise CheckpointError(f'Missing hashes.json: {path}')
        expected_payload = read_json(hashes_path)
        if not isinstance(expected_payload, dict) or any(
            not isinstance(relative, str) or not isinstance(digest, str)
            or len(digest) != 64 or any(character not in '0123456789abcdef' for character in digest)
            for relative, digest in expected_payload.items()
        ):
            raise CheckpointError(f'Invalid hash manifest structure: {path}')
        expected: dict[str, str] = expected_payload
        actual_files = {
            item.relative_to(path).as_posix()
            for item in path.rglob('*')
            if item.is_file() and item.name not in {'hashes.json', 'COMMITTED'}
        }
        if actual_files != set(expected):
            raise CheckpointError(f'Checkpoint file-set mismatch: {path}')
        for relative, digest in expected.items():
            if sha256_file(path / relative) != digest:
                raise CheckpointError(f'Checkpoint hash mismatch: {path / relative}')
        meta = read_json(path / 'meta.json')
        if not isinstance(meta, dict):
            raise CheckpointError(f'Invalid checkpoint metadata structure: {path}')
        if meta.get('config_hash') != self.config.config_hash():
            raise ConfigMismatchError(f'Checkpoint config mismatch: {path}')
        if meta.get('config') != self.config.canonical_payload():
            raise ConfigMismatchError(f'Checkpoint canonical config mismatch: {path}')
        if self.require_source_match and meta.get('source_hash') != self.source_hash:
            raise ConfigMismatchError(f'Checkpoint source hash mismatch: {path}')
        if self.require_source_match and meta.get('notebook_hash') != self.notebook_hash:
            raise ConfigMismatchError(f'Checkpoint notebook source hash mismatch: {path}')

    def load_latest(self, restore_rng: bool = True) -> LoadedCheckpoint | None:
        skipped: list[str] = []
        for path in self._candidate_paths():
            try:
                self.verify(path)
                state = RunState(**read_json(path / 'state.json'))
                state.assert_safe()
                if state.run_id != self.run_root.name:
                    raise CheckpointError(f'Checkpoint run_id/path mismatch: {path}')
                if state.config_hash != self.config.config_hash():
                    raise ConfigMismatchError(f'State config mismatch: {path}')
                if read_json(path / 'bounds.json') != state.bounds:
                    raise CheckpointError(f'Bounds/state mismatch: {path}')
                queue = WorkQueue.from_payload(read_json(path / 'work_queue.json'))
                try:
                    tensors = self.tensor_store.load(path / 'tensors')
                except Exception as exc:
                    raise CheckpointError(f'Tensor shard validation failed: {path}') from exc
                if restore_rng:
                    try:
                        with (path / 'rng_state.pkl').open('rb') as handle:
                            restore_rng_state(pickle.load(handle))
                    except Exception as exc:
                        raise CheckpointError(f'RNG restoration failed: {path}') from exc
                return LoadedCheckpoint(state, queue, path, tensors, tuple(skipped))
            except ConfigMismatchError:
                raise
            except (AttributeError, CheckpointError, EOFError, OSError, TypeError, ValueError, KeyError, pickle.PickleError) as exc:
                skipped.append(f'{path.name}: {exc}')
        return None

    def load_tensors(self, checkpoint_path: Path) -> dict[str, Any]:
        self.verify(checkpoint_path)
        return self.tensor_store.load(checkpoint_path / 'tensors')
