from __future__ import annotations

import importlib.util
import os
from dataclasses import asdict, dataclass
from typing import Any, Protocol, Sequence

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch


class BackendUnavailableError(RuntimeError):
    '''Raised when a required contraction backend is unavailable.'''


class ContractionBackend(Protocol):
    name: str
    device: torch.device
    dtype: torch.dtype
    is_cuda: bool

    def tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor: ...
    def to_numpy(self, value: torch.Tensor) -> np.ndarray: ...
    def matmul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor: ...
    def contract(
        self, expression: str, operands: Sequence[torch.Tensor],
        path: object | None = None,
    ) -> torch.Tensor: ...
    def synchronize(self) -> None: ...
    def memory_snapshot(self) -> dict[str, int | float | str | bool]: ...


@dataclass(frozen=True, slots=True)
class BackendSelection:
    name: str
    device: str
    dtype: str
    is_cuda: bool
    cuquantum_available: bool
    opt_einsum_available: bool
    torch_version: str
    cuda_runtime: str | None
    gpu_name: str | None

    def payload(self) -> dict[str, Any]:
        return asdict(self)


class TorchBackend:
    def __init__(self, *, use_cuda: bool, use_opt_einsum: bool = True) -> None:
        if use_cuda and not torch.cuda.is_available():
            raise BackendUnavailableError('CUDA was required but torch.cuda is unavailable.')
        self.device = torch.device('cuda:0' if use_cuda else 'cpu')
        self.dtype = torch.float64
        self.is_cuda = use_cuda
        self.use_opt_einsum = (
            use_opt_einsum and importlib.util.find_spec('opt_einsum') is not None
        )
        self.name = (
            'torch_cuda_opt_einsum'
            if self.is_cuda and self.use_opt_einsum
            else 'torch_cuda'
            if self.is_cuda
            else 'torch_cpu_reference'
        )
        if self.is_cuda:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    def tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value.to(device=self.device, dtype=self.dtype)
        array = np.asarray(value, dtype=np.float64, order='C')
        return torch.as_tensor(array, device=self.device, dtype=self.dtype)

    def to_numpy(self, value: torch.Tensor) -> np.ndarray:
        return value.detach().to(device='cpu', dtype=torch.float64).numpy().copy()

    def matmul(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return torch.matmul(left, right)

    def contract(
        self, expression: str, operands: Sequence[torch.Tensor],
        path: object | None = None,
    ) -> torch.Tensor:
        del path
        if self.use_opt_einsum:
            import opt_einsum
            return opt_einsum.contract(
                expression, *operands, backend='torch',
                optimize='auto-hq',
            )
        return torch.einsum(expression, *operands)

    def synchronize(self) -> None:
        if self.is_cuda:
            torch.cuda.synchronize(self.device)

    def memory_snapshot(self) -> dict[str, int | float | str | bool]:
        if not self.is_cuda:
            return {
                'device': 'cpu', 'cuda': False, 'allocated_bytes': 0,
                'reserved_bytes': 0, 'peak_allocated_bytes': 0,
                'free_bytes': 0, 'total_bytes': 0, 'free_fraction': 1.0,
            }
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        return {
            'device': str(self.device), 'cuda': True,
            'allocated_bytes': torch.cuda.memory_allocated(self.device),
            'reserved_bytes': torch.cuda.memory_reserved(self.device),
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(self.device),
            'free_bytes': free_bytes, 'total_bytes': total_bytes,
            'free_fraction': free_bytes / total_bytes,
        }


class CuQuantumBackend(TorchBackend):
    def __init__(self) -> None:
        if importlib.util.find_spec('cuquantum') is None:
            raise BackendUnavailableError('cuQuantum is not installed.')
        super().__init__(use_cuda=True, use_opt_einsum=False)
        import cuquantum

        if not hasattr(cuquantum, 'contract'):
            raise BackendUnavailableError('Installed cuQuantum lacks contract().')
        self._cuquantum = cuquantum
        self.name = 'cuquantum_cutensornet'

    def contract(
        self, expression: str, operands: Sequence[torch.Tensor],
        path: object | None = None,
    ) -> torch.Tensor:
        del path
        return self._cuquantum.contract(expression, *operands)


def select_backend(*, require_cuda: bool, prefer_cuquantum: bool = True) -> ContractionBackend:
    if require_cuda and not torch.cuda.is_available():
        raise BackendUnavailableError('M3 requires a CUDA GPU for the real pilot run.')
    if prefer_cuquantum and torch.cuda.is_available():
        try:
            return CuQuantumBackend()
        except (BackendUnavailableError, ImportError, RuntimeError):
            pass
    return TorchBackend(
        use_cuda=torch.cuda.is_available() if require_cuda else False,
        use_opt_einsum=True,
    )


def backend_selection(backend: ContractionBackend) -> BackendSelection:
    gpu_name = None
    if backend.is_cuda:
        gpu_name = torch.cuda.get_device_properties(backend.device).name
    return BackendSelection(
        name=backend.name, device=str(backend.device),
        dtype=str(backend.dtype).removeprefix('torch.'),
        is_cuda=backend.is_cuda,
        cuquantum_available=importlib.util.find_spec('cuquantum') is not None,
        opt_einsum_available=importlib.util.find_spec('opt_einsum') is not None,
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        gpu_name=gpu_name,
    )
