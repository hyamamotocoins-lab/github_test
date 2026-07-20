from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .forward_ad import DualTensor
from .source_channels import SOURCE_CLASSES, SourceClass


@dataclass(frozen=True, slots=True)
class NormalizationInfo:
    scale: float
    scale_derivatives: Mapping[SourceClass, float]
    formula: str = (
        'd(T/lambda)=dT/lambda-T*d(lambda)/lambda^2; '
        'lambda=Frobenius norm'
    )

    def payload(self) -> dict[str, object]:
        return {
            'scale': self.scale,
            'scale_derivatives': {
                source.value: self.scale_derivatives[source]
                for source in SOURCE_CLASSES
            },
            'formula': self.formula,
        }


def normalize_array(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64)
    scale = float(np.linalg.norm(value, 'fro'))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError('M4 normalization is nonpositive or nonfinite.')
    result = value / scale
    if not np.isfinite(result).all():
        raise FloatingPointError('M4 normalized tensor is nonfinite.')
    return result


def normalize_dual(value: DualTensor) -> tuple[DualTensor, NormalizationInfo]:
    scale = float(np.linalg.norm(value.primal, 'fro'))
    if not np.isfinite(scale) or scale <= 0.0:
        raise FloatingPointError('M4 normalization is nonpositive or nonfinite.')
    derivatives: dict[SourceClass, float] = {}
    tangent: dict[SourceClass, np.ndarray] = {}
    for source in SOURCE_CLASSES:
        derivative = float(
            np.vdot(value.primal, value.tangent[source]).real / scale
        )
        if not np.isfinite(derivative):
            raise FloatingPointError('M4 normalization derivative is nonfinite.')
        derivatives[source] = derivative
        tangent[source] = (
            value.tangent[source] / scale
            - value.primal * derivative / (scale * scale)
        )
    result = DualTensor(value.primal / scale, tangent)
    return result, NormalizationInfo(scale, derivatives)
