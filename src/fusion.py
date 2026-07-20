from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Callable, Iterable

from sympy import Expr, Matrix, Rational, eye, kronecker_product, simplify, zeros
from sympy.physics.wigner import clebsch_gordan


@dataclass(frozen=True, slots=True)
class FusionConvention:
    name: str = 'Condon-Shortley/SymPy exact'
    irrep_coordinate: str = 'j2=2j nonnegative integer'
    magnetic_order: str = 'm2=j2,j2-2,...,-j2'
    tensor_product_order: str = 'leg 0 is the leftmost Kronecker factor'
    fusion_tree: str = 'left-associated (((j0 x j1) x j2) x ...)'
    duality: str = 'C|j,m>=(-1)^(j-m)|j,-m>'
    orientation: str = '+1 fundamental, -1 mapped through C before fusion'
    normalization: str = 'orthonormal CG basis and normalized Haar projector'


FUSION_CONVENTION = FusionConvention()


def convention_payload() -> dict[str, str]:
    return {
        field: getattr(FUSION_CONVENTION, field)
        for field in FUSION_CONVENTION.__dataclass_fields__
    }


def convention_hash() -> str:
    payload = json.dumps(
        convention_payload(), sort_keys=True, separators=(',', ':'), ensure_ascii=False,
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def magnetic_values(j2: int) -> tuple[int, ...]:
    if not isinstance(j2, int) or isinstance(j2, bool) or j2 < 0:
        raise ValueError('j2 must be a nonnegative integer.')
    return tuple(range(j2, -j2 - 1, -2))


def _validate_magnetic(j2: int, m2: int) -> None:
    if not isinstance(m2, int) or isinstance(m2, bool) or m2 not in magnetic_values(j2):
        raise ValueError(f'm2={m2!r} is invalid for j2={j2}.')


def coupling_outputs(left_j2: int, right_j2: int) -> tuple[int, ...]:
    magnetic_values(left_j2)
    magnetic_values(right_j2)
    return tuple(range(abs(left_j2 - right_j2), left_j2 + right_j2 + 1, 2))


@lru_cache(maxsize=65536)
def cg_coefficient(
    left_j2: int, left_m2: int, right_j2: int, right_m2: int,
    total_j2: int, total_m2: int,
) -> Expr:
    _validate_magnetic(left_j2, left_m2)
    _validate_magnetic(right_j2, right_m2)
    _validate_magnetic(total_j2, total_m2)
    if total_j2 not in coupling_outputs(left_j2, right_j2):
        return Rational(0)
    if left_m2 + right_m2 != total_m2:
        return Rational(0)
    return simplify(clebsch_gordan(
        Rational(left_j2, 2), Rational(right_j2, 2), Rational(total_j2, 2),
        Rational(left_m2, 2), Rational(right_m2, 2), Rational(total_m2, 2),
    ))


def magnetic_basis(representations: Iterable[int]) -> tuple[tuple[int, ...], ...]:
    reps = tuple(representations)
    if not reps:
        raise ValueError('At least one representation leg is required.')
    values = tuple(magnetic_values(j2) for j2 in reps)
    return tuple(product(*values))


def fusion_paths(representations: Iterable[int], final_j2: int = 0) -> tuple[tuple[int, ...], ...]:
    reps = tuple(representations)
    if not reps:
        raise ValueError('At least one representation leg is required.')
    magnetic_values(final_j2)
    paths: list[tuple[int, ...]] = [(reps[0],)]
    for representation in reps[1:]:
        next_paths: list[tuple[int, ...]] = []
        for path in paths:
            for output in coupling_outputs(path[-1], representation):
                next_paths.append(path + (output,))
        paths = next_paths
    return tuple(path for path in sorted(paths) if path[-1] == final_j2)


CGProvider = Callable[[int, int, int, int, int, int], Expr]


def fusion_basis_vector(
    representations: Iterable[int], path: tuple[int, ...],
    cg_provider: CGProvider = cg_coefficient,
) -> Matrix:
    reps = tuple(representations)
    if len(path) != len(reps) or not reps or path[0] != reps[0]:
        raise ValueError('Fusion path does not match the representation legs.')
    for index in range(1, len(reps)):
        if path[index] not in coupling_outputs(path[index - 1], reps[index]):
            raise ValueError('Fusion path contains a forbidden intermediate irrep.')
    if path[-1] != 0:
        raise ValueError('M2 armillary basis requires a final singlet.')
    amplitudes: list[Expr] = []
    for state in magnetic_basis(reps):
        amplitude: Expr = Rational(1)
        running_m2 = state[0]
        for index in range(1, len(reps)):
            next_m2 = running_m2 + state[index]
            if next_m2 not in magnetic_values(path[index]):
                amplitude = Rational(0)
                break
            amplitude *= cg_provider(
                path[index - 1], running_m2, reps[index], state[index],
                path[index], next_m2,
            )
            running_m2 = next_m2
        amplitudes.append(simplify(amplitude if running_m2 == 0 else Rational(0)))
    return Matrix(amplitudes)


def fusion_basis_matrix(
    representations: Iterable[int], cg_provider: CGProvider = cg_coefficient,
) -> tuple[tuple[tuple[int, ...], ...], Matrix]:
    reps = tuple(representations)
    paths = fusion_paths(reps)
    dimension = len(magnetic_basis(reps))
    if not paths:
        return paths, zeros(dimension, 0)
    columns = [fusion_basis_vector(reps, path, cg_provider) for path in paths]
    basis = Matrix.hstack(*columns)
    gram = basis.T * basis
    if gram.applyfunc(simplify) != eye(len(paths)):
        raise ArithmeticError('Fusion basis is not exactly orthonormal.')
    return paths, basis.applyfunc(simplify)


def duality_matrix(j2: int) -> Matrix:
    values = magnetic_values(j2)
    index = {m2: position for position, m2 in enumerate(values)}
    result = zeros(j2 + 1)
    for column, m2 in enumerate(values):
        exponent = (j2 - m2) // 2
        result[index[-m2], column] = -1 if exponent % 2 else 1
    if (result.T * result) != eye(j2 + 1):
        raise ArithmeticError('SU(2) duality map is not orthogonal.')
    return result


def orientation_map(representations: Iterable[int], orientations: Iterable[int]) -> Matrix:
    reps = tuple(representations)
    signs = tuple(orientations)
    if len(reps) != len(signs) or not reps:
        raise ValueError('Representation/orientation lengths must agree and be nonempty.')
    factors: list[Matrix] = []
    for j2, sign in zip(reps, signs, strict=True):
        if sign not in {-1, 1}:
            raise ValueError('Orientation signs must be +1 or -1.')
        factors.append(eye(j2 + 1) if sign == 1 else duality_matrix(j2))
    result = Matrix([[1]])
    for factor in factors:
        result = kronecker_product(result, factor)
    return result


def representation_dimension(representations: Iterable[int]) -> int:
    dimension = 1
    for j2 in representations:
        magnetic_values(j2)
        dimension *= j2 + 1
    return dimension
