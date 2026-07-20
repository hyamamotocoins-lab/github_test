"""Legacy dense Haar projector construction (SymPy nullspace).

M2_PROOF_SCHEMA_V2 no longer uses ``build_dense_reference`` in the live
orchestrator path. The functions remain for regression comparisons against
integer multiplicity and for optional diagnostics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sympy import Matrix, Rational, eye, kronecker_product, simplify, sqrt, srepr, zeros

from .common import canonical_json_bytes
from .fusion import magnetic_basis, magnetic_values, orientation_map, representation_dimension


@dataclass(frozen=True, slots=True)
class DenseReferenceSector:
    representations: tuple[int, ...]
    orientations: tuple[int, ...]
    magnetic_ordering: tuple[tuple[int, ...], ...]
    projector: Matrix
    singlet_rank: int
    generator_residual_zero: bool


def _single_generators(j2: int) -> tuple[Matrix, Matrix, Matrix]:
    values = magnetic_values(j2)
    index = {m2: position for position, m2 in enumerate(values)}
    dimension = j2 + 1
    jz = zeros(dimension)
    raising = zeros(dimension)
    for column, m2 in enumerate(values):
        jz[column, column] = Rational(m2, 2)
        next_m2 = m2 + 2
        if next_m2 in index:
            coefficient = sqrt(Rational((j2 - m2) * (j2 + m2 + 2), 4))
            raising[index[next_m2], column] = coefficient
    lowering = raising.T
    return jz, raising, lowering


def _embedded_operator(representations: tuple[int, ...], leg: int, local: Matrix) -> Matrix:
    result = Matrix([[1]])
    for index, j2 in enumerate(representations):
        factor = local if index == leg else eye(j2 + 1)
        result = kronecker_product(result, factor)
    return result


def total_generators(reps: Iterable[int]) -> tuple[Matrix, Matrix, Matrix]:
    labels = tuple(reps)
    if not labels:
        raise ValueError('Dense reference requires representation legs.')
    dimension = representation_dimension(labels)
    totals = [zeros(dimension), zeros(dimension), zeros(dimension)]
    for leg, j2 in enumerate(labels):
        for component, local in enumerate(_single_generators(j2)):
            totals[component] += _embedded_operator(labels, leg, local)
    # Global applyfunc(simplify) on large Kronecker sums is O(dim^2) and
    # commonly silent-kills Paperspace workers on j2_max=2 tail sectors.
    if dimension <= 64:
        return tuple(matrix.applyfunc(simplify) for matrix in totals)  # type: ignore[return-value]
    return tuple(totals)  # type: ignore[return-value]


def _outgoing_singlet_projector(reps: tuple[int, ...]) -> tuple[Matrix, int, bool]:
    jz, raising, lowering = total_generators(reps)
    constraints = Matrix.vstack(jz, raising, lowering)
    nullspace = constraints.nullspace()
    dimension = representation_dimension(reps)
    if not nullspace:
        return zeros(dimension), 0, True
    basis = Matrix.hstack(*nullspace)
    if dimension <= 64:
        gram = (basis.T * basis).applyfunc(simplify)
        projector = (basis * gram.inv() * basis.T).applyfunc(simplify)
        residual_zero = all(
            (generator * projector).applyfunc(simplify) == zeros(dimension)
            for generator in (jz, raising, lowering)
        )
        if projector.T != projector or (projector * projector).applyfunc(simplify) != projector:
            raise ArithmeticError('Dense Haar singlet projector failed exact projector identities.')
        return projector, len(nullspace), residual_zero
    gram = basis.T * basis
    projector = basis * gram.inv() * basis.T
    residual_zero = all(
        generator * projector == zeros(dimension)
        for generator in (jz, raising, lowering)
    )
    if projector.T != projector or projector * projector != projector:
        projector = projector.applyfunc(simplify)
        residual_zero = all(
            (generator * projector).applyfunc(simplify) == zeros(dimension)
            for generator in (jz, raising, lowering)
        )
        if projector.T != projector or (projector * projector).applyfunc(simplify) != projector:
            raise ArithmeticError(
                'Dense Haar singlet projector failed exact projector identities.',
            )
    return projector, len(nullspace), residual_zero


def build_dense_reference(
    representations: Iterable[int], orientations: Iterable[int],
) -> DenseReferenceSector:
    reps = tuple(representations)
    signs = tuple(orientations)
    if len(reps) != 6 or len(signs) != 6:
        raise ValueError('M2 link star has exactly six plaquette legs in four dimensions.')
    outgoing, rank, residual = _outgoing_singlet_projector(reps)
    dual_map = orientation_map(reps, signs)
    physical = (dual_map.T * outgoing * dual_map).applyfunc(simplify)
    return DenseReferenceSector(
        reps, signs, magnetic_basis(reps), physical, rank, residual,
    )


def matrix_exact_payload(matrix: Matrix) -> dict[str, object]:
    entries = [srepr(simplify(value)) for value in matrix]
    return {'rows': matrix.rows, 'cols': matrix.cols, 'entries': entries}


def matrix_hash(matrix: Matrix) -> str:
    return hashlib.sha256(canonical_json_bytes(matrix_exact_payload(matrix))).hexdigest()


def matrix_to_float64(matrix: Matrix) -> np.ndarray:
    array = np.array([float(value.evalf(50)) for value in matrix], dtype=np.float64)
    return array.reshape((matrix.rows, matrix.cols))


def exact_matrix_difference_zero(left: Matrix, right: Matrix) -> bool:
    if left.shape != right.shape:
        return False
    return (left - right).applyfunc(simplify) == zeros(left.rows, left.cols)
