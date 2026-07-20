"""Apply total SU(2) generators to a basis without building D×D Kronecker matrices."""

from __future__ import annotations

from functools import reduce
from operator import mul

from sympy import Matrix, simplify, zeros

from .dense_reference import _single_generators
from .fusion import representation_dimension


def _leg_dims(reps: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(j2 + 1 for j2 in reps)


def _apply_local_to_vector(
    vector: list,
    dims: tuple[int, ...],
    leg: int,
    local: Matrix,
) -> list:
    """Apply a local operator on one Kronecker factor (leg 0 = leftmost)."""
    before = reduce(mul, dims[:leg], 1)
    d_leg = dims[leg]
    after = reduce(mul, dims[leg + 1 :], 1)
    out = [0] * len(vector)
    for b in range(before):
        for src_i in range(d_leg):
            for a in range(after):
                src = (b * d_leg + src_i) * after + a
                amp = vector[src]
                if amp == 0:
                    continue
                for dst_i in range(d_leg):
                    coeff = local[dst_i, src_i]
                    if coeff == 0:
                        continue
                    dst = (b * d_leg + dst_i) * after + a
                    out[dst] += coeff * amp
    return out


def apply_total_j_to_basis(
    basis: Matrix,
    reps: tuple[int, ...],
) -> tuple[Matrix, Matrix, Matrix]:
    """Return (Jz B, J+ B, J- B) acting in the outgoing magnetic product basis."""
    labels = tuple(reps)
    dimension = representation_dimension(labels)
    if basis.rows != dimension:
        raise ValueError(
            f'Basis rows {basis.rows} disagree with representation dimension {dimension}.',
        )
    rank = basis.cols
    if rank == 0:
        empty = zeros(dimension, 0)
        return empty, empty, empty
    dims = _leg_dims(labels)
    jz_cols: list[Matrix] = []
    jp_cols: list[Matrix] = []
    jm_cols: list[Matrix] = []
    local_gens = [_single_generators(j2) for j2 in labels]
    for col in range(rank):
        vector = [basis[row, col] for row in range(dimension)]
        jz_acc = [0] * dimension
        jp_acc = [0] * dimension
        jm_acc = [0] * dimension
        for leg, (jz_loc, jp_loc, jm_loc) in enumerate(local_gens):
            jz_part = _apply_local_to_vector(vector, dims, leg, jz_loc)
            jp_part = _apply_local_to_vector(vector, dims, leg, jp_loc)
            jm_part = _apply_local_to_vector(vector, dims, leg, jm_loc)
            for index in range(dimension):
                jz_acc[index] += jz_part[index]
                jp_acc[index] += jp_part[index]
                jm_acc[index] += jm_part[index]
        jz_cols.append(Matrix(jz_acc))
        jp_cols.append(Matrix(jp_acc))
        jm_cols.append(Matrix(jm_acc))
    return (
        Matrix.hstack(*jz_cols),
        Matrix.hstack(*jp_cols),
        Matrix.hstack(*jm_cols),
    )


def entry_exact_zero(value) -> bool:
    """Exact algebraic zero test; only simplifies nonzero-looking entries."""
    if value == 0:
        return True
    return simplify(value) == 0


def matrix_exact_zero(matrix: Matrix) -> bool:
    for value in matrix:
        if not entry_exact_zero(value):
            return False
    return True


def exact_generator_annihilation(basis: Matrix, reps: tuple[int, ...]) -> bool:
    """True iff Jz B = J+ B = J- B = 0 exactly."""
    jz_b, jp_b, jm_b = apply_total_j_to_basis(basis, reps)
    return all(matrix_exact_zero(action) for action in (jz_b, jp_b, jm_b))
