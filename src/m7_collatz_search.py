"""Collatz / Perron helpers for M7 S0 majorant sharpening."""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Sequence

from .certificate import collatz_certificate, nonnegative_interval_matrix, positive_rational_vector
from .exact_arithmetic import fraction_from_payload
from .interval_kernel import ProofInterval, construct


def coerce_interval(cell: Any) -> ProofInterval:
    if isinstance(cell, ProofInterval):
        return cell
    if isinstance(cell, dict) and 'lo' in cell and 'hi' in cell:
        return construct(
            fraction_from_payload(cell['lo']),
            fraction_from_payload(cell['hi']),
        )
    return construct(cell)


def matrix_midpoints(entries: Sequence[Sequence[Any]]) -> list[list[float]]:
    mid: list[list[float]] = []
    for row in entries:
        mid_row: list[float] = []
        for cell in row:
            iv = coerce_interval(cell)
            mid_row.append(float((iv.lo + iv.hi) / 2))
        mid.append(mid_row)
    return mid


def power_iteration_perron(
    mid: Sequence[Sequence[float]],
    *,
    iterations: int = 64,
) -> list[Fraction]:
    n = len(mid)
    if n == 0:
        return []
    vector = [1.0 for _ in range(n)]
    for _ in range(iterations):
        nxt = [0.0 for _ in range(n)]
        for i in range(n):
            for j in range(n):
                nxt[i] += mid[i][j] * vector[j]
        scale = max(nxt) if nxt else 1.0
        if scale <= 0:
            break
        vector = [value / scale for value in nxt]
    # Outward rationalize via Fraction.from_float then bump numerator.
    result: list[Fraction] = []
    for value in vector:
        frac = Fraction.from_float(max(value, 1e-15)).limit_denominator(10**6)
        if frac <= 0:
            frac = Fraction(1, 10**6)
        result.append(frac)
    return result


def inverse_row_sum_weights(mid: Sequence[Sequence[float]]) -> list[Fraction]:
    weights: list[Fraction] = []
    for row in mid:
        total = sum(row)
        if total <= 0:
            weights.append(Fraction(1))
        else:
            weights.append(Fraction(1) / Fraction.from_float(total).limit_denominator(10**6))
    return weights


def evaluate_collatz(
    entries: Sequence[Sequence[Any]],
    labels: Sequence[str],
    perron: Sequence[Any],
    *,
    outside_tail: Any = 0,
) -> dict[str, Any]:
    coerced = [[coerce_interval(cell) for cell in row] for row in entries]
    matrix = nonnegative_interval_matrix(coerced, labels)
    if isinstance(outside_tail, ProofInterval):
        tail = outside_tail
    elif isinstance(outside_tail, dict) and 'lo' in outside_tail:
        tail = coerce_interval(outside_tail)
    else:
        tail = construct(outside_tail)
    vector = positive_rational_vector(perron, labels)
    bound = collatz_certificate(matrix, vector, outside_matrix_tail=tail)
    return {
        'verdict': bound.verdict,
        'q_cert_lo': bound.q_cert.lo,
        'q_cert_hi': bound.q_cert.hi,
        'q_collatz_hi': bound.q_collatz.hi,
        'payload': bound.payload(),
        'certified': bound.q_cert.hi < 1,
    }


def diagonal_plus_l1_tail(
    entries: Sequence[Sequence[Any]],
) -> tuple[list[list[Any]], Fraction]:
    """Split full coupling into diagonal majorant + off-diagonal L1 tail."""
    n = len(entries)
    diag: list[list[Any]] = [[construct(0) for _ in range(n)] for _ in range(n)]
    tail = Fraction(0)
    for i, row in enumerate(entries):
        row_off = Fraction(0)
        for j, cell in enumerate(row):
            iv = coerce_interval(cell)
            if i == j:
                diag[i][j] = iv
            else:
                row_off += iv.hi
        tail = max(tail, row_off)
    return diag, tail
