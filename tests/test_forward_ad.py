from __future__ import annotations

import numpy as np
import pytest

from src.error_ledger import ErrorLedger, ErrorLedgerError
from src.forward_ad import (
    DualTensor, dual_matmul, dual_regroup, fixed_basis_project, zero_source_dual,
)
from src.normalization import normalize_dual
from src.source_channels import (
    SOURCE_CLASSES, SourceClass, generator_symmetry_residuals, source_generators,
)


def _dual(seed: int = 1) -> DualTensor:
    rng = np.random.default_rng(seed)
    primal = rng.standard_normal((4, 4))
    return DualTensor(
        primal,
        {
            source: rng.standard_normal((4, 4))
            for source in SOURCE_CLASSES
        },
    )


def test_source_generators_are_symmetry_reduced_and_zero_source_is_zero() -> None:
    generators = source_generators()
    assert set(generators) == set(SOURCE_CLASSES)
    assert next(iter(generators.values())).shape == (729,)
    assert max(generator_symmetry_residuals(generators).values()) == 0.0
    zero = zero_source_dual(np.eye(4))
    assert all(np.count_nonzero(zero.tangent[source]) == 0 for source in SOURCE_CLASSES)


def test_source_generators_support_j2_max_2_operator_dimension() -> None:
    generators = source_generators(operator_dimension=46656)
    assert set(generators) == set(SOURCE_CLASSES)
    assert next(iter(generators.values())).shape == (46656,)
    assert max(generator_symmetry_residuals(generators).values()) == 0.0
    rng = np.random.default_rng(0)
    left, _ = np.linalg.qr(rng.standard_normal((46656, 16)))
    right_basis, _ = np.linalg.qr(rng.standard_normal((46656, 16)))
    core = np.diag(np.linspace(1.0, 0.1, 16))
    right = right_basis.T
    from src.source_channels import projected_parent_dual
    dual = projected_parent_dual(left, core, right, left, generators)
    assert dual.primal.shape == (16, 16)


def test_forward_rules_match_centered_finite_difference() -> None:
    dual = _dual()
    basis, _ = np.linalg.qr(np.arange(24, dtype=np.float64).reshape(6, 4) + np.eye(6, 4))
    enlarged = DualTensor(
        basis @ dual.primal @ basis.T,
        {
            source: basis @ dual.tangent[source] @ basis.T
            for source in SOURCE_CLASSES
        },
    )
    projected = fixed_basis_project(enlarged, basis, basis)
    actual, _ = normalize_dual(dual_regroup(dual_matmul(projected, projected)))
    step = 1e-6
    for source in SOURCE_CLASSES:
        plus = DualTensor(
            projected.primal + step * projected.tangent[source],
            {item: np.zeros_like(projected.primal) for item in SOURCE_CLASSES},
        )
        minus = DualTensor(
            projected.primal - step * projected.tangent[source],
            {item: np.zeros_like(projected.primal) for item in SOURCE_CLASSES},
        )
        plus_value, _ = normalize_dual(dual_regroup(dual_matmul(plus, plus)))
        minus_value, _ = normalize_dual(dual_regroup(dual_matmul(minus, minus)))
        finite_difference = (plus_value.primal - minus_value.primal) / (2 * step)
        np.testing.assert_allclose(
            actual.tangent[source], finite_difference, rtol=2e-8, atol=2e-9,
        )


def test_normalization_fails_closed_for_nonpositive_or_nonfinite() -> None:
    with pytest.raises(FloatingPointError):
        normalize_dual(zero_source_dual(np.zeros((4, 4))))
    with pytest.raises(ValueError):
        zero_source_dual(np.full((4, 4), np.nan))


def test_error_ledger_roundtrip_and_double_counting_guard() -> None:
    ledger = ErrorLedger()
    exact = ledger.add_leaf(
        name='exact', category='basis_equivalence_error', applies_to='both',
        source_checkpoint='ckpt', formula='exact equality', estimate=0.0,
        deterministic_upper_bound=0.0, rigor='RIGOROUS', note='exact',
    )
    missing = ledger.add_leaf(
        name='missing', category='gpu_rounding_backward', applies_to='both',
        source_checkpoint='ckpt', formula='required formula', estimate=None,
        deterministic_upper_bound=None, rigor='MISSING', note='not bounded',
    )
    aggregate = ledger.add_sum(
        name='output', category='output_radius', applies_to='primal',
        parents=(exact, missing), source_checkpoint='ckpt',
        formula='sum unique leaves', note='partial only',
    )
    assert ledger.terms[aggregate].deterministic_upper_bound is None
    assert ledger.summary()['enclosure_ready'] is False
    restored = ErrorLedger.from_payload(ledger.payload())
    assert restored.payload() == ledger.payload()
    with pytest.raises(ErrorLedgerError, match='double count'):
        ledger.add_sum(
            name='bad', category='output_radius', applies_to='primal',
            parents=(exact, exact), source_checkpoint='ckpt',
            formula='bad sum', note='must fail',
        )


@pytest.mark.gpu
def test_forward_product_rule_cuda_matches_cpu() -> None:
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA is unavailable.')
    dual = _dual(4)
    expected = dual_matmul(dual, dual)
    primal = torch.as_tensor(dual.primal, device='cuda', dtype=torch.float64)
    actual = (primal @ primal).cpu().numpy()
    np.testing.assert_allclose(actual, expected.primal, rtol=1e-13, atol=1e-13)
    for source in SOURCE_CLASSES:
        tangent = torch.as_tensor(
            dual.tangent[source], device='cuda', dtype=torch.float64,
        )
        actual_tangent = (tangent @ primal + primal @ tangent).cpu().numpy()
        np.testing.assert_allclose(
            actual_tangent, expected.tangent[source], rtol=1e-13, atol=1e-13,
        )
    assert torch.backends.cuda.matmul.allow_tf32 is False
