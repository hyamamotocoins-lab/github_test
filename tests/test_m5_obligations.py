from __future__ import annotations

from fractions import Fraction

import numpy as np

from src.armillary import all_link_star_keys
from src.exact_arithmetic import fraction_from_payload
from src.m5_obligations import (
    _frobenius_fraction,
    _matrix_to_fractions,
    _sqrt_outward,
    evaluate_cutoff_rank_dependence,
    evaluate_gpu_rounding,
    evaluate_initial_representation_tail,
    evaluate_input_radius_propagation,
    evaluate_normalization_denominator,
    evaluate_omitted_fusion_channel_tail,
)


def test_fraction_frobenius_is_exact_for_binary_floats() -> None:
    matrix = np.array([[0.5, -0.25], [0.0, 0.125]], dtype=np.float64)
    square = _frobenius_fraction(_matrix_to_fractions(matrix))
    assert square == Fraction(1, 4) + Fraction(1, 16) + Fraction(1, 64)
    root = _sqrt_outward(square)
    assert root * root >= square


def test_gpu_rounding_closes_on_self_consistent_pipeline() -> None:
    from src.forward_ad import regroup_matrix
    from src.normalization import normalize_array

    projected = np.eye(16, dtype=np.float64)
    normalized = normalize_array(regroup_matrix(projected @ projected))
    result = evaluate_gpu_rounding(
        {'projected_primal': projected, 'normalized_primal': normalized},
        source_paths=('synthetic',),
        source_hashes=('a' * 64,),
    )
    assert result.status == 'RIGOROUS'
    assert result.upper_bound is not None


def test_normalization_requires_positive_center_scale() -> None:
    center = np.ones((16, 16), dtype=np.float64)
    result = evaluate_normalization_denominator(
        {'coarse_primal': center},
        source_paths=('synthetic',),
        source_hashes=('b' * 64,),
    )
    assert result.status == 'RIGOROUS'
    zero = evaluate_normalization_denominator(
        {'coarse_primal': np.zeros((16, 16), dtype=np.float64)},
        source_paths=('synthetic',),
        source_hashes=('c' * 64,),
    )
    assert zero.status == 'BLOCKED_MATH'


def test_input_radius_singleton_is_zero() -> None:
    result = evaluate_input_radius_propagation()
    assert result.status == 'RIGOROUS'
    assert result.upper_bound is not None
    assert fraction_from_payload(result.upper_bound['hi']) == 0


def test_cutoff_rank_fixed_scheme_is_rigorous() -> None:
    result = evaluate_cutoff_rank_dependence(None)
    assert result.status == 'RIGOROUS'
    assert result.upper_bound is not None
    assert fraction_from_payload(result.upper_bound['hi']) == 0


def test_omitted_channel_closes_on_full_projector_cover() -> None:
    tensors = {
        f'projector_{"".join(str(v) for v in key.representations)}': np.eye(2)
        for key in all_link_star_keys()
    }
    result = evaluate_omitted_fusion_channel_tail(
        tensors, source_paths=('synthetic',), source_hashes=('d' * 64,),
        j2_max=1,
    )
    assert result.status == 'RIGOROUS'
    incomplete = evaluate_omitted_fusion_channel_tail(
        {}, source_paths=('synthetic',), source_hashes=('e' * 64,),
        j2_max=1,
    )
    assert incomplete.status == 'BLOCKED_MATH'


def test_omitted_channel_respects_j2_max() -> None:
    tensors = {
        f'projector_{"".join(str(v) for v in key.representations)}': np.eye(1)
        for key in all_link_star_keys(1)
    }
    result = evaluate_omitted_fusion_channel_tail(
        tensors, source_paths=('synthetic',), source_hashes=('f' * 64,),
        j2_max=2,
    )
    assert result.status == 'BLOCKED_MATH'


def test_initial_tail_blocks_without_m1() -> None:
    result = evaluate_initial_representation_tail(None)
    assert result.status == 'BLOCKED_MATH'


def test_initial_tail_blocks_when_cutoff_too_low(tmp_path) -> None:
    from src.m5_parent_chain import AcceptedParentRef

    report = tmp_path / 'M1_report.json'
    # Only cutoff 1 available; j2_max=2 needs N>=3.
    atomic_write = __import__('src.common', fromlist=['atomic_write_json']).atomic_write_json
    atomic_write(report, {
        'results': {
            'M1_VALUE_TAIL': {
                'result': {
                    'rigor': 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
                    'entries': {
                        '1': {'tail': {'hi': {'numer': '1', 'denom': '10'}}},
                    },
                },
            },
            'M1_GRADIENT_TAIL': {
                'result': {
                    'entries': {
                        '1': {'tail': {'hi': {'numer': '1', 'denom': '10'}}},
                    },
                },
            },
        },
    })
    m1 = AcceptedParentRef(
        milestone='M1',
        run_id='M1-synthetic',
        audit_path=tmp_path / 'audit.json',
        audit={},
        run_root=tmp_path,
        checkpoint=tmp_path,
        report_path=report,
    )
    result = evaluate_initial_representation_tail(m1, j2_max=2)
    assert result.status == 'BLOCKED_MATH'
    assert 'No M1 value-tail cutoff' in (result.notes or '')


def test_read_j2_max_from_parent_m3_config(tmp_path) -> None:
    from src.common import atomic_write_json
    from src.m5_obligations import _read_j2_max_near_m4

    m4_run = tmp_path / 'runs' / 'M4-child'
    m3_run = tmp_path / 'runs' / 'M3-parent'
    m3_ckpt = m3_run / 'checkpoints' / 'ckpt_000010'
    m3_ckpt.mkdir(parents=True)
    (m3_ckpt / 'COMMITTED').write_text('ok', encoding='utf-8')
    m4_ckpt = m4_run / 'checkpoints' / 'ckpt_000027'
    m4_ckpt.mkdir(parents=True)
    atomic_write_json(m4_run / 'run_config.json', {
        'parent_run_id': 'M3-parent',
        'parent_checkpoint_path': str(m3_ckpt),
    })
    atomic_write_json(m3_run / 'run_config.json', {'j2_max': 2})
    assert _read_j2_max_near_m4(m4_ckpt) == 2

