from __future__ import annotations

from pathlib import Path

from sympy import Rational, eye, simplify

from src.common import sha256_file
from src.fusion import (
    convention_hash, coupling_outputs, duality_matrix, fusion_basis_matrix,
    magnetic_values,
)
from src.m2_config import M2Config
from src.wigner_cache import generate_low_cutoff_cache, validate_cache


def test_m2_config_is_fixed_low_cutoff_and_fail_closed() -> None:
    config = M2Config()
    assert config.j2_max == 1
    assert config.leg_count == 6
    assert config.orientations == (1, -1, 1, -1, 1, -1)
    assert config.certification_status == 'NOT_CERTIFIED'
    assert config.final_save_after_s <= 5 * 3600 + 20 * 60
    assert config.hard_return_s <= 5.5 * 3600


def test_half_spin_couplings_and_singlet_basis_are_exact() -> None:
    assert magnetic_values(1) == (1, -1)
    assert coupling_outputs(1, 1) == (0, 2)
    paths, basis = fusion_basis_matrix((1, 1, 1, 1, 1, 1))
    assert len(paths) == 5
    assert basis.shape == (64, 5)
    assert (basis.T * basis).applyfunc(simplify) == eye(5)


def test_duality_map_fixes_phase_and_square() -> None:
    dual = duality_matrix(1)
    assert dual.tolist() == [[0, -1], [1, 0]]
    assert dual.T * dual == eye(2)
    assert dual * dual == -eye(2)


def test_wigner_cache_regeneration_is_byte_deterministic(tmp_path: Path) -> None:
    first = tmp_path / 'first.json'
    second = tmp_path / 'second.json'
    first_digest = generate_low_cutoff_cache(first)
    second_digest = generate_low_cutoff_cache(second)
    assert first_digest == second_digest == sha256_file(first) == sha256_file(second)
    payload = validate_cache(first)
    assert payload['entry_count'] > 0
    assert payload['convention_hash'] == convention_hash()


def test_wigner_cache_uses_exact_expressions(tmp_path: Path) -> None:
    path = tmp_path / 'wigner.json'
    generate_low_cutoff_cache(path)
    payload = validate_cache(path)
    singlet = payload['entries']['3j:1,1,0,1,-1,0']['str']
    swapped = payload['entries']['3j:1,1,0,-1,1,0']['str']
    assert singlet == str(Rational(1, 1) / 2**Rational(1, 2))
    assert swapped == '-' + singlet
