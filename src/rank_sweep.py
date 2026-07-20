"""Nested high-rank RSVD truncation for Campaign C exploratory rank sweeps.

Computes one high-rank factorization and evaluates truncated ranks without
re-running full M3. All metrics are exploratory, not certificates.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .common import (
    atomic_write_json, hash_tree, sha256_bytes, canonical_json_bytes, utc_now,
)
from .contraction_backend import select_backend
from .cutoff_dims import cutoff_dimension_payload
from .linear_operator import ArmillaryLinearOperator, build_armillary_operator
from .m3_config import M3Config
from .m3_parent import verify_accepted_m2_parent
from .m7_lineage import effective_projected_rank
from .partial_error_budget import provisional_budget, select_rank_from_budgets
from .rsvd import array_sha256, randomized_svd
from .runtime import environment_info, runtime_compatibility_signature
from .spectral_cluster import (
    detect_approximate_clusters, rank_is_mid_cluster,
)


class RankSweepError(RuntimeError):
    """Raised when an exploratory rank sweep cannot proceed."""


DEFAULT_RANK_GRID: tuple[int, ...] = (
    16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512,
)


@dataclass(frozen=True, slots=True)
class NestedFactorization:
    left: np.ndarray
    singular_values: np.ndarray
    right_t: np.ndarray
    seed: int
    max_rank: int
    oversampling: int
    power_iterations: int
    elapsed_s: float
    operator_norm: float

    def truncate(self, rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not 1 <= rank <= self.max_rank:
            raise RankSweepError(f'rank={rank} outside nested factorization.')
        return (
            self.left[:, :rank].copy(),
            self.singular_values[:rank].copy(),
            self.right_t[:rank, :].copy(),
        )


def build_nested_factorization(
    operator: ArmillaryLinearOperator,
    *,
    max_rank: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> NestedFactorization:
    result = randomized_svd(
        operator,
        target_rank=max_rank,
        oversampling=oversampling,
        power_iterations=power_iterations,
        seed=seed,
    )
    return NestedFactorization(
        left=result.left,
        singular_values=result.singular_values,
        right_t=result.right_t,
        seed=seed,
        max_rank=max_rank,
        oversampling=oversampling,
        power_iterations=power_iterations,
        elapsed_s=result.elapsed_s,
        operator_norm=float(operator.frobenius_norm()),
    )


def evaluate_truncated_rank(
    operator: ArmillaryLinearOperator,
    nested: NestedFactorization,
    rank: int,
) -> dict[str, Any]:
    left, singular, right_t = nested.truncate(rank)
    residual = float(operator.factor_residual_frobenius(left, singular, right_t))
    norm = nested.operator_norm
    if not np.isfinite(residual) or not norm > 0.0:
        raise RankSweepError(f'Invalid residual at rank={rank}.')
    relative = residual / norm
    ortho = float(np.linalg.norm(left.T @ left - np.eye(rank), ord='fro'))
    gap = None
    if rank < nested.max_rank:
        gap = float(nested.singular_values[rank - 1] - nested.singular_values[rank])
    return {
        'rank': rank,
        'effective_projected_rank': effective_projected_rank(rank),
        'singular_values_head': singular.tolist(),
        'sigma_r': float(singular[-1]),
        'sigma_r_plus_1': (
            float(nested.singular_values[rank]) if rank < nested.max_rank else None
        ),
        'approximate_gap': gap,
        'residual_frobenius': residual,
        'relative_residual_frobenius': relative,
        'orthogonality_residual': ortho,
        'left_sha256': array_sha256(left),
        'right_sha256': array_sha256(right_t),
        'singular_sha256': array_sha256(singular),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }


def default_sweep_config(**overrides: Any) -> dict[str, Any]:
    base = {
        'schema_version': 1,
        'rank_grid': list(DEFAULT_RANK_GRID),
        'max_factor_rank': 128,
        'oversampling': 16,
        'power_iterations': 2,
        'seed': 20260720,
        'engineering_margin': 0.05,
        'relative_gap_threshold': 0.05,
        'absolute_gap_floor': 1e-12,
        'representation_tail_proxy': 0.0,
        'channel_tail_proxy': 0.0,
        'require_cuda': True,
        'sectors_per_shard': 8,
    }
    base.update(overrides)
    return base


def create_sweep_manifest(
    *,
    project_root: Path,
    package_root: Path,
    m2_run_id: str,
    m3_config: M3Config,
    sweep_config: dict[str, Any],
    candidate_id: str,
) -> dict[str, Any]:
    environment = environment_info()
    payload = {
        'schema_version': 1,
        'status': 'EXPLORATORY_NOT_CERTIFIED',
        'candidate_id': candidate_id,
        'parent_m2_run_id': m2_run_id,
        'j2_max': m3_config.j2_max,
        'operator_dimension': m3_config.operator_dimension,
        'sector_count': m3_config.sector_count,
        'cutoff_dims': cutoff_dimension_payload(m3_config.j2_max),
        'm3_config_hash': m3_config.config_hash(),
        'sweep_config': sweep_config,
        'source_hash': hash_tree(project_root / 'src'),
        'package_root': str(package_root),
        'environment': environment,
        'runtime_compatibility': runtime_compatibility_signature(environment),
        'created_at': utc_now(),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }
    payload['config_hash'] = sha256_bytes(canonical_json_bytes({
        key: payload[key] for key in (
            'candidate_id', 'parent_m2_run_id', 'j2_max', 'operator_dimension',
            'sweep_config', 'source_hash', 'm3_config_hash',
        )
    }))
    return payload


def run_rank_sweep(
    *,
    project_root: Path,
    persistent_root: Path,
    package_root: Path,
    m3_config: M3Config,
    candidate_id: str,
    sweep_config: dict[str, Any] | None = None,
    sweep_root: Path | None = None,
) -> dict[str, Any]:
    """Execute S0 exploratory nested RSVD rank/gap/budget sweep."""
    config = default_sweep_config(**(sweep_config or {}))
    evidence = verify_accepted_m2_parent(project_root, m3_config)
    manifest = create_sweep_manifest(
        project_root=project_root,
        package_root=package_root,
        m2_run_id=m3_config.parent_run_id,
        m3_config=m3_config,
        sweep_config=config,
        candidate_id=candidate_id,
    )
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    if sweep_root is None:
        sweep_root = (
            package_root / 'rank_sweep' / f"SWEEP-{stamp}-{manifest['config_hash'][:12]}"
        )
    sweep_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(sweep_root / 'sweep_manifest.json', manifest)

    backend = select_backend(require_cuda=bool(config['require_cuda']))
    operator = build_armillary_operator(
        evidence.projector_tensors,
        backend,
        sweep_root / 'path_cache.json',
        sectors_per_shard=int(config['sectors_per_shard']),
        j2_max=m3_config.j2_max,
    )
    max_rank = int(config['max_factor_rank'])
    grid = sorted({int(value) for value in config['rank_grid'] if 1 <= int(value)})
    if not grid:
        raise RankSweepError('rank_grid empty.')
    # Need one extra singular value beyond the largest evaluated rank for gaps.
    factor_rank = max(max_rank, max(grid) + 1)
    if factor_rank >= min(operator.shape):
        factor_rank = min(operator.shape) - 1
    if factor_rank <= max(grid):
        raise RankSweepError(
            'Operator too small to expose sigma_{r+1} for the requested grid.'
        )
    grid = [rank for rank in grid if rank < factor_rank]
    if not grid:
        raise RankSweepError('No ranks remain after factor_rank filtering.')

    started = time.monotonic()
    nested = build_nested_factorization(
        operator,
        max_rank=factor_rank,
        oversampling=int(config['oversampling']),
        power_iterations=int(config['power_iterations']),
        seed=int(config['seed']),
    )
    clusters = detect_approximate_clusters(
        nested.singular_values,
        relative_gap_threshold=float(config['relative_gap_threshold']),
        absolute_gap_floor=float(config['absolute_gap_floor']),
    )
    rows: list[dict[str, Any]] = []
    for rank in grid:
        metrics = evaluate_truncated_rank(operator, nested, rank)
        mid = rank_is_mid_cluster(rank, clusters)
        budget = provisional_budget(
            rank=rank,
            effective_projected_rank=metrics['effective_projected_rank'],
            relative_residual=float(metrics['relative_residual_frobenius']),
            approximate_gap=metrics['approximate_gap'],
            influence_proxy_value=float(metrics['singular_values_head'][0]),
            engineering_margin=float(config['engineering_margin']),
            representation_tail_proxy=float(config['representation_tail_proxy']),
            channel_tail_proxy=float(config['channel_tail_proxy']),
        )
        row = {
            **metrics,
            'is_cluster_terminus': not mid,
            'mid_cluster': mid,
            'budget': budget,
            'resource_ok': True,
            'nested_factor_elapsed_s': nested.elapsed_s,
        }
        rows.append(row)
        atomic_write_json(sweep_root / f'rank_{rank:04d}.json', row)

    selection = select_rank_from_budgets(rows)
    if selection['selection_status'] != 'SELECTED':
        # Resource-exhausted / no gap style archive signal.
        if not any(row.get('is_cluster_terminus') for row in rows):
            selection['selection_status'] = 'REJECT_CANDIDATE'
            selection['selection_reasons'] = (
                selection.get('selection_reasons') or []
            ) + ['no cluster terminus in evaluated grid']

    summary = {
        'schema_version': 1,
        'sweep_id': sweep_root.name,
        'candidate_id': candidate_id,
        'parent_m2_run_id': m3_config.parent_run_id,
        'status': 'EXPLORATORY_NOT_CERTIFIED',
        'rank_rows': rows,
        'cluster_rows': clusters,
        'nested_max_rank': factor_rank,
        'nested_singular_values': nested.singular_values.tolist(),
        'nested_elapsed_s': nested.elapsed_s,
        'wall_elapsed_s': time.monotonic() - started,
        'selected_rank': selection.get('selected_rank'),
        'selection_status': selection.get('selection_status'),
        'selection_reasons': selection.get('selection_reasons'),
        'feasible_ranks': selection.get('feasible_ranks'),
        'source_hash': manifest['source_hash'],
        'config_hash': manifest['config_hash'],
        'sweep_root': str(sweep_root),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
        'generated_at': utc_now(),
    }
    atomic_write_json(sweep_root / 'rank_sweep_summary.json', summary)
    atomic_write_json(sweep_root / 'rank_selection.json', {
        'schema_version': 1,
        **selection,
        'sweep_id': sweep_root.name,
        'candidate_id': candidate_id,
        'parent_m2_run_id': m3_config.parent_run_id,
        'certificate_usable': False,
        'generated_at': utc_now(),
    })
    # Persist truncated factors for selected rank only (exploratory).
    if selection.get('selected_rank') is not None:
        left, singular, right_t = nested.truncate(int(selection['selected_rank']))
        np.savez_compressed(
            sweep_root / 'selected_nested_factors.npz',
            left=left, singular_values=singular, right_t=right_t,
        )
    return summary
