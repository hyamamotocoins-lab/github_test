from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, atomic_write_text, read_json, sanitize_for_json,
    sha256_file, utc_now,
)
from .m3_config import M3Config
from .reporting import peak_memory_report
from .work_queue import WorkQueue

M3_PHASES = (
    'M3_BACKEND_DIAGNOSTIC', 'M3_OPERATOR_BUILD',
    'M3_MATRIX_FREE_VALIDATE', 'M3_RSVD', 'M3_TRIAD', 'M3_REPORT',
)


def load_m3_phase_results(
    run_root: Path, queue: WorkQueue,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for item in queue.items.values():
        if item.status != 'done' or not item.result_relpath:
            continue
        if item.phase in results:
            raise RuntimeError(f'Duplicate M3 phase artifact: {item.phase}')
        path = (run_root / item.result_relpath).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f'M3 phase artifact escapes run root: {item.phase}') from exc
        if not path.is_file() or sha256_file(path) != item.result_sha256:
            raise RuntimeError(f'M3 phase artifact hash mismatch: {item.phase}')
        payload = read_json(path)
        if (
            not isinstance(payload, dict)
            or payload.get('phase') != item.phase
            or payload.get('certification_status') != 'NOT_CERTIFIED'
        ):
            raise RuntimeError(f'Malformed or unsafe M3 phase artifact: {item.phase}')
        results[item.phase] = payload
    return results


def _finite_nonnegative(value: object) -> bool:
    import math
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0.0
    )


def m3_acceptance_gates(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any], *,
    sector_count: int = 64,
    operator_dimension: int = 729,
) -> dict[str, bool]:
    backend = results.get('M3_BACKEND_DIAGNOSTIC', {}).get('result', {})
    operator = results.get('M3_OPERATOR_BUILD', {}).get('result', {})
    validation = results.get('M3_MATRIX_FREE_VALIDATE', {}).get('result', {})
    rsvd = results.get('M3_RSVD', {}).get('result', {})
    triad = results.get('M3_TRIAD', {}).get('result', {})
    report = results.get('M3_REPORT', {}).get('result', {})
    statuses = [item.status for item in queue.items.values()]
    singular_values = rsvd.get('singular_values')
    return {
        'accepted_m2_parent_reverified': test_report.get('accepted_m2_parent') == 'PASS',
        'm0_m1_m2_regression_cpu_tests': (
            test_report.get('m0_m1_m2_regression_cpu_suite') == 'PASS'
        ),
        'm3_required_cpu_tests': test_report.get('m3_required_cpu_suite') == 'PASS',
        'm3_required_gpu_tests': test_report.get('m3_required_gpu_suite') == 'PASS',
        'fresh_process_resume': test_report.get('m3_fresh_process_resume') == 'PASS',
        'checkpoint_basis_restore': test_report.get('m3_checkpoint_basis_restore') == 'PASS',
        'oom_recovery': test_report.get('m3_oom_recovery') == 'PASS',
        'cuda_fp64_backend': (
            backend.get('status') == 'PASS'
            and backend.get('selection', {}).get('is_cuda') is True
            and backend.get('selection', {}).get('dtype') == 'float64'
            and backend.get('tf32_disabled') is True
        ),
        'operator_parent_shards_complete': (
            operator.get('status') == 'PASS'
            and operator.get('sector_count') == sector_count
            and operator.get('dimension') == operator_dimension
            and operator.get('parent_tensor_count') == sector_count
        ),
        'matrix_free_matches_explicit': (
            validation.get('status') == 'PASS'
            and _finite_nonnegative(validation.get('matvec_max_abs_error'))
            and validation.get('matvec_max_abs_error', float('inf')) <= 1e-12
        ),
        'adjoint_consistency': (
            _finite_nonnegative(validation.get('adjoint_relative_error'))
            and validation.get('adjoint_relative_error', float('inf')) <= 1e-12
        ),
        'path_cache_reused': validation.get('path_cache_reused') is True,
        'rsvd_finite_fixed_seed': (
            rsvd.get('status') == 'PASS'
            and isinstance(singular_values, list) and bool(singular_values)
            and all(_finite_nonnegative(value) for value in singular_values)
            and rsvd.get('rigor') == 'EXPLORATORY_FIXED_SEED_NOT_A_CERTIFICATE'
        ),
        'rsvd_explicit_low_cutoff_comparison': (
            _finite_nonnegative(rsvd.get('explicit_top_singular_max_abs_error'))
            and rsvd.get('explicit_top_singular_max_abs_error', float('inf')) <= 1e-5
            and _finite_nonnegative(rsvd.get('residual_to_explicit_optimal_ratio'))
            and rsvd.get('residual_to_explicit_optimal_ratio', float('inf')) <= 1.00001
        ),
        'triad_factors_reproduced': (
            triad.get('status') == 'PASS'
            and triad.get('rank') == rsvd.get('target_rank')
            and _finite_nonnegative(triad.get('relative_residual_frobenius'))
        ),
        'report_work_item_ready': report.get('status') == 'READY',
        'queue_complete': bool(statuses) and all(status == 'done' for status in statuses),
        'core_reproduced_only': (
            state.certification_status == 'NOT_CERTIFIED'
            and rsvd.get('milestone_status') == 'CORE_REPRODUCED'
        ),
    }


def validate_m3_acceptance(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any], *,
    sector_count: int = 64,
    operator_dimension: int = 729,
) -> dict[str, bool]:
    missing = [phase for phase in M3_PHASES if phase not in results]
    if missing:
        raise RuntimeError(f'M3 acceptance is missing phase artifacts: {missing}')
    gates = m3_acceptance_gates(
        state, queue, results, test_report,
        sector_count=sector_count,
        operator_dimension=operator_dimension,
    )
    failed = [name for name, value in gates.items() if not value]
    if failed:
        raise RuntimeError(f'M3 acceptance failed closed: {failed}')
    return gates


def _markdown(report: dict[str, Any]) -> str:
    results = report['results']
    backend = results['M3_BACKEND_DIAGNOSTIC']['result']
    validation = results['M3_MATRIX_FREE_VALIDATE']['result']
    rsvd = results['M3_RSVD']['result']
    triad = results['M3_TRIAD']['result']
    lines = [
        '# M3 GPU matrix-free Triad-ATRG pilot report', '',
        f"- run ID: {report['run_id']}",
        f"- accepted M2 parent: {report['parent']['parent_run_id']}",
        '- milestone status: CORE_REPRODUCED',
        '- certification status: NOT_CERTIFIED',
        '- interpretation: exploratory finite-core GPU pilot only', '',
        '## Runtime', '',
        f"- backend: {backend['selection']['name']}",
        f"- GPU: {backend['selection']['gpu_name']}",
        f"- dtype: {backend['selection']['dtype']}",
        f"- TF32 disabled: {backend['tf32_disabled']}", '',
        '## Matrix-free validation', '',
        f"- global operator dimension: {validation['dimension']}",
        f"- explicit matvec max abs error: {validation['matvec_max_abs_error']:.6e}",
        f"- adjoint relative error: {validation['adjoint_relative_error']:.6e}",
        f"- path cache reused: {validation['path_cache_reused']}", '',
        '## Fixed-seed RSVD and Triad', '',
        f"- target rank: {rsvd['target_rank']}",
        f"- relative Frobenius residual: {rsvd['relative_residual_frobenius']:.6e}",
        f"- explicit-optimal residual ratio: {rsvd['residual_to_explicit_optimal_ratio']:.9f}",
        f"- approximate influence proxy: {rsvd['influence_proxy']['value']:.9f}",
        f"- screening: {rsvd['influence_proxy']['screening']}",
        f"- triad factor bytes: {triad['factor_bytes']}", '',
        '## Nonclaims', '',
        '- RSVD probability, GPU rounding, singular decay, and influence proxy are heuristic.',
        '- No deterministic RSVD certificate or 4D RG error enclosure is claimed.',
        '- No mass-gap, thermodynamic-limit, continuum-limit, or CERTIFIED claim is made.',
        '- M4 may begin only after independent review of this report.', '',
    ]
    return '\n'.join(lines)


def write_m3_report_package(
    run_root: Path, config: M3Config, state: RunState, queue: WorkQueue,
    test_report: dict[str, Any], checkpoint: CheckpointSaveResult,
    manifest: dict[str, Any],
) -> dict[str, str]:
    results = load_m3_phase_results(run_root, queue)
    gates = validate_m3_acceptance(
        state, queue, results, test_report,
        sector_count=config.sector_count,
        operator_dimension=config.operator_dimension,
    )
    rsvd = results['M3_RSVD']['result']
    report = {
        'schema_version': 1, 'milestone': 'M3', 'phase': state.phase,
        'run_id': state.run_id, 'generated_at': utc_now(),
        'milestone_status': 'CORE_REPRODUCED',
        'certification_status': 'NOT_CERTIFIED',
        'parent': {key: manifest[key] for key in (
            'parent_milestone', 'parent_run_id', 'parent_checkpoint',
            'parent_checkpoint_path', 'parent_checkpoint_hash_manifest_sha256',
            'm2_report_sha256', 'm2_acceptance_sha256', 'm2_audit_sha256',
        )},
        'config': config.canonical_payload(), 'config_hash': config.config_hash(),
        'source_hash': manifest['source_hash'],
        'notebook_hash': manifest['notebook_hash'],
        'results': results, 'tests': test_report, 'acceptance_gates': gates,
        'proof_artifact_hashes': {
            item.phase: item.result_sha256 for item in queue.items.values()
            if item.status == 'done' and item.result_sha256
        },
        'checkpoint': {
            'path': str(checkpoint.path), 'index': checkpoint.index,
            'size_bytes': checkpoint.size_bytes, 'save_s': checkpoint.save_s,
            'verify_s': checkpoint.verify_s,
        },
        'memory': peak_memory_report(),
        'gpu_memory': results['M3_BACKEND_DIAGNOSTIC']['result']['memory_after'],
        'reproduced_results': [
            'matrix-free sector-shard matvec agrees with explicit low-cutoff matrix',
            'matrix-free adjoint consistency',
            'fixed-seed RSVD basis is checkpoint-reproducible',
            'three-factor Triad pilot reconstructed from the fixed RSVD result',
            'contraction path cache reuse and deterministic shard ordering',
        ],
        'heuristic_results': [
            'RSVD singular-value decay', 'RSVD residual estimate',
            'Triad truncation residual', 'approximate influence proxy',
            'screening decision ' + rsvd['influence_proxy']['screening'],
        ],
        'rigorous_bounds': [],
        'unresolved_issues': [
            'No deterministic RSVD residual enclosure exists at M3.',
            'GPU backward/rounding error is not yet bounded.',
            'The influence proxy is not a rigorous spectral-radius bound.',
            'M3 is a j2_max=1 finite-core pilot, not a 4D RG certificate.',
        ],
        'remaining_todos': [
            'independent M3 acceptance review', 'M4 forward derivatives',
            'M5 one-step validation', 'M6 multi-step certificate',
        ],
    }
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / 'M3_report.json'
    markdown_path = report_dir / 'M3_report.md'
    acceptance_path = report_dir / 'M3_acceptance.json'
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, _markdown(report))
    atomic_write_json(acceptance_path, {
        'milestone': 'M3', 'phase': 'M3_COMPLETE', 'status': 'PASS',
        'milestone_status': 'CORE_REPRODUCED',
        'certification_status': 'NOT_CERTIFIED',
        'gates': gates, 'generated_at': utc_now(),
    })
    return {
        'json': str(json_path), 'markdown': str(markdown_path),
        'acceptance': str(acceptance_path),
    }


def write_m3_session_artifacts(
    run_root: Path, state: RunState, queue: WorkQueue, stop_reason: str,
    elapsed_s: float, remaining_s: float, persistent_root: Path,
    project_root: Path,
) -> dict[str, str]:
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        status: sum(item.status == status for item in queue.items.values())
        for status in ('pending', 'running', 'done', 'failed', 'blocked_resource')
    }
    summary_path = report_dir / 'session_summary.json'
    metrics_path = report_dir / 'latest_metrics.json'
    plan_path = report_dir / 'next_session_plan.md'
    summary_payload, summary_nf = sanitize_for_json({
        'milestone': 'M3', 'run_id': state.run_id, 'phase': state.phase,
        'milestone_status': (
            'CORE_REPRODUCED' if state.phase == 'M3_COMPLETE' else 'EXPLORATORY'
        ),
        'certification_status': 'NOT_CERTIFIED', 'stop_reason': stop_reason,
        'elapsed_s': elapsed_s, 'remaining_s': remaining_s,
        'queue': counts, 'generated_at': utc_now(),
    })
    if summary_nf:
        summary_payload['nonfinite_values_present'] = True
        summary_payload['certification_status'] = 'NOT_CERTIFIED'
    atomic_write_json(summary_path, summary_payload)
    metrics_payload, metrics_nf = sanitize_for_json({
        'milestone': 'M3', 'run_id': state.run_id,
        'memory': peak_memory_report(), 'certification_status': 'NOT_CERTIFIED',
        'generated_at': utc_now(),
    })
    if metrics_nf:
        metrics_payload['nonfinite_values_present'] = True
        metrics_payload['certification_status'] = 'NOT_CERTIFIED'
    atomic_write_json(metrics_path, metrics_payload)
    if state.phase == 'M3_COMPLETE':
        plan = (
            '# Next session plan\n\nM3 is CORE_REPRODUCED. Review '
            'M3_report.json; do not start M4 automatically.\n'
        )
    else:
        plan = (
            '# Next session plan\n\n'
            f'1. Reopen {project_root} on the same CUDA runtime.\n'
            f'2. Set VALIDATED_RG_PERSIST_ROOT={persistent_root}.\n'
            '3. Set VALIDATED_RG_PERSIST_ACK=I_CONFIRM_THIS_PATH_IS_PERSISTENT.\n'
            f'4. Set VALIDATED_RG_M3_RUN_ID={state.run_id}.\n'
            '5. Use a fresh kernel and run notebooks/40_m3_gpu_triad_atrg.ipynb '
            'top-to-bottom.\n'
        )
    atomic_write_text(plan_path, plan)
    return {
        'session_summary': str(summary_path),
        'latest_metrics': str(metrics_path),
        'next_session_plan': str(plan_path),
    }
