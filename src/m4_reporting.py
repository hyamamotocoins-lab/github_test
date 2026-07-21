from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointSaveResult, RunState
from .common import atomic_write_json, atomic_write_text, read_json, sha256_file, utc_now
from .m4_config import M4Config
from .reporting import peak_memory_report
from .work_queue import WorkQueue

M4_PHASES = (
    'M4_SOURCE_CHANNELS', 'M4_DUAL_PIPELINE', 'M4_NORMALIZATION',
    'M4_FINITE_DIFFERENCE', 'M4_ERROR_LEDGER', 'M4_REPORT',
)


def load_m4_phase_results(
    run_root: Path, queue: WorkQueue,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for item in queue.items.values():
        if item.status != 'done' or not item.result_relpath:
            continue
        if item.phase in results:
            raise RuntimeError(f'Duplicate M4 phase artifact: {item.phase}')
        path = (run_root / item.result_relpath).resolve()
        try:
            path.relative_to(run_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f'M4 phase artifact escapes run root: {item.phase}') from exc
        if not path.is_file() or sha256_file(path) != item.result_sha256:
            raise RuntimeError(f'M4 phase artifact hash mismatch: {item.phase}')
        payload = read_json(path)
        if (
            not isinstance(payload, dict)
            or payload.get('phase') != item.phase
            or payload.get('certification_status') != 'NOT_CERTIFIED'
        ):
            raise RuntimeError(f'Malformed or unsafe M4 artifact: {item.phase}')
        results[item.phase] = payload
    return results


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0.0
    )


def m4_acceptance_gates(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any],
) -> dict[str, bool]:
    sources = results.get('M4_SOURCE_CHANNELS', {}).get('result', {})
    pipeline = results.get('M4_DUAL_PIPELINE', {}).get('result', {})
    normalization = results.get('M4_NORMALIZATION', {}).get('result', {})
    difference = results.get('M4_FINITE_DIFFERENCE', {}).get('result', {})
    ledger = results.get('M4_ERROR_LEDGER', {}).get('result', {})
    report = results.get('M4_REPORT', {}).get('result', {})
    statuses = [item.status for item in queue.items.values()]
    aggregate = ledger.get('aggregates', {})
    return {
        'accepted_m3_parent_reverified': (
            test_report.get('accepted_m3_parent') == 'PASS'
        ),
        'm0_through_m3_regression_cpu_tests': (
            test_report.get('m0_m1_m2_m3_regression_cpu_suite') == 'PASS'
        ),
        'm4_required_cpu_tests': test_report.get('m4_required_cpu_suite') == 'PASS',
        'm4_required_gpu_tests': test_report.get('m4_required_gpu_suite') == 'PASS',
        'fresh_process_resume': test_report.get('m4_fresh_process_resume') == 'PASS',
        'derivative_checkpoint_restore': (
            test_report.get('m4_derivative_checkpoint_restore') == 'PASS'
        ),
        'source_channels_symmetry_reduced': (
            sources.get('status') == 'PASS'
            and sources.get('source_count') == 5
            and sources.get('max_symmetry_residual') == 0.0
        ),
        'zero_source_tangents_zero': sources.get('zero_source_max_abs') == 0.0,
        'multilinear_forward_ad': (
            pipeline.get('status') == 'PASS'
            and pipeline.get('contraction_rule') == 'PRODUCT_RULE_COMPLETE'
            and pipeline.get('fixed_basis_projection') is True
            and pipeline.get('regrouping') is True
        ),
        'basis_variation_not_silently_zero': (
            pipeline.get('basis_variation_policy')
            == 'FIXED_BASIS_WITH_EXPLICIT_LEDGER_TERM'
            and ledger.get('basis_variation_accounted') is True
        ),
        'normalization_derivative_complete': (
            normalization.get('status') == 'PASS'
            and _finite_nonnegative(normalization.get('scale'))
            and normalization.get('scale', 0.0) > 0.0
            and abs(normalization.get('normalized_frobenius_norm', 0.0) - 1.0)
            <= 1e-12
            and normalization.get('all_outputs_finite') is True
        ),
        'finite_difference_converges': (
            difference.get('status') == 'PASS'
            and difference.get('all_channels_converged') is True
            and difference.get('finite_difference_is_proof_bound') is False
        ),
        'error_dag_complete_provenance': (
            ledger.get('status') == 'PASS'
            and ledger.get('summary', {}).get('double_counting_check') == 'PASS'
            and ledger.get('required_categories_complete') is True
        ),
        'all_partial_output_radii_finite_nonnegative': (
            bool(aggregate)
            and all(
                _finite_nonnegative(value.get('partial_estimate'))
                for value in aggregate.values()
            )
        ),
        'missing_bounds_block_enclosure': (
            ledger.get('summary', {}).get('enclosure_ready') is False
            and bool(
                ledger.get('summary', {}).get(
                    'missing_deterministic_bound_terms'
                )
            )
            and ledger.get('enclosure_status') == 'BLOCKED_MATH'
        ),
        'report_lists_every_error_term': report.get('status') == 'READY',
        'queue_complete': bool(statuses) and all(status == 'done' for status in statuses),
        'not_certified_invariant': (
            state.certification_status == 'NOT_CERTIFIED'
            and ledger.get('milestone_status') == 'BLOCKED_MATH'
        ),
    }


def validate_m4_acceptance(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any],
) -> dict[str, bool]:
    missing = [phase for phase in M4_PHASES if phase not in results]
    if missing:
        raise RuntimeError(f'M4 acceptance is missing phase artifacts: {missing}')
    gates = m4_acceptance_gates(state, queue, results, test_report)
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f'M4 acceptance failed closed: {failed}')
    return gates


def _markdown(report: dict[str, Any]) -> str:
    normalization = report['results']['M4_NORMALIZATION']['result']
    difference = report['results']['M4_FINITE_DIFFERENCE']['result']
    ledger = report['results']['M4_ERROR_LEDGER']['result']
    lines = [
        '# M4 forward derivative and error-ledger report', '',
        f"- run ID: {report['run_id']}",
        f"- accepted M3 parent: {report['parent']['parent_run_id']}",
        '- phase: M4_COMPLETE',
        '- milestone status: BLOCKED_MATH',
        '- certification status: NOT_CERTIFIED', '',
        '## Forward derivative', '',
        '- source channels: temporal link, spatial link, electric-like, '
        'magnetic-like, low-representation',
        f"- normalization scale: {normalization['scale']:.16e}",
        f"- normalized Frobenius norm: "
        f"{normalization['normalized_frobenius_norm']:.16e}",
        f"- maximum final finite-difference relative error: "
        f"{difference['max_final_relative_error']:.6e}",
        '- finite difference is regression evidence, not a proof bound.', '',
        '## Complete error ledger', '',
    ]
    for term in ledger['ledger']['terms']:
        lines.extend([
            f"### {term['name']}", '',
            f"- category: {term['category']}",
            f"- applies to: {term['applies_to']}",
            f"- rigor: {term['rigor']}",
            f"- estimate: {term['estimate']}",
            f"- deterministic upper bound: {term['deterministic_upper_bound']}",
            f"- source checkpoint: {term['source_checkpoint']}",
            f"- formula: {term['formula']}",
            f"- parents: {term['parents']}",
            f"- note: {term['note']}", '',
        ])
    lines.extend([
        '## Fail-closed verdict', '',
        f"- missing deterministic bounds: "
        f"{ledger['summary']['missing_deterministic_bound_terms']}",
        '- basis variation is explicit and is not set to zero.',
        '- No enclosed one-step RG output is claimed.',
        '- M5 must not begin while milestone status is BLOCKED_MATH.', '',
    ])
    return '\n'.join(lines)


def write_m4_report_package(
    run_root: Path, config: M4Config, state: RunState, queue: WorkQueue,
    test_report: dict[str, Any], checkpoint: CheckpointSaveResult,
    manifest: dict[str, Any],
) -> dict[str, str]:
    results = load_m4_phase_results(run_root, queue)
    gates = validate_m4_acceptance(state, queue, results, test_report)
    ledger = results['M4_ERROR_LEDGER']['result']
    report = {
        'schema_version': 1, 'milestone': 'M4', 'phase': state.phase,
        'run_id': state.run_id, 'generated_at': utc_now(),
        'milestone_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
        'enclosure_status': 'BLOCKED_MATH',
        'parent': {key: manifest[key] for key in (
            'parent_milestone', 'parent_run_id', 'parent_checkpoint',
            'parent_checkpoint_path', 'parent_checkpoint_hash_manifest_sha256',
            'm3_report_sha256', 'm3_acceptance_sha256', 'm3_audit_sha256',
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
        'gpu_memory': results['M4_DUAL_PIPELINE']['result']['gpu_memory'],
        'rigorous_error_terms': [
            term['name'] for term in ledger['ledger']['terms']
            if term['rigor'] == 'RIGOROUS'
        ],
        'heuristic_error_terms': [
            term['name'] for term in ledger['ledger']['terms']
            if term['rigor'] == 'HEURISTIC'
        ],
        'missing_bound_terms': ledger['summary']['missing_deterministic_bound_terms'],
        'unresolved_issues': [
            'No deterministic M3 RSVD residual enclosure.',
            'No GPU backward/rounding bound.',
            'No fixed-basis variation bound.',
            'No validated normalization lower bound.',
            'Cutoff/rank dependence remains unresolved after M3 screening.',
        ],
        'remaining_todos': [
            'supply every missing deterministic error bound',
            'investigate cutoff/rank dependence',
            'independent M4 review before any M5 implementation',
            'M5 one-step validation', 'M6 multi-step certificate',
        ],
    }
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / 'M4_report.json'
    markdown_path = report_dir / 'M4_report.md'
    acceptance_path = report_dir / 'M4_acceptance.json'
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, _markdown(report))
    atomic_write_json(acceptance_path, {
        'milestone': 'M4', 'phase': 'M4_COMPLETE',
        'status': 'PASS', 'milestone_status': 'BLOCKED_MATH',
        'enclosure_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
        'gates': gates, 'generated_at': utc_now(),
    })
    return {
        'json': str(json_path), 'markdown': str(markdown_path),
        'acceptance': str(acceptance_path),
    }


def write_m4_session_artifacts(
    run_root: Path, state: RunState, queue: WorkQueue, stop_reason: str,
    elapsed_s: float, remaining_s: float | None, persistent_root: Path,
    project_root: Path, session_policy: str,
) -> dict[str, str]:
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        status: sum(item.status == status for item in queue.items.values())
        for status in ('pending', 'running', 'done', 'failed', 'blocked_resource')
    }
    summary_path = report_dir / 'session_summary.json'
    metrics_path = report_dir / 'latest_metrics.json'
    plan_path = report_dir / 'next_session_plan.md'
    atomic_write_json(summary_path, {
        'milestone': 'M4', 'run_id': state.run_id, 'phase': state.phase,
        'milestone_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
        'session_policy': session_policy, 'stop_reason': stop_reason,
        'elapsed_s': elapsed_s, 'remaining_s': remaining_s,
        'queue': counts, 'generated_at': utc_now(),
    })
    atomic_write_json(metrics_path, {
        'milestone': 'M4', 'run_id': state.run_id,
        'memory': peak_memory_report(), 'certification_status': 'NOT_CERTIFIED',
        'generated_at': utc_now(),
    })
    if state.phase == 'M4_COMPLETE':
        plan = (
            '# Next session plan\n\nM4 derivative workflow is complete but '
            'BLOCKED_MATH. Review every missing term in M4_report.md; do not '
            'start M5.\n'
        )
    else:
        plan = (
            '# Next session plan\n\n'
            f'1. Reopen {project_root} on the same CUDA runtime.\n'
            f'2. Set VALIDATED_RG_PERSIST_ROOT={persistent_root}.\n'
            '3. Set VALIDATED_RG_PERSIST_ACK=I_CONFIRM_THIS_PATH_IS_PERSISTENT.\n'
            f'4. Set VALIDATED_RG_M3_RUN_ID=M3-20260720T013551Z-ae995e91e861.\n'
            f'5. Set VALIDATED_RG_M4_RUN_ID={state.run_id}.\n'
            '6. Use a fresh kernel and run notebooks/50_m4_derivatives.ipynb '
            'top-to-bottom. Resumed sessions use the normal 5 h 30 min limit.\n'
        )
    atomic_write_text(plan_path, plan)
    return {
        'session_summary': str(summary_path),
        'latest_metrics': str(metrics_path),
        'next_session_plan': str(plan_path),
    }
