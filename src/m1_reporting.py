from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .checkpoint import CheckpointSaveResult, RunState
from .common import atomic_write_json, atomic_write_text, read_json, sha256_file, utc_now
from .m1_config import M1Config
from .reporting import peak_memory_report
from .work_queue import WorkQueue

REQUIRED_PHASES = (
    'M1_COEFFICIENT_BATCH', 'M1_VALUE_TAIL', 'M1_GRADIENT_TAIL',
    'M1_RG_TRAJECTORY', 'M1_INDEPENDENT_VERIFY', 'M1_REPORT',
)


def load_phase_results(run_root: Path, queue: WorkQueue) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for item in queue.items.values():
        if item.status != 'done' or not item.result_relpath:
            continue
        if item.phase in results:
            raise RuntimeError(f'Duplicate completed M1 phase artifact: {item.phase}')
        result_path = (run_root / item.result_relpath).resolve()
        try:
            result_path.relative_to(run_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f'M1 phase artifact escapes run root: {item.phase}') from exc
        if not result_path.is_file() or sha256_file(result_path) != item.result_sha256:
            raise RuntimeError(f'M1 phase artifact hash mismatch: {item.phase}')
        payload = read_json(result_path)
        if not isinstance(payload, dict) or payload.get('phase') != item.phase:
            raise RuntimeError(f'Malformed M1 phase artifact: {item.phase}')
        results[item.phase] = payload
    return results


def _acceptance_gates(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]], test_report: dict[str, Any],
) -> dict[str, bool]:
    statuses = [item.status for item in queue.items.values()]
    return {
        'm0_regression_cpu_tests': test_report.get('m0_regression_cpu_suite') == 'PASS',
        'required_cpu_tests': test_report.get('m1_required_cpu_suite') == 'PASS',
        'gpu_smoke_if_available': test_report.get('optional_gpu_suite') in {'PASS', 'NOT_RUN_NO_CUDA'},
        'fresh_process_resume': test_report.get('m1_fresh_process_resume') == 'PASS',
        'coefficient_enclosures_rigorous': results.get('M1_COEFFICIENT_BATCH', {}).get('result', {}).get('rigor') == 'RIGOROUS_RATIONAL_POSITIVE_SERIES',
        'value_tail_rigorous': results.get('M1_VALUE_TAIL', {}).get('result', {}).get('rigor') == 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
        'gradient_tail_rigorous': results.get('M1_GRADIENT_TAIL', {}).get('result', {}).get('rigor') == 'RIGOROUS_RATIONAL_ANALYTIC_BOUND',
        'exact_2d_rg_rigorous': results.get('M1_RG_TRAJECTORY', {}).get('result', {}).get('rigor') == 'RIGOROUS_RATIONAL_INTERVAL_RECURRENCE',
        'independent_verifier': results.get('M1_INDEPENDENT_VERIFY', {}).get('result', {}).get('status') == 'PASS',
        'report_work_item_ready': results.get('M1_REPORT', {}).get('result', {}).get('status') == 'READY',
        'queue_complete': bool(statuses) and all(status == 'done' for status in statuses),
        'not_certified_invariant': state.certification_status == 'NOT_CERTIFIED',
    }


def validate_m1_acceptance(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]], test_report: dict[str, Any],
) -> dict[str, bool]:
    missing = [phase for phase in REQUIRED_PHASES if phase not in results]
    if missing:
        raise RuntimeError(f'M1 acceptance is missing phase artifacts: {missing}')
    gates = _acceptance_gates(state, queue, results, test_report)
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f'M1 acceptance gates failed closed: {failed}')
    return gates


def _markdown_report(report: dict[str, Any]) -> str:
    results = report['results']
    value_entries = results['M1_VALUE_TAIL']['result']['entries']
    gradient_entries = results['M1_GRADIENT_TAIL']['result']['entries']
    trajectories = results['M1_RG_TRAJECTORY']['result']['trajectories']
    lines = [
        '# M1 exact 2D SU(2) benchmark report', '',
        f"- run ID: `{report['run_id']}`",
        f"- parent M0 run: `{report['parent']['parent_run_id']}`",
        f"- parent checkpoint: `{report['parent']['parent_checkpoint']}`",
        '- status: `NOT_CERTIFIED`',
        '- scope: M1 only; no 4D armillary/RG/mass-gap claim', '',
        '## Conventions', '',
        r'\(j=j2/2\), \(d_j=j2+1\), \(C_2=j(j+1)\), \(\mathrm{Tr}U=2\cos\theta\).', '',
        r'\(\bar w_\beta=e^{\beta(\cos\theta-1)}\), '        r'\(a_n=2n I_n(\beta)/\beta\), and \(r_n=a_n/(n a_1)\).', '',
        'All proof-path endpoints are exact rational numbers serialized in hexadecimal. Decimal strings are outward-rounded displays.', '',
        '## Proof formulas used by the implementation', '',
        r'Positive series: \(I_n(\beta)=\sum_{k\ge0}(\beta/2)^{2k+n}/(k!(n+k)!)\). A decreasing next-term ratio bounds the omitted positive tail geometrically.', '',
        r'Value tail: \(\|\bar w-P_N\bar w\|_\infty\le e^{-\beta}\sum_{n>N}2n^2 I_n(\beta)/\beta\).', '',
        r'Gradient tail: use the explicit weight-sum bound \(\|\nabla\chi_j\|_\infty\le n^2/2\) and \(I_n\le(\beta/2)^n e^{\beta^2/(4(n+1))}/n!\).', '',
        r'Independent convolution: \(\chi_n*\chi_m=\delta_{nm}\chi_n/n\), so fourfold blocking sends the coefficient to \(a_n^4/n^3\) and the normalized ratio to \(r_n^4\).', '',
        '## Value tail', '', '| cutoff n | tail lower | tail upper |', '|---:|---:|---:|',
    ]
    for cutoff, entry in value_entries.items():
        interval = entry['tail']
        lines.append(f"| {cutoff} | {interval['decimal_lo']} | {interval['decimal_hi']} |")
    lines.extend(['', '## Casimir-gradient tail', '', '| cutoff n | tail lower | tail upper | 6× upper | 216× upper |', '|---:|---:|---:|---:|---:|'])
    for cutoff, entry in gradient_entries.items():
        interval = entry['tail']; fine = entry['fine_link_6']; cell = entry['coarse_cell_216']
        lines.append(f"| {cutoff} | {interval['decimal_lo']} | {interval['decimal_hi']} | {fine['decimal_hi']} | {cell['decimal_hi']} |")
    lines.extend(['', 'The factor 6 is the fine-link contact count. The 216 factor is recorded only as a deliberately coarse cell-wide comparison and is not substituted for the contact count.', '', '## Exact 2D 2×2 RG', ''])
    for dimension, steps in trajectories.items():
        lines.extend([f'### dimension n={dimension}', '', '| step | lower | upper |', '|---:|---:|---:|'])
        for step, interval in enumerate(steps):
            lines.append(f"| {step} | {interval['decimal_lo']} | {interval['decimal_hi']} |")
        lines.append('')
    verifier = results['M1_INDEPENDENT_VERIFY']['result']
    lines.extend([
        '## Independent verification', '',
        f"- status: `{verifier['status']}`",
        f"- method: {verifier['method']}",
        f"- Arb: `{verifier['arb_status']}`", '',
        '## Proven statements', '',
        '- Wilson character coefficients are enclosed by positive rational Bessel series with a term-ratio remainder.',
        '- For the listed cutoffs, the representation value tail and Casimir-gradient tail have rigorous rational upper bounds.',
        '- For n=2,3,4 and steps 0–3, the exact 2D recurrence is enclosed and independently reproduced by diagonal convolution.', '',
        '## Limitations', '',
        '- No four-dimensional armillary tensor or four-dimensional RG step is constructed.',
        '- No influence matrix, mass-gap bound, thermodynamic limit, or continuum statement is proved.',
        '- `CERTIFIED` remains forbidden; the milestone status is `NOT_CERTIFIED`.', '',
    ])
    return '\n'.join(lines)


def write_m1_report_package(
    run_root: Path, config: M1Config, state: RunState, queue: WorkQueue,
    test_report: dict[str, Any], last_checkpoint: CheckpointSaveResult, manifest: dict[str, Any],
) -> dict[str, str]:
    results = load_phase_results(run_root, queue)
    gates = validate_m1_acceptance(state, queue, results, test_report)
    report = {
        'schema_version': 1, 'milestone': 'M1', 'phase': state.phase,
        'run_id': state.run_id, 'generated_at': utc_now(),
        'certification_status': 'NOT_CERTIFIED',
        'parent': {key: manifest[key] for key in (
            'parent_milestone', 'parent_run_id', 'parent_checkpoint', 'parent_checkpoint_path',
            'parent_checkpoint_hash_manifest_sha256',
        )},
        'config': config.canonical_payload(), 'config_hash': config.config_hash(),
        'source_hash': manifest['source_hash'],
        'governing_document_hashes': manifest['governing_document_hashes'],
        'reference_artifact_hashes': manifest['reference_artifact_hashes'],
        'm0_acceptance_record_sha256': manifest['m0_acceptance_record_sha256'],
        'results': results, 'tests': test_report, 'acceptance_gates': gates,
        'proof_artifact_hashes': {
            item.phase: item.result_sha256 for item in queue.items.values()
            if item.status == 'done' and item.result_sha256 is not None
        },
        'checkpoint': {
            'path': str(last_checkpoint.path), 'index': last_checkpoint.index,
            'size_bytes': last_checkpoint.size_bytes, 'save_s': last_checkpoint.save_s,
            'verify_s': last_checkpoint.verify_s,
        },
        'memory': peak_memory_report(),
        'rigorous_results': [
            'positive-series Wilson coefficient enclosures', 'value-tail enclosures',
            'Casimir-gradient tail upper bounds', 'exact 2D r_n -> r_n^4 trajectories',
            'independent finite diagonal-convolution containment',
        ],
        'heuristic_results': [],
        'unresolved_issues': [
            'P0 transitive checkpoint hash-chain certification is not claimed at M1.',
            'Arb is optional; two independent rational paths are the required M1 verifier.',
            'No four-dimensional RG enclosure has been constructed.',
        ],
        'remaining_todos': ['M2 low-cutoff armillary', 'M3 GPU Triad-ATRG', 'M4 forward AD', 'M5 one-step validation', 'M6 multi-step certificate'],
    }
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / 'M1_report.json'; md_path = report_dir / 'M1_report.md'
    acceptance_path = report_dir / 'M1_acceptance.json'
    atomic_write_json(json_path, report)
    atomic_write_text(md_path, _markdown_report(report))
    atomic_write_json(acceptance_path, {
        'milestone': 'M1', 'phase': 'M1_COMPLETE', 'status': 'PASS',
        'certification_status': 'NOT_CERTIFIED', 'gates': gates, 'generated_at': utc_now(),
    })
    return {'json': str(json_path), 'markdown': str(md_path), 'acceptance': str(acceptance_path)}


def write_m1_session_artifacts(
    run_root: Path, state: RunState, queue: WorkQueue, stop_reason: str,
    elapsed_s: float, remaining_s: float, persistent_root: Path, project_root: Path,
) -> dict[str, str]:
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    counts = {status: sum(item.status == status for item in queue.items.values()) for status in ('pending', 'running', 'done', 'failed', 'blocked_resource')}
    unfinished = [item.item_id for item in queue.items.values() if item.status != 'done']
    summary_path = report_dir / 'session_summary.json'; metrics_path = report_dir / 'latest_metrics.json'
    plan_path = report_dir / 'next_session_plan.md'
    atomic_write_json(summary_path, {
        'milestone': 'M1', 'run_id': state.run_id, 'phase': state.phase,
        'certification_status': 'NOT_CERTIFIED', 'stop_reason': stop_reason,
        'elapsed_s': elapsed_s, 'remaining_s': remaining_s, 'queue': counts,
        'unfinished_work_items': unfinished, 'generated_at': utc_now(),
    })
    atomic_write_json(metrics_path, {
        'milestone': 'M1', 'run_id': state.run_id, 'memory': peak_memory_report(),
        'rigorous_bounds': sorted(state.bounds), 'approximate_spectral_radius': None,
        'certification_status': 'NOT_CERTIFIED', 'generated_at': utc_now(),
    })
    if state.phase == 'M1_COMPLETE':
        plan = '# Next session plan\n\nM1 is complete. Review M1_acceptance.json and do not start M2 automatically.\n'
    else:
        plan = (
            '# Next session plan\n\n'
            f'1. Reopen `{project_root}` with the same Paperspace runtime.\n'
            f'2. Set `VALIDATED_RG_PERSIST_ROOT={persistent_root}`.\n'
            '3. Set `VALIDATED_RG_PERSIST_ACK=I_CONFIRM_THIS_PATH_IS_PERSISTENT`.\n'
            f'4. Set `VALIDATED_RG_M1_RUN_ID={state.run_id}`.\n'
            '5. Use a fresh kernel, run the notebook top-to-bottom, then call `m1_orchestrator.run_until_checkpoint()`.\n'
        )
    atomic_write_text(plan_path, plan)
    return {'session_summary': str(summary_path), 'latest_metrics': str(metrics_path), 'next_session_plan': str(plan_path)}
