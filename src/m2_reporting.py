from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import CheckpointSaveResult, RunState
from .common import (
    atomic_write_json, atomic_write_text, read_json, safe_component, sha256_file,
    utc_now,
)
from .cutoff_dims import expected_m2_gate_counts
from .m2_batching import merge_m2_batch_payloads, proof_artifact_hash_map
from .m2_config import M2Config
from .reporting import peak_memory_report
from .work_queue import WorkQueue

REQUIRED_PHASES = (
    'M2_WIGNER_CACHE', 'M2_DENSE_REFERENCE', 'M2_ARMILLARY',
    'M2_EQUIVALENCE', 'M2_SYMMETRY', 'M2_REPORT',
)


def load_m2_phase_results(
    run_root: Path, queue: WorkQueue, *, j2_max: int = 1,
) -> dict[str, dict[str, Any]]:
    singles: dict[str, dict[str, Any]] = {}
    batches: dict[str, list[dict[str, Any]]] = {}
    for item in queue.items.values():
        if item.status != 'done' or not item.result_relpath:
            continue
        result_path = (run_root / item.result_relpath).resolve()
        try:
            result_path.relative_to(run_root.resolve())
        except ValueError as exc:
            raise RuntimeError(f'M2 phase artifact escapes run root: {item.phase}') from exc
        if not result_path.is_file() or sha256_file(result_path) != item.result_sha256:
            raise RuntimeError(f'M2 phase artifact hash mismatch: {item.phase}')
        payload = read_json(result_path)
        if not isinstance(payload, dict) or payload.get('phase') != item.phase:
            raise RuntimeError(f'Malformed M2 phase artifact: {item.phase}')
        if payload.get('certification_status') != 'NOT_CERTIFIED':
            raise RuntimeError(f'M2 phase artifact changed certification status: {item.phase}')
        if item.phase == 'M2_WIGNER_CACHE':
            result = payload.get('result')
            if not isinstance(result, dict):
                raise RuntimeError('Malformed M2 Wigner result.')
            for filename_key, digest_key in (
                ('cache_filename', 'cache_sha256'),
                ('regenerated_filename', 'regenerated_sha256'),
            ):
                filename = result.get(filename_key)
                digest = result.get(digest_key)
                if not isinstance(filename, str) or safe_component(filename) != filename:
                    raise RuntimeError('Unsafe M2 Wigner cache filename.')
                cache_path = result_path.parent / filename
                if not cache_path.is_file() or sha256_file(cache_path) != digest:
                    raise RuntimeError('M2 Wigner cache artifact hash mismatch.')
        if 'batch_index' in item.parameters:
            batches.setdefault(item.phase, []).append(payload)
        else:
            if item.phase in singles or item.phase in batches:
                raise RuntimeError(f'Duplicate completed M2 phase artifact: {item.phase}')
            singles[item.phase] = payload
    results = dict(singles)
    for phase, payloads in batches.items():
        if phase in singles:
            raise RuntimeError(f'Mixed batched/unbatched artifacts for {phase}')
        results[phase] = merge_m2_batch_payloads(phase, payloads, j2_max=j2_max)
    return results


def _acceptance_gates(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any], *, j2_max: int,
) -> dict[str, bool]:
    expected = expected_m2_gate_counts(j2_max)
    n = expected['sector_count']
    odd = expected['odd_half_zero_count']
    wigner = results.get('M2_WIGNER_CACHE', {}).get('result', {})
    dense = results.get('M2_DENSE_REFERENCE', {}).get('result', {})
    armillary = results.get('M2_ARMILLARY', {}).get('result', {})
    equivalence = results.get('M2_EQUIVALENCE', {}).get('result', {})
    symmetry = results.get('M2_SYMMETRY', {}).get('result', {})
    report = results.get('M2_REPORT', {}).get('result', {})
    statuses = [item.status for item in queue.items.values()]
    return {
        'accepted_m1_parent_reverified': test_report.get('accepted_m1_parent') == 'PASS',
        'm0_m1_regression_cpu_tests': test_report.get('m0_m1_regression_cpu_suite') == 'PASS',
        'required_m2_cpu_tests': test_report.get('m2_required_cpu_suite') == 'PASS',
        'fresh_process_resume': test_report.get('m2_fresh_process_resume') == 'PASS',
        'gpu_smoke_if_available': test_report.get('optional_gpu_suite') in {
            'PASS', 'NOT_RUN_NO_CUDA',
        },
        'wigner_cache_exact_and_deterministic': (
            wigner.get('status') == 'PASS'
            and isinstance(wigner.get('entry_count'), int)
            and wigner.get('entry_count', 0) > 0
            and wigner.get('regeneration_sha256_match') is True
        ),
        'dense_reference_all_sectors': (
            dense.get('status') == 'PASS'
            and dense.get('sector_count') == n
            and dense.get('generator_residual_zero_count') == n
        ),
        'gauge_noninvariant_sectors_vanish': dense.get('odd_half_zero_count') == odd,
        'armillary_all_sectors_exact_isometries': (
            armillary.get('status') == 'PASS'
            and armillary.get('sector_count') == n
            and armillary.get('isometry_exact_count') == n
        ),
        'dense_armillary_exact_equivalence': (
            equivalence.get('status') == 'PASS'
            and equivalence.get('exact_match_count') == n
            and equivalence.get('mismatches') == []
        ),
        'cubic_symmetry_deterministic': (
            symmetry.get('status') == 'PASS'
            and symmetry.get('group_order') == 48
            and symmetry.get('deterministic') is True
            and 1 < symmetry.get('canonical_sector_count', 0) < n
        ),
        'report_work_item_ready': report.get('status') == 'READY',
        'queue_complete': bool(statuses) and all(status == 'done' for status in statuses),
        'not_certified_invariant': state.certification_status == 'NOT_CERTIFIED',
    }


def validate_m2_acceptance(
    state: RunState, queue: WorkQueue, results: dict[str, dict[str, Any]],
    test_report: dict[str, Any], *, j2_max: int = 1,
) -> dict[str, bool]:
    missing = [phase for phase in REQUIRED_PHASES if phase not in results]
    if missing:
        raise RuntimeError(f'M2 acceptance is missing phase artifacts: {missing}')
    gates = _acceptance_gates(
        state, queue, results, test_report, j2_max=j2_max,
    )
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        raise RuntimeError(f'M2 acceptance gates failed closed: {failed}')
    return gates


def _markdown_report(report: dict[str, Any]) -> str:
    results = report['results']
    dense = results['M2_DENSE_REFERENCE']['result']
    armillary = results['M2_ARMILLARY']['result']
    equivalence = results['M2_EQUIVALENCE']['result']
    symmetry = results['M2_SYMMETRY']['result']
    j2_max = int(report.get('config', {}).get('j2_max', 1))
    expected = expected_m2_gate_counts(j2_max)
    n = expected['sector_count']
    odd = expected['odd_half_zero_count']
    return '\n'.join([
        '# M2 low-cutoff 4D SU(2) armillary report', '',
        f"- run ID: `{report['run_id']}`",
        f"- accepted M1 parent: `{report['parent']['parent_run_id']}`",
        '- human decision: M1 accepted for M2 implementation',
        '- status: `NOT_CERTIFIED`',
        f'- scope: local six-leg 4D link-star identity at `j2_max={j2_max}`; no 4D RG claim', '',
        '## Fixed conventions', '',
        '- irrep coordinate: `j2=2j`',
        '- magnetic order: `m2=j2,j2-2,...,-j2`',
        '- CG phase: exact Condon–Shortley convention implemented by SymPy',
        '- fusion tree: left-associated',
        '- orientations: `(+,-,+,-,+,-)`',
        '- duality: `C|j,m> = (-1)^(j-m)|j,-m>`',
        '- normalization: orthonormal CG basis and normalized Haar projector', '',
        '## Exact checks', '',
        f"- dense sectors with exact zero generator residual: {dense['generator_residual_zero_count']}/{n}",
        f"- gauge-noninvariant odd-half sectors exactly zero: {dense['odd_half_zero_count']}/{odd}",
        f"- armillary exact isometries: {armillary['isometry_exact_count']}/{n}",
        f"- dense–armillary exact matrix matches: {equivalence['exact_match_count']}/{n}",
        f"- transverse cubic actions: {symmetry['group_order']}",
        f"- deterministic canonical sectors: {symmetry['canonical_sector_count']}", '',
        'The dense reference is obtained from the exact simultaneous kernel of the total SU(2) generators. '
        'The armillary representation is obtained independently from a fixed exact CG fusion basis. '
        'Acceptance compares the resulting symbolic matrices exactly; float64 tensor shards are restart diagnostics only.', '',
        '## Proven at M2', '',
        f'- At `j2_max={j2_max}`, all {n} representation sectors of the six-leg link star have exact dense/armillary projector equality.',
        '- The phase, orientation, duality, fusion-tree, and normalization conventions are pinned by content hashes.',
        '- Deterministic cache regeneration, symmetry canonicalization, checkpoint fallback, and fresh-process resume pass.', '',
        '## Limitations', '',
        '- No tensor renormalization step is performed.',
        '- No approximate residual is promoted to a rigorous bound.',
        '- No mass gap, thermodynamic limit, continuum limit, or `CERTIFIED` claim is made.',
        '- M3 must not begin until this M2 report is independently reviewed and accepted.', '',
    ])


def write_m2_report_package(
    run_root: Path, config: M2Config, state: RunState, queue: WorkQueue,
    test_report: dict[str, Any], last_checkpoint: CheckpointSaveResult,
    manifest: dict[str, Any],
) -> dict[str, str]:
    results = load_m2_phase_results(run_root, queue, j2_max=config.j2_max)
    gates = validate_m2_acceptance(
        state, queue, results, test_report, j2_max=config.j2_max,
    )
    expected = expected_m2_gate_counts(config.j2_max)
    n = expected['sector_count']
    odd = expected['odd_half_zero_count']
    report = {
        'schema_version': 1, 'milestone': 'M2', 'phase': state.phase,
        'run_id': state.run_id, 'generated_at': utc_now(),
        'certification_status': 'NOT_CERTIFIED',
        'parent': {key: manifest[key] for key in (
            'parent_milestone', 'parent_run_id', 'parent_checkpoint',
            'parent_checkpoint_path', 'parent_checkpoint_hash_manifest_sha256',
            'm1_report_sha256', 'm1_acceptance_sha256', 'm1_audit_sha256',
        )},
        'config': config.canonical_payload(), 'config_hash': config.config_hash(),
        'source_hash': manifest['source_hash'],
        'notebook_hash': manifest['notebook_hash'],
        'convention_hash': manifest['convention_hash'],
        'governing_document_hashes': manifest['governing_document_hashes'],
        'results': results, 'tests': test_report, 'acceptance_gates': gates,
        'proof_artifact_hashes': proof_artifact_hash_map(queue.items.values()),
        'checkpoint': {
            'path': str(last_checkpoint.path), 'index': last_checkpoint.index,
            'size_bytes': last_checkpoint.size_bytes, 'save_s': last_checkpoint.save_s,
            'verify_s': last_checkpoint.verify_s,
        },
        'memory': peak_memory_report(),
        'rigorous_results': [
            f'exact total-generator dense Haar projectors for all {n} sectors',
            f'exact fixed-CG armillary basis maps for all {n} sectors',
            'exact symbolic equality of every dense and reconstructed projector',
            f'exact vanishing of all {odd} odd-half-spin sectors',
        ],
        'heuristic_results': [],
        'unresolved_issues': [
            'Float64 checkpoint tensor shards are diagnostic and have no certification role.',
            'M2 proves a local low-cutoff identity only; no 4D RG error enclosure exists.',
            'Independent milestone review is required before M3.',
        ],
        'remaining_todos': [
            'independent M2 acceptance review', 'M3 GPU Triad-ATRG',
            'M4 forward derivatives', 'M5 one-step validation',
            'M6 multi-step certificate',
        ],
    }
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / 'M2_report.json'
    markdown_path = report_dir / 'M2_report.md'
    acceptance_path = report_dir / 'M2_acceptance.json'
    atomic_write_json(json_path, report)
    atomic_write_text(markdown_path, _markdown_report(report))
    atomic_write_json(acceptance_path, {
        'milestone': 'M2', 'phase': 'M2_COMPLETE', 'status': 'PASS',
        'certification_status': 'NOT_CERTIFIED', 'gates': gates,
        'generated_at': utc_now(),
    })
    return {
        'json': str(json_path), 'markdown': str(markdown_path),
        'acceptance': str(acceptance_path),
    }


def write_m2_session_artifacts(
    run_root: Path, state: RunState, queue: WorkQueue, stop_reason: str,
    elapsed_s: float, remaining_s: float, persistent_root: Path,
    project_root: Path,
) -> dict[str, str]:
    report_dir = run_root / 'reports'; report_dir.mkdir(parents=True, exist_ok=True)
    counts = {
        status: sum(item.status == status for item in queue.items.values())
        for status in ('pending', 'running', 'done', 'failed', 'blocked_resource')
    }
    unfinished = [item.item_id for item in queue.items.values() if item.status != 'done']
    summary_path = report_dir / 'session_summary.json'
    metrics_path = report_dir / 'latest_metrics.json'
    plan_path = report_dir / 'next_session_plan.md'
    atomic_write_json(summary_path, {
        'milestone': 'M2', 'run_id': state.run_id, 'phase': state.phase,
        'certification_status': 'NOT_CERTIFIED', 'stop_reason': stop_reason,
        'elapsed_s': elapsed_s, 'remaining_s': remaining_s, 'queue': counts,
        'unfinished_work_items': unfinished, 'generated_at': utc_now(),
    })
    atomic_write_json(metrics_path, {
        'milestone': 'M2', 'run_id': state.run_id,
        'memory': peak_memory_report(), 'exact_checks': sorted(state.bounds),
        'approximate_residual': None, 'certification_status': 'NOT_CERTIFIED',
        'generated_at': utc_now(),
    })
    if state.phase == 'M2_COMPLETE':
        plan = (
            '# Next session plan\n\nM2 is complete. Independently review '
            '`M2_report.json` and `M2_acceptance.json`. Do not start M3 automatically.\n'
        )
    else:
        plan = (
            '# Next session plan\n\n'
            f'1. Reopen `{project_root}` with a fresh kernel.\n'
            f'2. Set `VALIDATED_RG_PERSIST_ROOT={persistent_root}`.\n'
            '3. Set `VALIDATED_RG_PERSIST_ACK=I_CONFIRM_THIS_PATH_IS_PERSISTENT`.\n'
            f'4. Set `VALIDATED_RG_M2_RUN_ID={state.run_id}`.\n'
            '5. Run `notebooks/30_m2_armillary.ipynb` top-to-bottom. It will verify '
            'hashes, recover interrupted items, and continue from the newest valid checkpoint.\n'
        )
    atomic_write_text(plan_path, plan)
    return {
        'session_summary': str(summary_path),
        'latest_metrics': str(metrics_path), 'next_session_plan': str(plan_path),
    }
