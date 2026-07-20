from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .checkpoint import TensorShardStore
from .common import read_json, safe_component, sha256_file
from .m4_status import (
    M4_CLOSED_OBLIGATIONS, M4_DERIVATIVE_ACCEPTED, M4_ENCLOSURE_BLOCKED,
    M4_IMPLEMENTATION_COMPLETE, M5_OPEN_PROOF_OBLIGATIONS,
    MIN_CENTERED_FD_ACCEPTANCE_ORDER,
)
from .source_channels import SOURCE_CLASSES
from .work_queue import WorkQueue


class M5ParentError(RuntimeError):
    """Raised when the accepted M4 derivative parent fails closed."""


@dataclass(frozen=True, slots=True)
class M5ParentEvidence:
    hashes: dict[str, str]
    bound_ledger: dict[str, list[str]]
    regression: dict[str, float]
    tensors: dict[str, np.ndarray]


def _sha(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        character not in '0123456789abcdef' for character in value
    ):
        raise M5ParentError(f'{label} is not a SHA-256 digest.')
    return value


def _finite_nonnegative(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and value >= 0.0
    )


def _all_true(value: object) -> bool:
    return (
        isinstance(value, dict)
        and bool(value)
        and all(item is True for item in value.values())
    )


def _verify_file(path: Path, digest: object, label: str) -> str:
    expected = _sha(digest, f'{label} expected digest')
    if path.is_symlink() or not path.is_file() or sha256_file(path) != expected:
        raise M5ParentError(f'{label} is missing, unsafe, or changed: {path}')
    return expected


def _verify_checkpoint(checkpoint: Path) -> str:
    if checkpoint.is_symlink() or not checkpoint.is_dir():
        raise M5ParentError('Accepted M4 checkpoint is missing or unsafe.')
    if not (checkpoint / 'COMMITTED').is_file():
        raise M5ParentError('Accepted M4 checkpoint is not committed.')
    if any(path.is_symlink() for path in checkpoint.rglob('*')):
        raise M5ParentError('Accepted M4 checkpoint contains a symlink.')
    hashes_path = checkpoint / 'hashes.json'
    expected = read_json(hashes_path) if hashes_path.is_file() else None
    if not isinstance(expected, dict):
        raise M5ParentError('Accepted M4 checkpoint hash manifest is malformed.')
    normalized: dict[str, str] = {}
    for relative, digest in expected.items():
        if not isinstance(relative, str):
            raise M5ParentError('Accepted M4 checkpoint path is not a string.')
        normalized[relative] = _sha(digest, f'checkpoint entry {relative}')
    actual = {
        path.relative_to(checkpoint).as_posix()
        for path in checkpoint.rglob('*')
        if path.is_file() and path.name not in {'hashes.json', 'COMMITTED'}
    }
    if actual != set(normalized):
        raise M5ParentError('Accepted M4 checkpoint file set changed.')
    for relative, digest in normalized.items():
        if sha256_file(checkpoint / relative) != digest:
            raise M5ParentError(f'Accepted M4 checkpoint hash mismatch: {relative}')
    return sha256_file(hashes_path)


def _verify_queue(checkpoint: Path, run_id: str) -> WorkQueue:
    state = read_json(checkpoint / 'state.json')
    if not isinstance(state, dict) or any((
        state.get('run_id') != run_id,
        state.get('phase') != 'M4_COMPLETE',
        state.get('checkpoint_index') != 14,
        state.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M5ParentError('Accepted M4 checkpoint state is invalid.')
    queue = WorkQueue.from_payload(read_json(checkpoint / 'work_queue.json'))
    if len(queue.items) != 6 or any(
        item.status != 'done' for item in queue.items.values()
    ):
        raise M5ParentError('Accepted M4 work queue is not complete.')
    run_root = checkpoint.parents[1]
    for item in queue.items.values():
        if not item.result_relpath or not item.result_sha256:
            raise M5ParentError(f'Accepted M4 item lacks metadata: {item.phase}')
        result = (run_root / item.result_relpath).resolve()
        try:
            result.relative_to(run_root.resolve())
        except ValueError as exc:
            raise M5ParentError('Accepted M4 result escapes its run root.') from exc
        marker_path = run_root / 'work_items' / f'{item.item_id}.done'
        marker = read_json(marker_path) if marker_path.is_file() else None
        if not result.is_file() or sha256_file(result) != item.result_sha256:
            raise M5ParentError(f'Accepted M4 artifact changed: {item.phase}')
        if not isinstance(marker, dict) or any((
            marker.get('item_id') != item.item_id,
            marker.get('result_relpath') != item.result_relpath,
            marker.get('result_sha256') != item.result_sha256,
        )):
            raise M5ParentError(f'Accepted M4 done marker changed: {item.phase}')
    return queue


def _verify_bound_handoff(value: object) -> dict[str, list[str]]:
    expected = {
        'closed_in_M4': list(M4_CLOSED_OBLIGATIONS),
        'open_for_M5': list(M5_OPEN_PROOF_OBLIGATIONS),
    }
    if value != expected:
        raise M5ParentError('M4-to-M5 proof-obligation ledger changed or is incomplete.')
    return expected


def _verify_regression(report: dict[str, Any]) -> dict[str, float]:
    results = report.get('results')
    if not isinstance(results, dict):
        raise M5ParentError('Accepted M4 results are missing.')
    sources = results.get('M4_SOURCE_CHANNELS', {}).get('result', {})
    if not isinstance(sources, dict) or any((
        sources.get('status') != 'PASS',
        sources.get('source_count') != len(SOURCE_CLASSES),
        sources.get('max_symmetry_residual') != 0.0,
        sources.get('zero_source_max_abs') != 0.0,
    )):
        raise M5ParentError('M4 zero-tangent or symmetry evidence failed.')
    expected_channels = {source.value for source in SOURCE_CLASSES}
    if set(sources.get('channels', ())) != expected_channels:
        raise M5ParentError('M4 source channel ordering/set changed.')

    pipeline = results.get('M4_DUAL_PIPELINE', {}).get('result', {})
    backend = pipeline.get('backend') if isinstance(pipeline, dict) else None
    cpu_backend = (
        isinstance(backend, dict)
        and backend.get('is_cuda') is False
        and str(backend.get('device', '')).startswith('cpu')
    )
    tf32_ok = pipeline.get('tf32_disabled') is True or cpu_backend
    if not isinstance(pipeline, dict) or any((
        pipeline.get('status') != 'PASS',
        pipeline.get('contraction_rule') != 'PRODUCT_RULE_COMPLETE',
        pipeline.get('fixed_basis_projection') is not True,
        pipeline.get('basis_variation_policy')
        != 'FIXED_BASIS_WITH_EXPLICIT_LEDGER_TERM',
        not tf32_ok,
    )):
        raise M5ParentError('M4 fixed-basis forward tangent evidence failed.')

    difference = results.get('M4_FINITE_DIFFERENCE', {}).get('result', {})
    channels = difference.get('channels') if isinstance(difference, dict) else None
    if (
        difference.get('status') != 'PASS'
        or difference.get('all_channels_converged') is not True
        or difference.get('finite_difference_is_proof_bound') is not False
        or not isinstance(channels, dict)
        or set(channels) != expected_channels
    ):
        raise M5ParentError('M4 finite-difference regression identity failed.')

    config = report.get('config')
    steps = config.get('finite_difference_steps') if isinstance(config, dict) else None
    tolerance = (
        config.get('finite_difference_relative_tolerance')
        if isinstance(config, dict) else None
    )
    if (
        not isinstance(steps, list)
        or len(steps) < 3
        or not _finite_nonnegative(tolerance)
    ):
        raise M5ParentError('M4 finite-difference policy is malformed.')

    minimum_order = math.inf
    maximum_final_relative = 0.0
    for name in sorted(expected_channels):
        channel = channels[name]
        entries = channel.get('steps') if isinstance(channel, dict) else None
        if (
            channel.get('converged') is not True
            or not isinstance(entries, list)
            or len(entries) != len(steps)
        ):
            raise M5ParentError(f'M4 regression channel failed: {name}')
        actual_steps: list[float] = []
        relative_errors: list[float] = []
        for entry in entries:
            if not isinstance(entry, dict) or not all(
                _finite_nonnegative(entry.get(key))
                for key in (
                    'step', 'absolute_error_frobenius',
                    'relative_error_frobenius',
                )
            ):
                raise M5ParentError(f'M4 regression has nonfinite data: {name}')
            actual_steps.append(float(entry['step']))
            relative_errors.append(float(entry['relative_error_frobenius']))
        if actual_steps != [float(step) for step in steps] or any(
            later >= earlier
            for earlier, later in zip(actual_steps, actual_steps[1:])
        ):
            raise M5ParentError(f'M4 regression step schedule changed: {name}')
        for earlier_h, later_h, earlier_e, later_e in zip(
            actual_steps, actual_steps[1:],
            relative_errors, relative_errors[1:],
        ):
            if later_e >= earlier_e or earlier_e <= 0.0 or later_e <= 0.0:
                raise M5ParentError(f'M4 regression does not converge: {name}')
            order = math.log(later_e / earlier_e) / math.log(later_h / earlier_h)
            if not math.isfinite(order):
                raise M5ParentError(f'M4 regression order is nonfinite: {name}')
            minimum_order = min(minimum_order, order)
        final_relative = relative_errors[-1]
        if final_relative > float(tolerance):
            raise M5ParentError(f'M4 regression tolerance failed: {name}')
        if channel.get('final_relative_error') != final_relative:
            raise M5ParentError(f'M4 final regression residual changed: {name}')
        maximum_final_relative = max(maximum_final_relative, final_relative)

    if minimum_order < MIN_CENTERED_FD_ACCEPTANCE_ORDER:
        raise M5ParentError('M4 centered finite difference lacks second-order convergence.')
    if difference.get('max_final_relative_error') != maximum_final_relative:
        raise M5ParentError('M4 maximum regression residual changed.')

    ledger = results.get('M4_ERROR_LEDGER', {}).get('result', {})
    ledger_summary = ledger.get('summary') if isinstance(ledger, dict) else None
    if (
        ledger.get('status') != 'PASS'
        or ledger.get('basis_variation_accounted') is not True
        or ledger.get('enclosure_status') != M4_ENCLOSURE_BLOCKED
        or not isinstance(ledger_summary, dict)
        or ledger_summary.get('enclosure_ready') is not False
        or ledger_summary.get('double_counting_check') != 'PASS'
    ):
        raise M5ParentError('M4 error DAG or basis-variation handoff failed.')

    return {
        'minimum_observed_centered_fd_order': minimum_order,
        'max_final_relative_error': maximum_final_relative,
        'symmetry_residual': 0.0,
        'zero_tangent_residual': 0.0,
    }


def _load_derivative_tensors(
    checkpoint: Path,
    *,
    projected_rank: int | None = None,
) -> dict[str, np.ndarray]:
    loaded = TensorShardStore(64 * 1024 * 1024).load(checkpoint / 'tensors')
    required = {
        'normalized_primal',
        *(f'normalized_tangent_{source.value}' for source in SOURCE_CLASSES),
    }
    if not required <= set(loaded):
        raise M5ParentError('Accepted M4 derivative checkpoint is incomplete.')
    result: dict[str, np.ndarray] = {}
    inferred_rank: int | None = None
    for name in sorted(required):
        value = np.asarray(loaded[name])
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise M5ParentError(f'Accepted M4 derivative tensor is invalid: {name}')
        if inferred_rank is None:
            inferred_rank = int(value.shape[0])
        if value.shape != (inferred_rank, inferred_rank) or not np.isfinite(value).all():
            raise M5ParentError(f'Accepted M4 derivative tensor is invalid: {name}')
        result[name] = value.copy()
    if projected_rank is not None and inferred_rank != int(projected_rank):
        raise M5ParentError(
            f'M4 derivative rank {inferred_rank} != expected {projected_rank}.'
        )
    return result


def verify_accepted_m4_parent(
    project_root: Path, persistent_root: Path, run_id: str,
) -> M5ParentEvidence:
    """Verify the frozen M4 derivative run before any M5 work may start."""
    try:
        safe_component(run_id)
    except ValueError as exc:
        raise M5ParentError('M4 run ID is unsafe.') from exc
    if not run_id.startswith('M4-'):
        raise M5ParentError('M4 run ID must use the M4 namespace.')

    audit_path = project_root.resolve() / 'audit/m4_accepted_parent.json'
    if audit_path.is_symlink() or not audit_path.is_file():
        raise M5ParentError('M4 derivative acceptance audit is missing or unsafe.')
    audit = read_json(audit_path)
    expected_audit: dict[str, Any] = {
        'schema_version': 1,
        'milestone_reviewed': 'M4',
        'accepted_for_next_milestone': 'M5',
        'accepted_phase': 'M4_COMPLETE',
        'accepted_run_id': run_id,
        'checkpoint_index': 14,
        'implementation_status': M4_IMPLEMENTATION_COMPLETE,
        'milestone_status': M4_DERIVATIVE_ACCEPTED,
        'enclosure_status': M4_ENCLOSURE_BLOCKED,
        'certification_status': 'NOT_CERTIFIED',
        'decision': 'ACCEPT_M4_DERIVATIVE_FOR_M5_ONE_STEP_VALIDATION',
        'independent_artifact_reload_performed': True,
    }
    if not isinstance(audit, dict) or any(
        audit.get(key) != value for key, value in expected_audit.items()
    ):
        raise M5ParentError('M4 acceptance audit identity or decision is invalid.')
    handoff = _verify_bound_handoff(audit.get('bound_ledger'))

    resolved_persistent = persistent_root.resolve()
    if persistent_root.is_symlink() or not resolved_persistent.is_dir():
        raise M5ParentError('M5 persistent root is missing or unsafe.')
    run_root = resolved_persistent / 'runs' / run_id
    if run_root.is_symlink() or not run_root.is_dir():
        raise M5ParentError('Accepted M4 run root is missing or unsafe.')
    try:
        run_root.resolve().relative_to(resolved_persistent)
    except ValueError as exc:
        raise M5ParentError('Accepted M4 run root escapes persistence.') from exc
    report_path = run_root / 'reports/M4_report.json'
    acceptance_path = run_root / 'reports/M4_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    checkpoint = run_root / 'checkpoints/ckpt_000014'
    for key, path in {
        'm4_report_path': report_path,
        'm4_acceptance_path': acceptance_path,
        'manifest_path': manifest_path,
        'checkpoint_path': checkpoint,
    }.items():
        audited = audit.get(key)
        if not isinstance(audited, str) or Path(audited).resolve() != path.resolve():
            raise M5ParentError(f'Accepted M4 path changed: {key}')

    report_hash = _verify_file(
        report_path, audit.get('m4_report_sha256'), 'accepted M4 report',
    )
    acceptance_hash = _verify_file(
        acceptance_path, audit.get('m4_acceptance_sha256'),
        'accepted M4 acceptance',
    )
    manifest_hash = _verify_file(
        manifest_path, audit.get('manifest_sha256'), 'accepted M4 manifest',
    )
    checkpoint_hash = _verify_checkpoint(checkpoint)
    if checkpoint_hash != audit.get('checkpoint_hash_manifest_sha256'):
        raise M5ParentError('Accepted M4 checkpoint hash manifest changed.')
    queue = _verify_queue(checkpoint, run_id)

    report = read_json(report_path)
    if not isinstance(report, dict) or any((
        report.get('milestone') != 'M4',
        report.get('run_id') != run_id,
        report.get('phase') != 'M4_COMPLETE',
        report.get('enclosure_status') != M4_ENCLOSURE_BLOCKED,
        report.get('certification_status') != 'NOT_CERTIFIED',
        not _all_true(report.get('acceptance_gates')),
        report.get('proof_artifact_hashes') != audit.get('proof_artifact_hashes'),
    )):
        raise M5ParentError('Accepted M4 report no longer satisfies every gate.')
    queue_hashes = {
        item.phase: item.result_sha256 for item in queue.items.values()
    }
    if report.get('proof_artifact_hashes') != queue_hashes:
        raise M5ParentError('Accepted M4 report hashes differ from its queue.')
    regression = _verify_regression(report)
    recorded_regression = audit.get('derivative_regression')
    expected_regression_record = {
        'classification': (
            'REPRODUCIBLE_REGRESSION_ACCEPTANCE_NOT_A_DETERMINISTIC_PROOF_BOUND'
        ),
        'all_channels_converged': True,
        'minimum_observed_centered_fd_order': (
            regression['minimum_observed_centered_fd_order']
        ),
        'max_final_relative_error': regression['max_final_relative_error'],
        'zero_tangent_residual': regression['zero_tangent_residual'],
        'symmetry_residual': regression['symmetry_residual'],
        'finite_difference_is_proof_bound': False,
    }
    if not isinstance(recorded_regression, dict) or any(
        recorded_regression.get(key) != value
        for key, value in expected_regression_record.items()
    ):
        raise M5ParentError('M4 derivative regression audit changed or is incomplete.')

    tests = report.get('tests')
    required_tests = {
        'accepted_m3_parent', 'm0_m1_m2_m3_regression_cpu_suite',
        'm4_required_cpu_suite', 'm4_required_gpu_suite',
        'm4_fresh_process_resume', 'm4_derivative_checkpoint_restore',
    }
    if not isinstance(tests, dict) or any(
        tests.get(name) != 'PASS' for name in required_tests
    ):
        raise M5ParentError('Accepted M4 test evidence is incomplete.')

    acceptance = read_json(acceptance_path)
    if not isinstance(acceptance, dict) or any((
        acceptance.get('milestone') != 'M4',
        acceptance.get('phase') != 'M4_COMPLETE',
        acceptance.get('status') != 'PASS',
        acceptance.get('enclosure_status') != M4_ENCLOSURE_BLOCKED,
        acceptance.get('certification_status') != 'NOT_CERTIFIED',
        not _all_true(acceptance.get('gates')),
    )):
        raise M5ParentError('Accepted M4 implementation acceptance changed.')

    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict) or any((
        manifest.get('milestone') != 'M4',
        manifest.get('run_id') != run_id,
        manifest.get('certification_status') != 'NOT_CERTIFIED',
    )):
        raise M5ParentError('Accepted M4 manifest identity changed.')

    tensors = _load_derivative_tensors(checkpoint)
    return M5ParentEvidence(
        hashes={
            'm4_audit_sha256': sha256_file(audit_path),
            'm4_report_sha256': report_hash,
            'm4_acceptance_sha256': acceptance_hash,
            'm4_manifest_sha256': manifest_hash,
            'parent_checkpoint_hash_manifest_sha256': checkpoint_hash,
        },
        bound_ledger=handoff,
        regression=regression,
        tensors=tensors,
    )
