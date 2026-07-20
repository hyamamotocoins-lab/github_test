"""Staged j2_max>=2 lineage automation: sector-batched M2 then child audits.

Screening q_cert estimates are never CERTIFIED. This module only automates
resource-gated M2 work that previously required manual notebooks.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from .common import atomic_write_json, read_json, sha256_file, utc_now
from .cutoff_dims import cutoff_dimension_payload, resource_gate
from .m2_batching import proof_artifact_hash_map
from .m2_config import M2Config
from .m2_orchestrator import create_or_resume_m2
from .m7_lineage import effective_projected_rank


class M7StagedLineageError(RuntimeError):
    """Raised when staged lineage automation fails closed."""


def default_sector_batch_size(j2_max: int, gate: dict[str, Any] | None = None) -> int:
    if gate and isinstance(gate.get('default_sector_batch_size'), int):
        return int(gate['default_sector_batch_size'])
    return 0 if j2_max <= 1 else 16


def inspect_staged_m2_progress(
    persistent_root: Path,
    *,
    run_id: str,
) -> dict[str, Any]:
    """Read-only progress for a staged child M2 run (safe across sessions)."""
    run_root = Path(persistent_root) / 'runs' / run_id
    if not run_root.is_dir():
        return {
            'run_id': run_id,
            'exists': False,
            'note': 'Run directory not found yet; first session creates it.',
        }
    config_path = run_root / 'run_config.json'
    config_payload = read_json(config_path) if config_path.is_file() else None
    ckpt_root = run_root / 'checkpoints'
    committed = sorted(
        path for path in ckpt_root.glob('ckpt_*')
        if (path / 'COMMITTED').is_file()
    )
    if not committed:
        return {
            'run_id': run_id,
            'exists': True,
            'committed_checkpoints': 0,
            'j2_max': (
                config_payload.get('j2_max')
                if isinstance(config_payload, dict) else None
            ),
            'note': 'No committed checkpoint yet.',
        }
    latest = committed[-1]
    state = read_json(latest / 'state.json')
    queue_payload = read_json(latest / 'work_queue.json')
    counts = {
        'pending': 0, 'running': 0, 'done': 0, 'failed': 0, 'blocked_resource': 0,
    }
    phase_pending: dict[str, int] = {}
    if isinstance(queue_payload, dict):
        items = queue_payload.get('items') or {}
        if isinstance(items, dict):
            for item in items.values():
                if not isinstance(item, dict):
                    continue
                status = str(item.get('status') or 'pending')
                counts[status] = counts.get(status, 0) + 1
                if status == 'pending':
                    phase = str(item.get('phase') or '?')
                    phase_pending[phase] = phase_pending.get(phase, 0) + 1
    acceptance = run_root / 'reports' / 'M2_acceptance.json'
    done = counts.get('done', 0)
    total = sum(counts.values())
    return {
        'run_id': run_id,
        'exists': True,
        'run_root': str(run_root),
        'latest_checkpoint': str(latest),
        'checkpoint_index': (
            state.get('checkpoint_index') if isinstance(state, dict) else None
        ),
        'phase': state.get('phase') if isinstance(state, dict) else None,
        'certification_status': (
            state.get('certification_status') if isinstance(state, dict) else None
        ),
        'queue_counts': counts,
        'pending_by_phase': phase_pending,
        'fraction_done': (done / total) if total else 0.0,
        'm2_complete': acceptance.is_file(),
        'j2_max': (
            config_payload.get('j2_max') if isinstance(config_payload, dict) else None
        ),
        'sector_batch_size': (
            config_payload.get('sector_batch_size')
            if isinstance(config_payload, dict) else None
        ),
        'total_items': total,
    }


def run_staged_m2_session(
    *,
    persistent_root: Path,
    project_root: Path,
    j2_max: int,
    run_id: str | None = None,
    sector_batch_size: int | None = None,
    test_report: dict[str, Any] | None = None,
    m2_config_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Create/resume sector-batched M2 and run one Paperspace-safe session."""
    import faulthandler
    import os
    import sys

    faulthandler.enable(file=sys.stderr, all_threads=True)
    os.environ.setdefault('VALIDATED_RG_CHECKPOINT_KEEP', '5')
    # High-dim tail sectors (dim hundreds) must run one-at-a-time.
    os.environ.setdefault('VALIDATED_RG_M2_SPLIT_BATCH_TO', '1')

    gate = resource_gate(j2_max)
    if j2_max == 1:
        batch = 0
    else:
        if not gate.get('staged_executable'):
            raise M7StagedLineageError(
                'Staged M2 blocked: '
                + '; '.join(gate.get('staged_blocked_reasons') or ['unknown'])
            )
        env_batch = os.environ.get('VALIDATED_RG_M2_SECTOR_BATCH_SIZE', '').strip()
        if sector_batch_size is not None:
            batch = int(sector_batch_size)
        elif env_batch:
            batch = int(env_batch)
        else:
            # Small default: large batches are often killed mid-SymPy with no traceback.
            batch = 2
        if batch < 1:
            raise M7StagedLineageError('Staged M2 requires sector_batch_size>=1.')

    # Resume must reuse the frozen run_config.json payload (config_hash pin).
    if run_id:
        existing_config = Path(persistent_root) / 'runs' / run_id / 'run_config.json'
        if existing_config.is_file():
            payload = read_json(existing_config)
            if not isinstance(payload, dict):
                raise M7StagedLineageError('Existing M2 run_config.json malformed.')
            if payload.get('proof_schema') != 'M2_PROOF_SCHEMA_V2':
                raise M7StagedLineageError(
                    f'Existing M2 run {run_id} uses proof schema '
                    f"{payload.get('proof_schema')!r}; "
                    'allocate a new child M2 run ID for M2_PROOF_SCHEMA_V2 '
                    '(invariant_subspace_uniqueness_v1).'
                )
            if 'orientations' in payload and isinstance(payload['orientations'], list):
                payload = {
                    **payload,
                    'orientations': tuple(payload['orientations']),
                }
            config = M2Config(**payload)
        else:
            base = asdict(M2Config())
            if m2_config_overrides:
                base.update(m2_config_overrides)
            base['j2_max'] = int(j2_max)
            base['sector_batch_size'] = int(batch)
            config = M2Config(**base)
    else:
        base = asdict(M2Config())
        if m2_config_overrides:
            base.update(m2_config_overrides)
        base['j2_max'] = int(j2_max)
        base['sector_batch_size'] = int(batch)
        config = M2Config(**base)

    orch = create_or_resume_m2(
        persistent_root,
        config,
        project_root,
        run_id=run_id,
        test_report=test_report,
        allow_code_drift=True,
    )
    summary = orch.run_until_checkpoint()
    summary['resource_gate'] = gate
    summary['sector_batch_size'] = config.sector_batch_size
    summary['cutoff_dims'] = cutoff_dimension_payload(j2_max)
    summary['m2_complete'] = (
        orch.state.phase == 'M2_COMPLETE'
        and (orch.run_root / 'reports' / 'M2_acceptance.json').is_file()
    )
    summary['run_root'] = str(orch.run_root)
    summary['progress'] = inspect_staged_m2_progress(
        persistent_root, run_id=orch.state.run_id,
    )
    return summary


def write_child_m2_acceptance_audit(
    project_root: Path,
    *,
    run_root: Path,
    audit_relative: str = 'audit/m2_accepted_parent.json',
) -> dict[str, Any]:
    """Rewrite M2→M3 parent audit from a completed child M2 run.

    Does not claim continuum results. Operator must still independently review
    before treating the child lineage as production-accepted.
    """
    run_root = run_root.resolve()
    report_path = run_root / 'reports' / 'M2_report.json'
    acceptance_path = run_root / 'reports' / 'M2_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    if not report_path.is_file() or not acceptance_path.is_file():
        raise M7StagedLineageError('Child M2 report/acceptance missing.')
    report = read_json(report_path)
    acceptance = read_json(acceptance_path)
    manifest = read_json(manifest_path)
    if not all(isinstance(doc, dict) for doc in (report, acceptance, manifest)):
        raise M7StagedLineageError('Child M2 artifacts malformed.')
    if report.get('phase') != 'M2_COMPLETE' or acceptance.get('status') != 'PASS':
        raise M7StagedLineageError('Child M2 is not acceptance-complete.')

    checkpoint_meta = report.get('checkpoint') or {}
    checkpoint_path = Path(str(checkpoint_meta.get('path', '')))
    if not checkpoint_path.is_dir():
        # Fall back to latest committed checkpoint under the run.
        ckpt_root = run_root / 'checkpoints'
        candidates = sorted(
            path for path in ckpt_root.glob('ckpt_*')
            if (path / 'COMMITTED').is_file()
        )
        if not candidates:
            raise M7StagedLineageError('No committed M2 checkpoint for audit rewrite.')
        checkpoint_path = candidates[-1]

    hashes_path = checkpoint_path / 'hashes.json'
    queue = read_json(checkpoint_path / 'work_queue.json')
    state = read_json(checkpoint_path / 'state.json')
    if not isinstance(queue, dict) or not isinstance(state, dict):
        raise M7StagedLineageError('Checkpoint queue/state malformed.')

    from .work_queue import WorkQueue
    work = WorkQueue.from_payload(queue)
    audit = {
        'milestone_reviewed': 'M2',
        'accepted_for_next_milestone': 'M3',
        'accepted_phase': 'M2_COMPLETE',
        'accepted_run_id': report.get('run_id'),
        'checkpoint_index': int(state.get('checkpoint_index', checkpoint_meta.get('index', 0))),
        'decision': 'ACCEPT_M2_FOR_M3_EXPLORATORY_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm2_report_path': str(report_path),
        'm2_acceptance_path': str(acceptance_path),
        'checkpoint_path': str(checkpoint_path),
        'manifest_path': str(manifest_path),
        'm2_report_sha256': sha256_file(report_path),
        'm2_acceptance_sha256': sha256_file(acceptance_path),
        'manifest_sha256': sha256_file(manifest_path),
        'checkpoint_hash_manifest_sha256': sha256_file(hashes_path),
        'proof_artifact_hashes': proof_artifact_hash_map(work.items.values()),
        'staged_child_lineage': True,
        'generated_at': utc_now(),
        'scope_limitation': (
            'Child M2 acceptance rewrite for staged j2_max>=2 lineage only; '
            'not a continuum or mass-gap claim.'
        ),
    }
    out = project_root / audit_relative
    atomic_write_json(out, audit)
    return audit


def write_child_m3_acceptance_audit(
    project_root: Path,
    *,
    run_root: Path,
    audit_relative: str = 'audit/m3_accepted_parent.json',
) -> dict[str, Any]:
    """Rewrite M3→M4 parent audit from a completed child M3 run.

    Does not claim continuum results. Operator must still independently review
    before treating the child lineage as production-accepted.
    """
    run_root = run_root.resolve()
    report_path = run_root / 'reports' / 'M3_report.json'
    acceptance_path = run_root / 'reports' / 'M3_acceptance.json'
    manifest_path = run_root / 'run_manifest.json'
    if not report_path.is_file() or not acceptance_path.is_file():
        raise M7StagedLineageError('Child M3 report/acceptance missing.')
    report = read_json(report_path)
    acceptance = read_json(acceptance_path)
    manifest = read_json(manifest_path)
    if not all(isinstance(doc, dict) for doc in (report, acceptance, manifest)):
        raise M7StagedLineageError('Child M3 artifacts malformed.')
    if (
        report.get('phase') != 'M3_COMPLETE'
        or report.get('milestone_status') != 'CORE_REPRODUCED'
        or acceptance.get('status') != 'PASS'
    ):
        raise M7StagedLineageError('Child M3 is not acceptance-complete.')

    checkpoint_meta = report.get('checkpoint') or {}
    raw_ckpt = checkpoint_meta.get('path')
    checkpoint_path = (
        Path(str(raw_ckpt)).resolve()
        if isinstance(raw_ckpt, str) and raw_ckpt.strip()
        else None
    )
    if checkpoint_path is None or not checkpoint_path.is_dir():
        ckpt_root = run_root / 'checkpoints'
        candidates = sorted(
            path for path in ckpt_root.glob('ckpt_*')
            if (path / 'COMMITTED').is_file()
        )
        if not candidates:
            raise M7StagedLineageError('No committed M3 checkpoint for audit rewrite.')
        checkpoint_path = candidates[-1]
    try:
        checkpoint_path.relative_to(run_root)
    except ValueError as exc:
        raise M7StagedLineageError(
            'Child M3 checkpoint escapes its run root.'
        ) from exc

    hashes_path = checkpoint_path / 'hashes.json'
    queue = read_json(checkpoint_path / 'work_queue.json')
    state = read_json(checkpoint_path / 'state.json')
    if not isinstance(queue, dict) or not isinstance(state, dict):
        raise M7StagedLineageError('Checkpoint queue/state malformed.')

    from .work_queue import WorkQueue
    work = WorkQueue.from_payload(queue)
    audit = {
        'schema_version': 1,
        'milestone_reviewed': 'M3',
        'accepted_for_next_milestone': 'M4',
        'accepted_phase': 'M3_COMPLETE',
        'accepted_run_id': report.get('run_id'),
        'checkpoint_index': int(
            state.get('checkpoint_index', checkpoint_meta.get('index', 0))
        ),
        'decision': 'ACCEPT_M3_FOR_M4_FORWARD_DERIVATIVE_IMPLEMENTATION',
        'certification_status': 'NOT_CERTIFIED',
        'independent_artifact_reload_performed': True,
        'm3_report_path': str(report_path),
        'm3_acceptance_path': str(acceptance_path),
        'checkpoint_path': str(checkpoint_path),
        'manifest_path': str(manifest_path),
        'm3_report_sha256': sha256_file(report_path),
        'm3_acceptance_sha256': sha256_file(acceptance_path),
        'manifest_sha256': sha256_file(manifest_path),
        'checkpoint_hash_manifest_sha256': sha256_file(hashes_path),
        'proof_artifact_hashes': {
            item.phase: item.result_sha256
            for item in work.items.values()
            if item.status == 'done' and item.result_sha256
        },
        'staged_child_lineage': True,
        'generated_at': utc_now(),
        'scope_limitation': (
            'Child M3 acceptance rewrite for staged j2_max>=2 lineage only; '
            'exploratory CORE_REPRODUCED pilot; not a continuum or mass-gap claim.'
        ),
    }
    out = project_root / audit_relative
    atomic_write_json(out, audit)
    return audit


def run_staged_lineage_from_package(
    package_root: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    rewrite_m2_audit: bool = True,
    sector_batch_size: int | None = None,
    test_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Drive one staged live session from a Campaign C package."""
    root = package_root.resolve()
    scheme = read_json(root / 'scheme.json')
    gate = read_json(root / 'resource_gate.json')
    child_ids = read_json(root / 'child_run_ids.json')
    if not all(isinstance(doc, dict) for doc in (scheme, gate, child_ids)):
        raise M7StagedLineageError('Lineage package incomplete.')
    j2_max = int(scheme.get('j2_max', 1))
    if j2_max <= 1:
        raise M7StagedLineageError('Staged path is for j2_max>=2; use instant live for j2=1.')
    if not gate.get('staged_executable'):
        raise M7StagedLineageError(
            'Package is not staged_executable: '
            + '; '.join(gate.get('staged_blocked_reasons') or [])
        )
    m2_id = child_ids.get('M2')
    if not isinstance(m2_id, str) or not m2_id.startswith('M2-'):
        raise M7StagedLineageError('Package child_run_ids.M2 is invalid.')

    summary = run_staged_m2_session(
        persistent_root=persistent_root,
        project_root=project_root,
        j2_max=j2_max,
        run_id=m2_id,
        sector_batch_size=sector_batch_size,
        test_report=test_report,
    )
    audit = None
    if summary.get('m2_complete') and rewrite_m2_audit:
        audit = write_child_m2_acceptance_audit(
            project_root, run_root=Path(summary['run_root']),
        )
        # Materialize M3 overrides already present; stamp next-step hint.
        dims = cutoff_dimension_payload(j2_max)
        next_steps = {
            'status': 'M2_COMPLETE_AUDIT_REWRITTEN',
            'next': [
                'create_or_resume_m3 with m3_config_overrides.json',
                'ACCEPT M3 → rewrite audit/m3_accepted_parent.json',
                'continue M4→M6 with child_run_ids',
            ],
            'cutoff_dims': dims,
            'effective_projected_rank': effective_projected_rank(
                min(16, dims['operator_dimension'] - 1),
            ),
            'generated_at': utc_now(),
        }
        atomic_write_json(root / 'staged_progress.json', {
            'm2_session': summary,
            'm2_audit': {
                'accepted_run_id': audit.get('accepted_run_id'),
                'checkpoint_index': audit.get('checkpoint_index'),
            },
            'next_steps': next_steps,
        })
    else:
        atomic_write_json(root / 'staged_progress.json', {
            'm2_session': summary,
            'm2_complete': bool(summary.get('m2_complete')),
            'note': (
                'Resume with the same execute_lineage.py --live --staged '
                'until M2_COMPLETE.'
            ),
            'generated_at': utc_now(),
        })
    return {
        'status': (
            'M2_COMPLETE_READY_FOR_M3'
            if summary.get('m2_complete') else 'M2_SESSION_CHECKPOINT'
        ),
        'm2_session': summary,
        'audit_rewritten': audit is not None,
        'package_root': str(root),
    }
