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
    gate = resource_gate(j2_max)
    if j2_max == 1:
        batch = 0
    else:
        if not gate.get('staged_executable'):
            raise M7StagedLineageError(
                'Staged M2 blocked: '
                + '; '.join(gate.get('staged_blocked_reasons') or ['unknown'])
            )
        batch = (
            int(sector_batch_size)
            if sector_batch_size is not None
            else default_sector_batch_size(j2_max, gate)
        )
        if batch < 1:
            raise M7StagedLineageError('Staged M2 requires sector_batch_size>=1.')

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


def run_staged_lineage_from_package(
    package_root: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    rewrite_m2_audit: bool = True,
    sector_batch_size: int | None = None,
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
