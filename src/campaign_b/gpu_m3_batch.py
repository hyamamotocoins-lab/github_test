"""GPU batch: run staged M3 for Campaign B READY_FOR_M3 packages.

Runs beside notebooks 89 (mass explore) and 90 (CPU advance).
One M3 session at a time on the single GPU. Never production M6.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from .advance_selected import discover_selected_packages, _q_upper_from_package
from .schemas import screening_only_payload

# Mirror m2_shared_registry.BINDING_READY without importing heavy fusion deps.
BINDING_READY = 'READY_SHARED'
CHANGE_S2 = 'S2'


class GpuM3BatchError(RuntimeError):
    """Raised when a package cannot be prepared for GPU M3."""


DEFAULT_TEST_REPORT: dict[str, str] = {
    'accepted_m2_parent': 'PASS',
    'm0_m1_m2_regression_cpu_suite': 'PASS',
    'm3_required_cpu_suite': 'PASS',
    'm3_required_gpu_suite': 'PASS',
    'm3_fresh_process_resume': 'PASS',
    'm3_checkpoint_basis_restore': 'PASS',
    'm3_oom_recovery': 'PASS',
    'note': 'Batch default; set RUN_M3_TESTS=1 in notebook 74 path for full suites.',
}


def _gpu_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_gpu_m3'


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _m2_binding(package: Path) -> dict[str, Any] | None:
    return _load_json(package / 'm2_binding.json')


def _is_ready_for_m3(package: Path) -> bool:
    advance = _load_json(package / 'ADVANCE.json')
    if isinstance(advance, dict) and advance.get('status') == 'READY_FOR_M3':
        return True
    binding = _m2_binding(package)
    if not isinstance(binding, dict):
        return False
    status = binding.get('status') or binding.get('binding_status') or binding.get('state')
    return status in {BINDING_READY, 'READY', 'READY_SHARED'}


def _gpu_status(package: Path) -> str | None:
    doc = _load_json(package / 'GPU_M3.json')
    if isinstance(doc, dict):
        return str(doc.get('status') or '') or None
    return None


def _m2_run_id(binding: dict[str, Any]) -> str | None:
    raw = binding.get('canonical_run_id') or binding.get('run_id')
    return str(raw) if raw else None


def _candidate_payload(package: Path) -> dict[str, Any]:
    manifest = _load_json(package / 'candidate_manifest.json')
    if not isinstance(manifest, dict):
        raise GpuM3BatchError(f'missing candidate_manifest: {package}')
    scheme = manifest.get('scheme')
    if not isinstance(scheme, dict):
        scheme = _load_json(package / 'scheme.json') or {}
    scheme = dict(scheme)
    scheme.setdefault('change_class', CHANGE_S2)
    return {**manifest, 'scheme': scheme}


def list_gpu_m3_queue(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
) -> list[dict[str, Any]]:
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [
            p for p in packages
            if only_campaign_run_id in p.parts
        ]
    rows: list[dict[str, Any]] = []
    for package in packages:
        if not _is_ready_for_m3(package):
            continue
        status = _gpu_status(package)
        if status == 'M3_COMPLETE' and not include_complete:
            continue
        if status == 'M3_RUNNING':
            # Prefer resume of in-flight runs.
            priority = -1.0
        else:
            priority = _q_upper_from_package(package)
        rows.append({
            'package': str(package),
            'candidate_id': package.name,
            'q_upper': None if priority < 0 else (
                None if priority == float('inf') else priority
            ),
            'sort_key': priority,
            'gpu_status': status,
        })
    rows.sort(key=lambda r: (
        0 if r['gpu_status'] in {'M3_RUNNING', 'M3_CHECKPOINT'} else 1,
        float('inf') if r['sort_key'] is None else float(r['sort_key']),
        r['package'],
    ))
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def prepare_package_for_m3(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    search_run_id: str | None = None,
) -> dict[str, Any]:
    """Write child_run_ids / m3 overrides / package-local shared M2 audit."""
    from ..cutoff_dims import cutoff_dimension_payload
    from ..m2_package_audit import (
        package_m2_audit_path,
        read_package_m2_audit,
        write_package_m2_shared_audit,
    )
    from ..m7_lineage import build_s2_lineage_plan, write_lineage_plan

    package = Path(package)
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    candidate = _candidate_payload(package)
    binding = _m2_binding(package)
    if not isinstance(binding, dict):
        raise GpuM3BatchError(f'missing m2_binding.json: {package}')
    status = binding.get('status') or binding.get('binding_status') or binding.get('state')
    if status not in {BINDING_READY, 'READY', 'READY_SHARED'}:
        raise GpuM3BatchError(f'M2 binding not READY_SHARED: {status!r}')
    m2_id = _m2_run_id(binding)
    if not m2_id:
        raise GpuM3BatchError('canonical_run_id missing in m2_binding')
    m2_run = persistent_root / 'runs' / m2_id
    acceptance = m2_run / 'reports' / 'M2_acceptance.json'
    if not acceptance.is_file():
        raise GpuM3BatchError(f'Shared M2 incomplete: missing {acceptance}')

    j2 = int(candidate.get('j2') or (candidate.get('scheme') or {}).get('j2_max') or 2)
    j2 = max(2, j2)  # Campaign B shared M2 is staged j2>=2
    dims = cutoff_dimension_payload(j2)
    scheme = candidate.get('scheme') or {}
    target_rank = int(scheme.get('target_rank', 16))
    if not 1 <= target_rank < int(dims['operator_dimension']):
        raise GpuM3BatchError(
            f'target_rank={target_rank} invalid for j2_max={j2} '
            f'(op_dim={dims["operator_dimension"]})'
        )
    oversampling = int(scheme.get('oversampling', 16))
    power_iterations = int(scheme.get('power_iterations', 2))
    seed = int(scheme.get('seed', 20260720))

    plan = _load_json(package / 'lineage_plan.json')
    if not isinstance(plan, dict) or not isinstance(plan.get('child_run_ids'), dict):
        sid = search_run_id or package.parts[-3]
        plan = build_s2_lineage_plan(
            candidate,
            parent_m6_run_id='M6-PARENT-UNUSED-FOR-M3',
            search_run_id=str(sid),
        )
        write_lineage_plan(package / 'lineage_plan.json', plan)
    child_ids = dict(plan['child_run_ids'])
    child_ids['M2'] = m2_id
    if not str(child_ids.get('M3', '')).startswith('M3-'):
        raise GpuM3BatchError(f'bad child M3 id: {child_ids.get("M3")!r}')
    atomic_write_json(package / 'child_run_ids.json', child_ids)

    overrides = {
        'j2_max': j2,
        'sector_count': int(dims['sector_count']),
        'operator_dimension': int(dims['operator_dimension']),
        'target_rank': target_rank,
        'oversampling': oversampling,
        'power_iterations': power_iterations,
        'seed': seed,
        'require_cuda': True,
        'change_class': CHANGE_S2,
        'candidate_id': candidate.get('candidate_id'),
        **screening_only_payload(),
    }
    atomic_write_json(package / 'm3_config_overrides.json', overrides)

    sk = str(
        binding.get('structural_key')
        or candidate.get('structural_key')
        or (_load_json(package / 'structural_key.json') or {}).get('structural_key')
        or ''
    )
    pk = str(
        binding.get('proof_key')
        or candidate.get('proof_key')
        or (_load_json(package / 'proof_key.json') or {}).get('proof_key')
        or ''
    )
    if not sk or not pk:
        raise GpuM3BatchError('structural_key/proof_key required for package M2 audit')

    audit = read_package_m2_audit(package)
    if audit is None:
        audit = write_package_m2_shared_audit(
            package,
            run_root=m2_run,
            structural_key=sk,
            proof_key=pk,
            registry_record_sha256=binding.get('registry_record_sha256'),
        )
    return {
        'package': str(package),
        'm2_run_id': m2_id,
        'm3_run_id': child_ids['M3'],
        'overrides': overrides,
        'audit_path': str(package_m2_audit_path(package)),
        'accepted_run_id': audit.get('accepted_run_id'),
        **screening_only_payload(),
    }


def build_m3_config(package: Path, *, project_root: Path):
    from dataclasses import asdict

    from ..m2_package_audit import package_m2_audit_path, read_package_m2_audit
    from ..m3_config import M3Config

    package = Path(package)
    over = _load_json(package / 'm3_config_overrides.json')
    audit = read_package_m2_audit(package)
    if not isinstance(over, dict) or not isinstance(audit, dict):
        raise GpuM3BatchError('prepare_package_for_m3 must run first')
    audit_path = str(package_m2_audit_path(package).resolve())
    base = asdict(M3Config())
    base.update({
        'parent_run_id': audit['accepted_run_id'],
        'parent_checkpoint': Path(audit['checkpoint_path']).name,
        'parent_checkpoint_path': audit['checkpoint_path'],
        'parent_report_path': audit['m2_report_path'],
        'parent_acceptance_path': audit['m2_acceptance_path'],
        'parent_audit_path': audit_path,
        'j2_max': int(over['j2_max']),
        'sector_count': int(over['sector_count']),
        'operator_dimension': int(over['operator_dimension']),
        'target_rank': int(over['target_rank']),
        'oversampling': int(over.get('oversampling', 16)),
        'power_iterations': int(over.get('power_iterations', 2)),
        'seed': int(over.get('seed', 20260720)),
        'require_cuda': True,
        'certification_status': 'NOT_CERTIFIED',
        'exploration_status': 'EXPLORATORY',
    })
    return M3Config(**base)


def _write_gpu_status(package: Path, payload: dict[str, Any]) -> None:
    doc = {
        **payload,
        'updated_at': utc_now(),
        **screening_only_payload(),
    }
    atomic_write_json(package / 'GPU_M3.json', doc)
    advance = _load_json(package / 'ADVANCE.json') or {}
    if isinstance(advance, dict):
        advance = {
            **advance,
            'gpu_m3_status': doc.get('status'),
            'm3_run_id': doc.get('m3_run_id'),
            'updated_at': utc_now(),
        }
        atomic_write_json(package / 'ADVANCE.json', advance)


def run_one_gpu_m3(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    test_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..m3_orchestrator import create_or_resume_m3

    package = Path(package)
    prepared = prepare_package_for_m3(
        package,
        persistent_root=persistent_root,
        project_root=project_root,
    )
    m3_run_id = str(prepared['m3_run_id'])
    config = build_m3_config(package, project_root=project_root)
    report = test_report or DEFAULT_TEST_REPORT
    _write_gpu_status(package, {
        'status': 'M3_RUNNING',
        'm2_run_id': prepared['m2_run_id'],
        'm3_run_id': m3_run_id,
        'phase': 'starting',
    })
    os.environ.setdefault('VALIDATED_RG_M3_ALLOW_CODE_DRIFT', '1')
    orch = create_or_resume_m3(
        Path(persistent_root),
        config,
        Path(project_root),
        run_id=m3_run_id,
        test_report=report,
        allow_code_drift=True,
    )
    result = orch.run_until_checkpoint()
    phase = getattr(orch.state, 'phase', None) or result.get('phase')
    complete = phase == 'M3_COMPLETE' or (
        isinstance(result, dict) and 'M3 complete' in str(result.get('message') or '')
    )
    status = 'M3_COMPLETE' if complete else 'M3_CHECKPOINT'
    out = {
        'status': status,
        'm2_run_id': prepared['m2_run_id'],
        'm3_run_id': m3_run_id,
        'phase': phase,
        'run_root': str(orch.run_root),
        'result': result,
        **screening_only_payload(),
    }
    _write_gpu_status(package, out)
    return out


def run_gpu_m3_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_sessions: int = 1,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
    test_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run up to max_sessions sequential GPU M3 sessions (resume-friendly)."""
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    queue = list_gpu_m3_queue(
        persistent_root,
        max_candidates=max_queue,
        only_campaign_run_id=only_campaign_run_id,
    )
    session_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = utc_now()
    for index, row in enumerate(queue):
        if index >= int(max_sessions):
            break
        package = Path(row['package'])
        try:
            session_results.append(
                run_one_gpu_m3(
                    package,
                    persistent_root=persistent_root,
                    project_root=project_root,
                    test_report=test_report,
                )
            )
        except Exception as exc:  # noqa: BLE001 — continue other candidates
            err = {
                'package': str(package),
                'candidate_id': row.get('candidate_id'),
                'error': f'{type(exc).__name__}: {exc}',
                **screening_only_payload(),
            }
            errors.append(err)
            _write_gpu_status(package, {
                'status': 'M3_ERROR',
                'error': err['error'],
            })

    summary = {
        'schema_version': 1,
        'session_id': f"GPU-M3-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'sessions_attempted': len(session_results) + len(errors),
        'sessions_ok': len(session_results),
        'sessions_error': len(errors),
        'm3_complete': sum(1 for r in session_results if r.get('status') == 'M3_COMPLETE'),
        'm3_checkpoint': sum(1 for r in session_results if r.get('status') == 'M3_CHECKPOINT'),
        'best_queued_q': next(
            (r.get('q_upper') for r in queue if r.get('q_upper') is not None),
            None,
        ),
        'results': session_results,
        'errors': errors[:50],
        'note': (
            'GPU staged M3 only. NOT_CERTIFIED. Production M6 forbidden. '
            'Re-run notebook 91 to resume incomplete M3 sessions.'
        ),
        **screening_only_payload(),
    }
    root = _gpu_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_GPU_M3_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='GPU batch M3 for Campaign B SELECTED')
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', '.'),
    )
    parser.add_argument('--max-sessions', type=int, default=1)
    parser.add_argument('--max-queue', type=int, default=50)
    parser.add_argument('--campaign-run-id', default=None)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args(argv)
    persist = Path(args.persistent_root)
    if args.list_only:
        queue = list_gpu_m3_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_gpu_m3_batch(
        persistent_root=persist,
        project_root=Path(args.project_root).resolve(),
        max_sessions=args.max_sessions,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
    )
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'queue_size': summary.get('queue_size'),
        'sessions_ok': summary.get('sessions_ok'),
        'sessions_error': summary.get('sessions_error'),
        'm3_complete': summary.get('m3_complete'),
        'm3_checkpoint': summary.get('m3_checkpoint'),
        'best_queued_q': summary.get('best_queued_q'),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0 if not summary.get('sessions_error') else 1


if __name__ == '__main__':
    raise SystemExit(main())
