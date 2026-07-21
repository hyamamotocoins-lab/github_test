"""Advance Campaign B packages from M3_COMPLETE through M4+M5 (stop before M6).

Uses the same staged path as notebooks 75/76:
  write_child_m3_acceptance_audit → create_or_resume_m4
  write_child_m4_acceptance_audit → create_or_resume_m5(mode=staged_child)

Never starts production M6. Always NOT_CERTIFIED / SCREENING_ONLY.
Global audits (audit/m3_accepted_parent.json, audit/m4_accepted_parent.json)
are rewritten per candidate — run only one package at a time on the GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from .advance_selected import discover_selected_packages, _q_upper_from_package
from .schemas import screening_only_payload


class PreM6BatchError(RuntimeError):
    """Raised when pre-M6 advancement cannot proceed safely."""


DEFAULT_M4_TEST_REPORT: dict[str, str] = {
    # Keys must match m4_acceptance_gates() / notebook 75 exactly.
    'accepted_m3_parent': 'PASS',
    'm0_m1_m2_m3_regression_cpu_suite': 'PASS',
    'm4_required_cpu_suite': 'PASS',
    'm4_required_gpu_suite': 'PASS',
    'm4_fresh_process_resume': 'PASS',
    'm4_derivative_checkpoint_restore': 'PASS',
    'note': 'Batch default aligned with notebook 75; full pytest optional.',
}


def _load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _ledger_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_pre_m6'


def _pre_m6_status(package: Path) -> str | None:
    doc = _load(package / 'PRE_M6.json')
    if isinstance(doc, dict) and doc.get('status'):
        return str(doc['status'])
    return None


def _gpu_m3_status(package: Path) -> str | None:
    doc = _load(package / 'GPU_M3.json')
    if isinstance(doc, dict) and doc.get('status'):
        return str(doc['status'])
    return None


def _child_ids(package: Path) -> dict[str, Any] | None:
    return _load(package / 'child_run_ids.json')


def _m3_complete_on_disk(persistent_root: Path, m3_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m3_run_id
    report = _load(root / 'reports' / 'M3_report.json')
    acceptance = _load(root / 'reports' / 'M3_acceptance.json')
    if not isinstance(report, dict) or not isinstance(acceptance, dict):
        return False
    return (
        report.get('phase') == 'M3_COMPLETE'
        and report.get('milestone_status') == 'CORE_REPRODUCED'
        and acceptance.get('status') == 'PASS'
    )


def _m4_complete_on_disk(persistent_root: Path, m4_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m4_run_id
    report = _load(root / 'reports' / 'M4_report.json')
    acceptance = _load(root / 'reports' / 'M4_acceptance.json')
    if not isinstance(report, dict) or not isinstance(acceptance, dict):
        return False
    return report.get('phase') == 'M4_COMPLETE' and acceptance.get('status') == 'PASS'


def _m5_done_on_disk(persistent_root: Path, m5_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m5_run_id
    if (root / 'reports' / 'M5_obligation_report.json').is_file():
        return True
    report = _load(root / 'reports' / 'M5_report.json')
    if not isinstance(report, dict):
        return False
    phase = str(report.get('phase') or '')
    return phase.startswith('M5_') and phase not in {'M5_IN_PROGRESS', 'M5_FAILED'}


def list_pre_m6_queue(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
) -> list[dict[str, Any]]:
    """Packages with completed M3 that still need M4/M5 (stop before M6)."""
    persistent_root = Path(persistent_root)
    rows: list[dict[str, Any]] = []
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        status = _pre_m6_status(package)
        if status == 'PRE_M6_READY' and not include_complete:
            continue
        child = _child_ids(package)
        if not isinstance(child, dict):
            continue
        m3_id = child.get('M3')
        if not isinstance(m3_id, str) or not m3_id.startswith('M3-'):
            continue
        gpu = _gpu_m3_status(package)
        if gpu != 'M3_COMPLETE' and not _m3_complete_on_disk(persistent_root, m3_id):
            continue
        q = _q_upper_from_package(package)
        stage = 'NEED_M4'
        if isinstance(child.get('M4'), str) and _m4_complete_on_disk(
            persistent_root, str(child['M4']),
        ):
            stage = 'NEED_M5'
        if isinstance(child.get('M5'), str) and _m5_done_on_disk(
            persistent_root, str(child['M5']),
        ):
            stage = 'PRE_M6_READY'
            if not include_complete:
                continue
        rows.append({
            'package': str(package),
            'candidate_id': package.name,
            'q_upper': None if q == float('inf') else q,
            'stage': stage,
            'm3_run_id': m3_id,
            'm4_run_id': child.get('M4'),
            'm5_run_id': child.get('M5'),
            'pre_m6_status': status,
        })
    rows.sort(key=lambda r: (
        0 if r['stage'] == 'NEED_M5' else 1,  # finish M5 before new M4
        float('inf') if r['q_upper'] is None else float(r['q_upper']),
        r['package'],
    ))
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def _write_pre_m6(package: Path, payload: dict[str, Any]) -> None:
    doc = {**payload, 'updated_at': utc_now(), **screening_only_payload()}
    atomic_write_json(package / 'PRE_M6.json', doc)
    advance = _load(package / 'ADVANCE.json') or {}
    if isinstance(advance, dict):
        advance = {
            **advance,
            'pre_m6_status': doc.get('status'),
            'm4_run_id': doc.get('m4_run_id'),
            'm5_run_id': doc.get('m5_run_id'),
            'updated_at': utc_now(),
        }
        atomic_write_json(package / 'ADVANCE.json', advance)


def prepare_m4_overrides(
    package: Path,
    *,
    persistent_root: Path,
) -> dict[str, Any]:
    from ..m7_lineage import effective_projected_rank

    package = Path(package)
    child = _child_ids(package)
    if not isinstance(child, dict):
        raise PreM6BatchError(f'missing child_run_ids.json: {package}')
    m3_id = str(child.get('M3') or '')
    if not m3_id.startswith('M3-'):
        raise PreM6BatchError(f'bad M3 id: {m3_id!r}')
    if not _m3_complete_on_disk(persistent_root, m3_id):
        raise PreM6BatchError(f'M3 not complete on disk: {m3_id}')

    m3_report = _load(Path(persistent_root) / 'runs' / m3_id / 'reports' / 'M3_report.json')
    assert isinstance(m3_report, dict)
    rsvd = (m3_report.get('results') or {}).get('M3_RSVD', {}).get('result') or {}
    m3_target_rank = int(rsvd.get('target_rank') or 0)
    if m3_target_rank < 1:
        # Fall back to package m3 overrides / scheme.
        over3 = _load(package / 'm3_config_overrides.json') or {}
        scheme = _load(package / 'scheme.json') or {}
        m3_target_rank = int(
            over3.get('target_rank')
            or scheme.get('target_rank')
            or 16
        )
    projected = effective_projected_rank(m3_target_rank)
    op_dim = int(
        (m3_report.get('config') or {}).get('operator_dimension')
        or rsvd.get('dimension')
        or (_load(package / 'm3_config_overrides.json') or {}).get('operator_dimension')
        or 0
    )
    if op_dim < 1:
        raise PreM6BatchError(f'operator_dimension missing for {m3_id}')
    if not 1 <= projected < op_dim:
        raise PreM6BatchError(
            f'projected_rank={projected} invalid for operator_dimension={op_dim}'
        )
    overrides = {
        'operator_dimension': op_dim,
        'projected_rank': projected,
        'm3_target_rank': m3_target_rank,
        'require_cuda': True,
        **screening_only_payload(),
    }
    atomic_write_json(package / 'm4_config_overrides.json', overrides)
    m4_id = str(child.get('M4') or '')
    if not m4_id.startswith('M4-'):
        raise PreM6BatchError(f'bad M4 id in child_run_ids: {m4_id!r}')
    return {
        'm3_run_id': m3_id,
        'm4_run_id': m4_id,
        'overrides': overrides,
    }


def build_m4_config(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
):
    from ..m4_config import M4Config
    from ..m7_staged_lineage import write_child_m3_acceptance_audit

    prepared = prepare_m4_overrides(package, persistent_root=persistent_root)
    m3_id = prepared['m3_run_id']
    over = prepared['overrides']
    audit = write_child_m3_acceptance_audit(
        Path(project_root),
        run_root=Path(persistent_root) / 'runs' / m3_id,
    )
    if audit.get('accepted_run_id') != m3_id:
        raise PreM6BatchError('M3 audit rewrite failed to pin child run')
    base = asdict(M4Config())
    base.update({
        'parent_run_id': audit['accepted_run_id'],
        'parent_checkpoint': Path(audit['checkpoint_path']).name,
        'parent_checkpoint_path': audit['checkpoint_path'],
        'parent_report_path': audit['m3_report_path'],
        'parent_acceptance_path': audit['m3_acceptance_path'],
        'parent_audit_path': 'audit/m3_accepted_parent.json',
        'operator_dimension': int(over['operator_dimension']),
        'projected_rank': int(over['projected_rank']),
        'require_cuda': True,
        'milestone_status': 'BLOCKED_MATH',
        'certification_status': 'NOT_CERTIFIED',
    })
    return M4Config(**base), prepared


def run_m4_session(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    test_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from ..m4_orchestrator import create_or_resume_m4

    config, prepared = build_m4_config(
        package, persistent_root=persistent_root, project_root=project_root,
    )
    m4_id = prepared['m4_run_id']
    _write_pre_m6(package, {
        'status': 'M4_RUNNING',
        'm3_run_id': prepared['m3_run_id'],
        'm4_run_id': m4_id,
        'stage': 'M4',
    })
    os.environ.setdefault('VALIDATED_RG_M4_ALLOW_CODE_DRIFT', '1')
    orch = create_or_resume_m4(
        Path(persistent_root),
        config,
        Path(project_root),
        run_id=m4_id,
        test_report=test_report or DEFAULT_M4_TEST_REPORT,
        allow_code_drift=True,
    )
    result = orch.run_until_checkpoint()
    complete = _m4_complete_on_disk(persistent_root, m4_id) or (
        getattr(orch.state, 'phase', None) == 'M4_COMPLETE'
    )
    status = 'M4_COMPLETE' if complete else 'M4_CHECKPOINT'
    m3_strip: dict[str, Any] | None = None
    if complete:
        # M4 no longer needs the M3 parent ckpt for resume; free ~hundreds of MiB now.
        from .m3_reclaim import strip_m3_after_m4_complete

        m3_strip = strip_m3_after_m4_complete(
            persistent_root,
            str(prepared['m3_run_id']),
            execute=True,
        )
    out = {
        'status': status,
        'm3_run_id': prepared['m3_run_id'],
        'm4_run_id': m4_id,
        'stage': 'M4',
        'phase': getattr(orch.state, 'phase', None),
        'run_root': str(orch.run_root),
        'result': result,
        'm3_reclaim_after_m4': m3_strip,
        **screening_only_payload(),
    }
    _write_pre_m6(package, out)
    return out


def run_m5_session(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    from ..m5_config import default_m5_config
    from ..m5_orchestrator import create_or_resume_m5
    from ..m5_parent import verify_accepted_m4_parent
    from ..m7_staged_lineage import write_child_m4_acceptance_audit

    package = Path(package)
    child = _child_ids(package)
    if not isinstance(child, dict):
        raise PreM6BatchError('missing child_run_ids.json')
    m4_id = str(child.get('M4') or '')
    m5_id = str(child.get('M5') or '')
    if not m4_id.startswith('M4-') or not m5_id.startswith('M5-'):
        raise PreM6BatchError(f'bad M4/M5 ids: {m4_id!r} {m5_id!r}')
    if not _m4_complete_on_disk(persistent_root, m4_id):
        raise PreM6BatchError(f'M4 not complete: {m4_id}')

    audit = write_child_m4_acceptance_audit(
        Path(project_root),
        run_root=Path(persistent_root) / 'runs' / m4_id,
    )
    if audit.get('accepted_run_id') != m4_id:
        raise PreM6BatchError('M4 audit rewrite failed to pin child run')

    m4_over = _load(package / 'm4_config_overrides.json') or {}
    projected = int(m4_over.get('projected_rank') or 16)
    m3_over = _load(package / 'm3_config_overrides.json') or {}
    scheme = _load(package / 'scheme.json') or {}
    j2_max = int(
        m3_over.get('j2_max')
        or m3_over.get('parent_m2_j2_max')
        or scheme.get('j2_max')
        or scheme.get('j2')
        or 2
    )
    # Prefer parent M3 run_config if present.
    m3_id = str(child.get('M3') or '')
    m3_cfg = _load(Path(persistent_root) / 'runs' / m3_id / 'run_config.json') or {}
    if m3_cfg.get('j2_max') is not None:
        j2_max = int(m3_cfg['j2_max'])

    m5_config = default_m5_config(
        parent_m4_run_id=m4_id,
        run_id=m5_id,
        cutoff=j2_max,
        bond_dimension=projected,
        mode='staged_child',
    )
    verify_accepted_m4_parent(Path(project_root), Path(persistent_root), m4_id)
    _write_pre_m6(package, {
        'status': 'M5_RUNNING',
        'm4_run_id': m4_id,
        'm5_run_id': m5_id,
        'stage': 'M5',
    })
    orch = create_or_resume_m5(
        Path(persistent_root),
        m5_config,
        Path(project_root),
        run_id=m5_id,
    )
    result = orch.run_until_checkpoint()
    # Stop before M6 regardless of ONE_STEP_CERTIFIED (unexpected for staged).
    cert = result.get('certification_status') if isinstance(result, dict) else None
    if cert is None:
        cert = getattr(getattr(orch, 'config', None), 'certification_status', None)
    out = {
        'status': 'PRE_M6_READY',
        'm4_run_id': m4_id,
        'm5_run_id': m5_id,
        'stage': 'PRE_M6',
        'm5_result': result,
        'certification_status_m5': cert,
        'prohibited_next': [
            'production M6',
            'notebook 81 production gate',
            'CERTIFIED claim',
        ],
        'allowed_next': [
            'Inspect M5_obligation_report.json',
            'Human review only; production M6 remains forbidden until ONE_STEP_CERTIFIED',
        ],
        **screening_only_payload(),
    }
    # Force screening vocabulary even if M5 unexpectedly flipped.
    out['certification_status'] = 'NOT_CERTIFIED'
    out['claim_scope'] = 'SCREENING_ONLY'
    _write_pre_m6(package, out)
    # Explicit marker that M6 must not auto-start.
    atomic_write_json(package / 'M6_GATE.json', {
        'status': 'BLOCKED_PRE_M6',
        'reason': 'Campaign B pre-M6 batch stops after M5; production M6 forbidden',
        'm5_run_id': m5_id,
        'm4_run_id': m4_id,
        **screening_only_payload(),
        'updated_at': utc_now(),
    })
    return out


def advance_one_toward_pre_m6(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    max_stage_sessions: int = 4,
) -> dict[str, Any]:
    """Run M4 (resume loop) then M5; never M6."""
    package = Path(package)
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    child = _child_ids(package)
    if not isinstance(child, dict):
        raise PreM6BatchError('missing child_run_ids.json')

    sessions: list[dict[str, Any]] = []
    m4_id = str(child.get('M4') or '')
    if not _m4_complete_on_disk(persistent_root, m4_id):
        for _ in range(int(max_stage_sessions)):
            if _m4_complete_on_disk(persistent_root, m4_id):
                break
            sessions.append(
                run_m4_session(
                    package,
                    persistent_root=persistent_root,
                    project_root=project_root,
                )
            )
            if sessions[-1].get('status') == 'M4_COMPLETE':
                break
        if not _m4_complete_on_disk(persistent_root, m4_id):
            return {
                'package': str(package),
                'status': 'M4_CHECKPOINT',
                'sessions': sessions,
                'note': 'Re-run notebook 92 to resume M4',
                **screening_only_payload(),
            }

    m5_id = str(child.get('M5') or '')
    if _m5_done_on_disk(persistent_root, m5_id) or _pre_m6_status(package) == 'PRE_M6_READY':
        # Ensure gate marker exists.
        if not (package / 'M6_GATE.json').is_file():
            atomic_write_json(package / 'M6_GATE.json', {
                'status': 'BLOCKED_PRE_M6',
                'reason': 'M5 already present; production M6 forbidden',
                'm5_run_id': m5_id,
                'm4_run_id': m4_id,
                **screening_only_payload(),
                'updated_at': utc_now(),
            })
        return {
            'package': str(package),
            'status': 'PRE_M6_READY',
            'sessions': sessions,
            **screening_only_payload(),
        }

    m5 = run_m5_session(
        package,
        persistent_root=persistent_root,
        project_root=project_root,
    )
    sessions.append(m5)
    return {
        'package': str(package),
        'status': m5.get('status'),
        'sessions': sessions,
        **screening_only_payload(),
    }


def run_pre_m6_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_packages: int = 4,
    max_stage_sessions: int = 4,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    queue = list_pre_m6_queue(
        persistent_root,
        max_candidates=max_queue,
        only_campaign_run_id=only_campaign_run_id,
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started = utc_now()
    for index, row in enumerate(queue):
        if index >= int(max_packages):
            break
        package = Path(row['package'])
        try:
            results.append(
                advance_one_toward_pre_m6(
                    package,
                    persistent_root=persistent_root,
                    project_root=project_root,
                    max_stage_sessions=max_stage_sessions,
                )
            )
        except Exception as exc:  # noqa: BLE001
            msg = f'{type(exc).__name__}: {exc}'
            if 'does not converge' in msg or 'regression' in msg.lower():
                status = 'M5_BLOCKED_M4_REGRESSION'
            elif 'M4' in msg and 'M5' not in msg:
                status = 'M4_BLOCKED'
            else:
                status = 'M5_BLOCKED'
            err = {
                'package': str(package),
                'candidate_id': row.get('candidate_id'),
                'error': msg,
                **screening_only_payload(),
            }
            errors.append(err)
            _write_pre_m6(package, {
                'status': status,
                'error': err['error'],
                'm4_complete': _m4_complete_on_disk(
                    persistent_root, str((_child_ids(package) or {}).get('M4') or ''),
                ),
            })

    summary = {
        'schema_version': 1,
        'session_id': f"PRE-M6-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'packages_attempted': len(results) + len(errors),
        'pre_m6_ready': sum(1 for r in results if r.get('status') == 'PRE_M6_READY'),
        'm4_checkpoint': sum(1 for r in results if r.get('status') == 'M4_CHECKPOINT'),
        'errors': errors[:50],
        'results': results,
        'note': (
            'Stops after M5. Production M6 is forbidden. '
            'See package M6_GATE.json = BLOCKED_PRE_M6.'
        ),
        **screening_only_payload(),
    }
    root = _ledger_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_PRE_M6_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Campaign B M4+M5 batch (stop before M6)')
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', '.'),
    )
    parser.add_argument('--max-packages', type=int, default=4)
    parser.add_argument('--max-stage-sessions', type=int, default=4)
    parser.add_argument('--max-queue', type=int, default=50)
    parser.add_argument('--campaign-run-id', default=None)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args(argv)
    persist = Path(args.persistent_root)
    if args.list_only:
        queue = list_pre_m6_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_pre_m6_batch(
        persistent_root=persist,
        project_root=Path(args.project_root).resolve(),
        max_packages=args.max_packages,
        max_stage_sessions=args.max_stage_sessions,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
    )
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'queue_size': summary.get('queue_size'),
        'packages_attempted': summary.get('packages_attempted'),
        'pre_m6_ready': summary.get('pre_m6_ready'),
        'm4_checkpoint': summary.get('m4_checkpoint'),
        'errors': summary.get('errors'),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0 if not summary.get('errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
