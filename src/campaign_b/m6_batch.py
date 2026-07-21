"""Batch staged/live-parent M6 for Campaign B packages with closed M5 obligations.

Uses mode=live_parent (not paperspace production gate 81).
May complete as NOT_CERTIFIED if q_cert_upper >= 1 — that is a verified
certificate failure, not a continuum claim and not fake CERTIFIED.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .advance_selected import discover_selected_packages, _q_upper_from_package
from .pre_m6_batch import _child_ids, _load
from .schemas import screening_only_payload


class M6BatchError(RuntimeError):
    """Raised when staged M6 cannot start safely."""


def _m5_ready_for_m6(persistent_root: Path, m5_run_id: str) -> dict[str, Any]:
    root = Path(persistent_root) / 'runs' / m5_run_id
    acceptance = _load(root / 'reports' / 'M5_acceptance.json')
    obl = _load(root / 'reports' / 'M5_obligation_report.json')
    package = root / 'artifacts' / 'one_step_certificate'
    verdict = _load(package / 'verdict.json') if package.is_dir() else None
    ok = (
        isinstance(acceptance, dict)
        and acceptance.get('phase') == 'M5_COMPLETE'
        and acceptance.get('status') == 'PASS'
        and acceptance.get('accepted_for_next_milestone') == 'M6'
        and acceptance.get('certification_status') in {
            'NOT_CERTIFIED', 'ONE_STEP_CERTIFIED',
        }
        and isinstance(obl, dict)
        and bool(obl.get('all_closed'))
        and package.is_dir()
        and isinstance(verdict, dict)
        and verdict.get('independent_verifier') == 'PASS'
    )
    return {
        'ok': ok,
        'acceptance': acceptance,
        'obligation': obl,
        'package_dir': str(package) if package.is_dir() else None,
        'certification_status': (
            acceptance.get('certification_status')
            if isinstance(acceptance, dict) else None
        ),
        'all_closed': bool((obl or {}).get('all_closed')) if isinstance(obl, dict) else False,
    }


def _m6_done(persistent_root: Path, m6_run_id: str) -> bool:
    root = Path(persistent_root) / 'runs' / m6_run_id
    report = _load(root / 'reports' / 'M6_report.json')
    acceptance = _load(root / 'reports' / 'M6_acceptance.json')
    if not isinstance(report, dict):
        return False
    phase = str(report.get('phase') or '')
    if phase == 'M6_COMPLETE' and isinstance(acceptance, dict):
        return True
    return phase == 'M6_COMPLETE'


def list_m6_queue(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
) -> list[dict[str, Any]]:
    persistent_root = Path(persistent_root)
    rows: list[dict[str, Any]] = []
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        child = _child_ids(package)
        if not isinstance(child, dict):
            continue
        m5_id = str(child.get('M5') or '')
        m6_id = str(child.get('M6') or '')
        if not m5_id.startswith('M5-') or not m6_id.startswith('M6-'):
            continue
        gate = _load(package / 'M6_GATE.json') or {}
        gate_status = str(gate.get('status') or '')
        ready = _m5_ready_for_m6(persistent_root, m5_id)
        if not ready['ok'] and gate_status != 'READY_FOR_STAGED_M6':
            continue
        if not ready['ok']:
            # Gate says ready but parent verify would fail — skip with note later.
            continue
        if _m6_done(persistent_root, m6_id) and not include_complete:
            continue
        q = _q_upper_from_package(package)
        rows.append({
            'package': str(package),
            'candidate_id': package.name,
            'q_upper': None if q == float('inf') else q,
            'm5_run_id': m5_id,
            'm6_run_id': m6_id,
            'm5_certification_status': ready.get('certification_status'),
            'gate_status': gate_status or 'READY_FOR_STAGED_M6',
        })
    rows.sort(key=lambda r: (
        float('inf') if r['q_upper'] is None else float(r['q_upper']),
        r['package'],
    ))
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def _m6_params(package: Path, persistent_root: Path, m5_run_id: str) -> dict[str, int]:
    m3_over = _load(package / 'm3_config_overrides.json') or {}
    m4_over = _load(package / 'm4_config_overrides.json') or {}
    scheme = _load(package / 'scheme.json') or {}
    m5_cfg = _load(Path(persistent_root) / 'runs' / m5_run_id / 'run_config.json') or {}
    cfg = m5_cfg.get('config') if isinstance(m5_cfg.get('config'), dict) else m5_cfg
    j2 = int(
        (cfg or {}).get('cutoff')
        or (cfg or {}).get('j2_max')
        or m3_over.get('j2_max')
        or m3_over.get('parent_m2_j2_max')
        or scheme.get('j2_max')
        or scheme.get('j2')
        or 2
    )
    bond = int(
        (cfg or {}).get('bond_dimension')
        or m4_over.get('projected_rank')
        or scheme.get('target_rank')
        or 16
    )
    num_steps = int(scheme.get('num_steps') or 3)
    if num_steps < 1:
        num_steps = 3
    return {'j2_max': j2, 'bond_dimension': bond, 'num_steps': num_steps}


def run_one_m6(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    from ..m6_config import default_m6_config
    from ..m6_orchestrator import create_or_resume_m6

    package = Path(package)
    child = _child_ids(package)
    if not isinstance(child, dict):
        raise M6BatchError('missing child_run_ids.json')
    m5_id = str(child.get('M5') or '')
    m6_id = str(child.get('M6') or '')
    if not m5_id.startswith('M5-') or not m6_id.startswith('M6-'):
        raise M6BatchError(f'bad M5/M6 ids: {m5_id!r} {m6_id!r}')
    ready = _m5_ready_for_m6(persistent_root, m5_id)
    if not ready['ok']:
        raise M6BatchError(
            f'M5 not ready for live M6: acceptance/obligations/one_step package '
            f'(all_closed={ready.get("all_closed")}, '
            f'cert={ready.get("certification_status")})'
        )
    params = _m6_params(package, persistent_root, m5_id)
    config = default_m6_config(
        parent_m5_run_id=m5_id,
        run_id=m6_id,
        mode='live_parent',
        j2_max=params['j2_max'],
        bond_dimension=params['bond_dimension'],
        num_steps=params['num_steps'],
    )
    atomic_write_json(package / 'M6_STATUS.json', {
        'status': 'M6_RUNNING',
        'm5_run_id': m5_id,
        'm6_run_id': m6_id,
        'mode': 'live_parent',
        'params': params,
        'updated_at': utc_now(),
        **screening_only_payload(),
    })
    orch = create_or_resume_m6(
        Path(persistent_root),
        config,
        Path(project_root),
        run_id=m6_id,
    )
    result = orch.run_until_checkpoint()
    cert = result.get('certification_status') if isinstance(result, dict) else None
    phase = result.get('phase') if isinstance(result, dict) else None
    # Never upgrade vocabulary; record orchestrator truth.
    out = {
        'status': 'M6_COMPLETE' if phase == 'M6_COMPLETE' else 'M6_FAILED',
        'package': str(package),
        'candidate_id': package.name,
        'm5_run_id': m5_id,
        'm6_run_id': m6_id,
        'mode': 'live_parent',
        'params': params,
        'phase': phase,
        'milestone_status': (
            result.get('milestone_status') if isinstance(result, dict) else None
        ),
        'certification_status_m6': cert,
        'q_cert_upper': (
            (result.get('verdict') or {}).get('q_cert_upper')
            if isinstance(result, dict) else None
        ),
        'q_cert_lower': (
            (result.get('verdict') or {}).get('q_cert_lower')
            if isinstance(result, dict) else None
        ),
        'run_root': str(orch.run_root),
        'result': result,
        'note': (
            'Staged live_parent M6 (notebook 70 path). '
            'NOT production paperspace gate 81. '
            'NOT_CERTIFIED means majorant did not prove q_cert<1.'
        ),
        **screening_only_payload(),
    }
    # Force screening claim_scope even if CERTIFIED somehow appeared —
    # Campaign B batch still records orchestrator cert separately.
    if cert == 'CERTIFIED':
        out['campaign_b_note'] = (
            'Orchestrator reported CERTIFIED for finite-cutoff finite-step '
            'majorant only; Campaign B claim_scope remains SCREENING_ONLY '
            'until human review of production gate 81.'
        )
    atomic_write_json(package / 'M6_STATUS.json', {
        **out,
        'updated_at': utc_now(),
    })
    atomic_write_json(package / 'M6_GATE.json', {
        'status': 'M6_DONE' if out['status'] == 'M6_COMPLETE' else 'M6_FAILED',
        'm5_run_id': m5_id,
        'm6_run_id': m6_id,
        'certification_status_m6': cert,
        'phase': phase,
        'updated_at': utc_now(),
        **screening_only_payload(),
    })
    return out


def run_m6_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_packages: int = 4,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    queue = list_m6_queue(
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
                run_one_m6(
                    package,
                    persistent_root=persistent_root,
                    project_root=project_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            err = {
                'package': str(package),
                'candidate_id': row.get('candidate_id'),
                'error': f'{type(exc).__name__}: {exc}',
                **screening_only_payload(),
            }
            errors.append(err)
            atomic_write_json(package / 'M6_STATUS.json', {
                'status': 'M6_ERROR',
                'error': err['error'],
                'updated_at': utc_now(),
                **screening_only_payload(),
            })

    summary = {
        'schema_version': 1,
        'session_id': f"M6B-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'attempted': len(results) + len(errors),
        'm6_complete': sum(1 for r in results if r.get('status') == 'M6_COMPLETE'),
        'm6_certified_count': sum(
            1 for r in results if r.get('certification_status_m6') == 'CERTIFIED'
        ),
        'm6_not_certified_count': sum(
            1 for r in results if r.get('certification_status_m6') == 'NOT_CERTIFIED'
        ),
        'errors': errors[:50],
        'results': [
            {
                'candidate_id': r.get('candidate_id'),
                'm6_run_id': r.get('m6_run_id'),
                'status': r.get('status'),
                'certification_status_m6': r.get('certification_status_m6'),
                'q_cert_upper': r.get('q_cert_upper'),
                'phase': r.get('phase'),
            }
            for r in results
        ],
        'note': (
            'live_parent M6 batch (70-equivalent). Not notebook 81 production gate. '
            'CERTIFIED only if orchestrator proves q_cert<1 on declared majorant; '
            'Campaign B claim_scope remains SCREENING_ONLY.'
        ),
        **screening_only_payload(),
    }
    root = Path(persistent_root) / 'campaign_b' / '_m6'
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_M6_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Campaign B live_parent M6 batch')
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', '.'),
    )
    parser.add_argument('--max-packages', type=int, default=4)
    parser.add_argument('--max-queue', type=int, default=50)
    parser.add_argument('--campaign-run-id', default=None)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args(argv)
    persist = Path(args.persistent_root)
    if args.list_only:
        queue = list_m6_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_m6_batch(
        persistent_root=persist,
        project_root=Path(args.project_root).resolve(),
        max_packages=args.max_packages,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
    )
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'queue_size': summary.get('queue_size'),
        'attempted': summary.get('attempted'),
        'm6_complete': summary.get('m6_complete'),
        'm6_certified_count': summary.get('m6_certified_count'),
        'm6_not_certified_count': summary.get('m6_not_certified_count'),
        'errors': summary.get('errors'),
        'results': summary.get('results'),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0 if not summary.get('errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
