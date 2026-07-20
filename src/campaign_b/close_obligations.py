"""Re-evaluate M5 proof obligations for Campaign B PRE_M6 packages.

Does not invent CERTIFIED. Only RIGOROUS obligation closures from
evaluate_all_obligations count. Re-runs staged M5 so live assembly may start
once all_closed becomes true.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, utc_now
from .advance_selected import discover_selected_packages, _q_upper_from_package
from .pre_m6_batch import (
    _child_ids,
    _load,
    _m4_complete_on_disk,
    _pre_m6_status,
    _write_pre_m6,
    run_m5_session,
)
from .schemas import screening_only_payload


def list_obligation_queue(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
) -> list[dict[str, Any]]:
    persistent_root = Path(persistent_root)
    rows: list[dict[str, Any]] = []
    for package in discover_selected_packages(persistent_root):
        if only_campaign_run_id and only_campaign_run_id not in package.parts:
            continue
        child = _child_ids(package)
        if not isinstance(child, dict):
            continue
        m4_id = str(child.get('M4') or '')
        m5_id = str(child.get('M5') or '')
        if not m4_id.startswith('M4-') or not m5_id.startswith('M5-'):
            continue
        if not _m4_complete_on_disk(persistent_root, m4_id):
            continue
        obl_path = (
            Path(persistent_root) / 'runs' / m5_id / 'reports' / 'M5_obligation_report.json'
        )
        open_ids: list[str] = []
        all_closed = False
        if obl_path.is_file():
            doc = _load(obl_path)
            if isinstance(doc, dict):
                open_ids = list(doc.get('open_obligations') or [])
                all_closed = bool(doc.get('all_closed'))
        if all_closed and not open_ids:
            continue
        q = _q_upper_from_package(package)
        rows.append({
            'package': str(package),
            'candidate_id': package.name,
            'q_upper': None if q == float('inf') else q,
            'm4_run_id': m4_id,
            'm5_run_id': m5_id,
            'open_obligations': open_ids,
            'pre_m6_status': _pre_m6_status(package),
        })
    rows.sort(key=lambda r: (
        float('inf') if r['q_upper'] is None else float(r['q_upper']),
        r['package'],
    ))
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def reevaluate_one(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
) -> dict[str, Any]:
    """Re-run staged M5 obligation evaluation for one package."""
    package = Path(package)
    result = run_m5_session(
        package,
        persistent_root=persistent_root,
        project_root=project_root,
    )
    child = _child_ids(package) or {}
    m5_id = str(child.get('M5') or '')
    obl_path = Path(persistent_root) / 'runs' / m5_id / 'reports' / 'M5_obligation_report.json'
    obl = _load(obl_path) if obl_path.is_file() else None
    open_ids = list((obl or {}).get('open_obligations') or [])
    all_closed = bool((obl or {}).get('all_closed'))
    acceptance = Path(persistent_root) / 'runs' / m5_id / 'reports' / 'M5_acceptance.json'
    m5_complete = False
    if acceptance.is_file():
        acc = _load(acceptance)
        m5_complete = (
            isinstance(acc, dict)
            and acc.get('phase') == 'M5_COMPLETE'
            and acc.get('status') == 'PASS'
        )
    status = 'OBLIGATIONS_CLOSED_M5_COMPLETE' if (all_closed and m5_complete) else (
        'OBLIGATIONS_CLOSED_AWAITING_ASSEMBLY' if all_closed else 'OBLIGATIONS_STILL_OPEN'
    )
    out = {
        'status': status,
        'package': str(package),
        'candidate_id': package.name,
        'm5_session': result,
        'all_closed': all_closed,
        'open_obligations': open_ids,
        'closed_obligations': list((obl or {}).get('closed_obligations') or []),
        'm5_complete': m5_complete,
        'j2_max': (obl or {}).get('j2_max'),
        'lineage_parents': (obl or {}).get('lineage_parents'),
        'parent_chain_error': (obl or {}).get('parent_chain_error'),
        **screening_only_payload(),
    }
    _write_pre_m6(package, {
        'status': 'PRE_M6_READY',
        'obligation_status': status,
        'all_closed': all_closed,
        'open_obligations': open_ids,
        'm5_run_id': m5_id,
        'm4_run_id': child.get('M4'),
        'j2_max': (obl or {}).get('j2_max'),
    })
    if all_closed and m5_complete:
        atomic_write_json(package / 'M6_GATE.json', {
            'status': 'READY_FOR_STAGED_M6',
            'reason': (
                'M5 obligations all RIGOROUS and M5_acceptance COMPLETE; '
                'production paperspace M6 (81) still requires ONE_STEP_CERTIFIED'
            ),
            'm5_run_id': m5_id,
            **screening_only_payload(),
            'updated_at': utc_now(),
        })
    elif not all_closed:
        atomic_write_json(package / 'M6_GATE.json', {
            'status': 'BLOCKED_PRE_M6',
            'reason': 'M5 obligations still open after re-evaluation',
            'open_obligations': open_ids,
            'm5_run_id': m5_id,
            **screening_only_payload(),
            'updated_at': utc_now(),
        })
    return out


def run_close_obligations_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_packages: int = 4,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    queue = list_obligation_queue(
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
                reevaluate_one(
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
    summary = {
        'schema_version': 1,
        'session_id': f"OBL-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'attempted': len(results) + len(errors),
        'all_closed_count': sum(1 for r in results if r.get('all_closed')),
        'm5_complete_count': sum(1 for r in results if r.get('m5_complete')),
        'still_open': [
            {
                'package': r.get('package'),
                'candidate_id': r.get('candidate_id'),
                'open_obligations': r.get('open_obligations'),
                'j2_max': r.get('j2_max'),
                'parent_chain_error': r.get('parent_chain_error'),
            }
            for r in results if not r.get('all_closed')
        ],
        'results': results,
        'errors': errors[:50],
        'note': (
            'Re-evaluated M5 obligations with j2-aware projector cover and live '
            'M3/M2 lineage parents. Does not fake CERTIFIED. Production M6 (81) '
            'still needs ONE_STEP_CERTIFIED.'
        ),
        **screening_only_payload(),
    }
    root = Path(persistent_root) / 'campaign_b' / '_obligations'
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_OBLIGATION_SESSION.json', summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='Re-evaluate Campaign B M5 obligations')
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
        queue = list_obligation_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_close_obligations_batch(
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
        'all_closed_count': summary.get('all_closed_count'),
        'm5_complete_count': summary.get('m5_complete_count'),
        'still_open': summary.get('still_open'),
        'errors': summary.get('errors'),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0 if not summary.get('errors') else 1


if __name__ == '__main__':
    raise SystemExit(main())
