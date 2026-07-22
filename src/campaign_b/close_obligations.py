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
    include_errors: bool = False,
) -> list[dict[str, Any]]:
    """Packages with M4 done that still need obligation re-eval.

    Durable PRE_M6 blocked statuses (M4_BLOCKED / M5_BLOCKED /
    M5_BLOCKED_M4_REGRESSION) are excluded unless include_errors=True,
    matching list_pre_m6_queue so one poison package cannot monopolize
    MAX_OBLIGATION_PACKAGES=1.
    """
    persistent_root = Path(persistent_root)
    from .queue_index import (
        _scan_obligation_rows,
        ensure_obligation_index,
        list_obligation_queue_indexed,
        rebuild_obligation_index,
    )

    if include_errors:
        return _scan_obligation_rows(
            persistent_root,
            only_campaign_run_id=only_campaign_run_id,
            max_candidates=max_candidates,
            include_errors=True,
        )

    ensure_obligation_index(
        persistent_root,
        only_campaign_run_id=only_campaign_run_id,
    )
    indexed = list_obligation_queue_indexed(
        persistent_root,
        max_candidates=max_candidates,
        only_campaign_run_id=only_campaign_run_id,
    )
    if indexed:
        return indexed

    rebuild_obligation_index(
        persistent_root,
        only_campaign_run_id=only_campaign_run_id,
    )
    indexed = list_obligation_queue_indexed(
        persistent_root,
        max_candidates=max_candidates,
        only_campaign_run_id=only_campaign_run_id,
    )
    if indexed:
        return indexed

    return _scan_obligation_rows(
        persistent_root,
        only_campaign_run_id=only_campaign_run_id,
        max_candidates=max_candidates,
    )


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
    try:
        from .queue_index import sync_m6_index_entry, sync_obligation_index_entry

        sync_obligation_index_entry(package, persistent_root)
        sync_m6_index_entry(package, persistent_root)
    except Exception:  # noqa: BLE001
        pass
    return out


def run_close_obligations_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_packages: int = 4,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
    include_errors: bool = False,
) -> dict[str, Any]:
    from .pre_m6_batch import (
        PRE_M6_BLOCKED_EXCLUDED,
        _classify_pre_m6_failure,
        _pre_m6_status,
    )
    from .queue_index import fetch_limit_for_batch, sync_obligation_index_entry

    fetch = fetch_limit_for_batch(
        max_items=int(max_packages),
        max_queue=int(max_queue),
    )
    # Oversample so mid-scan blocked skips can still fill max_packages.
    scan_limit = max(fetch, int(max_packages) * 8)
    scan_limit = min(scan_limit, int(max_queue)) if max_queue else scan_limit
    queue = list_obligation_queue(
        persistent_root,
        max_candidates=scan_limit,
        only_campaign_run_id=only_campaign_run_id,
        include_errors=include_errors,
    )
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    skipped_blocked: list[str] = []
    started = utc_now()
    attempted = 0
    for row in queue:
        if attempted >= int(max_packages):
            break
        package = Path(row['package'])
        current = _pre_m6_status(package)
        if current in PRE_M6_BLOCKED_EXCLUDED and not include_errors:
            skipped_blocked.append(str(package.name))
            continue
        attempted += 1
        try:
            results.append(
                reevaluate_one(
                    package,
                    persistent_root=persistent_root,
                    project_root=project_root,
                )
            )
        except Exception as exc:  # noqa: BLE001
            blocked = _classify_pre_m6_failure(exc)
            msg = f'{type(exc).__name__}: {exc}'
            child = _child_ids(package) or {}
            _write_pre_m6(package, {
                'status': blocked,
                'error': msg,
                'blocked_durable': True,
                'm4_run_id': child.get('M4'),
                'm5_run_id': child.get('M5'),
                'stage': 'OBLIGATIONS',
            })
            try:
                sync_obligation_index_entry(package, persistent_root)
            except Exception:  # noqa: BLE001
                pass
            err = {
                'package': str(package),
                'candidate_id': row.get('candidate_id'),
                'error': msg,
                'status': blocked,
                **screening_only_payload(),
            }
            errors.append(err)
    summary = {
        'schema_version': 1,
        'session_id': f"OBL-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'attempted': attempted,
        'skipped_blocked': skipped_blocked[:50],
        'include_errors': bool(include_errors),
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
            'still needs ONE_STEP_CERTIFIED. '
            'M4_BLOCKED / M5_BLOCKED / M5_BLOCKED_M4_REGRESSION leave the '
            'default obligation queue (include_errors to retry).'
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
    parser.add_argument(
        '--include-errors',
        action='store_true',
        help='Include durable M4/M5 blocked packages in the queue',
    )
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args(argv)
    persist = Path(args.persistent_root)
    if args.list_only:
        queue = list_obligation_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
            include_errors=args.include_errors,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_close_obligations_batch(
        persistent_root=persist,
        project_root=Path(args.project_root).resolve(),
        max_packages=args.max_packages,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
        include_errors=args.include_errors,
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
