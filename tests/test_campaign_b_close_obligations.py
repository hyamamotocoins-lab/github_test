"""Tests for obligation queue durable-block exclusion (CPU only)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.campaign_b.close_obligations import (
    list_obligation_queue,
    run_close_obligations_batch,
)
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json
from src.m5_parent import M5ParentError


def _obl_pkg(
    root: Path,
    cid: str,
    *,
    pre_status: str | None = 'PRE_M6_READY',
    error: str | None = None,
    open_obligations: list[str] | None = None,
) -> Path:
    pkg = root / 'campaign_b' / 'M7-A' / 'selected' / cid
    pkg.mkdir(parents=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': cid,
        'scheme': {'change_class': 'S2', 'target_rank': 16},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': 0.5})
    m4_id = f'M4-{cid}'
    m5_id = f'M5-{cid}'
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M2': f'M2-{cid}',
        'M3': f'M3-{cid}',
        'M4': m4_id,
        'M5': m5_id,
        'M6': f'M6-{cid}',
    })
    m4_reports = root / 'runs' / m4_id / 'reports'
    m4_reports.mkdir(parents=True)
    atomic_write_json(m4_reports / 'M4_report.json', {'phase': 'M4_COMPLETE'})
    atomic_write_json(m4_reports / 'M4_acceptance.json', {'status': 'PASS'})
    open_ids = (
        list(open_obligations)
        if open_obligations is not None
        else ['OBL-RSVD-RESIDUAL']
    )
    m5_reports = root / 'runs' / m5_id / 'reports'
    m5_reports.mkdir(parents=True)
    atomic_write_json(m5_reports / 'M5_obligation_report.json', {
        'all_closed': False,
        'open_obligations': open_ids,
        'closed_obligations': [],
    })
    if pre_status is not None:
        doc: dict = {
            'status': pre_status,
            'certification_status': CERTIFICATION_STATUS,
            'claim_scope': CLAIM_SCOPE,
        }
        if error is not None:
            doc['error'] = error
            doc['blocked_durable'] = True
        atomic_write_json(pkg / 'PRE_M6.json', doc)
    return pkg


def test_list_obligation_excludes_durable_m4_m5_blocked(tmp_path: Path) -> None:
    poison = _obl_pkg(
        tmp_path,
        'B-0b31d2ec0d8be5ce',
        pre_status='M5_BLOCKED_M4_REGRESSION',
        error=(
            'M5ParentError: M4 centered finite difference '
            'lacks second-order convergence.'
        ),
    )
    next_ok = _obl_pkg(tmp_path, 'B-next-ok')
    _obl_pkg(tmp_path, 'B-also-blocked', pre_status='M5_BLOCKED')
    _obl_pkg(tmp_path, 'B-m4-blocked', pre_status='M4_BLOCKED')

    queue = list_obligation_queue(tmp_path, max_candidates=20)
    ids = [r['candidate_id'] for r in queue]
    assert poison.name not in ids
    assert 'B-also-blocked' not in ids
    assert 'B-m4-blocked' not in ids
    assert ids == ['B-next-ok']

    with_err = list_obligation_queue(
        tmp_path, max_candidates=20, include_errors=True,
    )
    with_ids = [r['candidate_id'] for r in with_err]
    assert poison.name in with_ids
    assert next_ok.name in with_ids


def test_run_close_obligations_writes_durable_block_and_drops(
    tmp_path: Path,
) -> None:
    poison = _obl_pkg(tmp_path, 'B-00-fd-fail')
    nxt = _obl_pkg(tmp_path, 'B-01-after')

    def _raise_fd(package: Path, **_kwargs: object) -> dict:
        if package.name == poison.name:
            raise M5ParentError(
                'M4 centered finite difference lacks second-order convergence.',
            )
        return {
            'package': str(package),
            'status': 'OBLIGATIONS_STILL_OPEN',
            'all_closed': False,
            'open_obligations': ['OBL-RSVD-RESIDUAL'],
            'closed_obligations': [],
            'm5_complete': False,
            'certification_status': CERTIFICATION_STATUS,
            'claim_scope': CLAIM_SCOPE,
        }

    with patch(
        'src.campaign_b.close_obligations.reevaluate_one',
        side_effect=_raise_fd,
    ):
        first = run_close_obligations_batch(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_packages=1,
        )

    assert first['attempted'] == 1
    assert len(first['errors']) == 1
    assert first['errors'][0]['status'] == 'M5_BLOCKED_M4_REGRESSION'
    pre = (poison / 'PRE_M6.json').read_text(encoding='utf-8')
    assert 'M5_BLOCKED_M4_REGRESSION' in pre
    assert 'blocked_durable' in pre

    with patch(
        'src.campaign_b.close_obligations.reevaluate_one',
        side_effect=_raise_fd,
    ) as reeval:
        second = run_close_obligations_batch(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_packages=1,
        )

    assert second['attempted'] == 1
    assert second['errors'] == []
    assert reeval.call_count == 1
    assert Path(reeval.call_args.args[0]).name == nxt.name
    ids = [r['candidate_id'] for r in list_obligation_queue(tmp_path)]
    assert poison.name not in ids
    assert nxt.name in ids
