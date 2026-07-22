"""Tests for pre-M6 queue discovery (no CUDA)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.campaign_b.pre_m6_batch import (
    PRE_M6_BLOCKED_EXCLUDED,
    _classify_pre_m6_failure,
    list_pre_m6_queue,
    run_pre_m6_batch,
)
from src.campaign_b.schemas import CERTIFICATION_STATUS, CLAIM_SCOPE
from src.common import atomic_write_json
from src.m5_parent import M5ParentError


def _ready_pkg(
    root: Path,
    cid: str,
    *,
    q: float,
    pre_status: str | None = None,
    error: str | None = None,
) -> Path:
    pkg = root / 'campaign_b' / 'M7-A' / 'selected' / cid
    pkg.mkdir(parents=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': cid,
        'scheme': {'change_class': 'S2', 'target_rank': 16},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': q})
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M2': f'M2-{cid}',
        'M3': f'M3-{cid}',
        'M4': f'M4-{cid}',
        'M5': f'M5-{cid}',
        'M6': f'M6-{cid}',
    })
    atomic_write_json(pkg / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
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


def test_list_pre_m6_requires_m3_complete(tmp_path: Path) -> None:
    pkg = tmp_path / 'campaign_b' / 'M7-A' / 'selected' / 'B-aaa'
    pkg.mkdir(parents=True)
    atomic_write_json(pkg / 'candidate_manifest.json', {
        'candidate_id': 'B-aaa',
        'scheme': {'change_class': 'S2', 'target_rank': 16},
    })
    atomic_write_json(pkg / 's0_result.json', {'q_upper': 0.81})
    atomic_write_json(pkg / 'child_run_ids.json', {
        'M2': 'M2-X', 'M3': 'M3-X', 'M4': 'M4-X', 'M5': 'M5-X', 'M6': 'M6-X',
    })
    # Not complete yet
    assert list_pre_m6_queue(tmp_path) == []

    atomic_write_json(pkg / 'GPU_M3.json', {
        'status': 'M3_COMPLETE',
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
    })
    # Still need M3 on disk unless GPU says complete — GPU_M3 alone is enough
    queue = list_pre_m6_queue(tmp_path)
    assert len(queue) == 1
    assert queue[0]['stage'] == 'NEED_M4'
    assert queue[0]['candidate_id'] == 'B-aaa'


def test_list_pre_m6_excludes_durable_m4_m5_blocked(tmp_path: Path) -> None:
    """Poison package with M5ParentError FD status must leave default queue."""
    poison = _ready_pkg(
        tmp_path,
        'B-0b31d2ec0d8be5ce',
        q=0.40,
        pre_status='M5_BLOCKED_M4_REGRESSION',
        error=(
            'M5ParentError: M4 centered finite difference '
            'lacks second-order convergence.'
        ),
    )
    next_ok = _ready_pkg(tmp_path, 'B-next-ok', q=0.55)
    _ready_pkg(tmp_path, 'B-also-blocked', q=0.30, pre_status='M5_BLOCKED')
    _ready_pkg(tmp_path, 'B-m4-blocked', q=0.20, pre_status='M4_BLOCKED')

    queue = list_pre_m6_queue(tmp_path, max_candidates=20)
    ids = [r['candidate_id'] for r in queue]
    assert poison.name not in ids
    assert 'B-also-blocked' not in ids
    assert 'B-m4-blocked' not in ids
    assert ids == ['B-next-ok']

    with_err = list_pre_m6_queue(
        tmp_path, max_candidates=20, include_errors=True,
    )
    with_ids = [r['candidate_id'] for r in with_err]
    assert poison.name in with_ids
    assert 'B-next-ok' in with_ids


def test_classify_fd_order_failure_is_regression_block() -> None:
    exc = M5ParentError(
        'M4 centered finite difference lacks second-order convergence.',
    )
    assert _classify_pre_m6_failure(exc) == 'M5_BLOCKED_M4_REGRESSION'
    assert _classify_pre_m6_failure(exc) in PRE_M6_BLOCKED_EXCLUDED


def test_run_pre_m6_skips_blocked_and_advances_next(tmp_path: Path) -> None:
    """MAX_PRE_M6=1 must not retry a durable-blocked head forever."""
    _ready_pkg(
        tmp_path,
        'B-poison',
        q=0.10,
        pre_status='M5_BLOCKED_M4_REGRESSION',
        error=(
            'M5ParentError: M4 centered finite difference '
            'lacks second-order convergence.'
        ),
    )
    nxt = _ready_pkg(tmp_path, 'B-healthy', q=0.50)

    def _fake_advance(package: Path, **_kwargs: object) -> dict:
        return {
            'package': str(package),
            'status': 'M4_CHECKPOINT',
            'certification_status': CERTIFICATION_STATUS,
            'claim_scope': CLAIM_SCOPE,
        }

    # include_errors=True puts poison first by q; batch must skip and take next.
    with patch(
        'src.campaign_b.pre_m6_batch.advance_one_toward_pre_m6',
        side_effect=_fake_advance,
    ) as adv:
        summary = run_pre_m6_batch(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_packages=1,
            max_queue=50,
            include_errors=False,
        )

    assert summary['packages_attempted'] == 1
    assert summary['m4_checkpoint'] == 1
    assert summary['errors'] == []
    assert adv.call_count == 1
    assert Path(adv.call_args.args[0]).name == nxt.name


def test_run_pre_m6_writes_durable_block_and_drops_from_queue(
    tmp_path: Path,
) -> None:
    poison = _ready_pkg(tmp_path, 'B-fd-fail', q=0.10)
    nxt = _ready_pkg(tmp_path, 'B-after', q=0.60)

    def _raise_fd(package: Path, **_kwargs: object) -> dict:
        if package.name == poison.name:
            raise M5ParentError(
                'M4 centered finite difference lacks second-order convergence.',
            )
        return {
            'package': str(package),
            'status': 'PRE_M6_READY',
            'certification_status': CERTIFICATION_STATUS,
            'claim_scope': CLAIM_SCOPE,
        }

    with patch(
        'src.campaign_b.pre_m6_batch.advance_one_toward_pre_m6',
        side_effect=_raise_fd,
    ):
        first = run_pre_m6_batch(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_packages=1,
        )

    assert first['packages_attempted'] == 1
    assert len(first['errors']) == 1
    assert first['errors'][0]['status'] == 'M5_BLOCKED_M4_REGRESSION'
    pre = (poison / 'PRE_M6.json').read_text(encoding='utf-8')
    assert 'M5_BLOCKED_M4_REGRESSION' in pre
    assert 'blocked_durable' in pre

    # Second round: poison excluded; next candidate proceeds.
    with patch(
        'src.campaign_b.pre_m6_batch.advance_one_toward_pre_m6',
        side_effect=_raise_fd,
    ) as adv:
        second = run_pre_m6_batch(
            persistent_root=tmp_path,
            project_root=tmp_path,
            max_packages=1,
        )

    assert second['packages_attempted'] == 1
    assert second['pre_m6_ready'] == 1
    assert second['errors'] == []
    assert adv.call_count == 1
    assert Path(adv.call_args.args[0]).name == nxt.name
    ids = [r['candidate_id'] for r in list_pre_m6_queue(tmp_path)]
    assert poison.name not in ids
