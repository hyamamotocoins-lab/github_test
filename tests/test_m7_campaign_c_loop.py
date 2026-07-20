from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.common import atomic_write_json
from src.m2_shared_registry import BINDING_NEED, BINDING_READY
from src.m7_archive import write_archive, REASON_S0_NO_SELECTION
from src.m7_campaign_c_loop import run_campaign_c_queue_s0_loop


def _ranking_row(candidate_id: str, j2_max: int, q: float) -> dict:
    return {
        'candidate_id': candidate_id,
        'scheme_hash': f'hash-{candidate_id}',
        'q_cert_upper': str(q),
        'scheme': {'change_class': 'S3', 'j2_max': j2_max},
    }


def _make_search(tmp: Path) -> tuple[Path, Path]:
    persist = tmp / 'persist'
    search = persist / 'searches' / 'M7-fixture'
    auto = search / 'auto_execute'
    reports = search / 'reports'
    auto.mkdir(parents=True)
    reports.mkdir(parents=True)
    atomic_write_json(search / 'LOCK.json', {
        'run_id': 'M7-fixture',
        'parent_m6_run_id': 'M6-fixture',
    })
    ranking = {
        'ranking': [
            _ranking_row('CAND-a', 2, 0.5),
            _ranking_row('CAND-b', 2, 0.7),
        ]
    }
    atomic_write_json(reports / 'candidate_ranking.json', ranking)
    for cand in ('CAND-a', 'CAND-b'):
        pkg = auto / cand
        pkg.mkdir(parents=True)
        atomic_write_json(pkg / 'MANIFEST.json', {
            'candidate_id': cand,
            'package_root': str(pkg),
        })
        atomic_write_json(pkg / 'scheme.json', {'j2_max': 2})
        atomic_write_json(pkg / 'resource_gate.json', {
            'staged_executable': True,
            'instant_executable': False,
        })
        atomic_write_json(pkg / 'child_run_ids.json', {'M2': f'M2-{cand}'})
        atomic_write_json(pkg / 'm3_config_overrides.json', {
            'j2_max': 2,
            'sector_count': 729,
            'operator_dimension': 46656,
            'target_rank': 16,
        })
    return persist, search


def test_loop_archives_then_stops_on_need_canonical(tmp_path: Path) -> None:
    persist, search = _make_search(tmp_path)
    project = tmp_path / 'project'
    project.mkdir()
    (project / 'src').mkdir()

    calls = {'s0': 0}

    def fake_prepare(**kwargs):
        cid = kwargs['row']['candidate_id']
        if cid == 'CAND-a':
            return {
                'candidate_id': cid,
                'package_root': str(search / 'auto_execute' / cid),
                'binding': {'state': BINDING_READY, 'canonical_run_id': 'M2-SHARED-x'},
                'state': BINDING_READY,
                'canonical_run_id': 'M2-SHARED-x',
            }
        return {
            'candidate_id': cid,
            'package_root': str(search / 'auto_execute' / cid),
            'binding': {'state': BINDING_NEED, 'canonical_run_id': 'M2-SHARED-y'},
            'state': BINDING_NEED,
            'canonical_run_id': 'M2-SHARED-y',
        }

    def fake_s0(**kwargs):
        calls['s0'] += 1
        cid = kwargs['candidate_id']
        write_archive(
            Path(kwargs['package_root']),
            reason=REASON_S0_NO_SELECTION,
            details={'selection_status': 'NO_SELECTION'},
        )
        return {
            'status': 'EXPLORATORY_NOT_CERTIFIED',
            'series_status': 'ARCHIVED',
            'candidate_id': cid,
            'selection_status': 'NO_SELECTION',
        }

    with patch('src.m7_campaign_c_loop.materialize_top_k', return_value=[]), \
         patch('src.m7_campaign_c_loop.prepare_candidate_binding', side_effect=fake_prepare), \
         patch('src.m7_campaign_c_loop.run_s0_for_package', side_effect=fake_s0):
        result = run_campaign_c_queue_s0_loop(
            project_root=project,
            persistent_root=persist,
            search_root=search,
            campaign_run_id='M7-fixture',
            max_candidates=8,
        )

    assert calls['s0'] == 1
    assert result['series_status'] == BINDING_NEED
    assert result['candidate_id'] == 'CAND-b'
    assert result['next_notebook'].startswith('73_')
