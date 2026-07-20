from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json
from src.m7_generator import generate_campaign_c_candidates
from src.m7_q_lt1_hunt import diagnose_q_lt1_hunt, mint_m7c_qlt1_run_id, next_q_lt1_actions


def _ranking_row(candidate_id: str, j2_max: int, q: float) -> dict:
    return {
        'candidate_id': candidate_id,
        'scheme_hash': f'hash-{candidate_id}',
        'q_cert_upper': str(q),
        'scheme': {'change_class': 'S3', 'j2_max': j2_max},
    }


def test_mint_run_id_format() -> None:
    run_id = mint_m7c_qlt1_run_id(tag='qlt1c')
    assert run_id.startswith('M7-')
    assert 'qlt1c' in run_id


def test_diagnose_recommends_new_search_when_best_staged_ge_1(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    search = persist / 'searches' / 'M7-old'
    reports = search / 'reports'
    reports.mkdir(parents=True)
    atomic_write_json(search / 'LOCK.json', {'run_id': 'M7-old'})
    atomic_write_json(reports / 'candidate_ranking.json', {
        'ranking': [
            _ranking_row('CAND-a', 2, 1.011),
            _ranking_row('CAND-b', 1, 1.8),
        ],
    })
    for cand in ('CAND-a', 'CAND-b'):
        pkg = search / 'auto_execute' / cand
        pkg.mkdir(parents=True)
        atomic_write_json(pkg / 'MANIFEST.json', {'candidate_id': cand})
        atomic_write_json(pkg / 'child_run_ids.json', {'M2': f'M2-{cand}'})
        atomic_write_json(pkg / 'scheme.json', {
            'j2_max': 2 if cand == 'CAND-a' else 1,
        })
        atomic_write_json(pkg / 'resource_gate.json', {
            'staged_executable': cand == 'CAND-a',
            'instant_executable': cand == 'CAND-b',
            'executable': cand == 'CAND-b',
        })
    diag = diagnose_q_lt1_hunt(persistent_root=persist, search_run_id='M7-old')
    assert diag['recommendation'] == 'NEW_EXPANDED_CAMPAIGN_C_SEARCH'
    assert diag['best_staged']['estimated_q'] == 1.011
    actions = next_q_lt1_actions(diag)
    assert actions['action'] == 'notebook_86_new_search'
    assert actions['suggested_run_id'].startswith('M7-')


def test_diagnose_finds_staged_q_lt1(tmp_path: Path) -> None:
    persist = tmp_path / 'persist'
    search = persist / 'searches' / 'M7-hit'
    reports = search / 'reports'
    reports.mkdir(parents=True)
    atomic_write_json(search / 'LOCK.json', {'run_id': 'M7-hit'})
    atomic_write_json(reports / 'candidate_ranking.json', {
        'ranking': [
            _ranking_row('CAND-win', 2, 0.91),
            _ranking_row('CAND-b', 2, 1.2),
        ],
    })
    for cand in ('CAND-win', 'CAND-b'):
        pkg = search / 'auto_execute' / cand
        pkg.mkdir(parents=True)
        atomic_write_json(pkg / 'MANIFEST.json', {'candidate_id': cand})
        atomic_write_json(pkg / 'child_run_ids.json', {'M2': f'M2-{cand}'})
        atomic_write_json(pkg / 'scheme.json', {'j2_max': 2})
        atomic_write_json(pkg / 'resource_gate.json', {
            'staged_executable': True,
            'instant_executable': False,
            'executable': False,
        })
    diag = diagnose_q_lt1_hunt(persistent_root=persist, search_run_id='M7-hit')
    assert diag['recommendation'] == 'RUN_85_ON_STAGED_Q_LT1'
    assert diag['counts']['staged_q_lt1_live'] == 1


def test_expanded_campaign_c_prefers_j2_2_and_includes_new_coupling() -> None:
    rows = generate_campaign_c_candidates(
        parent_m6_run_id='M6-x',
        parent_scheme_hash='sha256:' + 'a' * 64,
        limit=20,
    )
    assert rows
    assert all(int(r['scheme']['j2_max']) == 2 for r in rows[:8])
    couplings = {r['scheme']['coupling_policy'] for r in rows}
    assert 'diagonal_plus_l1_tail' in couplings
    seeds = {r['scheme']['seed'] for r in rows}
    assert len(seeds) >= 2
