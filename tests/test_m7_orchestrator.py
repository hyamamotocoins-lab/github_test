from __future__ import annotations

from pathlib import Path

from src.m7_config import default_m7_config
from src.m7_orchestrator import create_or_resume_m7
from src.m7_status import (
    CERTIFIED_SCHEME_FOUND,
    M7_CERTIFIED_SCHEME_FOUND,
    M7_COMPLETE,
    M7_LINEAGE_PLANNED,
    M7_SEARCH_SPACE_EXHAUSTED,
)
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS
from src.common import atomic_write_text


def _seed(project: Path) -> None:
    (project / 'src').mkdir(parents=True, exist_ok=True)
    (project / 'audit').mkdir(parents=True, exist_ok=True)
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_write_text(path, f'# fixture\n{relative}\n')


def test_m7_fixture_finds_certified_scheme(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    _seed(project)
    config = default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-cert',
        mode='cpu_fixture_cert',
        max_candidates_total=8,
        max_rigorous_replays=4,
    )
    orch = create_or_resume_m7(persist, config, project)
    summary = orch.run_search()
    assert summary['phase'] == M7_COMPLETE
    assert summary['search_status'] == M7_CERTIFIED_SCHEME_FOUND
    assert summary['accepted'] is not None
    final = orch.search_root / 'final_package' / 'M7_acceptance.json'
    assert final.is_file()
    from src.common import read_json
    acceptance = read_json(final)
    assert acceptance['status'] == CERTIFIED_SCHEME_FOUND
    assert acceptance['independent_verifier'] == 'PASS'
    assert float(acceptance['q_cert_upper']) < 1


def test_m7_campaign_a_can_exhaust_without_claiming_nonexistence(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    _seed(project)
    from src.m7_orchestrator import M7Orchestrator
    orch2 = M7Orchestrator(project, persist, default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-exhaust',
        mode='cpu_fixture',
        max_candidates_total=3,
        max_rigorous_replays=3,
    ))
    summary = orch2.run_search()
    assert summary['phase'] == M7_COMPLETE
    assert summary['search_status'] == M7_SEARCH_SPACE_EXHAUSTED
    assert summary['accepted'] is None


def test_m7_campaign_b_plan_only_emits_lineage_plans(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    _seed(project)
    from src.m7_orchestrator import M7Orchestrator
    orch = M7Orchestrator(project, persist, default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-b-plan',
        mode='cpu_fixture',
        campaign='B',
        lineage_mode='plan_only',
        max_candidates_total=4,
        max_rigorous_replays=4,
    ))
    summary = orch.run_search()
    assert summary['phase'] == M7_COMPLETE
    assert summary['search_status'] == M7_LINEAGE_PLANNED
    assert summary['accepted'] is None
    assert summary['lineage_plans'] == 4
    plans = orch.search_root / 'reports' / 'lineage_plans.json'
    assert plans.is_file()
    from src.common import read_json
    doc = read_json(plans)
    assert doc['plans'][0]['change_class'] == 'S2'
    assert 'M3' in doc['plans'][0]['child_run_ids']


def test_m7_campaign_b_fixture_residual_can_certify(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    _seed(project)
    config = default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-b-cert',
        mode='cpu_fixture_campaign_b',
        campaign='B',
        lineage_mode='fixture_residual',
        max_candidates_total=4,
        max_rigorous_replays=2,
        max_lineage_replays=2,
        stop_on_first_certified=True,
    )
    orch = create_or_resume_m7(persist, config, project)
    summary = orch.run_search()
    assert summary['phase'] == M7_COMPLETE
    assert summary['search_status'] == M7_CERTIFIED_SCHEME_FOUND
    assert summary['accepted'] is not None
    assert float(summary['accepted']['q_cert_upper']) < 1
    from src.common import read_json
    acceptance = read_json(orch.search_root / 'final_package' / 'M7_acceptance.json')
    assert acceptance['independent_verifier'] == 'PASS'
    assert acceptance.get('campaign') == 'B'
