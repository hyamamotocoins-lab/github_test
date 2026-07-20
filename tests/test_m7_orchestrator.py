from __future__ import annotations

from pathlib import Path

from src.m7_config import default_m7_config
from src.m7_orchestrator import create_or_resume_m7
from src.m7_status import (
    CERTIFIED_SCHEME_FOUND,
    M7_CERTIFIED_SCHEME_FOUND,
    M7_COMPLETE,
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
    config = default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-search',
        mode='cpu_fixture_search',
        # Exclude the leading contractive candidate by using paperspace-like
        # generation only: use mode that still builds noncontractive parent but
        # force campaign candidates without fixture contractive by lowering to
        # paperspace generation via custom: use cpu_fixture without cert insert.
        max_candidates_total=4,
        max_rigorous_replays=4,
        stop_on_first_certified=True,
    )
    # cpu_fixture_search prepends contractive candidate — so it will find CERTIFIED.
    # For exhaustion, run paperspace-like generation against noncontractive only:
    orch = create_or_resume_m7(persist, config, project)
    # Monkeypatch by using mode that doesn't prepend: recreate with only campaign.
    from src.m7_orchestrator import M7Orchestrator
    orch2 = M7Orchestrator(project, persist, default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-exhaust',
        mode='cpu_fixture',  # not cert, not search — campaign only on noncontractive
        max_candidates_total=3,
        max_rigorous_replays=3,
    ))
    summary = orch2.run_search()
    assert summary['phase'] == M7_COMPLETE
    # Non-contractive parent + Campaign A reweight/product cannot certify.
    assert summary['search_status'] == M7_SEARCH_SPACE_EXHAUSTED
    assert summary['accepted'] is None
