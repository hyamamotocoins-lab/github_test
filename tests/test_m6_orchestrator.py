from __future__ import annotations

from pathlib import Path

from src.m6_config import default_m6_config
from src.m6_orchestrator import create_or_resume_m6
from src.m6_status import CERTIFIED, M6_COMPLETE, NOT_CERTIFIED
from tests.m6_helpers import make_synthetic_accepted_m5, seed_project_docs


def test_m6_fixture_cert_complete(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persistent = tmp_path / 'persist'
    seed_project_docs(project)
    config = default_m6_config(
        parent_m5_run_id='M5-fixture',
        run_id='M6-fixture-cert',
        mode='cpu_fixture_cert',
        num_steps=3,
    )
    orchestrator = create_or_resume_m6(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == M6_COMPLETE
    assert report['certification_status'] == CERTIFIED
    assert report['verdict']['independent_verifier'] == 'PASS'
    assert (orchestrator.package_root / 'verdict.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'M6_acceptance.json').is_file()
    assert (orchestrator.package_root / 'rg_step_00' / 'step_verdict.json').is_file()
    assert (orchestrator.package_root / 'rg_step_02' / 'step_verdict.json').is_file()


def test_m6_fixture_not_certified_complete(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persistent = tmp_path / 'persist'
    seed_project_docs(project)
    config = default_m6_config(
        parent_m5_run_id='M5-fixture',
        run_id='M6-fixture-not-cert',
        mode='cpu_fixture_not_certified',
        num_steps=3,
    )
    orchestrator = create_or_resume_m6(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == M6_COMPLETE
    assert report['certification_status'] == NOT_CERTIFIED
    assert report['verdict']['independent_verifier'] == 'PASS'
    assert report['verdict'].get('failure_reason') == 'verified q_cert_lower >= 1'


def test_m6_live_parent_package(tmp_path: Path) -> None:
    project, persistent, parent_run_id = make_synthetic_accepted_m5(tmp_path)
    config = default_m6_config(
        parent_m5_run_id=parent_run_id,
        run_id='M6-synthetic-live',
        mode='live_parent',
        num_steps=3,
    )
    orchestrator = create_or_resume_m6(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == M6_COMPLETE
    assert report['certification_status'] == NOT_CERTIFIED
    assert report['implementation_status'] == 'M6_IMPLEMENTATION_COMPLETE'
    assert (orchestrator.run_root / 'reports' / 'M6_lock.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'M6_acceptance.json').is_file()
    assert (project / 'audit' / 'm6_accepted_parent.json').is_file()
