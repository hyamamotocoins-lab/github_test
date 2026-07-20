from __future__ import annotations

from pathlib import Path

from src.m5_config import default_m5_config
from src.m5_orchestrator import create_or_resume_m5
from src.m5_status import M5_COMPLETE, NOT_CERTIFIED, ONE_STEP_CERTIFIED
from tests.m5_helpers import make_synthetic_accepted_m4


def test_m5_orchestrator_fixture_cert_complete(tmp_path: Path) -> None:
    project, persistent, parent_run_id = make_synthetic_accepted_m4(tmp_path)
    config = default_m5_config(
        parent_m4_run_id=parent_run_id,
        run_id='M5-synthetic-cert',
        mode='cpu_fixture_cert',
    )
    orchestrator = create_or_resume_m5(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == M5_COMPLETE
    assert report['milestone_status'] == ONE_STEP_CERTIFIED
    assert report['certification_status'] == ONE_STEP_CERTIFIED
    assert report['verdict']['independent_verifier'] == 'PASS'
    assert (orchestrator.package_root / 'verdict.json').is_file()
    assert (orchestrator.run_root / 'reports' / 'M5_report.json').is_file()
    assert (
        orchestrator.run_root / 'reports' / 'M5_independent_verifier_report.json'
    ).is_file()


def test_m5_orchestrator_fixture_not_certified_complete(tmp_path: Path) -> None:
    project, persistent, parent_run_id = make_synthetic_accepted_m4(tmp_path)
    config = default_m5_config(
        parent_m4_run_id=parent_run_id,
        run_id='M5-synthetic-not-cert',
        mode='cpu_fixture_not_certified',
    )
    orchestrator = create_or_resume_m5(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == M5_COMPLETE
    assert report['milestone_status'] == NOT_CERTIFIED
    assert report['certification_status'] == NOT_CERTIFIED
    assert report['verdict']['independent_verifier'] == 'PASS'


def test_m5_orchestrator_evaluates_handoff_obligations(
    tmp_path: Path,
) -> None:
    project, persistent, parent_run_id = make_synthetic_accepted_m4(tmp_path)
    # paperspace mode enforces frozen IDs; use a non-paperspace production-like mode.
    config = default_m5_config(
        parent_m4_run_id=parent_run_id,
        run_id='M5-synthetic-open',
        mode='live_parent_inventory',
    )
    orchestrator = create_or_resume_m5(persistent, config, project)
    report = orchestrator.run_until_checkpoint()
    assert report['phase'] == 'M5_IN_PROGRESS'
    assert report['implementation_status'] == 'M5_OBLIGATION_EVALUATION_COMPLETE'
    assert report['certification_status'] == NOT_CERTIFIED
    assert (
        orchestrator.run_root / 'reports' / 'M5_parent_artifact_inventory.json'
    ).is_file()
    assert (orchestrator.run_root / 'reports' / 'M5_schema_mapping.json').is_file()
    obligation_path = (
        orchestrator.run_root / 'reports' / 'M5_obligation_report.json'
    )
    assert obligation_path.is_file()
    evaluation = report['parent']['obligation_evaluation']
    assert 'closed_obligations' in evaluation
    assert 'open_obligations' in evaluation
    # Synthetic M4 lacks the full M1–M3 chain; some obligations stay open.
    assert evaluation['all_closed'] is False
    assert report['verdict']['closed_for_M5'] == evaluation['closed_obligations']
