from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json, atomic_write_text
from src.cutoff_dims import operator_dimension, resource_gate, sector_count
from src.m7_auto_execute import (
    dry_run_lineage_package,
    materialize_s3_lineage_package,
    run_campaign_c_automation,
    select_best_lineage_candidate,
    write_human_review_approval,
)
from src.m7_config import default_m7_config
from src.m7_orchestrator import M7Orchestrator
from src.m7_status import M7_COMPLETE, M7_LINEAGE_PLANNED
from src.orchestrator import GOVERNING_DOCUMENTS, REFERENCE_ARTIFACTS


def _seed(project: Path) -> None:
    (project / 'src').mkdir(parents=True, exist_ok=True)
    (project / 'audit').mkdir(parents=True, exist_ok=True)
    for relative in (*GOVERNING_DOCUMENTS, *REFERENCE_ARTIFACTS):
        path = project / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            atomic_write_text(path, f'# fixture\n{relative}\n')


def test_cutoff_dims_match_pilot() -> None:
    assert sector_count(1) == 64
    assert operator_dimension(1) == 729
    assert sector_count(2) == 729
    assert operator_dimension(2) == 46656


def test_resource_gate_blocks_j2_gt_1_for_live() -> None:
    gate = resource_gate(4, max_executable_j2_max=2)
    assert gate['executable'] is False
    assert gate['sector_count'] == 15625


def test_campaign_c_auto_materialize_and_dry_run(tmp_path: Path) -> None:
    project = tmp_path / 'project'
    persist = tmp_path / 'persist'
    _seed(project)
    orch = M7Orchestrator(project, persist, default_m7_config(
        parent_m6_run_id='M6-fixture',
        run_id='M7-fixture-c-auto',
        mode='cpu_fixture',
        campaign='C',
        lineage_mode='auto',
        auto_approve_for_materialize=True,
        max_candidates_total=4,
        max_rigorous_replays=4,
        max_executable_j2_max=2,
    ))
    summary = orch.run_search()
    assert summary['phase'] == M7_COMPLETE
    assert summary['search_status'] == M7_LINEAGE_PLANNED
    auto = summary.get('auto_execute')
    assert isinstance(auto, dict)
    assert auto['status'] in {
        'MATERIALIZED_RESOURCE_GATED',
        'READY_FOR_LIVE_EXECUTE',
        'WAITING_HUMAN_REVIEW',
    }
    # With auto_approve, should not wait.
    assert auto['status'] != 'WAITING_HUMAN_REVIEW'
    assert auto.get('dry_run', {}).get('status') == 'PASS'
    package = Path(auto['package']['package_root'])
    assert (package / 'execute_lineage.py').is_file()
    assert (package / 'dry_run_report.json').is_file()


def test_select_best_prefers_executable_over_lower_gated_q() -> None:
    ranking = {
        'ranking': [
            {
                'candidate_id': 'CAND-gated',
                'q_cert_upper': '0.81',
                'scheme': {'change_class': 'S3', 'j2_max': 4},
            },
            {
                'candidate_id': 'CAND-exec',
                'q_cert_upper': '1.9',
                'scheme': {'change_class': 'S3', 'j2_max': 1},
            },
        ]
    }
    best = select_best_lineage_candidate(ranking, max_executable_j2_max=2)
    assert best['candidate_id'] == 'CAND-exec'
    assert best['selection_policy'] == 'prefer_executable_lowest_q'
    assert best['screening_best_candidate_id'] == 'CAND-gated'


def test_select_best_and_manual_materialize(tmp_path: Path) -> None:
    ranking = {
        'ranking': [
            {
                'candidate_id': 'CAND-1',
                'q_cert_upper': '1.5',
                'scheme_hash': 'sha256:a',
                'scheme': {
                    'change_class': 'S3',
                    'j2_max': 1,
                    'channel_policy': 'complete_at_cutoff',
                    'block_geometry': 'current',
                },
            },
            {
                'candidate_id': 'CAND-0',
                'q_cert_upper': '0.8',
                'scheme_hash': 'sha256:b',
                'scheme': {
                    'change_class': 'S3',
                    'j2_max': 4,
                    'channel_policy': 'certified_pruned',
                    'block_geometry': 'approved_geometry_B',
                },
            },
        ]
    }
    best = select_best_lineage_candidate(ranking, max_executable_j2_max=2)
    assert best['candidate_id'] == 'CAND-1'
    root = tmp_path / 'search'
    (root / 'reports').mkdir(parents=True)
    atomic_write_json(root / 'reports' / 'candidate_ranking.json', ranking)
    # Prior gated review should be overridden by executable policy.
    write_human_review_approval(
        root, candidate_id='CAND-0', scheme=ranking['ranking'][1]['scheme'],
        reviewer='test',
    )
    summary = run_campaign_c_automation(
        root,
        parent_m6_run_id='M6-x',
        search_run_id='M7-x',
        human_review_approved=True,
        max_executable_j2_max=2,
    )
    assert summary['status'] == 'READY_FOR_LIVE_EXECUTE'
    assert summary['best']['candidate_id'] == 'CAND-1'
    dry = dry_run_lineage_package(Path(summary['package']['package_root']))
    assert dry['status'] == 'PASS'
    assert dry['live_execute_allowed'] is True
