from __future__ import annotations

from pathlib import Path

from src.common import atomic_write_json
from src.m7_archive import (
    REASON_S0_NO_SELECTION,
    archive_from_sweep,
    is_archived,
    write_advance,
    write_archive,
)
from src.m7_candidate_queue import (
    list_queue_rows,
    m2_ready,
    next_actionable_candidate,
)
from src.s0_series import run_s0_for_package


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
            _ranking_row('CAND-gated', 4, 0.1),
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
        atomic_write_json(pkg / 'child_run_ids.json', {
            'M2': f'M2-{cand}',
        })
        atomic_write_json(pkg / 'm3_config_overrides.json', {
            'j2_max': 2,
            'sector_count': 729,
            'operator_dimension': 46656,
            'target_rank': 16,
        })
    return persist, search


def test_archive_write_and_skip(tmp_path: Path) -> None:
    persist, search = _make_search(tmp_path)
    pkg = search / 'auto_execute' / 'CAND-a'
    sweep = pkg / 'rank_sweep' / 'SWEEP-test'
    sweep.mkdir(parents=True)
    arch = archive_from_sweep(
        pkg,
        sweep,
        selection_status='NO_SELECTION',
        selection_reasons=['rank=16: q_optimistic>=1'],
    )
    assert arch['reason'] == REASON_S0_NO_SELECTION
    assert is_archived(pkg)
    log = (search / 'reports' / 's0_series_log.jsonl').read_text(encoding='utf-8')
    assert 'ARCHIVE' in log

    nxt = next_actionable_candidate(search, persistent_root=persist)
    assert nxt is not None
    assert nxt['candidate_id'] == 'CAND-b'


def test_next_skips_archived_and_gated(tmp_path: Path) -> None:
    persist, search = _make_search(tmp_path)
    write_archive(
        search / 'auto_execute' / 'CAND-a',
        reason=REASON_S0_NO_SELECTION,
        details={},
    )
    rows = list_queue_rows(search, persistent_root=persist)
    ids = [row['candidate_id'] for row in rows]
    assert 'CAND-gated' in ids
    gated = next(row for row in rows if row['candidate_id'] == 'CAND-gated')
    assert gated['staged_executable'] is False
    nxt = next_actionable_candidate(search, persistent_root=persist)
    assert nxt['candidate_id'] == 'CAND-b'


def test_need_m2_without_acceptance(tmp_path: Path) -> None:
    persist, search = _make_search(tmp_path)
    project = tmp_path / 'project'
    project.mkdir()
    pkg = search / 'auto_execute' / 'CAND-a'
    assert m2_ready(pkg, persist) is False
    outcome = run_s0_for_package(
        project_root=project,
        persistent_root=persist,
        package_root=pkg,
        candidate_id='CAND-a',
    )
    assert outcome['status'] == 'NEED_M2'
    assert outcome['next_notebook'].startswith('73_')


def test_advance_and_archive_helpers(tmp_path: Path) -> None:
    persist, search = _make_search(tmp_path)
    pkg = search / 'auto_execute' / 'CAND-b'
    sweep = pkg / 'rank_sweep' / 'SWEEP-ok'
    sweep.mkdir(parents=True)
    adv = write_advance(
        pkg,
        selected_rank=36,
        sweep_root=sweep,
        selection_reasons=['cluster terminus'],
    )
    assert adv['selected_rank'] == 36
    assert (pkg / 'ADVANCE.json').is_file()
    # Advanced candidates are skipped by default.
    write_archive(
        search / 'auto_execute' / 'CAND-a',
        reason=REASON_S0_NO_SELECTION,
    )
    nxt = next_actionable_candidate(search, persistent_root=persist)
    assert nxt is None
