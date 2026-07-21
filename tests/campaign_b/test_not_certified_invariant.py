from __future__ import annotations

import ast
from pathlib import Path

from src.campaign_b.errors import InvariantViolation
from src.campaign_b.schemas import assert_not_certified, assert_phase_allowed


REPO = Path(__file__).resolve().parents[2]
PKG = REPO / 'src' / 'campaign_b'


def test_not_certified_required() -> None:
    assert_not_certified(
        {'certification_status': 'NOT_CERTIFIED', 'claim_scope': 'SCREENING_ONLY'},
        context='t',
    )
    try:
        assert_not_certified({'certification_status': 'CERTIFIED'}, context='t')
        raised = False
    except InvariantViolation:
        raised = True
    assert raised


def test_m6_phase_forbidden() -> None:
    try:
        assert_phase_allowed('M6_PRODUCTION')
        raised = False
    except InvariantViolation:
        raised = True
    assert raised


def test_no_m6_or_campaign_c_imports() -> None:
    # Staged live_parent M6 (notebook 94 / pipeline 95) may import the
    # orchestrator. Production paperspace gate helpers remain forbidden.
    allowed_m6_orchestrator = {'m6_batch.py'}
    forbidden_modules = {
        'm6_orchestrator',
        'm7_auto_execute',
    }
    forbidden_names = {
        'generate_campaign_c_candidates',
        'campaign_c_search_space',
        'run_production_m6',
    }
    for path in PKG.glob('*.py'):
        tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if path.name in allowed_m6_orchestrator:
                        continue
                    assert 'm6_orchestrator' not in alias.name
            if isinstance(node, ast.ImportFrom):
                mod = node.module or ''
                for part in forbidden_modules:
                    if part == 'm6_orchestrator' and path.name in allowed_m6_orchestrator:
                        continue
                    assert part not in mod, f'{path.name} imports {mod}'
                for alias in node.names:
                    assert alias.name not in forbidden_names, (
                        f'{path.name} imports {alias.name}'
                    )
