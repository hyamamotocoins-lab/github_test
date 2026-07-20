"""Advance Campaign B SELECTED packages toward M3–M6 (screening lineage).

Does NOT run production M6. Does NOT emit CERTIFIED.
CPU-safe steps (parallel with notebook 89 mass explore):
  1. Discover selected packages under campaign_b/*/selected/
  2. Build S2 lineage plans (child M3–M6 run ids)
  3. Evaluate S2 fixture residual against frozen parent M6 package when available
  4. Write ADVANCE.json + next-step hints (74/75/76; production M6 blocked)

Optional GPU M3 is off by default so this can run beside 89.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, utc_now
from ..m6_status import M6_RUN_ID_FROZEN
from ..m7_lineage import (
    build_s2_lineage_plan,
    evaluate_s2_fixture_residual,
    write_lineage_plan,
)
from ..m7_status import CHANGE_S2
from .schemas import screening_only_payload
from .resume_pointer import read_resume_id


class AdvanceSelectedError(RuntimeError):
    """Raised when advancement bookkeeping fails closed."""


def _advance_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_advance'


def discover_selected_packages(persistent_root: Path) -> list[Path]:
    root = Path(persistent_root) / 'campaign_b'
    if not root.is_dir():
        return []
    found: list[Path] = []
    for campaign in sorted(root.iterdir()):
        if not campaign.is_dir() or campaign.name.startswith('_'):
            continue
        selected = campaign / 'selected'
        if not selected.is_dir():
            continue
        for package in sorted(selected.iterdir()):
            if package.is_dir() and (package / 'candidate_manifest.json').is_file():
                found.append(package)
    return found


def _q_upper_from_package(package: Path) -> float:
    for name in ('s0_result.json', 'independent_verification.json', 'screening_result.json'):
        path = package / name
        if not path.is_file():
            continue
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        for key in ('q_upper', 'estimated_q', 'primary_q', 'verify_q'):
            if payload.get(key) is not None:
                try:
                    return float(payload[key])
                except (TypeError, ValueError):
                    pass
        verify = payload.get('verify_result')
        if isinstance(verify, dict) and verify.get('q_upper') is not None:
            try:
                return float(verify['q_upper'])
            except (TypeError, ValueError):
                pass
    return float('inf')


def find_parent_m6_package(
    persistent_root: Path,
    *,
    parent_m6_run_id: str | None = None,
) -> Path | None:
    """Locate a package that has final_influence_matrix.json for fixture residual."""
    run_id = parent_m6_run_id or M6_RUN_ID_FROZEN
    run_root = Path(persistent_root) / 'runs' / run_id
    candidates = [
        run_root / 'certificate_package',
        run_root / 'final_certificate',
        run_root / 'package',
        run_root,
    ]
    # Also scan recent M6 runs if frozen path missing.
    runs = Path(persistent_root) / 'runs'
    if runs.is_dir():
        for path in sorted(runs.glob('M6-*'), reverse=True)[:20]:
            candidates.extend([
                path / 'certificate_package',
                path / 'final_certificate',
                path,
            ])
    for path in candidates:
        if (path / 'final_influence_matrix.json').is_file() and (
            path / 'final_bound.json'
        ).is_file():
            return path
    return None


def _load_candidate(package: Path) -> dict[str, Any]:
    manifest = read_json(package / 'candidate_manifest.json')
    if not isinstance(manifest, dict):
        raise AdvanceSelectedError(f'bad candidate_manifest: {package}')
    scheme = manifest.get('scheme')
    if not isinstance(scheme, dict) and (package / 'scheme.json').is_file():
        scheme = read_json(package / 'scheme.json')
    if isinstance(scheme, dict):
        manifest = {**manifest, 'scheme': scheme}
        if scheme.get('change_class') is None:
            scheme['change_class'] = CHANGE_S2
            manifest['scheme'] = scheme
    return manifest


def advance_one_selected(
    package: Path,
    *,
    persistent_root: Path,
    search_run_id: str,
    parent_m6_run_id: str,
    parent_m6_package: Path | None,
    parent_rank: int = 16,
    force: bool = False,
) -> dict[str, Any]:
    package = Path(package)
    advance_path = package / 'ADVANCE.json'
    if advance_path.is_file() and not force:
        existing = read_json(advance_path)
        if isinstance(existing, dict) and existing.get('status') in {
            'LINEAGE_PLANNED', 'FIXTURE_RESIDUAL_DONE', 'READY_FOR_M3',
        }:
            return {
                'package': str(package),
                'skipped': True,
                'reason': 'already_advanced',
                'status': existing.get('status'),
                **screening_only_payload(),
            }

    candidate = _load_candidate(package)
    candidate_id = str(candidate.get('candidate_id') or package.name)
    plan = build_s2_lineage_plan(
        candidate,
        parent_m6_run_id=parent_m6_run_id,
        search_run_id=search_run_id,
    )
    write_lineage_plan(package / 'lineage_plan.json', plan)

    fixture: dict[str, Any] | None = None
    fixture_error: str | None = None
    if parent_m6_package is not None:
        try:
            # evaluate_s2 expects change_class on scheme
            scheme = dict(candidate.get('scheme') or {})
            scheme['change_class'] = CHANGE_S2
            cand = {**candidate, 'scheme': scheme}
            fixture = evaluate_s2_fixture_residual(
                parent_m6_package,
                cand,
                parent_rank=parent_rank,
            )
            atomic_write_json(package / 'fixture_residual_result.json', fixture)
        except Exception as exc:  # noqa: BLE001 — continue other candidates
            fixture_error = f'{type(exc).__name__}: {exc}'

    q_screen = _q_upper_from_package(package)
    q_fixture = None
    if isinstance(fixture, dict):
        raw = fixture.get('q_cert_upper')
        if raw is None:
            raw = fixture.get('q_upper')
        try:
            q_fixture = float(raw) if raw is not None else None
        except (TypeError, ValueError):
            q_fixture = None

    status = 'LINEAGE_PLANNED'
    if fixture is not None:
        status = 'FIXTURE_RESIDUAL_DONE'
    if (package / 'm2_binding.json').is_file():
        binding = read_json(package / 'm2_binding.json')
        if isinstance(binding, dict) and binding.get('status') in {
            'READY_SHARED', 'READY',
        }:
            status = 'READY_FOR_M3'

    advance = {
        'schema_version': 1,
        'candidate_id': candidate_id,
        'status': status,
        'advanced_at': utc_now(),
        'search_run_id': search_run_id,
        'parent_m6_run_id': parent_m6_run_id,
        'parent_m6_package': str(parent_m6_package) if parent_m6_package else None,
        'q_screen_upper': None if q_screen == float('inf') else q_screen,
        'q_fixture_upper': q_fixture,
        'lineage_plan_path': str(package / 'lineage_plan.json'),
        'child_run_ids': plan.get('child_run_ids'),
        'm4_geometry_compatible': plan.get('m4_geometry_compatible'),
        'fixture_error': fixture_error,
        'next_steps': [
            'Review lineage_plan.json and fixture_residual_result.json',
            'If READY_FOR_M3: notebook 74 (staged M3) with shared M2 audit',
            'Then 75 (M4), 76 (M5)',
            'Production M6 (81) blocked until ONE_STEP_CERTIFIED',
        ],
        'prohibited': [
            'production M6',
            'CERTIFIED claim',
            'continuum / mass-gap claim',
        ],
        **screening_only_payload(),
    }
    atomic_write_json(advance_path, advance)
    atomic_write_json(package / 'advance_result.json', advance)
    return {
        'package': str(package),
        'candidate_id': candidate_id,
        'skipped': False,
        'status': status,
        'q_screen_upper': advance['q_screen_upper'],
        'q_fixture_upper': q_fixture,
        **screening_only_payload(),
    }


def run_advance_selected(
    *,
    persistent_root: Path,
    max_candidates: int | None = None,
    force: bool = False,
    parent_m6_run_id: str | None = None,
    parent_rank: int = 16,
    only_campaign_run_id: str | None = None,
) -> dict[str, Any]:
    persistent_root = Path(persistent_root)
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        needle = f'/campaign_b/{only_campaign_run_id}/'
        packages = [p for p in packages if needle in str(p) or p.parts[-3] == only_campaign_run_id]

    ranked = sorted(packages, key=lambda p: (_q_upper_from_package(p), str(p)))
    if max_candidates is not None:
        ranked = ranked[: int(max_candidates)]

    parent_id = parent_m6_run_id or M6_RUN_ID_FROZEN
    parent_pkg = find_parent_m6_package(persistent_root, parent_m6_run_id=parent_id)
    search_run_id = (
        only_campaign_run_id
        or read_resume_id(persistent_root)
        or 'M7-B-ADVANCE'
    )

    results: list[dict[str, Any]] = []
    for package in ranked:
        try:
            results.append(
                advance_one_selected(
                    package,
                    persistent_root=persistent_root,
                    search_run_id=str(search_run_id),
                    parent_m6_run_id=parent_id,
                    parent_m6_package=parent_pkg,
                    parent_rank=parent_rank,
                    force=force,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append({
                'package': str(package),
                'error': f'{type(exc).__name__}: {exc}',
                **screening_only_payload(),
            })

    advanced = [r for r in results if not r.get('skipped') and not r.get('error')]
    session = {
        'schema_version': 1,
        'session_id': f"ADV-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': utc_now(),
        'finished_at': utc_now(),
        'search_run_id': search_run_id,
        'parent_m6_run_id': parent_id,
        'parent_m6_package': str(parent_pkg) if parent_pkg else None,
        'discovered': len(packages),
        'attempted': len(ranked),
        'advanced': len(advanced),
        'skipped': sum(1 for r in results if r.get('skipped')),
        'errors': sum(1 for r in results if r.get('error')),
        'ready_for_m3': sum(1 for r in results if r.get('status') == 'READY_FOR_M3'),
        'fixture_done': sum(1 for r in results if r.get('status') == 'FIXTURE_RESIDUAL_DONE'),
        'best_q_screen': min(
            (r['q_screen_upper'] for r in results if isinstance(r.get('q_screen_upper'), (int, float))),
            default=None,
        ),
        'best_q_fixture': min(
            (r['q_fixture_upper'] for r in results
             if isinstance(r.get('q_fixture_upper'), (int, float))),
            default=None,
        ),
        'results': results[:200],  # cap ledger size
        'note': (
            'CPU advancement only. Production M6 forbidden. '
            'Run notebook 74+ for GPU M3 on READY_FOR_M3 packages.'
        ),
        **screening_only_payload(),
    }
    root = _advance_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_ADVANCE_SESSION.json', session)
    atomic_write_json(root / f"{session['session_id']}_summary.json", session)
    return session


def main(argv: list[str] | None = None) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(description='Advance Campaign B SELECTED toward M3–M6 plan')
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument('--max-candidates', type=int, default=None)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--campaign-run-id', default=None)
    parser.add_argument('--parent-m6-run-id', default=None)
    args = parser.parse_args(argv)
    session = run_advance_selected(
        persistent_root=Path(args.persistent_root),
        max_candidates=args.max_candidates,
        force=args.force,
        parent_m6_run_id=args.parent_m6_run_id,
        only_campaign_run_id=args.campaign_run_id,
    )
    print(json.dumps({
        'session_id': session.get('session_id'),
        'discovered': session.get('discovered'),
        'attempted': session.get('attempted'),
        'advanced': session.get('advanced'),
        'ready_for_m3': session.get('ready_for_m3'),
        'fixture_done': session.get('fixture_done'),
        'best_q_screen': session.get('best_q_screen'),
        'best_q_fixture': session.get('best_q_fixture'),
        'parent_m6_package': session.get('parent_m6_package'),
        'certification_status': session.get('certification_status'),
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
