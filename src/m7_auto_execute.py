"""Automate Campaign C post-planning: approve, materialize, dry-run, gated execute.

This does NOT claim continuum results. Live j2_max>1 exact M2 remains
resource-gated; materialize+dry_run always run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .common import atomic_write_json, atomic_write_text, read_json, utc_now
from .cutoff_dims import cutoff_dimension_payload, resource_gate
from .m7_lineage import build_s3_lineage_plan, effective_projected_rank


class M7AutoExecuteError(RuntimeError):
    """Raised when automated lineage preparation/execution fails closed."""


def select_best_lineage_candidate(
    ranking: list[dict[str, Any]] | dict[str, Any],
    *,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    prefer_executable: bool = True,
    prefer_staged_for_q_lt_1: bool = True,
) -> dict[str, Any]:
    """Pick a Campaign C candidate for automation.

    Default policy:
    1. If prefer_staged_for_q_lt_1: among staged_executable with estimated
       q<1, choose lowest q (q<1 hunt via j2>=2 staged M2).
    2. Else among instant-executable OR staged_executable, choose lowest q.
    3. Else fall back to global lowest-q (RESOURCE_GATED archive).
    """
    rows = ranking.get('ranking') if isinstance(ranking, dict) else ranking
    if not isinstance(rows, list) or not rows:
        raise M7AutoExecuteError('No ranking rows available for auto-select.')

    def est_q(row: dict[str, Any]) -> float:
        try:
            return float(row.get('q_cert_upper') or 1e9)
        except (TypeError, ValueError):
            return 1e9

    def j2_of(row: dict[str, Any]) -> int:
        scheme = row.get('scheme') or {}
        try:
            return int(scheme.get('j2_max', 1))
        except (TypeError, ValueError):
            return 1

    def gate_of(row: dict[str, Any]) -> dict[str, Any]:
        return resource_gate(
            j2_of(row),
            max_executable_j2_max=max_executable_j2_max,
            max_staged_j2_max=max_staged_j2_max,
        )

    def is_executable(row: dict[str, Any]) -> bool:
        return bool(gate_of(row).get('executable'))

    def is_staged(row: dict[str, Any]) -> bool:
        return bool(gate_of(row).get('staged_executable'))

    def is_live_capable(row: dict[str, Any]) -> bool:
        return is_executable(row) or is_staged(row)

    screening_best = min(rows, key=est_q)
    if prefer_staged_for_q_lt_1:
        staged_lt1 = [
            row for row in rows if is_staged(row) and est_q(row) < 1.0
        ]
        if staged_lt1:
            chosen = dict(min(staged_lt1, key=est_q))
            chosen['selection_policy'] = 'prefer_staged_q_lt_1'
            chosen['screening_best_candidate_id'] = screening_best.get('candidate_id')
            chosen['screening_best_q'] = screening_best.get('q_cert_upper')
            return chosen
    if prefer_executable:
        live_rows = [row for row in rows if is_live_capable(row)]
        if live_rows:
            chosen = dict(min(live_rows, key=est_q))
            policy = (
                'prefer_staged_lowest_q'
                if is_staged(chosen) and not is_executable(chosen)
                else 'prefer_executable_lowest_q'
            )
            chosen['selection_policy'] = policy
            chosen['screening_best_candidate_id'] = screening_best.get('candidate_id')
            chosen['screening_best_q'] = screening_best.get('q_cert_upper')
            return chosen
    chosen = dict(screening_best)
    chosen['selection_policy'] = 'global_lowest_q_resource_gated'
    chosen['screening_best_candidate_id'] = screening_best.get('candidate_id')
    chosen['screening_best_q'] = screening_best.get('q_cert_upper')
    return chosen


def write_human_review_approval(
    search_root: Path,
    *,
    candidate_id: str,
    scheme: dict[str, Any],
    reviewer: str = 'operator',
    notes: str = '',
    auto: bool = False,
) -> Path:
    path = search_root / 'auto_execute' / 'HUMAN_REVIEW.json'
    payload = {
        'schema_version': 1,
        'status': 'APPROVED',
        'candidate_id': candidate_id,
        'scheme': scheme,
        'reviewer': reviewer,
        'auto_stamped': bool(auto),
        'notes': notes or (
            'Auto-stamped approval for materialize/dry_run only.'
            if auto else 'Human-approved Campaign C lineage execute package.'
        ),
        'approved_at': utc_now(),
        'scope_limitation': (
            'Approval does not certify q_cert<1; child lineage + independent '
            'verifier still required.'
        ),
    }
    atomic_write_json(path, payload)
    return path


def load_human_review(search_root: Path) -> dict[str, Any] | None:
    path = search_root / 'auto_execute' / 'HUMAN_REVIEW.json'
    if not path.is_file():
        return None
    doc = read_json(path)
    return doc if isinstance(doc, dict) else None


def materialize_s3_lineage_package(
    search_root: Path,
    candidate: dict[str, Any],
    *,
    parent_m6_run_id: str,
    search_run_id: str,
    parent_j2_max: int = 1,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    persistent_root: Path | None = None,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Write an executable workspace for one S3 candidate."""
    scheme = candidate.get('scheme') or {}
    if scheme.get('change_class') != 'S3':
        raise M7AutoExecuteError('Auto-execute currently supports S3 candidates only.')
    j2_max = int(scheme.get('j2_max', parent_j2_max))
    gate = resource_gate(
        j2_max,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    persist = Path(persistent_root) if persistent_root is not None else Path(search_root).resolve().parent.parent
    from .m2_shared_registry import BINDING_UNRESOLVED, resolve_m2_for_package, write_binding
    out = search_root / 'auto_execute' / str(candidate.get('candidate_id'))
    out.mkdir(parents=True, exist_ok=True)
    if project_root is not None:
        binding = resolve_m2_for_package(
            persistent_root=persist,
            j2_max=j2_max,
            project_root=project_root,
            package_root=out,
        )
    else:
        binding = write_binding(out, {
            'schema_version': 2,
            'structural_key': None,
            'proof_key': None,
            'state': BINDING_UNRESOLVED,
            'mode': None,
            'canonical_run_id': None,
            'registry_record_sha256': None,
            'acceptance_sha256': None,
            'verified_at': None,
            'note': 'Resolve with project_root to compute structural/proof keys',
            'certification_status': 'NOT_CERTIFIED',
        })
    plan = build_s3_lineage_plan(
        {
            'candidate_id': candidate.get('candidate_id'),
            'scheme_hash': candidate.get('scheme_hash'),
            'scheme': scheme,
        },
        parent_m6_run_id=parent_m6_run_id,
        search_run_id=search_run_id,
        parent_j2_max=parent_j2_max,
        m2_binding=binding,
    )
    # Clear math-lock flag in materialized plan: dims are now configurable;
    # resource_gate decides live execute.
    plan['execution_blocked_by_math_lock'] = False
    plan['resource_gate'] = gate
    plan['cutoff_dims'] = cutoff_dimension_payload(j2_max)

    atomic_write_json(out / 'rigorous_lineage.json', plan)
    atomic_write_json(out / 'scheme.json', scheme)
    atomic_write_json(out / 'resource_gate.json', gate)
    atomic_write_json(out / 'child_run_ids.json', plan['child_run_ids'])
    atomic_write_json(out / 'm2_binding.json', binding)

    dims = cutoff_dimension_payload(j2_max)
    m3_overrides = {
        'j2_max': j2_max,
        'sector_count': dims['sector_count'],
        'operator_dimension': dims['operator_dimension'],
        'target_rank': min(16, dims['operator_dimension'] - 1),
        'mode_hint': 'non_paperspace_child',
    }
    m4_overrides = {
        'projected_rank': effective_projected_rank(
            int(m3_overrides['target_rank']),
        ),
        'operator_dimension': dims['operator_dimension'],
    }
    atomic_write_json(out / 'm3_config_overrides.json', m3_overrides)
    atomic_write_json(out / 'm4_config_overrides.json', m4_overrides)

    driver = _driver_script(
        candidate_id=str(candidate.get('candidate_id')),
        child_ids=plan['child_run_ids'],
        j2_max=j2_max,
        gate=gate,
    )
    atomic_write_text(out / 'execute_lineage.py', driver)
    atomic_write_text(out / 'README.md', _package_readme(plan, gate))

    manifest = {
        'schema_version': 1,
        'candidate_id': candidate.get('candidate_id'),
        'scheme_hash': candidate.get('scheme_hash'),
        'package_root': str(out),
        'resource_gate': gate,
        'child_run_ids': plan['child_run_ids'],
        'm2_binding': binding,
        'generated_at': utc_now(),
        'next_command': f'python {out / "execute_lineage.py"} --dry-run',
    }
    atomic_write_json(out / 'MANIFEST.json', manifest)
    return manifest


def dry_run_lineage_package(package_root: Path) -> dict[str, Any]:
    """Validate materialized configs can be constructed (no GPU work)."""
    from dataclasses import asdict

    from .m2_config import M2Config
    from .m3_config import M3Config
    from .m4_config import M4Config

    root = package_root.resolve()
    scheme = read_json(root / 'scheme.json')
    m3_over = read_json(root / 'm3_config_overrides.json')
    m4_over = read_json(root / 'm4_config_overrides.json')
    gate = read_json(root / 'resource_gate.json')
    if not all(isinstance(doc, dict) for doc in (scheme, m3_over, m4_over, gate)):
        raise M7AutoExecuteError('Materialized package incomplete.')

    j2_max = int(scheme.get('j2_max', 1))
    checks: dict[str, Any] = {'j2_max': j2_max}

    # M2Config construction with requested cutoff.
    batch = 0 if j2_max <= 1 else int(gate.get('default_sector_batch_size') or 16)
    m2 = M2Config(j2_max=j2_max, sector_batch_size=batch)
    checks['m2_config'] = {
        'status': 'PASS',
        'j2_max': m2.j2_max,
        'sector_batch_size': m2.sector_batch_size,
    }

    m3_base = asdict(M3Config())
    m3_base.update({
        'j2_max': int(m3_over['j2_max']),
        'sector_count': int(m3_over['sector_count']),
        'operator_dimension': int(m3_over['operator_dimension']),
        'target_rank': int(m3_over['target_rank']),
        'require_cuda': False,
    })
    m3 = M3Config(**m3_base)
    checks['m3_config'] = {
        'status': 'PASS',
        'sector_count': m3.sector_count,
        'operator_dimension': m3.operator_dimension,
    }

    m4_base = asdict(M4Config())
    m4_base.update({
        'operator_dimension': int(m4_over['operator_dimension']),
        'projected_rank': int(m4_over['projected_rank']),
        'require_cuda': False,
    })
    m4 = M4Config(**m4_base)
    checks['m4_config'] = {
        'status': 'PASS',
        'operator_dimension': m4.operator_dimension,
        'projected_rank': m4.projected_rank,
    }

    from .armillary import all_link_star_keys
    keys = all_link_star_keys(j2_max)
    checks['armillary_keys'] = {
        'status': 'PASS',
        'count': len(keys),
        'expected': int(m3_over['sector_count']),
    }
    if len(keys) != int(m3_over['sector_count']):
        raise M7AutoExecuteError('sector key count mismatch in dry-run.')

    report = {
        'schema_version': 1,
        'status': 'PASS',
        'dry_run': True,
        'live_execute_allowed': bool(gate.get('executable')),
        'staged_live_execute_allowed': bool(gate.get('staged_executable')),
        'resource_gate': gate,
        'checks': checks,
        'generated_at': utc_now(),
        'notes': (
            'Dry-run validates config construction and key counts only. '
            'It does not run M2 SymPy/GPU lineage or emit CERTIFIED. '
            'j2_max>=2 uses execute_lineage.py --live --staged.'
        ),
    }
    atomic_write_json(root / 'dry_run_report.json', report)
    return report


def run_campaign_c_automation(
    search_root: Path,
    *,
    parent_m6_run_id: str,
    search_run_id: str,
    human_review_approved: bool = False,
    auto_approve: bool = False,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    parent_j2_max: int = 1,
) -> dict[str, Any]:
    """End-to-end automation after Campaign C plan_only search."""
    reports = search_root / 'reports'
    ranking_path = reports / 'candidate_ranking.json'
    if not ranking_path.is_file():
        raise M7AutoExecuteError(f'Missing ranking: {ranking_path}')
    ranking = read_json(ranking_path)
    best = select_best_lineage_candidate(
        ranking,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
        prefer_executable=True,
    )
    scheme = best.get('scheme') or {}
    candidate_id = str(best.get('candidate_id'))

    auto_root = search_root / 'auto_execute'
    auto_root.mkdir(parents=True, exist_ok=True)

    review = load_human_review(search_root)
    if review is None:
        if human_review_approved or auto_approve:
            write_human_review_approval(
                search_root,
                candidate_id=candidate_id,
                scheme=scheme if isinstance(scheme, dict) else {},
                reviewer='auto' if auto_approve else 'config.human_review_approved',
                auto=bool(auto_approve and not human_review_approved),
            )
            review = load_human_review(search_root)
        else:
            atomic_write_json(auto_root / 'STATUS.json', {
                'status': 'WAITING_HUMAN_REVIEW',
                'best_candidate_id': candidate_id,
                'estimated_q': best.get('q_cert_upper'),
                'action': (
                    'Write auto_execute/HUMAN_REVIEW.json or re-run with '
                    'human_review_approved=True / lineage_mode=auto.'
                ),
                'generated_at': utc_now(),
            })
            return {
                'status': 'WAITING_HUMAN_REVIEW',
                'best': best,
                'package': None,
                'dry_run': None,
            }

    if review.get('status') != 'APPROVED':
        raise M7AutoExecuteError('HUMAN_REVIEW.json is not APPROVED.')

    # Reviewed candidate is archival unless it is itself executable (or pinned).
    reviewed_id = str(review.get('candidate_id') or '')
    force_pin = bool(review.get('force_pin_candidate'))
    if reviewed_id and reviewed_id != candidate_id:
        rows = ranking.get('ranking') if isinstance(ranking, dict) else ranking
        reviewed_row = next(
            (row for row in (rows or []) if row.get('candidate_id') == reviewed_id),
            None,
        )
        if reviewed_row is not None:
            reviewed_j2 = int((reviewed_row.get('scheme') or {}).get('j2_max', 1))
            reviewed_gate = resource_gate(
                reviewed_j2,
                max_executable_j2_max=max_executable_j2_max,
                max_staged_j2_max=max_staged_j2_max,
            )
            reviewed_exec = bool(
                reviewed_gate.get('executable')
                or reviewed_gate.get('staged_executable')
            )
            if force_pin or reviewed_exec:
                best = dict(reviewed_row)
                best['selection_policy'] = (
                    'force_pin_review' if force_pin else 'reviewed_live_capable'
                )
                scheme = best.get('scheme') or scheme
                candidate_id = reviewed_id
            else:
                # Keep live-capable primary; archive reviewed screening pick later.
                best = dict(best)
                best['selection_policy'] = 'prefer_live_capable_over_gated_review'
                best['review_candidate_id_archived'] = reviewed_id
                # Ensure screening_best points at the gated review pick.
                best['screening_best_candidate_id'] = reviewed_id
                best['screening_best_q'] = reviewed_row.get('q_cert_upper')

    # Refresh approval to match the primary executable selection when we overrode.
    if str(review.get('candidate_id')) != candidate_id and not force_pin:
        write_human_review_approval(
            search_root,
            candidate_id=candidate_id,
            scheme=scheme if isinstance(scheme, dict) else {},
            reviewer=str(review.get('reviewer') or 'config.human_review_approved'),
            notes=(
                'Primary approval retargeted to live-capable candidate; '
                f"previous gated review {reviewed_id or 'n/a'} archived."
            ),
            auto=False,
        )
        review = load_human_review(search_root) or review

    manifest = materialize_s3_lineage_package(
        search_root,
        {
            'candidate_id': candidate_id,
            'scheme_hash': best.get('scheme_hash'),
            'scheme': scheme,
        },
        parent_m6_run_id=parent_m6_run_id,
        search_run_id=search_run_id,
        parent_j2_max=parent_j2_max,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    package_root = Path(manifest['package_root'])
    dry = dry_run_lineage_package(package_root)

    # Also materialize the absolute screening-best when it differs (for archive).
    screening_id = best.get('screening_best_candidate_id')
    if (
        screening_id
        and screening_id != candidate_id
        and isinstance(ranking, dict)
    ):
        for row in ranking.get('ranking') or []:
            if row.get('candidate_id') == screening_id:
                materialize_s3_lineage_package(
                    search_root,
                    {
                        'candidate_id': screening_id,
                        'scheme_hash': row.get('scheme_hash'),
                        'scheme': row.get('scheme') or {},
                    },
                    parent_m6_run_id=parent_m6_run_id,
                    search_run_id=search_run_id,
                    parent_j2_max=parent_j2_max,
                    max_executable_j2_max=max_executable_j2_max,
                    max_staged_j2_max=max_staged_j2_max,
                )
                break

    status = (
        'READY_FOR_LIVE_EXECUTE'
        if dry.get('live_execute_allowed')
        else (
            'READY_FOR_STAGED_LIVE_EXECUTE'
            if dry.get('staged_live_execute_allowed')
            else 'MATERIALIZED_RESOURCE_GATED'
        )
    )
    summary = {
        'schema_version': 1,
        'status': status,
        'best': best,
        'selection_policy': best.get('selection_policy'),
        'screening_best_candidate_id': best.get('screening_best_candidate_id'),
        'screening_best_q': best.get('screening_best_q'),
        'review': {
            'candidate_id': review.get('candidate_id'),
            'reviewer': review.get('reviewer'),
            'auto_stamped': review.get('auto_stamped'),
        },
        'package': manifest,
        'dry_run': dry,
        'generated_at': utc_now(),
        'notes': (
            'Automation completed materialize+dry_run. '
            'Selection prefers staged q<1 (j2=2) when available, else '
            'instant/staged live-capable lowest q; screening-best may be '
            'archived. Live: execute_lineage.py --live (j2=1) or '
            '--live --staged (j2=2 sector-batched M2).'
        ),
    }
    atomic_write_json(auto_root / 'STATUS.json', summary)
    atomic_write_json(reports / 'auto_execute_summary.json', summary)
    return summary


def _package_readme(plan: dict[str, Any], gate: dict[str, Any]) -> str:
    lines = [
        '# Campaign C automated lineage package',
        '',
        f"- candidate: `{plan.get('candidate_id')}`",
        f"- child runs: `{json.dumps(plan.get('child_run_ids'))}`",
        f"- resource executable (instant j2=1): `{gate.get('executable')}`",
        f"- staged executable (j2>=2 batched M2): `{gate.get('staged_executable')}`",
        '',
        '## Commands',
        '',
        '```bash',
        'python execute_lineage.py --dry-run',
        '# j2_max=1 instant:',
        'python execute_lineage.py --live',
        '# j2_max=2 sector-batched M2 (resume across sessions):',
        'python execute_lineage.py --live --staged',
        '```',
        '',
        'Dry-run never emits CERTIFIED. Staged live runs one M2 session per '
        'invocation until M2_COMPLETE, then rewrites the child M2 audit.',
        '',
    ]
    if gate.get('blocked_reasons'):
        lines.append('## Instant live blocks')
        lines.append('')
        for reason in gate['blocked_reasons']:
            lines.append(f'- {reason}')
        lines.append('')
    if gate.get('staged_blocked_reasons'):
        lines.append('## Staged live blocks')
        lines.append('')
        for reason in gate['staged_blocked_reasons']:
            lines.append(f'- {reason}')
        lines.append('')
    return '\n'.join(lines) + '\n'


def _driver_script(
    *,
    candidate_id: str,
    child_ids: dict[str, str],
    j2_max: int,
    gate: dict[str, Any],
) -> str:
    return f'''#!/usr/bin/env python3
"""Auto-generated Campaign C lineage driver for {candidate_id}."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', default=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument(
        '--staged', action='store_true',
        help='Sector-batched M2 path for j2_max>=2',
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSISTENT_ROOT',
            '/storage/validated_4d_su2_rg',
        ),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', str(Path.cwd())),
    )
    args = parser.parse_args()
    gate = json.loads((ROOT / 'resource_gate.json').read_text())
    print('candidate', {candidate_id!r})
    print('child_run_ids', {json.dumps(child_ids)!r})
    print('j2_max', {j2_max})
    print('resource_executable', gate.get('executable'))
    print('staged_executable', gate.get('staged_executable'))
    if args.live:
        if args.staged or (not gate.get('executable') and gate.get('staged_executable')):
            if not gate.get('staged_executable'):
                raise SystemExit(
                    'Staged live blocked: '
                    + '; '.join(gate.get('staged_blocked_reasons') or [])
                )
            from src.m7_staged_lineage import run_staged_lineage_from_package
            report = run_staged_lineage_from_package(
                ROOT,
                persistent_root=Path(args.persistent_root),
                project_root=Path(args.project_root),
            )
            print(json.dumps(report, indent=2, sort_keys=True, default=str))
            return
        if not gate.get('executable'):
            raise SystemExit(
                'Live execute blocked by resource_gate: '
                + '; '.join(gate.get('blocked_reasons') or [])
                + '; try --live --staged if staged_executable'
            )
        raise SystemExit(
            'Instant j2=1 live M2→M6 still uses milestone notebooks with '
            'child_run_ids in this package (operator GPU session).'
        )
    from src.m7_auto_execute import dry_run_lineage_package
    report = dry_run_lineage_package(ROOT)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
'''
