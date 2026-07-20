"""M6 report helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import atomic_write_json, atomic_write_text, utc_now


def write_m6_report_package(
    run_root: Path,
    *,
    report: Mapping[str, Any],
    independent_report: Mapping[str, Any] | None = None,
    lock: Mapping[str, Any] | None = None,
) -> dict[str, Path]:
    reports = run_root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    paths = {
        'json': reports / 'M6_report.json',
        'md': reports / 'M6_report.md',
    }
    atomic_write_json(paths['json'], dict(report))
    atomic_write_text(paths['md'], render_m6_markdown(report))
    if independent_report is not None:
        atomic_write_json(
            reports / 'M6_independent_verifier_report.json',
            dict(independent_report),
        )
    if lock is not None:
        atomic_write_json(reports / 'M6_lock.json', dict(lock))
    return paths


def render_m6_markdown(report: Mapping[str, Any]) -> str:
    verdict = report.get('verdict', {})
    lines = [
        '# M6 report',
        '',
        f"- generated_at: `{report.get('generated_at', utc_now())}`",
        f"- parent M5 run ID: `{report.get('parent_m5_run_id')}`",
        f"- M6 run ID: `{report.get('run_id')}`",
        f"- phase: `{report.get('phase')}`",
        f"- milestone_status: `{report.get('milestone_status')}`",
        f"- certification_status: `{report.get('certification_status')}`",
        f"- implementation_status: `{report.get('implementation_status')}`",
        f"- num_steps: `{report.get('num_steps')}`",
        f"- independent_verifier: `{verdict.get('independent_verifier')}`",
        f"- q_cert_upper: `{verdict.get('q_cert_upper')}`",
        f"- margin_lower: `{verdict.get('margin_lower')}`",
        '',
        '## Scope limitation',
        '',
        'Finite-cutoff, finite-step truncated SU(2) RG only.',
        'No continuum / thermodynamic-limit / mass-gap claim.',
        '',
    ]
    return '\n'.join(lines) + '\n'
