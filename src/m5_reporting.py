"""M5 report and freeze helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .common import atomic_write_json, atomic_write_text, utc_now
from .m5_status import M5_COMPLETE, NOT_CERTIFIED, ONE_STEP_CERTIFIED


def write_m5_report_package(
    run_root: Path,
    *,
    report: Mapping[str, Any],
    independent_report: Mapping[str, Any] | None,
    certificate_manifest: Mapping[str, Any] | None,
) -> dict[str, Path]:
    reports = run_root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)
    paths = {
        'json': reports / 'M5_report.json',
        'md': reports / 'M5_report.md',
    }
    atomic_write_json(paths['json'], dict(report))
    atomic_write_text(paths['md'], render_m5_markdown(report))
    if independent_report is not None:
        atomic_write_json(
            reports / 'M5_independent_verifier_report.json',
            dict(independent_report),
        )
    if certificate_manifest is not None:
        atomic_write_json(
            reports / 'M5_certificate_manifest.json',
            dict(certificate_manifest),
        )
    return paths


def render_m5_markdown(report: Mapping[str, Any]) -> str:
    verdict = report.get('verdict', {})
    obligations = verdict.get('proof_obligations', {})
    lines = [
        '# M5 report',
        '',
        f"- generated_at: `{report.get('generated_at', utc_now())}`",
        f"- parent M4 run ID: `{report.get('parent_m4_run_id')}`",
        f"- M5 run ID: `{report.get('run_id')}`",
        f"- phase: `{report.get('phase')}`",
        f"- milestone_status: `{report.get('milestone_status')}`",
        f"- certification_status: `{report.get('certification_status')}`",
        f"- implementation_status: `{report.get('implementation_status')}`",
        f"- independent_verifier: `{verdict.get('independent_verifier')}`",
        f"- q_cert_upper: `{verdict.get('q_cert_upper')}`",
        f"- margin_lower: `{verdict.get('margin_lower')}`",
        '',
        '## Proof obligations',
        '',
    ]
    for key in sorted(obligations):
        lines.append(f"- {key}: `{obligations[key]}`")
    lines.extend([
        '',
        '## Scope limitation',
        '',
        'M5 certifies at most one RG step at fixed cutoff/rank/source class.',
        'It does not authorize continuum, thermodynamic-limit, OS positivity,',
        'or mass-gap claims. M6 must not start unless milestone_status is',
        f'`{ONE_STEP_CERTIFIED}` or a `{NOT_CERTIFIED}` certificate-failure completion',
        f'with phase `{M5_COMPLETE}` (majorant failure ≠ proof of true-map expansion).',
        '',
    ])
    return '\n'.join(lines) + '\n'
