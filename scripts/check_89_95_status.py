#!/usr/bin/env python3
"""Check whether notebooks 89 (mass explore) and 95 (pipeline to M6) look alive.

Paperspace one-liner (from repo root):
  python scripts/check_89_95_status.py

Optional: VALIDATED_RG_PERSIST_ROOT=/storage/validated_4d_su2_rg \\
  python scripts/check_89_95_status.py --stale-minutes 90
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any


def _pgrep(patterns: list[str]) -> list[str]:
    """Return matching process lines (empty if none / pgrep missing)."""
    lines: list[str] = []
    for pat in patterns:
        try:
            proc = subprocess.run(
                ['pgrep', '-af', pat],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return []
        for line in (proc.stdout or '').splitlines():
            line = line.strip()
            if line and line not in lines:
                # Skip this status script itself.
                if 'check_89_95_status' in line:
                    continue
                lines.append(line)
    return lines


def _age_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    return max(0.0, time.time() - path.stat().st_mtime)


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return 'n/a'
    if seconds < 60:
        return f'{seconds:.0f}s'
    if seconds < 3600:
        return f'{seconds / 60:.1f}m'
    return f'{seconds / 3600:.2f}h'


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


def _label(
    *,
    procs: list[str],
    ledger: Path,
    payload: dict[str, Any] | None,
    age_s: float | None,
    stale_s: float,
    finished: bool,
) -> str:
    if procs:
        return 'RUNNING'
    if payload is None and not ledger.is_file():
        return 'UNKNOWN'
    if finished:
        return 'IDLE'
    # Incomplete session and no matching process.
    if age_s is not None and age_s <= stale_s:
        return 'STALE'  # recently touched → likely crashed mid-run
    if payload is not None:
        return 'STALE'  # old incomplete → orphaned / hung then died
    return 'IDLE'


def _print_block(title: str, rows: list[tuple[str, Any]]) -> None:
    print(f'=== {title} ===')
    for key, val in rows:
        print(f'  {key}: {val}')
    print()


def check_89(persist: Path, stale_s: float) -> str:
    root = persist / 'campaign_b' / '_mass_explore'
    ledger = root / 'LATEST_MASS_SESSION.json'
    seen = root / 'seen_normalized_schemes.json'
    payload = _load_json(ledger)
    age = _age_seconds(ledger)
    seen_age = _age_seconds(seen)
    procs = _pgrep([
        'src.campaign_b.mass_explore',
        'campaign_b/mass_explore',
        '89_campaign_b_mass_explore',
    ])
    finished = bool(
        payload
        and (
            payload.get('finished_at')
            or payload.get('status') == 'MASS_EXPLORE_COMPLETE'
        )
    )
    status = _label(
        procs=procs,
        ledger=ledger,
        payload=payload,
        age_s=age,
        stale_s=stale_s,
        finished=finished,
    )
    waves = (payload or {}).get('waves') or []
    rows: list[tuple[str, Any]] = [
        ('status', status),
        ('ledger', str(ledger)),
        ('ledger_age', _fmt_age(age)),
        ('seen_file_age', _fmt_age(seen_age)),
        ('session_id', (payload or {}).get('session_id')),
        ('waves_done', len(waves) if isinstance(waves, list) else None),
        ('selected_total', (payload or {}).get('selected_total')),
        ('archived_total', (payload or {}).get('archived_total')),
        ('processed_scheme_keys', (payload or {}).get('processed_scheme_keys')),
        ('certification_status', (payload or {}).get('certification_status')),
        ('finished_at', (payload or {}).get('finished_at')),
        ('processes', len(procs)),
    ]
    for line in procs[:5]:
        rows.append(('  proc', line[:160]))
    _print_block('Notebook 89 — mass explore', rows)
    return status


def check_95(persist: Path, stale_s: float) -> str:
    root = persist / 'campaign_b' / '_pipeline_to_m6'
    ledger = root / 'LATEST_PIPELINE_SESSION.json'
    payload = _load_json(ledger)
    age = _age_seconds(ledger)
    # Pipeline writes LATEST only at session end; stage ledgers update during a run.
    stage_paths = [
        persist / 'campaign_b' / '_advance' / 'LATEST_ADVANCE_SESSION.json',
        persist / 'campaign_b' / '_gpu_m3' / 'LATEST_GPU_M3_SESSION.json',
        persist / 'campaign_b' / '_pre_m6' / 'LATEST_PRE_M6_SESSION.json',
        persist / 'campaign_b' / '_obligations' / 'LATEST_OBLIGATION_SESSION.json',
        persist / 'campaign_b' / '_m6' / 'LATEST_M6_SESSION.json',
    ]
    stage_ages = [(p.name, _age_seconds(p)) for p in stage_paths if p.is_file()]
    freshest_stage = min((a for _, a in stage_ages if a is not None), default=None)
    activity_age = age
    if freshest_stage is not None:
        activity_age = freshest_stage if age is None else min(age, freshest_stage)

    procs = _pgrep([
        'src.campaign_b.pipeline_to_m6',
        'campaign_b/pipeline_to_m6',
        '95_campaign_b_pipeline_to_m6',
        'src.campaign_b.gpu_m3_batch',
        'src.campaign_b.pre_m6_batch',
        'src.campaign_b.m6_batch',
        'src.campaign_b.advance_selected',
        'src.campaign_b.close_obligations',
    ])
    finished = bool(payload and payload.get('finished_at'))
    status = _label(
        procs=procs,
        ledger=ledger,
        payload=payload,
        age_s=activity_age,
        stale_s=stale_s,
        finished=finished and not procs,
    )
    # If process alive, force RUNNING even when last pipeline ledger is finished
    # (a new round may be in progress before the next LATEST write).
    if procs:
        status = 'RUNNING'

    totals = (payload or {}).get('totals') or {}
    rows: list[tuple[str, Any]] = [
        ('status', status),
        ('ledger', str(ledger)),
        ('ledger_age', _fmt_age(age)),
        ('freshest_stage_age', _fmt_age(freshest_stage)),
        ('activity_age', _fmt_age(activity_age)),
        ('session_id', (payload or {}).get('session_id')),
        ('rounds_run', (payload or {}).get('rounds_run')),
        ('max_rounds', (payload or {}).get('max_rounds')),
        ('totals.advanced', totals.get('advanced') if isinstance(totals, dict) else None),
        ('totals.m3_complete', totals.get('m3_complete') if isinstance(totals, dict) else None),
        ('totals.pre_m6_ready', totals.get('pre_m6_ready') if isinstance(totals, dict) else None),
        ('totals.obligations_closed', totals.get('obligations_closed') if isinstance(totals, dict) else None),
        ('totals.m6_complete', totals.get('m6_complete') if isinstance(totals, dict) else None),
        ('totals.m6_certified', totals.get('m6_certified') if isinstance(totals, dict) else None),
        ('certification_status', (payload or {}).get('certification_status')),
        ('finished_at', (payload or {}).get('finished_at')),
        ('processes', len(procs)),
    ]
    for line in procs[:8]:
        rows.append(('  proc', line[:160]))
    _print_block('Notebook 95 — pipeline to M6', rows)
    return status


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Status check for Campaign B notebooks 89 and 95',
    )
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get(
            'VALIDATED_RG_PERSIST_ROOT',
            '/storage/validated_4d_su2_rg',
        ),
    )
    parser.add_argument(
        '--stale-minutes',
        type=float,
        default=90.0,
        help='Incomplete ledger older than this (no matching process) → STALE',
    )
    args = parser.parse_args()
    persist = Path(args.persistent_root)
    stale_s = float(args.stale_minutes) * 60.0

    print(f'persist_root: {persist}')
    print(f'stale_minutes: {args.stale_minutes}')
    print(f'checked_at: {time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}')
    print()
    print(
        'Labels: RUNNING=matching process; STALE=incomplete/old ledger, no process; '
        'IDLE=finished or quiet; UNKNOWN=no ledger/process.'
    )
    print(
        'Note: 95 writes LATEST_PIPELINE_SESSION only at session end; '
        'stage LATEST_* ages reflect in-flight work.'
    )
    print()

    s89 = check_89(persist, stale_s)
    s95 = check_95(persist, stale_s)
    print(f'SUMMARY  89={s89}  95={s95}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
