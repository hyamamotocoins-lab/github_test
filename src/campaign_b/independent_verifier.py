"""Independent re-screening in a separate Python process."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, hash_tree, read_json, utc_now
from .errors import CampaignFatalError
from .schemas import screening_only_payload


_VERIFY_WORKER = r'''
import json
import sys
from pathlib import Path

# Fresh process: import screening entrypoint independently.
from src.campaign_b.screening import run_primary_screening

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = run_primary_screening(
    payload["candidate"],
    parent_q_upper=float(payload["parent_q_upper"]),
    parent_rank=int(payload["parent_rank"]),
    screening_margin=float(payload["screening_margin"]),
)
Path(sys.argv[2]).write_text(json.dumps(result), encoding="utf-8")
'''


def _q_close(a: float, b: float, *, atol: float, rtol: float) -> bool:
    return abs(a - b) <= atol + rtol * abs(b)


def run_independent_verifier(
    *,
    candidate: dict[str, Any],
    primary_result: dict[str, Any],
    parent_q_upper: float,
    parent_rank: int,
    screening_margin: float,
    q_atol: float,
    q_rtol: float,
    repo_root: Path,
    work_dir: Path | None = None,
) -> dict[str, Any]:
    """Recompute screening in a subprocess; accept only if both q<1 and close."""
    root = Path(repo_root)
    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        tmp_path = Path(tmp)
        request_path = tmp_path / 'request.json'
        response_path = tmp_path / 'response.json'
        atomic_write_json(request_path, {
            'candidate': candidate,
            'parent_q_upper': parent_q_upper,
            'parent_rank': parent_rank,
            'screening_margin': screening_margin,
        })
        worker_path = tmp_path / 'verify_worker.py'
        worker_path.write_text(_VERIFY_WORKER, encoding='utf-8')
        env = os.environ.copy()
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = (
            str(root) if not existing else f'{root}{os.pathsep}{existing}'
        )
        proc = subprocess.run(
            [sys.executable, str(worker_path), str(request_path), str(response_path)],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if proc.returncode != 0 or not response_path.is_file():
            return {
                'schema_version': 1,
                'accepted': False,
                'reason': 'VERIFY_PROCESS_FAILED',
                'returncode': proc.returncode,
                'stderr': (proc.stderr or '')[-2000:],
                'stdout': (proc.stdout or '')[-1000:],
                'verified_at': utc_now(),
                **screening_only_payload(),
            }
        verify = read_json(response_path)
        if not isinstance(verify, dict):
            raise CampaignFatalError('independent verifier returned non-object')

    primary_q = float(primary_result.get('q_upper')
                      if primary_result.get('q_upper') is not None
                      else primary_result['estimated_q'])
    verify_q = float(verify.get('q_upper')
                     if verify.get('q_upper') is not None
                     else verify['estimated_q'])
    both_lt1 = (
        bool(primary_result.get('is_q_lt_1'))
        and bool(verify.get('is_q_lt_1'))
    )
    close = _q_close(primary_q, verify_q, atol=q_atol, rtol=q_rtol)
    # Compare operative hashes
    scheme_ok = (
        candidate.get('scheme_hash') == (verify.get('scheme_hash') or candidate.get('scheme_hash'))
    )
    j2_ok = int(candidate.get('j2') or 0) == int(verify.get('j2') or 0)
    accepted = both_lt1 and close and scheme_ok and j2_ok

    source_hash = None
    try:
        source_hash = hash_tree(root / 'src')
    except Exception:
        source_hash = None

    return {
        'schema_version': 1,
        'accepted': accepted,
        'reason': None if accepted else 'INDEPENDENT_VERIFY_MISMATCH',
        'primary_q': primary_q,
        'verify_q': verify_q,
        'q_atol': q_atol,
        'q_rtol': q_rtol,
        'both_q_lt_1': both_lt1,
        'q_close': close,
        'j2_ok': j2_ok,
        'scheme_ok': scheme_ok,
        'verify_result': verify,
        'verifier_source_tree_hash': source_hash,
        'verified_at': utc_now(),
        **screening_only_payload(),
    }
