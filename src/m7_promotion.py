"""Pre-M2 promotion stamps (Layer A). Never depends on post-M2 S0 results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .common import atomic_write_json, canonical_json_bytes, read_json, sha256_bytes, utc_now
from .cutoff_dims import resource_gate
from .m7_archive import is_archived, read_archive


class M7PromotionError(RuntimeError):
    """Raised when promotion bookkeeping fails closed."""


STATUS_REJECT = 'REJECT_SCREENING'
STATUS_PROMOTE = 'PROMOTE_TO_STRUCTURAL_PROOF'
STATUS_SCREENING_ONLY = 'SCREENING_ONLY'

# Back-compat alias used by older call sites.
STATUS_PROMOTE_LEGACY = 'PROMOTE_TO_PROOF'

LABELS = {
    'NOT_CERTIFIED': 'NOT_CERTIFIED',
    'NOT_ADMISSIBLE_AS_M5_BOUND': 'NOT_ADMISSIBLE_AS_M5_BOUND',
    'SCREENING_ONLY': 'SCREENING_ONLY',
}


def read_screening(package_root: Path) -> dict[str, Any] | None:
    path = Path(package_root) / 'SCREENING.json'
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def write_screening(
    package_root: Path,
    *,
    status: str,
    reasons: list[str],
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    allowed = {STATUS_REJECT, STATUS_PROMOTE, STATUS_SCREENING_ONLY, STATUS_PROMOTE_LEGACY}
    if status not in allowed:
        raise M7PromotionError(f'Unknown screening status: {status}')
    if status == STATUS_PROMOTE_LEGACY:
        status = STATUS_PROMOTE
    payload = {
        'schema_version': 2,
        'status': status,
        'reasons': list(reasons),
        'details': details or {},
        'labels': dict(LABELS),
        'certificate_usable': False,
        'admissible_as_m5_bound': False,
        'generated_at': utc_now(),
        'interpretation': 'HEURISTIC_EXPLORATORY_NOT_A_RIGOROUS_BOUND',
    }
    atomic_write_json(Path(package_root) / 'SCREENING.json', payload)
    return payload


def is_promoted(package_root: Path) -> bool:
    """Pre-M2 promotion only — does NOT consult S0 ADVANCE."""
    screening = read_screening(package_root)
    return bool(
        screening
        and screening.get('status') in {STATUS_PROMOTE, STATUS_PROMOTE_LEGACY}
    )


def evaluate_promotion(
    *,
    package_root: Path | None,
    j2_max: int,
    estimated_q: float,
    rank_among_executable: int | None,
    top_k: int = 3,
    max_executable_j2_max: int = 2,
    max_staged_j2_max: int = 2,
    force_promote: bool = False,
    force_reject: bool = False,
    campaign_run_id: str | None = None,
    ranking_snapshot_sha256: str | None = None,
    candidate_set_sha256: str | None = None,
    manual_override: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fixed pre-M2 screening rules. No S0 / post-M2 inputs."""
    reasons: list[str] = []
    gate = resource_gate(
        j2_max,
        max_executable_j2_max=max_executable_j2_max,
        max_staged_j2_max=max_staged_j2_max,
    )
    live = bool(gate.get('executable') or gate.get('staged_executable'))

    if force_reject:
        status = STATUS_REJECT
        reasons.append('force_reject')
    elif manual_override and manual_override.get('manual_override'):
        if not manual_override.get('reason') or not manual_override.get('operator'):
            raise M7PromotionError('manual_override requires reason and operator')
        status = STATUS_PROMOTE
        reasons.append('manual_override')
    elif force_promote:
        status = STATUS_PROMOTE
        reasons.append('force_promote')
    elif not live:
        status = STATUS_REJECT
        reasons.append('resource_gate blocked')
    elif package_root is not None and is_archived(package_root):
        status = STATUS_REJECT
        archive = read_archive(package_root) or {}
        reasons.append(f"archived:{archive.get('reason')}")
    elif rank_among_executable is not None and rank_among_executable < top_k:
        status = STATUS_PROMOTE
        reasons.append(f'top_k={top_k} (rank={rank_among_executable})')
        reasons.append(f'screening_estimated_q={estimated_q}')
    else:
        status = STATUS_SCREENING_ONLY
        reasons.append('not in campaign-snapshot top_k and not forced')
        reasons.append(f'screening_estimated_q={estimated_q}')

    details = {
        'campaign_run_id': campaign_run_id,
        'ranking_snapshot_sha256': ranking_snapshot_sha256,
        'candidate_set_sha256': candidate_set_sha256,
        'ranking_metric': 'screening_estimated_q',
        'ranking_direction': 'ascending',
        'top_k': int(top_k),
        'candidate_rank': rank_among_executable,
        'estimated_q': float(estimated_q),
        'manual_override': manual_override,
    }
    result = {
        'status': status,
        'decision': status,
        'reasons': reasons,
        'resource_gate': gate,
        **details,
        'labels': dict(LABELS),
    }
    if package_root is not None:
        write_screening(
            package_root,
            status=status,
            reasons=reasons,
            details=details,
        )
    return result


def ranking_snapshot_hashes(ranking_rows: list[dict[str, Any]]) -> dict[str, str]:
    ids = [str(row.get('candidate_id') or '') for row in ranking_rows]
    metric = [
        {'candidate_id': str(row.get('candidate_id') or ''), 'q': row.get('q_cert_upper')}
        for row in ranking_rows
    ]
    return {
        'candidate_set_sha256': sha256_bytes(canonical_json_bytes(sorted(ids))),
        'ranking_snapshot_sha256': sha256_bytes(canonical_json_bytes(metric)),
    }


def require_promote_for_canonical_m2(package_root: Path) -> None:
    if is_promoted(package_root):
        return
    screening = read_screening(package_root)
    status = (screening or {}).get('status')
    raise M7PromotionError(
        'Canonical shared M2 requires PROMOTE_TO_STRUCTURAL_PROOF '
        f'(got screening status={status!r}). '
        'Pre-M2 screening only; S0 SELECTED is not a promotion input.'
    )
