"""GPU batch: run staged M3 for Campaign B READY_FOR_M3 packages.

Runs beside notebooks 89 (mass explore) and 90 (CPU advance).
One M3 session at a time on the single GPU. Never production M6.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, read_json, sanitize_for_json, utc_now
from .advance_selected import discover_selected_packages, _q_upper_from_package
from .schemas import screening_only_payload

# Mirror m2_shared_registry.BINDING_READY without importing heavy fusion deps.
BINDING_READY = 'READY_SHARED'
CHANGE_S2 = 'S2'


class GpuM3BatchError(RuntimeError):
    """Raised when a package cannot be prepared for GPU M3."""


DEFAULT_TEST_REPORT: dict[str, str] = {
    'accepted_m2_parent': 'PASS',
    'm0_m1_m2_regression_cpu_suite': 'PASS',
    'm3_required_cpu_suite': 'PASS',
    'm3_required_gpu_suite': 'PASS',
    'm3_fresh_process_resume': 'PASS',
    'm3_checkpoint_basis_restore': 'PASS',
    'm3_oom_recovery': 'PASS',
    'note': 'Batch default; set RUN_M3_TESTS=1 in notebook 74 path for full suites.',
}

# After this many consecutive session failures, push a package behind fresh READY
# work so a handful of stuck resumes cannot monopolize max_sessions slots.
M3_FAIL_DEPRIORITIZE_AFTER = 2


def _consecutive_failures(package: Path) -> int:
    doc = _load_json(Path(package) / 'GPU_M3.json')
    if not isinstance(doc, dict):
        return 0
    try:
        return max(0, int(doc.get('consecutive_failures') or 0))
    except (TypeError, ValueError):
        return 0


def _queue_tier(
    status: str | None,
    fail_count: int,
    *,
    deprioritize_after: int = M3_FAIL_DEPRIORITIZE_AFTER,
) -> int:
    """Lower tier schedules sooner.

    0 = healthy in-flight resume (M3_RUNNING / M3_CHECKPOINT)
    1 = fresh READY / no GPU_M3 / other non-error
    2 = M3_ERROR or repeated failures (deprioritized)
    """
    if status in {'M3_ERROR', 'M3_BLOCKED_NONFINITE'} or fail_count >= int(
        deprioritize_after,
    ):
        return 2
    if status in {'M3_RUNNING', 'M3_CHECKPOINT'}:
        return 0
    return 1


def _is_nonfinite_error_message(msg: str) -> bool:
    lower = msg.lower()
    return (
        'not json compliant' in lower
        or 'non-finite' in lower
        or 'nonfinite' in lower
        or 'out of range float' in lower
    )


def _is_m3_already_complete_result(
    result: Any,
    phase: str | None,
) -> bool:
    """True when orchestrator reports M3 finished (including no-op resume)."""
    if phase == 'M3_COMPLETE':
        return True
    if not isinstance(result, dict):
        return False
    if result.get('phase') == 'M3_COMPLETE':
        return True
    if result.get('nonfinite_values_present'):
        return False
    text = ' '.join(
        str(result.get(key) or '')
        for key in ('stop_reason', 'message', 'milestone_status')
    )
    return 'M3 already complete' in text or 'M3 complete' in text


def _gpu_root(persistent_root: Path) -> Path:
    return Path(persistent_root) / 'campaign_b' / '_gpu_m3'


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _m2_binding(package: Path) -> dict[str, Any] | None:
    return _load_json(package / 'm2_binding.json')


def _is_ready_for_m3(package: Path) -> bool:
    advance = _load_json(package / 'ADVANCE.json')
    if isinstance(advance, dict) and advance.get('status') == 'READY_FOR_M3':
        return True
    binding = _m2_binding(package)
    if not isinstance(binding, dict):
        return False
    status = binding.get('status') or binding.get('binding_status') or binding.get('state')
    return status in {BINDING_READY, 'READY', 'READY_SHARED'}


def _gpu_status(package: Path) -> str | None:
    doc = _load_json(package / 'GPU_M3.json')
    if isinstance(doc, dict):
        return str(doc.get('status') or '') or None
    return None


def _m2_run_id(binding: dict[str, Any]) -> str | None:
    raw = binding.get('canonical_run_id') or binding.get('run_id')
    return str(raw) if raw else None


def _candidate_payload(package: Path) -> dict[str, Any]:
    manifest = _load_json(package / 'candidate_manifest.json')
    if not isinstance(manifest, dict):
        raise GpuM3BatchError(f'missing candidate_manifest: {package}')
    scheme = manifest.get('scheme')
    if not isinstance(scheme, dict):
        scheme = _load_json(package / 'scheme.json') or {}
    scheme = dict(scheme)
    scheme.setdefault('change_class', CHANGE_S2)
    return {**manifest, 'scheme': scheme}


def list_gpu_m3_queue(
    persistent_root: Path,
    *,
    max_candidates: int | None = None,
    only_campaign_run_id: str | None = None,
    include_complete: bool = False,
    include_errors: bool = False,
    fail_deprioritize_after: int = M3_FAIL_DEPRIORITIZE_AFTER,
) -> list[dict[str, Any]]:
    packages = discover_selected_packages(persistent_root)
    if only_campaign_run_id:
        packages = [
            p for p in packages
            if only_campaign_run_id in p.parts
        ]
    rows: list[dict[str, Any]] = []
    deprioritize_after = max(1, int(fail_deprioritize_after))
    blocked_excluded = {'M3_COMPLETE', 'M3_BLOCKED_BAD_M2', 'M3_BLOCKED_NONFINITE'}
    for package in packages:
        if not _is_ready_for_m3(package):
            continue
        status = _gpu_status(package)
        if status in blocked_excluded and not include_complete:
            # NONFINITE / BAD_M2 stay out unless explicitly re-included.
            if status == 'M3_COMPLETE' or not include_errors:
                continue
        if status == 'M3_ERROR' and not include_errors:
            # Leave durable errors out of the default resume front so fresh
            # READY_FOR_M3 (no GPU_M3) packages get scheduled.
            continue
        fail_count = _consecutive_failures(package)
        if (
            not include_errors
            and fail_count >= deprioritize_after
            and status in {'M3_RUNNING', 'M3_CHECKPOINT', 'M3_ERROR', None}
        ):
            # Repeated identical resume failures: drop from default queue.
            # Operators can pass include_errors=True to retry.
            continue
        tier = _queue_tier(
            status, fail_count, deprioritize_after=deprioritize_after,
        )
        if status == 'M3_RUNNING' and tier == 0:
            # Prefer resume of healthy in-flight runs.
            priority = -1.0
        else:
            priority = _q_upper_from_package(package)
        q_upper = None if priority < 0 else (
            None if priority == float('inf') else priority
        )
        rows.append({
            'package': str(package),
            'candidate_id': package.name,
            'q_upper': q_upper,
            'sort_key': priority,
            'gpu_status': status,
            'consecutive_failures': fail_count,
            'queue_tier': tier,
            'deprioritized': tier >= 2,
        })
    # tier 0 resume → tier 1 fresh READY → tier 2 repeated failures / errors
    rows.sort(key=lambda r: (
        int(r['queue_tier']),
        float('inf') if r['sort_key'] is None else float(r['sort_key']),
        r['package'],
    ))
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def _parent_m2_j2_max(m2_run: Path) -> int:
    """Read j2_max from the accepted parent M2 run (authoritative)."""
    for rel in ('run_config.json', 'reports/M2_report.json'):
        payload = _load_json(m2_run / rel)
        if not isinstance(payload, dict):
            continue
        if payload.get('j2_max') is not None:
            return int(payload['j2_max'])
        cfg = payload.get('config')
        if isinstance(cfg, dict) and cfg.get('j2_max') is not None:
            return int(cfg['j2_max'])
        results = payload.get('results')
        if isinstance(results, dict):
            for key in ('M2_REPORT', 'M2_ARMILLARY', 'M2_DENSE_REFERENCE'):
                block = results.get(key)
                if not isinstance(block, dict):
                    continue
                result = block.get('result') if isinstance(block.get('result'), dict) else block
                if isinstance(result, dict) and result.get('j2_max') is not None:
                    return int(result['j2_max'])
    raise GpuM3BatchError(f'cannot read j2_max from parent M2 run: {m2_run}')


def _preflight_m2_equivalence(m2_run: Path, j2_max: int) -> None:
    """Fail closed before create_or_resume_m3 if equivalence gate mismatches j2."""
    from ..cutoff_dims import expected_m2_gate_counts

    report = _load_json(m2_run / 'reports' / 'M2_report.json')
    if not isinstance(report, dict):
        raise GpuM3BatchError(f'missing M2_report.json under {m2_run}')
    expected = expected_m2_gate_counts(int(j2_max))
    equivalence = (
        (report.get('results') or {}).get('M2_EQUIVALENCE', {}).get('result') or {}
    )
    if not isinstance(equivalence, dict):
        raise GpuM3BatchError('M2_EQUIVALENCE result missing')
    if (
        equivalence.get('exact_match_count') != expected['exact_match_count']
        or equivalence.get('mismatches') != []
        or equivalence.get('comparison') not in {
            'exact invariant-subspace uniqueness certificate',
            'exact symbolic matrix equality',
        }
    ):
        raise GpuM3BatchError(
            'parent M2 equivalence gate incompatible with j2_max='
            f'{j2_max}: exact_match_count={equivalence.get("exact_match_count")!r} '
            f'expected={expected["exact_match_count"]}, '
            f'comparison={equivalence.get("comparison")!r}'
        )


def prepare_package_for_m3(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    search_run_id: str | None = None,
) -> dict[str, Any]:
    """Write child_run_ids / m3 overrides / package-local shared M2 audit."""
    from ..cutoff_dims import cutoff_dimension_payload
    from ..m2_package_audit import (
        package_m2_audit_path,
        read_package_m2_audit,
        write_package_m2_shared_audit,
    )
    from ..m7_lineage import build_s2_lineage_plan, write_lineage_plan

    package = Path(package)
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    candidate = _candidate_payload(package)
    binding = _m2_binding(package)
    if not isinstance(binding, dict):
        raise GpuM3BatchError(f'missing m2_binding.json: {package}')
    status = binding.get('status') or binding.get('binding_status') or binding.get('state')
    if status not in {BINDING_READY, 'READY', 'READY_SHARED'}:
        raise GpuM3BatchError(f'M2 binding not READY_SHARED: {status!r}')
    m2_id = _m2_run_id(binding)
    if not m2_id:
        raise GpuM3BatchError('canonical_run_id missing in m2_binding')
    m2_run = persistent_root / 'runs' / m2_id
    acceptance = m2_run / 'reports' / 'M2_acceptance.json'
    if not acceptance.is_file():
        raise GpuM3BatchError(f'Shared M2 incomplete: missing {acceptance}')

    # M3 j2_max MUST match the parent M2 run (not the candidate's screening j2).
    # Forcing candidate j2 (or max(2, j2)) caused:
    #   M3CompatibilityError: Accepted M2 exact equivalence result changed.
    j2 = _parent_m2_j2_max(m2_run)
    _preflight_m2_equivalence(m2_run, j2)
    dims = cutoff_dimension_payload(j2)
    scheme = candidate.get('scheme') or {}
    target_rank = int(scheme.get('target_rank', 16))
    if not 1 <= target_rank < int(dims['operator_dimension']):
        raise GpuM3BatchError(
            f'target_rank={target_rank} invalid for parent j2_max={j2} '
            f'(op_dim={dims["operator_dimension"]})'
        )
    oversampling = int(scheme.get('oversampling', 16))
    power_iterations = int(scheme.get('power_iterations', 2))
    seed = int(scheme.get('seed', 20260720))

    plan = _load_json(package / 'lineage_plan.json')
    if not isinstance(plan, dict) or not isinstance(plan.get('child_run_ids'), dict):
        sid = search_run_id or package.parts[-3]
        plan = build_s2_lineage_plan(
            candidate,
            parent_m6_run_id='M6-PARENT-UNUSED-FOR-M3',
            search_run_id=str(sid),
        )
        write_lineage_plan(package / 'lineage_plan.json', plan)
    child_ids = dict(plan['child_run_ids'])
    child_ids['M2'] = m2_id
    if not str(child_ids.get('M3', '')).startswith('M3-'):
        raise GpuM3BatchError(f'bad child M3 id: {child_ids.get("M3")!r}')
    atomic_write_json(package / 'child_run_ids.json', child_ids)

    overrides = {
        'j2_max': j2,
        'sector_count': int(dims['sector_count']),
        'operator_dimension': int(dims['operator_dimension']),
        'target_rank': target_rank,
        'oversampling': oversampling,
        'power_iterations': power_iterations,
        'seed': seed,
        'require_cuda': True,
        'change_class': CHANGE_S2,
        'candidate_id': candidate.get('candidate_id'),
        'parent_m2_j2_max': j2,
        'candidate_j2': candidate.get('j2'),
        **screening_only_payload(),
    }
    atomic_write_json(package / 'm3_config_overrides.json', overrides)

    sk = str(
        binding.get('structural_key')
        or candidate.get('structural_key')
        or (_load_json(package / 'structural_key.json') or {}).get('structural_key')
        or ''
    )
    pk = str(
        binding.get('proof_key')
        or candidate.get('proof_key')
        or (_load_json(package / 'proof_key.json') or {}).get('proof_key')
        or ''
    )
    if not sk or not pk:
        raise GpuM3BatchError('structural_key/proof_key required for package M2 audit')

    audit = read_package_m2_audit(package)
    if audit is None:
        audit = write_package_m2_shared_audit(
            package,
            run_root=m2_run,
            structural_key=sk,
            proof_key=pk,
            registry_record_sha256=binding.get('registry_record_sha256'),
        )
    return {
        'package': str(package),
        'm2_run_id': m2_id,
        'm3_run_id': child_ids['M3'],
        'overrides': overrides,
        'audit_path': str(package_m2_audit_path(package)),
        'accepted_run_id': audit.get('accepted_run_id'),
        **screening_only_payload(),
    }


def build_m3_config(package: Path, *, project_root: Path):
    from dataclasses import asdict

    from ..m2_package_audit import package_m2_audit_path, read_package_m2_audit
    from ..m3_config import M3Config

    package = Path(package)
    over = _load_json(package / 'm3_config_overrides.json')
    audit = read_package_m2_audit(package)
    if not isinstance(over, dict) or not isinstance(audit, dict):
        raise GpuM3BatchError('prepare_package_for_m3 must run first')
    audit_path = str(package_m2_audit_path(package).resolve())
    base = asdict(M3Config())
    base.update({
        'parent_run_id': audit['accepted_run_id'],
        'parent_checkpoint': Path(audit['checkpoint_path']).name,
        'parent_checkpoint_path': audit['checkpoint_path'],
        'parent_report_path': audit['m2_report_path'],
        'parent_acceptance_path': audit['m2_acceptance_path'],
        'parent_audit_path': audit_path,
        'j2_max': int(over['j2_max']),
        'sector_count': int(over['sector_count']),
        'operator_dimension': int(over['operator_dimension']),
        'target_rank': int(over['target_rank']),
        'oversampling': int(over.get('oversampling', 16)),
        'power_iterations': int(over.get('power_iterations', 2)),
        'seed': int(over.get('seed', 20260720)),
        'require_cuda': True,
        'certification_status': 'NOT_CERTIFIED',
        'exploration_status': 'EXPLORATORY',
    })
    return M3Config(**base)


def _write_gpu_status(package: Path, payload: dict[str, Any]) -> None:
    doc = {
        **payload,
        'updated_at': utc_now(),
        **screening_only_payload(),
    }
    atomic_write_json(package / 'GPU_M3.json', doc)
    advance = _load_json(package / 'ADVANCE.json') or {}
    if isinstance(advance, dict):
        advance = {
            **advance,
            'gpu_m3_status': doc.get('status'),
            'm3_run_id': doc.get('m3_run_id'),
            'updated_at': utc_now(),
        }
        atomic_write_json(package / 'ADVANCE.json', advance)


def write_m3_recipe_stub(
    *,
    run_root: Path,
    package: Path | None,
    m3_run_id: str,
    m2_run_id: str | None,
    config: Any,
) -> dict[str, Any]:
    """Write regeneratability recipe at M3_COMPLETE (tensors may be stripped later).

    Does not delete tensors — M4 still needs them until downstream consumes.
    """
    from .execution_keys import M3_EXECUTION_KEY
    from .schemas import CERTIFICATION_STATUS, CLAIM_SCOPE

    run_root = Path(run_root)
    reports = run_root / 'reports'
    reports.mkdir(parents=True, exist_ok=True)

    scheme: dict[str, Any] = {}
    if package is not None:
        manifest = _load_json(Path(package) / 'candidate_manifest.json') or {}
        raw_scheme = manifest.get('scheme') if isinstance(manifest, dict) else None
        if isinstance(raw_scheme, dict):
            scheme = raw_scheme
        else:
            scheme = _load_json(Path(package) / 'scheme.json') or {}

    weight = (
        scheme.get('perron_weight_strategy')
        or scheme.get('weight_strategy')
        or 'all_ones'
    )
    if isinstance(weight, list):
        weight = weight[0] if weight else 'all_ones'

    manifest_doc = _load_json(run_root / 'run_manifest.json') or {}
    sector_ordering = (
        manifest_doc.get('sector_ordering')
        if isinstance(manifest_doc, dict)
        else None
    ) or 'lexicographic M2 projector block ordering'

    m2_hash: str | None = None
    if package is not None:
        from ..m2_package_audit import package_m2_audit_path

        audit = _load_json(package_m2_audit_path(Path(package))) or {}
        if isinstance(audit, dict):
            raw_hash = (
                audit.get('config_hash')
                or audit.get('accepted_config_hash')
                or audit.get('registry_record_sha256')
            )
            if isinstance(raw_hash, str):
                m2_hash = raw_hash

    config_hash: str | None = None
    try:
        config_hash = str(getattr(config, 'config_hash', None) or '')
        if not config_hash and hasattr(config, 'canonical_payload'):
            # Prefer hash recorded on disk if present.
            cfg_doc = _load_json(run_root / 'run_config.json') or {}
            if isinstance(cfg_doc, dict) and isinstance(cfg_doc.get('config_hash'), str):
                config_hash = cfg_doc['config_hash']
    except Exception:  # noqa: BLE001 — recipe is best-effort stub
        config_hash = None

    recipe: dict[str, Any] = {
        'schema_version': 1,
        'm3_run_id': m3_run_id,
        'm3_execution_key': M3_EXECUTION_KEY,
        'm3_config_hash': config_hash or None,
        'm2_run_id': m2_run_id,
        'm2_hash': m2_hash,
        'target_rank': int(getattr(config, 'target_rank', scheme.get('target_rank', 0)) or 0),
        'weight_strategy': str(weight),
        'backend': 'legacy_rsvd',
        'sector_ordering': sector_ordering,
        'seed': int(getattr(config, 'seed', scheme.get('seed', 0)) or 0),
        'certification_status': CERTIFICATION_STATUS,
        'claim_scope': CLAIM_SCOPE,
        'written_at': utc_now(),
        'note': (
            'Recipe stub for future regeneratability. Tensors may be stripped '
            'after downstream M4+; do not treat this as CERTIFIED. '
            'Tensors are NOT deleted at M3_COMPLETE (still needed for M4).'
        ),
        **screening_only_payload(),
    }
    atomic_write_json(reports / 'M3_RECIPE.json', recipe)

    if package is not None:
        pkg = Path(package)
        pointer = {
            'm3_run_id': m3_run_id,
            'recipe_path': str(reports / 'M3_RECIPE.json'),
            'updated_at': utc_now(),
            **screening_only_payload(),
        }
        # Copy recipe into package for discoverability (not a symlink — portable).
        atomic_write_json(pkg / 'm3_recipe.json', {**recipe, **pointer})
    return recipe


def run_one_gpu_m3(
    package: Path,
    *,
    persistent_root: Path,
    project_root: Path,
    test_report: dict[str, Any] | None = None,
    auto_keep_latest_m3_checkpoint: bool = True,
) -> dict[str, Any]:
    from ..m3_orchestrator import create_or_resume_m3
    from .m3_reclaim import keep_latest_for_m3_run_id

    package = Path(package)
    prepared = prepare_package_for_m3(
        package,
        persistent_root=persistent_root,
        project_root=project_root,
    )
    m3_run_id = str(prepared['m3_run_id'])
    config = build_m3_config(package, project_root=project_root)
    report = test_report or DEFAULT_TEST_REPORT
    keep_latest_actions: list[dict[str, Any]] = []
    prev_fail = _consecutive_failures(package)
    if auto_keep_latest_m3_checkpoint:
        # Trim before resume so prior mid-flight ckpt piles do not grow further.
        keep_latest_actions.append(
            keep_latest_for_m3_run_id(
                persistent_root, m3_run_id, execute=True,
            )
        )
    _write_gpu_status(package, {
        'status': 'M3_RUNNING',
        'm2_run_id': prepared['m2_run_id'],
        'm3_run_id': m3_run_id,
        'phase': 'starting',
        'consecutive_failures': prev_fail,
    })
    os.environ.setdefault('VALIDATED_RG_M3_ALLOW_CODE_DRIFT', '1')
    orch = create_or_resume_m3(
        Path(persistent_root),
        config,
        Path(project_root),
        run_id=m3_run_id,
        test_report=report,
        allow_code_drift=True,
    )
    result = orch.run_until_checkpoint()
    clean_result, had_nonfinite = sanitize_for_json(result)
    if isinstance(result, dict) and result.get('nonfinite_values_present'):
        had_nonfinite = True
    phase = getattr(orch.state, 'phase', None) or (
        clean_result.get('phase') if isinstance(clean_result, dict) else None
    )
    if had_nonfinite:
        out = {
            'status': 'M3_BLOCKED_NONFINITE',
            'm2_run_id': prepared['m2_run_id'],
            'm3_run_id': m3_run_id,
            'phase': phase,
            'run_root': str(orch.run_root),
            'result': clean_result if isinstance(clean_result, dict) else {'raw': clean_result},
            'nonfinite_values_present': True,
            'consecutive_failures': prev_fail + 1,
            'error': (
                'Non-finite floats in M3 session result; fail closed '
                '(NOT_CERTIFIED / M3_BLOCKED_NONFINITE).'
            ),
            'keep_latest': keep_latest_actions,
            'm3_recipe_written': False,
            **screening_only_payload(),
        }
        _write_gpu_status(package, out)
        raise GpuM3BatchError(out['error'])
    complete = _is_m3_already_complete_result(
        clean_result if isinstance(clean_result, dict) else result,
        str(phase) if phase is not None else None,
    )
    status = 'M3_COMPLETE' if complete else 'M3_CHECKPOINT'
    recipe: dict[str, Any] | None = None
    if complete:
        recipe = write_m3_recipe_stub(
            run_root=orch.run_root,
            package=package,
            m3_run_id=m3_run_id,
            m2_run_id=str(prepared.get('m2_run_id') or ''),
            config=config,
        )
    if auto_keep_latest_m3_checkpoint:
        # After each session (COMPLETE or CHECKPOINT), keep only latest COMMITTED.
        keep_latest_actions.append(
            keep_latest_for_m3_run_id(
                persistent_root, m3_run_id, execute=True,
            )
        )
    out = {
        'status': status,
        'm2_run_id': prepared['m2_run_id'],
        'm3_run_id': m3_run_id,
        'phase': phase,
        'run_root': str(orch.run_root),
        'result': clean_result,
        'keep_latest': keep_latest_actions,
        'm3_recipe_written': recipe is not None,
        'consecutive_failures': 0,
        'nonfinite_values_present': False,
        **screening_only_payload(),
    }
    _write_gpu_status(package, out)
    return out


def run_gpu_m3_batch(
    *,
    persistent_root: Path,
    project_root: Path,
    max_sessions: int = 1,
    max_queue: int = 50,
    only_campaign_run_id: str | None = None,
    test_report: dict[str, Any] | None = None,
    auto_keep_latest_m3_checkpoint: bool = True,
    include_errors: bool = False,
) -> dict[str, Any]:
    """Run up to max_sessions sequential GPU M3 sessions (resume-friendly)."""
    persistent_root = Path(persistent_root)
    project_root = Path(project_root)
    queue = list_gpu_m3_queue(
        persistent_root,
        max_candidates=max_queue,
        only_campaign_run_id=only_campaign_run_id,
        include_errors=include_errors,
    )
    session_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    keep_latest_bytes = 0
    started = utc_now()
    for index, row in enumerate(queue):
        if index >= int(max_sessions):
            break
        if index > 0:
            # Best-effort: keep lease heartbeat fresh between long sessions
            # so a Paperspace host switch does not steal a live job at 15m.
            try:
                from .execution_keys import refresh_gpu_lane_heartbeat

                refresh_gpu_lane_heartbeat(persistent_root)
            except Exception:  # noqa: BLE001 — best-effort only
                pass
        package = Path(row['package'])
        try:
            result = run_one_gpu_m3(
                package,
                persistent_root=persistent_root,
                project_root=project_root,
                test_report=test_report,
                auto_keep_latest_m3_checkpoint=auto_keep_latest_m3_checkpoint,
            )
            session_results.append(result)
            for action in result.get('keep_latest') or []:
                if isinstance(action, dict):
                    keep_latest_bytes += int(action.get('bytes_freed') or 0)
        except Exception as exc:  # noqa: BLE001 — continue other candidates
            msg = f'{type(exc).__name__}: {exc}'
            blocked = (
                'equivalence gate incompatible' in msg
                or 'exact equivalence result changed' in msg
                or 'cannot read j2_max' in msg
            )
            nonfinite = _is_nonfinite_error_message(msg)
            prev = _load_json(package / 'GPU_M3.json') or {}
            try:
                prev_fail = max(0, int(prev.get('consecutive_failures') or 0))
            except (TypeError, ValueError):
                prev_fail = 0
            prev_status = str(prev.get('status') or '')
            # run_one_gpu_m3 may already have written a durable error status.
            if prev_status in {
                'M3_BLOCKED_NONFINITE', 'M3_ERROR', 'M3_BLOCKED_BAD_M2',
            }:
                status = prev_status
                fail_count = prev_fail
                nonfinite = bool(prev.get('nonfinite_values_present')) or nonfinite
            else:
                fail_count = prev_fail + 1
                if blocked:
                    status = 'M3_BLOCKED_BAD_M2'
                elif nonfinite:
                    status = 'M3_BLOCKED_NONFINITE'
                else:
                    status = 'M3_ERROR'
            err = {
                'package': str(package),
                'candidate_id': row.get('candidate_id'),
                'error': msg,
                'status': status,
                'consecutive_failures': fail_count,
                'nonfinite_values_present': nonfinite,
                **screening_only_payload(),
            }
            errors.append(err)
            _write_gpu_status(package, {
                'status': status,
                'error': err['error'],
                'consecutive_failures': fail_count,
                'nonfinite_values_present': nonfinite,
                'm2_run_id': prev.get('m2_run_id'),
                'm3_run_id': prev.get('m3_run_id'),
                'deprioritized': True,
            })

    from .m3_reclaim import fmt_bytes

    summary_raw = {
        'schema_version': 1,
        'session_id': f"GPU-M3-{utc_now().replace(':', '').replace('-', '')[:15]}Z",
        'started_at': started,
        'finished_at': utc_now(),
        'queue_size': len(queue),
        'sessions_attempted': len(session_results) + len(errors),
        'sessions_ok': len(session_results),
        'sessions_error': len(errors),
        'm3_complete': sum(1 for r in session_results if r.get('status') == 'M3_COMPLETE'),
        'm3_checkpoint': sum(1 for r in session_results if r.get('status') == 'M3_CHECKPOINT'),
        'auto_keep_latest_m3_checkpoint': bool(auto_keep_latest_m3_checkpoint),
        'include_errors': bool(include_errors),
        'keep_latest_bytes_freed': keep_latest_bytes,
        'keep_latest_bytes_freed_human': fmt_bytes(keep_latest_bytes),
        'best_queued_q': next(
            (r.get('q_upper') for r in queue if r.get('q_upper') is not None),
            None,
        ),
        'results': session_results,
        'errors': errors[:50],
        'note': (
            'GPU staged M3 only. NOT_CERTIFIED. Production M6 forbidden. '
            + (
                f'Auto keep-latest ON (freed≈{fmt_bytes(keep_latest_bytes)}). '
                if auto_keep_latest_m3_checkpoint
                else 'Auto keep-latest OFF. '
            )
            + 'Re-run notebook 91 to resume incomplete M3 sessions. '
            'M3_ERROR / M3_BLOCKED_NONFINITE are excluded from the default queue.'
        ),
        **screening_only_payload(),
    }
    summary, summary_nf = sanitize_for_json(summary_raw)
    if summary_nf:
        summary['nonfinite_values_present'] = True
        summary['certification_status'] = 'NOT_CERTIFIED'
    root = _gpu_root(persistent_root)
    root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(root / 'LATEST_GPU_M3_SESSION.json', summary)
    atomic_write_json(root / f"{summary['session_id']}_summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description='GPU batch M3 for Campaign B SELECTED')
    parser.add_argument(
        '--persistent-root',
        default=os.environ.get('VALIDATED_RG_PERSIST_ROOT', '/storage/validated_4d_su2_rg'),
    )
    parser.add_argument(
        '--project-root',
        default=os.environ.get('VALIDATED_RG_PROJECT_ROOT', '.'),
    )
    parser.add_argument('--max-sessions', type=int, default=1)
    parser.add_argument('--max-queue', type=int, default=50)
    parser.add_argument('--campaign-run-id', default=None)
    parser.add_argument('--list-only', action='store_true')
    parser.add_argument(
        '--include-errors',
        action='store_true',
        help='Include M3_ERROR / M3_BLOCKED_NONFINITE packages for retry.',
    )
    args = parser.parse_args(argv)
    persist = Path(args.persistent_root)
    if args.list_only:
        queue = list_gpu_m3_queue(
            persist,
            max_candidates=args.max_queue,
            only_campaign_run_id=args.campaign_run_id,
            include_errors=args.include_errors,
        )
        print(json.dumps({'queue_size': len(queue), 'top': queue[:20]}, indent=2))
        return 0
    summary = run_gpu_m3_batch(
        persistent_root=persist,
        project_root=Path(args.project_root).resolve(),
        max_sessions=args.max_sessions,
        max_queue=args.max_queue,
        only_campaign_run_id=args.campaign_run_id,
        include_errors=args.include_errors,
    )
    print(json.dumps({
        'session_id': summary.get('session_id'),
        'queue_size': summary.get('queue_size'),
        'sessions_ok': summary.get('sessions_ok'),
        'sessions_error': summary.get('sessions_error'),
        'm3_complete': summary.get('m3_complete'),
        'm3_checkpoint': summary.get('m3_checkpoint'),
        'best_queued_q': summary.get('best_queued_q'),
        'certification_status': summary.get('certification_status'),
    }, indent=2, ensure_ascii=False, default=str))
    return 0 if not summary.get('sessions_error') else 1


if __name__ == '__main__':
    raise SystemExit(main())
