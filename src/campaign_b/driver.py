"""Campaign B six-hour autonomous exploration driver."""

from __future__ import annotations

import platform
import sys
import time
from pathlib import Path
from typing import Any

from ..common import atomic_write_json, hash_tree, read_json, utc_now
from .archive import archive_and_note
from .audit import audit_lineage_package
from .budget import BudgetManager
from .candidate_generator import (
    assign_priorities,
    generate_campaign_b_queue_candidates,
)
from .config import (
    CampaignBConfig,
    load_campaign_b_config,
    mint_campaign_run_id,
    search_space_hash,
)
from .errors import (
    CampaignFatalError,
    InvariantViolation,
    NeedCanonicalM2,
    TimeBudgetClosed,
)
from .estimators import RuntimeEstimator
from .finalizer import finalize_campaign
from .independent_verifier import run_independent_verifier
from .lineage import build_lineage_package, resolve_shared_m2, run_s0_screening_record
from .queue_store import QueueStore
from .schemas import (
    TERMINAL_EXHAUSTED,
    TERMINAL_FAIL,
    TERMINAL_NEED_M2,
    TERMINAL_Q_LT_1,
    TERMINAL_TIME,
    assert_not_certified,
    assert_phase_allowed,
    screening_only_payload,
)
from .screening import run_primary_screening
from .state_machine import transition_campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _source_tree_hash(cfg: CampaignBConfig) -> str:
    roots = cfg.source_tree_roots or [Path('src')]
    # Hash first existing root under repo.
    repo = _repo_root()
    for root in roots:
        path = root if root.is_absolute() else repo / root
        if path.is_dir():
            return hash_tree(path)
    return hash_tree(repo / 'src')


def _environment_manifest() -> dict[str, Any]:
    return {
        'python': sys.version,
        'platform': platform.platform(),
        'executable': sys.executable,
        'at': utc_now(),
        **screening_only_payload(),
    }


def create_or_resume_manifest(cfg: CampaignBConfig) -> tuple[CampaignBConfig, dict[str, Any]]:
    if cfg.resume_campaign_run_id:
        cfg.campaign_run_id = cfg.resume_campaign_run_id
        root = cfg.campaign_root()
        path = root / 'campaign_manifest.json'
        if not path.is_file():
            raise CampaignFatalError(f'resume manifest missing: {path}')
        manifest = read_json(path)
        if not isinstance(manifest, dict):
            raise CampaignFatalError('corrupt campaign_manifest.json')
        assert_not_certified(manifest, context='resume_manifest')
        return cfg, manifest

    if not cfg.campaign_run_id:
        cfg.campaign_run_id = mint_campaign_run_id()
    root = cfg.campaign_root()
    root.mkdir(parents=True, exist_ok=True)
    source_hash = _source_tree_hash(cfg)
    structural_key = cfg.structural_key or f'campaign-b-{source_hash[:16]}'
    proof_key = cfg.proof_key or f'proof-b-{search_space_hash(cfg.search_space)[:16]}'
    cfg.structural_key = structural_key
    cfg.proof_key = proof_key

    manifest = {
        'schema_version': 1,
        'campaign_kind': 'B_S2',
        'campaign_run_id': cfg.campaign_run_id,
        'created_at': utc_now(),
        'time_budget_sec': cfg.time_budget_sec,
        'execution_policy': cfg.execution_policy(),
        'search_space_hash': search_space_hash(cfg.search_space),
        'source_tree_hash': source_hash,
        'environment_hash': None,
        'persistent_root': str(cfg.persistent_root),
        'parent_evidence': cfg.parent_evidence,
        'structural_key': structural_key,
        'proof_key': proof_key,
        'environment_manifest': _environment_manifest(),
        **screening_only_payload(),
    }
    env_bytes = str(manifest['environment_manifest']).encode()
    import hashlib
    manifest['environment_hash'] = hashlib.sha256(env_bytes).hexdigest()
    atomic_write_json(root / 'campaign_manifest.json', manifest)
    return cfg, manifest


def run_preflight(cfg: CampaignBConfig, manifest: dict[str, Any]) -> dict[str, Any]:
    assert_phase_allowed('B_QUEUE')
    issues: list[str] = []
    if not cfg.persistent_root:
        issues.append('persistent_root missing')
    Path(cfg.persistent_root).mkdir(parents=True, exist_ok=True)
    if manifest.get('execution_policy', {}).get('allow_production_m6'):
        issues.append('allow_production_m6 true')
    if manifest.get('execution_policy', {}).get('allow_campaign_c'):
        issues.append('allow_campaign_c true')
    # Fail closed if CERTIFIED sneaks in
    try:
        assert_not_certified(manifest, context='preflight')
    except InvariantViolation as exc:
        issues.append(str(exc))
    current_hash = _source_tree_hash(cfg)
    if current_hash != manifest.get('source_tree_hash'):
        issues.append(
            f'source_tree_hash drift: manifest={manifest.get("source_tree_hash")} '
            f'current={current_hash}'
        )
    ok = not issues
    return {
        'ok': ok,
        'issues': issues,
        'source_tree_hash': current_hash,
        **screening_only_payload(),
    }


def run_campaign_b(config_path: Path | str) -> dict[str, Any]:
    cfg = load_campaign_b_config(Path(config_path))
    budget = BudgetManager(
        hard_limit_sec=cfg.time_budget_sec,
        admission_close_sec=cfg.admission_close_sec,
        finalization_start_sec=cfg.finalization_start_sec,
        emergency_flush_sec=cfg.emergency_flush_sec,
        enforce_wall_clock=cfg.enforce_wall_clock,
    )
    cfg, manifest = create_or_resume_manifest(cfg)
    campaign_root = cfg.campaign_root()
    store = QueueStore(campaign_root, lease_sec=cfg.lease_sec)
    ledger = store.load_ledger()

    try:
        ledger['campaign_state'] = transition_campaign(
            str(ledger.get('campaign_state') or 'CREATED'), 'PREFLIGHT',
        )
        store.save_ledger(ledger)

        preflight = run_preflight(cfg, manifest)
        atomic_write_json(campaign_root / 'preflight.json', preflight)
        if not preflight['ok']:
            ledger['campaign_state'] = 'FAIL_CLOSED'
            store.set_terminal_reason(TERMINAL_FAIL)
            store.record_event({'type': 'preflight_failed', 'issues': preflight['issues']})
            return finalize_campaign(
                campaign_root=campaign_root,
                manifest=manifest,
                queue=store.load_or_init([], campaign_run_id=cfg.campaign_run_id or ''),
                ledger=store.load_ledger(),
                budget=budget,
                terminal_reason=TERMINAL_FAIL,
            )

        # Resume deadline if present
        resume_deadline = None
        if cfg.resume_campaign_run_id and cfg.inherit_deadline:
            prev = ledger.get('budget_deadline_monotonic')
            if isinstance(prev, (int, float)):
                resume_deadline = float(prev)
        budget.start(resume_deadline_at=resume_deadline)
        ledger['budget_deadline_monotonic'] = budget.deadline_at
        ledger['campaign_state'] = transition_campaign('PREFLIGHT', 'RUNNING')
        store.save_ledger(ledger)

        source_hash = str(manifest['source_tree_hash'])
        candidates = generate_campaign_b_queue_candidates(
            campaign_run_id=str(cfg.campaign_run_id),
            search_space=cfg.search_space,
            structural_key=str(cfg.structural_key),
            proof_key=str(cfg.proof_key),
            source_tree_hash=source_hash,
            parent_m6_run_id=cfg.parent_m6_run_id,
            parent_scheme_hash=cfg.parent_scheme_hash,
            limit=cfg.candidate_limit,
        )
        candidates = assign_priorities(candidates, parent_q_upper=cfg.parent_q_upper)
        queue = store.load_or_init(candidates, campaign_run_id=str(cfg.campaign_run_id))
        queue = store.recover_expired_leases(queue)

        estimator = RuntimeEstimator()
        archived_ids = set(ledger.get('archived_ids') or [])
        terminal_reason: str | None = None
        repo_root = _repo_root()

        while True:
            if budget.must_finalize():
                ledger = store.load_ledger()
                if budget.remaining_sec() <= 0:
                    ledger['campaign_state'] = transition_campaign(
                        str(ledger.get('campaign_state') or 'RUNNING'),
                        'TIME_BUDGET_EXHAUSTED',
                    )
                    terminal_reason = TERMINAL_TIME
                else:
                    ledger['campaign_state'] = transition_campaign(
                        str(ledger.get('campaign_state') or 'RUNNING'),
                        'FINALIZING',
                    )
                    if terminal_reason is None:
                        terminal_reason = TERMINAL_TIME
                store.save_ledger(ledger)
                break

            if budget.admission_closed():
                ledger = store.load_ledger()
                state = str(ledger.get('campaign_state') or 'RUNNING')
                if state == 'RUNNING':
                    ledger['campaign_state'] = transition_campaign(
                        state, 'ADMISSION_CLOSED',
                    )
                    store.save_ledger(ledger)

            candidate = store.next_admissible(
                queue,
                budget=budget,
                estimator=estimator,
                archived_ids=archived_ids,
            )
            if candidate is None:
                # Distinguish exhausted vs time-gated
                pending = [
                    c for c in (queue.get('candidates') or [])
                    if c.get('state') == 'PENDING' and c['candidate_id'] not in archived_ids
                ]
                if pending and budget.admission_closed():
                    terminal_reason = TERMINAL_TIME
                else:
                    terminal_reason = TERMINAL_EXHAUSTED
                break

            cand_id = str(candidate['candidate_id'])
            store.reserve(queue, cand_id)
            t0 = time.monotonic()
            try:
                assert_phase_allowed('B_SCREEN')
                store.update_candidate(queue, cand_id, state='SCREENING')
                screen = run_primary_screening(
                    candidate,
                    parent_q_upper=cfg.parent_q_upper,
                    parent_rank=cfg.parent_rank,
                    screening_margin=cfg.screening_margin,
                )
                estimator.observe('SCREENING', time.monotonic() - t0)
                atomic_write_json(
                    campaign_root / 'candidates' / cand_id / 'screening.json',
                    screen,
                )

                if screen.get('is_borderline'):
                    archive_and_note(
                        campaign_root / 'archive',
                        list(archived_ids),
                        candidate=candidate,
                        screening_result=screen,
                        reason_code='BORDERLINE_Q',
                    )
                    archived_ids.add(cand_id)
                    store.update_candidate(queue, cand_id, state='ARCHIVED', lease=None)
                    store.record_event({'type': 'archive', 'candidate_id': cand_id, 'reason': 'BORDERLINE_Q'})
                    continue

                if not screen.get('is_q_lt_1'):
                    archive_and_note(
                        campaign_root / 'archive',
                        list(archived_ids),
                        candidate=candidate,
                        screening_result=screen,
                        reason_code='Q_GE_1',
                    )
                    archived_ids.add(cand_id)
                    store.update_candidate(queue, cand_id, state='ARCHIVED', lease=None)
                    store.record_event({'type': 'archive', 'candidate_id': cand_id, 'reason': 'Q_GE_1'})
                    continue

                store.update_candidate(queue, cand_id, state='SCREENED_Q_LT_1')

                if not budget.may_start(
                    'M2_RESOLVE', estimator.upper_runtime_sec('M2_RESOLVE', candidate),
                ):
                    raise TimeBudgetClosed('M2_RESOLVE')

                store.update_candidate(queue, cand_id, state='M2_RESOLVE')
                m2 = resolve_shared_m2(
                    candidate=candidate,
                    persistent_root=cfg.persistent_root,
                    source_tree_hash=source_hash,
                    allow_generate_canonical=cfg.allow_generate_canonical_m2,
                    structural_key=cfg.structural_key,
                    proof_key=cfg.proof_key,
                )
                from ..m2_shared_registry import BINDING_NEED
                if (
                    m2.get('status') == BINDING_NEED
                    or m2.get('reason') == 'NEED_CANONICAL_M2'
                ):
                    raise NeedCanonicalM2(cand_id)

                store.update_candidate(queue, cand_id, state='READY_SHARED')
                store.update_candidate(queue, cand_id, state='S0')
                s0 = run_s0_screening_record(
                    candidate=candidate,
                    m2_binding=m2,
                    primary_screen=screen,
                )
                atomic_write_json(
                    campaign_root / 'candidates' / cand_id / 's0.json', s0,
                )

                store.update_candidate(queue, cand_id, state='INDEPENDENT_VERIFY')
                verify = run_independent_verifier(
                    candidate=candidate,
                    primary_result=screen,
                    parent_q_upper=cfg.parent_q_upper,
                    parent_rank=cfg.parent_rank,
                    screening_margin=cfg.screening_margin,
                    q_atol=cfg.q_atol,
                    q_rtol=cfg.q_rtol,
                    repo_root=repo_root,
                )
                if not verify.get('accepted'):
                    archive_and_note(
                        campaign_root / 'archive',
                        list(archived_ids),
                        candidate=candidate,
                        screening_result=screen,
                        reason_code='INDEPENDENT_VERIFY_MISMATCH',
                        extra={'verify': verify},
                    )
                    archived_ids.add(cand_id)
                    store.update_candidate(
                        queue, cand_id, state='VERIFY_REJECTED',
                    )
                    store.update_candidate(queue, cand_id, state='ARCHIVED', lease=None)
                    continue

                # Build package then audit (audit file written inside package)
                selected_dir = campaign_root / 'selected' / cand_id
                placeholder_audit = {
                    'accepted': True,
                    'pending_full_audit': True,
                    **screening_only_payload(),
                }
                build_lineage_package(
                    package_root=selected_dir,
                    candidate=candidate,
                    campaign_manifest=manifest,
                    m2_binding=m2,
                    s0_result=s0,
                    verification=verify,
                    package_audit=placeholder_audit,
                    source_tree_hash=source_hash,
                    environment_manifest=_environment_manifest(),
                )
                store.update_candidate(queue, cand_id, state='PACKAGE_AUDIT')
                audit = audit_lineage_package(selected_dir)
                atomic_write_json(selected_dir / 'package_audit.json', audit)
                # Refresh hashes after audit overwrite
                from ..common import sha256_file
                hash_lines = []
                for path in sorted(selected_dir.iterdir()):
                    if path.name in {'hashes.sha256', 'COMPLETED.json'} or not path.is_file():
                        continue
                    hash_lines.append(f'{sha256_file(path)}  {path.name}')
                (selected_dir / 'hashes.sha256').write_text(
                    '\n'.join(hash_lines) + '\n', encoding='utf-8',
                )
                audit = audit_lineage_package(selected_dir)
                atomic_write_json(selected_dir / 'package_audit.json', audit)

                if not audit.get('accepted'):
                    raise CampaignFatalError(f'lineage audit failed: {cand_id}')

                store.update_candidate(queue, cand_id, state='SELECTED', lease=None)
                store.record_selected(cand_id, str(selected_dir))
                store.record_event({
                    'type': 'selected',
                    'candidate_id': cand_id,
                    'q_upper': screen.get('q_upper'),
                })
                if cfg.stop_after_first_verified_q_lt_1:
                    terminal_reason = TERMINAL_Q_LT_1
                    break

            except NeedCanonicalM2 as exc:
                store.record_exception(exc)
                terminal_reason = TERMINAL_NEED_M2
                ledger = store.load_ledger()
                ledger['campaign_state'] = 'BLOCKED_NEED_CANONICAL_M2'
                store.save_ledger(ledger)
                break
            except TimeBudgetClosed as exc:
                store.record_exception(exc)
                archive_and_note(
                    campaign_root / 'archive',
                    list(archived_ids),
                    candidate=candidate,
                    screening_result=None,
                    reason_code='INSUFFICIENT_TIME_BUDGET',
                )
                archived_ids.add(cand_id)
                store.update_candidate(queue, cand_id, state='ARCHIVED', lease=None)
                terminal_reason = TERMINAL_TIME
                break
            except InvariantViolation as exc:
                store.record_exception(exc)
                terminal_reason = TERMINAL_FAIL
                ledger = store.load_ledger()
                ledger['campaign_state'] = 'FAIL_CLOSED'
                store.save_ledger(ledger)
                break
            except CampaignFatalError as exc:
                store.record_exception(exc)
                terminal_reason = TERMINAL_FAIL
                ledger = store.load_ledger()
                ledger['campaign_state'] = 'FAIL_CLOSED'
                store.save_ledger(ledger)
                break

        ledger = store.load_ledger()
        ledger['archived_ids'] = sorted(archived_ids)
        if terminal_reason:
            ledger['terminal_reason'] = terminal_reason
        if ledger.get('campaign_state') in {'RUNNING', 'ADMISSION_CLOSED', 'FINALIZING'}:
            ledger['campaign_state'] = 'COMPLETE'
        store.save_ledger(ledger)
        store.set_terminal_reason(str(ledger.get('terminal_reason') or terminal_reason or TERMINAL_EXHAUSTED))

    except CampaignFatalError as exc:
        store.record_exception(exc)
        store.set_terminal_reason(TERMINAL_FAIL)
        store.set_campaign_state('FAIL_CLOSED')
        ledger = store.load_ledger()
        queue = store.load_or_init([], campaign_run_id=cfg.campaign_run_id or '')
        return finalize_campaign(
            campaign_root=campaign_root,
            manifest=manifest,
            queue=queue,
            ledger=ledger,
            budget=budget,
            terminal_reason=TERMINAL_FAIL,
        )

    ledger = store.load_ledger()
    queue = read_json(campaign_root / 'queue.json')
    if not isinstance(queue, dict):
        queue = {'candidates': []}
    return finalize_campaign(
        campaign_root=campaign_root,
        manifest=manifest,
        queue=queue,
        ledger=ledger,
        budget=budget,
        terminal_reason=ledger.get('terminal_reason'),
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        print('usage: python -m src.campaign_b.driver <config.yaml>', file=sys.stderr)
        return 2
    summary = run_campaign_b(Path(args[0]))
    print(summary.get('terminal_reason'), summary.get('selected_count'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
