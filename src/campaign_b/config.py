"""Load and validate Campaign B six-hour autonomous config."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..common import canonical_json_bytes, sha256_bytes, utc_now
from .budget import (
    ADMISSION_CLOSE_SEC,
    EMERGENCY_FLUSH_SEC,
    FINALIZATION_START_SEC,
    HARD_LIMIT_SEC,
)
from .errors import CampaignFatalError, InvariantViolation
from .schemas import CERTIFICATION_STATUS, CLAIM_SCOPE, screening_only_payload


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    for index, char in enumerate(line):
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == '#' and not in_single and not in_double:
            return line[:index].rstrip()
    return line.rstrip()


def _parse_scalar(raw: str) -> Any:
    text = raw.strip()
    if not text:
        return ''
    if text in {'null', 'Null', 'NULL', '~'}:
        return None
    if text in {'true', 'True', 'TRUE'}:
        return True
    if text in {'false', 'False', 'FALSE'}:
        return False
    if (text.startswith('"') and text.endswith('"')) or (
        text.startswith("'") and text.endswith("'")
    ):
        return text[1:-1]
    if re.fullmatch(r'-?\d+', text):
        return int(text)
    if re.fullmatch(r'-?\d+(\.\d+)?([eE][+-]?\d+)?', text):
        return float(text)
    return text


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Minimal YAML subset loader for Campaign B configs (no anchors)."""
    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(-1, root)]
    pending_key: str | None = None

    for raw_line in text.splitlines():
        line = _strip_yaml_comment(raw_line)
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(' '))
        content = line.strip()
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith('- '):
            item_raw = content[2:].strip()
            if isinstance(parent, dict) and not parent:
                # Convert placeholder mapping into a list under the pending key.
                for depth, node in reversed(stack[:-1]):
                    if isinstance(node, dict):
                        for key, value in list(node.items()):
                            if value is parent:
                                new_list: list[Any] = []
                                node[key] = new_list
                                stack[-1] = (stack[-1][0], new_list)
                                parent = new_list
                                break
                        else:
                            continue
                        break
            if not isinstance(parent, list):
                raise CampaignFatalError('YAML list item without list parent')
            if ':' in item_raw and not item_raw.startswith('{'):
                key, _, value = item_raw.partition(':')
                item: dict[str, Any] = {key.strip(): _parse_scalar(value)}
                parent.append(item)
                stack.append((indent, item))
            else:
                parent.append(_parse_scalar(item_raw))
            continue

        if ':' not in content:
            raise CampaignFatalError(f'unsupported YAML line: {content!r}')
        key, _, value = content.partition(':')
        key = key.strip()
        value = value.strip()
        if not isinstance(parent, dict):
            raise CampaignFatalError('YAML mapping entry without dict parent')
        if value == '':
            # Lookahead-free: create dict; if next lines are list, replace.
            child: Any = {}
            parent[key] = child
            stack.append((indent, child))
            pending_key = key
            continue
        if value.startswith('[') and value.endswith(']'):
            inner = value[1:-1].strip()
            if not inner:
                parent[key] = []
            else:
                parent[key] = [_parse_scalar(part.strip()) for part in inner.split(',')]
            continue
        parent[key] = _parse_scalar(value)
        pending_key = None

    # Convert empty dicts that should be lists is not needed for our configs.
    _ = pending_key
    return root


def load_config_dict(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding='utf-8')
    if path.suffix.lower() == '.json':
        payload = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
            payload = yaml.safe_load(text)
        except Exception:
            payload = _load_simple_yaml(text)
    if not isinstance(payload, dict):
        raise CampaignFatalError(f'config root must be object: {path}')
    return payload


@dataclass
class CampaignBConfig:
    raw: dict[str, Any]
    config_path: Path
    persistent_root: Path
    time_budget_sec: float = HARD_LIMIT_SEC
    admission_close_sec: float = ADMISSION_CLOSE_SEC
    finalization_start_sec: float = FINALIZATION_START_SEC
    emergency_flush_sec: float = EMERGENCY_FLUSH_SEC
    # Resume-first: do not hard-stop on the six-hour window unless explicitly on.
    enforce_wall_clock: bool = False
    screening_margin: float = 1e-6
    stop_after_first_verified_q_lt_1: bool = True
    allow_generate_canonical_m2: bool = False
    on_missing_m2: str = 'stop_campaign'
    search_space: dict[str, Any] = field(default_factory=dict)
    search_space_path: Path | None = None
    parent_evidence: dict[str, Any] = field(default_factory=dict)
    parent_q_upper: float = 1.011045
    parent_rank: int = 16
    parent_m6_run_id: str = 'PARENT-M6-UNKNOWN'
    parent_scheme_hash: str = '0' * 64
    structural_key: str | None = None
    proof_key: str | None = None
    source_tree_roots: list[Path] = field(default_factory=list)
    campaign_run_id: str | None = None
    resume_campaign_run_id: str | None = None
    # Fresh session window on resume; queue/ledger still resume from disk.
    inherit_deadline: bool = False
    candidate_limit: int | None = None
    q_atol: float = 1e-9
    q_rtol: float = 1e-6
    lease_sec: int = 1800

    def campaign_root(self) -> Path:
        run_id = self.campaign_run_id or 'UNSET'
        return Path(self.persistent_root) / 'campaign_b' / run_id

    def execution_policy(self) -> dict[str, Any]:
        return {
            'staged_only': True,
            'minimum_j2': 2,
            'allow_campaign_c': False,
            'allow_production_m6': False,
            **screening_only_payload(),
        }


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise CampaignFatalError(f'missing config key: {key}')
    return mapping[key]


def validate_search_space(space: dict[str, Any]) -> dict[str, Any]:
    if space.get('campaign') not in {'B_S2', 'B'}:
        raise InvariantViolation('search space campaign must be B_S2 or B')
    staging = space.get('staging') or {}
    j2_values = [int(v) for v in staging.get('j2_values') or []]
    if not j2_values:
        raise CampaignFatalError('search space staging.j2_values required')
    if any(v < 2 for v in j2_values):
        raise InvariantViolation('forbid j2<2 in Campaign B search space')
    if staging.get('forbid_j2_1') is False:
        raise InvariantViolation('forbid_j2_1 must remain true')
    # Reject accidental Campaign C layers.
    layers = space.get('layers') or space
    for forbidden in ('j2_max', 'channel_policy', 'block_geometry'):
        if forbidden in layers and forbidden not in ('staging',):
            # j2_max is C-specific; staging.j2_values is allowed separately.
            if forbidden == 'j2_max' or forbidden in space.get('layers', {}):
                raise InvariantViolation(
                    f'Campaign C layer {forbidden!r} forbidden in B search space'
                )
    return space


def load_campaign_b_config(path: Path) -> CampaignBConfig:
    path = Path(path)
    raw = load_config_dict(path)
    if raw.get('campaign') not in {'B_S2', 'B', None}:
        # top-level may omit; nested ok
        pass
    campaign = raw.get('campaign') or raw.get('campaign_kind') or 'B_S2'
    if campaign not in {'B_S2', 'B'}:
        raise InvariantViolation(f'config campaign must be B_S2, got {campaign!r}')

    policy = raw.get('execution_policy') or {}
    if policy.get('allow_campaign_c') is True:
        raise InvariantViolation('allow_campaign_c must be false')
    if policy.get('allow_production_m6') is True:
        raise InvariantViolation('allow_production_m6 must be false')
    if policy.get('staged_only') is False:
        raise InvariantViolation('staged_only must be true')
    if int(policy.get('minimum_j2', 2)) < 2:
        raise InvariantViolation('minimum_j2 must be >= 2')
    if policy.get('certification_status') not in {None, CERTIFICATION_STATUS}:
        raise InvariantViolation('certification_status must be NOT_CERTIFIED')
    if policy.get('claim_scope') not in {None, CLAIM_SCOPE}:
        raise InvariantViolation('claim_scope must be SCREENING_ONLY')

    shared_m2 = raw.get('shared_m2') or {}
    allow_gen = bool(shared_m2.get('allow_generate_canonical', False))
    on_missing = str(shared_m2.get('on_missing', 'stop_campaign'))

    space_path = raw.get('search_space_path')
    if space_path:
        space_ref = Path(space_path)
        space_file = (
            space_ref.resolve()
            if space_ref.is_absolute()
            else (path.parent / space_ref).resolve()
        )
        space = load_config_dict(space_file)
    elif 'search_space' in raw:
        space = dict(raw['search_space'])
        space_file = None
    else:
        raise CampaignFatalError('search_space_path or search_space required')
    validate_search_space(space)

    parent = dict(raw.get('parent_evidence') or {})
    roots = [
        Path(p) for p in (raw.get('source_tree_roots') or ['src'])
    ]

    budget = raw.get('time_budget') or {}
    cfg = CampaignBConfig(
        raw=raw,
        config_path=path.resolve(),
        persistent_root=Path(_require(raw, 'persistent_root')),
        time_budget_sec=float(budget.get('hard_limit_sec', raw.get('time_budget_sec', HARD_LIMIT_SEC))),
        admission_close_sec=float(budget.get('admission_close_sec', ADMISSION_CLOSE_SEC)),
        finalization_start_sec=float(budget.get('finalization_start_sec', FINALIZATION_START_SEC)),
        emergency_flush_sec=float(budget.get('emergency_flush_sec', EMERGENCY_FLUSH_SEC)),
        enforce_wall_clock=bool(
            budget.get(
                'enforce_wall_clock',
                raw.get('enforce_wall_clock', False),
            )
        ),
        screening_margin=float(raw.get('screening_margin', 1e-6)),
        stop_after_first_verified_q_lt_1=bool(
            raw.get('stop_after_first_verified_q_lt_1', True)
        ),
        allow_generate_canonical_m2=allow_gen,
        on_missing_m2=on_missing,
        search_space=space,
        search_space_path=space_file,
        parent_evidence=parent,
        parent_q_upper=float(raw.get('parent_q_upper', parent.get('campaign_c_best_q', 1.011045))),
        parent_rank=int(raw.get('parent_rank', 16)),
        parent_m6_run_id=str(raw.get('parent_m6_run_id', 'PARENT-M6-UNKNOWN')),
        parent_scheme_hash=str(raw.get('parent_scheme_hash', '0' * 64)),
        structural_key=raw.get('structural_key'),
        proof_key=raw.get('proof_key'),
        source_tree_roots=roots,
        campaign_run_id=raw.get('campaign_run_id'),
        resume_campaign_run_id=raw.get('resume_campaign_run_id'),
        inherit_deadline=bool(raw.get('inherit_deadline', False)),
        candidate_limit=(
            int(raw['candidate_limit']) if raw.get('candidate_limit') is not None else None
        ),
        q_atol=float((raw.get('independent_verifier') or {}).get('q_atol', 1e-9)),
        q_rtol=float((raw.get('independent_verifier') or {}).get('q_rtol', 1e-6)),
        lease_sec=int(raw.get('lease_sec', 1800)),
    )
    return cfg


def mint_campaign_run_id() -> str:
    stamp = utc_now().replace('-', '').replace(':', '').replace('.', '')
    # Compact: YYYYMMDDTHHMMSSZ
    compact = stamp[:15] + 'Z' if 'T' in stamp else stamp
    digest = hashlib.sha256(f'{compact}-{utc_now()}'.encode()).hexdigest()[:12]
    return f'M7-{compact}-b-{digest}'


def search_space_hash(space: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json_bytes(space))
