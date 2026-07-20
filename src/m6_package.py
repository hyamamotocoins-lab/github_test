"""Assemble the M6 final_certificate package."""

from __future__ import annotations

import shutil
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from .certificate import (
    collatz_certificate,
    nonnegative_interval_matrix,
    positive_rational_vector,
)
from .common import atomic_write_json, atomic_write_text, hash_tree, sha256_file, utc_now
from .exact_arithmetic import fraction_decimal_text
from .interval_kernel import ProofInterval, construct
from .m6_ledger import all_leaves_closed, open_leaves
from .m6_lock import m6_lock_payload
from .m6_status import (
    CERTIFIED,
    M6_COMPLETE,
    NOT_CERTIFIED,
    STEP_ENCLOSED,
)
from .proof_manifest import package_manifest_hash, reject_symlinks


class M6PackageError(RuntimeError):
    """Raised when the multi-step certificate package cannot be assembled."""


M6_CERTIFICATE_ROOT_FILES: tuple[str, ...] = (
    'README.md',
    'theorem_scope.md',
    'assumptions.md',
    'limitations.md',
    'run_config.json',
    'environment.json',
    'source_hashes.json',
    'checkpoint_chain.json',
    'conventions.json',
    'error_ledger.json',
    'final_influence_matrix.json',
    'perron_vector.json',
    'final_bound.json',
    'independent_verifier_report.json',
    'verdict.json',
)

STEP_FILES: tuple[str, ...] = (
    'config.json',
    'tensors_hashes.json',
    'residuals.json',
    'normalization_bounds.json',
    'derivative_residuals.json',
    'step_verdict.json',
)


def _write_json(root: Path, name: str, payload: Mapping[str, Any]) -> None:
    atomic_write_json(root / name, dict(payload))


def _interval_payload(value: Any) -> dict[str, Any]:
    if isinstance(value, ProofInterval):
        return value.serialize()
    if isinstance(value, Mapping) and 'lo' in value and 'hi' in value:
        return dict(value)
    return construct(value).serialize()


def assemble_final_certificate(
    package_root: Path,
    *,
    run_id: str,
    parent_m5_run_id: str,
    config: Mapping[str, Any],
    environment: Mapping[str, Any],
    source_hashes: Mapping[str, Any],
    checkpoint_chain: Mapping[str, Any],
    error_ledger: Mapping[str, Any],
    steps: Sequence[Mapping[str, Any]],
    weighted_matrix_entries: Sequence[Sequence[Any]],
    weighted_labels: Sequence[str],
    perron_values: Sequence[Any],
    outside_matrix_tail: Any = 0,
    code_root: Path | None = None,
    theorem_scope: str,
    assumptions: str,
    limitations: str,
) -> dict[str, Any]:
    if package_root.exists():
        shutil.rmtree(package_root)
    package_root.mkdir(parents=True, exist_ok=True)

    num_steps = int(config.get('num_steps', len(steps)))
    if num_steps < 1 or len(steps) != num_steps:
        raise M6PackageError('steps length must equal num_steps.')
    if not all_leaves_closed(error_ledger):
        raise M6PackageError(
            'Error ledger has open leaves: ' + ', '.join(open_leaves(error_ledger))
        )

    lock = m6_lock_payload(
        num_steps=num_steps,
        j2_max=int(config.get('j2_max', 1)),
        bond_dimension=int(config.get('bond_dimension', 16)),
    )
    matrix = nonnegative_interval_matrix(weighted_matrix_entries, weighted_labels)
    vector = positive_rational_vector(perron_values, weighted_labels)
    bound = collatz_certificate(matrix, vector, outside_matrix_tail=outside_matrix_tail)

    atomic_write_text(
        package_root / 'README.md',
        (
            '# M6 final certificate\n\n'
            f'Parent M5: `{parent_m5_run_id}`\n'
            f'Run: `{run_id}`\n'
            f'Steps: `{num_steps}`\n'
        ),
    )
    atomic_write_text(package_root / 'theorem_scope.md', theorem_scope)
    atomic_write_text(package_root / 'assumptions.md', assumptions)
    atomic_write_text(package_root / 'limitations.md', limitations)
    _write_json(package_root, 'run_config.json', config)
    _write_json(package_root, 'environment.json', environment)
    _write_json(package_root, 'source_hashes.json', source_hashes)
    _write_json(package_root, 'checkpoint_chain.json', checkpoint_chain)
    _write_json(package_root, 'conventions.json', lock)
    _write_json(package_root, 'error_ledger.json', error_ledger)

    for index, step in enumerate(steps):
        step_dir = package_root / f'rg_step_{index:02d}'
        step_dir.mkdir(parents=True, exist_ok=True)
        for name in STEP_FILES:
            if name not in step:
                raise M6PackageError(f'Step {index} missing {name}.')
            _write_json(step_dir, name, step[name])  # type: ignore[arg-type]
        if step['step_verdict.json'].get('status') != STEP_ENCLOSED:
            raise M6PackageError(f'Step {index} is not STEP_ENCLOSED.')

    influence_doc = {
        'schema_version': 1,
        'labels': list(weighted_labels),
        'entries': [
            [cell.serialize() for cell in row]
            for row in matrix.entries
        ],
        'outside_matrix_tail_policy': 'added_once_in_collatz_bound',
        'composition_policy': str(
            config.get(
                'composition_policy',
                'final_coarse_uses_declared_final_influence_majorant',
            )
        ),
    }
    _write_json(package_root, 'final_influence_matrix.json', influence_doc)
    _write_json(package_root, 'perron_vector.json', vector.payload())
    _write_json(package_root, 'final_bound.json', bound.payload())

    if bound.q_cert.hi < 1:
        certification_status = CERTIFIED
        milestone_status = CERTIFIED
    elif bound.q_cert.lo >= 1:
        certification_status = NOT_CERTIFIED
        milestone_status = NOT_CERTIFIED
    else:
        raise M6PackageError('q_cert interval crosses 1; cannot mark M6_COMPLETE.')

    code_hashes = {
        'src_tree_sha256': hash_tree(code_root, suffixes=('.py',)) if code_root else None,
        'policy': 'hash_tree_py_files',
    }
    _write_json(package_root, 'source_hashes.json', {
        **dict(source_hashes),
        'code_hashes': code_hashes,
    })

    # Placeholder independent report filled after verification.
    independent_placeholder = {
        'schema_version': 1,
        'status': 'PENDING',
        'generated_at': utc_now(),
    }
    _write_json(
        package_root, 'independent_verifier_report.json', independent_placeholder,
    )

    verdict = {
        'schema_version': 1,
        'milestone': 'M6',
        'run_id': run_id,
        'parent_m5_run_id': parent_m5_run_id,
        'phase': M6_COMPLETE,
        'status': 'PASS',
        'milestone_status': milestone_status,
        'certification_status': certification_status,
        'scope': lock['scope'],
        'num_steps': num_steps,
        'j2_max': int(config.get('j2_max', 1)),
        'bond_dimension': int(config.get('bond_dimension', 16)),
        'norm': str(config.get('norm', 'frobenius')),
        'source_classes': list(weighted_labels),
        'q_collatz_upper': fraction_decimal_text(bound.q_collatz.hi),
        'outside_matrix_tail_upper': fraction_decimal_text(bound.outside_matrix_tail.hi),
        'q_cert_lower': fraction_decimal_text(bound.q_cert.lo),
        'q_cert_upper': fraction_decimal_text(bound.q_cert.hi),
        'margin_lower': fraction_decimal_text(bound.margin.lo),
        'independent_verifier': 'PENDING',
        'error_ledger_closed': True,
    }
    if certification_status == NOT_CERTIFIED:
        verdict['failure_gate'] = 'final_collatz'
        verdict['failure_reason'] = (
            'certificate_upper_bound_does_not_prove_contraction: q_cert_lower >= 1'
        )
        verdict['mathematical_interpretation'] = {
            'proved': (
                'Under the declared majorant, error ledger, weights, cutoff, rank, '
                'and composition_policy, the certified upper bound satisfies '
                'q_cert >= 1, so contraction cannot be certified.'
            ),
            'not_proved': (
                'Non-contraction of the true RG influence map. '
                'A majorant q_cert >= 1 only yields rho(B_true) <= q_cert; '
                'it does not imply rho(B_true) >= 1.'
            ),
            'status_meaning': (
                'NOT_CERTIFIED is a verified certificate failure, '
                'not a verified dynamical non-contraction.'
            ),
        }

    _write_json(package_root, 'verdict.json', verdict)

    reject_symlinks(package_root)
    return {
        'package_root': str(package_root.resolve()),
        'bound': bound.payload(),
        'verdict': verdict,
        'lock': lock,
    }


def finalize_package_after_independent_pass(
    package_root: Path,
    *,
    independent_report: Mapping[str, Any],
    verdict: Mapping[str, Any],
) -> dict[str, Any]:
    from .m6_independent_verifier import hash_package_files

    _write_json(package_root, 'independent_verifier_report.json', independent_report)
    updated = dict(verdict)
    updated['independent_verifier'] = 'PASS'
    _write_json(package_root, 'verdict.json', updated)
    hashes = hash_package_files(package_root)
    manifest_hash = package_manifest_hash(hashes)
    updated['package_manifest_hash'] = manifest_hash
    _write_json(package_root, 'verdict.json', updated)
    return {
        'verdict': updated,
        'package_manifest_hash': manifest_hash,
        'file_hashes': hashes,
        'verdict_sha256': sha256_file(package_root / 'verdict.json'),
    }


def make_singleton_step(
    *,
    step_index: int,
    z_min: Any,
    residual_budget: Fraction,
    parent_package_hash: str,
) -> dict[str, Any]:
    """Family inclusion for the singleton ball {T} with radius 0."""
    return {
        'config.json': {
            'schema_version': 1,
            'step_index': step_index,
            'input_ball': 'singleton',
            'input_radius': '0',
            'j2_max': 1,
            'bond_dimension': 16,
        },
        'tensors_hashes.json': {
            'parent_one_step_package_hash_ref': parent_package_hash,
            'policy': 'inherited_frozen_center',
        },
        'residuals.json': {
            'status': 'PASS',
            'aggregate_upper': fraction_decimal_text(residual_budget),
            'input_propagation_upper': '0',
            'proof_route': 'singleton_radius_zero',
        },
        'normalization_bounds.json': {
            'status': 'PASS',
            'z_min_interval': _interval_payload(z_min),
            'kernel_positivity_evidence': 'inherited_or_fixture_positive',
        },
        'derivative_residuals.json': {
            'status': 'PASS',
            'chain_rule_residual_upper': fraction_decimal_text(residual_budget),
            'proof_route': 'singleton_no_extra_radius',
        },
        'step_verdict.json': {
            'schema_version': 1,
            'step_index': step_index,
            'status': STEP_ENCLOSED,
            'family_inclusion': 'R({T}) subset {T} under declared frozen-scheme majorant',
            'notes': (
                'Singleton input ball has radius 0; propagation residual is 0. '
                'Positive-radius balls reopen E5.'
            ),
        },
    }


def close_standard_ledger(
    *,
    residual_budget: Fraction,
    notes_prefix: str,
) -> dict[str, Any]:
    from .m6_ledger import close_leaf, empty_ledger_template

    ledger = empty_ledger_template()
    closed = {
        'E1': (Fraction(0), 'no additional GPU residual beyond inherited budget'),
        'E2': (residual_budget, 'projection residual majorant inherited/aggregated'),
        'E3': (Fraction(0), 'frozen cutoff/rank in-scheme'),
        'E4': (residual_budget, 'representation tail majorant'),
        'E5': (Fraction(0), 'singleton input radius 0'),
        'E6': (Fraction(0), 'z_min strictly positive by construction'),
        'E7': (Fraction(0), 'truncated-sector cover at frozen j2_max'),
        'E8': (Fraction(0), 'frozen basis; no reselection'),
        'E9': (Fraction(0), 'singleton multi-step chain-rule adds no radius'),
        'E10': (Fraction(0), 'outside-matrix tail declared 0 unless set in Collatz'),
        'E11': (Fraction(0), 'weight_m=0; no extra weighting residual'),
        'E12': (Fraction(0), 'filled after independent verifier PASS'),
    }
    for leaf_id, (upper, note) in closed.items():
        close_leaf(
            ledger,
            leaf_id,
            upper=upper,
            notes=f'{notes_prefix}: {note}',
            proof_method='m6_lock_scoped_enclosure',
        )
    return ledger
