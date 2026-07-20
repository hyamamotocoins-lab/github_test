"""Exact file-set and dependency validation for one-step certificate packages."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .common import atomic_write_json, canonical_json_bytes, sha256_bytes, sha256_file


class ProofManifestError(RuntimeError):
    """Raised when a certificate package fails integrity validation."""


ONE_STEP_CERTIFICATE_FILES: tuple[str, ...] = (
    'theorem_statement.md',
    'config.json',
    'code_hashes.json',
    'conventions.json',
    'initial_tail.json',
    'basis_equivalence.json',
    'contraction_residuals.json',
    'derivative_residuals.json',
    'normalization_bounds.json',
    'influence_matrix_intervals.json',
    'perron_vector.json',
    'collatz_bound.json',
    'proof_dependencies.json',
    'verdict.json',
)


@dataclass(frozen=True, slots=True)
class ProofDependency:
    node_id: str
    depends_on: tuple[str, ...]
    artifact: str

    def payload(self) -> dict[str, Any]:
        return {
            'node_id': self.node_id,
            'depends_on': list(self.depends_on),
            'artifact': self.artifact,
        }


def canonical_relative_path(path: Path, root: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ProofManifestError(f'Path escapes package root: {path}') from exc
    text = relative.as_posix()
    if text.startswith('../') or text in {'.', '..'} or '\\' in text:
        raise ProofManifestError(f'Unsafe relative path: {text}')
    return text


def reject_symlinks(root: Path) -> None:
    if root.is_symlink():
        raise ProofManifestError(f'Package root must not be a symlink: {root}')
    for path in root.rglob('*'):
        if path.is_symlink():
            raise ProofManifestError(f'Symlink rejected in certificate package: {path}')


def exact_file_set(root: Path, expected: Sequence[str] = ONE_STEP_CERTIFICATE_FILES) -> set[str]:
    reject_symlinks(root)
    if not root.is_dir():
        raise ProofManifestError(f'Certificate package is missing: {root}')
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob('*')
        if path.is_file()
    }
    expected_set = set(expected)
    missing = expected_set - actual
    extra = actual - expected_set
    if missing:
        raise ProofManifestError(
            'Certificate package is missing required files: ' + ', '.join(sorted(missing))
        )
    if extra:
        raise ProofManifestError(
            'Certificate package has untracked extra files: ' + ', '.join(sorted(extra))
        )
    return actual


def file_hashes(root: Path, files: Iterable[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in sorted(files):
        path = root / relative
        if path.is_symlink() or not path.is_file():
            raise ProofManifestError(f'Missing or unsafe package file: {relative}')
        hashes[relative] = sha256_file(path)
    return hashes


def package_manifest_hash(file_hashes_map: Mapping[str, str]) -> str:
    return sha256_bytes(canonical_json_bytes(dict(sorted(file_hashes_map.items()))))


def validate_dependency_dag(nodes: Sequence[ProofDependency]) -> None:
    graph = {node.node_id: set(node.depends_on) for node in nodes}
    if len(graph) != len(nodes):
        raise ProofManifestError('Proof dependency node IDs are not unique.')
    for node in nodes:
        for parent in node.depends_on:
            if parent not in graph:
                raise ProofManifestError(f'Missing proof dependency: {parent}')
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ProofManifestError('Proof dependency DAG contains a cycle.')
        if node_id in visited:
            return
        visiting.add(node_id)
        for parent in graph[node_id]:
            visit(parent)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in graph:
        visit(node_id)


def verify_immutable_package(
    package_root: Path,
    *,
    expected_files: Sequence[str] = ONE_STEP_CERTIFICATE_FILES,
    expected_hashes: Mapping[str, str] | None = None,
    dependencies: Sequence[ProofDependency] | None = None,
) -> dict[str, Any]:
    files = exact_file_set(package_root, expected_files)
    hashes = file_hashes(package_root, files)
    if expected_hashes is not None:
        expected = dict(sorted(expected_hashes.items()))
        if hashes != expected:
            raise ProofManifestError('Certificate package hash tampering detected.')
    if dependencies is not None:
        validate_dependency_dag(dependencies)
        dependency_artifacts = {node.artifact for node in dependencies}
        if not dependency_artifacts <= files:
            raise ProofManifestError('Proof dependency references a missing artifact.')
    manifest = {
        'schema_version': 1,
        'package_root': str(package_root.resolve()),
        'exact_file_set': sorted(files),
        'file_hashes': hashes,
        'package_manifest_hash': package_manifest_hash(hashes),
    }
    return manifest


def write_certificate_manifest(
    report_path: Path,
    package_root: Path,
    *,
    dependencies: Sequence[ProofDependency] | None = None,
) -> dict[str, Any]:
    manifest = verify_immutable_package(
        package_root, dependencies=dependencies,
    )
    if dependencies is not None:
        manifest['dependencies'] = [node.payload() for node in dependencies]
    atomic_write_json(report_path, manifest)
    return manifest


def load_proof_dependencies(path: Path) -> list[ProofDependency]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    nodes = payload.get('nodes') if isinstance(payload, dict) else None
    if not isinstance(nodes, list):
        raise ProofManifestError('proof_dependencies.json is malformed.')
    result: list[ProofDependency] = []
    for item in nodes:
        if not isinstance(item, dict):
            raise ProofManifestError('Proof dependency node is malformed.')
        depends = item.get('depends_on', [])
        if not isinstance(depends, list) or any(not isinstance(x, str) for x in depends):
            raise ProofManifestError('Proof dependency edge list is malformed.')
        result.append(
            ProofDependency(
                node_id=str(item['node_id']),
                depends_on=tuple(depends),
                artifact=str(item['artifact']),
            )
        )
    validate_dependency_dag(result)
    return result
