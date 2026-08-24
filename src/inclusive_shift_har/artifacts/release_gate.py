"""Git-object release security scans and exact-candidate attestations.

The scanners in this module never walk raw-data directories. Repository content
is read through Git object IDs, and only blobs whose paths already pass the
forbidden-path/extension policy are inspected for disguised binary payloads.
Every generated record is create-only and canonically self-hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.0.0"
REPORT_KIND = "final_release_gate_report"
SCAN_KIND = "git_object_release_scan"
SECRET_KIND = "gitleaks_secret_scan_attestation"
LICENSE_KIND = "python_license_audit"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
GATE_NAMES = (
    "repository_scan",
    "secret_scan",
    "license_audit",
    "tests",
    "lint",
    "format",
    "types",
    "manifests",
    "splits",
    "configuration",
    "artifacts",
    "ci",
)
TEXT_SUFFIXES = {
    "",
    ".bib",
    ".cff",
    ".csv",
    ".gitignore",
    ".gitleaksignore",
    ".json",
    ".lock",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".typed",
    ".yaml",
    ".yml",
}


class ReleaseGateError(RuntimeError):
    """Raised when release evidence is unsafe, incomplete, or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseGateError(f"{name} must be a string-keyed object")
    return value


def _array(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseGateError(f"{name} must be an array")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseGateError(f"{name} must be a non-empty string")
    return value


def _commit(value: Any, *, name: str = "commit") -> str:
    result = _text(value, name=name).casefold()
    if COMMIT_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a full lowercase Git commit")
    return result


def _sha256(value: Any, *, name: str) -> str:
    result = _text(value, name=name).casefold()
    if SHA256_RE.fullmatch(result) is None:
        raise ReleaseGateError(f"{name} must be a lowercase SHA-256")
    return result


def _timestamp(value: Any) -> str:
    result = _text(value, name="created_at_utc")
    if not result.endswith("Z"):
        raise ReleaseGateError("created_at_utc must use a UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(result[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseGateError("created_at_utc must be valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseGateError("created_at_utc must be UTC")
    return result


def _safe_relative(value: Any, *, name: str) -> str:
    result = _text(value, name=name)
    path = PurePosixPath(result)
    if (
        "\\" in result
        or ":" in result
        or "\x00" in result
        or path.is_absolute()
        or path.as_posix() != result
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ReleaseGateError(f"{name} must be a portable POSIX relative path")
    return result


def _git(
    root: Path, arguments: Sequence[str], *, binary: bool = False, check: bool = True
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=check,
            capture_output=True,
            text=not binary,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseGateError(f"Git command failed: {' '.join(arguments)}") from exc


def _repository(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("repository_root may not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseGateError("repository_root must be a directory")
    inside = _git(root, ("rev-parse", "--is-inside-work-tree"))
    assert isinstance(inside.stdout, str)
    if inside.stdout.strip() != "true":
        raise ReleaseGateError("repository_root is not a Git worktree")
    return root


def _strict_json(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        return _mapping(load_json_strict(path), name=name)
    except (OSError, ValueError) as exc:
        raise ReleaseGateError(f"cannot load {name}: {exc}") from exc


def _policy(path: str | Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseGateError("policy may not be a symlink")
    payload = source.resolve(strict=True).read_bytes()
    value = _strict_json(source, name="release gate policy")
    if value.get("schema_version") != "1.0.0" or value.get("policy_kind") != (
        "final_release_gate_policy"
    ):
        raise ReleaseGateError("release gate policy version or kind differs")
    reference = {
        "path": "configs/release/release_gate_policy_v1.json",
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }
    return value, reference


def _write_new(record: dict[str, Any], destination: str | Path, *, root: Path) -> None:
    output = Path(destination)
    if not output.is_absolute():
        output = root / output
    prospective = output.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError("destination escapes repository_root") from exc
    if any(parent.is_symlink() for parent in (prospective, *prospective.parents) if parent != root):
        raise ReleaseGateError("destination may not traverse a symlink")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json_new(record, output, allowed_root=root)


def _self_hash(body: dict[str, Any]) -> dict[str, Any]:
    result = dict(body)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_self_hash(value: Mapping[str, Any], *, name: str) -> str:
    body = dict(value)
    observed = _sha256(body.pop("record_sha256", None), name=f"{name}.record_sha256")
    if canonical_json_sha256(body) != observed:
        raise ReleaseGateError(f"{name} self-hash does not reconstruct")
    return observed


def _tree_entries(root: Path, commit: str) -> list[tuple[str, str, str, int | None, str]]:
    result = _git(root, ("ls-tree", "-r", "-z", "-l", commit), binary=True)
    assert isinstance(result.stdout, bytes)
    entries: list[tuple[str, str, str, int | None, str]] = []
    for raw in (item for item in result.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, size_text = metadata.decode("ascii").split(" ", 3)
            path = raw_path.decode("utf-8")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git tree contains undecodable metadata or path") from exc
        size = None if size_text == "-" else int(size_text)
        entries.append((mode, kind, object_id, size, path))
    return entries


def _blob_prefixes(
    root: Path, object_ids: Sequence[str], *, maximum_bytes: int = 4096
) -> dict[str, bytes]:
    """Read many Git blobs through one batch process, retaining only prefixes."""

    if not object_ids:
        return {}
    request = ("\n".join(object_ids) + "\n").encode("ascii")
    try:
        result = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=root,
            input=request,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseGateError("Git batch blob inspection failed") from exc
    payload = result.stdout
    cursor = 0
    prefixes: dict[str, bytes] = {}
    for requested in object_ids:
        line_end = payload.find(b"\n", cursor)
        if line_end < 0:
            raise ReleaseGateError("Git batch response ended before its object header")
        try:
            object_id, kind, size_text = payload[cursor:line_end].decode("ascii").split(" ")
            size = int(size_text)
        except (UnicodeError, ValueError) as exc:
            raise ReleaseGateError("Git batch returned invalid object metadata") from exc
        if object_id != requested or kind != "blob":
            raise ReleaseGateError("Git batch returned an unexpected object")
        start = line_end + 1
        end = start + size
        if end >= len(payload) or payload[end : end + 1] != b"\n":
            raise ReleaseGateError("Git batch returned a truncated blob")
        prefixes[requested] = payload[start : min(end, start + maximum_bytes)]
        cursor = end + 1
    if cursor != len(payload):
        raise ReleaseGateError("Git batch response contains trailing bytes")
    return prefixes


def scan_repository(
    *,
    repository_root: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Scan the candidate tree and every reachable historical Git tree."""

    root = _repository(repository_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    exists = _git(root, ("cat-file", "-e", f"{candidate}^{{commit}}"), check=False)
    if exists.returncode != 0:
        raise ReleaseGateError("candidate_commit is not present in the repository")
    policy, policy_reference = _policy(policy_path)
    maximum = policy.get("maximum_blob_size_bytes")
    if not isinstance(maximum, int) or maximum < 1:
        raise ReleaseGateError("maximum_blob_size_bytes must be a positive integer")
    prefixes = tuple(
        _safe_relative(str(value).rstrip("/"), name="forbidden_path_prefix") + "/"
        for value in _array(policy.get("forbidden_path_prefixes"), name="forbidden prefixes")
    )
    suffixes = {
        _text(value, name="forbidden_suffix").casefold()
        for value in _array(policy.get("forbidden_suffixes"), name="forbidden suffixes")
    }
    signatures = {
        _text(name, name="signature name"): bytes.fromhex(_text(signature, name="signature"))
        for name, signature in _mapping(
            policy.get("disguised_binary_signatures"), name="binary signatures"
        ).items()
    }
    allowed_signatures = {
        str(name): {str(item).casefold() for item in _array(values, name="allowed suffixes")}
        for name, values in _mapping(
            policy.get("allowed_signature_suffixes"), name="allowed signatures"
        ).items()
    }
    revisions = _git(root, ("rev-list", "--all"))
    assert isinstance(revisions.stdout, str)
    commits = sorted(set(revisions.stdout.splitlines()) | {candidate})
    violations: list[dict[str, Any]] = []
    objects: dict[str, tuple[int, str]] = {}
    path_observations = 0
    inspection_targets: dict[str, tuple[str, str]] = {}
    for revision in commits:
        folded_paths: dict[str, str] = {}
        for mode, kind, object_id, size, path in _tree_entries(root, revision):
            path_observations += 1
            folded = path.casefold()
            previous = folded_paths.get(folded)
            if previous is not None and previous != path:
                violations.append(
                    {
                        "code": "case_collision",
                        "commit": revision,
                        "path": path,
                        "detail": f"collides with {previous}",
                    }
                )
            folded_paths[folded] = path
            if mode == "120000":
                violations.append(
                    {"code": "symlink", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if mode == "160000" or kind == "commit":
                violations.append(
                    {"code": "submodule", "commit": revision, "path": path, "detail": mode}
                )
                continue
            if kind != "blob" or size is None:
                violations.append(
                    {
                        "code": "unsupported_tree_entry",
                        "commit": revision,
                        "path": path,
                        "detail": f"{mode} {kind}",
                    }
                )
                continue
            objects.setdefault(object_id, (size, path))
            prohibited_path = any(folded.startswith(prefix.casefold()) for prefix in prefixes)
            prohibited_suffix = PurePosixPath(path).suffix.casefold() in suffixes
            if prohibited_path:
                violations.append(
                    {
                        "code": "forbidden_path",
                        "commit": revision,
                        "path": path,
                        "detail": "protected/raw/transient prefix",
                    }
                )
            if prohibited_suffix:
                violations.append(
                    {
                        "code": "forbidden_extension",
                        "commit": revision,
                        "path": path,
                        "detail": PurePosixPath(path).suffix.casefold(),
                    }
                )
            if size > maximum:
                violations.append(
                    {
                        "code": "oversized_blob",
                        "commit": revision,
                        "path": path,
                        "detail": str(size),
                    }
                )
            if prohibited_path or prohibited_suffix or size > maximum:
                continue
            inspection_targets.setdefault(object_id, (path, revision))
    prefixes_by_object = _blob_prefixes(root, sorted(inspection_targets))
    for object_id, (path, revision) in inspection_targets.items():
        prefix = prefixes_by_object[object_id]
        suffix = PurePosixPath(path).suffix.casefold()
        if suffix in TEXT_SUFFIXES and b"\x00" in prefix:
            violations.append(
                {
                    "code": "disguised_binary_nul",
                    "commit": revision,
                    "path": path,
                    "detail": "NUL byte in text-declared blob",
                }
            )
        for signature_name, signature in signatures.items():
            if prefix.startswith(signature) and suffix not in allowed_signatures.get(
                signature_name, set()
            ):
                violations.append(
                    {
                        "code": "disguised_binary_signature",
                        "commit": revision,
                        "path": path,
                        "detail": signature_name,
                    }
                )
                break
    # Deduplicate violations caused by the same path/object recurring unchanged in history.
    unique_violations = sorted(
        {(item["code"], item["path"], item["detail"]): item for item in violations}.values(),
        key=lambda item: (str(item["code"]), str(item["path"]), str(item["commit"])),
    )
    object_digest_rows = [f"{oid} {size} {path}" for oid, (size, path) in sorted(objects.items())]
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SCAN_KIND,
        "status": "pass" if not unique_violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "scan_scope": "candidate_tree_and_complete_reachable_git_history",
        "policy": policy_reference,
        "commit_count": len(commits),
        "unique_blob_count": len(objects),
        "tree_entry_observation_count": path_observations,
        "scanned_object_index_sha256": hashlib.sha256(
            ("\n".join(object_digest_rows) + "\n").encode()
        ).hexdigest(),
        "violations": unique_violations,
        "raw_worktree_paths_opened": False,
    }
    return _self_hash(body)


def validate_secret_scan(
    *,
    report_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    gitleaks_version: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate a Gitleaks JSON report and bind it to an exact candidate."""

    policy, policy_reference = _policy(policy_path)
    expected_version = _text(
        _mapping(policy.get("gitleaks"), name="gitleaks policy").get("version"),
        name="gitleaks.version",
    )
    if gitleaks_version != expected_version:
        raise ReleaseGateError("Gitleaks version differs from the policy pin")
    source = Path(report_path)
    if source.is_symlink():
        raise ReleaseGateError("Gitleaks report may not be a symlink")
    payload = source.resolve(strict=True).read_bytes()
    try:
        findings = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("Gitleaks report is not UTF-8 JSON") from exc
    if not isinstance(findings, list):
        raise ReleaseGateError("Gitleaks JSON report must be an array")
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": SECRET_KIND,
        "status": "pass" if not findings else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": _commit(candidate_commit, name="candidate_commit"),
        "scan_scope": "complete_reachable_git_history",
        "gitleaks_version": gitleaks_version,
        "policy": policy_reference,
        "raw_report": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "finding_count": len(findings),
    }
    return _self_hash(body)


def audit_licenses(
    *,
    inventory_path: str | Path,
    candidate_commit: str,
    policy_path: str | Path,
    pip_licenses_version: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate pip-licenses JSON against the versioned release policy."""

    if pip_licenses_version != "5.5.5":
        raise ReleaseGateError("pip-licenses version must equal the locked 5.5.5 release tool")
    policy, policy_reference = _policy(policy_path)
    source = Path(inventory_path)
    if source.is_symlink():
        raise ReleaseGateError("license inventory may not be a symlink")
    payload = source.resolve(strict=True).read_bytes()
    try:
        packages = json.loads(payload.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseGateError("license inventory is not UTF-8 JSON") from exc
    values = _array(packages, name="license inventory")
    prohibited = tuple(
        _text(value, name="prohibited license token").casefold()
        for value in _array(policy.get("prohibited_license_tokens"), name="license tokens")
    )
    exceptions = {
        _text(value, name="license exception").casefold()
        for value in _array(policy.get("license_exceptions"), name="license exceptions")
    }
    violations: list[dict[str, str]] = []
    normalized: list[str] = []
    for index, raw_package in enumerate(values):
        package = _mapping(raw_package, name=f"license inventory[{index}]")
        name = _text(package.get("Name"), name=f"license inventory[{index}].Name")
        version = _text(package.get("Version"), name=f"license inventory[{index}].Version")
        license_name = _text(package.get("License"), name=f"license inventory[{index}].License")
        identity = f"{name}@{version}".casefold()
        normalized.append(f"{name}\t{version}\t{license_name}")
        if identity not in exceptions and any(
            token in license_name.casefold() for token in prohibited
        ):
            violations.append({"package": name, "version": version, "license": license_name})
    body = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": LICENSE_KIND,
        "status": "pass" if not violations else "fail",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": _commit(candidate_commit, name="candidate_commit"),
        "pip_licenses_version": pip_licenses_version,
        "policy": policy_reference,
        "inventory": {
            "basename": source.name,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        "package_count": len(values),
        "normalized_inventory_sha256": hashlib.sha256(
            ("\n".join(sorted(normalized, key=str.casefold)) + "\n").encode()
        ).hexdigest(),
        "violations": sorted(violations, key=lambda item: item["package"].casefold()),
    }
    return _self_hash(body)


def _evidence_reference(root: Path, value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    path = _safe_relative(value.get("path"), name=f"{name}.path")
    expected = _sha256(value.get("expected_sha256"), name=f"{name}.expected_sha256")
    source = root / Path(path)
    if source.is_symlink():
        raise ReleaseGateError(f"{name} may not cite a symlink")
    resolved = source.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseGateError(f"{name} escapes repository_root") from exc
    payload = resolved.read_bytes()
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ReleaseGateError(f"{name} SHA-256 differs")
    return {"path": path, "size_bytes": len(payload), "file_sha256": observed}


def assemble_report(
    *,
    repository_root: str | Path,
    policy_path: str | Path,
    mode: str,
    created_at_utc: str,
    spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble a truthful tracked precommit report or exact-candidate attestation."""

    root = _repository(repository_root)
    _policy_value, policy_reference = _policy(policy_path)
    head_result = _git(root, ("rev-parse", "HEAD"))
    assert isinstance(head_result.stdout, str)
    head = _commit(head_result.stdout.strip(), name="HEAD")
    if mode == "tracked_precommit_report":
        if spec_path is not None:
            raise ReleaseGateError("tracked_precommit_report does not accept an attestation spec")
        return _self_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "record_kind": REPORT_KIND,
                "mode": mode,
                "status": "pending",
                "created_at_utc": _timestamp(created_at_utc),
                "repository": {
                    "parent_commit": head,
                    "content_commit": None,
                    "worktree_clean": False,
                },
                "policy": policy_reference,
                "gates": {name: {"status": "not_run", "evidence": None} for name in GATE_NAMES},
                "external_candidate_attestation": {
                    "status": "pending",
                    "expected_root": ".audit/release-attestations/<candidate_commit>/",
                    "report_path": None,
                },
            }
        )
    if mode != "exact_candidate_attestation" or spec_path is None:
        raise ReleaseGateError("exact_candidate_attestation requires --spec")
    spec = _strict_json(spec_path, name="exact candidate attestation spec")
    expected_keys = {
        "schema_version",
        "spec_kind",
        "created_at_utc",
        "candidate_commit",
        "parent_commit",
        "worktree_clean",
        "gates",
    }
    if set(spec) != expected_keys:
        raise ReleaseGateError("exact candidate attestation spec keys differ")
    if spec.get("schema_version") != "1.0.0" or spec.get("spec_kind") != (
        "final_release_gate_attestation_spec"
    ):
        raise ReleaseGateError("exact candidate attestation spec version or kind differs")
    candidate = _commit(spec.get("candidate_commit"), name="candidate_commit")
    parent = _commit(spec.get("parent_commit"), name="parent_commit")
    if spec.get("worktree_clean") is not True:
        raise ReleaseGateError("exact candidate attestation must record a clean worktree")
    if candidate != head:
        raise ReleaseGateError("candidate_commit differs from repository HEAD")
    status_result = _git(root, ("status", "--porcelain=v1", "--untracked-files=all"))
    assert isinstance(status_result.stdout, str)
    if status_result.stdout.strip():
        raise ReleaseGateError("exact candidate worktree is not clean")
    gates_spec = _mapping(spec.get("gates"), name="gates")
    if set(gates_spec) != set(GATE_NAMES):
        raise ReleaseGateError("exact candidate attestation must include every release gate")
    gates: dict[str, Any] = {}
    for name in GATE_NAMES:
        gate = _mapping(gates_spec[name], name=f"gates.{name}")
        if set(gate) != {"status", "path", "expected_sha256"} or gate.get("status") != "pass":
            raise ReleaseGateError(f"gates.{name} must be a pinned pass")
        reference = _evidence_reference(root, gate, name=f"gates.{name}")
        evidence = _strict_json(root / reference["path"], name=f"gates.{name} evidence")
        _verify_self_hash(evidence, name=f"gates.{name} evidence")
        if evidence.get("status") != "pass":
            raise ReleaseGateError(f"gates.{name} evidence is not a pass")
        if (
            name in {"repository_scan", "secret_scan", "license_audit"}
            and evidence.get("policy") != policy_reference
        ):
            raise ReleaseGateError(f"gates.{name} evidence used another release policy")
        declared_gate = evidence.get("gate")
        declared_gates = evidence.get("gates")
        if declared_gate is not None and declared_gate != name:
            raise ReleaseGateError(f"gates.{name} evidence names another gate")
        if declared_gates is not None and name not in _array(
            declared_gates, name=f"gates.{name} evidence.gates"
        ):
            raise ReleaseGateError(f"gates.{name} evidence omits the cited gate")
        evidence_commit = evidence.get("candidate_commit")
        if evidence_commit is not None and evidence_commit != candidate:
            raise ReleaseGateError(f"gates.{name} evidence is bound to another commit")
        gates[name] = {"status": "pass", "evidence": reference}
    report_path = f".audit/release-attestations/{candidate}/final_release_gate_report.json"
    return _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": REPORT_KIND,
            "mode": mode,
            "status": "pass",
            "created_at_utc": _timestamp(spec.get("created_at_utc")),
            "repository": {
                "parent_commit": parent,
                "content_commit": candidate,
                "worktree_clean": True,
            },
            "policy": policy_reference,
            "gates": gates,
            "external_candidate_attestation": {
                "status": "complete",
                "expected_root": ".audit/release-attestations/<candidate_commit>/",
                "report_path": report_path,
            },
        }
    )


def validate_report(
    report_path: str | Path, *, repository_root: str | Path | None = None
) -> dict[str, Any]:
    """Validate self-hash, exact keys, and mode-specific release semantics."""

    try:
        value = _strict_json(report_path, name="final release gate report")
        _verify_self_hash(value, name="final release gate report")
        expected = {
            "schema_version",
            "record_kind",
            "mode",
            "status",
            "created_at_utc",
            "repository",
            "policy",
            "gates",
            "external_candidate_attestation",
            "record_sha256",
        }
        if set(value) != expected:
            raise ReleaseGateError("final release gate report keys differ")
        if value.get("schema_version") != SCHEMA_VERSION or value.get("record_kind") != REPORT_KIND:
            raise ReleaseGateError("final release gate report version or kind differs")
        _timestamp(value.get("created_at_utc"))
        repository = _mapping(value.get("repository"), name="repository")
        if set(repository) != {"parent_commit", "content_commit", "worktree_clean"}:
            raise ReleaseGateError("repository binding keys differ")
        _commit(repository.get("parent_commit"), name="repository.parent_commit")
        content = repository.get("content_commit")
        if content is not None:
            _commit(content, name="repository.content_commit")
        gates = _mapping(value.get("gates"), name="gates")
        if set(gates) != set(GATE_NAMES):
            raise ReleaseGateError("release gate set differs")
        mode = value.get("mode")
        external = _mapping(
            value.get("external_candidate_attestation"), name="external_candidate_attestation"
        )
        if external.get("expected_root") != ".audit/release-attestations/<candidate_commit>/":
            raise ReleaseGateError("external candidate attestation root differs")
        if mode == "tracked_precommit_report":
            if (
                value.get("status") != "pending"
                or content is not None
                or repository.get("worktree_clean") is not False
                or external.get("status") != "pending"
                or external.get("report_path") is not None
                or any(
                    _mapping(gate, name="gate").get("status") != "not_run"
                    for gate in gates.values()
                )
            ):
                raise ReleaseGateError("tracked precommit report overstates release completion")
        elif mode == "exact_candidate_attestation":
            candidate = _commit(content, name="repository.content_commit")
            expected_path = (
                f".audit/release-attestations/{candidate}/final_release_gate_report.json"
            )
            if (
                value.get("status") != "pass"
                or repository.get("worktree_clean") is not True
                or external.get("status") != "complete"
                or external.get("report_path") != expected_path
                or any(
                    _mapping(gate, name="gate").get("status") != "pass" for gate in gates.values()
                )
            ):
                raise ReleaseGateError("exact candidate attestation is incomplete")
        else:
            raise ReleaseGateError("release gate mode is invalid")
        if repository_root is not None:
            root = _repository(repository_root)
            policy_ref = _mapping(value.get("policy"), name="policy")
            observed_ref = _evidence_reference(
                root,
                {
                    "path": policy_ref.get("path"),
                    "expected_sha256": policy_ref.get("file_sha256"),
                },
                name="policy",
            )
            if observed_ref != policy_ref:
                raise ReleaseGateError("policy reference size differs")
            if mode == "exact_candidate_attestation":
                assert content is not None
                for name, raw_gate in gates.items():
                    gate = _mapping(raw_gate, name=f"gates.{name}")
                    reference = _mapping(gate.get("evidence"), name=f"gates.{name}.evidence")
                    observed = _evidence_reference(
                        root,
                        {
                            "path": reference.get("path"),
                            "expected_sha256": reference.get("file_sha256"),
                        },
                        name=f"gates.{name}",
                    )
                    if observed != reference:
                        raise ReleaseGateError(f"gates.{name} evidence size differs")
                    evidence = _strict_json(root / observed["path"], name=f"gates.{name} evidence")
                    _verify_self_hash(evidence, name=f"gates.{name} evidence")
                    if evidence.get("status") != "pass":
                        raise ReleaseGateError(f"gates.{name} evidence is not a pass")
                    if evidence.get("candidate_commit") != content:
                        raise ReleaseGateError(f"gates.{name} evidence candidate differs")
                    if name in {"repository_scan", "secret_scan", "license_audit"} and (
                        evidence.get("policy") != policy_ref
                    ):
                        raise ReleaseGateError(f"gates.{name} evidence policy differs")
        return {
            "valid": True,
            "mode": mode,
            "status": value["status"],
            "record_sha256": value["record_sha256"],
            "errors": [],
        }
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def _write_and_status(record: dict[str, Any], args: argparse.Namespace) -> int:
    root = _repository(args.repository_root)
    _write_new(record, args.output, root=root)
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["status"] in {"pass", "pending"} else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    scan = subparsers.add_parser("scan-repository")
    scan.add_argument("--repository-root", type=Path, required=True)
    scan.add_argument("--candidate-commit", required=True)
    scan.add_argument("--policy", type=Path, required=True)
    scan.add_argument("--created-at-utc", required=True)
    scan.add_argument("--output", type=Path, required=True)
    secret = subparsers.add_parser("validate-secret-scan")
    secret.add_argument("--repository-root", type=Path, required=True)
    secret.add_argument("--report", type=Path, required=True)
    secret.add_argument("--candidate-commit", required=True)
    secret.add_argument("--policy", type=Path, required=True)
    secret.add_argument("--gitleaks-version", required=True)
    secret.add_argument("--created-at-utc", required=True)
    secret.add_argument("--output", type=Path, required=True)
    licenses = subparsers.add_parser("audit-licenses")
    licenses.add_argument("--repository-root", type=Path, required=True)
    licenses.add_argument("--inventory", type=Path, required=True)
    licenses.add_argument("--candidate-commit", required=True)
    licenses.add_argument("--policy", type=Path, required=True)
    licenses.add_argument("--pip-licenses-version", required=True)
    licenses.add_argument("--created-at-utc", required=True)
    licenses.add_argument("--output", type=Path, required=True)
    assemble = subparsers.add_parser("assemble")
    assemble.add_argument("--repository-root", type=Path, required=True)
    assemble.add_argument("--policy", type=Path, required=True)
    assemble.add_argument(
        "--mode",
        choices=("tracked_precommit_report", "exact_candidate_attestation"),
        required=True,
    )
    assemble.add_argument("--created-at-utc", required=True)
    assemble.add_argument("--spec", type=Path)
    assemble.add_argument("--output", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--report", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scan-repository":
            record = scan_repository(
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "validate-secret-scan":
            record = validate_secret_scan(
                report_path=args.report,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                gitleaks_version=args.gitleaks_version,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "audit-licenses":
            record = audit_licenses(
                inventory_path=args.inventory,
                candidate_commit=args.candidate_commit,
                policy_path=args.policy,
                pip_licenses_version=args.pip_licenses_version,
                created_at_utc=args.created_at_utc,
            )
            return _write_and_status(record, args)
        if args.command == "assemble":
            record = assemble_report(
                repository_root=args.repository_root,
                policy_path=args.policy,
                mode=args.mode,
                created_at_utc=args.created_at_utc,
                spec_path=args.spec,
            )
            return _write_and_status(record, args)
        report = validate_report(args.report, repository_root=args.repository_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report["valid"] is True else 2
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
