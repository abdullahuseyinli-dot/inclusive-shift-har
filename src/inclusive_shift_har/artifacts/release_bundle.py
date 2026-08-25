"""Create and validate an exact, fail-closed release evidence bundle.

The outer release ZIP is built only from a reviewed member spec.  The archive
contains that spec, a self-hashed manifest, and exactly the manifest members;
validation rejects additional, missing, duplicated, unsafe, or altered files.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from inclusive_shift_har.artifacts.github_evidence import EXPECTED_ARTIFACT_FILES
from inclusive_shift_har.artifacts.release_gate import (
    ReleaseGateError,
    validate_license_audit_semantics,
)
from inclusive_shift_har.artifacts.release_inventory import (
    validate_release_evidence_inventory_file,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.0.0"
SPEC_KIND = "release_evidence_bundle_spec"
MANIFEST_KIND = "release_evidence_bundle_manifest"
VALIDATION_KIND = "release_evidence_bundle_validation"
SPEC_ARCHIVE_PATH = "_bundle/release_bundle_spec.json"
MANIFEST_ARCHIVE_PATH = "_bundle/release_bundle_manifest.json"
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
WINDOWS_USER_PATH_RE = re.compile(rb"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\r\n]+")
MAC_USER_PATH_RE = re.compile(rb"/Users/[^/\r\n]+/")
ZIP_SIGNATURES = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"PK\x01\x02",
    b"PK\x06\x06",
    b"PK\x06\x07",
)
MAX_MEMBERS = 256
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_COMPRESSION_RATIO = 1_000
FORBIDDEN_SUFFIXES = frozenset({".ckpt", ".npy", ".npz", ".onnx", ".pt", ".pth"})
FORBIDDEN_BASENAMES = frozenset(
    {
        ".env",
        "credentials.json",
        "gitleaks.json",
        "python_licenses.json",
        "secrets.json",
        "secrets.yaml",
        "secrets.yml",
    }
)
FORBIDDEN_COMPONENTS = frozenset({"checkpoints", "data", "raw", "secrets"})
ALLOWED_ROLES = frozenset(
    {
        "release_inventory",
        "release_inventory_spec",
        "github_remote_evidence",
        "github_completed_ci_evidence",
        "github_api_response",
        "github_actions_archive",
        "ci_mirror",
        "candidate_attestation",
        "candidate_attestation_spec",
        "staged_index_scan",
        "staged_secret_evidence",
        "local_cuda_evidence",
        "local_license_evidence",
        "local_quality_log",
    }
)
REQUIRED_SINGLETON_ROLES = frozenset(
    {
        "release_inventory",
        "release_inventory_spec",
        "github_remote_evidence",
        "github_completed_ci_evidence",
        "github_actions_archive",
        "candidate_attestation",
        "candidate_attestation_spec",
        "staged_index_scan",
        "staged_secret_evidence",
        "local_cuda_evidence",
        "local_license_evidence",
    }
)
WINDOWS_RESERVED_NAMES = frozenset(
    {
        "aux",
        "con",
        "nul",
        "prn",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)

LOCAL_GATE_COMMANDS: dict[str, list[str]] = {
    "lock": ["uv", "lock", "--check"],
    "tests": ["python", "-m", "pytest", "-q"],
    "lint": ["python", "-m", "ruff", "check", "src", "tests"],
    "format": ["python", "-m", "ruff", "format", "--check", "src", "tests"],
    "types": ["python", "-m", "mypy", "src", "tests"],
    "manifests": [
        "python",
        "-m",
        "inclusive_shift_har.cli",
        "validate-manifests",
        "--json",
    ],
    "splits": [
        "python",
        "-m",
        "inclusive_shift_har.cli",
        "audit-splits",
        "--split-manifest",
        "results/protocol/splits/inclusivehar_v4_released_block_v1_2.json",
        "--json",
    ],
    "configuration": [
        "python",
        "-m",
        "pytest",
        "-q",
        "tests/test_few_person_protocol.py",
        "tests/test_protocol_splits.py",
        "tests/test_schema_documents.py",
        "tests/test_uci_source_protocol.py",
    ],
    "artifacts": [
        "python",
        "-m",
        "inclusive_shift_har.cli",
        "validate-artifacts",
        "--artifact-root",
        "results",
        "--require-artifacts",
        "--json",
    ],
    "git_integrity": ["git", "fsck", "--full"],
}
STAGED_INDEX_RECORD_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "created_at_utc",
    "head_commit",
    "scan_scope",
    "policy",
    "index_entry_count",
    "staged_change_count",
    "unique_blob_count",
    "replace_ref_count",
    "grafts_file_present",
    "index_entry_index_sha256",
    "violations",
    "raw_worktree_paths_opened",
    "git_objects_written",
    "record_sha256",
}
LOCAL_RECORD_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "created_at_utc",
    "candidate_commit",
    "parent_commit",
    "scope",
    "confirmatory_evaluator_invoked",
    "cuda_required",
    "cuda",
    "cuda_log",
    "machine",
    "tool_versions",
    "gates",
    "record_sha256",
}
STAGED_SECRET_RECORD_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "created_at_utc",
    "candidate_commit",
    "parent_commit",
    "scan_scope",
    "gitleaks_version",
    "gitleaks_exit_code",
    "finding_count",
    "gitleaks_executable",
    "raw_report",
    "staged_index_scan",
    "policy",
    "config",
    "ignore",
    "record_sha256",
}
LICENSE_RECORD_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "created_at_utc",
    "candidate_commit",
    "pip_licenses_version",
    "policy",
    "inventory",
    "audit_environment",
    "packages",
    "package_count",
    "normalized_inventory_sha256",
    "exceptions_applied",
    "violations",
    "record_sha256",
}


class ReleaseBundleError(RuntimeError):
    """Raised when release bundle evidence is incomplete or unsafe."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseBundleError(f"{name} must be a string-keyed object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ReleaseBundleError(f"{name} must be an array")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseBundleError(f"{name} must be a non-empty string")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    if set(value) != expected:
        raise ReleaseBundleError(f"{name} keys differ from the exact contract")


def _commit(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if COMMIT_RE.fullmatch(text) is None:
        raise ReleaseBundleError(f"{name} must be a lowercase 40-character Git commit")
    return text


def _sha256(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if SHA256_RE.fullmatch(text) is None:
        raise ReleaseBundleError(f"{name} must be a lowercase SHA-256")
    return text


def _timestamp(value: Any) -> str:
    text = _text(value, name="created_at_utc")
    if TIMESTAMP_RE.fullmatch(text) is None:
        raise ReleaseBundleError("created_at_utc must be canonical UTC ISO-8601")
    try:
        parsed = datetime.fromisoformat(text.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise ReleaseBundleError("created_at_utc is not a valid timestamp") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseBundleError("created_at_utc must be UTC")
    return text


def _safe_relative(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if (
        "\\" in text
        or ":" in text
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ReleaseBundleError(f"{name} must be a portable POSIX relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ReleaseBundleError(f"{name} must be a canonical relative path")
    for part in path.parts:
        if part.endswith((" ", ".")) or part.split(".", 1)[0].casefold() in WINDOWS_RESERVED_NAMES:
            raise ReleaseBundleError(f"{name} is not portable across release platforms")
    return text


def _load_record(path: Path, *, name: str) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ReleaseBundleError(f"{name} must be a regular non-symlink file")
    try:
        return _mapping(load_json_strict(path), name=name)
    except Exception as exc:
        raise ReleaseBundleError(f"cannot load {name}: {exc}") from exc


def _resolve_input(path: str | Path, *, name: str, directory: bool) -> Path:
    raw = Path(path)
    if any(part in {".", ".."} for part in raw.parts):
        raise ReleaseBundleError(f"{name} path must not contain dot traversal")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ReleaseBundleError(f"{name} may not traverse a symlink")
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise ReleaseBundleError(f"{name} does not exist") from exc
    if directory and not resolved.is_dir():
        raise ReleaseBundleError(f"{name} must be an existing directory")
    if not directory and not resolved.is_file():
        raise ReleaseBundleError(f"{name} must be a regular file")
    return resolved


def _require_passing_candidate_record(
    path: Path, *, name: str, record_kind: str, candidate: str
) -> Mapping[str, Any]:
    record = _load_record(path, name=name)
    body = dict(record)
    observed = _sha256(body.pop("record_sha256", None), name=f"{name} record_sha256")
    if (
        canonical_json_sha256(body) != observed
        or record.get("schema_version") != SCHEMA_VERSION
        or record.get("record_kind") != record_kind
        or record.get("status") != "pass"
        or record.get("candidate_commit") != candidate
    ):
        raise ReleaseBundleError(f"{name} is not a passing candidate-bound record")
    return record


def _strict_json_payload(payload: bytes, *, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ReleaseBundleError(f"duplicate JSON key in {name}: {key!r}")
            result[key] = value
        return result

    def reject_non_finite(token: str) -> None:
        raise ReleaseBundleError(f"non-finite JSON number in {name}: {token}")

    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseBundleError(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    return _mapping(value, name=name)


def _exact_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        raise ReleaseBundleError(f"{name} must be an integer >= {minimum}")
    return value


def _self_hashed_record(payload: bytes, *, name: str, expected_keys: set[str]) -> Mapping[str, Any]:
    record = _strict_json_payload(payload, name=name)
    _exact_keys(record, expected_keys, name=name)
    body = dict(record)
    observed = _sha256(body.pop("record_sha256", None), name=f"{name}.record_sha256")
    if canonical_json_sha256(body) != observed:
        raise ReleaseBundleError(f"{name} self-hash does not reconstruct")
    return record


def _reference(
    value: Any,
    *,
    name: str,
    expected_keys: set[str],
    expected_basename: str | None = None,
    expected_path: str | None = None,
) -> Mapping[str, Any]:
    reference = _mapping(value, name=name)
    _exact_keys(reference, expected_keys, name=name)
    if "basename" in expected_keys:
        basename = _text(reference.get("basename"), name=f"{name}.basename")
        if PurePosixPath(basename).name != basename or "\\" in basename or ":" in basename:
            raise ReleaseBundleError(f"{name}.basename must be portable")
        if expected_basename is not None and basename != expected_basename:
            raise ReleaseBundleError(f"{name}.basename differs from the exact contract")
    if "path" in expected_keys:
        path = _safe_relative(reference.get("path"), name=f"{name}.path")
        if expected_path is not None and path != expected_path:
            raise ReleaseBundleError(f"{name}.path differs from the exact contract")
    _exact_integer(reference.get("size_bytes"), name=f"{name}.size_bytes")
    _sha256(reference.get("file_sha256"), name=f"{name}.file_sha256")
    return reference


def _git_text(root: Path, *arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseBundleError(f"Git command failed: {' '.join(arguments)}") from exc
    return result.stdout


def _git_bytes(root: Path, *arguments: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseBundleError(f"Git command failed: {' '.join(arguments)}") from exc
    return result.stdout


def _candidate_repository_context(
    repository_root: str | Path, *, candidate: str
) -> tuple[Path, str, Mapping[str, Any], dict[str, Any], dict[str, Any]]:
    root = _resolve_input(repository_root, name="repository_root", directory=True)
    top_level = Path(_git_text(root, "rev-parse", "--show-toplevel").strip()).resolve(strict=True)
    if not root.samefile(top_level):
        raise ReleaseBundleError("repository_root must be the Git worktree top-level")
    if _git_text(root, "rev-parse", "HEAD").strip() != candidate:
        raise ReleaseBundleError("repository HEAD differs from candidate_commit")
    if _git_text(root, "branch", "--show-current").strip() != "main":
        raise ReleaseBundleError("candidate must be checked out on local main")
    if _git_text(root, "status", "--porcelain=v1", "--untracked-files=all").strip():
        raise ReleaseBundleError("candidate repository must be clean")
    parent = _commit(_git_text(root, "rev-parse", f"{candidate}^").strip(), name="candidate parent")

    def candidate_blob(path: str) -> bytes:
        return _git_bytes(root, "show", f"{candidate}:{path}")

    policy_path = "configs/release/release_gate_policy_v1.json"
    policy_payload = candidate_blob(policy_path)
    policy = _strict_json_payload(policy_payload, name="candidate release policy")
    if (
        policy.get("schema_version") != SCHEMA_VERSION
        or policy.get("policy_kind") != "final_release_gate_policy"
    ):
        raise ReleaseBundleError("candidate release policy schema or kind differs")
    policy_reference = {"path": policy_path, **_payload_reference(policy_payload)}
    gitleaks = _mapping(policy.get("gitleaks"), name="candidate gitleaks policy")
    config_path = _safe_relative(gitleaks.get("config_path"), name="gitleaks.config_path")
    ignore_path = _safe_relative(gitleaks.get("ignore_path"), name="gitleaks.ignore_path")
    config_payload = candidate_blob(config_path)
    ignore_payload = candidate_blob(ignore_path)
    config_reference = {"path": config_path, **_payload_reference(config_payload)}
    ignore_reference = {"path": ignore_path, **_payload_reference(ignore_payload)}
    if config_reference["file_sha256"] != _sha256(
        gitleaks.get("config_sha256"), name="gitleaks.config_sha256"
    ):
        raise ReleaseBundleError("candidate Gitleaks configuration differs from policy")
    return (
        root,
        parent,
        policy,
        policy_reference,
        {
            "config": config_reference,
            "ignore": ignore_reference,
        },
    )


def _candidate_tree_index(root: Path, candidate: str) -> tuple[int, str]:
    payload = _git_bytes(root, "ls-tree", "-r", "-z", "-l", candidate)
    rows: list[str] = []
    for raw in (item for item in payload.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, raw_size = metadata.decode("ascii").split(" ", 3)
            path = _safe_relative(raw_path.decode("utf-8"), name="candidate tree path")
            size: int | None = None if raw_size == "-" else int(raw_size)
        except (UnicodeError, ValueError) as exc:
            raise ReleaseBundleError("candidate tree contains invalid metadata") from exc
        rows.append(f"{mode} {object_id} 0 {kind} {size} {path}")
    return len(rows), hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def _source_root(path: str | Path) -> Path:
    return _resolve_input(path, name="source_root", directory=True)


def _create_new_directory(path: str | Path, *, name: str) -> Path:
    raw = Path(path)
    if any(part in {"", ".", ".."} for part in raw.parts):
        raise ReleaseBundleError(f"{name} must be a canonical path")
    absolute = raw if raw.is_absolute() else Path.cwd() / raw
    parent = _resolve_input(absolute.parent, name=f"{name} parent", directory=True)
    destination = parent / absolute.name
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite {name}: {destination}")
    destination.mkdir()
    return destination.resolve(strict=True)


def _source_file(root: Path, relative: str, *, name: str) -> Path:
    path = PurePosixPath(relative)
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseBundleError(f"{name} may not traverse a symlink")
    resolved = current.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleError(f"{name} escapes source_root") from exc
    if not resolved.is_file() or resolved.is_symlink():
        raise ReleaseBundleError(f"{name} must be a regular non-symlink file")
    return resolved


def _payload_reference(payload: bytes) -> dict[str, int | str]:
    return {"size_bytes": len(payload), "file_sha256": hashlib.sha256(payload).hexdigest()}


def _validate_standard_candidate_evidence(
    *, payloads: Mapping[str, bytes], candidate: str, repository_root: str | Path
) -> list[str]:
    """Validate every local-only record and bind it to the candidate repository."""

    root, parent, policy, policy_reference, gitleaks_references = _candidate_repository_context(
        repository_root, candidate=candidate
    )

    def required_payload(path: str) -> bytes:
        try:
            return payloads[path]
        except KeyError as exc:
            raise ReleaseBundleError(f"standard bundle evidence is missing: {path}") from exc

    local_path = "local/local_cuda_gate.json"
    local_record = _self_hashed_record(
        required_payload(local_path),
        name="local CUDA record",
        expected_keys=LOCAL_RECORD_KEYS,
    )
    _timestamp(local_record.get("created_at_utc"))
    if (
        local_record.get("schema_version") != SCHEMA_VERSION
        or local_record.get("record_kind") != "local_cuda_release_gate_evidence"
        or local_record.get("status") != "pass"
        or local_record.get("candidate_commit") != candidate
        or local_record.get("parent_commit") != parent
        or local_record.get("scope")
        != "candidate_bound_local_release_gates_without_target_evaluation"
        or local_record.get("confirmatory_evaluator_invoked") is not False
        or local_record.get("cuda_required") is not True
    ):
        raise ReleaseBundleError("local CUDA record identity or scope differs")
    cuda = _mapping(local_record.get("cuda"), name="local CUDA result")
    _exact_keys(
        cuda,
        {
            "status",
            "started_at_utc",
            "completed_at_utc",
            "torch_version",
            "cuda_build",
            "device",
            "device_count",
            "nvidia_smi",
            "peak_allocated_bytes",
            "allocation_checksum",
        },
        name="local CUDA result",
    )
    _timestamp(cuda.get("started_at_utc"))
    _timestamp(cuda.get("completed_at_utc"))
    checksum = cuda.get("allocation_checksum")
    if (
        cuda.get("status") != "pass"
        or any(
            not isinstance(cuda.get(field), str) or not str(cuda.get(field)).strip()
            for field in (
                "torch_version",
                "cuda_build",
                "device",
                "nvidia_smi",
            )
        )
        or _exact_integer(cuda.get("device_count"), name="cuda.device_count", minimum=1) < 1
        or _exact_integer(
            cuda.get("peak_allocated_bytes"), name="cuda.peak_allocated_bytes", minimum=1
        )
        < 1
        or not isinstance(checksum, (int, float))
        or isinstance(checksum, bool)
        or not math.isfinite(float(checksum))
    ):
        raise ReleaseBundleError("local CUDA allocation evidence is incomplete")

    machine = _mapping(local_record.get("machine"), name="local machine")
    _exact_keys(machine, {"platform", "architecture", "python"}, name="local machine")
    if any(not isinstance(value, str) or not value.strip() for value in machine.values()):
        raise ReleaseBundleError("local machine metadata is incomplete")
    tools = _mapping(local_record.get("tool_versions"), name="local tool versions")
    _exact_keys(tools, {"uv", "pytest", "ruff", "mypy"}, name="local tool versions")
    if any(not isinstance(value, str) or not value.strip() for value in tools.values()):
        raise ReleaseBundleError("local tool-version metadata is incomplete")

    cuda_log_path = "local/cuda_allocation.log"
    cuda_log = _reference(
        local_record.get("cuda_log"),
        name="local CUDA log",
        expected_keys={"basename", "size_bytes", "file_sha256", "path"},
        expected_basename="cuda_allocation.log",
        expected_path=cuda_log_path,
    )
    if _payload_reference(required_payload(cuda_log_path)) != {
        "size_bytes": cuda_log["size_bytes"],
        "file_sha256": cuda_log["file_sha256"],
    }:
        raise ReleaseBundleError("local CUDA log bytes differ from its record")

    gates = _mapping(local_record.get("gates"), name="local CUDA gates")
    _exact_keys(gates, set(LOCAL_GATE_COMMANDS), name="local CUDA gates")
    local_logs = [cuda_log_path]
    for gate_name, expected_command in LOCAL_GATE_COMMANDS.items():
        gate = _mapping(gates.get(gate_name), name=f"local gate {gate_name}")
        _exact_keys(
            gate,
            {
                "status",
                "command",
                "started_at_utc",
                "completed_at_utc",
                "exit_code",
                "raw_log",
                "sanitized_log",
            },
            name=f"local gate {gate_name}",
        )
        _timestamp(gate.get("started_at_utc"))
        _timestamp(gate.get("completed_at_utc"))
        if gate.get("status") != "pass" or gate.get("command") != expected_command:
            raise ReleaseBundleError(f"local gate {gate_name} did not run the exact command")
        if _exact_integer(gate.get("exit_code"), name=f"{gate_name}.exit_code") != 0:
            raise ReleaseBundleError(f"local gate {gate_name} did not exit successfully")
        raw_log = _reference(
            gate.get("raw_log"),
            name=f"local gate {gate_name} raw log",
            expected_keys={
                "basename",
                "size_bytes",
                "file_sha256",
                "retained_locally_only",
            },
            expected_basename=f"{gate_name}.raw.log",
        )
        if raw_log.get("retained_locally_only") is not True:
            raise ReleaseBundleError(f"local gate {gate_name} raw log must remain local")
        log_path = f"local/{gate_name}.log"
        sanitized = _reference(
            gate.get("sanitized_log"),
            name=f"local gate {gate_name} sanitized log",
            expected_keys={"basename", "size_bytes", "file_sha256", "path", "replacements"},
            expected_basename=f"{gate_name}.log",
            expected_path=log_path,
        )
        replacements = _sequence(
            sanitized.get("replacements"), name=f"local gate {gate_name} replacements"
        )
        if list(replacements) != sorted(set(replacements)) or any(
            value not in {"repository_root", "user_profile"} for value in replacements
        ):
            raise ReleaseBundleError(f"local gate {gate_name} replacements differ")
        if _payload_reference(required_payload(log_path)) != {
            "size_bytes": sanitized["size_bytes"],
            "file_sha256": sanitized["file_sha256"],
        }:
            raise ReleaseBundleError(f"local gate {gate_name} sanitized log bytes differ")
        local_logs.append(log_path)

    staged_path = f".audit/release-attestations/{candidate}/staged_index_scan.json"
    staged_payload = required_payload(staged_path)
    staged = _self_hashed_record(
        staged_payload,
        name="staged index record",
        expected_keys=STAGED_INDEX_RECORD_KEYS,
    )
    _timestamp(staged.get("created_at_utc"))
    entry_count = _exact_integer(
        staged.get("index_entry_count"), name="staged index entry_count", minimum=1
    )
    staged_change_count = _exact_integer(
        staged.get("staged_change_count"), name="staged change_count", minimum=1
    )
    unique_blob_count = _exact_integer(
        staged.get("unique_blob_count"), name="staged unique_blob_count", minimum=1
    )
    _exact_integer(staged.get("replace_ref_count"), name="staged replace_ref_count")
    tree_count, tree_digest = _candidate_tree_index(root, candidate)
    staged_digest = _sha256(staged.get("index_entry_index_sha256"), name="staged index digest")
    if (
        staged.get("schema_version") != SCHEMA_VERSION
        or staged.get("record_kind") != "git_index_release_scan"
        or staged.get("status") != "pass"
        or staged.get("head_commit") != parent
        or staged.get("scan_scope") != "exact_git_index"
        or staged.get("policy") != policy_reference
        or staged.get("replace_ref_count") != 0
        or staged.get("grafts_file_present") is not False
        or staged.get("violations") != []
        or staged.get("raw_worktree_paths_opened") is not False
        or staged.get("git_objects_written") is not False
        or staged_change_count < 1
        or unique_blob_count > entry_count
        or entry_count != tree_count
        or staged_digest != tree_digest
    ):
        raise ReleaseBundleError("staged index record does not reconstruct the candidate tree")

    staged_secret = _self_hashed_record(
        required_payload("local/staged_secret_scan.json"),
        name="staged Gitleaks record",
        expected_keys=STAGED_SECRET_RECORD_KEYS,
    )
    _timestamp(staged_secret.get("created_at_utc"))
    gitleaks_policy = _mapping(policy.get("gitleaks"), name="candidate gitleaks policy")
    executable = _reference(
        staged_secret.get("gitleaks_executable"),
        name="staged Gitleaks executable",
        expected_keys={"basename", "size_bytes", "file_sha256"},
        expected_basename=_text(
            gitleaks_policy.get("windows_x64_executable"), name="Gitleaks executable pin"
        ),
    )
    raw_report = _reference(
        staged_secret.get("raw_report"),
        name="staged Gitleaks raw report",
        expected_keys={
            "basename",
            "size_bytes",
            "file_sha256",
            "included_in_release_bundle",
        },
        expected_basename="staged-gitleaks.json",
    )
    staged_reference = _reference(
        staged_secret.get("staged_index_scan"),
        name="staged Gitleaks index reference",
        expected_keys={
            "basename",
            "size_bytes",
            "file_sha256",
            "record_sha256",
            "index_entry_index_sha256",
        },
        expected_basename="staged_index_scan.json",
    )
    _sha256(staged_reference.get("record_sha256"), name="staged index record reference")
    _sha256(staged_reference.get("index_entry_index_sha256"), name="staged index digest reference")
    if (
        staged_secret.get("schema_version") != SCHEMA_VERSION
        or staged_secret.get("record_kind") != "staged_gitleaks_evidence"
        or staged_secret.get("status") != "pass"
        or staged_secret.get("candidate_commit") != candidate
        or staged_secret.get("parent_commit") != parent
        or staged_secret.get("scan_scope") != "exact_precommit_git_index"
        or staged_secret.get("gitleaks_version") != gitleaks_policy.get("version")
        or staged_secret.get("gitleaks_exit_code") != 0
        or isinstance(staged_secret.get("gitleaks_exit_code"), bool)
        or staged_secret.get("finding_count") != 0
        or isinstance(staged_secret.get("finding_count"), bool)
        or executable.get("size_bytes") != gitleaks_policy.get("windows_x64_executable_size_bytes")
        or executable.get("file_sha256") != gitleaks_policy.get("windows_x64_executable_sha256")
        or raw_report.get("included_in_release_bundle") is not False
        or staged_secret.get("policy") != policy_reference
        or staged_secret.get("config") != gitleaks_references["config"]
        or staged_secret.get("ignore") != gitleaks_references["ignore"]
        or _payload_reference(staged_payload)
        != {
            "size_bytes": staged_reference["size_bytes"],
            "file_sha256": staged_reference["file_sha256"],
        }
        or staged_reference.get("record_sha256") != staged.get("record_sha256")
        or staged_reference.get("index_entry_index_sha256") != staged_digest
    ):
        raise ReleaseBundleError("staged Gitleaks record is not bound to the candidate index")

    license_record = _self_hashed_record(
        required_payload("local/license_audit.json"),
        name="local licence record",
        expected_keys=LICENSE_RECORD_KEYS,
    )
    _timestamp(license_record.get("created_at_utc"))
    inventory = _reference(
        license_record.get("inventory"),
        name="local licence inventory",
        expected_keys={"basename", "size_bytes", "file_sha256"},
        expected_basename="python_licenses.json",
    )
    package_count = _exact_integer(
        license_record.get("package_count"), name="local licence package_count", minimum=1
    )
    _sha256(
        license_record.get("normalized_inventory_sha256"),
        name="local licence normalized inventory digest",
    )
    try:
        validate_license_audit_semantics(
            license_record,
            root=root,
            candidate=candidate,
            policy=policy,
        )
    except ReleaseGateError as exc:
        raise ReleaseBundleError("local licence exceptions differ from policy") from exc
    if (
        license_record.get("schema_version") != SCHEMA_VERSION
        or license_record.get("record_kind") != "python_license_audit"
        or license_record.get("status") != "pass"
        or license_record.get("candidate_commit") != candidate
        or license_record.get("pip_licenses_version") != "5.5.5"
        or license_record.get("policy") != policy_reference
        or package_count < 1
        or inventory.get("size_bytes") == 0
        or license_record.get("violations") != []
    ):
        raise ReleaseBundleError("local licence record is incomplete or candidate-unbound")
    return local_logs


def _validate_member_payload(*, role: str, path: str, payload: bytes) -> None:
    pure = PurePosixPath(path)
    basename = pure.name.casefold()
    parts = {part.casefold() for part in pure.parts}
    if len(payload) > MAX_MEMBER_BYTES:
        raise ReleaseBundleError(f"bundle member exceeds size cap: {path}")
    if pure.suffix.casefold() in FORBIDDEN_SUFFIXES:
        raise ReleaseBundleError(f"bundle member has a forbidden model/array suffix: {path}")
    if basename in FORBIDDEN_BASENAMES or basename.startswith(".env."):
        raise ReleaseBundleError(f"bundle member has a forbidden sensitive/raw basename: {path}")
    if parts & FORBIDDEN_COMPONENTS:
        raise ReleaseBundleError(f"bundle member enters a forbidden raw/checkpoint path: {path}")
    if pure.suffix.casefold() == ".zip" and role != "github_actions_archive":
        raise ReleaseBundleError(f"only the reviewed GitHub Actions archive may be nested: {path}")
    if payload.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise ReleaseBundleError(f"Git LFS pointers are forbidden in the bundle: {path}")
    if role == "github_actions_archive":
        if not payload.startswith(b"PK\x03\x04"):
            raise ReleaseBundleError("reviewed GitHub Actions archive is not a non-empty ZIP")
        return
    if payload.startswith(ZIP_SIGNATURES):
        raise ReleaseBundleError(f"unreviewed nested ZIP payload is forbidden: {path}")
    if b"\x00" in payload:
        raise ReleaseBundleError(f"non-archive bundle members must be UTF-8 text: {path}")
    try:
        payload.decode("utf-8")
    except UnicodeError as exc:
        raise ReleaseBundleError(f"text bundle member is not UTF-8: {path}") from exc
    if WINDOWS_USER_PATH_RE.search(payload) or MAC_USER_PATH_RE.search(payload):
        raise ReleaseBundleError(f"bundle member exposes an absolute user path: {path}")


def _parse_spec(value: Mapping[str, Any]) -> tuple[str, str, str, list[tuple[str, str]]]:
    _exact_keys(
        value,
        {
            "schema_version",
            "spec_kind",
            "created_at_utc",
            "candidate_commit",
            "inventory_path",
            "release_spec_path",
            "members",
        },
        name="bundle spec",
    )
    if value.get("schema_version") != SCHEMA_VERSION or value.get("spec_kind") != SPEC_KIND:
        raise ReleaseBundleError("bundle spec schema or kind differs")
    _timestamp(value.get("created_at_utc"))
    candidate = _commit(value.get("candidate_commit"), name="candidate_commit")
    inventory_path = _safe_relative(value.get("inventory_path"), name="inventory_path")
    release_spec_path = _safe_relative(value.get("release_spec_path"), name="release_spec_path")
    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    counts: Counter[str] = Counter()
    for index, raw in enumerate(_sequence(value.get("members"), name="members")):
        member = _mapping(raw, name=f"members[{index}]")
        _exact_keys(member, {"role", "path"}, name=f"members[{index}]")
        role = _text(member.get("role"), name=f"members[{index}].role")
        if role not in ALLOWED_ROLES:
            raise ReleaseBundleError(f"members[{index}].role is not allowlisted")
        path = _safe_relative(member.get("path"), name=f"members[{index}].path")
        folded = path.casefold()
        if folded in seen or folded in {
            SPEC_ARCHIVE_PATH.casefold(),
            MANIFEST_ARCHIVE_PATH.casefold(),
        }:
            raise ReleaseBundleError("bundle member paths collide or use reserved paths")
        seen.add(folded)
        counts[role] += 1
        rows.append((role, path))
    if len(rows) > MAX_MEMBERS:
        raise ReleaseBundleError("bundle member count exceeds the cap")
    if any(counts[role] != 1 for role in REQUIRED_SINGLETON_ROLES):
        raise ReleaseBundleError("bundle singleton evidence roles must each occur exactly once")
    if counts["github_api_response"] != 4:
        raise ReleaseBundleError("bundle must contain exactly four saved GitHub API responses")
    if counts["ci_mirror"] != 45:
        raise ReleaseBundleError("bundle must contain exactly the 45 validated CI mirror members")
    if counts["local_quality_log"] < 1:
        raise ReleaseBundleError("bundle must retain at least one sanitized local quality log")
    role_by_path = {path: role for role, path in rows}
    if role_by_path.get(inventory_path) != "release_inventory":
        raise ReleaseBundleError("inventory_path must identify the release_inventory member")
    if role_by_path.get(release_spec_path) != "release_inventory_spec":
        raise ReleaseBundleError(
            "release_spec_path must identify the release_inventory_spec member"
        )
    return candidate, inventory_path, release_spec_path, rows


def _validate_manifest(value: Mapping[str, Any]) -> tuple[str, str, str, list[Mapping[str, Any]]]:
    _exact_keys(
        value,
        {
            "schema_version",
            "record_kind",
            "created_at_utc",
            "candidate_commit",
            "inventory_path",
            "release_spec_path",
            "source_spec",
            "member_count",
            "total_uncompressed_bytes",
            "members",
            "member_index_sha256",
            "record_sha256",
        },
        name="bundle manifest",
    )
    record = dict(value)
    observed_record_hash = _sha256(record.pop("record_sha256", None), name="record_sha256")
    if canonical_json_sha256(record) != observed_record_hash:
        raise ReleaseBundleError("bundle manifest self-hash does not reconstruct")
    if value.get("schema_version") != SCHEMA_VERSION or value.get("record_kind") != MANIFEST_KIND:
        raise ReleaseBundleError("bundle manifest schema or kind differs")
    _timestamp(value.get("created_at_utc"))
    candidate = _commit(value.get("candidate_commit"), name="candidate_commit")
    inventory_path = _safe_relative(value.get("inventory_path"), name="inventory_path")
    release_spec_path = _safe_relative(value.get("release_spec_path"), name="release_spec_path")
    source_spec = _mapping(value.get("source_spec"), name="source_spec")
    _exact_keys(source_spec, {"size_bytes", "file_sha256"}, name="source_spec")
    if not isinstance(source_spec.get("size_bytes"), int) or isinstance(
        source_spec.get("size_bytes"), bool
    ):
        raise ReleaseBundleError("source_spec.size_bytes must be an integer")
    _sha256(source_spec.get("file_sha256"), name="source_spec.file_sha256")
    members = [
        _mapping(item, name=f"members[{index}]")
        for index, item in enumerate(_sequence(value.get("members"), name="members"))
    ]
    seen: set[str] = set()
    index_rows: list[str] = []
    total = 0
    for index, member in enumerate(members):
        _exact_keys(
            member,
            {"role", "path", "size_bytes", "file_sha256"},
            name=f"members[{index}]",
        )
        role = _text(member.get("role"), name=f"members[{index}].role")
        if role not in ALLOWED_ROLES:
            raise ReleaseBundleError("bundle manifest contains an unallowlisted role")
        path = _safe_relative(member.get("path"), name=f"members[{index}].path")
        folded = path.casefold()
        if folded in seen:
            raise ReleaseBundleError("bundle manifest contains case-colliding paths")
        seen.add(folded)
        size = member.get("size_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or not 0 <= size <= MAX_MEMBER_BYTES:
            raise ReleaseBundleError("bundle manifest member size is invalid")
        digest = _sha256(member.get("file_sha256"), name=f"members[{index}].file_sha256")
        total += size
        index_rows.append(f"{role}\t{path}\t{size}\t{digest}")
    if value.get("member_count") != len(members) or len(members) > MAX_MEMBERS:
        raise ReleaseBundleError("bundle manifest member count differs")
    if value.get("total_uncompressed_bytes") != total or total > MAX_TOTAL_BYTES:
        raise ReleaseBundleError("bundle manifest total size differs or exceeds the cap")
    expected_index = hashlib.sha256(("\n".join(sorted(index_rows)) + "\n").encode()).hexdigest()
    if value.get("member_index_sha256") != expected_index:
        raise ReleaseBundleError("bundle manifest member index does not reconstruct")
    return candidate, inventory_path, release_spec_path, members


def assemble_release_bundle_manifest(
    *,
    bundle_spec_path: str | Path,
    source_root: str | Path,
    created_at_utc: str,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Create a self-hashed manifest from an exact reviewed member spec."""

    root = _source_root(source_root)
    spec_path = _resolve_input(bundle_spec_path, name="bundle spec", directory=False)
    spec_payload = spec_path.read_bytes()
    spec = _mapping(load_json_strict(spec_path), name="bundle spec")
    candidate, inventory_path, release_spec_path, rows = _parse_spec(spec)
    manifest_members: list[dict[str, int | str]] = []
    index_rows: list[str] = []
    total = 0
    for role, relative in sorted(rows, key=lambda item: item[1].casefold()):
        source = _source_file(root, relative, name=f"bundle source {relative}")
        payload = source.read_bytes()
        _validate_member_payload(role=role, path=relative, payload=payload)
        reference = _payload_reference(payload)
        size = int(reference["size_bytes"])
        digest = str(reference["file_sha256"])
        total += size
        manifest_members.append({"role": role, "path": relative, **reference})
        index_rows.append(f"{role}\t{relative}\t{size}\t{digest}")
    if total > MAX_TOTAL_BYTES:
        raise ReleaseBundleError("bundle uncompressed content exceeds the total cap")
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": MANIFEST_KIND,
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "inventory_path": inventory_path,
        "release_spec_path": release_spec_path,
        "source_spec": _payload_reference(spec_payload),
        "member_count": len(manifest_members),
        "total_uncompressed_bytes": total,
        "members": manifest_members,
        "member_index_sha256": hashlib.sha256(
            ("\n".join(sorted(index_rows)) + "\n").encode()
        ).hexdigest(),
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    output_root = Path(allowed_output_root).resolve(strict=True)
    atomic_write_json_new(manifest, output_path, allowed_root=output_root)
    return manifest


def create_standard_release_bundle_spec(
    *,
    source_root: str | Path,
    repository_root: str | Path,
    candidate_commit: str,
    created_at_utc: str,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Create the exact standard member spec from validated evidence locations."""

    root = _source_root(source_root)
    candidate = _commit(candidate_commit, name="candidate_commit")
    mirror = f".audit/release-attestations/{candidate}"
    local = "local"
    local_record_path = f"{local}/local_cuda_gate.json"
    local_logs = [
        f"{local}/cuda_allocation.log",
        *(f"{local}/{gate_name}.log" for gate_name in LOCAL_GATE_COMMANDS),
    ]
    rows: list[dict[str, str]] = [
        {"role": "release_inventory", "path": "final_release_evidence_inventory.json"},
        {"role": "release_inventory_spec", "path": "release-spec.json"},
        {"role": "github_remote_evidence", "path": "remote.json"},
        {"role": "github_completed_ci_evidence", "path": "ci.json"},
        {"role": "github_api_response", "path": "repository-api.json"},
        {"role": "github_api_response", "path": "main-ref-api.json"},
        {"role": "github_api_response", "path": "workflow-run-api.json"},
        {"role": "github_api_response", "path": "run-artifacts-api.json"},
        {
            "role": "github_actions_archive",
            "path": f"release-security-{candidate}.zip",
        },
        {
            "role": "candidate_attestation",
            "path": f"{mirror}/final_release_gate_report.json",
        },
        {
            "role": "candidate_attestation_spec",
            "path": f"{mirror}/attestation_spec.json",
        },
        {"role": "staged_index_scan", "path": f"{mirror}/staged_index_scan.json"},
        {"role": "staged_secret_evidence", "path": f"{local}/staged_secret_scan.json"},
        {"role": "local_cuda_evidence", "path": local_record_path},
        {"role": "local_license_evidence", "path": f"{local}/license_audit.json"},
        *(
            {"role": "ci_mirror", "path": f"{mirror}/{basename}"}
            for basename in sorted(EXPECTED_ARTIFACT_FILES)
        ),
        *({"role": "local_quality_log", "path": path} for path in sorted(local_logs)),
    ]
    spec: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "spec_kind": SPEC_KIND,
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "inventory_path": "final_release_evidence_inventory.json",
        "release_spec_path": "release-spec.json",
        "members": rows,
    }
    _parse_spec(spec)
    payloads: dict[str, bytes] = {}
    for item in rows:
        payloads[item["path"]] = _source_file(
            root, item["path"], name=f"standard bundle member {item['path']}"
        ).read_bytes()
    observed_logs = _validate_standard_candidate_evidence(
        payloads=payloads,
        candidate=candidate,
        repository_root=repository_root,
    )
    if observed_logs != local_logs:
        raise ReleaseBundleError("standard local log membership differs from exact gate set")
    atomic_write_json_new(spec, output_path, allowed_root=allowed_output_root)
    return spec


def _publish_archive_new(
    *, target: Path, allowed_output_root: Path, writer: Any
) -> tuple[int, str]:
    root = allowed_output_root.resolve(strict=True)
    raw = target
    target = raw if raw.is_absolute() else root / raw
    try:
        relative = target.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleError("bundle output escapes allowed_output_root") from exc
    if (
        any(part in {"", ".", ".."} for part in relative.parts)
        or target.suffix.casefold() != ".zip"
    ):
        raise ReleaseBundleError("bundle output must be a canonical .zip path")
    current = root
    for part in relative.parent.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseBundleError("bundle output may not traverse a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent.resolve(strict=True)
    try:
        parent.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleError("bundle output escapes allowed_output_root") from exc
    target = parent / target.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite release bundle: {target}")
    temporary = parent / f".{target.name}.partial.{uuid4().hex}"
    try:
        writer(temporary)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except Exception as exc:
        raise ReleaseBundleError(
            f"bundle publication failed; partial retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()
    payload = target.read_bytes()
    return len(payload), hashlib.sha256(payload).hexdigest()


def _zip_info(path: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
    # Stored members make the complete container reproducible across zlib
    # implementations; the already-compressed Actions ZIP gains nothing from
    # another compression layer.
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = (stat.S_IFREG | 0o644) << 16
    return info


def _canonical_archive_bytes(
    *, spec_payload: bytes, manifest_payload: bytes, members: Sequence[tuple[str, bytes]]
) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, mode="w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr(_zip_info(SPEC_ARCHIVE_PATH), spec_payload)
        archive.writestr(_zip_info(MANIFEST_ARCHIVE_PATH), manifest_payload)
        for relative, payload in members:
            archive.writestr(_zip_info(relative), payload)
    return stream.getvalue()


def build_release_bundle(
    *,
    bundle_spec_path: str | Path,
    manifest_path: str | Path,
    source_root: str | Path,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Build a deterministic create-only ZIP from a validated manifest."""

    root = _source_root(source_root)
    spec_path = _resolve_input(bundle_spec_path, name="bundle spec", directory=False)
    manifest_source = _resolve_input(manifest_path, name="bundle manifest", directory=False)
    spec_payload = spec_path.read_bytes()
    manifest_payload = manifest_source.read_bytes()
    spec = _mapping(load_json_strict(spec_path), name="bundle spec")
    manifest = _mapping(load_json_strict(manifest_source), name="bundle manifest")
    spec_candidate, spec_inventory, spec_release_spec, spec_rows = _parse_spec(spec)
    candidate, inventory_path, release_spec_path, members = _validate_manifest(manifest)
    if (candidate, inventory_path, release_spec_path) != (
        spec_candidate,
        spec_inventory,
        spec_release_spec,
    ):
        raise ReleaseBundleError("bundle spec and manifest identity differ")
    source_spec = _mapping(manifest["source_spec"], name="source_spec")
    if _payload_reference(spec_payload) != dict(source_spec):
        raise ReleaseBundleError("bundle spec bytes differ from the manifest")
    if [(item["role"], item["path"]) for item in members] != sorted(
        spec_rows, key=lambda item: item[1].casefold()
    ):
        raise ReleaseBundleError("bundle manifest membership differs from the reviewed spec")
    payloads: list[tuple[str, bytes]] = []
    for member in members:
        role = str(member["role"])
        relative = str(member["path"])
        payload = _source_file(root, relative, name=f"bundle source {relative}").read_bytes()
        _validate_member_payload(role=role, path=relative, payload=payload)
        if _payload_reference(payload) != {
            "size_bytes": member["size_bytes"],
            "file_sha256": member["file_sha256"],
        }:
            raise ReleaseBundleError(f"bundle source changed after manifest review: {relative}")
        payloads.append((relative, payload))

    archive_payload = _canonical_archive_bytes(
        spec_payload=spec_payload,
        manifest_payload=manifest_payload,
        members=payloads,
    )

    def write_archive(path: Path) -> None:
        with path.open("xb") as handle:
            handle.write(archive_payload)
            handle.flush()
            os.fsync(handle.fileno())

    size, digest = _publish_archive_new(
        target=Path(output_path),
        allowed_output_root=Path(allowed_output_root),
        writer=write_archive,
    )
    return {
        "status": "pass",
        "candidate_commit": candidate,
        "member_count": len(payloads) + 2,
        "archive_size_bytes": size,
        "archive_sha256": digest,
        "manifest_record_sha256": manifest["record_sha256"],
    }


def _zip_member_mode(info: zipfile.ZipInfo) -> int:
    return (info.external_attr >> 16) & 0xFFFF


def validate_release_bundle_archive(
    *,
    archive_path: str | Path,
    bundle_spec_path: str | Path,
    manifest_path: str | Path,
    candidate_commit: str,
    repository_root: str | Path | None = None,
    extraction_root: str | Path | None = None,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate exact ZIP membership and optionally reconstruct inventory offline."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    spec_source = _resolve_input(bundle_spec_path, name="bundle spec", directory=False)
    manifest_source = _resolve_input(manifest_path, name="bundle manifest", directory=False)
    archive_source = _resolve_input(archive_path, name="bundle archive", directory=False)
    spec_payload = spec_source.read_bytes()
    manifest_payload = manifest_source.read_bytes()
    spec = _mapping(load_json_strict(spec_source), name="bundle spec")
    manifest = _mapping(load_json_strict(manifest_source), name="bundle manifest")
    spec_candidate, spec_inventory, spec_release_spec, spec_rows = _parse_spec(spec)
    manifest_candidate, inventory_path, release_spec_path, members = _validate_manifest(manifest)
    if candidate != spec_candidate or candidate != manifest_candidate:
        raise ReleaseBundleError("bundle candidate identity differs")
    if (inventory_path, release_spec_path) != (spec_inventory, spec_release_spec):
        raise ReleaseBundleError("bundle inventory/spec paths differ")
    if _payload_reference(spec_payload) != dict(
        _mapping(manifest["source_spec"], name="source_spec")
    ):
        raise ReleaseBundleError("bundle spec bytes differ from manifest reference")
    if [(item["role"], item["path"]) for item in members] != sorted(
        spec_rows, key=lambda item: item[1].casefold()
    ):
        raise ReleaseBundleError("bundle reviewed membership differs from manifest")
    expected = {
        SPEC_ARCHIVE_PATH: spec_payload,
        MANIFEST_ARCHIVE_PATH: manifest_payload,
    }
    member_by_path = {str(item["path"]): item for item in members}
    archive_payload = archive_source.read_bytes()
    if len(archive_payload) > MAX_TOTAL_BYTES:
        raise ReleaseBundleError("compressed release bundle exceeds the size cap")
    extracted_payloads: dict[str, bytes] = {}
    try:
        with zipfile.ZipFile(archive_source) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len({name.casefold() for name in names}):
                raise ReleaseBundleError("bundle ZIP contains duplicate or case-colliding members")
            if set(names) != set(expected) | set(member_by_path):
                raise ReleaseBundleError("bundle ZIP member set differs from the exact manifest")
            if len(names) != len(expected) + len(member_by_path):
                raise ReleaseBundleError("bundle ZIP member count differs")
            total = 0
            for info in infos:
                name = _safe_relative(info.filename, name="ZIP member")
                if info.is_dir() or info.flag_bits & 0x1:
                    raise ReleaseBundleError("bundle ZIP contains a directory or encrypted member")
                mode = _zip_member_mode(info)
                if stat.S_ISLNK(mode) or (mode and not stat.S_ISREG(mode)):
                    raise ReleaseBundleError("bundle ZIP contains a symlink or non-file member")
                if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                    raise ReleaseBundleError("bundle ZIP uses an unsupported compression method")
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ReleaseBundleError("bundle ZIP member exceeds the size cap")
                if (
                    info.compress_size > 0
                    and info.file_size > info.compress_size * MAX_COMPRESSION_RATIO
                ):
                    raise ReleaseBundleError("bundle ZIP member exceeds the compression-ratio cap")
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise ReleaseBundleError("bundle ZIP member size differs from metadata")
                total += len(payload)
                if name in expected:
                    if payload != expected[name]:
                        raise ReleaseBundleError("embedded bundle control record differs")
                else:
                    member = member_by_path[name]
                    _validate_member_payload(role=str(member["role"]), path=name, payload=payload)
                    if _payload_reference(payload) != {
                        "size_bytes": member["size_bytes"],
                        "file_sha256": member["file_sha256"],
                    }:
                        raise ReleaseBundleError(f"bundle ZIP member bytes differ: {name}")
                    extracted_payloads[name] = payload
            if total > MAX_TOTAL_BYTES:
                raise ReleaseBundleError("bundle ZIP uncompressed size exceeds the total cap")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ReleaseBundleError("release bundle is not a valid ZIP archive") from exc
    canonical_archive = _canonical_archive_bytes(
        spec_payload=spec_payload,
        manifest_payload=manifest_payload,
        members=[(str(item["path"]), extracted_payloads[str(item["path"])]) for item in members],
    )
    if archive_payload != canonical_archive:
        raise ReleaseBundleError(
            "release bundle container differs from the canonical deterministic ZIP bytes"
        )

    inventory_validation: Mapping[str, Any] | None = None
    if repository_root is not None or extraction_root is not None:
        if repository_root is None or extraction_root is None:
            raise ReleaseBundleError(
                "repository_root and extraction_root are both required for offline reconstruction"
            )
        resolved_destination = _create_new_directory(extraction_root, name="bundle extraction root")
        for relative, payload in sorted(extracted_payloads.items()):
            target = resolved_destination.joinpath(*PurePosixPath(relative).parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            if any(
                resolved_destination.joinpath(*PurePosixPath(relative).parts[:index]).is_symlink()
                for index in range(1, len(PurePosixPath(relative).parts) + 1)
            ):
                raise ReleaseBundleError("bundle extraction path traverses a symlink")
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _validate_standard_candidate_evidence(
            payloads=extracted_payloads,
            candidate=candidate,
            repository_root=repository_root,
        )
        inventory_validation = validate_release_evidence_inventory_file(
            resolved_destination.joinpath(*PurePosixPath(inventory_path).parts),
            spec_path=resolved_destination.joinpath(*PurePosixPath(release_spec_path).parts),
            repository_root=repository_root,
            expected_candidate_commit=candidate,
        )
        if (
            inventory_validation.get("valid") is not True
            or inventory_validation.get("status") != "ready"
        ):
            raise ReleaseBundleError(
                "offline release inventory must reconstruct with ready status: "
                f"{inventory_validation.get('errors')}"
            )
    index_rows = [
        f"{item['role']}\t{item['path']}\t{item['size_bytes']}\t{item['file_sha256']}"
        for item in members
    ]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "record_kind": VALIDATION_KIND,
        "status": "pass",
        "created_at_utc": _timestamp(created_at_utc),
        "candidate_commit": candidate,
        "archive": {
            "size_bytes": len(archive_payload),
            "file_sha256": hashlib.sha256(archive_payload).hexdigest(),
        },
        "manifest_record_sha256": manifest["record_sha256"],
        "member_count": len(members) + 2,
        "member_index_sha256": hashlib.sha256(
            ("\n".join(sorted(index_rows)) + "\n").encode()
        ).hexdigest(),
        "extra_member_count": 0,
        "offline_inventory_reconstructed": inventory_validation is not None,
        "inventory_record_sha256": (
            inventory_validation.get("record_sha256") if inventory_validation else None
        ),
    }
    report["record_sha256"] = canonical_json_sha256(report)
    return report


def write_release_notes(
    *,
    inventory_path: str | Path,
    manifest_path: str | Path,
    archive_path: str | Path,
    candidate_commit: str,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Write deterministic, create-only private-prerelease notes from exact assets."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    inventory_source = _resolve_input(inventory_path, name="release inventory", directory=False)
    manifest_source = _resolve_input(manifest_path, name="bundle manifest", directory=False)
    archive_source = _resolve_input(archive_path, name="bundle archive", directory=False)
    inventory_payload = inventory_source.read_bytes()
    inventory = _strict_json_payload(inventory_payload, name="release inventory")
    inventory_body = dict(inventory)
    inventory_hash = _sha256(
        inventory_body.pop("record_sha256", None), name="inventory.record_sha256"
    )
    if canonical_json_sha256(inventory_body) != inventory_hash:
        raise ReleaseBundleError("release inventory self-hash does not reconstruct")
    repository = _mapping(inventory.get("repository"), name="inventory.repository")
    confirmatory = _mapping(
        inventory.get("confirmatory_state"), name="inventory.confirmatory_state"
    )
    if (
        inventory.get("status") != "ready"
        or repository.get("code_commit") != candidate
        or confirmatory.get("opening_count") != 1
        or isinstance(confirmatory.get("opening_count"), bool)
        or confirmatory.get("target_rerun_permitted") is not False
    ):
        raise ReleaseBundleError("release notes require a ready one-opening inventory")
    manifest = _mapping(load_json_strict(manifest_source), name="bundle manifest")
    manifest_candidate, _inventory, _release_spec, _members = _validate_manifest(manifest)
    if manifest_candidate != candidate:
        raise ReleaseBundleError("release notes assets identify different candidates")
    inventory_reference = _payload_reference(inventory_payload)
    archive_payload = archive_source.read_bytes()
    archive_reference = _payload_reference(archive_payload)
    notes = (
        "# InclusiveShift-HAR benchmark v0.1.7 prerelease\n\n"
        f"Candidate commit: `{candidate}`\n\n"
        "This private prerelease preserves the validated benchmark code and its exact "
        "release-evidence assets. The locked target was opened once; no target rerun was "
        "performed for this release. No DOI was minted.\n\n"
        "Assets:\n\n"
        f"- `{inventory_source.name}` — {inventory_reference['size_bytes']} bytes; "
        f"SHA-256 `{inventory_reference['file_sha256']}`; record "
        f"`{inventory_hash}`.\n"
        f"- `{archive_source.name}` — {archive_reference['size_bytes']} bytes; "
        f"SHA-256 `{archive_reference['file_sha256']}`; bundle manifest record "
        f"`{manifest['record_sha256']}`.\n"
    )
    root = _resolve_input(allowed_output_root, name="allowed_output_root", directory=True)
    raw_output = Path(output_path)
    output = raw_output if raw_output.is_absolute() else root / raw_output
    try:
        relative = output.relative_to(root)
    except ValueError as exc:
        raise ReleaseBundleError("release-notes output escapes allowed_output_root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts) or output.suffix.casefold() != ".md":
        raise ReleaseBundleError("release-notes output must be a canonical Markdown path")
    current = root
    for part in relative.parent.parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseBundleError("release-notes output may not traverse a symlink")
        current.mkdir(exist_ok=True)
    output = output.parent.resolve(strict=True) / output.name
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to overwrite release notes: {output}")
    with output.open("xb") as handle:
        handle.write(notes.encode("utf-8"))
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "pass",
        "candidate_commit": candidate,
        "output_basename": output.name,
        "size_bytes": output.stat().st_size,
        "file_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    standard = commands.add_parser("standard-spec")
    standard.add_argument("--source-root", type=Path, required=True)
    standard.add_argument("--repository-root", type=Path, required=True)
    standard.add_argument("--candidate-commit", required=True)
    standard.add_argument("--created-at-utc", required=True)
    standard.add_argument("--output", type=Path, required=True)
    standard.add_argument("--allowed-output-root", type=Path, required=True)
    manifest = commands.add_parser("manifest")
    manifest.add_argument("--spec", type=Path, required=True)
    manifest.add_argument("--source-root", type=Path, required=True)
    manifest.add_argument("--created-at-utc", required=True)
    manifest.add_argument("--output", type=Path, required=True)
    manifest.add_argument("--allowed-output-root", type=Path, required=True)
    build = commands.add_parser("build")
    build.add_argument("--spec", type=Path, required=True)
    build.add_argument("--manifest", type=Path, required=True)
    build.add_argument("--source-root", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--allowed-output-root", type=Path, required=True)
    notes = commands.add_parser("notes")
    notes.add_argument("--inventory", type=Path, required=True)
    notes.add_argument("--manifest", type=Path, required=True)
    notes.add_argument("--archive", type=Path, required=True)
    notes.add_argument("--candidate-commit", required=True)
    notes.add_argument("--output", type=Path, required=True)
    notes.add_argument("--allowed-output-root", type=Path, required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--archive", type=Path, required=True)
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--candidate-commit", required=True)
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--extraction-root", type=Path, required=True)
    validate.add_argument("--created-at-utc", required=True)
    validate.add_argument("--output", type=Path, required=True)
    validate.add_argument("--allowed-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "standard-spec":
            result = create_standard_release_bundle_spec(
                source_root=args.source_root,
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                created_at_utc=args.created_at_utc,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
            summary = {
                "status": "pass",
                "member_count": len(result["members"]),
                "spec_sha256": hashlib.sha256(
                    Path(args.output).resolve(strict=True).read_bytes()
                ).hexdigest(),
            }
        elif args.command == "manifest":
            result = assemble_release_bundle_manifest(
                bundle_spec_path=args.spec,
                source_root=args.source_root,
                created_at_utc=args.created_at_utc,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
            summary = {
                "status": "pass",
                "member_count": result["member_count"],
                "record_sha256": result["record_sha256"],
            }
        elif args.command == "build":
            summary = build_release_bundle(
                bundle_spec_path=args.spec,
                manifest_path=args.manifest,
                source_root=args.source_root,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
        elif args.command == "notes":
            summary = write_release_notes(
                inventory_path=args.inventory,
                manifest_path=args.manifest,
                archive_path=args.archive,
                candidate_commit=args.candidate_commit,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
        else:
            summary = validate_release_bundle_archive(
                archive_path=args.archive,
                bundle_spec_path=args.spec,
                manifest_path=args.manifest,
                candidate_commit=args.candidate_commit,
                repository_root=args.repository_root,
                extraction_root=args.extraction_root,
                created_at_utc=args.created_at_utc,
            )
            atomic_write_json_new(summary, args.output, allowed_root=args.allowed_output_root)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
