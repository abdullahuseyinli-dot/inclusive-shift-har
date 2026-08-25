"""Create-only generation and offline validation of final release evidence inventories."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.artifacts.github_evidence import (
    EXPECTED_ARTIFACT_FILES,
    MAX_ARCHIVE_BYTES,
    GitHubEvidenceError,
    validate_ci_archive_payload,
)
from inclusive_shift_har.artifacts.release_gate import (
    SECRET_SCAN_SCOPE,
    ReleaseGateError,
    inspect_git_ref_state,
    validate_license_audit_semantics,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
)

SCHEMA_VERSION = "1.1.0"
SPEC_SCHEMA_VERSION = "1.0.0"
INVENTORY_KIND = "final_release_evidence_inventory"
SPEC_KIND = "release_evidence_inventory_spec"
MAX_EVIDENCE_BYTES = 16 * 1024 * 1024
ACTIVE_PROTOCOL_TAG = "protocol-v1.2.0"

REQUIRED_ROLES = (
    "legacy_audit",
    "legacy_verification",
    "literature_matrix",
    "dataset_manifest",
    "dataset_card",
    "data_audit",
    "ontology_lock",
    "locked_protocol",
    "split_manifest_and_audit",
    "source_artifact_freeze",
    "opening_receipt",
    "locked_target_index",
    "participant_statistics",
    "publication_report",
    "benchmark_card",
    "model_card",
    "baseline_coverage",
    "experiment_runbook",
    "paper_outline",
    "release_gate_report",
)
CANONICAL_REQUIRED_ROLE_PATHS = {
    "legacy_audit": "docs/LEGACY_AUDIT.md",
    "legacy_verification": "legacy/verification_results.json",
    "literature_matrix": "docs/LITERATURE_MATRIX.md",
    "dataset_manifest": "manifests/datasets/inclusivehar_v4.json",
    "dataset_card": "docs/data/INCLUSIVEHAR_V4.md",
    "data_audit": "results/data_audit/inclusivehar_v4.audit.json",
    "ontology_lock": "results/protocol/ontology_lock_v1_1.json",
    "locked_protocol": "docs/LOCKED_PROTOCOL.md",
    "split_manifest_and_audit": "results/protocol/split_audit_v1_2.json",
    "source_artifact_freeze": "results/protocol/final_source_artifact_freeze_v1.json",
    "opening_receipt": "results/protocol/confirmatory_target_opening_1.json",
    "locked_target_index": (
        "results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json"
    ),
    "participant_statistics": "results/confirmatory/zero_shot_v1/participant_statistics.json",
    "publication_report": "results/confirmatory/zero_shot_v1/publication_report_v1.json",
    "benchmark_card": "docs/BENCHMARK_CARD.md",
    "model_card": "docs/MODEL_CARD_MORE_HAR.md",
    "baseline_coverage": "docs/baselines/BASELINE_COVERAGE_AND_OMISSIONS.md",
    "experiment_runbook": "docs/EXPERIMENT_RUNBOOK.md",
    "paper_outline": "paper/OUTLINE.md",
    "release_gate_report": "results/release/final_release_gate_report.json",
}

POSTCONFIRMATORY_ROLE_BY_TRACK = {
    "ccil_bpd_postconfirmatory_v1": "postconfirmatory_ccil_bpd",
    "cross_source_pretraining": "postconfirmatory_cross_source",
    "few_person_inclusion_curve_v1_1": "postconfirmatory_few_person",
    "neural_efficiency_profile_v1": "postconfirmatory_efficiency",
    "raw_total_acceleration_sensitivity_v1": "postconfirmatory_raw_total",
    "sensor_reliability_stress_v1": "postconfirmatory_sensor_stress",
    "si_unit_sensitivity_v1": "postconfirmatory_si_unit",
    "ssl_foundation_comparison": "postconfirmatory_ssl_foundation",
    "uci_corrected_reproduction_v1_1": "postconfirmatory_uci_reproduction",
    "within_group_cross_subject": "postconfirmatory_within_group",
}
POSTCONFIRMATORY_TRACKS = tuple(sorted(POSTCONFIRMATORY_ROLE_BY_TRACK))
OPTIONAL_ARTIFACT_ROLES = (
    *sorted(set(POSTCONFIRMATORY_ROLE_BY_TRACK.values())),
    "environment_record",
    "failure_or_deviation",
    "license_notice",
)
ARTIFACT_ROLES = frozenset((*REQUIRED_ROLES, *OPTIONAL_ARTIFACT_ROLES))
GATE_NAMES = (
    "tests",
    "lint",
    "format",
    "types",
    "manifests",
    "splits",
    "configuration",
    "artifacts",
    "staged_index_scan",
    "tracked_file_scan",
    "secret_scan",
    "license_audit",
    "ci",
)
QUALITY_GATE_NAMES = frozenset(
    {
        "tests",
        "lint",
        "format",
        "types",
        "manifests",
        "splits",
        "configuration",
        "artifacts",
    }
)
QUALITY_EVIDENCE_KIND = "ci_release_quality_gate_evidence"
QUALITY_EVIDENCE_SCOPE = "within_job_quality_proxy_not_completed_workflow"
COMPLETED_CI_EVIDENCE_SCOPE = "completed_remote_workflow_run"
NON_CI_GATE_RECORD_KINDS = {
    "staged_index_scan": "git_index_release_scan",
    "tracked_file_scan": "git_object_release_scan",
    "secret_scan": "gitleaks_secret_scan_attestation",
    "license_audit": "python_license_audit",
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
QUALITY_EVIDENCE_KEYS = {
    "schema_version",
    "record_kind",
    "status",
    "candidate_commit",
    "created_at_utc",
    "gates",
    "gate_statuses",
    "scope",
    "ci_evidence_scope",
    "external_completed_ci_required_for_release_inventory",
    "gate_evidence",
    "tool_versions",
    "machine",
    "record_sha256",
}
NON_CI_GATE_RECORD_KEYS = {
    "tracked_file_scan": {
        "schema_version",
        "record_kind",
        "status",
        "created_at_utc",
        "candidate_commit",
        "scan_scope",
        "policy",
        "commit_count",
        "unique_blob_count",
        "tree_entry_observation_count",
        "ref_count",
        "tag_ref_count",
        "validated_annotated_tag_count",
        "invalid_ref_target_count",
        "ref_metadata_index_sha256",
        "shallow_repository",
        "replace_ref_count",
        "grafts_file_present",
        "scanned_object_index_sha256",
        "violations",
        "raw_worktree_paths_opened",
        "record_sha256",
    },
    "secret_scan": {
        "schema_version",
        "record_kind",
        "status",
        "created_at_utc",
        "candidate_commit",
        "scan_scope",
        "ref_metadata_index_sha256",
        "validated_annotated_tag_count",
        "gitleaks_version",
        "gitleaks_exit_code",
        "execution_status",
        "policy",
        "config",
        "ignore",
        "raw_report",
        "finding_count",
        "record_sha256",
    },
    "license_audit": {
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
    },
}

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_GITHUB_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9._-]{1,100}$"
)
_ARTIFACT_STATUSES = {
    "pass",
    "conditional",
    "failed",
    "quarantined",
    "not_run",
    "incomplete",
    "blocked",
}
_GATE_STATUSES = {"pass", "fail", "not_run", "not_applicable"}
_POSTCONFIRMATORY_STATUSES = {"completed", "incomplete", "not_run", "blocked", "not_applicable"}
_INVENTORY_STATUSES = {"draft", "blocked", "ready"}
_BLOCKER_STATUSES = {"open", "resolved", "accepted_limitation"}
_FORBIDDEN_PREFIXES = (
    (".audit",),
    ("data", "raw"),
    ("data", "cache"),
    ("data", "processed"),
    ("data", "restricted"),
    ("legacy", "source_archives"),
    ("legacy", "extracted"),
    ("checkpoints",),
)
FORBIDDEN_ARTIFACT_SUFFIXES = frozenset(
    {
        ".7z",
        ".bin",
        ".bz2",
        ".ckpt",
        ".db",
        ".dill",
        ".dll",
        ".docx",
        ".exe",
        ".feather",
        ".gz",
        ".h5",
        ".hdf5",
        ".jar",
        ".joblib",
        ".key",
        ".mat",
        ".npy",
        ".npz",
        ".onnx",
        ".p12",
        ".parquet",
        ".pem",
        ".pfx",
        ".pickle",
        ".pkl",
        ".pt",
        ".pth",
        ".rar",
        ".safetensors",
        ".so",
        ".sqlite",
        ".tar",
        ".tflite",
        ".tgz",
        ".whl",
        ".xz",
        ".zip",
        ".zst",
    }
)


class ReleaseEvidenceError(RuntimeError):
    """Raised when release evidence cannot be generated or validated safely."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ReleaseEvidenceError(f"{name} must be a string-keyed object")
    return value


def _sequence(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ReleaseEvidenceError(f"{name} must be an array")
    return value


def _exact_keys(value: Mapping[str, Any], expected: set[str], *, name: str) -> None:
    observed = set(value)
    if observed != expected:
        raise ReleaseEvidenceError(
            f"{name} keys differ; missing={sorted(expected - observed)}, "
            f"unexpected={sorted(observed - expected)}"
        )


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ReleaseEvidenceError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    text = _string(value, name=name)
    if _SHA256_RE.fullmatch(text) is None:
        raise ReleaseEvidenceError(f"{name} must be a lowercase SHA-256")
    return text


def _commit(value: Any, *, name: str) -> str:
    text = _string(value, name=name)
    if _COMMIT_RE.fullmatch(text) is None:
        raise ReleaseEvidenceError(f"{name} must be a full lowercase Git commit")
    return text


def _exact_integer(
    value: Any,
    *,
    name: str,
    expected: int | None = None,
    minimum: int | None = None,
) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ReleaseEvidenceError(f"{name} must be an exact JSON integer")
    if expected is not None and value != expected:
        raise ReleaseEvidenceError(f"{name} must equal {expected}")
    if minimum is not None and value < minimum:
        raise ReleaseEvidenceError(f"{name} must be at least {minimum}")
    return value


def _nullable_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _github_repository(value: Any, *, name: str) -> str:
    repository = _string(value, name=name)
    repository_name = repository.split("/", 1)[1] if "/" in repository else ""
    if (
        _GITHUB_REPOSITORY_RE.fullmatch(repository) is None
        or repository.casefold().endswith(".git")
        or repository_name in {".", ".."}
        or ".." in repository_name
    ):
        raise ReleaseEvidenceError(f"{name} must be a canonical owner/repository identity")
    return repository


def _timestamp(value: Any) -> str:
    text = _string(value, name="created_at_utc")
    if _TIMESTAMP_RE.fullmatch(text) is None:
        raise ReleaseEvidenceError("created_at_utc must use canonical ISO-8601 UTC Z form")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseEvidenceError("created_at_utc must be valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseEvidenceError("created_at_utc must be UTC")
    return text


def _safe_relative_path(value: Any, *, name: str) -> str:
    text = _string(value, name=name)
    if (
        "\\" in text
        or ":" in text
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
    ):
        raise ReleaseEvidenceError(f"{name} must be a portable POSIX relative path")
    path = PurePosixPath(text)
    if (
        path.is_absolute()
        or path.as_posix() != text
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ReleaseEvidenceError(f"{name} must be a portable POSIX relative path")
    folded = tuple(part.casefold() for part in path.parts)
    if any(folded[: len(prefix)] == prefix for prefix in _FORBIDDEN_PREFIXES):
        raise ReleaseEvidenceError(f"{name} points into protected raw/cache evidence")
    if path.suffix.casefold() in FORBIDDEN_ARTIFACT_SUFFIXES:
        raise ReleaseEvidenceError(f"{name} has a prohibited raw/checkpoint extension")
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ReleaseEvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise ReleaseEvidenceError(f"non-finite JSON number is forbidden: {token}")


def _load_json_bytes(payload: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
        value = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ReleaseEvidenceError(f"{name} is not strict UTF-8 JSON: {exc}") from exc
    return _mapping(value, name=name)


def _git(
    repository_root: Path,
    arguments: Sequence[str],
    *,
    text: bool,
    check: bool = True,
) -> subprocess.CompletedProcess[str] | subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            ["git", "--no-replace-objects", *arguments],
            cwd=repository_root,
            check=check,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseEvidenceError(f"Git command failed: {' '.join(arguments)}") from exc


def _repository(path: str | Path) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise ReleaseEvidenceError("repository_root may not be a symlink")
    root = source.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseEvidenceError("repository_root must be an existing directory")
    inside = _git(root, ("rev-parse", "--is-inside-work-tree"), text=True)
    assert isinstance(inside.stdout, str)
    if inside.stdout.strip() != "true":
        raise ReleaseEvidenceError("repository_root is not a Git worktree")
    top_level = _git(root, ("rev-parse", "--show-toplevel"), text=True)
    assert isinstance(top_level.stdout, str)
    try:
        actual_root = Path(top_level.stdout.strip()).resolve(strict=True)
    except OSError as exc:
        raise ReleaseEvidenceError("Git worktree root cannot be resolved") from exc
    if not root.samefile(actual_root):
        raise ReleaseEvidenceError("repository_root must equal the Git worktree top-level")
    shallow = _git(root, ("rev-parse", "--is-shallow-repository"), text=True)
    assert isinstance(shallow.stdout, str)
    if shallow.stdout.strip() != "false":
        raise ReleaseEvidenceError("release inventory requires complete, non-shallow history")
    replace_refs = _git(root, ("for-each-ref", "--format=%(refname)", "refs/replace"), text=True)
    assert isinstance(replace_refs.stdout, str)
    if replace_refs.stdout.strip():
        raise ReleaseEvidenceError("release inventory forbids Git replace refs")
    grafts_result = _git(root, ("rev-parse", "--git-path", "info/grafts"), text=True)
    assert isinstance(grafts_result.stdout, str)
    raw_grafts = Path(grafts_result.stdout.strip())
    grafts = root / raw_grafts if not raw_grafts.is_absolute() else raw_grafts
    if grafts.is_symlink() or (grafts.is_file() and grafts.stat().st_size > 0):
        raise ReleaseEvidenceError("release inventory forbids legacy Git grafts")
    return root


def _git_state(repository_root: Path) -> tuple[str, str]:
    commit_result = _git(repository_root, ("rev-parse", "HEAD"), text=True)
    status_result = _git(
        repository_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        text=True,
    )
    assert isinstance(commit_result.stdout, str)
    assert isinstance(status_result.stdout, str)
    return _commit(
        commit_result.stdout.strip(), name="repository HEAD"
    ), status_result.stdout.strip()


def _require_commit(repository_root: Path, commit: str) -> None:
    result = _git(
        repository_root,
        ("cat-file", "-e", f"{commit}^{{commit}}"),
        text=False,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseEvidenceError("candidate_commit does not identify a Git commit")


def _protocol_tag_binding(
    repository_root: Path, tag_name: str, *, candidate_commit: str
) -> tuple[str, str, str]:
    name = _string(tag_name, name="protocol_tag")
    if _IDENTIFIER_RE.fullmatch(name) is None or name != ACTIVE_PROTOCOL_TAG:
        raise ReleaseEvidenceError(
            f"protocol_tag must be the active protocol anchor {ACTIVE_PROTOCOL_TAG}"
        )
    _policy_reference, policy = _candidate_policy(repository_root, candidate_commit)
    git_refs = _mapping(policy.get("git_refs"), name="candidate git_refs policy")
    required_tags = [
        _string(value, name="required release tag")
        for value in _sequence(
            git_refs.get("required_for_release_tags"),
            name="required release tags",
        )
    ]
    if name not in required_tags:
        raise ReleaseEvidenceError("active protocol_tag is not required by the candidate policy")
    reference = f"refs/tags/{name}"
    object_result = _git(
        repository_root, ("rev-parse", "--verify", reference), text=True, check=False
    )
    if object_result.returncode != 0:
        raise ReleaseEvidenceError("protocol_tag does not exist")
    assert isinstance(object_result.stdout, str)
    object_id = _commit(object_result.stdout.strip(), name="protocol_tag object ID")
    kind_result = _git(repository_root, ("cat-file", "-t", object_id), text=True)
    assert isinstance(kind_result.stdout, str)
    if kind_result.stdout.strip() != "tag":
        raise ReleaseEvidenceError("protocol_tag must be an annotated tag")
    target_result = _git(repository_root, ("rev-parse", f"{reference}^{{commit}}"), text=True)
    assert isinstance(target_result.stdout, str)
    target = _commit(target_result.stdout.strip(), name="protocol_tag target")
    ancestry = _git(
        repository_root,
        ("merge-base", "--is-ancestor", target, candidate_commit),
        text=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ReleaseEvidenceError("protocol_tag target is not an ancestor of the candidate")
    try:
        _ref_metrics, ref_violations = inspect_git_ref_state(
            repository_root, policy, candidate=candidate_commit
        )
    except ReleaseGateError as exc:
        raise ReleaseEvidenceError("protocol_tag Git ref policy validation failed") from exc
    if ref_violations:
        raise ReleaseEvidenceError("protocol_tag or another required Git ref violates policy")
    pinned = _mapping(git_refs.get("pinned_annotated_tags"), name="pinned annotated tags").get(name)
    if pinned is not None:
        pinned_mapping = _mapping(pinned, name=f"pinned tag {name}")
        if (
            pinned_mapping.get("object_id") != object_id
            or pinned_mapping.get("target_commit") != target
        ):
            raise ReleaseEvidenceError("protocol_tag differs from its candidate-policy pin")
    return name, object_id, target


def _git_blob(repository_root: Path, commit: str, path: str, *, name: str) -> bytes:
    relative = _safe_relative_path(path, name=name)
    tree = _git(
        repository_root,
        ("ls-tree", "-z", commit, "--", relative),
        text=False,
    )
    assert isinstance(tree.stdout, bytes)
    entries = [entry for entry in tree.stdout.split(b"\x00") if entry]
    if len(entries) != 1:
        raise ReleaseEvidenceError(f"{name} is not one tracked file at candidate_commit")
    try:
        metadata, observed_path = entries[0].split(b"\t", 1)
        mode, kind, _object_id = metadata.decode("ascii").split(" ", 2)
        decoded_path = observed_path.decode("utf-8")
    except (UnicodeError, ValueError) as exc:
        raise ReleaseEvidenceError(f"{name} has invalid Git tree metadata") from exc
    if decoded_path != relative or kind != "blob" or mode not in {"100644", "100755"}:
        raise ReleaseEvidenceError(f"{name} must be a regular tracked file, not a symlink")
    blob = _git(repository_root, ("cat-file", "blob", f"{commit}:{relative}"), text=False)
    assert isinstance(blob.stdout, bytes)
    if len(blob.stdout) > MAX_EVIDENCE_BYTES:
        raise ReleaseEvidenceError(f"{name} exceeds the {MAX_EVIDENCE_BYTES}-byte evidence cap")
    return blob.stdout


def _candidate_tree_index(repository_root: Path, commit: str) -> tuple[int, str]:
    result = _git(
        repository_root,
        ("ls-tree", "-r", "-z", "-l", commit),
        text=False,
    )
    assert isinstance(result.stdout, bytes)
    rows: list[str] = []
    for raw in (item for item in result.stdout.split(b"\x00") if item):
        try:
            metadata, raw_path = raw.split(b"\t", 1)
            mode, kind, object_id, size_text = metadata.decode("ascii").split(" ", 3)
            path = _safe_relative_path(raw_path.decode("utf-8"), name="candidate tree path")
        except (UnicodeError, ValueError) as exc:
            raise ReleaseEvidenceError("candidate tree contains invalid metadata") from exc
        size: int | None
        try:
            size = None if size_text == "-" else int(size_text)
        except ValueError as exc:
            raise ReleaseEvidenceError("candidate tree contains an invalid object size") from exc
        rows.append(f"{mode} {object_id} 0 {kind} {size} {path}")
    return len(rows), hashlib.sha256(("\n".join(sorted(rows)) + "\n").encode()).hexdigest()


def _blob_reference(
    repository_root: Path,
    commit: str,
    path: str,
    expected_sha256: str,
    *,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    relative = _safe_relative_path(path, name=f"{name}.path")
    expected = _sha256(expected_sha256, name=f"{name}.expected_sha256")
    payload = _git_blob(repository_root, commit, relative, name=name)
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ReleaseEvidenceError(f"{name} file SHA-256 differs from its reviewed pin")
    return {"path": relative, "size_bytes": len(payload), "file_sha256": observed}, payload


def _candidate_policy(
    repository_root: Path, commit: str
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    path = "configs/release/release_gate_policy_v1.json"
    payload = _git_blob(repository_root, commit, path, name="release gate policy")
    policy = _load_json_bytes(payload, name="release gate policy")
    if (
        policy.get("schema_version") != "1.0.0"
        or policy.get("policy_kind") != "final_release_gate_policy"
    ):
        raise ReleaseEvidenceError("candidate release gate policy version or kind differs")
    return (
        {
            "path": path,
            "size_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        },
        policy,
    )


def _validate_gitleaks_ignore_entries(policy: Mapping[str, Any], payload: bytes) -> None:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeError as exc:
        raise ReleaseEvidenceError("Gitleaks ignore file must be UTF-8") from exc
    gitleaks = _mapping(policy.get("gitleaks"), name="candidate Gitleaks policy")
    reviewed = [
        _string(value, name="reviewed Gitleaks ignore entry")
        for value in _sequence(
            gitleaks.get("reviewed_ignore_entries"),
            name="reviewed Gitleaks ignore entries",
        )
    ]
    observed = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if observed != reviewed:
        raise ReleaseEvidenceError("Gitleaks ignore entries differ from the candidate policy")


def _safe_spec_evidence_path(value: Any, *, name: str, candidate: str) -> str:
    text = _string(value, name=name)
    path = PurePosixPath(text)
    if (
        "\\" in text
        or ":" in text
        or any(ord(character) < 32 or ord(character) == 127 for character in text)
        or path.is_absolute()
        or path.as_posix() != text
        or any(part in {".", ".."} for part in path.parts)
    ):
        raise ReleaseEvidenceError(f"{name} must be a portable POSIX relative path")
    candidate_prefix = (".audit", "release-attestations", candidate)
    if path.parts[:3] == candidate_prefix and len(path.parts) == 4:
        if path.suffix.casefold() in FORBIDDEN_ARTIFACT_SUFFIXES:
            raise ReleaseEvidenceError(f"{name} has a prohibited raw/checkpoint extension")
        return text
    return _safe_relative_path(text, name=name)


def _spec_root_blob(
    spec_root: Path,
    path: str,
    expected_sha256: str,
    *,
    name: str,
    candidate: str,
) -> tuple[dict[str, Any], bytes]:
    relative = _safe_spec_evidence_path(
        path,
        name=f"{name}.path",
        candidate=candidate,
    )
    source = spec_root / Path(relative)
    current = spec_root
    for part in PurePosixPath(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ReleaseEvidenceError(f"{name} may not traverse a symlink")
    try:
        resolved = source.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise ReleaseEvidenceError(f"{name} is missing from spec_root") from exc
    try:
        resolved.relative_to(spec_root)
    except ValueError as exc:
        raise ReleaseEvidenceError(f"{name} escapes spec_root") from exc
    if not resolved.is_file():
        raise ReleaseEvidenceError(f"{name} must be a regular file")
    if resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseEvidenceError(f"{name} exceeds the evidence size cap")
    payload = resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseEvidenceError(f"{name} exceeds the evidence size cap")
    expected = _sha256(expected_sha256, name=f"{name}.expected_sha256")
    observed = hashlib.sha256(payload).hexdigest()
    if observed != expected:
        raise ReleaseEvidenceError(f"{name} file SHA-256 differs from its reviewed pin")
    return {
        "scope": "spec_root",
        "path": relative,
        "size_bytes": len(payload),
        "file_sha256": observed,
    }, payload


def _evidence_reference(
    *,
    source: str,
    path: str,
    expected_sha256: str,
    repository_root: Path,
    candidate_commit: str,
    spec_root: Path,
    name: str,
) -> tuple[dict[str, Any], bytes]:
    if source == "repository_commit":
        reference, payload = _blob_reference(
            repository_root, candidate_commit, path, expected_sha256, name=name
        )
        return {"scope": "repository_root", **reference}, payload
    if source == "spec_root":
        return _spec_root_blob(
            spec_root,
            path,
            expected_sha256,
            name=name,
            candidate=candidate_commit,
        )
    raise ReleaseEvidenceError(f"{name}.evidence_source is invalid")


def _embedded_self_hash(payload: bytes, *, field: str, name: str) -> str:
    if _IDENTIFIER_RE.fullmatch(field) is None:
        raise ReleaseEvidenceError(f"{name} embedded hash field is invalid")
    record = dict(_load_json_bytes(payload, name=name))
    observed = _sha256(record.pop(field, None), name=f"{name}.{field}")
    if canonical_json_sha256(record) != observed:
        raise ReleaseEvidenceError(f"{name} embedded self-hash does not reconstruct")
    return observed


def _candidate_first_parent(repository_root: Path, commit: str) -> str:
    result = _git(
        repository_root,
        ("rev-parse", f"{commit}^"),
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseEvidenceError("candidate commit has no resolvable first parent")
    assert isinstance(result.stdout, str)
    return _commit(result.stdout.strip(), name="candidate first parent")


def _validate_tracked_release_report(
    record: Mapping[str, Any], *, repository_root: Path, commit: str
) -> None:
    _exact_keys(
        record,
        {
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
        },
        name="tracked release gate report",
    )
    _timestamp(record.get("created_at_utc"))
    repository = _mapping(record.get("repository"), name="tracked release gate report repository")
    _exact_keys(
        repository,
        {"parent_commit", "content_commit", "worktree_clean"},
        name="tracked release gate report repository",
    )
    parent = _commit(
        repository.get("parent_commit"),
        name="tracked release gate report repository.parent_commit",
    )
    if parent != _candidate_first_parent(repository_root, commit):
        raise ReleaseEvidenceError(
            "tracked release gate report parent differs from candidate first parent"
        )
    policy = _mapping(record.get("policy"), name="tracked release gate report policy")
    _exact_keys(
        policy,
        {"path", "size_bytes", "file_sha256"},
        name="tracked release gate report policy",
    )
    policy_path = _safe_relative_path(
        policy.get("path"), name="tracked release gate report policy.path"
    )
    policy_size = policy.get("size_bytes")
    policy_sha256 = _sha256(
        policy.get("file_sha256"),
        name="tracked release gate report policy.file_sha256",
    )
    if not isinstance(policy_size, int) or isinstance(policy_size, bool) or policy_size < 0:
        raise ReleaseEvidenceError(
            "tracked release gate report policy.size_bytes must be a non-negative integer"
        )
    observed_policy, _policy_value = _candidate_policy(repository_root, commit)
    normalized_policy = {
        "path": policy_path,
        "size_bytes": policy_size,
        "file_sha256": policy_sha256,
    }
    if normalized_policy != observed_policy:
        raise ReleaseEvidenceError(
            "tracked release gate report policy differs from the candidate policy blob"
        )
    gates = _mapping(record.get("gates"), name="tracked release gate report gates")
    expected_gates = {
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
    }
    _exact_keys(gates, expected_gates, name="tracked release gate report gates")
    for gate_name, raw_gate in gates.items():
        gate = _mapping(raw_gate, name=f"tracked release gate report gates.{gate_name}")
        _exact_keys(
            gate,
            {"status", "evidence"},
            name=f"tracked release gate report gates.{gate_name}",
        )
        if gate.get("status") != "not_run" or gate.get("evidence") is not None:
            raise ReleaseEvidenceError(
                f"tracked release gate report gate {gate_name} must remain not_run"
            )
    external = _mapping(
        record.get("external_candidate_attestation"),
        name="tracked release gate report external attestation",
    )
    _exact_keys(
        external,
        {"status", "expected_root", "report_path"},
        name="tracked release gate report external attestation",
    )
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("record_kind") != "final_release_gate_report"
        or record.get("mode") != "tracked_precommit_report"
        or record.get("status") != "pending"
        or repository.get("content_commit") is not None
        or repository.get("worktree_clean") is not False
        or external.get("status") != "pending"
        or external.get("expected_root") != ".audit/release-attestations/<candidate_commit>/"
        or external.get("report_path") is not None
    ):
        raise ReleaseEvidenceError("tracked release_gate_report semantics are incompatible")


def _validate_opening_receipt(record: Mapping[str, Any]) -> None:
    _exact_keys(
        record,
        {
            "final_freeze_inventory_sha256",
            "frozen_artifact_set_sha256",
            "machine_record_sha256",
            "opened_at_utc",
            "record_kind",
            "record_sha256",
            "schema_version",
            "split_manifest_sha256",
            "status",
            "target_opening_number",
            "target_predictions_or_performance_accessed_at_receipt_time",
            "target_seal_id",
            "target_signals_materialized_at_receipt_time",
            "unlock_record_sha256",
        },
        name="opening_receipt",
    )
    _timestamp(record.get("opened_at_utc"))
    for field in (
        "target_seal_id",
        "split_manifest_sha256",
        "final_freeze_inventory_sha256",
        "frozen_artifact_set_sha256",
        "machine_record_sha256",
        "unlock_record_sha256",
    ):
        _sha256(record.get(field), name=f"opening_receipt.{field}")
    _exact_integer(
        record.get("target_opening_number"),
        name="opening_receipt.target_opening_number",
        expected=1,
    )
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("record_kind") != "confirmatory_target_opening_receipt"
        or record.get("status") != "unlock_consumed_before_materialization"
        or record.get("target_predictions_or_performance_accessed_at_receipt_time") is not False
        or record.get("target_signals_materialized_at_receipt_time") is not False
    ):
        raise ReleaseEvidenceError("opening_receipt confirmatory semantics are incompatible")


def _validate_locked_target_index(record: Mapping[str, Any]) -> None:
    _exact_keys(
        record,
        {
            "class_names",
            "evidence_status",
            "final_freeze_inventory_sha256",
            "frozen_artifact_set_sha256",
            "model_seed_result_count",
            "opening_receipt_file_sha256",
            "opening_receipt_record_sha256",
            "participant_count",
            "record_kind",
            "record_sha256",
            "results",
            "schema_version",
            "split_manifest_sha256",
            "status",
            "target_information_used_for_model_selection",
            "target_opening_number",
            "target_seal_id",
            "window_count",
        },
        name="locked_target_index",
    )
    for field in (
        "target_seal_id",
        "split_manifest_sha256",
        "final_freeze_inventory_sha256",
        "frozen_artifact_set_sha256",
        "opening_receipt_record_sha256",
        "opening_receipt_file_sha256",
    ):
        _sha256(record.get(field), name=f"locked_target_index.{field}")
    class_names = [
        _string(value, name="locked_target_index class name")
        for value in _sequence(record.get("class_names"), name="locked_target_index class_names")
    ]
    results = _sequence(record.get("results"), name="locked_target_index results")
    result_count = record.get("model_seed_result_count")
    participant_count = record.get("participant_count")
    window_count = record.get("window_count")
    if (
        class_names != ["mobility", "sitting", "standing"]
        or not isinstance(result_count, int)
        or isinstance(result_count, bool)
        or result_count < 1
        or result_count != len(results)
        or not isinstance(participant_count, int)
        or isinstance(participant_count, bool)
        or participant_count < 1
        or not isinstance(window_count, int)
        or isinstance(window_count, bool)
        or window_count < 1
    ):
        raise ReleaseEvidenceError("locked_target_index coverage is incompatible")
    seen_model_seeds: set[tuple[str, int]] = set()
    for index, raw_result in enumerate(results):
        result = _mapping(raw_result, name=f"locked_target_index.results[{index}]")
        _exact_keys(
            result,
            {
                "array_path",
                "array_sha256",
                "model_id",
                "record_file_sha256",
                "record_path",
                "record_sha256",
                "seed",
            },
            name=f"locked_target_index.results[{index}]",
        )
        model_id = _string(result.get("model_id"), name="target result model_id")
        if _IDENTIFIER_RE.fullmatch(model_id) is None:
            raise ReleaseEvidenceError("locked_target_index result model_id is invalid")
        seed = result.get("seed")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ReleaseEvidenceError("locked_target_index result seed is invalid")
        key = (model_id, seed)
        if key in seen_model_seeds:
            raise ReleaseEvidenceError("locked_target_index duplicates a model/seed result")
        seen_model_seeds.add(key)
        for field in ("array_sha256", "record_file_sha256", "record_sha256"):
            _sha256(result.get(field), name=f"locked_target_index result {field}")
        for field in ("array_path", "record_path"):
            path = _string(result.get(field), name=f"locked_target_index result {field}")
            if any(ord(character) < 32 or ord(character) == 127 for character in path):
                raise ReleaseEvidenceError(
                    f"locked_target_index result {field} contains control characters"
                )
    _exact_integer(
        record.get("target_opening_number"),
        name="locked_target_index.target_opening_number",
        expected=1,
    )
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("record_kind") != "locked_target_evaluation_index"
        or record.get("status") != "complete_create_only"
        or record.get("evidence_status") != "locked_confirmatory_target_opening_1"
        or record.get("target_information_used_for_model_selection") is not False
    ):
        raise ReleaseEvidenceError("locked_target_index confirmatory semantics are incompatible")


def _artifact_entries(
    spec: Mapping[str, Any], repository_root: Path, commit: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
    critical_records: dict[str, tuple[Mapping[str, Any], bytes, str]] = {}
    for index, value in enumerate(_sequence(spec.get("artifacts"), name="artifacts")):
        item = _mapping(value, name=f"artifacts[{index}]")
        _exact_keys(
            item,
            {
                "role",
                "path",
                "expected_file_sha256",
                "evidence_status",
                "validation_status",
                "required_for_release",
                "claim_scope",
                "embedded_self_hash_field",
            },
            name=f"artifacts[{index}]",
        )
        role = _string(item["role"], name=f"artifacts[{index}].role")
        if role not in ARTIFACT_ROLES or role in seen_roles:
            raise ReleaseEvidenceError(f"artifact role is unsupported or duplicated: {role}")
        seen_roles.add(role)
        path = _safe_relative_path(item["path"], name=f"artifacts[{index}].path")
        canonical_path = CANONICAL_REQUIRED_ROLE_PATHS.get(role)
        if canonical_path is not None and path != canonical_path:
            raise ReleaseEvidenceError(
                f"artifact role {role} must use canonical path {canonical_path}"
            )
        if path.casefold() in seen_paths:
            raise ReleaseEvidenceError(f"artifact path is duplicated: {path}")
        seen_paths.add(path.casefold())
        required = item["required_for_release"]
        if not isinstance(required, bool):
            raise ReleaseEvidenceError("required_for_release must be boolean")
        if role in REQUIRED_ROLES and required is not True:
            raise ReleaseEvidenceError(f"required role {role} must set required_for_release=true")
        validation_status = _string(
            item["validation_status"], name=f"artifacts[{index}].validation_status"
        )
        if validation_status not in _ARTIFACT_STATUSES:
            raise ReleaseEvidenceError(f"invalid artifact validation status: {validation_status}")
        reference, payload = _blob_reference(
            repository_root,
            commit,
            path,
            _string(item["expected_file_sha256"], name="expected_file_sha256"),
            name=f"artifact {role}",
        )
        embedded_field = _nullable_string(
            item["embedded_self_hash_field"], name="embedded_self_hash_field"
        )
        record_sha256 = (
            None
            if embedded_field is None
            else _embedded_self_hash(payload, field=embedded_field, name=f"artifact {role}")
        )
        if role in {"opening_receipt", "locked_target_index", "release_gate_report"}:
            record = _load_json_bytes(payload, name=f"artifact {role}")
            assert record_sha256 is not None
            critical_records[role] = (record, payload, record_sha256)
        entries.append(
            {
                "role": role,
                **reference,
                "record_sha256": record_sha256,
                "embedded_self_hash_field": embedded_field,
                "evidence_status": _string(item["evidence_status"], name="evidence_status"),
                "validation_status": validation_status,
                "required_for_release": required,
                "claim_scope": _nullable_string(item["claim_scope"], name="claim_scope"),
            }
        )
    missing = set(REQUIRED_ROLES) - seen_roles
    if missing:
        raise ReleaseEvidenceError(f"required release roles are missing: {sorted(missing)}")
    for role in ("opening_receipt", "locked_target_index", "release_gate_report"):
        entry = next(value for value in entries if value["role"] == role)
        if entry["embedded_self_hash_field"] != "record_sha256" or entry["record_sha256"] is None:
            raise ReleaseEvidenceError(f"{role} must declare and validate record_sha256")
    release_report, _release_report_payload, _release_report_hash = critical_records[
        "release_gate_report"
    ]
    _validate_tracked_release_report(
        release_report,
        repository_root=repository_root,
        commit=commit,
    )
    opening, opening_payload, opening_record_sha256 = critical_records["opening_receipt"]
    target_index, _target_index_payload, _target_index_record_sha256 = critical_records[
        "locked_target_index"
    ]
    _validate_opening_receipt(opening)
    _validate_locked_target_index(target_index)
    if (
        target_index.get("opening_receipt_record_sha256") != opening_record_sha256
        or target_index.get("opening_receipt_file_sha256")
        != hashlib.sha256(opening_payload).hexdigest()
        or any(
            target_index.get(field) != opening.get(field)
            for field in (
                "target_seal_id",
                "split_manifest_sha256",
                "final_freeze_inventory_sha256",
                "frozen_artifact_set_sha256",
            )
        )
    ):
        raise ReleaseEvidenceError(
            "locked_target_index does not cross-bind the one-time opening receipt"
        )
    machine_payload = _git_blob(
        repository_root,
        commit,
        "results/environment/confirmatory_machine_20260824.json",
        name="confirmatory machine record",
    )
    machine = _load_json_bytes(machine_payload, name="confirmatory machine record")
    machine_record_sha256 = _embedded_self_hash(
        machine_payload,
        field="record_sha256",
        name="confirmatory machine record",
    )
    if (
        machine.get("schema_version") != "1.0.0"
        or machine.get("record_kind") != "confirmatory_machine_environment"
        or machine.get("status") != "observed_read_only_cuda_validated"
        or opening.get("machine_record_sha256") != machine_record_sha256
    ):
        raise ReleaseEvidenceError("opening_receipt machine provenance does not cross-bind")
    unlock_payload = _git_blob(
        repository_root,
        commit,
        "results/protocol/final_evaluation_unlock_v1.json",
        name="confirmatory unlock record",
    )
    unlock = _load_json_bytes(unlock_payload, name="confirmatory unlock record")
    _embedded_self_hash(
        unlock_payload,
        field="record_sha256",
        name="confirmatory unlock record",
    )
    _exact_integer(
        unlock.get("target_opening_number"),
        name="confirmatory unlock target_opening_number",
        expected=1,
    )
    _exact_integer(
        unlock.get("maximum_target_openings"),
        name="confirmatory unlock maximum_target_openings",
        expected=1,
    )
    if (
        unlock.get("schema_version") != "1.0.0"
        or unlock.get("gate") != "final_evaluation_unlock"
        or unlock.get("status") != "approved"
        or unlock.get("target_performance_previously_accessed") is not False
        or unlock.get("target_subject_or_window_records_loaded_at_approval") is not False
        or unlock.get("target_predictions_or_performance_accessed_at_approval") is not False
        or unlock.get("source_models_and_calibrators_frozen") is not True
        or opening.get("unlock_record_sha256") != canonical_json_sha256(unlock)
        or any(
            unlock.get(field) != opening.get(field)
            for field in (
                "target_seal_id",
                "split_manifest_sha256",
                "final_freeze_inventory_sha256",
                "frozen_artifact_set_sha256",
            )
        )
    ):
        raise ReleaseEvidenceError("opening_receipt unlock provenance does not cross-bind")
    freeze_payload = _git_blob(
        repository_root,
        commit,
        CANONICAL_REQUIRED_ROLE_PATHS["source_artifact_freeze"],
        name="source artifact freeze",
    )
    freeze = _load_json_bytes(freeze_payload, name="source artifact freeze")
    split_payload = _git_blob(
        repository_root,
        commit,
        CANONICAL_REQUIRED_ROLE_PATHS["split_manifest_and_audit"],
        name="split audit",
    )
    split_audit = _load_json_bytes(split_payload, name="split audit")
    target_state = _mapping(freeze.get("target_state"), name="source freeze target_state")
    _exact_keys(
        target_state,
        {
            "target_predictions_or_performance_accessed",
            "target_subject_or_window_records_loaded",
            "unlock_record",
        },
        name="source freeze target_state",
    )
    if (
        freeze.get("schema_version") != "1.0.0"
        or freeze.get("record_kind") != "final_source_artifact_freeze"
        or freeze.get("status") != "frozen_before_target_unlock"
        or target_state.get("target_predictions_or_performance_accessed") is not False
        or target_state.get("target_subject_or_window_records_loaded") is not False
        or target_state.get("unlock_record") is not None
        or freeze.get("target_seal_id") != opening.get("target_seal_id")
        or freeze.get("split_manifest_sha256") != opening.get("split_manifest_sha256")
        or freeze.get("inventory_sha256") != opening.get("final_freeze_inventory_sha256")
        or freeze.get("frozen_artifact_set_sha256") != opening.get("frozen_artifact_set_sha256")
        or split_audit.get("schema_version") != "1.0.0"
        or split_audit.get("status") != "pass_conditional_released_block"
        or split_audit.get("valid") is not True
        or split_audit.get("target_performance_or_prediction_accessed") is not False
        or split_audit.get("split_manifest_sha256") != opening.get("split_manifest_sha256")
    ):
        raise ReleaseEvidenceError("opening_receipt freeze or split provenance does not cross-bind")
    return sorted(entries, key=lambda value: str(value["role"]))


def _validate_gate_record_common(
    payload: bytes,
    *,
    name: str,
    expected_kind: str,
    expected_status: str,
    commit: str,
) -> tuple[Mapping[str, Any], str]:
    record = _load_json_bytes(payload, name=name)
    record_hash = _embedded_self_hash(payload, field="record_sha256", name=name)
    if record.get("schema_version") != "1.0.0":
        raise ReleaseEvidenceError(f"{name} schema version is incompatible")
    if record.get("record_kind") != expected_kind:
        raise ReleaseEvidenceError(f"{name} record kind is incompatible")
    if record.get("status") != expected_status:
        raise ReleaseEvidenceError(f"{name} status differs from the generation spec")
    if record.get("candidate_commit") != commit:
        raise ReleaseEvidenceError(f"{name} is bound to another candidate commit")
    return record, record_hash


def _validate_non_ci_gate_record(
    payload: bytes,
    *,
    gate_name: str,
    status: str,
    commit: str,
    repository_root: Path,
    spec_root: Path,
    policy_reference: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[str, str]:
    if gate_name == "staged_index_scan":
        record = _load_json_bytes(payload, name="staged_index_scan gate evidence")
        _exact_keys(record, STAGED_INDEX_RECORD_KEYS, name="staged_index_scan gate evidence")
        record_hash = _embedded_self_hash(
            payload, field="record_sha256", name="staged_index_scan gate evidence"
        )
        parent_result = _git(repository_root, ("rev-parse", f"{commit}^"), text=True, check=False)
        if parent_result.returncode != 0:
            raise ReleaseEvidenceError("candidate commit has no resolvable first parent")
        assert isinstance(parent_result.stdout, str)
        parent = _commit(parent_result.stdout.strip(), name="candidate first parent")
        entry_count = record.get("index_entry_count")
        staged_change_count = record.get("staged_change_count")
        unique_blob_count = record.get("unique_blob_count")
        digest = record.get("index_entry_index_sha256")
        violations = _sequence(record.get("violations"), name="staged_index_scan violations")
        _timestamp(record.get("created_at_utc"))
        _exact_integer(
            record.get("replace_ref_count"),
            name="staged_index_scan.replace_ref_count",
            expected=0,
        )
        candidate_count, candidate_digest = _candidate_tree_index(repository_root, commit)
        if (
            record.get("schema_version") != "1.0.0"
            or record.get("record_kind") != NON_CI_GATE_RECORD_KINDS[gate_name]
            or record.get("status") != status
            or record.get("head_commit") != parent
            or record.get("scan_scope") != "exact_git_index"
            or record.get("policy") != policy_reference
            or record.get("raw_worktree_paths_opened") is not False
            or record.get("git_objects_written") is not False
            or record.get("grafts_file_present") is not False
            or (status == "pass" and violations)
            or (status == "fail" and not violations)
            or not isinstance(entry_count, int)
            or isinstance(entry_count, bool)
            or entry_count < 1
            or not isinstance(staged_change_count, int)
            or isinstance(staged_change_count, bool)
            or staged_change_count < 1
            or not isinstance(unique_blob_count, int)
            or isinstance(unique_blob_count, bool)
            or unique_blob_count < 1
            or unique_blob_count > entry_count
            or entry_count != candidate_count
            or digest != candidate_digest
        ):
            raise ReleaseEvidenceError(
                "staged_index_scan does not prove the reviewed candidate tree and parent"
            )
        return NON_CI_GATE_RECORD_KINDS[gate_name], record_hash

    if gate_name in QUALITY_GATE_NAMES:
        record, record_hash = _validate_gate_record_common(
            payload,
            name=f"{gate_name} gate evidence",
            expected_kind=QUALITY_EVIDENCE_KIND,
            expected_status=status,
            commit=commit,
        )
        _exact_keys(record, QUALITY_EVIDENCE_KEYS, name=f"{gate_name} gate evidence")
        declared_gates = [
            _string(value, name=f"{gate_name} gate coverage")
            for value in _sequence(record.get("gates"), name=f"{gate_name} gate coverage")
        ]
        covered_gates = set(declared_gates)
        expected_gates = set(QUALITY_GATE_NAMES) | {"ci_quality_proxy"}
        if len(covered_gates) != len(declared_gates):
            raise ReleaseEvidenceError(f"{gate_name} quality evidence duplicates gate coverage")
        statuses = _mapping(record.get("gate_statuses"), name=f"{gate_name} gate_statuses")
        gate_evidence = _mapping(record.get("gate_evidence"), name=f"{gate_name} gate_evidence")
        executed_gates = set(QUALITY_GATE_NAMES)
        tools = _mapping(record.get("tool_versions"), name=f"{gate_name} tool_versions")
        machine = _mapping(record.get("machine"), name=f"{gate_name} machine")
        if set(statuses) != covered_gates or any(
            value not in {"pass", "fail"} for value in statuses.values()
        ):
            raise ReleaseEvidenceError(
                f"{gate_name} quality evidence coverage and status mapping differ"
            )
        derived_status = "pass" if all(value == "pass" for value in statuses.values()) else "fail"
        if derived_status != record.get("status"):
            raise ReleaseEvidenceError(
                f"{gate_name} quality evidence overall status is inconsistent"
            )
        if (
            covered_gates != expected_gates
            or gate_name not in covered_gates
            or statuses.get(gate_name) != status
            or record.get("ci_evidence_scope") != QUALITY_EVIDENCE_SCOPE
            or record.get("external_completed_ci_required_for_release_inventory") is not True
            or record.get("scope")
            != "upstream synthetic-validation matrix and preceding release-security steps"
            or set(gate_evidence) != executed_gates
            or set(tools) != {"python", "uv", "pytest", "ruff", "mypy"}
            or any(not isinstance(value, str) or not value for value in tools.values())
            or not str(tools.get("uv", "")).startswith("uv 0.11.29")
            or set(machine) != {"runner_os", "architecture", "platform"}
            or machine.get("runner_os") != "Linux"
            or any(not isinstance(machine.get(key), str) or not machine.get(key) for key in machine)
        ):
            raise ReleaseEvidenceError(
                f"{gate_name} quality evidence does not cover the cited gate and status"
            )
        _timestamp(record.get("created_at_utc"))
        for executed_gate in sorted(executed_gates):
            item = _mapping(
                gate_evidence[executed_gate], name=f"{gate_name}.{executed_gate} provenance"
            )
            _exact_keys(
                item,
                {
                    "exit_code",
                    "started_at_utc",
                    "completed_at_utc",
                    "log_path",
                    "log_size_bytes",
                    "log_sha256",
                },
                name=f"{gate_name}.{executed_gate} provenance",
            )
            exit_code = item.get("exit_code")
            started = _timestamp(item.get("started_at_utc"))
            completed = _timestamp(item.get("completed_at_utc"))
            log_path = _string(item.get("log_path"), name=f"{executed_gate} log path")
            log_pure = PurePosixPath(log_path)
            log_size = item.get("log_size_bytes")
            if (
                datetime.fromisoformat(completed[:-1] + "+00:00")
                < datetime.fromisoformat(started[:-1] + "+00:00")
                or not isinstance(exit_code, int)
                or isinstance(exit_code, bool)
                or (statuses[executed_gate] == "pass" and exit_code != 0)
                or (statuses[executed_gate] == "fail" and exit_code == 0)
                or log_pure.is_absolute()
                or log_pure.as_posix() != log_path
                or "\\" in log_path
                or any(ord(character) < 32 or ord(character) == 127 for character in log_path)
                or any(part in {".", ".."} for part in log_pure.parts)
                or not log_path.startswith(f".audit/release-attestations/{commit}/")
                or not isinstance(log_size, int)
                or isinstance(log_size, bool)
                or log_size < 0
                or not isinstance(item.get("log_sha256"), str)
                or _SHA256_RE.fullmatch(item["log_sha256"]) is None
            ):
                raise ReleaseEvidenceError(
                    f"{gate_name}.{executed_gate} quality provenance is inconsistent"
                )
            log_source = spec_root / Path(log_path)
            if log_source.is_symlink():
                raise ReleaseEvidenceError(f"{gate_name}.{executed_gate} log may not be a symlink")
            log_resolved = log_source.resolve(strict=True)
            try:
                log_resolved.relative_to(spec_root)
            except ValueError as exc:
                raise ReleaseEvidenceError(
                    f"{gate_name}.{executed_gate} log escapes spec_root"
                ) from exc
            if (
                not log_resolved.is_file()
                or log_resolved.stat().st_size > MAX_EVIDENCE_BYTES
                or log_resolved.stat().st_size != log_size
            ):
                raise ReleaseEvidenceError(
                    f"{gate_name}.{executed_gate} log exceeds the evidence size cap or size pin"
                )
            log_payload = log_resolved.read_bytes()
            if len(log_payload) != log_size or hashlib.sha256(log_payload).hexdigest() != item.get(
                "log_sha256"
            ):
                raise ReleaseEvidenceError(
                    f"{gate_name}.{executed_gate} log bytes differ from provenance"
                )
        return QUALITY_EVIDENCE_KIND, record_hash

    expected_kind = NON_CI_GATE_RECORD_KINDS.get(gate_name)
    if expected_kind is None:
        raise ReleaseEvidenceError(f"no semantic validator exists for gate {gate_name}")
    record, record_hash = _validate_gate_record_common(
        payload,
        name=f"{gate_name} gate evidence",
        expected_kind=expected_kind,
        expected_status=status,
        commit=commit,
    )
    _exact_keys(
        record,
        NON_CI_GATE_RECORD_KEYS[gate_name],
        name=f"{gate_name} gate evidence",
    )
    _timestamp(record.get("created_at_utc"))
    if record.get("policy") != policy_reference:
        raise ReleaseEvidenceError(f"{gate_name} evidence used another release policy")
    if gate_name == "secret_scan":
        finding_count = record.get("finding_count")
        config = _mapping(record.get("config"), name="secret_scan config")
        ignore = _mapping(record.get("ignore"), name="secret_scan ignore")
        raw_report = _mapping(record.get("raw_report"), name="secret_scan raw_report")
        _exact_keys(config, {"path", "size_bytes", "file_sha256"}, name="secret_scan config")
        _exact_keys(ignore, {"path", "size_bytes", "file_sha256"}, name="secret_scan ignore")
        _exact_keys(
            raw_report,
            {"basename", "size_bytes", "file_sha256"},
            name="secret_scan raw_report",
        )
        raw_basename = _string(raw_report.get("basename"), name="secret_scan raw report basename")
        raw_size = raw_report.get("size_bytes")
        if (
            PurePosixPath(raw_basename).name != raw_basename
            or "\\" in raw_basename
            or ":" in raw_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_basename)
            or not isinstance(raw_size, int)
            or isinstance(raw_size, bool)
            or raw_size < 0
        ):
            raise ReleaseEvidenceError("secret_scan raw report reference is invalid")
        _sha256(raw_report.get("file_sha256"), name="secret_scan raw report file_sha256")
        observed_config, _payload = _blob_reference(
            repository_root,
            commit,
            _string(config.get("path"), name="secret_scan config path"),
            _string(config.get("file_sha256"), name="secret_scan config SHA-256"),
            name="Gitleaks configuration",
        )
        observed_ignore, ignore_payload = _blob_reference(
            repository_root,
            commit,
            _string(ignore.get("path"), name="secret_scan ignore path"),
            _string(ignore.get("file_sha256"), name="secret_scan ignore SHA-256"),
            name="Gitleaks ignore file",
        )
        _validate_gitleaks_ignore_entries(policy, ignore_payload)
        gitleaks_policy = _mapping(policy.get("gitleaks"), name="candidate Gitleaks policy")
        try:
            ref_metrics, ref_violations = inspect_git_ref_state(
                repository_root, policy, candidate=commit
            )
        except ReleaseGateError as exc:
            raise ReleaseEvidenceError("secret_scan Git ref policy validation failed") from exc
        gitleaks_exit = record.get("gitleaks_exit_code")
        _exact_integer(gitleaks_exit, name="secret_scan.gitleaks_exit_code", minimum=0)
        execution_status = record.get("execution_status")
        execution_consistent = (
            status == "pass"
            and gitleaks_exit == 0
            and execution_status == "completed"
            and finding_count == 0
        ) or (
            status == "fail"
            and (
                (
                    gitleaks_exit == 1
                    and execution_status == "completed"
                    and isinstance(finding_count, int)
                    and finding_count > 0
                )
                or execution_status == "operational_error"
            )
        )
        if (
            not isinstance(finding_count, int)
            or isinstance(finding_count, bool)
            or finding_count < 0
            or record.get("scan_scope") != SECRET_SCAN_SCOPE
            or ref_violations
            or record.get("ref_metadata_index_sha256") != ref_metrics["ref_metadata_index_sha256"]
            or record.get("validated_annotated_tag_count")
            != ref_metrics["validated_annotated_tag_count"]
            or record.get("gitleaks_version") != gitleaks_policy.get("version")
            or gitleaks_policy.get("required_scope") != SECRET_SCAN_SCOPE
            or config.get("path") != gitleaks_policy.get("config_path")
            or config.get("file_sha256") != gitleaks_policy.get("config_sha256")
            or ignore.get("path") != gitleaks_policy.get("ignore_path")
            or not isinstance(gitleaks_exit, int)
            or isinstance(gitleaks_exit, bool)
            or not execution_consistent
            or observed_config != dict(config)
            or observed_ignore != dict(ignore)
        ):
            raise ReleaseEvidenceError("secret_scan finding count or scope is inconsistent")
        return expected_kind, record_hash

    violations = _sequence(record.get("violations"), name=f"{gate_name} evidence violations")
    if (status == "pass" and violations) or (status == "fail" and not violations):
        raise ReleaseEvidenceError(f"{gate_name} violations are inconsistent with its status")
    if gate_name == "tracked_file_scan":
        commit_count = record.get("commit_count")
        unique_blob_count = record.get("unique_blob_count")
        observation_count = record.get("tree_entry_observation_count")
        try:
            ref_metrics, ref_violations = inspect_git_ref_state(
                repository_root, policy, candidate=commit
            )
        except ReleaseGateError as exc:
            raise ReleaseEvidenceError(
                "tracked_file_scan Git ref policy validation failed"
            ) from exc
        _exact_integer(
            record.get("replace_ref_count"),
            name="tracked_file_scan.replace_ref_count",
            expected=0,
        )
        for metric in (
            "ref_count",
            "tag_ref_count",
            "validated_annotated_tag_count",
            "invalid_ref_target_count",
        ):
            _exact_integer(
                record.get(metric),
                name=f"tracked_file_scan.{metric}",
                expected=int(ref_metrics[metric]),
            )
        if (
            record.get("scan_scope") != "candidate_tree_and_complete_reachable_git_history"
            or record.get("raw_worktree_paths_opened") is not False
            or record.get("shallow_repository") is not False
            or record.get("grafts_file_present") is not False
            or ref_violations
            or not isinstance(commit_count, int)
            or isinstance(commit_count, bool)
            or commit_count < 1
            or not isinstance(unique_blob_count, int)
            or isinstance(unique_blob_count, bool)
            or unique_blob_count < 1
            or not isinstance(observation_count, int)
            or isinstance(observation_count, bool)
            or observation_count < unique_blob_count
            or not isinstance(record.get("ref_count"), int)
            or isinstance(record.get("ref_count"), bool)
            or record.get("ref_count", 0) < 1
            or any(
                record.get(metric) != ref_metrics[metric]
                for metric in (
                    "ref_count",
                    "tag_ref_count",
                    "validated_annotated_tag_count",
                    "invalid_ref_target_count",
                    "ref_metadata_index_sha256",
                )
            )
        ):
            raise ReleaseEvidenceError("tracked_file_scan scope is incomplete or unsafe")
        _sha256(
            record.get("scanned_object_index_sha256"),
            name="tracked_file_scan scanned_object_index_sha256",
        )
    if gate_name == "license_audit":
        package_count = record.get("package_count")
        inventory = _mapping(record.get("inventory"), name="license_audit inventory")
        _exact_keys(
            inventory,
            {"basename", "size_bytes", "file_sha256"},
            name="license_audit inventory",
        )
        inventory_basename = _string(
            inventory.get("basename"), name="license_audit inventory basename"
        )
        inventory_size = inventory.get("size_bytes")
        if (
            record.get("pip_licenses_version") != "5.5.5"
            or not isinstance(package_count, int)
            or isinstance(package_count, bool)
            or package_count < 1
            or PurePosixPath(inventory_basename).name != inventory_basename
            or "\\" in inventory_basename
            or ":" in inventory_basename
            or any(ord(character) < 32 or ord(character) == 127 for character in inventory_basename)
            or not isinstance(inventory_size, int)
            or isinstance(inventory_size, bool)
            or inventory_size < 0
        ):
            raise ReleaseEvidenceError("license_audit inventory coverage is incomplete")
        _sha256(inventory.get("file_sha256"), name="license_audit inventory file_sha256")
        _sha256(
            record.get("normalized_inventory_sha256"),
            name="license_audit normalized_inventory_sha256",
        )
        try:
            validate_license_audit_semantics(
                record,
                root=repository_root,
                candidate=commit,
                policy=policy,
            )
        except ReleaseGateError as exc:
            raise ReleaseEvidenceError(
                "license_audit exceptions differ from the release policy"
            ) from exc
    return expected_kind, record_hash


def _validate_file_references(
    value: Any, *, expected_names: set[str], name: str
) -> dict[str, dict[str, Any]]:
    references = _mapping(value, name=name)
    _exact_keys(references, expected_names, name=name)
    normalized: dict[str, dict[str, Any]] = {}
    seen_basenames: set[str] = set()
    for key, raw_reference in references.items():
        reference = _mapping(raw_reference, name=f"{name}.{key}")
        _exact_keys(
            reference,
            {"basename", "size_bytes", "file_sha256"},
            name=f"{name}.{key}",
        )
        basename = _string(reference.get("basename"), name=f"{name}.{key}.basename")
        size = reference.get("size_bytes")
        if (
            PurePosixPath(basename).name != basename
            or "\\" in basename
            or ":" in basename
            or any(ord(character) < 32 or ord(character) == 127 for character in basename)
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
        ):
            raise ReleaseEvidenceError(f"{name}.{key} reference is invalid")
        if basename.casefold() in seen_basenames:
            raise ReleaseEvidenceError(f"{name} basenames must be distinct")
        seen_basenames.add(basename.casefold())
        normalized[key] = {
            "basename": basename,
            "size_bytes": size,
            "file_sha256": _sha256(reference.get("file_sha256"), name=f"{name}.{key}.file_sha256"),
        }
    return normalized


def _validate_external_pinned_files(
    spec_root: Path,
    references: Mapping[str, Mapping[str, Any]],
    *,
    name: str,
    max_bytes: int = MAX_EVIDENCE_BYTES,
) -> dict[str, bytes]:
    payloads: dict[str, bytes] = {}
    for key, reference in references.items():
        source = spec_root / str(reference["basename"])
        if source.is_symlink():
            raise ReleaseEvidenceError(f"{name}.{key} may not be a symlink")
        try:
            resolved = source.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise ReleaseEvidenceError(f"{name}.{key} is not retained at spec_root") from exc
        try:
            resolved.relative_to(spec_root)
        except ValueError as exc:
            raise ReleaseEvidenceError(f"{name}.{key} escapes spec_root") from exc
        if not resolved.is_file() or resolved.stat().st_size > max_bytes:
            raise ReleaseEvidenceError(f"{name}.{key} is not retained within the evidence cap")
        payload = resolved.read_bytes()
        if (
            len(payload) > max_bytes
            or source.is_symlink()
            or not resolved.is_file()
            or resolved.stat().st_size != len(payload)
            or len(payload) != reference["size_bytes"]
            or hashlib.sha256(payload).hexdigest() != reference["file_sha256"]
        ):
            raise ReleaseEvidenceError(f"{name}.{key} retained bytes differ from the pin")
        payloads[key] = payload
    return payloads


def _validate_external_api_response_files(
    spec_root: Path, references: Mapping[str, Mapping[str, Any]], *, name: str
) -> dict[str, Mapping[str, Any]]:
    payloads = _validate_external_pinned_files(spec_root, references, name=name)
    return {
        key: _load_json_bytes(payload, name=f"{name}.{key} retained response")
        for key, payload in payloads.items()
    }


def _validate_ci_record(
    payload: bytes, *, commit: str, expected_status: str, spec_root: Path
) -> tuple[str, str, str, dict[str, str], dict[str, str], set[str]]:
    record = _load_json_bytes(payload, name="CI evidence")
    _exact_keys(
        record,
        {
            "schema_version",
            "record_kind",
            "status",
            "candidate_commit",
            "workflow_status",
            "conclusion",
            "ci_evidence_scope",
            "event",
            "head_branch",
            "head_sha",
            "repository",
            "workflow_path",
            "run_id",
            "run_attempt",
            "workflow_url",
            "artifact_name",
            "artifact_id",
            "artifact_expired",
            "artifact_workflow_run_id",
            "artifact_workflow_run_head_sha",
            "artifact_api_url",
            "artifacts_list_api_url",
            "artifact_archive_download_url",
            "artifact_size_in_bytes",
            "artifact_digest",
            "downloaded_artifact_archive_sha256",
            "downloaded_artifact_archive",
            "artifact_member_sha256s",
            "release_gate_file_sha256s",
            "api_response_files",
            "queried_at_utc",
            "record_sha256",
        },
        name="CI evidence",
    )
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("record_kind") != "ci_run_evidence"
        or record.get("status") != expected_status
        or record.get("candidate_commit") != commit
        or record.get("ci_evidence_scope") != COMPLETED_CI_EVIDENCE_SCOPE
        or record.get("workflow_status") != "completed"
        or record.get("event") != "push"
        or record.get("head_branch") != "main"
        or record.get("head_sha") != commit
        or record.get("workflow_path") != ".github/workflows/ci.yml"
    ):
        raise ReleaseEvidenceError("CI evidence has incompatible status or commit")
    repository = _github_repository(record.get("repository"), name="CI evidence repository")
    run_id = record.get("run_id")
    run_attempt = record.get("run_attempt")
    artifact_id = record.get("artifact_id")
    workflow_url = record.get("workflow_url")
    match = (
        re.fullmatch(
            r"https://github\.com/(?P<repository>[^/]+/[^/]+)/actions/runs/(?P<run>[1-9][0-9]*)",
            workflow_url,
        )
        if isinstance(workflow_url, str)
        else None
    )
    if match is None:
        raise ReleaseEvidenceError("CI evidence lacks a canonical completed workflow URL")
    release_hashes = _mapping(
        record.get("release_gate_file_sha256s"), name="CI release gate file hashes"
    )
    expected_release_files = {
        "repository_scan",
        "secret_scan",
        "license_audit",
        "quality_gates",
        "ci_bundle_validation",
    }
    if set(release_hashes) != expected_release_files or any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in release_hashes.values()
    ):
        raise ReleaseEvidenceError("CI release gate file hashes are incomplete")
    if (
        match.group("repository").casefold() != repository.casefold()
        or not isinstance(run_id, int)
        or isinstance(run_id, bool)
        or run_id < 1
        or int(match.group("run")) != run_id
        or not isinstance(run_attempt, int)
        or isinstance(run_attempt, bool)
        or run_attempt < 1
        or record.get("artifact_name") != f"release-security-{commit}"
        or not isinstance(artifact_id, int)
        or isinstance(artifact_id, bool)
        or artifact_id < 1
        or record.get("artifact_expired") is not False
        or not isinstance(record.get("artifact_workflow_run_id"), int)
        or isinstance(record.get("artifact_workflow_run_id"), bool)
        or record.get("artifact_workflow_run_id", 0) < 1
        or record.get("artifact_workflow_run_id") != run_id
        or record.get("artifact_workflow_run_head_sha") != commit
        or record.get("artifact_api_url")
        != f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}"
        or record.get("artifacts_list_api_url")
        != (
            f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts"
            f"?name=release-security-{commit}"
        )
        or record.get("artifact_archive_download_url")
        != f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}/zip"
        or not isinstance(record.get("artifact_size_in_bytes"), int)
        or isinstance(record.get("artifact_size_in_bytes"), bool)
        or record.get("artifact_size_in_bytes", 0) < 1
        or not isinstance(record.get("artifact_digest"), str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", record["artifact_digest"]) is None
        or not isinstance(record.get("downloaded_artifact_archive_sha256"), str)
        or _SHA256_RE.fullmatch(record["downloaded_artifact_archive_sha256"]) is None
        or record.get("artifact_digest")
        != f"sha256:{record.get('downloaded_artifact_archive_sha256')}"
    ):
        raise ReleaseEvidenceError("CI run identity is inconsistent")
    member_hashes = _mapping(
        record.get("artifact_member_sha256s"), name="CI artifact member hashes"
    )
    if set(member_hashes) != EXPECTED_ARTIFACT_FILES or any(
        not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None
        for value in member_hashes.values()
    ):
        raise ReleaseEvidenceError("CI artifact member hashes are incomplete")
    component_member_names = {
        "repository_scan": "repository_scan.json",
        "secret_scan": "secret_scan.json",
        "license_audit": "license_audit.json",
        "quality_gates": "quality_gates.json",
        "ci_bundle_validation": "ci_bundle_validation.json",
    }
    if any(
        release_hashes[name] != member_hashes[basename]
        for name, basename in component_member_names.items()
    ):
        raise ReleaseEvidenceError("CI component hashes differ from archive member hashes")
    api_references = _validate_file_references(
        record.get("api_response_files"),
        expected_names={"workflow_run", "run_artifacts"},
        name="CI API response files",
    )
    api_responses = _validate_external_api_response_files(
        spec_root,
        api_references,
        name="CI API response files",
    )
    workflow_response = api_responses["workflow_run"]
    workflow_repository = _mapping(
        workflow_response.get("repository"), name="retained workflow repository"
    )
    retained_run_id = _exact_integer(
        workflow_response.get("id"), name="retained workflow run id", minimum=1
    )
    retained_run_attempt = _exact_integer(
        workflow_response.get("run_attempt"),
        name="retained workflow run attempt",
        minimum=1,
    )
    if (
        retained_run_id != run_id
        or retained_run_attempt != run_attempt
        or workflow_response.get("status") != record.get("workflow_status")
        or workflow_response.get("conclusion") != record.get("conclusion")
        or workflow_response.get("event") != record.get("event")
        or workflow_response.get("head_branch") != record.get("head_branch")
        or workflow_response.get("head_sha") != record.get("head_sha")
        or workflow_response.get("path") != record.get("workflow_path")
        or workflow_response.get("html_url") != workflow_url
        or workflow_repository.get("full_name") != repository
    ):
        raise ReleaseEvidenceError("retained workflow-run API response differs from CI evidence")
    artifact_list_response = api_responses["run_artifacts"]
    retained_artifacts = artifact_list_response.get("artifacts")
    _exact_integer(
        artifact_list_response.get("total_count"),
        name="retained artifact total_count",
        expected=1,
    )
    if not isinstance(retained_artifacts, list) or len(retained_artifacts) != 1:
        raise ReleaseEvidenceError("retained run-artifacts response is not an exact singleton")
    retained_artifact = _mapping(retained_artifacts[0], name="retained CI artifact")
    retained_workflow = _mapping(
        retained_artifact.get("workflow_run"), name="retained artifact workflow_run"
    )
    retained_artifact_id = _exact_integer(
        retained_artifact.get("id"), name="retained artifact id", minimum=1
    )
    retained_artifact_size = _exact_integer(
        retained_artifact.get("size_in_bytes"),
        name="retained artifact size_in_bytes",
        minimum=1,
    )
    retained_artifact_run_id = _exact_integer(
        retained_workflow.get("id"), name="retained artifact workflow run id", minimum=1
    )
    if (
        retained_artifact_id != artifact_id
        or retained_artifact.get("name") != record.get("artifact_name")
        or retained_artifact.get("expired") is not False
        or retained_artifact.get("url") != record.get("artifact_api_url")
        or retained_artifact.get("archive_download_url")
        != record.get("artifact_archive_download_url")
        or retained_artifact_size != record.get("artifact_size_in_bytes")
        or retained_artifact.get("digest") != record.get("artifact_digest")
        or retained_artifact_run_id != record.get("artifact_workflow_run_id")
        or retained_workflow.get("head_sha") != record.get("artifact_workflow_run_head_sha")
    ):
        raise ReleaseEvidenceError("retained run-artifacts API response differs from CI evidence")
    archive_references = _validate_file_references(
        {"archive": record.get("downloaded_artifact_archive")},
        expected_names={"archive"},
        name="downloaded CI artifact archive",
    )
    if archive_references["archive"]["basename"].casefold() in {
        reference["basename"].casefold() for reference in api_references.values()
    }:
        raise ReleaseEvidenceError("downloaded CI artifact basename collides with API evidence")
    archive_payload = _validate_external_pinned_files(
        spec_root,
        archive_references,
        name="downloaded CI artifact archive",
        max_bytes=MAX_ARCHIVE_BYTES,
    )["archive"]
    try:
        archive_validation = validate_ci_archive_payload(
            archive_payload,
            candidate_commit=commit,
        )
    except GitHubEvidenceError as exc:
        raise ReleaseEvidenceError(f"retained CI artifact archive is invalid: {exc}") from exc
    if (
        archive_validation["archive_sha256"] != record.get("downloaded_artifact_archive_sha256")
        or archive_validation["size_bytes"] != record.get("artifact_size_in_bytes")
        or archive_validation["member_file_sha256s"] != dict(member_hashes)
    ):
        raise ReleaseEvidenceError("retained CI artifact archive differs from CI evidence")
    _timestamp(record.get("queried_at_utc"))
    conclusion = record.get("conclusion")
    if (expected_status == "pass" and conclusion != "success") or (
        expected_status == "fail" and conclusion == "success"
    ):
        raise ReleaseEvidenceError("CI evidence conclusion is inconsistent with its status")
    record_hash = _embedded_self_hash(payload, field="record_sha256", name="CI evidence")
    return (
        "ci_run_evidence",
        record_hash,
        repository,
        dict(release_hashes),
        dict(member_hashes),
        {
            *(reference["basename"].casefold() for reference in api_references.values()),
            archive_references["archive"]["basename"].casefold(),
        },
    )


def _derived_gate_tool_version(gate_name: str, payload: bytes, *, commit: str) -> str:
    record = _load_json_bytes(payload, name=f"{gate_name} tool-version evidence")
    if gate_name in QUALITY_GATE_NAMES:
        tools = _mapping(record.get("tool_versions"), name=f"{gate_name} tool versions")
        tool_key = {
            "tests": "pytest",
            "lint": "ruff",
            "format": "ruff",
            "types": "mypy",
            "configuration": "pytest",
        }.get(gate_name)
        if tool_key is not None:
            return _string(tools.get(tool_key), name=f"{gate_name} tool version")
        return f"inclusive-shift-har@{commit}"
    if gate_name in {"staged_index_scan", "tracked_file_scan"}:
        return f"inclusive-shift-har@{commit}"
    if gate_name == "secret_scan":
        return f"gitleaks {_string(record.get('gitleaks_version'), name='Gitleaks version')}"
    if gate_name == "license_audit":
        return (
            "pip-licenses "
            f"{_string(record.get('pip_licenses_version'), name='pip-licenses version')}"
        )
    if gate_name == "ci":
        run_id = record.get("run_id")
        attempt = record.get("run_attempt")
        if (
            not isinstance(run_id, int)
            or isinstance(run_id, bool)
            or not isinstance(attempt, int)
            or isinstance(attempt, bool)
        ):
            raise ReleaseEvidenceError("CI run tool-version provenance is invalid")
        return f"github-actions run {run_id} attempt {attempt}"
    raise ReleaseEvidenceError(f"no tool-version derivation exists for gate {gate_name}")


def _validate_mirrored_ci_bundle(
    *,
    spec_root: Path,
    repository_root: Path,
    commit: str,
    release_hashes: Mapping[str, str],
    member_hashes: Mapping[str, str],
) -> None:
    prefix = f".audit/release-attestations/{commit}"
    if set(member_hashes) != EXPECTED_ARTIFACT_FILES:
        raise ReleaseEvidenceError("mirrored CI member hash set is incomplete")
    for basename, expected_hash in member_hashes.items():
        _spec_root_blob(
            spec_root,
            f"{prefix}/{basename}",
            _sha256(expected_hash, name=f"mirrored CI member {basename} SHA-256"),
            name=f"mirrored CI member {basename}",
            candidate=commit,
        )
    component_basenames = {
        "repository_scan": "repository_scan.json",
        "secret_scan": "secret_scan.json",
        "license_audit": "license_audit.json",
        "quality_gates": "quality_gates.json",
    }
    component_references: dict[str, dict[str, Any]] = {}
    for name, basename in component_basenames.items():
        reference, _payload = _spec_root_blob(
            spec_root,
            f"{prefix}/{basename}",
            _sha256(release_hashes.get(name), name=f"CI release hash {name}"),
            name=f"mirrored CI component {name}",
            candidate=commit,
        )
        component_references[name] = {
            "path": reference["path"],
            "size_bytes": reference["size_bytes"],
            "file_sha256": reference["file_sha256"],
        }
    bundle_reference, bundle_payload = _spec_root_blob(
        spec_root,
        f"{prefix}/ci_bundle_validation.json",
        _sha256(
            release_hashes.get("ci_bundle_validation"),
            name="CI release hash ci_bundle_validation",
        ),
        name="mirrored CI bundle validation",
        candidate=commit,
    )
    bundle = _load_json_bytes(bundle_payload, name="mirrored CI bundle validation")
    _exact_keys(
        bundle,
        {
            "schema_version",
            "record_kind",
            "status",
            "created_at_utc",
            "candidate_commit",
            "parent_commit",
            "ci_evidence_scope",
            "external_completed_ci_required_for_release_inventory",
            "validated_gates",
            "evidence",
            "record_sha256",
        },
        name="mirrored CI bundle validation",
    )
    _embedded_self_hash(
        bundle_payload,
        field="record_sha256",
        name="mirrored CI bundle validation",
    )
    _timestamp(bundle.get("created_at_utc"))
    expected_gates = {
        "repository_scan",
        "secret_scan",
        "license_audit",
        *QUALITY_GATE_NAMES,
        "ci_quality_proxy",
    }
    validated_gates = bundle.get("validated_gates")
    evidence = _mapping(bundle.get("evidence"), name="mirrored CI bundle evidence")
    if (
        bundle.get("schema_version") != "1.0.0"
        or bundle.get("record_kind") != "ci_release_gate_bundle_validation"
        or bundle.get("status") != "pass"
        or bundle.get("candidate_commit") != commit
        or bundle.get("parent_commit") != _candidate_first_parent(repository_root, commit)
        or bundle.get("ci_evidence_scope") != QUALITY_EVIDENCE_SCOPE
        or bundle.get("external_completed_ci_required_for_release_inventory") is not True
        or not isinstance(validated_gates, list)
        or validated_gates != sorted(expected_gates)
        or set(evidence) != expected_gates
    ):
        raise ReleaseEvidenceError("mirrored CI bundle validation semantics differ")
    reference_name_by_gate = {
        "repository_scan": "repository_scan",
        "secret_scan": "secret_scan",
        "license_audit": "license_audit",
        **{name: "quality_gates" for name in QUALITY_GATE_NAMES},
        "ci_quality_proxy": "quality_gates",
    }
    for gate_name, component_name in reference_name_by_gate.items():
        bundle_gate_reference = _mapping(
            evidence[gate_name], name=f"CI bundle evidence {gate_name}"
        )
        _exact_keys(
            bundle_gate_reference,
            {"path", "size_bytes", "file_sha256"},
            name=f"CI bundle evidence {gate_name}",
        )
        if dict(bundle_gate_reference) != component_references[component_name]:
            raise ReleaseEvidenceError(f"CI bundle evidence {gate_name} differs from mirror")
    if bundle_reference["file_sha256"] != release_hashes["ci_bundle_validation"]:
        raise ReleaseEvidenceError("CI bundle validation hash differs from CI evidence")


def _exact_candidate_attestation(
    spec: Mapping[str, Any],
    *,
    status: str,
    gates: Mapping[str, Mapping[str, Any]],
    spec_root: Path,
    repository_root: Path,
    commit: str,
) -> dict[str, Any]:
    item = _mapping(spec.get("exact_candidate_attestation"), name="exact_candidate_attestation")
    _exact_keys(
        item,
        {"status", "evidence_source", "evidence_path", "expected_sha256"},
        name="exact_candidate_attestation",
    )
    attestation_status = _string(item.get("status"), name="exact_candidate_attestation.status")
    source = _nullable_string(
        item.get("evidence_source"), name="exact_candidate_attestation.evidence_source"
    )
    path = _nullable_string(
        item.get("evidence_path"), name="exact_candidate_attestation.evidence_path"
    )
    expected = _nullable_string(
        item.get("expected_sha256"), name="exact_candidate_attestation.expected_sha256"
    )
    if attestation_status == "not_run":
        if source is not None or path is not None or expected is not None:
            raise ReleaseEvidenceError("not_run exact_candidate_attestation may not cite evidence")
        if status == "ready":
            raise ReleaseEvidenceError("ready inventory requires exact candidate attestation")
        return {
            "status": "not_run",
            "evidence_scope": None,
            "evidence_path": None,
            "evidence_sha256": None,
            "record_sha256": None,
        }
    if attestation_status != "pass" or source != "spec_root" or path is None or expected is None:
        raise ReleaseEvidenceError(
            "exact_candidate_attestation must be not_run or a spec_root pass"
        )
    expected_path = f".audit/release-attestations/{commit}/final_release_gate_report.json"
    if path != expected_path:
        raise ReleaseEvidenceError("exact candidate attestation path differs")
    reference, payload = _spec_root_blob(
        spec_root,
        path,
        expected,
        name="exact candidate attestation",
        candidate=commit,
    )
    report = _load_json_bytes(payload, name="exact candidate attestation")
    _exact_keys(
        report,
        {
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
        },
        name="exact candidate attestation",
    )
    record_sha256 = _embedded_self_hash(
        payload,
        field="record_sha256",
        name="exact candidate attestation",
    )
    _timestamp(report.get("created_at_utc"))
    repository = _mapping(report.get("repository"), name="exact attestation repository")
    _exact_keys(
        repository,
        {"parent_commit", "content_commit", "worktree_clean"},
        name="exact attestation repository",
    )
    policy = _mapping(report.get("policy"), name="exact attestation policy")
    _exact_keys(
        policy,
        {"path", "size_bytes", "file_sha256"},
        name="exact attestation policy",
    )
    candidate_policy, _policy_value = _candidate_policy(repository_root, commit)
    external = _mapping(
        report.get("external_candidate_attestation"),
        name="exact attestation external status",
    )
    _exact_keys(
        external,
        {"status", "expected_root", "report_path"},
        name="exact attestation external status",
    )
    report_gates = _mapping(report.get("gates"), name="exact attestation gates")
    expected_report_gates = {
        "staged_index_scan",
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
        "ci_quality_proxy",
    }
    _exact_keys(report_gates, expected_report_gates, name="exact attestation gates")
    if (
        report.get("schema_version") != "1.0.0"
        or report.get("record_kind") != "final_release_gate_report"
        or report.get("mode") != "exact_candidate_attestation"
        or report.get("status") != "pass"
        or repository.get("parent_commit") != _candidate_first_parent(repository_root, commit)
        or repository.get("content_commit") != commit
        or repository.get("worktree_clean") is not True
        or dict(policy) != candidate_policy
        or external.get("status") != "complete"
        or external.get("expected_root") != ".audit/release-attestations/<candidate_commit>/"
        or external.get("report_path") != expected_path
    ):
        raise ReleaseEvidenceError("exact candidate attestation semantics differ")
    inventory_gate_by_report_gate = {
        "staged_index_scan": "staged_index_scan",
        "repository_scan": "tracked_file_scan",
        "secret_scan": "secret_scan",
        "license_audit": "license_audit",
        **{name: name for name in QUALITY_GATE_NAMES},
        "ci_quality_proxy": "tests",
    }
    for report_gate, inventory_gate in inventory_gate_by_report_gate.items():
        gate = _mapping(report_gates[report_gate], name=f"exact gate {report_gate}")
        _exact_keys(gate, {"status", "evidence"}, name=f"exact gate {report_gate}")
        evidence = _mapping(gate.get("evidence"), name=f"exact gate {report_gate} evidence")
        _exact_keys(
            evidence,
            {"path", "size_bytes", "file_sha256"},
            name=f"exact gate {report_gate} evidence",
        )
        if gate.get("status") != "pass":
            raise ReleaseEvidenceError(f"exact gate {report_gate} did not pass")
        observed, _gate_payload = _spec_root_blob(
            spec_root,
            _string(evidence.get("path"), name=f"exact gate {report_gate} path"),
            _string(evidence.get("file_sha256"), name=f"exact gate {report_gate} SHA-256"),
            name=f"exact gate {report_gate}",
            candidate=commit,
        )
        inventory_reference = gates[inventory_gate]
        normalized_observed = {
            "path": observed["path"],
            "size_bytes": observed["size_bytes"],
            "file_sha256": observed["file_sha256"],
        }
        if (
            dict(evidence) != normalized_observed
            or inventory_reference.get("status") != "pass"
            or inventory_reference.get("evidence_sha256") != observed["file_sha256"]
        ):
            raise ReleaseEvidenceError(
                f"exact gate {report_gate} differs from release inventory evidence"
            )
    return {
        "status": "pass",
        "evidence_scope": "spec_root",
        "evidence_path": reference["path"],
        "evidence_sha256": reference["file_sha256"],
        "record_sha256": record_sha256,
    }


def _gate_entries(
    spec: Mapping[str, Any], repository_root: Path, commit: str, spec_root: Path
) -> tuple[dict[str, dict[str, Any]], str | None, set[str]]:
    gates = _mapping(spec.get("gates"), name="gates")
    _exact_keys(gates, set(GATE_NAMES), name="gates")
    policy_reference, policy = _candidate_policy(repository_root, commit)
    result: dict[str, dict[str, Any]] = {}
    ci_repository: str | None = None
    ci_release_hashes: dict[str, str] | None = None
    ci_member_hashes: dict[str, str] | None = None
    ci_external_basenames: set[str] = set()
    for gate_name in GATE_NAMES:
        item = _mapping(gates[gate_name], name=f"gates.{gate_name}")
        _exact_keys(
            item,
            {
                "status",
                "evidence_source",
                "evidence_path",
                "expected_sha256",
                "tool_version",
                "note",
            },
            name=f"gates.{gate_name}",
        )
        status = _string(item["status"], name=f"gates.{gate_name}.status")
        if status not in _GATE_STATUSES:
            raise ReleaseEvidenceError(f"invalid gate status for {gate_name}: {status}")
        source = _nullable_string(
            item["evidence_source"], name=f"gates.{gate_name}.evidence_source"
        )
        path = _nullable_string(item["evidence_path"], name=f"gates.{gate_name}.evidence_path")
        expected = _nullable_string(
            item["expected_sha256"], name=f"gates.{gate_name}.expected_sha256"
        )
        tool_version = _nullable_string(
            item["tool_version"], name=f"gates.{gate_name}.tool_version"
        )
        if status in {"not_run", "not_applicable"}:
            if source is not None or path is not None or expected is not None:
                raise ReleaseEvidenceError(f"{gate_name} {status} must not cite pass evidence")
            reference = {"scope": None, "path": None, "file_sha256": None}
            payload = None
        else:
            if source is None or path is None or expected is None:
                raise ReleaseEvidenceError(f"{gate_name} {status} requires pinned evidence")
            if tool_version is None:
                raise ReleaseEvidenceError(f"{gate_name} {status} requires a tool version")
            full_reference, payload = _evidence_reference(
                source=source,
                path=path,
                expected_sha256=expected,
                repository_root=repository_root,
                candidate_commit=commit,
                spec_root=spec_root,
                name=f"gate {gate_name}",
            )
            reference = {
                "scope": full_reference["scope"],
                "path": full_reference["path"],
                "file_sha256": full_reference["file_sha256"],
            }
        if gate_name == "ci" and status in {"pass", "fail"}:
            assert payload is not None
            (
                _kind,
                _record_hash,
                ci_repository,
                ci_release_hashes,
                ci_member_hashes,
                ci_external_basenames,
            ) = _validate_ci_record(
                payload, commit=commit, expected_status=status, spec_root=spec_root
            )
        elif status in {"pass", "fail"}:
            assert payload is not None
            _validate_non_ci_gate_record(
                payload,
                gate_name=gate_name,
                status=status,
                commit=commit,
                repository_root=repository_root,
                spec_root=spec_root,
                policy_reference=policy_reference,
                policy=policy,
            )
        if status in {"pass", "fail"}:
            assert payload is not None
            derived_tool_version = _derived_gate_tool_version(gate_name, payload, commit=commit)
            if tool_version != derived_tool_version:
                raise ReleaseEvidenceError(f"{gate_name} tool_version differs from parsed evidence")
            tool_version = derived_tool_version
        result[gate_name] = {
            "status": status,
            "evidence_scope": reference["scope"],
            "evidence_path": reference["path"],
            "evidence_sha256": reference["file_sha256"],
            "tool_version": tool_version,
            "note": _nullable_string(item["note"], name=f"gates.{gate_name}.note"),
        }
    if ci_release_hashes is not None:
        assert ci_member_hashes is not None
        _validate_mirrored_ci_bundle(
            spec_root=spec_root,
            repository_root=repository_root,
            commit=commit,
            release_hashes=ci_release_hashes,
            member_hashes=ci_member_hashes,
        )
        expected_gate_hashes = {
            "repository_scan": result["tracked_file_scan"]["evidence_sha256"],
            "secret_scan": result["secret_scan"]["evidence_sha256"],
            "license_audit": result["license_audit"]["evidence_sha256"],
            "quality_gates": result["tests"]["evidence_sha256"],
        }
        if any(
            ci_release_hashes[name] != expected for name, expected in expected_gate_hashes.items()
        ) or any(
            result[name]["evidence_sha256"] != ci_release_hashes["quality_gates"]
            for name in QUALITY_GATE_NAMES
        ):
            raise ReleaseEvidenceError("CI artifact file hashes differ from cited gate evidence")
    return result, ci_repository, ci_external_basenames


def _validate_remote_record(
    payload: bytes,
    *,
    commit: str,
    visibility: str,
    remote_url: str,
    spec_root: Path,
) -> set[str]:
    record = _load_json_bytes(payload, name="remote evidence")
    _exact_keys(
        record,
        {
            "schema_version",
            "record_kind",
            "status",
            "candidate_commit",
            "visibility",
            "remote_url",
            "repository",
            "default_branch",
            "ref_name",
            "ref_commit",
            "repository_api_url",
            "ref_api_url",
            "api_response_files",
            "queried_at_utc",
            "record_sha256",
        },
        name="remote evidence",
    )
    if (
        record.get("schema_version") != "1.0.0"
        or record.get("record_kind") != "github_remote_evidence"
        or record.get("status") != "pass"
        or record.get("candidate_commit") != commit
        or record.get("visibility") != visibility
        or record.get("remote_url") != remote_url
        or record.get("default_branch") != "main"
        or record.get("ref_name") != "refs/heads/main"
        or record.get("ref_commit") != commit
    ):
        raise ReleaseEvidenceError("remote evidence has incompatible state or commit")
    match = re.fullmatch(r"https://github\.com/(?P<repository>[^/]+/[^/]+)", remote_url)
    repository = _github_repository(record.get("repository"), name="remote evidence repository")
    if (
        match is None
        or match.group("repository").casefold() != repository.casefold()
        or record.get("repository_api_url") != f"https://api.github.com/repos/{repository}"
        or record.get("ref_api_url")
        != f"https://api.github.com/repos/{repository}/git/refs/heads/main"
    ):
        raise ReleaseEvidenceError("remote evidence API and repository identity differ")
    api_references = _validate_file_references(
        record.get("api_response_files"),
        expected_names={"repository", "main_ref"},
        name="remote API response files",
    )
    api_responses = _validate_external_api_response_files(
        spec_root,
        api_references,
        name="remote API response files",
    )
    repository_response = api_responses["repository"]
    private = repository_response.get("private")
    api_visibility = repository_response.get("visibility")
    if private is True and api_visibility in {None, "private"}:
        retained_visibility = "private"
    elif private is False and api_visibility in {None, "public"}:
        retained_visibility = "public"
    else:
        raise ReleaseEvidenceError("retained repository visibility fields are inconsistent")
    ref_response = api_responses["main_ref"]
    ref_object = _mapping(ref_response.get("object"), name="retained main-ref object")
    if (
        repository_response.get("full_name") != repository
        or repository_response.get("html_url") != remote_url
        or repository_response.get("url") != record.get("repository_api_url")
        or repository_response.get("default_branch") != "main"
        or retained_visibility != visibility
        or ref_response.get("ref") != "refs/heads/main"
        or ref_response.get("url") != record.get("ref_api_url")
        or ref_object.get("type") != "commit"
        or ref_object.get("sha") != commit
    ):
        raise ReleaseEvidenceError("retained repository/main-ref API responses differ")
    _timestamp(record.get("queried_at_utc"))
    _embedded_self_hash(payload, field="record_sha256", name="remote evidence")
    return {reference["basename"].casefold() for reference in api_references.values()}


def _remote_state(
    spec: Mapping[str, Any], repository_root: Path, commit: str, spec_root: Path
) -> tuple[dict[str, Any], str | None, set[str]]:
    item = _mapping(spec.get("remote"), name="remote")
    _exact_keys(
        item,
        {"visibility", "url", "evidence_source", "evidence_path", "expected_sha256"},
        name="remote",
    )
    visibility = _string(item["visibility"], name="remote.visibility")
    if visibility not in {"absent", "private", "public"}:
        raise ReleaseEvidenceError("remote.visibility is invalid")
    url = _nullable_string(item["url"], name="remote.url")
    source = _nullable_string(item["evidence_source"], name="remote.evidence_source")
    path = _nullable_string(item["evidence_path"], name="remote.evidence_path")
    expected = _nullable_string(item["expected_sha256"], name="remote.expected_sha256")
    if visibility == "absent":
        if url is not None or source is not None or path is not None or expected is not None:
            raise ReleaseEvidenceError("absent remote must not carry URL or evidence")
        return (
            {
                "remote_visibility": "absent",
                "remote_url": None,
                "remote_evidence_scope": None,
                "remote_evidence_path": None,
                "remote_evidence_sha256": None,
            },
            None,
            set(),
        )
    match = re.fullmatch(r"https://github\.com/(?P<repository>[^/?#\s]+/[^/?#\s]+)", url or "")
    if url is None or match is None:
        raise ReleaseEvidenceError("configured remote requires a canonical GitHub HTTPS URL")
    remote_repository = _github_repository(
        match.group("repository"), name="configured remote repository"
    )
    if source is None or path is None or expected is None:
        raise ReleaseEvidenceError("configured remote requires pinned validation evidence")
    reference, payload = _evidence_reference(
        source=source,
        path=path,
        expected_sha256=expected,
        repository_root=repository_root,
        candidate_commit=commit,
        spec_root=spec_root,
        name="remote evidence",
    )
    api_basenames = _validate_remote_record(
        payload,
        commit=commit,
        visibility=visibility,
        remote_url=url,
        spec_root=spec_root,
    )
    return (
        {
            "remote_visibility": visibility,
            "remote_url": url,
            "remote_evidence_scope": reference["scope"],
            "remote_evidence_path": reference["path"],
            "remote_evidence_sha256": reference["file_sha256"],
        },
        remote_repository,
        api_basenames,
    )


def _postconfirmatory_entries(
    spec: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    artifact_by_role = {str(value["role"]): value for value in artifacts}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(
        _sequence(spec.get("postconfirmatory_statuses"), name="postconfirmatory_statuses")
    ):
        item = _mapping(value, name=f"postconfirmatory_statuses[{index}]")
        _exact_keys(
            item,
            {"track", "status", "evidence_role", "note", "primary_claim_eligible"},
            name=f"postconfirmatory_statuses[{index}]",
        )
        track = _string(item["track"], name="postconfirmatory track")
        if track not in POSTCONFIRMATORY_ROLE_BY_TRACK or track in seen:
            raise ReleaseEvidenceError(
                f"postconfirmatory track is unsupported or duplicated: {track}"
            )
        seen.add(track)
        status = _string(item["status"], name=f"{track}.status")
        if status not in _POSTCONFIRMATORY_STATUSES:
            raise ReleaseEvidenceError(f"invalid postconfirmatory status for {track}")
        if item["primary_claim_eligible"] is not False:
            raise ReleaseEvidenceError(f"{track} must set primary_claim_eligible=false")
        evidence_role = _nullable_string(item["evidence_role"], name=f"{track}.evidence_role")
        expected_role = POSTCONFIRMATORY_ROLE_BY_TRACK[track]
        if evidence_role is not None and evidence_role != expected_role:
            raise ReleaseEvidenceError(f"{track} evidence role must equal {expected_role}")
        if status in {"not_run", "not_applicable"} and evidence_role is not None:
            raise ReleaseEvidenceError(f"{track} {status} must not cite completed evidence")
        if status == "completed":
            if evidence_role is None or evidence_role not in artifact_by_role:
                raise ReleaseEvidenceError(f"completed track {track} lacks its evidence artifact")
            if artifact_by_role[evidence_role]["validation_status"] not in {"pass", "conditional"}:
                raise ReleaseEvidenceError(f"completed track {track} evidence did not validate")
        if evidence_role is not None and evidence_role not in artifact_by_role:
            raise ReleaseEvidenceError(f"{track} cites an absent artifact role")
        entries.append(
            {
                "track": track,
                "status": status,
                "evidence_role": evidence_role,
                "note": _string(item["note"], name=f"{track}.note"),
                "primary_claim_eligible": False,
            }
        )
    missing = set(POSTCONFIRMATORY_TRACKS) - seen
    if missing:
        raise ReleaseEvidenceError(
            f"postconfirmatory status entries are missing: {sorted(missing)}"
        )
    return sorted(entries, key=lambda value: str(value["track"]))


def _blocker_entries(
    spec: Mapping[str, Any], artifacts: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    roles = {str(value["role"]) for value in artifacts}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(_sequence(spec.get("blockers"), name="blockers")):
        item = _mapping(value, name=f"blockers[{index}]")
        _exact_keys(
            item,
            {"blocker_id", "status", "description", "evidence_role"},
            name=f"blockers[{index}]",
        )
        blocker_id = _string(item["blocker_id"], name="blocker_id")
        if _IDENTIFIER_RE.fullmatch(blocker_id) is None or blocker_id in seen:
            raise ReleaseEvidenceError(f"blocker_id is invalid or duplicated: {blocker_id}")
        seen.add(blocker_id)
        status = _string(item["status"], name=f"{blocker_id}.status")
        if status not in _BLOCKER_STATUSES:
            raise ReleaseEvidenceError(f"invalid blocker status: {status}")
        evidence_role = _nullable_string(item["evidence_role"], name="evidence_role")
        if evidence_role is not None and evidence_role not in roles:
            raise ReleaseEvidenceError(f"blocker {blocker_id} cites an absent evidence role")
        entries.append(
            {
                "blocker_id": blocker_id,
                "status": status,
                "description": _string(item["description"], name="blocker description"),
                "evidence_role": evidence_role,
            }
        )
    return sorted(entries, key=lambda value: str(value["blocker_id"]))


def _build_inventory(
    *,
    spec: Mapping[str, Any],
    spec_reference: Mapping[str, Any],
    spec_payload: bytes,
    spec_root: Path,
    repository_root: Path,
    candidate_commit: str,
) -> dict[str, Any]:
    _exact_keys(
        spec,
        {
            "schema_version",
            "spec_kind",
            "requested_status",
            "created_at_utc",
            "protocol_tag",
            "remote",
            "exact_candidate_attestation",
            "confirmatory_state",
            "artifacts",
            "gates",
            "postconfirmatory_statuses",
            "blockers",
        },
        name="generation spec",
    )
    if spec.get("schema_version") != SPEC_SCHEMA_VERSION or spec.get("spec_kind") != SPEC_KIND:
        raise ReleaseEvidenceError("generation spec version or kind differs")
    status = _string(spec["requested_status"], name="requested_status")
    if status not in _INVENTORY_STATUSES:
        raise ReleaseEvidenceError("requested_status is invalid")
    if status == "ready" and spec_reference.get("scope") != "external_spec":
        raise ReleaseEvidenceError(
            f"{status} inventory requires a reviewed generation spec outside the repository"
        )
    confirmatory = _mapping(spec["confirmatory_state"], name="confirmatory_state")
    _exact_keys(
        confirmatory,
        {"opening_count", "target_rerun_permitted"},
        name="confirmatory_state",
    )
    _exact_integer(
        confirmatory["opening_count"],
        name="confirmatory_state.opening_count",
        expected=1,
    )
    if confirmatory["target_rerun_permitted"] is not False:
        raise ReleaseEvidenceError("confirmatory state must preserve one opening and forbid rerun")

    artifacts = _artifact_entries(spec, repository_root, candidate_commit)
    artifact_by_role = {str(value["role"]): value for value in artifacts}
    gates, ci_repository, ci_external_basenames = _gate_entries(
        spec, repository_root, candidate_commit, spec_root
    )
    exact_attestation = _exact_candidate_attestation(
        spec,
        status=status,
        gates=gates,
        spec_root=spec_root,
        repository_root=repository_root,
        commit=candidate_commit,
    )
    remote, remote_repository, remote_api_basenames = _remote_state(
        spec, repository_root, candidate_commit, spec_root
    )
    if (
        ci_repository is not None
        and remote_repository is not None
        and ci_repository.casefold() != remote_repository.casefold()
    ):
        raise ReleaseEvidenceError("CI evidence repository differs from the configured remote")
    external_github_basenames = ci_external_basenames | remote_api_basenames
    if (ci_external_basenames and len(ci_external_basenames) != 3) or (
        remote_api_basenames and len(remote_api_basenames) != 2
    ):
        raise ReleaseEvidenceError("GitHub API/archive evidence basenames are internally aliased")
    if ci_external_basenames and remote_api_basenames and len(external_github_basenames) != 5:
        raise ReleaseEvidenceError(
            "four GitHub API responses and the Actions archive require distinct basenames"
        )
    postconfirmatory = _postconfirmatory_entries(spec, artifacts)
    blockers = _blocker_entries(spec, artifacts)
    open_blockers = [value for value in blockers if value["status"] == "open"]
    if status == "blocked" and not open_blockers:
        raise ReleaseEvidenceError("blocked inventory requires at least one open blocker")
    if status == "ready":
        if open_blockers:
            raise ReleaseEvidenceError(f"{status} inventory cannot contain open blockers")
        if any(value["status"] != "pass" for value in gates.values()):
            raise ReleaseEvidenceError(f"{status} inventory requires every release gate to pass")
        if remote["remote_visibility"] == "absent":
            raise ReleaseEvidenceError(f"{status} inventory requires validated remote evidence")
        if exact_attestation["status"] != "pass":
            raise ReleaseEvidenceError(
                f"{status} inventory requires a validated exact candidate attestation"
            )
        non_external_gates = [
            gate_name
            for gate_name, value in gates.items()
            if value["evidence_scope"] != "spec_root"
        ]
        if non_external_gates:
            raise ReleaseEvidenceError(
                f"{status} inventory requires sanitized spec_root evidence for every gate: "
                f"{non_external_gates}"
            )
        if remote["remote_evidence_scope"] != "spec_root":
            raise ReleaseEvidenceError(
                f"{status} inventory requires sanitized spec_root remote evidence"
            )
        invalid_required = [
            value["role"]
            for value in artifacts
            if value["required_for_release"] is True
            and value["validation_status"] not in {"pass", "conditional"}
        ]
        if invalid_required:
            raise ReleaseEvidenceError(
                f"{status} inventory has invalid required artifacts: {invalid_required}"
            )

    opening = artifact_by_role["opening_receipt"]
    target_index = artifact_by_role["locked_target_index"]
    protocol_tag, protocol_tag_object, protocol_tag_target = _protocol_tag_binding(
        repository_root,
        _string(spec["protocol_tag"], name="protocol_tag"),
        candidate_commit=candidate_commit,
    )
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "inventory_kind": INVENTORY_KIND,
        "delivery_mode": "external_release_asset",
        "tracked": False,
        "status": status,
        "created_at_utc": _timestamp(spec["created_at_utc"]),
        "generation_spec": {
            **spec_reference,
            "size_bytes": len(spec_payload),
            "file_sha256": hashlib.sha256(spec_payload).hexdigest(),
        },
        "repository": {
            "code_commit": candidate_commit,
            "worktree_clean": True,
            "protocol_tag": protocol_tag,
            "protocol_tag_object": protocol_tag_object,
            "protocol_tag_target": protocol_tag_target,
            **remote,
        },
        "confirmatory_state": {
            "opening_count": 1,
            "target_rerun_permitted": False,
            "opening_receipt_path": opening["path"],
            "opening_receipt_record_sha256": opening["record_sha256"],
            "target_index_path": target_index["path"],
            "target_index_record_sha256": target_index["record_sha256"],
        },
        "exact_candidate_attestation": exact_attestation,
        "required_roles": list(REQUIRED_ROLES),
        "artifacts": artifacts,
        "gates": gates,
        "postconfirmatory_statuses": postconfirmatory,
        "blockers": blockers,
    }
    body["record_sha256"] = canonical_json_sha256(body)
    return body


def _spec_input(
    spec_path: str | Path, *, repository_root: Path
) -> tuple[dict[str, Any], bytes, Path]:
    source = Path(spec_path)
    if source.is_symlink():
        raise ReleaseEvidenceError("generation spec may not be a symlink")
    resolved = source.resolve(strict=True)
    if not resolved.is_file():
        raise ReleaseEvidenceError("generation spec must be a regular file")
    if resolved.stat().st_size > MAX_EVIDENCE_BYTES:
        raise ReleaseEvidenceError("generation spec exceeds the evidence size cap")
    payload = resolved.read_bytes()
    if len(payload) > MAX_EVIDENCE_BYTES:
        raise ReleaseEvidenceError("generation spec exceeds the evidence size cap")
    try:
        relative = resolved.relative_to(repository_root)
    except ValueError:
        basename = _safe_relative_path(resolved.name, name="generation spec basename")
        reference = {
            "scope": "external_spec",
            "basename": basename,
            "absolute_path_recorded": False,
        }
    else:
        reference = {
            "scope": "repository_root",
            "path": _safe_relative_path(relative.as_posix(), name="generation spec"),
            "absolute_path_recorded": False,
        }
    return reference, payload, resolved.parent


def _new_output(path: str | Path, *, allowed_root: Path, repository_root: Path) -> Path:
    if allowed_root.is_symlink():
        raise ReleaseEvidenceError("allowed_output_root may not be a symlink")
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseEvidenceError("allowed_output_root must be an existing directory")
    try:
        root.relative_to(repository_root)
    except ValueError as exc:
        raise ReleaseEvidenceError("allowed_output_root escapes repository_root") from exc
    raw = Path(path)
    if any(part == ".." for part in raw.parts):
        raise ReleaseEvidenceError("inventory destination may not contain parent traversal")
    candidate = raw if raw.is_absolute() else root / raw
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ReleaseEvidenceError("inventory destination escapes allowed_output_root") from exc
    for component_count in range(1, len(relative.parts) + 1):
        component = root.joinpath(*relative.parts[:component_count])
        if component.is_symlink():
            raise ReleaseEvidenceError("inventory destination may not traverse a symlink")
    prospective = candidate.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise ReleaseEvidenceError("inventory destination escapes allowed_output_root") from exc
    if os.path.lexists(prospective):
        raise FileExistsError(f"refusing to overwrite release inventory: {prospective}")
    return candidate


def generate_release_evidence_inventory(
    *,
    spec_path: str | Path,
    repository_root: str | Path,
    candidate_commit: str,
    require_clean_worktree: bool,
    destination: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Generate and create-only publish a release inventory from a reviewed spec."""

    root = _repository(repository_root)
    output_root_path = Path(allowed_output_root)
    if not output_root_path.is_absolute():
        output_root_path = root / output_root_path
    output = _new_output(
        destination,
        allowed_root=output_root_path,
        repository_root=root,
    )
    commit = _commit(candidate_commit, name="candidate_commit")
    if require_clean_worktree is not True:
        raise ReleaseEvidenceError("--require-clean-worktree acknowledgement is mandatory")
    observed_commit, status = _git_state(root)
    if observed_commit != commit:
        raise ReleaseEvidenceError("candidate_commit differs from repository HEAD")
    if status:
        raise ReleaseEvidenceError("candidate repository worktree is not clean")
    spec_reference, spec_payload, spec_root = _spec_input(spec_path, repository_root=root)
    spec = _load_json_bytes(spec_payload, name="generation spec")
    inventory = _build_inventory(
        spec=spec,
        spec_reference=spec_reference,
        spec_payload=spec_payload,
        spec_root=spec_root,
        repository_root=root,
        candidate_commit=commit,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output = _new_output(
        output,
        allowed_root=output_root_path,
        repository_root=root,
    )
    atomic_write_json_new(inventory, output, allowed_root=output_root_path)
    return inventory


def validate_release_evidence_inventory_file(
    inventory_path: str | Path,
    *,
    spec_path: str | Path,
    repository_root: str | Path,
    expected_candidate_commit: str | None = None,
) -> dict[str, Any]:
    """Validate an inventory by deterministic reconstruction from its tracked spec."""

    source = Path(inventory_path)
    try:
        root = _repository(repository_root)
        value = _mapping(load_json_strict(source), name="release inventory")
        record = dict(value)
        observed_hash = _sha256(record.pop("record_sha256", None), name="record_sha256")
        if canonical_json_sha256(record) != observed_hash:
            raise ReleaseEvidenceError("release inventory self-hash does not reconstruct")
        repository = _mapping(value.get("repository"), name="repository")
        commit = _commit(repository.get("code_commit"), name="repository.code_commit")
        if expected_candidate_commit is not None and commit != _commit(
            expected_candidate_commit, name="expected_candidate_commit"
        ):
            raise ReleaseEvidenceError("inventory code_commit differs from expected candidate")
        _require_commit(root, commit)
        spec_reference = _mapping(value.get("generation_spec"), name="generation_spec")
        observed_spec_reference, spec_payload, spec_root = _spec_input(
            spec_path, repository_root=root
        )
        if (
            any(
                spec_reference.get(key) != reference_value
                for key, reference_value in observed_spec_reference.items()
            )
            or spec_reference.get("size_bytes") != len(spec_payload)
            or spec_reference.get("file_sha256") != hashlib.sha256(spec_payload).hexdigest()
        ):
            raise ReleaseEvidenceError("generation spec reference differs from supplied spec")
        expected = _build_inventory(
            spec=_load_json_bytes(spec_payload, name="generation spec"),
            spec_reference=observed_spec_reference,
            spec_payload=spec_payload,
            spec_root=spec_root,
            repository_root=root,
            candidate_commit=commit,
        )
        if canonical_json_sha256(value) != canonical_json_sha256(expected):
            raise ReleaseEvidenceError("inventory differs from deterministic reconstruction")
        return {
            "valid": True,
            "status": value["status"],
            "code_commit": commit,
            "artifact_count": len(_sequence(value.get("artifacts"), name="artifacts")),
            "required_role_count": len(REQUIRED_ROLES),
            "postconfirmatory_track_count": len(POSTCONFIRMATORY_TRACKS),
            "record_sha256": observed_hash,
            "errors": [],
        }
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)]}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    generate = subparsers.add_parser("generate")
    generate.add_argument("--spec", type=Path, required=True)
    generate.add_argument("--repository-root", type=Path, required=True)
    generate.add_argument("--candidate-commit", required=True)
    generate.add_argument("--require-clean-worktree", action="store_true", required=True)
    generate.add_argument("--destination", type=Path, required=True)
    generate.add_argument("--allowed-output-root", type=Path, required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--inventory", type=Path, required=True)
    validate.add_argument("--spec", type=Path, required=True)
    validate.add_argument("--repository-root", type=Path, required=True)
    validate.add_argument("--candidate-commit")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "generate":
        try:
            inventory = generate_release_evidence_inventory(
                spec_path=args.spec,
                repository_root=args.repository_root,
                candidate_commit=args.candidate_commit,
                require_clean_worktree=args.require_clean_worktree,
                destination=args.destination,
                allowed_output_root=args.allowed_output_root,
            )
        except Exception as exc:
            print(
                json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr
            )
            return 2
        print(
            json.dumps(
                {
                    "status": inventory["status"],
                    "record_sha256": inventory["record_sha256"],
                    "artifact_count": len(inventory["artifacts"]),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    report = validate_release_evidence_inventory_file(
        args.inventory,
        spec_path=args.spec,
        repository_root=args.repository_root,
        expected_candidate_commit=args.candidate_commit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["valid"] is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
