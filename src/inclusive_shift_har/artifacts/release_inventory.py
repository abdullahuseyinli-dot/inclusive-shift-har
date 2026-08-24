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
    "tracked_file_scan",
    "secret_scan",
    "license_audit",
    "ci",
)

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
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
_INVENTORY_STATUSES = {"draft", "blocked", "ready", "released"}
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
_FORBIDDEN_SUFFIXES = {
    ".ckpt",
    ".docx",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".zip",
}


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


def _nullable_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name=name)


def _timestamp(value: Any) -> str:
    text = _string(value, name="created_at_utc")
    if not text.endswith("Z"):
        raise ReleaseEvidenceError("created_at_utc must use the UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise ReleaseEvidenceError("created_at_utc must be valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise ReleaseEvidenceError("created_at_utc must be UTC")
    return text


def _safe_relative_path(value: Any, *, name: str) -> str:
    text = _string(value, name=name)
    if "\\" in text or ":" in text or "\x00" in text:
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
    if path.suffix.casefold() in _FORBIDDEN_SUFFIXES:
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
            ["git", *arguments],
            cwd=repository_root,
            check=check,
            capture_output=True,
            text=text,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ReleaseEvidenceError(f"Git command failed: {' '.join(arguments)}") from exc


def _git_state(repository_root: Path) -> tuple[str, str]:
    commit_result = _git(repository_root, ("rev-parse", "HEAD"), text=True)
    status_result = _git(
        repository_root,
        ("status", "--porcelain=v1", "--untracked-files=all"),
        text=True,
    )
    assert isinstance(commit_result.stdout, str)
    assert isinstance(status_result.stdout, str)
    return commit_result.stdout.strip().casefold(), status_result.stdout.strip()


def _require_commit(repository_root: Path, commit: str) -> None:
    result = _git(
        repository_root,
        ("cat-file", "-e", f"{commit}^{{commit}}"),
        text=False,
        check=False,
    )
    if result.returncode != 0:
        raise ReleaseEvidenceError("candidate_commit does not identify a Git commit")


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


def _spec_root_blob(
    spec_root: Path, path: str, expected_sha256: str, *, name: str
) -> tuple[dict[str, Any], bytes]:
    relative = _safe_relative_path(path, name=f"{name}.path")
    candidate = spec_root / Path(relative)
    if candidate.is_symlink():
        raise ReleaseEvidenceError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(spec_root)
    except ValueError as exc:
        raise ReleaseEvidenceError(f"{name} escapes spec_root") from exc
    if not resolved.is_file():
        raise ReleaseEvidenceError(f"{name} must be a regular file")
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
        return _spec_root_blob(spec_root, path, expected_sha256, name=name)
    raise ReleaseEvidenceError(f"{name}.evidence_source is invalid")


def _embedded_self_hash(payload: bytes, *, field: str, name: str) -> str:
    if _IDENTIFIER_RE.fullmatch(field) is None:
        raise ReleaseEvidenceError(f"{name} embedded hash field is invalid")
    record = dict(_load_json_bytes(payload, name=name))
    observed = _sha256(record.pop(field, None), name=f"{name}.{field}")
    if canonical_json_sha256(record) != observed:
        raise ReleaseEvidenceError(f"{name} embedded self-hash does not reconstruct")
    return observed


def _artifact_entries(
    spec: Mapping[str, Any], repository_root: Path, commit: str
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    seen_roles: set[str] = set()
    seen_paths: set[str] = set()
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
    for role in ("opening_receipt", "locked_target_index"):
        entry = next(value for value in entries if value["role"] == role)
        if entry["embedded_self_hash_field"] != "record_sha256" or entry["record_sha256"] is None:
            raise ReleaseEvidenceError(f"{role} must declare and validate record_sha256")
    return sorted(entries, key=lambda value: str(value["role"]))


def _validate_ci_record(payload: bytes, *, commit: str) -> None:
    record = _load_json_bytes(payload, name="CI evidence")
    if (
        record.get("record_kind") != "ci_run_evidence"
        or record.get("status") != "pass"
        or record.get("conclusion") != "success"
        or record.get("candidate_commit") != commit
    ):
        raise ReleaseEvidenceError("CI pass evidence has incompatible status or commit")
    _embedded_self_hash(payload, field="record_sha256", name="CI evidence")


def _gate_entries(
    spec: Mapping[str, Any], repository_root: Path, commit: str, spec_root: Path
) -> dict[str, dict[str, Any]]:
    gates = _mapping(spec.get("gates"), name="gates")
    _exact_keys(gates, set(GATE_NAMES), name="gates")
    result: dict[str, dict[str, Any]] = {}
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
        if status in {"not_run", "not_applicable"}:
            if source is not None or path is not None or expected is not None:
                raise ReleaseEvidenceError(f"{gate_name} {status} must not cite pass evidence")
            reference = {"scope": None, "path": None, "file_sha256": None}
            payload = None
        else:
            if source is None or path is None or expected is None:
                raise ReleaseEvidenceError(f"{gate_name} {status} requires pinned evidence")
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
        if gate_name == "ci" and status == "pass":
            assert payload is not None
            _validate_ci_record(payload, commit=commit)
        result[gate_name] = {
            "status": status,
            "evidence_scope": reference["scope"],
            "evidence_path": reference["path"],
            "evidence_sha256": reference["file_sha256"],
            "tool_version": _nullable_string(
                item["tool_version"], name=f"gates.{gate_name}.tool_version"
            ),
            "note": _nullable_string(item["note"], name=f"gates.{gate_name}.note"),
        }
    return result


def _validate_remote_record(
    payload: bytes, *, commit: str, visibility: str, remote_url: str
) -> None:
    record = _load_json_bytes(payload, name="remote evidence")
    if (
        record.get("record_kind") != "github_remote_evidence"
        or record.get("status") != "pass"
        or record.get("candidate_commit") != commit
        or record.get("visibility") != visibility
        or record.get("remote_url") != remote_url
    ):
        raise ReleaseEvidenceError("remote evidence has incompatible state or commit")
    _embedded_self_hash(payload, field="record_sha256", name="remote evidence")


def _remote_state(
    spec: Mapping[str, Any], repository_root: Path, commit: str, spec_root: Path
) -> dict[str, Any]:
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
        return {
            "remote_visibility": "absent",
            "remote_url": None,
            "remote_evidence_scope": None,
            "remote_evidence_path": None,
            "remote_evidence_sha256": None,
        }
    if url is None or re.fullmatch(r"https://github\.com/[^/]+/[^/]+", url) is None:
        raise ReleaseEvidenceError("configured remote requires a canonical GitHub HTTPS URL")
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
    _validate_remote_record(payload, commit=commit, visibility=visibility, remote_url=url)
    return {
        "remote_visibility": visibility,
        "remote_url": url,
        "remote_evidence_scope": reference["scope"],
        "remote_evidence_path": reference["path"],
        "remote_evidence_sha256": reference["file_sha256"],
    }


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
    confirmatory = _mapping(spec["confirmatory_state"], name="confirmatory_state")
    _exact_keys(
        confirmatory,
        {"opening_count", "target_rerun_permitted"},
        name="confirmatory_state",
    )
    if confirmatory["opening_count"] != 1 or confirmatory["target_rerun_permitted"] is not False:
        raise ReleaseEvidenceError("confirmatory state must preserve one opening and forbid rerun")

    artifacts = _artifact_entries(spec, repository_root, candidate_commit)
    artifact_by_role = {str(value["role"]): value for value in artifacts}
    gates = _gate_entries(spec, repository_root, candidate_commit, spec_root)
    remote = _remote_state(spec, repository_root, candidate_commit, spec_root)
    postconfirmatory = _postconfirmatory_entries(spec, artifacts)
    blockers = _blocker_entries(spec, artifacts)
    open_blockers = [value for value in blockers if value["status"] == "open"]
    if status == "blocked" and not open_blockers:
        raise ReleaseEvidenceError("blocked inventory requires at least one open blocker")
    if status in {"ready", "released"}:
        if open_blockers:
            raise ReleaseEvidenceError(f"{status} inventory cannot contain open blockers")
        if any(value["status"] != "pass" for value in gates.values()):
            raise ReleaseEvidenceError(f"{status} inventory requires every release gate to pass")
        if remote["remote_visibility"] == "absent":
            raise ReleaseEvidenceError(f"{status} inventory requires validated remote evidence")
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
            "protocol_tag": _string(spec["protocol_tag"], name="protocol_tag"),
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
    candidate = raw if raw.is_absolute() else root / raw
    prospective = candidate.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise ReleaseEvidenceError("inventory destination escapes allowed_output_root") from exc
    if os.path.lexists(prospective):
        raise FileExistsError(f"refusing to overwrite release inventory: {prospective}")
    return prospective


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

    root_path = Path(repository_root)
    if root_path.is_symlink():
        raise ReleaseEvidenceError("repository_root may not be a symlink")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise ReleaseEvidenceError("repository_root must be an existing directory")
    output_root_path = Path(allowed_output_root)
    if not output_root_path.is_absolute():
        output_root_path = root / output_root_path
    output = _new_output(
        destination,
        allowed_root=output_root_path,
        repository_root=root,
    )
    commit = candidate_commit.casefold()
    if _COMMIT_RE.fullmatch(commit) is None:
        raise ReleaseEvidenceError("candidate_commit must be a full lowercase Git commit")
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
        root_path = Path(repository_root)
        if root_path.is_symlink():
            raise ReleaseEvidenceError("repository_root may not be a symlink")
        root = root_path.resolve(strict=True)
        value = _mapping(load_json_strict(source), name="release inventory")
        record = dict(value)
        observed_hash = _sha256(record.pop("record_sha256", None), name="record_sha256")
        if canonical_json_sha256(record) != observed_hash:
            raise ReleaseEvidenceError("release inventory self-hash does not reconstruct")
        repository = _mapping(value.get("repository"), name="repository")
        commit = _string(repository.get("code_commit"), name="repository.code_commit").casefold()
        if _COMMIT_RE.fullmatch(commit) is None:
            raise ReleaseEvidenceError("inventory code_commit is invalid")
        if expected_candidate_commit is not None and commit != expected_candidate_commit.casefold():
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
