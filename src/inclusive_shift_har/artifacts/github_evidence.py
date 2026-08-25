"""Create-only capture of GitHub remote and completed-CI release evidence.

The command consumes saved ``gh api`` JSON responses and the downloaded Actions
artifact. It never calls GitHub itself, so the exact API responses remain
reviewable inputs and capture can be repeated offline in a fresh directory.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import stat
import sys
import zipfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
)

SCHEMA_VERSION = "1.0.0"
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
GITHUB_REPOSITORY_RE = re.compile(
    r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?/[A-Za-z0-9._-]{1,100}$"
)
QUALITY_GATES = (
    "tests",
    "lint",
    "format",
    "types",
    "manifests",
    "splits",
    "configuration",
    "artifacts",
)
CI_QUALITY_GATE_NAMES = (*QUALITY_GATES, "ci_quality_proxy")
COMPONENT_FILES = (
    "repository_scan.json",
    "secret_scan.json",
    "license_audit.json",
    "quality_gates.json",
    "ci_bundle_validation.json",
)
PROVENANCE_FILES = tuple(
    f"{gate}.{suffix}"
    for gate in QUALITY_GATES
    for suffix in ("status", "exit", "started", "completed", "log")
)
EXPECTED_ARTIFACT_FILES = frozenset((*COMPONENT_FILES, *PROVENANCE_FILES))
MAX_ARCHIVE_MEMBERS = len(EXPECTED_ARTIFACT_FILES)


class GitHubEvidenceError(RuntimeError):
    """Raised when saved GitHub evidence is incomplete, unsafe, or inconsistent."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise GitHubEvidenceError(f"{name} must be a string-keyed object")
    return value


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise GitHubEvidenceError(f"{name} must be a non-empty string")
    return value


def _positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise GitHubEvidenceError(f"{name} must be a positive integer")
    return value


def _commit(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if COMMIT_RE.fullmatch(text) is None:
        raise GitHubEvidenceError(f"{name} must be a full lowercase Git commit")
    return text


def _timestamp(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    if TIMESTAMP_RE.fullmatch(text) is None:
        raise GitHubEvidenceError(f"{name} must use canonical ISO-8601 UTC Z form")
    try:
        parsed = datetime.fromisoformat(text[:-1] + "+00:00")
    except ValueError as exc:
        raise GitHubEvidenceError(f"{name} is not a valid UTC timestamp") from exc
    if parsed.tzinfo != UTC:
        raise GitHubEvidenceError(f"{name} must be UTC")
    return text


def _repository(value: Any, *, name: str) -> str:
    text = _text(value, name=name)
    repository_name = text.split("/", 1)[1] if "/" in text else ""
    if (
        GITHUB_REPOSITORY_RE.fullmatch(text) is None
        or text.casefold().endswith(".git")
        or repository_name in {".", ".."}
        or ".." in repository_name
    ):
        raise GitHubEvidenceError(f"{name} must be a canonical owner/repository identity")
    return text


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise GitHubEvidenceError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise GitHubEvidenceError(f"non-finite JSON number is forbidden: {token}")


def _basename_reference(source: Path, payload: bytes, *, name: str) -> dict[str, Any]:
    basename = source.name
    if (
        PurePosixPath(basename).name != basename
        or "\\" in basename
        or ":" in basename
        or any(ord(character) < 32 or ord(character) == 127 for character in basename)
    ):
        raise GitHubEvidenceError(f"{name} basename is not portable")
    return {
        "basename": basename,
        "size_bytes": len(payload),
        "file_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _load_api(path: str | Path, *, name: str) -> tuple[Mapping[str, Any], dict[str, Any]]:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise GitHubEvidenceError(f"{name} must be a regular non-symlink file")
    if source.stat().st_size > MAX_INPUT_BYTES:
        raise GitHubEvidenceError(f"{name} exceeds the evidence size cap")
    payload = source.read_bytes()
    if len(payload) > MAX_INPUT_BYTES:
        raise GitHubEvidenceError(f"{name} exceeds the evidence size cap")
    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GitHubEvidenceError(f"{name} must be strict UTF-8 JSON") from exc
    return _mapping(value, name=name), _basename_reference(source, payload, name=name)


def _self_hash(body: dict[str, Any]) -> dict[str, Any]:
    record = dict(body)
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def _new_json(record: dict[str, Any], path: str | Path, *, allowed_root: str | Path) -> None:
    root, destination = _preflight_new_json(path, allowed_root=allowed_root)
    atomic_write_json_new(record, destination, allowed_root=root)


def _preflight_new_json(path: str | Path, *, allowed_root: str | Path) -> tuple[Path, Path]:
    root_path = Path(allowed_root)
    if root_path.is_symlink():
        raise GitHubEvidenceError("allowed output root may not be a symlink")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise GitHubEvidenceError("allowed output root must be an existing directory")
    destination = Path(path)
    if not destination.is_absolute():
        destination = root / destination
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise GitHubEvidenceError("evidence output escapes allowed output root") from exc
    if any(part in {"", ".", ".."} for part in relative.parts):
        raise GitHubEvidenceError("evidence output path is not canonical")
    for count in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:count]).is_symlink():
            raise GitHubEvidenceError("evidence output may not traverse a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    for count in range(1, len(relative.parts) + 1):
        if root.joinpath(*relative.parts[:count]).is_symlink():
            raise GitHubEvidenceError("evidence output may not traverse a symlink")
    resolved_parent = destination.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise GitHubEvidenceError("evidence output escapes allowed output root") from exc
    destination = resolved_parent / destination.name
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite existing file: {destination}")
    return root, destination


def capture_remote_evidence(
    *,
    repository_response_path: str | Path,
    main_ref_response_path: str | Path,
    candidate_commit: str,
    queried_at_utc: str,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Validate saved repository/ref responses and create a canonical remote record."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    queried = _timestamp(queried_at_utc, name="queried_at_utc")
    repository_response, repository_input = _load_api(
        repository_response_path, name="repository API response"
    )
    ref_response, ref_input = _load_api(main_ref_response_path, name="main-ref API response")
    if "temp_clone_token" in repository_response:
        raise GitHubEvidenceError(
            "repository API response contains forbidden temp_clone_token field"
        )
    repository = _repository(repository_response.get("full_name"), name="repository full_name")
    remote_url = f"https://github.com/{repository}"
    expected_api_url = f"https://api.github.com/repos/{repository}"
    private = repository_response.get("private")
    visibility = repository_response.get("visibility")
    if private is True and visibility in {None, "private"}:
        normalized_visibility = "private"
    elif private is False and visibility in {None, "public"}:
        normalized_visibility = "public"
    else:
        raise GitHubEvidenceError("repository visibility fields are inconsistent")
    ref_object = _mapping(ref_response.get("object"), name="main-ref object")
    ref_api_url = f"{expected_api_url}/git/refs/heads/main"
    if (
        repository_response.get("html_url") != remote_url
        or repository_response.get("url") != expected_api_url
        or repository_response.get("default_branch") != "main"
        or ref_response.get("ref") != "refs/heads/main"
        or ref_response.get("url") != ref_api_url
        or ref_object.get("type") != "commit"
        or ref_object.get("sha") != candidate
    ):
        raise GitHubEvidenceError("GitHub repository/ref response does not bind main to candidate")
    record = _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "github_remote_evidence",
            "status": "pass",
            "candidate_commit": candidate,
            "visibility": normalized_visibility,
            "remote_url": remote_url,
            "repository": repository,
            "default_branch": "main",
            "ref_name": "refs/heads/main",
            "ref_commit": candidate,
            "repository_api_url": expected_api_url,
            "ref_api_url": ref_api_url,
            "api_response_files": {
                "repository": repository_input,
                "main_ref": ref_input,
            },
            "queried_at_utc": queried,
        }
    )
    _new_json(record, output_path, allowed_root=allowed_output_root)
    return record


def _member_basename(info: zipfile.ZipInfo, *, candidate: str) -> str:
    raw = info.filename
    if "\\" in raw or any(ord(character) < 32 or ord(character) == 127 for character in raw):
        raise GitHubEvidenceError("CI artifact contains an unsafe member path")
    path = PurePosixPath(raw)
    if (
        path.is_absolute()
        or path.as_posix() != raw
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise GitHubEvidenceError("CI artifact contains a noncanonical member path")
    if info.is_dir():
        raise GitHubEvidenceError("CI artifact may not contain directory entries")
    mode = info.external_attr >> 16
    if stat.S_ISLNK(mode):
        raise GitHubEvidenceError("CI artifact may not contain symlinks")
    basename = path.name
    allowed_prefix = PurePosixPath(".audit") / "release-attestations" / candidate
    if path != PurePosixPath(basename) and path.parent != allowed_prefix:
        raise GitHubEvidenceError("CI artifact member is outside the candidate evidence root")
    if basename not in EXPECTED_ARTIFACT_FILES:
        raise GitHubEvidenceError(f"CI artifact contains an unexpected file: {basename}")
    return basename


def _read_artifact_payload(archive_payload: bytes, *, candidate: str) -> dict[str, bytes]:
    if len(archive_payload) > MAX_ARCHIVE_BYTES:
        raise GitHubEvidenceError("downloaded CI artifact exceeds the archive size cap")
    files: dict[str, bytes] = {}
    total_uncompressed = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive_payload)) as archive:
            members = archive.infolist()
            if len(members) != MAX_ARCHIVE_MEMBERS:
                raise GitHubEvidenceError(
                    "CI artifact member count differs from the exact evidence set"
                )
            for info in members:
                basename = _member_basename(info, candidate=candidate)
                if info.flag_bits & 0x1:
                    raise GitHubEvidenceError("CI artifact may not contain encrypted members")
                if info.file_size > MAX_ARCHIVE_MEMBER_BYTES:
                    raise GitHubEvidenceError("CI artifact member exceeds the evidence size cap")
                total_uncompressed += info.file_size
                if total_uncompressed > MAX_ARCHIVE_BYTES:
                    raise GitHubEvidenceError("CI artifact uncompressed content exceeds the cap")
                if basename in files:
                    raise GitHubEvidenceError(f"CI artifact duplicates {basename}")
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise GitHubEvidenceError("CI artifact member size differs from ZIP metadata")
                files[basename] = payload
    except (OSError, zipfile.BadZipFile) as exc:
        raise GitHubEvidenceError("downloaded CI artifact is not a valid ZIP archive") from exc
    missing = EXPECTED_ARTIFACT_FILES - set(files)
    if missing:
        raise GitHubEvidenceError(f"CI artifact is missing files: {sorted(missing)}")
    if set(files) != EXPECTED_ARTIFACT_FILES:
        raise GitHubEvidenceError("CI artifact file set differs from the exact evidence set")
    return files


def _read_artifact_files(
    archive_path: str | Path, *, candidate: str
) -> tuple[str, int, dict[str, bytes], dict[str, Any]]:
    source = Path(archive_path)
    if source.is_symlink() or not source.is_file():
        raise GitHubEvidenceError("downloaded CI artifact must be a regular non-symlink file")
    if source.stat().st_size > MAX_ARCHIVE_BYTES:
        raise GitHubEvidenceError("downloaded CI artifact exceeds the archive size cap")
    archive_payload = source.read_bytes()
    if (
        len(archive_payload) > MAX_ARCHIVE_BYTES
        or source.is_symlink()
        or not source.is_file()
        or source.stat().st_size != len(archive_payload)
    ):
        raise GitHubEvidenceError("downloaded CI artifact changed or exceeds the archive size cap")
    files = _read_artifact_payload(archive_payload, candidate=candidate)
    return (
        hashlib.sha256(archive_payload).hexdigest(),
        len(archive_payload),
        files,
        _basename_reference(source, archive_payload, name="downloaded CI artifact"),
    )


def _safe_evidence_parent(extraction_root: str | Path) -> Path:
    root_path = Path(extraction_root)
    if root_path.is_symlink():
        raise GitHubEvidenceError("extraction root may not be a symlink")
    root = root_path.resolve(strict=True)
    if not root.is_dir():
        raise GitHubEvidenceError("extraction root must be an existing directory")
    current = root
    for component in (".audit", "release-attestations"):
        current = current / component
        if current.is_symlink():
            raise GitHubEvidenceError("candidate extraction path may not traverse a symlink")
        if current.exists() and not current.is_dir():
            raise GitHubEvidenceError("candidate extraction parent is not a directory")
        current.mkdir(exist_ok=True)
        if current.is_symlink():
            raise GitHubEvidenceError("candidate extraction path may not traverse a symlink")
        resolved = current.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise GitHubEvidenceError("candidate extraction path escapes extraction root") from exc
        if not resolved.is_dir():
            raise GitHubEvidenceError("candidate extraction parent is not a directory")
        current = resolved
    return current


def _publish_artifact_files(
    files: Mapping[str, bytes], *, candidate: str, extraction_root: str | Path
) -> Path:
    parent = _safe_evidence_parent(extraction_root)
    destination_root = parent / candidate
    if os.path.lexists(destination_root):
        raise FileExistsError(f"refusing to overwrite extracted CI evidence: {destination_root}")
    staging = parent / f".{candidate}.partial.{uuid4().hex}"
    staging.mkdir()
    try:
        for basename in sorted(EXPECTED_ARTIFACT_FILES):
            payload = files[basename]
            destination = staging / basename
            with destination.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        if {path.name for path in staging.iterdir()} != EXPECTED_ARTIFACT_FILES:
            raise GitHubEvidenceError("staged CI evidence file set differs before publication")
        for basename in sorted(EXPECTED_ARTIFACT_FILES):
            staged_file = staging / basename
            if staged_file.is_symlink() or not staged_file.is_file():
                raise GitHubEvidenceError("staged CI evidence contains a non-regular file")
            payload = staged_file.read_bytes()
            if payload != files[basename]:
                raise GitHubEvidenceError("staged CI evidence bytes changed before publication")
        staging.rename(destination_root)
    except Exception as exc:
        raise GitHubEvidenceError(
            f"CI evidence publication failed; partial retained at {staging}: {exc}"
        ) from exc
    return destination_root


def _strict_component(payload: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8-sig"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_non_finite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise GitHubEvidenceError(f"{name} must be strict UTF-8 JSON") from exc
    return _mapping(value, name=name)


def _verify_self_hash(record: Mapping[str, Any], *, name: str) -> None:
    body = dict(record)
    observed = body.pop("record_sha256", None)
    if not isinstance(observed, str) or SHA256_RE.fullmatch(observed) is None:
        raise GitHubEvidenceError(f"{name} lacks a lowercase record_sha256")
    if canonical_json_sha256(body) != observed:
        raise GitHubEvidenceError(f"{name} record_sha256 does not reconstruct")


def _validate_component_records(files: Mapping[str, bytes], *, candidate: str) -> None:
    expected_kinds = {
        "repository_scan.json": "git_object_release_scan",
        "secret_scan.json": "gitleaks_secret_scan_attestation",
        "license_audit.json": "python_license_audit",
        "quality_gates.json": "ci_release_quality_gate_evidence",
    }
    records: dict[str, Mapping[str, Any]] = {}
    for basename, expected_kind in expected_kinds.items():
        record = _strict_component(files[basename], name=basename)
        _verify_self_hash(record, name=basename)
        _timestamp(record.get("created_at_utc"), name=f"{basename}.created_at_utc")
        if (
            record.get("schema_version") != SCHEMA_VERSION
            or record.get("record_kind") != expected_kind
            or record.get("status") != "pass"
            or record.get("candidate_commit") != candidate
        ):
            raise GitHubEvidenceError(f"{basename} status, kind, or candidate differs")
        records[basename] = record

    bundle = _strict_component(files["ci_bundle_validation.json"], name="ci_bundle_validation.json")
    _verify_self_hash(bundle, name="ci_bundle_validation.json")
    expected_bundle_keys = {
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
    }
    if set(bundle) != expected_bundle_keys:
        raise GitHubEvidenceError("ci_bundle_validation.json keys differ")
    _timestamp(bundle.get("created_at_utc"), name="ci_bundle_validation.created_at_utc")
    _commit(bundle.get("parent_commit"), name="ci_bundle_validation.parent_commit")
    expected_gates = {
        "repository_scan",
        "secret_scan",
        "license_audit",
        *CI_QUALITY_GATE_NAMES,
    }
    validated_gates = bundle.get("validated_gates")
    evidence = _mapping(bundle.get("evidence"), name="ci_bundle_validation.evidence")
    if (
        bundle.get("schema_version") != SCHEMA_VERSION
        or bundle.get("record_kind") != "ci_release_gate_bundle_validation"
        or bundle.get("status") != "pass"
        or bundle.get("candidate_commit") != candidate
        or bundle.get("ci_evidence_scope") != "within_job_quality_proxy_not_completed_workflow"
        or bundle.get("external_completed_ci_required_for_release_inventory") is not True
        or not isinstance(validated_gates, list)
        or validated_gates != sorted(expected_gates)
        or set(evidence) != expected_gates
    ):
        raise GitHubEvidenceError("ci_bundle_validation.json semantics differ")
    basename_by_gate = {
        "repository_scan": "repository_scan.json",
        "secret_scan": "secret_scan.json",
        "license_audit": "license_audit.json",
        **{name: "quality_gates.json" for name in CI_QUALITY_GATE_NAMES},
    }
    prefix = f".audit/release-attestations/{candidate}/"
    for gate_name, basename in basename_by_gate.items():
        reference = _mapping(evidence[gate_name], name=f"ci_bundle_validation.evidence.{gate_name}")
        if set(reference) != {"path", "size_bytes", "file_sha256"}:
            raise GitHubEvidenceError(f"CI bundle reference keys differ for {gate_name}")
        payload = files[basename]
        if (
            reference.get("path") != f"{prefix}{basename}"
            or reference.get("size_bytes") != len(payload)
            or reference.get("file_sha256") != hashlib.sha256(payload).hexdigest()
        ):
            raise GitHubEvidenceError(f"CI bundle reference differs for {gate_name}")


def validate_ci_archive_payload(archive_payload: bytes, *, candidate_commit: str) -> dict[str, Any]:
    """Reconstruct a retained Actions archive without writing extracted members."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    if not isinstance(archive_payload, bytes):
        raise GitHubEvidenceError("downloaded CI artifact payload must be bytes")
    files = _read_artifact_payload(archive_payload, candidate=candidate)
    _validate_component_records(files, candidate=candidate)
    return {
        "archive_sha256": hashlib.sha256(archive_payload).hexdigest(),
        "size_bytes": len(archive_payload),
        "member_file_sha256s": {
            basename: hashlib.sha256(payload).hexdigest()
            for basename, payload in sorted(files.items())
        },
    }


def capture_ci_evidence(
    *,
    workflow_run_response_path: str | Path,
    artifact_response_path: str | Path,
    downloaded_archive_path: str | Path,
    candidate_commit: str,
    queried_at_utc: str,
    extraction_root: str | Path,
    output_path: str | Path,
    allowed_output_root: str | Path,
) -> dict[str, Any]:
    """Validate completed Actions/API evidence, mirror sanitized files, and attest it."""

    candidate = _commit(candidate_commit, name="candidate_commit")
    queried = _timestamp(queried_at_utc, name="queried_at_utc")
    run, run_input = _load_api(workflow_run_response_path, name="workflow-run API response")
    artifact_list, artifact_input = _load_api(
        artifact_response_path, name="run-artifacts API response"
    )
    run_repository = _mapping(run.get("repository"), name="workflow-run repository")
    repository = _repository(run_repository.get("full_name"), name="workflow-run repository")
    run_id = _positive_int(run.get("id"), name="workflow run id")
    run_attempt = _positive_int(run.get("run_attempt"), name="workflow run attempt")
    expected_run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    if (
        run.get("status") != "completed"
        or run.get("conclusion") != "success"
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != candidate
        or run.get("path") != ".github/workflows/ci.yml"
        or run.get("html_url") != expected_run_url
    ):
        raise GitHubEvidenceError("workflow run is not the completed successful main push")
    artifacts = artifact_list.get("artifacts")
    total_count = _positive_int(artifact_list.get("total_count"), name="artifact total_count")
    if not isinstance(artifacts, list) or total_count != 1 or len(artifacts) != 1:
        raise GitHubEvidenceError("run-artifacts response must contain exactly one artifact")
    artifact = _mapping(artifacts[0], name="release artifact")
    artifact_id = _positive_int(artifact.get("id"), name="artifact id")
    artifact_workflow = _mapping(artifact.get("workflow_run"), name="artifact workflow_run")
    artifact_workflow_id = _positive_int(
        artifact_workflow.get("id"), name="artifact workflow run id"
    )
    expected_artifact_api_url = (
        f"https://api.github.com/repos/{repository}/actions/artifacts/{artifact_id}"
    )
    expected_archive_url = f"{expected_artifact_api_url}/zip"
    expected_artifacts_list_url = (
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/artifacts"
        f"?name=release-security-{candidate}"
    )
    if (
        artifact.get("name") != f"release-security-{candidate}"
        or artifact.get("expired") is not False
        or artifact.get("url") != expected_artifact_api_url
        or artifact.get("archive_download_url") != expected_archive_url
        or artifact_workflow_id != run_id
        or artifact_workflow.get("head_sha") != candidate
    ):
        raise GitHubEvidenceError("Actions artifact is not the candidate run's release bundle")
    archive_sha256, archive_size, files, archive_input = _read_artifact_files(
        downloaded_archive_path,
        candidate=candidate,
    )
    artifact_digest = artifact.get("digest")
    artifact_size = artifact.get("size_in_bytes")
    if (
        artifact_digest != f"sha256:{archive_sha256}"
        or not isinstance(artifact_size, int)
        or isinstance(artifact_size, bool)
        or artifact_size != archive_size
    ):
        raise GitHubEvidenceError("Actions artifact digest or size differs from downloaded archive")
    _validate_component_records(files, candidate=candidate)
    component_hashes = {
        "repository_scan": hashlib.sha256(files["repository_scan.json"]).hexdigest(),
        "secret_scan": hashlib.sha256(files["secret_scan.json"]).hexdigest(),
        "license_audit": hashlib.sha256(files["license_audit.json"]).hexdigest(),
        "quality_gates": hashlib.sha256(files["quality_gates.json"]).hexdigest(),
        "ci_bundle_validation": hashlib.sha256(files["ci_bundle_validation.json"]).hexdigest(),
    }
    if any(SHA256_RE.fullmatch(value) is None for value in component_hashes.values()):
        raise GitHubEvidenceError("captured component hash is invalid")
    member_hashes = {
        basename: hashlib.sha256(payload).hexdigest() for basename, payload in sorted(files.items())
    }
    record = _self_hash(
        {
            "schema_version": SCHEMA_VERSION,
            "record_kind": "ci_run_evidence",
            "status": "pass",
            "candidate_commit": candidate,
            "workflow_status": "completed",
            "conclusion": "success",
            "ci_evidence_scope": "completed_remote_workflow_run",
            "event": "push",
            "head_branch": "main",
            "head_sha": candidate,
            "repository": repository,
            "workflow_path": ".github/workflows/ci.yml",
            "run_id": run_id,
            "run_attempt": run_attempt,
            "workflow_url": expected_run_url,
            "artifact_name": f"release-security-{candidate}",
            "artifact_id": artifact_id,
            "artifact_expired": False,
            "artifact_workflow_run_id": run_id,
            "artifact_workflow_run_head_sha": candidate,
            "artifact_api_url": expected_artifact_api_url,
            "artifacts_list_api_url": expected_artifacts_list_url,
            "artifact_archive_download_url": expected_archive_url,
            "artifact_size_in_bytes": artifact_size,
            "artifact_digest": artifact_digest,
            "downloaded_artifact_archive_sha256": archive_sha256,
            "downloaded_artifact_archive": archive_input,
            "artifact_member_sha256s": member_hashes,
            "release_gate_file_sha256s": component_hashes,
            "api_response_files": {
                "workflow_run": run_input,
                "run_artifacts": artifact_input,
            },
            "queried_at_utc": queried,
        }
    )
    _preflight_new_json(output_path, allowed_root=allowed_output_root)
    _publish_artifact_files(
        files,
        candidate=candidate,
        extraction_root=extraction_root,
    )
    _new_json(record, output_path, allowed_root=allowed_output_root)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    remote = subparsers.add_parser("remote")
    remote.add_argument("--repository-response", type=Path, required=True)
    remote.add_argument("--main-ref-response", type=Path, required=True)
    remote.add_argument("--candidate-commit", required=True)
    remote.add_argument("--queried-at-utc", required=True)
    remote.add_argument("--output", type=Path, required=True)
    remote.add_argument("--allowed-output-root", type=Path, required=True)
    ci = subparsers.add_parser("ci")
    ci.add_argument("--workflow-run-response", type=Path, required=True)
    ci.add_argument("--artifact-response", type=Path, required=True)
    ci.add_argument("--downloaded-archive", type=Path, required=True)
    ci.add_argument("--candidate-commit", required=True)
    ci.add_argument("--queried-at-utc", required=True)
    ci.add_argument("--extraction-root", type=Path, required=True)
    ci.add_argument("--output", type=Path, required=True)
    ci.add_argument("--allowed-output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "remote":
            record = capture_remote_evidence(
                repository_response_path=args.repository_response,
                main_ref_response_path=args.main_ref_response,
                candidate_commit=args.candidate_commit,
                queried_at_utc=args.queried_at_utc,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
        else:
            record = capture_ci_evidence(
                workflow_run_response_path=args.workflow_run_response,
                artifact_response_path=args.artifact_response,
                downloaded_archive_path=args.downloaded_archive,
                candidate_commit=args.candidate_commit,
                queried_at_utc=args.queried_at_utc,
                extraction_root=args.extraction_root,
                output_path=args.output,
                allowed_output_root=args.allowed_output_root,
            )
        print(json.dumps(record, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
