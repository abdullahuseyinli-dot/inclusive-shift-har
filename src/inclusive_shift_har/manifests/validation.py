"""Validation for immutable dataset provenance manifests."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

from .canonical import (
    CanonicalJSONError,
    canonical_json_sha256,
    load_json_strict,
)

MANIFEST_SCHEMA_VERSION = "1.0.0"
_IDENTIFIER_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ALLOWED_HASH_PROVENANCE = {
    "provider_sha256",
    "locally_observed_official_https",
    "not_published",
}
_ALLOWED_LICENSE_STATUS = {"clear", "conflicting_notices"}
PRIMARY_SIX_CHANNEL_ALLOWLISTS: dict[str, list[str]] = {
    "inclusivehar_v4": [
        "motionUserAccelerationX",
        "motionUserAccelerationY",
        "motionUserAccelerationZ",
        "motionRotationRateX",
        "motionRotationRateY",
        "motionRotationRateZ",
    ],
    "uci_har_v1": [
        "body_acc_x",
        "body_acc_y",
        "body_acc_z",
        "body_gyro_x",
        "body_gyro_y",
        "body_gyro_z",
    ],
}


class ManifestValidationError(ValueError):
    """Raised when a caller requires a valid manifest but validation fails."""


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    location: str = "$"

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


@dataclass
class ManifestValidationResult:
    path: str | None
    dataset_id: str | None
    manifest_sha256: str | None
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "errors": [issue.to_dict() for issue in self.errors],
            "manifest_sha256": self.manifest_sha256,
            "path": self.path,
            "valid": self.valid,
            "warnings": [issue.to_dict() for issue in self.warnings],
        }


@dataclass
class ManifestDirectoryReport:
    root: str
    results: list[ManifestValidationResult]
    errors: list[ValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors and bool(self.results) and all(item.valid for item in self.results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "errors": [issue.to_dict() for issue in self.errors],
            "manifest_count": len(self.results),
            "results": [result.to_dict() for result in self.results],
            "root": self.root,
            "valid": self.valid,
        }


def _is_https_url(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and parsed.username is None


def _is_safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    candidate = PurePosixPath(value)
    return (
        not candidate.is_absolute()
        and candidate.as_posix() == value
        and ".." not in candidate.parts
        and "." not in candidate.parts
        and ":" not in candidate.parts[0]
    )


def _mapping(
    value: Any,
    *,
    location: str,
    errors: list[ValidationIssue],
) -> Mapping[str, Any] | None:
    if not isinstance(value, Mapping):
        errors.append(ValidationIssue("TYPE_OBJECT", "must be a JSON object", location))
        return None
    return value


def _non_empty_string(
    value: Any,
    *,
    location: str,
    errors: list[ValidationIssue],
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        errors.append(
            ValidationIssue("TYPE_NONEMPTY_STRING", "must be a non-empty string", location)
        )
        return None
    return value


def _string_list(
    value: Any,
    *,
    location: str,
    errors: list[ValidationIssue],
    exact_length: int | None = None,
) -> list[str] | None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        errors.append(
            ValidationIssue("TYPE_STRING_ARRAY", "must be an array of non-empty strings", location)
        )
        return None
    if exact_length is not None and len(value) != exact_length:
        errors.append(
            ValidationIssue(
                "ARRAY_LENGTH",
                f"must contain exactly {exact_length} entries",
                location,
            )
        )
    if len(set(value)) != len(value):
        errors.append(ValidationIssue("ARRAY_DUPLICATE", "must not contain duplicates", location))
    return value


def validate_manifest(manifest: Any, *, path: str | None = None) -> ManifestValidationResult:
    """Validate a parsed dataset manifest without modifying it."""

    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []
    dataset_id: str | None = None
    manifest_hash: str | None = None
    if not isinstance(manifest, Mapping):
        return ManifestValidationResult(
            path=path,
            dataset_id=None,
            manifest_sha256=None,
            errors=[ValidationIssue("ROOT_OBJECT", "manifest root must be an object")],
        )
    try:
        manifest_hash = canonical_json_sha256(manifest)
    except Exception as exc:
        errors.append(ValidationIssue("CANONICAL_JSON", str(exc)))

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append(
            ValidationIssue(
                "SCHEMA_VERSION",
                f"schema_version must equal {MANIFEST_SCHEMA_VERSION!r}",
                "$.schema_version",
            )
        )
    if manifest.get("manifest_kind") != "dataset":
        errors.append(
            ValidationIssue(
                "MANIFEST_KIND", "manifest_kind must equal 'dataset'", "$.manifest_kind"
            )
        )
    dataset_id = _non_empty_string(
        manifest.get("dataset_id"), location="$.dataset_id", errors=errors
    )
    if dataset_id is not None and not _IDENTIFIER_RE.fullmatch(dataset_id):
        errors.append(
            ValidationIssue(
                "DATASET_ID", "dataset_id contains unsupported characters", "$.dataset_id"
            )
        )
    _non_empty_string(manifest.get("title"), location="$.title", errors=errors)
    if manifest.get("record_status") != "locked_starter":
        errors.append(
            ValidationIssue(
                "RECORD_STATUS", "record_status must equal 'locked_starter'", "$.record_status"
            )
        )

    release = _mapping(manifest.get("release"), location="$.release", errors=errors)
    if release is not None:
        _non_empty_string(release.get("version"), location="$.release.version", errors=errors)
        _non_empty_string(release.get("doi"), location="$.release.doi", errors=errors)
        if not _is_https_url(release.get("landing_url")):
            errors.append(
                ValidationIssue(
                    "HTTPS_URL", "landing_url must be an HTTPS URL", "$.release.landing_url"
                )
            )
        _non_empty_string(
            release.get("published_date"), location="$.release.published_date", errors=errors
        )

    provenance = _mapping(manifest.get("provenance"), location="$.provenance", errors=errors)
    if provenance is not None:
        for key in ("official_repository_url", "metadata_endpoint"):
            if not _is_https_url(provenance.get(key)):
                errors.append(
                    ValidationIssue(
                        "HTTPS_URL", f"{key} must be an HTTPS URL", f"$.provenance.{key}"
                    )
                )
        _non_empty_string(
            provenance.get("metadata_verified_date"),
            location="$.provenance.metadata_verified_date",
            errors=errors,
        )

    licence = _mapping(manifest.get("license"), location="$.license", errors=errors)
    if licence is not None:
        if licence.get("status") not in _ALLOWED_LICENSE_STATUS:
            errors.append(
                ValidationIssue(
                    "LICENSE_STATUS",
                    f"status must be one of {sorted(_ALLOWED_LICENSE_STATUS)}",
                    "$.license.status",
                )
            )
        notices = licence.get("notices")
        if not isinstance(notices, list) or not notices:
            errors.append(
                ValidationIssue(
                    "LICENSE_NOTICES",
                    "at least one license notice is required",
                    "$.license.notices",
                )
            )
        elif any(not isinstance(item, Mapping) for item in notices):
            errors.append(
                ValidationIssue(
                    "LICENSE_NOTICES", "each license notice must be an object", "$.license.notices"
                )
            )
        if licence.get("redistribution_policy") != "do_not_redistribute_raw":
            errors.append(
                ValidationIssue(
                    "REDISTRIBUTION_POLICY",
                    "raw-data redistribution policy must be 'do_not_redistribute_raw'",
                    "$.license.redistribution_policy",
                )
            )

    artifacts = manifest.get("artifacts")
    artifact_ids: set[str] = set()
    artifact_storage_paths: set[str] = set()
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(
            ValidationIssue("ARTIFACTS", "artifacts must be a non-empty array", "$.artifacts")
        )
    else:
        for index, artifact_value in enumerate(artifacts):
            location = f"$.artifacts[{index}]"
            artifact = _mapping(artifact_value, location=location, errors=errors)
            if artifact is None:
                continue
            artifact_id = _non_empty_string(
                artifact.get("artifact_id"), location=f"{location}.artifact_id", errors=errors
            )
            if artifact_id is not None:
                if artifact_id in artifact_ids:
                    errors.append(
                        ValidationIssue(
                            "ARTIFACT_ID_DUPLICATE",
                            "artifact_id must be unique",
                            f"{location}.artifact_id",
                        )
                    )
                artifact_ids.add(artifact_id)
            for key in ("role", "provider_filename", "media_type", "git_policy"):
                _non_empty_string(artifact.get(key), location=f"{location}.{key}", errors=errors)
            if artifact.get("git_policy") != "exclude_raw_from_git":
                errors.append(
                    ValidationIssue(
                        "GIT_POLICY",
                        "raw artifacts must use git_policy 'exclude_raw_from_git'",
                        f"{location}.git_policy",
                    )
                )
            if not _is_https_url(artifact.get("download_url")):
                errors.append(
                    ValidationIssue(
                        "HTTPS_URL", "download_url must be an HTTPS URL", f"{location}.download_url"
                    )
                )
            if not _is_safe_relative_path(artifact.get("storage_path")):
                errors.append(
                    ValidationIssue(
                        "STORAGE_PATH",
                        "storage_path must be a safe POSIX-style relative path",
                        f"{location}.storage_path",
                    )
                )
            else:
                storage_key = artifact["storage_path"].casefold()
                if storage_key in artifact_storage_paths:
                    errors.append(
                        ValidationIssue(
                            "STORAGE_PATH_COLLISION",
                            "artifact storage paths must be unique under case-folding",
                            f"{location}.storage_path",
                        )
                    )
                artifact_storage_paths.add(storage_key)
            size = artifact.get("expected_size_bytes")
            if size is not None and (
                isinstance(size, bool) or not isinstance(size, int) or size <= 0
            ):
                errors.append(
                    ValidationIssue(
                        "EXPECTED_SIZE",
                        "expected_size_bytes must be a positive integer or null",
                        f"{location}.expected_size_bytes",
                    )
                )
            sha256 = artifact.get("expected_sha256")
            if sha256 is not None and (
                not isinstance(sha256, str) or not _SHA256_RE.fullmatch(sha256)
            ):
                errors.append(
                    ValidationIssue(
                        "EXPECTED_SHA256",
                        "expected_sha256 must be a lowercase 64-character SHA-256 or null",
                        f"{location}.expected_sha256",
                    )
                )
            hash_provenance = artifact.get("hash_provenance")
            if hash_provenance not in _ALLOWED_HASH_PROVENANCE:
                errors.append(
                    ValidationIssue(
                        "HASH_PROVENANCE",
                        f"hash_provenance must be one of {sorted(_ALLOWED_HASH_PROVENANCE)}",
                        f"{location}.hash_provenance",
                    )
                )
            if hash_provenance == "provider_sha256" and sha256 is None:
                errors.append(
                    ValidationIssue(
                        "PROVIDER_HASH_MISSING",
                        "provider_sha256 provenance requires expected_sha256",
                        location,
                    )
                )
            if sha256 is None:
                warnings.append(
                    ValidationIssue(
                        "UNPINNED_PROVIDER_ARTIFACT",
                        "provider publishes no SHA-256; first official retrieval must compute and lock one",
                        location,
                    )
                )

    _mapping(manifest.get("expected_data"), location="$.expected_data", errors=errors)
    feature_policy = _mapping(
        manifest.get("feature_policy"), location="$.feature_policy", errors=errors
    )
    if feature_policy is not None:
        if feature_policy.get("model_input_policy") != "exact_allowlist_only":
            errors.append(
                ValidationIssue(
                    "MODEL_INPUT_POLICY",
                    "model_input_policy must equal 'exact_allowlist_only'",
                    "$.feature_policy.model_input_policy",
                )
            )
        channels = _string_list(
            feature_policy.get("primary_six_channel_allowlist"),
            location="$.feature_policy.primary_six_channel_allowlist",
            errors=errors,
            exact_length=6,
        )
        expected_channels = PRIMARY_SIX_CHANNEL_ALLOWLISTS.get(dataset_id or "")
        if channels is not None and expected_channels is not None and channels != expected_channels:
            errors.append(
                ValidationIssue(
                    "PRIMARY_CHANNEL_ALLOWLIST",
                    f"primary channel allowlist must exactly equal the locked {dataset_id} profile",
                    "$.feature_policy.primary_six_channel_allowlist",
                )
            )
        exclusions = _mapping(
            feature_policy.get("sensitive_and_non_model_exclusions"),
            location="$.feature_policy.sensitive_and_non_model_exclusions",
            errors=errors,
        )
        if exclusions is not None:
            exact_names = _string_list(
                exclusions.get("exact_names"),
                location="$.feature_policy.sensitive_and_non_model_exclusions.exact_names",
                errors=errors,
            )
            prefixes = _string_list(
                exclusions.get("casefold_prefixes"),
                location="$.feature_policy.sensitive_and_non_model_exclusions.casefold_prefixes",
                errors=errors,
            )
            if exact_names is not None:
                required = {"disabled", "userid", "disability", "assistive_device"}
                missing = required - {name.casefold() for name in exact_names}
                if missing:
                    errors.append(
                        ValidationIssue(
                            "SENSITIVE_EXCLUSIONS",
                            f"missing required sensitive exclusions: {sorted(missing)}",
                            "$.feature_policy.sensitive_and_non_model_exclusions.exact_names",
                        )
                    )
                expected_data = manifest.get("expected_data")
                is_pre_audit_snapshot = (
                    isinstance(expected_data, Mapping)
                    and expected_data.get("claims_status")
                    == "provenance_verified_schema_pending_data_gate"
                )
                if dataset_id == "inclusivehar_v4" and not is_pre_audit_snapshot:
                    required_label_spellings = {"Label", "label"}
                    missing_label_spellings = required_label_spellings - set(exact_names)
                    if missing_label_spellings:
                        errors.append(
                            ValidationIssue(
                                "RAW_LABEL_EXCLUSIONS",
                                "InclusiveHAR exclusions must retain both the observed lowercase label and legacy uppercase Label spellings",
                                "$.feature_policy.sensitive_and_non_model_exclusions.exact_names",
                            )
                        )
                if channels is not None and {item.casefold() for item in channels} & {
                    item.casefold() for item in exact_names
                }:
                    errors.append(
                        ValidationIssue(
                            "ALLOWLIST_EXCLUSION_OVERLAP",
                            "model allowlist overlaps an excluded name",
                            "$.feature_policy",
                        )
                    )
            if prefixes is not None:
                required_prefixes = {"gps", "location"}
                missing_prefixes = required_prefixes - {prefix.casefold() for prefix in prefixes}
                if missing_prefixes:
                    errors.append(
                        ValidationIssue(
                            "GPS_EXCLUSIONS",
                            f"missing required GPS/location prefixes: {sorted(missing_prefixes)}",
                            "$.feature_policy.sensitive_and_non_model_exclusions.casefold_prefixes",
                        )
                    )
                if channels is not None:
                    forbidden_prefixes = tuple(prefix.casefold() for prefix in prefixes)
                    forbidden_channels = [
                        channel
                        for channel in channels
                        if channel.casefold().startswith(forbidden_prefixes)
                    ]
                    if forbidden_channels:
                        errors.append(
                            ValidationIssue(
                                "GPS_FEATURE",
                                f"model allowlist contains GPS/sensitive-prefixed channels: {forbidden_channels}",
                                "$.feature_policy.primary_six_channel_allowlist",
                            )
                        )

    gate_policy = _mapping(manifest.get("gate_policy"), location="$.gate_policy", errors=errors)
    if gate_policy is not None:
        required_gates = {
            "raw_data_read_requires": "raw_data_read_access_approved",
            "split_build_requires": "data_audit_pass",
            "training_requires": "split_audit_pass",
            "confirmatory_evaluation_requires": "final_evaluation_unlock",
        }
        for key, expected in required_gates.items():
            if gate_policy.get(key) != expected:
                errors.append(
                    ValidationIssue(
                        "GATE_POLICY",
                        f"{key} must equal {expected!r}",
                        f"$.gate_policy.{key}",
                    )
                )

    return ManifestValidationResult(
        path=path,
        dataset_id=dataset_id,
        manifest_sha256=manifest_hash,
        errors=errors,
        warnings=warnings,
    )


def validate_manifest_file(path: str | Path) -> ManifestValidationResult:
    """Load and validate one manifest file."""

    source = Path(path)
    try:
        value = load_json_strict(source)
    except (CanonicalJSONError, OSError) as exc:
        return ManifestValidationResult(
            path=str(source),
            dataset_id=None,
            manifest_sha256=None,
            errors=[ValidationIssue("LOAD_JSON", str(exc))],
        )
    return validate_manifest(value, path=str(source))


def require_valid_manifest(path: str | Path) -> tuple[Mapping[str, Any], ManifestValidationResult]:
    """Return a parsed valid manifest or raise :class:`ManifestValidationError`."""

    result = validate_manifest_file(path)
    if not result.valid:
        messages = "; ".join(f"{issue.location}: {issue.message}" for issue in result.errors)
        raise ManifestValidationError(f"invalid dataset manifest {path}: {messages}")
    value = load_json_strict(path)
    assert isinstance(value, Mapping)
    return value, result


def validate_manifest_directory(root: str | Path) -> ManifestDirectoryReport:
    """Validate all ``*.json`` dataset manifests below *root* deterministically."""

    directory = Path(root)
    if not directory.exists():
        return ManifestDirectoryReport(
            root=str(directory),
            results=[],
            errors=[
                ValidationIssue(
                    "MANIFEST_ROOT_MISSING", "manifest root does not exist", str(directory)
                )
            ],
        )
    if not directory.is_dir():
        return ManifestDirectoryReport(
            root=str(directory),
            results=[],
            errors=[
                ValidationIssue(
                    "MANIFEST_ROOT_TYPE", "manifest root is not a directory", str(directory)
                )
            ],
        )
    paths = sorted(directory.rglob("*.json"), key=lambda item: item.as_posix().casefold())
    if not paths:
        return ManifestDirectoryReport(
            root=str(directory),
            results=[],
            errors=[ValidationIssue("NO_MANIFESTS", "no JSON manifests found", str(directory))],
        )
    results = [validate_manifest_file(path) for path in paths]
    ids = [result.dataset_id for result in results if result.dataset_id is not None]
    errors: list[ValidationIssue] = []
    duplicates = sorted({dataset for dataset in ids if ids.count(dataset) > 1})
    if duplicates:
        errors.append(
            ValidationIssue(
                "DUPLICATE_DATASET_ID",
                f"dataset_id appears in multiple manifests: {duplicates}",
                str(directory),
            )
        )
    return ManifestDirectoryReport(root=str(directory), results=results, errors=errors)
