"""Read-only validation of result artifact manifests and referenced files."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

ARTIFACT_SCHEMA_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_VISIBLE_STATUSES = {"complete", "conditional", "failed", "quarantined"}
CHECKPOINT_REQUIRED_FILE_ROLES = frozenset(
    {
        "environment_metadata",
        "label_schema",
        "model_state",
        "normalization",
        "optimizer_state",
        "preprocessing",
        "rng_states",
    }
)


@dataclass(frozen=True)
class ArtifactValidationIssue:
    code: str
    message: str
    location: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


@dataclass
class ArtifactManifestResult:
    manifest_path: str
    artifact_id: str | None
    artifact_manifest_sha256: str | None
    checked_files: int = 0
    errors: list[ArtifactValidationIssue] = field(default_factory=list)
    warnings: list[ArtifactValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "artifact_manifest_sha256": self.artifact_manifest_sha256,
            "checked_files": self.checked_files,
            "errors": [item.to_dict() for item in self.errors],
            "manifest_path": self.manifest_path,
            "valid": self.valid,
            "warnings": [item.to_dict() for item in self.warnings],
        }


@dataclass
class ArtifactValidationReport:
    artifact_root: str
    results: list[ArtifactManifestResult]
    errors: list[ArtifactValidationIssue] = field(default_factory=list)
    warnings: list[ArtifactValidationIssue] = field(default_factory=list)

    @property
    def checked_manifests(self) -> int:
        return len(self.results)

    @property
    def checked_files(self) -> int:
        return sum(result.checked_files for result in self.results)

    @property
    def valid(self) -> bool:
        return not self.errors and all(result.valid for result in self.results)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "artifact_root": self.artifact_root,
            "checked_files": self.checked_files,
            "checked_manifests": self.checked_manifests,
            "errors": [item.to_dict() for item in self.errors],
            "results": [item.to_dict() for item in self.results],
            "valid": self.valid,
            "warnings": [item.to_dict() for item in self.warnings],
        }
        payload["report_sha256"] = canonical_json_sha256(payload)
        return payload


def _safe_relative_path(value: Any) -> bool:
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


def _resolve_referenced_file(root: Path, relative_path: str) -> Path:
    resolved_root = root.resolve(strict=True)
    candidate = (resolved_root / Path(relative_path)).resolve(strict=False)
    try:
        candidate.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"referenced path escapes artifact root: {relative_path}") from exc
    return candidate


def validate_artifact_manifest_file(
    manifest_path: str | Path,
    *,
    artifact_root: str | Path,
) -> ArtifactManifestResult:
    source = Path(manifest_path)
    errors: list[ArtifactValidationIssue] = []
    warnings: list[ArtifactValidationIssue] = []
    try:
        manifest = load_json_strict(source)
    except Exception as exc:
        return ArtifactManifestResult(
            manifest_path=str(source),
            artifact_id=None,
            artifact_manifest_sha256=None,
            errors=[ArtifactValidationIssue("LOAD_JSON", str(exc), str(source))],
        )
    if not isinstance(manifest, Mapping):
        return ArtifactManifestResult(
            manifest_path=str(source),
            artifact_id=None,
            artifact_manifest_sha256=None,
            errors=[
                ArtifactValidationIssue(
                    "ROOT_OBJECT", "artifact manifest root must be an object", "$"
                )
            ],
        )
    manifest_hash = canonical_json_sha256(manifest)
    artifact_id = (
        manifest.get("artifact_id") if isinstance(manifest.get("artifact_id"), str) else None
    )
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        errors.append(
            ArtifactValidationIssue(
                "SCHEMA_VERSION",
                f"schema_version must equal {ARTIFACT_SCHEMA_VERSION!r}",
                "$.schema_version",
            )
        )
    if manifest.get("manifest_kind") != "artifact":
        errors.append(
            ArtifactValidationIssue(
                "MANIFEST_KIND", "manifest_kind must equal 'artifact'", "$.manifest_kind"
            )
        )
    if artifact_id is None or not artifact_id:
        errors.append(
            ArtifactValidationIssue(
                "ARTIFACT_ID", "artifact_id must be a non-empty string", "$.artifact_id"
            )
        )
    artifact_type = manifest.get("artifact_type")
    if not isinstance(artifact_type, str) or not artifact_type:
        errors.append(
            ArtifactValidationIssue(
                "ARTIFACT_TYPE", "artifact_type must be a non-empty string", "$.artifact_type"
            )
        )
    status = manifest.get("status")
    if status not in _VISIBLE_STATUSES:
        errors.append(
            ArtifactValidationIssue(
                "STATUS",
                f"status must be one of {sorted(_VISIBLE_STATUSES)}",
                "$.status",
            )
        )
    provenance = manifest.get("provenance")
    if not isinstance(provenance, Mapping):
        errors.append(
            ArtifactValidationIssue("PROVENANCE", "provenance must be an object", "$.provenance")
        )
    else:
        required_provenance = ["code_commit", "configuration_sha256", "dataset_manifest_sha256"]
        if artifact_type == "checkpoint":
            required_provenance.append("split_manifest_sha256")
        for key in required_provenance:
            value = provenance.get(key)
            if not isinstance(value, str) or not value:
                errors.append(
                    ArtifactValidationIssue(
                        "PROVENANCE_FIELD",
                        f"{key} must be a non-empty string",
                        f"$.provenance.{key}",
                    )
                )
            elif key.endswith("_sha256") and not _SHA256_RE.fullmatch(value):
                errors.append(
                    ArtifactValidationIssue(
                        "PROVENANCE_SHA256",
                        f"{key} must be a lowercase 64-character SHA-256",
                        f"$.provenance.{key}",
                    )
                )

    files = manifest.get("files")
    if not isinstance(files, list):
        errors.append(ArtifactValidationIssue("FILES", "files must be an array", "$.files"))
        files = []
    if status in {"complete", "conditional"} and not files:
        errors.append(
            ArtifactValidationIssue(
                "FILES_REQUIRED", "complete/conditional artifacts must reference files", "$.files"
            )
        )
    seen_paths: set[str] = set()
    seen_roles: set[str] = set()
    checked_files = 0
    root = Path(artifact_root)
    for index, file_entry in enumerate(files):
        location = f"$.files[{index}]"
        if not isinstance(file_entry, Mapping):
            errors.append(
                ArtifactValidationIssue("FILE_ENTRY", "file entry must be an object", location)
            )
            continue
        role = file_entry.get("role")
        if not isinstance(role, str) or not role:
            errors.append(
                ArtifactValidationIssue(
                    "FILE_ROLE", "role must be a non-empty string", f"{location}.role"
                )
            )
        elif role in seen_roles:
            errors.append(
                ArtifactValidationIssue(
                    "FILE_ROLE_DUPLICATE", "file roles must be unique", f"{location}.role"
                )
            )
        else:
            seen_roles.add(role)
        relative_path_value = file_entry.get("path")
        if not isinstance(relative_path_value, str) or not _safe_relative_path(relative_path_value):
            errors.append(
                ArtifactValidationIssue(
                    "FILE_PATH",
                    "path must be a safe POSIX-style relative path",
                    f"{location}.path",
                )
            )
            continue
        relative_path = relative_path_value
        collision_key = relative_path.casefold()
        if collision_key in seen_paths:
            errors.append(
                ArtifactValidationIssue(
                    "FILE_PATH_COLLISION",
                    "file paths must be unique under case-folding",
                    f"{location}.path",
                )
            )
            continue
        seen_paths.add(collision_key)
        expected_size = file_entry.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            errors.append(
                ArtifactValidationIssue(
                    "FILE_SIZE",
                    "size_bytes must be a non-negative integer",
                    f"{location}.size_bytes",
                )
            )
            continue
        expected_hash = file_entry.get("sha256")
        if not isinstance(expected_hash, str) or not _SHA256_RE.fullmatch(expected_hash):
            errors.append(
                ArtifactValidationIssue(
                    "FILE_SHA256",
                    "sha256 must be a lowercase 64-character SHA-256",
                    f"{location}.sha256",
                )
            )
            continue
        try:
            referenced = _resolve_referenced_file(root, relative_path)
        except (OSError, ValueError) as exc:
            errors.append(ArtifactValidationIssue("FILE_PATH", str(exc), f"{location}.path"))
            continue
        if referenced.is_symlink():
            errors.append(
                ArtifactValidationIssue(
                    "FILE_SYMLINK", "referenced files may not be symlinks", relative_path
                )
            )
            continue
        if not referenced.is_file():
            errors.append(
                ArtifactValidationIssue("FILE_MISSING", "referenced file is missing", relative_path)
            )
            continue
        checked_files += 1
        observed_size = referenced.stat().st_size
        if observed_size != expected_size:
            errors.append(
                ArtifactValidationIssue(
                    "FILE_SIZE_MISMATCH",
                    f"expected {expected_size}, observed {observed_size}",
                    relative_path,
                )
            )
        observed_hash = sha256_file(referenced)
        if observed_hash != expected_hash:
            errors.append(
                ArtifactValidationIssue(
                    "FILE_HASH_MISMATCH",
                    f"expected {expected_hash}, observed {observed_hash}",
                    relative_path,
                )
            )
    if artifact_type == "checkpoint" and status in {"complete", "conditional"}:
        missing_roles = sorted(CHECKPOINT_REQUIRED_FILE_ROLES - seen_roles)
        if missing_roles:
            errors.append(
                ArtifactValidationIssue(
                    "CHECKPOINT_COMPONENT_MISSING",
                    f"checkpoint is missing required file roles: {missing_roles}",
                    "$.files",
                )
            )
    if status in {"failed", "quarantined"}:
        warnings.append(
            ArtifactValidationIssue(
                "VISIBLE_NONPASS_STATUS",
                f"artifact status remains visibly {status}",
                "$.status",
            )
        )
    return ArtifactManifestResult(
        manifest_path=str(source),
        artifact_id=artifact_id,
        artifact_manifest_sha256=manifest_hash,
        checked_files=checked_files,
        errors=errors,
        warnings=warnings,
    )


def validate_artifact_directory(
    artifact_root: str | Path,
    *,
    require_artifacts: bool = False,
) -> ArtifactValidationReport:
    """Validate artifact manifests recursively without modifying the tree."""

    root = Path(artifact_root)
    if not root.exists():
        issue = ArtifactValidationIssue(
            "ARTIFACT_ROOT_MISSING", "artifact root does not exist", str(root)
        )
        return ArtifactValidationReport(
            artifact_root=str(root),
            results=[],
            errors=[issue] if require_artifacts else [],
            warnings=[] if require_artifacts else [issue],
        )
    if not root.is_dir():
        return ArtifactValidationReport(
            artifact_root=str(root),
            results=[],
            errors=[
                ArtifactValidationIssue(
                    "ARTIFACT_ROOT_TYPE", "artifact root is not a directory", str(root)
                )
            ],
        )
    discovered = set(root.rglob("artifact_manifest.json")) | set(root.rglob("*.artifact.json"))
    paths = sorted(discovered, key=lambda item: item.as_posix().casefold())
    if not paths:
        issue = ArtifactValidationIssue(
            "NO_ARTIFACT_MANIFESTS", "no artifact manifests found", str(root)
        )
        return ArtifactValidationReport(
            artifact_root=str(root),
            results=[],
            errors=[issue] if require_artifacts else [],
            warnings=[] if require_artifacts else [issue],
        )
    results = [validate_artifact_manifest_file(path, artifact_root=root) for path in paths]
    ids = [result.artifact_id for result in results if result.artifact_id is not None]
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    errors: list[ArtifactValidationIssue] = []
    if duplicates:
        errors.append(
            ArtifactValidationIssue(
                "DUPLICATE_ARTIFACT_ID",
                f"artifact_id appears more than once: {duplicates}",
                str(root),
            )
        )
    return ArtifactValidationReport(artifact_root=str(root), results=results, errors=errors)
