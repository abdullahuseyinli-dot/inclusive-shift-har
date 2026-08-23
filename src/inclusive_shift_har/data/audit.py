"""Dry-run and explicitly gated read-only data integrity audits."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.manifests.validation import ManifestValidationError, require_valid_manifest

READ_ACCESS_GATE = "raw_data_read_access"
READ_ACCESS_APPROVAL = "approved"


class DataGateError(PermissionError):
    """Raised when a requested data read has not been explicitly unlocked."""


class DataAuditMode(StrEnum):
    DRY_RUN = "dry-run"
    READ_ONLY = "read-only"


@dataclass(frozen=True)
class ArtifactObservation:
    artifact_id: str
    storage_path: str
    status: str
    expected_size_bytes: int | None
    observed_size_bytes: int | None = None
    expected_sha256: str | None = None
    observed_sha256: str | None = None
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "errors": list(self.errors),
            "expected_sha256": self.expected_sha256,
            "expected_size_bytes": self.expected_size_bytes,
            "observed_sha256": self.observed_sha256,
            "observed_size_bytes": self.observed_size_bytes,
            "status": self.status,
            "storage_path": self.storage_path,
        }


@dataclass
class DataAuditReport:
    dataset_id: str
    manifest_path: str
    manifest_sha256: str
    mode: str
    data_access: str
    scope: str
    primary_six_channel_allowlist: list[str]
    sensitive_and_non_model_exclusions: Mapping[str, Any]
    observations: list[ArtifactObservation] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors and all(not item.errors for item in self.observations)

    def _payload(self) -> dict[str, Any]:
        return {
            "data_access": self.data_access,
            "dataset_id": self.dataset_id,
            "errors": self.errors,
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "mode": self.mode,
            "observations": [item.to_dict() for item in self.observations],
            "primary_six_channel_allowlist": self.primary_six_channel_allowlist,
            "scope": self.scope,
            "sensitive_and_non_model_exclusions": dict(self.sensitive_and_non_model_exclusions),
            "valid": self.valid,
            "warnings": self.warnings,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["report_sha256"] = canonical_json_sha256(payload)
        return payload


def _safe_artifact_path(data_root: Path, storage_path: str) -> Path:
    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"data root is not a directory: {root}")
    candidate = (root / Path(storage_path)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"artifact path escapes data root: {storage_path}") from exc
    return candidate


def verify_read_access_gate(
    gate_record_path: str | os.PathLike[str],
    *,
    dataset_id: str,
    manifest_sha256: str,
) -> Mapping[str, Any]:
    gate = load_json_strict(gate_record_path)
    if not isinstance(gate, Mapping):
        raise DataGateError("raw-data read gate record must be a JSON object")
    expected = {
        "schema_version": "1.0.0",
        "gate": READ_ACCESS_GATE,
        "status": READ_ACCESS_APPROVAL,
        "dataset_id": dataset_id,
        "manifest_sha256": manifest_sha256,
    }
    mismatches = [key for key, value in expected.items() if gate.get(key) != value]
    if mismatches:
        raise DataGateError(f"raw-data read gate is absent or does not match: {mismatches}")
    if not isinstance(gate.get("approved_at_utc"), str) or not gate["approved_at_utc"]:
        raise DataGateError("raw-data read gate lacks approved_at_utc")
    return gate


def audit_manifest_data(
    manifest_path: str | os.PathLike[str],
    *,
    mode: DataAuditMode | str = DataAuditMode.DRY_RUN,
    data_root: str | os.PathLike[str] | None = None,
    gate_record_path: str | os.PathLike[str] | None = None,
) -> DataAuditReport:
    """Plan an audit or verify raw artifact integrity without modifying data.

    The Stage 2 read-only implementation intentionally stops at whole-file size
    and checksum verification.  Row/schema/trial inspection belongs to the later
    data gate and must extend this report without changing its access semantics.
    """

    selected_mode = mode if isinstance(mode, DataAuditMode) else DataAuditMode(mode)
    try:
        manifest, validation = require_valid_manifest(Path(manifest_path))
    except ManifestValidationError:
        raise
    assert validation.manifest_sha256 is not None
    dataset_id = str(manifest["dataset_id"])
    feature_policy = manifest["feature_policy"]
    report = DataAuditReport(
        dataset_id=dataset_id,
        manifest_path=str(Path(manifest_path)),
        manifest_sha256=validation.manifest_sha256,
        mode=selected_mode.value,
        data_access="none"
        if selected_mode is DataAuditMode.DRY_RUN
        else "read_bytes_integrity_only",
        scope="artifact_plan_only"
        if selected_mode is DataAuditMode.DRY_RUN
        else "whole_file_integrity_only",
        primary_six_channel_allowlist=list(feature_policy["primary_six_channel_allowlist"]),
        sensitive_and_non_model_exclusions=feature_policy["sensitive_and_non_model_exclusions"],
    )

    root: Path | None
    if selected_mode is DataAuditMode.READ_ONLY:
        if data_root is None:
            raise DataGateError("read-only audit requires an explicit data_root")
        if gate_record_path is None:
            raise DataGateError("read-only audit requires a matching raw-data read gate record")
        verify_read_access_gate(
            gate_record_path,
            dataset_id=dataset_id,
            manifest_sha256=validation.manifest_sha256,
        )
        root = Path(data_root)
    else:
        root = Path(data_root) if data_root is not None else None

    for artifact in manifest["artifacts"]:
        expected_size = artifact.get("expected_size_bytes")
        expected_hash = artifact.get("expected_sha256")
        if selected_mode is DataAuditMode.DRY_RUN:
            report.observations.append(
                ArtifactObservation(
                    artifact_id=artifact["artifact_id"],
                    storage_path=artifact["storage_path"],
                    status="planned_no_filesystem_access",
                    expected_size_bytes=expected_size,
                    expected_sha256=expected_hash,
                )
            )
            continue

        assert root is not None
        errors: list[str] = []
        try:
            path = _safe_artifact_path(root, artifact["storage_path"])
        except (OSError, ValueError) as exc:
            report.observations.append(
                ArtifactObservation(
                    artifact_id=artifact["artifact_id"],
                    storage_path=artifact["storage_path"],
                    status="invalid_path",
                    expected_size_bytes=expected_size,
                    expected_sha256=expected_hash,
                    errors=(str(exc),),
                )
            )
            continue
        if path.is_symlink():
            errors.append("symbolic links are forbidden for raw artifacts")
        if not path.is_file():
            errors.append("raw artifact is missing or not a regular file")
            observed_size = None
            observed_hash = None
        else:
            observed_size = path.stat().st_size
            observed_hash = sha256_file(path)
            if expected_size is not None and observed_size != expected_size:
                errors.append(f"size mismatch: expected {expected_size}, observed {observed_size}")
            if expected_hash is not None and observed_hash != expected_hash:
                errors.append(
                    f"SHA-256 mismatch: expected {expected_hash}, observed {observed_hash}"
                )
            if expected_hash is None:
                report.warnings.append(
                    f"{artifact['artifact_id']}: provider did not publish a hash; observed SHA-256 must be reviewed and locked"
                )
        report.observations.append(
            ArtifactObservation(
                artifact_id=artifact["artifact_id"],
                storage_path=artifact["storage_path"],
                status="pass" if not errors else "fail",
                expected_size_bytes=expected_size,
                observed_size_bytes=observed_size,
                expected_sha256=expected_hash,
                observed_sha256=observed_hash,
                errors=tuple(errors),
            )
        )
    for observation in report.observations:
        report.errors.extend(
            f"{observation.artifact_id}: {message}" for message in observation.errors
        )
    if selected_mode is DataAuditMode.DRY_RUN:
        report.warnings.append("dry-run performed no filesystem or raw-data access")
    else:
        report.warnings.append(
            "row/schema/trial audit is not implemented in the Stage 2 integrity-only command"
        )
    return report
