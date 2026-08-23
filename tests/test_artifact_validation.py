"""Artifact-lineage and checkpoint-completeness tests."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

from inclusive_shift_har.artifacts.validation import (
    validate_artifact_directory,
    validate_artifact_manifest_file,
)

from ._synthetic import (
    CHECKPOINT_FILE_ROLES,
    materialize_checkpoint_artifact,
    synthetic_checkpoint_manifest,
)


def test_complete_synthetic_checkpoint_validates(tmp_path: Path) -> None:
    manifest_path, _ = materialize_checkpoint_artifact(tmp_path)
    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    report = validate_artifact_directory(tmp_path, require_artifacts=True)

    assert result.valid, result.to_dict()
    assert result.checked_files == len(CHECKPOINT_FILE_ROLES)
    assert report.valid, report.to_dict()
    assert report.checked_manifests == 1
    assert report.checked_files == len(CHECKPOINT_FILE_ROLES)
    assert report.to_dict() == report.to_dict()


@pytest.mark.parametrize("missing_role", CHECKPOINT_FILE_ROLES)
def test_checkpoint_rejects_missing_required_component(
    tmp_path: Path,
    missing_role: str,
) -> None:
    manifest_path, manifest = materialize_checkpoint_artifact(tmp_path)
    manifest["files"] = [entry for entry in manifest["files"] if entry["role"] != missing_role]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert not result.valid, f"checkpoint silently omitted {missing_role}"
    assert any(
        issue.code
        in {
            "CHECKPOINT_COMPONENT_MISSING",
            "CHECKPOINT_ROLES_MISSING",
            "FILE_ROLE_REQUIRED",
        }
        and missing_role in issue.message
        for issue in result.errors
    )


def test_checkpoint_requires_split_manifest_lineage(tmp_path: Path) -> None:
    manifest_path, manifest = materialize_checkpoint_artifact(tmp_path)
    del manifest["provenance"]["split_manifest_sha256"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert not result.valid
    assert any(
        issue.code == "PROVENANCE_FIELD" and issue.location == "$.provenance.split_manifest_sha256"
        for issue in result.errors
    )


def test_checkpoint_rejects_duplicate_file_role(tmp_path: Path) -> None:
    manifest_path, manifest = materialize_checkpoint_artifact(tmp_path)
    manifest["files"][1]["role"] = manifest["files"][0]["role"]
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert not result.valid
    assert any(
        issue.code in {"FILE_ROLE_DUPLICATE", "CHECKPOINT_COMPONENT_MISSING"}
        for issue in result.errors
    )


def test_artifact_detects_content_hash_mutation(tmp_path: Path) -> None:
    manifest_path, _ = materialize_checkpoint_artifact(tmp_path)
    target = tmp_path / "run-001" / "model_state.bin"
    original = target.read_bytes()
    target.write_bytes(b"X" + original[1:])

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert not result.valid
    assert any(issue.code == "FILE_HASH_MISMATCH" for issue in result.errors)


def test_artifact_rejects_casefold_file_collision(tmp_path: Path) -> None:
    manifest_path, manifest = materialize_checkpoint_artifact(tmp_path)
    duplicate = deepcopy(manifest["files"][0])
    duplicate["role"] = "extra_diagnostic"
    duplicate["path"] = duplicate["path"].upper()
    manifest["files"].append(duplicate)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert not result.valid
    assert any(issue.code == "FILE_PATH_COLLISION" for issue in result.errors)


def test_empty_artifact_root_fails_when_artifacts_are_required(tmp_path: Path) -> None:
    report = validate_artifact_directory(tmp_path, require_artifacts=True)
    assert not report.valid
    assert report.errors[0].code == "NO_ARTIFACT_MANIFESTS"


def test_failed_artifact_remains_visible_without_fabricating_files(tmp_path: Path) -> None:
    manifest, _ = synthetic_checkpoint_manifest()
    manifest["status"] = "failed"
    manifest["files"] = []
    run_root = tmp_path / "failed-run"
    run_root.mkdir()
    manifest_path = run_root / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    result = validate_artifact_manifest_file(manifest_path, artifact_root=tmp_path)
    assert result.valid, result.to_dict()
    assert any(issue.code == "VISIBLE_NONPASS_STATUS" for issue in result.warnings)
