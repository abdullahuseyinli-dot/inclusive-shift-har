"""Machine-readable schema documents must match the runtime contract."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from inclusive_shift_har.artifacts.validation import ARTIFACT_SCHEMA_VERSION
from inclusive_shift_har.manifests.validation import (
    MANIFEST_SCHEMA_VERSION,
    validate_manifest_directory,
)

from ._synthetic import CHECKPOINT_FILE_ROLES, synthetic_dataset_manifest


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _references(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "$ref":
                assert isinstance(child, str)
                yield child
            else:
                yield from _references(child)
    elif isinstance(value, list):
        for child in value:
            yield from _references(child)


def test_schema_documents_are_valid_json_with_only_local_references(
    repository_root: Path,
) -> None:
    schema_root = repository_root / "configs" / "schema"
    paths = sorted(schema_root.glob("*.schema.json"))
    assert {path.name for path in paths} == {
        "artifact_manifest.schema.json",
        "dataset_manifest.schema.json",
        "final_freeze.schema.json",
        "locked_target_result.schema.json",
        "source_calibrator.schema.json",
    }
    for path in paths:
        schema = _load(path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert all(reference.startswith("#/") for reference in _references(schema))


def test_dataset_schema_covers_runtime_required_fields(repository_root: Path) -> None:
    schema = _load(repository_root / "configs" / "schema" / "dataset_manifest.schema.json")
    fixture = synthetic_dataset_manifest()
    assert schema["properties"]["schema_version"]["const"] == MANIFEST_SCHEMA_VERSION
    assert set(fixture) == set(schema["required"])
    assert set(fixture) <= set(schema["properties"])
    artifact_required = set(schema["$defs"]["provider_artifact"]["required"])
    assert set(fixture["artifacts"][0]) == artifact_required


def test_locked_starter_manifests_use_declared_top_level_schema_fields(
    repository_root: Path,
) -> None:
    schema = _load(repository_root / "configs" / "schema" / "dataset_manifest.schema.json")
    declared = set(schema["properties"])
    manifest_root = repository_root / "manifests" / "datasets"
    manifests = [_load(path) for path in sorted(manifest_root.glob("*.json"))]
    assert manifests
    for manifest in manifests:
        assert set(manifest) <= declared


def test_locked_starter_manifest_directory_validates(repository_root: Path) -> None:
    report = validate_manifest_directory(repository_root / "manifests" / "datasets")
    assert report.valid, report.to_dict()
    assert {result.dataset_id for result in report.results} == {
        "inclusivehar_v4",
        "uci_har_v1",
    }
    assert all(result.manifest_sha256 for result in report.results)


def test_artifact_schema_declares_complete_checkpoint_contract(
    repository_root: Path,
) -> None:
    schema = _load(repository_root / "configs" / "schema" / "artifact_manifest.schema.json")
    assert schema["properties"]["schema_version"]["const"] == ARTIFACT_SCHEMA_VERSION
    schema_text = json.dumps(schema, sort_keys=True)
    for role in CHECKPOINT_FILE_ROLES:
        assert f'"const": "{role}"' in schema_text
    provenance_required = {
        "code_commit",
        "configuration_sha256",
        "dataset_manifest_sha256",
        "split_manifest_sha256",
    }
    assert provenance_required <= set(schema_text.split('"'))
