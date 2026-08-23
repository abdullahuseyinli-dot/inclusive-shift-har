"""Canonical JSON, hashing, and immutable manifest validation tests."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import pytest

from inclusive_shift_har.manifests.canonical import (
    CanonicalJSONError,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.manifests.validation import (
    validate_manifest,
    validate_manifest_directory,
    validate_manifest_file,
)

from ._synthetic import synthetic_dataset_manifest, with_second_artifact


def test_canonical_hash_is_stable_across_mapping_order() -> None:
    left = {"z": [3, 2, 1], "nested": {"b": "β", "a": True}, "a": 7}
    right = {"a": 7, "nested": {"a": True, "b": "β"}, "z": [3, 2, 1]}

    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert canonical_json_sha256(left) == canonical_json_sha256(right)
    assert canonical_json_bytes(left).startswith(b'{"a":7,"nested":')


def test_canonical_hash_changes_on_semantic_change() -> None:
    original = {"values": [1, 2, 3]}
    reordered = {"values": [3, 2, 1]}
    assert canonical_json_sha256(original) != canonical_json_sha256(reordered)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_numbers(value: float) -> None:
    with pytest.raises(CanonicalJSONError):
        canonical_json_bytes({"invalid": value})


def test_strict_loader_rejects_duplicate_keys(tmp_path: Path) -> None:
    source = tmp_path / "duplicate.json"
    source.write_text('{"dataset_id":"one","dataset_id":"two"}', encoding="utf-8")
    with pytest.raises(CanonicalJSONError, match="duplicate"):
        load_json_strict(source)


def test_sha256_file_matches_independent_digest(tmp_path: Path) -> None:
    payload = b"small deterministic payload\x00\xff"
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)
    assert sha256_file(source, chunk_size=3) == hashlib.sha256(payload).hexdigest()


def test_valid_synthetic_manifest_has_deterministic_hash() -> None:
    manifest = synthetic_dataset_manifest()
    first = validate_manifest(manifest)
    second = validate_manifest(deepcopy(manifest))
    assert first.valid, first.to_dict()
    assert second.valid, second.to_dict()
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.manifest_sha256 == canonical_json_sha256(manifest)


def test_manifest_schema_failure_is_machine_readable() -> None:
    manifest = synthetic_dataset_manifest()
    del manifest["license"]
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(
        issue.code == "TYPE_OBJECT" and issue.location == "$.license" for issue in result.errors
    )


def test_manifest_rejects_malformed_checksum() -> None:
    manifest = synthetic_dataset_manifest()
    manifest["artifacts"][0]["expected_sha256"] = "NOT-A-SHA256"
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(issue.code == "EXPECTED_SHA256" for issue in result.errors)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "../outside.bin",
        "data/raw/../../outside.bin",
        "/absolute/outside.bin",
        "data\\raw\\outside.bin",
        "./data/raw/outside.bin",
    ],
)
def test_manifest_rejects_path_traversal(unsafe_path: str) -> None:
    manifest = synthetic_dataset_manifest(storage_path=unsafe_path)
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(issue.code == "STORAGE_PATH" for issue in result.errors)


def test_manifest_rejects_casefold_path_collision() -> None:
    manifest = with_second_artifact(synthetic_dataset_manifest())
    manifest["artifacts"][1]["storage_path"] = manifest["artifacts"][0]["storage_path"].upper()
    result = validate_manifest(manifest)
    assert not result.valid, "case-only path aliases are unsafe across filesystems"
    assert any("CASE" in issue.code or "COLLISION" in issue.code for issue in result.errors)


def test_directory_rejects_duplicate_dataset_ids(tmp_path: Path) -> None:
    import json

    manifest = synthetic_dataset_manifest()
    for name in ("one.json", "two.json"):
        (tmp_path / name).write_text(json.dumps(manifest), encoding="utf-8")
    report = validate_manifest_directory(tmp_path)
    assert not report.valid
    assert any(issue.code == "DUPLICATE_DATASET_ID" for issue in report.errors)


def test_manifest_file_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    source = tmp_path / "invalid.json"
    source.write_text('{"schema_version":"1.0.0","schema_version":"2"}', encoding="utf-8")
    result = validate_manifest_file(source)
    assert not result.valid
    assert result.errors[0].code == "LOAD_JSON"
