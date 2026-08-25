"""Machine-readable schema documents must match the runtime contract."""

from __future__ import annotations

import json
import re
import subprocess
import tomllib
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from inclusive_shift_har.artifacts.release_gate import EXACT_GATE_NAMES, TRACKED_GATE_NAMES
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


def test_ci_locked_sync_disables_global_uv_frozen_for_install_steps(
    repository_root: Path,
) -> None:
    workflow = yaml.load(
        (repository_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    assert isinstance(workflow, dict)
    assert workflow["env"]["UV_FROZEN"] == "1"
    expected = {
        "synthetic-validation": "Install locked development environment",
        "release-security": "Install locked release environment",
    }
    for job_name, step_name in expected.items():
        steps = workflow["jobs"][job_name]["steps"]
        matches = [step for step in steps if step.get("name") == step_name]
        assert len(matches) == 1
        install = matches[0]
        assert install["env"]["UV_FROZEN"] == "false"
        assert "uv sync --locked" in install["run"]


def test_release_version_is_consistent_across_package_metadata(repository_root: Path) -> None:
    project = tomllib.loads((repository_root / "pyproject.toml").read_text(encoding="utf-8"))
    lock = tomllib.loads((repository_root / "uv.lock").read_text(encoding="utf-8"))
    citation = yaml.safe_load((repository_root / "CITATION.cff").read_text(encoding="utf-8"))
    zenodo = _load(repository_root / ".zenodo.json")
    version = project["project"]["version"]
    locked_package = [
        package for package in lock["package"] if package["name"] == "inclusive-shift-har"
    ]
    fallback = re.search(
        r'^\s*__version__\s*=\s*"([^"]+)"\s*$',
        (repository_root / "src" / "inclusive_shift_har" / "__init__.py").read_text(
            encoding="utf-8"
        ),
        flags=re.MULTILINE,
    )
    assert len(locked_package) == 1
    assert fallback is not None
    assert {
        locked_package[0]["version"],
        citation["version"],
        zenodo["version"],
        fallback.group(1),
    } == {version}


def test_ci_and_gitattributes_preserve_hash_bound_files_on_windows(
    repository_root: Path,
) -> None:
    workflow = yaml.load(
        (repository_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )
    steps = workflow["jobs"]["synthetic-validation"]["steps"]
    assert steps[0] == {
        "name": "Preserve Git blob line endings on Windows",
        "if": "runner.os == 'Windows'",
        "shell": "pwsh",
        "run": "git config --global core.autocrlf false",
    }
    assert steps[1]["name"] == "Check out repository"
    attributes = (repository_root / ".gitattributes").read_text(encoding="utf-8").splitlines()
    assert attributes[-1] == "* -text"
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    checked = subprocess.run(
        ["git", "check-attr", "-z", "--stdin", "text"],
        cwd=repository_root,
        input=tracked,
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")
    assert checked.pop() == b""
    triplets = list(zip(checked[0::3], checked[1::3], checked[2::3], strict=True))
    assert triplets
    assert all(attribute == b"text" and value == b"unset" for _, attribute, value in triplets)


def test_schema_documents_are_valid_json_with_only_local_references(
    repository_root: Path,
) -> None:
    schema_root = repository_root / "configs" / "schema"
    paths = sorted(schema_root.glob("*.schema.json"))
    assert {path.name for path in paths} == {
        "artifact_manifest.schema.json",
        "dataset_manifest.schema.json",
        "final_freeze.schema.json",
        "final_release_gate_report.schema.json",
        "locked_target_result.schema.json",
        "release_evidence_inventory.schema.json",
        "source_calibrator.schema.json",
    }
    for path in paths:
        schema = _load(path)
        assert schema["$schema"] == "https://json-schema.org/draft/2020-12/schema"
        assert schema["type"] == "object"
        assert all(reference.startswith("#/") for reference in _references(schema))
        Draft202012Validator.check_schema(schema)


def _release_gate_schema_fixture(mode: str) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "path": "evidence.json",
        "size_bytes": 1,
        "file_sha256": "a" * 64,
    }
    gates: dict[str, dict[str, Any]]
    external: dict[str, Any]
    if mode == "tracked_precommit_report":
        gates = {name: {"status": "not_run", "evidence": None} for name in TRACKED_GATE_NAMES}
        status = "pending"
        content_commit = None
        clean = False
        external = {
            "status": "pending",
            "expected_root": ".audit/release-attestations/<candidate_commit>/",
            "report_path": None,
        }
    else:
        gates = {name: {"status": "pass", "evidence": reference} for name in EXACT_GATE_NAMES}
        status = "pass"
        content_commit = "b" * 40
        clean = True
        external = {
            "status": "complete",
            "expected_root": ".audit/release-attestations/<candidate_commit>/",
            "report_path": (
                f".audit/release-attestations/{content_commit}/final_release_gate_report.json"
            ),
        }
    return {
        "schema_version": "1.0.0",
        "record_kind": "final_release_gate_report",
        "mode": mode,
        "status": status,
        "created_at_utc": "2026-08-24T12:00:00Z",
        "repository": {
            "parent_commit": "a" * 40,
            "content_commit": content_commit,
            "worktree_clean": clean,
        },
        "policy": reference,
        "gates": gates,
        "external_candidate_attestation": external,
        "record_sha256": "c" * 64,
    }


def test_final_release_gate_schema_enforces_mode_specific_gate_semantics(
    repository_root: Path,
) -> None:
    schema = _load(repository_root / "configs" / "schema" / "final_release_gate_report.schema.json")
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    tracked = _release_gate_schema_fixture("tracked_precommit_report")
    exact = _release_gate_schema_fixture("exact_candidate_attestation")
    validator.validate(tracked)
    validator.validate(exact)

    tracked_with_evidence = deepcopy(tracked)
    tracked_with_evidence["gates"]["tests"]["evidence"] = tracked["policy"]
    with pytest.raises(ValidationError):
        validator.validate(tracked_with_evidence)

    exact_without_evidence = deepcopy(exact)
    exact_without_evidence["gates"]["tests"]["evidence"] = None
    with pytest.raises(ValidationError):
        validator.validate(exact_without_evidence)

    exact_with_extra_gate_key = deepcopy(exact)
    exact_with_extra_gate_key["gates"]["tests"]["unexpected"] = True
    with pytest.raises(ValidationError):
        validator.validate(exact_with_extra_gate_key)

    invalid_timestamp = deepcopy(tracked)
    invalid_timestamp["created_at_utc"] = "not-a-date"
    with pytest.raises(ValidationError):
        validator.validate(invalid_timestamp)

    offset_timestamp = deepcopy(tracked)
    offset_timestamp["created_at_utc"] = "2026-08-24T13:00:00+01:00"
    with pytest.raises(ValidationError):
        validator.validate(offset_timestamp)


@pytest.mark.parametrize(
    "schema_name",
    ["final_release_gate_report.schema.json", "release_evidence_inventory.schema.json"],
)
def test_release_schemas_reject_control_paths_and_non_z_timestamps(
    repository_root: Path, schema_name: str
) -> None:
    schema = _load(repository_root / "configs" / "schema" / schema_name)
    path_validator = Draft202012Validator(schema["$defs"]["safe_relative_path"])
    timestamp_validator = Draft202012Validator(
        schema["properties"]["created_at_utc"],
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    for unsafe in (
        "bad\npath.json",
        "bad\rpath.json",
        "bad\x7fpath.json",
        "evidence/",
    ):
        with pytest.raises(ValidationError):
            path_validator.validate(unsafe)
    with pytest.raises(ValidationError):
        timestamp_validator.validate("2026-08-24T13:00:00+01:00")
    with pytest.raises(ValidationError):
        timestamp_validator.validate("2026-99-99T99:99:99Z")


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
