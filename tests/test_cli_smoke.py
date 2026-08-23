"""End-to-end CLI smoke tests over tiny synthetic or manifest-only inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from ._synthetic import materialize_checkpoint_artifact, synthetic_dataset_manifest


def _run_cli(
    arguments: list[str],
    *,
    repository_root: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "inclusive_shift_har.cli", *arguments],
        cwd=repository_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=20,
    )


def _json_stdout(completed: subprocess.CompletedProcess[str]) -> dict[str, Any]:
    assert completed.stdout.strip(), completed.stderr
    payload = json.loads(completed.stdout)
    assert isinstance(payload, dict)
    return payload


def test_cli_help_lists_every_stage_2_command(
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        ["--help"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    assert completed.returncode == 0, completed.stderr
    for command in (
        "validate-manifests",
        "audit-data",
        "build-splits",
        "audit-splits",
        "train",
        "evaluate",
        "validate-artifacts",
    ):
        assert command in completed.stdout


def test_validate_manifests_cli_passes_locked_starters(
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        ["validate-manifests", "--manifest-root", "manifests/datasets", "--json"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 0, payload
    assert payload["status"] == "pass"
    assert payload["report"]["manifest_count"] == 2


def test_audit_data_cli_dry_run_reads_no_raw_data(
    tmp_path: Path,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(
        json.dumps(synthetic_dataset_manifest(), sort_keys=True),
        encoding="utf-8",
    )
    completed = _run_cli(
        ["audit-data", "--manifest", str(manifest_path), "--dry-run", "--json"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 0, payload
    assert payload["status"] == "dry_run_pass"
    assert payload["report"]["data_access"] == "none"
    assert payload["report"]["observations"][0]["status"] == ("planned_no_filesystem_access")


def test_audit_data_cli_fails_closed_without_read_gate(
    tmp_path: Path,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(
        json.dumps(synthetic_dataset_manifest(), sort_keys=True),
        encoding="utf-8",
    )
    data_root = tmp_path / "raw"
    data_root.mkdir()
    completed = _run_cli(
        [
            "audit-data",
            "--manifest",
            str(manifest_path),
            "--read-only",
            "--data-root",
            str(data_root),
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 3
    assert payload["status"] == "gated"
    assert payload["code"] == "RAW_DATA_READ_GATE_CLOSED"


def test_full_inclusivehar_profile_cannot_run_as_ungated_dry_run(
    tmp_path: Path,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    manifest_path = tmp_path / "synthetic-manifest.json"
    manifest_path.write_text(
        json.dumps(synthetic_dataset_manifest(), sort_keys=True),
        encoding="utf-8",
    )

    completed = _run_cli(
        [
            "audit-data",
            "--manifest",
            str(manifest_path),
            "--profile",
            "inclusivehar-v4",
            "--dry-run",
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)

    assert completed.returncode == 3
    assert payload["status"] == "gated"
    assert payload["code"] == "FULL_AUDIT_REQUIRES_READ_ONLY_GATE"


def test_validate_artifacts_cli_checks_complete_synthetic_checkpoint(
    tmp_path: Path,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    materialize_checkpoint_artifact(tmp_path)
    completed = _run_cli(
        [
            "validate-artifacts",
            "--artifact-root",
            str(tmp_path),
            "--require-artifacts",
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 0, payload
    assert payload["status"] == "pass"
    assert payload["report"]["checked_manifests"] == 1


@pytest.mark.parametrize(
    ("command", "expected_code", "required_gate"),
    [
        ("build-splits", "DATA_AUDIT_GATE_CLOSED", "data_audit_pass"),
        (
            "audit-splits",
            "SPLIT_MANIFEST_GATE_CLOSED",
            "data_audit_pass_and_split_manifest_present",
        ),
        ("train", "TRAINING_GATE_CLOSED_NOT_IMPLEMENTED", "split_audit_pass"),
        (
            "evaluate",
            "EVALUATION_GATE_CLOSED_NOT_IMPLEMENTED",
            "final_evaluation_unlock",
        ),
    ],
)
def test_unimplemented_cli_stages_are_visible_gates_not_false_successes(
    command: str,
    expected_code: str,
    required_gate: str,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        [command, "--json"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 3
    assert payload == {
        "code": expected_code,
        "command": command,
        "exit_code": 3,
        "message": payload["message"],
        "required_gate": required_gate,
        "status": "gated_not_implemented",
    }
    assert payload["message"]
