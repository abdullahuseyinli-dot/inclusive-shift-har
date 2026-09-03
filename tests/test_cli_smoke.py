"""End-to-end CLI smoke tests over tiny synthetic or manifest-only inputs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har import cli as cli_module

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


def test_cli_help_lists_current_direct_and_dispatch_commands(
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
        "aggregate-source-cv",
        "aggregate-uci-source",
        "train",
        "evaluate",
        "validate-artifacts",
    ):
        assert command in completed.stdout


def test_train_and_evaluate_help_lists_truthful_safe_tracks(
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    train = _run_cli(
        ["train", "--help"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    evaluate = _run_cli(
        ["evaluate", "--help"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )

    assert train.returncode == 0, train.stderr
    assert evaluate.returncode == 0, evaluate.stderr
    for track in (
        "inclusivehar-source",
        "final-source-suite",
        "uci-source-fold",
        "few-person",
        "within-group",
        "raw-total-sensitivity",
    ):
        assert track in train.stdout
    for track in (
        "source-cv",
        "uci-source",
        "few-person-statistics",
        "within-group-statistics",
        "efficiency",
        "sensor-stress",
        "raw-total-statistics",
    ):
        assert track in evaluate.stdout
    combined = (train.stdout + evaluate.stdout).casefold()
    assert "not implemented" not in combined
    assert "no confirmatory target was opened" not in combined
    assert "stage 2 provenance scaffold" not in combined
    assert "one-time confirmatory target-opening operation is never routed" in combined
    assert "no locked-report track is exposed" in combined


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
    assert payload["report"]["manifest_count"] == 3


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


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_dispatchers_require_a_track_with_machine_readable_errors(
    command: str,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        [command, "--json"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)
    assert completed.returncode == 2
    assert payload["code"] == f"{command.upper()}_TRACK_REQUIRED"
    assert payload["command"] == command
    assert payload["exit_code"] == 2
    assert payload["one_time_confirmatory_target_operation_exposed"] is False
    assert payload["status"] == "fail"
    assert payload["track"] is None
    assert payload["message"]


@pytest.mark.parametrize("command", ["train", "evaluate"])
def test_dispatchers_never_route_confirmatory_target_operation(
    command: str,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        [command, "--json", "confirmatory-target"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)

    assert completed.returncode == 2
    assert payload["code"] == f"UNKNOWN_{command.upper()}_TRACK"
    assert payload["one_time_confirmatory_target_operation_exposed"] is False
    assert payload["track"] == "confirmatory-target"


def test_forwarded_parser_failure_remains_machine_readable(
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        ["train", "--json", "uci-source-fold"],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    payload = _json_stdout(completed)

    assert completed.returncode == 2
    assert payload["code"] == "TRAIN_TRACK_ARGUMENT_ERROR"
    assert payload["status"] == "fail"
    assert payload["track"] == "uci-source-fold"
    assert "required" in completed.stderr


@pytest.mark.parametrize(
    ("arguments", "expected_fragment"),
    [
        (["train", "uci-source-fold", "--", "--help"], "run-uci-source-fold"),
        (["train", "inclusivehar-source", "--", "--help"], "--source-manifest"),
        (
            ["train", "within-group", "--", "--help"],
            "--primary-cache-record-file-sha256",
        ),
        (
            ["evaluate", "within-group-statistics", "--", "--help"],
            "--result-root",
        ),
        (["evaluate", "efficiency", "--", "--help"], "--created-at-utc"),
    ],
)
def test_dispatchers_forward_child_help_without_running_work(
    arguments: list[str],
    expected_fragment: str,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        arguments,
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert expected_fragment in completed.stdout


@pytest.mark.parametrize(
    "arguments",
    [
        ["train", "within-group", "--", "--help"],
        ["evaluate", "within-group-statistics", "--", "--help"],
    ],
)
def test_within_group_dispatchers_expose_cache_evidence_but_no_opening_or_raw_route(
    arguments: list[str],
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    completed = _run_cli(
        arguments,
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )

    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout.casefold()
    assert "--primary-cache-record" in help_text
    assert "--raw-csv" not in help_text
    assert "--unlock-record" not in help_text
    assert "--opening-acknowledgement" not in help_text
    assert "--device" not in help_text


def test_dispatcher_forwards_exact_argv_without_shell_interpretation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[tuple[str, str, list[str] | None]] = []

    def fake_entrypoint(arguments: list[str] | None) -> int:
        observed.append(("called", "entrypoint", arguments))
        return 17

    def fake_resolver(
        workflow: str,
        track: str,
    ) -> tuple[cli_module.WorkflowMain, tuple[str, ...], bool]:
        observed.append((workflow, track, None))
        return fake_entrypoint, ("fixed-prefix",), True

    monkeypatch.setattr(cli_module, "_workflow_entrypoint", fake_resolver)

    exit_code = cli_module.main(
        [
            "evaluate",
            "--json",
            "uci-source",
            "--",
            "--literal",
            "value with spaces",
            "semi;colon",
        ]
    )

    assert exit_code == 17
    assert observed == [
        ("evaluate", "uci-source", None),
        (
            "called",
            "entrypoint",
            ["fixed-prefix", "--literal", "value with spaces", "semi;colon", "--json"],
        ),
    ]


@pytest.mark.parametrize("model", ["compact_residual_96", "xgboost"])
def test_unified_inclusivehar_route_rejects_cpu_for_neural_and_xgboost(
    model: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def forbidden_resolver(
        workflow: str, track: str
    ) -> tuple[cli_module.WorkflowMain, tuple[str, ...], bool]:
        raise AssertionError(f"device policy must reject before routing {workflow}/{track}")

    monkeypatch.setattr(cli_module, "_workflow_entrypoint", forbidden_resolver)
    exit_code = cli_module.main(
        [
            "train",
            "--json",
            "inclusivehar-source",
            "--",
            "--model",
            model,
            "--device",
            "cpu",
        ]
    )
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert payload["code"] == "TRAIN_TRACK_DEVICE_POLICY_ERROR"
    assert payload["track"] == "inclusivehar-source"
    assert "cuda" in payload["message"].casefold()


@pytest.mark.parametrize("model", ["random_forest", "svm_rbf", "logistic_regression"])
def test_unified_inclusivehar_route_preserves_native_cpu_classical_models(
    model: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str] | None] = []

    def fake_entrypoint(arguments: list[str] | None) -> int:
        observed.append(arguments)
        return 0

    monkeypatch.setattr(
        cli_module,
        "_workflow_entrypoint",
        lambda workflow, track: (fake_entrypoint, (), False),
    )
    exit_code = cli_module.main(
        [
            "train",
            "inclusivehar-source",
            "--",
            "--model",
            model,
            "--device=cpu",
        ]
    )

    assert exit_code == 0
    assert observed == [["--model", model, "--device=cpu"]]


def test_build_and_audit_splits_cli_never_opens_target_performance(
    tmp_path: Path,
    repository_root: Path,
    offline_subprocess_environment: dict[str, str],
) -> None:
    split_path = tmp_path / "split.json"
    built = _run_cli(
        [
            "build-splits",
            "--output",
            str(split_path),
            "--allowed-root",
            str(tmp_path),
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    built_payload = _json_stdout(built)
    assert built.returncode == 0, built_payload
    assert built_payload["status"] == "built_conditional_released_block"
    assert built_payload["window_count"] == 3042
    assert built_payload["target_performance_or_prediction_accessed"] is False

    audited = _run_cli(
        [
            "audit-splits",
            "--split-manifest",
            str(split_path),
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    audited_payload = _json_stdout(audited)
    assert audited.returncode == 0, audited_payload
    assert audited_payload["status"] == "pass_conditional_released_block"
    assert audited_payload["report"]["target_performance_or_prediction_accessed"] is False

    source_path = tmp_path / "source-windows.json"
    source_built = _run_cli(
        [
            "build-source-windows",
            "--split-manifest",
            str(split_path),
            "--output",
            str(source_path),
            "--allowed-root",
            str(tmp_path),
            "--json",
        ],
        repository_root=repository_root,
        environment=offline_subprocess_environment,
    )
    source_payload = _json_stdout(source_built)
    assert source_built.returncode == 0, source_payload
    assert source_payload["source_window_count"] == 1443
    assert source_payload["target_subject_or_window_records_included"] is False
