from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import inclusive_shift_har.experiments.postconfirmatory_efficiency as runner
from inclusive_shift_har.evaluation.efficiency import (
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_COUNTED_MODULES,
    EFFICIENCY_EXCLUDED_OPERATIONS,
    EFFICIENCY_PRECISIONS,
    EFFICIENCY_TIMING_METHOD,
    EFFICIENCY_TIMING_SCOPE,
    EFFICIENCY_VRAM_SCOPE,
    FROZEN_NEURAL_MODEL_IDS,
    FROZEN_NEURAL_SEEDS,
    RECURRENT_CUDNN_DISABLED_MODEL_IDS,
    efficiency_semantic_contract,
    load_efficiency_profile_config,
)
from inclusive_shift_har.evaluation.secondary_aggregation import (
    SecondaryAggregationError,
    aggregate_efficiency_profiles,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


@pytest.fixture(autouse=True)
def _repository_head_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "inclusive_shift_har.evaluation.secondary_aggregation._require_repository_head",
        lambda _root, *, expected_commit: expected_commit,
    )


def _write(path: Path, value: dict[str, Any]) -> None:
    value["record_sha256"] = canonical_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_index(path: Path, value: dict[str, Any]) -> None:
    value.pop("record_sha256", None)
    _write(path, value)


def _contention_snapshot(phase: str, profile_identity: dict[str, Any] | None) -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "gpu_contention_snapshot",
        "captured_at_utc": "2099-01-01T00:00:00Z",
        "phase": phase,
        "profile_identity": profile_identity,
        "profiler_code_commit": "9" * 40,
        "device_index": 0,
        "monitor": "nvidia_smi_selected_device_compute_apps",
        "monitor_return_code": 0,
        "monitor_error": None,
        "allowed_ambient_process_names": ["dwm.exe", "explorer.exe"],
        "observed_compute_processes": [],
        "unapproved_competing_process_count": 0,
        "status": "pass_no_unapproved_compute_processes",
    }
    snapshot["record_sha256"] = canonical_json_sha256(snapshot)
    return snapshot


def _exact_fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    config_path = root / "configs/experiments/neural_efficiency_profile_v1.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_bytes(
        (
            Path(__file__).parents[1] / "configs/experiments/neural_efficiency_profile_v1.yaml"
        ).read_bytes()
    )
    config = load_efficiency_profile_config(config_path)
    environment = {
        "torch_version": "2.12.0+cu132",
        "torch_cuda_version": "13.2",
        "device_type": "cuda",
        "device": "cuda:0",
        "device_index": 0,
        "device_name": "Synthetic CUDA GPU",
        "device_total_memory_bytes": 12_000_000_000,
        "device_compute_capability": [12, 0],
        "cudnn_version": 92000,
    }
    validations: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
    contention_snapshots = [_contention_snapshot("before_run", None)]
    for model_index, model_id in enumerate(FROZEN_NEURAL_MODEL_IDS):
        for seed in FROZEN_NEURAL_SEEDS:
            disabled = model_id in RECURRENT_CUDNN_DISABLED_MODEL_IDS
            configuration_hash = canonical_json_sha256({"model_id": model_id, "seed": seed})
            checkpoint = root / "checkpoints" / model_id / f"seed-{seed}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            checkpoint.write_bytes(f"{model_id}:{seed}".encode())
            checkpoint_hash = sha256_file(checkpoint)
            validation_path = root / "validations" / f"{model_id}--{seed}.json"
            validation: dict[str, Any] = {
                "schema_version": "1.0.0",
                "record_kind": "fixed_epoch_checkpoint_validation",
                "status": "pass_source_only",
                "checkpoint_sha256": checkpoint_hash,
                "training_configuration_sha256": configuration_hash,
                "split_manifest_sha256": "d" * 64,
                "code_commit": "frozen-commit",
                "model_name": model_id,
                "seed": seed,
                "target_information_used_for_selection": False,
            }
            _write(validation_path, validation)
            validation_entry = {
                "model_id": model_id,
                "model_name": model_id,
                "seed": seed,
                "training_configuration_sha256": configuration_hash,
                "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "disable_cudnn": disabled,
                "path": validation_path.relative_to(root).as_posix(),
                "file_sha256": sha256_file(validation_path),
                "record_sha256": validation["record_sha256"],
            }
            validations.append(validation_entry)
            for batch_size in EFFICIENCY_BATCH_SIZES:
                for precision in EFFICIENCY_PRECISIONS:
                    identity = {
                        "model_id": model_id,
                        "seed": seed,
                        "batch_size": batch_size,
                        "precision": precision,
                    }
                    contention_before = _contention_snapshot("before_profile", identity)
                    contention_after = _contention_snapshot("after_profile", identity)
                    contention_snapshots.extend((contention_before, contention_after))
                    profile_path = (
                        root / "profiles" / f"{model_id}--{seed}--{batch_size}--{precision}.json"
                    )
                    latency = 1.0 + model_index / 100.0 + seed / 10000.0
                    profile: dict[str, Any] = {
                        "schema_version": "1.0.0",
                        "record_kind": "cuda_neural_efficiency_profile",
                        "status": "profile_complete",
                        "evidence_status": "measured_cuda_efficiency_not_model_selection_evidence",
                        "required_device": "cuda",
                        "execution_device_type": "cuda",
                        "cuda_available_at_profile": True,
                        "profile_id": config.profile_id,
                        "profile_config_path": config_path.relative_to(root).as_posix(),
                        "profile_config_sha256": config.config_sha256,
                        "profile_config_file_sha256": config.file_sha256,
                        "model_id": model_id,
                        "seed": seed,
                        "precision": precision,
                        "input_shape": [batch_size, 128, 6],
                        "training_configuration_sha256": configuration_hash,
                        "split_manifest_sha256": "d" * 64,
                        "frozen_code_commit": "frozen-commit",
                        "profiler_code_commit": "9" * 40,
                        "final_freeze_inventory_sha256": "a" * 64,
                        "frozen_artifact_set_sha256": "c" * 64,
                        "locked_target_index_record_sha256": "e" * 64,
                        "opening_receipt_record_sha256": "f" * 64,
                        "checkpoint_path": checkpoint.relative_to(root).as_posix(),
                        "checkpoint_validation": {
                            "path": validation_entry["path"],
                            "file_sha256": validation_entry["file_sha256"],
                            "record_sha256": validation_entry["record_sha256"],
                        },
                        "cudnn_policy": {
                            "checkpoint_configuration_disable_cudnn": disabled,
                            "backend_enabled_before_profile": True,
                            "backend_enabled_during_profile": not disabled,
                            "backend_enabled_after_profile": True,
                            "original_backend_state_restored": True,
                        },
                        "environment": environment,
                        "gpu_contention_samples": {
                            "before_record_sha256": contention_before["record_sha256"],
                            "after_record_sha256": contention_after["record_sha256"],
                        },
                        "parameters": {
                            "total": 100 + model_index,
                            "trainable": 90 + model_index,
                            "state_tensor_bytes": 400,
                        },
                        "complexity": {
                            "definition": "one_scalar_multiply_accumulate",
                            "counted_modules": list(EFFICIENCY_COUNTED_MODULES),
                            "excluded_operations": list(EFFICIENCY_EXCLUDED_OPERATIONS),
                            "supported_operator_macs_scope": "one_profiled_batch",
                            "supported_operator_macs_per_window": 1000 + model_index,
                            "estimated_flops_from_supported_macs_per_window": 2000
                            + 2 * model_index,
                            "flop_conversion": "2_flops_per_mac",
                            "coverage_status": "supported_operator_subset_only",
                        },
                        "latency_ms_per_batch": {
                            "warmup_iterations": 20,
                            "measured_iterations": 100,
                            "mean": latency,
                            "median": latency,
                            "minimum": latency - 0.1,
                            "maximum": latency + 0.1,
                            "sample_standard_deviation": 0.01,
                            "mean_per_window": latency / batch_size,
                            "percentiles": {"50.0": latency, "95.0": latency + 0.1},
                            "timing_method": EFFICIENCY_TIMING_METHOD,
                            "scope": EFFICIENCY_TIMING_SCOPE,
                        },
                        "vram_bytes": {
                            "baseline_allocated": 500,
                            "baseline_reserved": 1000,
                            "peak_allocated": 1000 + seed,
                            "peak_reserved": 2000 + seed,
                            "incremental_peak_allocated": 500 + seed,
                            "incremental_peak_reserved": 1000 + seed,
                            "cache_cleared_before_profile": True,
                            "measurement_scope": EFFICIENCY_VRAM_SCOPE,
                        },
                        "model_size": {
                            "checkpoint_file_bytes": checkpoint.stat().st_size,
                            "checkpoint_sha256": checkpoint_hash,
                            "checkpoint_validation_record_sha256": validation["record_sha256"],
                        },
                        "model_selection_use": False,
                    }
                    _write(profile_path, profile)
                    profiles.append(
                        {
                            "model_id": model_id,
                            "seed": seed,
                            "batch_size": batch_size,
                            "precision": precision,
                            "training_configuration_sha256": configuration_hash,
                            "checkpoint_sha256": checkpoint_hash,
                            "checkpoint_validation_path": validation_entry["path"],
                            "checkpoint_validation_file_sha256": validation_entry["file_sha256"],
                            "checkpoint_validation_record_sha256": validation_entry[
                                "record_sha256"
                            ],
                            "disable_cudnn": disabled,
                            "path": profile_path.relative_to(root).as_posix(),
                            "file_sha256": sha256_file(profile_path),
                            "record_sha256": profile["record_sha256"],
                        }
                    )
    contention_path = root / "gpu-contention-attestation.json"
    contention: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "gpu_contention_attestation",
        "status": "pass_no_unapproved_compute_processes_sampled",
        "created_at_utc": "2099-01-01T00:00:00Z",
        "profiler_code_commit": "9" * 40,
        "profile_config_sha256": config.config_sha256,
        "profile_config_file_sha256": config.file_sha256,
        "device_index": 0,
        "monitor": "nvidia_smi_selected_device_compute_apps",
        "sampling_policy": "before_run_and_immediately_before_and_after_every_profile",
        "sampling_limit": "sampled_process_gate_not_continuous_utilization_monitoring",
        "allowed_ambient_process_names": ["dwm.exe", "explorer.exe"],
        "observed_allowlisted_ambient_process_names": [],
        "timing_validity": "valid_exclusive_compute_process_samples",
        "expected_snapshot_count": 641,
        "observed_snapshot_count": 641,
        "unapproved_competing_process_count": 0,
        "snapshots": contention_snapshots,
    }
    _write(contention_path, contention)
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_index",
        "status": "complete_create_only",
        "execution": "sequential_cuda_one_frozen_model_seed_at_a_time",
        "required_device": "cuda",
        "cuda_available_at_start": True,
        "profile_config_path": config_path.relative_to(root).as_posix(),
        "profile_config_sha256": config.config_sha256,
        "profile_config_file_sha256": config.file_sha256,
        "profile_semantics": efficiency_semantic_contract(),
        "execution_environment": environment,
        "final_freeze_inventory_sha256": "a" * 64,
        "frozen_artifact_set_sha256": "c" * 64,
        "split_manifest_sha256": "d" * 64,
        "frozen_code_commit": "frozen-commit",
        "profiler_code_commit": "9" * 40,
        "gpu_contention_attestation": {
            "path": contention_path.relative_to(root).as_posix(),
            "file_sha256": sha256_file(contention_path),
            "record_sha256": contention["record_sha256"],
            "status": contention["status"],
            "timing_validity": contention["timing_validity"],
        },
        "locked_target_index_record_sha256": "e" * 64,
        "opening_receipt_record_sha256": "f" * 64,
        "expected_model_ids": list(FROZEN_NEURAL_MODEL_IDS),
        "required_seed_order": list(FROZEN_NEURAL_SEEDS),
        "required_batch_sizes": list(EFFICIENCY_BATCH_SIZES),
        "required_precisions": list(EFFICIENCY_PRECISIONS),
        "neural_model_count": 16,
        "neural_model_seed_count": 80,
        "profiles_per_model_seed": 4,
        "expected_profile_count": 320,
        "checkpoint_validation_count": 80,
        "profile_count": 320,
        "checkpoint_validations": validations,
        "profiles": profiles,
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }
    index_path = root / "efficiency-index.json"
    _write(index_path, index)
    return index_path, index


def _aggregate(index_path: Path, root: Path) -> dict[str, Any]:
    return aggregate_efficiency_profiles(
        index_path,
        artifact_root=root,
        destination=root / "aggregate.json",
        created_at_utc="2099-01-01T00:00:00Z",
        aggregation_code_commit="a" * 40,
    )


def test_exact_320_cell_efficiency_matrix_aggregates(tmp_path: Path) -> None:
    index_path, _ = _exact_fixture(tmp_path)
    result = _aggregate(index_path, tmp_path)
    assert result["validated_checkpoint_count"] == 80
    assert result["validated_profile_count"] == 320
    assert result["profiler_code_commit"] == "9" * 40
    assert result["gpu_timing_validity"] == "valid_exclusive_compute_process_samples"
    assert result["execution_environment"]["device"] == "cuda:0"
    assert result["profile_semantics"] == efficiency_semantic_contract()
    assert len(result["tables"]["model_batch_precision_aggregates"]) == 64


def test_missing_profile_combination_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    index["profiles"].pop()
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="incomplete"):
        _aggregate(index_path, tmp_path)


def test_bad_opening_lineage_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    index["opening_receipt_record_sha256"] = "0" * 64
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="lineage differs"):
        _aggregate(index_path, tmp_path)


def test_recurrent_cudnn_policy_change_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    recurrent = next(
        entry for entry in index["checkpoint_validations"] if entry["model_id"] == "deepconvlstm"
    )
    recurrent["disable_cudnn"] = False
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="cuDNN policy"):
        _aggregate(index_path, tmp_path)


def test_efficiency_profile_escape_path_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    index["profiles"][0]["path"] = "../escaped.json"
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="escapes artifact_root"):
        _aggregate(index_path, tmp_path)


def test_profiler_commit_lineage_change_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    index["profiler_code_commit"] = "8" * 40
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="contention attestation contract"):
        _aggregate(index_path, tmp_path)


def test_profile_contention_link_change_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    entry = index["profiles"][0]
    profile_path = tmp_path / entry["path"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.pop("record_sha256")
    profile["gpu_contention_samples"]["before_record_sha256"] = "0" * 64
    _write(profile_path, profile)
    entry["file_sha256"] = sha256_file(profile_path)
    entry["record_sha256"] = profile["record_sha256"]
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="contention samples"):
        _aggregate(index_path, tmp_path)


def test_profile_semantic_or_environment_drift_is_rejected(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    entry = index["profiles"][0]
    profile_path = tmp_path / entry["path"]
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    profile.pop("record_sha256")
    profile["latency_ms_per_batch"]["warmup_iterations"] = 19
    _write(profile_path, profile)
    entry["file_sha256"] = sha256_file(profile_path)
    entry["record_sha256"] = profile["record_sha256"]
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="latency measurement semantics"):
        _aggregate(index_path, tmp_path)


def test_contention_snapshot_and_profile_device_must_match(tmp_path: Path) -> None:
    index_path, index = _exact_fixture(tmp_path)
    index["execution_environment"]["device"] = "cuda:1"
    index["execution_environment"]["device_index"] = 1
    _rewrite_index(index_path, index)
    with pytest.raises(SecondaryAggregationError, match="contention device differ"):
        _aggregate(index_path, tmp_path)


def test_runner_rejects_escape_before_creating_directory(tmp_path: Path) -> None:
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped"
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="escapes output_root"):
        runner._output_directory(escaped, output_root=tmp_path)
    assert not escaped.exists()


def test_runner_confines_profile_config_and_rejects_symlink(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-profile.yaml"
    outside.write_text("profile", encoding="utf-8")
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="escapes artifact_root"):
        runner._resolve_artifact_input(
            outside, artifact_root=tmp_path.resolve(), name="efficiency profile config"
        )

    link = tmp_path / "profile-link.yaml"
    try:
        link.symlink_to(outside)
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="symlink"):
        runner._resolve_artifact_input(
            link, artifact_root=tmp_path.resolve(), name="efficiency profile config"
        )


def test_runner_recurrent_policy_validation_is_strict() -> None:
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="cuDNN policy"):
        runner._validate_recurrent_cudnn_policy(
            model_id="deepconvlstm",
            configuration={"disable_cudnn": False},
            model=nn.LSTM(3, 4, batch_first=True),
        )


def test_runner_applies_and_restores_cudnn_without_cuda_execution(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_efficiency_profile_config(
        repository_root / "configs/experiments/neural_efficiency_profile_v1.yaml"
    )
    observed: list[bool] = []

    def fake_profile(*_: Any, **__: Any) -> dict[str, Any]:
        observed.append(bool(torch.backends.cudnn.enabled))
        return {}

    monkeypatch.setattr(runner, "profile_neural_model", fake_profile)
    before = bool(torch.backends.cudnn.enabled)
    _, policy = runner._profile_with_cudnn_policy(
        nn.Linear(2, 2),
        torch.zeros((1, 128, 6)),
        config=config,
        precision="float32",
        model_id="deepconvlstm",
        seed=11,
        checkpoint_path=tmp_path / "unused.pt",
        checkpoint_sha256="a" * 64,
        checkpoint_validation_record_sha256="b" * 64,
        disable_cudnn=True,
    )
    assert observed == [False]
    assert bool(torch.backends.cudnn.enabled) is before
    assert policy["original_backend_state_restored"] is True


def test_profiler_commit_must_equal_repository_head(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(runner, "_repository_head", lambda _: "1" * 40)
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="differs"):
        runner._require_profiler_head(tmp_path, "2" * 40)
    assert runner._require_profiler_head(tmp_path, "1" * 40) == "1" * 40


def test_protected_windows_dwm_is_resolved_by_exact_pid(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_efficiency_profile_config(
        repository_root / "configs/experiments/neural_efficiency_profile_v1.yaml"
    )
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"],
        returncode=0,
        stdout="2268, [Insufficient Permissions]\n",
        stderr="",
    )
    monkeypatch.setattr(
        "inclusive_shift_har.experiments.postconfirmatory_efficiency.subprocess.run",
        lambda *_, **__: completed,
    )
    monkeypatch.setattr(runner, "_resolve_windows_process_basename", lambda pid: ("dwm.exe", None))
    snapshot = runner._capture_gpu_contention_snapshot(
        config=config,
        device=torch.device("cuda:0"),
        profiler_code_commit="9" * 40,
        phase="before_run",
        profile_identity=None,
    )
    assert snapshot["status"] == "pass_no_unapproved_compute_processes"
    process = snapshot["observed_compute_processes"][0]
    assert process["pid"] == 2268
    assert process["process_name_basename"] == "dwm.exe"
    assert process["process_name_resolution"] == "windows_get_process_exact_pid"
    assert process["classification"] == "allowlisted_ambient_process"


def test_unapproved_gpu_process_fails_contention_gate(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_efficiency_profile_config(
        repository_root / "configs/experiments/neural_efficiency_profile_v1.yaml"
    )
    completed = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="9999, python.exe\n", stderr=""
    )
    monkeypatch.setattr(
        "inclusive_shift_har.experiments.postconfirmatory_efficiency.subprocess.run",
        lambda *_, **__: completed,
    )
    snapshot = runner._capture_gpu_contention_snapshot(
        config=config,
        device=torch.device("cuda:0"),
        profiler_code_commit="9" * 40,
        phase="before_run",
        profile_identity=None,
    )
    assert snapshot["status"] == "fail_contention_gate"
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="contention gate"):
        runner._require_contention_pass(snapshot)


def test_repository_freeze_has_exact_runner_inventory(repository_root: Path) -> None:
    inventory = json.loads(
        (repository_root / "results/protocol/final_source_artifact_freeze_v1.json").read_text(
            encoding="utf-8"
        )
    )
    entries = runner._validate_neural_inventory(inventory)
    assert len(entries) == 80
    assert {(entry["model_id"], entry["seed"]) for entry in entries} == {
        (model_id, seed) for model_id in FROZEN_NEURAL_MODEL_IDS for seed in FROZEN_NEURAL_SEEDS
    }


def test_failure_and_failed_index_are_create_only_and_preserve_partial_counts(
    tmp_path: Path,
) -> None:
    output = tmp_path / "efficiency"
    output.mkdir()
    runner._publish_failure_artifacts(
        output=output,
        output_root=tmp_path,
        timestamp="2099-01-01T00:00:00Z",
        profile_config_path="configs/profile.yaml",
        profile_config_sha256="a" * 64,
        profile_config_file_sha256="b" * 64,
        inventory=None,
        context=None,
        profile_entries=[{"model_id": "partial"}],
        validation_entries=[{"model_id": "partial"}],
        profiler_code_commit="9" * 40,
        contention_snapshots=[],
        execution_environment=None,
        error=RuntimeError("synthetic failure"),
    )
    failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    index = json.loads(
        (output / "neural_efficiency_profile_index.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_preserved_create_only"
    assert index["status"] == "failed_preserved_create_only"
    assert index["profile_count"] == 1
    assert index["checkpoint_validation_count"] == 1


def test_efficiency_main_returns_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(**_: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic runner failure")

    monkeypatch.setattr(runner, "run_frozen_efficiency_profiles", fail)
    exit_code = runner.main(
        [
            "--profile-config",
            "config.yaml",
            "--expected-profile-config-file-sha256",
            "a" * 64,
            "--final-freeze-inventory",
            "freeze.json",
            "--opening-receipt",
            "receipt.json",
            "--locked-target-index",
            "index.json",
            "--artifact-root",
            ".",
            "--output-directory",
            "results",
            "--output-root",
            ".",
            "--created-at-utc",
            "2099-01-01T00:00:00Z",
            "--profiler-code-commit",
            "9" * 40,
        ]
    )
    assert exit_code == 1
    assert "synthetic runner failure" in capsys.readouterr().err
