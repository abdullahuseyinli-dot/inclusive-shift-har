from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import inclusive_shift_har.experiments.postconfirmatory_efficiency as runner
from inclusive_shift_har.evaluation.efficiency import (
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_PRECISIONS,
    FROZEN_NEURAL_MODEL_IDS,
    FROZEN_NEURAL_SEEDS,
    RECURRENT_CUDNN_DISABLED_MODEL_IDS,
    load_efficiency_profile_config,
)
from inclusive_shift_har.evaluation.secondary_aggregation import (
    SecondaryAggregationError,
    aggregate_efficiency_profiles,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write(path: Path, value: dict[str, Any]) -> None:
    value["record_sha256"] = canonical_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _rewrite_index(path: Path, value: dict[str, Any]) -> None:
    value.pop("record_sha256", None)
    _write(path, value)


def _exact_fixture(root: Path) -> tuple[Path, dict[str, Any]]:
    validations: list[dict[str, Any]] = []
    profiles: list[dict[str, Any]] = []
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
                        "profile_config_sha256": "b" * 64,
                        "model_id": model_id,
                        "seed": seed,
                        "precision": precision,
                        "input_shape": [batch_size, 128, 6],
                        "training_configuration_sha256": configuration_hash,
                        "split_manifest_sha256": "d" * 64,
                        "frozen_code_commit": "frozen-commit",
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
                        "environment": {"device_type": "cuda", "device": "cuda:0"},
                        "parameters": {
                            "total": 100 + model_index,
                            "trainable": 90 + model_index,
                            "state_tensor_bytes": 400,
                        },
                        "complexity": {
                            "supported_operator_macs_per_window": 1000 + model_index,
                            "estimated_flops_from_supported_macs_per_window": 2000
                            + 2 * model_index,
                            "coverage_status": "supported_operator_subset_only",
                        },
                        "latency_ms_per_batch": {
                            "mean": latency,
                            "median": latency,
                            "mean_per_window": latency / batch_size,
                            "percentiles": {"95.0": latency + 0.1},
                        },
                        "vram_bytes": {
                            "peak_allocated": 1000 + seed,
                            "peak_reserved": 2000 + seed,
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
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_index",
        "status": "complete_create_only",
        "execution": "sequential_cuda_one_frozen_model_seed_at_a_time",
        "required_device": "cuda",
        "cuda_available_at_start": True,
        "profile_config_sha256": "b" * 64,
        "final_freeze_inventory_sha256": "a" * 64,
        "frozen_artifact_set_sha256": "c" * 64,
        "split_manifest_sha256": "d" * 64,
        "frozen_code_commit": "frozen-commit",
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
    )


def test_exact_320_cell_efficiency_matrix_aggregates(tmp_path: Path) -> None:
    index_path, _ = _exact_fixture(tmp_path)
    result = _aggregate(index_path, tmp_path)
    assert result["validated_checkpoint_count"] == 80
    assert result["validated_profile_count"] == 320
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


def test_runner_rejects_escape_before_creating_directory(tmp_path: Path) -> None:
    escaped = tmp_path.parent / f"{tmp_path.name}-escaped"
    with pytest.raises(runner.PostconfirmatoryEfficiencyError, match="escapes output_root"):
        runner._output_directory(escaped, output_root=tmp_path)
    assert not escaped.exists()


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
        profile_config_sha256="a" * 64,
        inventory=None,
        context=None,
        profile_entries=[{"model_id": "partial"}],
        validation_entries=[{"model_id": "partial"}],
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
        ]
    )
    assert exit_code == 1
    assert "synthetic runner failure" in capsys.readouterr().err
