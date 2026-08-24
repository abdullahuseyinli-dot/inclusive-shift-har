from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from inclusive_shift_har.artifacts.final_freeze import (
    FrozenModelInput,
    build_final_freeze_inventory,
    write_final_freeze_inventory_new,
)
from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.evaluation.locked_target import (
    CANONICAL_OPENING_RECEIPT_NAME,
    LOCKED_TARGET_EVIDENCE_STATUS,
    LockedTargetEvaluationError,
    run_locked_target_evaluation,
)
from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    write_source_temperature_calibrator_new,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    save_classical_checkpoint_create_only,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.protocols.seal import TargetSealError

CLASSES = ("mobility", "sitting", "standing")
CODE_COMMIT = "synthetic-locked-runner-commit"


def _target_manifest() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_kind": "released_block_split",
        "target_performance_or_prediction_accessed": False,
        "target_seal": {
            "maximum_confirmatory_openings": 1,
            "performance_inspection": "forbidden",
            "seal_id": "e" * 64,
            "status": "sealed",
            "subject_ids": ["11", "12"],
            "unlock_record": None,
        },
        "windows": [{"partition": "target_sealed", "window_id": "metadata-only"}],
    }
    payload["split_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _separable_windows(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % len(CLASSES)
    windows = generator.normal(scale=0.05, size=(count, 128, 6)).astype(np.float32)
    windows[:, :, 0] += labels[:, None]
    return windows, labels


def _materialized_target() -> MaterializedWindows:
    windows, labels = _separable_windows(6, seed=7)
    return MaterializedWindows(
        signals=windows,
        labels=labels,
        window_ids=tuple(f"target-window-{index}" for index in range(6)),
        participant_ids=("11", "11", "11", "12", "12", "12"),
        released_labels=("Walking", "Sitting", "Standing") * 2,
        partitions=("target_sealed",) * 6,
        class_names=CLASSES,
        ontology_track="functional_core",
    )


def _unlock(inventory: dict[str, Any], inventory_path: Path) -> dict[str, Any]:
    return {
        "approved_at_utc": "2099-01-01T01:00:00Z",
        "code_commit": CODE_COMMIT,
        "final_gates": {
            "artifact_validation_passed": True,
            "configuration_validation_passed": True,
            "manifest_validation_passed": True,
            "protocol_lock_present": True,
            "split_audit_passed": True,
            "tests_passed": True,
            "type_checks_passed": True,
            "working_tree_clean_or_documented": True,
        },
        "gate": "final_evaluation_unlock",
        "protocol_lock_sha256": "d" * 64,
        "reason": "locked_confirmatory_evaluation",
        "schema_version": "1.0.0",
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "status": "approved",
        "target_opening_number": 1,
        "target_performance_previously_accessed": False,
        "target_seal_id": "e" * 64,
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "final_freeze_file_sha256": sha256_file(inventory_path),
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "source_models_and_calibrators_frozen": True,
    }


def _classical_freeze(
    tmp_path: Path,
    *,
    mismatched_inventory_config: bool = False,
) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    target_manifest = _target_manifest()
    train_windows, train_labels = _separable_windows(18, seed=2)
    train_participants = ["1"] * 9 + ["2"] * 9
    normalizer = ChannelStandardizer.fit(
        train_windows,
        train_participants,
        declared_training_participants={"1", "2"},
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
    )
    config = ClassicalConfig("logistic_regression", num_classes=3, seed=11)
    lineage = {
        "dataset_manifest_sha256": "a" * 64,
        "split_manifest_sha256": target_manifest["split_manifest_sha256"],
        "preprocessing_config_sha256": "c" * 64,
        "ontology_sha256": "f" * 64,
        "code_commit": CODE_COMMIT,
        "evidence_status": "source_only_final_training_target_sealed",
        "label_schema": list(CLASSES),
        "normalization": normalizer.to_dict(),
    }
    fitted = fit_classical_model(
        normalizer.transform(train_windows),
        train_labels,
        train_participants,
        config=config,
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
        lineage=lineage,
    )
    checkpoint_path = tmp_path / "frozen" / "logistic-seed11.pkl"
    save_classical_checkpoint_create_only(fitted, checkpoint_path)
    inventory_config = asdict(config)
    if mismatched_inventory_config:
        inventory_config["xgboost_device"] = "cuda"
    configuration_sha256 = canonical_json_sha256(inventory_config)
    calibration = build_source_temperature_calibrator_record(
        np.asarray(
            [[3.0, 0.0, -1.0], [0.0, 3.0, -1.0], [-1.0, 0.0, 3.0]],
            dtype=np.float64,
        ),
        np.asarray([0, 1, 2], dtype=np.int64),
        ["source-validation-1", "source-validation-2", "source-validation-3"],
        fit_partition="source_validation",
        checkpoint_sha256=sha256_file(checkpoint_path),
        training_configuration_sha256=configuration_sha256,
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        class_names=CLASSES,
    )
    calibrator_path = tmp_path / "frozen" / "logistic-seed11.calibrator.json"
    write_source_temperature_calibrator_new(
        calibration,
        calibrator_path,
        allowed_root=tmp_path,
    )
    inventory = build_final_freeze_inventory(
        [
            FrozenModelInput(
                model_id="logistic-regression",
                seed=11,
                training_regime="deterministic_classical",
                training_configuration=inventory_config,
                checkpoint_path=checkpoint_path,
                calibrator_path=calibrator_path,
                selected_epoch=None,
            )
        ],
        artifact_root=tmp_path,
        created_at_utc="2099-01-01T00:00:00Z",
        code_commit=CODE_COMMIT,
        dataset_manifest_sha256="a" * 64,
        source_window_manifest_sha256="b" * 64,
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        protocol_lock_sha256="d" * 64,
        target_seal_id="e" * 64,
        required_seed_order=[11],
    )
    inventory_path = tmp_path / "final-freeze.json"
    write_final_freeze_inventory_new(inventory, inventory_path, allowed_root=tmp_path)
    return inventory, inventory_path, target_manifest


def _run(
    tmp_path: Path,
    inventory: dict[str, Any],
    inventory_path: Path,
    target_manifest: dict[str, Any],
    materializer: Any,
    *,
    output_name: str = "target-results",
    unlock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return run_locked_target_evaluation(
        target_manifest,
        _unlock(inventory, inventory_path) if unlock is None else unlock,
        final_freeze_inventory_path=inventory_path,
        artifact_root=tmp_path,
        receipt_root=tmp_path,
        materialize_target=materializer,
        output_directory=tmp_path / output_name,
        output_root=tmp_path,
        opened_at_utc="2099-01-01T02:00:00Z",
        machine_record_sha256="9" * 64,
        device=torch.device("cpu"),
    )


def test_locked_runner_records_opening_before_materializing_and_preserves_order(
    tmp_path: Path,
) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(tmp_path)
    calls = 0

    def materializer(authorized: Any) -> MaterializedWindows:
        nonlocal calls
        calls += 1
        assert authorized["target_opening_number"] == 1
        assert (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).is_file()
        return _materialized_target()

    run = _run(tmp_path, inventory, inventory_path, target_manifest, materializer)
    assert calls == 1
    assert run["result_count"] == 1
    index = load_json_strict(run["index_path"])
    assert index["evidence_status"] == LOCKED_TARGET_EVIDENCE_STATUS
    result = load_json_strict(index["results"][0]["record_path"])
    assert result["evidence_status"] == LOCKED_TARGET_EVIDENCE_STATUS
    assert result["participant_level_report"]["participant_count"] == 2
    with np.load(index["results"][0]["array_path"], allow_pickle=False) as arrays:
        assert arrays["window_ids"].tolist() == list(_materialized_target().window_ids)
        assert arrays["participant_ids"].tolist() == list(_materialized_target().participant_ids)
        assert arrays["true_labels"].tolist() == _materialized_target().labels.tolist()
        assert np.allclose(arrays["calibrated_probabilities"].sum(axis=1), 1.0)
        assert arrays["evidence_status"].tolist() == [LOCKED_TARGET_EVIDENCE_STATUS]


def test_locked_runner_refuses_reused_canonical_receipt_before_materialization(
    tmp_path: Path,
) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(tmp_path)
    _run(tmp_path, inventory, inventory_path, target_manifest, lambda _: _materialized_target())
    called = False

    def forbidden(_: Any) -> MaterializedWindows:
        nonlocal called
        called = True
        return _materialized_target()

    with pytest.raises(TargetSealError, match="already has an opening receipt"):
        _run(
            tmp_path,
            inventory,
            inventory_path,
            target_manifest,
            forbidden,
            output_name="second-target-results",
        )
    assert called is False


@pytest.mark.parametrize("wrong_field", ["split_manifest_sha256", "target_seal_id"])
def test_locked_runner_refuses_wrong_unlock_lineage_without_materializing(
    tmp_path: Path,
    wrong_field: str,
) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(tmp_path)
    unlock = _unlock(inventory, inventory_path)
    unlock[wrong_field] = "0" * 64
    called = False

    def forbidden(_: Any) -> MaterializedWindows:
        nonlocal called
        called = True
        return _materialized_target()

    with pytest.raises(TargetSealError, match="mismatch"):
        _run(
            tmp_path,
            inventory,
            inventory_path,
            target_manifest,
            forbidden,
            unlock=unlock,
        )
    assert called is False
    assert not (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).exists()


def test_locked_runner_refuses_inventory_checkpoint_configuration_mismatch(
    tmp_path: Path,
) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(
        tmp_path, mismatched_inventory_config=True
    )
    with pytest.raises(LockedTargetEvaluationError, match="configuration differs"):
        _run(tmp_path, inventory, inventory_path, target_manifest, lambda _: _materialized_target())
    assert not (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).exists()


@pytest.mark.parametrize("artifact_role", ["checkpoint", "calibrator"])
def test_locked_runner_refuses_mutated_frozen_file_without_materializing(
    tmp_path: Path,
    artifact_role: str,
) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(tmp_path)
    artifact = tmp_path / inventory["models"][0][artifact_role]["path"]
    artifact.write_bytes(artifact.read_bytes() + b"mutation retained by synthetic test")
    called = False

    def forbidden(_: Any) -> MaterializedWindows:
        nonlocal called
        called = True
        return _materialized_target()

    with pytest.raises(LockedTargetEvaluationError, match="final freeze inventory is invalid"):
        _run(tmp_path, inventory, inventory_path, target_manifest, forbidden)
    assert called is False
    assert not (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).exists()


def test_locked_runner_refuses_missing_frozen_file_without_materializing(tmp_path: Path) -> None:
    inventory, inventory_path, target_manifest = _classical_freeze(tmp_path)
    checkpoint = tmp_path / inventory["models"][0]["checkpoint"]["path"]
    checkpoint.unlink()
    with pytest.raises(LockedTargetEvaluationError, match="final freeze inventory is invalid"):
        _run(tmp_path, inventory, inventory_path, target_manifest, lambda _: _materialized_target())
    assert not (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).exists()


def test_locked_runner_refuses_cpu_when_frozen_inventory_contains_neural_model(
    tmp_path: Path,
) -> None:
    _, classical_inventory_path, target_manifest = _classical_freeze(tmp_path)
    classical_inventory = load_json_strict(classical_inventory_path)
    checkpoint_path = tmp_path / classical_inventory["models"][0]["checkpoint"]["path"]
    neural_config = {
        "model_name": "compact96",
        "num_classes": 3,
        "seed": 11,
        "epochs": 2,
        "checkpoint_selection_rule": "fixed_last_epoch",
    }
    calibrator = build_source_temperature_calibrator_record(
        np.asarray([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0], [-1.0, 0.0, 2.0]]),
        np.asarray([0, 1, 2], dtype=np.int64),
        ["source-validation-a", "source-validation-b", "source-validation-c"],
        fit_partition="source_validation",
        checkpoint_sha256=sha256_file(checkpoint_path),
        training_configuration_sha256=canonical_json_sha256(neural_config),
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        class_names=CLASSES,
    )
    calibrator_path = tmp_path / "frozen" / "neural.calibrator.json"
    write_source_temperature_calibrator_new(calibrator, calibrator_path, allowed_root=tmp_path)
    inventory = build_final_freeze_inventory(
        [
            FrozenModelInput(
                model_id="compact96",
                seed=11,
                training_regime="fixed_epoch_neural",
                training_configuration=neural_config,
                checkpoint_path=checkpoint_path,
                calibrator_path=calibrator_path,
                selected_epoch=2,
            )
        ],
        artifact_root=tmp_path,
        created_at_utc="2099-01-01T00:00:00Z",
        code_commit=CODE_COMMIT,
        dataset_manifest_sha256="a" * 64,
        source_window_manifest_sha256="b" * 64,
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        protocol_lock_sha256="d" * 64,
        target_seal_id="e" * 64,
        required_seed_order=[11],
    )
    inventory_path = tmp_path / "neural-freeze.json"
    write_final_freeze_inventory_new(inventory, inventory_path, allowed_root=tmp_path)
    with pytest.raises(LockedTargetEvaluationError, match="requires CUDA"):
        _run(tmp_path, inventory, inventory_path, target_manifest, lambda _: _materialized_target())
    assert not (tmp_path / CANONICAL_OPENING_RECEIPT_NAME).exists()


def test_result_schema_example_matches_runtime_required_fields(tmp_path: Path) -> None:
    schema = json.loads(
        (Path(__file__).parents[1] / "configs/schema/locked_target_result.schema.json").read_text(
            encoding="utf-8"
        )
    )
    assert schema["properties"]["schema_version"]["const"] == "1.0.0"
    assert schema["properties"]["evidence_status"]["const"] == LOCKED_TARGET_EVIDENCE_STATUS
    assert "ordered_alignment" in schema["required"]
