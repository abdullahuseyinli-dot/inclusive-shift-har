from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.artifacts.final_freeze import validate_final_freeze_inventory_file
from inclusive_shift_har.artifacts.source_finalization import (
    SourceFinalizationError,
    freeze_final_source_runs,
    load_final_selection_plan,
)
from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    predict_classical_model,
    save_classical_checkpoint_create_only,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer

CLASSES = ("mobility", "sitting", "standing")


def _windows(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 3
    windows = generator.normal(scale=0.1, size=(count, 128, 6)).astype(np.float32)
    windows[:, :, 0] += labels[:, None]
    return windows, labels


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    split_hash = "c" * 64
    train_windows, train_labels = _windows(18, seed=1)
    validation_windows, validation_labels = _windows(6, seed=2)
    train_participants = ["1"] * 9 + ["2"] * 9
    validation_participants = ["8"] * 3 + ["10"] * 3
    normalizer = ChannelStandardizer.fit(
        train_windows,
        train_participants,
        declared_training_participants={"1", "2"},
        split_manifest_sha256=split_hash,
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
    )
    config = ClassicalConfig("logistic_regression", num_classes=3, seed=11)
    fitted = fit_classical_model(
        normalizer.transform(train_windows),
        train_labels,
        train_participants,
        config=config,
        channel_names=INCLUSIVEHAR_PRIMARY_CHANNELS,
        lineage={
            "dataset_manifest_sha256": "a" * 64,
            "split_manifest_sha256": split_hash,
            "preprocessing_config_sha256": "f" * 64,
            "ontology_sha256": "9" * 64,
            "code_commit": "final-code-commit",
            "evidence_status": "source_only_final_training_target_sealed",
            "label_schema": list(CLASSES),
            "normalization": normalizer.to_dict(),
        },
    )
    checkpoint = tmp_path / "runs" / "logistic" / "seed-11" / "selected.pkl"
    checkpoint_record = save_classical_checkpoint_create_only(fitted, checkpoint)
    logits, probabilities, report = predict_classical_model(
        fitted,
        normalizer.transform(validation_windows),
        validation_labels,
        validation_participants,
        class_names=CLASSES,
    )
    prediction_path = checkpoint.parent / "source_validation_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=logits,
            probabilities=probabilities,
            labels=validation_labels,
            participant_ids=np.asarray(validation_participants),
            window_ids=np.asarray([f"source-validation-{index}" for index in range(6)]),
        )
    configuration = asdict(config)
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "source_development_complete_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "model_name": "logistic_regression",
        "seed": 11,
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "source_window_manifest_sha256": "b" * 64,
        "split_manifest_sha256": split_hash,
        "source_split_id": "final_source_split",
        "class_names": list(CLASSES),
        "normalization": normalizer.to_dict(),
        "train_window_count": 18,
        "validation_window_count": 6,
        "train_participants": ["1", "2"],
        "validation_participants": ["8", "10"],
        "validation_report": report,
        "checkpoint": checkpoint_record,
        "prediction_artifact": {
            "path": prediction_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "parameter_count": None,
        "best_epoch": None,
        "training_history": [],
        "elapsed_seconds": 1.0,
        "peak_vram_bytes": None,
        "requested_device": "cuda",
        "model_device": "cpu",
        "code_commit": "final-code-commit",
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    summary_path = tmp_path / "records" / "logistic--seed-11.json"
    _write_json(summary_path, summary)
    plan: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "final_source_selection_plan",
        "status": "predeclared_source_only_target_sealed",
        "created_at_utc": "2099-01-01T00:00:00Z",
        "code_commit_policy": "exact_current_commit_at_training_and_freeze",
        "dataset_manifest_sha256": "a" * 64,
        "source_window_manifest_sha256": "b" * 64,
        "split_manifest_sha256": split_hash,
        "protocol_lock_sha256": "d" * 64,
        "target_seal_id": "e" * 64,
        "required_seed_order": [11],
        "final_source_split_id": "final_source_split",
        "final_train_participants": ["1", "2"],
        "final_validation_participants": ["8", "10"],
        "final_train_window_count": 18,
        "final_validation_window_count": 6,
        "class_names": list(CLASSES),
        "models": [
            {
                "model_id": "logistic-regression",
                "runner_model_name": "logistic_regression",
                "training_regime": "deterministic_classical",
                "summary_path_template": "records/logistic--seed-{seed}.json",
                "run_directory_template": "runs/logistic--seed-{seed}",
                "runner_arguments": {
                    "epochs": 1,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                },
                "configuration_expectations": {
                    "model_name": "logistic_regression",
                    "num_classes": 3,
                    "xgboost_device": "cpu",
                },
            }
        ],
        "target_subject_or_window_records_used": False,
        "target_predictions_or_performance_accessed": False,
    }
    plan["selection_plan_sha256"] = canonical_json_sha256(plan)
    plan_path = tmp_path / "selection-plan.json"
    _write_json(plan_path, plan)
    return plan_path, summary_path


def test_source_finalization_freezes_exact_source_only_artifacts(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    result = freeze_final_source_runs(
        plan_path,
        artifact_root=tmp_path,
        calibrator_directory="calibrators",
        inventory_path="final-freeze.json",
        created_at_utc="2099-01-01T01:00:00Z",
        expected_code_commit="final-code-commit",
    )
    assert result["status"] == "frozen_source_only_target_sealed"
    assert result["model_seed_count"] == 1
    report = validate_final_freeze_inventory_file(
        tmp_path / "final-freeze.json", artifact_root=tmp_path
    )
    assert report.valid, report.to_dict()


def test_source_finalization_rejects_changed_summary_before_calibration(tmp_path: Path) -> None:
    plan_path, summary_path = _fixture(tmp_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["validation_window_count"] = 5
    _write_json(summary_path, summary)
    with pytest.raises(SourceFinalizationError, match="self-hash"):
        freeze_final_source_runs(
            plan_path,
            artifact_root=tmp_path,
            calibrator_directory="calibrators",
            inventory_path="final-freeze.json",
            created_at_utc="2099-01-01T01:00:00Z",
            expected_code_commit="final-code-commit",
        )
    assert not (tmp_path / "calibrators").exists()


def test_final_selection_plan_rejects_target_access_claim(tmp_path: Path) -> None:
    plan_path, _ = _fixture(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["target_predictions_or_performance_accessed"] = True
    plan.pop("selection_plan_sha256")
    plan["selection_plan_sha256"] = canonical_json_sha256(plan)
    _write_json(plan_path, plan)
    with pytest.raises(SourceFinalizationError, match="contract mismatch"):
        load_final_selection_plan(plan_path)
