from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from inclusive_shift_har.data.uci_har import UCI_HAR_CHANNELS, UCI_HAR_WINDOW_LENGTH
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.uci_reporting import (
    DEFAULT_UCI_REPRODUCTION_SEEDS,
    UCIReportingError,
    build_uci_reproduction_report,
    write_uci_reproduction_exports_new,
)
from inclusive_shift_har.experiments.uci_source import (
    UCI_CLASS_NAMES,
    UCI_CORRECTED_MODEL_IDS,
    build_uci_training_config,
    load_uci_reproduction_config,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

EXPERIMENT_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "uci_har_corrected_reproduction_v1_1.yaml"
)
EXPERIMENT_CONFIG_FILE_SHA256 = sha256_file(EXPERIMENT_CONFIG_PATH)


@pytest.fixture(autouse=True)
def _lightweight_checkpoint_reconstruction(monkeypatch: pytest.MonkeyPatch) -> None:
    def reconstruct(path: Path, *, device: torch.device) -> tuple[object, dict[str, Any]]:
        del device
        return object(), torch.load(path, map_location="cpu", weights_only=False)

    monkeypatch.setattr(
        "inclusive_shift_har.evaluation.uci_reporting.reconstruct_checkpoint", reconstruct
    )


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _window_hash(values: list[str]) -> str:
    return canonical_json_sha256(sorted(values))


def _materialize_complete_matrix(tmp_path: Path) -> tuple[Path, Path]:
    experiment_config = load_uci_reproduction_config(
        EXPERIMENT_CONFIG_PATH,
        expected_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
    )
    record_root = tmp_path / "records"
    artifact_root = tmp_path / "runs"
    record_root.mkdir()
    artifact_root.mkdir()
    folds: list[dict[str, Any]] = []
    all_window_ids: list[str] = []
    # Include a two-digit identifier so the fixture exercises the normalizer's
    # canonical lexicographic participant ordering rather than accidentally
    # matching the protocol's numeric subject order.
    synthetic_participants = ("1", "2", "3", "4", "11")
    for index, participant in enumerate(synthetic_participants, start=1):
        fold_id = f"uci_source_cv_{index:02d}"
        validation_ids = [f"uci_har:train:{participant}:{offset:06d}" for offset in range(6)]
        all_window_ids.extend(validation_ids)
        training_participants = [
            candidate for candidate in synthetic_participants if candidate != participant
        ]
        training_window_ids = [
            f"uci_har:train:{candidate}:{offset:06d}"
            for candidate in training_participants
            for offset in range(6)
        ]
        folds.append(
            {
                "fold_id": fold_id,
                "normalization_fit_subjects": training_participants,
                "train_subject_ids": training_participants,
                "train_window_count": len(training_window_ids),
                "train_window_ids_sha256": _window_hash(training_window_ids),
                "validation_subject_ids": [participant],
                "validation_window_count": 6,
                "validation_window_ids_sha256": _window_hash(validation_ids),
            }
        )
    protocol: dict[str, Any] = {
        "dataset_id": "uci_har_v1",
        "dataset_manifest_sha256": "b" * 64,
        "input": {
            "archive_sha256": "c" * 64,
            "released_split": "train",
            "window_count": 30,
            "window_ids_sha256": _window_hash(all_window_ids),
        },
        "leakage_controls": {"official_test_status": "legacy_exploratory_development_consumed"},
        "source_pretraining_policy": {"official_test_allowed_for_new_claims": False},
        "fold_assignment": {"folds": folds},
    }
    protocol["protocol_sha256"] = canonical_json_sha256(protocol)
    protocol_path = tmp_path / "protocol.json"
    _write_json(protocol_path, protocol)

    for model_index, model in enumerate(UCI_CORRECTED_MODEL_IDS):
        for seed in DEFAULT_UCI_REPRODUCTION_SEEDS:
            for fold_index, fold in enumerate(folds):
                fold_id = str(fold["fold_id"])
                participant = str(fold["validation_subject_ids"][0])
                labels = np.arange(6, dtype=np.int64)
                probabilities = np.full((6, 6), 0.02, dtype=np.float64)
                predicted = labels.copy()
                if model_index and (seed + fold_index) % (model_index + 2) == 0:
                    predicted[-1] = 0
                probabilities[np.arange(6), predicted] = 0.9
                probabilities /= probabilities.sum(axis=1, keepdims=True)
                participants = np.asarray([participant] * 6)
                window_ids = np.asarray(
                    [f"uci_har:train:{participant}:{offset:06d}" for offset in range(6)]
                )
                stem = f"{model}--seed-{seed}--{fold_id}"
                run_dir = artifact_root / stem / "attempt-001"
                run_dir.mkdir(parents=True)
                prediction_path = run_dir / "source_validation_predictions.npz"
                with prediction_path.open("xb") as stream:
                    np.savez_compressed(
                        stream,
                        logits=np.log(probabilities),
                        probabilities=probabilities,
                        labels=labels,
                        participant_ids=participants,
                        window_ids=window_ids,
                    )
                report = classification_report(
                    labels,
                    probabilities,
                    participants.tolist(),
                    class_names=UCI_CLASS_NAMES,
                )
                configuration = asdict(
                    build_uci_training_config(
                        experiment_config,
                        model_name=model,
                        seed=seed,
                    )
                )
                normalization = {
                    "schema_version": "1.0.0",
                    "method": "per_channel_population_standardization",
                    "mean": [0.0] * len(UCI_HAR_CHANNELS),
                    "scale": [1.0] * len(UCI_HAR_CHANNELS),
                    "training_participants": sorted(fold["train_subject_ids"]),
                    "split_manifest_sha256": protocol["protocol_sha256"],
                    "channel_names": list(UCI_HAR_CHANNELS),
                    "fitted_value_count_per_channel": (
                        int(fold["train_window_count"]) * UCI_HAR_WINDOW_LENGTH
                    ),
                    "fit_scope": "training_partition_only",
                }
                parameter_count = 1_000 + model_index
                selected_epoch = 3
                checkpoint_path = run_dir / "selected.pt"
                torch.save(
                    {
                        "configuration": configuration,
                        "configuration_sha256": canonical_json_sha256(configuration),
                        "checkpoint_selection_rule": "source_validation_best",
                        "checkpoint_role": "source_validation_selected",
                        "selected_epoch": selected_epoch,
                        "parameter_count": parameter_count,
                        "evidence_status": (
                            "corrected_uci_source_grouped_development_official_test_unopened"
                        ),
                        "label_schema": list(UCI_CLASS_NAMES),
                        "lineage": {
                            "dataset_manifest_sha256": "b" * 64,
                            "split_manifest_sha256": protocol["protocol_sha256"],
                            "code_commit": "a" * 40,
                            "normalization": normalization,
                        },
                    },
                    checkpoint_path,
                )
                record: dict[str, Any] = {
                    "schema_version": "1.0.0",
                    "status": "corrected_uci_source_fold_complete_official_test_unopened",
                    "evidence_status": "source_grouped_development_not_confirmatory",
                    "dataset_id": "uci_har_v1",
                    "dataset_manifest_sha256": "b" * 64,
                    "processed_archive_sha256": "c" * 64,
                    "source_protocol_sha256": protocol["protocol_sha256"],
                    "released_split_loaded": "train",
                    "official_test_member_opened": False,
                    "official_test_performance_or_prediction_accessed": False,
                    "inclusivehar_data_or_target_accessed": False,
                    "model_name": model,
                    "seed": seed,
                    "attempt": 1,
                    "prior_attempt_failures": [],
                    "code_commit": "a" * 40,
                    "experiment_config": {
                        "experiment_id": experiment_config.experiment_id,
                        "file_sha256": experiment_config.file_sha256,
                        "canonical_sha256": experiment_config.canonical_sha256,
                    },
                    "fold": fold,
                    "class_names": list(UCI_CLASS_NAMES),
                    "configuration": configuration,
                    "configuration_sha256": canonical_json_sha256(configuration),
                    "normalization": normalization,
                    "training": {
                        "parameter_count": parameter_count,
                        "selected_epoch": selected_epoch,
                        "checkpoint_path": checkpoint_path.relative_to(tmp_path).as_posix(),
                        "checkpoint_path_base": "result_root",
                        "checkpoint_sha256": sha256_file(checkpoint_path),
                    },
                    "validation_report": report,
                    "prediction_artifact": {
                        "path": prediction_path.relative_to(tmp_path).as_posix(),
                        "path_base": "result_root",
                        "sha256": sha256_file(prediction_path),
                    },
                    "device": {
                        "requested": "cuda",
                        "actual": "cuda",
                        "cudnn_enabled": not experiment_config.disable_cudnn_by_model[model],
                        "peak_vram_bytes": 123_456,
                    },
                    "elapsed_seconds": 1.5,
                }
                record["record_sha256"] = canonical_json_sha256(record)
                record_directory = record_root / stem
                record_directory.mkdir()
                _write_json(record_directory / "attempt-001.json", record)
    return record_root, protocol_path


def test_complete_uci_matrix_is_reconstructed_and_exported_create_only(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)

    report = build_uci_reproduction_report(
        record_directory=record_root,
        protocol_path=protocol_path,
        experiment_config_path=EXPERIMENT_CONFIG_PATH,
        expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
        bootstrap_resamples=100,
    )

    assert report["status"] == "corrected_uci_source_grouped_reproduction_complete"
    assert report["official_test_member_opened"] is False
    assert len(report["models"]) == 3
    assert all(row["participant_count"] == 5 for row in report["models"])
    assert len(report["all_pairwise_comparisons"]) == 3
    outputs = write_uci_reproduction_exports_new(report, output_directory=tmp_path / "exports")
    assert set(outputs) == {"json", "csv", "markdown"}
    with pytest.raises(FileExistsError):
        write_uci_reproduction_exports_new(report, output_directory=tmp_path / "exports")


def test_missing_run_fails_closed(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    next(record_root.rglob("*.json")).unlink()

    with pytest.raises(UCIReportingError, match="expected exactly"):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


def test_prediction_hash_tampering_fails_closed(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    (tmp_path / record["prediction_artifact"]["path"]).write_bytes(b"tampered")

    with pytest.raises(UCIReportingError, match="prediction artifact hash mismatch"):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    ("replacement", "message"),
    (("absolute", "output-root-relative"), ("../outside.npz", "escapes the run root")),
)
def test_prediction_reference_must_be_relative_and_confined(
    tmp_path: Path, replacement: str, message: str
) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if replacement == "absolute":
        replacement = str((tmp_path / record["prediction_artifact"]["path"]).resolve())
    record["prediction_artifact"]["path"] = replacement
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(UCIReportingError, match=message):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    (
        ("dataset_manifest_sha256", "dataset manifest lineage mismatch"),
        ("processed_archive_sha256", "processed archive lineage mismatch"),
    ),
)
def test_run_dataset_lineage_must_match_protocol_pins(
    tmp_path: Path, field: str, message: str
) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record[field] = "d" * 64
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(UCIReportingError, match=message):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


def test_locked_experiment_rejects_recurrent_backend_or_hyperparameter_drift(
    tmp_path: Path,
) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("legacy_bilstm_h192*/attempt-*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["configuration"]["disable_cudnn"] = False
    record["configuration_sha256"] = canonical_json_sha256(record["configuration"])
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(UCIReportingError, match="differs from the locked experiment"):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


def test_misaligned_prediction_vectors_fail_closed(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    prediction_path = tmp_path / record["prediction_artifact"]["path"]
    with np.load(prediction_path, allow_pickle=False) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    payload["labels"] = payload["labels"].reshape(2, 3)
    prediction_path.unlink()
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    record["prediction_artifact"]["sha256"] = sha256_file(prediction_path)
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(UCIReportingError, match="array alignment"):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


def test_window_label_identity_must_match_across_runs(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    record_path = next(record_root.rglob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    prediction_path = tmp_path / record["prediction_artifact"]["path"]
    with np.load(prediction_path, allow_pickle=False) as arrays:
        payload = {name: arrays[name] for name in arrays.files}
    labels = np.asarray(payload["labels"]).copy()
    labels[0] = 1
    payload["labels"] = labels
    prediction_path.unlink()
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **payload)
    record["prediction_artifact"]["sha256"] = sha256_file(prediction_path)
    record["validation_report"] = classification_report(
        labels,
        np.asarray(payload["probabilities"]),
        np.asarray(payload["participant_ids"]).tolist(),
        class_names=UCI_CLASS_NAMES,
    )
    record.pop("record_sha256")
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(record_path, record)

    with pytest.raises(UCIReportingError, match="identity differs across runs"):
        build_uci_reproduction_report(
            record_directory=record_root,
            protocol_path=protocol_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            bootstrap_resamples=100,
        )


def test_export_collision_is_preflighted_before_any_bundle_member(tmp_path: Path) -> None:
    record_root, protocol_path = _materialize_complete_matrix(tmp_path)
    report = build_uci_reproduction_report(
        record_directory=record_root,
        protocol_path=protocol_path,
        experiment_config_path=EXPERIMENT_CONFIG_PATH,
        expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
        bootstrap_resamples=100,
    )
    output = tmp_path / "exports"
    output.mkdir()
    (output / "uci_source_grouped_model_summary_v1.csv").write_text("preserve me", encoding="utf-8")

    with pytest.raises(FileExistsError, match="partial export bundle"):
        write_uci_reproduction_exports_new(report, output_directory=output)

    assert not (output / "uci_source_grouped_report_v1.json").exists()
    assert not (output / "uci_source_grouped_model_summary_v1.md").exists()
