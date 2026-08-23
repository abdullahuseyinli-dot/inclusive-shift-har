"""Reproducible source-only InclusiveHAR development runner; target access is impossible here."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    predict_classical_model,
    save_classical_checkpoint_create_only,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    train_source_model,
)

PRIMARY_CHANNELS = (
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
)


def _load_source_manifest(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    claimed_hash = payload.get("source_window_manifest_sha256")
    without_hash = dict(payload)
    without_hash.pop("source_window_manifest_sha256", None)
    if claimed_hash != canonical_json_sha256(without_hash):
        raise ValueError("source window manifest self-hash does not validate")
    if (
        payload.get("target_subject_or_window_records_included") is not False
        or payload.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("source runner refuses any target-bearing manifest")
    windows = payload.get("windows")
    if not isinstance(windows, list) or any(
        item.get("partition") not in {"source_train", "source_validation"} for item in windows
    ):
        raise PermissionError("source runner received an unauthorized partition")
    return payload


def _materialize_source(manifest: dict[str, Any], raw_csv: Path) -> tuple[Any, tuple[str, ...]]:
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    track_schema = manifest["ontology"]["runnable_track_schemas"]["functional_core"]
    class_names = tuple(str(value) for value in track_schema["class_order"])
    batch = materialize_inclusivehar_windows(
        raw_csv,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    return batch, class_names


def _normalize_source_split(
    batch: Any,
    manifest: dict[str, Any],
    *,
    fold_id: str,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int64],
    list[str],
    tuple[str, ...],
    NDArray[np.float32],
    NDArray[np.int64],
    list[str],
    tuple[str, ...],
    ChannelStandardizer,
]:
    participant_array = np.asarray(batch.participant_ids)
    if fold_id == "final_source_split":
        partitions = np.asarray(batch.partitions)
        train_mask = partitions == "source_train"
        validation_mask = partitions == "source_validation"
    else:
        fold = next(
            (
                candidate
                for candidate in manifest["source_cv_folds"]
                if candidate["fold_id"] == fold_id
            ),
            None,
        )
        if fold is None:
            choices = [candidate["fold_id"] for candidate in manifest["source_cv_folds"]]
            raise ValueError(f"unknown source fold {fold_id!r}; choices: {choices}")
        train_mask = np.isin(participant_array, fold["train_subjects"])
        validation_mask = np.isin(participant_array, fold["validation_subjects"])
    if not train_mask.any() or not validation_mask.any() or np.any(train_mask & validation_mask):
        raise ValueError("locked final source split is incomplete or overlapping")
    standardizer = ChannelStandardizer.fit(
        batch.signals[train_mask],
        participant_array[train_mask].tolist(),
        declared_training_participants=set(participant_array[train_mask].tolist()),
        split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        channel_names=PRIMARY_CHANNELS,
    )
    train_window_ids = tuple(np.asarray(batch.window_ids)[train_mask].tolist())
    validation_window_ids = tuple(np.asarray(batch.window_ids)[validation_mask].tolist())
    return (
        standardizer.transform(batch.signals[train_mask]),
        np.asarray(batch.labels[train_mask], dtype=np.int64),
        participant_array[train_mask].tolist(),
        train_window_ids,
        standardizer.transform(batch.signals[validation_mask]),
        np.asarray(batch.labels[validation_mask], dtype=np.int64),
        participant_array[validation_mask].tolist(),
        validation_window_ids,
        standardizer,
    )


def _save_predictions_create_only(
    path: Path,
    *,
    logits: NDArray[np.float64],
    probabilities: NDArray[np.float64],
    labels: NDArray[np.int64],
    participant_ids: list[str],
    window_ids: tuple[str, ...],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            logits=logits,
            probabilities=probabilities,
            labels=labels,
            participant_ids=np.asarray(participant_ids),
            window_ids=np.asarray(window_ids),
        )
    return sha256_file(path)


def _write_summary_create_only(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return sha256_file(path)


def run_source_development(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    model_name: str,
    seed: int,
    run_directory: Path,
    summary_path: Path,
    code_commit: str,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    device_name: str,
    fold_id: str = "final_source_split",
    use_augmentation: bool = False,
    use_content_objective: bool = False,
    use_realization_factorization: bool = False,
    use_group_dro: bool = False,
    coral_weight: float = 0.0,
    disable_cudnn: bool = False,
) -> dict[str, Any]:
    """Execute one immutable final-source-split development trial."""

    manifest = _load_source_manifest(source_manifest_path)
    batch, class_names = _materialize_source(manifest, raw_csv_path)
    (
        train_windows,
        train_labels,
        train_participants,
        _,
        validation_windows,
        validation_labels,
        validation_participants,
        validation_window_ids,
        standardizer,
    ) = _normalize_source_split(batch, manifest, fold_id=fold_id)
    run_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    classical_names = {"random_forest", "xgboost", "svm_rbf", "logistic_regression"}
    if model_name in classical_names:
        classical_config = ClassicalConfig(
            model_name=model_name,
            num_classes=len(class_names),
            seed=seed,
            xgboost_device="cuda" if model_name == "xgboost" and device.type == "cuda" else "cpu",
        )
        fitted = fit_classical_model(
            train_windows,
            train_labels,
            train_participants,
            config=classical_config,
            channel_names=PRIMARY_CHANNELS,
            lineage={
                "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
                "split_manifest_sha256": manifest["source_split_manifest_sha256"],
                "normalization": standardizer.to_dict(),
                "code_commit": code_commit,
                "evidence_status": "source_development_only_target_sealed",
            },
        )
        logits, probabilities, report = predict_classical_model(
            fitted,
            validation_windows,
            validation_labels,
            validation_participants,
            class_names=class_names,
        )
        checkpoint_record = save_classical_checkpoint_create_only(
            fitted, run_directory / "selected.pkl"
        )
        configuration = asdict(classical_config)
        parameter_count = None
        best_epoch = None
        history: list[dict[str, Any]] = []
        model_device = (
            "cuda"
            if model_name == "xgboost" and classical_config.xgboost_device == "cuda"
            else "cpu"
        )
    else:
        neural_config = TrainingConfig(
            model_name=model_name,
            num_classes=len(class_names),
            seed=seed,
            epochs=epochs,
            batch_size=256,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            patience=min(12, max(2, epochs // 3)),
            minimum_epochs=min(8, epochs),
            mixed_precision="float16" if device.type == "cuda" else "disabled",
            checkpoint_interval=max(epochs, 1),
            disable_cudnn=disable_cudnn,
            use_augmentation=use_augmentation,
            use_content_objective=use_content_objective,
            use_realization_factorization=use_realization_factorization,
            use_group_dro=use_group_dro,
            coral_weight=coral_weight,
        )
        lineage = TrainingLineage(
            dataset_manifest_sha256=sha256_file(dataset_manifest_path),
            split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
            preprocessing_config_sha256=str(manifest["preprocessing"]["config_sha256"]),
            ontology_sha256=str(manifest["ontology"]["config_sha256"]),
            code_commit=code_commit,
            evidence_status="source_development_only_target_sealed",
            label_schema=class_names,
            normalization=standardizer.to_dict(),
        )
        trained = train_source_model(
            train_windows,
            train_labels,
            train_participants,
            validation_windows,
            validation_labels,
            validation_participants,
            config=neural_config,
            lineage=lineage,
            output_directory=run_directory,
            device=device,
        )
        logits = trained.pop("validation_logits")
        probabilities = trained.pop("validation_probabilities")
        report = trained.pop("validation_report")
        checkpoint_record = {
            "path": trained["checkpoint_path"],
            "sha256": trained["checkpoint_sha256"],
        }
        configuration = asdict(neural_config)
        parameter_count = trained["parameter_count"]
        best_epoch = trained["best_epoch"]
        history = trained["history"]
        model_device = device.type

    prediction_path = run_directory / "source_validation_predictions.npz"
    prediction_sha256 = _save_predictions_create_only(
        prediction_path,
        logits=logits,
        probabilities=probabilities,
        labels=validation_labels,
        participant_ids=validation_participants,
        window_ids=validation_window_ids,
    )
    elapsed = time.perf_counter() - started
    peak_vram = int(torch.cuda.max_memory_allocated(device)) if model_device == "cuda" else None
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "source_development_complete_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "model_name": model_name,
        "seed": seed,
        "configuration": configuration,
        "configuration_sha256": canonical_json_sha256(configuration),
        "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
        "split_manifest_sha256": manifest["source_split_manifest_sha256"],
        "source_split_id": fold_id,
        "class_names": list(class_names),
        "normalization": standardizer.to_dict(),
        "train_window_count": int(train_labels.size),
        "validation_window_count": int(validation_labels.size),
        "train_participants": sorted(set(train_participants), key=int),
        "validation_participants": sorted(set(validation_participants), key=int),
        "validation_report": report,
        "checkpoint": checkpoint_record,
        "prediction_artifact": {
            "path": str(prediction_path),
            "sha256": prediction_sha256,
        },
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "training_history": history,
        "elapsed_seconds": elapsed,
        "peak_vram_bytes": peak_vram,
        "requested_device": device.type,
        "model_device": model_device,
        "code_commit": code_commit,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_summary_create_only(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--fold-id", default="final_source_split")
    parser.add_argument("--augmentation", action="store_true")
    parser.add_argument("--content-objective", action="store_true")
    parser.add_argument("--realization-factorization", action="store_true")
    parser.add_argument("--group-dro", action="store_true")
    parser.add_argument("--coral-weight", type=float, default=0.0)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="keep CUDA tensors but use the non-cuDNN recurrent backend",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_source_development(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        model_name=args.model,
        seed=args.seed,
        run_directory=args.run_directory,
        summary_path=args.summary,
        code_commit=args.code_commit,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device_name=args.device,
        fold_id=args.fold_id,
        use_augmentation=args.augmentation,
        use_content_objective=args.content_objective,
        use_realization_factorization=args.realization_factorization,
        use_group_dro=args.group_dro,
        coral_weight=args.coral_weight,
        disable_cudnn=args.disable_cudnn,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
