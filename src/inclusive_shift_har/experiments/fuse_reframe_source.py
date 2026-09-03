"""Run one immutable, target-sealed FuSE-ReFrame source-development experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.v2_augmentation import PhysicalAugmentationConfig
from inclusive_shift_har.training.v2_engine import (
    V2TrainingConfig,
    exact_allowed_classes,
    train_fuse_reframe,
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
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("source manifest root must be an object")
    claimed = payload.get("source_window_manifest_sha256")
    unhashed = dict(payload)
    unhashed.pop("source_window_manifest_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError("source manifest self-hash does not validate")
    if payload.get("target_subject_or_window_records_included") is not False:
        raise PermissionError("FuSE-ReFrame runner refuses a target-bearing source manifest")
    if payload.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("FuSE-ReFrame runner refuses target-informed source evidence")
    records = payload.get("windows")
    if not isinstance(records, list) or any(
        item.get("partition") not in {"source_train", "source_validation"} for item in records
    ):
        raise PermissionError("FuSE-ReFrame runner received an unauthorized partition")
    return cast(dict[str, Any], payload)


def _partition_masks(
    participant_ids: NDArray[np.str_],
    partitions: NDArray[np.str_],
    manifest: dict[str, Any],
    *,
    fold_id: str,
) -> tuple[NDArray[np.bool_], NDArray[np.bool_]]:
    if fold_id == "final_source_split":
        train = partitions == "source_train"
        validation = partitions == "source_validation"
    else:
        fold = next(
            (item for item in manifest["source_cv_folds"] if str(item["fold_id"]) == fold_id),
            None,
        )
        if fold is None:
            raise ValueError(f"unknown source fold {fold_id!r}")
        train = np.isin(participant_ids, fold["train_subjects"])
        validation = np.isin(participant_ids, fold["validation_subjects"])
    if not train.any() or not validation.any() or np.any(train & validation):
        raise ValueError("source development masks are empty or overlapping")
    if set(participant_ids[train].tolist()) & set(participant_ids[validation].tolist()):
        raise ValueError("source participants overlap across train and validation")
    return np.asarray(train, dtype=np.bool_), np.asarray(validation, dtype=np.bool_)


def run_fuse_reframe_source(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    output_directory: Path,
    code_commit: str,
    fold_id: str,
    config: V2TrainingConfig,
    device_name: str,
) -> dict[str, Any]:
    """Materialize source windows only, fit source moments, and train one v2 cell."""

    manifest = _load_source_manifest(source_manifest_path)
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    class_names_raw = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names_raw != ("mobility", "sitting", "standing"):
        raise ValueError("FuSE hierarchy requires mobility/sitting/standing class order")
    class_names = class_names_raw
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    partitions = np.asarray(batch.partitions, dtype=np.str_)
    train_mask, validation_mask = _partition_masks(
        participants,
        partitions,
        manifest,
        fold_id=fold_id,
    )
    train_signals = np.asarray(batch.signals[train_mask], dtype=np.float32)
    train_labels = np.asarray(batch.labels[train_mask], dtype=np.int64)
    train_participants = participants[train_mask].tolist()
    validation_signals = np.asarray(batch.signals[validation_mask], dtype=np.float32)
    validation_labels = np.asarray(batch.labels[validation_mask], dtype=np.int64)
    validation_participants = participants[validation_mask].tolist()
    preprocessor = V2PhysicalPreprocessor.fit(
        train_signals,
        train_participants,
        declared_training_participants=set(train_participants),
        split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        channel_names=PRIMARY_CHANNELS,
    )
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    lineage = {
        "dataset_manifest_path": dataset_manifest_path.as_posix(),
        "dataset_manifest_sha256": sha256_file(dataset_manifest_path),
        "source_manifest_path": source_manifest_path.as_posix(),
        "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
        "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
        "preprocessing_config_sha256": manifest["preprocessing"]["config_sha256"],
        "ontology_sha256": manifest["ontology"]["config_sha256"],
        "source_fold_id": fold_id,
        "code_commit": code_commit,
        "evidence_status": "source_development_not_confirmatory",
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    return train_fuse_reframe(
        train_signals,
        exact_allowed_classes(train_labels),
        train_participants,
        validation_signals,
        validation_labels,
        validation_participants,
        preprocessor=preprocessor,
        config=config,
        class_names=class_names,
        lineage=lineage,
        output_directory=output_directory,
        device=torch.device(device_name),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--fold-id", default="final_source_split")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--minimum-epochs", type=int, default=12)
    parser.add_argument("--hidden-channels", type=int, default=64)
    parser.add_argument("--embedding-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--backbone",
        choices=("multiscale", "compact_residual", "inception", "tinyhar"),
        default="multiscale",
    )
    parser.add_argument(
        "--fusion-mode", choices=("raw", "invariant", "static", "gated"), default="gated"
    )
    parser.add_argument("--normalization", choices=("batch", "group", "layer"), default="group")
    parser.add_argument("--flat-head", action="store_true")
    parser.add_argument("--no-balanced-sampler", action="store_true")
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument("--augmented-classification-weight", type=float, default=0.5)
    parser.add_argument("--consistency-weight", type=float, default=0.1)
    parser.add_argument("--reconstruction-weight", type=float, default=0.05)
    parser.add_argument("--participant-cvar-weight", type=float, default=0.0)
    parser.add_argument("--participant-cvar-fraction", type=float, default=0.3)
    parser.add_argument("--branch-supervision-weight", type=float, default=0.0)
    parser.add_argument("--gate-supervision-weight", type=float, default=0.0)
    parser.add_argument("--gate-oracle-temperature", type=float, default=0.25)
    parser.add_argument("--vicreg-weight", type=float, default=0.0)
    parser.add_argument("--domain-adversarial-weight", type=float, default=0.0)
    parser.add_argument("--domain-grl-max-strength", type=float, default=1.0)
    parser.add_argument("--domain-grl-warmup-epochs", type=int, default=10)
    parser.add_argument(
        "--weight-averaging", choices=("none", "ema", "swa", "swad"), default="swad"
    )
    parser.add_argument("--swa-start-fraction", type=float, default=0.5)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = V2TrainingConfig(
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        patience=args.patience,
        minimum_epochs=args.minimum_epochs,
        hidden_channels=args.hidden_channels,
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        backbone=args.backbone,
        hierarchical=not args.flat_head,
        fusion_mode=args.fusion_mode,
        normalization=args.normalization,
        balanced_sampler=not args.no_balanced_sampler,
        use_physical_augmentation=not args.no_augmentation,
        augmentation=PhysicalAugmentationConfig(),
        augmented_classification_weight=args.augmented_classification_weight,
        consistency_weight=args.consistency_weight,
        reconstruction_weight=args.reconstruction_weight,
        participant_cvar_weight=args.participant_cvar_weight,
        participant_cvar_fraction=args.participant_cvar_fraction,
        branch_supervision_weight=args.branch_supervision_weight,
        gate_supervision_weight=args.gate_supervision_weight,
        gate_oracle_temperature=args.gate_oracle_temperature,
        vicreg_weight=args.vicreg_weight,
        domain_adversarial_weight=args.domain_adversarial_weight,
        domain_grl_max_strength=args.domain_grl_max_strength,
        domain_grl_warmup_epochs=args.domain_grl_warmup_epochs,
        weight_averaging=args.weight_averaging,
        swa_start_fraction=args.swa_start_fraction,
        mixed_precision=not args.no_mixed_precision,
    )
    result = run_fuse_reframe_source(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
        fold_id=args.fold_id,
        config=config,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
