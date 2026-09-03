"""Run one true nested, target-sealed FuSE-ReFrame outer source fold."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.v2_augmentation import PhysicalAugmentationConfig
from inclusive_shift_har.training.v2_engine import (
    V2TrainingConfig,
    exact_allowed_classes,
    train_fuse_reframe,
    train_fuse_reframe_fixed_evaluation,
)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_candidates(path: Path, *, seed: int) -> list[tuple[str, V2TrainingConfig]]:
    payload = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="candidate file")
    defaults = _mapping(payload.get("defaults", {}), name="candidate defaults")
    raw_candidates = payload.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise ValueError("candidate file must contain at least one candidate")
    candidates: list[tuple[str, V2TrainingConfig]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_candidates):
        candidate = _mapping(raw, name=f"candidates[{index}]")
        candidate_id = str(candidate.get("id", ""))
        if (
            not candidate_id
            or any(
                character not in "abcdefghijklmnopqrstuvwxyz0123456789_-"
                for character in candidate_id
            )
            or candidate_id in seen
        ):
            raise ValueError(f"invalid or duplicate candidate id {candidate_id!r}")
        seen.add(candidate_id)
        values = {**defaults, **{key: value for key, value in candidate.items() if key != "id"}}
        values["seed"] = seed
        augmentation = values.get("augmentation")
        if isinstance(augmentation, dict):
            values["augmentation"] = PhysicalAugmentationConfig(**augmentation)
        candidates.append((candidate_id, V2TrainingConfig(**values)))
    return candidates


def _mask_for_subjects(participant_ids: NDArray[np.str_], subjects: list[str]) -> NDArray[np.bool_]:
    mask = np.asarray(np.isin(participant_ids, subjects), dtype=np.bool_)
    if not mask.any() or set(participant_ids[mask].tolist()) != set(subjects):
        raise ValueError("declared fold subjects are absent from materialized source windows")
    return mask


def _lower_fraction_mean(values: list[float], fraction: float = 0.3) -> float:
    if not values or not 0 < fraction <= 1:
        raise ValueError("lower-tail summary inputs are invalid")
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    mass = fraction * ordered.size
    full = math.floor(mass)
    remainder = mass - full
    numerator = float(ordered[:full].sum())
    if remainder > 0:
        numerator += remainder * float(ordered[full])
    return numerator / mass


def _candidate_summary(candidate_id: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    participant_rows: list[dict[str, Any]] = []
    confusion = np.zeros((3, 3), dtype=np.int64)
    for record in records:
        report = _mapping(record["validation_report"], name="inner validation report")
        participant_rows.extend(cast(list[dict[str, Any]], report["participants"]))
        confusion += np.asarray(
            report["window_level_diagnostics"]["confusion_matrix"], dtype=np.int64
        )
    identifiers = [str(row["participant_id"]) for row in participant_rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("an inner-validation participant was evaluated more than once")
    macro = [float(row["macro_f1"]) for row in participant_rows]
    recalls = np.diag(confusion) / np.maximum(confusion.sum(axis=1), 1)
    first = records[0]
    return {
        "candidate_id": candidate_id,
        "configuration": first["configuration"],
        "configuration_sha256": first["configuration_sha256"],
        "inner_fold_count": len(records),
        "inner_participant_count": len(participant_rows),
        "mean_participant_macro_f1": float(np.mean(macro)),
        "lower_30_percent_participant_macro_f1": _lower_fraction_mean(macro),
        "worst_participant_macro_f1": min(macro),
        "class_recall": {
            "mobility": float(recalls[0]),
            "sitting": float(recalls[1]),
            "standing": float(recalls[2]),
        },
        "parameter_count": int(first["parameter_count"]),
        "inner_selected_epochs": [int(record["selection"]["epoch"]) for record in records],
        "inner_epochs_completed": [len(record["history"]) for record in records],
        "inner_selection": [record["selection"] for record in records],
        "inner_result_files": [record["_result_reference"] for record in records],
    }


def _selection_key(summary: dict[str, Any]) -> tuple[float, float, int, float, str]:
    configuration = _mapping(summary["configuration"], name="candidate configuration")
    return (
        -float(summary["mean_participant_macro_f1"]),
        -float(summary["lower_30_percent_participant_macro_f1"]),
        int(summary["parameter_count"]),
        float(configuration["learning_rate"]),
        str(summary["candidate_id"]),
    )


def _inner_derived_schedule(
    summary: dict[str, Any], config: V2TrainingConfig
) -> tuple[int, tuple[int, int] | None]:
    selections = cast(list[dict[str, Any]], summary["inner_selection"])
    if config.weight_averaging == "none":
        epochs = int(statistics.median_low(cast(list[int], summary["inner_selected_epochs"])))
        return max(1, epochs), None
    completed = cast(list[int], summary["inner_epochs_completed"])
    epochs = max(1, int(statistics.median_low(completed)))
    if config.weight_averaging != "swad":
        return epochs, None
    starts = [int(item["start_epoch"]) for item in selections if item.get("method") == "swad"]
    ends = [int(item["end_epoch"]) for item in selections if item.get("method") == "swad"]
    if len(starts) != len(selections) or len(ends) != len(selections):
        raise ValueError("SWAD may advance only when every inner fold produced a SWAD interval")
    start = int(statistics.median_low(starts))
    end = min(epochs, int(statistics.median_low(ends)))
    if start > end:
        raise ValueError("inner-derived SWAD interval is empty in the outer refit schedule")
    return epochs, (start, end)


def run_nested_outer_fold(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    candidate_path: Path,
    output_directory: Path,
    code_commit: str,
    outer_fold_id: str,
    seed: int,
    device_name: str,
) -> dict[str, Any]:
    """Select in inner folds, fixed-refit, and open one source outer fold once."""

    if output_directory.exists():
        raise FileExistsError(f"nested output directory already exists: {output_directory}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = _load_source_manifest(source_manifest_path)
    candidates = _load_candidates(candidate_path, seed=seed)
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("nested FuSE evaluation requires the functional three-class ontology")
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(batch.signals, dtype=np.float32)
    labels = np.asarray(batch.labels, dtype=np.int64)
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    outer = next(
        (
            item
            for item in cast(list[dict[str, Any]], manifest["source_nested_cv"])
            if str(item["outer_fold_id"]) == outer_fold_id
        ),
        None,
    )
    if outer is None:
        raise ValueError(f"unknown nested outer fold {outer_fold_id!r}")
    output_directory.mkdir(parents=True)
    inner_candidate_summaries: list[dict[str, Any]] = []
    for candidate_id, config in candidates:
        inner_records: list[dict[str, Any]] = []
        for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
            inner_train_subjects = [str(item) for item in inner["train_subjects"]]
            inner_validation_subjects = [str(item) for item in inner["validation_subjects"]]
            train_mask = _mask_for_subjects(participants, inner_train_subjects)
            validation_mask = _mask_for_subjects(participants, inner_validation_subjects)
            if np.any(train_mask & validation_mask):
                raise ValueError("nested inner train and validation masks overlap")
            preprocessor = V2PhysicalPreprocessor.fit(
                signals[train_mask],
                participants[train_mask].tolist(),
                declared_training_participants=set(inner_train_subjects),
                split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
                channel_names=PRIMARY_CHANNELS,
            )
            inner_id = str(inner["fold_id"])
            inner_output = output_directory / "inner" / candidate_id / inner_id
            result = train_fuse_reframe(
                signals[train_mask],
                exact_allowed_classes(labels[train_mask]),
                participants[train_mask].tolist(),
                signals[validation_mask],
                labels[validation_mask],
                participants[validation_mask].tolist(),
                preprocessor=preprocessor,
                config=config,
                class_names=class_names,
                lineage={
                    "code_commit": code_commit,
                    "candidate_id": candidate_id,
                    "outer_fold_id": outer_fold_id,
                    "inner_fold_id": inner_id,
                    "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
                    "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
                    "candidate_file_sha256": sha256_file(candidate_path),
                    "evidence_status": "nested_inner_source_development_not_confirmatory",
                    "target_subject_or_window_records_loaded": False,
                    "target_performance_or_prediction_accessed": False,
                },
                output_directory=inner_output,
                device=torch.device(device_name),
            )
            result_path = inner_output / "result.json"
            result["_result_reference"] = {
                "path": result_path.as_posix(),
                "sha256": sha256_file(result_path),
            }
            inner_records.append(result)
        expected_inner_subjects = set(str(item) for item in outer["outer_test_subjects"])
        observed_validation_subjects = {
            str(participant)
            for record in inner_records
            for participant in record["validation_participants"]
        }
        if expected_inner_subjects & observed_validation_subjects:
            raise PermissionError("outer-test participant leaked into inner model selection")
        inner_candidate_summaries.append(_candidate_summary(candidate_id, inner_records))

    ranked = sorted(inner_candidate_summaries, key=_selection_key)
    selected_summary = ranked[0]
    selected_id = str(selected_summary["candidate_id"])
    selected_config = next(
        config for candidate_id, config in candidates if candidate_id == selected_id
    )
    fixed_epochs, averaging_interval = _inner_derived_schedule(selected_summary, selected_config)
    outer_test_subjects = [str(item) for item in outer["outer_test_subjects"]]
    outer_train_subjects = sorted(set(participants.tolist()) - set(outer_test_subjects))
    outer_train_mask = _mask_for_subjects(participants, outer_train_subjects)
    outer_test_mask = _mask_for_subjects(participants, outer_test_subjects)
    preprocessor = V2PhysicalPreprocessor.fit(
        signals[outer_train_mask],
        participants[outer_train_mask].tolist(),
        declared_training_participants=set(outer_train_subjects),
        split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        channel_names=PRIMARY_CHANNELS,
    )
    ranking_digest = canonical_json_sha256(ranked)
    outer_output = output_directory / "outer" / selected_id
    outer_result = train_fuse_reframe_fixed_evaluation(
        signals[outer_train_mask],
        exact_allowed_classes(labels[outer_train_mask]),
        participants[outer_train_mask].tolist(),
        signals[outer_test_mask],
        labels[outer_test_mask],
        participants[outer_test_mask].tolist(),
        preprocessor=preprocessor,
        config=selected_config,
        fixed_epochs=fixed_epochs,
        averaging_interval=averaging_interval,
        class_names=class_names,
        lineage={
            "code_commit": code_commit,
            "candidate_id": selected_id,
            "outer_fold_id": outer_fold_id,
            "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
            "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
            "candidate_file_sha256": sha256_file(candidate_path),
            "inner_candidate_ranking_sha256": ranking_digest,
            "fixed_schedule_derived_without_outer_evaluation": True,
            "outer_test_used_for_selection": False,
            "evidence_status": "nested_outer_source_development_not_confirmatory",
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        },
        output_directory=outer_output,
        device=torch.device(device_name),
    )
    outer_result_path = outer_output / "result.json"
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_nested_outer_selection_and_evaluation",
        "status": "complete_target_sealed",
        "evidence_status": "nested_outer_source_development_not_confirmatory",
        "outer_fold_id": outer_fold_id,
        "seed": seed,
        "candidate_file": {
            "path": candidate_path.as_posix(),
            "sha256": sha256_file(candidate_path),
        },
        "candidate_count": len(ranked),
        "inner_candidate_ranking": ranked,
        "inner_candidate_ranking_sha256": ranking_digest,
        "selection": {
            "candidate_id": selected_id,
            "configuration": asdict(selected_config),
            "fixed_epochs": fixed_epochs,
            "averaging_interval": list(averaging_interval) if averaging_interval else None,
            "metric": "inner_mean_participant_macro_f1",
            "outer_fold_accessed_for_selection": False,
        },
        "outer_result": {
            "path": outer_result_path.as_posix(),
            "sha256": sha256_file(outer_result_path),
            "evaluation_report": outer_result["evaluation_report"],
            "evaluation_access_count_during_training": 0,
            "evaluation_access_count_after_training": 1,
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
            "canonical_record_sha256": manifest["source_window_manifest_sha256"],
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "code_commit": code_commit,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    summary_path = output_directory / "nested_result.json"
    with summary_path.open("xb") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--candidate-configs", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--outer-fold-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_nested_outer_fold(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        candidate_path=args.candidate_configs,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
        outer_fold_id=args.outer_fold_id,
        seed=args.seed,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
