"""Replay frozen nested Geometric Spectral Pyramid models on source corruptions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.v2_corruptions import CorruptionSpec, apply_sensor_corruption
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import _probabilities
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> tuple[dict[str, Any], tuple[CorruptionSpec, ...]]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("selection_use_prohibited") is not True:
        raise ValueError("corruption diagnostics must be prohibited from model selection")
    if config.get("target_data_allowed") is not False:
        raise ValueError("corruption diagnostics must explicitly forbid target data")
    raw_suite = config.get("suite")
    if not isinstance(raw_suite, list) or not raw_suite:
        raise ValueError("corruption config requires a non-empty suite")
    specs = tuple(
        CorruptionSpec(
            cast(Any, _mapping(item, name=f"suite[{index}]")["name"]),
            float(_mapping(item, name=f"suite[{index}]")["severity"]),
        )
        for index, item in enumerate(raw_suite)
    )
    keys = [(spec.name, spec.severity) for spec in specs]
    if keys.count(("clean", 0.0)) != 1 or len(set(keys)) != len(keys):
        raise ValueError("corruption suite must contain one clean case and unique cases")
    if float(config["sampling_rate_hz"]) <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not 0.95 <= float(config["clipping_quantile"]) < 1.0:
        raise ValueError("clipping_quantile must lie in [0.95, 1.0)")
    return config, specs


def train_only_corruption_scales(
    signals: NDArray[np.floating[Any]], *, clipping_quantile: float
) -> tuple[FloatArray, FloatArray]:
    """Estimate corruption magnitudes from an outer training partition only."""

    values = np.asarray(signals, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[2] != 6:
        raise ValueError("corruption scale input must be non-empty [window,time,6]")
    if not np.isfinite(values).all():
        raise ValueError("corruption scale input must be finite")
    if not 0.95 <= clipping_quantile < 1.0:
        raise ValueError("clipping_quantile must lie in [0.95, 1.0)")
    channel_scale = np.maximum(values.reshape(-1, 6).std(axis=0), 1e-8)
    clipping_thresholds = np.maximum(
        np.quantile(np.abs(values), clipping_quantile, axis=(0, 1)),
        1e-6,
    )
    return np.asarray(channel_scale), np.asarray(clipping_thresholds)


def _case_id(spec: CorruptionSpec) -> str:
    return f"{spec.name}__{spec.severity:.6f}".replace(".", "p")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_geometric_pyramid_corruptions(
    *,
    nested_result_path: Path,
    source_manifest_path: Path,
    raw_csv_path: Path,
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Evaluate frozen outer models; fail if their clean predictions do not replay."""

    if output_directory.exists():
        raise FileExistsError(f"corruption output already exists: {output_directory}")
    config, specs = _load_config(config_path)
    nested = _mapping(load_json_strict(nested_result_path), name="nested result")
    if nested.get("status") != "complete_target_sealed":
        raise PermissionError("corruption runner requires a complete target-sealed nested result")
    if nested.get("target_subject_or_window_records_loaded") is not False:
        raise PermissionError("corruption runner refuses a target-bearing result")
    if nested.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("corruption runner refuses a target-informed result")
    nested_source = _mapping(nested.get("source_manifest"), name="nested source manifest")
    if sha256_file(source_manifest_path) != str(nested_source.get("sha256")):
        raise ValueError("source manifest does not match the frozen nested run")

    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("corruption runner requires the functional-core ontology")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
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
    window_ids = np.asarray(batch.window_ids, dtype=np.str_)
    sampling_rate = float(config["sampling_rate_hz"])
    clipping_quantile = float(config["clipping_quantile"])
    seed = int(config["seed"])

    outer_by_id = {
        str(item["outer_fold_id"]): item
        for item in cast(list[dict[str, Any]], manifest["source_nested_cv"])
    }
    accumulators: dict[str, dict[str, list[Any]]] = {
        _case_id(spec): {
            "labels": [],
            "probabilities": [],
            "participants": [],
            "windows": [],
            "affected": [],
            "counts": [],
        }
        for spec in specs
    }
    fold_records: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True)
    for fold_index, frozen_fold in enumerate(cast(list[dict[str, Any]], nested["folds"])):
        fold_id = str(frozen_fold["outer_fold_id"])
        outer = _mapping(outer_by_id.get(fold_id), name=f"outer split {fold_id}")
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        train_mask = ~outer_mask
        fold_path = Path(str(frozen_fold["path"]))
        if sha256_file(fold_path) != str(frozen_fold["sha256"]):
            raise ValueError(f"frozen fold record hash mismatch: {fold_id}")
        fold = _mapping(load_json_strict(fold_path), name=f"fold result {fold_id}")
        if fold.get("target_subject_or_window_records_loaded") is not False:
            raise PermissionError(f"fold {fold_id} is target-bearing")
        if fold.get("target_performance_or_prediction_accessed") is not False:
            raise PermissionError(f"fold {fold_id} is target-informed")
        model_record = _mapping(fold.get("model"), name=f"model record {fold_id}")
        model_path = Path(str(model_record["path"]))
        if sha256_file(model_path) != str(model_record["sha256"]):
            raise ValueError(f"frozen model hash mismatch: {fold_id}")
        with model_path.open("rb") as stream:
            estimator = pickle.load(stream)
        saved_record = _mapping(fold.get("predictions"), name=f"prediction record {fold_id}")
        saved_path = Path(str(saved_record["path"]))
        if sha256_file(saved_path) != str(saved_record["sha256"]):
            raise ValueError(f"frozen prediction hash mismatch: {fold_id}")
        with np.load(saved_path, allow_pickle=False) as saved:
            saved_probability = np.asarray(saved["probabilities"], dtype=np.float64)
            np.testing.assert_array_equal(saved["labels"], labels[outer_mask])
            np.testing.assert_array_equal(saved["participant_ids"], participants[outer_mask])
            np.testing.assert_array_equal(saved["window_ids"], window_ids[outer_mask])

        channel_scale, clipping_thresholds = train_only_corruption_scales(
            signals[train_mask], clipping_quantile=clipping_quantile
        )
        fold_cases: list[dict[str, Any]] = []
        for case_index, spec in enumerate(specs):
            corrupted = apply_sensor_corruption(
                torch.from_numpy(np.ascontiguousarray(signals[outer_mask])),
                spec,
                channel_scale=torch.from_numpy(channel_scale.astype(np.float32)),
                clipping_thresholds=torch.from_numpy(clipping_thresholds.astype(np.float32)),
                seed=seed + fold_index * 1_000_003 + case_index * 100_003,
            )
            features = extract_geometric_spectral_pyramid_features(
                corrupted.signals.numpy(), sampling_rate_hz=sampling_rate
            )
            probability = _probabilities(estimator, features)
            if spec.name == "clean":
                np.testing.assert_array_equal(
                    probability.argmax(axis=1), saved_probability.argmax(axis=1)
                )
                np.testing.assert_allclose(probability, saved_probability, rtol=1e-12, atol=1e-15)
            report = classification_report(
                labels[outer_mask],
                probability,
                participants[outer_mask].tolist(),
                class_names=class_names,
            )
            case_id = _case_id(spec)
            accumulator = accumulators[case_id]
            accumulator["labels"].append(labels[outer_mask])
            accumulator["probabilities"].append(probability)
            accumulator["participants"].append(participants[outer_mask])
            accumulator["windows"].append(window_ids[outer_mask])
            accumulator["affected"].append(corrupted.affected_fraction)
            accumulator["counts"].append(int(outer_mask.sum()))
            fold_cases.append(
                {
                    "case_id": case_id,
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "affected_fraction": corrupted.affected_fraction,
                    "report": report,
                }
            )
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "selected_candidate_id": frozen_fold["selected_candidate_id"],
                "frozen_model_sha256": model_record["sha256"],
                "training_scale_participants": sorted(set(participants[train_mask].tolist())),
                "cases": fold_cases,
            }
        )

    aggregate_cases: list[dict[str, Any]] = []
    prediction_payload: dict[str, NDArray[Any]] = {}
    for spec in specs:
        case_id = _case_id(spec)
        accumulator = accumulators[case_id]
        aggregate_labels = np.concatenate(accumulator["labels"])
        aggregate_probability = np.concatenate(accumulator["probabilities"])
        aggregate_participants = np.concatenate(accumulator["participants"])
        aggregate_windows = np.concatenate(accumulator["windows"])
        if set(aggregate_windows.tolist()) != set(window_ids.tolist()):
            raise ValueError(f"corruption case does not cover every source window: {case_id}")
        report = classification_report(
            aggregate_labels,
            aggregate_probability,
            aggregate_participants.tolist(),
            class_names=class_names,
        )
        counts = np.asarray(accumulator["counts"], dtype=np.float64)
        affected = np.asarray(accumulator["affected"], dtype=np.float64)
        aggregate_cases.append(
            {
                "case_id": case_id,
                "corruption": spec.name,
                "severity": spec.severity,
                "affected_fraction": float(np.average(affected, weights=counts)),
                "report": report,
            }
        )
        prediction_payload[f"{case_id}__probabilities"] = aggregate_probability
        prediction_payload[f"{case_id}__labels"] = aggregate_labels
        prediction_payload[f"{case_id}__participant_ids"] = aggregate_participants
        prediction_payload[f"{case_id}__window_ids"] = aggregate_windows

    clean = next(item for item in aggregate_cases if item["corruption"] == "clean")
    clean_score = float(clean["report"]["primary"]["mean_participant_macro_f1"])
    frozen_score = float(nested["aggregate_report"]["primary"]["mean_participant_macro_f1"])
    if clean_score != frozen_score:
        raise ValueError("clean corruption replay does not match frozen nested aggregate")
    corrupted_scores = [
        float(item["report"]["primary"]["mean_participant_macro_f1"])
        for item in aggregate_cases
        if item["corruption"] != "clean"
    ]
    predictions_path = output_directory / "all_corruption_predictions.npz"
    with predictions_path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "geometric_spectral_pyramid_source_corruption_diagnostic",
        "status": "complete_target_sealed",
        "evidence_status": "post_selection_source_only_diagnostic_not_confirmatory",
        "selection_use_prohibited": True,
        "nested_result": {
            "path": nested_result_path.as_posix(),
            "sha256": sha256_file(nested_result_path),
            "canonical_record_sha256": nested["record_sha256"],
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "folds": fold_records,
        "suite": {
            "case_count": len(aggregate_cases),
            "clean_mean_participant_macro_f1": clean_score,
            "mean_corrupted_participant_macro_f1": float(np.mean(corrupted_scores)),
            "worst_corrupted_participant_macro_f1": min(corrupted_scores),
            "cases": aggregate_cases,
        },
        "predictions": {
            "path": predictions_path.as_posix(),
            "sha256": sha256_file(predictions_path),
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_geometric_pyramid_corruptions(
        nested_result_path=args.nested_result,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        config_path=args.config,
        output_directory=args.output_directory,
    )
    suite = cast(dict[str, Any], result["suite"])
    print(
        json.dumps(
            {
                "status": result["status"],
                "case_count": suite["case_count"],
                "clean_mean_participant_macro_f1": suite["clean_mean_participant_macro_f1"],
                "mean_corrupted_participant_macro_f1": suite["mean_corrupted_participant_macro_f1"],
                "worst_corrupted_participant_macro_f1": suite[
                    "worst_corrupted_participant_macro_f1"
                ],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
