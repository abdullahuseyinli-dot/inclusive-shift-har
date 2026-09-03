"""Nested heterogeneous OOF routing for two target-sealed source models."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _candidate_summary,
    _inner_derived_schedule,
    _load_candidates,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.v2_engine import (
    exact_allowed_classes,
    train_fuse_reframe,
    train_fuse_reframe_fixed_evaluation,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

GATE_FEATURE_NAMES = (
    "base_a_p_mobility",
    "base_a_p_sitting",
    "base_a_p_standing",
    "base_b_p_mobility",
    "base_b_p_sitting",
    "base_b_p_standing",
    "p_difference_mobility",
    "p_difference_sitting",
    "p_difference_standing",
    "absolute_p_difference_mobility",
    "absolute_p_difference_sitting",
    "absolute_p_difference_standing",
    "base_a_confidence",
    "base_b_confidence",
    "base_a_margin",
    "base_b_margin",
    "base_a_entropy",
    "base_b_entropy",
    "prediction_disagreement",
    "jensen_shannon_divergence",
    "near_zero_fraction",
    "source_clipping_fraction",
    "acceleration_norm_mean",
    "acceleration_norm_sd",
    "gyroscope_norm_mean",
    "gyroscope_norm_sd",
    "acceleration_delta_norm_mean",
    "gyroscope_delta_norm_mean",
    "minimum_channel_sd",
    "maximum_channel_sd",
)


def _uncertainty_features(probabilities: FloatArray) -> FloatArray:
    clipped = np.clip(probabilities, 1e-12, 1.0)
    ordered = np.sort(probabilities, axis=1)
    confidence = probabilities.max(axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    entropy = -np.sum(clipped * np.log(clipped), axis=1) / math.log(probabilities.shape[1])
    return np.stack((confidence, margin, entropy), axis=1)


def sensor_health_features(
    signals: NDArray[np.float32],
    clipping_thresholds: NDArray[np.float64],
) -> FloatArray:
    """Compute identity-free native-signal health descriptors per window."""

    values = np.asarray(signals, dtype=np.float64)
    thresholds = np.asarray(clipping_thresholds, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (128, 6):
        raise ValueError("router health features require [window,128,6] signals")
    if thresholds.shape != (6,) or np.any(thresholds <= 0):
        raise ValueError("router clipping thresholds must contain six positive values")
    if not np.isfinite(values).all():
        raise ValueError("router health features require finite input")
    acceleration = values[:, :, :3]
    gyroscope = values[:, :, 3:]
    acceleration_norm = np.linalg.norm(acceleration, axis=2)
    gyroscope_norm = np.linalg.norm(gyroscope, axis=2)
    acceleration_delta = np.diff(acceleration, axis=1)
    gyroscope_delta = np.diff(gyroscope, axis=1)
    channel_sd = values.std(axis=1)
    return np.asarray(
        np.stack(
            (
                np.mean(np.abs(values) <= 1e-8, axis=(1, 2)),
                np.mean(np.abs(values) >= thresholds[None, None, :], axis=(1, 2)),
                acceleration_norm.mean(axis=1),
                acceleration_norm.std(axis=1),
                gyroscope_norm.mean(axis=1),
                gyroscope_norm.std(axis=1),
                np.linalg.norm(acceleration_delta, axis=2).mean(axis=1),
                np.linalg.norm(gyroscope_delta, axis=2).mean(axis=1),
                channel_sd.min(axis=1),
                channel_sd.max(axis=1),
            ),
            axis=1,
        ),
        dtype=np.float64,
    )


def router_features(
    probabilities_a: FloatArray,
    probabilities_b: FloatArray,
    signals: NDArray[np.float32],
    clipping_thresholds: NDArray[np.float64],
) -> FloatArray:
    """Build the predeclared gate matrix without participant or cohort metadata."""

    first = np.asarray(probabilities_a, dtype=np.float64)
    second = np.asarray(probabilities_b, dtype=np.float64)
    if first.shape != second.shape or first.shape != (signals.shape[0], 3):
        raise ValueError("router base probabilities must align as [window,3]")
    if (
        not np.isfinite(first).all()
        or not np.isfinite(second).all()
        or np.any(first < 0)
        or np.any(second < 0)
        or not np.allclose(first.sum(axis=1), 1.0, atol=1e-6)
        or not np.allclose(second.sum(axis=1), 1.0, atol=1e-6)
    ):
        raise ValueError("router received invalid probability rows")
    first_uncertainty = _uncertainty_features(first)
    second_uncertainty = _uncertainty_features(second)
    mixture = np.clip(0.5 * (first + second), 1e-12, 1.0)
    js_divergence = 0.5 * np.sum(
        np.clip(first, 1e-12, 1.0) * np.log(np.clip(first, 1e-12, 1.0) / mixture)
        + np.clip(second, 1e-12, 1.0) * np.log(np.clip(second, 1e-12, 1.0) / mixture),
        axis=1,
    )
    features = np.concatenate(
        (
            first,
            second,
            first - second,
            np.abs(first - second),
            first_uncertainty[:, [0]],
            second_uncertainty[:, [0]],
            first_uncertainty[:, [1]],
            second_uncertainty[:, [1]],
            first_uncertainty[:, [2]],
            second_uncertainty[:, [2]],
            (first.argmax(axis=1) != second.argmax(axis=1)).astype(np.float64)[:, None],
            js_divergence[:, None],
            sensor_health_features(signals, clipping_thresholds),
        ),
        axis=1,
    )
    if features.shape[1] != len(GATE_FEATURE_NAMES) or not np.isfinite(features).all():
        raise AssertionError("router feature contract changed")
    return np.asarray(features, dtype=np.float64)


def _fit_gate(
    features: FloatArray,
    labels: IntArray,
    probabilities_a: FloatArray,
    probabilities_b: FloatArray,
    *,
    seed: int,
) -> tuple[Any | None, int | None, dict[str, Any]]:
    prediction_a = probabilities_a.argmax(axis=1)
    prediction_b = probabilities_b.argmax(axis=1)
    correct_a = prediction_a == labels
    correct_b = prediction_b == labels
    disagreement = prediction_a != prediction_b
    decisive = disagreement & (correct_a ^ correct_b)
    targets = correct_a[decisive].astype(np.int64)
    counts = np.bincount(targets, minlength=2)
    model: Any | None = None
    constant_choice: int | None = None
    if targets.size == 0:
        constant_choice = 0
        status = "constant_base_b_no_decisive_oof_rows"
    elif np.unique(targets).size == 1:
        constant_choice = int(targets[0])
        status = "constant_single_oof_preference"
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
                solver="lbfgs",
            ),
        )
        model.fit(features[decisive], targets)
        status = "fitted_oof_logistic_router"
    record = {
        "status": status,
        "feature_names": list(GATE_FEATURE_NAMES),
        "all_oof_row_count": int(labels.size),
        "base_prediction_disagreement_count": int(disagreement.sum()),
        "decisive_training_row_count": int(decisive.sum()),
        "neither_correct_disagreement_count": int((disagreement & ~correct_a & ~correct_b).sum()),
        "base_a_decisive_win_count": int(counts[1]),
        "base_b_decisive_win_count": int(counts[0]),
        "participant_identity_feature_used": False,
        "cohort_or_disability_feature_used": False,
        "outer_label_used": False,
        "regularization_c": 1.0,
        "class_weight": "balanced",
        "threshold": 0.5,
        "constant_choice": constant_choice,
    }
    return model, constant_choice, record


def _gate_probability(
    model: Any | None,
    constant_choice: int | None,
    features: FloatArray,
) -> FloatArray:
    if model is None:
        if constant_choice not in {0, 1}:
            raise ValueError("constant router choice is missing")
        return np.full(features.shape[0], float(constant_choice), dtype=np.float64)
    raw = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    location = np.flatnonzero(classes == 1)
    if raw.shape[0] != features.shape[0] or location.size != 1:
        raise ValueError("router probability output is invalid")
    return np.asarray(raw[:, int(location[0])], dtype=np.float64)


def _load_probabilities(
    result: dict[str, Any],
    expected_labels: IntArray,
    expected_participants: NDArray[np.str_],
) -> FloatArray:
    path = Path(str(result["predictions"]["path"]))
    if sha256_file(path) != result["predictions"]["sha256"]:
        raise ValueError("base prediction artifact hash changed")
    with np.load(path, allow_pickle=False) as arrays:
        probabilities = np.asarray(arrays["probabilities"], dtype=np.float64)
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = np.asarray(arrays["participant_ids"], dtype=np.str_)
    if not np.array_equal(labels, expected_labels) or not np.array_equal(
        participants, expected_participants
    ):
        raise ValueError("base predictions do not align with the declared evaluation rows")
    return probabilities


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_heterogeneous_router_outer_fold(
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
    """Fit a gate on inner OOF outputs, refit bases, then report one outer pair."""

    if output_directory.exists():
        raise FileExistsError(f"router output directory already exists: {output_directory}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    manifest = _load_source_manifest(source_manifest_path)
    candidates = _load_candidates(candidate_path, seed=seed)
    candidate_ids = [candidate_id for candidate_id, _ in candidates]
    if candidate_ids != ["compact_residual_dann", "inception_erm"]:
        raise ValueError("router candidate order must be compact_residual_dann then inception_erm")
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("heterogeneous router requires the functional three-class ontology")
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
    outer_test_subjects = [str(item) for item in outer["outer_test_subjects"]]
    outer_train_subjects = sorted(set(participants.tolist()) - set(outer_test_subjects), key=int)
    outer_train_mask = _mask_for_subjects(participants, outer_train_subjects)
    outer_test_mask = _mask_for_subjects(participants, outer_test_subjects)
    if np.any(outer_train_mask & outer_test_mask):
        raise ValueError("router outer masks overlap")
    meta_preprocessor = V2PhysicalPreprocessor.fit(
        signals[outer_train_mask],
        participants[outer_train_mask].tolist(),
        declared_training_participants=set(outer_train_subjects),
        split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        channel_names=PRIMARY_CHANNELS,
    )
    output_directory.mkdir(parents=True)
    inner_records: dict[str, list[dict[str, Any]]] = {name: [] for name in candidate_ids}
    oof_probabilities: dict[str, list[FloatArray]] = {name: [] for name in candidate_ids}
    oof_labels: list[IntArray] = []
    oof_signals: list[NDArray[np.float32]] = []
    oof_participants: list[NDArray[np.str_]] = []
    oof_windows: list[NDArray[np.str_]] = []
    for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
        inner_id = str(inner["fold_id"])
        train_subjects = [str(item) for item in inner["train_subjects"]]
        validation_subjects = [str(item) for item in inner["validation_subjects"]]
        train_mask = _mask_for_subjects(participants, train_subjects)
        validation_mask = _mask_for_subjects(participants, validation_subjects)
        if np.any(train_mask & validation_mask) or np.any(validation_mask & outer_test_mask):
            raise PermissionError("outer participant entered router inner development")
        preprocessor = V2PhysicalPreprocessor.fit(
            signals[train_mask],
            participants[train_mask].tolist(),
            declared_training_participants=set(train_subjects),
            split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
            channel_names=PRIMARY_CHANNELS,
        )
        for candidate_id, config in candidates:
            inner_output = output_directory / "inner" / inner_id / candidate_id
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
                    "evidence_status": "router_inner_oof_source_development",
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
            inner_records[candidate_id].append(result)
            oof_probabilities[candidate_id].append(
                _load_probabilities(
                    result,
                    labels[validation_mask],
                    participants[validation_mask],
                )
            )
        oof_labels.append(labels[validation_mask])
        oof_signals.append(signals[validation_mask])
        oof_participants.append(participants[validation_mask])
        oof_windows.append(window_ids[validation_mask])

    all_oof_labels = np.concatenate(oof_labels)
    all_oof_signals = np.concatenate(oof_signals)
    all_oof_participants = np.concatenate(oof_participants)
    all_oof_windows = np.concatenate(oof_windows)
    probabilities_a = np.concatenate(oof_probabilities[candidate_ids[0]])
    probabilities_b = np.concatenate(oof_probabilities[candidate_ids[1]])
    if (
        set(all_oof_participants.tolist()) != set(outer_train_subjects)
        or len(set(all_oof_windows.tolist())) != all_oof_windows.size
        or set(all_oof_windows.tolist()) != set(window_ids[outer_train_mask].tolist())
    ):
        raise ValueError("inner OOF rows do not cover every outer-training window exactly once")
    gate_features = router_features(
        probabilities_a,
        probabilities_b,
        all_oof_signals,
        meta_preprocessor.clipping_thresholds,
    )
    gate_model, constant_choice, gate_training = _fit_gate(
        gate_features,
        all_oof_labels,
        probabilities_a,
        probabilities_b,
        seed=seed,
    )
    gate_path = output_directory / "oof_router.pkl"
    with gate_path.open("xb") as stream:
        pickle.dump(
            {
                "model": gate_model,
                "constant_choice": constant_choice,
                "feature_names": GATE_FEATURE_NAMES,
                "training": gate_training,
            },
            stream,
            protocol=5,
        )
    candidate_summaries = {
        candidate_id: _candidate_summary(candidate_id, inner_records[candidate_id])
        for candidate_id in candidate_ids
    }
    outer_results: dict[str, dict[str, Any]] = {}
    outer_probabilities: dict[str, FloatArray] = {}
    schedules: dict[str, dict[str, Any]] = {}
    for candidate_id, config in candidates:
        fixed_epochs, averaging_interval = _inner_derived_schedule(
            candidate_summaries[candidate_id], config
        )
        schedules[candidate_id] = {
            "fixed_epochs": fixed_epochs,
            "averaging_interval": list(averaging_interval) if averaging_interval else None,
            "derived_from_inner_only": True,
        }
        outer_output = output_directory / "outer" / candidate_id
        result = train_fuse_reframe_fixed_evaluation(
            signals[outer_train_mask],
            exact_allowed_classes(labels[outer_train_mask]),
            participants[outer_train_mask].tolist(),
            signals[outer_test_mask],
            labels[outer_test_mask],
            participants[outer_test_mask].tolist(),
            preprocessor=meta_preprocessor,
            config=config,
            fixed_epochs=fixed_epochs,
            averaging_interval=averaging_interval,
            class_names=class_names,
            lineage={
                "code_commit": code_commit,
                "candidate_id": candidate_id,
                "outer_fold_id": outer_fold_id,
                "source_window_manifest_sha256": manifest["source_window_manifest_sha256"],
                "source_split_manifest_sha256": manifest["source_split_manifest_sha256"],
                "candidate_file_sha256": sha256_file(candidate_path),
                "router_sha256": sha256_file(gate_path),
                "fixed_schedule_derived_without_outer_evaluation": True,
                "outer_test_used_for_selection": False,
                "evidence_status": "router_outer_source_development",
                "target_subject_or_window_records_loaded": False,
                "target_performance_or_prediction_accessed": False,
            },
            output_directory=outer_output,
            device=torch.device(device_name),
        )
        outer_results[candidate_id] = result
        outer_probabilities[candidate_id] = _load_probabilities(
            result,
            labels[outer_test_mask],
            participants[outer_test_mask],
        )

    outer_a = outer_probabilities[candidate_ids[0]]
    outer_b = outer_probabilities[candidate_ids[1]]
    outer_features = router_features(
        outer_a,
        outer_b,
        signals[outer_test_mask],
        meta_preprocessor.clipping_thresholds,
    )
    gate_probability = _gate_probability(gate_model, constant_choice, outer_features)
    choose_a = gate_probability >= 0.5
    max_confidence_a = outer_a.max(axis=1) >= outer_b.max(axis=1)
    method_probabilities = {
        candidate_ids[0]: outer_a,
        candidate_ids[1]: outer_b,
        "fixed_mean": 0.5 * (outer_a + outer_b),
        "max_confidence": np.where(max_confidence_a[:, None], outer_a, outer_b),
        "learned_hard": np.where(choose_a[:, None], outer_a, outer_b),
        "learned_soft": gate_probability[:, None] * outer_a
        + (1.0 - gate_probability[:, None]) * outer_b,
    }
    reports = {
        method: classification_report(
            labels[outer_test_mask],
            probabilities,
            participants[outer_test_mask].tolist(),
            class_names=class_names,
        )
        for method, probabilities in method_probabilities.items()
    }
    prediction_a = outer_a.argmax(axis=1)
    prediction_b = outer_b.argmax(axis=1)
    true_outer = labels[outer_test_mask]
    oracle_correct = (prediction_a == true_outer) | (prediction_b == true_outer)
    oracle_probabilities = np.where(
        (prediction_a == true_outer)[:, None],
        outer_a,
        outer_b,
    )
    prediction_path = output_directory / "outer_router_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            labels=true_outer,
            participant_ids=participants[outer_test_mask],
            window_ids=window_ids[outer_test_mask],
            base_a_probabilities=outer_a,
            base_b_probabilities=outer_b,
            fixed_mean_probabilities=method_probabilities["fixed_mean"],
            max_confidence_probabilities=method_probabilities["max_confidence"],
            learned_hard_probabilities=method_probabilities["learned_hard"],
            learned_soft_probabilities=method_probabilities["learned_soft"],
            gate_probability_choose_base_a=gate_probability,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_nested_heterogeneous_router_outer_result",
        "status": "complete_target_sealed",
        "evidence_status": "nested_outer_source_development_not_confirmatory",
        "outer_fold_id": outer_fold_id,
        "seed": seed,
        "candidate_file": {
            "path": candidate_path.as_posix(),
            "sha256": sha256_file(candidate_path),
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
        "base_ids": candidate_ids,
        "base_configurations": {
            candidate_id: asdict(config) for candidate_id, config in candidates
        },
        "inner_candidate_summaries": candidate_summaries,
        "outer_schedules": schedules,
        "gate_training": gate_training,
        "gate": {"path": gate_path.as_posix(), "sha256": sha256_file(gate_path)},
        "outer_reports": reports,
        "outer_base_results": {
            candidate_id: {
                "path": (output_directory / "outer" / candidate_id / "result.json").as_posix(),
                "sha256": sha256_file(output_directory / "outer" / candidate_id / "result.json"),
                "outer_label_used_for_training_or_selection": False,
            }
            for candidate_id in candidate_ids
        },
        "oracle_headroom_diagnostic": {
            "selection_use_prohibited": True,
            "either_base_correct_fraction": float(oracle_correct.mean()),
            "both_wrong_count": int((~oracle_correct).sum()),
            "report": classification_report(
                true_outer,
                oracle_probabilities,
                participants[outer_test_mask].tolist(),
                class_names=class_names,
            ),
        },
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "selection": {
            "outer_labels_used_for_gate_or_base_selection": False,
            "reported_methods_predeclared": list(method_probabilities),
            "post_outer_method_selection_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json_create_only(output_directory / "result.json", summary)
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
    result = run_heterogeneous_router_outer_fold(
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
    compact = {
        "status": result["status"],
        "outer_fold_id": result["outer_fold_id"],
        "seed": result["seed"],
        "outer_primary": {
            method: report["primary"] for method, report in result["outer_reports"].items()
        },
        "record_sha256": result["record_sha256"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
