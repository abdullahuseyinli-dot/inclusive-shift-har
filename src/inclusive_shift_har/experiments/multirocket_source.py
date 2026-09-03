"""Run a fixed MultiRocket control over all target-sealed source folds.

This is deliberately a control, not a tuning surface. The official aeon
MultiRocket defaults are used except for the explicitly recorded kernel count,
job count, and random seed. Each outer source pair is evaluated exactly once
after fitting preprocessing and the classifier on the other eight people.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer


class _Classifier(Protocol):
    classes_: NDArray[np.integer[Any]]

    def fit(self, x: NDArray[np.float32], y: NDArray[np.int64]) -> _Classifier: ...

    def predict(self, x: NDArray[np.float32]) -> NDArray[np.integer[Any]]: ...

    def predict_proba(self, x: NDArray[np.float32]) -> NDArray[np.floating[Any]]: ...


@dataclass(frozen=True)
class MultiRocketControlConfig:
    """The intentionally small, prospectively fixed control surface."""

    seed: int = 11
    n_kernels: int = 10_000
    max_dilations_per_kernel: int = 32
    n_features_per_kernel: int = 4
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if (
            self.seed < 0
            or min(
                self.n_kernels,
                self.max_dilations_per_kernel,
                self.n_features_per_kernel,
                self.n_jobs,
            )
            < 1
        ):
            raise ValueError("invalid MultiRocket control configuration")


def _aligned_probabilities(
    classifier: _Classifier,
    signals: NDArray[np.float32],
    *,
    num_classes: int,
) -> tuple[NDArray[np.float64], str]:
    """Return class-aligned scores while exposing non-probabilistic fallback use."""

    predicted = np.asarray(classifier.predict(signals), dtype=np.int64)
    probabilities = np.asarray(classifier.predict_proba(signals), dtype=np.float64)
    classes = np.asarray(classifier.classes_, dtype=np.int64)
    if probabilities.shape != (signals.shape[0], classes.size):
        raise ValueError("MultiRocket probability output has an unexpected shape")
    if set(classes.tolist()) != set(range(num_classes)):
        raise ValueError("MultiRocket did not retain every declared source class")
    aligned = np.zeros((signals.shape[0], num_classes), dtype=np.float64)
    for source_column, class_index in enumerate(classes.tolist()):
        aligned[:, class_index] = probabilities[:, source_column]
    if not np.isfinite(aligned).all() or np.any(aligned < 0):
        raise ValueError("MultiRocket returned invalid class scores")
    sums = aligned.sum(axis=1, keepdims=True)
    if np.any(sums <= 0):
        raise ValueError("MultiRocket returned an empty class-score row")
    aligned /= sums
    if not np.array_equal(aligned.argmax(axis=1), predicted):
        raise ValueError("MultiRocket predictions disagree with its class-score output")
    one_hot_rows = np.all(np.isclose(aligned, 0.0) | np.isclose(aligned, 1.0))
    calibration_status = (
        "unsupported_one_hot_from_nonprobabilistic_ridge_control"
        if one_hot_rows
        else "estimator_probability_output"
    )
    return aligned, calibration_status


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    _write_bytes_create_only(
        path,
        json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n",
    )


def _build_classifier(config: MultiRocketControlConfig) -> _Classifier:
    try:
        from aeon.classification.convolution_based import (  # type: ignore[import-untyped]
            MultiRocketClassifier,
        )
    except ImportError as exc:  # pragma: no cover - exercised by installation gate
        raise RuntimeError(
            "MultiRocket control requires the locked research dependency group"
        ) from exc
    return cast(
        _Classifier,
        MultiRocketClassifier(
            n_kernels=config.n_kernels,
            max_dilations_per_kernel=config.max_dilations_per_kernel,
            n_features_per_kernel=config.n_features_per_kernel,
            n_jobs=config.n_jobs,
            random_state=config.seed,
        ),
    )


def run_multirocket_source_cv(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    output_directory: Path,
    code_commit: str,
    config: MultiRocketControlConfig,
) -> dict[str, Any]:
    """Fit the fixed control independently in all five grouped source folds."""

    if output_directory.exists():
        raise FileExistsError(f"MultiRocket output directory already exists: {output_directory}")
    manifest = _load_source_manifest(source_manifest_path)
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("MultiRocket control requires the functional three-class ontology")
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
    output_directory.mkdir(parents=True)

    config_record = asdict(config)
    fold_results: list[dict[str, Any]] = []
    aggregate_labels: list[NDArray[np.int64]] = []
    aggregate_probabilities: list[NDArray[np.float64]] = []
    aggregate_participants: list[NDArray[np.str_]] = []
    aggregate_windows: list[NDArray[np.str_]] = []
    for fold in cast(list[dict[str, Any]], manifest["source_cv_folds"]):
        fold_id = str(fold["fold_id"])
        train_subjects = [str(item) for item in fold["train_subjects"]]
        validation_subjects = [str(item) for item in fold["validation_subjects"]]
        train_mask = np.asarray(np.isin(participants, train_subjects), dtype=np.bool_)
        validation_mask = np.asarray(np.isin(participants, validation_subjects), dtype=np.bool_)
        if (
            not train_mask.any()
            or not validation_mask.any()
            or np.any(train_mask & validation_mask)
            or set(participants[train_mask].tolist()) != set(train_subjects)
            or set(participants[validation_mask].tolist()) != set(validation_subjects)
        ):
            raise ValueError(f"invalid participant-exclusive masks for {fold_id}")
        standardizer = ChannelStandardizer.fit(
            signals[train_mask],
            participants[train_mask],
            declared_training_participants=set(train_subjects),
            split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
            channel_names=PRIMARY_CHANNELS,
        )
        train_x = np.transpose(standardizer.transform(signals[train_mask]), (0, 2, 1))
        validation_x = np.transpose(standardizer.transform(signals[validation_mask]), (0, 2, 1))
        classifier = _build_classifier(config)
        classifier.fit(train_x, labels[train_mask])
        probabilities, calibration_status = _aligned_probabilities(
            classifier,
            validation_x,
            num_classes=len(class_names),
        )
        report = classification_report(
            labels[validation_mask],
            probabilities,
            participants[validation_mask].tolist(),
            class_names=class_names,
        )
        report["calibration"]["status"] = calibration_status
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "model.pkl"
        _write_bytes_create_only(model_path, pickle.dumps(classifier, protocol=5))
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                probabilities=probabilities,
                labels=labels[validation_mask],
                participant_ids=participants[validation_mask],
                window_ids=window_ids[validation_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "multirocket_source_fold_control",
            "fold_id": fold_id,
            "train_participants": train_subjects,
            "evaluation_participants": validation_subjects,
            "train_window_count": int(train_mask.sum()),
            "evaluation_window_count": int(validation_mask.sum()),
            "configuration": config_record,
            "configuration_sha256": canonical_json_sha256(config_record),
            "preprocessor": standardizer.to_dict(),
            "evaluation_report": report,
            "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "outer_evaluation_access_count_during_training": 0,
            "outer_evaluation_access_count_after_training": 1,
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json_create_only(fold_path, fold_record)
        fold_results.append(
            {
                "fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "evaluation_report": report,
            }
        )
        aggregate_labels.append(labels[validation_mask])
        aggregate_probabilities.append(probabilities)
        aggregate_participants.append(participants[validation_mask])
        aggregate_windows.append(window_ids[validation_mask])

    all_labels = np.concatenate(aggregate_labels)
    all_probabilities = np.concatenate(aggregate_probabilities)
    all_participants = np.concatenate(aggregate_participants)
    all_windows = np.concatenate(aggregate_windows)
    if len(set(all_windows.tolist())) != all_windows.size:
        raise ValueError("a source window was evaluated in more than one MultiRocket fold")
    if set(all_windows.tolist()) != set(window_ids.tolist()):
        raise ValueError("MultiRocket outer folds do not cover the source windows exactly once")
    aggregate_report = classification_report(
        all_labels,
        all_probabilities,
        all_participants.tolist(),
        class_names=class_names,
    )
    aggregate_report["calibration"]["status"] = (
        "unsupported_one_hot_from_nonprobabilistic_ridge_control"
    )
    aggregate_prediction_path = output_directory / "all_outer_predictions.npz"
    with aggregate_prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=all_probabilities,
            labels=all_labels,
            participant_ids=all_participants,
            window_ids=all_windows,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "multirocket_grouped_source_cv_control",
        "status": "complete_target_sealed",
        "evidence_status": "outer_source_development_control_not_confirmatory",
        "configuration": config_record,
        "configuration_sha256": canonical_json_sha256(config_record),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("aeon", "numba", "numpy", "scikit-learn")
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
        "fold_count": len(fold_results),
        "folds": fold_results,
        "aggregate_report": aggregate_report,
        "predictions": {
            "path": aggregate_prediction_path.as_posix(),
            "sha256": sha256_file(aggregate_prediction_path),
        },
        "selection": {
            "method": "none_fixed_external_control",
            "outer_results_used_for_hyperparameter_selection": False,
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
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--n-kernels", type=int, default=10_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_multirocket_source_cv(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
        config=MultiRocketControlConfig(
            seed=args.seed,
            n_kernels=args.n_kernels,
            n_jobs=args.n_jobs,
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
