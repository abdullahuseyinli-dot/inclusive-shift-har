"""Nested participant-exclusive evaluation of deterministic SpectralShape models."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.discriminant_analysis import (  # type: ignore[import-untyped]
    LinearDiscriminantAnalysis,
)
from sklearn.ensemble import (  # type: ignore[import-untyped]
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import RobustScaler, StandardScaler  # type: ignore[import-untyped]
from sklearn.svm import SVC  # type: ignore[import-untyped]

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _lower_fraction_mean,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.spectral_shape import (
    extract_spectral_shape_features,
    spectral_shape_feature_names,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _ProbabilisticEstimator(Protocol):
    classes_: NDArray[np.integer[Any]]

    def fit(self, x: FloatArray, y: IntArray) -> _ProbabilisticEstimator: ...

    def predict_proba(self, x: FloatArray) -> NDArray[np.floating[Any]]: ...


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    payload = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("SpectralShape config requires candidates")
    identifiers = []
    ranks = []
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, name=f"candidates[{index}]")
        identifiers.append(str(candidate.get("id", "")))
        ranks.append(int(candidate.get("complexity_rank", -1)))
    if (
        len(set(identifiers)) != len(identifiers)
        or any(not identifier for identifier in identifiers)
        or sorted(ranks) != list(range(1, len(ranks) + 1))
    ):
        raise ValueError("SpectralShape candidate ids/ranks are invalid")
    return payload


def _build_estimator(candidate_id: str, *, seed: int, n_jobs: int) -> _ProbabilisticEstimator:
    estimators: dict[str, Any] = {
        "lda_shrinkage": make_pipeline(
            StandardScaler(),
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        ),
        "logistic_c01": make_pipeline(
            RobustScaler(),
            LogisticRegression(
                C=0.1,
                class_weight="balanced",
                max_iter=4_000,
                random_state=seed,
            ),
        ),
        "logistic_c1": make_pipeline(
            RobustScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=4_000,
                random_state=seed,
            ),
        ),
        "svm_rbf_c1": make_pipeline(
            RobustScaler(),
            SVC(
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
        ),
        "svm_rbf_c10": make_pipeline(
            RobustScaler(),
            SVC(
                C=10.0,
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=seed,
            ),
        ),
        "extra_trees_leaf3": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "extra_trees_leaf1": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=1,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "histogram_gradient": make_pipeline(
            RobustScaler(),
            HistGradientBoostingClassifier(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=15,
                l2_regularization=1.0,
                class_weight="balanced",
                random_state=seed,
            ),
        ),
    }
    try:
        return cast(_ProbabilisticEstimator, estimators[candidate_id])
    except KeyError as exc:
        raise ValueError(f"unknown SpectralShape candidate {candidate_id!r}") from exc


def _probabilities(
    estimator: _ProbabilisticEstimator,
    features: FloatArray,
    *,
    num_classes: int,
) -> FloatArray:
    values = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if values.shape != (features.shape[0], classes.size) or set(classes.tolist()) != set(
        range(num_classes)
    ):
        raise ValueError("SpectralShape estimator returned misaligned probabilities")
    result = np.zeros((features.shape[0], num_classes), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        result[:, class_index] = values[:, column]
    result /= result.sum(axis=1, keepdims=True)
    return result


def _candidate_inner_summary(
    candidate_id: str,
    complexity_rank: int,
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    participant_rows = [
        row for report in reports for row in cast(list[dict[str, Any]], report["participants"])
    ]
    identifiers = [str(row["participant_id"]) for row in participant_rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("SpectralShape inner participant was evaluated more than once")
    values = [float(row["macro_f1"]) for row in participant_rows]
    return {
        "candidate_id": candidate_id,
        "complexity_rank": complexity_rank,
        "inner_fold_count": len(reports),
        "participant_count": len(values),
        "mean_participant_macro_f1": float(np.mean(values)),
        "lower_30_percent_participant_macro_f1": _lower_fraction_mean(values),
        "worst_participant_macro_f1": min(values),
    }


def _selection_key(record: dict[str, Any]) -> tuple[float, float, int, str]:
    return (
        -float(record["mean_participant_macro_f1"]),
        -float(record["lower_30_percent_participant_macro_f1"]),
        int(record["complexity_rank"]),
        str(record["candidate_id"]),
    )


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_spectral_shape_nested_cv(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run the complete fixed five-outer-fold nested classical experiment."""

    if output_directory.exists():
        raise FileExistsError(f"SpectralShape output directory already exists: {output_directory}")
    config = _load_config(config_path)
    seed = int(config["seed"])
    n_jobs = int(config["n_jobs"])
    if seed < 0 or n_jobs < 1:
        raise ValueError("SpectralShape seed/job settings are invalid")
    raw_candidates = cast(list[dict[str, Any]], config["candidates"])
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("SpectralShape requires the functional three-class ontology")
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
    features = extract_spectral_shape_features(signals)
    output_directory.mkdir(parents=True)
    all_labels: list[IntArray] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    all_probabilities: list[FloatArray] = []
    fold_records: list[dict[str, Any]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        outer_fold_id = str(outer["outer_fold_id"])
        outer_test_subjects = [str(item) for item in outer["outer_test_subjects"]]
        outer_test_mask = _mask_for_subjects(participants, outer_test_subjects)
        outer_train_subjects = sorted(
            set(participants.tolist()) - set(outer_test_subjects), key=int
        )
        outer_train_mask = _mask_for_subjects(participants, outer_train_subjects)
        inner_summaries: list[dict[str, Any]] = []
        for raw_candidate in raw_candidates:
            candidate_id = str(raw_candidate["id"])
            reports: list[dict[str, Any]] = []
            for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
                inner_train = _mask_for_subjects(
                    participants, [str(item) for item in inner["train_subjects"]]
                )
                inner_validation = _mask_for_subjects(
                    participants, [str(item) for item in inner["validation_subjects"]]
                )
                if np.any(inner_train & inner_validation) or np.any(
                    inner_validation & outer_test_mask
                ):
                    raise PermissionError("SpectralShape nested partition leakage")
                estimator = _build_estimator(candidate_id, seed=seed, n_jobs=n_jobs)
                estimator.fit(features[inner_train], labels[inner_train])
                probability = _probabilities(
                    estimator, features[inner_validation], num_classes=len(class_names)
                )
                reports.append(
                    classification_report(
                        labels[inner_validation],
                        probability,
                        participants[inner_validation].tolist(),
                        class_names=class_names,
                    )
                )
            inner_summaries.append(
                _candidate_inner_summary(
                    candidate_id,
                    int(raw_candidate["complexity_rank"]),
                    reports,
                )
            )
        ranked = sorted(inner_summaries, key=_selection_key)
        selected_id = str(ranked[0]["candidate_id"])
        selected = _build_estimator(selected_id, seed=seed, n_jobs=n_jobs)
        selected.fit(features[outer_train_mask], labels[outer_train_mask])
        outer_probability = _probabilities(
            selected, features[outer_test_mask], num_classes=len(class_names)
        )
        outer_report = classification_report(
            labels[outer_test_mask],
            outer_probability,
            participants[outer_test_mask].tolist(),
            class_names=class_names,
        )
        fold_directory = output_directory / outer_fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "selected_model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(selected, stream, protocol=5)
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                probabilities=outer_probability,
                labels=labels[outer_test_mask],
                participant_ids=participants[outer_test_mask],
                window_ids=window_ids[outer_test_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "spectral_shape_nested_outer_source_result",
            "status": "complete_target_sealed",
            "evidence_status": "post_analysis_nested_outer_source_development",
            "outer_fold_id": outer_fold_id,
            "inner_candidate_ranking": ranked,
            "selection": {
                "candidate_id": selected_id,
                "outer_labels_used": False,
                "metric": "inner_mean_participant_macro_f1",
            },
            "outer_report": outer_report,
            "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json_create_only(fold_path, fold_record)
        fold_records.append(
            {
                "outer_fold_id": outer_fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "selected_candidate_id": selected_id,
                "outer_report": outer_report,
            }
        )
        all_labels.append(labels[outer_test_mask])
        all_participants.append(participants[outer_test_mask])
        all_windows.append(window_ids[outer_test_mask])
        all_probabilities.append(outer_probability)

    aggregate_labels = np.concatenate(all_labels)
    aggregate_participants = np.concatenate(all_participants)
    aggregate_windows = np.concatenate(all_windows)
    aggregate_probabilities = np.concatenate(all_probabilities)
    if len(set(aggregate_windows.tolist())) != aggregate_windows.size or set(
        aggregate_windows.tolist()
    ) != set(window_ids.tolist()):
        raise ValueError("SpectralShape outer folds do not cover each source window once")
    aggregate_report = classification_report(
        aggregate_labels,
        aggregate_probabilities,
        aggregate_participants.tolist(),
        class_names=class_names,
    )
    participant_values = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], aggregate_report["participants"])
    }
    aggregate_prediction_path = output_directory / "all_outer_predictions.npz"
    with aggregate_prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=aggregate_probabilities,
            labels=aggregate_labels,
            participant_ids=aggregate_participants,
            window_ids=aggregate_windows,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "spectral_shape_nested_source_cv_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "method_origin": config["method_origin"],
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
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
        "feature_count": features.shape[1],
        "feature_names_sha256": canonical_json_sha256(list(spectral_shape_feature_names())),
        "library_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "folds": fold_records,
        "aggregate_report": aggregate_report,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {
            "path": aggregate_prediction_path.as_posix(),
            "sha256": sha256_file(aggregate_prediction_path),
        },
        "selection": {
            "outer_labels_used_for_candidate_selection": False,
            "method_family_inspired_by_prior_source_results": True,
            "confirmatory_claim_allowed": False,
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
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_spectral_shape_nested_cv(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    compact = {
        "status": result["status"],
        "evidence_status": result["evidence_status"],
        "feature_count": result["feature_count"],
        "selected_by_fold": {
            item["outer_fold_id"]: item["selected_candidate_id"] for item in result["folds"]
        },
        "aggregate_primary": result["aggregate_report"]["primary"],
        "class_recall": result["aggregate_report"]["window_level_diagnostics"]["per_class_recall"],
        "record_sha256": result["record_sha256"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
