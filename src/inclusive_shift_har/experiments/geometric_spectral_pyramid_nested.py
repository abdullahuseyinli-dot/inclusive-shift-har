"""Nested source evaluation of the Geometric Spectral Pyramid invention."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import (  # type: ignore[import-untyped]
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
)

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
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    payload = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    candidates = payload.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("geometric-pyramid config requires candidates")
    ids: list[str] = []
    ranks: list[int] = []
    for index, item in enumerate(candidates):
        candidate = _mapping(item, name=f"candidates[{index}]")
        ids.append(str(candidate.get("id", "")))
        ranks.append(int(candidate.get("complexity_rank", -1)))
    if (
        len(set(ids)) != len(ids)
        or any(not item for item in ids)
        or sorted(ranks) != list(range(1, len(ranks) + 1))
    ):
        raise ValueError("geometric-pyramid candidate ids/ranks are invalid")
    return payload


def _participant_class_weights(labels: IntArray, participants: NDArray[np.str_]) -> FloatArray:
    if labels.shape != participants.shape or labels.size == 0:
        raise ValueError("participant-class weights require aligned non-empty vectors")
    weights = np.empty(labels.size, dtype=np.float64)
    cells = sorted(set(zip(participants.tolist(), labels.tolist(), strict=True)))
    for participant, label in cells:
        mask = (participants == participant) & (labels == label)
        weights[mask] = 1.0 / float(mask.sum())
    weights *= labels.size / weights.sum()
    return weights


class _HierarchicalForest:
    def __init__(self, *, seed: int, n_jobs: int, n_estimators: int = 400) -> None:
        settings = {
            "n_estimators": n_estimators,
            "max_features": "sqrt",
            "min_samples_leaf": 3,
            "class_weight": "balanced",
            "n_jobs": n_jobs,
        }
        self.mobility = ExtraTreesClassifier(random_state=seed, **settings)
        self.posture = ExtraTreesClassifier(random_state=seed + 1, **settings)
        self.classes_ = np.arange(3, dtype=np.int64)

    def fit(self, x: FloatArray, y: IntArray, sample_weight: FloatArray) -> _HierarchicalForest:
        self.mobility.fit(x, (y == 0).astype(np.int64), sample_weight=sample_weight)
        stationary = y != 0
        self.posture.fit(
            x[stationary],
            (y[stationary] == 1).astype(np.int64),
            sample_weight=sample_weight[stationary],
        )
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        mobility = np.asarray(self.mobility.predict_proba(x), dtype=np.float64)[:, 1]
        sitting = np.asarray(self.posture.predict_proba(x), dtype=np.float64)[:, 1]
        return np.stack(
            (mobility, (1.0 - mobility) * sitting, (1.0 - mobility) * (1.0 - sitting)),
            axis=1,
        )


class _ParticipantJackknifeForest:
    def __init__(self, *, seed: int, n_jobs: int) -> None:
        self.seed = seed
        self.n_jobs = n_jobs
        self.models: list[ExtraTreesClassifier] = []
        self.classes_ = np.arange(3, dtype=np.int64)

    def fit(
        self, x: FloatArray, y: IntArray, participants: NDArray[np.str_]
    ) -> _ParticipantJackknifeForest:
        self.models = []
        for index, excluded in enumerate(sorted(set(participants.tolist()), key=int)):
            mask = participants != excluded
            weights = _participant_class_weights(y[mask], participants[mask])
            model = ExtraTreesClassifier(
                n_estimators=96,
                max_features="sqrt",
                min_samples_leaf=3,
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=self.seed + index,
            )
            model.fit(x[mask], y[mask], sample_weight=weights)
            self.models.append(model)
        if len(self.models) < 2:
            raise ValueError("participant jackknife needs at least two training participants")
        return self

    def predict_proba(self, x: FloatArray) -> FloatArray:
        return np.asarray(
            np.mean(
                np.stack([model.predict_proba(x) for model in self.models], axis=0),
                axis=0,
            ),
            dtype=np.float64,
        )


def _build_estimator(candidate_id: str, *, seed: int, n_jobs: int) -> Any:
    if candidate_id in {"extra_trees_leaf3", "extra_trees_leaf1"}:
        return ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            min_samples_leaf=3 if candidate_id.endswith("leaf3") else 1,
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate_id == "hierarchical_forest_leaf3":
        return _HierarchicalForest(seed=seed, n_jobs=n_jobs)
    if candidate_id == "participant_jackknife_leaf3":
        return _ParticipantJackknifeForest(seed=seed, n_jobs=n_jobs)
    if candidate_id == "histogram_gradient":
        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        )
    if candidate_id == "xgboost_depth3":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.65,
            reg_lambda=5.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=n_jobs,
            random_state=seed,
        )
    if candidate_id == "rotation_forest":
        from aeon.classification.sklearn import (  # type: ignore[import-untyped]
            RotationForestClassifier,
        )

        return RotationForestClassifier(n_estimators=200, n_jobs=n_jobs, random_state=seed)
    raise ValueError(f"unknown geometric-pyramid candidate {candidate_id!r}")


def _fit_estimator(
    estimator: Any,
    candidate_id: str,
    features: FloatArray,
    labels: IntArray,
    participants: NDArray[np.str_],
) -> Any:
    if candidate_id == "participant_jackknife_leaf3":
        return estimator.fit(features, labels, participants)
    if candidate_id == "rotation_forest":
        return estimator.fit(features, labels)
    return estimator.fit(
        features,
        labels,
        **{"sample_weight": _participant_class_weights(labels, participants)},
    )


def _probabilities(estimator: Any, features: FloatArray) -> FloatArray:
    values = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if values.shape != (features.shape[0], classes.size) or set(classes.tolist()) != {0, 1, 2}:
        raise ValueError("geometric-pyramid estimator returned misaligned probabilities")
    result = np.zeros((features.shape[0], 3), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        result[:, class_index] = values[:, column]
    result /= result.sum(axis=1, keepdims=True)
    return result


def _candidate_summary(candidate: dict[str, Any], reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for report in reports for row in cast(list[dict[str, Any]], report["participants"])]
    identifiers = [str(row["participant_id"]) for row in rows]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("inner participant was evaluated more than once")
    values = [float(row["macro_f1"]) for row in rows]
    return {
        "candidate_id": str(candidate["id"]),
        "complexity_rank": int(candidate["complexity_rank"]),
        "participant_class_weighted": bool(candidate["participant_class_weighted"]),
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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_geometric_spectral_pyramid_nested_cv(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run nested participant-exclusive selection and outer source evaluation."""

    if output_directory.exists():
        raise FileExistsError(f"geometric-pyramid output already exists: {output_directory}")
    config = _load_config(config_path)
    seed, n_jobs = int(config["seed"]), int(config["n_jobs"])
    candidates = cast(list[dict[str, Any]], config["candidates"])
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("geometric-pyramid requires the functional-core ontology")
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
    sampling_rate = float(config["feature_contract"]["sampling_rate_hz"])
    features = extract_geometric_spectral_pyramid_features(signals, sampling_rate_hz=sampling_rate)
    output_directory.mkdir(parents=True)
    fold_records: list[dict[str, Any]] = []
    all_labels: list[IntArray] = []
    all_probabilities: list[FloatArray] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_subjects = [str(item) for item in outer["outer_test_subjects"]]
        outer_mask = _mask_for_subjects(participants, outer_subjects)
        train_mask = ~outer_mask
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            reports: list[dict[str, Any]] = []
            for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
                inner_train = _mask_for_subjects(
                    participants, [str(item) for item in inner["train_subjects"]]
                )
                inner_validation = _mask_for_subjects(
                    participants, [str(item) for item in inner["validation_subjects"]]
                )
                if np.any(inner_train & inner_validation) or np.any(inner_validation & outer_mask):
                    raise PermissionError("geometric-pyramid nested partition leakage")
                estimator = _build_estimator(candidate_id, seed=seed, n_jobs=n_jobs)
                _fit_estimator(
                    estimator,
                    candidate_id,
                    features[inner_train],
                    labels[inner_train],
                    participants[inner_train],
                )
                reports.append(
                    classification_report(
                        labels[inner_validation],
                        _probabilities(estimator, features[inner_validation]),
                        participants[inner_validation].tolist(),
                        class_names=class_names,
                    )
                )
            summaries.append(_candidate_summary(candidate, reports))
        ranking = sorted(summaries, key=_selection_key)
        selected_id = str(ranking[0]["candidate_id"])
        selected = _build_estimator(selected_id, seed=seed, n_jobs=n_jobs)
        _fit_estimator(
            selected,
            selected_id,
            features[train_mask],
            labels[train_mask],
            participants[train_mask],
        )
        probability = _probabilities(selected, features[outer_mask])
        report = classification_report(
            labels[outer_mask],
            probability,
            participants[outer_mask].tolist(),
            class_names=class_names,
        )
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "selected_model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(selected, stream, protocol=5)
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                probabilities=probability,
                labels=labels[outer_mask],
                participant_ids=participants[outer_mask],
                window_ids=window_ids[outer_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "geometric_spectral_pyramid_nested_outer_result",
            "status": "complete_target_sealed",
            "evidence_status": "post_analysis_nested_outer_source_development",
            "outer_fold_id": fold_id,
            "inner_candidate_ranking": ranking,
            "selection": {
                "candidate_id": selected_id,
                "metric": "inner_mean_participant_macro_f1",
                "outer_labels_used": False,
            },
            "outer_report": report,
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
        _write_json(fold_path, fold_record)
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "selected_candidate_id": selected_id,
                "outer_report": report,
            }
        )
        all_labels.append(labels[outer_mask])
        all_probabilities.append(probability)
        all_participants.append(participants[outer_mask])
        all_windows.append(window_ids[outer_mask])

    aggregate_labels = np.concatenate(all_labels)
    aggregate_probabilities = np.concatenate(all_probabilities)
    aggregate_participants = np.concatenate(all_participants)
    aggregate_windows = np.concatenate(all_windows)
    if (
        set(aggregate_windows.tolist()) != set(window_ids.tolist())
        or len(set(aggregate_windows.tolist())) != aggregate_windows.size
    ):
        raise ValueError("geometric-pyramid outer folds do not cover source exactly once")
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
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=aggregate_probabilities,
            labels=aggregate_labels,
            participant_ids=aggregate_participants,
            window_ids=aggregate_windows,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "geometric_spectral_pyramid_nested_source_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "method_origin": config["method_origin"],
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "code_commit": code_commit,
        "feature_count": features.shape[1],
        "feature_names_sha256": canonical_json_sha256(
            list(geometric_spectral_pyramid_feature_names())
        ),
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("aeon", "numpy", "scikit-learn", "scipy", "xgboost")
        },
        "folds": fold_records,
        "aggregate_report": aggregate_report,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "outer_labels_used_for_candidate_selection": False,
            "method_family_inspired_by_prior_source_results": True,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json(output_directory / "result.json", summary)
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
    result = run_geometric_spectral_pyramid_nested_cv(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_status": result["evidence_status"],
                "feature_count": result["feature_count"],
                "selected_by_fold": {
                    item["outer_fold_id"]: item["selected_candidate_id"] for item in result["folds"]
                },
                "aggregate_primary": result["aggregate_report"]["primary"],
                "class_recall": result["aggregate_report"]["window_level_diagnostics"][
                    "per_class_recall"
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
