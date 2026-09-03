"""Nested source evaluation of the Robust Multiscale Residual Pyramid invention."""

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

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _candidate_summary,
    _fit_estimator,
    _mapping,
    _probabilities,
    _selection_key,
    _write_json,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_feature_names,
    robust_multiscale_feature_views,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("robust multiscale config requires candidates")
    ids: list[str] = []
    ranks: list[int] = []
    allowed_views = {"base", "denoised", "residual", "base_denoised", "tri_view"}
    allowed_estimators = {
        "extra_trees_leaf1",
        "extra_trees_leaf3",
        "participant_jackknife_leaf3",
    }
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, name=f"candidates[{index}]")
        ids.append(str(candidate.get("id", "")))
        ranks.append(int(candidate.get("complexity_rank", -1)))
        if candidate.get("view") not in allowed_views:
            raise ValueError("robust multiscale candidate has an invalid feature view")
        if candidate.get("estimator") not in allowed_estimators:
            raise ValueError("robust multiscale candidate has an invalid estimator")
    if len(set(ids)) != len(ids) or any(not item for item in ids):
        raise ValueError("robust multiscale candidate ids must be non-empty and unique")
    if sorted(ranks) != list(range(1, len(ranks) + 1)):
        raise ValueError("robust multiscale complexity ranks must be consecutive")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("post_corruption_method_development") is not True
        or policy.get("corruption_result_reuse_for_selection") is not False
    ):
        raise ValueError("robust multiscale method must disclose post-corruption development")
    if policy.get("confirmatory_claim_allowed") is not False:
        raise ValueError("robust multiscale source development cannot be confirmatory")
    return config


def run_robust_multiscale_residual_nested_cv(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run nested participant CV without using outer labels for selection."""

    if output_directory.exists():
        raise FileExistsError(f"robust multiscale output already exists: {output_directory}")
    config = _load_config(config_path)
    seed, n_jobs = int(config["seed"]), int(config["n_jobs"])
    candidates = cast(list[dict[str, Any]], config["candidates"])
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("robust multiscale experiment requires functional core")
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
    views = robust_multiscale_feature_views(
        signals, sampling_rate_hz=float(config["feature_contract"]["sampling_rate_hz"])
    )
    output_directory.mkdir(parents=True)
    folds: list[dict[str, Any]] = []
    all_labels: list[IntArray] = []
    all_probabilities: list[FloatArray] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        train_mask = ~outer_mask
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            view = str(candidate["view"])
            estimator_id = str(candidate["estimator"])
            reports: list[dict[str, Any]] = []
            for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
                inner_train = _mask_for_subjects(
                    participants, [str(item) for item in inner["train_subjects"]]
                )
                inner_validation = _mask_for_subjects(
                    participants, [str(item) for item in inner["validation_subjects"]]
                )
                if np.any(inner_train & inner_validation) or np.any(inner_validation & outer_mask):
                    raise PermissionError("robust multiscale nested partition leakage")
                estimator = _build_estimator(estimator_id, seed=seed, n_jobs=n_jobs)
                _fit_estimator(
                    estimator,
                    estimator_id,
                    views[view][inner_train],
                    labels[inner_train],
                    participants[inner_train],
                )
                reports.append(
                    classification_report(
                        labels[inner_validation],
                        _probabilities(estimator, views[view][inner_validation]),
                        participants[inner_validation].tolist(),
                        class_names=class_names,
                    )
                )
            summary = _candidate_summary(candidate, reports)
            summary.update({"feature_view": view, "estimator": estimator_id})
            summaries.append(summary)
        ranking = sorted(summaries, key=_selection_key)
        selected_id = str(ranking[0]["candidate_id"])
        selected_config = next(item for item in candidates if item["id"] == selected_id)
        selected_view = str(selected_config["view"])
        selected_estimator_id = str(selected_config["estimator"])
        selected = _build_estimator(selected_estimator_id, seed=seed, n_jobs=n_jobs)
        _fit_estimator(
            selected,
            selected_estimator_id,
            views[selected_view][train_mask],
            labels[train_mask],
            participants[train_mask],
        )
        probability = _probabilities(selected, views[selected_view][outer_mask])
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
            pickle.dump(
                {
                    "candidate_id": selected_id,
                    "feature_view": selected_view,
                    "estimator_id": selected_estimator_id,
                    "estimator": selected,
                },
                stream,
                protocol=5,
            )
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
            "record_kind": "robust_multiscale_residual_nested_outer_result",
            "status": "complete_target_sealed",
            "evidence_status": "post_corruption_analysis_source_development",
            "outer_fold_id": fold_id,
            "inner_candidate_ranking": ranking,
            "selection": {
                "candidate_id": selected_id,
                "feature_view": selected_view,
                "estimator": selected_estimator_id,
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
        folds.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "selected_candidate_id": selected_id,
                "selected_feature_view": selected_view,
                "outer_report": report,
            }
        )
        all_labels.append(labels[outer_mask])
        all_probabilities.append(probability)
        all_participants.append(participants[outer_mask])
        all_windows.append(window_ids[outer_mask])

    aggregate_labels = np.concatenate(all_labels)
    aggregate_probability = np.concatenate(all_probabilities)
    aggregate_participants = np.concatenate(all_participants)
    aggregate_windows = np.concatenate(all_windows)
    if (
        set(aggregate_windows.tolist()) != set(window_ids.tolist())
        or len(set(aggregate_windows.tolist())) != aggregate_windows.size
    ):
        raise ValueError("robust multiscale folds do not cover source exactly once")
    aggregate_report = classification_report(
        aggregate_labels,
        aggregate_probability,
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
            probabilities=aggregate_probability,
            labels=aggregate_labels,
            participant_ids=aggregate_participants,
            window_ids=aggregate_windows,
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "robust_multiscale_residual_nested_source_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "post_corruption_analysis_source_development_not_independent",
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
        "raw_source": {"path": raw_csv_path.as_posix(), "sha256": sha256_file(raw_csv_path)},
        "code_commit": code_commit,
        "feature_views": {
            view: {
                "feature_count": values.shape[1],
                "feature_names_sha256": canonical_json_sha256(
                    list(robust_multiscale_feature_names(view))
                ),
            }
            for view, values in views.items()
        },
        "library_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "folds": folds,
        "aggregate_report": aggregate_report,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "outer_labels_used_for_candidate_selection": False,
            "method_family_inspired_by_prior_source_corruption_results": True,
            "corruption_result_reused_as_training_data": False,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


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
    result = run_robust_multiscale_residual_nested_cv(
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
