"""Evaluate a fixed multi-seed Geometric Spectral Pyramid probability ensemble."""

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
from inclusive_shift_har.evaluation.statistics import (
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _fit_estimator,
    _mapping,
    _probabilities,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def _verify_record(record: dict[str, Any], *, name: str) -> None:
    claimed = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError(f"{name} self-hash changed")


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    base = _mapping(config.get("base_result_contract"), name="base result contract")
    ensemble = _mapping(config.get("ensemble"), name="ensemble")
    selection = _mapping(config.get("selection"), name="selection")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    seeds = ensemble.get("seeds")
    if not isinstance(seeds, list) or [int(item) for item in seeds] != [11, 23, 47, 89, 131]:
        raise ValueError("seed ensemble requires the five predeclared seeds")
    if int(ensemble.get("frozen_base_seed", -1)) != 11:
        raise ValueError("seed 11 must be the frozen base seed")
    if ensemble.get("aggregation") != "equal_arithmetic_probability_mean":
        raise ValueError("seed ensemble must use the fixed arithmetic probability mean")
    if (
        ensemble.get("fitted_weights") is not False
        or ensemble.get("every_seed_reported") is not True
    ):
        raise ValueError("seed ensemble cannot fit weights or suppress seed reports")
    if (
        base.get("reuse_selected_candidate_per_outer_fold") is not True
        or base.get("repeat_inner_selection") is not False
    ):
        raise ValueError("seed ensemble must freeze the prior nested architecture choices")
    if (
        selection.get("outer_labels_used") is not False
        or selection.get("target_data_used") is not False
    ):
        raise ValueError("seed ensemble selection contract permits no outer or target labels")
    if selection.get("winner_selection_from_this_run_allowed") is not False:
        raise ValueError("seed ensemble result cannot be used for another source winner search")
    if policy.get("confirmatory_claim_allowed") is not False:
        raise ValueError("seed ensemble is not confirmatory")
    return config


def equal_seed_probability_mean(probabilities: NDArray[np.floating[Any]]) -> FloatArray:
    """Validate and average fixed seed probabilities with equal weights."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] < 2 or values.shape[2] != 3:
        raise ValueError("seed probabilities must have shape [seed,window,3]")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=2), 1.0, rtol=0.0, atol=1e-8)
    ):
        raise ValueError("seed probabilities are invalid")
    result = values.mean(axis=0)
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def _participant_values(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], report["participants"])
    }


def _load_frozen_predictions(record: dict[str, Any]) -> dict[str, NDArray[Any]]:
    reference = _mapping(record.get("predictions"), name="base predictions")
    path = Path(str(reference["path"]))
    if sha256_file(path) != str(reference["sha256"]):
        raise ValueError("frozen base prediction artifact hash changed")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: np.asarray(payload[name]) for name in payload.files}
    if set(arrays) != {"probabilities", "labels", "participant_ids", "window_ids"}:
        raise ValueError("frozen base prediction artifact fields changed")
    return arrays


def run_geometric_pyramid_seed_ensemble(
    *,
    nested_result_path: Path,
    raw_csv_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Train four added seeds and pool them with the frozen seed-11 OOF predictions."""

    if output_directory.exists():
        raise FileExistsError(f"seed-ensemble output already exists: {output_directory}")
    config = _load_config(config_path)
    nested = _mapping(load_json_strict(nested_result_path), name="nested result")
    _verify_record(nested, name="nested result")
    base_contract = _mapping(config["base_result_contract"], name="base result contract")
    if (
        nested.get("record_kind") != base_contract["record_kind"]
        or nested.get("status") != base_contract["status"]
    ):
        raise ValueError("nested result does not satisfy the configured base contract")
    if (
        nested.get("target_subject_or_window_records_loaded") is not False
        or nested.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("seed ensemble refuses target-bearing results")

    source_reference = _mapping(nested.get("source_manifest"), name="source manifest")
    source_manifest_path = Path(str(source_reference["path"]))
    if sha256_file(source_manifest_path) != str(source_reference["sha256"]):
        raise ValueError("source manifest hash changed")
    dataset_reference = _mapping(nested.get("dataset_manifest"), name="dataset manifest")
    dataset_manifest_path = Path(str(dataset_reference["path"]))
    if sha256_file(dataset_manifest_path) != str(dataset_reference["sha256"]):
        raise ValueError("dataset manifest hash changed")
    source_manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in source_manifest["ontology"]["runnable_track_schemas"]["functional_core"][
            "class_order"
        ]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("seed ensemble requires the functional-core ontology")
    records = tuple(WindowRecord(**record) for record in source_manifest["windows"])
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(source_manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(batch.signals, dtype=np.float32)
    labels = np.asarray(batch.labels, dtype=np.int64)
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    windows = np.asarray(batch.window_ids, dtype=np.str_)
    sampling_rate = float(config["feature_contract"]["sampling_rate_hz"])
    features = extract_geometric_spectral_pyramid_features(signals, sampling_rate_hz=sampling_rate)
    expected_names_sha = canonical_json_sha256(list(geometric_spectral_pyramid_feature_names()))
    if (
        int(nested.get("feature_count", -1)) != features.shape[1]
        or nested.get("feature_names_sha256") != expected_names_sha
    ):
        raise ValueError("frozen base feature contract changed")

    frozen = _load_frozen_predictions(nested)
    frozen_index = {str(window_id): index for index, window_id in enumerate(frozen["window_ids"])}
    if set(frozen_index) != set(windows.tolist()) or len(frozen_index) != windows.size:
        raise ValueError("frozen base windows do not cover the materialized source exactly once")
    base_order = np.asarray([frozen_index[item] for item in windows.tolist()], dtype=np.int64)
    for field, current in (
        ("labels", labels),
        ("participant_ids", participants),
        ("window_ids", windows),
    ):
        if not np.array_equal(frozen[field][base_order], current):
            raise ValueError(f"frozen base {field} does not align with materialized source")
    base_probabilities = np.asarray(frozen["probabilities"], dtype=np.float64)[base_order]

    fold_inputs = {
        str(item["outer_fold_id"]): item for item in cast(list[dict[str, Any]], nested["folds"])
    }
    seeds = [int(item) for item in config["ensemble"]["seeds"]]
    base_seed = int(config["ensemble"]["frozen_base_seed"])
    seed_probabilities = {
        seed: np.full((labels.size, 3), np.nan, dtype=np.float64) for seed in seeds
    }
    seed_probabilities[base_seed][:] = base_probabilities
    fold_records: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True)
    for outer in cast(list[dict[str, Any]], source_manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        if fold_id not in fold_inputs:
            raise ValueError(f"frozen base is missing outer fold {fold_id}")
        base_fold = _mapping(fold_inputs[fold_id], name=f"base fold {fold_id}")
        candidate_id = str(base_fold["selected_candidate_id"])
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        train_mask = ~outer_mask
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_records: list[dict[str, Any]] = []
        for seed in seeds:
            if seed == base_seed:
                model_records.append(
                    {
                        "seed": seed,
                        "role": "frozen_base_model_not_duplicated",
                        "source_nested_result": nested_result_path.as_posix(),
                    }
                )
                continue
            estimator = _build_estimator(candidate_id, seed=seed, n_jobs=4)
            _fit_estimator(
                estimator,
                candidate_id,
                features[train_mask],
                labels[train_mask],
                participants[train_mask],
            )
            seed_probabilities[seed][outer_mask] = _probabilities(estimator, features[outer_mask])
            model_path = fold_directory / f"seed-{seed}.pkl"
            with model_path.open("xb") as stream:
                pickle.dump(estimator, stream, protocol=5)
            model_records.append(
                {
                    "seed": seed,
                    "role": "additional_fixed_seed_model",
                    "path": model_path.as_posix(),
                    "sha256": sha256_file(model_path),
                }
            )
        fold_stack = np.stack([seed_probabilities[seed][outer_mask] for seed in seeds])
        ensemble_probability = equal_seed_probability_mean(fold_stack)
        fold_prediction_path = fold_directory / "predictions.npz"
        with fold_prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                seeds=np.asarray(seeds, dtype=np.int64),
                probabilities=fold_stack,
                ensemble_probabilities=ensemble_probability,
                labels=labels[outer_mask],
                participant_ids=participants[outer_mask],
                window_ids=windows[outer_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "geometric_pyramid_fixed_seed_ensemble_outer_result",
            "status": "complete_target_sealed",
            "evidence_status": "post_analysis_source_development_not_independent",
            "outer_fold_id": fold_id,
            "selected_candidate_id_frozen_from_nested_result": candidate_id,
            "inner_selection_repeated": False,
            "outer_labels_used_for_training_or_selection": False,
            "seeds": seeds,
            "models": model_records,
            "seed_reports": {
                str(seed): classification_report(
                    labels[outer_mask],
                    seed_probabilities[seed][outer_mask],
                    participants[outer_mask].tolist(),
                    class_names=class_names,
                )
                for seed in seeds
            },
            "ensemble_report": classification_report(
                labels[outer_mask],
                ensemble_probability,
                participants[outer_mask].tolist(),
                class_names=class_names,
            ),
            "predictions": {
                "path": fold_prediction_path.as_posix(),
                "sha256": sha256_file(fold_prediction_path),
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
            }
        )

    if any(not np.isfinite(seed_probabilities[seed]).all() for seed in seeds):
        raise ValueError("seed ensemble outer folds did not cover every source window")
    stacked = np.stack([seed_probabilities[seed] for seed in seeds])
    ensemble_probabilities = equal_seed_probability_mean(stacked)
    reports = {
        **{
            f"seed_{seed}": classification_report(
                labels,
                seed_probabilities[seed],
                participants.tolist(),
                class_names=class_names,
            )
            for seed in seeds
        },
        "equal_seed_ensemble": classification_report(
            labels, ensemble_probabilities, participants.tolist(), class_names=class_names
        ),
    }
    participant_values = {name: _participant_values(report) for name, report in reports.items()}
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            seeds=np.asarray(seeds, dtype=np.int64),
            probabilities=stacked,
            ensemble_probabilities=ensemble_probabilities,
            labels=labels,
            participant_ids=participants,
            window_ids=windows,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "geometric_pyramid_fixed_seed_ensemble_source_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "method_origin": config["method_origin"],
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "base_nested_result": {
            "path": nested_result_path.as_posix(),
            "sha256": sha256_file(nested_result_path),
            "record_sha256": nested["record_sha256"],
        },
        "source_manifest": source_reference,
        "dataset_manifest": dataset_reference,
        "raw_source": {"path": raw_csv_path.as_posix(), "sha256": sha256_file(raw_csv_path)},
        "code_commit": code_commit,
        "library_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "feature_count": features.shape[1],
        "feature_names_sha256": expected_names_sha,
        "seeds": seeds,
        "aggregation": "equal_arithmetic_probability_mean",
        "folds": fold_records,
        "reports": reports,
        "participant_bootstrap": {
            name: participant_bootstrap_interval(values)
            for name, values in participant_values.items()
        },
        "paired_comparison_ensemble_vs_frozen_seed_11": paired_participant_comparison(
            participant_values["seed_11"], participant_values["equal_seed_ensemble"]
        ),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "candidate_per_fold_frozen_from_prior_nested_result": True,
            "outer_labels_used_for_training_or_selection": False,
            "target_data_used": False,
            "winner_selection_from_this_run_allowed": False,
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
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_geometric_pyramid_seed_ensemble(
        nested_result_path=args.nested_result,
        raw_csv_path=args.raw_csv,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary": {name: report["primary"] for name, report in result["reports"].items()},
                "class_recall": result["reports"]["equal_seed_ensemble"][
                    "window_level_diagnostics"
                ]["per_class_recall"],
                "paired_comparison": result["paired_comparison_ensemble_vs_frozen_seed_11"],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
