"""Frozen explanatory ablations for the completed MPG-RMRP experiment."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _add_bottom_tail,
    _candidate_seed,
    _fit_posture_estimator,
    _load_config,
    _materialize_source,
    _positive_probability,
    _read_hashed_record,
    _selected_candidates,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.microstate_posture_graph import (
    MicrostateCodebook,
    MicrostateFeatureSpec,
    compose_mobility_posture_probabilities,
    extract_microstate_posture_features,
    extract_microstate_state_vectors,
    extract_undetrended_microstate_state_vectors,
    fit_microstate_codebook,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

_CLASS_NAMES = ("mobility", "sitting", "standing")
_METHODS = (
    "flat_rmrp_reference",
    "occupancy_only",
    "occupancy_plus_lag_1_transition",
    "full_multiscale_transition_graph",
    "no_detrending",
    "hard_assignment",
    "remove_first_half_minus_second_half_direction",
    "rist_posture_control",
    "gsp_instead_of_rmrp_mobility",
    "same_rmrp_representation_for_both_heads",
)
_SPECS = {
    "occupancy_only": MicrostateFeatureSpec(
        transition_lags=(1,),
        include_transitions=False,
        include_persistence_dwell=False,
        include_half_direction=False,
    ),
    "occupancy_plus_lag_1_transition": MicrostateFeatureSpec(
        transition_lags=(1,),
        include_transitions=True,
        include_persistence_dwell=False,
        include_half_direction=False,
    ),
    "hard_assignment": MicrostateFeatureSpec(assignment="hard"),
    "remove_first_half_minus_second_half_direction": MicrostateFeatureSpec(
        include_half_direction=False
    ),
}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def _load_fold_bundle(
    fold_reference: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, NDArray[Any]]]:
    fold_path = Path(str(fold_reference["path"]))
    if sha256_file(fold_path) != str(fold_reference["sha256"]):
        raise ValueError(f"frozen fold file hash changed: {fold_path}")
    fold = _read_hashed_record(fold_path)
    model_record = cast(dict[str, Any], fold["models"])
    model_path = Path(str(model_record["path"]))
    if sha256_file(model_path) != str(model_record["sha256"]):
        raise ValueError(f"frozen model hash changed: {model_path}")
    with model_path.open("rb") as stream:
        models = cast(dict[str, Any], pickle.load(stream))
    prediction_record = cast(dict[str, Any], fold["predictions"])
    prediction_path = Path(str(prediction_record["path"]))
    if sha256_file(prediction_path) != str(prediction_record["sha256"]):
        raise ValueError(f"frozen prediction hash changed: {prediction_path}")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        predictions = {key: np.asarray(loaded[key]) for key in loaded.files}
    return fold, models, predictions


def _variant_probability(
    *,
    name: str,
    state_vectors: FloatArray,
    codebook: MicrostateCodebook,
    training_mask: NDArray[np.bool_],
    evaluation_mask: NDArray[np.bool_],
    labels: IntArray,
    participants: StringArray,
    estimator_id: str,
    mobility_probability: FloatArray,
    seed: int,
    n_jobs: int,
) -> tuple[FloatArray, Any]:
    spec = _SPECS[name]
    training_features = extract_microstate_posture_features(
        state_vectors[training_mask], codebook, feature_spec=spec
    )
    evaluation_features = extract_microstate_posture_features(
        state_vectors[evaluation_mask], codebook, feature_spec=spec
    )
    estimator = _fit_posture_estimator(
        estimator_id,
        training_features,
        labels[training_mask],
        participants[training_mask],
        seed=seed,
        n_jobs=n_jobs,
    )
    probability = compose_mobility_posture_probabilities(
        mobility_probability,
        _positive_probability(estimator, evaluation_features),
    )
    return probability, estimator


def run_microstate_posture_graph_ablations(
    *,
    base_result_path: Path,
    source_manifest_path: Path,
    raw_csv_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run all predeclared ablations once at the frozen selection seed."""

    if output_directory.exists():
        raise FileExistsError(f"MPG-RMRP ablation output already exists: {output_directory}")
    config = _load_config(config_path)
    expected_ablations = tuple(
        str(item) for item in config["ablations_after_freeze_without_retuning"]
    )
    if expected_ablations != _METHODS[1:]:
        raise ValueError("configured MPG-RMRP ablations changed or are out of order")
    base = _read_hashed_record(base_result_path)
    seed = int(config["selection"]["seed"])
    if (
        int(base.get("seed", -1)) != seed
        or base.get("status") != "complete_target_sealed"
        or base.get("target_subject_or_window_records_loaded") is not False
        or base.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("ablations require the complete target-sealed seed-11 result")
    if str(base["config"]["sha256"]) != sha256_file(config_path) or str(
        base["source_manifest"]["sha256"]
    ) != sha256_file(source_manifest_path):
        raise ValueError("ablation inputs do not match the frozen primary evaluation")

    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path,
        raw_csv_path=raw_csv_path,
    )
    sampling_rate = float(config["input_contract"]["sampling_rate_hz"])
    n_jobs = int(config["n_jobs"])
    selected_by_fold = _selected_candidates(
        _read_hashed_record(Path(str(base["selection_record"]["path"])))
    )
    candidates = {
        str(item["id"]): item for item in cast(list[dict[str, Any]], config["candidates"])
    }
    selected_spans = sorted(
        {float(candidates[item]["detrend_span_seconds"]) for item in selected_by_fold.values()}
    )
    state_vectors = {
        span: extract_microstate_state_vectors(
            signals,
            sampling_rate_hz=sampling_rate,
            detrend_span_seconds=span,
        )
        for span in selected_spans
    }
    raw_state_vectors = extract_undetrended_microstate_state_vectors(
        signals, sampling_rate_hz=sampling_rate
    )
    fold_by_id = {
        str(item["outer_fold_id"]): item for item in cast(list[dict[str, Any]], base["folds"])
    }
    aggregate: dict[str, dict[str, list[NDArray[Any]]]] = {
        method: {"labels": [], "probabilities": [], "participants": [], "windows": []}
        for method in _METHODS
    }
    fold_records: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True)
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        evaluation_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        training_mask = ~evaluation_mask
        frozen_fold, frozen_models, saved = _load_fold_bundle(fold_by_id[fold_id])
        np.testing.assert_array_equal(saved["labels"], labels[evaluation_mask])
        np.testing.assert_array_equal(saved["participant_ids"], participants[evaluation_mask])
        np.testing.assert_array_equal(saved["window_ids"], windows[evaluation_mask])
        candidate = candidates[selected_by_fold[fold_id]]
        primary_bundle = cast(dict[str, Any], frozen_models["mpg_rmrp"])
        codebook = cast(MicrostateCodebook, primary_bundle["codebook"])
        if codebook.cluster_count != int(candidate["cluster_count"]):
            raise ValueError(f"frozen codebook/candidate mismatch: {fold_id}")
        primary_vectors = state_vectors[float(candidate["detrend_span_seconds"])]
        full_evaluation_features = extract_microstate_posture_features(
            primary_vectors[evaluation_mask], codebook
        )
        full_posture_probability = _positive_probability(
            primary_bundle["posture"], full_evaluation_features
        )
        q_rmrp = np.asarray(saved["mpg_rmrp_probabilities"][:, 0], dtype=np.float64)
        full_probability = compose_mobility_posture_probabilities(q_rmrp, full_posture_probability)
        np.testing.assert_allclose(
            full_probability,
            saved["mpg_rmrp_probabilities"],
            rtol=1e-12,
            atol=1e-15,
        )
        probabilities: dict[str, FloatArray] = {
            "flat_rmrp_reference": np.asarray(saved["flat_rmrp_probabilities"], dtype=np.float64),
            "full_multiscale_transition_graph": full_probability,
            "rist_posture_control": np.asarray(
                saved["rmrp_mobility_rist_posture_probabilities"], dtype=np.float64
            ),
            "gsp_instead_of_rmrp_mobility": compose_mobility_posture_probabilities(
                np.asarray(saved["hierarchical_gsp_probabilities"][:, 0], dtype=np.float64),
                full_posture_probability,
            ),
            "same_rmrp_representation_for_both_heads": np.asarray(
                saved["hierarchical_rmrp_probabilities"], dtype=np.float64
            ),
        }
        fitted: dict[str, Any] = {}
        for name in _SPECS:
            probability, estimator = _variant_probability(
                name=name,
                state_vectors=primary_vectors,
                codebook=codebook,
                training_mask=training_mask,
                evaluation_mask=evaluation_mask,
                labels=labels,
                participants=participants,
                estimator_id=str(candidate["posture_estimator"]),
                mobility_probability=q_rmrp,
                seed=seed,
                n_jobs=n_jobs,
            )
            probabilities[name] = probability
            fitted[name] = estimator
        raw_codebook = fit_microstate_codebook(
            raw_state_vectors[training_mask],
            labels[training_mask],
            participants[training_mask],
            cluster_count=int(candidate["cluster_count"]),
            detrend_span_seconds=float(candidate["detrend_span_seconds"]),
            sampling_rate_hz=sampling_rate,
            seed=_candidate_seed(candidate, seed=seed),
        )
        raw_training_features = extract_microstate_posture_features(
            raw_state_vectors[training_mask], raw_codebook
        )
        raw_evaluation_features = extract_microstate_posture_features(
            raw_state_vectors[evaluation_mask], raw_codebook
        )
        raw_posture = _fit_posture_estimator(
            str(candidate["posture_estimator"]),
            raw_training_features,
            labels[training_mask],
            participants[training_mask],
            seed=seed,
            n_jobs=n_jobs,
        )
        probabilities["no_detrending"] = compose_mobility_posture_probabilities(
            q_rmrp, _positive_probability(raw_posture, raw_evaluation_features)
        )
        fitted["no_detrending"] = {"codebook": raw_codebook, "posture": raw_posture}

        reports: dict[str, dict[str, Any]] = {}
        for method in _METHODS:
            probability = probabilities[method]
            report = classification_report(
                labels[evaluation_mask],
                probability,
                participants[evaluation_mask].tolist(),
                class_names=_CLASS_NAMES,
            )
            _add_bottom_tail(report)
            reports[method] = report
            aggregate[method]["labels"].append(labels[evaluation_mask])
            aggregate[method]["probabilities"].append(probability)
            aggregate[method]["participants"].append(participants[evaluation_mask])
            aggregate[method]["windows"].append(windows[evaluation_mask])
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        models_path = fold_directory / "ablation_models.pkl"
        with models_path.open("xb") as stream:
            pickle.dump(fitted, stream, protocol=5)
        predictions_path = fold_directory / "predictions.npz"
        fold_prediction_payload: dict[str, Any] = {
            "labels": labels[evaluation_mask],
            "participant_ids": participants[evaluation_mask],
            "window_ids": windows[evaluation_mask],
            **{f"{key}_probabilities": value for key, value in probabilities.items()},
        }
        with predictions_path.open("xb") as stream:
            np.savez_compressed(stream, **fold_prediction_payload)
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "microstate_posture_graph_frozen_ablation_fold",
            "status": "complete_source_development_not_confirmatory",
            "seed": seed,
            "outer_fold_id": fold_id,
            "selected_candidate_id": str(candidate["id"]),
            "base_fold_record_sha256": frozen_fold["record_sha256"],
            "reports": reports,
            "models": {"path": models_path.as_posix(), "sha256": sha256_file(models_path)},
            "predictions": {
                "path": predictions_path.as_posix(),
                "sha256": sha256_file(predictions_path),
            },
            "outer_labels_used_for_selection_or_retuning": False,
            "target_subject_or_window_records_loaded": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json(fold_path, fold_record)
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "record_sha256": fold_record["record_sha256"],
            }
        )

    aggregate_reports: dict[str, dict[str, Any]] = {}
    prediction_payload: dict[str, NDArray[Any]] = {}
    for method in _METHODS:
        method_labels = np.concatenate(aggregate[method]["labels"])
        method_probability = np.concatenate(aggregate[method]["probabilities"])
        method_participants = np.concatenate(aggregate[method]["participants"])
        method_windows = np.concatenate(aggregate[method]["windows"])
        if len(set(method_windows.tolist())) != windows.size or set(method_windows.tolist()) != set(
            windows.tolist()
        ):
            raise ValueError(f"ablation does not cover source exactly once: {method}")
        report = classification_report(
            method_labels,
            method_probability,
            method_participants.tolist(),
            class_names=_CLASS_NAMES,
        )
        _add_bottom_tail(report)
        aggregate_reports[method] = report
        prediction_payload[f"{method}_probabilities"] = method_probability
    prediction_payload.update(
        {
            "labels": np.concatenate(aggregate[_METHODS[0]]["labels"]),
            "participant_ids": np.concatenate(aggregate[_METHODS[0]]["participants"]),
            "window_ids": np.concatenate(aggregate[_METHODS[0]]["windows"]),
        }
    )
    predictions_path = output_directory / "all_outer_predictions.npz"
    with predictions_path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "microstate_posture_graph_frozen_ablation_aggregate",
        "status": "complete_source_development_not_confirmatory",
        "evidence_status": "post_primary_explanatory_ablations_without_retuning",
        "seed": seed,
        "code_commit": code_commit,
        "base_result": {
            "path": base_result_path.as_posix(),
            "sha256": sha256_file(base_result_path),
            "record_sha256": base["record_sha256"],
        },
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "methods": list(_METHODS),
        "folds": fold_records,
        "aggregate_reports": aggregate_reports,
        "predictions": {
            "path": predictions_path.as_posix(),
            "sha256": sha256_file(predictions_path),
        },
        "outer_labels_used_for_selection_or_retuning": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "confirmatory_claim_allowed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_microstate_posture_graph_ablations(
        base_result_path=args.base_result,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary": {
                    method: report["primary"]
                    for method, report in result["aggregate_reports"].items()
                },
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
