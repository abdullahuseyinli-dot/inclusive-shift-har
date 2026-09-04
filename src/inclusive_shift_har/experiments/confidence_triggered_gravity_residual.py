"""Nested selection and evaluation of confidence-triggered gravity assistance."""

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

from inclusive_shift_har.data.auxiliary import materialize_inclusivehar_source_gravity
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _lower_fraction_mean,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _candidate_summary,
    _fit_estimator,
    _mapping,
    _probabilities,
    _write_json,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _fit_posture_estimator,
    _materialize_source,
    _positive_probability,
    _read_hashed_record,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
)
from inclusive_shift_har.models.gravity_posture_reference import (
    extract_gravity_posture_reference_features,
    gravity_posture_reference_feature_names,
)
from inclusive_shift_har.models.microstate_posture_graph import (
    compose_mobility_posture_probabilities,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_signal_views,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

_CLASS_NAMES = ("mobility", "sitting", "standing")
_VIEW_ORDER = ("physics", "physics_plus_rmrp", "total_gsp", "dual_gsp")
_ESTIMATOR_ORDER = ("extra_trees_leaf3", "extra_trees_leaf1")


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    method = _mapping(config.get("method"), name="method")
    if tuple(method.get("expert_views", [])) != _VIEW_ORDER:
        raise ValueError("CTGR expert view grid changed")
    if tuple(method.get("posture_estimators", [])) != _ESTIMATOR_ORDER:
        raise ValueError("CTGR estimator grid changed")
    thresholds = tuple(float(item) for item in method.get("confidence_thresholds", []))
    weights = tuple(float(item) for item in method.get("blend_weights", []))
    if thresholds != (0.45, 0.50, 0.55, 0.60, 0.65):
        raise ValueError("CTGR confidence threshold grid changed")
    if weights != (0.50, 0.75, 1.00):
        raise ValueError("CTGR blend-weight grid changed")
    if method.get("include_base_no_route") is not True:
        raise ValueError("CTGR must include the unchanged RMRP control as a candidate")
    selection = _mapping(config.get("selection"), name="selection")
    if (
        int(selection.get("seed", -1)) != 11
        or float(selection.get("mean_tie_tolerance", -1.0)) != 0.005
    ):
        raise ValueError("CTGR selection rule changed")
    if tuple(int(item) for item in config.get("fixed_evaluation_seeds", [])) != (
        11,
        23,
        47,
        89,
        131,
    ):
        raise ValueError("CTGR fixed evaluation seeds changed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("primary_six_channel_claim_allowed") is not False
        or policy.get("confirmatory_claim_allowed") is not False
        or policy.get("post_hoc_source_inspiration_disclosed") is not True
    ):
        raise ValueError("CTGR claim boundary changed")
    return config


def _candidates(config: dict[str, Any]) -> list[dict[str, Any]]:
    method = cast(dict[str, Any], config["method"])
    candidates: list[dict[str, Any]] = [
        {
            "id": "base_no_route",
            "expert_view": None,
            "posture_estimator": None,
            "confidence_threshold": 0.0,
            "blend_weight": 0.0,
            "complexity_rank": 0,
            "participant_class_weighted": True,
        }
    ]
    rank = 1
    for view in cast(list[str], method["expert_views"]):
        for estimator in cast(list[str], method["posture_estimators"]):
            for threshold in cast(list[float], method["confidence_thresholds"]):
                for weight in cast(list[float], method["blend_weights"]):
                    candidates.append(
                        {
                            "id": (
                                f"{view}__{estimator}__t{round(100 * float(threshold)):03d}"
                                f"__w{round(100 * float(weight)):03d}"
                            ),
                            "expert_view": view,
                            "posture_estimator": estimator,
                            "confidence_threshold": float(threshold),
                            "blend_weight": float(weight),
                            "complexity_rank": rank,
                            "participant_class_weighted": True,
                        }
                    )
                    rank += 1
    if len(candidates) != 121 or len({str(item["id"]) for item in candidates}) != 121:
        raise AssertionError("CTGR candidate expansion changed")
    return candidates


def _materialize_gravity(
    *,
    manifest: dict[str, Any],
    raw_csv_path: Path,
    expected_participants: StringArray,
    expected_windows: StringArray,
) -> FloatArray:
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    gravity = materialize_inclusivehar_source_gravity(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        allowed_partitions={"source_train", "source_validation"},
    )
    if gravity.participant_ids != tuple(expected_participants.tolist()):
        raise ValueError("gravity participant order does not match the six-channel source")
    if gravity.window_ids != tuple(expected_windows.tolist()):
        raise ValueError("gravity window order does not match the six-channel source")
    return np.asarray(gravity.signals, dtype=np.float64)


def _feature_views(
    signals: FloatArray,
    gravity: FloatArray,
    *,
    sampling_rate_hz: float,
) -> tuple[FloatArray, dict[str, FloatArray]]:
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=sampling_rate_hz)[
        "denoised"
    ]
    rmrp = extract_geometric_spectral_pyramid_features(denoised, sampling_rate_hz=sampling_rate_hz)
    physics = extract_gravity_posture_reference_features(
        denoised, gravity, sampling_rate_hz=sampling_rate_hz
    )
    total = denoised.copy()
    total[:, :, :3] += gravity
    total_gsp = extract_geometric_spectral_pyramid_features(
        total, sampling_rate_hz=sampling_rate_hz
    )
    return rmrp, {
        "physics": physics,
        "physics_plus_rmrp": np.concatenate((physics, rmrp), axis=1),
        "total_gsp": total_gsp,
        "dual_gsp": np.concatenate((rmrp, total_gsp), axis=1),
    }


def _fit_base(
    features: FloatArray,
    labels: IntArray,
    participants: StringArray,
    mask: BoolArray,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    model = _build_estimator("extra_trees_leaf1", seed=seed, n_jobs=n_jobs)
    _fit_estimator(
        model,
        "extra_trees_leaf1",
        features[mask],
        labels[mask],
        participants[mask],
    )
    return model


def _fit_expert(
    estimator_id: str,
    features: FloatArray,
    labels: IntArray,
    participants: StringArray,
    mask: BoolArray,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    return _fit_posture_estimator(
        estimator_id,
        features[mask],
        labels[mask],
        participants[mask],
        seed=seed,
        n_jobs=n_jobs,
    )


def apply_confidence_triggered_gravity_residual(
    base_probability: NDArray[np.floating[Any]],
    gravity_probability: NDArray[np.floating[Any]],
    *,
    confidence_threshold: float,
    blend_weight: float,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Blend a gravity expert only where the unchanged base is insufficiently confident."""

    base = np.asarray(base_probability, dtype=np.float64)
    expert = np.asarray(gravity_probability, dtype=np.float64)
    if (
        base.ndim != 2
        or base.shape != expert.shape
        or base.shape[1] != 3
        or not np.isfinite(base).all()
        or not np.isfinite(expert).all()
    ):
        raise ValueError("CTGR probabilities must be aligned finite three-class matrices")
    if not 0.0 <= confidence_threshold <= 1.0 or not 0.0 <= blend_weight <= 1.0:
        raise ValueError("CTGR threshold and blend weight must lie in [0,1]")
    trigger = np.max(base, axis=1) < confidence_threshold
    result = base.copy()
    result[trigger] = (1.0 - blend_weight) * base[trigger] + blend_weight * expert[trigger]
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64), np.asarray(trigger, dtype=np.bool_)


def _gravity_probability(
    base_probability: FloatArray,
    posture_model: Any,
    expert_features: FloatArray,
) -> FloatArray:
    return compose_mobility_posture_probabilities(
        base_probability[:, 0], _positive_probability(posture_model, expert_features)
    )


def _selection_order(
    summaries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    by_id = {str(item["id"]): item for item in candidates}
    ceiling = max(float(item["mean_participant_macro_f1"]) for item in summaries)
    eligible = {
        str(item["candidate_id"])
        for item in summaries
        if float(item["mean_participant_macro_f1"]) >= ceiling - tolerance
    }

    def key(item: dict[str, Any]) -> tuple[float | int | str, ...]:
        candidate_id = str(item["candidate_id"])
        candidate = by_id[candidate_id]
        if candidate_id in eligible:
            return (
                0,
                -float(item["lower_30_percent_participant_macro_f1"]),
                float(item["mean_trigger_fraction"]),
                int(candidate["complexity_rank"]),
                candidate_id,
            )
        return (
            1,
            -float(item["mean_participant_macro_f1"]),
            -float(item["lower_30_percent_participant_macro_f1"]),
            int(candidate["complexity_rank"]),
            candidate_id,
        )

    return sorted(summaries, key=key), sorted(eligible)


def run_ctgr_selection(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Select a CTGR candidate inside each outer training partition and freeze it."""

    if output_directory.exists():
        raise FileExistsError(f"CTGR selection output already exists: {output_directory}")
    config = _load_config(config_path)
    candidates = _candidates(config)
    seed = int(config["selection"]["seed"])
    n_jobs = int(config["n_jobs"])
    sampling_rate = float(config["input_contract"]["sampling_rate_hz"])
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path, raw_csv_path=raw_csv_path
    )
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_csv_path,
        expected_participants=participants,
        expected_windows=windows,
    )
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=sampling_rate)
    folds: list[dict[str, Any]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        reports_by_candidate: dict[str, list[dict[str, Any]]] = {
            str(item["id"]): [] for item in candidates
        }
        triggers_by_candidate: dict[str, list[float]] = {str(item["id"]): [] for item in candidates}
        for inner_index, inner in enumerate(cast(list[dict[str, Any]], outer["inner_folds"])):
            training_mask = _mask_for_subjects(
                participants, [str(item) for item in inner["train_subjects"]]
            )
            validation_mask = _mask_for_subjects(
                participants, [str(item) for item in inner["validation_subjects"]]
            )
            if np.any(training_mask & validation_mask) or np.any(validation_mask & outer_mask):
                raise PermissionError("CTGR nested partition leakage")
            base_model = _fit_base(
                rmrp,
                labels,
                participants,
                training_mask,
                seed=seed + inner_index,
                n_jobs=n_jobs,
            )
            base_probability = _probabilities(base_model, rmrp[validation_mask])
            reports_by_candidate["base_no_route"].append(
                classification_report(
                    labels[validation_mask],
                    base_probability,
                    participants[validation_mask].tolist(),
                    class_names=_CLASS_NAMES,
                )
            )
            triggers_by_candidate["base_no_route"].append(0.0)
            for view_index, view in enumerate(_VIEW_ORDER):
                for estimator_index, estimator_id in enumerate(_ESTIMATOR_ORDER):
                    expert = _fit_expert(
                        estimator_id,
                        views[view],
                        labels,
                        participants,
                        training_mask,
                        seed=seed + 101 * view_index + 17 * estimator_index + inner_index,
                        n_jobs=n_jobs,
                    )
                    gravity_probability = _gravity_probability(
                        base_probability, expert, views[view][validation_mask]
                    )
                    matching = [
                        item
                        for item in candidates
                        if item["expert_view"] == view and item["posture_estimator"] == estimator_id
                    ]
                    for candidate in matching:
                        probability, trigger = apply_confidence_triggered_gravity_residual(
                            base_probability,
                            gravity_probability,
                            confidence_threshold=float(candidate["confidence_threshold"]),
                            blend_weight=float(candidate["blend_weight"]),
                        )
                        candidate_id = str(candidate["id"])
                        reports_by_candidate[candidate_id].append(
                            classification_report(
                                labels[validation_mask],
                                probability,
                                participants[validation_mask].tolist(),
                                class_names=_CLASS_NAMES,
                            )
                        )
                        triggers_by_candidate[candidate_id].append(float(trigger.mean()))
        summaries: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["id"])
            summary = _candidate_summary(candidate, reports_by_candidate[candidate_id])
            summary["mean_trigger_fraction"] = float(np.mean(triggers_by_candidate[candidate_id]))
            summaries.append(summary)
        ranking, eligible = _selection_order(
            summaries,
            candidates,
            tolerance=float(config["selection"]["mean_tie_tolerance"]),
        )
        folds.append(
            {
                "outer_fold_id": fold_id,
                "outer_test_subjects": list(outer["outer_test_subjects"]),
                "selected_candidate_id": ranking[0]["candidate_id"],
                "eligible_within_mean_tolerance": eligible,
                "candidate_ranking": ranking,
                "outer_labels_used": False,
                "outer_evaluation_performed": False,
            }
        )
    output_directory.mkdir(parents=True)
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confidence_triggered_gravity_residual_inner_selection_freeze",
        "status": "selection_frozen_before_outer_evaluation",
        "evidence_status": "iterative_source_development_after_disclosed_post_hoc_inspiration",
        "code_commit": code_commit,
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
        "candidate_count": len(candidates),
        "folds": folds,
        "outer_labels_used_for_selection": False,
        "outer_evaluation_performed": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "confirmatory_claim_allowed": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(output_directory / "selection.json", record)
    return record


def _selected_by_fold(selection: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["outer_fold_id"]): str(item["selected_candidate_id"])
        for item in cast(list[dict[str, Any]], selection["folds"])
    }


def _add_bottom_tail(report: dict[str, Any]) -> None:
    values = [
        float(item["macro_f1"]) for item in cast(list[dict[str, Any]], report["participants"])
    ]
    report["primary"]["bottom_30_percent_participant_macro_f1"] = _lower_fraction_mean(values)


def run_ctgr_evaluation(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    selection_record_path: Path,
    output_directory: Path,
    code_commit: str,
    seed: int,
) -> dict[str, Any]:
    """Evaluate the frozen CTGR map at one locked seed."""

    if output_directory.exists():
        raise FileExistsError(f"CTGR evaluation output already exists: {output_directory}")
    config = _load_config(config_path)
    if seed not in cast(list[int], config["fixed_evaluation_seeds"]):
        raise ValueError("CTGR evaluation seed is not locked")
    selection = _read_hashed_record(selection_record_path)
    if (
        selection.get("status") != "selection_frozen_before_outer_evaluation"
        or selection.get("outer_labels_used_for_selection") is not False
        or selection.get("outer_evaluation_performed") is not False
    ):
        raise PermissionError("CTGR evaluation requires an unopened selection freeze")
    if (
        selection["config"]["sha256"] != sha256_file(config_path)
        or selection["source_manifest"]["sha256"] != sha256_file(source_manifest_path)
        or selection["dataset_manifest"]["sha256"] != sha256_file(dataset_manifest_path)
    ):
        raise ValueError("CTGR selection lineage does not match evaluation inputs")
    candidates = {str(item["id"]): item for item in _candidates(config)}
    selected = _selected_by_fold(selection)
    n_jobs = int(config["n_jobs"])
    sampling_rate = float(config["input_contract"]["sampling_rate_hz"])
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path, raw_csv_path=raw_csv_path
    )
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_csv_path,
        expected_participants=participants,
        expected_windows=windows,
    )
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=sampling_rate)
    output_directory.mkdir(parents=True)
    methods = ("flat_rmrp", "gravity_posture_expert", "ctgr")
    aggregate: dict[str, dict[str, list[NDArray[Any]]]] = {
        method: {"labels": [], "probabilities": [], "participants": [], "windows": []}
        for method in methods
    }
    folds: list[dict[str, Any]] = []
    positive_folds = 0
    for outer_index, outer in enumerate(cast(list[dict[str, Any]], manifest["source_nested_cv"])):
        fold_id = str(outer["outer_fold_id"])
        evaluation_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        training_mask = ~evaluation_mask
        candidate = candidates[selected[fold_id]]
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training_mask,
            seed=seed,
            n_jobs=n_jobs,
        )
        base_probability = _probabilities(base_model, rmrp[evaluation_mask])
        expert_view = candidate["expert_view"] or "physics"
        estimator_id = candidate["posture_estimator"] or "extra_trees_leaf3"
        expert = _fit_expert(
            str(estimator_id),
            views[str(expert_view)],
            labels,
            participants,
            training_mask,
            seed=seed + 101 * outer_index,
            n_jobs=n_jobs,
        )
        gravity_probability = _gravity_probability(
            base_probability, expert, views[str(expert_view)][evaluation_mask]
        )
        if candidate["id"] == "base_no_route":
            ctgr_probability = base_probability.copy()
            trigger = np.zeros(base_probability.shape[0], dtype=np.bool_)
        else:
            ctgr_probability, trigger = apply_confidence_triggered_gravity_residual(
                base_probability,
                gravity_probability,
                confidence_threshold=float(candidate["confidence_threshold"]),
                blend_weight=float(candidate["blend_weight"]),
            )
        probabilities = {
            "flat_rmrp": base_probability,
            "gravity_posture_expert": gravity_probability,
            "ctgr": ctgr_probability,
        }
        reports: dict[str, dict[str, Any]] = {}
        for method, probability in probabilities.items():
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
        if (
            reports["ctgr"]["primary"]["mean_participant_macro_f1"]
            > reports["flat_rmrp"]["primary"]["mean_participant_macro_f1"]
        ):
            positive_folds += 1
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "models.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(
                {"base": base_model, "gravity_posture_expert": expert, "candidate": candidate},
                stream,
                protocol=5,
            )
        prediction_path = fold_directory / "predictions.npz"
        fold_prediction_payload: dict[str, Any] = {
            "labels": labels[evaluation_mask],
            "participant_ids": participants[evaluation_mask],
            "window_ids": windows[evaluation_mask],
            "trigger": trigger,
            **{f"{key}_probabilities": value for key, value in probabilities.items()},
        }
        with prediction_path.open("xb") as prediction_stream:
            np.savez_compressed(
                prediction_stream,
                **fold_prediction_payload,
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "confidence_triggered_gravity_residual_outer_fold",
            "status": "complete_target_sealed",
            "evidence_status": "iterative_nine_channel_source_development_not_confirmatory",
            "seed": seed,
            "outer_fold_id": fold_id,
            "selected_candidate": candidate,
            "trigger_fraction": float(trigger.mean()),
            "reports": reports,
            "models": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "outer_labels_used_for_selection": False,
            "target_subject_or_window_records_loaded": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json(fold_path, fold_record)
        folds.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "record_sha256": fold_record["record_sha256"],
                "selected_candidate_id": candidate["id"],
                "trigger_fraction": float(trigger.mean()),
                "reports": reports,
            }
        )
    aggregate_reports: dict[str, dict[str, Any]] = {}
    bootstraps: dict[str, dict[str, Any]] = {}
    prediction_payload: dict[str, NDArray[Any]] = {}
    for method in methods:
        method_labels = np.concatenate(aggregate[method]["labels"])
        probability = np.concatenate(aggregate[method]["probabilities"])
        method_participants = np.concatenate(aggregate[method]["participants"])
        method_windows = np.concatenate(aggregate[method]["windows"])
        if len(set(method_windows.tolist())) != windows.size or set(method_windows.tolist()) != set(
            windows.tolist()
        ):
            raise ValueError(f"CTGR outer folds do not cover source exactly once: {method}")
        report = classification_report(
            method_labels,
            probability,
            method_participants.tolist(),
            class_names=_CLASS_NAMES,
        )
        _add_bottom_tail(report)
        aggregate_reports[method] = report
        participant_values = {
            str(item["participant_id"]): float(item["macro_f1"])
            for item in cast(list[dict[str, Any]], report["participants"])
        }
        bootstraps[method] = participant_bootstrap_interval(participant_values)
        prediction_payload[f"{method}_probabilities"] = probability
    prediction_payload.update(
        {
            "labels": np.concatenate(aggregate["ctgr"]["labels"]),
            "participant_ids": np.concatenate(aggregate["ctgr"]["participants"]),
            "window_ids": np.concatenate(aggregate["ctgr"]["windows"]),
        }
    )
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as prediction_stream:
        np.savez_compressed(prediction_stream, **cast(dict[str, Any], prediction_payload))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confidence_triggered_gravity_residual_fixed_seed_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "iterative_nine_channel_source_development_not_confirmatory",
        "seed": seed,
        "code_commit": code_commit,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "selection_record": {
            "path": selection_record_path.as_posix(),
            "sha256": sha256_file(selection_record_path),
            "record_sha256": selection["record_sha256"],
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "feature_counts": {
            "rmrp": int(rmrp.shape[1]),
            **{key: int(value.shape[1]) for key, value in views.items()},
        },
        "physics_feature_names_sha256": canonical_json_sha256(
            list(gravity_posture_reference_feature_names())
        ),
        "library_versions": {
            name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn", "scipy")
        },
        "folds": folds,
        "positive_fold_count": positive_folds,
        "aggregate_reports": aggregate_reports,
        "participant_bootstrap": bootstraps,
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "claim_scope": "nine_channel_gravity_sensor_sufficiency_only",
        "post_hoc_source_inspiration_disclosed": True,
        "outer_labels_used_for_selection": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "confirmatory_claim_allowed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def _summary_metrics(report: dict[str, Any]) -> dict[str, float]:
    primary = cast(dict[str, Any], report["primary"])
    recall = cast(dict[str, Any], report["window_level_diagnostics"])["per_class_recall"]
    calibration = cast(dict[str, Any], report["calibration"])
    return {
        "mean_participant_macro_f1": float(primary["mean_participant_macro_f1"]),
        "bottom_30_percent_participant_macro_f1": float(
            primary["bottom_30_percent_participant_macro_f1"]
        ),
        "worst_participant_macro_f1": float(primary["worst_participant_macro_f1"]),
        "mobility_recall": float(recall["mobility"]),
        "sitting_recall": float(recall["sitting"]),
        "standing_recall": float(recall["standing"]),
        "negative_log_likelihood": float(calibration["negative_log_likelihood"]),
        "multiclass_brier_score": float(calibration["multiclass_brier_score"]),
    }


def run_ctgr_summary(
    *, result_paths: list[Path], config_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Aggregate all five fixed CTGR seeds and apply the secondary-lane gate."""

    if output_directory.exists():
        raise FileExistsError(f"CTGR summary output already exists: {output_directory}")
    config = _load_config(config_path)
    pairs = [(path, _read_hashed_record(path)) for path in result_paths]
    expected = [int(item) for item in config["fixed_evaluation_seeds"]]
    if sorted(int(record["seed"]) for _, record in pairs) != sorted(expected):
        raise ValueError("CTGR summary requires every locked seed exactly once")
    if len({str(record["selection_record"]["record_sha256"]) for _, record in pairs}) != 1:
        raise ValueError("CTGR results do not share one selection freeze")
    pairs.sort(key=lambda item: int(item[1]["seed"]))
    per_seed = {
        str(record["seed"]): {
            method: _summary_metrics(record["aggregate_reports"][method])
            for method in ("flat_rmrp", "gravity_posture_expert", "ctgr")
        }
        for _, record in pairs
    }
    method_summary: dict[str, dict[str, float]] = {}
    for method in ("flat_rmrp", "gravity_posture_expert", "ctgr"):
        keys = tuple(next(iter(per_seed.values()))[method])
        method_summary[method] = {
            key: float(np.mean([per_seed[str(seed)][method][key] for seed in expected]))
            for key in keys
        }
        method_summary[method]["seed_standard_deviation_mean_participant_macro_f1"] = float(
            np.std(
                [per_seed[str(seed)][method]["mean_participant_macro_f1"] for seed in expected],
                ddof=1,
            )
        )
    positive_seeds = sum(
        per_seed[str(seed)]["ctgr"]["mean_participant_macro_f1"]
        > per_seed[str(seed)]["flat_rmrp"]["mean_participant_macro_f1"]
        for seed in expected
    )
    gate = cast(dict[str, Any], config["advancement_gate"])
    values = method_summary["ctgr"]
    checks = {
        "mean_participant_macro_f1": values["mean_participant_macro_f1"]
        >= float(gate["minimum_mean_participant_macro_f1"]),
        "bottom_30_percent_participant_macro_f1": values["bottom_30_percent_participant_macro_f1"]
        >= float(gate["minimum_bottom_30_percent_participant_macro_f1"]),
        "mobility_recall": values["mobility_recall"] >= float(gate["minimum_mobility_recall"]),
        "sitting_recall": values["sitting_recall"] >= float(gate["minimum_sitting_recall"]),
        "standing_recall": values["standing_recall"] >= float(gate["minimum_standing_recall"]),
        "positive_seed_count": positive_seeds >= int(gate["minimum_positive_seed_count"]),
        "average_positive_fold_count": float(
            np.mean([record["positive_fold_count"] for _, record in pairs])
        )
        >= float(gate["minimum_positive_fold_count"]),
        "negative_log_likelihood": values["negative_log_likelihood"]
        <= float(gate["maximum_negative_log_likelihood"]),
        "multiclass_brier_score": values["multiclass_brier_score"]
        <= float(gate["maximum_multiclass_brier_score"]),
    }
    output_directory.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confidence_triggered_gravity_residual_multiseed_summary",
        "status": "complete_iterative_nine_channel_source_development_not_confirmatory",
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "result_files": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "record_sha256": record["record_sha256"],
                "seed": record["seed"],
            }
            for path, record in pairs
        ],
        "per_seed": per_seed,
        "method_summary_across_seeds": method_summary,
        "positive_seed_count": positive_seeds,
        "average_positive_fold_count": float(
            np.mean([record["positive_fold_count"] for _, record in pairs])
        ),
        "advancement_gate": {"checks": checks, "passed": all(checks.values())},
        "claim_scope": "nine_channel_gravity_sensor_sufficiency_only",
        "post_hoc_source_inspiration_disclosed": True,
        "confirmatory_claim_allowed": False,
        "independent_new_ability_relevant_cohort_required": True,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    select = subparsers.add_parser("select")
    for current in (select,):
        current.add_argument("--source-manifest", type=Path, required=True)
        current.add_argument("--raw-csv", type=Path, required=True)
        current.add_argument("--dataset-manifest", type=Path, required=True)
        current.add_argument("--config", type=Path, required=True)
        current.add_argument("--output-directory", type=Path, required=True)
        current.add_argument("--code-commit", required=True)
    evaluate = subparsers.add_parser("evaluate")
    evaluate.add_argument("--source-manifest", type=Path, required=True)
    evaluate.add_argument("--raw-csv", type=Path, required=True)
    evaluate.add_argument("--dataset-manifest", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--selection-record", type=Path, required=True)
    evaluate.add_argument("--output-directory", type=Path, required=True)
    evaluate.add_argument("--code-commit", required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    summarize = subparsers.add_parser("summary")
    summarize.add_argument("--result", type=Path, action="append", required=True)
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "select":
        result = run_ctgr_selection(
            source_manifest_path=args.source_manifest,
            raw_csv_path=args.raw_csv,
            dataset_manifest_path=args.dataset_manifest,
            config_path=args.config,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
        )
        payload = {
            "status": result["status"],
            "selected_by_fold": _selected_by_fold(result),
            "record_sha256": result["record_sha256"],
        }
    elif args.operation == "evaluate":
        result = run_ctgr_evaluation(
            source_manifest_path=args.source_manifest,
            raw_csv_path=args.raw_csv,
            dataset_manifest_path=args.dataset_manifest,
            config_path=args.config,
            selection_record_path=args.selection_record,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
            seed=args.seed,
        )
        payload = {
            "status": result["status"],
            "seed": result["seed"],
            "primary": {
                method: report["primary"] for method, report in result["aggregate_reports"].items()
            },
            "record_sha256": result["record_sha256"],
        }
    else:
        result = run_ctgr_summary(
            result_paths=args.result,
            config_path=args.config,
            output_directory=args.output_directory,
        )
        payload = {
            "status": result["status"],
            "method_summary": result["method_summary_across_seeds"],
            "advancement_gate": result["advancement_gate"],
            "record_sha256": result["record_sha256"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
