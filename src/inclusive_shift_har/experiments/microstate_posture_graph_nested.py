"""Prospectively locked nested selection and fixed-seed evaluation of MPG-RMRP."""

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
from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.aeon_source_controls import _build_classifier
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _lower_fraction_mean,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _candidate_summary,
    _fit_estimator,
    _mapping,
    _participant_class_weights,
    _probabilities,
    _write_json,
)
from inclusive_shift_har.experiments.multirocket_source import _aligned_probabilities
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
)
from inclusive_shift_har.models.microstate_posture_graph import (
    MicrostateCodebook,
    MicrostateFeatureSpec,
    compose_mobility_posture_probabilities,
    extract_microstate_posture_features,
    extract_microstate_state_vectors,
    fit_microstate_codebook,
    microstate_posture_feature_names,
    microstate_state_vector_names,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_signal_views,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

_CLASS_NAMES = ("mobility", "sitting", "standing")
_METHODS = (
    "flat_rmrp",
    "hierarchical_rmrp",
    "hierarchical_gsp",
    "rmrp_mobility_rist_posture",
    "mpg_rmrp",
)
_PRIMARY_FEATURE_SPEC = MicrostateFeatureSpec()


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    candidates_raw = config.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) != 18:
        raise ValueError("MPG-RMRP protocol requires exactly 18 posture candidates")
    candidates = [_mapping(item, name=f"candidates[{index}]") for index, item in enumerate(candidates_raw)]
    identifiers = [str(item.get("id", "")) for item in candidates]
    if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("MPG-RMRP candidate ids must be non-empty and unique")
    expected = {
        (span, clusters, estimator)
        for span in (0.2, 0.5)
        for clusters in (4, 6, 8)
        for estimator in ("extra_trees_leaf1", "extra_trees_leaf3", "logistic_l2_c1")
    }
    observed = {
        (
            float(item.get("detrend_span_seconds", -1.0)),
            int(item.get("cluster_count", -1)),
            str(item.get("posture_estimator", "")),
        )
        for item in candidates
    }
    if observed != expected:
        raise ValueError("MPG-RMRP candidate grid does not equal the locked 2x3x3 design")
    ranks = sorted(int(item.get("complexity_rank", -1)) for item in candidates)
    if ranks != list(range(1, 19)):
        raise ValueError("MPG-RMRP complexity ranks must be consecutive")
    selection = _mapping(config.get("selection"), name="selection")
    if (
        int(selection.get("seed", -1)) != 11
        or float(selection.get("mean_tie_tolerance", -1.0)) != 0.005
        or selection.get("outer_labels_used") is not False
    ):
        raise ValueError("MPG-RMRP selection contract changed")
    if tuple(int(value) for value in config.get("fixed_evaluation_seeds", [])) != (
        11,
        23,
        47,
        89,
        131,
    ):
        raise ValueError("MPG-RMRP fixed evaluation seeds changed")
    controls = _mapping(config.get("controls"), name="controls")
    if tuple(controls) != _METHODS[:-1]:
        raise ValueError("MPG-RMRP pure-control order changed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("confirmatory_claim_allowed") is not False
        or policy.get("consumed_target_reuse_allowed") is not False
        or policy.get("independent_new_ability_cohort_required") is not True
    ):
        raise ValueError("MPG-RMRP claim boundary changed")
    return config


def _read_hashed_record(path: Path) -> dict[str, Any]:
    record = _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    claimed = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError(f"record self-hash changed: {path}")
    return record


def _materialize_source(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
) -> tuple[dict[str, Any], FloatArray, IntArray, StringArray, StringArray]:
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"][
            "class_order"
        ]
    )
    if class_names != _CLASS_NAMES:
        raise ValueError("MPG-RMRP requires the three-class functional-core ontology")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(batch.signals, dtype=np.float64)
    labels = np.asarray(batch.labels, dtype=np.int64)
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    windows = np.asarray(batch.window_ids, dtype=np.str_)
    if set(participants.tolist()) != {str(index) for index in range(1, 11)}:
        raise PermissionError("MPG-RMRP source materialization crossed the participant 1-10 boundary")
    return manifest, signals, labels, participants, windows


def _binary_forest(*, seed: int, n_jobs: int, minimum_leaf: int = 1) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=minimum_leaf,
        class_weight="balanced",
        n_jobs=n_jobs,
        random_state=seed,
    )


def _fit_binary_forest(
    features: FloatArray,
    targets: IntArray,
    participants: StringArray,
    *,
    seed: int,
    n_jobs: int,
    minimum_leaf: int = 1,
) -> ExtraTreesClassifier:
    if set(targets.tolist()) != {0, 1}:
        raise ValueError("binary forest training requires both classes")
    estimator = _binary_forest(seed=seed, n_jobs=n_jobs, minimum_leaf=minimum_leaf)
    estimator.fit(
        features,
        targets,
        sample_weight=_participant_class_weights(targets, participants),
    )
    return estimator


def _positive_probability(estimator: Any, features: FloatArray) -> FloatArray:
    probability = np.asarray(estimator.predict_proba(features), dtype=np.float64)
    classes = np.asarray(estimator.classes_, dtype=np.int64)
    if probability.shape != (features.shape[0], 2) or set(classes.tolist()) != {0, 1}:
        raise ValueError("binary estimator returned misaligned probabilities")
    column = int(np.flatnonzero(classes == 1)[0])
    return np.asarray(probability[:, column], dtype=np.float64)


def _posture_estimator(
    estimator_id: str,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    if estimator_id in {"extra_trees_leaf1", "extra_trees_leaf3"}:
        return _binary_forest(
            seed=seed,
            n_jobs=n_jobs,
            minimum_leaf=1 if estimator_id.endswith("leaf1") else 3,
        )
    if estimator_id == "logistic_l2_c1":
        return LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2_000,
            class_weight="balanced",
            random_state=seed,
        )
    raise ValueError(f"unknown MPG posture estimator {estimator_id!r}")


def _fit_posture_estimator(
    estimator_id: str,
    features: FloatArray,
    labels: IntArray,
    participants: StringArray,
    *,
    seed: int,
    n_jobs: int,
) -> Any:
    stationary = labels != 0
    targets = (labels[stationary] == 1).astype(np.int64)
    estimator = _posture_estimator(estimator_id, seed=seed, n_jobs=n_jobs)
    estimator.fit(
        features[stationary],
        targets,
        sample_weight=_participant_class_weights(targets, participants[stationary]),
    )
    return estimator


def _candidate_seed(candidate: dict[str, Any], *, seed: int) -> int:
    span_offset = round(float(candidate["detrend_span_seconds"]) * 1_000)
    return seed + span_offset + 100 * int(candidate["cluster_count"])


def _fit_codebook_and_features(
    state_vectors: FloatArray,
    labels: IntArray,
    participants: StringArray,
    training_mask: BoolArray,
    evaluation_mask: BoolArray,
    candidate: dict[str, Any],
    *,
    sampling_rate_hz: float,
    seed: int,
) -> tuple[MicrostateCodebook, FloatArray, FloatArray]:
    codebook = fit_microstate_codebook(
        state_vectors[training_mask],
        labels[training_mask],
        participants[training_mask],
        cluster_count=int(candidate["cluster_count"]),
        detrend_span_seconds=float(candidate["detrend_span_seconds"]),
        sampling_rate_hz=sampling_rate_hz,
        seed=_candidate_seed(candidate, seed=seed),
    )
    training_features = extract_microstate_posture_features(
        state_vectors[training_mask], codebook, feature_spec=_PRIMARY_FEATURE_SPEC
    )
    evaluation_features = extract_microstate_posture_features(
        state_vectors[evaluation_mask], codebook, feature_spec=_PRIMARY_FEATURE_SPEC
    )
    return codebook, training_features, evaluation_features


def _selection_order(
    summaries: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    *,
    tolerance: float,
) -> tuple[list[dict[str, Any]], list[str]]:
    if not summaries or tolerance < 0:
        raise ValueError("candidate selection requires summaries and a non-negative tolerance")
    by_id = {str(item["id"]): item for item in candidates}
    ceiling = max(float(item["mean_participant_macro_f1"]) for item in summaries)
    eligible = {
        str(item["candidate_id"])
        for item in summaries
        if float(item["mean_participant_macro_f1"]) >= ceiling - tolerance
    }

    def key(item: dict[str, Any]) -> tuple[float | int | str, ...]:
        candidate = by_id[str(item["candidate_id"])]
        candidate_id = str(item["candidate_id"])
        if candidate_id in eligible:
            return (
                0,
                -float(item["lower_30_percent_participant_macro_f1"]),
                int(candidate["cluster_count"]),
                int(candidate["estimator_complexity_rank"]),
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


def run_microstate_posture_graph_selection(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Select one MPG posture candidate per outer fold using inner participants only."""

    if output_directory.exists():
        raise FileExistsError(f"MPG-RMRP selection output already exists: {output_directory}")
    config = _load_config(config_path)
    seed = int(config["selection"]["seed"])
    n_jobs = int(config["n_jobs"])
    sampling_rate = float(config["input_contract"]["sampling_rate_hz"])
    candidates = cast(list[dict[str, Any]], config["candidates"])
    manifest, signals, labels, participants, _ = _materialize_source(
        source_manifest_path=source_manifest_path,
        raw_csv_path=raw_csv_path,
    )
    denoised = robust_multiscale_signal_views(
        signals, sampling_rate_hz=sampling_rate
    )["denoised"]
    mobility_features = extract_geometric_spectral_pyramid_features(
        denoised, sampling_rate_hz=sampling_rate
    )
    spans = sorted({float(candidate["detrend_span_seconds"]) for candidate in candidates})
    state_vectors = {
        span: extract_microstate_state_vectors(
            signals,
            sampling_rate_hz=sampling_rate,
            detrend_span_seconds=span,
        )
        for span in spans
    }
    output_directory.mkdir(parents=True)
    fold_references: list[dict[str, Any]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        reports_by_candidate: dict[str, list[dict[str, Any]]] = {
            str(candidate["id"]): [] for candidate in candidates
        }
        for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
            training_mask = _mask_for_subjects(
                participants, [str(item) for item in inner["train_subjects"]]
            )
            validation_mask = _mask_for_subjects(
                participants, [str(item) for item in inner["validation_subjects"]]
            )
            if np.any(training_mask & validation_mask) or np.any(validation_mask & outer_mask):
                raise PermissionError("MPG-RMRP nested selection partition leakage")
            mobility_target = (labels[training_mask] == 0).astype(np.int64)
            mobility_model = _fit_binary_forest(
                mobility_features[training_mask],
                mobility_target,
                participants[training_mask],
                seed=seed,
                n_jobs=n_jobs,
            )
            mobility_probability = _positive_probability(
                mobility_model, mobility_features[validation_mask]
            )
            feature_cache: dict[tuple[float, int], tuple[FloatArray, FloatArray]] = {}
            for candidate in candidates:
                span = float(candidate["detrend_span_seconds"])
                clusters = int(candidate["cluster_count"])
                cache_key = (span, clusters)
                if cache_key not in feature_cache:
                    _, training_features, validation_features = _fit_codebook_and_features(
                        state_vectors[span],
                        labels,
                        participants,
                        training_mask,
                        validation_mask,
                        candidate,
                        sampling_rate_hz=sampling_rate,
                        seed=seed,
                    )
                    feature_cache[cache_key] = (training_features, validation_features)
                training_features, validation_features = feature_cache[cache_key]
                estimator_id = str(candidate["posture_estimator"])
                posture_model = _fit_posture_estimator(
                    estimator_id,
                    training_features,
                    labels[training_mask],
                    participants[training_mask],
                    seed=seed,
                    n_jobs=n_jobs,
                )
                sitting_probability = _positive_probability(posture_model, validation_features)
                probability = compose_mobility_posture_probabilities(
                    mobility_probability, sitting_probability
                )
                reports_by_candidate[str(candidate["id"])].append(
                    classification_report(
                        labels[validation_mask],
                        probability,
                        participants[validation_mask].tolist(),
                        class_names=_CLASS_NAMES,
                    )
                )
        summaries = [
            _candidate_summary(candidate, reports_by_candidate[str(candidate["id"])])
            for candidate in candidates
        ]
        ranking, eligible = _selection_order(
            summaries,
            candidates,
            tolerance=float(config["selection"]["mean_tie_tolerance"]),
        )
        selected_id = str(ranking[0]["candidate_id"])
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "microstate_posture_graph_inner_selection",
            "status": "selection_frozen_before_outer_evaluation",
            "evidence_status": "prospective_nested_inner_source_development",
            "outer_fold_id": fold_id,
            "outer_test_subjects": [str(item) for item in outer["outer_test_subjects"]],
            "inner_candidate_ranking": ranking,
            "within_mean_tolerance_candidate_ids": eligible,
            "selection": {
                "candidate_id": selected_id,
                "mean_tie_tolerance": float(config["selection"]["mean_tie_tolerance"]),
                "tie_breakers": config["selection"]["tie_breakers"],
                "outer_labels_used": False,
            },
            "outer_evaluation_performed": False,
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        fold_path = fold_directory / "selection.json"
        _write_json(fold_path, fold_record)
        fold_references.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "record_sha256": fold_record["record_sha256"],
                "selected_candidate_id": selected_id,
            }
        )
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "microstate_posture_graph_nested_selection_freeze",
        "status": "selection_frozen_before_outer_evaluation",
        "evidence_status": "prospective_source_development_not_confirmatory",
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
        "selection_seed": seed,
        "candidate_count": len(candidates),
        "folds": fold_references,
        "outer_labels_used_for_selection": False,
        "outer_evaluation_performed": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    _write_json(output_directory / "selection.json", record)
    return record


def _selected_candidates(selection: dict[str, Any]) -> dict[str, str]:
    return {
        str(item["outer_fold_id"]): str(item["selected_candidate_id"])
        for item in cast(list[dict[str, Any]], selection["folds"])
    }


def _add_bottom_tail(report: dict[str, Any]) -> None:
    participants = cast(list[dict[str, Any]], report["participants"])
    values = [float(item["macro_f1"]) for item in participants]
    report["primary"]["bottom_30_percent_participant_macro_f1"] = _lower_fraction_mean(values)


def _fit_predict_hierarchy(
    training_features: FloatArray,
    evaluation_features: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    *,
    seed: int,
    n_jobs: int,
) -> tuple[FloatArray, dict[str, Any]]:
    mobility_model = _fit_binary_forest(
        training_features,
        (training_labels == 0).astype(np.int64),
        training_participants,
        seed=seed,
        n_jobs=n_jobs,
    )
    posture_model = _fit_posture_estimator(
        "extra_trees_leaf1",
        training_features,
        training_labels,
        training_participants,
        seed=seed + 1,
        n_jobs=n_jobs,
    )
    probability = compose_mobility_posture_probabilities(
        _positive_probability(mobility_model, evaluation_features),
        _positive_probability(posture_model, evaluation_features),
    )
    return probability, {"mobility": mobility_model, "posture": posture_model}


def _fit_predict_rist_posture(
    training_signals: FloatArray,
    evaluation_signals: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    *,
    parameters: dict[str, Any],
    seed: int,
    n_jobs: int,
    split_manifest_sha256: str,
) -> tuple[FloatArray, dict[str, Any]]:
    standardizer = ChannelStandardizer.fit(
        training_signals,
        training_participants,
        declared_training_participants=set(training_participants.tolist()),
        split_manifest_sha256=split_manifest_sha256,
        channel_names=PRIMARY_CHANNELS,
    )
    training_x = np.asarray(
        np.transpose(standardizer.transform(training_signals), (0, 2, 1)), dtype=np.float64
    )
    evaluation_x = np.asarray(
        np.transpose(standardizer.transform(evaluation_signals), (0, 2, 1)), dtype=np.float64
    )
    stationary = training_labels != 0
    targets = (training_labels[stationary] == 1).astype(np.int64)
    classifier = _build_classifier(
        "rist_budgeted",
        parameters,
        seed=seed,
        n_jobs=n_jobs,
    )
    classifier.fit(training_x[stationary], targets)
    probability, status = _aligned_probabilities(
        classifier,
        evaluation_x,  # type: ignore[arg-type]  # RIST's Numba path requires float64.
        num_classes=2,
    )
    return np.asarray(probability[:, 1], dtype=np.float64), {
        "classifier": classifier,
        "standardizer": standardizer,
        "probability_status": status,
    }


def run_microstate_posture_graph_evaluation(
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
    """Evaluate frozen fold-specific MPG candidates and pure controls at one fixed seed."""

    if output_directory.exists():
        raise FileExistsError(f"MPG-RMRP evaluation output already exists: {output_directory}")
    config = _load_config(config_path)
    if seed not in [int(value) for value in config["fixed_evaluation_seeds"]]:
        raise ValueError("evaluation seed is not in the prospectively locked list")
    selection = _read_hashed_record(selection_record_path)
    if (
        selection.get("status") != "selection_frozen_before_outer_evaluation"
        or selection.get("outer_evaluation_performed") is not False
        or selection.get("outer_labels_used_for_selection") is not False
    ):
        raise PermissionError("MPG-RMRP evaluation requires an unopened selection freeze")
    if (
        selection["config"]["sha256"] != sha256_file(config_path)
        or selection["source_manifest"]["sha256"] != sha256_file(source_manifest_path)
        or selection["dataset_manifest"]["sha256"] != sha256_file(dataset_manifest_path)
    ):
        raise ValueError("MPG-RMRP selection freeze lineage does not match evaluation inputs")
    selected_by_fold = _selected_candidates(selection)
    candidates = {
        str(item["id"]): item for item in cast(list[dict[str, Any]], config["candidates"])
    }
    n_jobs = int(config["n_jobs"])
    sampling_rate = float(config["input_contract"]["sampling_rate_hz"])
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path,
        raw_csv_path=raw_csv_path,
    )
    signal_views = robust_multiscale_signal_views(signals, sampling_rate_hz=sampling_rate)
    base_features = extract_geometric_spectral_pyramid_features(
        signal_views["base"], sampling_rate_hz=sampling_rate
    )
    denoised_features = extract_geometric_spectral_pyramid_features(
        signal_views["denoised"], sampling_rate_hz=sampling_rate
    )
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
    rist_parameters = _mapping(
        config["controls"]["rmrp_mobility_rist_posture"], name="RIST control"
    )
    output_directory.mkdir(parents=True)
    fold_records: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[NDArray[Any]]]] = {
        method: {"labels": [], "probabilities": [], "participants": [], "windows": []}
        for method in _METHODS
    }
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        training_mask = ~outer_mask
        candidate = candidates[selected_by_fold[fold_id]]
        flat_model = _build_estimator("extra_trees_leaf1", seed=seed, n_jobs=n_jobs)
        _fit_estimator(
            flat_model,
            "extra_trees_leaf1",
            denoised_features[training_mask],
            labels[training_mask],
            participants[training_mask],
        )
        probabilities: dict[str, FloatArray] = {
            "flat_rmrp": _probabilities(flat_model, denoised_features[outer_mask])
        }
        hierarchical_rmrp, hierarchical_rmrp_models = _fit_predict_hierarchy(
            denoised_features[training_mask],
            denoised_features[outer_mask],
            labels[training_mask],
            participants[training_mask],
            seed=seed,
            n_jobs=n_jobs,
        )
        probabilities["hierarchical_rmrp"] = hierarchical_rmrp
        hierarchical_gsp, hierarchical_gsp_models = _fit_predict_hierarchy(
            base_features[training_mask],
            base_features[outer_mask],
            labels[training_mask],
            participants[training_mask],
            seed=seed,
            n_jobs=n_jobs,
        )
        probabilities["hierarchical_gsp"] = hierarchical_gsp
        shared_mobility = hierarchical_rmrp_models["mobility"]
        shared_mobility_probability = _positive_probability(
            shared_mobility, denoised_features[outer_mask]
        )
        sitting_rist, rist_artifact = _fit_predict_rist_posture(
            signal_views["denoised"][training_mask],
            signal_views["denoised"][outer_mask],
            labels[training_mask],
            participants[training_mask],
            parameters=rist_parameters,
            seed=seed,
            n_jobs=n_jobs,
            split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        )
        probabilities["rmrp_mobility_rist_posture"] = (
            compose_mobility_posture_probabilities(shared_mobility_probability, sitting_rist)
        )
        span = float(candidate["detrend_span_seconds"])
        codebook, training_graph, evaluation_graph = _fit_codebook_and_features(
            state_vectors[span],
            labels,
            participants,
            training_mask,
            outer_mask,
            candidate,
            sampling_rate_hz=sampling_rate,
            seed=seed,
        )
        posture_model = _fit_posture_estimator(
            str(candidate["posture_estimator"]),
            training_graph,
            labels[training_mask],
            participants[training_mask],
            seed=seed,
            n_jobs=n_jobs,
        )
        probabilities["mpg_rmrp"] = compose_mobility_posture_probabilities(
            shared_mobility_probability,
            _positive_probability(posture_model, evaluation_graph),
        )
        reports: dict[str, dict[str, Any]] = {}
        for method, probability in probabilities.items():
            report = classification_report(
                labels[outer_mask],
                probability,
                participants[outer_mask].tolist(),
                class_names=_CLASS_NAMES,
            )
            _add_bottom_tail(report)
            reports[method] = report
            aggregate[method]["labels"].append(labels[outer_mask])
            aggregate[method]["probabilities"].append(probability)
            aggregate[method]["participants"].append(participants[outer_mask])
            aggregate[method]["windows"].append(windows[outer_mask])
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "models.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(
                {
                    "flat_rmrp": flat_model,
                    "hierarchical_rmrp": hierarchical_rmrp_models,
                    "hierarchical_gsp": hierarchical_gsp_models,
                    "rmrp_mobility_rist_posture": rist_artifact,
                    "mpg_rmrp": {"codebook": codebook, "posture": posture_model},
                    "selected_candidate": candidate,
                },
                stream,
                protocol=5,
            )
        prediction_path = fold_directory / "predictions.npz"
        fold_prediction_payload: dict[str, Any] = {
            "labels": labels[outer_mask],
            "participant_ids": participants[outer_mask],
            "window_ids": windows[outer_mask],
            **{f"{name}_probabilities": value for name, value in probabilities.items()},
        }
        with prediction_path.open("xb") as stream:
            np.savez_compressed(stream, **fold_prediction_payload)
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "microstate_posture_graph_fixed_outer_evaluation",
            "status": "complete_target_sealed",
            "evidence_status": "prospective_nested_source_development_not_confirmatory",
            "seed": seed,
            "outer_fold_id": fold_id,
            "selection": {
                "candidate_id": str(candidate["id"]),
                "selection_record_sha256": selection["record_sha256"],
                "outer_labels_used": False,
            },
            "microstate_codebook": codebook.audit_record(),
            "reports": reports,
            "rist_probability_status": rist_artifact["probability_status"],
            "models": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
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
        _write_json(fold_path, fold_record)
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "record_sha256": fold_record["record_sha256"],
                "selected_candidate_id": str(candidate["id"]),
                "reports": reports,
            }
        )
    aggregate_reports: dict[str, dict[str, Any]] = {}
    bootstraps: dict[str, dict[str, Any]] = {}
    prediction_payload: dict[str, NDArray[Any]] = {}
    reference_windows: set[str] | None = None
    for method in _METHODS:
        method_labels = np.concatenate(aggregate[method]["labels"])
        method_probability = np.concatenate(aggregate[method]["probabilities"])
        method_participants = np.concatenate(aggregate[method]["participants"])
        method_windows = np.concatenate(aggregate[method]["windows"])
        observed_windows = set(method_windows.tolist())
        if (
            observed_windows != set(windows.tolist())
            or len(observed_windows) != method_windows.size
            or (reference_windows is not None and observed_windows != reference_windows)
        ):
            raise ValueError("MPG-RMRP outer folds do not cover source windows exactly once")
        reference_windows = observed_windows
        report = classification_report(
            method_labels,
            method_probability,
            method_participants.tolist(),
            class_names=_CLASS_NAMES,
        )
        _add_bottom_tail(report)
        aggregate_reports[method] = report
        participant_values = {
            str(row["participant_id"]): float(row["macro_f1"])
            for row in cast(list[dict[str, Any]], report["participants"])
        }
        bootstraps[method] = participant_bootstrap_interval(participant_values)
        prediction_payload[f"{method}_probabilities"] = method_probability
    prediction_payload.update(
        {
            "labels": np.concatenate(aggregate["mpg_rmrp"]["labels"]),
            "participant_ids": np.concatenate(aggregate["mpg_rmrp"]["participants"]),
            "window_ids": np.concatenate(aggregate["mpg_rmrp"]["windows"]),
        }
    )
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "microstate_posture_graph_fixed_seed_source_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "prospective_source_development_not_confirmatory",
        "seed": seed,
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
        "raw_source": {"path": raw_csv_path.as_posix(), "sha256": sha256_file(raw_csv_path)},
        "code_commit": code_commit,
        "state_vector_names": list(microstate_state_vector_names()),
        "graph_feature_count_by_k": {
            str(k): len(microstate_posture_feature_names(k, _PRIMARY_FEATURE_SPEC))
            for k in (4, 6, 8)
        },
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("aeon", "numpy", "scikit-learn", "scipy")
        },
        "folds": fold_records,
        "aggregate_reports": aggregate_reports,
        "participant_bootstrap": bootstraps,
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "frozen_before_outer_evaluation": True,
            "outer_labels_used": False,
            "selected_by_fold": selected_by_fold,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "confirmatory_claim_allowed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def _report_metrics(report: dict[str, Any]) -> dict[str, float]:
    primary = _mapping(report["primary"], name="primary")
    diagnostics = _mapping(report["window_level_diagnostics"], name="diagnostics")
    recall = _mapping(diagnostics["per_class_recall"], name="per-class recall")
    calibration = _mapping(report["calibration"], name="calibration")
    return {
        "mean_participant_macro_f1": float(primary["mean_participant_macro_f1"]),
        "bottom_30_percent_participant_macro_f1": float(
            primary["bottom_30_percent_participant_macro_f1"]
        ),
        "lower_decile_participant_macro_f1": float(
            primary["lower_decile_participant_macro_f1"]
        ),
        "worst_participant_macro_f1": float(primary["worst_participant_macro_f1"]),
        "mobility_recall": float(recall["mobility"]),
        "sitting_recall": float(recall["sitting"]),
        "standing_recall": float(recall["standing"]),
        "negative_log_likelihood": float(calibration["negative_log_likelihood"]),
        "multiclass_brier_score": float(calibration["multiclass_brier_score"]),
    }


def run_microstate_posture_graph_multiseed_summary(
    *,
    result_paths: list[Path],
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Validate and aggregate the complete frozen five-seed MPG-RMRP matrix."""

    if output_directory.exists():
        raise FileExistsError(f"MPG-RMRP summary output already exists: {output_directory}")
    config = _load_config(config_path)
    path_record_pairs = [(path, _read_hashed_record(path)) for path in result_paths]
    records = [record for _, record in path_record_pairs]
    expected_seeds = [int(value) for value in config["fixed_evaluation_seeds"]]
    if sorted(int(item["seed"]) for item in records) != sorted(expected_seeds):
        raise ValueError("MPG-RMRP summary requires every locked seed exactly once")
    config_hash = sha256_file(config_path)
    selection_hashes = {str(item["selection_record"]["record_sha256"]) for item in records}
    if len(selection_hashes) != 1 or any(item["config"]["sha256"] != config_hash for item in records):
        raise ValueError("MPG-RMRP seed results do not share one config and selection freeze")
    path_record_pairs.sort(key=lambda item: int(item[1]["seed"]))
    records = [record for _, record in path_record_pairs]
    by_seed: dict[str, dict[str, dict[str, float]]] = {}
    for record in records:
        by_seed[str(record["seed"])] = {
            method: _report_metrics(record["aggregate_reports"][method]) for method in _METHODS
        }
    method_summary: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        keys = tuple(next(iter(by_seed.values()))[method])
        method_summary[method] = {
            key: float(np.mean([by_seed[str(seed)][method][key] for seed in sorted(expected_seeds)]))
            for key in keys
        }
        method_summary[method]["seed_standard_deviation_mean_participant_macro_f1"] = float(
            np.std(
                [
                    by_seed[str(seed)][method]["mean_participant_macro_f1"]
                    for seed in sorted(expected_seeds)
                ],
                ddof=1,
            )
        )
    positive_seeds = sum(
        by_seed[str(seed)]["mpg_rmrp"]["mean_participant_macro_f1"]
        > by_seed[str(seed)]["flat_rmrp"]["mean_participant_macro_f1"]
        for seed in expected_seeds
    )
    fold_differences: dict[str, float] = {}
    for fold_id in [f"source_cv_{index:02d}" for index in range(1, 6)]:
        values: list[float] = []
        for record in records:
            fold = next(item for item in record["folds"] if item["outer_fold_id"] == fold_id)
            values.append(
                float(fold["reports"]["mpg_rmrp"]["primary"]["mean_participant_macro_f1"])
                - float(fold["reports"]["flat_rmrp"]["primary"]["mean_participant_macro_f1"])
            )
        fold_differences[fold_id] = float(np.mean(values))
    positive_folds = sum(value > 0 for value in fold_differences.values())
    gate = _mapping(config["advancement_gate"], name="advancement gate")
    mpg = method_summary["mpg_rmrp"]
    checks = {
        "mean_participant_macro_f1": mpg["mean_participant_macro_f1"]
        >= float(gate["minimum_mean_participant_macro_f1"]),
        "bottom_30_percent_participant_macro_f1": mpg[
            "bottom_30_percent_participant_macro_f1"
        ]
        >= float(gate["minimum_bottom_30_percent_participant_macro_f1"]),
        "mobility_recall": mpg["mobility_recall"] >= float(gate["minimum_mobility_recall"]),
        "sitting_recall": mpg["sitting_recall"] >= float(gate["minimum_sitting_recall"]),
        "standing_recall": mpg["standing_recall"] >= float(gate["minimum_standing_recall"]),
        "positive_seed_count": positive_seeds >= int(gate["minimum_positive_seed_count"]),
        "positive_fold_count": positive_folds >= int(gate["minimum_positive_fold_count"]),
        "negative_log_likelihood": mpg["negative_log_likelihood"]
        <= float(gate["maximum_negative_log_likelihood"]),
        "multiclass_brier_score": mpg["multiclass_brier_score"]
        <= float(gate["maximum_multiclass_brier_score"]),
    }
    stretch = mpg["worst_participant_macro_f1"] >= float(
        gate["stretch_minimum_worst_participant_macro_f1"]
    )
    output_directory.mkdir(parents=True)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "microstate_posture_graph_multiseed_summary",
        "status": "complete_source_development_not_confirmatory",
        "evidence_status": "prospective_source_development_not_independent_confirmation",
        "config": {"path": config_path.as_posix(), "sha256": config_hash},
        "selection_record_sha256": next(iter(selection_hashes)),
        "result_files": [
            {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "record_sha256": record["record_sha256"],
                "seed": record["seed"],
            }
            for path, record in path_record_pairs
        ],
        "per_seed": by_seed,
        "method_summary_across_seeds": method_summary,
        "paired_mpg_minus_flat": {
            "positive_seed_count": positive_seeds,
            "positive_fold_count": positive_folds,
            "seed_count": len(expected_seeds),
            "fold_count": len(fold_differences),
            "mean_fold_differences": fold_differences,
        },
        "advancement_gate": {
            "checks": checks,
            "passed": all(checks.values()),
            "stretch_worst_participant_passed": stretch,
        },
        "confirmatory_claim_allowed": False,
        "independent_new_ability_cohort_required": True,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--source-manifest", type=Path, required=True)
    shared.add_argument("--raw-csv", type=Path, required=True)
    shared.add_argument("--dataset-manifest", type=Path, required=True)
    shared.add_argument("--config", type=Path, required=True)
    shared.add_argument("--output-directory", type=Path, required=True)
    shared.add_argument("--code-commit", required=True)
    subparsers.add_parser("select", parents=[shared])
    evaluate = subparsers.add_parser("evaluate", parents=[shared])
    evaluate.add_argument("--selection-record", type=Path, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--result", type=Path, action="append", required=True)
    summarize.add_argument("--config", type=Path, required=True)
    summarize.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "select":
        result = run_microstate_posture_graph_selection(
            source_manifest_path=args.source_manifest,
            raw_csv_path=args.raw_csv,
            dataset_manifest_path=args.dataset_manifest,
            config_path=args.config,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
        )
        payload = {
            "status": result["status"],
            "selected_by_fold": _selected_candidates(result),
            "record_sha256": result["record_sha256"],
        }
    elif args.command == "evaluate":
        result = run_microstate_posture_graph_evaluation(
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
                method: report["primary"]
                for method, report in result["aggregate_reports"].items()
            },
            "record_sha256": result["record_sha256"],
        }
    else:
        result = run_microstate_posture_graph_multiseed_summary(
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
