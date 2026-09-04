"""Fully nested retrospective evaluation of frozen CAGE-HAR on source participants.

This lane intentionally reuses InclusiveHAR source participants 1--10 and is
therefore development evidence, not independent validation.  Participants 11--20
and all DAGHAR evidence remain inaccessible to this runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.experiments.cage_har import (
    _METHOD_NAMES,
    _apply_selected_repair,
    _fixed_gate,
    _report,
    _routing_audit,
    _select_probability_repair,
    _tail_and_harm_weights,
    _write_result,
)
from inclusive_shift_har.experiments.cage_har import (
    _load_config as _load_cage_config,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _ESTIMATOR_ORDER,
    _VIEW_ORDER,
    _feature_views,
    _fit_base,
    _fit_expert,
    _gravity_probability,
    _materialize_gravity,
    _selected_by_fold,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates as _ctgr_candidates,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _load_config as _load_ctgr_config,
)
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _materialize_source,
    _read_hashed_record,
)
from inclusive_shift_har.manifests.canonical import sha256_file
from inclusive_shift_har.models.cage_har import (
    apply_cage_trust_region,
    build_advantage_router_features,
    cage_expert_reliability,
    counterfactual_log_loss_advantage,
    extract_cage_gauge_context,
    extract_cage_sensor_health,
    participant_jackknife_advantage_prediction,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

_EXPECTED_SEEDS = (11, 23, 47, 89, 131)
_EXPECTED_PARTICIPANTS = tuple(str(index) for index in range(1, 11))
_RETROSPECTIVE_METHODS = (*_METHOD_NAMES, "frozen_ctgr")


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def load_cage_retrospective_config(path: Path) -> dict[str, Any]:
    """Load the immutable retrospective execution contract."""

    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("status") != "locked_before_fully_nested_retrospective_source_run":
        raise ValueError("retrospective CAGE-HAR protocol is not locked")
    if tuple(int(item) for item in config.get("fixed_seeds", [])) != _EXPECTED_SEEDS:
        raise ValueError("retrospective CAGE-HAR seed set changed")
    if tuple(str(item) for item in config.get("participant_ids", [])) != _EXPECTED_PARTICIPANTS:
        raise ValueError("retrospective CAGE-HAR source participants changed")
    if int(config.get("outer_fold_count", -1)) != 5:
        raise ValueError("retrospective CAGE-HAR outer fold count changed")
    if int(config.get("inner_fold_count_per_outer", -1)) != 4:
        raise ValueError("retrospective CAGE-HAR inner fold count changed")
    if tuple(str(item) for item in config.get("methods", [])) != _RETROSPECTIVE_METHODS:
        raise ValueError("retrospective CAGE-HAR method set changed")

    freeze = _mapping(config.get("method_freeze"), name="method freeze")
    if (
        freeze.get("tag") != "cage-har-implementation-v1"
        or freeze.get("commit") != "3d1e8ae444b9976c7d532b39142ae58d4fe41aad"
        or freeze.get("parameters_may_change_after_preliminary_probe") is not False
    ):
        raise ValueError("CAGE-HAR method freeze changed")
    nesting = _mapping(config.get("nesting_contract"), name="nesting contract")
    required_true = {
        "frozen_ctgr_candidate_map_reused",
        "base_and_expert_inner_oof_regenerated_inside_each_outer_training_partition",
        "decision_repair_fit_on_inner_oof_only",
        "advantage_router_fit_on_inner_oof_only",
        "participant_jackknife_inside_outer_training_partition",
        "preserved_outer_ctgr_probabilities_used_for_evaluation",
        "participant_is_inference_unit",
    }
    if any(nesting.get(key) is not True for key in required_true):
        raise ValueError("retrospective CAGE-HAR nesting guarantee changed")
    if nesting.get("outer_labels_used_for_training_selection_or_repair") is not False:
        raise ValueError("outer labels must remain unavailable to CAGE fitting")
    if nesting.get("context_channel_scale") != "outer_training_median_absolute_clipped_at_1e-6":
        raise ValueError("retrospective CAGE-HAR context scaling changed")
    if (
        nesting.get("inner_base_seed_formula") != "seed + inner_index"
        or nesting.get("inner_expert_seed_formula")
        != "seed + 101*view_index + 17*estimator_index + inner_index"
    ):
        raise ValueError("retrospective CAGE-HAR inner seed schedule changed")

    probe = _mapping(config.get("preliminary_probe_disclosure"), name="probe disclosure")
    if (
        probe.get("performed_before_this_fully_nested_protocol") is not True
        or probe.get("method_or_hyperparameters_changed_after_probe") is not False
        or probe.get("probe_is_publication_evidence") is not False
    ):
        raise ValueError("preliminary feasibility probe is not fully disclosed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    forbidden_false = (
        "participants_11_through_20_may_be_loaded",
        "daghar_may_be_loaded",
        "independent_validation_claim_allowed",
        "confirmatory_claim_allowed",
        "state_of_the_art_claim_allowed",
    )
    if any(policy.get(key) is not False for key in forbidden_false):
        raise ValueError("retrospective CAGE-HAR claim boundary changed")
    return config


def _validated_path(reference: object, *, name: str) -> Path:
    item = _mapping(reference, name=name)
    path = Path(str(item.get("path", "")))
    expected = item.get("sha256")
    if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
        raise ValueError(f"{name} is missing or changed: {path}")
    return path


def _outer_context(
    signals: FloatArray,
    gravity: FloatArray,
    training_mask: BoolArray,
    *,
    sampling_rate_hz: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build label-free context with scale fitted only on outer training rows."""

    combined = np.concatenate((signals, gravity), axis=2)
    validity = np.ones(combined.shape, dtype=np.bool_)
    scale = np.maximum(np.median(np.abs(combined[training_mask]), axis=(0, 1)), 1e-6)
    gauge = extract_cage_gauge_context(
        signals,
        gravity,
        sampling_rate_hz=sampling_rate_hz,
        validity_mask=validity,
    )
    health = extract_cage_sensor_health(
        combined,
        validity_mask=validity,
        channel_scale=scale,
    )
    context = np.concatenate((gauge.features, health.features), axis=1)
    reliability = cage_expert_reliability(
        health,
        required_modalities=("accelerometer", "gyroscope", "gravity"),
        gravity_gauge=gauge,
    )
    return (
        np.asarray(context, dtype=np.float64),
        np.asarray(reliability, dtype=np.float64),
        np.asarray(scale, dtype=np.float64),
    )


def evaluate_cage_outer_arrays(
    *,
    training_base_probability: FloatArray,
    training_expert_probability: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    training_context: FloatArray,
    training_reliability: FloatArray,
    evaluation_base_probability: FloatArray,
    evaluation_expert_probability: FloatArray,
    evaluation_ctgr_probability: FloatArray,
    evaluation_context: FloatArray,
    evaluation_reliability: FloatArray,
    expert_name: str,
    cage_config: dict[str, Any],
) -> tuple[dict[str, FloatArray], dict[str, Any], dict[str, NDArray[Any]]]:
    """Fit CAGE only on inner-OOF arrays and return label-unaware outer predictions."""

    router = cast(dict[str, Any], cage_config["router"])
    ablations = cast(dict[str, Any], cage_config["ablations"])
    base_selection, base_candidates = _select_probability_repair(
        training_base_probability,
        training_labels,
        training_participants,
        cage_config,
    )
    expert_selection, expert_candidates = _select_probability_repair(
        training_expert_probability,
        training_labels,
        training_participants,
        cage_config,
    )
    repaired_base_training = _apply_selected_repair(training_base_probability, base_selection)
    repaired_expert_training = _apply_selected_repair(training_expert_probability, expert_selection)
    repaired_base_evaluation = _apply_selected_repair(evaluation_base_probability, base_selection)
    repaired_expert_evaluation = _apply_selected_repair(
        evaluation_expert_probability, expert_selection
    )
    training_features, feature_names = build_advantage_router_features(
        repaired_base_training,
        repaired_expert_training,
        context_features=training_context,
        expert_reliability=training_reliability,
    )
    evaluation_features, evaluation_feature_names = build_advantage_router_features(
        repaired_base_evaluation,
        repaired_expert_evaluation,
        context_features=evaluation_context,
        expert_reliability=evaluation_reliability,
    )
    if feature_names != evaluation_feature_names:
        raise AssertionError("retrospective CAGE router feature order changed")
    advantage = counterfactual_log_loss_advantage(
        repaired_base_training,
        repaired_expert_training,
        training_labels,
    )
    weights = _tail_and_harm_weights(
        repaired_base_training,
        repaired_expert_training,
        training_labels,
        training_participants,
        bottom_tail_fraction=float(router["bottom_tail_fraction"]),
        bottom_tail_multiplier=float(router["bottom_tail_weight_multiplier"]),
        harm_multiplier=float(router["harm_weight_multiplier"]),
    )
    prediction = participant_jackknife_advantage_prediction(
        training_features,
        advantage,
        training_participants,
        evaluation_features,
        ridge_penalty=float(router["ridge_penalty"]),
        lower_bound_z=float(router["lower_bound_z"]),
        training_sample_weight=weights,
    )
    expert_stack = repaired_expert_evaluation[:, None, :]
    reliability_stack = evaluation_reliability[:, None]
    posture_only = np.asarray([True], dtype=np.bool_)
    cage = apply_cage_trust_region(
        repaired_base_evaluation,
        expert_stack,
        prediction.lower_bound[:, None],
        reliability_stack,
        posture_only,
        minimum_advantage=float(router["minimum_advantage"]),
        minimum_reliability=float(router["minimum_reliability"]),
        maximum_mix_weight=float(router["maximum_mix_weight"]),
        advantage_scale=float(router["advantage_scale"]),
        maximum_kl=float(router["maximum_kl"]),
    )
    all_rows = np.ones(repaired_base_evaluation.shape[0], dtype=np.bool_)
    global_blend = _fixed_gate(
        repaired_base_evaluation,
        expert_stack,
        reliability_stack,
        posture_only,
        selected_expert=0,
        gate=all_rows,
        blend_weight=float(ablations["fixed_blend_weight"]),
        minimum_reliability=float(router["minimum_reliability"]),
        maximum_kl=float(router["maximum_kl"]),
    )
    confidence_blend = _fixed_gate(
        repaired_base_evaluation,
        expert_stack,
        reliability_stack,
        posture_only,
        selected_expert=0,
        gate=repaired_base_evaluation.max(axis=1) < float(ablations["confidence_threshold"]),
        blend_weight=float(ablations["fixed_blend_weight"]),
        minimum_reliability=float(router["minimum_reliability"]),
        maximum_kl=float(router["maximum_kl"]),
    )
    disagreement_blend = _fixed_gate(
        repaired_base_evaluation,
        expert_stack,
        reliability_stack,
        posture_only,
        selected_expert=0,
        gate=repaired_base_evaluation.argmax(axis=1) != repaired_expert_evaluation.argmax(axis=1),
        blend_weight=float(ablations["fixed_blend_weight"]),
        minimum_reliability=float(router["minimum_reliability"]),
        maximum_kl=float(router["maximum_kl"]),
    )
    routes = {
        "global_trust_blend": global_blend,
        "confidence_trust_gate": confidence_blend,
        "disagreement_trust_gate": disagreement_blend,
        "cage_har": cage,
    }
    methods = {
        "base_uncalibrated": np.asarray(evaluation_base_probability, dtype=np.float64),
        "base_decision_repaired": repaired_base_evaluation,
        **{name: value.probabilities for name, value in routes.items()},
        "frozen_ctgr": np.asarray(evaluation_ctgr_probability, dtype=np.float64),
    }
    diagnostics: dict[str, Any] = {
        "base_repair": base_selection,
        "base_repair_candidate_count": len(base_candidates),
        "expert_repair": expert_selection,
        "expert_repair_candidate_count": len(expert_candidates),
        "expert_name": expert_name,
        "router_feature_count": len(feature_names),
        "router_jackknife_model_count": prediction.model_count,
        "mean_training_counterfactual_advantage": float(np.mean(advantage)),
        "mean_predicted_advantage": float(np.mean(prediction.mean)),
        "mean_predicted_advantage_lower_bound": float(np.mean(prediction.lower_bound)),
        "outer_labels_used_for_training_selection_or_repair": False,
    }
    arrays: dict[str, NDArray[Any]] = {
        **{f"{name}_route_mask": value.routed for name, value in routes.items()},
        **{f"{name}_chosen_expert": value.chosen_expert for name, value in routes.items()},
        "cage_mix_weight": cage.mix_weight,
        "cage_achieved_kl": cage.achieved_kl,
        "cage_predicted_advantage_mean": prediction.mean,
        "cage_predicted_advantage_lower_bound": prediction.lower_bound,
    }
    return methods, diagnostics, arrays


def _load_preserved_seed(
    config: dict[str, Any],
    *,
    seed: int,
    canonical_windows: StringArray,
    canonical_labels: IntArray,
    canonical_participants: StringArray,
    source_manifest_sha256: str,
    ctgr_config_sha256: str,
    selection_sha256: str,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    inputs = cast(dict[str, Any], config["inputs"])
    pattern = str(inputs["ctgr_evaluation_record_pattern"])
    record_path = Path(pattern.format(seed=seed))
    record = _read_hashed_record(record_path)
    if (
        int(record.get("seed", -1)) != seed
        or record.get("status") != "complete_target_sealed"
        or record.get("target_performance_or_prediction_accessed") is not False
        or record.get("target_subject_or_window_records_loaded") is not False
        or record["source_manifest"]["sha256"] != source_manifest_sha256
        or record["config"]["sha256"] != ctgr_config_sha256
        or record["selection_record"]["sha256"] != selection_sha256
    ):
        raise PermissionError(f"preserved CTGR seed {seed} violates retrospective lineage")
    prediction_ref = _mapping(record.get("predictions"), name="CTGR prediction reference")
    prediction_path = Path(str(prediction_ref.get("path", "")))
    if not prediction_path.is_file() or sha256_file(prediction_path) != prediction_ref.get(
        "sha256"
    ):
        raise ValueError(f"preserved CTGR prediction artifact changed for seed {seed}")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        source_windows = np.asarray(loaded["window_ids"], dtype=np.str_)
        if len(set(source_windows.tolist())) != canonical_windows.size:
            raise ValueError(f"preserved CTGR seed {seed} does not cover source exactly once")
        source_index = {str(window): index for index, window in enumerate(source_windows.tolist())}
        try:
            order = np.asarray(
                [source_index[str(window)] for window in canonical_windows], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(f"preserved CTGR seed {seed} window coverage changed") from exc
        labels = np.asarray(loaded["labels"], dtype=np.int64)[order]
        participants = np.asarray(loaded["participant_ids"], dtype=np.str_)[order]
        if not np.array_equal(labels, canonical_labels) or not np.array_equal(
            participants, canonical_participants
        ):
            raise ValueError(f"preserved CTGR seed {seed} labels/participants do not align")
        probabilities = {
            "base": np.asarray(loaded["flat_rmrp_probabilities"], dtype=np.float64)[order],
            "expert": np.asarray(loaded["gravity_posture_expert_probabilities"], dtype=np.float64)[
                order
            ],
            "ctgr": np.asarray(loaded["ctgr_probabilities"], dtype=np.float64)[order],
        }
    aggregate_reports = _mapping(record.get("aggregate_reports"), name="CTGR aggregate reports")
    expected_means = {
        "base_uncalibrated": float(
            aggregate_reports["flat_rmrp"]["primary"]["mean_participant_macro_f1"]
        ),
        "gravity_posture_expert": float(
            aggregate_reports["gravity_posture_expert"]["primary"]["mean_participant_macro_f1"]
        ),
        "frozen_ctgr": float(aggregate_reports["ctgr"]["primary"]["mean_participant_macro_f1"]),
    }
    return probabilities, {
        "path": record_path.as_posix(),
        "sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "expected_mean_participant_macro_f1": expected_means,
    }


def _inner_oof_probabilities(
    *,
    outer: dict[str, Any],
    candidate: dict[str, Any],
    seed: int,
    n_jobs: int,
    rmrp: FloatArray,
    views: dict[str, FloatArray],
    labels: IntArray,
    participants: StringArray,
) -> tuple[FloatArray, FloatArray, BoolArray]:
    outer_test = [str(item) for item in cast(list[Any], outer["outer_test_subjects"])]
    outer_training = ~_mask_for_subjects(participants, outer_test)
    base_oof = np.full((labels.size, 3), np.nan, dtype=np.float64)
    expert_oof = np.full_like(base_oof, np.nan)
    covered = np.zeros(labels.size, dtype=np.bool_)
    view = str(candidate["expert_view"])
    estimator = str(candidate["posture_estimator"])
    view_index = _VIEW_ORDER.index(view)
    estimator_index = _ESTIMATOR_ORDER.index(estimator)
    inner_folds = cast(list[dict[str, Any]], outer["inner_folds"])
    if len(inner_folds) != 4:
        raise ValueError("retrospective CAGE outer fold must contain four inner folds")
    for inner_index, inner in enumerate(inner_folds):
        training = _mask_for_subjects(
            participants, [str(item) for item in cast(list[Any], inner["train_subjects"])]
        )
        validation = _mask_for_subjects(
            participants, [str(item) for item in cast(list[Any], inner["validation_subjects"])]
        )
        if (
            np.any(training & validation)
            or np.any(validation & ~outer_training)
            or np.any(training & ~outer_training)
            or np.any(covered & validation)
        ):
            raise PermissionError("retrospective CAGE inner/outer partition leakage")
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training,
            seed=seed + inner_index,
            n_jobs=n_jobs,
        )
        base_probability = np.asarray(base_model.predict_proba(rmrp[validation]), dtype=np.float64)
        expert_model = _fit_expert(
            estimator,
            views[view],
            labels,
            participants,
            training,
            seed=seed + 101 * view_index + 17 * estimator_index + inner_index,
            n_jobs=n_jobs,
        )
        expert_probability = _gravity_probability(
            base_probability,
            expert_model,
            views[view][validation],
        )
        base_oof[validation] = base_probability
        expert_oof[validation] = expert_probability
        covered[validation] = True
    if not np.array_equal(covered, outer_training):
        raise ValueError("retrospective CAGE inner folds do not cover outer training exactly once")
    if (
        not np.isfinite(base_oof[outer_training]).all()
        or not np.isfinite(expert_oof[outer_training]).all()
    ):
        raise ValueError("retrospective CAGE inner OOF probabilities are incomplete")
    return base_oof, expert_oof, outer_training


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_cage_retrospective(
    *,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Execute the predeclared nested retrospective run across all five seeds."""

    if output_directory.exists():
        raise FileExistsError(f"retrospective CAGE output already exists: {output_directory}")
    config = load_cage_retrospective_config(config_path)
    inputs = cast(dict[str, Any], config["inputs"])
    freeze = cast(dict[str, Any], config["method_freeze"])
    source_manifest_path = _validated_path(inputs["source_manifest"], name="source manifest")
    dataset_manifest_path = _validated_path(inputs["dataset_manifest"], name="dataset manifest")
    raw_source_path = _validated_path(inputs["raw_source"], name="raw source")
    ctgr_config_path = _validated_path(inputs["ctgr_config"], name="CTGR config")
    selection_path = _validated_path(inputs["ctgr_selection"], name="CTGR selection")
    cage_config_path = _validated_path(freeze["method_config"], name="CAGE method config")
    cage_config = _load_cage_config(cage_config_path)
    ctgr_config = _load_ctgr_config(ctgr_config_path)
    selection = _read_hashed_record(selection_path)
    if (
        selection.get("status") != "selection_frozen_before_outer_evaluation"
        or selection.get("outer_labels_used_for_selection") is not False
        or selection.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("retrospective CAGE requires the frozen source-only CTGR map")

    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path,
        raw_csv_path=raw_source_path,
    )
    if tuple(sorted(set(participants.tolist()), key=int)) != _EXPECTED_PARTICIPANTS:
        raise PermissionError("retrospective CAGE source participant set changed")
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_source_path,
        expected_participants=participants,
        expected_windows=windows,
    )
    sampling_rate_hz = float(ctgr_config["input_contract"]["sampling_rate_hz"])
    if sampling_rate_hz != 50.0:
        raise ValueError("retrospective CAGE sampling rate changed")
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=sampling_rate_hz)
    candidates = {str(item["id"]): item for item in _ctgr_candidates(ctgr_config)}
    selected = _selected_by_fold(selection)
    outer_folds = cast(list[dict[str, Any]], manifest["source_nested_cv"])
    if len(outer_folds) != 5:
        raise ValueError("retrospective CAGE source manifest must contain five outer folds")
    n_jobs = int(ctgr_config["n_jobs"])

    prediction_payload: dict[str, NDArray[Any]] = {
        "labels": labels,
        "participant_ids": participants,
        "window_ids": windows,
    }
    seed_records: list[dict[str, Any]] = []
    reports_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for seed in _EXPECTED_SEEDS:
        preserved, preserved_record = _load_preserved_seed(
            config,
            seed=seed,
            canonical_windows=windows,
            canonical_labels=labels,
            canonical_participants=participants,
            source_manifest_sha256=str(inputs["source_manifest"]["sha256"]),
            ctgr_config_sha256=str(inputs["ctgr_config"]["sha256"]),
            selection_sha256=str(inputs["ctgr_selection"]["sha256"]),
        )
        method_probabilities = {
            name: np.full((labels.size, 3), np.nan, dtype=np.float64)
            for name in _RETROSPECTIVE_METHODS
        }
        method_routes = {
            name: np.zeros(labels.size, dtype=np.bool_)
            for name in _METHOD_NAMES
            if name not in {"base_uncalibrated", "base_decision_repaired"}
        }
        method_experts = {name: np.full(labels.size, -1, dtype=np.int64) for name in method_routes}
        cage_mix_weight = np.zeros(labels.size, dtype=np.float64)
        cage_kl = np.zeros(labels.size, dtype=np.float64)
        cage_advantage = np.full(labels.size, np.nan, dtype=np.float64)
        cage_lower_bound = np.full(labels.size, np.nan, dtype=np.float64)
        fold_records: list[dict[str, Any]] = []
        for outer in outer_folds:
            fold_id = str(outer["outer_fold_id"])
            candidate_id = selected[fold_id]
            candidate = candidates[candidate_id]
            if (
                candidate.get("expert_view") not in _VIEW_ORDER
                or candidate.get("posture_estimator") not in _ESTIMATOR_ORDER
            ):
                raise ValueError(f"outer fold {fold_id} lacks a usable frozen gravity expert")
            base_oof, expert_oof, training = _inner_oof_probabilities(
                outer=outer,
                candidate=candidate,
                seed=seed,
                n_jobs=n_jobs,
                rmrp=rmrp,
                views=views,
                labels=labels,
                participants=participants,
            )
            evaluation = ~training
            context, reliability, scale = _outer_context(
                signals,
                gravity,
                training,
                sampling_rate_hz=sampling_rate_hz,
            )
            methods, diagnostics, arrays = evaluate_cage_outer_arrays(
                training_base_probability=base_oof[training],
                training_expert_probability=expert_oof[training],
                training_labels=labels[training],
                training_participants=participants[training],
                training_context=context[training],
                training_reliability=reliability[training],
                evaluation_base_probability=preserved["base"][evaluation],
                evaluation_expert_probability=preserved["expert"][evaluation],
                evaluation_ctgr_probability=preserved["ctgr"][evaluation],
                evaluation_context=context[evaluation],
                evaluation_reliability=reliability[evaluation],
                expert_name=candidate_id,
                cage_config=cage_config,
            )
            for name, probability in methods.items():
                method_probabilities[name][evaluation] = probability
            for name in method_routes:
                method_routes[name][evaluation] = np.asarray(
                    arrays[f"{name}_route_mask"], dtype=np.bool_
                )
                method_experts[name][evaluation] = np.asarray(
                    arrays[f"{name}_chosen_expert"], dtype=np.int64
                )
            cage_mix_weight[evaluation] = np.asarray(arrays["cage_mix_weight"], dtype=np.float64)
            cage_kl[evaluation] = np.asarray(arrays["cage_achieved_kl"], dtype=np.float64)
            cage_advantage[evaluation] = np.asarray(
                arrays["cage_predicted_advantage_mean"], dtype=np.float64
            )
            cage_lower_bound[evaluation] = np.asarray(
                arrays["cage_predicted_advantage_lower_bound"], dtype=np.float64
            )
            fold_records.append(
                {
                    "outer_fold_id": fold_id,
                    "outer_test_subjects": list(outer["outer_test_subjects"]),
                    "selected_ctgr_candidate_id": candidate_id,
                    "outer_training_participant_count": int(np.unique(participants[training]).size),
                    "inner_oof_window_count": int(training.sum()),
                    "outer_evaluation_window_count": int(evaluation.sum()),
                    "outer_training_channel_scale": scale.tolist(),
                    **diagnostics,
                }
            )
        if any(not np.isfinite(probability).all() for probability in method_probabilities.values()):
            raise ValueError(f"retrospective CAGE seed {seed} predictions are incomplete")
        reports = {
            name: _report(labels, probability, participants)
            for name, probability in method_probabilities.items()
        }
        expected_means = cast(
            dict[str, float], preserved_record["expected_mean_participant_macro_f1"]
        )
        replay_values = {
            "base_uncalibrated": float(
                reports["base_uncalibrated"]["primary"]["mean_participant_macro_f1"]
            ),
            "gravity_posture_expert": float(
                _report(labels, preserved["expert"], participants)["primary"][
                    "mean_participant_macro_f1"
                ]
            ),
            "frozen_ctgr": float(reports["frozen_ctgr"]["primary"]["mean_participant_macro_f1"]),
        }
        if any(
            not np.isclose(replay_values[name], expected, atol=1e-12, rtol=0.0)
            for name, expected in expected_means.items()
        ):
            raise ValueError(f"preserved CTGR report replay changed for seed {seed}")
        preserved_record["aggregate_report_replay_exact"] = True
        routing = {
            name: _routing_audit(
                labels,
                participants,
                method_probabilities["base_decision_repaired"],
                method_probabilities[name],
                method_routes[name],
                method_experts[name],
                ("gravity_posture_expert",),
            )
            for name in method_routes
        }
        reports_by_seed[seed] = reports
        seed_records.append(
            {
                "seed": seed,
                "preserved_ctgr_record": preserved_record,
                "folds": fold_records,
                "reports": reports,
                "routing_audits": routing,
                "outer_labels_used_for_training_selection_or_repair": False,
                "participants_11_through_20_loaded": False,
                "daghar_loaded": False,
            }
        )
        for name, probability in method_probabilities.items():
            prediction_payload[f"seed_{seed}__{name}_probabilities"] = probability
        for name, mask in method_routes.items():
            prediction_payload[f"seed_{seed}__{name}_route_mask"] = mask
            prediction_payload[f"seed_{seed}__{name}_chosen_expert"] = method_experts[name]
        prediction_payload[f"seed_{seed}__cage_mix_weight"] = cage_mix_weight
        prediction_payload[f"seed_{seed}__cage_achieved_kl"] = cage_kl
        prediction_payload[f"seed_{seed}__cage_predicted_advantage_mean"] = cage_advantage
        prediction_payload[f"seed_{seed}__cage_predicted_advantage_lower_bound"] = cage_lower_bound

    aggregate: dict[str, dict[str, float]] = {}
    for method in _RETROSPECTIVE_METHODS:
        method_seed_reports = [reports_by_seed[seed][method] for seed in _EXPECTED_SEEDS]
        aggregate[method] = {
            "mean_participant_macro_f1": _mean(
                [
                    float(item["primary"]["mean_participant_macro_f1"])
                    for item in method_seed_reports
                ]
            ),
            "bottom_30_percent_participant_macro_f1": _mean(
                [
                    float(item["primary"]["bottom_30_percent_participant_macro_f1"])
                    for item in method_seed_reports
                ]
            ),
            "worst_participant_macro_f1": _mean(
                [
                    float(item["primary"]["worst_participant_macro_f1"])
                    for item in method_seed_reports
                ]
            ),
            "negative_log_likelihood": _mean(
                [
                    float(item["calibration"]["negative_log_likelihood"])
                    for item in method_seed_reports
                ]
            ),
            "multiclass_brier_score": _mean(
                [
                    float(item["calibration"]["multiclass_brier_score"])
                    for item in method_seed_reports
                ]
            ),
        }
    cage_mean = aggregate["cage_har"]["mean_participant_macro_f1"]
    comparisons = {
        comparator: cage_mean - aggregate[comparator]["mean_participant_macro_f1"]
        for comparator in ("base_uncalibrated", "base_decision_repaired", "frozen_ctgr")
    }
    cage_audits = [
        cast(dict[str, Any], item["routing_audits"])["cage_har"] for item in seed_records
    ]
    rescues = sum(int(item["rescue_count"]) for item in cage_audits)
    harms = sum(int(item["harm_count"]) for item in cage_audits)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "cage_har_fully_nested_retrospective_source_result",
        "status": "complete_reused_source_development_not_independent",
        "evidence_status": config["evidence_status"],
        "code_commit": code_commit,
        "method_freeze": freeze,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "ctgr_selection": {
            "path": selection_path.as_posix(),
            "sha256": sha256_file(selection_path),
            "record_sha256": selection["record_sha256"],
        },
        "participant_count": int(np.unique(participants).size),
        "window_count": int(labels.size),
        "seeds": seed_records,
        "aggregate_five_seed_means": aggregate,
        "cage_har_mean_macro_f1_differences": comparisons,
        "cage_har_aggregate_routing": {
            "route_count": sum(int(item["routed_count"]) for item in cage_audits),
            "changed_count": sum(int(item["changed_count"]) for item in cage_audits),
            "rescue_count": rescues,
            "harm_count": harms,
            "intervention_precision": (
                None if rescues + harms == 0 else float(rescues / (rescues + harms))
            ),
        },
        "preliminary_probe_disclosed": True,
        "method_or_hyperparameters_changed_after_preliminary_probe": False,
        "participants_11_through_20_loaded": False,
        "daghar_loaded": False,
        "independent_validation_claim_allowed": False,
        "confirmatory_claim_allowed": False,
        "state_of_the_art_claim_allowed": False,
        "new_cohort_still_required": True,
    }
    return _write_result(output_directory, result, prediction_payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_cage_retrospective(
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_status": result["evidence_status"],
                "aggregate_five_seed_means": result["aggregate_five_seed_means"],
                "cage_har_mean_macro_f1_differences": result["cage_har_mean_macro_f1_differences"],
                "cage_har_aggregate_routing": result["cage_har_aggregate_routing"],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
