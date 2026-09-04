"""Fully nested retrospective screen of the frozen HERA-CTGR v2 invention.

The runner is intentionally restricted to exhausted InclusiveHAR source
participants 1--10.  It cannot provide independent or confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.experiments.cage_har import _report, _routing_audit, _write_result
from inclusive_shift_har.experiments.cage_har_retrospective import (
    _load_preserved_seed,
    _outer_context,
    _validated_path,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _ESTIMATOR_ORDER,
    _VIEW_ORDER,
    _candidates,
    _feature_views,
    _fit_base,
    _fit_expert,
    _gravity_probability,
    _load_config,
    _materialize_gravity,
    _selected_by_fold,
    apply_confidence_triggered_gravity_residual,
)
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.hera_ctgr_retrospective import (
    _apply_candidate,
    _bootstrap_intervals,
    _slice_context,
    _top_ranked_candidates,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _materialize_source,
    _positive_probability,
    _read_hashed_record,
)
from inclusive_shift_har.manifests.canonical import sha256_file
from inclusive_shift_har.models.hera_ctgr import (
    apply_physics_reference,
    extract_gravity_kinematic_context,
    fit_physics_reference,
    standardized_nearest_support_distance,
)
from inclusive_shift_har.models.hera_ctgr_v2 import (
    BinaryJackknifePrediction,
    SentinelRoutingResult,
    apply_one_query_state_selection,
    apply_rank_locked_posture_offset,
    apply_rescue_harm_sentinel,
    build_rescue_harm_features,
    counterfactual_rescue_harm_targets,
    extract_dual_frame_posture_features,
    participant_jackknife_binary_prediction,
    stability_weighted_candidate_marginalization,
    temperature_scale_probabilities,
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

_EXPECTED_SEEDS = (11, 23, 47, 89, 131)
_EXPECTED_PARTICIPANTS = tuple(str(index) for index in range(1, 11))
_METHODS = (
    "base_uncalibrated",
    "frozen_ctgr",
    "frozen_hera_ctgr_v1_strict",
    "equal_top3_ctgr",
    "stability_weighted_top3_ctgr",
    "temperature_only_core",
    "decision_separated_core",
    "dual_frame_global_candidate",
    "rescue_harm_logits_only",
    "rescue_harm_physics",
    "hera_ctgr_v2_full",
    "one_query_matched_core",
    "one_query_personalization",
    "participant_oracle_diagnostic",
    "window_oracle_diagnostic",
)
_SCIENTIFIC_ZERO_SHOT = _METHODS[:11]
_DIAGNOSTIC_METHODS = _METHODS[-2:]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def load_hera_v2_retrospective_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on any change to the v2 retrospective contract."""

    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("status") != "locked_before_fully_nested_retrospective_source_run":
        raise ValueError("HERA-CTGR v2 retrospective protocol is not locked")
    if tuple(int(item) for item in config.get("fixed_seeds", [])) != _EXPECTED_SEEDS:
        raise ValueError("HERA-CTGR v2 seed set changed")
    if tuple(str(item) for item in config.get("participant_ids", [])) != _EXPECTED_PARTICIPANTS:
        raise ValueError("HERA-CTGR v2 participant set changed")
    if tuple(str(item) for item in config.get("methods", [])) != _METHODS:
        raise ValueError("HERA-CTGR v2 method matrix changed")
    freeze = _mapping(config.get("method_freeze"), name="method freeze")
    if (
        freeze.get("tag") != "hera-ctgr-v2-implementation-v1"
        or freeze.get("commit") != "5def59c1bf1c9276a10b1fccbf7b023975d962a2"
        or freeze.get("parameters_may_change_after_run") is not False
    ):
        raise ValueError("HERA-CTGR v2 implementation freeze changed")
    fixed = _mapping(config.get("fixed_method"), name="fixed method")
    expected = {
        "candidate_count": 3,
        "divergence_temperature": 0.05,
        "minimum_candidate_weight": 0.05,
        "temperature_candidates": [0.75, 0.9, 1.0, 1.1, 1.25],
        "posture_logit_offset_candidates": [-0.2, -0.1, 0.0, 0.1, 0.2],
        "dual_estimator": "extra_trees_leaf3",
        "dual_confidence_thresholds": [0.5, 0.55, 0.6, 0.65],
        "dual_blend_weights": [0.25, 0.5, 0.75, 1.0],
        "dual_selection_mean_tolerance": 0.005,
        "physics_veto_quantile": 0.99,
        "physics_minimum_valid_pair_fraction": 0.98,
        "binary_ridge_penalty": 25.0,
        "jackknife_interval_z": 1.0,
        "harm_penalty": 2.0,
        "minimum_physics_reliability": 0.2,
        "maximum_support_distance": 3.0,
        "minimum_rescue_lower_bound_grid": [0.05, 0.1, 0.2, 0.3],
        "maximum_harm_upper_bound_grid": [0.05, 0.1, 0.15, 0.25],
        "minimum_net_benefit_grid": [0.0, 0.05, 0.1, 0.2],
        "training_minimum_precision": 0.8,
        "training_maximum_harmful_fraction": 0.15,
        "one_query_minimum_stationary_mass": 0.8,
    }
    if fixed != expected:
        raise ValueError("HERA-CTGR v2 fixed settings changed")
    nesting = _mapping(config.get("nesting_contract"), name="nesting contract")
    required = (
        "frozen_ctgr_and_hera_v1_predictions_replayed",
        "base_candidate_and_dual_predictions_cross_fitted_by_participant",
        "temperature_offset_dual_and_sentinel_thresholds_selected_on_outer_training_oof_only",
        "binary_heads_trained_on_participant_exclusive_oof_disagreements",
        "outer_labels_used_only_after_all_zero_shot_predictions_are_fixed",
        "participant_is_inference_unit",
    )
    if any(nesting.get(item) is not True for item in required):
        raise ValueError("HERA-CTGR v2 nesting guarantee changed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    forbidden = (
        "participants_11_through_20_may_be_loaded",
        "daghar_may_be_loaded",
        "independent_validation_claim_allowed",
        "confirmatory_claim_allowed",
        "state_of_the_art_claim_allowed",
    )
    if any(policy.get(item) is not False for item in forbidden):
        raise ValueError("HERA-CTGR v2 claim boundary changed")
    return config


def _rank_lock_candidate(base: FloatArray, candidate: FloatArray) -> FloatArray:
    result = np.asarray(candidate, dtype=np.float64).copy()
    base_class = base.argmax(axis=1)
    candidate_class = result.argmax(axis=1)
    invalid = (base_class == 0) | (candidate_class == 0)
    result[invalid] = base[invalid]
    if np.any((result.argmax(axis=1) == 0) != (base_class == 0)):
        raise AssertionError("posture candidate changed mobility membership")
    return result


def _inner_paths(
    *,
    outer: dict[str, Any],
    top_candidates: tuple[dict[str, Any], ...],
    seed: int,
    n_jobs: int,
    rmrp: FloatArray,
    views: dict[str, FloatArray],
    dual_features: FloatArray,
    labels: IntArray,
    participants: StringArray,
) -> tuple[dict[str, FloatArray], BoolArray]:
    outer_test = [str(item) for item in cast(list[Any], outer["outer_test_subjects"])]
    outer_training = ~_mask_for_subjects(participants, outer_test)
    arrays = {
        name: np.full((labels.size, 3), np.nan, dtype=np.float64)
        for name in ("base", "candidate_0", "candidate_1", "candidate_2", "dual_expert")
    }
    covered = np.zeros(labels.size, dtype=np.bool_)
    inner_folds = cast(list[dict[str, Any]], outer["inner_folds"])
    if len(inner_folds) != 4:
        raise ValueError("HERA-CTGR v2 requires four inner folds")
    for inner_index, inner in enumerate(inner_folds):
        training = _mask_for_subjects(
            participants, [str(item) for item in cast(list[Any], inner["train_subjects"])]
        )
        validation = _mask_for_subjects(
            participants, [str(item) for item in cast(list[Any], inner["validation_subjects"])]
        )
        if (
            np.any(training & validation)
            or np.any(training & ~outer_training)
            or np.any(validation & ~outer_training)
            or np.any(covered & validation)
        ):
            raise PermissionError("HERA-CTGR v2 inner/outer participant leakage")
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training,
            seed=seed + inner_index,
            n_jobs=n_jobs,
        )
        base = np.asarray(base_model.predict_proba(rmrp[validation]), dtype=np.float64)
        arrays["base"][validation] = base
        expert_cache: dict[tuple[str, str], FloatArray] = {}
        for candidate_index, candidate in enumerate(top_candidates):
            view = str(candidate["expert_view"])
            estimator = str(candidate["posture_estimator"])
            if view not in _VIEW_ORDER or estimator not in _ESTIMATOR_ORDER:
                raise ValueError("frozen CTGR candidate is unavailable to HERA-CTGR v2")
            key = (view, estimator)
            if key not in expert_cache:
                model = _fit_expert(
                    estimator,
                    views[view],
                    labels,
                    participants,
                    training,
                    seed=seed
                    + 101 * _VIEW_ORDER.index(view)
                    + 17 * _ESTIMATOR_ORDER.index(estimator)
                    + inner_index,
                    n_jobs=n_jobs,
                )
                expert_cache[key] = _gravity_probability(base, model, views[view][validation])
            arrays[f"candidate_{candidate_index}"][validation] = _apply_candidate(
                base, expert_cache[key], candidate
            )
        dual_model = _fit_expert(
            "extra_trees_leaf3",
            dual_features,
            labels,
            participants,
            training,
            seed=seed + 701 + inner_index,
            n_jobs=n_jobs,
        )
        arrays["dual_expert"][validation] = compose_mobility_posture_probabilities(
            base[:, 0], _positive_probability(dual_model, dual_features[validation])
        )
        covered[validation] = True
    if not np.array_equal(covered, outer_training):
        raise ValueError("HERA-CTGR v2 inner folds do not cover outer training exactly once")
    if any(not np.isfinite(item[outer_training]).all() for item in arrays.values()):
        raise ValueError("HERA-CTGR v2 inner predictions are incomplete")
    return arrays, outer_training


def _outer_paths(
    *,
    top_candidates: tuple[dict[str, Any], ...],
    preserved: dict[str, FloatArray],
    evaluation: BoolArray,
    training: BoolArray,
    outer_index: int,
    seed: int,
    n_jobs: int,
    views: dict[str, FloatArray],
    dual_features: FloatArray,
    labels: IntArray,
    participants: StringArray,
) -> tuple[FloatArray, FloatArray, tuple[str, ...]]:
    base = preserved["base"][evaluation]
    selected = top_candidates[0]
    selected_key = (str(selected["expert_view"]), str(selected["posture_estimator"]))
    expert_cache: dict[tuple[str, str], FloatArray] = {
        selected_key: preserved["expert"][evaluation]
    }
    paths: list[FloatArray] = []
    for candidate in top_candidates:
        view = str(candidate["expert_view"])
        estimator = str(candidate["posture_estimator"])
        key = (view, estimator)
        if key not in expert_cache:
            model = _fit_expert(
                estimator,
                views[view],
                labels,
                participants,
                training,
                seed=seed + 101 * outer_index,
                n_jobs=n_jobs,
            )
            expert_cache[key] = _gravity_probability(base, model, views[view][evaluation])
        paths.append(_apply_candidate(base, expert_cache[key], candidate))
    if not np.allclose(paths[0], preserved["ctgr"][evaluation], atol=1e-12, rtol=0.0):
        raise ValueError("HERA-CTGR v2 no longer replays frozen CTGR")
    dual_model = _fit_expert(
        "extra_trees_leaf3",
        dual_features,
        labels,
        participants,
        training,
        seed=seed + 1701 + outer_index,
        n_jobs=n_jobs,
    )
    dual_probability = compose_mobility_posture_probabilities(
        base[:, 0], _positive_probability(dual_model, dual_features[evaluation])
    )
    return (
        np.stack(paths, axis=0),
        dual_probability,
        tuple(str(item["id"]) for item in top_candidates),
    )


def _marginalize(
    stack: FloatArray, fixed: dict[str, Any]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    equal = np.asarray(stack.mean(axis=0), dtype=np.float64)
    weighted = stability_weighted_candidate_marginalization(
        stack,
        divergence_temperature=float(fixed["divergence_temperature"]),
        minimum_candidate_weight=float(fixed["minimum_candidate_weight"]),
    )
    return equal, weighted.probabilities, weighted.disagreement


def _select_temperature(
    probability: FloatArray,
    labels: IntArray,
    participants: StringArray,
    fixed: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    summaries: list[dict[str, float]] = []
    for temperature in cast(list[float], fixed["temperature_candidates"]):
        candidate = temperature_scale_probabilities(probability, temperature=float(temperature))
        report = _report(labels, candidate, participants)
        summaries.append(
            {
                "temperature": float(temperature),
                "negative_log_likelihood": float(report["calibration"]["negative_log_likelihood"]),
                "mean_participant_macro_f1": float(report["primary"]["mean_participant_macro_f1"]),
            }
        )
    summaries.sort(
        key=lambda item: (
            item["negative_log_likelihood"],
            abs(np.log(item["temperature"])),
            item["temperature"],
        )
    )
    return summaries[0]["temperature"], summaries


def _select_offset(
    probability: FloatArray,
    labels: IntArray,
    participants: StringArray,
    fixed: dict[str, Any],
) -> tuple[float, list[dict[str, float]]]:
    summaries: list[dict[str, float]] = []
    for offset in cast(list[float], fixed["posture_logit_offset_candidates"]):
        candidate, changed = apply_rank_locked_posture_offset(
            probability, posture_logit_offset=float(offset)
        )
        report = _report(labels, candidate, participants)
        summaries.append(
            {
                "posture_logit_offset": float(offset),
                "mean_participant_macro_f1": float(report["primary"]["mean_participant_macro_f1"]),
                "bottom_30_percent_participant_macro_f1": float(
                    report["primary"]["bottom_30_percent_participant_macro_f1"]
                ),
                "hard_change_count": float(changed.sum()),
            }
        )
    summaries.sort(
        key=lambda item: (
            -item["mean_participant_macro_f1"],
            -item["bottom_30_percent_participant_macro_f1"],
            abs(item["posture_logit_offset"]),
            item["posture_logit_offset"],
        )
    )
    return summaries[0]["posture_logit_offset"], summaries


def _select_dual_candidate(
    core: FloatArray,
    dual_expert: FloatArray,
    labels: IntArray,
    participants: StringArray,
    fixed: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    summaries: list[dict[str, float]] = []
    for threshold in cast(list[float], fixed["dual_confidence_thresholds"]):
        for weight in cast(list[float], fixed["dual_blend_weights"]):
            candidate, trigger = apply_confidence_triggered_gravity_residual(
                core,
                dual_expert,
                confidence_threshold=float(threshold),
                blend_weight=float(weight),
            )
            candidate = _rank_lock_candidate(core, candidate)
            report = _report(labels, candidate, participants)
            target = counterfactual_rescue_harm_targets(core, candidate, labels)
            summaries.append(
                {
                    "confidence_threshold": float(threshold),
                    "blend_weight": float(weight),
                    "mean_participant_macro_f1": float(
                        report["primary"]["mean_participant_macro_f1"]
                    ),
                    "bottom_30_percent_participant_macro_f1": float(
                        report["primary"]["bottom_30_percent_participant_macro_f1"]
                    ),
                    "negative_log_likelihood": float(
                        report["calibration"]["negative_log_likelihood"]
                    ),
                    "trigger_fraction": float(trigger.mean()),
                    "rescue_count": float(target.rescue.sum()),
                    "harm_count": float(target.harm.sum()),
                }
            )
    ceiling = max(item["mean_participant_macro_f1"] for item in summaries)
    tolerance = float(fixed["dual_selection_mean_tolerance"])
    eligible = [
        item for item in summaries if item["mean_participant_macro_f1"] >= ceiling - tolerance
    ]
    eligible.sort(
        key=lambda item: (
            -item["bottom_30_percent_participant_macro_f1"],
            item["harm_count"],
            -item["rescue_count"],
            item["negative_log_likelihood"],
            item["trigger_fraction"],
            item["blend_weight"],
            item["confidence_threshold"],
        )
    )
    selected = eligible[0]
    return {
        "confidence_threshold": selected["confidence_threshold"],
        "blend_weight": selected["blend_weight"],
    }, summaries


def _apply_dual_candidate(
    core: FloatArray, dual_expert: FloatArray, selection: dict[str, float]
) -> FloatArray:
    candidate, _ = apply_confidence_triggered_gravity_residual(
        core,
        dual_expert,
        confidence_threshold=float(selection["confidence_threshold"]),
        blend_weight=float(selection["blend_weight"]),
    )
    return _rank_lock_candidate(core, candidate)


def _blank_binary(size: int, *, rescue: bool) -> BinaryJackknifePrediction:
    value = 0.0 if rescue else 1.0
    array = np.full(size, value, dtype=np.float64)
    return BinaryJackknifePrediction(array, np.zeros(size), array, array, 0)


def _meta_binary_predictions(
    features: FloatArray,
    targets: FloatArray,
    participants: StringArray,
    eligible: BoolArray,
    *,
    ridge_penalty: float,
    interval_z: float,
) -> tuple[BinaryJackknifePrediction, FloatArray] | None:
    eligible_participants = np.unique(participants[eligible])
    if eligible_participants.size < 5:
        return None
    mean = np.zeros(features.shape[0], dtype=np.float64)
    deviation = np.zeros(features.shape[0], dtype=np.float64)
    lower = np.zeros(features.shape[0], dtype=np.float64)
    upper = np.ones(features.shape[0], dtype=np.float64)
    support = np.zeros(features.shape[0], dtype=np.float64)
    minimum_models = 10_000
    for held in eligible_participants:
        evaluate = eligible & (participants == held)
        training = eligible & (participants != held)
        if np.unique(participants[training]).size < 4:
            return None
        prediction = participant_jackknife_binary_prediction(
            features[training],
            targets[training],
            participants[training],
            features[evaluate],
            ridge_penalty=ridge_penalty,
            interval_z=interval_z,
        )
        mean[evaluate] = prediction.mean
        deviation[evaluate] = prediction.standard_deviation
        lower[evaluate] = prediction.lower_bound
        upper[evaluate] = prediction.upper_bound
        support[evaluate] = standardized_nearest_support_distance(
            features[training], features[evaluate]
        )
        minimum_models = min(minimum_models, prediction.model_count)
    return (
        BinaryJackknifePrediction(mean, deviation, lower, upper, minimum_models),
        support,
    )


def _outer_binary_predictions(
    training_features: FloatArray,
    evaluation_features: FloatArray,
    targets: FloatArray,
    participants: StringArray,
    eligible: BoolArray,
    *,
    ridge_penalty: float,
    interval_z: float,
) -> tuple[BinaryJackknifePrediction, FloatArray] | None:
    if np.unique(participants[eligible]).size < 4:
        return None
    prediction = participant_jackknife_binary_prediction(
        training_features[eligible],
        targets[eligible],
        participants[eligible],
        evaluation_features,
        ridge_penalty=ridge_penalty,
        interval_z=interval_z,
    )
    support = standardized_nearest_support_distance(
        training_features[eligible], evaluation_features
    )
    return prediction, support


def _disabled_route(
    core: FloatArray,
    rescue: BinaryJackknifePrediction,
    harm: BinaryJackknifePrediction,
    support: FloatArray,
) -> SentinelRoutingResult:
    return SentinelRoutingResult(
        probabilities=core.copy(),
        routed=np.zeros(core.shape[0], dtype=np.bool_),
        eligible=np.zeros(core.shape[0], dtype=np.bool_),
        rescue_lower_bound=rescue.lower_bound,
        harm_upper_bound=harm.upper_bound,
        net_benefit_lower_bound=rescue.lower_bound - 2.0 * harm.upper_bound,
        support_distance=support,
    )


def _select_sentinel(
    core: FloatArray,
    candidate: FloatArray,
    labels: IntArray,
    participants: StringArray,
    rescue: BinaryJackknifePrediction,
    harm: BinaryJackknifePrediction,
    reliability: FloatArray,
    trusted: BoolArray,
    support: FloatArray,
    fixed: dict[str, Any],
    *,
    use_physics: bool,
    use_support: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    applied_reliability = reliability if use_physics else np.ones_like(reliability)
    applied_trusted = trusted if use_physics else np.ones_like(trusted)
    applied_support = support if use_support else np.zeros_like(support)
    for rescue_minimum in cast(list[float], fixed["minimum_rescue_lower_bound_grid"]):
        for harm_maximum in cast(list[float], fixed["maximum_harm_upper_bound_grid"]):
            for net_minimum in cast(list[float], fixed["minimum_net_benefit_grid"]):
                result = apply_rescue_harm_sentinel(
                    core,
                    candidate,
                    rescue,
                    harm,
                    physics_reliability=applied_reliability,
                    physics_trusted=applied_trusted,
                    support_distance=applied_support,
                    minimum_rescue_lower_bound=float(rescue_minimum),
                    maximum_harm_upper_bound=float(harm_maximum),
                    harm_penalty=float(fixed["harm_penalty"]),
                    minimum_net_benefit=float(net_minimum),
                    minimum_physics_reliability=(
                        float(fixed["minimum_physics_reliability"]) if use_physics else 0.0
                    ),
                    maximum_support_distance=(
                        float(fixed["maximum_support_distance"]) if use_support else 1.0
                    ),
                )
                audit = _routing_audit(
                    labels,
                    participants,
                    core,
                    result.probabilities,
                    result.routed,
                    result.routed.astype(np.int64),
                    ("CORE", "DUAL"),
                )
                report = _report(labels, result.probabilities, participants)
                precision = audit["intervention_precision_rescue_over_rescue_plus_harm"]
                harmful = audit["harmful_changed_fraction"]
                passes = (
                    precision is not None
                    and harmful is not None
                    and float(precision) >= float(fixed["training_minimum_precision"])
                    and float(harmful) <= float(fixed["training_maximum_harmful_fraction"])
                )
                summaries.append(
                    {
                        "minimum_rescue_lower_bound": float(rescue_minimum),
                        "maximum_harm_upper_bound": float(harm_maximum),
                        "minimum_net_benefit": float(net_minimum),
                        "training_gate_passed": passes,
                        "mean_participant_macro_f1": float(
                            report["primary"]["mean_participant_macro_f1"]
                        ),
                        "bottom_30_percent_participant_macro_f1": float(
                            report["primary"]["bottom_30_percent_participant_macro_f1"]
                        ),
                        "routed_count": int(audit["routed_count"]),
                        "rescue_count": int(audit["rescue_count"]),
                        "harm_count": int(audit["harm_count"]),
                        "intervention_precision": precision,
                        "harmful_changed_fraction": harmful,
                    }
                )
    eligible = [item for item in summaries if bool(item["training_gate_passed"])]
    if not eligible:
        return {"enabled": False, "reason": "no_training_threshold_passed"}, summaries
    eligible.sort(
        key=lambda item: (
            -float(item["mean_participant_macro_f1"]),
            -float(item["bottom_30_percent_participant_macro_f1"]),
            -float(item["intervention_precision"]),
            int(item["harm_count"]),
            int(item["routed_count"]),
            -float(item["minimum_rescue_lower_bound"]),
            float(item["maximum_harm_upper_bound"]),
            -float(item["minimum_net_benefit"]),
        )
    )
    selected = eligible[0]
    return {
        "enabled": True,
        "minimum_rescue_lower_bound": selected["minimum_rescue_lower_bound"],
        "maximum_harm_upper_bound": selected["maximum_harm_upper_bound"],
        "minimum_net_benefit": selected["minimum_net_benefit"],
    }, summaries


def _apply_selected_sentinel(
    core: FloatArray,
    candidate: FloatArray,
    rescue: BinaryJackknifePrediction,
    harm: BinaryJackknifePrediction,
    reliability: FloatArray,
    trusted: BoolArray,
    support: FloatArray,
    selection: dict[str, Any],
    fixed: dict[str, Any],
    *,
    use_physics: bool,
    use_support: bool,
) -> SentinelRoutingResult:
    if not bool(selection["enabled"]):
        return _disabled_route(core, rescue, harm, support)
    return apply_rescue_harm_sentinel(
        core,
        candidate,
        rescue,
        harm,
        physics_reliability=reliability if use_physics else np.ones_like(reliability),
        physics_trusted=trusted if use_physics else np.ones_like(trusted),
        support_distance=support if use_support else np.zeros_like(support),
        minimum_rescue_lower_bound=float(selection["minimum_rescue_lower_bound"]),
        maximum_harm_upper_bound=float(selection["maximum_harm_upper_bound"]),
        harm_penalty=float(fixed["harm_penalty"]),
        minimum_net_benefit=float(selection["minimum_net_benefit"]),
        minimum_physics_reliability=(
            float(fixed["minimum_physics_reliability"]) if use_physics else 0.0
        ),
        maximum_support_distance=(float(fixed["maximum_support_distance"]) if use_support else 1.0),
    )


def _fit_route_lane(
    *,
    training_core: FloatArray,
    training_candidate: FloatArray,
    training_features: FloatArray,
    training_labels: IntArray,
    training_participants: StringArray,
    training_reliability: FloatArray,
    training_trusted: BoolArray,
    evaluation_core: FloatArray,
    evaluation_candidate: FloatArray,
    evaluation_features: FloatArray,
    evaluation_reliability: FloatArray,
    evaluation_trusted: BoolArray,
    fixed: dict[str, Any],
    use_physics: bool,
    use_support: bool,
) -> tuple[SentinelRoutingResult, dict[str, Any], int]:
    targets = counterfactual_rescue_harm_targets(training_core, training_candidate, training_labels)
    semantic = (
        targets.disagreement
        & (training_core.argmax(axis=1) != 0)
        & (training_candidate.argmax(axis=1) != 0)
    )
    rescue_meta = _meta_binary_predictions(
        training_features,
        targets.rescue,
        training_participants,
        semantic,
        ridge_penalty=float(fixed["binary_ridge_penalty"]),
        interval_z=float(fixed["jackknife_interval_z"]),
    )
    harm_meta = _meta_binary_predictions(
        training_features,
        targets.harm,
        training_participants,
        semantic,
        ridge_penalty=float(fixed["binary_ridge_penalty"]),
        interval_z=float(fixed["jackknife_interval_z"]),
    )
    if rescue_meta is None or harm_meta is None:
        rescue = _blank_binary(evaluation_core.shape[0], rescue=True)
        harm = _blank_binary(evaluation_core.shape[0], rescue=False)
        route = _disabled_route(
            evaluation_core, rescue, harm, np.full(evaluation_core.shape[0], np.inf)
        )
        return (
            route,
            {"enabled": False, "reason": "fewer_than_five_training_responders"},
            int(semantic.sum()),
        )
    rescue_training, support_training = rescue_meta
    harm_training, _ = harm_meta
    selection, summaries = _select_sentinel(
        training_core,
        training_candidate,
        training_labels,
        training_participants,
        rescue_training,
        harm_training,
        training_reliability,
        training_trusted,
        support_training,
        fixed,
        use_physics=use_physics,
        use_support=use_support,
    )
    rescue_outer = _outer_binary_predictions(
        training_features,
        evaluation_features,
        targets.rescue,
        training_participants,
        semantic,
        ridge_penalty=float(fixed["binary_ridge_penalty"]),
        interval_z=float(fixed["jackknife_interval_z"]),
    )
    harm_outer = _outer_binary_predictions(
        training_features,
        evaluation_features,
        targets.harm,
        training_participants,
        semantic,
        ridge_penalty=float(fixed["binary_ridge_penalty"]),
        interval_z=float(fixed["jackknife_interval_z"]),
    )
    if rescue_outer is None or harm_outer is None:
        raise AssertionError("meta and outer binary-head eligibility diverged")
    rescue_evaluation, support_evaluation = rescue_outer
    harm_evaluation, _ = harm_outer
    route = _apply_selected_sentinel(
        evaluation_core,
        evaluation_candidate,
        rescue_evaluation,
        harm_evaluation,
        evaluation_reliability,
        evaluation_trusted,
        support_evaluation,
        selection,
        fixed,
        use_physics=use_physics,
        use_support=use_support,
    )
    selection_record = dict(selection)
    selection_record["candidate_count"] = len(summaries)
    selection_record["training_semantic_disagreement_count"] = int(semantic.sum())
    selection_record["training_semantic_participant_count"] = int(
        np.unique(training_participants[semantic]).size
    )
    return route, selection_record, int(semantic.sum())


def _participant_oracle(
    labels: IntArray,
    participants: StringArray,
    methods: tuple[FloatArray, ...],
    names: tuple[str, ...],
) -> tuple[FloatArray, dict[str, str]]:
    output = np.empty_like(methods[0])
    choices: dict[str, str] = {}
    for participant in np.unique(participants):
        rows = participants == participant
        scores = [
            float(
                _report(labels[rows], item[rows], participants[rows])["primary"][
                    "mean_participant_macro_f1"
                ]
            )
            for item in methods
        ]
        selected = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        output[rows] = methods[selected][rows]
        choices[str(participant)] = names[selected]
    return output, choices


def _window_oracle(labels: IntArray, methods: tuple[FloatArray, ...]) -> FloatArray:
    rows = np.arange(labels.size)
    truth_probability = np.stack([item[rows, labels] for item in methods], axis=0)
    selected = truth_probability.argmax(axis=0)
    output = np.empty_like(methods[0])
    for index, method in enumerate(methods):
        mask = selected == index
        output[mask] = method[mask]
    return output


def _aggregate_reports(
    reports_by_seed: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        reports = [reports_by_seed[seed][method] for seed in _EXPECTED_SEEDS]
        result[method] = {
            "mean_participant_macro_f1": float(
                np.mean([item["primary"]["mean_participant_macro_f1"] for item in reports])
            ),
            "bottom_30_percent_participant_macro_f1": float(
                np.mean(
                    [item["primary"]["bottom_30_percent_participant_macro_f1"] for item in reports]
                )
            ),
            "worst_participant_macro_f1": float(
                np.mean([item["primary"]["worst_participant_macro_f1"] for item in reports])
            ),
            "mobility_recall": float(
                np.mean(
                    [
                        item["window_level_diagnostics"]["per_class_recall"]["mobility"]
                        for item in reports
                    ]
                )
            ),
            "sitting_recall": float(
                np.mean(
                    [
                        item["window_level_diagnostics"]["per_class_recall"]["sitting"]
                        for item in reports
                    ]
                )
            ),
            "standing_recall": float(
                np.mean(
                    [
                        item["window_level_diagnostics"]["per_class_recall"]["standing"]
                        for item in reports
                    ]
                )
            ),
            "negative_log_likelihood": float(
                np.mean([item["calibration"]["negative_log_likelihood"] for item in reports])
            ),
            "multiclass_brier_score": float(
                np.mean([item["calibration"]["multiclass_brier_score"] for item in reports])
            ),
            "mean_evaluated_window_count": float(
                np.mean([item["sample_count"] for item in reports])
            ),
        }
    return result


def _load_v1_predictions(
    *,
    result_path: Path,
    prediction_path: Path,
    labels: IntArray,
    participants: StringArray,
    windows: StringArray,
) -> tuple[dict[int, FloatArray], dict[str, Any]]:
    result = _read_hashed_record(result_path)
    if result.get("status") != "complete_reused_source_development_not_independent":
        raise ValueError("frozen HERA v1 result status changed")
    with np.load(prediction_path, allow_pickle=False) as payload:
        np.testing.assert_array_equal(payload["labels"], labels)
        np.testing.assert_array_equal(payload["participant_ids"], participants)
        np.testing.assert_array_equal(payload["window_ids"], windows)
        probabilities = {
            seed: np.asarray(
                payload[f"seed_{seed}__hera_ctgr_strict_probabilities"], dtype=np.float64
            )
            for seed in _EXPECTED_SEEDS
        }
    return probabilities, result


def run_hera_v2_retrospective(
    *, config_path: Path, output_directory: Path, code_commit: str
) -> dict[str, Any]:
    """Execute the one locked HERA-CTGR v2 reused-source screen."""

    if output_directory.exists():
        raise FileExistsError(f"HERA-CTGR v2 output already exists: {output_directory}")
    config = load_hera_v2_retrospective_config(config_path)
    inputs = cast(dict[str, Any], config["inputs"])
    fixed = cast(dict[str, Any], config["fixed_method"])
    freeze = cast(dict[str, Any], config["method_freeze"])
    source_manifest_path = _validated_path(inputs["source_manifest"], name="source manifest")
    dataset_manifest_path = _validated_path(inputs["dataset_manifest"], name="dataset manifest")
    raw_source_path = _validated_path(inputs["raw_source"], name="raw source")
    ctgr_config_path = _validated_path(inputs["ctgr_config"], name="CTGR config")
    selection_path = _validated_path(inputs["ctgr_selection"], name="CTGR selection")
    v1_result_path = _validated_path(inputs["hera_v1_result"], name="HERA v1 result")
    v1_prediction_path = _validated_path(inputs["hera_v1_predictions"], name="HERA v1 predictions")
    _validated_path(freeze["method_config"], name="HERA v2 method config")
    _validated_path(freeze["method_implementation"], name="HERA v2 implementation")

    ctgr_config = _load_config(ctgr_config_path)
    selection = _read_hashed_record(selection_path)
    if selection.get("status") != "selection_frozen_before_outer_evaluation":
        raise PermissionError("HERA-CTGR v2 requires the frozen CTGR selection")
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path, raw_csv_path=raw_source_path
    )
    if tuple(sorted(set(participants.tolist()), key=int)) != _EXPECTED_PARTICIPANTS:
        raise PermissionError("HERA-CTGR v2 participant set changed")
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_source_path,
        expected_participants=participants,
        expected_windows=windows,
    )
    sampling_rate = float(ctgr_config["input_contract"]["sampling_rate_hz"])
    if sampling_rate != 50.0:
        raise ValueError("HERA-CTGR v2 source sampling rate changed")
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=sampling_rate)
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=sampling_rate)["denoised"]
    dual = extract_dual_frame_posture_features(denoised, gravity, sampling_rate_hz=sampling_rate)
    kinematic = extract_gravity_kinematic_context(signals, gravity, sampling_rate_hz=sampling_rate)
    candidates = {str(item["id"]): item for item in _candidates(ctgr_config)}
    selected_by_fold = _selected_by_fold(selection)
    top_by_fold = _top_ranked_candidates(selection, candidates)
    outer_folds = cast(list[dict[str, Any]], manifest["source_nested_cv"])
    if len(outer_folds) != 5:
        raise ValueError("HERA-CTGR v2 requires five outer folds")
    n_jobs = int(ctgr_config["n_jobs"])
    v1_predictions, v1_result = _load_v1_predictions(
        result_path=v1_result_path,
        prediction_path=v1_prediction_path,
        labels=labels,
        participants=participants,
        windows=windows,
    )

    prediction_payload: dict[str, NDArray[Any]] = {
        "labels": labels,
        "participant_ids": participants,
        "window_ids": windows,
        "dual_frame_compact_context": dual.compact_context,
    }
    reports_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    audits_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    seed_records: list[dict[str, Any]] = []
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
        probabilities = {
            name: np.full((labels.size, 3), np.nan, dtype=np.float64) for name in _METHODS
        }
        one_query_mask = np.ones(labels.size, dtype=np.bool_)
        routed_masks = {
            name: np.zeros(labels.size, dtype=np.bool_)
            for name in ("rescue_harm_logits_only", "rescue_harm_physics", "hera_ctgr_v2_full")
        }
        fold_records: list[dict[str, Any]] = []
        for outer_index, outer in enumerate(outer_folds):
            fold_id = str(outer["outer_fold_id"])
            top_candidates = top_by_fold[fold_id]
            if str(top_candidates[0]["id"]) != selected_by_fold[fold_id]:
                raise ValueError("HERA-CTGR v2 selected CTGR candidate changed")
            inner, training = _inner_paths(
                outer=outer,
                top_candidates=top_candidates,
                seed=seed,
                n_jobs=n_jobs,
                rmrp=rmrp,
                views=views,
                dual_features=dual.features,
                labels=labels,
                participants=participants,
            )
            evaluation = ~training
            outer_stack, outer_dual_expert, top_ids = _outer_paths(
                top_candidates=top_candidates,
                preserved=preserved,
                evaluation=evaluation,
                training=training,
                outer_index=outer_index,
                seed=seed,
                n_jobs=n_jobs,
                views=views,
                dual_features=dual.features,
                labels=labels,
                participants=participants,
            )
            inner_stack = np.stack(
                [inner[f"candidate_{index}"][training] for index in range(3)], axis=0
            )
            _inner_equal, inner_weighted, inner_disagreement = _marginalize(inner_stack, fixed)
            outer_equal, outer_weighted, outer_disagreement = _marginalize(outer_stack, fixed)
            temperature, temperature_candidates = _select_temperature(
                inner_weighted, labels[training], participants[training], fixed
            )
            inner_temperature = temperature_scale_probabilities(
                inner_weighted, temperature=temperature
            )
            outer_temperature = temperature_scale_probabilities(
                outer_weighted, temperature=temperature
            )
            offset, offset_candidates = _select_offset(
                inner_temperature, labels[training], participants[training], fixed
            )
            inner_core, _ = apply_rank_locked_posture_offset(
                inner_temperature, posture_logit_offset=offset
            )
            outer_core, _ = apply_rank_locked_posture_offset(
                outer_temperature, posture_logit_offset=offset
            )
            dual_selection, dual_candidates = _select_dual_candidate(
                inner_core,
                inner["dual_expert"][training],
                labels[training],
                participants[training],
                fixed,
            )
            inner_dual = _apply_dual_candidate(
                inner_core, inner["dual_expert"][training], dual_selection
            )
            outer_dual = _apply_dual_candidate(outer_core, outer_dual_expert, dual_selection)

            _cage_context, cage_reliability, channel_scale = _outer_context(
                signals, gravity, training, sampling_rate_hz=sampling_rate
            )
            physics_reference = fit_physics_reference(
                _slice_context(kinematic, training),
                veto_quantile=float(fixed["physics_veto_quantile"]),
                minimum_valid_pair_fraction=float(fixed["physics_minimum_valid_pair_fraction"]),
            )
            physics_training = apply_physics_reference(
                _slice_context(kinematic, training), physics_reference
            )
            physics_evaluation = apply_physics_reference(
                _slice_context(kinematic, evaluation), physics_reference
            )
            reliability_training = np.minimum(
                cage_reliability[training], physics_training.reliability
            )
            reliability_evaluation = np.minimum(
                cage_reliability[evaluation], physics_evaluation.reliability
            )
            logits_training, logits_names = build_rescue_harm_features(
                inner_core,
                inner_dual,
                candidate_disagreement=inner_disagreement,
                physics_reliability=reliability_training,
                dual_frame_context=np.empty((int(training.sum()), 0), dtype=np.float64),
            )
            logits_evaluation, _ = build_rescue_harm_features(
                outer_core,
                outer_dual,
                candidate_disagreement=outer_disagreement,
                physics_reliability=reliability_evaluation,
                dual_frame_context=np.empty((int(evaluation.sum()), 0), dtype=np.float64),
            )
            physics_features_training, physics_names = build_rescue_harm_features(
                inner_core,
                inner_dual,
                candidate_disagreement=inner_disagreement,
                physics_reliability=reliability_training,
                dual_frame_context=dual.compact_context[training],
            )
            physics_features_evaluation, _ = build_rescue_harm_features(
                outer_core,
                outer_dual,
                candidate_disagreement=outer_disagreement,
                physics_reliability=reliability_evaluation,
                dual_frame_context=dual.compact_context[evaluation],
            )
            logits_route, logits_selection, _ = _fit_route_lane(
                training_core=inner_core,
                training_candidate=inner_dual,
                training_features=logits_training,
                training_labels=labels[training],
                training_participants=participants[training],
                training_reliability=reliability_training,
                training_trusted=physics_training.trusted,
                evaluation_core=outer_core,
                evaluation_candidate=outer_dual,
                evaluation_features=logits_evaluation,
                evaluation_reliability=reliability_evaluation,
                evaluation_trusted=physics_evaluation.trusted,
                fixed=fixed,
                use_physics=False,
                use_support=False,
            )
            physics_route, physics_selection, _ = _fit_route_lane(
                training_core=inner_core,
                training_candidate=inner_dual,
                training_features=physics_features_training,
                training_labels=labels[training],
                training_participants=participants[training],
                training_reliability=reliability_training,
                training_trusted=physics_training.trusted,
                evaluation_core=outer_core,
                evaluation_candidate=outer_dual,
                evaluation_features=physics_features_evaluation,
                evaluation_reliability=reliability_evaluation,
                evaluation_trusted=physics_evaluation.trusted,
                fixed=fixed,
                use_physics=True,
                use_support=False,
            )
            full_route, full_selection, _ = _fit_route_lane(
                training_core=inner_core,
                training_candidate=inner_dual,
                training_features=physics_features_training,
                training_labels=labels[training],
                training_participants=participants[training],
                training_reliability=reliability_training,
                training_trusted=physics_training.trusted,
                evaluation_core=outer_core,
                evaluation_candidate=outer_dual,
                evaluation_features=physics_features_evaluation,
                evaluation_reliability=reliability_evaluation,
                evaluation_trusted=physics_evaluation.trusted,
                fixed=fixed,
                use_physics=True,
                use_support=True,
            )

            query_probability = outer_core.copy()
            query_evaluation = np.ones(int(evaluation.sum()), dtype=np.bool_)
            query_records: dict[str, Any] = {}
            evaluation_participants = participants[evaluation]
            evaluation_windows = windows[evaluation]
            evaluation_labels = labels[evaluation]
            for participant in np.unique(evaluation_participants):
                rows = evaluation_participants == participant
                query = apply_one_query_state_selection(
                    outer_core[rows],
                    outer_dual[rows],
                    reliability_evaluation[rows],
                    evaluation_windows[rows],
                    evaluation_labels[rows],
                    minimum_stationary_mass=float(fixed["one_query_minimum_stationary_mass"]),
                )
                query_probability[rows] = query.probabilities
                query_evaluation[rows] = query.evaluation_mask
                query_records[str(participant)] = {
                    "queried_window_id": (
                        None
                        if query.queried_index is None
                        else str(evaluation_windows[rows][query.queried_index])
                    ),
                    "selected_state": query.selected_state,
                    "query_excluded_from_evaluation": query.queried_index is not None,
                }

            fold_probability = {
                "base_uncalibrated": preserved["base"][evaluation],
                "frozen_ctgr": preserved["ctgr"][evaluation],
                "frozen_hera_ctgr_v1_strict": v1_predictions[seed][evaluation],
                "equal_top3_ctgr": outer_equal,
                "stability_weighted_top3_ctgr": outer_weighted,
                "temperature_only_core": outer_temperature,
                "decision_separated_core": outer_core,
                "dual_frame_global_candidate": outer_dual,
                "rescue_harm_logits_only": logits_route.probabilities,
                "rescue_harm_physics": physics_route.probabilities,
                "hera_ctgr_v2_full": full_route.probabilities,
                "one_query_matched_core": outer_core,
                "one_query_personalization": query_probability,
            }
            for name, value in fold_probability.items():
                probabilities[name][evaluation] = value
            one_query_mask[evaluation] = query_evaluation
            routed_masks["rescue_harm_logits_only"][evaluation] = logits_route.routed
            routed_masks["rescue_harm_physics"][evaluation] = physics_route.routed
            routed_masks["hera_ctgr_v2_full"][evaluation] = full_route.routed
            fold_records.append(
                {
                    "outer_fold_id": fold_id,
                    "outer_test_subjects": list(outer["outer_test_subjects"]),
                    "top_three_candidate_ids": list(top_ids),
                    "temperature": temperature,
                    "temperature_candidate_count": len(temperature_candidates),
                    "posture_logit_offset": offset,
                    "posture_offset_candidate_count": len(offset_candidates),
                    "dual_candidate_selection": dual_selection,
                    "dual_candidate_count": len(dual_candidates),
                    "logits_sentinel_selection": logits_selection,
                    "physics_sentinel_selection": physics_selection,
                    "full_sentinel_selection": full_selection,
                    "logits_feature_count": len(logits_names),
                    "physics_feature_count": len(physics_names),
                    "mean_candidate_disagreement_training": float(inner_disagreement.mean()),
                    "mean_candidate_disagreement_evaluation": float(outer_disagreement.mean()),
                    "outer_training_channel_scale": channel_scale.tolist(),
                    "physics_reference_threshold": physics_reference.normalized_residual_p90_threshold,
                    "query_records": query_records,
                    "outer_labels_used_for_zero_shot_selection_or_routing": False,
                    "outer_labels_used_for_one_query_personalization": True,
                }
            )

        if any(not np.isfinite(probabilities[name]).all() for name in _METHODS[:-2]):
            raise ValueError(f"HERA-CTGR v2 seed {seed} predictions are incomplete")
        oracle_methods = (
            probabilities["base_uncalibrated"],
            probabilities["frozen_ctgr"],
            probabilities["frozen_hera_ctgr_v1_strict"],
            probabilities["decision_separated_core"],
            probabilities["dual_frame_global_candidate"],
        )
        participant_oracle, oracle_choices = _participant_oracle(
            labels,
            participants,
            oracle_methods,
            ("BASE", "CTGR", "HERA_V1", "V2_CORE", "DUAL"),
        )
        window_oracle = _window_oracle(labels, oracle_methods)
        probabilities["participant_oracle_diagnostic"] = participant_oracle
        probabilities["window_oracle_diagnostic"] = window_oracle
        reports: dict[str, dict[str, Any]] = {}
        for name, probability in probabilities.items():
            if name in {"one_query_matched_core", "one_query_personalization"}:
                reports[name] = _report(
                    labels[one_query_mask],
                    probability[one_query_mask],
                    participants[one_query_mask],
                )
            else:
                reports[name] = _report(labels, probability, participants)
        expected = cast(dict[str, float], preserved_record["expected_mean_participant_macro_f1"])
        if not np.isclose(
            reports["base_uncalibrated"]["primary"]["mean_participant_macro_f1"],
            expected["base_uncalibrated"],
            atol=1e-12,
            rtol=0.0,
        ) or not np.isclose(
            reports["frozen_ctgr"]["primary"]["mean_participant_macro_f1"],
            expected["frozen_ctgr"],
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("HERA-CTGR v2 preserved CTGR replay changed")
        v1_seed_record = next(
            item
            for item in cast(list[dict[str, Any]], v1_result["seeds"])
            if int(item["seed"]) == seed
        )
        expected_v1 = float(
            v1_seed_record["reports"]["hera_ctgr_strict"]["primary"]["mean_participant_macro_f1"]
        )
        if not np.isclose(
            reports["frozen_hera_ctgr_v1_strict"]["primary"]["mean_participant_macro_f1"],
            expected_v1,
            atol=1e-12,
            rtol=0.0,
        ):
            raise ValueError("HERA-CTGR v2 preserved HERA v1 replay changed")

        audits: dict[str, dict[str, Any]] = {}
        for name in ("rescue_harm_logits_only", "rescue_harm_physics", "hera_ctgr_v2_full"):
            routed = routed_masks[name]
            audits[name] = _routing_audit(
                labels,
                participants,
                probabilities["decision_separated_core"],
                probabilities[name],
                routed,
                routed.astype(np.int64),
                ("CORE", "DUAL"),
            )
            mobility_change = (probabilities["decision_separated_core"].argmax(axis=1) == 0) != (
                probabilities[name].argmax(axis=1) == 0
            )
            audits[name]["mobility_membership_change_count"] = int(mobility_change.sum())
        reports_by_seed[seed] = reports
        audits_by_seed[seed] = audits
        seed_records.append(
            {
                "seed": seed,
                "preserved_ctgr_record": preserved_record,
                "folds": fold_records,
                "reports": reports,
                "routing_audits_against_decision_core": audits,
                "participant_oracle_choices": oracle_choices,
                "one_query_count": int((~one_query_mask).sum()),
                "outer_labels_used_for_zero_shot_training_selection_or_routing": False,
                "one_query_labels_used_only_after_label_free_query_selection": True,
                "participants_11_through_20_loaded": False,
                "daghar_loaded": False,
            }
        )
        for name, probability in probabilities.items():
            prediction_payload[f"seed_{seed}__{name}_probabilities"] = probability
        for name, routed in routed_masks.items():
            prediction_payload[f"seed_{seed}__{name}_routed"] = routed
        prediction_payload[f"seed_{seed}__one_query_evaluation_mask"] = one_query_mask

    aggregate = _aggregate_reports(reports_by_seed)
    comparison_v1 = _bootstrap_intervals(
        reports_by_seed,
        method="hera_ctgr_v2_full",
        comparator="frozen_hera_ctgr_v1_strict",
        seed=20260905,
    )
    comparison_ctgr = _bootstrap_intervals(
        reports_by_seed,
        method="hera_ctgr_v2_full",
        comparator="frozen_ctgr",
        seed=20260906,
    )
    full = aggregate["hera_ctgr_v2_full"]
    v1 = aggregate["frozen_hera_ctgr_v1_strict"]
    ctgr = aggregate["frozen_ctgr"]
    delta_v1 = {key: full[key] - v1[key] for key in full if key != "mean_evaluated_window_count"}
    delta_ctgr = {
        key: full[key] - ctgr[key] for key in full if key != "mean_evaluated_window_count"
    }
    full_audits = [audits_by_seed[seed]["hera_ctgr_v2_full"] for seed in _EXPECTED_SEEDS]
    rescues = sum(int(item["rescue_count"]) for item in full_audits)
    harms = sum(int(item["harm_count"]) for item in full_audits)
    changed = sum(int(item["changed_count"]) for item in full_audits)
    routing = {
        "opportunity_count": labels.size * len(_EXPECTED_SEEDS),
        "routed_count": sum(int(item["routed_count"]) for item in full_audits),
        "hard_label_change_count": changed,
        "rescue_count": rescues,
        "harm_count": harms,
        "net_rescues": rescues - harms,
        "intervention_precision": None if rescues + harms == 0 else rescues / (rescues + harms),
        "harmful_changed_fraction": None if changed == 0 else harms / changed,
        "mobility_membership_change_count": sum(
            int(item["mobility_membership_change_count"]) for item in full_audits
        ),
    }
    posture_difference = min(delta_v1["sitting_recall"], delta_v1["standing_recall"])
    mean_ci = comparison_v1["mean_difference_familywise_97_5_percent_bootstrap_interval"]
    bottom_ci = comparison_ctgr[
        "bottom_30_percent_difference_familywise_97_5_percent_bootstrap_interval"
    ]
    advancement_gate = {
        "mean_gain_over_hera_v1_at_least_0_010": {
            "observed": delta_v1["mean_participant_macro_f1"],
            "passed": delta_v1["mean_participant_macro_f1"] >= 0.010,
        },
        "bottom_30_gain_over_ctgr_at_least_0_015": {
            "observed": delta_ctgr["bottom_30_percent_participant_macro_f1"],
            "passed": delta_ctgr["bottom_30_percent_participant_macro_f1"] >= 0.015,
        },
        "intervention_precision_at_least_0_80": {
            "observed": routing["intervention_precision"],
            "passed": routing["intervention_precision"] is not None
            and float(routing["intervention_precision"]) >= 0.80,
        },
        "harmful_changed_fraction_at_most_0_15": {
            "observed": routing["harmful_changed_fraction"],
            "passed": routing["harmful_changed_fraction"] is not None
            and float(routing["harmful_changed_fraction"]) <= 0.15,
        },
        "zero_mobility_membership_changes": {
            "observed": routing["mobility_membership_change_count"],
            "passed": routing["mobility_membership_change_count"] == 0,
        },
        "worst_participant_noninferiority_minus_0_010": {
            "observed": delta_v1["worst_participant_macro_f1"],
            "passed": delta_v1["worst_participant_macro_f1"] >= -0.010,
        },
        "posture_recall_noninferiority_minus_0_010": {
            "observed": posture_difference,
            "passed": posture_difference >= -0.010,
        },
        "negative_log_likelihood_not_worse": {
            "observed": delta_v1["negative_log_likelihood"],
            "passed": delta_v1["negative_log_likelihood"] <= 0.0,
        },
        "multiclass_brier_not_worse": {
            "observed": delta_v1["multiclass_brier_score"],
            "passed": delta_v1["multiclass_brier_score"] <= 0.0,
        },
        "participant_interval_lower_bound_above_zero": {
            "observed": mean_ci[0],
            "passed": float(mean_ci[0]) > 0.0,
        },
    }
    advancement_gate["overall_passed"] = all(
        bool(item["passed"]) for item in advancement_gate.values() if isinstance(item, dict)
    )
    breakthrough_gate = {
        "mean_gain_over_ctgr_at_least_0_020": delta_ctgr["mean_participant_macro_f1"] >= 0.020,
        "familywise_mean_interval_over_hera_v1_lower_bound_above_zero": float(mean_ci[0]) > 0.0,
        "familywise_bottom_interval_over_ctgr_lower_bound_above_zero": float(bottom_ci[0]) > 0.0,
    }
    breakthrough_gate["overall_passed"] = all(breakthrough_gate.values())
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "hera_ctgr_v2_fully_nested_retrospective_source_result",
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
        "hera_v1_result": {
            "path": v1_result_path.as_posix(),
            "sha256": sha256_file(v1_result_path),
            "record_sha256": v1_result["record_sha256"],
        },
        "participant_count": int(np.unique(participants).size),
        "window_count": int(labels.size),
        "seeds": seed_records,
        "aggregate_five_seed_means": aggregate,
        "hera_ctgr_v2_full_minus_hera_v1_strict": delta_v1,
        "hera_ctgr_v2_full_minus_frozen_ctgr": delta_ctgr,
        "hera_ctgr_v2_full_routing_against_decision_core": routing,
        "descriptive_participant_bootstrap_against_hera_v1": comparison_v1,
        "descriptive_participant_bootstrap_against_ctgr": comparison_ctgr,
        "advancement_gate": advancement_gate,
        "breakthrough_gate": breakthrough_gate,
        "scientific_zero_shot_methods": list(_SCIENTIFIC_ZERO_SHOT),
        "one_query_methods": ["one_query_matched_core", "one_query_personalization"],
        "diagnostic_only_methods": list(_DIAGNOSTIC_METHODS),
        "bout_lane_evaluated": False,
        "bout_lane_reason": "released_source_has_no_authentic_session_trial_or_timestamp_order",
        "outer_labels_used_for_zero_shot_training_selection_or_routing": False,
        "one_query_labels_used_after_label_free_query_selection": True,
        "oracle_labels_used_for_hindsight_diagnostics_only": True,
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
    result = run_hera_v2_retrospective(
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "aggregate_five_seed_means": result["aggregate_five_seed_means"],
                "advancement_gate": result["advancement_gate"],
                "breakthrough_gate": result["breakthrough_gate"],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
