"""Fully nested retrospective evaluation of frozen HERA-CTGR on source participants.

This is deliberately reused-source hypothesis-generation evidence.  It cannot
load InclusiveHAR participants 11--20 or DAGHAR and cannot support an
independent, confirmatory, or state-of-the-art claim.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.experiments.cage_har import _load_config as _load_cage_config
from inclusive_shift_har.experiments.cage_har import _report, _routing_audit, _write_result
from inclusive_shift_har.experiments.cage_har_retrospective import (
    _load_preserved_seed,
    _outer_context,
    _validated_path,
    evaluate_cage_outer_arrays,
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
    apply_confidence_triggered_gravity_residual,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates as _ctgr_candidates,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _load_config as _load_ctgr_config,
)
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _lower_fraction_mean,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _materialize_source,
    _read_hashed_record,
)
from inclusive_shift_har.manifests.canonical import sha256_file
from inclusive_shift_har.models.cage_har import participant_jackknife_advantage_prediction
from inclusive_shift_har.models.hera_ctgr import (
    GravityKinematicContext,
    HeraRoutingResult,
    apply_physics_reference,
    apply_physics_veto,
    apply_posture_conditional_calibration,
    apply_tri_state_responder_controller,
    average_probability_candidates,
    build_responder_signatures,
    extract_gravity_kinematic_context,
    fit_physics_reference,
    standardized_nearest_support_distance,
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
    "global_trust_blend",
    "ctgr_aggregate_calibrated",
    "ctgr_physics_veto",
    "ctgr_candidate_ensemble_top3",
    "hera_ctgr_strict",
    "hera_ctgr_context_mean",
    "hera_ctgr_context_safe",
    "hera_ctgr_full",
    "matched_random_state_control",
    "label_informed_three_way_oracle_diagnostic",
)
_SCIENTIFIC_METHODS = _METHODS[:-2]
_ROUTED_METHODS = (
    "hera_ctgr_context_mean",
    "hera_ctgr_context_safe",
    "hera_ctgr_full",
    "matched_random_state_control",
)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def load_hera_retrospective_config(path: Path) -> dict[str, Any]:
    """Load and fail closed on any change to the retrospective contract."""

    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("status") != "locked_before_fully_nested_retrospective_source_run":
        raise ValueError("HERA-CTGR retrospective protocol is not locked")
    if tuple(int(item) for item in config.get("fixed_seeds", [])) != _EXPECTED_SEEDS:
        raise ValueError("HERA-CTGR seed set changed")
    if tuple(str(item) for item in config.get("participant_ids", [])) != _EXPECTED_PARTICIPANTS:
        raise ValueError("HERA-CTGR retrospective participant set changed")
    if (
        int(config.get("outer_fold_count", -1)) != 5
        or int(config.get("inner_fold_count_per_outer", -1)) != 4
    ):
        raise ValueError("HERA-CTGR nested fold counts changed")
    if tuple(str(item) for item in config.get("methods", [])) != _METHODS:
        raise ValueError("HERA-CTGR retrospective method matrix changed")

    freeze = _mapping(config.get("method_freeze"), name="method freeze")
    if (
        freeze.get("tag") != "hera-ctgr-implementation-v1"
        or freeze.get("commit") != "b274e54992bbfe18aa43243059e3c977b3f7ee5e"
        or freeze.get("parameters_may_change_after_run") is not False
    ):
        raise ValueError("HERA-CTGR implementation freeze changed")
    fixed = _mapping(config.get("fixed_method"), name="fixed method")
    expected = {
        "candidate_top_k": 3,
        "candidate_aggregation": "equal_probability_mean",
        "candidate_source": "first_three_entries_of_each_frozen_ctgr_candidate_ranking",
        "physics_veto_quantile": 0.99,
        "physics_minimum_valid_pair_fraction": 0.98,
        "calibration_temperatures": [0.8, 1.0, 1.25],
        "calibration_posture_logit_offsets": [-0.15, 0.0, 0.15],
        "calibration_mean_tie_tolerance": 0.005,
        "responder_low_confidence_threshold": 0.6,
        "responder_ridge_penalty": 25.0,
        "responder_mean_lower_bound_z": 0.0,
        "responder_safe_lower_bound_z": 1.0,
        "responder_minimum_advantage": 0.005,
        "responder_maximum_support_distance": 3.0,
        "uncertain_default_state": "PULSE",
        "matched_random_seed_offset": 20260904,
    }
    if fixed != expected:
        raise ValueError("HERA-CTGR fixed method settings changed")
    nesting = _mapping(config.get("nesting_contract"), name="nesting contract")
    required_true = (
        "frozen_ctgr_candidate_map_reused",
        "top_three_frozen_ranked_candidates_reused_without_outcome_reweighting",
        "base_and_expert_inner_oof_regenerated_inside_each_outer_training_partition",
        "broad_path_fit_on_inner_oof_only",
        "aggregate_calibration_fit_on_inner_oof_only",
        "responder_fit_at_participant_level_inside_outer_training_partition",
        "full_calibrator_fit_on_leave_one_training_participant_controller_predictions",
        "preserved_outer_base_ctgr_and_selected_expert_used_for_evaluation",
        "participant_is_inference_unit",
    )
    if any(nesting.get(key) is not True for key in required_true):
        raise ValueError("HERA-CTGR nesting guarantee changed")
    if nesting.get("outer_labels_used_for_training_selection_calibration_or_routing") is not False:
        raise ValueError("outer labels must remain unavailable to HERA-CTGR fitting")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    forbidden_false = (
        "participants_11_through_20_may_be_loaded",
        "daghar_may_be_loaded",
        "independent_validation_claim_allowed",
        "confirmatory_claim_allowed",
        "state_of_the_art_claim_allowed",
    )
    if any(policy.get(key) is not False for key in forbidden_false):
        raise ValueError("HERA-CTGR retrospective claim boundary changed")
    return config


def _top_ranked_candidates(
    selection: dict[str, Any],
    candidates: dict[str, dict[str, Any]],
) -> dict[str, tuple[dict[str, Any], ...]]:
    result: dict[str, tuple[dict[str, Any], ...]] = {}
    for fold in cast(list[dict[str, Any]], selection["folds"]):
        fold_id = str(fold["outer_fold_id"])
        ranking = cast(list[dict[str, Any]], fold["candidate_ranking"])
        ids = tuple(str(item["candidate_id"]) for item in ranking[:3])
        eligible = {str(item) for item in cast(list[Any], fold["eligible_within_mean_tolerance"])}
        if len(ids) != 3 or ids[0] != str(fold["selected_candidate_id"]):
            raise ValueError(f"frozen top-three ranking changed for {fold_id}")
        if not set(ids).issubset(eligible):
            raise ValueError(f"top-three ranking escaped the frozen eligible set for {fold_id}")
        try:
            result[fold_id] = tuple(candidates[item] for item in ids)
        except KeyError as exc:
            raise ValueError(f"frozen CTGR candidate is unavailable for {fold_id}") from exc
    if len(result) != 5:
        raise ValueError("frozen CTGR selection must contain five folds")
    return result


def _apply_candidate(
    base_probability: FloatArray,
    expert_probability: FloatArray,
    candidate: dict[str, Any],
) -> FloatArray:
    if str(candidate["id"]) == "base_no_route":
        return np.asarray(base_probability, dtype=np.float64).copy()
    probability, _ = apply_confidence_triggered_gravity_residual(
        base_probability,
        expert_probability,
        confidence_threshold=float(candidate["confidence_threshold"]),
        blend_weight=float(candidate["blend_weight"]),
    )
    return probability


def _slice_context(context: GravityKinematicContext, mask: BoolArray) -> GravityKinematicContext:
    return GravityKinematicContext(
        features=np.asarray(context.features[mask], dtype=np.float64),
        normalized_residual_p90=np.asarray(context.normalized_residual_p90[mask], dtype=np.float64),
        valid_pair_fraction=np.asarray(context.valid_pair_fraction[mask], dtype=np.float64),
        feature_names=context.feature_names,
    )


def _inner_oof_paths(
    *,
    outer: dict[str, Any],
    top_candidates: tuple[dict[str, Any], ...],
    seed: int,
    n_jobs: int,
    rmrp: FloatArray,
    views: dict[str, FloatArray],
    labels: IntArray,
    participants: StringArray,
) -> tuple[dict[str, FloatArray], BoolArray]:
    """Regenerate participant-exclusive inner predictions for all HERA paths."""

    selected = top_candidates[0]
    outer_test = [str(item) for item in cast(list[Any], outer["outer_test_subjects"])]
    outer_training = ~_mask_for_subjects(participants, outer_test)
    arrays = {
        name: np.full((labels.size, 3), np.nan, dtype=np.float64)
        for name in ("base", "selected_expert", "pulse", "top3")
    }
    covered = np.zeros(labels.size, dtype=np.bool_)
    inner_folds = cast(list[dict[str, Any]], outer["inner_folds"])
    if len(inner_folds) != 4:
        raise ValueError("HERA-CTGR requires four inner folds")
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
            raise PermissionError("HERA-CTGR inner/outer participant leakage")
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training,
            seed=seed + inner_index,
            n_jobs=n_jobs,
        )
        base_probability = np.asarray(base_model.predict_proba(rmrp[validation]), dtype=np.float64)
        expert_cache: dict[tuple[str, str], FloatArray] = {}
        candidate_paths: list[FloatArray] = []
        for candidate in top_candidates:
            view = str(candidate["expert_view"])
            estimator = str(candidate["posture_estimator"])
            if view not in _VIEW_ORDER or estimator not in _ESTIMATOR_ORDER:
                raise ValueError("HERA-CTGR top-three candidate lacks a usable gravity expert")
            key = (view, estimator)
            if key not in expert_cache:
                view_index = _VIEW_ORDER.index(view)
                estimator_index = _ESTIMATOR_ORDER.index(estimator)
                model = _fit_expert(
                    estimator,
                    views[view],
                    labels,
                    participants,
                    training,
                    seed=seed + 101 * view_index + 17 * estimator_index + inner_index,
                    n_jobs=n_jobs,
                )
                expert_cache[key] = _gravity_probability(
                    base_probability, model, views[view][validation]
                )
            candidate_paths.append(_apply_candidate(base_probability, expert_cache[key], candidate))
        selected_key = (str(selected["expert_view"]), str(selected["posture_estimator"]))
        arrays["base"][validation] = base_probability
        arrays["selected_expert"][validation] = expert_cache[selected_key]
        arrays["pulse"][validation] = candidate_paths[0]
        arrays["top3"][validation] = average_probability_candidates(
            np.stack(candidate_paths, axis=0)
        )
        covered[validation] = True
    if not np.array_equal(covered, outer_training):
        raise ValueError("HERA-CTGR inner folds do not cover outer training exactly once")
    if any(not np.isfinite(value[outer_training]).all() for value in arrays.values()):
        raise ValueError("HERA-CTGR inner OOF paths are incomplete")
    return arrays, outer_training


def _outer_top3_path(
    *,
    top_candidates: tuple[dict[str, Any], ...],
    preserved: dict[str, FloatArray],
    evaluation: BoolArray,
    training: BoolArray,
    outer_index: int,
    seed: int,
    n_jobs: int,
    views: dict[str, FloatArray],
    labels: IntArray,
    participants: StringArray,
) -> tuple[FloatArray, tuple[str, ...]]:
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
        raise ValueError("HERA-CTGR no longer exactly replays the frozen selected CTGR path")
    return (
        average_probability_candidates(np.stack(paths, axis=0)),
        tuple(str(item["id"]) for item in top_candidates),
    )


def _select_posture_calibration(
    probabilities: FloatArray,
    labels: IntArray,
    participants: StringArray,
    fixed: dict[str, Any],
    *,
    base_probability: FloatArray | None = None,
    restore_mask: BoolArray | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """Select the frozen conditional posture calibrator on training rows only."""

    summaries: list[dict[str, float]] = []
    for temperature in cast(list[float], fixed["calibration_temperatures"]):
        for offset in cast(list[float], fixed["calibration_posture_logit_offsets"]):
            candidate = apply_posture_conditional_calibration(
                probabilities,
                temperature=float(temperature),
                posture_logit_offset=float(offset),
            )
            if restore_mask is not None:
                if base_probability is None or restore_mask.shape != (candidate.shape[0],):
                    raise ValueError("calibration restoration inputs do not align")
                candidate[restore_mask] = base_probability[restore_mask]
            report = _report(labels, candidate, participants)
            summaries.append(
                {
                    "temperature": float(temperature),
                    "posture_logit_offset": float(offset),
                    "mean_participant_macro_f1": float(
                        report["primary"]["mean_participant_macro_f1"]
                    ),
                    "bottom_30_percent_participant_macro_f1": float(
                        report["primary"]["bottom_30_percent_participant_macro_f1"]
                    ),
                    "negative_log_likelihood": float(
                        report["calibration"]["negative_log_likelihood"]
                    ),
                }
            )
    ceiling = max(item["mean_participant_macro_f1"] for item in summaries)
    tolerance = float(fixed["calibration_mean_tie_tolerance"])
    eligible = [
        item for item in summaries if item["mean_participant_macro_f1"] >= ceiling - tolerance
    ]
    eligible.sort(
        key=lambda item: (
            item["negative_log_likelihood"],
            abs(np.log(item["temperature"])) + abs(item["posture_logit_offset"]),
            -item["bottom_30_percent_participant_macro_f1"],
            item["temperature"],
            item["posture_logit_offset"],
        )
    )
    selected = eligible[0]
    return (
        {
            "temperature": selected["temperature"],
            "posture_logit_offset": selected["posture_logit_offset"],
        },
        summaries,
    )


def _apply_calibration(
    probabilities: FloatArray,
    selection: dict[str, float],
    *,
    base_probability: FloatArray | None = None,
    restore_mask: BoolArray | None = None,
) -> FloatArray:
    result = apply_posture_conditional_calibration(
        probabilities,
        temperature=float(selection["temperature"]),
        posture_logit_offset=float(selection["posture_logit_offset"]),
    )
    if restore_mask is not None:
        if base_probability is None or restore_mask.shape != (result.shape[0],):
            raise ValueError("calibration restoration inputs do not align")
        result[restore_mask] = base_probability[restore_mask]
    return result


def _participant_scores(
    labels: IntArray,
    probabilities: FloatArray,
    participants: StringArray,
) -> dict[str, float]:
    report = _report(labels, probabilities, participants)
    return {
        str(item["participant_id"]): float(item["macro_f1"])
        for item in cast(list[dict[str, Any]], report["participants"])
    }


def _utility_targets(
    labels: IntArray,
    participants: StringArray,
    base: FloatArray,
    pulse: FloatArray,
    broad: FloatArray,
    signature_ids: StringArray,
) -> tuple[FloatArray, FloatArray]:
    base_scores = _participant_scores(labels, base, participants)
    pulse_scores = _participant_scores(labels, pulse, participants)
    broad_scores = _participant_scores(labels, broad, participants)
    return (
        np.asarray(
            [base_scores[item] - pulse_scores[item] for item in signature_ids.tolist()],
            dtype=np.float64,
        ),
        np.asarray(
            [broad_scores[item] - pulse_scores[item] for item in signature_ids.tolist()],
            dtype=np.float64,
        ),
    )


def _controller(
    *,
    training_features: FloatArray,
    training_ids: StringArray,
    base_utility: FloatArray,
    broad_utility: FloatArray,
    evaluation_features: FloatArray,
    evaluation_signature_ids: StringArray,
    evaluation_base: FloatArray,
    evaluation_pulse: FloatArray,
    evaluation_broad: FloatArray,
    evaluation_participants: StringArray,
    physics_trusted: BoolArray,
    fixed: dict[str, Any],
    lower_bound_z: float,
) -> HeraRoutingResult:
    base_prediction = participant_jackknife_advantage_prediction(
        training_features,
        base_utility,
        training_ids,
        evaluation_features,
        ridge_penalty=float(fixed["responder_ridge_penalty"]),
        lower_bound_z=lower_bound_z,
    )
    broad_prediction = participant_jackknife_advantage_prediction(
        training_features,
        broad_utility,
        training_ids,
        evaluation_features,
        ridge_penalty=float(fixed["responder_ridge_penalty"]),
        lower_bound_z=lower_bound_z,
    )
    support = standardized_nearest_support_distance(training_features, evaluation_features)
    return apply_tri_state_responder_controller(
        evaluation_base,
        evaluation_pulse,
        evaluation_broad,
        evaluation_participants,
        evaluation_signature_ids,
        base_prediction.lower_bound,
        broad_prediction.lower_bound,
        support,
        physics_trusted,
        minimum_advantage=float(fixed["responder_minimum_advantage"]),
        maximum_support_distance=float(fixed["responder_maximum_support_distance"]),
    )


def _meta_training_controller(
    *,
    signatures: FloatArray,
    signature_ids: StringArray,
    base_utility: FloatArray,
    broad_utility: FloatArray,
    base: FloatArray,
    pulse: FloatArray,
    broad: FloatArray,
    participants: StringArray,
    physics_trusted: BoolArray,
    fixed: dict[str, Any],
) -> tuple[FloatArray, IntArray, BoolArray]:
    """Produce controller predictions excluding each training participant in turn."""

    probabilities = np.full_like(base, np.nan)
    states = np.full(base.shape[0], -1, dtype=np.int64)
    vetoed = np.zeros(base.shape[0], dtype=np.bool_)
    for held_index, held in enumerate(signature_ids.tolist()):
        train = np.arange(signature_ids.size) != held_index
        rows = participants == held
        result = _controller(
            training_features=signatures[train],
            training_ids=signature_ids[train],
            base_utility=base_utility[train],
            broad_utility=broad_utility[train],
            evaluation_features=signatures[held_index : held_index + 1],
            evaluation_signature_ids=signature_ids[held_index : held_index + 1],
            evaluation_base=base[rows],
            evaluation_pulse=pulse[rows],
            evaluation_broad=broad[rows],
            evaluation_participants=participants[rows],
            physics_trusted=physics_trusted[rows],
            fixed=fixed,
            lower_bound_z=float(fixed["responder_safe_lower_bound_z"]),
        )
        probabilities[rows] = result.probabilities
        states[rows] = result.state_index
        vetoed[rows] = result.physics_vetoed
    if not np.isfinite(probabilities).all() or np.any(states < 0):
        raise ValueError("HERA-CTGR meta-controller predictions are incomplete")
    return probabilities, states, vetoed


def _random_state_control(
    *,
    base: FloatArray,
    pulse: FloatArray,
    broad: FloatArray,
    participants: StringArray,
    participant_state: dict[str, str],
    physics_trusted: BoolArray,
    calibration: dict[str, float],
    seed: int,
) -> tuple[FloatArray, IntArray, BoolArray, dict[str, str]]:
    ids = np.unique(participants)
    states = [participant_state[str(item)] for item in ids.tolist()]
    generator = np.random.default_rng(seed)
    shuffled = list(np.asarray(states, dtype=np.str_)[generator.permutation(len(states))])
    output = pulse.copy()
    indices = np.ones(base.shape[0], dtype=np.int64)
    assigned: dict[str, str] = {}
    for participant, state_value in zip(ids.tolist(), shuffled, strict=True):
        state = str(state_value)
        rows = participants == participant
        assigned[str(participant)] = state
        if state == "OFF":
            output[rows] = base[rows]
            indices[rows] = 0
        elif state == "BROAD":
            output[rows] = broad[rows]
            indices[rows] = 2
    intervention = indices != 0
    physical_veto = intervention & ~physics_trusted
    restore = (indices == 0) | physical_veto
    output = _apply_calibration(
        output,
        calibration,
        base_probability=base,
        restore_mask=restore,
    )
    return output, indices, physical_veto, assigned


def _label_informed_oracle(
    labels: IntArray,
    participants: StringArray,
    methods: tuple[FloatArray, FloatArray, FloatArray],
) -> tuple[FloatArray, dict[str, str]]:
    names = ("OFF", "PULSE", "BROAD")
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
        best = int(np.argmax(np.asarray(scores, dtype=np.float64)))
        output[rows] = methods[best][rows]
        choices[str(participant)] = names[best]
    return output, choices


def _mean(values: list[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _aggregate_reports(
    reports_by_seed: dict[int, dict[str, dict[str, Any]]],
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for method in _METHODS:
        reports = [reports_by_seed[seed][method] for seed in _EXPECTED_SEEDS]
        result[method] = {
            "mean_participant_macro_f1": _mean(
                [float(item["primary"]["mean_participant_macro_f1"]) for item in reports]
            ),
            "bottom_30_percent_participant_macro_f1": _mean(
                [
                    float(item["primary"]["bottom_30_percent_participant_macro_f1"])
                    for item in reports
                ]
            ),
            "worst_participant_macro_f1": _mean(
                [float(item["primary"]["worst_participant_macro_f1"]) for item in reports]
            ),
            "mobility_recall": _mean(
                [
                    float(item["window_level_diagnostics"]["per_class_recall"]["mobility"])
                    for item in reports
                ]
            ),
            "sitting_recall": _mean(
                [
                    float(item["window_level_diagnostics"]["per_class_recall"]["sitting"])
                    for item in reports
                ]
            ),
            "standing_recall": _mean(
                [
                    float(item["window_level_diagnostics"]["per_class_recall"]["standing"])
                    for item in reports
                ]
            ),
            "negative_log_likelihood": _mean(
                [float(item["calibration"]["negative_log_likelihood"]) for item in reports]
            ),
            "multiclass_brier_score": _mean(
                [float(item["calibration"]["multiclass_brier_score"]) for item in reports]
            ),
        }
    return result


def _participant_mean_scores(
    reports_by_seed: dict[int, dict[str, dict[str, Any]]], method: str
) -> dict[str, float]:
    values: dict[str, list[float]] = {item: [] for item in _EXPECTED_PARTICIPANTS}
    for seed in _EXPECTED_SEEDS:
        for row in cast(list[dict[str, Any]], reports_by_seed[seed][method]["participants"]):
            values[str(row["participant_id"])].append(float(row["macro_f1"]))
    return {key: _mean(item) for key, item in values.items()}


def _bootstrap_intervals(
    reports_by_seed: dict[int, dict[str, dict[str, Any]]],
    *,
    method: str,
    comparator: str,
    resamples: int = 100_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    left = _participant_mean_scores(reports_by_seed, method)
    right = _participant_mean_scores(reports_by_seed, comparator)
    left_values = np.asarray([left[item] for item in _EXPECTED_PARTICIPANTS], dtype=np.float64)
    right_values = np.asarray([right[item] for item in _EXPECTED_PARTICIPANTS], dtype=np.float64)
    differences = left_values - right_values
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, differences.size, size=(resamples, differences.size))
    mean_draw = differences[draw].mean(axis=1)
    bottom_draw = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        selected = draw[index]
        bottom_draw[index] = _lower_fraction_mean(
            left_values[selected].tolist()
        ) - _lower_fraction_mean(right_values[selected].tolist())
    return {
        "participant_mean_differences": {
            participant: float(left[participant] - right[participant])
            for participant in _EXPECTED_PARTICIPANTS
        },
        "positive_participant_count": int(np.sum(differences > 0.0)),
        "negative_participant_count": int(np.sum(differences < 0.0)),
        "mean_difference": float(np.mean(differences)),
        "mean_difference_familywise_97_5_percent_bootstrap_interval": [
            float(np.quantile(mean_draw, 0.0125)),
            float(np.quantile(mean_draw, 0.9875)),
        ],
        "bottom_30_percent_difference_familywise_97_5_percent_bootstrap_interval": [
            float(np.quantile(bottom_draw, 0.0125)),
            float(np.quantile(bottom_draw, 0.9875)),
        ],
        "bootstrap_resamples": resamples,
        "bootstrap_seed": seed,
        "interpretation": "descriptive reused-source participant bootstrap; seeds are averaged within participant",
    }


def run_hera_retrospective(
    *,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Execute the locked five-seed HERA-CTGR retrospective experiment."""

    if output_directory.exists():
        raise FileExistsError(f"HERA-CTGR output already exists: {output_directory}")
    config = load_hera_retrospective_config(config_path)
    inputs = cast(dict[str, Any], config["inputs"])
    fixed = cast(dict[str, Any], config["fixed_method"])
    freeze = cast(dict[str, Any], config["method_freeze"])
    source_manifest_path = _validated_path(inputs["source_manifest"], name="source manifest")
    dataset_manifest_path = _validated_path(inputs["dataset_manifest"], name="dataset manifest")
    raw_source_path = _validated_path(inputs["raw_source"], name="raw source")
    ctgr_config_path = _validated_path(inputs["ctgr_config"], name="CTGR config")
    selection_path = _validated_path(inputs["ctgr_selection"], name="CTGR selection")
    cage_config_path = _validated_path(inputs["cage_config"], name="CAGE config")
    _validated_path(freeze["method_config"], name="HERA method config")
    _validated_path(freeze["method_implementation"], name="HERA implementation")
    ctgr_config = _load_ctgr_config(ctgr_config_path)
    cage_config = _load_cage_config(cage_config_path)
    selection = _read_hashed_record(selection_path)
    if (
        selection.get("status") != "selection_frozen_before_outer_evaluation"
        or selection.get("outer_labels_used_for_selection") is not False
        or selection.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("HERA-CTGR requires the frozen source-only CTGR ranking")

    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path,
        raw_csv_path=raw_source_path,
    )
    if tuple(sorted(set(participants.tolist()), key=int)) != _EXPECTED_PARTICIPANTS:
        raise PermissionError("HERA-CTGR retrospective source participant set changed")
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_source_path,
        expected_participants=participants,
        expected_windows=windows,
    )
    sampling_rate_hz = float(ctgr_config["input_contract"]["sampling_rate_hz"])
    if sampling_rate_hz != 50.0:
        raise ValueError("HERA-CTGR source sampling rate changed")
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=sampling_rate_hz)
    candidates = {str(item["id"]): item for item in _ctgr_candidates(ctgr_config)}
    selected = _selected_by_fold(selection)
    top_by_fold = _top_ranked_candidates(selection, candidates)
    outer_folds = cast(list[dict[str, Any]], manifest["source_nested_cv"])
    if len(outer_folds) != 5:
        raise ValueError("HERA-CTGR source manifest must contain five outer folds")
    n_jobs = int(ctgr_config["n_jobs"])
    kinematic = extract_gravity_kinematic_context(
        signals,
        gravity,
        sampling_rate_hz=sampling_rate_hz,
    )

    prediction_payload: dict[str, NDArray[Any]] = {
        "labels": labels,
        "participant_ids": participants,
        "window_ids": windows,
        "kinematic_features": kinematic.features,
    }
    reports_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    seed_records: list[dict[str, Any]] = []
    audits_by_seed: dict[int, dict[str, Any]] = {}
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
        state_indices = {name: np.full(labels.size, -1, dtype=np.int64) for name in _ROUTED_METHODS}
        physical_vetoes = {
            name: np.zeros(labels.size, dtype=np.bool_) for name in _SCIENTIFIC_METHODS[3:]
        }
        fold_records: list[dict[str, Any]] = []
        for outer_index, outer in enumerate(outer_folds):
            fold_id = str(outer["outer_fold_id"])
            top_candidates = top_by_fold[fold_id]
            if str(top_candidates[0]["id"]) != selected[fold_id]:
                raise ValueError(f"HERA-CTGR selected candidate mismatch for {fold_id}")
            inner, training = _inner_oof_paths(
                outer=outer,
                top_candidates=top_candidates,
                seed=seed,
                n_jobs=n_jobs,
                rmrp=rmrp,
                views=views,
                labels=labels,
                participants=participants,
            )
            evaluation = ~training
            evaluation_top3, top_ids = _outer_top3_path(
                top_candidates=top_candidates,
                preserved=preserved,
                evaluation=evaluation,
                training=training,
                outer_index=outer_index,
                seed=seed,
                n_jobs=n_jobs,
                views=views,
                labels=labels,
                participants=participants,
            )
            cage_context, cage_reliability, channel_scale = _outer_context(
                signals,
                gravity,
                training,
                sampling_rate_hz=sampling_rate_hz,
            )
            all_base = np.concatenate((inner["base"][training], preserved["base"][evaluation]))
            all_expert = np.concatenate(
                (inner["selected_expert"][training], preserved["expert"][evaluation])
            )
            all_ctgr = np.concatenate((inner["pulse"][training], preserved["ctgr"][evaluation]))
            all_context = np.concatenate((cage_context[training], cage_context[evaluation]))
            all_reliability = np.concatenate(
                (cage_reliability[training], cage_reliability[evaluation])
            )
            cage_methods, cage_diagnostics, _ = evaluate_cage_outer_arrays(
                training_base_probability=inner["base"][training],
                training_expert_probability=inner["selected_expert"][training],
                training_labels=labels[training],
                training_participants=participants[training],
                training_context=cage_context[training],
                training_reliability=cage_reliability[training],
                evaluation_base_probability=all_base,
                evaluation_expert_probability=all_expert,
                evaluation_ctgr_probability=all_ctgr,
                evaluation_context=all_context,
                evaluation_reliability=all_reliability,
                expert_name=str(top_candidates[0]["id"]),
                cage_config=cage_config,
            )
            broad_all = cage_methods["global_trust_blend"]
            training_count = int(training.sum())
            broad_training = broad_all[:training_count]
            broad_evaluation = broad_all[training_count:]

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
            combined_training_reliability = np.minimum(
                cage_reliability[training], physics_training.reliability
            )
            combined_evaluation_reliability = np.minimum(
                cage_reliability[evaluation], physics_evaluation.reliability
            )

            ctgr_calibration, ctgr_calibration_candidates = _select_posture_calibration(
                inner["pulse"][training],
                labels[training],
                participants[training],
                fixed,
            )
            calibrated_ctgr = _apply_calibration(preserved["ctgr"][evaluation], ctgr_calibration)
            ctgr_veto, ctgr_vetoed = apply_physics_veto(
                preserved["base"][evaluation],
                preserved["ctgr"][evaluation],
                physics_evaluation.trusted,
            )
            strict_calibration, strict_calibration_candidates = _select_posture_calibration(
                inner["top3"][training],
                labels[training],
                participants[training],
                fixed,
            )
            strict_pre_veto = _apply_calibration(evaluation_top3, strict_calibration)
            strict, strict_vetoed = apply_physics_veto(
                preserved["base"][evaluation],
                strict_pre_veto,
                physics_evaluation.trusted,
            )

            training_signatures = build_responder_signatures(
                inner["base"][training],
                inner["selected_expert"][training],
                inner["pulse"][training],
                broad_training,
                participants[training],
                gravity_reliability=combined_training_reliability,
                kinematic_context=_slice_context(kinematic, training),
                physics_trusted=physics_training.trusted,
                low_confidence_threshold=float(fixed["responder_low_confidence_threshold"]),
            )
            evaluation_signatures = build_responder_signatures(
                preserved["base"][evaluation],
                preserved["expert"][evaluation],
                preserved["ctgr"][evaluation],
                broad_evaluation,
                participants[evaluation],
                gravity_reliability=combined_evaluation_reliability,
                kinematic_context=_slice_context(kinematic, evaluation),
                physics_trusted=physics_evaluation.trusted,
                low_confidence_threshold=float(fixed["responder_low_confidence_threshold"]),
            )
            base_utility, broad_utility = _utility_targets(
                labels[training],
                participants[training],
                inner["base"][training],
                inner["pulse"][training],
                broad_training,
                training_signatures.participant_ids,
            )
            mean_route = _controller(
                training_features=training_signatures.features,
                training_ids=training_signatures.participant_ids,
                base_utility=base_utility,
                broad_utility=broad_utility,
                evaluation_features=evaluation_signatures.features,
                evaluation_signature_ids=evaluation_signatures.participant_ids,
                evaluation_base=preserved["base"][evaluation],
                evaluation_pulse=preserved["ctgr"][evaluation],
                evaluation_broad=broad_evaluation,
                evaluation_participants=participants[evaluation],
                physics_trusted=physics_evaluation.trusted,
                fixed=fixed,
                lower_bound_z=float(fixed["responder_mean_lower_bound_z"]),
            )
            safe_route = _controller(
                training_features=training_signatures.features,
                training_ids=training_signatures.participant_ids,
                base_utility=base_utility,
                broad_utility=broad_utility,
                evaluation_features=evaluation_signatures.features,
                evaluation_signature_ids=evaluation_signatures.participant_ids,
                evaluation_base=preserved["base"][evaluation],
                evaluation_pulse=preserved["ctgr"][evaluation],
                evaluation_broad=broad_evaluation,
                evaluation_participants=participants[evaluation],
                physics_trusted=physics_evaluation.trusted,
                fixed=fixed,
                lower_bound_z=float(fixed["responder_safe_lower_bound_z"]),
            )
            meta_probability, meta_state, meta_veto = _meta_training_controller(
                signatures=training_signatures.features,
                signature_ids=training_signatures.participant_ids,
                base_utility=base_utility,
                broad_utility=broad_utility,
                base=inner["base"][training],
                pulse=inner["pulse"][training],
                broad=broad_training,
                participants=participants[training],
                physics_trusted=physics_training.trusted,
                fixed=fixed,
            )
            meta_restore = (meta_state == 0) | meta_veto
            full_calibration, full_calibration_candidates = _select_posture_calibration(
                meta_probability,
                labels[training],
                participants[training],
                fixed,
                base_probability=inner["base"][training],
                restore_mask=meta_restore,
            )
            full_restore = (safe_route.state_index == 0) | safe_route.physics_vetoed
            full_probability = _apply_calibration(
                safe_route.probabilities,
                full_calibration,
                base_probability=preserved["base"][evaluation],
                restore_mask=full_restore,
            )
            random_probability, random_state, _random_veto, random_assignment = (
                _random_state_control(
                    base=preserved["base"][evaluation],
                    pulse=preserved["ctgr"][evaluation],
                    broad=broad_evaluation,
                    participants=participants[evaluation],
                    participant_state=safe_route.participant_state,
                    physics_trusted=physics_evaluation.trusted,
                    calibration=full_calibration,
                    seed=int(fixed["matched_random_seed_offset"]) + seed * 10 + outer_index,
                )
            )

            fold_probabilities = {
                "base_uncalibrated": preserved["base"][evaluation],
                "frozen_ctgr": preserved["ctgr"][evaluation],
                "global_trust_blend": broad_evaluation,
                "ctgr_aggregate_calibrated": calibrated_ctgr,
                "ctgr_physics_veto": ctgr_veto,
                "ctgr_candidate_ensemble_top3": evaluation_top3,
                "hera_ctgr_strict": strict,
                "hera_ctgr_context_mean": mean_route.probabilities,
                "hera_ctgr_context_safe": safe_route.probabilities,
                "hera_ctgr_full": full_probability,
                "matched_random_state_control": random_probability,
            }
            for name, value in fold_probabilities.items():
                probabilities[name][evaluation] = value
            state_indices["hera_ctgr_context_mean"][evaluation] = mean_route.state_index
            state_indices["hera_ctgr_context_safe"][evaluation] = safe_route.state_index
            state_indices["hera_ctgr_full"][evaluation] = safe_route.state_index
            state_indices["matched_random_state_control"][evaluation] = random_state
            physical_vetoes["ctgr_aggregate_calibrated"][evaluation] = False
            physical_vetoes["ctgr_physics_veto"][evaluation] = ctgr_vetoed
            physical_vetoes["ctgr_candidate_ensemble_top3"][evaluation] = False
            physical_vetoes["hera_ctgr_strict"][evaluation] = strict_vetoed
            physical_vetoes["hera_ctgr_context_mean"][evaluation] = mean_route.physics_vetoed
            physical_vetoes["hera_ctgr_context_safe"][evaluation] = safe_route.physics_vetoed
            physical_vetoes["hera_ctgr_full"][evaluation] = safe_route.physics_vetoed

            fold_records.append(
                {
                    "outer_fold_id": fold_id,
                    "outer_test_subjects": list(outer["outer_test_subjects"]),
                    "selected_ctgr_candidate_id": selected[fold_id],
                    "top_three_candidate_ids": list(top_ids),
                    "outer_training_participant_count": int(np.unique(participants[training]).size),
                    "inner_oof_window_count": int(training.sum()),
                    "outer_evaluation_window_count": int(evaluation.sum()),
                    "outer_training_channel_scale": channel_scale.tolist(),
                    "physics_reference": {
                        "normalized_residual_p90_threshold": physics_reference.normalized_residual_p90_threshold,
                        "training_trusted_fraction": float(physics_training.trusted.mean()),
                        "evaluation_trusted_fraction": float(physics_evaluation.trusted.mean()),
                    },
                    "ctgr_calibration": ctgr_calibration,
                    "ctgr_calibration_candidate_count": len(ctgr_calibration_candidates),
                    "strict_calibration": strict_calibration,
                    "strict_calibration_candidate_count": len(strict_calibration_candidates),
                    "full_calibration": full_calibration,
                    "full_calibration_candidate_count": len(full_calibration_candidates),
                    "mean_participant_state": mean_route.participant_state,
                    "safe_participant_state": safe_route.participant_state,
                    "random_participant_state": random_assignment,
                    "safe_support_distance": safe_route.participant_support_distance,
                    "safe_base_advantage_lower_bound": safe_route.participant_base_advantage_lower_bound,
                    "safe_broad_advantage_lower_bound": safe_route.participant_broad_advantage_lower_bound,
                    "training_base_utility": {
                        item: float(value)
                        for item, value in zip(
                            training_signatures.participant_ids.tolist(),
                            base_utility,
                            strict=True,
                        )
                    },
                    "training_broad_utility": {
                        item: float(value)
                        for item, value in zip(
                            training_signatures.participant_ids.tolist(),
                            broad_utility,
                            strict=True,
                        )
                    },
                    "cage_broad_fit": cage_diagnostics,
                    "outer_labels_used_for_training_selection_calibration_or_routing": False,
                }
            )

        if any(
            not np.isfinite(probabilities[name]).all()
            for name in _METHODS
            if name != "label_informed_three_way_oracle_diagnostic"
        ):
            raise ValueError(f"HERA-CTGR seed {seed} predictions are incomplete")
        oracle, oracle_choices = _label_informed_oracle(
            labels,
            participants,
            (
                probabilities["base_uncalibrated"],
                probabilities["frozen_ctgr"],
                probabilities["global_trust_blend"],
            ),
        )
        probabilities["label_informed_three_way_oracle_diagnostic"] = oracle
        reports = {
            name: _report(labels, value, participants) for name, value in probabilities.items()
        }
        expected = cast(dict[str, float], preserved_record["expected_mean_participant_macro_f1"])
        replay = {
            "base_uncalibrated": float(
                reports["base_uncalibrated"]["primary"]["mean_participant_macro_f1"]
            ),
            "frozen_ctgr": float(reports["frozen_ctgr"]["primary"]["mean_participant_macro_f1"]),
        }
        if any(
            not np.isclose(replay[name], expected[name], atol=1e-12, rtol=0.0) for name in replay
        ):
            raise ValueError(f"preserved CTGR report replay changed for seed {seed}")
        preserved_record["aggregate_report_replay_exact"] = True

        audits: dict[str, Any] = {}
        for name in _SCIENTIFIC_METHODS[2:]:
            changed_probability = (
                np.max(np.abs(probabilities[name] - probabilities["frozen_ctgr"]), axis=1) > 1e-12
            )
            chosen = state_indices.get(name, np.full(labels.size, -1, dtype=np.int64))
            audits[name] = _routing_audit(
                labels,
                participants,
                probabilities["frozen_ctgr"],
                probabilities[name],
                changed_probability,
                chosen,
                ("OFF", "PULSE", "BROAD"),
            )
        audits_by_seed[seed] = audits
        reports_by_seed[seed] = reports
        seed_records.append(
            {
                "seed": seed,
                "preserved_ctgr_record": preserved_record,
                "folds": fold_records,
                "reports": reports,
                "routing_audits_against_frozen_ctgr": audits,
                "label_informed_oracle_participant_choices": oracle_choices,
                "oracle_labels_used_for_model_or_hyperparameter_selection": False,
                "outer_labels_used_for_training_selection_calibration_or_routing": False,
                "participants_11_through_20_loaded": False,
                "daghar_loaded": False,
            }
        )
        for name, probability_value in probabilities.items():
            prediction_payload[f"seed_{seed}__{name}_probabilities"] = probability_value
        for name, state_value in state_indices.items():
            prediction_payload[f"seed_{seed}__{name}_state_index"] = state_value
        for name, veto_value in physical_vetoes.items():
            prediction_payload[f"seed_{seed}__{name}_physics_veto"] = veto_value

    aggregate = _aggregate_reports(reports_by_seed)
    full_audits = [audits_by_seed[seed]["hera_ctgr_full"] for seed in _EXPECTED_SEEDS]
    rescues = sum(int(item["rescue_count"]) for item in full_audits)
    harms = sum(int(item["harm_count"]) for item in full_audits)
    changed = sum(int(item["changed_count"]) for item in full_audits)
    routing = {
        "opportunity_count": labels.size * len(_EXPECTED_SEEDS),
        "probability_intervention_count": sum(int(item["routed_count"]) for item in full_audits),
        "hard_label_change_count": changed,
        "rescue_count": rescues,
        "harm_count": harms,
        "net_rescues": rescues - harms,
        "intervention_precision": None if rescues + harms == 0 else rescues / (rescues + harms),
        "harmful_changed_fraction": None if changed == 0 else harms / changed,
    }
    comparison = _bootstrap_intervals(
        reports_by_seed,
        method="hera_ctgr_full",
        comparator="frozen_ctgr",
    )
    full = aggregate["hera_ctgr_full"]
    ctgr = aggregate["frozen_ctgr"]
    differences = {key: full[key] - ctgr[key] for key in full}
    posture_difference = min(differences["sitting_recall"], differences["standing_recall"])
    mean_ci = comparison["mean_difference_familywise_97_5_percent_bootstrap_interval"]
    bottom_ci = comparison[
        "bottom_30_percent_difference_familywise_97_5_percent_bootstrap_interval"
    ]
    advancement_gate = {
        "mean_improvement_at_least_0_010": {
            "observed": differences["mean_participant_macro_f1"],
            "passed": differences["mean_participant_macro_f1"] >= 0.010,
        },
        "bottom_30_percent_improvement_at_least_0_015": {
            "observed": differences["bottom_30_percent_participant_macro_f1"],
            "passed": differences["bottom_30_percent_participant_macro_f1"] >= 0.015,
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
        "worst_participant_noninferiority_minus_0_010": {
            "observed": differences["worst_participant_macro_f1"],
            "passed": differences["worst_participant_macro_f1"] >= -0.010,
        },
        "posture_recall_noninferiority_minus_0_010": {
            "observed": posture_difference,
            "passed": posture_difference >= -0.010,
        },
        "negative_log_likelihood_not_worse": {
            "observed": differences["negative_log_likelihood"],
            "passed": differences["negative_log_likelihood"] <= 0.0,
        },
        "multiclass_brier_not_worse": {
            "observed": differences["multiclass_brier_score"],
            "passed": differences["multiclass_brier_score"] <= 0.0,
        },
        "improvement_not_only_one_participant": {
            "observed_positive_participants": comparison["positive_participant_count"],
            "passed": int(comparison["positive_participant_count"]) >= 2,
        },
    }
    advancement_gate["overall_passed"] = all(
        bool(item["passed"])
        for item in advancement_gate.values()
        if isinstance(item, dict) and "passed" in item
    )
    breakthrough_gate = {
        "mean_improvement_at_least_0_020": differences["mean_participant_macro_f1"] >= 0.020,
        "familywise_mean_interval_lower_bound_above_zero": float(mean_ci[0]) > 0.0,
        "familywise_bottom_interval_lower_bound_above_zero": float(bottom_ci[0]) > 0.0,
    }
    breakthrough_gate["overall_passed"] = all(breakthrough_gate.values())

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "hera_ctgr_fully_nested_retrospective_source_result",
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
        "hera_ctgr_full_minus_frozen_ctgr": differences,
        "hera_ctgr_full_routing_against_frozen_ctgr": routing,
        "descriptive_participant_bootstrap": comparison,
        "advancement_gate": advancement_gate,
        "breakthrough_gate": breakthrough_gate,
        "strict_window_lane": [
            "ctgr_aggregate_calibrated",
            "ctgr_physics_veto",
            "ctgr_candidate_ensemble_top3",
            "hera_ctgr_strict",
        ],
        "transductive_unlabelled_participant_context_lane": [
            "hera_ctgr_context_mean",
            "hera_ctgr_context_safe",
            "hera_ctgr_full",
        ],
        "diagnostic_only_methods": [
            "matched_random_state_control",
            "label_informed_three_way_oracle_diagnostic",
        ],
        "outer_labels_used_for_training_selection_calibration_or_routing": False,
        "oracle_labels_used_for_hindsight_diagnostic_only": True,
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
    result = run_hera_retrospective(
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
                "hera_ctgr_full_minus_frozen_ctgr": result["hera_ctgr_full_minus_frozen_ctgr"],
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
