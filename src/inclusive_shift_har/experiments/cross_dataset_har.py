"""Prospective external-development runner for CTGR/CAGE/HERA adaptations.

This runner never opens the sealed WearGait-PD target.  It evaluates the derived-gravity
adaptations on public development cohorts using participant-exclusive nested folds and
retains the outer labels exclusively for fold scoring.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import (
    CORE_CLASS_NAMES,
    ExternalHARWindows,
    load_fog_star,
    load_imu_har_il,
    participant_fold_assignment,
)
from inclusive_shift_har.experiments.cage_har import _report
from inclusive_shift_har.experiments.cage_har_retrospective import (
    _outer_context,
    evaluate_cage_outer_arrays,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _ESTIMATOR_ORDER,
    _VIEW_ORDER,
    _candidates,
    _feature_views,
    _fit_base,
    _fit_expert,
    _gravity_probability,
    _selection_order,
)
from inclusive_shift_har.experiments.hera_ctgr_retrospective import (
    _apply_calibration,
    _apply_candidate,
    _controller,
    _meta_training_controller,
    _select_posture_calibration,
    _slice_context,
    _utility_targets,
)
from inclusive_shift_har.experiments.hera_ctgr_v2_retrospective import (
    _apply_dual_candidate,
    _fit_route_lane,
    _marginalize,
    _participant_oracle,
    _select_dual_candidate,
    _select_offset,
    _select_temperature,
    _window_oracle,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _positive_probability,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    predict_classical_probabilities,
)
from inclusive_shift_har.models.hera_ctgr import (
    apply_physics_reference,
    apply_physics_veto,
    build_responder_signatures,
    extract_gravity_kinematic_context,
    fit_physics_reference,
)
from inclusive_shift_har.models.hera_ctgr_v2 import (
    apply_rank_locked_posture_offset,
    build_rescue_harm_features,
    extract_dual_frame_posture_features,
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

_INVENTION_METHODS = (
    "RMRP-DG",
    "CTGR-DG",
    "CAGE-DG",
    "CTGR-DG-top3-equal",
    "HERA-DG-strict",
    "HERA-DG-context-safe",
    "HERA-DG-full",
    "HERA-DG-v2-weighted",
    "HERA-DG-v2-core",
    "HERA-DG-v2-dual",
    "HERA-DG-v2-full",
)
_CLASSICAL_METHODS = ("RandomForest-6ch", "XGBoost-6ch")
_METHODS = _INVENTION_METHODS + _CLASSICAL_METHODS


@dataclass(frozen=True, slots=True)
class _InnerPredictions:
    base: FloatArray
    expert_by_key: dict[tuple[str, str], FloatArray]
    dual_expert: FloatArray
    selected_candidate: dict[str, Any]
    selected_expert_candidate: dict[str, Any]
    top_three_expert_candidates: tuple[dict[str, Any], ...]
    candidate_ranking: list[dict[str, Any]]
    eligible_ids: tuple[str, ...]
    folds: tuple[dict[str, Any], ...]


def _read_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"configuration must be an object with string keys: {path}")
    return cast(dict[str, Any], value)


def _three_probabilities(model: Any, features: FloatArray) -> FloatArray:
    probability = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    if probability.shape != (features.shape[0], classes.size) or set(classes.tolist()) != {
        0,
        1,
        2,
    }:
        raise ValueError("three-class estimator returned a misaligned probability matrix")
    aligned = np.zeros((features.shape[0], 3), dtype=np.float64)
    for column, class_index in enumerate(classes.tolist()):
        aligned[:, class_index] = probability[:, column]
    aligned /= aligned.sum(axis=1, keepdims=True)
    return aligned


def _participant_mask(participants: StringArray, selected: set[str]) -> BoolArray:
    return np.asarray(np.isin(participants, sorted(selected)), dtype=np.bool_)


def _inner_oof_and_selection(
    *,
    rmrp: FloatArray,
    views: dict[str, FloatArray],
    dual_features: FloatArray,
    labels: IntArray,
    participants: StringArray,
    outer_training: BoolArray,
    candidates: list[dict[str, Any]],
    outer_index: int,
    seed: int,
    inner_fold_count: int,
    n_jobs: int,
) -> _InnerPredictions:
    training_ids = sorted(np.unique(participants[outer_training]).tolist())
    assignment = participant_fold_assignment(
        training_ids,
        fold_count=inner_fold_count,
        seed=seed + 10_000 + outer_index,
    )
    base = np.full((labels.size, 3), np.nan, dtype=np.float64)
    keys = tuple((view, estimator) for view in _VIEW_ORDER for estimator in _ESTIMATOR_ORDER)
    expert_by_key = {key: np.full((labels.size, 3), np.nan, dtype=np.float64) for key in keys}
    dual_expert = np.full((labels.size, 3), np.nan, dtype=np.float64)
    covered = np.zeros(labels.size, dtype=np.bool_)
    fold_records: list[dict[str, Any]] = []
    for inner_index in range(inner_fold_count):
        validation_ids = {
            participant for participant, fold in assignment.items() if fold == inner_index
        }
        validation = outer_training & _participant_mask(participants, validation_ids)
        training = outer_training & ~validation
        if np.any(training & validation) or not validation.any():
            raise PermissionError("inner participant partitions overlap or are empty")
        if set(np.unique(labels[training]).tolist()) != {0, 1, 2}:
            raise ValueError("an inner training fold lacks one of the three core classes")
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training,
            seed=seed + 101 * outer_index + inner_index,
            n_jobs=n_jobs,
        )
        fold_base = _three_probabilities(base_model, rmrp[validation])
        base[validation] = fold_base
        for key_index, (view, estimator) in enumerate(keys):
            expert_model = _fit_expert(
                estimator,
                views[view],
                labels,
                participants,
                training,
                seed=seed + 1_000 * outer_index + 37 * key_index + inner_index,
                n_jobs=n_jobs,
            )
            expert_by_key[(view, estimator)][validation] = _gravity_probability(
                fold_base, expert_model, views[view][validation]
            )
        dual_model = _fit_expert(
            "extra_trees_leaf3",
            dual_features,
            labels,
            participants,
            training,
            seed=seed + 7_001 + 101 * outer_index + inner_index,
            n_jobs=n_jobs,
        )
        dual_expert[validation] = compose_mobility_posture_probabilities(
            fold_base[:, 0], _positive_probability(dual_model, dual_features[validation])
        )
        covered[validation] = True
        fold_records.append(
            {
                "inner_fold": inner_index,
                "training_participants": sorted(np.unique(participants[training]).tolist()),
                "validation_participants": sorted(validation_ids),
                "outer_evaluation_labels_accessed": False,
            }
        )
    if not np.array_equal(covered, outer_training):
        raise PermissionError("inner folds did not cover each outer-training window exactly once")
    if not np.isfinite(base[outer_training]).all() or any(
        not np.isfinite(value[outer_training]).all() for value in expert_by_key.values()
    ):
        raise ValueError("inner OOF prediction matrix is incomplete")

    summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        if str(candidate["id"]) == "base_no_route":
            probability = base[outer_training]
        else:
            key = (str(candidate["expert_view"]), str(candidate["posture_estimator"]))
            probability = _apply_candidate(
                base[outer_training], expert_by_key[key][outer_training], candidate
            )
        report = _report(labels[outer_training], probability, participants[outer_training])
        summaries.append(
            {
                "candidate_id": str(candidate["id"]),
                "mean_participant_macro_f1": float(report["primary"]["mean_participant_macro_f1"]),
                "lower_30_percent_participant_macro_f1": float(
                    report["primary"]["bottom_30_percent_participant_macro_f1"]
                ),
                "worst_participant_macro_f1": float(
                    report["primary"]["worst_participant_macro_f1"]
                ),
                "mean_trigger_fraction": float(
                    np.mean(
                        np.max(base[outer_training], axis=1)
                        < float(candidate["confidence_threshold"])
                    )
                    if str(candidate["id"]) != "base_no_route"
                    else 0.0
                ),
            }
        )
    ranking, eligible = _selection_order(summaries, candidates, tolerance=0.005)
    by_id = {str(item["id"]): item for item in candidates}
    selected = by_id[str(ranking[0]["candidate_id"])]
    eligible_experts = [
        by_id[str(item["candidate_id"])]
        for item in ranking
        if str(item["candidate_id"]) in set(eligible)
        and str(item["candidate_id"]) != "base_no_route"
    ]
    ranked_experts = [
        by_id[str(item["candidate_id"])]
        for item in ranking
        if str(item["candidate_id"]) != "base_no_route"
    ]
    top = (eligible_experts + [item for item in ranked_experts if item not in eligible_experts])[:3]
    if len(top) != 3:
        raise ValueError("CTGR family did not produce three usable expert candidates")
    selected_expert = selected if str(selected["id"]) != "base_no_route" else top[0]
    return _InnerPredictions(
        base=base,
        expert_by_key=expert_by_key,
        dual_expert=dual_expert,
        selected_candidate=selected,
        selected_expert_candidate=selected_expert,
        top_three_expert_candidates=tuple(top),
        candidate_ranking=ranking,
        eligible_ids=tuple(eligible),
        folds=tuple(fold_records),
    )


def _candidate_probability(
    base: FloatArray,
    expert_by_key: dict[tuple[str, str], FloatArray],
    candidate: dict[str, Any],
) -> FloatArray:
    if str(candidate["id"]) == "base_no_route":
        return base.copy()
    key = (str(candidate["expert_view"]), str(candidate["posture_estimator"]))
    return _apply_candidate(base, expert_by_key[key], candidate)


def _bottom_fraction(values: list[float], fraction: float = 0.30) -> float:
    count = max(1, int(np.ceil(fraction * len(values))))
    return float(np.mean(sorted(values)[:count]))


def _paired_bootstrap(
    report: dict[str, Any],
    comparator: dict[str, Any],
    *,
    resamples: int = 20_000,
    seed: int = 20260904,
) -> dict[str, Any]:
    left = {
        str(item["participant_id"]): float(item["macro_f1"])
        for item in cast(list[dict[str, Any]], report["participants"])
    }
    right = {
        str(item["participant_id"]): float(item["macro_f1"])
        for item in cast(list[dict[str, Any]], comparator["participants"])
    }
    if set(left) != set(right):
        raise ValueError("paired bootstrap reports do not cover the same participants")
    identifiers = sorted(left)
    left_values = np.asarray([left[item] for item in identifiers], dtype=np.float64)
    right_values = np.asarray([right[item] for item in identifiers], dtype=np.float64)
    differences = left_values - right_values
    generator = np.random.default_rng(seed)
    draw = generator.integers(0, len(identifiers), size=(resamples, len(identifiers)))
    mean_draw = differences[draw].mean(axis=1)
    bottom_draw = np.asarray(
        [
            _bottom_fraction(left_values[row].tolist())
            - _bottom_fraction(right_values[row].tolist())
            for row in draw
        ],
        dtype=np.float64,
    )
    return {
        "participant_count": len(identifiers),
        "mean_difference": float(np.mean(differences)),
        "mean_difference_95_percent_bootstrap_interval": [
            float(np.quantile(mean_draw, 0.025)),
            float(np.quantile(mean_draw, 0.975)),
        ],
        "bottom_30_percent_difference": _bottom_fraction(left_values.tolist())
        - _bottom_fraction(right_values.tolist()),
        "bottom_30_percent_difference_95_percent_bootstrap_interval": [
            float(np.quantile(bottom_draw, 0.025)),
            float(np.quantile(bottom_draw, 0.975)),
        ],
        "positive_participant_count": int(np.sum(differences > 0.0)),
        "negative_participant_count": int(np.sum(differences < 0.0)),
        "zero_participant_count": int(np.sum(differences == 0.0)),
        "resamples": resamples,
        "seed": seed,
    }


def _classical_outer_predictions(
    data: ExternalHARWindows,
    *,
    training: BoolArray,
    evaluation: BoolArray,
    seed: int,
) -> dict[str, FloatArray]:
    output: dict[str, FloatArray] = {}
    for model_name, result_name in (
        ("random_forest", "RandomForest-6ch"),
        ("xgboost", "XGBoost-6ch"),
    ):
        fitted = fit_classical_model(
            data.signals[training],
            data.labels[training],
            data.participant_ids[training].tolist(),
            config=ClassicalConfig(model_name=model_name, num_classes=3, seed=seed),
            channel_names=("lin_acc_x", "lin_acc_y", "lin_acc_z", "gyro_x", "gyro_y", "gyro_z"),
            lineage={
                "evidence_status": "external_development_only",
                "participant_exclusive": True,
                "dataset_id": data.dataset_id,
            },
        )
        _, probability = predict_classical_probabilities(
            fitted,
            data.signals[evaluation],
        )
        output[result_name] = probability
    return output


def _evaluate_seed(
    data: ExternalHARWindows,
    *,
    seed: int,
    outer_fold_count: int,
    inner_fold_count: int,
    n_jobs: int,
    repository_root: Path,
    include_classical: bool,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    labels = data.labels
    participants = data.participant_ids
    signals = np.asarray(data.signals, dtype=np.float64)
    gravity = np.asarray(data.gravity, dtype=np.float64)
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=data.sampling_rate_hz)
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=data.sampling_rate_hz)[
        "denoised"
    ]
    dual = extract_dual_frame_posture_features(
        denoised, gravity, sampling_rate_hz=data.sampling_rate_hz
    )
    kinematic = extract_gravity_kinematic_context(
        denoised, gravity, sampling_rate_hz=data.sampling_rate_hz
    )
    ctgr_config = _read_mapping(
        repository_root / "configs/experiments/confidence_triggered_gravity_residual_v1.yaml"
    )
    candidates = _candidates(ctgr_config)
    cage_config = _read_mapping(repository_root / "configs/experiments/cage_har_v1.yaml")
    v1_fixed = cast(
        dict[str, Any],
        _read_mapping(repository_root / "configs/experiments/hera_ctgr_retrospective_v1.yaml")[
            "fixed_method"
        ],
    )
    v2_fixed = cast(
        dict[str, Any],
        _read_mapping(repository_root / "configs/experiments/hera_ctgr_v2_retrospective_v1.yaml")[
            "fixed_method"
        ],
    )
    assignment = participant_fold_assignment(
        np.unique(participants).tolist(), fold_count=outer_fold_count, seed=seed
    )
    active_methods = _METHODS if include_classical else _INVENTION_METHODS
    probability = {
        name: np.full((labels.size, 3), np.nan, dtype=np.float64) for name in active_methods
    }
    fold_records: list[dict[str, Any]] = []
    for outer_index in range(outer_fold_count):
        evaluation_ids = {
            participant for participant, fold in assignment.items() if fold == outer_index
        }
        evaluation = _participant_mask(participants, evaluation_ids)
        training = ~evaluation
        if set(np.unique(labels[training]).tolist()) != {0, 1, 2}:
            raise ValueError("outer training partition lacks a core class")
        inner = _inner_oof_and_selection(
            rmrp=rmrp,
            views=views,
            dual_features=dual.features,
            labels=labels,
            participants=participants,
            outer_training=training,
            candidates=candidates,
            outer_index=outer_index,
            seed=seed,
            inner_fold_count=inner_fold_count,
            n_jobs=n_jobs,
        )
        base_model = _fit_base(
            rmrp,
            labels,
            participants,
            training,
            seed=seed + 500 * outer_index,
            n_jobs=n_jobs,
        )
        outer_base = _three_probabilities(base_model, rmrp[evaluation])
        needed = {
            (
                str(candidate["expert_view"]),
                str(candidate["posture_estimator"]),
            )
            for candidate in (
                inner.selected_expert_candidate,
                *inner.top_three_expert_candidates,
            )
        }
        if str(inner.selected_candidate["id"]) != "base_no_route":
            needed.add(
                (
                    str(inner.selected_candidate["expert_view"]),
                    str(inner.selected_candidate["posture_estimator"]),
                )
            )
        outer_expert_by_key: dict[tuple[str, str], FloatArray] = {}
        for key_index, (view, estimator) in enumerate(sorted(needed)):
            model = _fit_expert(
                estimator,
                views[view],
                labels,
                participants,
                training,
                seed=seed + 9_001 + 101 * outer_index + key_index,
                n_jobs=n_jobs,
            )
            outer_expert_by_key[(view, estimator)] = _gravity_probability(
                outer_base, model, views[view][evaluation]
            )
        selected_expert_key = (
            str(inner.selected_expert_candidate["expert_view"]),
            str(inner.selected_expert_candidate["posture_estimator"]),
        )
        inner_selected_expert = inner.expert_by_key[selected_expert_key][training]
        outer_selected_expert = outer_expert_by_key[selected_expert_key]
        inner_ctgr = _candidate_probability(
            inner.base[training],
            {key: value[training] for key, value in inner.expert_by_key.items()},
            inner.selected_candidate,
        )
        outer_ctgr = _candidate_probability(
            outer_base, outer_expert_by_key, inner.selected_candidate
        )
        inner_stack = np.stack(
            [
                _candidate_probability(
                    inner.base[training],
                    {key: value[training] for key, value in inner.expert_by_key.items()},
                    candidate,
                )
                for candidate in inner.top_three_expert_candidates
            ],
            axis=0,
        )
        outer_stack = np.stack(
            [
                _candidate_probability(outer_base, outer_expert_by_key, candidate)
                for candidate in inner.top_three_expert_candidates
            ],
            axis=0,
        )
        inner_equal = np.mean(inner_stack, axis=0)
        outer_equal = np.mean(outer_stack, axis=0)

        context, cage_reliability, channel_scale = _outer_context(
            signals,
            gravity,
            training,
            sampling_rate_hz=data.sampling_rate_hz,
        )
        all_base = np.concatenate((inner.base[training], outer_base), axis=0)
        all_expert = np.concatenate((inner_selected_expert, outer_selected_expert), axis=0)
        all_ctgr = np.concatenate((inner_ctgr, outer_ctgr), axis=0)
        all_context = np.concatenate((context[training], context[evaluation]), axis=0)
        all_reliability = np.concatenate(
            (cage_reliability[training], cage_reliability[evaluation]), axis=0
        )
        cage_methods, cage_diagnostics, cage_arrays = evaluate_cage_outer_arrays(
            training_base_probability=inner.base[training],
            training_expert_probability=inner_selected_expert,
            training_labels=labels[training],
            training_participants=participants[training],
            training_context=context[training],
            training_reliability=cage_reliability[training],
            evaluation_base_probability=all_base,
            evaluation_expert_probability=all_expert,
            evaluation_ctgr_probability=all_ctgr,
            evaluation_context=all_context,
            evaluation_reliability=all_reliability,
            expert_name=str(inner.selected_expert_candidate["id"]),
            cage_config=cage_config,
        )
        training_count = int(training.sum())
        broad_training = cage_methods["global_trust_blend"][:training_count]
        broad_evaluation = cage_methods["global_trust_blend"][training_count:]
        cage_evaluation = cage_methods["cage_har"][training_count:]

        physics_reference = fit_physics_reference(
            _slice_context(kinematic, training),
            veto_quantile=float(v1_fixed["physics_veto_quantile"]),
            minimum_valid_pair_fraction=float(v1_fixed["physics_minimum_valid_pair_fraction"]),
        )
        physics_training = apply_physics_reference(
            _slice_context(kinematic, training), physics_reference
        )
        physics_evaluation = apply_physics_reference(
            _slice_context(kinematic, evaluation), physics_reference
        )
        reliability_training = np.minimum(cage_reliability[training], physics_training.reliability)
        reliability_evaluation = np.minimum(
            cage_reliability[evaluation], physics_evaluation.reliability
        )
        strict_selection, strict_candidates = _select_posture_calibration(
            inner_equal,
            labels[training],
            participants[training],
            v1_fixed,
        )
        strict_pre_veto = _apply_calibration(outer_equal, strict_selection)
        strict, strict_vetoed = apply_physics_veto(
            outer_base, strict_pre_veto, physics_evaluation.trusted
        )
        training_signatures = build_responder_signatures(
            inner.base[training],
            inner_selected_expert,
            inner_ctgr,
            broad_training,
            participants[training],
            gravity_reliability=reliability_training,
            kinematic_context=_slice_context(kinematic, training),
            physics_trusted=physics_training.trusted,
            low_confidence_threshold=float(v1_fixed["responder_low_confidence_threshold"]),
        )
        evaluation_signatures = build_responder_signatures(
            outer_base,
            outer_selected_expert,
            outer_ctgr,
            broad_evaluation,
            participants[evaluation],
            gravity_reliability=reliability_evaluation,
            kinematic_context=_slice_context(kinematic, evaluation),
            physics_trusted=physics_evaluation.trusted,
            low_confidence_threshold=float(v1_fixed["responder_low_confidence_threshold"]),
        )
        base_utility, broad_utility = _utility_targets(
            labels[training],
            participants[training],
            inner.base[training],
            inner_ctgr,
            broad_training,
            training_signatures.participant_ids,
        )
        safe_route = _controller(
            training_features=training_signatures.features,
            training_ids=training_signatures.participant_ids,
            base_utility=base_utility,
            broad_utility=broad_utility,
            evaluation_features=evaluation_signatures.features,
            evaluation_signature_ids=evaluation_signatures.participant_ids,
            evaluation_base=outer_base,
            evaluation_pulse=outer_ctgr,
            evaluation_broad=broad_evaluation,
            evaluation_participants=participants[evaluation],
            physics_trusted=physics_evaluation.trusted,
            fixed=v1_fixed,
            lower_bound_z=float(v1_fixed["responder_safe_lower_bound_z"]),
        )
        meta_probability, meta_state, meta_veto = _meta_training_controller(
            signatures=training_signatures.features,
            signature_ids=training_signatures.participant_ids,
            base_utility=base_utility,
            broad_utility=broad_utility,
            base=inner.base[training],
            pulse=inner_ctgr,
            broad=broad_training,
            participants=participants[training],
            physics_trusted=physics_training.trusted,
            fixed=v1_fixed,
        )
        meta_restore = (meta_state == 0) | meta_veto
        full_selection, full_candidates = _select_posture_calibration(
            meta_probability,
            labels[training],
            participants[training],
            v1_fixed,
            base_probability=inner.base[training],
            restore_mask=meta_restore,
        )
        full_restore = (safe_route.state_index == 0) | safe_route.physics_vetoed
        v1_full = _apply_calibration(
            safe_route.probabilities,
            full_selection,
            base_probability=outer_base,
            restore_mask=full_restore,
        )

        _inner_equal_v2, inner_weighted, inner_disagreement = _marginalize(inner_stack, v2_fixed)
        _outer_equal_v2, outer_weighted, outer_disagreement = _marginalize(outer_stack, v2_fixed)
        temperature, temperature_candidates = _select_temperature(
            inner_weighted, labels[training], participants[training], v2_fixed
        )
        inner_temperature = temperature_scale_probabilities(inner_weighted, temperature=temperature)
        outer_temperature = temperature_scale_probabilities(outer_weighted, temperature=temperature)
        offset, offset_candidates = _select_offset(
            inner_temperature, labels[training], participants[training], v2_fixed
        )
        inner_core, _ = apply_rank_locked_posture_offset(
            inner_temperature, posture_logit_offset=offset
        )
        outer_core, _ = apply_rank_locked_posture_offset(
            outer_temperature, posture_logit_offset=offset
        )
        dual_model = _fit_expert(
            "extra_trees_leaf3",
            dual.features,
            labels,
            participants,
            training,
            seed=seed + 17_001 + outer_index,
            n_jobs=n_jobs,
        )
        outer_dual_expert = compose_mobility_posture_probabilities(
            outer_base[:, 0], _positive_probability(dual_model, dual.features[evaluation])
        )
        dual_selection, dual_candidates = _select_dual_candidate(
            inner_core,
            inner.dual_expert[training],
            labels[training],
            participants[training],
            v2_fixed,
        )
        inner_dual = _apply_dual_candidate(inner_core, inner.dual_expert[training], dual_selection)
        outer_dual = _apply_dual_candidate(outer_core, outer_dual_expert, dual_selection)
        route_training_features, route_names = build_rescue_harm_features(
            inner_core,
            inner_dual,
            candidate_disagreement=inner_disagreement,
            physics_reliability=reliability_training,
            dual_frame_context=dual.compact_context[training],
        )
        route_evaluation_features, evaluation_route_names = build_rescue_harm_features(
            outer_core,
            outer_dual,
            candidate_disagreement=outer_disagreement,
            physics_reliability=reliability_evaluation,
            dual_frame_context=dual.compact_context[evaluation],
        )
        if route_names != evaluation_route_names:
            raise AssertionError("HERA-DG-v2 route feature order changed")
        v2_route, v2_selection, semantic_disagreements = _fit_route_lane(
            training_core=inner_core,
            training_candidate=inner_dual,
            training_features=route_training_features,
            training_labels=labels[training],
            training_participants=participants[training],
            training_reliability=reliability_training,
            training_trusted=physics_training.trusted,
            evaluation_core=outer_core,
            evaluation_candidate=outer_dual,
            evaluation_features=route_evaluation_features,
            evaluation_reliability=reliability_evaluation,
            evaluation_trusted=physics_evaluation.trusted,
            fixed=v2_fixed,
            use_physics=True,
            use_support=True,
        )

        fold_probability: dict[str, FloatArray] = {
            "RMRP-DG": outer_base,
            "CTGR-DG": outer_ctgr,
            "CAGE-DG": cage_evaluation,
            "CTGR-DG-top3-equal": outer_equal,
            "HERA-DG-strict": strict,
            "HERA-DG-context-safe": safe_route.probabilities,
            "HERA-DG-full": v1_full,
            "HERA-DG-v2-weighted": outer_weighted,
            "HERA-DG-v2-core": outer_core,
            "HERA-DG-v2-dual": outer_dual,
            "HERA-DG-v2-full": v2_route.probabilities,
        }
        if include_classical:
            fold_probability.update(
                _classical_outer_predictions(
                    data, training=training, evaluation=evaluation, seed=seed + outer_index
                )
            )
        for method, values in fold_probability.items():
            probability[method][evaluation] = values
        fold_records.append(
            {
                "outer_fold": outer_index,
                "training_participants": sorted(np.unique(participants[training]).tolist()),
                "evaluation_participants": sorted(evaluation_ids),
                "selected_ctgr_candidate": str(inner.selected_candidate["id"]),
                "selected_cage_expert": str(inner.selected_expert_candidate["id"]),
                "top_three_hera_candidates": [
                    str(item["id"]) for item in inner.top_three_expert_candidates
                ],
                "eligible_ctgr_candidate_count": len(inner.eligible_ids),
                "candidate_ranking": inner.candidate_ranking,
                "inner_folds": list(inner.folds),
                "cage_diagnostics": cage_diagnostics,
                "cage_evaluation_route_count": int(
                    np.sum(cage_arrays["cage_har_route_mask"][training_count:])
                ),
                "channel_scale_fit_on_outer_training": channel_scale.tolist(),
                "physics_reference_threshold": (
                    physics_reference.normalized_residual_p90_threshold
                ),
                "physics_evaluation_trusted_fraction": float(physics_evaluation.trusted.mean()),
                "hera_v1_strict_calibration": strict_selection,
                "hera_v1_strict_calibration_candidates": len(strict_candidates),
                "hera_v1_full_calibration": full_selection,
                "hera_v1_full_calibration_candidates": len(full_candidates),
                "hera_v2_temperature": temperature,
                "hera_v2_temperature_candidates": len(temperature_candidates),
                "hera_v2_posture_offset": offset,
                "hera_v2_posture_offset_candidates": len(offset_candidates),
                "hera_v2_dual_selection": dual_selection,
                "hera_v2_dual_candidates": len(dual_candidates),
                "hera_v2_route_selection": v2_selection,
                "hera_v2_semantic_training_disagreements": semantic_disagreements,
                "hera_v1_strict_veto_count": int(strict_vetoed.sum()),
                "outer_evaluation_labels_used_before_predictions_fixed": False,
            }
        )
    if any(not np.isfinite(values).all() for values in probability.values()):
        raise ValueError("outer OOF probability coverage is incomplete")
    return probability, {"seed": seed, "folds": fold_records}


def evaluate_external_development(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
    repository_root: Path,
    n_jobs: int = -1,
    include_classical: bool = True,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    """Run matched nested participant-exclusive development and return full evidence."""

    data.validate()
    if data.channel_lane != "derived-gravity-9ch" or data.class_names != CORE_CLASS_NAMES:
        raise ValueError("three-class invention runner requires the derived-gravity lane")
    participant_count = np.unique(data.participant_ids).size
    if participant_count < 12:
        raise ValueError("external invention development requires at least 12 participants")
    seed_probabilities: dict[int, dict[str, FloatArray]] = {}
    seed_records: list[dict[str, Any]] = []
    reports_by_seed: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        probabilities, seed_record = _evaluate_seed(
            data,
            seed=seed,
            outer_fold_count=5,
            inner_fold_count=4,
            n_jobs=n_jobs,
            repository_root=repository_root,
            include_classical=include_classical,
        )
        seed_probabilities[seed] = probabilities
        reports_by_seed[str(seed)] = {
            method: _report(data.labels, values, data.participant_ids)
            for method, values in probabilities.items()
        }
        seed_records.append(seed_record)
    ensemble = {
        method: np.mean(
            np.stack([seed_probabilities[seed][method] for seed in seeds], axis=0), axis=0
        )
        for method in seed_probabilities[seeds[0]]
    }
    reports = {
        method: _report(data.labels, values, data.participant_ids)
        for method, values in ensemble.items()
    }
    participant_oracle, participant_choices = _participant_oracle(
        data.labels,
        data.participant_ids,
        tuple(ensemble[name] for name in ("RMRP-DG", "CTGR-DG", "HERA-DG-full", "HERA-DG-v2-full")),
        ("RMRP-DG", "CTGR-DG", "HERA-DG-full", "HERA-DG-v2-full"),
    )
    window_oracle = _window_oracle(
        data.labels,
        tuple(ensemble[name] for name in ("RMRP-DG", "CTGR-DG", "HERA-DG-full", "HERA-DG-v2-full")),
    )
    diagnostics = {
        "participant_oracle_not_scientific_method": _report(
            data.labels, participant_oracle, data.participant_ids
        ),
        "participant_oracle_choices": participant_choices,
        "window_oracle_not_scientific_method": _report(
            data.labels, window_oracle, data.participant_ids
        ),
    }
    comparison_methods = (
        "CTGR-DG",
        "CAGE-DG",
        "CTGR-DG-top3-equal",
        "HERA-DG-strict",
        "HERA-DG-full",
        "HERA-DG-v2-dual",
        "HERA-DG-v2-full",
    )
    comparisons = {
        method: _paired_bootstrap(reports[method], reports["RMRP-DG"], seed=20260904 + index)
        for index, method in enumerate(comparison_methods)
    }
    v2 = comparisons["HERA-DG-v2-full"]
    advancement_gate = {
        "method": "HERA-DG-v2-full",
        "comparator": "RMRP-DG",
        "minimum_mean_gain": 0.020,
        "observed_mean_gain": v2["mean_difference"],
        "mean_gain_passed": float(v2["mean_difference"]) >= 0.020,
        "mean_interval_lower_bound_above_zero": float(
            v2["mean_difference_95_percent_bootstrap_interval"][0]
        )
        > 0.0,
        "bottom_interval_lower_bound_above_zero": float(
            v2["bottom_30_percent_difference_95_percent_bootstrap_interval"][0]
        )
        > 0.0,
        "overall_passed": bool(
            float(v2["mean_difference"]) >= 0.020
            and float(v2["mean_difference_95_percent_bootstrap_interval"][0]) > 0.0
            and float(v2["bottom_30_percent_difference_95_percent_bootstrap_interval"][0]) > 0.0
        ),
        "interpretation": "development advancement only; never a confirmatory or SOTA gate",
    }
    best_invention = max(
        _INVENTION_METHODS,
        key=lambda name: float(reports[name]["primary"]["mean_participant_macro_f1"]),
    )
    available_controls = [name for name in _CLASSICAL_METHODS if name in reports]
    strongest_control = (
        max(
            available_controls,
            key=lambda name: float(reports[name]["primary"]["mean_participant_macro_f1"]),
        )
        if available_controls
        else None
    )
    benchmark_comparison = (
        None
        if strongest_control is None
        else {
            "best_invention": best_invention,
            "strongest_control": strongest_control,
            "paired_participant_bootstrap": _paired_bootstrap(
                reports[best_invention], reports[strongest_control], seed=20261004
            ),
            "selection_note": (
                "descriptive post-evaluation best-method comparison; not a multiplicity-"
                "controlled superiority test"
            ),
        }
    )
    result = {
        "schema_version": "1.0.0",
        "experiment_id": "cross-dataset-har-rnd-v1",
        "dataset": data.summary(),
        "evidence_status": "EXTERNAL_DEVELOPMENT_NOT_CONFIRMATORY",
        "method_lane": "derived-gravity adaptations",
        "seeds": list(seeds),
        "reports": reports,
        "reports_by_seed": reports_by_seed,
        "paired_participant_bootstrap": comparisons,
        "development_advancement_gate": advancement_gate,
        "descriptive_benchmark_comparison": benchmark_comparison,
        "diagnostics": diagnostics,
        "fold_records": seed_records,
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "native_and_derived_lane_pooling_allowed": False,
            "wear_gait_opened": False,
            "outer_labels_used_before_prediction_freeze": False,
        },
    }
    return result, ensemble


def _write_json_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "worktree_dirty": bool(status), "status_entries": status}


def _source_input_manifest(repository_root: Path) -> dict[str, Any]:
    """Hash executable source and the declared external protocol at run start."""

    candidates = list((repository_root / "src").rglob("*.py"))
    candidates.extend(
        repository_root / relative
        for relative in (
            "configs/datasets/external_har_portfolio_v1.yaml",
            "configs/experiments/cross_dataset_har_rnd_v1.yaml",
            "docs/research/CROSS_DATASET_HAR_RND_V1_PROTOCOL.md",
            "pyproject.toml",
            "uv.lock",
            "requirements/external-har-research.in",
            "requirements/external-har-research.lock",
        )
    )
    files = {
        path.relative_to(repository_root).as_posix(): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }
    return {
        "file_count": len(files),
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
        "captured_at": datetime.now(UTC).isoformat(),
    }


def run_and_write(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    n_jobs: int,
    include_classical: bool,
) -> dict[str, Any]:
    """Execute the run in a create-only directory and retain success or failure evidence."""

    git_at_launch = _git_state(repository_root)
    source_input_manifest = _source_input_manifest(repository_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "dataset": data.summary(),
        "source_receipts": [item.to_dict() for item in data.receipts],
        "storage_disclosure": {
            "raw_local_mirror": False,
            "processing": "streamed provider bytes and in-memory materialization",
        },
        "source_input_manifest": source_input_manifest,
        "git_at_launch": git_at_launch,
    }
    _write_json_create_only(output_directory / "data_audit.json", audit)
    try:
        result, predictions = evaluate_external_development(
            data,
            seeds=seeds,
            repository_root=repository_root,
            n_jobs=n_jobs,
            include_classical=include_classical,
        )
        prediction_path = output_directory / "predictions.npz"
        if prediction_path.exists():
            raise FileExistsError(prediction_path)
        np.savez_compressed(
            prediction_path,
            labels=data.labels,
            participant_ids=data.participant_ids,
            session_ids=data.session_ids,
            trial_ids=data.trial_ids,
            window_ids=data.window_ids,
            **{  # type: ignore[arg-type]
                f"probability__{name}": value for name, value in predictions.items()
            },
        )
        result["created_at"] = datetime.now(UTC).isoformat()
        result["started_at"] = started
        result["git"] = _git_state(repository_root)
        result["git_at_launch"] = git_at_launch
        result["source_input_manifest"] = source_input_manifest
        result["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
        result["inputs"] = {
            "portfolio_config": {
                "path": "configs/datasets/external_har_portfolio_v1.yaml",
                "sha256": sha256_file(
                    repository_root / "configs/datasets/external_har_portfolio_v1.yaml"
                ),
            },
            "experiment_config": {
                "path": "configs/experiments/cross_dataset_har_rnd_v1.yaml",
                "sha256": sha256_file(
                    repository_root / "configs/experiments/cross_dataset_har_rnd_v1.yaml"
                ),
            },
        }
        result["prediction_artifact"] = {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        }
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output_directory / "result.json", result)
        return result
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_PRESERVED",
            "started_at": started,
            "failed_at": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "git": _git_state(repository_root),
            "git_at_launch": git_at_launch,
        }
        _write_json_create_only(output_directory / "failure.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fog-star", "imu-har-il"), required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--participant-limit", type=int)
    parser.add_argument("--repetition-limit", type=int)
    parser.add_argument(
        "--gravity-cutoff-hz",
        type=float,
        help="Declared derived-gravity sensitivity value; defaults to the primary cutoff.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-classical", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    experiment_config = _read_mapping(root / "configs/experiments/cross_dataset_har_rnd_v1.yaml")
    preprocessing = cast(dict[str, Any], experiment_config["preprocessing"])
    derived = cast(dict[str, Any], preprocessing["derived_gravity"])
    declared_cutoffs = tuple(float(value) for value in derived["cutoff_sensitivity_hz"])
    gravity_cutoff_hz = (
        float(derived["cutoff_hz"])
        if args.gravity_cutoff_hz is None
        else float(args.gravity_cutoff_hz)
    )
    if not any(np.isclose(gravity_cutoff_hz, value) for value in declared_cutoffs):
        raise ValueError(
            f"gravity cutoff {gravity_cutoff_hz} is outside the declared sensitivity set "
            f"{declared_cutoffs}"
        )
    common = {
        "participant_limit": args.participant_limit,
        "target_rate_hz": float(preprocessing["target_sampling_rate_hz"]),
        "window_samples": int(preprocessing["window_samples"]),
        "gravity_cutoff_hz": gravity_cutoff_hz,
    }
    if args.dataset == "fog-star":
        data = load_fog_star(**common)
    else:
        data = load_imu_har_il(
            **common,
            repetition_limit=args.repetition_limit,
        )
    result = run_and_write(
        data=data,
        output_directory=args.output_directory.resolve(),
        repository_root=root,
        seeds=tuple(args.seeds),
        n_jobs=args.n_jobs,
        include_classical=not args.skip_classical,
    )
    summary = {
        name: values["primary"]["mean_participant_macro_f1"]
        for name, values in cast(dict[str, dict[str, Any]], result["reports"]).items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
