"""Prospective CAGE-HAR development and synthetic contract runner.

Real evaluation accepts only participant-exclusive out-of-fold predictions from
a genuinely new development cohort.  The locked InclusiveHAR participants and
consumed DAGHAR domains fail closed at the bundle-manifest boundary.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_nested import _lower_fraction_mean
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.cage_har import (
    CageRoutingResult,
    apply_cage_trust_region,
    build_advantage_router_features,
    counterfactual_log_loss_advantage,
    participant_jackknife_advantage_prediction,
)
from inclusive_shift_har.protocols.cage_cohort import load_cage_cohort_manifest

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

_CLASS_NAMES = ("mobility", "sitting", "standing")
_METHOD_NAMES = (
    "base_uncalibrated",
    "base_decision_repaired",
    "global_trust_blend",
    "confidence_trust_gate",
    "disagreement_trust_gate",
    "cage_har",
)


@dataclass(frozen=True, slots=True)
class CageDevelopmentBundle:
    """Labelled new-development OOF predictions and label-free router context."""

    base_probability: FloatArray
    expert_probabilities: FloatArray
    labels: IntArray
    participant_ids: StringArray
    window_ids: StringArray
    context_features: FloatArray
    expert_reliability: FloatArray
    expert_names: tuple[str, ...]
    expert_posture_only: BoolArray


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("status") != "implemented_awaiting_new_development_cohort":
        raise ValueError("CAGE-HAR configuration status changed")
    input_contract = _mapping(config.get("input_contract"), name="input contract")
    if tuple(input_contract.get("class_names", [])) != _CLASS_NAMES:
        raise ValueError("CAGE-HAR class order changed")
    if int(input_contract.get("required_channel_count", -1)) != 9:
        raise ValueError("CAGE-HAR requires the separate nine-channel track")
    if input_contract.get("predictions_must_be_participant_exclusive_oof") is not True:
        raise ValueError("CAGE-HAR requires participant-exclusive OOF inputs")
    forbidden = tuple(str(item) for item in input_contract.get("forbidden_participant_ids", []))
    if forbidden != tuple(str(index) for index in range(1, 21)):
        raise ValueError("CAGE-HAR consumed-participant denial list changed")
    repair = _mapping(config.get("decision_repair"), name="decision repair")
    temperatures = tuple(float(item) for item in repair.get("temperatures", []))
    offsets = tuple(float(item) for item in repair.get("posture_logit_offsets", []))
    if 1.0 not in temperatures or 0.0 not in offsets:
        raise ValueError("decision repair must include the identity candidate")
    router = _mapping(config.get("router"), name="router")
    required_router = {
        "ridge_penalty",
        "lower_bound_z",
        "minimum_advantage",
        "minimum_reliability",
        "maximum_mix_weight",
        "advantage_scale",
        "maximum_kl",
        "harm_weight_multiplier",
        "bottom_tail_fraction",
        "bottom_tail_weight_multiplier",
    }
    if set(router) != required_router:
        raise ValueError("CAGE-HAR router settings changed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("confirmatory_claim_allowed") is not False
        or policy.get("locked_inclusivehar_reuse_allowed") is not False
        or policy.get("consumed_daghar_reuse_allowed") is not False
        or policy.get("synthetic_performance_claim_allowed") is not False
    ):
        raise ValueError("CAGE-HAR evidence boundary changed")
    return config


def apply_probability_repair(
    probabilities: NDArray[np.floating[Any]],
    *,
    temperature: float,
    posture_logit_offset: float,
) -> FloatArray:
    """Apply a global temperature and antisymmetric sitting/standing offset."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or values.shape[0] == 0:
        raise ValueError("probability repair requires [sample,3] probabilities")
    if (
        not np.isfinite(values).all()
        or np.any(values < 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("probability repair input is invalid")
    if not np.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be finite and positive")
    if not np.isfinite(posture_logit_offset):
        raise ValueError("posture logit offset must be finite")
    logits = np.log(np.clip(values, 1e-12, 1.0)) / temperature
    logits[:, 1] += 0.5 * posture_logit_offset
    logits[:, 2] -= 0.5 * posture_logit_offset
    logits -= logits.max(axis=1, keepdims=True)
    result = np.exp(logits)
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def _add_bottom_tail(report: dict[str, Any]) -> None:
    values = [
        float(item["macro_f1"]) for item in cast(list[dict[str, Any]], report["participants"])
    ]
    report["primary"]["bottom_30_percent_participant_macro_f1"] = _lower_fraction_mean(values)


def _report(labels: IntArray, probability: FloatArray, participants: StringArray) -> dict[str, Any]:
    report = classification_report(
        labels,
        probability,
        participants.tolist(),
        class_names=_CLASS_NAMES,
    )
    _add_bottom_tail(report)
    return report


def _select_probability_repair(
    probabilities: FloatArray,
    labels: IntArray,
    participants: StringArray,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[dict[str, float]]]:
    repair = cast(dict[str, Any], config["decision_repair"])
    summaries: list[dict[str, float]] = []
    for temperature in cast(list[float], repair["temperatures"]):
        for offset in cast(list[float], repair["posture_logit_offsets"]):
            candidate = apply_probability_repair(
                probabilities,
                temperature=float(temperature),
                posture_logit_offset=float(offset),
            )
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
    tolerance = float(repair["mean_tie_tolerance"])
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


def _apply_selected_repair(probabilities: FloatArray, selection: dict[str, float]) -> FloatArray:
    return apply_probability_repair(
        probabilities,
        temperature=float(selection["temperature"]),
        posture_logit_offset=float(selection["posture_logit_offset"]),
    )


def _tail_and_harm_weights(
    base_probability: FloatArray,
    expert_probability: FloatArray,
    labels: IntArray,
    participants: StringArray,
    *,
    bottom_tail_fraction: float,
    bottom_tail_multiplier: float,
    harm_multiplier: float,
) -> FloatArray:
    if not 0.0 < bottom_tail_fraction <= 1.0:
        raise ValueError("bottom-tail fraction must lie in (0,1]")
    if bottom_tail_multiplier < 1.0 or harm_multiplier < 1.0:
        raise ValueError("tail and harm multipliers must be at least one")
    base_report = _report(labels, base_probability, participants)
    participant_rows = cast(list[dict[str, Any]], base_report["participants"])
    count = max(1, int(np.ceil(bottom_tail_fraction * len(participant_rows))))
    tail = {
        str(item["participant_id"])
        for item in sorted(participant_rows, key=lambda item: float(item["macro_f1"]))[:count]
    }
    weights = np.ones(labels.size, dtype=np.float64)
    weights[np.isin(participants, list(tail))] *= bottom_tail_multiplier
    base_label = base_probability.argmax(axis=1)
    expert_label = expert_probability.argmax(axis=1)
    harm = (base_label == labels) & (expert_label != labels)
    weights[harm] *= harm_multiplier
    return weights


def _fixed_gate(
    base_probability: FloatArray,
    expert_probabilities: FloatArray,
    expert_reliability: FloatArray,
    expert_posture_only: BoolArray,
    *,
    selected_expert: int,
    gate: BoolArray,
    blend_weight: float,
    minimum_reliability: float,
    maximum_kl: float,
) -> CageRoutingResult:
    bounds = np.full(expert_reliability.shape, -1.0, dtype=np.float64)
    bounds[gate, selected_expert] = 1.0
    return apply_cage_trust_region(
        base_probability,
        expert_probabilities,
        bounds,
        expert_reliability,
        expert_posture_only,
        minimum_advantage=0.0,
        minimum_reliability=minimum_reliability,
        maximum_mix_weight=blend_weight,
        advantage_scale=1.0,
        maximum_kl=maximum_kl,
    )


def _routing_audit(
    labels: IntArray,
    participants: StringArray,
    base_probability: FloatArray,
    routed_probability: FloatArray,
    routed: BoolArray,
    chosen_expert: IntArray,
    expert_names: tuple[str, ...],
) -> dict[str, Any]:
    base_label = base_probability.argmax(axis=1)
    routed_label = routed_probability.argmax(axis=1)
    changed = base_label != routed_label
    rescue = changed & (base_label != labels) & (routed_label == labels)
    harm = changed & (base_label == labels) & (routed_label != labels)
    both_wrong = changed & (base_label != labels) & (routed_label != labels)
    decisive = int(rescue.sum() + harm.sum())
    changed_count = int(changed.sum())
    per_participant: list[dict[str, Any]] = []
    for participant in np.unique(participants):
        mask = participants == participant
        per_participant.append(
            {
                "participant_id": str(participant),
                "routed": int(np.sum(routed & mask)),
                "changed": int(np.sum(changed & mask)),
                "rescues": int(np.sum(rescue & mask)),
                "harms": int(np.sum(harm & mask)),
                "net_rescues": int(np.sum(rescue & mask) - np.sum(harm & mask)),
            }
        )
    return {
        "routed_count": int(routed.sum()),
        "route_fraction": float(routed.mean()),
        "changed_count": changed_count,
        "changed_fraction": float(changed.mean()),
        "rescue_count": int(rescue.sum()),
        "harm_count": int(harm.sum()),
        "both_wrong_changed_count": int(both_wrong.sum()),
        "net_rescues": int(rescue.sum() - harm.sum()),
        "intervention_precision_rescue_over_rescue_plus_harm": (
            None if decisive == 0 else float(rescue.sum() / decisive)
        ),
        "harmful_changed_fraction": (
            None if changed_count == 0 else float(harm.sum() / changed_count)
        ),
        "chosen_expert_counts": {
            name: int(np.sum(chosen_expert == index)) for index, name in enumerate(expert_names)
        },
        "per_participant": per_participant,
    }


def _validate_bundle(bundle: CageDevelopmentBundle, *, minimum_participants: int) -> None:
    base = bundle.base_probability
    experts = bundle.expert_probabilities
    labels = bundle.labels
    participants = bundle.participant_ids
    windows = bundle.window_ids
    if base.ndim != 2 or base.shape[1] != 3 or base.shape[0] == 0:
        raise ValueError("CAGE bundle base probabilities must be [sample,3]")
    if experts.ndim != 3 or experts.shape[0] != base.shape[0] or experts.shape[2] != 3:
        raise ValueError("CAGE bundle expert probabilities must be [sample,expert,3]")
    if (
        labels.shape != (base.shape[0],)
        or participants.shape != labels.shape
        or windows.shape != labels.shape
        or bundle.context_features.ndim != 2
        or bundle.context_features.shape[0] != labels.size
        or bundle.expert_reliability.shape != experts.shape[:2]
        or bundle.expert_posture_only.shape != (experts.shape[1],)
        or len(bundle.expert_names) != experts.shape[1]
    ):
        raise ValueError("CAGE bundle arrays are not aligned")
    if (
        not np.isfinite(base).all()
        or not np.isfinite(experts).all()
        or not np.isfinite(bundle.context_features).all()
        or not np.isfinite(bundle.expert_reliability).all()
        or np.any(base < 0.0)
        or np.any(experts < 0.0)
        or not np.allclose(base.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6)
        or not np.allclose(experts.sum(axis=2), 1.0, atol=1e-6, rtol=1e-6)
        or np.any((bundle.expert_reliability < 0.0) | (bundle.expert_reliability > 1.0))
    ):
        raise ValueError("CAGE bundle contains invalid probability/context values")
    if labels.min() < 0 or labels.max() > 2 or set(labels.tolist()) != {0, 1, 2}:
        raise ValueError("CAGE bundle must contain all three locked classes")
    if len(set(windows.tolist())) != windows.size:
        raise ValueError("CAGE bundle window identifiers must be unique")
    if np.unique(participants).size < minimum_participants:
        raise ValueError("CAGE bundle has too few independent participants")
    if len(set(bundle.expert_names)) != len(bundle.expert_names):
        raise ValueError("CAGE expert names must be unique")


def run_cage_cross_fitted_arrays(
    bundle: CageDevelopmentBundle,
    *,
    config: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, NDArray[Any]]]:
    """Evaluate fixed CAGE mechanics by leave-one-participant-out cross-fitting."""

    input_contract = cast(dict[str, Any], config["input_contract"])
    _validate_bundle(bundle, minimum_participants=int(input_contract["minimum_participants"]))
    router = cast(dict[str, Any], config["router"])
    ablations = cast(dict[str, Any], config["ablations"])
    methods: dict[str, FloatArray] = {
        name: np.empty_like(bundle.base_probability) for name in _METHOD_NAMES
    }
    route_masks: dict[str, BoolArray] = {
        name: np.zeros(bundle.labels.size, dtype=np.bool_)
        for name in _METHOD_NAMES
        if name not in {"base_uncalibrated", "base_decision_repaired"}
    }
    chosen_experts: dict[str, IntArray] = {
        name: np.full(bundle.labels.size, -1, dtype=np.int64) for name in route_masks
    }
    mix_weights = np.zeros(bundle.labels.size, dtype=np.float64)
    achieved_kl = np.zeros(bundle.labels.size, dtype=np.float64)
    advantage_lower_bound = np.full(bundle.labels.size, -np.inf, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    for held_participant in np.unique(bundle.participant_ids):
        evaluation = bundle.participant_ids == held_participant
        training = ~evaluation
        training_participants = bundle.participant_ids[training]
        base_selection, base_candidates = _select_probability_repair(
            bundle.base_probability[training],
            bundle.labels[training],
            training_participants,
            config,
        )
        repaired_base_training = _apply_selected_repair(
            bundle.base_probability[training], base_selection
        )
        repaired_base_evaluation = _apply_selected_repair(
            bundle.base_probability[evaluation], base_selection
        )
        repaired_expert_training = np.empty_like(bundle.expert_probabilities[training])
        repaired_expert_evaluation = np.empty_like(bundle.expert_probabilities[evaluation])
        expert_repair_records: list[dict[str, Any]] = []
        mean_advantages: list[float] = []
        lower_bounds: list[FloatArray] = []
        model_counts: list[int] = []
        for expert_index, expert_name in enumerate(bundle.expert_names):
            selection, candidates = _select_probability_repair(
                bundle.expert_probabilities[training, expert_index],
                bundle.labels[training],
                training_participants,
                config,
            )
            train_expert = _apply_selected_repair(
                bundle.expert_probabilities[training, expert_index], selection
            )
            evaluate_expert = _apply_selected_repair(
                bundle.expert_probabilities[evaluation, expert_index], selection
            )
            repaired_expert_training[:, expert_index] = train_expert
            repaired_expert_evaluation[:, expert_index] = evaluate_expert
            train_features, feature_names = build_advantage_router_features(
                repaired_base_training,
                train_expert,
                context_features=bundle.context_features[training],
                expert_reliability=bundle.expert_reliability[training, expert_index],
            )
            evaluation_features, evaluation_feature_names = build_advantage_router_features(
                repaired_base_evaluation,
                evaluate_expert,
                context_features=bundle.context_features[evaluation],
                expert_reliability=bundle.expert_reliability[evaluation, expert_index],
            )
            if feature_names != evaluation_feature_names:
                raise AssertionError("CAGE router feature order changed between partitions")
            advantage = counterfactual_log_loss_advantage(
                repaired_base_training,
                train_expert,
                bundle.labels[training],
            )
            sample_weights = _tail_and_harm_weights(
                repaired_base_training,
                train_expert,
                bundle.labels[training],
                training_participants,
                bottom_tail_fraction=float(router["bottom_tail_fraction"]),
                bottom_tail_multiplier=float(router["bottom_tail_weight_multiplier"]),
                harm_multiplier=float(router["harm_weight_multiplier"]),
            )
            advantage_prediction = participant_jackknife_advantage_prediction(
                train_features,
                advantage,
                training_participants,
                evaluation_features,
                ridge_penalty=float(router["ridge_penalty"]),
                lower_bound_z=float(router["lower_bound_z"]),
                training_sample_weight=sample_weights,
            )
            lower_bounds.append(advantage_prediction.lower_bound)
            model_counts.append(advantage_prediction.model_count)
            mean_advantages.append(float(np.mean(advantage)))
            expert_repair_records.append(
                {
                    "expert_name": expert_name,
                    "selection": selection,
                    "eligible_candidate_count": sum(
                        item["mean_participant_macro_f1"]
                        >= max(row["mean_participant_macro_f1"] for row in candidates)
                        - float(config["decision_repair"]["mean_tie_tolerance"])
                        for item in candidates
                    ),
                    "router_feature_count": len(feature_names),
                    "mean_training_counterfactual_advantage": float(np.mean(advantage)),
                    "jackknife_model_count": advantage_prediction.model_count,
                }
            )
        bound_matrix = np.column_stack(lower_bounds)
        cage = apply_cage_trust_region(
            repaired_base_evaluation,
            repaired_expert_evaluation,
            bound_matrix,
            bundle.expert_reliability[evaluation],
            bundle.expert_posture_only,
            minimum_advantage=float(router["minimum_advantage"]),
            minimum_reliability=float(router["minimum_reliability"]),
            maximum_mix_weight=float(router["maximum_mix_weight"]),
            advantage_scale=float(router["advantage_scale"]),
            maximum_kl=float(router["maximum_kl"]),
        )
        selected_expert = int(np.argmax(mean_advantages))
        all_rows = np.ones(int(evaluation.sum()), dtype=np.bool_)
        global_blend = _fixed_gate(
            repaired_base_evaluation,
            repaired_expert_evaluation,
            bundle.expert_reliability[evaluation],
            bundle.expert_posture_only,
            selected_expert=selected_expert,
            gate=all_rows,
            blend_weight=float(ablations["fixed_blend_weight"]),
            minimum_reliability=float(router["minimum_reliability"]),
            maximum_kl=float(router["maximum_kl"]),
        )
        confidence_blend = _fixed_gate(
            repaired_base_evaluation,
            repaired_expert_evaluation,
            bundle.expert_reliability[evaluation],
            bundle.expert_posture_only,
            selected_expert=selected_expert,
            gate=repaired_base_evaluation.max(axis=1) < float(ablations["confidence_threshold"]),
            blend_weight=float(ablations["fixed_blend_weight"]),
            minimum_reliability=float(router["minimum_reliability"]),
            maximum_kl=float(router["maximum_kl"]),
        )
        disagreement_blend = _fixed_gate(
            repaired_base_evaluation,
            repaired_expert_evaluation,
            bundle.expert_reliability[evaluation],
            bundle.expert_posture_only,
            selected_expert=selected_expert,
            gate=repaired_base_evaluation.argmax(axis=1)
            != repaired_expert_evaluation[:, selected_expert].argmax(axis=1),
            blend_weight=float(ablations["fixed_blend_weight"]),
            minimum_reliability=float(router["minimum_reliability"]),
            maximum_kl=float(router["maximum_kl"]),
        )
        methods["base_uncalibrated"][evaluation] = bundle.base_probability[evaluation]
        methods["base_decision_repaired"][evaluation] = repaired_base_evaluation
        route_results = {
            "global_trust_blend": global_blend,
            "confidence_trust_gate": confidence_blend,
            "disagreement_trust_gate": disagreement_blend,
            "cage_har": cage,
        }
        for method_name, route_result in route_results.items():
            methods[method_name][evaluation] = route_result.probabilities
            route_masks[method_name][evaluation] = route_result.routed
            chosen_experts[method_name][evaluation] = route_result.chosen_expert
        mix_weights[evaluation] = cage.mix_weight
        achieved_kl[evaluation] = cage.achieved_kl
        advantage_lower_bound[evaluation] = cage.predicted_advantage_lower_bound
        fold_records.append(
            {
                "held_participant_id": str(held_participant),
                "training_participant_count": int(np.unique(training_participants).size),
                "base_decision_repair": base_selection,
                "base_repair_candidate_count": len(base_candidates),
                "expert_repairs": expert_repair_records,
                "reference_expert": bundle.expert_names[selected_expert],
                "outer_labels_used_for_selection_or_training": False,
                "router_jackknife_model_counts": model_counts,
            }
        )

    reports = {
        name: _report(bundle.labels, probability, bundle.participant_ids)
        for name, probability in methods.items()
    }
    routing = {
        name: _routing_audit(
            bundle.labels,
            bundle.participant_ids,
            methods["base_decision_repaired"],
            methods[name],
            route_masks[name],
            chosen_experts[name],
            bundle.expert_names,
        )
        for name in route_masks
    }
    base_mean = float(reports["base_decision_repaired"]["primary"]["mean_participant_macro_f1"])
    comparisons = {
        name: {
            "mean_participant_macro_f1_difference_from_repaired_base": float(
                report["primary"]["mean_participant_macro_f1"] - base_mean
            ),
            "bottom_30_percent_difference_from_repaired_base": float(
                report["primary"]["bottom_30_percent_participant_macro_f1"]
                - reports["base_decision_repaired"]["primary"][
                    "bottom_30_percent_participant_macro_f1"
                ]
            ),
        }
        for name, report in reports.items()
    }
    result = {
        "folds": fold_records,
        "reports": reports,
        "comparisons": comparisons,
        "routing_audits": routing,
        "participant_count": int(np.unique(bundle.participant_ids).size),
        "window_count": int(bundle.labels.size),
        "expert_names": list(bundle.expert_names),
        "outer_participant_labels_used_for_router_or_repair": False,
        "participant_is_inference_unit": True,
    }
    prediction_payload: dict[str, NDArray[Any]] = {
        **{f"{name}_probabilities": probability for name, probability in methods.items()},
        **{f"{name}_route_mask": mask for name, mask in route_masks.items()},
        **{f"{name}_chosen_expert": values for name, values in chosen_experts.items()},
        "cage_mix_weight": mix_weights,
        "cage_achieved_kl": achieved_kl,
        "cage_predicted_advantage_lower_bound": advantage_lower_bound,
        "labels": bundle.labels,
        "participant_ids": bundle.participant_ids,
        "window_ids": bundle.window_ids,
    }
    return result, prediction_payload


def _load_bundle(path: Path) -> CageDevelopmentBundle:
    with np.load(path, allow_pickle=False) as loaded:
        return CageDevelopmentBundle(
            base_probability=np.asarray(loaded["base_probabilities"], dtype=np.float64),
            expert_probabilities=np.asarray(loaded["expert_probabilities"], dtype=np.float64),
            labels=np.asarray(loaded["labels"], dtype=np.int64),
            participant_ids=np.asarray(loaded["participant_ids"], dtype=np.str_),
            window_ids=np.asarray(loaded["window_ids"], dtype=np.str_),
            context_features=np.asarray(loaded["context_features"], dtype=np.float64),
            expert_reliability=np.asarray(loaded["expert_reliability"], dtype=np.float64),
            expert_names=tuple(np.asarray(loaded["expert_names"], dtype=np.str_).tolist()),
            expert_posture_only=np.asarray(loaded["expert_posture_only"], dtype=np.bool_),
        )


def _read_self_hashed_json(path: Path) -> dict[str, Any]:
    record = _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    claimed = record.pop("record_sha256", None)
    observed = canonical_json_sha256(record)
    record["record_sha256"] = claimed
    if not isinstance(claimed, str) or claimed != observed:
        raise ValueError(f"record self-hash mismatch: {path}")
    return record


def _load_guarded_bundle(
    manifest_path: Path,
    config: dict[str, Any],
) -> tuple[CageDevelopmentBundle, dict[str, Any]]:
    manifest = _read_self_hashed_json(manifest_path)
    if (
        manifest.get("record_kind") != "cage_har_cross_fitted_development_bundle"
        or manifest.get("status") != "complete_new_development_oof"
        or manifest.get("cohort_role") != "new_development"
        or int(manifest.get("channel_count", -1)) != 9
    ):
        raise PermissionError("CAGE-HAR accepts only a new nine-channel OOF development bundle")
    contract = _mapping(manifest.get("oof_contract"), name="bundle OOF contract")
    boundary = _mapping(manifest.get("evidence_boundary"), name="bundle evidence boundary")
    if (
        contract.get("base_predictions_participant_exclusive") is not True
        or contract.get("expert_predictions_participant_exclusive") is not True
        or contract.get("router_features_label_free") is not True
        or boundary.get("includes_inclusivehar_participants_1_through_20") is not False
        or boundary.get("includes_consumed_daghar_targets") is not False
        or boundary.get("target_or_confirmatory_cohort") is not False
    ):
        raise PermissionError("CAGE-HAR bundle violates the new-development evidence boundary")
    dataset_id = str(manifest.get("dataset_id", ""))
    forbidden_datasets = {str(item) for item in config["input_contract"]["forbidden_dataset_ids"]}
    if dataset_id in forbidden_datasets:
        raise PermissionError(f"CAGE-HAR refuses consumed dataset {dataset_id!r}")
    bundle_record = _mapping(manifest.get("bundle"), name="bundle reference")
    bundle_path = Path(str(bundle_record.get("path", "")))
    if not bundle_path.is_file() or sha256_file(bundle_path) != bundle_record.get("sha256"):
        raise ValueError("CAGE-HAR bundle is missing or changed")
    bundle = _load_bundle(bundle_path)
    forbidden_participants = {
        str(item) for item in config["input_contract"]["forbidden_participant_ids"]
    }
    overlap = sorted(set(bundle.participant_ids.tolist()) & forbidden_participants)
    if overlap:
        raise PermissionError(f"CAGE-HAR refuses consumed participant identifiers: {overlap}")
    declared_participants = sorted(str(item) for item in manifest.get("participant_ids", []))
    if declared_participants != sorted(set(bundle.participant_ids.tolist())):
        raise ValueError("CAGE-HAR bundle participants do not match its manifest")
    cohort_reference = _mapping(manifest.get("cohort_manifest"), name="cohort manifest reference")
    cohort_path = Path(str(cohort_reference.get("path", "")))
    if not cohort_path.is_file() or sha256_file(cohort_path) != cohort_reference.get("sha256"):
        raise ValueError("CAGE-HAR cohort manifest is missing or changed")
    cohort = load_cage_cohort_manifest(cohort_path)
    if (
        cohort["cohort_role"] != "new_development"
        or cohort["dataset_id"] != dataset_id
        or cohort["validated_participant_ids"] != declared_participants
    ):
        raise PermissionError("CAGE-HAR cohort and OOF bundle contracts do not align")
    return bundle, manifest


def _write_result(
    output_directory: Path,
    result: dict[str, Any],
    prediction_payload: dict[str, NDArray[Any]],
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"CAGE-HAR output already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    prediction_path = output_directory / "cross_fitted_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result["predictions"] = {
        "path": prediction_path.as_posix(),
        "sha256": sha256_file(prediction_path),
    }
    result["record_sha256"] = canonical_json_sha256(result)
    result_path = output_directory / "result.json"
    with result_path.open("xb") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return result


def run_cage_development(
    *,
    config_path: Path,
    bundle_manifest_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run CAGE only on a manifest-proven new OOF development bundle."""

    config = _load_config(config_path)
    bundle, manifest = _load_guarded_bundle(bundle_manifest_path, config)
    body, predictions = run_cage_cross_fitted_arrays(bundle, config=config)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "cage_har_new_development_cross_fitted_result",
        "status": "complete_new_development_not_confirmatory",
        "evidence_status": "new_development_hypothesis_selection_only",
        "code_commit": code_commit,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "bundle_manifest": {
            "path": bundle_manifest_path.as_posix(),
            "sha256": sha256_file(bundle_manifest_path),
            "record_sha256": manifest["record_sha256"],
        },
        **body,
        "locked_inclusivehar_participants_loaded": False,
        "consumed_daghar_targets_loaded": False,
        "confirmatory_claim_allowed": False,
        "independent_sealed_cohort_required": True,
    }
    return _write_result(output_directory, result, predictions)


def synthetic_cage_bundle(*, seed: int = 20260904) -> CageDevelopmentBundle:
    """Create a deterministic contract fixture; never use it as performance evidence."""

    generator = np.random.default_rng(seed)
    participant_count = 12
    samples_per_participant = 90
    sample_count = participant_count * samples_per_participant
    labels = np.tile(np.repeat(np.arange(3, dtype=np.int64), 30), participant_count)
    participants = np.repeat(
        np.asarray([f"synthetic-{index:02d}" for index in range(participant_count)]),
        samples_per_participant,
    )
    windows = np.asarray([f"synthetic-window-{index:05d}" for index in range(sample_count)])
    gauge_stability = generator.beta(3.0, 1.8, sample_count)
    fault_score = generator.beta(1.5, 4.0, sample_count)
    posture_signal = generator.normal(size=sample_count)
    base = np.full((sample_count, 3), 0.13, dtype=np.float64)
    base[np.arange(sample_count), labels] = 0.74
    base_error = (labels != 0) & (np.abs(posture_signal) < 0.42)
    fault_error = fault_score > 0.72
    for index in np.flatnonzero(base_error | fault_error):
        alternative = 2 if labels[index] == 1 else 1
        base[index, labels[index]] = 0.42
        base[index, alternative] = 0.45
    base += generator.normal(scale=0.012, size=base.shape)
    base = np.clip(base, 0.01, None)
    base /= base.sum(axis=1, keepdims=True)

    gravity_expert = base.copy()
    helpful_gravity = base_error & (gauge_stability > 0.52)
    harmful_gravity = (labels != 0) & ~base_error & (gauge_stability < 0.28)
    for index in np.flatnonzero(helpful_gravity):
        stationary_mass = 1.0 - base[index, 0]
        gravity_expert[index, labels[index]] = 0.88 * stationary_mass
        other = 2 if labels[index] == 1 else 1
        gravity_expert[index, other] = 0.12 * stationary_mass
    for index in np.flatnonzero(harmful_gravity):
        stationary_mass = 1.0 - base[index, 0]
        other = 2 if labels[index] == 1 else 1
        gravity_expert[index, labels[index]] = 0.20 * stationary_mass
        gravity_expert[index, other] = 0.80 * stationary_mass

    health_expert = base.copy()
    helpful_health = fault_error
    for index in np.flatnonzero(helpful_health):
        health_expert[index] = 0.075
        health_expert[index, labels[index]] = 0.85
    experts = np.stack((gravity_expert, health_expert), axis=1)
    context = np.column_stack(
        (
            gauge_stability,
            fault_score,
            posture_signal,
            np.abs(posture_signal),
            generator.normal(scale=0.2, size=sample_count),
        )
    )
    reliability = np.column_stack((gauge_stability, np.clip(0.55 + fault_score, 0.0, 1.0)))
    return CageDevelopmentBundle(
        base_probability=base,
        expert_probabilities=experts,
        labels=labels,
        participant_ids=np.asarray(participants, dtype=np.str_),
        window_ids=np.asarray(windows, dtype=np.str_),
        context_features=np.asarray(context, dtype=np.float64),
        expert_reliability=np.asarray(reliability, dtype=np.float64),
        expert_names=("gravity_posture", "health_fallback"),
        expert_posture_only=np.asarray([True, False], dtype=np.bool_),
    )


def run_cage_synthetic_smoke(
    *,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
    seed: int,
) -> dict[str, Any]:
    """Exercise every CAGE route using deterministic artificial data."""

    config = _load_config(config_path)
    bundle = synthetic_cage_bundle(seed=seed)
    body, predictions = run_cage_cross_fitted_arrays(bundle, config=config)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "cage_har_synthetic_contract_smoke",
        "status": "pass_synthetic_contract_only",
        "evidence_status": "synthetic_contract_test_not_scientific_performance",
        "seed": seed,
        "code_commit": code_commit,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        **body,
        "locked_inclusivehar_participants_loaded": False,
        "consumed_daghar_targets_loaded": False,
        "real_har_performance_claim_allowed": False,
        "confirmatory_claim_allowed": False,
    }
    return _write_result(output_directory, result, predictions)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--bundle-manifest", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--code-commit", required=True)
    smoke = subparsers.add_parser("synthetic-smoke")
    smoke.add_argument("--config", type=Path, required=True)
    smoke.add_argument("--output-directory", type=Path, required=True)
    smoke.add_argument("--code-commit", required=True)
    smoke.add_argument("--seed", type=int, default=20260904)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.operation == "run":
        result = run_cage_development(
            config_path=args.config,
            bundle_manifest_path=args.bundle_manifest,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
        )
    else:
        result = run_cage_synthetic_smoke(
            config_path=args.config,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
            seed=args.seed,
        )
    summaries = {name: report["primary"] for name, report in result["reports"].items()}
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_status": result["evidence_status"],
                "summaries": summaries,
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
