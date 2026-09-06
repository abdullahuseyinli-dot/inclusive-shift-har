"""Explicit per-method inference units; no annotation-selected batch masquerades as a window."""

from __future__ import annotations

from typing import Any

PARTICIPANT_CONTEXT_METHODS = frozenset({"HERA-DG-context-safe", "HERA-DG-full"})
ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS = frozenset(
    {
        "CAGE-DG",
        "HERA-DG-context-safe",
        "HERA-DG-full",
        "HERA-DG-strict",
        "HERA-DG-v2-full",
    }
)
OBSERVABLE_CONTEXT_PROTOCOL = "external-har-observable-context-v1"
OBSERVABLE_CONTEXT_TRAINING_PROTOCOL = "external-har-observable-context-training-population-v1"
SOLE_OBSERVABLE_SESSION_TEMPORAL_PROTOCOL = "sole-harmony-observable-session-temporal-v1"


def method_inference_contracts(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = result.get("target_dataset", result.get("dataset", {}))
    dataset_id = dataset.get("dataset_id")
    methods = result.get("primary_seed_averaged", {}).get("methods", result.get("reports", {}))
    output: dict[str, dict[str, Any]] = {}
    for name in methods:
        context = name in PARTICIPANT_CONTEXT_METHODS
        context_fit = name in ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS
        unit, population, status = "independent_fixed_window", "one_window", "window_local"
        annotation_selected_context = False
        annotation_selected_training_context = False
        if context_fit and dataset_id == "fog_star_v3":
            annotation_selected_training_context = (
                result.get("observable_context_training_protocol")
                != OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
            )
            if annotation_selected_training_context:
                status = "diagnostic_annotation_selected_training_context"
        if context:
            if dataset_id == "fog_star_v3":
                observable_evaluation = (
                    result.get("observable_context_protocol") == OBSERVABLE_CONTEXT_PROTOCOL
                    and dataset.get("observable_candidate_pool", {}).get("protocol_id")
                    == OBSERVABLE_CONTEXT_PROTOCOL
                )
                observable_training = (
                    result.get("observable_context_training_protocol")
                    == OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
                )
                if observable_evaluation and observable_training:
                    population = "all_observable_candidates_of_inner_and_outer_heldout_participants"
                elif observable_evaluation:
                    population = (
                        "outer_evaluation_candidates_complete_but_training_context_"
                        "annotation_selected"
                    )
                else:
                    population = "annotation_admitted_windows_only"
                unit = "noncausal_participant_batch"
                annotation_selected_context = not observable_evaluation
                status = (
                    "development_noncausal_context"
                    if observable_evaluation and observable_training
                    else "diagnostic_annotation_selected_context"
                )
            else:
                unit, population, status = (
                    "noncausal_participant_batch",
                    "declared_selected_scripted_trials",
                    "development_scripted_trial_context_not_natural_stream",
                )
        if dataset_id == "sole_harmony_v1":
            boundary = dataset.get("boundary_provenance", {})
            session_observable = (
                isinstance(boundary, dict)
                and boundary.get("boundary_mode") == "session_observable"
                and boundary.get("repository_signal_grid_annotation_independent") is True
                and result.get("temporal_contract", {}).get("protocol_id")
                == SOLE_OBSERVABLE_SESSION_TEMPORAL_PROTOCOL
            )
            if session_observable:
                temporal = name != "XGBoost-6ch-unsmoothed"
                unit = "causal_observable_physical_run_history" if temporal else unit
                population = "all_observable_session_candidates_before_post_hoc_scoring_eligibility"
                status = "development_session_observable_offline_resampling"
                annotation_selected_context = False
            else:
                unit, population, status = (
                    "camera_bout_oracle_pipeline",
                    "camera_annotated_bouts",
                    "oracle_diagnostic",
                )
                annotation_selected_context = True
        channels = (
            6
            if name.endswith("-6ch")
            or name.startswith("XGBoost-6ch")
            or name == "RMRP-DG"
            or name.startswith(("HPF-", "PB-HPF"))
            else 9
            if name.endswith(("-N9", "-D9")) or "-DG" in name
            else result.get("input_channels")
        )
        output[name] = {
            "inference_unit": unit,
            "context_population": population,
            "annotation_selected_evaluation_context": annotation_selected_context,
            "annotation_selected_training_context": annotation_selected_training_context,
            "scientific_role": status,
            "input_channel_count": channels,
            "dataset_channel_lane": dataset.get("channel_lane"),
            "zero_lookahead_streaming_claim_allowed": False,
            "differences_vs_independent_six_channel_control": (
                ([] if channels == 6 else ["input_channels"])
                + (
                    []
                    if unit == "independent_fixed_window"
                    else ["inference_unit_and_context_budget"]
                )
            ),
            "scope": "Inference/channel descriptors only; dataset, split, supervision, seeds and metrics require their own comparison checks.",
        }
    return output


def annotation_selected_context_methods(result: dict[str, Any]) -> list[str]:
    return sorted(
        name
        for name, contract in method_inference_contracts(result).items()
        if contract["annotation_selected_evaluation_context"]
        or contract["annotation_selected_training_context"]
    )
