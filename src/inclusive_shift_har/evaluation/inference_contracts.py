"""Explicit per-method inference units; no annotation-selected batch masquerades as a window."""

from __future__ import annotations

from typing import Any

PARTICIPANT_CONTEXT_METHODS = frozenset({"HERA-DG-context-safe", "HERA-DG-full"})
OBSERVABLE_CONTEXT_PROTOCOL = "external-har-observable-context-v1"


def method_inference_contracts(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    dataset = result.get("target_dataset", result.get("dataset", {}))
    dataset_id = dataset.get("dataset_id")
    methods = result.get("primary_seed_averaged", {}).get("methods", result.get("reports", {}))
    output: dict[str, dict[str, Any]] = {}
    for name in methods:
        context = name in PARTICIPANT_CONTEXT_METHODS
        unit, population, status = "independent_fixed_window", "one_window", "window_local"
        annotation_selected_context = False
        if context:
            if dataset_id == "fog_star_v3":
                observable = (
                    result.get("observable_context_protocol") == OBSERVABLE_CONTEXT_PROTOCOL
                    and dataset.get("observable_candidate_pool", {}).get("protocol_id")
                    == OBSERVABLE_CONTEXT_PROTOCOL
                )
                population = (
                    "all_observable_candidates_of_retained_participant"
                    if observable
                    else "annotation_admitted_windows_only"
                )
                unit = "noncausal_participant_batch"
                annotation_selected_context = not observable
                status = (
                    "development_noncausal_context"
                    if observable
                    else "diagnostic_annotation_selected_context"
                )
            else:
                unit, population, status = (
                    "noncausal_participant_batch",
                    "declared_selected_scripted_trials",
                    "development_scripted_trial_context_not_natural_stream",
                )
        if dataset_id == "sole_harmony_v1":
            unit, population, status = (
                "camera_bout_oracle_pipeline",
                "camera_annotated_bouts",
                "oracle_diagnostic",
            )
        channels = (
            6
            if name.endswith("-6ch") or name == "RMRP-DG" or name.startswith(("HPF-", "PB-HPF"))
            else 9
            if name.endswith(("-N9", "-D9")) or "-DG" in name
            else result.get("input_channels")
        )
        output[name] = {
            "inference_unit": unit,
            "context_population": population,
            "annotation_selected_evaluation_context": annotation_selected_context,
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
    )
