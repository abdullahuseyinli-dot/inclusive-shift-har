from __future__ import annotations

from typing import Any

import pytest

from inclusive_shift_har.evaluation.inference_contracts import (
    OBSERVABLE_CONTEXT_PROTOCOL,
    annotation_selected_context_methods,
    method_inference_contracts,
)
from inclusive_shift_har.experiments.external_evidence_validate import _method_contract_errors


def _result() -> dict[str, Any]:
    return {
        "dataset": {"dataset_id": "fog_star_v3", "channel_lane": "derived-gravity-9ch"},
        "reports": {
            name: {}
            for name in (
                "RMRP-DG",
                "XGBoost-6ch",
                "HERA-DG-strict",
                "HERA-DG-full",
                "HERA-DG-context-safe",
            )
        },
    }


def test_annotation_selected_context_is_not_independent_window_evidence() -> None:
    result = _result()
    contracts = method_inference_contracts(result)
    assert annotation_selected_context_methods(result) == ["HERA-DG-context-safe", "HERA-DG-full"]
    assert contracts["HERA-DG-full"]["scientific_role"] == "diagnostic_annotation_selected_context"
    assert contracts["RMRP-DG"]["input_channel_count"] == 6
    assert contracts["HERA-DG-strict"]["inference_unit"] == "independent_fixed_window"
    assert contracts["HERA-DG-strict"]["input_channel_count"] == 9
    assert contracts["HERA-DG-full"]["differences_vs_independent_six_channel_control"] == [
        "input_channels",
        "inference_unit_and_context_budget",
    ]


def test_observable_pool_and_executed_protocol_are_both_required() -> None:
    result = _result()
    result["observable_context_protocol"] = OBSERVABLE_CONTEXT_PROTOCOL
    assert annotation_selected_context_methods(result)
    result["dataset"]["observable_candidate_pool"] = {"protocol_id": OBSERVABLE_CONTEXT_PROTOCOL}
    assert annotation_selected_context_methods(result) == []
    contract = method_inference_contracts(result)["HERA-DG-full"]
    assert contract["scientific_role"] == "development_noncausal_context"
    assert contract["inference_unit"] == "noncausal_participant_batch"
    assert not contract["zero_lookahead_streaming_claim_allowed"]
    errors = _method_contract_errors(result, {})
    assert "FoG observable inference pool is incomplete or label-bearing" in errors
    assert "FoG observable inference population is not recorded per fit" in errors


def test_transfer_context_uses_target_contract_not_source_name() -> None:
    result = _result()
    result["target_dataset"] = result.pop("dataset")
    result["source_dataset"] = {"dataset_id": "imu_har_il_v1"}
    assert annotation_selected_context_methods(result)


def test_scripted_trial_and_oracle_contexts_do_not_become_streaming_claims() -> None:
    result = _result()
    result["dataset"]["dataset_id"] = "imu_har_il_v1"
    contract = method_inference_contracts(result)["HERA-DG-full"]
    assert contract["context_population"] == "declared_selected_scripted_trials"
    assert "not_natural_stream" in contract["scientific_role"]
    result["dataset"]["dataset_id"] = "sole_harmony_v1"
    assert all(
        item["scientific_role"] == "oracle_diagnostic"
        for item in method_inference_contracts(result).values()
    )


def test_posture_controls_keep_matched_six_channel_and_separate_d9_ablation() -> None:
    result = _result()
    result["reports"] = {
        name: {}
        for name in ("PB-HPF", "HPF-unweighted", "PB-HPF-shuffled-posture", "PB-RF-6ch", "PB-RF-D9")
    }
    contracts = method_inference_contracts(result)
    assert contracts["PB-HPF"]["differences_vs_independent_six_channel_control"] == []
    assert contracts["PB-RF-D9"]["differences_vs_independent_six_channel_control"] == [
        "input_channels"
    ]


def _complete_pool_result(*, transfer: bool) -> dict[str, Any]:
    result = _result()
    result["seeds"] = [11, 23, 47]
    result["primary_seed_averaged"] = {"methods": result["reports"]}
    result["observable_context_protocol"] = OBSERVABLE_CONTEXT_PROTOCOL
    result["dataset"].update(
        {
            "window_count": 1,
            "preprocessing_audit": [
                {
                    "protocol_id": "external-har-session-grid-v3",
                    "annotation_dependency": False,
                    "resampling_passes": 1,
                    "window_samples": 128,
                    "candidate_start_samples": [0, 128],
                    "candidate_window_count": 2,
                    "admitted_candidate_indices": [0],
                    "excluded_candidate_indices": {"missing": [1]},
                }
            ],
            "observable_candidate_pool": {
                "protocol_id": OBSERVABLE_CONTEXT_PROTOCOL,
                "annotation_fields_present": False,
                "window_count": 2,
                "participant_window_counts": {"p1": 2},
                "arrays": {
                    name: "a" * 64
                    for name in (
                        "signals",
                        "gravity",
                        "participant_ids",
                        "session_ids",
                        "trial_ids",
                        "window_ids",
                    )
                },
            },
        }
    )
    if transfer:
        result["target_dataset"] = result.pop("dataset")
        result["source_dataset"] = {"dataset_id": "imu_har_il_v1"}
        result["seed_records"] = [
            {
                "target_participants": ["p1"],
                "target_inference_candidate_window_count": 2,
            }
        ]
    else:
        result["fold_records"] = [
            {
                "folds": [
                    {
                        "evaluation_participants": ["p1"],
                        "evaluation_candidate_window_count": 2,
                    }
                ]
            }
        ]
    return result


@pytest.mark.parametrize("transfer", [False, True])
@pytest.mark.parametrize("mutation", ["none", "labels", "missing_hash", "count", "missing_fit"])
def test_validator_requires_complete_observable_population_per_fit(
    transfer: bool, mutation: str
) -> None:
    result = _complete_pool_result(transfer=transfer)
    pool = result["target_dataset" if transfer else "dataset"]["observable_candidate_pool"]
    if mutation == "labels":
        pool["annotation_fields_present"] = True
    elif mutation == "missing_hash":
        del pool["arrays"]["gravity"]
    elif mutation == "count":
        record = result["seed_records"][0] if transfer else result["fold_records"][0]["folds"][0]
        record[
            "target_inference_candidate_window_count"
            if transfer
            else "evaluation_candidate_window_count"
        ] = 1
    elif mutation == "missing_fit":
        result["seed_records" if transfer else "fold_records"] = []
    errors = _method_contract_errors(result, {})
    assert bool(errors) is (mutation != "none")
    if mutation == "count":
        assert (
            "FoG fit did not evaluate every observable candidate of its held-out participants"
            in errors
        )


def test_unknown_method_channels_remain_unknown() -> None:
    result = _result()
    result["reports"] = {"unrecognized": {}}
    assert method_inference_contracts(result)["unrecognized"]["input_channel_count"] is None
