from __future__ import annotations

from typing import Any

import numpy as np

from inclusive_shift_har.data.physical_information_signal import WindowTable
from inclusive_shift_har.experiments.physical_information_pooled_reference import (
    analyze_pooled_reference,
)


def _fixture(
    swapped_across_motion: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], WindowTable]:
    wearers = [f"pilot:{index:03d}" for index in range(1, 7)]
    bouts: list[dict[str, Any]] = []
    projections: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    ids: list[str] = []
    vectors: list[np.ndarray] = []
    for wearer in wearers:
        for block in range(4):
            block_id = f"{wearer}:b{block}"
            support_ids: list[str] = []
            query_ids: list[str] = []
            for role in ("support", "query"):
                for activity in ("sitting", "standing"):
                    for motion in ("quiet", "upper_body_motion"):
                        bout_id = f"{block_id}:{role}:{activity}:{motion}"
                        (support_ids if role == "support" else query_ids).append(bout_id)
                        bouts.append(
                            {
                                "bout_id": bout_id,
                                "block_id": block_id,
                                "wearer_id": wearer,
                                "role": role,
                                "activity": activity,
                                "motion": motion,
                            }
                        )
                        window_id = f"window:{bout_id}"
                        ids.append(window_id)
                        if swapped_across_motion and motion == "upper_body_motion":
                            direction = (
                                np.array([0.0, 1.0, 0.0])
                                if activity == "sitting"
                                else np.array([1.0, 0.0, 0.0])
                            )
                        else:
                            direction = (
                                np.array([1.0, 0.0, 0.0])
                                if activity == "sitting"
                                else np.array([0.0, 1.0, 0.0])
                            )
                        vectors.append(direction)
                        projections.append(
                            {"bout_id": bout_id, "eligible": True, "valid_window_ids": [window_id]}
                        )
            comparisons.append(
                {
                    "comparison_id": f"fresh::{block_id}",
                    "comparison_type": "fresh",
                    "fresh_support_bout_ids": support_ids,
                    "query_bout_ids": query_ids,
                }
            )
    count = len(ids)
    windows = WindowTable(
        window_ids=tuple(ids),
        recording_ids=tuple("r" for _ in ids),
        block_ids=tuple("b" for _ in ids),
        wearer_ids=tuple("w" for _ in ids),
        run_indices=np.zeros(count, dtype=np.int64),
        source_run_indices=np.zeros(count, dtype=np.int64),
        source_start_sample_indices=np.arange(count),
        source_stop_sample_indices_exclusive=np.arange(count) + 1,
        starts_seconds=np.arange(count, dtype=float),
        ends_seconds=np.arange(count, dtype=float) + 1,
        gravity_directions=np.stack(vectors),
        dynamic_acceleration_rms=np.zeros(count),
        gyroscope_rms=np.zeros(count),
        valid=np.ones(count, dtype=np.bool_),
        invalid_reasons=tuple("valid" for _ in ids),
    )
    return (
        {
            "record_kind": "physical_information_pilot_plan",
            "wearers": [{"wearer_id": wearer} for wearer in wearers],
            "bouts": bouts,
            "comparisons": comparisons,
        },
        projections,
        windows,
    )


def test_pooled_companion_passes_separable_components_without_query_stratum() -> None:
    plan, projections, windows = _fixture()
    result = analyze_pooled_reference(plan=plan, projections=projections, windows=windows)
    assert result["model_fit_count"] == 0
    assert result["pooled_reference_prerequisite_gate"]["status"] == "pass"
    assert all(not row["prediction_used_query_motion"] for row in result["query_bout_rows"])
    assert all(row["mean_signed_margin_radians"] > 0 for row in result["wearer_motion_rows"])


def test_stratum_separable_but_pooled_indistinguishable_counterexample_fails() -> None:
    plan, projections, windows = _fixture(swapped_across_motion=True)
    result = analyze_pooled_reference(plan=plan, projections=projections, windows=windows)
    assert result["pooled_reference_prerequisite_gate"]["status"] == "fail"
    assert all(row["mean_signed_margin_radians"] == 0 for row in result["wearer_motion_rows"])
    assert len(result["pooled_reference_prerequisite_gate"]["failures"]) == 12


def test_missing_support_remains_visible_and_cannot_pass() -> None:
    plan, projections, windows = _fixture()
    projections[0]["eligible"] = False
    projections[0]["valid_window_ids"] = []
    result = analyze_pooled_reference(plan=plan, projections=projections, windows=windows)
    assert result["pooled_reference_prerequisite_gate"]["status"] == "fail"
    affected = [
        row for row in result["query_bout_rows"] if row["comparison_id"] == "fresh::pilot:001:b0"
    ]
    assert all(row["eligibility_reason"] == "pooled_support_invalid" for row in affected)
