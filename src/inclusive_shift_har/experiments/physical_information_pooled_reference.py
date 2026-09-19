"""Annotation-blind pooled-support companion for the physical information pilot.

The four labelled support bouts are legitimate calibration inputs. Query activity and
motion annotations are used only after predictions are formed, for signed scoring and
reporting. This module performs no fitting and does not alter the frozen pilot analysis.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.physical_information_signal import (
    SignalContractError,
    WindowTable,
    student_t_summary,
    window_table_hash,
)

POSTURES = ("sitting", "standing")
MOTIONS = ("quiet", "upper_body_motion")


FloatArray = NDArray[np.float64]


def _unit_mean(vectors: FloatArray, epsilon: float) -> FloatArray | None:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] == 0:
        return None
    mean = np.asarray(vectors, dtype=np.float64).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if not math.isfinite(norm) or norm <= epsilon:
        return None
    return np.asarray(mean / norm, dtype=np.float64)


def _angle_rows(queries: FloatArray, supports: FloatArray) -> FloatArray:
    return np.asarray(np.arccos(np.clip(queries @ supports.T, -1.0, 1.0)), dtype=np.float64)


def analyze_pooled_reference(
    *,
    plan: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
    windows: WindowTable,
    norm_epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Evaluate two components per posture without query-stratum prototype selection."""

    if not math.isfinite(norm_epsilon) or norm_epsilon <= 0.0:
        raise SignalContractError("norm_epsilon must be finite and positive")
    if plan.get("record_kind") != "physical_information_pilot_plan":
        raise SignalContractError("pooled companion requires a physical pilot plan")
    roster = [str(row["wearer_id"]) for row in cast(list[Mapping[str, Any]], plan["wearers"])]
    if len(roster) != len(set(roster)):
        raise SignalContractError("pilot roster contains duplicate wearer identifiers")
    bouts = cast(list[Mapping[str, Any]], plan["bouts"])
    planned = {str(row["bout_id"]): row for row in bouts}
    if len(planned) != len(bouts):
        raise SignalContractError("planned bout identifiers are not unique")
    projected = {str(row["bout_id"]): row for row in projections}
    if len(projected) != len(projections) or set(projected) != set(planned):
        raise SignalContractError("bout projections must cover the exact planned bout set")
    lookup = {identifier: index for index, identifier in enumerate(windows.window_ids)}
    if len(lookup) != len(windows.window_ids):
        raise SignalContractError("window identifiers are not unique")

    component_records: list[dict[str, Any]] = []
    bout_rows: list[dict[str, Any]] = []
    fresh = [
        row
        for row in cast(list[Mapping[str, Any]], plan["comparisons"])
        if row.get("comparison_type") == "fresh"
    ]
    for comparison in fresh:
        comparison_id = str(comparison["comparison_id"])
        support_ids = [
            str(value) for value in cast(list[object], comparison["fresh_support_bout_ids"])
        ]
        query_ids = [str(value) for value in cast(list[object], comparison["query_bout_ids"])]
        if len(support_ids) != 4 or len(set(support_ids)) != 4:
            raise SignalContractError(
                f"pooled companion requires exactly four unique support bouts: {comparison_id}"
            )
        if len(query_ids) != len(set(query_ids)):
            raise SignalContractError(
                f"pooled companion query bouts must be unique: {comparison_id}"
            )
        if set(support_ids) & set(query_ids):
            raise SignalContractError(f"support/query bout overlap: {comparison_id}")
        support_windows = {
            identifier
            for bout_id in support_ids
            for identifier in cast(list[str], projected[bout_id].get("valid_window_ids", []))
        }
        query_windows = {
            identifier
            for bout_id in query_ids
            for identifier in cast(list[str], projected[bout_id].get("valid_window_ids", []))
        }
        if support_windows & query_windows:
            raise SignalContractError(f"support/query window overlap: {comparison_id}")

        components: dict[str, list[FloatArray]] = {posture: [] for posture in POSTURES}
        component_failure: list[str] = []
        for posture in POSTURES:
            for motion in MOTIONS:
                matches = [
                    bout_id
                    for bout_id in support_ids
                    if planned[bout_id].get("activity") == posture
                    and planned[bout_id].get("motion") == motion
                ]
                vector: FloatArray | None = None
                reason: str | None = None
                window_ids: list[str] = []
                if len(matches) != 1:
                    reason = "support_condition_not_unique"
                else:
                    projection = projected[matches[0]]
                    if projection.get("eligible") is not True:
                        reason = "support_bout_ineligible"
                    else:
                        window_ids = cast(list[str], projection["valid_window_ids"])
                        if any(identifier not in lookup for identifier in window_ids):
                            raise SignalContractError(
                                "support projection references unknown window"
                            )
                        vector = _unit_mean(
                            windows.gravity_directions[[lookup[item] for item in window_ids]],
                            norm_epsilon,
                        )
                        if vector is None:
                            reason = "support_resultant_norm_at_or_below_epsilon"
                        else:
                            components[posture].append(vector)
                component_records.append(
                    {
                        "comparison_id": comparison_id,
                        "posture": posture,
                        "support_motion": motion,
                        "support_bout_id": matches[0] if len(matches) == 1 else None,
                        "valid": reason is None,
                        "invalid_reason": reason,
                        "valid_window_ids": window_ids,
                        "direction": None if vector is None else vector.tolist(),
                    }
                )
                if reason is not None:
                    component_failure.append(f"{posture}:{motion}:{reason}")

        for bout_id in query_ids:
            row = planned[bout_id]
            activity = str(row.get("activity"))
            motion = str(row.get("motion"))
            if activity not in POSTURES or motion not in MOTIONS:
                continue
            projection = projected[bout_id]
            window_ids = cast(list[str], projection.get("valid_window_ids", []))
            query_reason: str | None = None
            raw_ratios: list[float] = []
            signed: list[float] = []
            if projection.get("eligible") is not True:
                query_reason = "query_bout_ineligible"
            elif component_failure:
                query_reason = "pooled_support_invalid"
            elif any(identifier not in lookup for identifier in window_ids):
                raise SignalContractError("query projection references unknown window")
            else:
                query = windows.gravity_directions[[lookup[item] for item in window_ids]]
                sitting = np.stack(components["sitting"])
                standing = np.stack(components["standing"])
                d_sitting = _angle_rows(query, sitting).min(axis=1)
                d_standing = _angle_rows(query, standing).min(axis=1)
                raw = d_standing - d_sitting
                sign = 1.0 if activity == "sitting" else -1.0
                raw_ratios = [float(value) for value in raw]
                signed = [float(sign * value) for value in raw]
            bout_rows.append(
                {
                    "comparison_id": comparison_id,
                    "query_bout_id": bout_id,
                    "wearer_id": str(row["wearer_id"]),
                    "activity_for_evaluation_only": activity,
                    "motion_for_reporting_only": motion,
                    "prediction_used_query_activity": False,
                    "prediction_used_query_motion": False,
                    "eligible": query_reason is None,
                    "eligibility_reason": query_reason,
                    "valid_window_ids": window_ids,
                    "raw_standing_minus_sitting_distance": raw_ratios,
                    "signed_correct_posture_margins_radians": signed,
                    "bout_median_signed_margin_radians": (
                        float(np.median(signed)) if query_reason is None and signed else None
                    ),
                }
            )

    wearer_rows: list[dict[str, Any]] = []
    wearer_motion_means: dict[tuple[str, str], float] = {}
    for wearer in roster:
        for motion in MOTIONS:
            rows = [
                row
                for row in bout_rows
                if row["wearer_id"] == wearer and row["motion_for_reporting_only"] == motion
            ]
            eligible = [row for row in rows if row["eligible"] is True]
            expected = 8
            complete = len(rows) == expected and len(eligible) == expected
            values = [float(row["bout_median_signed_margin_radians"]) for row in eligible]
            mean = float(np.mean(values)) if complete else None
            if mean is not None:
                wearer_motion_means[(wearer, motion)] = mean
            wearer_rows.append(
                {
                    "wearer_id": wearer,
                    "motion_for_reporting_only": motion,
                    "expected_bout_count": expected,
                    "observed_bout_count": len(rows),
                    "eligible_bout_count": len(eligible),
                    "complete": complete,
                    "bout_values": values,
                    "mean_signed_margin_radians": mean,
                    "exclusions": [
                        {"query_bout_id": row["query_bout_id"], "reason": row["eligibility_reason"]}
                        for row in rows
                        if row["eligible"] is not True
                    ],
                }
            )

    failures: list[str] = []
    for wearer in roster:
        for motion in MOTIONS:
            value = wearer_motion_means.get((wearer, motion))
            if value is None:
                failures.append(f"{wearer}:{motion}:complete_mean_missing")
            elif value <= 0.0:
                failures.append(f"{wearer}:{motion}:mean_not_strictly_positive")
    if len(roster) != 6:
        failures.append("roster_is_not_exactly_six_wearers")
    summaries: dict[str, Any] = {}
    for motion in MOTIONS:
        summaries[motion] = student_t_summary(
            [
                wearer_motion_means[(wearer, motion)]
                for wearer in roster
                if (wearer, motion) in wearer_motion_means
            ]
        )
    return {
        "schema_version": "1.0.0",
        "record_kind": "physical_information_pooled_reference_companion_analysis",
        "evidence_status": "inherits_physical_signal_input_status",
        "model_fit_count": 0,
        "window_table_sha256": window_table_hash(windows),
        "support_components": component_records,
        "query_bout_rows": bout_rows,
        "wearer_motion_rows": wearer_rows,
        "equal_wearer_summaries": summaries,
        "pooled_reference_prerequisite_gate": {
            "status": "pass" if not failures else "fail",
            "all_six_complete": len(roster) == 6 and all(row["complete"] for row in wearer_rows),
            "failures": failures,
            "decision_scope": "permits_separate_supervised_protocol_only",
        },
        "claims": {
            "physical_identifiability_demonstrated": False,
            "population_superiority_demonstrated": False,
            "architecture_eligible": False,
        },
        "inference_firewall": {
            "support_activity_used": True,
            "support_motion_used_to_retain_two_independent_components": True,
            "query_activity_used_by_predictor": False,
            "query_motion_used_by_predictor": False,
            "query_annotations_used_for_evaluation_only": True,
        },
    }
