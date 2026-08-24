"""Numerical audit of acceleration-unit sensitivity under training-only z-scoring."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.preprocessing.units import (
    STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED,
    acceleration_g_to_metres_per_second_squared,
)


def _converted_copy(windows: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    array = np.asarray(windows, dtype=np.float64)
    if (
        array.ndim != 3
        or array.shape[0] == 0
        or array.shape[1] == 0
        or array.shape[2] != 6
        or not np.isfinite(array).all()
    ):
        raise ValueError("unit sensitivity requires non-empty finite [window,time,6] tensors")
    converted = array.copy()
    converted[:, :, :3] = acceleration_g_to_metres_per_second_squared(converted[:, :, :3])
    return converted


def build_acceleration_unit_sensitivity_record(
    source_windows_g: NDArray[np.floating[Any]],
    evaluation_windows_g: NDArray[np.floating[Any]],
    source_participant_ids: Sequence[str],
    *,
    declared_source_participants: set[str],
    split_manifest_sha256: str,
    channel_names: tuple[str, ...],
    lineage: Mapping[str, Any],
    equivalence_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Compare normalized tensors when acceleration is represented in g versus SI.

    This is a preprocessing sensitivity, not another trained-model result.  A positive affine
    unit scale cancels algebraically when moments are fit on the same source partition.  The
    returned diagnostics make the finite-precision residual explicit for both source and an
    evaluation tensor.
    """

    if (
        len(channel_names) != 6
        or len(set(channel_names)) != 6
        or any(not isinstance(name, str) or not name for name in channel_names)
    ):
        raise ValueError("exactly six unique channel names are required")
    if equivalence_tolerance <= 0 or not np.isfinite(equivalence_tolerance):
        raise ValueError("equivalence_tolerance must be positive and finite")
    source = np.asarray(source_windows_g)
    evaluation = np.asarray(evaluation_windows_g)
    source_si = _converted_copy(source)
    evaluation_si = _converted_copy(evaluation)
    if source.shape[0] != len(source_participant_ids):
        raise ValueError("source participant ids do not align with source windows")
    if any(
        not isinstance(participant, str) or not participant
        for participant in source_participant_ids
    ):
        raise ValueError("source participant ids must be non-empty strings")
    if not isinstance(split_manifest_sha256, str):
        raise ValueError("split_manifest_sha256 must be a full SHA-256")
    normalized_split_hash = split_manifest_sha256.casefold()
    if len(normalized_split_hash) != 64 or any(
        character not in "0123456789abcdef" for character in normalized_split_hash
    ):
        raise ValueError("split_manifest_sha256 must be a full SHA-256")
    standardizer_g = ChannelStandardizer.fit(
        source,
        list(source_participant_ids),
        declared_training_participants=declared_source_participants,
        split_manifest_sha256=normalized_split_hash,
        channel_names=channel_names,
    )
    standardizer_si = ChannelStandardizer.fit(
        source_si,
        list(source_participant_ids),
        declared_training_participants=declared_source_participants,
        split_manifest_sha256=normalized_split_hash,
        channel_names=channel_names,
    )
    normalized_source_g = standardizer_g.transform(source)
    normalized_source_si = standardizer_si.transform(source_si)
    normalized_evaluation_g = standardizer_g.transform(evaluation)
    normalized_evaluation_si = standardizer_si.transform(evaluation_si)
    source_residual = np.abs(normalized_source_g.astype(np.float64) - normalized_source_si)
    evaluation_residual = np.abs(
        normalized_evaluation_g.astype(np.float64) - normalized_evaluation_si
    )
    maximum = max(float(source_residual.max()), float(evaluation_residual.max()))
    mean_ratio_residual = np.abs(
        standardizer_si.mean[:3]
        - standardizer_g.mean[:3] * STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED
    )
    scale_ratio_residual = np.abs(
        standardizer_si.scale[:3]
        - standardizer_g.scale[:3] * STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED
    )
    equivalent = maximum <= equivalence_tolerance
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": (
            "numerically_equivalent_under_training_only_zscore"
            if equivalent
            else "finite_precision_equivalence_tolerance_exceeded"
        ),
        "evidence_status": "post_confirmatory_preprocessing_sensitivity_not_model_result",
        "conversion": {
            "input_acceleration_unit": "g",
            "output_acceleration_unit": "m/s^2",
            "multiplier": STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED,
            "converted_channel_indices": [0, 1, 2],
            "unchanged_channel_indices": [3, 4, 5],
            "timing": "before_source_training_only_standardization",
        },
        "source_window_count": int(source.shape[0]),
        "evaluation_window_count": int(evaluation.shape[0]),
        "source_participants": sorted(declared_source_participants),
        "split_manifest_sha256": normalized_split_hash,
        "channel_names": list(channel_names),
        "equivalence_tolerance": equivalence_tolerance,
        "normalized_tensor_diagnostics": {
            "source_maximum_absolute_difference": float(source_residual.max()),
            "source_mean_absolute_difference": float(source_residual.mean()),
            "evaluation_maximum_absolute_difference": float(evaluation_residual.max()),
            "evaluation_mean_absolute_difference": float(evaluation_residual.mean()),
            "overall_maximum_absolute_difference": maximum,
            "within_tolerance": equivalent,
        },
        "moment_scaling_diagnostics": {
            "acceleration_mean_maximum_absolute_residual": float(mean_ratio_residual.max()),
            "acceleration_scale_maximum_absolute_residual": float(scale_ratio_residual.max()),
        },
        "interpretation": (
            "Changing only the acceleration unit is algebraically cancelled by fitting the same "
            "per-channel z-score on the same source partition. This record tests preprocessing "
            "equivalence and is not an independent accuracy experiment."
        ),
        "lineage": dict(lineage),
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload
