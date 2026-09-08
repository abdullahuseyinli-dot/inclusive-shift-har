"""Label-blind FoG ankle-context reconstruction and zero-fit qualification.

The functions in this module reconstruct physical signal support only.  Activity
annotations are deliberately absent from every context-boundary interface; scoring
metadata is loaded only after the context mask has been frozen by the path-based
preflight.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from torch import Tensor, nn
from torch.nn import functional as F

from inclusive_shift_har.data.external_har import (
    UniformPhysicalSegment,
    _contiguous_signal_runs,
    resample_physical_segment,
)
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _require,
)
from inclusive_shift_har.experiments.fog_spatial_information_probe import (
    ANKLE_FEATURE_SHA256,
    ANKLE_GRAVITY_SHA256,
    ANKLE_SIGNAL_SHA256,
    COVERAGE_SHA256,
    MASK_SHA256,
    ORDERED_ID_SHA256,
    RAW_SHA256,
    RAW_SIZE,
    _availability_array_sha256,
    availability_mask_from_rows,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]
SessionKey: TypeAlias = tuple[str, str]
SensorRunMap: TypeAlias = Mapping[SessionKey, Mapping[int, UniformPhysicalSegment]]

OBSERVABLE_COUNT = 1_939
SCORED_COUNT = 1_213
CURRENT_AVAILABLE_COUNT = 1_817
FULL_CONTEXT_OBSERVABLE_COUNT = 1_724
FULL_CONTEXT_SCORED_COUNT = 1_098
CURRENT_SAMPLES = 128
CONTEXT_SAMPLES = 500
SOURCE_RATE_HZ = 60.0
TARGET_RATE_HZ = 50.0
GRAVITY_CUTOFF_HZ = 0.30
MAXIMUM_GAP_FACTOR = 3.0
STANDARD_GRAVITY_M_S2 = 9.80665

EXPECTED_SCORED_CLASS_COUNTS = (954, 74, 185)
EXPECTED_FULL_CONTEXT_CLASS_COUNTS = (906, 32, 160)
EXPECTED_FULL_CONTEXT_EVALUATION_ROWS = (194, 318, 196, 318, 72)
EXPECTED_FULL_CONTEXT_TRAINING_ROWS = (904, 780, 902, 780, 1_026)
EXPECTED_FULL_CONTEXT_TRAINING_CLASSES = (
    (762, 19, 123),
    (641, 28, 111),
    (753, 21, 128),
    (619, 28, 133),
    (849, 32, 145),
)

_WINDOW_COORDINATES = re.compile(r"/run-(?P<run>[0-9]+)-finite-[0-9]+/window-(?P<window>[0-9]+)$")


@dataclass(frozen=True, slots=True)
class FogPhysicalRuns:
    """Annotation-independent back and left-ankle signal grids."""

    back: dict[SessionKey, dict[int, UniformPhysicalSegment]]
    ankle: dict[SessionKey, dict[int, UniformPhysicalSegment]]
    segment_receipts: tuple[dict[str, Any], ...]
    parsed_columns: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FogMotionContexts:
    """Right-aligned signal contexts in frozen observable-candidate order.

    ``signals`` has layout ``[observable, time, channel]`` and contains ankle
    linear acceleration XYZ followed by angular velocity XYZ. ``derived_gravity``
    uses the same time grid but is audit-only. Invalid/padded positions are exactly
    zero and identified by ``observation_mask``; timestamps are never model inputs.
    """

    signals: FloatArray
    derived_gravity: FloatArray
    timestamps: FloatArray
    observation_mask: BoolArray
    current_availability_mask: BoolArray
    full_context_mask: BoolArray
    observable_window_ids: StringArray
    alignment_receipts: tuple[dict[str, Any], ...]
    segment_receipts: tuple[dict[str, Any], ...]

    @property
    def context_indices(self) -> IntArray:
        """Return observable indices with the complete fixed context."""

        return np.flatnonzero(self.full_context_mask).astype(np.int64, copy=False)

    def validate(self, *, context_samples: int, current_samples: int) -> None:
        """Validate array alignment, masking, and current-query preservation."""

        count = self.observable_window_ids.size
        _require(
            self.signals.shape == (count, context_samples, 6),
            "motion-context signal shape changed",
        )
        _require(
            self.derived_gravity.shape == (count, context_samples, 3),
            "motion-context gravity shape changed",
        )
        _require(
            self.timestamps.shape == self.observation_mask.shape == (count, context_samples),
            "motion-context timestamp or observation-mask shape changed",
        )
        _require(
            self.current_availability_mask.shape == self.full_context_mask.shape == (count,),
            "motion-context row-mask shape changed",
        )
        _require(
            len(self.alignment_receipts) == count,
            "motion-context alignment receipt count changed",
        )
        _require(
            bool(np.isfinite(self.signals).all())
            and bool(np.isfinite(self.derived_gravity).all())
            and bool(np.isfinite(self.timestamps).all()),
            "motion-context arrays contain nonfinite values",
        )
        _require(
            np.array_equal(self.full_context_mask, self.observation_mask.all(axis=1)),
            "full-context mask differs from per-sample support",
        )
        _require(
            not bool(np.any(self.full_context_mask & ~self.current_availability_mask)),
            "full context exists without a qualified current ankle query",
        )
        current_support = self.observation_mask[:, -current_samples:].all(axis=1)
        _require(
            np.array_equal(current_support, self.current_availability_mask),
            "current-window support differs from the frozen availability mask",
        )
        _require(
            bool(np.all(self.signals[~self.observation_mask] == 0.0))
            and bool(np.all(self.derived_gravity[~self.observation_mask] == 0.0))
            and bool(np.all(self.timestamps[~self.observation_mask] == 0.0)),
            "masked context positions must be exactly zero",
        )


@dataclass(frozen=True, slots=True)
class FogMotionContextPreflight:
    """Qualified zero-fit context replay and its machine-readable receipt."""

    contexts: FogMotionContexts
    scoring_indices: IntArray
    scoring_eligibility: BoolArray
    scored_labels: IntArray
    observable_participant_ids: StringArray
    observable_fold_index: IntArray
    receipt: dict[str, Any]


def _window_coordinates(window_id: str) -> tuple[int, int]:
    match = _WINDOW_COORDINATES.search(window_id)
    if match is None:
        raise ValueError(f"unrecognized corrected-FoG window ID: {window_id}")
    return int(match.group("run")), int(match.group("window"))


def _sensor_columns(sensor: str) -> list[str]:
    return [f"{sensor}_{modality}_{axis}" for modality in ("acc", "gyro") for axis in "xyz"]


def _session_identity(subject: Any, session_value: Any) -> SessionKey:
    participant = f"fogstar:{int(subject):03d}"
    return participant, f"{participant}:session-{int(session_value):03d}"


def _materialize_sensor_runs(
    *,
    timestamps: FloatArray,
    raw_values: FloatArray,
    participant_id: str,
    session_id: str,
    sensor: str,
    source_rate_hz: float,
    target_rate_hz: float,
    gravity_cutoff_hz: float,
    maximum_gap_factor: float,
    current_samples: int,
) -> tuple[dict[int, UniformPhysicalSegment], list[dict[str, Any]]]:
    total_acceleration = np.asarray(raw_values[:, :3], dtype=np.float64) * STANDARD_GRAVITY_M_S2
    angular_velocity = np.asarray(raw_values[:, 3:], dtype=np.float64) * np.pi / 180.0
    boundaries = _contiguous_signal_runs(
        timestamps,
        total_acceleration,
        angular_velocity,
        nominal_rate_hz=source_rate_hz,
        maximum_gap_factor=maximum_gap_factor,
    )
    runs: dict[int, UniformPhysicalSegment] = {}
    receipts: list[dict[str, Any]] = []
    for run_index, (left, right) in enumerate(boundaries):
        if right - left < 3:
            receipts.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "sensor": sensor,
                    "physical_run_index": run_index,
                    "source_left_index": left,
                    "source_right_index_exclusive": right,
                    "status": "excluded_short_physical_run",
                }
            )
            continue
        segment = resample_physical_segment(
            timestamps=np.asarray(timestamps[left:right], dtype=np.float64),
            acceleration=np.asarray(total_acceleration[left:right], dtype=np.float64),
            gyroscope=np.asarray(angular_velocity[left:right], dtype=np.float64),
            gravity=None,
            source_rate_hz=source_rate_hz,
            target_rate_hz=target_rate_hz,
            gravity_cutoff_hz=gravity_cutoff_hz,
            window_samples=current_samples,
        )
        runs[run_index] = segment
        receipts.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "sensor": sensor,
                "physical_run_index": run_index,
                "source_left_index": left,
                "source_right_index_exclusive": right,
                "status": "materialized",
                "audit": segment.audit(),
            }
        )
    return runs, receipts


def reconstruct_fog_physical_runs(
    raw_path: Path,
    *,
    source_rate_hz: float = SOURCE_RATE_HZ,
    target_rate_hz: float = TARGET_RATE_HZ,
    gravity_cutoff_hz: float = GRAVITY_CUTOFF_HZ,
    maximum_gap_factor: float = MAXIMUM_GAP_FACTOR,
    current_samples: int = CURRENT_SAMPLES,
) -> FogPhysicalRuns:
    """Reconstruct back and ankle grids without reading an annotation column.

    Released acceleration is interpreted as total acceleration in g and converted
    to m/s^2. Gravity is derived causally on each native-rate finite run before the
    nine physical channels are jointly resampled once. Gyroscope values are converted
    from degrees/s to radians/s.
    """

    _require(raw_path.is_file(), "FoG raw source is missing")
    _require(
        source_rate_hz > 0.0
        and target_rate_hz > 0.0
        and gravity_cutoff_hz > 0.0
        and maximum_gap_factor > 0.0
        and current_samples > 0,
        "physical reconstruction parameters must be positive",
    )
    columns = ["timestamp", "subjectID", "sessionID"]
    for sensor in ("back", "ankleL"):
        columns.extend(_sensor_columns(sensor))
    frame = pd.read_csv(raw_path, usecols=columns)
    _require(set(frame.columns) == set(columns), "raw FoG sensor column set changed")
    back: dict[SessionKey, dict[int, UniformPhysicalSegment]] = {}
    ankle: dict[SessionKey, dict[int, UniformPhysicalSegment]] = {}
    receipts: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["subjectID", "sessionID"], sort=True, dropna=False):
        subject, session_value = cast(tuple[Any, Any], keys)
        participant_id, session_id = _session_identity(subject, session_value)
        session_key = (participant_id, session_id)
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        for sensor, destination in (("back", back), ("ankleL", ankle)):
            values = group[_sensor_columns(sensor)].to_numpy(dtype=np.float64)
            materialized, sensor_receipts = _materialize_sensor_runs(
                timestamps=timestamps,
                raw_values=values,
                participant_id=participant_id,
                session_id=session_id,
                sensor=sensor,
                source_rate_hz=source_rate_hz,
                target_rate_hz=target_rate_hz,
                gravity_cutoff_hz=gravity_cutoff_hz,
                maximum_gap_factor=maximum_gap_factor,
                current_samples=current_samples,
            )
            destination[session_key] = materialized
            receipts.extend(sensor_receipts)
    return FogPhysicalRuns(
        back=back,
        ankle=ankle,
        segment_receipts=tuple(receipts),
        parsed_columns=tuple(columns),
    )


def _qualified_current_mask(rows: Sequence[Mapping[str, Any]]) -> BoolArray:
    return np.asarray(
        [
            bool(row["ankle_finite_run_contains_window"])
            and bool(row["ankle_own_continuous_50hz_grid_matches_all128_times"])
            and len(cast(Sequence[Any], row["ankle_matching_grid_run_indices"])) == 1
            for row in rows
        ],
        dtype=np.bool_,
    )


def _required_run(
    runs: SensorRunMap,
    session_key: SessionKey,
    run_index: int,
    *,
    sensor: str,
    window_id: str,
) -> UniformPhysicalSegment:
    session_runs = runs.get(session_key)
    if session_runs is None:
        raise ValueError(f"missing {sensor} session for {window_id}")
    segment = session_runs.get(run_index)
    if segment is None:
        raise ValueError(f"missing {sensor} physical run for {window_id}")
    return segment


def materialize_fog_motion_contexts(
    coverage_rows: Sequence[Mapping[str, Any]],
    current_availability_mask: BoolArray,
    back_runs: SensorRunMap,
    ankle_runs: SensorRunMap,
    *,
    context_samples: int = CONTEXT_SAMPLES,
    current_samples: int = CURRENT_SAMPLES,
    target_rate_hz: float = TARGET_RATE_HZ,
    segment_receipts: Sequence[Mapping[str, Any]] = (),
) -> FogMotionContexts:
    """Build right-aligned ankle contexts on the intersection of physical runs.

    The function never reads annotation fields. The back run identified by the
    inherited observable-window ID limits history even when an ankle clock continues
    across a back nonfinite break. Conversely, the explicit matching ankle run limits
    history even when the back run continues. This prevents reset clocks or sensor-only
    gaps from leaking samples across physical boundaries.
    """

    count = len(coverage_rows)
    availability = np.asarray(current_availability_mask, dtype=np.bool_)
    _require(availability.shape == (count,), "current availability is misaligned")
    _require(context_samples >= current_samples > 0, "context geometry is invalid")
    _require(target_rate_hz > 0.0, "target rate must be positive")
    identifiers = np.asarray([str(row["window_id"]) for row in coverage_rows], dtype=np.str_)
    _require(len(set(identifiers.tolist())) == count, "observable window IDs are not unique")
    _require(
        np.array_equal(availability, _qualified_current_mask(coverage_rows)),
        "current availability differs from the label-blind coverage fields",
    )

    signals = np.zeros((count, context_samples, 6), dtype=np.float64)
    gravity = np.zeros((count, context_samples, 3), dtype=np.float64)
    timestamps = np.zeros((count, context_samples), dtype=np.float64)
    observation = np.zeros((count, context_samples), dtype=np.bool_)
    receipts: list[dict[str, Any]] = []

    for global_index, row in enumerate(coverage_rows):
        window_id = str(row["window_id"])
        participant_id = str(row["participant_id"])
        session_id = str(row["session_id"])
        session_key = (participant_id, session_id)
        back_run_index, window_index = _window_coordinates(window_id)
        back_segment = _required_run(
            back_runs,
            session_key,
            back_run_index,
            sensor="back",
            window_id=window_id,
        )
        back_offset = window_index * current_samples
        back_stop = back_offset + current_samples
        _require(
            back_stop <= back_segment.timestamps.size,
            f"observable current window exceeds its back run: {window_id}",
        )
        start_time = float(row["start_time"])
        expected_current_times = start_time + np.arange(current_samples) / target_rate_hz
        back_current_times = np.asarray(
            back_segment.timestamps[back_offset:back_stop], dtype=np.float64
        )
        _require(
            np.allclose(back_current_times, expected_current_times, rtol=0.0, atol=1e-10),
            f"observable back-grid correspondence changed: {window_id}",
        )

        receipt: dict[str, Any] = {
            "global_candidate_index": global_index,
            "window_id": window_id,
            "participant_id": participant_id,
            "session_id": session_id,
            "back_run_index": back_run_index,
            "back_current_offset": back_offset,
            "current_ankle_available": bool(availability[global_index]),
            "history_boundary_inputs": "participant/session/back-run/ankle-run/timestamp/finite",
            "annotation_fields_accessed": False,
        }
        if not availability[global_index]:
            receipt.update(
                {
                    "status": "current_ankle_unavailable",
                    "history_sample_count": 0,
                    "full_context": False,
                }
            )
            receipts.append(receipt)
            continue

        matches = tuple(int(value) for value in row["ankle_matching_grid_run_indices"])
        _require(len(matches) == 1, f"qualified ankle match count changed: {window_id}")
        ankle_run_index = matches[0]
        ankle_segment = _required_run(
            ankle_runs,
            session_key,
            ankle_run_index,
            sensor="ankle",
            window_id=window_id,
        )
        ankle_offset = int(
            np.round((start_time - float(ankle_segment.timestamps[0])) * target_rate_hz)
        )
        ankle_stop = ankle_offset + current_samples
        _require(
            0 <= ankle_offset and ankle_stop <= ankle_segment.timestamps.size,
            f"qualified ankle current window exceeds its run: {window_id}",
        )
        ankle_current_times = np.asarray(
            ankle_segment.timestamps[ankle_offset:ankle_stop], dtype=np.float64
        )
        _require(
            np.allclose(ankle_current_times, expected_current_times, rtol=0.0, atol=1e-10),
            f"qualified ankle-grid correspondence changed: {window_id}",
        )

        history_count = min(context_samples, back_stop, ankle_stop)
        ankle_left = ankle_stop - history_count
        back_left = back_stop - history_count
        ankle_times = np.asarray(ankle_segment.timestamps[ankle_left:ankle_stop], dtype=np.float64)
        back_times = np.asarray(back_segment.timestamps[back_left:back_stop], dtype=np.float64)
        _require(
            ankle_times.shape == back_times.shape == (history_count,)
            and np.allclose(ankle_times, back_times, rtol=0.0, atol=1e-10),
            f"back/ankle history grids diverged: {window_id}",
        )
        target_left = context_samples - history_count
        selected_signal = np.asarray(ankle_segment.signals[ankle_left:ankle_stop], dtype=np.float64)
        selected_gravity = np.asarray(
            ankle_segment.gravity[ankle_left:ankle_stop], dtype=np.float64
        )
        _require(
            selected_signal.shape == (history_count, 6)
            and selected_gravity.shape == (history_count, 3)
            and bool(np.isfinite(selected_signal).all())
            and bool(np.isfinite(selected_gravity).all()),
            f"ankle history payload is invalid: {window_id}",
        )
        signals[global_index, target_left:] = selected_signal
        gravity[global_index, target_left:] = selected_gravity
        timestamps[global_index, target_left:] = ankle_times
        observation[global_index, target_left:] = True
        reasons: list[str] = []
        if back_stop < context_samples:
            reasons.append("back_run_history")
        if ankle_stop < context_samples:
            reasons.append("ankle_run_history")
        status = "full_context" if not reasons else "insufficient_" + "_and_".join(reasons)
        receipt.update(
            {
                "status": status,
                "ankle_run_index": ankle_run_index,
                "ankle_current_offset": ankle_offset,
                "history_sample_count": history_count,
                "full_context": history_count == context_samples,
                "history_start_time": float(ankle_times[0]),
                "current_start_time": start_time,
                "context_end_time_exclusive": float(expected_current_times[-1])
                + 1.0 / target_rate_hz,
                "timestamp_sha256": _array_sha256(ankle_times),
                "signal_sha256": _array_sha256(selected_signal),
                "gravity_sha256": _array_sha256(selected_gravity),
            }
        )
        receipts.append(receipt)

    full_context = np.asarray(observation.all(axis=1), dtype=np.bool_)
    contexts = FogMotionContexts(
        signals=signals,
        derived_gravity=gravity,
        timestamps=timestamps,
        observation_mask=observation,
        current_availability_mask=availability.copy(),
        full_context_mask=full_context,
        observable_window_ids=identifiers,
        alignment_receipts=tuple(receipts),
        segment_receipts=tuple(dict(item) for item in segment_receipts),
    )
    contexts.validate(context_samples=context_samples, current_samples=current_samples)
    return contexts


def _load_scoring_contract(
    spatial_cache_path: Path,
    *,
    coverage_ids: StringArray,
    current_availability: BoolArray,
) -> dict[str, NDArray[Any]]:
    with np.load(spatial_cache_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    required = {
        "back_values",
        "back_names",
        "ankle_values",
        "ankle_names",
        "available_indices",
        "availability_mask",
        "observable_window_ids",
        "observable_participant_ids",
        "observable_fold_index",
        "scoring_indices",
        "scoring_eligibility",
        "scored_labels",
        "scored_participant_ids",
        "scored_window_ids",
        "scored_fold_index",
    }
    _require(required <= arrays.keys(), "spatial cache lacks required base-contract arrays")
    observable_ids = np.asarray(arrays["observable_window_ids"], dtype=np.str_)
    scoring_indices = np.asarray(arrays["scoring_indices"], dtype=np.int64)
    scoring_eligibility = np.asarray(arrays["scoring_eligibility"], dtype=np.bool_)
    _require(
        observable_ids.shape == coverage_ids.shape and np.array_equal(observable_ids, coverage_ids),
        "coverage and cached observable IDs differ",
    )
    _require(
        scoring_indices.shape == (SCORED_COUNT,)
        and bool(np.all(np.diff(scoring_indices) > 0))
        and int(scoring_indices[0]) >= 0
        and int(scoring_indices[-1]) < coverage_ids.size,
        "scoring projection is invalid",
    )
    expected_scoring = np.zeros(coverage_ids.size, dtype=np.bool_)
    expected_scoring[scoring_indices] = True
    _require(
        np.array_equal(scoring_eligibility, expected_scoring),
        "scoring eligibility differs from scoring indices",
    )
    _require(
        np.array_equal(
            np.asarray(arrays["availability_mask"], dtype=np.bool_), current_availability
        )
        and np.array_equal(
            np.asarray(arrays["available_indices"], dtype=np.int64),
            np.flatnonzero(current_availability),
        ),
        "cached current-ankle availability changed",
    )
    for observable_name, scored_name in (
        ("observable_window_ids", "scored_window_ids"),
        ("observable_participant_ids", "scored_participant_ids"),
        ("observable_fold_index", "scored_fold_index"),
    ):
        _require(
            np.array_equal(arrays[observable_name][scoring_indices], arrays[scored_name]),
            f"cached scoring projection changed: {scored_name}",
        )
    labels = np.asarray(arrays["scored_labels"], dtype=np.int64)
    _require(
        tuple(np.bincount(labels, minlength=3).tolist()) == EXPECTED_SCORED_CLASS_COUNTS,
        "base scored class support changed",
    )
    _require(
        np.asarray(arrays["back_values"]).shape == (OBSERVABLE_COUNT, 80),
        "base back feature shape changed",
    )
    ankle_features = np.asarray(arrays["ankle_values"], dtype=np.float64)
    _require(
        ankle_features.shape == (CURRENT_AVAILABLE_COUNT, 80)
        and _array_sha256(ankle_features) == ANKLE_FEATURE_SHA256,
        "base ankle feature cache changed",
    )
    return arrays


def _support_receipt(
    contexts: FogMotionContexts,
    arrays: Mapping[str, NDArray[Any]],
) -> dict[str, Any]:
    scoring = np.asarray(arrays["scoring_eligibility"], dtype=np.bool_)
    scoring_indices = np.asarray(arrays["scoring_indices"], dtype=np.int64)
    labels = np.asarray(arrays["scored_labels"], dtype=np.int64)
    folds = np.asarray(arrays["observable_fold_index"], dtype=np.int64)
    people = np.asarray(arrays["observable_participant_ids"], dtype=np.str_)
    full_scored = contexts.full_context_mask[scoring_indices]
    _require(
        int(contexts.full_context_mask.sum()) == FULL_CONTEXT_OBSERVABLE_COUNT,
        "full-context observable forecast failed",
    )
    _require(
        int(full_scored.sum()) == FULL_CONTEXT_SCORED_COUNT, "full-context score forecast failed"
    )
    _require(
        tuple(np.bincount(labels[full_scored], minlength=3).tolist())
        == EXPECTED_FULL_CONTEXT_CLASS_COUNTS,
        "full-context class forecast failed",
    )
    evaluation_rows: list[int] = []
    training_rows: list[int] = []
    training_classes: list[tuple[int, int, int]] = []
    observable_labels = np.zeros(contexts.observable_window_ids.size, dtype=np.int64)
    observable_labels[scoring] = labels
    for fold in range(5):
        evaluation = scoring & contexts.full_context_mask & (folds == fold)
        training = scoring & contexts.full_context_mask & (folds != fold)
        evaluation_rows.append(int(evaluation.sum()))
        training_rows.append(int(training.sum()))
        class_counts = np.bincount(observable_labels[training], minlength=3)
        training_classes.append((int(class_counts[0]), int(class_counts[1]), int(class_counts[2])))
    _require(
        tuple(evaluation_rows) == EXPECTED_FULL_CONTEXT_EVALUATION_ROWS,
        "full-context evaluation-fold forecast failed",
    )
    _require(
        tuple(training_rows) == EXPECTED_FULL_CONTEXT_TRAINING_ROWS,
        "full-context training-fold forecast failed",
    )
    _require(
        tuple(training_classes) == EXPECTED_FULL_CONTEXT_TRAINING_CLASSES,
        "full-context training class forecast failed",
    )
    p017_scored = scoring & (people == "fogstar:017")
    _require(
        int((p017_scored & contexts.full_context_mask).sum()) == 0,
        "P017 structural non-intervention forecast failed",
    )
    available_scored = int((scoring & contexts.current_availability_mask).sum())
    _require(available_scored == 1_154, "current-ankle scored support changed")
    return {
        "observable_count": int(contexts.observable_window_ids.size),
        "scored_count": int(scoring.sum()),
        "current_ankle_available_observable": int(contexts.current_availability_mask.sum()),
        "current_ankle_available_scored": available_scored,
        "full_context_observable": int(contexts.full_context_mask.sum()),
        "full_context_scored": int(full_scored.sum()),
        "full_context_scored_class_counts": list(EXPECTED_FULL_CONTEXT_CLASS_COUNTS),
        "full_context_evaluation_rows_by_fold": evaluation_rows,
        "full_context_training_rows_by_fold": training_rows,
        "full_context_training_class_counts_by_fold": [list(row) for row in training_classes],
        "current_available_short_context_scored": available_scored - int(full_scored.sum()),
        "ankle_unavailable_scored": int(scoring.sum()) - available_scored,
        "P017_full_context_scored": 0,
    }


def preflight_fog_motion_context(
    *,
    raw_path: Path,
    coverage_path: Path,
    spatial_cache_path: Path,
) -> FogMotionContextPreflight:
    """Replay the exact source/base/context qualification without fitting a model."""

    _require(
        raw_path.is_file()
        and raw_path.stat().st_size == RAW_SIZE
        and sha256_file(raw_path) == RAW_SHA256,
        "pinned FoG source object changed",
    )
    _require(
        coverage_path.is_file() and sha256_file(coverage_path) == COVERAGE_SHA256,
        "pinned FoG coverage audit changed",
    )
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    _require(isinstance(coverage, dict), "FoG coverage audit must be an object")
    rows = cast(list[dict[str, Any]], coverage.get("candidate_availability"))
    _require(isinstance(rows, list) and len(rows) == OBSERVABLE_COUNT, "observable count changed")
    current_availability = availability_mask_from_rows(rows)
    coverage_ids = np.asarray([str(row["window_id"]) for row in rows], dtype=np.str_)
    _require(
        _availability_array_sha256(current_availability) == MASK_SHA256
        and _availability_array_sha256(coverage_ids) == ORDERED_ID_SHA256,
        "frozen availability or observable-ID hash changed",
    )

    physical = reconstruct_fog_physical_runs(raw_path)
    contexts = materialize_fog_motion_contexts(
        rows,
        current_availability,
        physical.back,
        physical.ankle,
        segment_receipts=physical.segment_receipts,
    )
    current_signals = contexts.signals[current_availability, -CURRENT_SAMPLES:]
    current_gravity = contexts.derived_gravity[current_availability, -CURRENT_SAMPLES:]
    _require(
        _array_sha256(current_signals) == ANKLE_SIGNAL_SHA256,
        "reconstructed current ankle signals changed",
    )
    _require(
        _array_sha256(current_gravity) == ANKLE_GRAVITY_SHA256,
        "reconstructed current ankle gravity changed",
    )
    current_features = extract_engineered_features(
        current_signals,
        channel_names=(
            "lin_acc_x",
            "lin_acc_y",
            "lin_acc_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
        ),
    )
    _require(
        _array_sha256(current_features.values) == ANKLE_FEATURE_SHA256,
        "reconstructed current ankle features changed",
    )

    # Scoring metadata is intentionally opened only after the physical context is frozen.
    arrays = _load_scoring_contract(
        spatial_cache_path,
        coverage_ids=coverage_ids,
        current_availability=current_availability,
    )
    support = _support_receipt(contexts, arrays)
    receipt_without_hash: dict[str, Any] = {
        "record_kind": "fog_motion_context_preflight_v1",
        "status": "pass",
        "model_fits": 0,
        "raw_source": {
            "path": str(raw_path),
            "sha256": RAW_SHA256,
            "size_bytes": RAW_SIZE,
        },
        "coverage_audit": {"path": str(coverage_path), "sha256": COVERAGE_SHA256},
        "spatial_cache_path": str(spatial_cache_path),
        "parsed_columns": list(physical.parsed_columns),
        "annotation_columns_parsed_for_context": [],
        "context_frozen_before_scoring_metadata_load": True,
        "segmentation_inputs": [
            "participant_id",
            "session_id",
            "timestamp_monotonicity",
            "timestamp_gap",
            "finite_sensor_rows",
            "back_physical_run",
            "ankle_physical_run",
        ],
        "source_acceleration_unit": "g",
        "model_linear_acceleration_unit": "m/s^2",
        "source_gyroscope_unit": "degree/s",
        "model_gyroscope_unit": "radian/s",
        "gravity_mode": "causal_native_rate_derived_from_total_acceleration",
        "resampling": "single_joint_offline_polyphase_60_to_50_hz_per_physical_run",
        "zero_lookahead_streaming_claim": False,
        "context_samples": CONTEXT_SAMPLES,
        "current_samples": CURRENT_SAMPLES,
        "signals_sha256": _array_sha256(contexts.signals),
        "derived_gravity_sha256": _array_sha256(contexts.derived_gravity),
        "timestamps_sha256": _array_sha256(contexts.timestamps),
        "observation_mask_sha256": _array_sha256(contexts.observation_mask),
        "full_context_mask_sha256": _array_sha256(contexts.full_context_mask),
        "alignment_receipts_sha256": canonical_json_sha256(list(contexts.alignment_receipts)),
        "segment_receipts_sha256": canonical_json_sha256(list(contexts.segment_receipts)),
        "historical_current_window_hashes_verified": {
            "signal": ANKLE_SIGNAL_SHA256,
            "gravity": ANKLE_GRAVITY_SHA256,
            "features": ANKLE_FEATURE_SHA256,
        },
        "support": support,
        "InclusiveHAR_P11_P20_loaded": False,
    }
    receipt = dict(receipt_without_hash)
    receipt["record_sha256"] = canonical_json_sha256(receipt_without_hash)
    return FogMotionContextPreflight(
        contexts=contexts,
        scoring_indices=np.asarray(arrays["scoring_indices"], dtype=np.int64),
        scoring_eligibility=np.asarray(arrays["scoring_eligibility"], dtype=np.bool_),
        scored_labels=np.asarray(arrays["scored_labels"], dtype=np.int64),
        observable_participant_ids=np.asarray(arrays["observable_participant_ids"], dtype=np.str_),
        observable_fold_index=np.asarray(arrays["observable_fold_index"], dtype=np.int64),
        receipt=receipt,
    )


def context_reason_counts(contexts: FogMotionContexts) -> dict[str, int]:
    """Count explicit context outcomes without consulting scoring labels."""

    counts: defaultdict[str, int] = defaultdict(int)
    for row in contexts.alignment_receipts:
        counts[str(row["status"])] += 1
    return dict(sorted(counts.items()))


def participant_class_weights(labels: IntArray, participant_ids: StringArray) -> FloatArray:
    """Return mean-one 1/(k_i*n_ic) weights for represented original classes."""

    y = np.asarray(labels, dtype=np.int64)
    people = np.asarray(participant_ids, dtype=np.str_)
    _require(
        y.ndim == people.ndim == 1 and y.size == people.size and y.size > 0,
        "weight arrays are not aligned",
    )
    _require(set(np.unique(y).tolist()).issubset({0, 1, 2}), "original class schema changed")
    weights = np.empty(y.size, dtype=np.float64)
    for participant in np.unique(people):
        selected = people == participant
        represented = np.unique(y[selected])
        _require(represented.size > 0, "participant has no eligible training row")
        for label in represented:
            cell = selected & (y == label)
            weights[cell] = 1.0 / (int(represented.size) * int(cell.sum()))
    weights /= weights.mean()
    _require(
        bool(np.isfinite(weights).all() and np.all(weights > 0))
        and abs(float(weights.mean()) - 1.0) <= 1e-12,
        "participant-class weights are invalid",
    )
    return weights


def history_block_permutation(window_id: str) -> IntArray:
    """Return the fixed content-independent order of the three 124-sample history blocks."""

    _require(bool(window_id), "window ID must be nonempty")
    order = sorted(
        range(3),
        key=lambda index: (
            hashlib.sha256(f"motion-history-v1|{window_id}|{index}".encode()).digest(),
            index,
        ),
    )
    return np.asarray(order, dtype=np.int64)


def apply_history_block_permutation(
    values: NDArray[np.floating],
    observation_mask: NDArray[np.generic],
    permutation: NDArray[np.integer] | Sequence[int],
) -> tuple[NDArray[Any], NDArray[Any]]:
    """Reorder only three past blocks, jointly preserving channels, mask, and current query."""

    signal = np.asarray(values)
    mask = np.asarray(observation_mask)
    order = np.asarray(permutation, dtype=np.int64)
    _require(signal.ndim == 2 and signal.shape[0] == 500, "history signal shape changed")
    _require(mask.shape == (500,), "history mask shape changed")
    _require(sorted(order.tolist()) == [0, 1, 2], "history permutation is invalid")
    indices = np.concatenate(
        (*[np.arange(index * 124, (index + 1) * 124) for index in order], np.arange(372, 500))
    )
    return signal[indices].copy(), mask[indices].copy()


def compose_motion_probability(
    motion_probability: NDArray[np.floating],
    l9v_probabilities: NDArray[np.floating],
    intervention_mask: NDArray[np.bool_],
    b0_probabilities: NDArray[np.floating],
) -> FloatArray:
    """Replace only motion mass while retaining L9v conditional posture and exact fallbacks."""

    motion = np.asarray(motion_probability, dtype=np.float64)
    baseline = np.asarray(l9v_probabilities, dtype=np.float64)
    intervene = np.asarray(intervention_mask, dtype=np.bool_)
    b0 = np.asarray(b0_probabilities, dtype=np.float64)
    _require(
        motion.ndim == 1
        and baseline.shape == b0.shape == (motion.size, 3)
        and intervene.shape == motion.shape,
        "composition arrays are not aligned",
    )
    _require(
        bool(np.isfinite(motion).all())
        and bool(np.all((motion >= 0.0) & (motion <= 1.0)))
        and bool(np.isfinite(baseline).all())
        and bool(np.isfinite(b0).all()),
        "composition probabilities are invalid",
    )
    _require(
        np.allclose(baseline.sum(axis=1), 1.0, rtol=0.0, atol=1e-12)
        and np.allclose(b0.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "reference probabilities do not sum to one",
    )
    result = baseline.copy()
    stationary = baseline[:, 1] + baseline[:, 2]
    eligible = intervene & (stationary > 0.0)
    q = np.divide(baseline[:, 1], stationary, out=np.zeros_like(stationary), where=stationary > 0.0)
    result[eligible, 0] = motion[eligible]
    result[eligible, 1] = (1.0 - motion[eligible]) * q[eligible]
    result[eligible, 2] = (1.0 - motion[eligible]) * (1.0 - q[eligible])
    _require(
        np.allclose(result.sum(axis=1), 1.0, rtol=0.0, atol=1e-12),
        "composed probabilities do not sum to one",
    )
    return result


class _CausalMaskedSublayer(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.left_padding = 4 * dilation
        self.convolution = nn.Conv1d(
            32, 32, kernel_size=5, stride=1, padding=0, dilation=dilation, bias=True
        )
        self.normalization = nn.LayerNorm(32, eps=1e-5, elementwise_affine=True)
        self.activation = nn.GELU(approximate="none")
        self.dropout = nn.Dropout(0.1)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        hidden = self.convolution(F.pad(values, (self.left_padding, 0)))
        hidden = self.normalization(hidden.transpose(1, 2)).transpose(1, 2)
        hidden = self.dropout(self.activation(hidden))
        return cast(Tensor, hidden * mask[:, None, :].to(hidden.dtype))


class _CausalResidualBlock(nn.Module):
    def __init__(self, dilation: int) -> None:
        super().__init__()
        self.first = _CausalMaskedSublayer(dilation)
        self.second = _CausalMaskedSublayer(dilation)

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        hidden = self.first(values, mask)
        hidden = self.second(hidden, mask)
        return cast(Tensor, (values + hidden) * mask[:, None, :].to(hidden.dtype))


class MotionResidualTCN(nn.Module):
    """Fixed 62,881-parameter causal signal residual on a frozen E2 motion logit."""

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Conv1d(7, 32, kernel_size=1, bias=True)
        self.blocks = nn.ModuleList(
            [_CausalResidualBlock(dilation) for dilation in (1, 2, 4, 8, 16, 32)]
        )
        self.head = nn.Linear(32, 1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def encode_sequence(self, values: Tensor, observation_mask: Tensor) -> Tensor:
        if values.ndim != 3 or values.shape[1:] != (500, 6):
            raise ValueError("motion model expects [batch,500,6]")
        if observation_mask.shape != values.shape[:2]:
            raise ValueError("motion observation mask is not aligned")
        mask = observation_mask.to(dtype=values.dtype)
        inputs = torch.cat((values * mask[:, :, None], mask[:, :, None]), dim=2)
        hidden = self.stem(inputs.transpose(1, 2))
        hidden = hidden * mask[:, None, :]
        for block in self.blocks:
            hidden = block(hidden, observation_mask)
        return cast(Tensor, hidden.transpose(1, 2))

    def forward(self, values: Tensor, observation_mask: Tensor, e2_logit: Tensor) -> Tensor:
        if e2_logit.shape != (values.shape[0],):
            raise ValueError("E2 logits are not aligned")
        encoded = self.encode_sequence(values, observation_mask)
        residual = self.head(encoded[:, -128:].mean(dim=1)).squeeze(1)
        return cast(Tensor, e2_logit + residual)
