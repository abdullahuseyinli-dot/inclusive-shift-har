"""Deterministic, annotation-independent signal analysis for the physical pilot.

The code in this module starts at a device-specific *canonical* recording.  It
does not parse a vendor format and it deliberately has no model-fitting path.
Device identity, units, channel semantics, gravity provenance, and the frozen
processing choices must be supplied explicitly before any signal result can be
computed.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

from inclusive_shift_har.manifests.canonical import canonical_json_bytes

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]
IntArray: TypeAlias = npt.NDArray[np.int64]

GRAVITY_STANDARD = 9.80665
POSTURES = ("sitting", "standing")
MOTIONS = ("quiet", "upper_body_motion")
T_CRITICAL_975 = {
    1: 12.706204736432095,
    2: 4.302652729749462,
    3: 3.182446305284263,
    4: 2.7764451051977987,
    5: 2.570581835636314,
}


class SignalContractError(ValueError):
    """Raised when canonical input is ambiguous or violates its frozen contract."""


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SignalContractError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _exact_fields(
    value: Mapping[str, Any], required: set[str], optional: set[str], name: str
) -> None:
    missing = sorted(required - set(value))
    unexpected = sorted(set(value) - required - optional)
    if missing:
        raise SignalContractError(f"{name} fields missing: {missing}")
    if unexpected:
        raise SignalContractError(f"unexpected {name} fields: {unexpected}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SignalContractError(f"{name} must be a nonempty string")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SignalContractError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result) or (positive and result <= 0.0):
        raise SignalContractError(f"{name} must be a finite positive number")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SignalContractError(f"{name} must be an integer")
    if positive and value <= 0:
        raise SignalContractError(f"{name} must be positive")
    return value


def _sha256(value: object, name: str) -> str:
    result = _string(value, name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SignalContractError(f"{name} must be a lowercase SHA-256")
    return result


@dataclass(frozen=True)
class ProcessingContract:
    """Frozen numeric processing choices for one qualified device stream."""

    source_rate_hz: float
    target_rate_hz: float
    gravity_source: str
    gravity_cutoff_hz: float | None
    maximum_gap_seconds: float
    resampling_method: str
    window_samples: int
    window_stride_samples: int


@dataclass(frozen=True)
class DeviceSignalContract:
    """Explicit bridge from a qualified device export to canonical SI arrays."""

    record_kind: str
    schema_version: str
    status: str
    evidence_status: str
    device_id: str
    manufacturer: str
    model: str
    firmware: str
    logger: str
    axis_convention: str
    clock_domain: str
    channel_names: dict[str, str]
    timestamp_unit: str
    acceleration_unit: str
    angular_velocity_unit: str
    gravity_unit: str
    acceleration_semantics: str
    native_gravity_available: bool
    processing: ProcessingContract
    frame_conditioning_limit: float | None
    equipment_receipt_sha256: str
    placement_instruction_sha256: str
    bench_qualification_sha256: str
    canonicalization_receipt_sha256: str
    processing_frozen_at_utc: str


CONTRACT_FIELDS = {
    "record_kind",
    "schema_version",
    "status",
    "evidence_status",
    "device_id",
    "manufacturer",
    "model",
    "firmware",
    "logger",
    "axis_convention",
    "clock_domain",
    "channel_names",
    "units",
    "acceleration_semantics",
    "native_gravity_available",
    "processing_contract",
    "frame_conditioning_limit",
    "equipment_receipt_sha256",
    "placement_instruction_sha256",
    "bench_qualification_sha256",
    "canonicalization_receipt_sha256",
    "processing_frozen_at_utc",
}
BASE_CHANNEL_KEYS = {
    "sample_id",
    "timestamp",
    "recording_break_before",
    "ax",
    "ay",
    "az",
    "gx",
    "gy",
    "gz",
}
GRAVITY_CHANNEL_KEYS = {"gravity_x", "gravity_y", "gravity_z"}


def parse_device_signal_contract(value: Mapping[str, Any]) -> DeviceSignalContract:
    """Parse a strict contract; no device-specific value receives a default."""

    _exact_fields(value, CONTRACT_FIELDS, set(), "signal contract")
    if value.get("record_kind") != "physical_information_signal_contract":
        raise SignalContractError("unexpected signal contract record_kind")
    if value.get("schema_version") != "1.0.0":
        raise SignalContractError("unexpected signal contract schema_version")
    if value.get("status") != "qualified_and_processing_frozen":
        raise SignalContractError("device contract is not qualified and frozen")
    evidence_status = _string(value.get("evidence_status"), "evidence_status")
    if evidence_status not in {
        "development_physical_measurement",
        "synthetic_fixture_not_physical_evidence",
    }:
        raise SignalContractError("unsupported evidence_status")
    native = value.get("native_gravity_available")
    if not isinstance(native, bool):
        raise SignalContractError("native_gravity_available must be boolean")
    channels = _mapping(value.get("channel_names"), "channel_names")
    expected_channel_keys = BASE_CHANNEL_KEYS | (GRAVITY_CHANNEL_KEYS if native else set())
    _exact_fields(channels, expected_channel_keys, set(), "channel_names")
    channel_names = {key: _string(item, f"channel_names.{key}") for key, item in channels.items()}
    if len(set(channel_names.values())) != len(channel_names):
        raise SignalContractError("channel_names values must be unique")
    units = _mapping(value.get("units"), "units")
    _exact_fields(
        units,
        {"timestamp", "acceleration", "angular_velocity", "gravity"},
        set(),
        "units",
    )
    timestamp_unit = _string(units.get("timestamp"), "units.timestamp")
    acceleration_unit = _string(units.get("acceleration"), "units.acceleration")
    angular_velocity_unit = _string(units.get("angular_velocity"), "units.angular_velocity")
    gravity_unit = _string(units.get("gravity"), "units.gravity")
    if timestamp_unit not in {"s", "ms"}:
        raise SignalContractError("timestamp unit must be 's' or 'ms'")
    if acceleration_unit not in {"m/s^2", "g"} or gravity_unit not in {"m/s^2", "g"}:
        raise SignalContractError("acceleration and gravity units must be 'm/s^2' or 'g'")
    if angular_velocity_unit not in {"rad/s", "deg/s"}:
        raise SignalContractError("angular velocity unit must be 'rad/s' or 'deg/s'")
    semantics = _string(value.get("acceleration_semantics"), "acceleration_semantics")
    if semantics not in {
        "total_acceleration_including_gravity",
        "linear_acceleration_gravity_removed",
    }:
        raise SignalContractError("unsupported acceleration_semantics")
    processing_value = _mapping(value.get("processing_contract"), "processing_contract")
    _exact_fields(
        processing_value,
        {
            "source_rate_hz",
            "target_rate_hz",
            "gravity_source",
            "gravity_cutoff_hz",
            "maximum_gap_seconds",
            "resampling_method",
            "window_samples",
            "window_stride_samples",
        },
        set(),
        "processing_contract",
    )
    source_rate = _number(
        processing_value.get("source_rate_hz"), "processing_contract.source_rate_hz", positive=True
    )
    target_rate = _number(
        processing_value.get("target_rate_hz"), "processing_contract.target_rate_hz", positive=True
    )
    maximum_gap = _number(
        processing_value.get("maximum_gap_seconds"),
        "processing_contract.maximum_gap_seconds",
        positive=True,
    )
    if maximum_gap < 1.0 / source_rate:
        raise SignalContractError("maximum_gap_seconds is below the nominal source interval")
    gravity_source = _string(
        processing_value.get("gravity_source"), "processing_contract.gravity_source"
    )
    if gravity_source not in {"provider_native_gravity", "derived_causal_lowpass"}:
        raise SignalContractError("unsupported gravity_source")
    cutoff_value = processing_value.get("gravity_cutoff_hz")
    cutoff = (
        None
        if cutoff_value is None
        else _number(cutoff_value, "processing_contract.gravity_cutoff_hz", positive=True)
    )
    if gravity_source == "provider_native_gravity":
        if not native or cutoff is not None:
            raise SignalContractError("native gravity requires native channels and a null cutoff")
    else:
        if semantics != "total_acceleration_including_gravity":
            raise SignalContractError("derived gravity requires total acceleration")
        if cutoff is None or cutoff >= 0.5 * source_rate or cutoff >= 0.5 * target_rate:
            raise SignalContractError("derived gravity cutoff must be below both Nyquist rates")
    method = _string(
        processing_value.get("resampling_method"), "processing_contract.resampling_method"
    )
    if method != "linear_interpolation_v1":
        raise SignalContractError("resampling_method must be linear_interpolation_v1")
    processing = ProcessingContract(
        source_rate_hz=source_rate,
        target_rate_hz=target_rate,
        gravity_source=gravity_source,
        gravity_cutoff_hz=cutoff,
        maximum_gap_seconds=maximum_gap,
        resampling_method=method,
        window_samples=_integer(
            processing_value.get("window_samples"),
            "processing_contract.window_samples",
            positive=True,
        ),
        window_stride_samples=_integer(
            processing_value.get("window_stride_samples"),
            "processing_contract.window_stride_samples",
            positive=True,
        ),
    )
    frame_limit_value = value.get("frame_conditioning_limit")
    frame_limit = (
        None
        if frame_limit_value is None
        else _number(frame_limit_value, "frame_conditioning_limit", positive=True)
    )
    frozen = _string(value.get("processing_frozen_at_utc"), "processing_frozen_at_utc")
    if not frozen.endswith("Z"):
        raise SignalContractError("processing_frozen_at_utc must be explicit UTC ending in Z")
    return DeviceSignalContract(
        record_kind="physical_information_signal_contract",
        schema_version="1.0.0",
        status="qualified_and_processing_frozen",
        evidence_status=evidence_status,
        device_id=_string(value.get("device_id"), "device_id"),
        manufacturer=_string(value.get("manufacturer"), "manufacturer"),
        model=_string(value.get("model"), "model"),
        firmware=_string(value.get("firmware"), "firmware"),
        logger=_string(value.get("logger"), "logger"),
        axis_convention=_string(value.get("axis_convention"), "axis_convention"),
        clock_domain=_string(value.get("clock_domain"), "clock_domain"),
        channel_names=channel_names,
        timestamp_unit=timestamp_unit,
        acceleration_unit=acceleration_unit,
        angular_velocity_unit=angular_velocity_unit,
        gravity_unit=gravity_unit,
        acceleration_semantics=semantics,
        native_gravity_available=native,
        processing=processing,
        frame_conditioning_limit=frame_limit,
        equipment_receipt_sha256=_sha256(
            value.get("equipment_receipt_sha256"), "equipment_receipt_sha256"
        ),
        placement_instruction_sha256=_sha256(
            value.get("placement_instruction_sha256"), "placement_instruction_sha256"
        ),
        bench_qualification_sha256=_sha256(
            value.get("bench_qualification_sha256"), "bench_qualification_sha256"
        ),
        canonicalization_receipt_sha256=_sha256(
            value.get("canonicalization_receipt_sha256"),
            "canonicalization_receipt_sha256",
        ),
        processing_frozen_at_utc=frozen,
    )


@dataclass(frozen=True)
class CanonicalRecording:
    """One attachment-bound canonical continuous recording before segmentation."""

    recording_id: str
    block_id: str
    wearer_id: str
    visit_number: int
    attachment_number: int
    sample_ids: IntArray
    timestamps: FloatArray
    acceleration_xyz: FloatArray
    gyroscope_xyz: FloatArray
    recording_break_before: BoolArray
    native_gravity_xyz: FloatArray | None = None


@dataclass(frozen=True)
class ResampledRun:
    recording_id: str
    block_id: str
    wearer_id: str
    run_index: int
    source_start_index: int
    source_stop_index_exclusive: int
    timestamps: FloatArray
    total_acceleration_xyz: FloatArray
    linear_acceleration_xyz: FloatArray
    gyroscope_xyz: FloatArray
    gravity_xyz: FloatArray


@dataclass(frozen=True)
class WindowTable:
    """Global signal-derived window table; annotations never enter its construction."""

    window_ids: tuple[str, ...]
    recording_ids: tuple[str, ...]
    block_ids: tuple[str, ...]
    wearer_ids: tuple[str, ...]
    run_indices: IntArray
    source_run_indices: IntArray
    source_start_sample_indices: IntArray
    source_stop_sample_indices_exclusive: IntArray
    starts_seconds: FloatArray
    ends_seconds: FloatArray
    gravity_directions: FloatArray
    dynamic_acceleration_rms: FloatArray
    gyroscope_rms: FloatArray
    valid: BoolArray
    invalid_reasons: tuple[str, ...]


def _as_float_matrix(value: npt.ArrayLike, name: str, rows: int) -> FloatArray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (rows, 3):
        raise SignalContractError(f"{name} must have shape ({rows}, 3)")
    return result


def canonical_recording_from_arrays(
    *,
    metadata: Mapping[str, Any],
    arrays: Mapping[str, npt.ArrayLike],
    contract: DeviceSignalContract,
) -> CanonicalRecording:
    """Validate canonical array names/shapes and convert declared units to SI."""

    required_arrays = {
        "sample_ids",
        "timestamps",
        "acceleration_xyz",
        "gyroscope_xyz",
        "recording_break_before",
    }
    if contract.native_gravity_available:
        required_arrays.add("native_gravity_xyz")
    _exact_fields(arrays, required_arrays, set(), "canonical arrays")
    sample_ids = np.asarray(arrays["sample_ids"])
    if sample_ids.ndim != 1 or sample_ids.dtype.kind not in "iu":
        raise SignalContractError("sample_ids must be a one-dimensional integer array")
    sample_ids = sample_ids.astype(np.int64, copy=False)
    rows = int(sample_ids.size)
    if rows == 0 or np.unique(sample_ids).size != rows:
        raise SignalContractError("sample_ids must be nonempty and unique")
    timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
    breaks = np.asarray(arrays["recording_break_before"])
    if timestamps.shape != (rows,) or breaks.shape != (rows,) or breaks.dtype.kind != "b":
        raise SignalContractError("timestamps and recording_break_before must be length-N arrays")
    acceleration = _as_float_matrix(arrays["acceleration_xyz"], "acceleration_xyz", rows)
    gyroscope = _as_float_matrix(arrays["gyroscope_xyz"], "gyroscope_xyz", rows)
    gravity = (
        _as_float_matrix(arrays["native_gravity_xyz"], "native_gravity_xyz", rows)
        if "native_gravity_xyz" in arrays
        else None
    )
    timestamp_scale = 0.001 if contract.timestamp_unit == "ms" else 1.0
    acceleration_scale = GRAVITY_STANDARD if contract.acceleration_unit == "g" else 1.0
    gyro_scale = math.pi / 180.0 if contract.angular_velocity_unit == "deg/s" else 1.0
    gravity_scale = GRAVITY_STANDARD if contract.gravity_unit == "g" else 1.0
    return CanonicalRecording(
        recording_id=_string(metadata.get("recording_id"), "recording_id"),
        block_id=_string(metadata.get("block_id"), "block_id"),
        wearer_id=_string(metadata.get("wearer_id"), "wearer_id"),
        visit_number=_integer(metadata.get("visit_number"), "visit_number", positive=True),
        attachment_number=_integer(
            metadata.get("attachment_number"), "attachment_number", positive=True
        ),
        sample_ids=sample_ids.copy(),
        timestamps=timestamps * timestamp_scale,
        acceleration_xyz=acceleration * acceleration_scale,
        gyroscope_xyz=gyroscope * gyro_scale,
        recording_break_before=breaks.astype(np.bool_, copy=True),
        native_gravity_xyz=None if gravity is None else gravity * gravity_scale,
    )


def _finite_row(recording: CanonicalRecording, *, include_native_gravity: bool = True) -> BoolArray:
    result = np.isfinite(recording.timestamps)
    result &= np.isfinite(recording.acceleration_xyz).all(axis=1)
    result &= np.isfinite(recording.gyroscope_xyz).all(axis=1)
    if include_native_gravity and recording.native_gravity_xyz is not None:
        result &= np.isfinite(recording.native_gravity_xyz).all(axis=1)
    return result


def continuous_run_slices(
    recording: CanonicalRecording,
    *,
    maximum_gap_seconds: float,
    include_native_gravity: bool = True,
) -> list[tuple[int, int]]:
    """Return half-open finite runs, cutting only at the frozen reset boundaries."""

    if maximum_gap_seconds <= 0.0 or not math.isfinite(maximum_gap_seconds):
        raise SignalContractError("maximum_gap_seconds must be finite and positive")
    finite = _finite_row(recording, include_native_gravity=include_native_gravity)
    runs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index in range(recording.timestamps.size):
        if not finite[index]:
            if start is not None:
                runs.append((start, index))
            start = None
            previous = None
            continue
        split = bool(recording.recording_break_before[index])
        if previous is not None:
            delta = float(recording.timestamps[index] - recording.timestamps[previous])
            split = split or delta <= 0.0 or delta > maximum_gap_seconds
        if split and start is not None:
            runs.append((start, index))
            start = index
        elif start is None:
            start = index
        previous = index
    if start is not None:
        runs.append((start, int(recording.timestamps.size)))
    return [
        (start_index, stop_index) for start_index, stop_index in runs if stop_index > start_index
    ]


def _causal_gravity(total: FloatArray, timestamps: FloatArray, cutoff_hz: float) -> FloatArray:
    gravity = np.empty_like(total)
    gravity[0] = total[0]
    for index in range(1, total.shape[0]):
        delta = float(timestamps[index] - timestamps[index - 1])
        alpha = 1.0 - math.exp(-2.0 * math.pi * cutoff_hz * delta)
        gravity[index] = gravity[index - 1] + alpha * (total[index] - gravity[index - 1])
    return gravity


def _interp_matrix(x: FloatArray, y: FloatArray, query: FloatArray) -> FloatArray:
    return np.column_stack([np.interp(query, x, y[:, axis]) for axis in range(3)])


def resample_recording(
    recording: CanonicalRecording, contract: DeviceSignalContract
) -> list[ResampledRun]:
    """Split, resolve gravity, and linearly resample each independent continuous run."""

    runs: list[ResampledRun] = []
    slices = continuous_run_slices(
        recording,
        maximum_gap_seconds=contract.processing.maximum_gap_seconds,
        include_native_gravity=contract.processing.gravity_source == "provider_native_gravity",
    )
    for run_index, (start, stop) in enumerate(slices):
        source_time = recording.timestamps[start:stop]
        duration = float(source_time[-1] - source_time[0])
        count = math.floor(duration * contract.processing.target_rate_hz + 1.0e-9) + 1
        target_time = source_time[0] + np.arange(count, dtype=np.float64) / (
            contract.processing.target_rate_hz
        )
        target_time = target_time[target_time <= source_time[-1] + 1.0e-10]
        source_acceleration = recording.acceleration_xyz[start:stop]
        source_gyro = recording.gyroscope_xyz[start:stop]
        if contract.processing.gravity_source == "provider_native_gravity":
            if recording.native_gravity_xyz is None:
                raise SignalContractError("provider native gravity array is missing")
            source_gravity = recording.native_gravity_xyz[start:stop]
            gravity = _interp_matrix(source_time, source_gravity, target_time)
            acceleration = _interp_matrix(source_time, source_acceleration, target_time)
            if contract.acceleration_semantics == "total_acceleration_including_gravity":
                total = acceleration
                linear = acceleration - gravity
            else:
                linear = acceleration
                total = acceleration + gravity
        else:
            cutoff = contract.processing.gravity_cutoff_hz
            if cutoff is None:
                raise SignalContractError("derived gravity cutoff is missing")
            source_gravity = _causal_gravity(source_acceleration, source_time, cutoff)
            total = _interp_matrix(source_time, source_acceleration, target_time)
            gravity = _interp_matrix(source_time, source_gravity, target_time)
            linear = total - gravity
        runs.append(
            ResampledRun(
                recording_id=recording.recording_id,
                block_id=recording.block_id,
                wearer_id=recording.wearer_id,
                run_index=run_index,
                source_start_index=start,
                source_stop_index_exclusive=stop,
                timestamps=target_time,
                total_acceleration_xyz=total,
                linear_acceleration_xyz=linear,
                gyroscope_xyz=_interp_matrix(source_time, source_gyro, target_time),
                gravity_xyz=gravity,
            )
        )
    return runs


def _timestamp_epoch_slices(recording: CanonicalRecording) -> list[tuple[int, int]]:
    """Split only where a monotonic time grid cannot be defined.

    Positive gaps stay inside an epoch so their nominal target-grid slots remain
    explicit invalid candidates. Sensor nonfinites also stay inside the time
    epoch; they reset signal processing but must not disappear from denominators.
    """

    finite_timestamp = np.isfinite(recording.timestamps)
    epochs: list[tuple[int, int]] = []
    start: int | None = None
    previous: int | None = None
    for index in range(recording.timestamps.size):
        if not finite_timestamp[index]:
            if start is not None:
                epochs.append((start, index))
            start = None
            previous = None
            continue
        split = bool(recording.recording_break_before[index])
        if previous is not None:
            split = split or recording.timestamps[index] <= recording.timestamps[previous]
        if split and start is not None:
            epochs.append((start, index))
            start = index
        elif start is None:
            start = index
        previous = index
    if start is not None:
        epochs.append((start, int(recording.timestamps.size)))
    return [
        (start_index, stop_index) for start_index, stop_index in epochs if stop_index > start_index
    ]


def _processed_source_values(
    recording: CanonicalRecording,
    contract: DeviceSignalContract,
    start: int,
    stop: int,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    timestamps = recording.timestamps[start:stop]
    acceleration = recording.acceleration_xyz[start:stop]
    gyroscope = recording.gyroscope_xyz[start:stop]
    if contract.processing.gravity_source == "provider_native_gravity":
        if recording.native_gravity_xyz is None:
            raise SignalContractError("provider native gravity array is missing")
        gravity = recording.native_gravity_xyz[start:stop]
        if contract.acceleration_semantics == "total_acceleration_including_gravity":
            total = acceleration
            linear = acceleration - gravity
        else:
            linear = acceleration
            total = acceleration + gravity
    else:
        cutoff = contract.processing.gravity_cutoff_hz
        if cutoff is None:
            raise SignalContractError("derived gravity cutoff is missing")
        total = acceleration
        gravity = _causal_gravity(total, timestamps, cutoff)
        linear = total - gravity
    return total, linear, gyroscope, gravity


def _nominal_source_support(
    source_times: FloatArray, query_times: FloatArray, *, source_offset: int
) -> tuple[IntArray, IntArray]:
    starts = np.empty(query_times.size, dtype=np.int64)
    stops = np.empty(query_times.size, dtype=np.int64)
    for index, query in enumerate(query_times):
        insertion = int(np.searchsorted(source_times, query, side="left"))
        exact = (
            insertion < source_times.size and abs(float(source_times[insertion] - query)) <= 1.0e-10
        )
        if exact:
            left = insertion
            right_exclusive = insertion + 1
        else:
            left = max(0, insertion - 1)
            right_exclusive = min(source_times.size, insertion + 1)
        starts[index] = source_offset + left
        stops[index] = source_offset + right_exclusive
    return starts, stops


def build_window_table(
    recordings: Sequence[CanonicalRecording],
    contract: DeviceSignalContract,
    *,
    norm_epsilon: float = 1.0e-12,
) -> WindowTable:
    """Create the frozen left-aligned grid without accepting annotation inputs."""

    if norm_epsilon <= 0.0:
        raise SignalContractError("norm_epsilon must be positive")
    ids: list[str] = []
    recording_ids: list[str] = []
    block_ids: list[str] = []
    wearer_ids: list[str] = []
    run_indices: list[int] = []
    source_run_indices: list[int] = []
    source_starts: list[int] = []
    source_stops: list[int] = []
    starts: list[float] = []
    ends: list[float] = []
    directions: list[FloatArray] = []
    dynamic_rms: list[float] = []
    gyro_rms: list[float] = []
    valid: list[bool] = []
    reasons: list[str] = []
    length = contract.processing.window_samples
    stride = contract.processing.window_stride_samples
    rate = contract.processing.target_rate_hz
    seen_recordings: set[str] = set()
    for recording in recordings:
        if recording.recording_id in seen_recordings:
            raise SignalContractError(f"duplicate recording_id: {recording.recording_id}")
        seen_recordings.add(recording.recording_id)
        finite_signal = _finite_row(
            recording,
            include_native_gravity=contract.processing.gravity_source == "provider_native_gravity",
        )
        signal_slices = continuous_run_slices(
            recording,
            maximum_gap_seconds=contract.processing.maximum_gap_seconds,
            include_native_gravity=contract.processing.gravity_source == "provider_native_gravity",
        )
        for epoch_index, (epoch_start, epoch_stop) in enumerate(_timestamp_epoch_slices(recording)):
            epoch_times = recording.timestamps[epoch_start:epoch_stop]
            duration = float(epoch_times[-1] - epoch_times[0])
            target_count = math.floor(duration * rate + 1.0e-9) + 1
            target_times = epoch_times[0] + np.arange(target_count, dtype=np.float64) / rate
            target_times = target_times[target_times <= epoch_times[-1] + 1.0e-10]
            target_total = np.full((target_times.size, 3), np.nan, dtype=np.float64)
            target_linear = np.full((target_times.size, 3), np.nan, dtype=np.float64)
            target_gyro = np.full((target_times.size, 3), np.nan, dtype=np.float64)
            target_gravity = np.full((target_times.size, 3), np.nan, dtype=np.float64)
            target_source_run = np.full(target_times.size, -1, dtype=np.int64)
            nominal_starts, nominal_stops = _nominal_source_support(
                epoch_times, target_times, source_offset=epoch_start
            )
            for signal_run_index, (signal_start, signal_stop) in enumerate(signal_slices):
                if signal_start < epoch_start or signal_stop > epoch_stop:
                    continue
                source_times = recording.timestamps[signal_start:signal_stop]
                mask = (target_times >= source_times[0] - 1.0e-10) & (
                    target_times <= source_times[-1] + 1.0e-10
                )
                if not np.any(mask):
                    continue
                total, linear, gyroscope, gravity = _processed_source_values(
                    recording, contract, signal_start, signal_stop
                )
                target_total[mask] = _interp_matrix(source_times, total, target_times[mask])
                target_linear[mask] = _interp_matrix(source_times, linear, target_times[mask])
                target_gyro[mask] = _interp_matrix(source_times, gyroscope, target_times[mask])
                target_gravity[mask] = _interp_matrix(source_times, gravity, target_times[mask])
                target_source_run[mask] = signal_run_index
            for offset in range(0, target_times.size - length + 1, stride):
                stop = offset + length
                gravity_window = target_gravity[offset:stop]
                linear_window = target_linear[offset:stop]
                gyro_window = target_gyro[offset:stop]
                mean_gravity = gravity_window.mean(axis=0)
                norm = float(np.linalg.norm(mean_gravity))
                window_source_start = int(np.min(nominal_starts[offset:stop]))
                window_source_stop = int(np.max(nominal_stops[offset:stop]))
                window_source_runs = target_source_run[offset:stop]
                same_signal_run = bool(
                    np.all(window_source_runs >= 0) and np.unique(window_source_runs).size == 1
                )
                finite = bool(
                    np.isfinite(gravity_window).all()
                    and np.isfinite(linear_window).all()
                    and np.isfinite(gyro_window).all()
                    and math.isfinite(norm)
                )
                reason = "valid"
                source_slice = slice(window_source_start, window_source_stop)
                source_times = recording.timestamps[source_slice]
                source_has_gap = bool(
                    source_times.size > 1
                    and np.any(np.diff(source_times) > contract.processing.maximum_gap_seconds)
                )
                source_has_nonfinite = not bool(np.all(finite_signal[source_slice]))
                if source_has_gap:
                    reason = "timestamp_gap_above_frozen_maximum"
                elif source_has_nonfinite or not finite or not same_signal_run:
                    reason = "nonfinite_inertial_or_gravity_sample"
                elif norm <= norm_epsilon:
                    reason = "gravity_mean_norm_at_or_below_epsilon"
                is_valid = reason == "valid"
                direction = mean_gravity / norm if is_valid else np.full(3, np.nan)
                ids.append(f"{recording.recording_id}/epoch-{epoch_index:04d}/window-{offset:08d}")
                recording_ids.append(recording.recording_id)
                block_ids.append(recording.block_id)
                wearer_ids.append(recording.wearer_id)
                run_indices.append(epoch_index)
                source_run_indices.append(
                    int(window_source_runs[0]) if is_valid and same_signal_run else -1
                )
                source_starts.append(window_source_start)
                source_stops.append(window_source_stop)
                starts.append(float(target_times[offset]))
                ends.append(float(target_times[offset]) + length / rate)
                directions.append(direction)
                dynamic_rms.append(float(np.sqrt(np.mean(np.sum(linear_window**2, axis=1)))))
                gyro_rms.append(float(np.sqrt(np.mean(np.sum(gyro_window**2, axis=1)))))
                valid.append(is_valid)
                reasons.append(reason)
    direction_array = (
        np.vstack(directions).astype(np.float64, copy=False)
        if directions
        else np.empty((0, 3), dtype=np.float64)
    )
    return WindowTable(
        window_ids=tuple(ids),
        recording_ids=tuple(recording_ids),
        block_ids=tuple(block_ids),
        wearer_ids=tuple(wearer_ids),
        run_indices=np.asarray(run_indices, dtype=np.int64),
        source_run_indices=np.asarray(source_run_indices, dtype=np.int64),
        source_start_sample_indices=np.asarray(source_starts, dtype=np.int64),
        source_stop_sample_indices_exclusive=np.asarray(source_stops, dtype=np.int64),
        starts_seconds=np.asarray(starts, dtype=np.float64),
        ends_seconds=np.asarray(ends, dtype=np.float64),
        gravity_directions=direction_array,
        dynamic_acceleration_rms=np.asarray(dynamic_rms, dtype=np.float64),
        gyroscope_rms=np.asarray(gyro_rms, dtype=np.float64),
        valid=np.asarray(valid, dtype=np.bool_),
        invalid_reasons=tuple(reasons),
    )


def build_signal_audit(
    recordings: Sequence[CanonicalRecording],
    contract: DeviceSignalContract,
    windows: WindowTable,
) -> dict[str, Any]:
    """Describe every reset/exclusion and window denominator without annotations."""

    rows: list[dict[str, Any]] = []
    for recording in recordings:
        timestamps = recording.timestamps
        finite = _finite_row(
            recording,
            include_native_gravity=contract.processing.gravity_source == "provider_native_gravity",
        )
        explicit_break_count = int(np.count_nonzero(recording.recording_break_before & finite))
        nonpositive_timestamp_count = 0
        above_maximum_gap_count = 0
        for index in range(1, timestamps.size):
            if not finite[index - 1] or not finite[index]:
                continue
            delta = float(timestamps[index] - timestamps[index - 1])
            if delta <= 0.0:
                nonpositive_timestamp_count += 1
            elif delta > contract.processing.maximum_gap_seconds:
                above_maximum_gap_count += 1
        slices = continuous_run_slices(
            recording,
            maximum_gap_seconds=contract.processing.maximum_gap_seconds,
            include_native_gravity=contract.processing.gravity_source == "provider_native_gravity",
        )
        indices = [
            index
            for index, identifier in enumerate(windows.recording_ids)
            if identifier == recording.recording_id
        ]
        reason_counts: dict[str, int] = defaultdict(int)
        for index in indices:
            reason_counts[windows.invalid_reasons[index]] += 1
        rows.append(
            {
                "recording_id": recording.recording_id,
                "block_id": recording.block_id,
                "wearer_id": recording.wearer_id,
                "source_sample_count": int(timestamps.size),
                "finite_source_sample_count": int(np.count_nonzero(finite)),
                "nonfinite_source_sample_count": int(np.count_nonzero(~finite)),
                "explicit_recording_break_count": explicit_break_count,
                "nonpositive_timestamp_discontinuity_count": nonpositive_timestamp_count,
                "above_maximum_gap_count": above_maximum_gap_count,
                "continuous_run_count": len(slices),
                "continuous_source_run_slices": [list(item) for item in slices],
                "candidate_window_count": len(indices),
                "window_validity_reason_counts": dict(sorted(reason_counts.items())),
            }
        )
    return {
        "record_kind": "physical_information_signal_preprocessing_audit",
        "schema_version": "1.0.0",
        "annotation_inputs_accepted": False,
        "reset_boundaries": [
            "attachment_change_or_input_recording_boundary",
            "recording_break_before",
            "timestamp_nonincrease",
            "timestamp_gap_above_frozen_maximum",
            "nonfinite_sensor_row",
        ],
        "non_boundaries": [
            "activity_annotation",
            "support_query_role",
            "bout_boundary",
            "scoring_admission",
        ],
        "resampling_method": contract.processing.resampling_method,
        "target_rate_hz": contract.processing.target_rate_hz,
        "window_samples": contract.processing.window_samples,
        "window_stride_samples": contract.processing.window_stride_samples,
        "recordings": rows,
        "total_source_samples": sum(row["source_sample_count"] for row in rows),
        "total_candidate_windows": len(windows.window_ids),
        "total_valid_windows": int(np.count_nonzero(windows.valid)),
        "window_table_sha256": window_table_hash(windows),
        "model_fit_count": 0,
    }


def array_sha256(array: npt.NDArray[Any]) -> str:
    """Hash dtype, shape, and C-order bytes so arrays cannot be reinterpreted silently."""

    contiguous = np.ascontiguousarray(array)
    descriptor = canonical_json_bytes(
        {"dtype": contiguous.dtype.str, "shape": list(contiguous.shape)}
    )
    return hashlib.sha256(descriptor + b"\0" + contiguous.tobytes(order="C")).hexdigest()


def window_table_hash(table: WindowTable) -> str:
    payload = {
        "window_ids": list(table.window_ids),
        "recording_ids": list(table.recording_ids),
        "block_ids": list(table.block_ids),
        "wearer_ids": list(table.wearer_ids),
        "run_indices": array_sha256(table.run_indices),
        "source_run_indices": array_sha256(table.source_run_indices),
        "source_start_sample_indices": array_sha256(table.source_start_sample_indices),
        "source_stop_sample_indices_exclusive": array_sha256(
            table.source_stop_sample_indices_exclusive
        ),
        "starts_seconds": array_sha256(table.starts_seconds),
        "ends_seconds": array_sha256(table.ends_seconds),
        "gravity_directions": array_sha256(table.gravity_directions),
        "dynamic_acceleration_rms": array_sha256(table.dynamic_acceleration_rms),
        "gyroscope_rms": array_sha256(table.gyroscope_rms),
        "valid": array_sha256(table.valid),
        "invalid_reasons": list(table.invalid_reasons),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def project_bouts(
    *,
    plan_bouts: Sequence[Mapping[str, Any]],
    annotations: Sequence[Mapping[str, Any]],
    recordings: Sequence[CanonicalRecording],
    windows: WindowTable,
    minimum_valid_window_count: int = 3,
    minimum_valid_window_fraction: float = 0.80,
    source_rate_hz: float | None = None,
) -> list[dict[str, Any]]:
    """Project adjudicated intervals onto the already-created signal window grid."""

    if minimum_valid_window_count <= 0 or not 0.0 <= minimum_valid_window_fraction <= 1.0:
        raise SignalContractError("invalid bout eligibility thresholds")
    annotation_by_bout: dict[str, Mapping[str, Any]] = {}
    for annotation in annotations:
        bout_id = str(annotation.get("bout_id", ""))
        if not bout_id or bout_id in annotation_by_bout:
            raise SignalContractError(f"missing or duplicate annotation bout_id: {bout_id!r}")
        annotation_by_bout[bout_id] = annotation
    recording_by_id = {item.recording_id: item for item in recordings}
    if len(recording_by_id) != len(recordings):
        raise SignalContractError("duplicate recording_id")
    window_indices: dict[str, list[int]] = defaultdict(list)
    for index, recording_id in enumerate(windows.recording_ids):
        window_indices[recording_id].append(index)
    results: list[dict[str, Any]] = []
    seen_bouts: set[str] = set()
    for planned in plan_bouts:
        bout_id = str(planned.get("bout_id", ""))
        if not bout_id or bout_id in seen_bouts:
            raise SignalContractError(f"missing or duplicate planned bout_id: {bout_id!r}")
        seen_bouts.add(bout_id)
        base: dict[str, Any] = {
            "bout_id": bout_id,
            "block_id": planned.get("block_id"),
            "wearer_id": planned.get("wearer_id"),
            "role": planned.get("role"),
            "activity": planned.get("activity"),
            "motion": planned.get("motion"),
            "eligible": False,
            "eligibility_reason": None,
            "candidate_window_count": 0,
            "valid_window_count": 0,
            "valid_window_fraction": 0.0,
            "candidate_window_ids": [],
            "valid_window_ids": [],
            "candidate_window_source_support": [],
            "candidate_invalid_reason_counts": {},
            "trimmed_start_seconds": None,
            "trimmed_stop_seconds_exclusive": None,
        }
        bout_annotation = annotation_by_bout.get(bout_id)
        reason: str | None = None
        recording: CanonicalRecording | None = None
        if bout_annotation is None:
            reason = "annotation_missing"
        elif bout_annotation.get("adjudication_status") != "adjudicated":
            reason = "annotation_not_adjudicated"
        elif bout_annotation.get("observed_activity") != planned.get(
            "activity"
        ) or bout_annotation.get("observed_motion") != planned.get("motion"):
            reason = "adjudicated_label_does_not_match_plan"
        else:
            recording = recording_by_id.get(str(bout_annotation.get("recording_id", "")))
            if recording is None:
                reason = "recording_missing"
            elif recording.block_id != planned.get("block_id"):
                reason = "annotation_recording_block_mismatch"
        start_value = bout_annotation.get("start_sample") if bout_annotation is not None else None
        stop_value = (
            bout_annotation.get("stop_sample_exclusive") if bout_annotation is not None else None
        )
        if reason is None and recording is not None:
            if (
                isinstance(start_value, bool)
                or not isinstance(start_value, int)
                or isinstance(stop_value, bool)
                or not isinstance(stop_value, int)
                or start_value < 0
                or stop_value <= start_value
                or stop_value > recording.timestamps.size
            ):
                reason = "annotation_sample_interval_invalid"
        if reason is None and recording is not None:
            assert isinstance(start_value, int)
            assert isinstance(stop_value, int)
            start_time = float(recording.timestamps[start_value])
            if stop_value < recording.timestamps.size:
                next_time = float(recording.timestamps[stop_value])
                previous_time = float(recording.timestamps[stop_value - 1])
                next_starts_new_epoch = bool(recording.recording_break_before[stop_value]) or (
                    not math.isfinite(next_time)
                    or not math.isfinite(previous_time)
                    or next_time <= previous_time
                )
                if next_starts_new_epoch and source_rate_hz is not None:
                    stop_time = previous_time + 1.0 / source_rate_hz
                else:
                    stop_time = next_time
            elif source_rate_hz is not None:
                if not math.isfinite(source_rate_hz) or source_rate_hz <= 0.0:
                    raise SignalContractError("source_rate_hz must be finite and positive")
                stop_time = float(recording.timestamps[-1]) + 1.0 / source_rate_hz
            else:
                finite_delta = np.diff(recording.timestamps)
                finite_delta = finite_delta[np.isfinite(finite_delta) & (finite_delta > 0.0)]
                stop_time = (
                    float(recording.timestamps[-1] + np.median(finite_delta))
                    if finite_delta.size
                    else float("nan")
                )
            trim = float(planned.get("interior_trim_seconds_each_end", 5))
            trimmed_start = start_time + trim
            trimmed_stop = stop_time - trim
            if (
                not math.isfinite(trimmed_start)
                or not math.isfinite(trimmed_stop)
                or trimmed_stop <= trimmed_start
            ):
                reason = "trimmed_interval_invalid"
            else:
                indices = [
                    index
                    for index in window_indices[recording.recording_id]
                    if windows.starts_seconds[index] >= trimmed_start - 1.0e-10
                    and windows.ends_seconds[index] <= trimmed_stop + 1.0e-10
                    and windows.source_start_sample_indices[index] >= start_value
                    and windows.source_stop_sample_indices_exclusive[index] <= stop_value
                ]
                valid_indices = [index for index in indices if bool(windows.valid[index])]
                fraction = len(valid_indices) / len(indices) if indices else 0.0
                invalid_reason_counts: dict[str, int] = defaultdict(int)
                for index in indices:
                    if not bool(windows.valid[index]):
                        invalid_reason_counts[windows.invalid_reasons[index]] += 1
                base.update(
                    {
                        "candidate_window_count": len(indices),
                        "valid_window_count": len(valid_indices),
                        "valid_window_fraction": fraction,
                        "candidate_window_ids": [windows.window_ids[index] for index in indices],
                        "valid_window_ids": [windows.window_ids[index] for index in valid_indices],
                        "candidate_window_source_support": [
                            {
                                "window_id": windows.window_ids[index],
                                "timestamp_epoch_index": int(windows.run_indices[index]),
                                "source_signal_run_index": int(windows.source_run_indices[index]),
                                "source_start_sample": int(
                                    windows.source_start_sample_indices[index]
                                ),
                                "source_stop_sample_exclusive": int(
                                    windows.source_stop_sample_indices_exclusive[index]
                                ),
                                "valid": bool(windows.valid[index]),
                                "invalid_reason": (
                                    None
                                    if bool(windows.valid[index])
                                    else windows.invalid_reasons[index]
                                ),
                            }
                            for index in indices
                        ],
                        "candidate_invalid_reason_counts": dict(
                            sorted(invalid_reason_counts.items())
                        ),
                        "trimmed_start_seconds": trimmed_start,
                        "trimmed_stop_seconds_exclusive": trimmed_stop,
                    }
                )
                if not indices:
                    reason = "no_grid_windows_fully_inside_trimmed_interval"
                elif len(valid_indices) < minimum_valid_window_count:
                    reason = "valid_window_count_below_minimum"
                elif fraction < minimum_valid_window_fraction:
                    reason = "valid_window_fraction_below_minimum"
                else:
                    reason = "eligible"
                    base["eligible"] = True
        base["eligibility_reason"] = reason
        results.append(base)
    return results


def _unit_mean(vectors: FloatArray, epsilon: float) -> FloatArray | None:
    if vectors.ndim != 2 or vectors.shape[1] != 3 or vectors.shape[0] == 0:
        return None
    result = vectors.mean(axis=0)
    norm = float(np.linalg.norm(result))
    return None if not math.isfinite(norm) or norm <= epsilon else result / norm


def angular_distance(left: npt.ArrayLike, right: npt.ArrayLike) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != (3,) or right_array.shape != (3,):
        raise SignalContractError("angular distance inputs must be length-three vectors")
    return float(math.acos(float(np.clip(np.dot(left_array, right_array), -1.0, 1.0))))


def student_t_summary(values: Sequence[float]) -> dict[str, Any]:
    """Report wearer values and a fixed exact df<=5 Student-t interval."""

    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    count = len(finite_values)
    mean = float(np.mean(finite_values)) if finite_values else None
    standard_deviation = float(np.std(finite_values, ddof=1)) if count >= 2 else None
    interval: list[float] | None = None
    suppression: str | None = None
    if count < 2:
        suppression = "suppressed_below_two_complete_wearers"
    else:
        degrees = count - 1
        if degrees not in T_CRITICAL_975:
            raise SignalContractError("fixed pilot interval supports at most six wearers")
        assert mean is not None and standard_deviation is not None
        half_width = T_CRITICAL_975[degrees] * standard_deviation / math.sqrt(count)
        interval = [mean - half_width, mean + half_width]
    return {
        "wearer_values": finite_values,
        "count": count,
        "mean": mean,
        "sample_standard_deviation": standard_deviation,
        "student_t_95_interval": interval,
        "interval_degrees_of_freedom": count - 1 if count >= 2 else None,
        "interval_suppression_reason": suppression,
    }


def _window_lookup(windows: WindowTable) -> dict[str, int]:
    result = {identifier: index for index, identifier in enumerate(windows.window_ids)}
    if len(result) != len(windows.window_ids):
        raise SignalContractError("window identifiers are not unique")
    return result


def _build_prototypes(
    support_ids: Sequence[str],
    projection_by_bout: Mapping[str, Mapping[str, Any]],
    planned_by_bout: Mapping[str, Mapping[str, Any]],
    windows: WindowTable,
    lookup: Mapping[str, int],
    *,
    epsilon: float,
    frame_limit: float | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"valid": True, "invalid_reasons": [], "motion_strata": {}}
    for motion in MOTIONS:
        motion_record: dict[str, Any] = {"postures": {}}
        prototypes: dict[str, FloatArray] = {}
        for posture in POSTURES:
            matched = [
                identifier
                for identifier in support_ids
                if planned_by_bout.get(identifier, {}).get("activity") == posture
                and planned_by_bout.get(identifier, {}).get("motion") == motion
            ]
            reason: str | None = None
            vector: FloatArray | None = None
            spread_median: float | None = None
            spread_p90: float | None = None
            window_ids: list[str] = []
            if len(matched) != 1:
                reason = "support_condition_not_unique"
            else:
                projection = projection_by_bout.get(matched[0])
                if projection is None or projection.get("eligible") is not True:
                    reason = "support_bout_ineligible"
                else:
                    window_ids = cast(list[str], projection["valid_window_ids"])
                    vectors = windows.gravity_directions[[lookup[item] for item in window_ids]]
                    vector = _unit_mean(vectors, epsilon)
                    if vector is None:
                        reason = "prototype_resultant_norm_at_or_below_epsilon"
                    else:
                        angles = [angular_distance(item, vector) for item in vectors]
                        spread_median = float(np.median(angles))
                        spread_p90 = float(np.percentile(angles, 90.0))
                        prototypes[posture] = vector
            motion_record["postures"][posture] = {
                "valid": reason is None,
                "invalid_reason": reason,
                "support_bout_id": matched[0] if len(matched) == 1 else None,
                "valid_window_ids": window_ids,
                "prototype": None if vector is None else vector.tolist(),
                "median_window_angle_radians": spread_median,
                "p90_window_angle_radians": spread_p90,
            }
            if reason is not None:
                result["valid"] = False
                result["invalid_reasons"].append(f"{motion}:{posture}:{reason}")
        separation: float | None = None
        inverse_condition: float | None = None
        numerical_nondegenerate = False
        frame_eligible = False
        frame_reason = "prototype_invalid"
        if all(posture in prototypes for posture in POSTURES):
            separation = angular_distance(prototypes["sitting"], prototypes["standing"])
            denominator = abs(math.sin(separation))
            numerical_nondegenerate = denominator > epsilon
            inverse_condition = 1.0 / denominator if numerical_nondegenerate else None
            if not numerical_nondegenerate:
                frame_reason = "inverse_sine_condition_degenerate"
            elif frame_limit is None:
                frame_reason = "bench_frame_conditioning_limit_missing"
            elif inverse_condition is not None and inverse_condition <= frame_limit:
                frame_eligible = True
                frame_reason = "eligible"
            else:
                frame_reason = "inverse_sine_condition_exceeds_bench_limit"
        motion_record.update(
            {
                "between_posture_separation_radians": separation,
                "inverse_absolute_sine_condition_number": inverse_condition,
                "inverse_condition_display": (
                    "infinity_denominator_at_or_below_epsilon"
                    if separation is not None and not numerical_nondegenerate
                    else None
                ),
                "numerically_nondegenerate": numerical_nondegenerate,
                "frame_eligible": frame_eligible,
                "frame_eligibility_reason": frame_reason,
            }
        )
        result["motion_strata"][motion] = motion_record
    return result


def _signed_margin_rows(
    *,
    query_ids: Sequence[str],
    prototype_record: Mapping[str, Any],
    projection_by_bout: Mapping[str, Mapping[str, Any]],
    planned_by_bout: Mapping[str, Mapping[str, Any]],
    windows: WindowTable,
    lookup: Mapping[str, int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    strata = _mapping(prototype_record.get("motion_strata"), "prototype motion strata")
    for bout_id in query_ids:
        planned = planned_by_bout[bout_id]
        activity = str(planned.get("activity"))
        motion = str(planned.get("motion"))
        if activity not in POSTURES or motion not in MOTIONS:
            continue
        projection = projection_by_bout[bout_id]
        window_ids = cast(list[str], projection.get("valid_window_ids", []))
        reason: str | None = None
        median: float | None = None
        margins: list[float] = []
        if projection.get("eligible") is not True:
            reason = "query_bout_ineligible"
        else:
            stratum = _mapping(strata.get(motion), f"prototype stratum {motion}")
            postures = _mapping(stratum.get("postures"), "prototype postures")
            correct = _mapping(postures.get(activity), "correct prototype")
            wrong_activity = "standing" if activity == "sitting" else "sitting"
            wrong = _mapping(postures.get(wrong_activity), "wrong prototype")
            if correct.get("valid") is not True or wrong.get("valid") is not True:
                reason = "prototype_invalid"
            else:
                correct_vector = np.asarray(correct["prototype"], dtype=np.float64)
                wrong_vector = np.asarray(wrong["prototype"], dtype=np.float64)
                for identifier in window_ids:
                    vector = windows.gravity_directions[lookup[identifier]]
                    margins.append(
                        angular_distance(vector, wrong_vector)
                        - angular_distance(vector, correct_vector)
                    )
                median = float(np.median(margins)) if margins else None
                reason = "eligible" if margins else "no_identical_query_windows"
        rows.append(
            {
                "bout_id": bout_id,
                "wearer_id": planned.get("wearer_id"),
                "activity": activity,
                "motion": motion,
                "eligible": reason == "eligible",
                "eligibility_reason": reason,
                "query_window_ids": window_ids,
                "query_window_ids_sha256": hashlib.sha256(
                    canonical_json_bytes(window_ids)
                ).hexdigest(),
                "window_signed_margins_radians": margins,
                "bout_median_signed_margin_radians": median,
            }
        )
    return rows


def _condition_wearer_aggregate(
    fresh_rows: Sequence[Mapping[str, Any]],
    signal_complete_wearers: Sequence[str],
    roster: Sequence[str],
) -> dict[str, Any]:
    condition_rows: list[dict[str, Any]] = []
    for row in fresh_rows:
        value = row.get("bout_median_signed_margin_radians")
        condition_rows.append(
            {
                "wearer_id": str(row["wearer_id"]),
                "activity": str(row["activity"]),
                "motion": str(row["motion"]),
                "reference_scope": str(row["comparison_id"]),
                "bout_id": str(row["bout_id"]),
                "estimand_eligible": row.get("eligible") is True,
                "exclusion_reason": (
                    None if row.get("eligible") is True else row.get("eligibility_reason")
                ),
                "bout_value": (
                    float(value)
                    if row.get("eligible") is True and isinstance(value, (float, int))
                    else None
                ),
            }
        )
    condition_rows.sort(
        key=lambda row: (
            str(row["wearer_id"]),
            str(row["motion"]),
            str(row["reference_scope"]),
            str(row["activity"]),
        )
    )
    wearer_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for motion in MOTIONS:
        for wearer in roster:
            rows = [
                row
                for row in condition_rows
                if row["wearer_id"] == wearer and row["motion"] == motion
            ]
            eligible_rows = [row for row in rows if row["estimand_eligible"] is True]
            values = [float(row["bout_value"]) for row in eligible_rows]
            exclusions = [
                {
                    "bout_id": row["bout_id"],
                    "reference_scope": row["reference_scope"],
                    "reason": row["exclusion_reason"],
                }
                for row in rows
                if row["estimand_eligible"] is not True
            ]
            if wearer not in signal_complete_wearers:
                exclusions.append({"reason": "wearer_signal_completeness_failed"})
            if len(rows) != 8:
                exclusions.append(
                    {
                        "reason": "fresh_condition_reference_count_mismatch",
                        "expected": 8,
                        "observed": len(rows),
                    }
                )
            estimand_complete = (
                wearer in signal_complete_wearers and len(rows) == 8 and len(eligible_rows) == 8
            )
            wearer_rows.append(
                {
                    "wearer_id": wearer,
                    "motion": motion,
                    "expected_condition_reference_count": 8,
                    "observed_condition_reference_count": len(rows),
                    "eligible_condition_reference_count": len(eligible_rows),
                    "available_condition_values": values,
                    "wearer_mean": float(np.mean(values)) if estimand_complete else None,
                    "estimand_complete": estimand_complete,
                    "included_in_independent_summary": estimand_complete,
                    "exclusions": exclusions,
                }
            )
        included = [
            row
            for row in wearer_rows
            if row["motion"] == motion and row["included_in_independent_summary"] is True
        ]
        summary = student_t_summary([float(row["wearer_mean"]) for row in included])
        summary.update(
            {
                "expected_wearer_ids": list(roster),
                "included_wearer_ids": [row["wearer_id"] for row in included],
                "excluded_wearers": [
                    {
                        "wearer_id": row["wearer_id"],
                        "exclusions": row["exclusions"],
                    }
                    for row in wearer_rows
                    if row["motion"] == motion
                    and row["included_in_independent_summary"] is not True
                ],
            }
        )
        summaries[motion] = summary
    return {
        "bout_to_condition_rows": condition_rows,
        "condition_to_wearer_rows": wearer_rows,
        "equal_wearer_summaries": summaries,
    }


def _common_language_probability(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left or not right:
        return None
    wins = sum(a > b for a in left for b in right)
    ties = sum(a == b for a in left for b in right)
    return (wins + 0.5 * ties) / (len(left) * len(right))


def analyze_physical_information(
    *,
    plan: Mapping[str, Any],
    projections: Sequence[Mapping[str, Any]],
    windows: WindowTable,
    contract: DeviceSignalContract,
    norm_epsilon: float = 1.0e-12,
) -> dict[str, Any]:
    """Compute frozen fresh/stale posture and energy estimands with zero fits."""

    planned_bouts = cast(list[Mapping[str, Any]], plan.get("bouts", []))
    comparisons = cast(list[Mapping[str, Any]], plan.get("comparisons", []))
    roster = [str(item) for item in cast(list[object], plan.get("wearers", []))]
    planned_by_bout = {str(item["bout_id"]): item for item in planned_bouts}
    projection_by_bout = {str(item["bout_id"]): item for item in projections}
    if set(planned_by_bout) != set(projection_by_bout):
        raise SignalContractError("bout projection does not cover the exact plan")
    lookup = _window_lookup(windows)
    fresh_rows: list[dict[str, Any]] = []
    stale_rows: list[dict[str, Any]] = []
    reference_records: list[dict[str, Any]] = []
    prototype_cache: dict[tuple[str, ...], dict[str, Any]] = {}

    def prototypes(support: Sequence[str]) -> dict[str, Any]:
        key = tuple(str(item) for item in support)
        if key not in prototype_cache:
            prototype_cache[key] = _build_prototypes(
                key,
                projection_by_bout,
                planned_by_bout,
                windows,
                lookup,
                epsilon=norm_epsilon,
                frame_limit=contract.frame_conditioning_limit,
            )
        return prototype_cache[key]

    for comparison in comparisons:
        comparison_type = str(comparison.get("comparison_type"))
        query_ids = [str(item) for item in cast(list[object], comparison.get("query_bout_ids", []))]
        fresh_support = [
            str(item) for item in cast(list[object], comparison.get("fresh_support_bout_ids", []))
        ]
        fresh_prototypes = prototypes(fresh_support)
        if comparison_type == "fresh":
            rows = _signed_margin_rows(
                query_ids=query_ids,
                prototype_record=fresh_prototypes,
                projection_by_bout=projection_by_bout,
                planned_by_bout=planned_by_bout,
                windows=windows,
                lookup=lookup,
            )
            for row in rows:
                row["comparison_id"] = comparison.get("comparison_id")
                row["reference_scope"] = "fresh"
            fresh_rows.extend(rows)
            reference_records.append(
                {
                    "comparison_id": comparison.get("comparison_id"),
                    "comparison_type": comparison_type,
                    "fresh_support_bout_ids": fresh_support,
                    "fresh_prototypes": fresh_prototypes,
                }
            )
            continue
        stale_support_value = comparison.get("stale_support_bout_ids")
        if not isinstance(stale_support_value, list):
            raise SignalContractError("stale comparison is missing stale support")
        stale_support = [str(item) for item in stale_support_value]
        stale_prototypes = prototypes(stale_support)
        fresh_comparison_rows = _signed_margin_rows(
            query_ids=query_ids,
            prototype_record=fresh_prototypes,
            projection_by_bout=projection_by_bout,
            planned_by_bout=planned_by_bout,
            windows=windows,
            lookup=lookup,
        )
        stale_comparison_rows = _signed_margin_rows(
            query_ids=query_ids,
            prototype_record=stale_prototypes,
            projection_by_bout=projection_by_bout,
            planned_by_bout=planned_by_bout,
            windows=windows,
            lookup=lookup,
        )
        for fresh_row, stale_row in zip(fresh_comparison_rows, stale_comparison_rows, strict=True):
            fresh_ids = cast(list[str], fresh_row["query_window_ids"])
            stale_ids = cast(list[str], stale_row["query_window_ids"])
            intersection = [identifier for identifier in fresh_ids if identifier in set(stale_ids)]
            if fresh_ids != stale_ids or intersection != fresh_ids:
                raise SignalContractError("fresh/stale identical-query window invariant failed")
            difference: float | None = None
            if fresh_row["eligible"] is True and stale_row["eligible"] is True:
                difference = float(fresh_row["bout_median_signed_margin_radians"]) - float(
                    stale_row["bout_median_signed_margin_radians"]
                )
            exclusion_reasons: list[str] = []
            if fresh_row["eligible"] is not True:
                exclusion_reasons.append(f"fresh:{fresh_row.get('eligibility_reason', 'unknown')}")
            if stale_row["eligible"] is not True:
                exclusion_reasons.append(f"stale:{stale_row.get('eligibility_reason', 'unknown')}")
            stale_rows.append(
                {
                    "comparison_id": comparison.get("comparison_id"),
                    "comparison_type": comparison_type,
                    "bout_id": fresh_row["bout_id"],
                    "wearer_id": fresh_row["wearer_id"],
                    "activity": fresh_row["activity"],
                    "motion": fresh_row["motion"],
                    "identical_query_window_ids": intersection,
                    "identical_query_window_count": len(intersection),
                    "identical_query_window_ids_sha256": hashlib.sha256(
                        canonical_json_bytes(intersection)
                    ).hexdigest(),
                    "fresh_median_signed_margin_radians": fresh_row[
                        "bout_median_signed_margin_radians"
                    ],
                    "stale_median_signed_margin_radians": stale_row[
                        "bout_median_signed_margin_radians"
                    ],
                    "fresh_minus_stale_median_margin_radians": difference,
                    "eligible": difference is not None,
                    "eligibility_reason": (
                        "eligible" if difference is not None else ";".join(exclusion_reasons)
                    ),
                }
            )
        reference_records.append(
            {
                "comparison_id": comparison.get("comparison_id"),
                "comparison_type": comparison_type,
                "fresh_support_bout_ids": fresh_support,
                "stale_support_bout_ids": stale_support,
                "fresh_prototypes": fresh_prototypes,
                "stale_prototypes": stale_prototypes,
            }
        )

    # Completeness is metadata/signal eligibility, independent of whether prototypes work.
    fresh_by_wearer: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in fresh_rows:
        fresh_by_wearer[str(row["wearer_id"])].append(row)
    completeness: list[dict[str, Any]] = []
    wearer_motion_means: dict[tuple[str, str], float] = {}
    for wearer in roster:
        rows = fresh_by_wearer[wearer]
        expected_query_ids = [
            str(item["bout_id"])
            for item in planned_bouts
            if item.get("wearer_id") == wearer
            and item.get("role") == "query"
            and item.get("activity") in POSTURES
        ]
        signal_eligible_query_ids = [
            identifier
            for identifier in expected_query_ids
            if projection_by_bout[identifier].get("eligible") is True
        ]
        complete = len(expected_query_ids) == 16 and len(signal_eligible_query_ids) == 16
        eligible = [row for row in rows if row["eligible"] is True]
        missing = sorted(
            identifier
            for identifier in expected_query_ids
            if projection_by_bout[identifier].get("eligible") is not True
        )
        if len(expected_query_ids) != 16:
            missing.append(
                f"expected_16_fresh_posture_bouts_in_plan_observed_{len(expected_query_ids)}"
            )
        motion_means: dict[str, float | None] = {}
        for motion in MOTIONS:
            values = [
                float(row["bout_median_signed_margin_radians"])
                for row in eligible
                if row["motion"] == motion
            ]
            motion_means[motion] = float(np.mean(values)) if len(values) == 8 else None
            if motion_means[motion] is not None:
                wearer_motion_means[(wearer, motion)] = cast(float, motion_means[motion])
        completeness.append(
            {
                "wearer_id": wearer,
                "complete": complete,
                "expected_fresh_posture_bout_count": 16,
                "planned_fresh_posture_bout_count": len(expected_query_ids),
                "signal_eligible_fresh_posture_bout_count": len(signal_eligible_query_ids),
                "margin_eligible_fresh_posture_bout_count": len(eligible),
                "incomplete_reasons": missing,
                "fresh_mean_margin_by_motion_radians": motion_means,
            }
        )
    complete_wearers = [str(row["wearer_id"]) for row in completeness if row["complete"]]
    gate_failures: list[str] = []
    if complete_wearers != roster or len(roster) != 6:
        gate_failures.append("not_all_six_frozen_roster_wearers_complete")
    for wearer in roster:
        for motion in MOTIONS:
            value = wearer_motion_means.get((wearer, motion))
            if value is None:
                gate_failures.append(f"{wearer}:{motion}:complete_wearer_mean_missing")
            elif value <= 0.0:
                gate_failures.append(f"{wearer}:{motion}:mean_not_strictly_positive")
    gate_pass = len(gate_failures) == 0

    # Energy summaries operate on bout medians and retain each wearer's raw values.
    energy_records: list[dict[str, Any]] = []
    for wearer in roster:
        wearer_bouts = [
            item
            for item in projections
            if item.get("wearer_id") == wearer and item.get("eligible") is True
        ]
        for feature_name, feature in (
            ("dynamic_acceleration_rms", windows.dynamic_acceleration_rms),
            ("gyroscope_rms", windows.gyroscope_rms),
        ):
            mobility: list[float] = []
            upper: list[float] = []
            for bout in wearer_bouts:
                identifiers = cast(list[str], bout["valid_window_ids"])
                if not identifiers:
                    continue
                median = float(np.median([feature[lookup[item]] for item in identifiers]))
                if bout.get("activity") == "mobility":
                    mobility.append(median)
                elif bout.get("motion") == "upper_body_motion" and bout.get("activity") in POSTURES:
                    upper.append(median)
            energy_records.append(
                {
                    "wearer_id": wearer,
                    "feature": feature_name,
                    "mobility_bout_medians": mobility,
                    "upper_body_motion_posture_bout_medians": upper,
                    "common_language_probability": _common_language_probability(mobility, upper),
                }
            )

    aggregation = _condition_wearer_aggregate(fresh_rows, complete_wearers, roster)
    stale_summaries: dict[str, Any] = {}
    for comparison_type in ("attachment_stale", "visit_stale"):
        condition_reference_rows = [
            {
                "wearer_id": row["wearer_id"],
                "activity": row["activity"],
                "motion": row["motion"],
                "reference_scope": row["comparison_id"],
                "bout_value": row["fresh_minus_stale_median_margin_radians"],
                "estimand_eligible": row["eligible"],
                "exclusion_reason": (
                    None if row["eligible"] is True else row["eligibility_reason"]
                ),
            }
            for row in stale_rows
            if row["comparison_type"] == comparison_type
        ]
        wearer_rows: list[dict[str, Any]] = []
        for wearer in roster:
            rows = [row for row in condition_reference_rows if row["wearer_id"] == wearer]
            eligible_rows = [row for row in rows if row["estimand_eligible"] is True]
            values = [float(row["bout_value"]) for row in eligible_rows]
            exclusions = [
                {
                    "reference_scope": row["reference_scope"],
                    "activity": row["activity"],
                    "motion": row["motion"],
                    "reason": row["exclusion_reason"],
                }
                for row in rows
                if row["estimand_eligible"] is not True
            ]
            if len(rows) != 8:
                exclusions.append(
                    {
                        "reason": "stale_condition_reference_count_mismatch",
                        "expected": 8,
                        "observed": len(rows),
                    }
                )
            if wearer not in complete_wearers:
                exclusions.append({"reason": "wearer_signal_completeness_failed"})
            estimand_complete = (
                wearer in complete_wearers and len(rows) == 8 and len(eligible_rows) == 8
            )
            wearer_rows.append(
                {
                    "wearer_id": wearer,
                    "expected_condition_reference_count": 8,
                    "observed_condition_reference_count": len(rows),
                    "eligible_condition_reference_count": len(eligible_rows),
                    "available_condition_reference_values": values,
                    "wearer_mean": float(np.mean(values)) if estimand_complete else None,
                    "estimand_complete": estimand_complete,
                    "included_in_independent_summary": estimand_complete,
                    "exclusions": exclusions,
                }
            )
        included = [row for row in wearer_rows if row["included_in_independent_summary"] is True]
        equal_summary = student_t_summary([float(row["wearer_mean"]) for row in included])
        equal_summary.update(
            {
                "expected_wearer_ids": roster,
                "included_wearer_ids": [row["wearer_id"] for row in included],
                "excluded_wearers": [
                    {
                        "wearer_id": row["wearer_id"],
                        "exclusions": row["exclusions"],
                    }
                    for row in wearer_rows
                    if row["included_in_independent_summary"] is not True
                ],
            }
        )
        stale_summaries[comparison_type] = {
            "condition_reference_rows": condition_reference_rows,
            "wearer_rows": wearer_rows,
            "equal_wearer_summary": equal_summary,
            "threshold_applied": False,
        }
    return {
        "record_kind": "physical_information_signal_analysis",
        "schema_version": "1.0.0",
        "model_fit_count": 0,
        "evidence_status": contract.evidence_status,
        "window_table_sha256": window_table_hash(windows),
        "reference_records": reference_records,
        "fresh_signed_margin_rows": fresh_rows,
        "fresh_aggregation": aggregation,
        "stale_comparison_rows": stale_rows,
        "stale_descriptive_summaries": stale_summaries,
        "wearer_completeness": completeness,
        "energy_overlap_records": energy_records,
        "physical_identifiability_gate": {
            "status": "pass" if gate_pass else "fail",
            "all_six_complete": complete_wearers == roster and len(roster) == 6,
            "complete_wearer_ids": complete_wearers,
            "failures": gate_failures,
            "requires_every_complete_wearer_strictly_positive_in": list(MOTIONS),
            "decision_scope": "permits_new_protocol_draft_only",
        },
        "claims": {
            "physical_identifiability_demonstrated": False,
            "population_superiority_demonstrated": False,
            "architecture_eligible": False,
        },
    }


def window_table_arrays(table: WindowTable) -> dict[str, npt.NDArray[Any]]:
    """Return portable non-object arrays for a deterministic NPZ cache."""

    return {
        "window_ids": np.asarray(table.window_ids, dtype=np.str_),
        "recording_ids": np.asarray(table.recording_ids, dtype=np.str_),
        "block_ids": np.asarray(table.block_ids, dtype=np.str_),
        "wearer_ids": np.asarray(table.wearer_ids, dtype=np.str_),
        "run_indices": table.run_indices,
        "source_run_indices": table.source_run_indices,
        "source_start_sample_indices": table.source_start_sample_indices,
        "source_stop_sample_indices_exclusive": table.source_stop_sample_indices_exclusive,
        "starts_seconds": table.starts_seconds,
        "ends_seconds": table.ends_seconds,
        "gravity_directions": table.gravity_directions,
        "dynamic_acceleration_rms": table.dynamic_acceleration_rms,
        "gyroscope_rms": table.gyroscope_rms,
        "valid": table.valid,
        "invalid_reasons": np.asarray(table.invalid_reasons, dtype=np.str_),
    }


def window_table_from_arrays(arrays: Mapping[str, npt.ArrayLike]) -> WindowTable:
    """Restore a cache while rejecting unexpected or object-typed content."""

    required = {
        "window_ids",
        "recording_ids",
        "block_ids",
        "wearer_ids",
        "run_indices",
        "source_run_indices",
        "source_start_sample_indices",
        "source_stop_sample_indices_exclusive",
        "starts_seconds",
        "ends_seconds",
        "gravity_directions",
        "dynamic_acceleration_rms",
        "gyroscope_rms",
        "valid",
        "invalid_reasons",
    }
    _exact_fields(arrays, required, set(), "window cache arrays")
    converted = {key: np.asarray(value) for key, value in arrays.items()}
    if any(value.dtype.kind == "O" for value in converted.values()):
        raise SignalContractError("object arrays are forbidden")
    count = int(converted["window_ids"].size)
    for key, value in converted.items():
        if key == "gravity_directions":
            if value.shape != (count, 3):
                raise SignalContractError("gravity_directions cache shape mismatch")
        elif value.shape != (count,):
            raise SignalContractError(f"{key} cache shape mismatch")
    return WindowTable(
        window_ids=tuple(str(item) for item in converted["window_ids"]),
        recording_ids=tuple(str(item) for item in converted["recording_ids"]),
        block_ids=tuple(str(item) for item in converted["block_ids"]),
        wearer_ids=tuple(str(item) for item in converted["wearer_ids"]),
        run_indices=converted["run_indices"].astype(np.int64, copy=False),
        source_run_indices=converted["source_run_indices"].astype(np.int64, copy=False),
        source_start_sample_indices=converted["source_start_sample_indices"].astype(
            np.int64, copy=False
        ),
        source_stop_sample_indices_exclusive=converted[
            "source_stop_sample_indices_exclusive"
        ].astype(np.int64, copy=False),
        starts_seconds=converted["starts_seconds"].astype(np.float64, copy=False),
        ends_seconds=converted["ends_seconds"].astype(np.float64, copy=False),
        gravity_directions=converted["gravity_directions"].astype(np.float64, copy=False),
        dynamic_acceleration_rms=converted["dynamic_acceleration_rms"].astype(
            np.float64, copy=False
        ),
        gyroscope_rms=converted["gyroscope_rms"].astype(np.float64, copy=False),
        valid=converted["valid"].astype(np.bool_, copy=False),
        invalid_reasons=tuple(str(item) for item in converted["invalid_reasons"]),
    )
