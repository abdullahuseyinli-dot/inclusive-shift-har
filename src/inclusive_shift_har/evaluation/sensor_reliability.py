"""Deterministic, window-local inertial sensor reliability stressors.

This module is infrastructure for the secondary, post-confirmatory stress track.
It does not select stress levels, models, calibration, or thresholds from target
performance.  Corruptions are applied independently within already materialized
windows and therefore cannot cross participant, trial, label, or split boundaries.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.evaluation._strict_config import (
    StrictConfigError,
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_bytes,
    canonical_json_sha256,
)

FloatWindows = NDArray[np.floating[Any]]


class SensorReliabilityError(ValueError):
    """Raised when a stress configuration or corruption request fails closed."""


@dataclass(frozen=True, slots=True)
class GaussianNoise:
    condition_id: str
    channel_std: tuple[float, ...]
    condition_sha256: str
    kind: Literal["gaussian_noise"] = "gaussian_noise"


@dataclass(frozen=True, slots=True)
class ContiguousDropout:
    condition_id: str
    duration_samples: int
    channel_indices: tuple[int, ...]
    fill_value: float
    condition_sha256: str
    kind: Literal["contiguous_dropout"] = "contiguous_dropout"


@dataclass(frozen=True, slots=True)
class MissingAxis:
    condition_id: str
    channel_index: int
    fill_value: float
    condition_sha256: str
    kind: Literal["missing_axis"] = "missing_axis"


@dataclass(frozen=True, slots=True)
class MissingModality:
    condition_id: str
    modality: Literal["accelerometer", "gyroscope"]
    fill_value: float
    condition_sha256: str
    kind: Literal["missing_modality"] = "missing_modality"


@dataclass(frozen=True, slots=True)
class RateReduction:
    condition_id: str
    factor: int
    condition_sha256: str
    kind: Literal["rate_reduction"] = "rate_reduction"


@dataclass(frozen=True, slots=True)
class LinearDrift:
    condition_id: str
    endpoint_offsets: tuple[float, ...]
    condition_sha256: str
    kind: Literal["linear_drift"] = "linear_drift"


StressCondition: TypeAlias = (
    GaussianNoise | ContiguousDropout | MissingAxis | MissingModality | RateReduction | LinearDrift
)


@dataclass(frozen=True, slots=True)
class SensorReliabilityConfig:
    """A self-hashed, immutable secondary stress-track configuration."""

    schema_version: str
    config_id: str
    config_sha256: str
    status: str
    track_role: str
    input_space: str
    window_length_samples: int
    sampling_rate_hz: float
    channels: tuple[str, ...]
    channel_units: tuple[str, ...]
    seed_namespace: str
    global_seed: int
    model_family_id: str
    eligible_training_model_names: tuple[str, ...]
    required_training_regime: str
    source_partition: str
    target_partition: str
    target_clean_reference: str
    conditions: tuple[StressCondition, ...]

    def condition(self, condition_id: str) -> StressCondition:
        matches = [item for item in self.conditions if item.condition_id == condition_id]
        if len(matches) != 1:
            raise SensorReliabilityError(f"unknown stress condition: {condition_id!r}")
        return matches[0]


@dataclass(frozen=True, slots=True)
class CorruptedWindows:
    """A newly allocated corrupted tensor and its non-performance lineage."""

    windows: FloatWindows
    metadata: dict[str, Any]


_TOP_LEVEL_KEYS = {
    "schema_version",
    "config_id",
    "config_sha256",
    "status",
    "track_role",
    "input_space",
    "window_length_samples",
    "sampling_rate_hz",
    "channels",
    "channel_units",
    "deterministic_seed",
    "preconditions",
    "policy",
    "operation",
    "conditions",
}
_PRECONDITION_KEYS = {
    "primary_confirmatory_status",
    "frozen_checkpoint_sha256_required",
    "base_prediction_sha256_required",
    "frozen_normalization_sha256_required",
}
_POLICY_KEYS = {
    "primary_claim_eligible",
    "used_for_model_selection",
    "calibration_refit_allowed",
    "threshold_refit_allowed",
    "target_based_tuning_allowed",
    "participant_level_reporting_required",
    "stress_levels_empirically_calibrated",
}
_OPERATION_KEYS = {"model_family", "cohorts"}


def _finite_number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SensorReliabilityError(f"{location} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise SensorReliabilityError(f"{location} must be finite")
    return result


def _integer(value: Any, *, location: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise SensorReliabilityError(f"{location} must be an integer >= {minimum}")
    return int(value)


def _float_tuple(value: Any, *, location: str, length: int) -> tuple[float, ...]:
    if not isinstance(value, list) or len(value) != length:
        raise SensorReliabilityError(f"{location} must contain exactly {length} values")
    return tuple(
        _finite_number(item, location=f"{location}[{index}]") for index, item in enumerate(value)
    )


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    try:
        return require_mapping(value, location=location)
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc


def _condition_from_mapping(
    value: Mapping[str, Any], *, channel_count: int, window_length: int
) -> StressCondition:
    require_exact_keys(value, {"condition_id", "type", "parameters"}, location="condition")
    condition_id = value["condition_id"]
    kind = value["type"]
    if not isinstance(condition_id, str) or not condition_id.strip():
        raise SensorReliabilityError("condition.condition_id must be a non-empty string")
    if not isinstance(kind, str):
        raise SensorReliabilityError("condition.type must be a string")
    parameters = _mapping(value["parameters"], location=f"condition {condition_id}")
    condition_sha256 = canonical_json_sha256(dict(value))

    if kind == "gaussian_noise":
        require_exact_keys(parameters, {"channel_std"}, location=condition_id)
        channel_std = _float_tuple(
            parameters["channel_std"], location=f"{condition_id}.channel_std", length=channel_count
        )
        if any(item < 0.0 for item in channel_std) or not any(item > 0.0 for item in channel_std):
            raise SensorReliabilityError(
                f"{condition_id}.channel_std must be non-negative with at least one positive value"
            )
        return GaussianNoise(condition_id, channel_std, condition_sha256)
    if kind == "contiguous_dropout":
        require_exact_keys(
            parameters,
            {"duration_samples", "channel_indices", "fill_value"},
            location=condition_id,
        )
        duration = _integer(
            parameters["duration_samples"],
            location=f"{condition_id}.duration_samples",
            minimum=1,
        )
        if duration > window_length:
            raise SensorReliabilityError(f"{condition_id}.duration_samples exceeds window length")
        raw_indices = parameters["channel_indices"]
        if not isinstance(raw_indices, list) or not raw_indices:
            raise SensorReliabilityError(f"{condition_id}.channel_indices must be non-empty")
        indices = tuple(
            _integer(item, location=f"{condition_id}.channel_indices", minimum=0)
            for item in raw_indices
        )
        if tuple(sorted(set(indices))) != indices or indices[-1] >= channel_count:
            raise SensorReliabilityError(
                f"{condition_id}.channel_indices must be sorted, unique, and in range"
            )
        return ContiguousDropout(
            condition_id,
            duration,
            indices,
            _finite_number(parameters["fill_value"], location=f"{condition_id}.fill_value"),
            condition_sha256,
        )
    if kind == "missing_axis":
        require_exact_keys(parameters, {"channel_index", "fill_value"}, location=condition_id)
        index = _integer(
            parameters["channel_index"], location=f"{condition_id}.channel_index", minimum=0
        )
        if index >= channel_count:
            raise SensorReliabilityError(f"{condition_id}.channel_index is out of range")
        return MissingAxis(
            condition_id,
            index,
            _finite_number(parameters["fill_value"], location=f"{condition_id}.fill_value"),
            condition_sha256,
        )
    if kind == "missing_modality":
        require_exact_keys(parameters, {"modality", "fill_value"}, location=condition_id)
        modality = parameters["modality"]
        if modality not in {"accelerometer", "gyroscope"}:
            raise SensorReliabilityError(
                f"{condition_id}.modality must be accelerometer or gyroscope"
            )
        return MissingModality(
            condition_id,
            modality,
            _finite_number(parameters["fill_value"], location=f"{condition_id}.fill_value"),
            condition_sha256,
        )
    if kind == "rate_reduction":
        require_exact_keys(parameters, {"factor"}, location=condition_id)
        factor = _integer(parameters["factor"], location=f"{condition_id}.factor", minimum=2)
        if factor >= window_length:
            raise SensorReliabilityError(
                f"{condition_id}.factor must be smaller than window length"
            )
        return RateReduction(condition_id, factor, condition_sha256)
    if kind == "linear_drift":
        require_exact_keys(parameters, {"endpoint_offsets"}, location=condition_id)
        offsets = _float_tuple(
            parameters["endpoint_offsets"],
            location=f"{condition_id}.endpoint_offsets",
            length=channel_count,
        )
        if not any(item != 0.0 for item in offsets):
            raise SensorReliabilityError(f"{condition_id}.endpoint_offsets cannot all be zero")
        return LinearDrift(condition_id, offsets, condition_sha256)
    raise SensorReliabilityError(f"unsupported stress condition type: {kind!r}")


def load_sensor_reliability_config(path: str | Path) -> SensorReliabilityConfig:
    """Load and self-hash-validate the locked secondary stress configuration."""

    try:
        parsed = load_strict_yaml_mapping(path)
        require_exact_keys(parsed, _TOP_LEVEL_KEYS, location="stress configuration")
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc
    claimed_hash = parsed["config_sha256"]
    if not isinstance(claimed_hash, str):
        raise SensorReliabilityError("config_sha256 must be a lowercase SHA-256")
    unhashed = dict(parsed)
    unhashed.pop("config_sha256")
    observed_hash = canonical_json_sha256(unhashed)
    if claimed_hash != observed_hash:
        raise SensorReliabilityError(
            f"stress configuration self-hash mismatch: claimed={claimed_hash}, observed={observed_hash}"
        )
    exact_values = {
        "schema_version": "1.0.0",
        "status": "configured_not_run",
        "track_role": "secondary_post_confirmatory",
        "input_space": "physical_pre_normalization",
    }
    for key, expected in exact_values.items():
        if parsed[key] != expected:
            raise SensorReliabilityError(f"{key} must be {expected!r}")
    config_id = parsed["config_id"]
    if not isinstance(config_id, str) or not config_id.strip():
        raise SensorReliabilityError("config_id must be a non-empty string")
    window_length = _integer(
        parsed["window_length_samples"], location="window_length_samples", minimum=2
    )
    sampling_rate = _finite_number(parsed["sampling_rate_hz"], location="sampling_rate_hz")
    if sampling_rate <= 0.0:
        raise SensorReliabilityError("sampling_rate_hz must be positive")
    raw_channels = parsed["channels"]
    if not isinstance(raw_channels, list) or not all(
        isinstance(item, str) for item in raw_channels
    ):
        raise SensorReliabilityError("channels must be a list of strings")
    channels = tuple(raw_channels)
    if channels != INCLUSIVEHAR_PRIMARY_CHANNELS:
        raise SensorReliabilityError(
            "channels must exactly match the locked accelerometer-then-gyroscope interface"
        )
    raw_units = parsed["channel_units"]
    if not isinstance(raw_units, list) or not all(isinstance(item, str) for item in raw_units):
        raise SensorReliabilityError("channel_units must be a list of strings")
    units = tuple(raw_units)
    if len(units) != len(channels):
        raise SensorReliabilityError("channel_units must align with channels")

    seed = _mapping(parsed["deterministic_seed"], location="deterministic_seed")
    try:
        require_exact_keys(seed, {"namespace", "global_seed"}, location="deterministic_seed")
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc
    namespace = seed["namespace"]
    if not isinstance(namespace, str) or not namespace.strip():
        raise SensorReliabilityError("deterministic_seed.namespace must be non-empty")
    global_seed = _integer(seed["global_seed"], location="global_seed", minimum=0)

    preconditions = _mapping(parsed["preconditions"], location="preconditions")
    policy = _mapping(parsed["policy"], location="policy")
    operation = _mapping(parsed["operation"], location="operation")
    try:
        require_exact_keys(preconditions, _PRECONDITION_KEYS, location="preconditions")
        require_exact_keys(policy, _POLICY_KEYS, location="policy")
        require_exact_keys(operation, _OPERATION_KEYS, location="operation")
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc
    required_preconditions = {
        "primary_confirmatory_status": "complete_before_stress_opening",
        "frozen_checkpoint_sha256_required": True,
        "base_prediction_sha256_required": True,
        "frozen_normalization_sha256_required": True,
    }
    if preconditions != required_preconditions:
        raise SensorReliabilityError("post-confirmatory precondition contract was changed")
    required_policy = {
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "calibration_refit_allowed": False,
        "threshold_refit_allowed": False,
        "target_based_tuning_allowed": False,
        "participant_level_reporting_required": True,
        "stress_levels_empirically_calibrated": False,
    }
    if policy != required_policy:
        raise SensorReliabilityError("secondary stress policy contract was changed")

    model_family = _mapping(operation["model_family"], location="operation.model_family")
    cohorts = _mapping(operation["cohorts"], location="operation.cohorts")
    try:
        require_exact_keys(
            model_family,
            {
                "family_id",
                "inventory_training_regime",
                "eligible_training_model_names",
                "selection_rule",
            },
            location="operation.model_family",
        )
        require_exact_keys(cohorts, {"source", "target"}, location="operation.cohorts")
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc
    family_id = model_family["family_id"]
    if not isinstance(family_id, str) or not family_id.strip():
        raise SensorReliabilityError("operation.model_family.family_id must be non-empty")
    if model_family["inventory_training_regime"] != "fixed_epoch_neural":
        raise SensorReliabilityError("stress operation is restricted to frozen neural entries")
    if model_family["selection_rule"] != "exact_training_configuration_model_name":
        raise SensorReliabilityError("stress model-family selection rule changed")
    raw_model_names = model_family["eligible_training_model_names"]
    if (
        not isinstance(raw_model_names, list)
        or not raw_model_names
        or any(not isinstance(item, str) or not item for item in raw_model_names)
        or len(set(raw_model_names)) != len(raw_model_names)
    ):
        raise SensorReliabilityError("eligible training model names must be unique strings")
    source_cohort = _mapping(cohorts["source"], location="operation.cohorts.source")
    target_cohort = _mapping(cohorts["target"], location="operation.cohorts.target")
    try:
        require_exact_keys(
            source_cohort,
            {"required_partition", "clean_reference"},
            location="operation.cohorts.source",
        )
        require_exact_keys(
            target_cohort,
            {"required_partition", "clean_reference"},
            location="operation.cohorts.target",
        )
    except StrictConfigError as exc:
        raise SensorReliabilityError(str(exc)) from exc
    expected_cohorts = {
        "source_partition": "source_validation",
        "source_clean": "recompute_once_with_frozen_pipeline",
        "target_partition": "target_sealed",
        "target_clean": "reuse_locked_target_opening_1_artifact",
    }
    observed_cohorts = {
        "source_partition": source_cohort["required_partition"],
        "source_clean": source_cohort["clean_reference"],
        "target_partition": target_cohort["required_partition"],
        "target_clean": target_cohort["clean_reference"],
    }
    if observed_cohorts != expected_cohorts:
        raise SensorReliabilityError("stress cohort or clean-reference contract changed")

    raw_conditions = parsed["conditions"]
    if not isinstance(raw_conditions, list) or not raw_conditions:
        raise SensorReliabilityError("conditions must be a non-empty list")
    conditions: list[StressCondition] = []
    for index, item in enumerate(raw_conditions):
        try:
            mapping = _mapping(item, location=f"conditions[{index}]")
            conditions.append(
                _condition_from_mapping(
                    mapping, channel_count=len(channels), window_length=window_length
                )
            )
        except StrictConfigError as exc:
            raise SensorReliabilityError(str(exc)) from exc
    ids = [item.condition_id for item in conditions]
    if len(set(ids)) != len(ids):
        raise SensorReliabilityError("condition IDs must be unique")
    required_kinds = {
        "gaussian_noise",
        "contiguous_dropout",
        "missing_axis",
        "missing_modality",
        "rate_reduction",
        "linear_drift",
    }
    if {item.kind for item in conditions} != required_kinds:
        raise SensorReliabilityError(
            "locked stress configuration must contain every required reliability stress type"
        )
    return SensorReliabilityConfig(
        schema_version="1.0.0",
        config_id=config_id,
        config_sha256=claimed_hash,
        status="configured_not_run",
        track_role="secondary_post_confirmatory",
        input_space="physical_pre_normalization",
        window_length_samples=window_length,
        sampling_rate_hz=sampling_rate,
        channels=channels,
        channel_units=units,
        seed_namespace=namespace,
        global_seed=global_seed,
        model_family_id=family_id,
        eligible_training_model_names=tuple(raw_model_names),
        required_training_regime="fixed_epoch_neural",
        source_partition="source_validation",
        target_partition="target_sealed",
        target_clean_reference="reuse_locked_target_opening_1_artifact",
        conditions=tuple(conditions),
    )


def _window_seed(config: SensorReliabilityConfig, condition_id: str, window_id: str) -> int:
    material = {
        "namespace": config.seed_namespace,
        "global_seed": config.global_seed,
        "condition_id": condition_id,
        "window_id": window_id,
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).digest()
    return int.from_bytes(digest[:16], byteorder="big", signed=False)


def _validate_windows(
    windows: FloatWindows, window_ids: Sequence[str], config: SensorReliabilityConfig
) -> tuple[FloatWindows, tuple[str, ...]]:
    array = np.asarray(windows)
    if array.ndim != 3 or array.shape[1:] != (
        config.window_length_samples,
        len(config.channels),
    ):
        raise SensorReliabilityError(
            "windows must have shape [N, locked_window_length, locked_channel_count]"
        )
    if array.dtype.kind != "f" or not np.isfinite(array).all():
        raise SensorReliabilityError("windows must use a finite floating dtype")
    ids = tuple(window_ids)
    if len(ids) != array.shape[0] or any(not item for item in ids):
        raise SensorReliabilityError("window IDs must be non-empty and align with windows")
    if len(set(ids)) != len(ids):
        raise SensorReliabilityError("window IDs must be unique")
    return array, ids


def _reduce_rate(window: NDArray[np.float64], *, factor: int) -> NDArray[np.float64]:
    sample_count = window.shape[0]
    retained = np.arange(0, sample_count, factor, dtype=np.int64)
    if retained[-1] != sample_count - 1:
        retained = np.concatenate((retained, np.asarray([sample_count - 1], dtype=np.int64)))
    full = np.arange(sample_count, dtype=np.float64)
    return np.column_stack(
        [
            np.interp(full, retained.astype(np.float64), window[retained, channel])
            for channel in range(window.shape[1])
        ]
    )


def apply_sensor_reliability_stress(
    windows: FloatWindows,
    window_ids: Sequence[str],
    *,
    config: SensorReliabilityConfig,
    condition_id: str,
) -> CorruptedWindows:
    """Apply one locked condition independently to each physical-unit window.

    Random conditions derive one PCG64 seed from the immutable configuration,
    condition ID, and stable window ID.  Results are therefore invariant to
    batching and row order.  The input is never mutated.
    """

    source, ids = _validate_windows(windows, window_ids, config)
    condition = config.condition(condition_id)
    result = np.array(source, copy=True)
    work = result.astype(np.float64, copy=False)
    channel_count = len(config.channels)

    for row, window_id in enumerate(ids):
        if isinstance(condition, GaussianNoise):
            generator = np.random.Generator(
                np.random.PCG64(_window_seed(config, condition.condition_id, window_id))
            )
            noise = generator.normal(size=work[row].shape) * np.asarray(condition.channel_std)
            work[row] += noise
        elif isinstance(condition, ContiguousDropout):
            generator = np.random.Generator(
                np.random.PCG64(_window_seed(config, condition.condition_id, window_id))
            )
            maximum_start = config.window_length_samples - condition.duration_samples
            start = int(generator.integers(0, maximum_start + 1))
            stop = start + condition.duration_samples
            work[row, start:stop, list(condition.channel_indices)] = condition.fill_value
        elif isinstance(condition, MissingAxis):
            work[row, :, condition.channel_index] = condition.fill_value
        elif isinstance(condition, MissingModality):
            selected = slice(0, 3) if condition.modality == "accelerometer" else slice(3, 6)
            work[row, :, selected] = condition.fill_value
        elif isinstance(condition, RateReduction):
            work[row] = _reduce_rate(work[row], factor=condition.factor)
        elif isinstance(condition, LinearDrift):
            scale = np.linspace(0.0, 1.0, config.window_length_samples, dtype=np.float64)
            work[row] += scale[:, None] * np.asarray(condition.endpoint_offsets)[None, :]
        else:  # pragma: no cover - exhaustive union guard
            raise SensorReliabilityError("unhandled stress condition")

    if work is not result:
        result = work.astype(source.dtype, copy=False)
    if (
        result.shape != source.shape
        or result.dtype != source.dtype
        or not np.isfinite(result).all()
    ):
        raise SensorReliabilityError("corruption produced an invalid tensor")
    metadata = {
        "schema_version": "1.0.0",
        "record_kind": "deterministic_sensor_reliability_corruption",
        "track_role": config.track_role,
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_based_tuning": False,
        "input_space": config.input_space,
        "stress_config_id": config.config_id,
        "stress_config_sha256": config.config_sha256,
        "condition_id": condition.condition_id,
        "condition_type": condition.kind,
        "condition_sha256": condition.condition_sha256,
        "window_count": len(ids),
        "window_ids_sha256": canonical_json_sha256(list(ids)),
        "seed_derivation": "sha256(namespace,global_seed,condition_id,window_id)->pcg64",
        "boundary_contract": "each_already_materialized_window_independently",
        "channel_count": channel_count,
    }
    metadata["record_sha256"] = canonical_json_sha256(metadata)
    return CorruptedWindows(windows=result, metadata=metadata)
