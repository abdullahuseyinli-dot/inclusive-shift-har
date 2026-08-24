"""CUDA-only post-confirmatory raw/total-acceleration sensitivity runner.

The source stage is completed and create-only locked before the consumed target
opening is loaded.  This is an exploratory signal-definition sensitivity, not a
new confirmatory opening, an exact UCI signal match, or trial-safe evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.artifacts.source_finalization import load_final_selection_plan
from inclusive_shift_har.data.materialize import MaterializedWindows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation._strict_config import (
    StrictConfigError,
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
    require_utc_timestamp,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    load_consumed_target_context,
    load_target_clean_reference,
)
from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    load_source_temperature_calibrator_file,
    write_source_temperature_calibrator_new,
)
from inclusive_shift_har.experiments.final_source_suite import (
    validate_plan_configuration_contract,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    predict_model,
    reconstruct_checkpoint,
    train_source_model,
)

RAW_TOTAL_CHANNELS = (
    "accelerometerAccelerationX",
    "accelerometerAccelerationY",
    "accelerometerAccelerationZ",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
)
MODEL_IDS = ("compact-erm", "more-har-full")
SEED_ORDER = (11, 23, 47, 89, 131)
SOURCE_TRAIN_PARTICIPANTS = ("1", "2", "3", "4", "5", "6", "7", "9")
SOURCE_VALIDATION_PARTICIPANTS = ("8", "10")
TARGET_PARTICIPANTS = tuple(str(value) for value in range(11, 21))
EVIDENCE_STATUS = "post_confirmatory_raw_total_acceleration_sensitivity"
TRACK_ROLE = "post_confirmatory_signal_definition_sensitivity"

_LOCKED_PREDICTION_ARRAY_KEYS = {
    "evidence_status",
    "window_ids",
    "participant_ids",
    "true_labels",
    "logits",
    "uncalibrated_probabilities",
    "calibrated_probabilities",
    "predicted_labels",
}
_SENSITIVITY_ARRAY_KEYS = {
    "evidence_status",
    "model_id",
    "seed",
    "cohort",
    "window_ids",
    "participant_ids",
    "true_labels",
    "logits",
    "uncalibrated_probabilities",
    "source_temperature_probabilities",
    "predicted_labels",
}


class RawTotalSensitivityError(RuntimeError):
    """Raised when the exploratory raw/total signal contract fails closed."""


@dataclass(frozen=True, slots=True)
class RawTotalModelSpec:
    """One architecture with the unchanged primary-track training budget."""

    model_id: str
    runner_model_name: str
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    patience: int
    minimum_epochs: int
    use_augmentation: bool
    use_content_objective: bool
    use_realization_factorization: bool
    use_group_dro: bool

    def training_config(self, seed: int) -> TrainingConfig:
        """Construct the exact fixed-epoch neural configuration for one seed."""

        return TrainingConfig(
            model_name=self.runner_model_name,
            num_classes=3,
            seed=seed,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            patience=self.patience,
            minimum_epochs=self.minimum_epochs,
            mixed_precision="float16",
            data_loader_workers=0,
            checkpoint_interval=self.epochs,
            checkpoint_selection_rule="fixed_last_epoch",
            use_augmentation=self.use_augmentation,
            use_content_objective=self.use_content_objective,
            use_realization_factorization=self.use_realization_factorization,
            use_group_dro=self.use_group_dro,
        )


@dataclass(frozen=True, slots=True)
class RawTotalSensitivityConfig:
    """Strict validated experiment configuration."""

    path: Path
    file_sha256: str
    experiment_id: str
    dataset: Mapping[str, Any]
    signal_interface: Mapping[str, Any]
    normalization: Mapping[str, Any]
    execution: Mapping[str, Any]
    opening_context: Mapping[str, Any]
    selection_lineage: Mapping[str, Any]
    models: tuple[RawTotalModelSpec, ...]
    calibration: Mapping[str, Any]
    outputs: Mapping[str, Any]
    aggregation: Mapping[str, Any]
    forbidden_claims: tuple[str, ...]


def _string(value: Any, *, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise StrictConfigError(f"{location} must be a non-empty string")
    return value


def _integer(value: Any, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise StrictConfigError(f"{location} must be an integer")
    return int(value)


def _number(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StrictConfigError(f"{location} must be a finite number")
    result = float(value)
    if not np.isfinite(result):
        raise StrictConfigError(f"{location} must be a finite number")
    return result


def _strings(value: Any, *, location: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise StrictConfigError(f"{location} must be an array of non-empty strings")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise StrictConfigError(f"{location} contains duplicates")
    return result


def _integers(value: Any, *, location: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise StrictConfigError(f"{location} must be an integer array")
    result = tuple(_integer(item, location=f"{location}[]") for item in value)
    if len(set(result)) != len(result):
        raise StrictConfigError(f"{location} contains duplicates")
    return result


def _require_sha256(value: Any, *, location: str) -> str:
    text = _string(value, location=location)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise StrictConfigError(f"{location} must be a lowercase SHA-256")
    return text


def _require_constant(actual: Any, expected: Any, *, location: str) -> None:
    if actual != expected:
        raise StrictConfigError(f"{location} must equal {expected!r}")


def load_raw_total_sensitivity_config(path: str | Path) -> RawTotalSensitivityConfig:
    """Load the exact v1 YAML schema and reject every undeclared degree of freedom."""

    source = Path(path).resolve(strict=True)
    raw = load_strict_yaml_mapping(source)
    require_exact_keys(
        raw,
        {
            "schema_version",
            "experiment_id",
            "status",
            "track_role",
            "primary_claim_eligible",
            "selection_basis",
            "dataset",
            "signal_interface",
            "normalization",
            "execution",
            "opening_context",
            "selection_lineage",
            "models",
            "calibration",
            "outputs",
            "aggregation",
            "forbidden_claims",
        },
        location="$",
    )
    constants = {
        "schema_version": "1.0.0",
        "experiment_id": "raw-total-acceleration-sensitivity-v1",
        "status": "predeclared_post_confirmatory_exploratory",
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "selection_basis": (
            "compact_erm_backbone_and_more_har_full_predeclared_independent_of_observed_ranking"
        ),
    }
    for key, expected in constants.items():
        _require_constant(raw.get(key), expected, location=key)

    dataset = require_mapping(raw["dataset"], location="dataset")
    require_exact_keys(
        dataset,
        {
            "dataset_manifest_path",
            "dataset_manifest_file_sha256",
            "source_window_manifest_path",
            "source_window_manifest_file_sha256",
            "source_window_manifest_sha256",
            "split_manifest_path",
            "split_manifest_file_sha256",
            "split_manifest_sha256",
            "raw_sensor_artifact_sha256",
            "ontology_config_sha256",
            "ontology_track",
            "class_names",
            "source_train_participants",
            "source_validation_participants",
            "target_participants",
            "expected_source_train_windows",
            "expected_source_validation_windows",
            "expected_target_windows",
        },
        location="dataset",
    )
    for key in (
        "dataset_manifest_file_sha256",
        "source_window_manifest_file_sha256",
        "source_window_manifest_sha256",
        "split_manifest_file_sha256",
        "split_manifest_sha256",
        "raw_sensor_artifact_sha256",
        "ontology_config_sha256",
    ):
        _require_sha256(dataset.get(key), location=f"dataset.{key}")
    for key in (
        "dataset_manifest_path",
        "source_window_manifest_path",
        "split_manifest_path",
    ):
        _string(dataset.get(key), location=f"dataset.{key}")
    _require_constant(dataset.get("ontology_track"), "functional_core", location="ontology")
    _require_constant(
        _strings(dataset.get("class_names"), location="dataset.class_names"),
        ("mobility", "sitting", "standing"),
        location="dataset.class_names",
    )
    participant_contracts = (
        ("source_train_participants", SOURCE_TRAIN_PARTICIPANTS),
        ("source_validation_participants", SOURCE_VALIDATION_PARTICIPANTS),
        ("target_participants", TARGET_PARTICIPANTS),
    )
    for key, expected in participant_contracts:
        _require_constant(
            _strings(dataset.get(key), location=f"dataset.{key}"), expected, location=key
        )
    count_contracts = {
        "expected_source_train_windows": 582,
        "expected_source_validation_windows": 143,
        "expected_target_windows": 807,
    }
    for key, expected in count_contracts.items():
        _require_constant(_integer(dataset.get(key), location=key), expected, location=key)

    signal = require_mapping(raw["signal_interface"], location="signal_interface")
    require_exact_keys(
        signal,
        {
            "channels",
            "acceleration_definition",
            "gyroscope_definition",
            "acceleration_unit_status",
            "preprocessing_config_path",
            "preprocessing_config_file_sha256",
            "window_length_samples",
            "window_stride_samples",
            "uci_body_acceleration_equivalent",
            "trial_safe",
            "trial_boundary_status",
        },
        location="signal_interface",
    )
    _require_constant(
        _strings(signal.get("channels"), location="signal_interface.channels"),
        RAW_TOTAL_CHANNELS,
        location="signal_interface.channels",
    )
    signal_constants = {
        "acceleration_definition": "released_raw_or_total_device_acceleration_including_gravity",
        "gyroscope_definition": "released_motion_rotation_rate",
        "acceleration_unit_status": "not_encoded_in_released_csv",
        "window_length_samples": 128,
        "window_stride_samples": 128,
        "uci_body_acceleration_equivalent": False,
        "trial_safe": False,
        "trial_boundary_status": "unrecoverable",
    }
    for key, expected in signal_constants.items():
        _require_constant(signal.get(key), expected, location=f"signal_interface.{key}")
    _string(signal.get("preprocessing_config_path"), location="preprocessing config path")
    _require_sha256(
        signal.get("preprocessing_config_file_sha256"), location="preprocessing config hash"
    )

    normalization = require_mapping(raw["normalization"], location="normalization")
    require_exact_keys(
        normalization,
        {"method", "fit_partition", "fit_participants", "target_in_fit"},
        location="normalization",
    )
    normalization_constants = {
        "method": "per_channel_population_standardization",
        "fit_partition": "source_train",
        "target_in_fit": False,
    }
    for key, expected in normalization_constants.items():
        _require_constant(normalization.get(key), expected, location=f"normalization.{key}")
    _require_constant(
        _strings(normalization.get("fit_participants"), location="normalization participants"),
        SOURCE_TRAIN_PARTICIPANTS,
        location="normalization.fit_participants",
    )

    execution = require_mapping(raw["execution"], location="execution")
    require_exact_keys(
        execution,
        {
            "neural_device",
            "mixed_precision",
            "data_loader_workers",
            "sequential_runs",
            "require_clean_worktree",
            "source_stage_must_complete_before_target_access",
            "source_checkpoint_rule",
            "required_seed_order",
        },
        location="execution",
    )
    execution_constants = {
        "neural_device": "cuda",
        "mixed_precision": "float16",
        "data_loader_workers": 0,
        "sequential_runs": True,
        "require_clean_worktree": True,
        "source_stage_must_complete_before_target_access": True,
        "source_checkpoint_rule": "fixed_last_epoch",
    }
    for key, expected in execution_constants.items():
        _require_constant(execution.get(key), expected, location=f"execution.{key}")
    _require_constant(
        _integers(execution.get("required_seed_order"), location="execution seeds"),
        SEED_ORDER,
        location="execution.required_seed_order",
    )

    opening = require_mapping(raw["opening_context"], location="opening_context")
    require_exact_keys(
        opening,
        {
            "opening_number",
            "opening_receipt_path",
            "opening_receipt_file_sha256",
            "opening_receipt_record_sha256",
            "locked_target_index_path",
            "locked_target_index_file_sha256",
            "locked_target_index_record_sha256",
            "target_seal_id",
            "unlock_or_new_opening_invocation_allowed",
        },
        location="opening_context",
    )
    _require_constant(opening.get("opening_number"), 1, location="opening number")
    _require_constant(
        opening.get("unlock_or_new_opening_invocation_allowed"),
        False,
        location="opening unlock policy",
    )
    for key in ("opening_receipt_path", "locked_target_index_path"):
        _string(opening.get(key), location=f"opening_context.{key}")
    for key in (
        "opening_receipt_file_sha256",
        "opening_receipt_record_sha256",
        "locked_target_index_file_sha256",
        "locked_target_index_record_sha256",
        "target_seal_id",
    ):
        _require_sha256(opening.get(key), location=f"opening_context.{key}")

    selection = require_mapping(raw["selection_lineage"], location="selection_lineage")
    require_exact_keys(
        selection,
        {
            "final_source_selection_plan_path",
            "final_source_selection_plan_file_sha256",
            "final_source_selection_plan_sha256",
            "target_metrics_used_to_select_architectures",
        },
        location="selection_lineage",
    )
    _string(selection.get("final_source_selection_plan_path"), location="selection path")
    for key in (
        "final_source_selection_plan_file_sha256",
        "final_source_selection_plan_sha256",
    ):
        _require_sha256(selection.get(key), location=f"selection_lineage.{key}")
    _require_constant(
        selection.get("target_metrics_used_to_select_architectures"),
        False,
        location="selection target-metric policy",
    )

    models_value = raw["models"]
    if not isinstance(models_value, list) or len(models_value) != 2:
        raise StrictConfigError("models must contain exactly two architecture specifications")
    model_keys = {
        "model_id",
        "runner_model_name",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "minimum_epochs",
        "use_augmentation",
        "use_content_objective",
        "use_realization_factorization",
        "use_group_dro",
    }
    models: list[RawTotalModelSpec] = []
    for index, value in enumerate(models_value):
        item = require_mapping(value, location=f"models[{index}]")
        require_exact_keys(item, model_keys, location=f"models[{index}]")
        models.append(
            RawTotalModelSpec(
                model_id=_string(item.get("model_id"), location="model_id"),
                runner_model_name=_string(
                    item.get("runner_model_name"), location="runner_model_name"
                ),
                epochs=_integer(item.get("epochs"), location="epochs"),
                batch_size=_integer(item.get("batch_size"), location="batch_size"),
                learning_rate=_number(item.get("learning_rate"), location="learning_rate"),
                weight_decay=_number(item.get("weight_decay"), location="weight_decay"),
                patience=_integer(item.get("patience"), location="patience"),
                minimum_epochs=_integer(item.get("minimum_epochs"), location="minimum_epochs"),
                use_augmentation=item.get("use_augmentation") is True,
                use_content_objective=item.get("use_content_objective") is True,
                use_realization_factorization=item.get("use_realization_factorization") is True,
                use_group_dro=item.get("use_group_dro") is True,
            )
        )
        for flag in (
            "use_augmentation",
            "use_content_objective",
            "use_realization_factorization",
            "use_group_dro",
        ):
            if not isinstance(item.get(flag), bool):
                raise StrictConfigError(f"models[{index}].{flag} must be boolean")
    if tuple(model.model_id for model in models) != MODEL_IDS:
        raise StrictConfigError("models must be ordered compact-erm then more-har-full")
    expected_configs = {
        "compact-erm": (
            "compact_residual_96",
            23,
            256,
            0.001,
            0.0001,
            7,
            8,
            False,
            False,
            False,
            False,
        ),
        "more-har-full": (
            "more_har",
            27,
            256,
            0.0003,
            0.0001,
            9,
            8,
            True,
            True,
            True,
            True,
        ),
    }
    for model in models:
        observed = (
            model.runner_model_name,
            model.epochs,
            model.batch_size,
            model.learning_rate,
            model.weight_decay,
            model.patience,
            model.minimum_epochs,
            model.use_augmentation,
            model.use_content_objective,
            model.use_realization_factorization,
            model.use_group_dro,
        )
        _require_constant(observed, expected_configs[model.model_id], location=model.model_id)

    calibration = require_mapping(raw["calibration"], location="calibration")
    require_exact_keys(
        calibration,
        {
            "method",
            "fit_partition",
            "target_labels_or_metrics_used_for_fit",
            "target_calibration_or_refit_allowed",
        },
        location="calibration",
    )
    calibration_constants = {
        "method": "scalar_temperature",
        "fit_partition": "source_validation",
        "target_labels_or_metrics_used_for_fit": False,
        "target_calibration_or_refit_allowed": False,
    }
    for key, expected in calibration_constants.items():
        _require_constant(calibration.get(key), expected, location=f"calibration.{key}")

    outputs = require_mapping(raw["outputs"], location="outputs")
    require_exact_keys(
        outputs,
        {
            "output_directory",
            "source_stage_lock_filename",
            "index_filename",
            "aggregate_json_filename",
            "aggregate_csv_filename",
            "aggregate_markdown_filename",
            "failure_directory_name",
        },
        location="outputs",
    )
    for key, value in outputs.items():
        _string(value, location=f"outputs.{key}")

    aggregation = require_mapping(raw["aggregation"], location="aggregation")
    require_exact_keys(
        aggregation,
        {
            "statistical_unit",
            "seed_handling",
            "paired_bootstrap_resamples",
            "paired_bootstrap_seed",
            "confidence",
            "multiple_comparison_status",
        },
        location="aggregation",
    )
    aggregation_constants = {
        "statistical_unit": "participant",
        "seed_handling": ("participant_metric_averaged_over_five_locked_seeds_before_summary"),
        "paired_bootstrap_resamples": 10000,
        "paired_bootstrap_seed": 1729,
        "confidence": 0.95,
        "multiple_comparison_status": "exploratory_no_confirmatory_claim",
    }
    for key, expected in aggregation_constants.items():
        _require_constant(aggregation.get(key), expected, location=f"aggregation.{key}")

    forbidden = _strings(raw["forbidden_claims"], location="forbidden_claims")
    _require_constant(
        forbidden,
        ("confirmatory", "exact_UCI_signal_match", "trial_safe", "causal_disability_effect"),
        location="forbidden_claims",
    )
    return RawTotalSensitivityConfig(
        path=source,
        file_sha256=sha256_file(source),
        experiment_id="raw-total-acceleration-sensitivity-v1",
        dataset=dataset,
        signal_interface=signal,
        normalization=normalization,
        execution=execution,
        opening_context=opening,
        selection_lineage=selection,
        models=tuple(models),
        calibration=calibration,
        outputs=outputs,
        aggregation=aggregation,
        forbidden_claims=forbidden,
    )


def _selected_records(
    records: Sequence[WindowRecord],
    *,
    ontology_track: str,
    allowed_partitions: frozenset[str],
) -> list[WindowRecord]:
    selected = [
        record
        for record in records
        if record.partition in allowed_partitions and ontology_track in record.canonical_labels
    ]
    selected.sort(key=lambda record: record.start_row_inclusive)
    if not selected:
        raise RawTotalSensitivityError("no windows match the authorized sensitivity partition")
    for record in selected:
        if (
            record.length_samples != 128
            or record.stride_samples != 128
            or record.end_row_inclusive - record.start_row_inclusive + 1 != 128
            or record.trial_id is not None
            or record.trial_status != "unrecoverable"
        ):
            raise RawTotalSensitivityError("window or unrecoverable-trial contract changed")
    for previous, current in pairwise(selected):
        if current.start_row_inclusive <= previous.end_row_inclusive:
            raise RawTotalSensitivityError("selected raw intervals overlap")
    return selected


def materialize_signal_definition_windows(
    csv_path: str | Path,
    records: Sequence[WindowRecord],
    *,
    expected_source_sha256: str,
    channels: tuple[str, ...],
    ontology_track: str,
    class_names: tuple[str, ...],
    allowed_partitions: frozenset[str],
    target_context: ConsumedTargetContext | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_target_seal_id: str | None = None,
) -> MaterializedWindows:
    """Read one exact channel schema without invoking an unlock or new opening."""

    source = Path(csv_path)
    if channels != RAW_TOTAL_CHANNELS:
        raise RawTotalSensitivityError("only the raw/total six-channel interface is allowed")
    if len(class_names) != 3 or len(set(class_names)) != 3:
        raise RawTotalSensitivityError("functional-core class schema is invalid")
    target_requested = any(partition.startswith("target") for partition in allowed_partitions)
    if target_requested:
        if target_context is None:
            raise PermissionError("target signal materialization requires consumed opening 1")
        if (
            expected_split_manifest_sha256 is None
            or expected_target_seal_id is None
            or target_context.receipt.get("split_manifest_sha256") != expected_split_manifest_sha256
            or target_context.index.get("split_manifest_sha256") != expected_split_manifest_sha256
            or target_context.receipt.get("target_seal_id") != expected_target_seal_id
            or target_context.index.get("target_seal_id") != expected_target_seal_id
        ):
            raise PermissionError("consumed opening context differs from split/seal lineage")
    elif target_context is not None:
        raise RawTotalSensitivityError("source materialization may not receive target context")
    if sha256_file(source) != expected_source_sha256:
        raise RawTotalSensitivityError("raw sensor artifact hash mismatch")
    selected = _selected_records(
        records, ontology_track=ontology_track, allowed_partitions=allowed_partitions
    )
    class_to_index = {name: index for index, name in enumerate(class_names)}
    signals = np.empty((len(selected), 128, 6), dtype=np.float32)
    labels = np.empty(len(selected), dtype=np.int64)
    window_ids: list[str] = []
    participant_ids: list[str] = []
    released_labels: list[str] = []
    partitions: list[str] = []
    next_window = 0
    active_values: list[list[float]] = []
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RawTotalSensitivityError("InclusiveHAR CSV is empty") from exc
        if len(header) != len(set(header)):
            raise RawTotalSensitivityError("InclusiveHAR CSV header contains duplicate names")
        header_index = {name: index for index, name in enumerate(header)}
        required = {*channels, "label", "UserID"}
        missing = sorted(required - set(header_index))
        if missing:
            raise RawTotalSensitivityError(f"raw CSV lacks required sensitivity columns: {missing}")
        channel_indices = [header_index[name] for name in channels]
        for data_row, row in enumerate(reader, start=1):
            if next_window >= len(selected):
                break
            active = selected[next_window]
            if data_row < active.start_row_inclusive:
                continue
            if data_row > active.end_row_inclusive:
                raise RawTotalSensitivityError("materializer skipped an expected window row")
            if len(row) != len(header):
                raise RawTotalSensitivityError("raw CSV row width differs from its header")
            if row[header_index["label"]].strip() != active.activity_label:
                raise RawTotalSensitivityError("window crosses or mismatches an activity label")
            try:
                participant = str(int(row[header_index["UserID"]].strip()))
            except ValueError as exc:
                raise RawTotalSensitivityError("window contains an invalid participant ID") from exc
            if participant != active.subject_id:
                raise RawTotalSensitivityError("window crosses or mismatches a participant")
            try:
                active_values.append([float(row[index]) for index in channel_indices])
            except ValueError as exc:
                raise RawTotalSensitivityError("window contains a nonnumeric signal value") from exc
            if data_row == active.end_row_inclusive:
                array = np.asarray(active_values, dtype=np.float32)
                if array.shape != (128, 6) or not np.isfinite(array).all():
                    raise RawTotalSensitivityError("window violates the finite [128,6] contract")
                canonical_label = active.canonical_labels[ontology_track]
                if canonical_label not in class_to_index:
                    raise RawTotalSensitivityError("window label lies outside functional-core")
                signals[next_window] = array
                labels[next_window] = class_to_index[canonical_label]
                window_ids.append(active.window_id)
                participant_ids.append(active.subject_id)
                released_labels.append(active.activity_label)
                partitions.append(active.partition)
                next_window += 1
                active_values = []
    if next_window != len(selected):
        raise RawTotalSensitivityError(
            f"raw CSV ended after {next_window} of {len(selected)} authorized windows"
        )
    return MaterializedWindows(
        signals=signals,
        labels=labels,
        window_ids=tuple(window_ids),
        participant_ids=tuple(participant_ids),
        released_labels=tuple(released_labels),
        partitions=tuple(partitions),
        class_names=class_names,
        ontology_track=ontology_track,
    )


@dataclass(frozen=True, slots=True)
class _SourceArrays:
    train_windows: NDArray[np.float32]
    train_labels: NDArray[np.int64]
    train_participants: list[str]
    validation_windows: NDArray[np.float32]
    validation_labels: NDArray[np.int64]
    validation_participants: list[str]
    validation_window_ids: tuple[str, ...]
    standardizer: ChannelStandardizer


def _confined_path(
    value: str | Path,
    *,
    root: Path,
    role: str,
    must_exist: bool,
) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise RawTotalSensitivityError(f"{role} may not be a symlink")
    if must_exist:
        resolved = candidate.resolve(strict=True)
    else:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        resolved = candidate.parent.resolve(strict=True) / candidate.name
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RawTotalSensitivityError(f"{role} escapes the repository root") from exc
    if must_exist and not resolved.is_file():
        raise RawTotalSensitivityError(f"{role} is not a regular file")
    return resolved


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawTotalSensitivityError(f"{name} must be an object")
    return value


def _validate_self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise RawTotalSensitivityError(f"{name} self-hash does not validate")
    return claimed


def _load_json_with_file_hash(
    path_value: Any,
    hash_value: Any,
    *,
    root: Path,
    role: str,
) -> tuple[Path, Mapping[str, Any]]:
    path = _confined_path(_string(path_value, location=role), root=root, role=role, must_exist=True)
    expected = _require_sha256(hash_value, location=f"{role} file hash")
    if sha256_file(path) != expected:
        raise RawTotalSensitivityError(f"{role} file hash changed")
    return path, _mapping(load_json_strict(path), name=role)


def _windows_from_manifest(manifest: Mapping[str, Any], *, role: str) -> tuple[WindowRecord, ...]:
    values = manifest.get("windows")
    if not isinstance(values, list) or not values:
        raise RawTotalSensitivityError(f"{role} has no windows")
    records: list[WindowRecord] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            raise RawTotalSensitivityError(f"{role} window {index} is not an object")
        try:
            records.append(WindowRecord(**cast(dict[str, Any], value)))
        except (TypeError, ValueError) as exc:
            raise RawTotalSensitivityError(f"{role} window {index} is invalid") from exc
    return tuple(records)


def _validate_source_lineage(
    config: RawTotalSensitivityConfig,
    *,
    root: Path,
) -> tuple[tuple[WindowRecord, ...], tuple[str, ...]]:
    dataset = config.dataset
    dataset_path = _confined_path(
        _string(dataset["dataset_manifest_path"], location="dataset manifest path"),
        root=root,
        role="dataset manifest",
        must_exist=True,
    )
    if sha256_file(dataset_path) != dataset["dataset_manifest_file_sha256"]:
        raise RawTotalSensitivityError("dataset manifest hash changed")
    preprocessing_path = _confined_path(
        _string(
            config.signal_interface["preprocessing_config_path"],
            location="preprocessing config path",
        ),
        root=root,
        role="raw/total preprocessing config",
        must_exist=True,
    )
    if (
        sha256_file(preprocessing_path)
        != config.signal_interface["preprocessing_config_file_sha256"]
    ):
        raise RawTotalSensitivityError("raw/total preprocessing config hash changed")

    _, source_manifest = _load_json_with_file_hash(
        dataset["source_window_manifest_path"],
        dataset["source_window_manifest_file_sha256"],
        root=root,
        role="source window manifest",
    )
    source_hash = _validate_self_hash(
        source_manifest, field="source_window_manifest_sha256", name="source window manifest"
    )
    source_required = {
        "source_split_manifest_sha256": dataset["split_manifest_sha256"],
        "source_artifact_sha256": dataset["raw_sensor_artifact_sha256"],
        "trial_boundary_status": "unrecoverable",
        "window_length_samples": 128,
        "window_stride_samples": 128,
        "target_subject_or_window_records_included": False,
        "target_performance_or_prediction_accessed": False,
    }
    if source_hash != dataset["source_window_manifest_sha256"] or any(
        source_manifest.get(key) != expected for key, expected in source_required.items()
    ):
        raise RawTotalSensitivityError("source window-manifest lineage changed")
    ontology = _mapping(source_manifest.get("ontology"), name="source ontology")
    if ontology.get("config_sha256") != dataset["ontology_config_sha256"]:
        raise RawTotalSensitivityError("functional-core ontology lineage changed")
    schemas = _mapping(ontology.get("runnable_track_schemas"), name="ontology schemas")
    functional = _mapping(schemas.get("functional_core"), name="functional-core schema")
    class_names = tuple(str(value) for value in cast(list[Any], functional.get("class_order")))
    if class_names != tuple(cast(list[str], dataset["class_names"])):
        raise RawTotalSensitivityError("functional-core class order changed")

    selection = config.selection_lineage
    plan_path = _confined_path(
        _string(selection["final_source_selection_plan_path"], location="selection plan path"),
        root=root,
        role="final source selection plan",
        must_exist=True,
    )
    if sha256_file(plan_path) != selection["final_source_selection_plan_file_sha256"]:
        raise RawTotalSensitivityError("final source selection-plan file hash changed")
    plan = load_final_selection_plan(plan_path)
    if plan.get("selection_plan_sha256") != selection["final_source_selection_plan_sha256"]:
        raise RawTotalSensitivityError("final source selection-plan record hash changed")
    if (
        plan.get("required_seed_order") != list(SEED_ORDER)
        or plan.get("class_names") != list(class_names)
        or plan.get("final_train_participants") != list(SOURCE_TRAIN_PARTICIPANTS)
        or plan.get("final_validation_participants") != list(SOURCE_VALIDATION_PARTICIPANTS)
        or plan.get("final_train_window_count") != dataset["expected_source_train_windows"]
        or plan.get("final_validation_window_count")
        != dataset["expected_source_validation_windows"]
    ):
        raise RawTotalSensitivityError("source split or seed budget differs from final plan")
    reconstructed = {
        str(row["model_id"]): str(row["seed_11_configuration_sha256"])
        for row in validate_plan_configuration_contract(plan)
    }
    for model in config.models:
        expected_hash = canonical_json_sha256(asdict(model.training_config(11)))
        if reconstructed.get(model.model_id) != expected_hash:
            raise RawTotalSensitivityError(
                f"{model.model_id} training budget differs from the frozen primary plan"
            )
    return _windows_from_manifest(source_manifest, role="source window manifest"), class_names


def _source_arrays(
    batch: MaterializedWindows,
    config: RawTotalSensitivityConfig,
) -> _SourceArrays:
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    partitions = np.asarray(batch.partitions, dtype=np.str_)
    train_mask = partitions == "source_train"
    validation_mask = partitions == "source_validation"
    if not train_mask.any() or not validation_mask.any() or np.any(train_mask & validation_mask):
        raise RawTotalSensitivityError("source train/validation masks are incomplete")
    train_ids = participants[train_mask].tolist()
    validation_ids = participants[validation_mask].tolist()
    if (
        tuple(sorted(set(train_ids), key=int)) != SOURCE_TRAIN_PARTICIPANTS
        or tuple(sorted(set(validation_ids), key=int)) != SOURCE_VALIDATION_PARTICIPANTS
        or int(train_mask.sum()) != config.dataset["expected_source_train_windows"]
        or int(validation_mask.sum()) != config.dataset["expected_source_validation_windows"]
    ):
        raise RawTotalSensitivityError("source materialization differs from the fixed split")
    standardizer = ChannelStandardizer.fit(
        batch.signals[train_mask],
        train_ids,
        declared_training_participants=set(SOURCE_TRAIN_PARTICIPANTS),
        split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
        channel_names=RAW_TOTAL_CHANNELS,
    )
    if standardizer.training_participants != tuple(sorted(SOURCE_TRAIN_PARTICIPANTS)):
        raise RawTotalSensitivityError("normalization was not fit on source training only")
    return _SourceArrays(
        train_windows=standardizer.transform(batch.signals[train_mask]),
        train_labels=np.asarray(batch.labels[train_mask], dtype=np.int64),
        train_participants=train_ids,
        validation_windows=standardizer.transform(batch.signals[validation_mask]),
        validation_labels=np.asarray(batch.labels[validation_mask], dtype=np.int64),
        validation_participants=validation_ids,
        validation_window_ids=tuple(np.asarray(batch.window_ids)[validation_mask].tolist()),
        standardizer=standardizer,
    )


def _git_state(root: Path) -> tuple[str, str]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return commit, status


def _write_prediction_new(
    path: Path,
    *,
    model_id: str,
    seed: int,
    cohort: str,
    window_ids: Sequence[str],
    participant_ids: Sequence[str],
    labels: NDArray[np.int64],
    logits: NDArray[np.float64],
    uncalibrated: NDArray[np.float64],
    calibrated: NDArray[np.float64],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite sensitivity prediction: {path}")
    predicted = calibrated.argmax(axis=1).astype(np.int64)
    with path.open("xb") as stream:
        np.savez_compressed(
            stream,
            evidence_status=np.asarray([EVIDENCE_STATUS]),
            model_id=np.asarray([model_id]),
            seed=np.asarray([seed], dtype=np.int64),
            cohort=np.asarray([cohort]),
            window_ids=np.asarray(window_ids),
            participant_ids=np.asarray(participant_ids),
            true_labels=np.asarray(labels, dtype=np.int64),
            logits=np.asarray(logits, dtype=np.float64),
            uncalibrated_probabilities=np.asarray(uncalibrated, dtype=np.float64),
            source_temperature_probabilities=np.asarray(calibrated, dtype=np.float64),
            predicted_labels=predicted,
        )
    return sha256_file(path)


def _result_record(
    *,
    model_id: str,
    seed: int,
    cohort: str,
    report: Mapping[str, Any],
    prediction_path: Path,
    prediction_sha256: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    configuration: TrainingConfig,
    calibrator_path: Path,
    calibrator_file_sha256: str,
    calibrator_record_sha256: str,
    normalization: Mapping[str, Any],
    config: RawTotalSensitivityConfig,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    source_stage_lock_sha256: str | None,
    opening_context: ConsumedTargetContext | None,
) -> dict[str, Any]:
    target = cohort == "target_consumed_opening_1"
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "postconfirmatory_raw_total_acceleration_evaluation",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "model_id": model_id,
        "seed": seed,
        "cohort": cohort,
        "class_names": list(cast(Sequence[Any], config.dataset["class_names"])),
        "signal_definition": {
            "channels": list(RAW_TOTAL_CHANNELS),
            "acceleration": config.signal_interface["acceleration_definition"],
            "acceleration_unit_status": config.signal_interface["acceleration_unit_status"],
            "uci_body_acceleration_equivalent": False,
            "trial_safe": False,
        },
        "training_configuration": asdict(configuration),
        "training_configuration_sha256": canonical_json_sha256(asdict(configuration)),
        "checkpoint": {
            "path": checkpoint_path.relative_to(root).as_posix(),
            "sha256": checkpoint_sha256,
        },
        "source_temperature_calibrator": {
            "path": calibrator_path.relative_to(root).as_posix(),
            "file_sha256": calibrator_file_sha256,
            "record_sha256": calibrator_record_sha256,
            "fit_partition": "source_validation",
        },
        "normalization": dict(normalization),
        "prediction_artifact": {
            "path": prediction_path.relative_to(root).as_posix(),
            "sha256": prediction_sha256,
            "array_keys": sorted(_SENSITIVITY_ARRAY_KEYS),
        },
        "participant_level_report": dict(report),
        "lineage": {
            "experiment_config_file_sha256": config.file_sha256,
            "dataset_manifest_file_sha256": config.dataset["dataset_manifest_file_sha256"],
            "source_window_manifest_sha256": config.dataset["source_window_manifest_sha256"],
            "split_manifest_sha256": config.dataset["split_manifest_sha256"],
            "preprocessing_config_file_sha256": config.signal_interface[
                "preprocessing_config_file_sha256"
            ],
            "code_commit": code_commit,
            "source_stage_lock_record_sha256": source_stage_lock_sha256,
            "opening_receipt_record_sha256": (
                opening_context.receipt["record_sha256"] if opening_context else None
            ),
            "locked_target_index_record_sha256": (
                opening_context.index["record_sha256"] if opening_context else None
            ),
        },
        "source_stage_completed_before_target_access": True if target else None,
        "target_signals_or_labels_accessed": target,
        "target_tuning": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "interpretation": "post_confirmatory_exploratory_signal_definition_sensitivity",
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def _index_entry(
    *,
    model_id: str,
    seed: int,
    record_path: Path,
    record: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    prediction = _mapping(record.get("prediction_artifact"), name="prediction artifact")
    return {
        "model_id": model_id,
        "seed": seed,
        "record_path": record_path.relative_to(root).as_posix(),
        "record_file_sha256": sha256_file(record_path),
        "record_sha256": record["record_sha256"],
        "prediction_path": prediction["path"],
        "prediction_sha256": prediction["sha256"],
    }


def _failure_record(
    *,
    output: Path,
    root: Path,
    config: RawTotalSensitivityConfig,
    stage: str,
    model_id: str | None,
    seed: int | None,
    exc: Exception,
    code_commit: str,
    created_at_utc: str,
    target_accessed: bool,
) -> dict[str, Any]:
    identity = "track" if model_id is None else f"{model_id}--seed-{seed}"
    path = output / str(config.outputs["failure_directory_name"]) / f"{stage}--{identity}.json"
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "raw_total_acceleration_sensitivity_failure",
        "status": "failed_preserved",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "stage": stage,
        "model_id": model_id,
        "seed": seed,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "code_commit": code_commit,
        "target_accessed_before_failure": target_accessed,
        "target_tuning_or_refit": False,
        "partial_outputs_preserved": True,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "stage": stage,
        "model_id": model_id,
        "seed": seed,
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
    }


@dataclass(frozen=True, slots=True)
class _PreparedSourceRun:
    model: RawTotalModelSpec
    seed: int
    configuration: TrainingConfig
    checkpoint_path: Path
    checkpoint_sha256: str
    calibrator_path: Path
    calibrator_file_sha256: str
    calibrator_record_sha256: str


def _write_index(
    *,
    output: Path,
    root: Path,
    config: RawTotalSensitivityConfig,
    status: str,
    code_commit: str,
    created_at_utc: str,
    normalization_entry: Mapping[str, Any] | None,
    source_stage_lock_entry: Mapping[str, Any] | None,
    source_results: Sequence[Mapping[str, Any]],
    target_results: Sequence[Mapping[str, Any]],
    primary_references: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    target_accessed: bool,
    context: ConsumedTargetContext | None,
) -> dict[str, Any]:
    index_path = output / str(config.outputs["index_filename"])
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "postconfirmatory_raw_total_acceleration_sensitivity_index",
        "status": status,
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "track_role": TRACK_ROLE,
        "primary_claim_eligible": False,
        "model_selection_use": False,
        "experiment_id": config.experiment_id,
        "experiment_config": {
            "path": config.path.relative_to(root).as_posix(),
            "file_sha256": config.file_sha256,
        },
        "code_commit": code_commit,
        "signal_definition": {
            "channels": list(RAW_TOTAL_CHANNELS),
            "acceleration": config.signal_interface["acceleration_definition"],
            "acceleration_unit_status": config.signal_interface["acceleration_unit_status"],
            "uci_body_acceleration_equivalent": False,
            "trial_safe": False,
            "trial_boundary_status": "unrecoverable",
        },
        "class_names": list(cast(Sequence[Any], config.dataset["class_names"])),
        "required_model_order": list(MODEL_IDS),
        "required_seed_order": list(SEED_ORDER),
        "expected_model_seed_count": len(MODEL_IDS) * len(SEED_ORDER),
        "normalization": dict(normalization_entry) if normalization_entry else None,
        "source_stage_lock": dict(source_stage_lock_entry) if source_stage_lock_entry else None,
        "source_result_count": len(source_results),
        "target_result_count": len(target_results),
        "primary_reference_count": len(primary_references),
        "failure_count": len(failures),
        "source_results": [dict(value) for value in source_results],
        "target_results": [dict(value) for value in target_results],
        "primary_target_references": [dict(value) for value in primary_references],
        "failures": [dict(value) for value in failures],
        "source_stage_completed_before_target_access": source_stage_lock_entry is not None,
        "target_signals_or_labels_accessed": target_accessed,
        "opening_receipt_record_sha256": (
            context.receipt["record_sha256"] if context is not None else None
        ),
        "locked_target_index_record_sha256": (
            context.index["record_sha256"] if context is not None else None
        ),
        "target_tuning": False,
        "target_calibration_or_refit": False,
        "new_opening_or_unlock_invoked": False,
        "source_selection_consumed": True,
        "interpretation": "post_confirmatory_exploratory_not_primary_claim_evidence",
        "forbidden_claims": list(config.forbidden_claims),
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, index_path, allowed_root=root)
    return payload


def _normalization_record(
    standardizer: ChannelStandardizer,
    *,
    config: RawTotalSensitivityConfig,
    output: Path,
    root: Path,
    created_at_utc: str,
) -> dict[str, Any]:
    path = output / "source_training_normalization.json"
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "raw_total_acceleration_training_normalization",
        "status": "complete_create_only",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "normalization": standardizer.to_dict(),
        "fit_partition": "source_train",
        "fit_participants": list(SOURCE_TRAIN_PARTICIPANTS),
        "target_in_fit": False,
        "split_manifest_sha256": config.dataset["split_manifest_sha256"],
        "preprocessing_config_file_sha256": config.signal_interface[
            "preprocessing_config_file_sha256"
        ],
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
        "normalization_sha256": canonical_json_sha256(standardizer.to_dict()),
    }


def _train_source_runs(
    source: _SourceArrays,
    *,
    config: RawTotalSensitivityConfig,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> tuple[
    list[_PreparedSourceRun],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    prepared: list[_PreparedSourceRun] = []
    result_entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    normalization = source.standardizer.to_dict()
    lineage = TrainingLineage(
        dataset_manifest_sha256=str(config.dataset["dataset_manifest_file_sha256"]),
        split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
        preprocessing_config_sha256=str(
            config.signal_interface["preprocessing_config_file_sha256"]
        ),
        ontology_sha256=str(config.dataset["ontology_config_sha256"]),
        code_commit=code_commit,
        evidence_status=EVIDENCE_STATUS,
        label_schema=tuple(cast(list[str], config.dataset["class_names"])),
        normalization=normalization,
    )
    for model in config.models:
        for seed in SEED_ORDER:
            run_directory = output / "runs" / model.model_id / f"seed-{seed}"
            try:
                run_directory.mkdir(parents=True, exist_ok=False)
                configuration = model.training_config(seed)
                trained = train_source_model(
                    source.train_windows,
                    source.train_labels,
                    source.train_participants,
                    source.validation_windows,
                    source.validation_labels,
                    source.validation_participants,
                    config=configuration,
                    lineage=lineage,
                    output_directory=run_directory,
                    device=device,
                )
                checkpoint_path = Path(str(trained["checkpoint_path"])).resolve(strict=True)
                checkpoint_sha = str(trained["checkpoint_sha256"])
                try:
                    checkpoint_path.relative_to(root)
                except ValueError as exc:
                    raise RawTotalSensitivityError("checkpoint escaped repository root") from exc
                if sha256_file(checkpoint_path) != checkpoint_sha:
                    raise RawTotalSensitivityError("new source checkpoint hash changed")
                configuration_sha = canonical_json_sha256(asdict(configuration))
                validation_logits = np.asarray(trained["validation_logits"], dtype=np.float64)
                validation_uncalibrated = np.asarray(
                    trained["validation_probabilities"], dtype=np.float64
                )
                calibrator_record = build_source_temperature_calibrator_record(
                    validation_logits,
                    source.validation_labels,
                    source.validation_window_ids,
                    fit_partition="source_validation",
                    checkpoint_sha256=checkpoint_sha,
                    training_configuration_sha256=configuration_sha,
                    split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
                    class_names=tuple(cast(list[str], config.dataset["class_names"])),
                )
                calibrator_path = output / "calibrators" / f"{model.model_id}--seed-{seed}.json"
                calibrator_written = write_source_temperature_calibrator_new(
                    calibrator_record, calibrator_path, allowed_root=root
                )
                _, calibrator = load_source_temperature_calibrator_file(
                    calibrator_path,
                    expected_checkpoint_sha256=checkpoint_sha,
                    expected_training_configuration_sha256=configuration_sha,
                    expected_split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
                )
                validation_calibrated = calibrator.probabilities(validation_logits)
                source_report = classification_report(
                    source.validation_labels,
                    validation_calibrated,
                    source.validation_participants,
                    class_names=tuple(cast(list[str], config.dataset["class_names"])),
                )
                prediction_path = (
                    output / "predictions" / f"{model.model_id}--seed-{seed}--source.npz"
                )
                prediction_sha = _write_prediction_new(
                    prediction_path,
                    model_id=model.model_id,
                    seed=seed,
                    cohort="source_validation",
                    window_ids=source.validation_window_ids,
                    participant_ids=source.validation_participants,
                    labels=source.validation_labels,
                    logits=validation_logits,
                    uncalibrated=validation_uncalibrated,
                    calibrated=validation_calibrated,
                )
                record = _result_record(
                    model_id=model.model_id,
                    seed=seed,
                    cohort="source_validation",
                    report=source_report,
                    prediction_path=prediction_path,
                    prediction_sha256=prediction_sha,
                    checkpoint_path=checkpoint_path,
                    checkpoint_sha256=checkpoint_sha,
                    configuration=configuration,
                    calibrator_path=calibrator_path,
                    calibrator_file_sha256=str(calibrator_written["file_sha256"]),
                    calibrator_record_sha256=str(calibrator_written["record_sha256"]),
                    normalization=normalization,
                    config=config,
                    root=root,
                    code_commit=code_commit,
                    created_at_utc=created_at_utc,
                    source_stage_lock_sha256=None,
                    opening_context=None,
                )
                record_path = output / "records" / f"{model.model_id}--seed-{seed}--source.json"
                atomic_write_json_new(record, record_path, allowed_root=root)
                result_entries.append(
                    _index_entry(
                        model_id=model.model_id,
                        seed=seed,
                        record_path=record_path,
                        record=record,
                        root=root,
                    )
                )
                prepared.append(
                    _PreparedSourceRun(
                        model=model,
                        seed=seed,
                        configuration=configuration,
                        checkpoint_path=checkpoint_path,
                        checkpoint_sha256=checkpoint_sha,
                        calibrator_path=calibrator_path,
                        calibrator_file_sha256=str(calibrator_written["file_sha256"]),
                        calibrator_record_sha256=str(calibrator_written["record_sha256"]),
                    )
                )
            except Exception as exc:
                failures.append(
                    _failure_record(
                        output=output,
                        root=root,
                        config=config,
                        stage="source_training_or_calibration",
                        model_id=model.model_id,
                        seed=seed,
                        exc=exc,
                        code_commit=code_commit,
                        created_at_utc=created_at_utc,
                        target_accessed=False,
                    )
                )
            finally:
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()
    return prepared, result_entries, failures


def _source_stage_lock(
    *,
    prepared: Sequence[_PreparedSourceRun],
    source_results: Sequence[Mapping[str, Any]],
    normalization_entry: Mapping[str, Any],
    config: RawTotalSensitivityConfig,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    path = output / str(config.outputs["source_stage_lock_filename"])
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "raw_total_acceleration_source_stage_lock",
        "status": "complete_create_only_target_not_yet_accessed",
        "created_at_utc": created_at_utc,
        "evidence_status": EVIDENCE_STATUS,
        "experiment_config_file_sha256": config.file_sha256,
        "code_commit": code_commit,
        "required_model_order": list(MODEL_IDS),
        "required_seed_order": list(SEED_ORDER),
        "model_seed_count": len(prepared),
        "normalization": dict(normalization_entry),
        "source_results": [dict(entry) for entry in source_results],
        "source_artifacts": [
            {
                "model_id": run.model.model_id,
                "seed": run.seed,
                "training_configuration_sha256": canonical_json_sha256(asdict(run.configuration)),
                "checkpoint_path": run.checkpoint_path.relative_to(root).as_posix(),
                "checkpoint_sha256": run.checkpoint_sha256,
                "calibrator_path": run.calibrator_path.relative_to(root).as_posix(),
                "calibrator_file_sha256": run.calibrator_file_sha256,
                "calibrator_record_sha256": run.calibrator_record_sha256,
            }
            for run in prepared
        ],
        "whole_raw_artifact_integrity_bytes_hashed_before_lock": True,
        "target_rows_parsed_or_materialized_before_lock": False,
        "target_context_loaded": False,
        "target_signals_labels_or_metrics_accessed": False,
        "source_checkpoint_selection_used_target": False,
        "calibration_fit_partition": "source_validation",
        "target_calibration_or_refit": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, path, allowed_root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "file_sha256": sha256_file(path),
        "record_sha256": payload["record_sha256"],
    }


def _load_target_stage(
    config: RawTotalSensitivityConfig,
    *,
    root: Path,
    raw_csv_path: Path,
    class_names: tuple[str, ...],
) -> tuple[ConsumedTargetContext, MaterializedWindows]:
    opening = config.opening_context
    context = load_consumed_target_context(
        opening_receipt_path=str(opening["opening_receipt_path"]),
        locked_target_index_path=str(opening["locked_target_index_path"]),
        artifact_root=root,
    )
    if (
        context.receipt_file_sha256 != opening["opening_receipt_file_sha256"]
        or context.receipt.get("record_sha256") != opening["opening_receipt_record_sha256"]
        or context.index_file_sha256 != opening["locked_target_index_file_sha256"]
        or context.index.get("record_sha256") != opening["locked_target_index_record_sha256"]
        or context.index.get("target_seal_id") != opening["target_seal_id"]
    ):
        raise RawTotalSensitivityError("consumed opening-1 lineage differs from configuration")
    _, split = _load_json_with_file_hash(
        config.dataset["split_manifest_path"],
        config.dataset["split_manifest_file_sha256"],
        root=root,
        role="full split manifest",
    )
    split_hash = _validate_self_hash(split, field="split_manifest_sha256", name="split manifest")
    target_seal = _mapping(split.get("target_seal"), name="target seal")
    if (
        split_hash != config.dataset["split_manifest_sha256"]
        or target_seal.get("seal_id") != opening["target_seal_id"]
        or target_seal.get("subject_ids") != list(TARGET_PARTICIPANTS)
        or split.get("trial_boundary_status") != "unrecoverable"
    ):
        raise RawTotalSensitivityError("target split/seal lineage changed")
    target = materialize_signal_definition_windows(
        raw_csv_path,
        _windows_from_manifest(split, role="full split manifest"),
        expected_source_sha256=str(config.dataset["raw_sensor_artifact_sha256"]),
        channels=RAW_TOTAL_CHANNELS,
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions=frozenset({"target_sealed"}),
        target_context=context,
        expected_split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
        expected_target_seal_id=str(opening["target_seal_id"]),
    )
    if (
        target.signals.shape[0] != config.dataset["expected_target_windows"]
        or tuple(sorted(set(target.participant_ids), key=int)) != TARGET_PARTICIPANTS
    ):
        raise RawTotalSensitivityError("raw/total target materialization changed cohort coverage")
    _validate_target_identity_against_opening(context, target)
    return context, target


def _validate_target_identity_against_opening(
    context: ConsumedTargetContext,
    target: MaterializedWindows,
) -> None:
    key = ("compact-erm", SEED_ORDER[0])
    if key not in context.result_entries:
        raise RawTotalSensitivityError("opening-1 index lacks the alignment reference")
    entry = context.result_entries[key]
    clean = load_target_clean_reference(context, model_id=key[0], seed=key[1])
    prediction_path = clean.prediction_path
    if sha256_file(prediction_path) != entry.get("array_sha256"):
        raise RawTotalSensitivityError("opening-1 alignment prediction hash changed")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != _LOCKED_PREDICTION_ARRAY_KEYS:
            raise RawTotalSensitivityError("opening-1 alignment prediction schema changed")
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
    if (
        window_ids != target.window_ids
        or participant_ids != target.participant_ids
        or not np.array_equal(labels, target.labels)
    ):
        raise RawTotalSensitivityError(
            "raw/total target identity differs from consumed opening-1 alignment"
        )


def _primary_reference_entry(
    context: ConsumedTargetContext,
    *,
    model_id: str,
    seed: int,
    root: Path,
) -> dict[str, Any]:
    reference = load_target_clean_reference(context, model_id=model_id, seed=seed)
    return {
        "model_id": model_id,
        "seed": seed,
        "record_path": reference.record_path.relative_to(root).as_posix(),
        "record_file_sha256": reference.record_file_sha256,
        "record_sha256": reference.record_sha256,
        "prediction_sha256": reference.prediction_sha256,
        "source": "consumed_locked_target_opening_1_primary_signal_definition",
    }


def _evaluate_target_runs(
    prepared: Sequence[_PreparedSourceRun],
    target: MaterializedWindows,
    context: ConsumedTargetContext,
    *,
    source_stage_lock_sha256: str,
    standardizer: ChannelStandardizer,
    config: RawTotalSensitivityConfig,
    output: Path,
    root: Path,
    code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = standardizer.transform(target.signals)
    result_entries: list[dict[str, Any]] = []
    primary_references: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in prepared:
        try:
            neural, checkpoint = reconstruct_checkpoint(run.checkpoint_path, device=device)
            if (
                canonical_json_sha256(checkpoint.get("configuration"))
                != canonical_json_sha256(asdict(run.configuration))
                or checkpoint.get("lineage", {}).get("preprocessing_config_sha256")
                != config.signal_interface["preprocessing_config_file_sha256"]
                or canonical_json_sha256(checkpoint.get("normalization"))
                != canonical_json_sha256(standardizer.to_dict())
            ):
                raise RawTotalSensitivityError("reconstructed source checkpoint lineage changed")
            logits, uncalibrated, _ = predict_model(
                neural,
                normalized,
                target.labels,
                list(target.participant_ids),
                class_names=tuple(cast(list[str], config.dataset["class_names"])),
                batch_size=run.configuration.batch_size,
                device=device,
                mixed_precision=run.configuration.mixed_precision,
            )
            calibrator_record, calibrator = load_source_temperature_calibrator_file(
                run.calibrator_path,
                expected_checkpoint_sha256=run.checkpoint_sha256,
                expected_training_configuration_sha256=canonical_json_sha256(
                    asdict(run.configuration)
                ),
                expected_split_manifest_sha256=str(config.dataset["split_manifest_sha256"]),
            )
            if sha256_file(run.calibrator_path) != run.calibrator_file_sha256:
                raise RawTotalSensitivityError("source calibrator bytes changed")
            probabilities = calibrator.probabilities(logits)
            report = classification_report(
                target.labels,
                probabilities,
                list(target.participant_ids),
                class_names=tuple(cast(list[str], config.dataset["class_names"])),
            )
            prediction_path = (
                output / "predictions" / f"{run.model.model_id}--seed-{run.seed}--target.npz"
            )
            prediction_sha = _write_prediction_new(
                prediction_path,
                model_id=run.model.model_id,
                seed=run.seed,
                cohort="target_consumed_opening_1",
                window_ids=target.window_ids,
                participant_ids=target.participant_ids,
                labels=target.labels,
                logits=logits,
                uncalibrated=uncalibrated,
                calibrated=probabilities,
            )
            record = _result_record(
                model_id=run.model.model_id,
                seed=run.seed,
                cohort="target_consumed_opening_1",
                report=report,
                prediction_path=prediction_path,
                prediction_sha256=prediction_sha,
                checkpoint_path=run.checkpoint_path,
                checkpoint_sha256=run.checkpoint_sha256,
                configuration=run.configuration,
                calibrator_path=run.calibrator_path,
                calibrator_file_sha256=run.calibrator_file_sha256,
                calibrator_record_sha256=str(calibrator_record["record_sha256"]),
                normalization=standardizer.to_dict(),
                config=config,
                root=root,
                code_commit=code_commit,
                created_at_utc=created_at_utc,
                source_stage_lock_sha256=source_stage_lock_sha256,
                opening_context=context,
            )
            record_path = output / "records" / f"{run.model.model_id}--seed-{run.seed}--target.json"
            atomic_write_json_new(record, record_path, allowed_root=root)
            result_entries.append(
                _index_entry(
                    model_id=run.model.model_id,
                    seed=run.seed,
                    record_path=record_path,
                    record=record,
                    root=root,
                )
            )
            primary_references.append(
                _primary_reference_entry(
                    context, model_id=run.model.model_id, seed=run.seed, root=root
                )
            )
        except Exception as exc:
            failures.append(
                _failure_record(
                    output=output,
                    root=root,
                    config=config,
                    stage="consumed_target_evaluation",
                    model_id=run.model.model_id,
                    seed=run.seed,
                    exc=exc,
                    code_commit=code_commit,
                    created_at_utc=created_at_utc,
                    target_accessed=True,
                )
            )
        finally:
            if "neural" in locals():
                del neural
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
    return result_entries, primary_references, failures


def run_raw_total_acceleration_sensitivity(
    *,
    config_path: str | Path,
    repository_root: str | Path,
    raw_csv_path: str | Path,
    expected_code_commit: str,
    created_at_utc: str,
    device: torch.device,
) -> dict[str, Any]:
    """Run source locking then consumed-target evaluation, preserving every failure."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RawTotalSensitivityError(
            "CUDA is required before configuration, raw data, or target context is opened"
        )
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    config = load_raw_total_sensitivity_config(config_path)
    root = Path(repository_root).resolve(strict=True)
    try:
        config.path.relative_to(root)
    except ValueError as exc:
        raise RawTotalSensitivityError("experiment config escapes repository root") from exc
    current_commit, status = _git_state(root)
    if current_commit != expected_code_commit:
        raise RawTotalSensitivityError("current Git commit differs from expected execution commit")
    if status:
        raise RawTotalSensitivityError(f"sensitivity execution requires a clean worktree: {status}")
    output = _confined_path(
        str(config.outputs["output_directory"]),
        root=root,
        role="sensitivity output directory",
        must_exist=False,
    )
    if os.path.lexists(output):
        raise FileExistsError(f"refusing to reuse sensitivity output directory: {output}")
    output.mkdir(parents=True, exist_ok=False)
    failures: list[dict[str, Any]] = []
    source_entries: list[dict[str, Any]] = []
    normalization_entry: dict[str, Any] | None = None
    source_lock_entry: dict[str, Any] | None = None
    raw_path = Path(raw_csv_path)
    try:
        source_records, class_names = _validate_source_lineage(config, root=root)
        source_batch = materialize_signal_definition_windows(
            raw_path,
            source_records,
            expected_source_sha256=str(config.dataset["raw_sensor_artifact_sha256"]),
            channels=RAW_TOTAL_CHANNELS,
            ontology_track="functional_core",
            class_names=class_names,
            allowed_partitions=frozenset({"source_train", "source_validation"}),
        )
        source = _source_arrays(source_batch, config)
        normalization_entry = _normalization_record(
            source.standardizer,
            config=config,
            output=output,
            root=root,
            created_at_utc=timestamp,
        )
    except Exception as exc:
        failures.append(
            _failure_record(
                output=output,
                root=root,
                config=config,
                stage="source_lineage_or_materialization",
                model_id=None,
                seed=None,
                exc=exc,
                code_commit=current_commit,
                created_at_utc=timestamp,
                target_accessed=False,
            )
        )
        return _write_index(
            output=output,
            root=root,
            config=config,
            status="source_materialization_failed_target_not_accessed",
            code_commit=current_commit,
            created_at_utc=timestamp,
            normalization_entry=None,
            source_stage_lock_entry=None,
            source_results=[],
            target_results=[],
            primary_references=[],
            failures=failures,
            target_accessed=False,
            context=None,
        )

    prepared, source_entries, source_failures = _train_source_runs(
        source,
        config=config,
        output=output,
        root=root,
        code_commit=current_commit,
        created_at_utc=timestamp,
        device=device,
    )
    failures.extend(source_failures)
    if failures or len(prepared) != len(MODEL_IDS) * len(SEED_ORDER):
        return _write_index(
            output=output,
            root=root,
            config=config,
            status="source_stage_failed_preserved_target_not_accessed",
            code_commit=current_commit,
            created_at_utc=timestamp,
            normalization_entry=normalization_entry,
            source_stage_lock_entry=None,
            source_results=source_entries,
            target_results=[],
            primary_references=[],
            failures=failures,
            target_accessed=False,
            context=None,
        )
    source_lock_entry = _source_stage_lock(
        prepared=prepared,
        source_results=source_entries,
        normalization_entry=cast(Mapping[str, Any], normalization_entry),
        config=config,
        output=output,
        root=root,
        code_commit=current_commit,
        created_at_utc=timestamp,
    )

    try:
        context, target = _load_target_stage(
            config, root=root, raw_csv_path=raw_path, class_names=class_names
        )
    except Exception as exc:
        failures.append(
            _failure_record(
                output=output,
                root=root,
                config=config,
                stage="consumed_target_lineage_or_materialization",
                model_id=None,
                seed=None,
                exc=exc,
                code_commit=current_commit,
                created_at_utc=timestamp,
                target_accessed=True,
            )
        )
        return _write_index(
            output=output,
            root=root,
            config=config,
            status="target_materialization_failed_preserved",
            code_commit=current_commit,
            created_at_utc=timestamp,
            normalization_entry=normalization_entry,
            source_stage_lock_entry=source_lock_entry,
            source_results=source_entries,
            target_results=[],
            primary_references=[],
            failures=failures,
            target_accessed=True,
            context=None,
        )
    target_entries, primary_references, target_failures = _evaluate_target_runs(
        prepared,
        target,
        context,
        source_stage_lock_sha256=str(source_lock_entry["record_sha256"]),
        standardizer=source.standardizer,
        config=config,
        output=output,
        root=root,
        code_commit=current_commit,
        created_at_utc=timestamp,
        device=device,
    )
    failures.extend(target_failures)
    complete = (
        not failures
        and len(source_entries) == len(MODEL_IDS) * len(SEED_ORDER)
        and len(target_entries) == len(MODEL_IDS) * len(SEED_ORDER)
        and len(primary_references) == len(MODEL_IDS) * len(SEED_ORDER)
    )
    return _write_index(
        output=output,
        root=root,
        config=config,
        status="complete_create_only" if complete else "target_stage_failed_preserved_partial",
        code_commit=current_commit,
        created_at_utc=timestamp,
        normalization_entry=normalization_entry,
        source_stage_lock_entry=source_lock_entry,
        source_results=source_entries,
        target_results=target_entries,
        primary_references=primary_references,
        failures=failures,
        target_accessed=True,
        context=context,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_raw_total_acceleration_sensitivity(
        config_path=args.config,
        repository_root=args.repository_root,
        raw_csv_path=args.raw_csv,
        expected_code_commit=args.code_commit,
        created_at_utc=args.created_at_utc,
        device=torch.device("cuda"),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete_create_only" else 2


if __name__ == "__main__":
    raise SystemExit(main())
