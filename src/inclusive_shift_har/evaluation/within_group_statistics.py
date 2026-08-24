"""Validate and aggregate disabled-cohort within-group cross-subject cells."""

from __future__ import annotations

import argparse
import csv
import io
import itertools
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.protocols.within_group import (
    FUNCTIONAL_CORE_CLASS_ORDER,
    REQUIRED_SEEDS,
    SUPPORTED_MODEL_IDS,
    TARGET_PARTICIPANTS,
    WITHIN_GROUP_EVIDENCE_STATUS,
    WITHIN_GROUP_PROTOCOL_ID,
    WITHIN_GROUP_SCHEMA_VERSION,
)
from inclusive_shift_har.training.engine import build_model, training_config_from_dict

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
_UTC_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")
_PREDICTION_KEYS = {
    "evidence_status",
    "window_ids",
    "participant_ids",
    "true_labels",
    "logits",
    "uncalibrated_probabilities",
    "calibrated_probabilities",
    "predicted_labels",
}


class WithinGroupStatisticsError(RuntimeError):
    """Raised when cell evidence or aggregate statistics fail closed."""


@dataclass(frozen=True, slots=True)
class CellKey:
    fold_id: str
    model_id: str
    seed: int

    @property
    def cell_id(self) -> str:
        return f"{self.fold_id}__{self.model_id}__seed-{self.seed}"


@dataclass(frozen=True, slots=True)
class ValidatedCell:
    key: CellKey
    record_path: Path
    record_file_sha256: str
    record_sha256: str
    execution_code_commit: str
    window_ids: tuple[str, ...]
    participant_ids: tuple[str, ...]
    labels: NDArray[np.int64]
    calibrated_probabilities: NDArray[np.float64]
    participant_macro_f1: Mapping[str, float]


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise WithinGroupStatisticsError(f"{name} must be an object")
    return value


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    value = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(value, str) or value != canonical_json_sha256(body):
        raise WithinGroupStatisticsError(f"{name} self-hash does not validate")
    return value


def _require_repository_head(root: Path, *, expected_commit: str) -> str:
    """Bind aggregation code lineage to the repository's actual HEAD."""

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise WithinGroupStatisticsError("could not resolve repository Git HEAD") from exc
    observed = completed.stdout.strip().casefold()
    if _COMMIT_RE.fullmatch(observed) is None:
        raise WithinGroupStatisticsError("repository Git HEAD is not a full object ID")
    if observed != expected_commit:
        raise WithinGroupStatisticsError("aggregation code commit differs from repository Git HEAD")
    return observed


def _resolve_file(value: Any, *, root: Path, name: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise WithinGroupStatisticsError(f"{name} must be a portable relative path")
    candidate = root / value
    if candidate.is_symlink():
        raise WithinGroupStatisticsError(f"{name} may not be a symlink")
    path = candidate.resolve(strict=True)
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise WithinGroupStatisticsError(f"{name} escapes artifact root") from exc
    if path.is_symlink() or not path.is_file():
        raise WithinGroupStatisticsError(f"{name} must be a regular file")
    return path


def _load_plan(path: Path) -> tuple[dict[str, Any], str]:
    value = load_json_strict(path)
    plan = dict(_mapping(value, name="within-group manifest"))
    plan_sha256 = _self_hash(plan, field="manifest_sha256", name="within-group manifest")
    required = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "status": "ready_no_cell_run",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "model_order": list(SUPPORTED_MODEL_IDS),
        "seed_order": list(REQUIRED_SEEDS),
        "expected_cell_count": 75,
    }
    mismatches = [key for key, expected in required.items() if plan.get(key) != expected]
    if mismatches:
        raise WithinGroupStatisticsError(f"manifest contract differs: {mismatches}")
    return plan, plan_sha256


def expected_cells(plan: Mapping[str, Any]) -> tuple[CellKey, ...]:
    folds = plan.get("folds")
    models = plan.get("model_order")
    seeds = plan.get("seed_order")
    if not isinstance(folds, list) or not isinstance(models, list) or not isinstance(seeds, list):
        raise WithinGroupStatisticsError("manifest lacks fold/model/seed inventory")
    fold_ids: list[str] = []
    for value in folds:
        fold = _mapping(value, name="fold")
        fold_id = fold.get("fold_id")
        if not isinstance(fold_id, str) or not fold_id or fold_id in fold_ids:
            raise WithinGroupStatisticsError("manifest fold identities are invalid")
        fold_ids.append(fold_id)
    cells = tuple(
        CellKey(fold_id=fold_id, model_id=str(model), seed=int(seed))
        for fold_id in fold_ids
        for model in models
        for seed in seeds
    )
    if len(cells) != plan.get("expected_cell_count") or len(set(cells)) != len(cells):
        raise WithinGroupStatisticsError("manifest cell inventory is incomplete")
    entries = plan.get("models")
    if not isinstance(entries, list):
        raise WithinGroupStatisticsError("manifest lacks expanded model entries")
    entry_keys = [
        (
            _mapping(value, name="model entry").get("model_id"),
            _mapping(value, name="model entry").get("seed"),
        )
        for value in entries
    ]
    expected_model_keys = {
        (model_id, seed) for model_id in SUPPORTED_MODEL_IDS for seed in REQUIRED_SEEDS
    }
    if (
        len(entry_keys) != len(expected_model_keys)
        or len(set(entry_keys)) != len(entry_keys)
        or set(entry_keys) != expected_model_keys
    ):
        raise WithinGroupStatisticsError("expanded model entry inventory changed")
    return cells


def _fold_by_id(plan: Mapping[str, Any], fold_id: str) -> Mapping[str, Any]:
    folds = cast(list[Any], plan["folds"])
    return next(
        _mapping(value, name="fold")
        for value in folds
        if isinstance(value, Mapping) and value.get("fold_id") == fold_id
    )


def _model_by_key(plan: Mapping[str, Any], model_id: str, seed: int) -> Mapping[str, Any]:
    models = plan.get("models")
    if not isinstance(models, list):
        raise WithinGroupStatisticsError("manifest lacks model cells")
    try:
        return next(
            _mapping(value, name="model entry")
            for value in models
            if isinstance(value, Mapping)
            and value.get("model_id") == model_id
            and value.get("seed") == seed
        )
    except StopIteration as exc:
        raise WithinGroupStatisticsError("manifest lacks expected model/seed") from exc


def _expected_evaluation_rows(
    split: Mapping[str, Any], participants: set[str]
) -> tuple[tuple[str, ...], tuple[str, ...], NDArray[np.int64]]:
    windows = split.get("windows")
    if not isinstance(windows, list):
        raise WithinGroupStatisticsError("parent split lacks windows")
    records = [
        value
        for value in windows
        if isinstance(value, Mapping)
        and value.get("partition") == "target_sealed"
        and value.get("subject_id") in participants
        and isinstance(value.get("canonical_labels"), Mapping)
        and _mapping(value["canonical_labels"], name="window labels").get("functional_core")
        in FUNCTIONAL_CORE_CLASS_ORDER
    ]
    records.sort(key=lambda item: int(cast(int, item["start_row_inclusive"])))
    class_to_index = {name: index for index, name in enumerate(FUNCTIONAL_CORE_CLASS_ORDER)}
    return (
        tuple(str(value["window_id"]) for value in records),
        tuple(str(value["subject_id"]) for value in records),
        np.asarray(
            [
                class_to_index[
                    str(
                        _mapping(value["canonical_labels"], name="window labels")["functional_core"]
                    )
                ]
                for value in records
            ],
            dtype=np.int64,
        ),
    )


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return np.asarray(exponent / exponent.sum(axis=1, keepdims=True), dtype=np.float64)


def _participant_values(report: Mapping[str, Any]) -> dict[str, float]:
    values = report.get("participants")
    if not isinstance(values, list):
        raise WithinGroupStatisticsError("report lacks participant rows")
    result: dict[str, float] = {}
    for value in values:
        row = _mapping(value, name="participant report row")
        participant = row.get("participant_id")
        metric = row.get("macro_f1")
        if (
            not isinstance(participant, str)
            or participant in result
            or isinstance(metric, bool)
            or not isinstance(metric, (int, float))
            or not np.isfinite(metric)
            or not 0 <= metric <= 1
        ):
            raise WithinGroupStatisticsError("participant report row is invalid")
        result[participant] = float(metric)
    return result


def validate_cell(
    *,
    key: CellKey,
    plan: Mapping[str, Any],
    plan_sha256: str,
    split: Mapping[str, Any],
    result_root: Path,
    artifact_root: Path,
    primary_cache_record_file_sha256: str,
) -> ValidatedCell:
    directory = result_root / key.fold_id / key.model_id / f"seed-{key.seed}"
    if not directory.is_dir() or directory.is_symlink():
        raise WithinGroupStatisticsError(f"missing cell directory: {key.cell_id}")
    failure = directory / "failure.json"
    if failure.exists():
        failure_record = _mapping(load_json_strict(failure), name="cell failure")
        _self_hash(failure_record, field="record_sha256", name="cell failure")
        raise WithinGroupStatisticsError(f"cell has preserved failure: {key.cell_id}")
    expected_names = {"selected.pt", "calibrator.json", "predictions.npz", "result.json"}
    observed_names = {path.name for path in directory.iterdir()}
    if observed_names != expected_names:
        raise WithinGroupStatisticsError(f"cell artifact set differs: {key.cell_id}")
    if any(not path.is_file() or path.is_symlink() for path in directory.iterdir()):
        raise WithinGroupStatisticsError(f"cell contains an unsafe artifact: {key.cell_id}")
    record_path = directory / "result.json"
    record_value = load_json_strict(record_path)
    record = _mapping(record_value, name="cell result")
    record_sha256 = _self_hash(record, field="record_sha256", name="cell result")
    split_reference = _mapping(plan.get("split_manifest"), name="manifest split")
    opening = _mapping(plan.get("opening_1"), name="manifest opening")
    required = {
        "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
        "record_kind": "within_group_fold_model_seed_result",
        "status": "complete_create_only",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "scientific_role": "secondary_descriptive_not_locked_confirmatory",
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "manifest_sha256": plan_sha256,
        "split_manifest_sha256": split_reference["record_sha256"],
        "source_artifact_sha256": plan["source_artifact_sha256"],
        "opening_receipt_record_sha256": opening["receipt_record_sha256"],
        "opening_index_record_sha256": opening["index_record_sha256"],
        "class_names": list(FUNCTIONAL_CORE_CLASS_ORDER),
        "participant_assignment_before_array_loading": True,
        "normalization_fit_on_training_only": True,
        "checkpoint_frozen_before_validation_loading": True,
        "calibration_fit_on_validation_only": True,
        "checkpoint_and_calibrator_frozen_before_evaluation_loading": True,
        "evaluation_used_for_training_selection_or_calibration": False,
        "disability_or_assistive_device_metadata_used_as_model_input": False,
        "raw_dataset_file_accessed": False,
        "whole_target_cache_array_accessed": False,
        "opening_1_prediction_array_accessed": False,
        "new_target_opening_created": False,
        "cpu_neural_fallback_used": False,
    }
    mismatches = [name for name, expected in required.items() if record.get(name) != expected]
    if mismatches:
        raise WithinGroupStatisticsError(f"cell contract differs {key.cell_id}: {mismatches}")
    execution_code_commit = record.get("code_commit")
    if (
        not isinstance(execution_code_commit, str)
        or _COMMIT_RE.fullmatch(execution_code_commit) is None
    ):
        raise WithinGroupStatisticsError("cell execution code commit is invalid")
    model_entry = _model_by_key(plan, key.model_id, key.seed)
    configuration = _mapping(record.get("configuration"), name="result configuration")
    if configuration != _mapping(
        model_entry.get("source_locked_training_configuration"),
        name="manifest configuration",
    ) or record.get("configuration_sha256") != model_entry.get(
        "source_locked_training_configuration_sha256"
    ):
        raise WithinGroupStatisticsError("cell training configuration changed")
    fold = _fold_by_id(plan, key.fold_id)
    fold_roles = _mapping(record.get("fold_roles"), name="result fold roles")
    for role in ("training", "validation", "evaluation"):
        declared = _mapping(fold.get(role), name=f"manifest {role}")
        observed = _mapping(fold_roles.get(role), name=f"result {role}")
        for field in (
            "participant_ids",
            "window_count",
            "window_ids_sha256",
            "ordered_window_ids_sha256",
        ):
            if observed.get(field) != declared.get(field):
                raise WithinGroupStatisticsError(f"cell {role} lineage changed")
    cache = _mapping(record.get("primary_cache"), name="result cache")
    if cache.get("record_file_sha256") != primary_cache_record_file_sha256:
        raise WithinGroupStatisticsError("cell cache file pin differs")
    cache_path = _resolve_file(cache.get("record_path"), root=artifact_root, name="cache record")
    if sha256_file(cache_path) != primary_cache_record_file_sha256:
        raise WithinGroupStatisticsError("cell cache record bytes changed")
    cache_record = _mapping(load_json_strict(cache_path), name="primary cache record")
    cache_record_sha256 = _self_hash(
        cache_record, field="record_sha256", name="primary cache record"
    )
    if cache.get("record_sha256") != cache_record_sha256:
        raise WithinGroupStatisticsError("cell cache self-hash lineage changed")

    checkpoint = _mapping(record.get("checkpoint"), name="result checkpoint")
    checkpoint_path = _resolve_file(
        checkpoint.get("path"), root=artifact_root, name="cell checkpoint"
    )
    if checkpoint_path != directory / "selected.pt" or sha256_file(checkpoint_path) != (
        checkpoint.get("sha256")
    ):
        raise WithinGroupStatisticsError("cell checkpoint path/hash changed")
    payload_value = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload = _mapping(payload_value, name="checkpoint payload")
    checkpoint_required = {
        "record_kind": "within_group_fixed_epoch_checkpoint",
        "status": "frozen_before_validation_or_evaluation_loading",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "checkpoint_selection_rule": "fixed_last_epoch",
        "manifest_sha256": plan_sha256,
        "split_manifest_sha256": split_reference["record_sha256"],
        "primary_cache_record_file_sha256": primary_cache_record_file_sha256,
        "validation_arrays_parsed_before_checkpoint_freeze": False,
        "evaluation_arrays_parsed_before_checkpoint_freeze": False,
    }
    if any(payload.get(name) != value for name, value in checkpoint_required.items()):
        raise WithinGroupStatisticsError("checkpoint embedded lineage changed")
    if any(
        name not in payload
        for name in (
            "model_state",
            "optimizer_state",
            "scheduler_state",
            "scaler_state",
            "rng_states",
        )
    ):
        raise WithinGroupStatisticsError("checkpoint reconstruction state is incomplete")
    checkpoint_configuration = _mapping(
        payload.get("configuration"), name="checkpoint configuration"
    )
    if (
        canonical_json_sha256(checkpoint_configuration) != record.get("configuration_sha256")
        or payload.get("configuration_sha256") != record.get("configuration_sha256")
        or payload.get("label_schema") != list(FUNCTIONAL_CORE_CLASS_ORDER)
        or payload.get("epoch") != checkpoint.get("selected_epoch")
        or payload.get("selected_epoch") != checkpoint.get("selected_epoch")
    ):
        raise WithinGroupStatisticsError("checkpoint configuration/schema changed")
    reconstruction_config = training_config_from_dict(dict(checkpoint_configuration))
    reconstructed = build_model(reconstruction_config)
    model_state = _mapping(payload.get("model_state"), name="checkpoint model state")
    try:
        reconstructed.load_state_dict(dict(model_state), strict=True)
    except RuntimeError as exc:
        raise WithinGroupStatisticsError(
            "checkpoint model state does not reconstruct strictly"
        ) from exc
    normalization = _mapping(payload.get("normalization"), name="checkpoint normalization")
    train_participants = _mapping(fold.get("training"), name="fold training").get("participant_ids")
    if normalization.get("training_participants") != sorted(cast(list[str], train_participants)):
        raise WithinGroupStatisticsError("checkpoint normalization participants changed")

    calibrator_reference = _mapping(record.get("calibrator"), name="result calibrator")
    calibrator_path = _resolve_file(
        calibrator_reference.get("path"), root=artifact_root, name="cell calibrator"
    )
    if calibrator_path != directory / "calibrator.json" or sha256_file(calibrator_path) != (
        calibrator_reference.get("file_sha256")
    ):
        raise WithinGroupStatisticsError("cell calibrator path/hash changed")
    calibrator = _mapping(load_json_strict(calibrator_path), name="calibrator record")
    calibrator_sha256 = _self_hash(calibrator, field="record_sha256", name="calibrator")
    validation_participants = _mapping(fold.get("validation"), name="fold validation").get(
        "participant_ids"
    )
    calibrator_required = {
        "record_kind": "within_group_validation_temperature_calibrator",
        "status": "frozen_before_evaluation_loading",
        "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
        "fold_id": key.fold_id,
        "model_id": key.model_id,
        "seed": key.seed,
        "fit_role": "validation",
        "fit_participants": validation_participants,
        "checkpoint_sha256": checkpoint["sha256"],
        "evaluation_arrays_parsed_before_calibrator_freeze": False,
        "checkpoint_selection_used_validation": False,
        "target_validation_used_for_calibration": True,
    }
    if calibrator_sha256 != calibrator_reference.get("record_sha256") or any(
        calibrator.get(name) != value for name, value in calibrator_required.items()
    ):
        raise WithinGroupStatisticsError("calibrator embedded lineage changed")
    temperature = calibrator.get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise WithinGroupStatisticsError("calibrator temperature is invalid")

    prediction_reference = _mapping(record.get("prediction_array"), name="prediction ref")
    prediction_path = _resolve_file(
        prediction_reference.get("path"), root=artifact_root, name="prediction array"
    )
    if prediction_path != directory / "predictions.npz" or sha256_file(prediction_path) != (
        prediction_reference.get("sha256")
    ):
        raise WithinGroupStatisticsError("prediction path/hash changed")
    with np.load(prediction_path, allow_pickle=False) as arrays:
        if set(arrays.files) != _PREDICTION_KEYS:
            raise WithinGroupStatisticsError("prediction array keys changed")
        evidence = tuple(str(value) for value in arrays["evidence_status"].tolist())
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        raw_labels = np.asarray(arrays["true_labels"])
        raw_logits = np.asarray(arrays["logits"])
        raw_uncalibrated = np.asarray(arrays["uncalibrated_probabilities"])
        raw_calibrated = np.asarray(arrays["calibrated_probabilities"])
        raw_predictions = np.asarray(arrays["predicted_labels"])
    labels = np.asarray(raw_labels, dtype=np.int64)
    logits = np.asarray(raw_logits, dtype=np.float64)
    uncalibrated = np.asarray(raw_uncalibrated, dtype=np.float64)
    calibrated = np.asarray(raw_calibrated, dtype=np.float64)
    predicted = np.asarray(raw_predictions, dtype=np.int64)
    evaluation_participants = set(
        cast(
            list[str],
            _mapping(fold.get("evaluation"), name="fold evaluation")["participant_ids"],
        )
    )
    expected_ids, expected_participants, expected_labels = _expected_evaluation_rows(
        split, evaluation_participants
    )
    count = len(expected_ids)
    if (
        evidence != (WITHIN_GROUP_EVIDENCE_STATUS,)
        or raw_labels.dtype != np.dtype(np.int64)
        or raw_logits.dtype != np.dtype(np.float64)
        or raw_uncalibrated.dtype != np.dtype(np.float64)
        or raw_calibrated.dtype != np.dtype(np.float64)
        or raw_predictions.dtype != np.dtype(np.int64)
        or window_ids != expected_ids
        or participant_ids != expected_participants
        or not np.array_equal(labels, expected_labels)
        or logits.shape != (count, len(FUNCTIONAL_CORE_CLASS_ORDER))
        or calibrated.shape != logits.shape
        or uncalibrated.shape != logits.shape
        or predicted.shape != (count,)
        or not np.isfinite(logits).all()
        or not np.allclose(uncalibrated, _softmax(logits), atol=1e-7, rtol=1e-7)
        or not np.allclose(calibrated, _softmax(logits / float(temperature)), atol=1e-7, rtol=1e-7)
        or not np.array_equal(predicted, calibrated.argmax(axis=1))
    ):
        raise WithinGroupStatisticsError("prediction numerical/alignment contract changed")
    report = classification_report(
        labels,
        calibrated,
        participant_ids,
        class_names=FUNCTIONAL_CORE_CLASS_ORDER,
    )
    if canonical_json_sha256(report) != canonical_json_sha256(
        _mapping(record.get("participant_level_report"), name="stored report")
    ):
        raise WithinGroupStatisticsError("stored participant report does not recompute")
    participant_values = _participant_values(report)
    if set(participant_values) != evaluation_participants:
        raise WithinGroupStatisticsError("cell report participant set changed")
    return ValidatedCell(
        key=key,
        record_path=record_path,
        record_file_sha256=sha256_file(record_path),
        record_sha256=record_sha256,
        execution_code_commit=execution_code_commit,
        window_ids=window_ids,
        participant_ids=participant_ids,
        labels=labels,
        calibrated_probabilities=calibrated,
        participant_macro_f1=participant_values,
    )


def participant_cluster_bootstrap_ci(
    values: NDArray[np.float64], *, replicates: int, seed: int
) -> list[float]:
    """Bootstrap participants after averaging the five seed replicates."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise WithinGroupStatisticsError("bootstrap requires [seed,participant] values")
    if replicates < 1000 or seed < 0 or not np.isfinite(matrix).all():
        raise WithinGroupStatisticsError("bootstrap configuration/data are invalid")
    participant_values = matrix.mean(axis=0)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, participant_values.size, size=(replicates, participant_values.size))
    statistics = participant_values[indices].mean(axis=1)
    return [
        float(np.quantile(statistics, 0.025, method="linear")),
        float(np.quantile(statistics, 0.975, method="linear")),
    ]


def hierarchical_bootstrap_ci(
    values: NDArray[np.float64], *, replicates: int, seed: int
) -> list[float]:
    """Resample both participants and training seeds for the mean endpoint."""

    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2 or matrix.shape[1] < 2:
        raise WithinGroupStatisticsError("hierarchical bootstrap needs a matrix")
    rng = np.random.default_rng(seed)
    statistics = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        seed_indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        participant_indices = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        statistics[index] = matrix[np.ix_(seed_indices, participant_indices)].mean()
    return [
        float(np.quantile(statistics, 0.025, method="linear")),
        float(np.quantile(statistics, 0.975, method="linear")),
    ]


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise WithinGroupStatisticsError("paired differences are invalid")
    observed = abs(float(values.mean()))
    exceed = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=values.size):
        statistic = abs(float(np.mean(values * np.asarray(signs, dtype=np.float64))))
        exceed += int(statistic >= observed - 1e-15)
        total += 1
    return exceed / total


def _holm_adjust(rows: list[dict[str, Any]]) -> None:
    ordered = sorted(range(len(rows)), key=lambda index: float(rows[index]["p_value_raw"]))
    running = 0.0
    count = len(rows)
    for rank, index in enumerate(ordered):
        adjusted = min(1.0, (count - rank) * float(rows[index]["p_value_raw"]))
        running = max(running, adjusted)
        rows[index]["p_value_holm"] = running


def _csv_bytes(rows: Sequence[Mapping[str, Any]], fields: Sequence[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_bytes_new(path: Path, payload: bytes) -> str:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    return sha256_file(path)


def _validate_result_tree(result_root: Path, cells: Sequence[CellKey]) -> None:
    """Reject unexpected fold/model/seed directories before aggregation."""

    expected: dict[str, dict[str, set[str]]] = {}
    for cell in cells:
        expected.setdefault(cell.fold_id, {}).setdefault(cell.model_id, set()).add(
            f"seed-{cell.seed}"
        )
    observed_folds = {path.name for path in result_root.iterdir()}
    if observed_folds != set(expected):
        raise WithinGroupStatisticsError("result-root fold directory set differs")
    for fold_id, models in expected.items():
        fold_path = result_root / fold_id
        if not fold_path.is_dir() or fold_path.is_symlink():
            raise WithinGroupStatisticsError(f"result fold path is unsafe: {fold_id}")
        observed_models = {path.name for path in fold_path.iterdir()}
        if observed_models != set(models):
            raise WithinGroupStatisticsError(f"result model set differs for {fold_id}")
        for model_id, seeds in models.items():
            model_path = fold_path / model_id
            if not model_path.is_dir() or model_path.is_symlink():
                raise WithinGroupStatisticsError("result model path is unsafe")
            observed_seeds = {path.name for path in model_path.iterdir()}
            if observed_seeds != seeds:
                raise WithinGroupStatisticsError(
                    f"result seed set differs for {fold_id}/{model_id}"
                )


def aggregate_within_group_statistics(
    *,
    manifest_path: Path,
    split_manifest_path: Path,
    primary_cache_record_path: Path,
    primary_cache_record_file_sha256: str,
    result_root: Path,
    artifact_root: Path,
    output_directory: Path,
    output_root: Path,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Validate the exact 75-cell matrix and publish participant-level statistics."""

    root = artifact_root.resolve(strict=True)
    results = result_root.resolve(strict=True)
    allowed_output = output_root.resolve(strict=True)
    for path, name in ((results, "result root"), (allowed_output, "output root")):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise WithinGroupStatisticsError(f"{name} escapes artifact root") from exc
    commit = code_commit.casefold()
    if _COMMIT_RE.fullmatch(commit) is None or _UTC_RE.fullmatch(created_at_utc) is None:
        raise WithinGroupStatisticsError("aggregate requires full commit and UTC timestamp")
    _require_repository_head(root, expected_commit=commit)
    manifest_file = manifest_path.resolve(strict=True)
    split_file = split_manifest_path.resolve(strict=True)
    cache_file = primary_cache_record_path.resolve(strict=True)
    for path, name in (
        (manifest_file, "manifest"),
        (split_file, "split"),
        (cache_file, "cache record"),
    ):
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise WithinGroupStatisticsError(f"{name} escapes artifact root") from exc
    if sha256_file(cache_file) != primary_cache_record_file_sha256:
        raise WithinGroupStatisticsError("primary cache external file pin changed")
    plan, plan_sha256 = _load_plan(manifest_file)
    split_value = load_json_strict(split_file)
    split = _mapping(split_value, name="parent split")
    split_hash = _self_hash(split, field="split_manifest_sha256", name="parent split")
    split_reference = _mapping(plan.get("split_manifest"), name="manifest split")
    if split_hash != split_reference.get("record_sha256") or sha256_file(
        split_file
    ) != split_reference.get("file_sha256"):
        raise WithinGroupStatisticsError("parent split differs from manifest")
    destination = (allowed_output / output_directory).resolve(strict=False)
    try:
        destination.relative_to(allowed_output)
    except ValueError as exc:
        raise WithinGroupStatisticsError("aggregate destination escapes output root") from exc
    if os.path.lexists(destination):
        raise FileExistsError(f"refusing to overwrite aggregate directory: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    execution_code_commit: str | None = None
    try:
        expected = expected_cells(plan)
        _validate_result_tree(results, expected)
        cells = [
            validate_cell(
                key=key,
                plan=plan,
                plan_sha256=plan_sha256,
                split=split,
                result_root=results,
                artifact_root=root,
                primary_cache_record_file_sha256=primary_cache_record_file_sha256,
            )
            for key in expected
        ]
        execution_commits = {cell.execution_code_commit for cell in cells}
        if len(execution_commits) != 1:
            raise WithinGroupStatisticsError("cell execution commits are mixed")
        execution_code_commit = next(iter(execution_commits))
        participants = sorted(TARGET_PARTICIPANTS, key=int)
        policy = _mapping(plan.get("statistics_policy"), name="statistics policy")
        replicates = int(cast(int, policy["bootstrap_replicates"]))
        bootstrap_seed = int(cast(int, policy["bootstrap_seed"]))
        summaries: list[dict[str, Any]] = []
        participant_rows: list[dict[str, Any]] = []
        matrices: dict[str, NDArray[np.float64]] = {}
        for model_index, model_id in enumerate(SUPPORTED_MODEL_IDS):
            matrix = np.empty((len(REQUIRED_SEEDS), len(participants)), dtype=np.float64)
            seed_reports: list[dict[str, Any]] = []
            for seed_index, seed in enumerate(REQUIRED_SEEDS):
                selected = [
                    cell
                    for cell in cells
                    if cell.key.model_id == model_id and cell.key.seed == seed
                ]
                if len(selected) != 5:
                    raise WithinGroupStatisticsError("model/seed fold coverage is incomplete")
                participant_values: dict[str, float] = {}
                for cell in selected:
                    overlap = set(participant_values) & set(cell.participant_macro_f1)
                    if overlap:
                        raise WithinGroupStatisticsError("participant repeats across outer folds")
                    participant_values.update(cell.participant_macro_f1)
                if set(participant_values) != TARGET_PARTICIPANTS:
                    raise WithinGroupStatisticsError(
                        "model/seed lacks complete participant coverage"
                    )
                matrix[seed_index] = np.asarray(
                    [participant_values[participant] for participant in participants],
                    dtype=np.float64,
                )
                labels = np.concatenate([cell.labels for cell in selected])
                probabilities = np.concatenate(
                    [cell.calibrated_probabilities for cell in selected], axis=0
                )
                pooled_participants = tuple(
                    participant for cell in selected for participant in cell.participant_ids
                )
                pooled = classification_report(
                    labels,
                    probabilities,
                    pooled_participants,
                    class_names=FUNCTIONAL_CORE_CLASS_ORDER,
                )
                seed_reports.append(
                    {
                        "seed": seed,
                        "participant_level_report": pooled,
                    }
                )
            matrices[model_id] = matrix
            averaged = matrix.mean(axis=0)
            for participant_index, participant in enumerate(participants):
                participant_rows.append(
                    {
                        "model_id": model_id,
                        "participant_id": participant,
                        "seed_mean_macro_f1": float(averaged[participant_index]),
                        "seed_standard_deviation_macro_f1": float(
                            matrix[:, participant_index].std(ddof=1)
                        ),
                        **{
                            f"seed_{seed}_macro_f1": float(matrix[seed_index, participant_index])
                            for seed_index, seed in enumerate(REQUIRED_SEEDS)
                        },
                    }
                )
            summaries.append(
                {
                    "model_id": model_id,
                    "seed_count": len(REQUIRED_SEEDS),
                    "participant_count": len(participants),
                    "mean_participant_macro_f1": float(averaged.mean()),
                    "worst_participant_macro_f1": float(averaged.min()),
                    "lower_decile_participant_macro_f1": float(
                        np.quantile(averaged, 0.1, method="linear")
                    ),
                    "participant_cluster_bootstrap_95_ci": participant_cluster_bootstrap_ci(
                        matrix,
                        replicates=replicates,
                        seed=bootstrap_seed + model_index * 2,
                    ),
                    "hierarchical_bootstrap_95_ci": hierarchical_bootstrap_ci(
                        matrix,
                        replicates=replicates,
                        seed=bootstrap_seed + model_index * 2 + 1,
                    ),
                    "per_seed_reports": seed_reports,
                }
            )
        reference_id = str(policy["paired_reference_model_id"])
        reference = matrices[reference_id].mean(axis=0)
        comparisons: list[dict[str, Any]] = []
        for model_id in SUPPORTED_MODEL_IDS:
            if model_id == reference_id:
                continue
            differences = matrices[model_id].mean(axis=0) - reference
            standard_deviation = float(differences.std(ddof=1))
            comparisons.append(
                {
                    "reference_model_id": reference_id,
                    "candidate_model_id": model_id,
                    "participant_count": len(participants),
                    "mean_paired_difference": float(differences.mean()),
                    "median_paired_difference": float(np.median(differences)),
                    "paired_cohens_dz": (
                        None
                        if np.isclose(standard_deviation, 0.0)
                        else float(differences.mean() / standard_deviation)
                    ),
                    "p_value_raw": exact_sign_flip_pvalue(differences),
                    "p_value_holm": None,
                    "test": "exact_two_sided_participant_level_sign_flip",
                }
            )
        _holm_adjust(comparisons)
        participant_fields = [
            "model_id",
            "participant_id",
            "seed_mean_macro_f1",
            "seed_standard_deviation_macro_f1",
            *(f"seed_{seed}_macro_f1" for seed in REQUIRED_SEEDS),
        ]
        comparison_fields = [
            "reference_model_id",
            "candidate_model_id",
            "participant_count",
            "mean_paired_difference",
            "median_paired_difference",
            "paired_cohens_dz",
            "p_value_raw",
            "p_value_holm",
            "test",
        ]
        participant_csv = destination / "participant_metrics.csv"
        comparison_csv = destination / "comparisons.csv"
        participant_csv_hash = _write_bytes_new(
            participant_csv, _csv_bytes(participant_rows, participant_fields)
        )
        comparison_csv_hash = _write_bytes_new(
            comparison_csv, _csv_bytes(comparisons, comparison_fields)
        )
        markdown_lines = [
            "# Disabled-cohort within-group descriptive results",
            "",
            "Post-confirmatory descriptive evidence; not the locked zero-shot endpoint.",
            "",
            "| Model | Mean participant macro-F1 | Worst participant | Lower decile |",
            "|---|---:|---:|---:|",
        ]
        for model_summary in summaries:
            markdown_lines.append(
                "| {model_id} | {mean_participant_macro_f1:.6f} | "
                "{worst_participant_macro_f1:.6f} | "
                "{lower_decile_participant_macro_f1:.6f} |".format(**model_summary)
            )
        markdown_lines.extend(
            [
                "",
                "Participants, not overlapping windows, are the inferential units.",
                "",
            ]
        )
        markdown_path = destination / "SUMMARY.md"
        markdown_hash = _write_bytes_new(markdown_path, "\n".join(markdown_lines).encode("utf-8"))
        summary: dict[str, Any] = {
            "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
            "record_kind": "within_group_cross_subject_aggregate",
            "status": "complete_create_only",
            "evidence_status": WITHIN_GROUP_EVIDENCE_STATUS,
            "scientific_role": "secondary_descriptive_not_locked_confirmatory",
            "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
            "created_at_utc": created_at_utc,
            "code_commit": commit,
            "code_commit_role": "aggregation_implementation",
            "aggregation_code_commit": commit,
            "cell_execution_code_commit": execution_code_commit,
            "manifest": {
                "path": manifest_file.relative_to(root).as_posix(),
                "record_sha256": plan_sha256,
                "file_sha256": sha256_file(manifest_file),
            },
            "split_manifest": {
                "path": split_file.relative_to(root).as_posix(),
                "record_sha256": split_hash,
                "file_sha256": sha256_file(split_file),
            },
            "primary_cache": {
                "path": cache_file.relative_to(root).as_posix(),
                "file_sha256": primary_cache_record_file_sha256,
            },
            "validated_cell_count": len(cells),
            "failed_cell_count": 0,
            "expected_cell_count": 75,
            "models": summaries,
            "paired_comparisons": comparisons,
            "statistics_policy": dict(policy),
            "artifacts": {
                "participant_metrics_csv": {
                    "path": participant_csv.relative_to(root).as_posix(),
                    "sha256": participant_csv_hash,
                },
                "comparisons_csv": {
                    "path": comparison_csv.relative_to(root).as_posix(),
                    "sha256": comparison_csv_hash,
                },
                "markdown_summary": {
                    "path": markdown_path.relative_to(root).as_posix(),
                    "sha256": markdown_hash,
                },
            },
            "cell_records": [
                {
                    "cell_id": cell.key.cell_id,
                    "record_path": cell.record_path.relative_to(root).as_posix(),
                    "record_file_sha256": cell.record_file_sha256,
                    "record_sha256": cell.record_sha256,
                }
                for cell in cells
            ],
            "windows_treated_as_independent_observations": False,
            "target_opening_number": 1,
            "new_target_opening_created": False,
            "raw_dataset_file_accessed": False,
        }
        summary["record_sha256"] = canonical_json_sha256(summary)
        atomic_write_json_new(summary, destination / "summary.json", allowed_root=destination)
        return summary
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": WITHIN_GROUP_SCHEMA_VERSION,
            "record_kind": "within_group_aggregation_failure",
            "status": "failed_preserved_create_only",
            "evidence_status": "failed_aggregation_not_result_evidence",
            "protocol_id": WITHIN_GROUP_PROTOCOL_ID,
            "manifest_sha256": plan_sha256,
            "created_at_utc": created_at_utc,
            "code_commit": commit,
            "code_commit_role": "aggregation_implementation",
            "aggregation_code_commit": commit,
            "cell_execution_code_commit": execution_code_commit,
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "partial_artifacts_preserved": True,
            "raw_dataset_file_accessed": False,
            "new_target_opening_created": False,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, destination / "failure.json", allowed_root=destination)
        raise WithinGroupStatisticsError(
            f"aggregation failed; evidence preserved at {destination / 'failure.json'}"
        ) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--primary-cache-record", type=Path, required=True)
    parser.add_argument("--primary-cache-record-file-sha256", required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = aggregate_within_group_statistics(
            manifest_path=args.manifest,
            split_manifest_path=args.split_manifest,
            primary_cache_record_path=args.primary_cache_record,
            primary_cache_record_file_sha256=args.primary_cache_record_file_sha256,
            result_root=args.result_root,
            artifact_root=args.artifact_root,
            output_directory=args.output_directory,
            output_root=args.output_root,
            code_commit=args.code_commit,
            created_at_utc=args.created_at_utc,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    print(
        json.dumps(
            {
                "status": summary["status"],
                "record_sha256": summary["record_sha256"],
                "validated_cell_count": summary["validated_cell_count"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
