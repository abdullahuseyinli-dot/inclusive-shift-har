"""Create-only aggregation for efficiency and secondary sensor-stress evidence."""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.efficiency import (
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_PRECISIONS,
    FROZEN_NEURAL_MODEL_IDS,
    FROZEN_NEURAL_SEEDS,
    RECURRENT_CUDNN_DISABLED_MODEL_IDS,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


class SecondaryAggregationError(ValueError):
    """Raised when secondary records are incomplete, changed, or misaligned."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SecondaryAggregationError(f"{name} must be an object")
    return value


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise SecondaryAggregationError(f"{name} must be a list")
    return value


def _self_hash(record: Mapping[str, Any], *, field: str, name: str) -> str:
    claimed = record.get(field)
    body = dict(record)
    body.pop(field, None)
    if not isinstance(claimed, str) or claimed != canonical_json_sha256(body):
        raise SecondaryAggregationError(f"{name} self-hash does not validate")
    return claimed


def _resolve_file(value: Any, *, root: Path, name: str, allow_absolute: bool = False) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise SecondaryAggregationError(f"{name} path is missing")
    raw = Path(value)
    if raw.is_absolute() and not allow_absolute:
        raise SecondaryAggregationError(f"{name} path must be artifact-root-relative")
    if not raw.is_absolute() and ("\\" in str(value) or ".." in raw.parts):
        raise SecondaryAggregationError(f"{name} escapes artifact_root")
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise SecondaryAggregationError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SecondaryAggregationError(f"{name} escapes artifact_root") from exc
    if not resolved.is_file():
        raise SecondaryAggregationError(f"{name} is not a regular file")
    return resolved


def _load_index(path: str | Path, *, root: Path, kind: str) -> tuple[Path, Mapping[str, Any]]:
    index_path = _resolve_file(path, root=root, name=kind, allow_absolute=True)
    index = _mapping(load_json_strict(index_path), name=kind)
    _self_hash(index, field="record_sha256", name=kind)
    return index_path, index


def _finite(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SecondaryAggregationError(f"{name} must be finite")
    result = float(value)
    if not np.isfinite(result):
        raise SecondaryAggregationError(f"{name} must be finite")
    return result


def _integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SecondaryAggregationError(f"{name} must be an integer")
    return int(value)


def _string(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise SecondaryAggregationError(f"{name} must be a non-empty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    result = _string(value, name=name)
    if len(result) != 64 or any(character not in "0123456789abcdef" for character in result):
        raise SecondaryAggregationError(f"{name} must be a lowercase SHA-256")
    return result


def _boolean(value: Any, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise SecondaryAggregationError(f"{name} must be boolean")
    return value


def _confined_new_destination(value: str | Path, *, root: Path) -> Path:
    """Validate confinement without creating a parent directory."""

    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise SecondaryAggregationError("aggregate destination may not be a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise SecondaryAggregationError("aggregate destination escapes artifact_root") from exc
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite aggregate: {resolved}")
    return resolved


def _participant_metrics(report_value: Any) -> dict[str, dict[str, float]]:
    report = _mapping(report_value, name="participant-level report")
    rows = _sequence(report.get("participants"), name="participant rows")
    if not rows:
        raise SecondaryAggregationError("participant report is empty")
    result: dict[str, dict[str, float]] = {}
    for value in rows:
        row = _mapping(value, name="participant row")
        participant = row.get("participant_id")
        if not isinstance(participant, str) or not participant or participant in result:
            raise SecondaryAggregationError("participant row identity is invalid")
        result[participant] = {
            "macro_f1": _finite(row.get("macro_f1"), name="participant macro-F1"),
            "balanced_accuracy": _finite(
                row.get("balanced_accuracy"), name="participant balanced accuracy"
            ),
        }
    return result


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_efficiency_profiles(
    index_path: str | Path,
    *,
    artifact_root: str | Path,
    destination: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Aggregate only the complete, lineage-validated 320-cell CUDA matrix."""

    root_candidate = Path(artifact_root)
    if root_candidate.is_symlink():
        raise SecondaryAggregationError("artifact_root may not be a symlink")
    root = root_candidate.resolve(strict=True)
    target = _confined_new_destination(destination, root=root)
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    _, index = _load_index(index_path, root=root, kind="frozen neural efficiency profile index")
    required = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_index",
        "status": "complete_create_only",
        "execution": "sequential_cuda_one_frozen_model_seed_at_a_time",
        "required_device": "cuda",
        "cuda_available_at_start": True,
        "expected_model_ids": list(FROZEN_NEURAL_MODEL_IDS),
        "required_seed_order": list(FROZEN_NEURAL_SEEDS),
        "required_batch_sizes": list(EFFICIENCY_BATCH_SIZES),
        "required_precisions": list(EFFICIENCY_PRECISIONS),
        "neural_model_count": len(FROZEN_NEURAL_MODEL_IDS),
        "neural_model_seed_count": len(FROZEN_NEURAL_MODEL_IDS) * len(FROZEN_NEURAL_SEEDS),
        "profiles_per_model_seed": len(EFFICIENCY_BATCH_SIZES) * len(EFFICIENCY_PRECISIONS),
        "expected_profile_count": len(FROZEN_NEURAL_MODEL_IDS)
        * len(FROZEN_NEURAL_SEEDS)
        * len(EFFICIENCY_BATCH_SIZES)
        * len(EFFICIENCY_PRECISIONS),
        "checkpoint_validation_count": len(FROZEN_NEURAL_MODEL_IDS) * len(FROZEN_NEURAL_SEEDS),
        "profile_count": len(FROZEN_NEURAL_MODEL_IDS)
        * len(FROZEN_NEURAL_SEEDS)
        * len(EFFICIENCY_BATCH_SIZES)
        * len(EFFICIENCY_PRECISIONS),
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }
    mismatches = [key for key, expected in required.items() if index.get(key) != expected]
    if mismatches:
        raise SecondaryAggregationError(f"efficiency index contract mismatch: {mismatches}")
    lineage_fields = (
        "profile_config_sha256",
        "final_freeze_inventory_sha256",
        "frozen_artifact_set_sha256",
        "split_manifest_sha256",
        "locked_target_index_record_sha256",
        "opening_receipt_record_sha256",
    )
    lineage = {field: _sha256(index.get(field), name=field) for field in lineage_fields}
    frozen_code_commit = _string(index.get("frozen_code_commit"), name="frozen code commit")

    validation_values = _sequence(
        index.get("checkpoint_validations"), name="checkpoint validations"
    )
    expected_model_seeds = {
        (model_id, seed) for model_id in FROZEN_NEURAL_MODEL_IDS for seed in FROZEN_NEURAL_SEEDS
    }
    validations: dict[tuple[str, int], Mapping[str, Any]] = {}
    for value in validation_values:
        entry = _mapping(value, name="checkpoint validation entry")
        validation_identity = (
            _string(entry.get("model_id"), name="checkpoint model_id"),
            _integer(entry.get("seed"), name="checkpoint seed"),
        )
        if validation_identity not in expected_model_seeds or validation_identity in validations:
            raise SecondaryAggregationError(
                "checkpoint validation identity is unexpected or duplicated"
            )
        expected_disable = validation_identity[0] in RECURRENT_CUDNN_DISABLED_MODEL_IDS
        if (
            _boolean(entry.get("disable_cudnn"), name="checkpoint disable_cudnn")
            is not expected_disable
        ):
            raise SecondaryAggregationError(
                "checkpoint recurrent cuDNN policy differs from the lock"
            )
        path = _resolve_file(entry.get("path"), root=root, name="checkpoint validation")
        if sha256_file(path) != _sha256(entry.get("file_sha256"), name="validation file hash"):
            raise SecondaryAggregationError("checkpoint validation file hash changed")
        validation = _mapping(load_json_strict(path), name="checkpoint validation")
        validation_hash = _self_hash(
            validation, field="record_sha256", name="checkpoint validation"
        )
        checkpoint_hash = _sha256(entry.get("checkpoint_sha256"), name="checkpoint hash")
        configuration_hash = _sha256(
            entry.get("training_configuration_sha256"), name="configuration hash"
        )
        checkpoint_path = _resolve_file(
            entry.get("checkpoint_path"), root=root, name="frozen checkpoint"
        )
        expected_validation = {
            "schema_version": "1.0.0",
            "record_kind": "fixed_epoch_checkpoint_validation",
            "status": "pass_source_only",
            "checkpoint_sha256": checkpoint_hash,
            "training_configuration_sha256": configuration_hash,
            "split_manifest_sha256": lineage["split_manifest_sha256"],
            "code_commit": frozen_code_commit,
            "model_name": _string(entry.get("model_name"), name="checkpoint model_name"),
            "seed": validation_identity[1],
            "target_information_used_for_selection": False,
        }
        validation_mismatches = [
            key for key, expected in expected_validation.items() if validation.get(key) != expected
        ]
        if (
            validation_mismatches
            or validation_hash != entry.get("record_sha256")
            or sha256_file(checkpoint_path) != checkpoint_hash
        ):
            raise SecondaryAggregationError(
                f"checkpoint validation lineage differs from index: {validation_mismatches}"
            )
        validations[validation_identity] = entry
    if set(validations) != expected_model_seeds or len(validation_values) != len(
        expected_model_seeds
    ):
        raise SecondaryAggregationError("checkpoint validation matrix is incomplete")

    entries = _sequence(index.get("profiles"), name="efficiency profiles")
    expected_profiles = {
        (model_id, seed, batch_size, precision)
        for model_id in FROZEN_NEURAL_MODEL_IDS
        for seed in FROZEN_NEURAL_SEEDS
        for batch_size in EFFICIENCY_BATCH_SIZES
        for precision in EFFICIENCY_PRECISIONS
    }
    seed_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, str]] = set()
    for value in entries:
        entry = _mapping(value, name="efficiency profile entry")
        profile_identity = (
            _string(entry.get("model_id"), name="profile model_id"),
            _integer(entry.get("seed"), name="profile seed"),
            _integer(entry.get("batch_size"), name="profile batch size"),
            _string(entry.get("precision"), name="profile precision"),
        )
        if profile_identity not in expected_profiles or profile_identity in seen:
            raise SecondaryAggregationError(
                "efficiency profile identity is unexpected or duplicated"
            )
        seen.add(profile_identity)
        validation_entry = validations[(profile_identity[0], profile_identity[1])]
        expected_disable = profile_identity[0] in RECURRENT_CUDNN_DISABLED_MODEL_IDS
        if (
            _boolean(entry.get("disable_cudnn"), name="profile disable_cudnn")
            is not expected_disable
        ):
            raise SecondaryAggregationError("profile recurrent cuDNN policy differs from the lock")
        entry_link_fields = {
            "training_configuration_sha256": validation_entry["training_configuration_sha256"],
            "checkpoint_sha256": validation_entry["checkpoint_sha256"],
            "checkpoint_validation_path": validation_entry["path"],
            "checkpoint_validation_file_sha256": validation_entry["file_sha256"],
            "checkpoint_validation_record_sha256": validation_entry["record_sha256"],
        }
        if any(entry.get(key) != expected for key, expected in entry_link_fields.items()):
            raise SecondaryAggregationError("profile entry checkpoint link differs from validation")
        path = _resolve_file(entry.get("path"), root=root, name="efficiency profile")
        if sha256_file(path) != _sha256(entry.get("file_sha256"), name="profile file hash"):
            raise SecondaryAggregationError("efficiency profile file hash changed")
        profile = _mapping(load_json_strict(path), name="efficiency profile")
        profile_hash = _self_hash(profile, field="record_sha256", name="efficiency profile")
        if profile_hash != entry.get("record_sha256"):
            raise SecondaryAggregationError("efficiency profile record differs from index")
        environment = _mapping(profile.get("environment"), name="CUDA environment")
        checkpoint_link = _mapping(
            profile.get("checkpoint_validation"), name="checkpoint validation link"
        )
        model_size = _mapping(profile.get("model_size"), name="model-size profile")
        cudnn = _mapping(profile.get("cudnn_policy"), name="cuDNN policy")
        if (
            profile.get("schema_version") != "1.0.0"
            or profile.get("record_kind") != "cuda_neural_efficiency_profile"
            or profile.get("status") != "profile_complete"
            or profile.get("evidence_status")
            != "measured_cuda_efficiency_not_model_selection_evidence"
            or profile.get("required_device") != "cuda"
            or profile.get("execution_device_type") != "cuda"
            or profile.get("cuda_available_at_profile") is not True
            or environment.get("device_type") != "cuda"
            or not str(environment.get("device", "")).startswith("cuda")
            or profile.get("model_selection_use") is not False
            or profile.get("model_id") != profile_identity[0]
            or profile.get("seed") != profile_identity[1]
            or profile.get("input_shape") != [profile_identity[2], 128, 6]
            or profile.get("precision") != profile_identity[3]
        ):
            raise SecondaryAggregationError(
                "efficiency profile kind, status, or CUDA identity is invalid"
            )
        expected_profile_lineage = {
            **lineage,
            "frozen_code_commit": frozen_code_commit,
            "training_configuration_sha256": validation_entry["training_configuration_sha256"],
            "checkpoint_path": validation_entry["checkpoint_path"],
        }
        if any(profile.get(key) != expected for key, expected in expected_profile_lineage.items()):
            raise SecondaryAggregationError("efficiency profile lineage differs from index")
        if (
            checkpoint_link
            != {
                "path": validation_entry["path"],
                "file_sha256": validation_entry["file_sha256"],
                "record_sha256": validation_entry["record_sha256"],
            }
            or model_size.get("checkpoint_sha256") != validation_entry["checkpoint_sha256"]
            or model_size.get("checkpoint_validation_record_sha256")
            != validation_entry["record_sha256"]
        ):
            raise SecondaryAggregationError("efficiency profile checkpoint link is invalid")
        before = _boolean(cudnn.get("backend_enabled_before_profile"), name="cuDNN before")
        after = _boolean(cudnn.get("backend_enabled_after_profile"), name="cuDNN after")
        if (
            cudnn.get("checkpoint_configuration_disable_cudnn") is not expected_disable
            or cudnn.get("backend_enabled_during_profile") is not (not expected_disable)
            or before is not after
            or cudnn.get("original_backend_state_restored") is not True
        ):
            raise SecondaryAggregationError("profile did not apply and restore the cuDNN policy")
        parameters = _mapping(profile.get("parameters"), name="parameter profile")
        complexity = _mapping(profile.get("complexity"), name="complexity profile")
        latency = _mapping(profile.get("latency_ms_per_batch"), name="latency profile")
        vram = _mapping(profile.get("vram_bytes"), name="VRAM profile")
        percentiles = _mapping(latency.get("percentiles"), name="latency percentiles")
        if complexity.get("coverage_status") != "supported_operator_subset_only":
            raise SecondaryAggregationError(
                "MAC coverage scope is not the declared operator subset"
            )
        seed_rows.append(
            {
                "model_id": profile_identity[0],
                "seed": profile_identity[1],
                "batch_size": profile_identity[2],
                "precision": profile_identity[3],
                "disable_cudnn": expected_disable,
                "parameters_total": int(parameters["total"]),
                "parameters_trainable": int(parameters["trainable"]),
                "state_tensor_bytes": int(parameters["state_tensor_bytes"]),
                "supported_operator_macs_per_window": int(
                    complexity["supported_operator_macs_per_window"]
                ),
                "estimated_flops_per_window": int(
                    complexity["estimated_flops_from_supported_macs_per_window"]
                ),
                "mac_coverage_status": complexity["coverage_status"],
                "latency_mean_ms_per_batch": _finite(latency.get("mean"), name="latency"),
                "latency_median_ms_per_batch": _finite(
                    latency.get("median"), name="median latency"
                ),
                "latency_p95_ms_per_batch": _finite(percentiles.get("95.0"), name="p95 latency"),
                "latency_mean_ms_per_window": _finite(
                    latency.get("mean_per_window"), name="per-window latency"
                ),
                "peak_vram_allocated_bytes": int(vram["peak_allocated"]),
                "peak_vram_reserved_bytes": int(vram["peak_reserved"]),
                "checkpoint_file_bytes": int(model_size["checkpoint_file_bytes"]),
                "record_sha256": profile_hash,
            }
        )
    if seen != expected_profiles or len(entries) != len(expected_profiles):
        raise SecondaryAggregationError("efficiency profile matrix is incomplete")

    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in seed_rows:
        grouped[(str(row["model_id"]), int(row["batch_size"]), str(row["precision"]))].append(row)
    for (model_id, _batch_size, _precision), rows in grouped.items():
        if {int(row["seed"]) for row in rows} != set(FROZEN_NEURAL_SEEDS):
            raise SecondaryAggregationError(
                f"{model_id} efficiency combinations do not contain all frozen seeds"
            )
    aggregate_rows: list[dict[str, Any]] = []
    for (model_id, batch_size, precision), rows in sorted(grouped.items()):
        for field in (
            "parameters_total",
            "parameters_trainable",
            "supported_operator_macs_per_window",
            "estimated_flops_per_window",
        ):
            if len({int(row[field]) for row in rows}) != 1:
                raise SecondaryAggregationError(
                    f"{model_id} architecture quantity {field} differs across seeds"
                )
        aggregate_rows.append(
            {
                "model_id": model_id,
                "batch_size": batch_size,
                "precision": precision,
                "seed_count": len(rows),
                "seed_order": sorted(int(row["seed"]) for row in rows),
                "parameters_total": rows[0]["parameters_total"],
                "parameters_trainable": rows[0]["parameters_trainable"],
                "supported_operator_macs_per_window": rows[0]["supported_operator_macs_per_window"],
                "estimated_flops_per_window": rows[0]["estimated_flops_per_window"],
                "latency_mean_ms_per_batch_seed_mean": _mean(
                    [float(row["latency_mean_ms_per_batch"]) for row in rows]
                ),
                "latency_p95_ms_per_batch_seed_mean": _mean(
                    [float(row["latency_p95_ms_per_batch"]) for row in rows]
                ),
                "latency_mean_ms_per_window_seed_mean": _mean(
                    [float(row["latency_mean_ms_per_window"]) for row in rows]
                ),
                "peak_vram_allocated_bytes_max": max(
                    int(row["peak_vram_allocated_bytes"]) for row in rows
                ),
                "peak_vram_reserved_bytes_max": max(
                    int(row["peak_vram_reserved_bytes"]) for row in rows
                ),
                "checkpoint_file_bytes_seed_mean": _mean(
                    [float(row["checkpoint_file_bytes"]) for row in rows]
                ),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_aggregate",
        "status": "complete_create_only",
        "created_at_utc": timestamp,
        "efficiency_index_record_sha256": index["record_sha256"],
        **lineage,
        "frozen_code_commit": frozen_code_commit,
        "validated_checkpoint_count": len(validations),
        "validated_profile_count": len(seen),
        "validated_combinatorics": "16_models_x_5_seeds_x_2_batches_x_2_precisions",
        "cudnn_policy_validation": "applied_from_each_frozen_configuration_and_restored",
        "tables": {
            "seed_profiles": seed_rows,
            "model_batch_precision_aggregates": aggregate_rows,
        },
        "mac_scope_warning": "reported MAC/FLOP values cover only declared supported operators",
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    atomic_write_json_new(payload, target, allowed_root=root)
    return payload


def _load_record_from_entry(
    entry: Mapping[str, Any], *, root: Path, expected_kind: str | None = None
) -> Mapping[str, Any]:
    path = _resolve_file(entry.get("record_path"), root=root, name="secondary result record")
    if sha256_file(path) != entry.get("record_file_sha256"):
        raise SecondaryAggregationError("secondary result record file hash changed")
    record = _mapping(load_json_strict(path), name="secondary result record")
    record_hash = _self_hash(record, field="record_sha256", name="secondary result record")
    if record_hash != entry.get("record_sha256"):
        raise SecondaryAggregationError("secondary result record differs from index")
    if expected_kind is not None and record.get("record_kind") != expected_kind:
        raise SecondaryAggregationError("secondary result record kind is unexpected")
    return record


def aggregate_sensor_stress(
    index_path: str | Path,
    *,
    artifact_root: str | Path,
    destination: str | Path,
    created_at_utc: str,
) -> dict[str, Any]:
    """Create participant-seed and seed-averaged clean-to-stress delta tables."""

    root = Path(artifact_root).resolve(strict=True)
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    _, index = _load_index(index_path, root=root, kind="sensor reliability stress index")
    required = {
        "record_kind": "secondary_sensor_reliability_index",
        "status": "complete_create_only",
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_clean_inference_rerun": False,
        "raw_target_materialization_invoked": False,
        "opening_or_unlock_invoked": False,
    }
    mismatches = [key for key, expected in required.items() if index.get(key) != expected]
    if mismatches:
        raise SecondaryAggregationError(f"stress index contract mismatch: {mismatches}")
    model_seed_count = _integer(index.get("model_seed_count"), name="model-seed count")
    condition_count = _integer(index.get("condition_count"), name="condition count")
    if condition_count != 6:
        raise SecondaryAggregationError("secondary stress index must contain all six conditions")
    references: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for cohort, field, kind in (
        ("source", "source_clean_references", "secondary_source_clean_reference"),
        ("target", "target_clean_references", "locked_target_per_seed_result"),
    ):
        for value in _sequence(index.get(field), name=field):
            entry = _mapping(value, name="clean reference entry")
            key = (
                _string(entry.get("model_id"), name="clean reference model_id"),
                _integer(entry.get("seed"), name="clean reference seed"),
                cohort,
            )
            if key in references:
                raise SecondaryAggregationError("clean reference is duplicated")
            references[key] = _load_record_from_entry(entry, root=root, expected_kind=kind)
    if len(references) != model_seed_count * 2:
        raise SecondaryAggregationError("clean-reference index does not cover both cohorts")
    stress_values = _sequence(index.get("stress_results"), name="stress results")
    if (
        len(stress_values) != index.get("stress_result_count")
        or len(stress_values) != model_seed_count * condition_count * 2
    ):
        raise SecondaryAggregationError("stress result index is incomplete")
    participant_seed_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str, str]] = set()
    for value in stress_values:
        entry = _mapping(value, name="stress result entry")
        identity = (
            _string(entry.get("model_id"), name="stress model_id"),
            _integer(entry.get("seed"), name="stress seed"),
            _string(entry.get("cohort"), name="stress cohort"),
            _string(entry.get("condition_id"), name="stress condition"),
        )
        if identity in seen or identity[2] not in {"source", "target"}:
            raise SecondaryAggregationError("stress result identity is invalid or duplicated")
        seen.add(identity)
        record = _load_record_from_entry(
            entry, root=root, expected_kind="secondary_sensor_reliability_evaluation"
        )
        required_record = {
            "model_id": identity[0],
            "seed": identity[1],
            "cohort": identity[2],
            "track_role": "secondary_post_confirmatory",
            "primary_claim_eligible": False,
            "used_for_model_selection": False,
            "target_based_tuning": False,
            "stress_config_sha256": index["stress_config_sha256"],
        }
        if any(record.get(key) != expected for key, expected in required_record.items()):
            raise SecondaryAggregationError("stress result lineage differs from index")
        condition = _mapping(record.get("condition"), name="stress condition metadata")
        if condition.get("condition_id") != identity[3]:
            raise SecondaryAggregationError("stress condition differs from index")
        try:
            clean_record = references[(identity[0], identity[1], identity[2])]
        except KeyError as exc:
            raise SecondaryAggregationError("stress result lacks a clean reference") from exc
        clean = _participant_metrics(clean_record.get("participant_level_report"))
        stressed = _participant_metrics(record.get("participant_level_report"))
        if set(clean) != set(stressed):
            raise SecondaryAggregationError("clean and stressed participant sets differ")
        for participant in sorted(clean, key=lambda item: (len(item), item)):
            clean_row = clean[participant]
            stress_row = stressed[participant]
            participant_seed_rows.append(
                {
                    "model_id": identity[0],
                    "seed": identity[1],
                    "cohort": identity[2],
                    "condition_id": identity[3],
                    "participant_id": participant,
                    "clean_macro_f1": clean_row["macro_f1"],
                    "stress_macro_f1": stress_row["macro_f1"],
                    "delta_macro_f1": stress_row["macro_f1"] - clean_row["macro_f1"],
                    "clean_balanced_accuracy": clean_row["balanced_accuracy"],
                    "stress_balanced_accuracy": stress_row["balanced_accuracy"],
                    "delta_balanced_accuracy": stress_row["balanced_accuracy"]
                    - clean_row["balanced_accuracy"],
                }
            )
    coverage: dict[tuple[str, int, str], set[str]] = defaultdict(set)
    for model_id, seed, cohort, condition_id in seen:
        coverage[(model_id, seed, cohort)].add(condition_id)
    if set(coverage) != set(references) or any(
        len(conditions) != condition_count for conditions in coverage.values()
    ):
        raise SecondaryAggregationError("stress condition coverage is incomplete")

    grouped_participants: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in participant_seed_rows:
        grouped_participants[
            (
                str(row["model_id"]),
                str(row["cohort"]),
                str(row["condition_id"]),
                str(row["participant_id"]),
            )
        ].append(row)
    expected_seeds: dict[tuple[str, str], set[int]] = defaultdict(set)
    for model_id, seed, cohort in references:
        expected_seeds[(model_id, cohort)].add(seed)
    for model_id in {key[0] for key in expected_seeds}:
        if expected_seeds[(model_id, "source")] != expected_seeds[(model_id, "target")]:
            raise SecondaryAggregationError(
                f"{model_id} source and target clean references use different seeds"
            )
    participant_averaged_rows: list[dict[str, Any]] = []
    for (model_id, cohort, condition_id, participant), rows in sorted(grouped_participants.items()):
        observed_seeds = {int(row["seed"]) for row in rows}
        if observed_seeds != expected_seeds[(model_id, cohort)]:
            raise SecondaryAggregationError(
                f"{model_id} participant {participant} lacks a complete stress seed set"
            )
        participant_averaged_rows.append(
            {
                "model_id": model_id,
                "cohort": cohort,
                "condition_id": condition_id,
                "participant_id": participant,
                "seed_count": len(rows),
                "seed_order": sorted(int(row["seed"]) for row in rows),
                "clean_macro_f1_seed_mean": _mean([float(row["clean_macro_f1"]) for row in rows]),
                "stress_macro_f1_seed_mean": _mean([float(row["stress_macro_f1"]) for row in rows]),
                "delta_macro_f1_seed_mean": _mean([float(row["delta_macro_f1"]) for row in rows]),
                "delta_balanced_accuracy_seed_mean": _mean(
                    [float(row["delta_balanced_accuracy"]) for row in rows]
                ),
            }
        )
    grouped_conditions: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in participant_averaged_rows:
        grouped_conditions[
            (str(row["model_id"]), str(row["cohort"]), str(row["condition_id"]))
        ].append(row)
    cohort_condition_rows: list[dict[str, Any]] = []
    for (model_id, cohort, condition_id), rows in sorted(grouped_conditions.items()):
        stress_values_array = np.asarray(
            [float(row["stress_macro_f1_seed_mean"]) for row in rows], dtype=np.float64
        )
        cohort_condition_rows.append(
            {
                "model_id": model_id,
                "cohort": cohort,
                "condition_id": condition_id,
                "participant_count": len(rows),
                "clean_mean_participant_macro_f1": _mean(
                    [float(row["clean_macro_f1_seed_mean"]) for row in rows]
                ),
                "stress_mean_participant_macro_f1": float(stress_values_array.mean()),
                "mean_delta_participant_macro_f1": _mean(
                    [float(row["delta_macro_f1_seed_mean"]) for row in rows]
                ),
                "stress_worst_participant_macro_f1": float(stress_values_array.min()),
                "stress_lower_decile_participant_macro_f1": float(
                    np.quantile(stress_values_array, 0.1, method="linear")
                ),
            }
        )
    by_group = {
        (str(row["model_id"]), str(row["condition_id"]), str(row["cohort"])): row
        for row in cohort_condition_rows
    }
    cohort_gap_rows: list[dict[str, Any]] = []
    model_conditions = sorted(
        {(key[0], key[1]) for key in by_group}, key=lambda item: (item[0].casefold(), item[1])
    )
    for model_id, condition_id in model_conditions:
        if (model_id, condition_id, "source") not in by_group or (
            model_id,
            condition_id,
            "target",
        ) not in by_group:
            raise SecondaryAggregationError("stress aggregate lacks one cohort")
        source = by_group[(model_id, condition_id, "source")]
        target = by_group[(model_id, condition_id, "target")]
        clean_gap = float(source["clean_mean_participant_macro_f1"]) - float(
            target["clean_mean_participant_macro_f1"]
        )
        stress_gap = float(source["stress_mean_participant_macro_f1"]) - float(
            target["stress_mean_participant_macro_f1"]
        )
        cohort_gap_rows.append(
            {
                "model_id": model_id,
                "condition_id": condition_id,
                "clean_source_minus_target_gap": clean_gap,
                "stress_source_minus_target_gap": stress_gap,
                "gap_change_under_stress": stress_gap - clean_gap,
            }
        )
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "secondary_sensor_reliability_participant_aggregate",
        "status": "complete_create_only",
        "created_at_utc": timestamp,
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "stress_index_record_sha256": index["record_sha256"],
        "stress_config_sha256": index["stress_config_sha256"],
        "final_freeze_inventory_sha256": index["final_freeze_inventory_sha256"],
        "locked_target_index_record_sha256": index["locked_target_index_record_sha256"],
        "tables": {
            "participant_seed_deltas": participant_seed_rows,
            "participant_seed_averaged_deltas": participant_averaged_rows,
            "cohort_condition_aggregates": cohort_condition_rows,
            "source_target_gap_deltas": cohort_gap_rows,
        },
        "statistical_unit": "participant",
        "seed_handling": "participant metrics averaged over frozen seeds before cohort summary",
        "independence_note": "windows are not treated as statistical replicates",
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    destination_path = Path(destination)
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    atomic_write_json_new(payload, destination_path, allowed_root=root)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="operation", required=True)
    for name in ("efficiency", "sensor-stress"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--index", type=Path, required=True)
        subparser.add_argument("--artifact-root", type=Path, required=True)
        subparser.add_argument("--destination", type=Path, required=True)
        subparser.add_argument("--created-at-utc", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    kwargs = {
        "index_path": args.index,
        "artifact_root": args.artifact_root,
        "destination": args.destination,
        "created_at_utc": args.created_at_utc,
    }
    if args.operation == "efficiency":
        result = aggregate_efficiency_profiles(**kwargs)
    else:
        result = aggregate_sensor_stress(**kwargs)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
