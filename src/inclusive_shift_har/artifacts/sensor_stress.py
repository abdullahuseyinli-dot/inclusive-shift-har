"""Create-only evidence artifacts for secondary sensor-reliability evaluation."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.sensor_reliability import SensorReliabilityConfig
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    sha256_file,
)


class SensorStressArtifactError(ValueError):
    """Raised when secondary stress evidence is incomplete or unsafe to publish."""


_PREDICTION_KEYS = (
    "logits",
    "probabilities",
    "labels",
    "participant_ids",
    "window_ids",
)


def _sha256(value: str, *, description: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise SensorStressArtifactError(f"{description} must be a full SHA-256")
    return normalized


def _timestamp(value: str) -> str:
    if not value.endswith("Z"):
        raise SensorStressArtifactError("created_at_utc must use the canonical UTC Z suffix")
    try:
        parsed = datetime.fromisoformat(value[:-1] + "+00:00")
    except ValueError as exc:
        raise SensorStressArtifactError("created_at_utc is not valid ISO-8601") from exc
    if parsed.tzinfo != UTC:
        raise SensorStressArtifactError("created_at_utc must be UTC")
    return value


def _confined_new_path(path: Path, *, allowed_root: Path, description: str) -> Path:
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise SensorStressArtifactError(f"allowed_root is not a directory: {root}")
    candidate = path.resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SensorStressArtifactError(f"{description} escapes allowed_root") from exc
    if candidate == root:
        raise SensorStressArtifactError(f"{description} cannot be allowed_root itself")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise SensorStressArtifactError(
            f"{description} parent resolves outside allowed_root"
        ) from exc
    target = resolved_parent / candidate.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing {description}: {target}")
    return target


def _write_npz_new(path: Path, arrays: Mapping[str, NDArray[Any]]) -> str:
    """Publish a compressed NPZ with create-if-absent hard-link semantics."""

    temporary = path.parent / f".{path.name}.partial.{uuid4().hex}"
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **cast(dict[str, Any], dict(arrays)))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except Exception as exc:
        raise OSError(
            f"prediction publication failed; partial evidence retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()
    return sha256_file(path)


def _validated_condition_metadata(
    value: Mapping[str, Any], *, config: SensorReliabilityConfig
) -> dict[str, Any]:
    metadata = dict(value)
    claimed = metadata.get("record_sha256")
    unhashed = dict(metadata)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise SensorStressArtifactError("condition metadata self-hash does not validate")
    required = {
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "target_based_tuning": False,
        "stress_config_sha256": config.config_sha256,
        "input_space": "physical_pre_normalization",
    }
    for key, expected in required.items():
        if metadata.get(key) != expected:
            raise SensorStressArtifactError(f"condition metadata contract mismatch at {key}")
    condition_id = metadata.get("condition_id")
    if not isinstance(condition_id, str):
        raise SensorStressArtifactError("condition metadata lacks condition_id")
    condition = config.condition(condition_id)
    if metadata.get("condition_sha256") != condition.condition_sha256:
        raise SensorStressArtifactError("condition hash disagrees with locked configuration")
    return metadata


def create_secondary_stress_artifacts(
    *,
    prediction_path: Path,
    record_path: Path,
    allowed_root: Path,
    logits: NDArray[np.floating[Any]],
    probabilities: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    participant_ids: Sequence[str],
    window_ids: Sequence[str],
    class_names: Sequence[str],
    config: SensorReliabilityConfig,
    condition_metadata: Mapping[str, Any],
    model_id: str,
    seed: int,
    frozen_checkpoint_sha256: str,
    frozen_normalization_sha256: str,
    base_prediction_sha256: str,
    primary_confirmatory_completion_record_sha256: str,
    created_at_utc: str,
) -> dict[str, Any]:
    """Publish aligned predictions plus participant-level secondary metrics.

    The caller supplies immutable lineage hashes.  This function deliberately
    does not open a base prediction, checkpoint, or confirmatory record and thus
    cannot use target performance for tuning or selection.
    """

    if not model_id.strip():
        raise SensorStressArtifactError("model_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SensorStressArtifactError("seed must be a non-negative integer")
    metadata = _validated_condition_metadata(condition_metadata, config=config)
    checkpoint_hash = _sha256(frozen_checkpoint_sha256, description="checkpoint hash")
    normalization_hash = _sha256(
        frozen_normalization_sha256, description="normalization hash"
    )
    base_hash = _sha256(base_prediction_sha256, description="base prediction hash")
    completion_hash = _sha256(
        primary_confirmatory_completion_record_sha256,
        description="primary confirmatory completion record hash",
    )
    timestamp = _timestamp(created_at_utc)

    logit_array = np.asarray(logits, dtype=np.float64)
    probability_array = np.asarray(probabilities, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    participants = tuple(participant_ids)
    windows = tuple(window_ids)
    if logit_array.ndim != 2 or probability_array.shape != logit_array.shape:
        raise SensorStressArtifactError("logits and probabilities must be aligned matrices")
    if label_array.ndim != 1 or label_array.size != logit_array.shape[0]:
        raise SensorStressArtifactError("labels must align with prediction rows")
    if len(participants) != label_array.size or len(windows) != label_array.size:
        raise SensorStressArtifactError("participant/window IDs must align with predictions")
    if any(not isinstance(item, str) or not item for item in participants + windows):
        raise SensorStressArtifactError("participant/window IDs must be non-empty strings")
    if len(set(windows)) != len(windows):
        raise SensorStressArtifactError("window IDs must be unique")
    if not np.isfinite(logit_array).all() or not np.isfinite(probability_array).all():
        raise SensorStressArtifactError("prediction arrays must be finite")
    if metadata.get("window_count") != label_array.size:
        raise SensorStressArtifactError("condition metadata window count is misaligned")
    if metadata.get("window_ids_sha256") != canonical_json_sha256(list(windows)):
        raise SensorStressArtifactError("condition metadata window IDs are misaligned")

    report = classification_report(
        label_array,
        probability_array,
        participants,
        class_names=tuple(class_names),
    )
    prediction_target = _confined_new_path(
        prediction_path, allowed_root=allowed_root, description="stress prediction artifact"
    )
    record_target = _confined_new_path(
        record_path, allowed_root=allowed_root, description="stress evaluation record"
    )
    arrays: dict[str, NDArray[Any]] = {
        "logits": logit_array,
        "probabilities": probability_array,
        "labels": label_array,
        "participant_ids": np.asarray(participants, dtype=np.str_),
        "window_ids": np.asarray(windows, dtype=np.str_),
    }
    prediction_hash = _write_npz_new(prediction_target, arrays)
    root = allowed_root.resolve(strict=True)
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "secondary_sensor_reliability_evaluation",
        "status": "secondary_post_confirmatory_stress_complete",
        "created_at_utc": timestamp,
        "track_role": "secondary_post_confirmatory",
        "primary_claim_eligible": False,
        "used_for_model_selection": False,
        "calibration_refit": False,
        "threshold_refit": False,
        "target_based_tuning": False,
        "stress_levels_empirically_calibrated": False,
        "model_id": model_id,
        "seed": seed,
        "class_names": list(class_names),
        "stress_config_id": config.config_id,
        "stress_config_sha256": config.config_sha256,
        "condition": metadata,
        "frozen_lineage": {
            "checkpoint_sha256": checkpoint_hash,
            "normalization_sha256": normalization_hash,
            "base_prediction_sha256": base_hash,
            "primary_confirmatory_completion_record_sha256": completion_hash,
        },
        "prediction_artifact": {
            "path": prediction_target.relative_to(root).as_posix(),
            "sha256": prediction_hash,
            "format": "npz",
            "keys": list(_PREDICTION_KEYS),
            "create_only": True,
        },
        "participant_level_report": report,
        "independence_note": "windows are not treated as independent statistical units",
    }
    record["record_sha256"] = canonical_json_sha256(record)
    atomic_write_json_new(record, record_target, allowed_root=root)
    return record
