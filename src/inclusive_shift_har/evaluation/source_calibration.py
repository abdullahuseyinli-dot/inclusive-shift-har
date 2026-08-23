"""Immutable source-validation-only temperature-calibration records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.calibration import TemperatureCalibrator, fit_temperature
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_bytes,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

SOURCE_CALIBRATOR_SCHEMA_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: str, *, name: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase full SHA-256")


def _array_sha256(array: NDArray[Any], *, dtype: np.dtype[Any]) -> str:
    canonical = np.ascontiguousarray(array, dtype=dtype)
    header = canonical_json_bytes(
        {
            "algorithm": "numpy-c-contiguous-little-endian-v1",
            "dtype": canonical.dtype.str,
            "shape": list(canonical.shape),
        }
    )
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def build_source_temperature_calibrator_record(
    validation_logits: NDArray[np.float64],
    validation_labels: NDArray[np.int64],
    validation_window_ids: Sequence[str],
    *,
    fit_partition: str,
    checkpoint_sha256: str,
    training_configuration_sha256: str,
    split_manifest_sha256: str,
    class_names: Sequence[str],
) -> dict[str, Any]:
    """Fit temperature scaling with an explicit, fail-closed source-only contract."""

    if fit_partition != "source_validation":
        raise PermissionError("temperature calibration is restricted to source_validation")
    for name, value in (
        ("checkpoint_sha256", checkpoint_sha256),
        ("training_configuration_sha256", training_configuration_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
    ):
        _require_sha256(value, name=name)
    logits = np.asarray(validation_logits, dtype=np.float64)
    labels = np.asarray(validation_labels, dtype=np.int64)
    window_ids = tuple(str(value) for value in validation_window_ids)
    classes = tuple(str(value) for value in class_names)
    if logits.ndim != 2 or labels.ndim != 1 or logits.shape[0] != labels.size:
        raise ValueError("source-validation logits and labels are not aligned")
    if len(window_ids) != labels.size or len(set(window_ids)) != len(window_ids):
        raise ValueError("source-validation window IDs must be aligned and unique")
    if len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError("class_names must be a unique, ordered schema")
    if logits.shape[1] != len(classes):
        raise ValueError("logit columns do not match the locked class schema")

    window_ids_sha256 = canonical_json_sha256(list(window_ids))
    calibrator = fit_temperature(
        logits,
        labels,
        validation_split_sha256=split_manifest_sha256,
    )
    payload: dict[str, Any] = {
        "schema_version": SOURCE_CALIBRATOR_SCHEMA_VERSION,
        "record_kind": "source_temperature_calibrator",
        "status": "frozen_source_validation",
        "evidence_status": "source_validation_calibration_target_sealed",
        "method": "scalar_temperature",
        "fit_partition": "source_validation",
        "checkpoint_sha256": checkpoint_sha256,
        "training_configuration_sha256": training_configuration_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "validation_window_ids_sha256": window_ids_sha256,
        "class_names": list(classes),
        "class_schema_sha256": canonical_json_sha256(list(classes)),
        "fit_input": {
            "sample_count": labels.size,
            "logits_sha256": _array_sha256(logits, dtype=np.dtype("<f8")),
            "labels_sha256": _array_sha256(labels, dtype=np.dtype("<i8")),
        },
        "temperature": calibrator.temperature,
        "nll_before": calibrator.nll_before,
        "nll_after": calibrator.nll_after,
        "target_subject_or_window_records_used": False,
        "target_labels_or_performance_used": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    return payload


def validate_source_temperature_calibrator_record(
    record: Mapping[str, Any],
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_training_configuration_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
) -> TemperatureCalibrator:
    """Validate a serialized calibrator and reconstruct its numerical transform."""

    required_values = {
        "schema_version": SOURCE_CALIBRATOR_SCHEMA_VERSION,
        "record_kind": "source_temperature_calibrator",
        "status": "frozen_source_validation",
        "evidence_status": "source_validation_calibration_target_sealed",
        "fit_partition": "source_validation",
        "method": "scalar_temperature",
        "target_subject_or_window_records_used": False,
        "target_labels_or_performance_used": False,
    }
    mismatches = [key for key, value in required_values.items() if record.get(key) != value]
    if mismatches:
        raise ValueError(f"source calibrator contract mismatch: {mismatches}")
    claimed_hash = record.get("record_sha256")
    body = dict(record)
    body.pop("record_sha256", None)
    if claimed_hash != canonical_json_sha256(body):
        raise ValueError("source calibrator self-hash does not validate")
    expected_values = {
        "checkpoint_sha256": expected_checkpoint_sha256,
        "training_configuration_sha256": expected_training_configuration_sha256,
        "split_manifest_sha256": expected_split_manifest_sha256,
    }
    for name in (
        "checkpoint_sha256",
        "training_configuration_sha256",
        "split_manifest_sha256",
        "validation_window_ids_sha256",
        "class_schema_sha256",
    ):
        value = record.get(name)
        if not isinstance(value, str):
            raise ValueError(f"source calibrator lacks {name}")
        _require_sha256(value, name=name)
        expected = expected_values.get(name)
        if expected is not None and value != expected:
            raise ValueError(f"source calibrator {name} differs from the frozen artifact")
    classes = record.get("class_names")
    if (
        not isinstance(classes, list)
        or len(classes) < 2
        or any(not isinstance(value, str) or not value for value in classes)
        or len(set(classes)) != len(classes)
        or canonical_json_sha256(classes) != record.get("class_schema_sha256")
    ):
        raise ValueError("source calibrator class schema does not validate")
    temperature = record.get("temperature")
    fit_input = record.get("fit_input")
    if not isinstance(fit_input, Mapping):
        raise ValueError("source calibrator lacks fit-input provenance")
    sample_count = fit_input.get("sample_count")
    for name in ("logits_sha256", "labels_sha256"):
        value = fit_input.get(name)
        if not isinstance(value, str):
            raise ValueError(f"source calibrator lacks {name}")
        _require_sha256(value, name=name)
    nll_before = record.get("nll_before")
    nll_after = record.get("nll_after")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not np.isfinite(temperature)
        or temperature <= 0
    ):
        raise ValueError("source calibrator temperature must be finite and positive")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("source calibrator sample count must be positive")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(value)
        or value < 0
        for value in (nll_before, nll_after)
    ):
        raise ValueError("source calibrator NLL values must be finite")
    return TemperatureCalibrator(
        temperature=float(temperature),
        validation_split_sha256=str(record["split_manifest_sha256"]),
        sample_count=sample_count,
        nll_before=float(cast(float, nll_before)),
        nll_after=float(cast(float, nll_after)),
    )


def write_source_temperature_calibrator_new(
    record: Mapping[str, Any],
    destination: str | Path,
    *,
    allowed_root: str | Path,
) -> dict[str, str]:
    """Validate and publish a calibrator once; existing evidence is never replaced."""

    validate_source_temperature_calibrator_record(record)
    path = atomic_write_json_new(dict(record), destination, allowed_root=allowed_root)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "record_sha256": str(record["record_sha256"]),
    }


def load_source_temperature_calibrator_file(
    path: str | Path,
    *,
    expected_checkpoint_sha256: str | None = None,
    expected_training_configuration_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
) -> tuple[dict[str, Any], TemperatureCalibrator]:
    """Read a strict JSON record and validate all requested frozen lineage hashes."""

    payload = load_json_strict(path)
    if not isinstance(payload, dict):
        raise ValueError("source calibrator record root must be an object")
    calibrator = validate_source_temperature_calibrator_record(
        payload,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        expected_training_configuration_sha256=expected_training_configuration_sha256,
        expected_split_manifest_sha256=expected_split_manifest_sha256,
    )
    return payload, calibrator
