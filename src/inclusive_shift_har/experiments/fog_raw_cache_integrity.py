"""Independent, zero-fit FoG raw-to-cache integrity audit.

This module deliberately does not import the production FoG loader, resampling
helpers, engineered-feature extractor, or any estimator. The reference cache is
used only after raw rows independently determine the complete physical grid.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import platform
import subprocess
import sys
import time
import traceback
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import requests
import scipy  # type: ignore[import-untyped]
import yaml
from numpy.typing import NDArray
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
StringArray = NDArray[np.str_]

SOURCE_COLUMNS = (
    "timestamp",
    "back_acc_x",
    "back_acc_y",
    "back_acc_z",
    "back_gyro_x",
    "back_gyro_y",
    "back_gyro_z",
    "activity",
    "subjectID",
    "sessionID",
)
DESCRIPTOR_NAMES = (
    "accelerometer_norm__rms",
    "accelerometer_norm__std",
    "gyroscope_norm__rms",
    "gyroscope_norm__std",
)
LABEL_MAP = {1: 0, 2: 1, 3: 2, 6: 0, 7: 0}
REQUIRED_RUN_FILES = (
    "config_snapshot.yaml",
    "per_session_integrity.json",
    "preflight.json",
    "protocol_snapshot.md",
    "p7_descriptor_confirmation.json",
    "raw_cache_comparison.json",
    "result.json",
    "runtime.json",
    "source_receipt.json",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _read_json(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), str(path))


def _read_yaml(path: Path) -> dict[str, Any]:
    return _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), str(path))


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_create_only(path: Path, payload: Mapping[str, Any]) -> None:
    body = (
        json.dumps(
            dict(payload), sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        + b"\n"
    )
    _write_bytes_create_only(path, body)


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    _require("record_sha256" not in result, "record_sha256 is reserved")
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_sealed(payload: Mapping[str, Any], name: str) -> None:
    body = dict(payload)
    declared = body.pop("record_sha256", None)
    _require(
        isinstance(declared, str) and declared == canonical_json_sha256(body),
        f"{name} self-hash is invalid",
    )


def _external_array_sha256(values: NDArray[Any]) -> str:
    """Match the external loader's shape/dtype-prefixed array convention."""

    array = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str((array.shape, array.dtype.str)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _cache_array_sha256(values: NDArray[Any]) -> str:
    """Match the factorial cache's dtype/pipe/shape-prefixed convention."""

    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"|")
    digest.update(",".join(str(item) for item in array.shape).encode("ascii"))
    digest.update(b"|")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _git_output(repository_root: Path, *arguments: str) -> str:
    return subprocess.check_output(
        ("git", *arguments), cwd=repository_root, text=True, encoding="utf-8"
    ).strip()


def _git_state(repository_root: Path) -> dict[str, Any]:
    status = _git_output(repository_root, "status", "--porcelain=v1")
    return {
        "branch": _git_output(repository_root, "branch", "--show-current"),
        "commit": _git_output(repository_root, "rev-parse", "HEAD"),
        "clean": not status,
        "porcelain": status.splitlines() if status else [],
    }


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject drift from the frozen integrity contract."""

    source = _mapping(config.get("source"), "source")
    reference = _mapping(config.get("reference_run"), "reference_run")
    reconstruction = _mapping(config.get("reconstruction"), "reconstruction")
    tolerances = _mapping(config.get("numerical_tolerances"), "numerical_tolerances")
    resources = _mapping(config.get("resources"), "resources")
    claims = _mapping(config.get("claims"), "claims")
    expected_roster = [f"fogstar:{index:03d}" for index in range(1, 23)]
    _require(config.get("schema_version") == "1.0.0", "unexpected config schema")
    _require(config.get("audit_id") == "fog-raw-cache-integrity-v1", "unexpected audit id")
    _require(config.get("status") == "frozen_before_source_replay", "config is not frozen")
    _require(source.get("dataset_id") == "fog_star_v3", "wrong dataset")
    _require(
        source.get("content_url")
        == "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content",
        "wrong source URL",
    )
    _require(source.get("declared_size_bytes") == 119_629_580, "wrong source size")
    _require(source.get("declared_md5") == "952a37ab147da35e6d4e7a1e9bac44cb", "wrong MD5")
    _require(
        source.get("expected_sha256")
        == "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477",
        "wrong source SHA-256",
    )
    _require(source.get("participant_roster") == expected_roster, "wrong roster")
    _require(source.get("raw_local_mirror") is False, "raw mirror must remain disabled")
    _require(
        reference.get("run_id") == "fog-rf-feature-weight-factorial-seed11-20260907-001",
        "wrong reference run",
    )
    expected_counts = _mapping(reference.get("expected_counts"), "expected_counts")
    _require(
        expected_counts
        == {
            "participants": 22,
            "sessions": 31,
            "physical_segments": 33,
            "finite_source_samples": 300669,
            "omitted_source_rows": 28358,
            "resampled_samples": 250571,
            "dropped_tail_samples": 2379,
            "observable_candidates": 1939,
            "missing_or_unsupported_candidates": 226,
            "mixed_annotation_candidates": 500,
            "scored_windows": 1213,
            "class_windows": [954, 74, 185],
        },
        "expected count contract changed",
    )
    expected_reconstruction: dict[str, object] = {
        "source_rate_hz": 60.0,
        "target_rate_hz": 50.0,
        "window_samples": 128,
        "maximum_gap_factor": 3.0,
        "gravity_cutoff_hz": 0.30,
        "acceleration_scale_m_s2_per_g": 9.80665,
        "gyroscope_scale_rad_per_deg": math.pi / 180.0,
        "label_map": {1: 0, 2: 1, 3: 2, 6: 0, 7: 0},
        "homogeneity_basis": "original_provider_activity_code_before_mapping",
        "segment_basis": "participant_session_timestamp_gap_nonfinite_run",
        "gravity_initial_state": "first_total_acceleration_sample_of_physical_segment",
        "transform_order": (
            "si_conversion_then_native_rate_gravity_then_single_joint_polyphase_resample"
        ),
        "candidate_materialization_dtype": "float32",
        "selected_descriptors": list(DESCRIPTOR_NAMES),
        "outer_fold_seed": 11,
        "outer_fold_count": 5,
    }
    _require(reconstruction == expected_reconstruction, "reconstruction contract changed")
    _require(
        tolerances
        == {
            "descriptor_absolute": 1e-12,
            "descriptor_relative": 1e-12,
            "segment_float64_exact_hash_required": True,
            "candidate_float32_exact_hash_required": True,
            "structural_exact_required": True,
        },
        "numerical tolerances changed",
    )
    _require(
        resources
        == {
            "controller_count": 1,
            "engineering_wall_seconds": 3600,
            "compute_wall_seconds": 1800,
            "model_fit_budget": 0,
            "raw_source_written_to_disk": False,
            "automatic_follow_on_allowed": False,
        },
        "resource contract changed",
    )
    _require(all(value is False for value in claims.values()), "claim policy changed")


def independent_gravity(
    total_acceleration: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
    cutoff_hz: float,
) -> FloatArray:
    """First-order causal gravity implementation kept separate from production."""

    values = np.asarray(total_acceleration, dtype=np.float64)
    _require(values.ndim == 2 and values.shape[0] > 0 and values.shape[1] == 3, "bad gravity input")
    _require(bool(np.isfinite(values).all()), "gravity input is not finite")
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sampling_rate_hz)
    result = np.empty_like(values)
    result[0] = values[0]
    for index in range(1, values.shape[0]):
        result[index] = result[index - 1] + alpha * (values[index] - result[index - 1])
    return np.asarray(result, dtype=np.float64)


def independent_true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=np.bool_)
    padded = np.concatenate((np.asarray([False]), values, np.asarray([False]))).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return [(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)]


def independent_physical_runs(
    timestamps: FloatArray,
    signals: FloatArray,
    *,
    nominal_rate_hz: float,
    maximum_gap_factor: float,
) -> list[tuple[int, int]]:
    """Discover finite, monotonic, gap-bounded runs without annotations."""

    _require(
        timestamps.ndim == 1 and signals.shape == (timestamps.size, 6),
        "timestamp/signal arrays are not aligned",
    )
    finite = np.isfinite(timestamps) & np.isfinite(signals).all(axis=1)
    output: list[tuple[int, int]] = []
    for finite_start, finite_stop in independent_true_runs(finite):
        values = timestamps[finite_start:finite_stop]
        if values.size == 0:
            continue
        boundaries = np.ones(values.size, dtype=np.bool_)
        if values.size > 1:
            delta = np.diff(values)
            boundaries[1:] = (delta <= 0.0) | (delta > maximum_gap_factor / nominal_rate_hz)
        starts = np.flatnonzero(boundaries)
        stops = np.concatenate((starts[1:], np.asarray([values.size], dtype=np.int64)))
        output.extend(
            (finite_start + int(start), finite_start + int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )
    return output


def independent_si_channels(frame: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    """Convert the provider back sensor columns with the frozen operation order."""

    total = frame[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(dtype=np.float64) * 9.80665
    gyroscope = (
        frame[["back_gyro_x", "back_gyro_y", "back_gyro_z"]].to_numpy(dtype=np.float64)
        * np.pi
        / 180.0
    )
    return np.asarray(total, dtype=np.float64), np.asarray(gyroscope, dtype=np.float64)


def independent_resample_segment(
    *,
    timestamps: FloatArray,
    total_acceleration: FloatArray,
    gyroscope: FloatArray,
    source_rate_hz: float,
    target_rate_hz: float,
    gravity_cutoff_hz: float,
    window_samples: int,
) -> dict[str, NDArray[Any]]:
    """Independently apply the frozen native-rate filter and one joint resample."""

    gravity = independent_gravity(
        total_acceleration,
        sampling_rate_hz=source_rate_hz,
        cutoff_hz=gravity_cutoff_hz,
    )
    linear = np.asarray(total_acceleration, dtype=np.float64) - gravity
    combined = np.column_stack((linear, np.asarray(gyroscope, dtype=np.float64), gravity))
    ratio = Fraction(str(target_rate_hz / source_rate_hz)).limit_denominator(10_000)
    expected_count = (
        combined.shape[0] * ratio.numerator + ratio.denominator - 1
    ) // ratio.denominator
    resampled = np.asarray(
        resample_poly(combined, ratio.numerator, ratio.denominator, axis=0),
        dtype=np.float64,
    )
    _require(resampled.shape == (expected_count, 9), "independent resample length changed")
    grid = timestamps[0] + np.arange(resampled.shape[0], dtype=np.float64) / target_rate_hz
    keep = grid < timestamps[-1] + 1.0 / source_rate_hz
    grid = np.asarray(grid[keep], dtype=np.float64)
    resampled = np.asarray(resampled[keep], dtype=np.float64)
    starts = np.arange(grid.size // window_samples, dtype=np.int64) * window_samples
    return {
        "source_timestamps": np.asarray(timestamps, dtype=np.float64).copy(),
        "timestamps": grid,
        "signals": np.asarray(resampled[:, :6], dtype=np.float64),
        "gravity": np.asarray(resampled[:, 6:], dtype=np.float64),
        "candidate_starts": starts,
    }


def independent_annotation_admission(
    *,
    source_timestamps: FloatArray,
    source_labels: FloatArray,
    target_timestamps: FloatArray,
    candidate_starts: IntArray,
    target_rate_hz: float,
    window_samples: int,
) -> tuple[list[int], dict[str, list[int]], list[int]]:
    """Project annotations after grid construction using original-code homogeneity."""

    source_indices = np.searchsorted(source_timestamps, target_timestamps, side="right") - 1
    projected = source_labels[np.clip(source_indices, 0, source_labels.size - 1)]
    admitted: list[int] = []
    rejected: dict[str, list[int]] = {
        "missing_or_unsupported_annotation": [],
        "mixed_annotation": [],
    }
    admitted_labels: list[int] = []
    for window_index, start in enumerate(candidate_starts.tolist()):
        stop = start + window_samples
        start_time = target_timestamps[start]
        stop_time = start_time + window_samples / target_rate_hz
        left = max(
            0,
            int(np.searchsorted(source_timestamps, start_time, side="right")) - 1,
        )
        right = int(np.searchsorted(source_timestamps, stop_time, side="left"))
        annotations = np.concatenate((projected[start:stop], source_labels[left:right]))
        supported = bool(
            np.isfinite(annotations).all()
            and all(value in LABEL_MAP for value in np.unique(annotations))
        )
        if not supported:
            rejected["missing_or_unsupported_annotation"].append(window_index)
            continue
        if bool(np.any(annotations != annotations[0])):
            rejected["mixed_annotation"].append(window_index)
            continue
        admitted.append(window_index)
        admitted_labels.append(LABEL_MAP[int(annotations[0])])
    return admitted, rejected, admitted_labels


def independent_magnitude_descriptors(windows: Float32Array) -> FloatArray:
    """Compute the four frozen descriptors without the production extractor."""

    values = np.asarray(windows, dtype=np.float64)
    _require(values.ndim == 3 and values.shape[1:] == (128, 6), "bad candidate tensor")
    accelerometer_norm = np.sqrt(np.sum(values[:, :, :3] ** 2, axis=2))
    gyroscope_norm = np.sqrt(np.sum(values[:, :, 3:6] ** 2, axis=2))
    return np.column_stack(
        (
            np.sqrt(np.mean(accelerometer_norm**2, axis=1)),
            np.std(accelerometer_norm, axis=1, ddof=0),
            np.sqrt(np.mean(gyroscope_norm**2, axis=1)),
            np.std(gyroscope_norm, axis=1, ddof=0),
        )
    ).astype(np.float64, copy=False)


def independent_outer_assignment(roster: Sequence[str], *, seed: int, count: int) -> dict[str, int]:
    participants = sorted(set(roster))
    order = np.random.default_rng(seed).permutation(len(participants))
    return {participants[int(index)]: position % count for position, index in enumerate(order)}


def _download_source(
    url: str,
    *,
    deadline: float,
    timeout_seconds: float = 180.0,
) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        _require(time.perf_counter() < deadline, "compute cap reached before source download")
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.5 * 2**attempt)
    raise RuntimeError("failed to stream pinned FoG source after retries") from last_error


def _verify_reference(
    *,
    repository_root: Path,
    reference_run: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    reference = _mapping(config["reference_run"], "reference_run")
    _require(reference_run.is_dir(), "reference run directory does not exist")
    _require(reference_run.name == reference["run_id"], "reference run id mismatch")
    files = _mapping(reference["files"], "reference files")
    file_checks: dict[str, bool] = {}
    for name, expected in files.items():
        path = reference_run / name
        file_checks[name] = path.is_file() and sha256_file(path) == str(expected)
    _require(all(file_checks.values()), "reference file hash mismatch")
    manifest = _read_json(reference_run / "artifact_manifest.json")
    _verify_sealed(manifest, "reference artifact manifest")
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    manifest_checks: dict[str, bool] = {}
    for record in artifacts:
        relative = str(record["path"])
        path = (reference_run / relative).resolve()
        manifest_checks[relative] = (
            _is_relative_to(path, reference_run)
            and path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(record["size_bytes"])
            and sha256_file(path) == str(record["sha256"])
        )
    _require(all(manifest_checks.values()), "reference artifact manifest verification failed")
    preflight = _read_json(reference_run / "preflight.json")
    _verify_sealed(preflight, "reference preflight")
    source_manifest = _mapping(preflight["source_manifest"], "reference source manifest")
    source_checks: dict[str, bool] = {}
    for record in cast(list[dict[str, Any]], source_manifest["files"]):
        relative = str(record["path"])
        path = (repository_root / relative).resolve()
        source_checks[relative] = (
            _is_relative_to(path, repository_root)
            and path.is_file()
            and sha256_file(path) == str(record["sha256"])
        )
    _require(all(source_checks.values()), "reference source code no longer matches frozen files")
    for name in ("data_audit.json", "feature_cache_metadata.json", "source_receipt.json"):
        _verify_sealed(_read_json(reference_run / name), f"reference {name}")
    return {
        "reference_file_checks": file_checks,
        "artifact_manifest_entry_count": len(artifacts),
        "artifact_manifest_checks_passed": all(manifest_checks.values()),
        "source_manifest_file_count": len(source_checks),
        "source_manifest_checks": source_checks,
        "reference_validation_status": _read_json(reference_run / "validation.json").get("status"),
    }


def _snapshot_inputs(
    *,
    repository_root: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
) -> None:
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    _write_bytes_create_only(output_directory / "protocol_snapshot.md", protocol_path.read_bytes())
    code_path = Path(__file__).resolve()
    preflight = _sealed(
        {
            "record_kind": "fog_raw_cache_integrity_preflight",
            "created_at_utc": _now(),
            "outcomes_available_when_written": False,
            "zero_model_fits": True,
            "raw_source_written_to_disk": False,
            "implementation_independence": {
                "production_loader_called": False,
                "production_gravity_or_resampling_called": False,
                "production_feature_extractor_called": False,
                "estimator_imported_or_called": False,
                "shared_numeric_dependency": "scipy.signal.resample_poly",
            },
            "source_manifest": {
                "audit_code": {
                    "path": code_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(code_path),
                },
                "config": {
                    "path": config_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(config_path),
                },
                "protocol": {
                    "path": protocol_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(protocol_path),
                },
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
            },
        }
    )
    _write_json_create_only(output_directory / "preflight.json", preflight)


def _comparison(name: str, actual: object, expected: object) -> dict[str, Any]:
    return {"name": name, "passed": actual == expected, "actual": actual, "expected": expected}


def _segment_compare(
    segment: Mapping[str, NDArray[Any]],
    *,
    participant: str,
    session: str,
    trial: str,
    run_index: int,
    admitted: list[int],
    rejected: dict[str, list[int]],
    reference: Mapping[str, Any] | None,
    source_rate_hz: float,
    target_rate_hz: float,
    window_samples: int,
) -> dict[str, Any]:
    timestamps = np.asarray(segment["timestamps"], dtype=np.float64)
    starts = np.asarray(segment["candidate_starts"], dtype=np.int64)
    actual = {
        "protocol_id": "external-har-session-grid-v3",
        "source_samples": int(np.asarray(segment["source_timestamps"]).size),
        "resampled_samples": int(timestamps.size),
        "source_rate_hz": source_rate_hz,
        "target_rate_hz": target_rate_hz,
        "window_samples": window_samples,
        "candidate_start_samples": starts.tolist(),
        "candidate_window_count": int(starts.size),
        "dropped_tail_samples": int(timestamps.size % window_samples),
        "start_timestamp": float(timestamps[0]),
        "last_timestamp": float(timestamps[-1]),
        "source_timestamps_sha256": _external_array_sha256(
            np.asarray(segment["source_timestamps"], dtype=np.float64)
        ),
        "timestamps_sha256": _external_array_sha256(timestamps),
        "signals_sha256": _external_array_sha256(np.asarray(segment["signals"], dtype=np.float64)),
        "gravity_sha256": _external_array_sha256(np.asarray(segment["gravity"], dtype=np.float64)),
        "candidate_grid_sha256": _external_array_sha256(starts),
        "resampling_passes": 1,
        "within_declared_segment_transform_annotation_dependency": False,
        "participant_id": participant,
        "session_id": session,
        "trial_id": trial,
        "run_index": run_index,
        "segment_boundary_annotation_conditioned": False,
        "admitted_candidate_indices": admitted,
        "excluded_candidate_indices": rejected,
    }
    if reference is None:
        return {
            "participant_id": participant,
            "session_id": session,
            "run_index": run_index,
            "status": "unresolved_missing_reference_segment",
            "checks": {},
            "actual": actual,
        }
    fields = tuple(actual)
    checks = {field: actual[field] == reference.get(field) for field in fields}
    return {
        "participant_id": participant,
        "session_id": session,
        "run_index": run_index,
        "status": (
            "no_defect_detected_under_frozen_contract"
            if all(checks.values())
            else "defect_detected"
        ),
        "checks": checks,
        "actual": actual,
        "reference": {field: reference.get(field) for field in fields},
    }


def _build_replay(
    *,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    reference_data_audit: Mapping[str, Any],
    deadline: float,
) -> dict[str, Any]:
    reconstruction = _mapping(config["reconstruction"], "reconstruction")
    source_rate = float(reconstruction["source_rate_hz"])
    target_rate = float(reconstruction["target_rate_hz"])
    window_samples = int(reconstruction["window_samples"])
    gap_factor = float(reconstruction["maximum_gap_factor"])
    cutoff = float(reconstruction["gravity_cutoff_hz"])
    raw_roster = [
        f"fogstar:{participant:03d}"
        for participant in sorted(frame["subjectID"].dropna().astype(int).unique().tolist())
    ]
    dataset = _mapping(reference_data_audit["dataset"], "reference dataset audit")
    reference_segments = cast(list[dict[str, Any]], dataset["preprocessing_audit"])
    reference_by_key = {
        (str(item["participant_id"]), str(item["session_id"]), int(item["run_index"])): item
        for item in reference_segments
    }
    seen_reference_keys: set[tuple[str, str, int]] = set()

    candidate_signals: list[Float32Array] = []
    candidate_gravity: list[Float32Array] = []
    candidate_people: list[str] = []
    candidate_sessions: list[str] = []
    candidate_trials: list[str] = []
    candidate_ids: list[str] = []
    scoring_indices: list[int] = []
    scored_labels: list[int] = []
    scored_people: list[str] = []
    scored_ids: list[str] = []
    segment_reports: list[dict[str, Any]] = []
    session_records: list[dict[str, Any]] = []
    processed_source_samples = 0
    omitted_source_rows = 0
    resampled_samples = 0
    dropped_tail_samples = 0
    missing_count = 0
    mixed_count = 0

    grouped = frame.groupby(["subjectID", "sessionID"], sort=True, dropna=False)
    for keys, trial_frame in grouped:
        _require(time.perf_counter() < deadline, "compute cap reached during raw replay")
        subject, session_value = cast(tuple[Any, Any], keys)
        if pd.isna(subject) or pd.isna(session_value):
            session_records.append(
                {
                    "participant_id": None if pd.isna(subject) else f"fogstar:{int(subject):03d}",
                    "session_id": None,
                    "raw_row_count": int(trial_frame.shape[0]),
                    "status": "unresolved_missing_participant_or_session_identifier",
                }
            )
            continue
        participant = f"fogstar:{int(subject):03d}"
        session = f"{participant}:session-{int(session_value):03d}"
        trial = f"{session}:recording"
        timestamps = trial_frame["timestamp"].to_numpy(dtype=np.float64)
        activities = trial_frame["activity"].to_numpy(dtype=np.float64)
        total, gyro = independent_si_channels(trial_frame)
        signals = np.column_stack((total, gyro))
        physical_runs = independent_physical_runs(
            timestamps,
            signals,
            nominal_rate_hz=source_rate,
            maximum_gap_factor=gap_factor,
        )
        omitted = int(timestamps.size - sum(stop - start for start, stop in physical_runs))
        omitted_source_rows += omitted
        session_candidate_start = len(candidate_ids)
        session_score_start = len(scoring_indices)
        session_segment_start = len(segment_reports)
        short_runs: list[dict[str, int]] = []
        for run_index, (physical_start, physical_stop) in enumerate(physical_runs):
            run_size = physical_stop - physical_start
            if run_size < 3:
                short_runs.append({"run_index": run_index, "source_rows": run_size})
                continue
            segment = independent_resample_segment(
                timestamps=timestamps[physical_start:physical_stop],
                total_acceleration=total[physical_start:physical_stop],
                gyroscope=gyro[physical_start:physical_stop],
                source_rate_hz=source_rate,
                target_rate_hz=target_rate,
                gravity_cutoff_hz=cutoff,
                window_samples=window_samples,
            )
            processed_source_samples += run_size
            grid = np.asarray(segment["timestamps"], dtype=np.float64)
            starts = np.asarray(segment["candidate_starts"], dtype=np.int64)
            segment_signals = np.asarray(segment["signals"], dtype=np.float64)
            segment_gravity = np.asarray(segment["gravity"], dtype=np.float64)
            resampled_samples += int(grid.size)
            dropped_tail_samples += int(grid.size % window_samples)
            source_times = timestamps[physical_start:physical_stop]
            source_labels = activities[physical_start:physical_stop]
            admitted, rejected, admitted_labels = independent_annotation_admission(
                source_timestamps=source_times,
                source_labels=source_labels,
                target_timestamps=grid,
                candidate_starts=starts,
                target_rate_hz=target_rate,
                window_samples=window_samples,
            )
            admitted_by_index = dict(zip(admitted, admitted_labels, strict=True))
            for window_index, start in enumerate(starts.tolist()):
                stop = start + window_samples
                identifier = (
                    f"{participant}/{session}/{trial}/run-{run_index:04d}-"
                    f"finite-000/window-{window_index:06d}"
                )
                candidate_signals.append(np.asarray(segment_signals[start:stop], dtype=np.float32))
                candidate_gravity.append(np.asarray(segment_gravity[start:stop], dtype=np.float32))
                candidate_people.append(participant)
                candidate_sessions.append(session)
                candidate_trials.append(trial)
                candidate_ids.append(identifier)
                if window_index in rejected["missing_or_unsupported_annotation"]:
                    missing_count += 1
                    continue
                if window_index in rejected["mixed_annotation"]:
                    mixed_count += 1
                    continue
                scoring_indices.append(len(candidate_ids) - 1)
                scored_labels.append(admitted_by_index[window_index])
                scored_people.append(participant)
                scored_ids.append(identifier)
            key = (participant, session, run_index)
            reference_segment = reference_by_key.get(key)
            if reference_segment is not None:
                seen_reference_keys.add(key)
            segment_reports.append(
                _segment_compare(
                    segment,
                    participant=participant,
                    session=session,
                    trial=trial,
                    run_index=run_index,
                    admitted=admitted,
                    rejected=rejected,
                    reference=reference_segment,
                    source_rate_hz=source_rate,
                    target_rate_hz=target_rate,
                    window_samples=window_samples,
                )
            )
        these_segments = segment_reports[session_segment_start:]
        session_records.append(
            {
                "participant_id": participant,
                "session_id": session,
                "trial_id": trial,
                "raw_row_count": int(timestamps.size),
                "omitted_nonfinite_source_rows": omitted,
                "discovered_physical_run_count": len(physical_runs),
                "short_physical_runs": short_runs,
                "processed_segment_count": len(these_segments),
                "processed_source_samples": sum(
                    int(item["actual"]["source_samples"]) for item in these_segments
                ),
                "resampled_samples": sum(
                    int(item["actual"]["resampled_samples"]) for item in these_segments
                ),
                "dropped_tail_samples": sum(
                    int(item["actual"]["dropped_tail_samples"]) for item in these_segments
                ),
                "observable_candidate_count": len(candidate_ids) - session_candidate_start,
                "scored_window_count": len(scoring_indices) - session_score_start,
                "missing_or_unsupported_candidates": sum(
                    len(
                        cast(
                            list[int],
                            item["actual"]["excluded_candidate_indices"][
                                "missing_or_unsupported_annotation"
                            ],
                        )
                    )
                    for item in these_segments
                ),
                "mixed_annotation_candidates": sum(
                    len(
                        cast(
                            list[int],
                            item["actual"]["excluded_candidate_indices"]["mixed_annotation"],
                        )
                    )
                    for item in these_segments
                ),
                "segment_statuses": [str(item["status"]) for item in these_segments],
                "status": (
                    "no_defect_detected_under_frozen_contract"
                    if these_segments
                    and all(
                        item["status"] == "no_defect_detected_under_frozen_contract"
                        for item in these_segments
                    )
                    else (
                        "defect_detected"
                        if any(item["status"] == "defect_detected" for item in these_segments)
                        else "unresolved_no_processed_segment"
                    )
                ),
            }
        )

    _require(bool(candidate_signals), "raw replay produced no observable candidates")
    signals_array = np.stack(candidate_signals).astype(np.float32, copy=False)
    gravity_array = np.stack(candidate_gravity).astype(np.float32, copy=False)
    people_array = np.asarray(candidate_people, dtype=np.str_)
    sessions_array = np.asarray(candidate_sessions, dtype=np.str_)
    trials_array = np.asarray(candidate_trials, dtype=np.str_)
    ids_array = np.asarray(candidate_ids, dtype=np.str_)
    scoring_array = np.asarray(scoring_indices, dtype=np.int64)
    eligibility = np.zeros(ids_array.size, dtype=np.bool_)
    eligibility[scoring_array] = True
    labels_array = np.asarray(scored_labels, dtype=np.int64)
    scored_people_array = np.asarray(scored_people, dtype=np.str_)
    scored_ids_array = np.asarray(scored_ids, dtype=np.str_)
    descriptors = independent_magnitude_descriptors(signals_array)
    missing_reference = sorted(set(reference_by_key) - seen_reference_keys)

    return {
        "raw_roster": raw_roster,
        "signals": signals_array,
        "gravity": gravity_array,
        "people": people_array,
        "sessions": sessions_array,
        "trials": trials_array,
        "window_ids": ids_array,
        "scoring_indices": scoring_array,
        "eligibility": eligibility,
        "labels": labels_array,
        "scored_people": scored_people_array,
        "scored_window_ids": scored_ids_array,
        "descriptors": descriptors,
        "segment_reports": segment_reports,
        "session_records": session_records,
        "missing_reference_segments": [
            {"participant_id": key[0], "session_id": key[1], "run_index": key[2]}
            for key in missing_reference
        ],
        "counts": {
            "participants": len(raw_roster),
            "sessions": len(session_records),
            "physical_segments": len(segment_reports),
            "finite_source_samples": processed_source_samples,
            "omitted_source_rows": omitted_source_rows,
            "resampled_samples": resampled_samples,
            "dropped_tail_samples": dropped_tail_samples,
            "observable_candidates": int(ids_array.size),
            "missing_or_unsupported_candidates": missing_count,
            "mixed_annotation_candidates": mixed_count,
            "scored_windows": int(scoring_array.size),
            "class_windows": np.bincount(labels_array, minlength=3).astype(int).tolist(),
        },
    }


def _compare_replay(
    *,
    replay: Mapping[str, Any],
    config: Mapping[str, Any],
    reference_data_audit: Mapping[str, Any],
    feature_metadata: Mapping[str, Any],
    cache: Mapping[str, NDArray[Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source = _mapping(config["source"], "source")
    reference = _mapping(config["reference_run"], "reference_run")
    expected_counts = _mapping(reference["expected_counts"], "expected_counts")
    expected_hashes = _mapping(reference["expected_hashes"], "expected hashes")
    tolerances = _mapping(config["numerical_tolerances"], "tolerances")
    signals = np.asarray(replay["signals"], dtype=np.float32)
    gravity = np.asarray(replay["gravity"], dtype=np.float32)
    ids = np.asarray(replay["window_ids"], dtype=np.str_)
    people = np.asarray(replay["people"], dtype=np.str_)
    scoring = np.asarray(replay["scoring_indices"], dtype=np.int64)
    eligibility = np.asarray(replay["eligibility"], dtype=np.bool_)
    labels = np.asarray(replay["labels"], dtype=np.int64)
    scored_people = np.asarray(replay["scored_people"], dtype=np.str_)
    scored_ids = np.asarray(replay["scored_window_ids"], dtype=np.str_)
    descriptors = np.asarray(replay["descriptors"], dtype=np.float64)
    cache_names = np.asarray(cache["six_names"], dtype=np.str_)
    cache_values = np.asarray(cache["six_values"], dtype=np.float64)
    descriptor_indices = [int(np.flatnonzero(cache_names == name)[0]) for name in DESCRIPTOR_NAMES]
    cached_descriptors = cache_values[:, descriptor_indices]
    residual = np.abs(descriptors - cached_descriptors)
    close = np.isclose(
        descriptors,
        cached_descriptors,
        rtol=float(tolerances["descriptor_relative"]),
        atol=float(tolerances["descriptor_absolute"]),
    )
    assignment = independent_outer_assignment(
        cast(list[str], replay["raw_roster"]),
        seed=int(_mapping(config["reconstruction"], "reconstruction")["outer_fold_seed"]),
        count=int(_mapping(config["reconstruction"], "reconstruction")["outer_fold_count"]),
    )
    observable_fold = np.asarray([assignment[str(item)] for item in people], dtype=np.int64)
    scored_fold = observable_fold[scoring]
    dataset = _mapping(reference_data_audit["dataset"], "reference dataset")
    pool = _mapping(dataset["observable_candidate_pool"], "observable pool")
    pool_hashes = _mapping(pool["arrays"], "observable pool arrays")

    checks = [
        _comparison("raw_roster", replay["raw_roster"], source["participant_roster"]),
        _comparison("aggregate_counts", replay["counts"], expected_counts),
        _comparison("missing_reference_segments", replay["missing_reference_segments"], []),
        _comparison(
            "all_segment_replays",
            all(
                item["status"] == "no_defect_detected_under_frozen_contract"
                for item in cast(list[dict[str, Any]], replay["segment_reports"])
            ),
            True,
        ),
        _comparison(
            "candidate_signal_external_hash",
            _external_array_sha256(signals),
            pool_hashes["signals"],
        ),
        _comparison(
            "candidate_gravity_external_hash",
            _external_array_sha256(gravity),
            pool_hashes["gravity"],
        ),
        _comparison(
            "candidate_nine_external_hash",
            _external_array_sha256(np.concatenate((signals, gravity), axis=2)),
            pool_hashes["nine_channel_signals"],
        ),
        _comparison(
            "candidate_signal_frozen_hash",
            _external_array_sha256(signals),
            expected_hashes["candidate_signals"],
        ),
        _comparison(
            "candidate_gravity_frozen_hash",
            _external_array_sha256(gravity),
            expected_hashes["candidate_gravity"],
        ),
        _comparison(
            "candidate_ids_exact",
            ids.tolist(),
            np.asarray(cache["observable_window_ids"], dtype=np.str_).tolist(),
        ),
        _comparison(
            "candidate_people_exact",
            people.tolist(),
            np.asarray(cache["observable_participant_ids"], dtype=np.str_).tolist(),
        ),
        _comparison(
            "candidate_ids_cache_hash",
            _cache_array_sha256(ids),
            expected_hashes["observable_window_ids"],
        ),
        _comparison(
            "scoring_indices_exact",
            scoring.tolist(),
            np.asarray(cache["scoring_indices"], dtype=np.int64).tolist(),
        ),
        _comparison(
            "scoring_indices_hash",
            _cache_array_sha256(scoring),
            expected_hashes["scoring_indices"],
        ),
        _comparison(
            "eligibility_exact",
            eligibility.tolist(),
            np.asarray(cache["scoring_eligibility"], dtype=np.bool_).tolist(),
        ),
        _comparison(
            "eligibility_hash",
            _cache_array_sha256(eligibility),
            expected_hashes["scoring_eligibility"],
        ),
        _comparison(
            "scored_labels_exact",
            labels.tolist(),
            np.asarray(cache["scored_labels"], dtype=np.int64).tolist(),
        ),
        _comparison(
            "scored_labels_hash",
            _cache_array_sha256(labels),
            expected_hashes["scored_labels"],
        ),
        _comparison(
            "scored_people_exact",
            scored_people.tolist(),
            np.asarray(cache["scored_participant_ids"], dtype=np.str_).tolist(),
        ),
        _comparison(
            "scored_window_ids_exact",
            scored_ids.tolist(),
            np.asarray(cache["scored_window_ids"], dtype=np.str_).tolist(),
        ),
        _comparison(
            "observable_fold_exact",
            observable_fold.tolist(),
            np.asarray(cache["observable_fold_index"], dtype=np.int64).tolist(),
        ),
        _comparison(
            "scored_fold_exact",
            scored_fold.tolist(),
            np.asarray(cache["scored_fold_index"], dtype=np.int64).tolist(),
        ),
        _comparison(
            "descriptor_names", list(DESCRIPTOR_NAMES), cache_names[descriptor_indices].tolist()
        ),
        _comparison("descriptor_all_within_tolerance", bool(close.all()), True),
    ]
    compact_checks = [
        {
            **item,
            **(
                {
                    "actual_count": len(item["actual"]),
                    "expected_count": len(item["expected"]),
                    "actual": None,
                    "expected": None,
                }
                if isinstance(item["actual"], list)
                and isinstance(item["expected"], list)
                and len(item["actual"]) > 50
                else {}
            ),
        }
        for item in checks
    ]
    comparison = _sealed(
        {
            "record_kind": "fog_raw_cache_comparison",
            "checks": compact_checks,
            "all_checks_passed": all(bool(item["passed"]) for item in checks),
            "descriptor_comparison": {
                "names": list(DESCRIPTOR_NAMES),
                "shape": list(descriptors.shape),
                "absolute_tolerance": tolerances["descriptor_absolute"],
                "relative_tolerance": tolerances["descriptor_relative"],
                "mismatch_value_count": int((~close).sum()),
                "maximum_absolute_difference": float(residual.max(initial=0.0)),
                "per_descriptor_maximum_absolute_difference": residual.max(axis=0).tolist(),
                "independent_values_cache_hash": _cache_array_sha256(descriptors),
                "reference_values_cache_hash": _cache_array_sha256(cached_descriptors),
            },
            "candidate_hashes": {
                "signals_external_convention": _external_array_sha256(signals),
                "gravity_external_convention": _external_array_sha256(gravity),
                "nine_channel_external_convention": _external_array_sha256(
                    np.concatenate((signals, gravity), axis=2)
                ),
                "window_ids_cache_convention": _cache_array_sha256(ids),
            },
            "scope_limit": (
                "four preselected magnitude descriptors and all upstream signal/gravity "
                "arrays were checked; unselected engineered feature columns were not "
                "independently recomputed"
            ),
        }
    )

    p7 = scored_people == "fogstar:007"
    p7_rows: list[dict[str, Any]] = []
    for label, class_name in enumerate(("mobility", "sitting", "standing")):
        selected_scored = p7 & (labels == label)
        observable_rows = scoring[selected_scored]
        p7_rows.append(
            {
                "class_name": class_name,
                "scored_window_count": int(selected_scored.sum()),
                "independent_descriptor_medians": (
                    np.median(descriptors[observable_rows], axis=0).tolist()
                    if observable_rows.size
                    else None
                ),
                "cache_descriptor_medians": (
                    np.median(cached_descriptors[observable_rows], axis=0).tolist()
                    if observable_rows.size
                    else None
                ),
                "maximum_absolute_difference": (
                    float(residual[observable_rows].max()) if observable_rows.size else None
                ),
            }
        )
    p7_report = _sealed(
        {
            "record_kind": "fog_p7_raw_descriptor_confirmation",
            "participant_id": "fogstar:007",
            "descriptor_names": list(DESCRIPTOR_NAMES),
            "source": "independent reconstruction from pinned raw bytes",
            "modalities_checked": ["linear_acceleration", "gyroscope"],
            "classes": p7_rows,
            "interpretation": (
                "The audit checks whether the cached amplitude pattern survives raw replay. "
                "It does not identify physiology, sensor condition, or annotation correctness."
            ),
        }
    )
    return comparison, p7_report


def _write_artifact_manifest(output_directory: Path) -> dict[str, Any]:
    paths = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    )
    manifest = _sealed(
        {
            "record_kind": "fog_raw_cache_integrity_artifact_manifest",
            "artifacts": [
                {
                    "path": path.relative_to(output_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in paths
            ],
        }
    )
    _write_json_create_only(output_directory / "artifact_manifest.json", manifest)
    return manifest


def run_audit(
    *,
    repository_root: Path,
    evidence_root: Path,
    reference_run: Path,
    config_path: Path,
    protocol_path: Path,
    output_directory: Path,
    code_commit: str,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Execute the frozen audit with zero model fits."""

    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    reference_run = reference_run.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    output_directory = output_directory.resolve()
    config = _read_yaml(config_path)
    validate_config(config)
    _require(config_path.parent.parent.parent == repository_root, "config must be in repository")
    _require(
        protocol_path.parent.parent.parent == repository_root, "protocol must be in repository"
    )
    _require(
        output_directory.parent == evidence_root / ".audit" / "fog_raw_cache_integrity",
        "output must be a create-only run under the dedicated evidence root",
    )
    _require(not output_directory.exists(), "output directory already exists")
    state = _git_state(repository_root)
    _require(state["clean"] is True, "audit source worktree must be clean")
    _require(state["commit"] == code_commit, "code commit does not match HEAD")
    _require(timeout_seconds == 1800, "compute cap must remain 1800 seconds")
    output_directory.mkdir(parents=True, exist_ok=False)
    started_at = _now()
    started = time.perf_counter()
    deadline = started + timeout_seconds
    _snapshot_inputs(
        repository_root=repository_root,
        config_path=config_path,
        protocol_path=protocol_path,
        output_directory=output_directory,
    )
    reference_verification: dict[str, Any] | None = None
    try:
        reference_verification = _verify_reference(
            repository_root=repository_root,
            reference_run=reference_run,
            config=config,
        )
        source = _mapping(config["source"], "source")
        payload = _download_source(str(source["content_url"]), deadline=deadline)
        source_size = len(payload)
        source_md5 = hashlib.md5(payload).hexdigest()
        source_sha256 = hashlib.sha256(payload).hexdigest()
        _require(source_size == int(source["declared_size_bytes"]), "source byte count mismatch")
        _require(source_md5 == source["declared_md5"], "source MD5 mismatch")
        _require(source_sha256 == source["expected_sha256"], "source SHA-256 mismatch")
        _write_json_create_only(
            output_directory / "source_receipt.json",
            _sealed(
                {
                    "record_kind": "fog_raw_cache_integrity_source_receipt",
                    "accessed_at_utc": _now(),
                    "dataset_id": source["dataset_id"],
                    "provider_record": source["provider_record"],
                    "locator": source["content_url"],
                    "received_size_bytes": source_size,
                    "computed_md5": source_md5,
                    "computed_sha256": source_sha256,
                    "source_hash_match": True,
                    "raw_local_mirror": False,
                    "verified_before_signal_parse": True,
                }
            ),
        )
        frame = pd.read_csv(io.BytesIO(payload), usecols=list(SOURCE_COLUMNS))
        del payload
        reference_data_audit = _read_json(reference_run / "data_audit.json")
        feature_metadata = _read_json(reference_run / "feature_cache_metadata.json")
        _verify_sealed(reference_data_audit, "reference data audit")
        _verify_sealed(feature_metadata, "reference feature metadata")
        replay = _build_replay(
            frame=frame,
            config=config,
            reference_data_audit=reference_data_audit,
            deadline=deadline,
        )
        del frame
        with np.load(reference_run / "feature_cache.npz", allow_pickle=False) as loaded:
            cache = {name: loaded[name] for name in loaded.files}
        comparison, p7_report = _compare_replay(
            replay=replay,
            config=config,
            reference_data_audit=reference_data_audit,
            feature_metadata=feature_metadata,
            cache=cache,
        )
        _write_json_create_only(output_directory / "raw_cache_comparison.json", comparison)
        _write_json_create_only(output_directory / "p7_descriptor_confirmation.json", p7_report)
        sessions = cast(list[dict[str, Any]], replay["session_records"])
        segment_reports = cast(list[dict[str, Any]], replay["segment_reports"])
        per_session = _sealed(
            {
                "record_kind": "fog_raw_cache_per_session_integrity",
                "raw_discovery_scope": "all provider participant/session groups",
                "session_count": len(sessions),
                "segment_count": len(segment_reports),
                "sessions": sessions,
                "segments": segment_reports,
                "all_sessions_retained": True,
                "status_counts": dict(
                    sorted(
                        (
                            status,
                            sum(item["status"] == status for item in sessions),
                        )
                        for status in sorted({str(item["status"]) for item in sessions})
                    )
                ),
                "semantic_qualification": {
                    "provider_timestamp_unit": "unresolved_documentation_discrepancy",
                    "fixed_numeric_interpretation": "seconds",
                    "annotation_adjudication": "unresolved_not_performed",
                    "clinical_cause": "unresolved_not_inferable",
                },
            }
        )
        _write_json_create_only(output_directory / "per_session_integrity.json", per_session)
        implementation_status = (
            "no_defect_detected_under_frozen_contract"
            if comparison["all_checks_passed"]
            and all(
                item["status"] == "no_defect_detected_under_frozen_contract" for item in sessions
            )
            else "defect_detected"
        )
        result = _sealed(
            {
                "record_kind": "fog_raw_cache_integrity_result",
                "audit_id": config["audit_id"],
                "status": "complete_awaiting_artifact_validation",
                "implementation_verdict": implementation_status,
                "semantic_verdict": "unresolved",
                "semantic_reasons": [
                    "provider_timestamp_documentation conflicts with released increments",
                    "activity annotations were not clinically adjudicated",
                    "signal agreement cannot identify physiology or device condition",
                ],
                "counts": replay["counts"],
                "all_raw_cache_checks_passed": comparison["all_checks_passed"],
                "reference_verification": reference_verification,
                "zero_model_fits": True,
                "model_fit_count": 0,
                "target_or_other_external_dataset_loaded": False,
                "automatic_follow_on_launched": False,
                "raw_local_mirror": False,
                "code_commit": code_commit,
            }
        )
        _write_json_create_only(output_directory / "result.json", result)
        elapsed = time.perf_counter() - started
        _write_json_create_only(
            output_directory / "runtime.json",
            _sealed(
                {
                    "record_kind": "fog_raw_cache_integrity_runtime",
                    "started_at_utc": started_at,
                    "completed_at_utc": _now(),
                    "wall_seconds": elapsed,
                    "compute_cap_seconds": timeout_seconds,
                    "within_compute_cap": elapsed <= timeout_seconds,
                    "controller_count": 1,
                    "model_fit_count": 0,
                }
            ),
        )
        _write_artifact_manifest(output_directory)
        return result
    except Exception as exc:
        elapsed = time.perf_counter() - started
        failure = _sealed(
            {
                "record_kind": "fog_raw_cache_integrity_failure",
                "status": "incomplete",
                "verdict": "unresolved",
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "reference_verification": reference_verification,
                "wall_seconds": elapsed,
                "model_fit_count": 0,
                "raw_local_mirror": False,
            }
        )
        _write_json_create_only(output_directory / "failure.json", failure)
        if not (output_directory / "runtime.json").exists():
            _write_json_create_only(
                output_directory / "runtime.json",
                _sealed(
                    {
                        "record_kind": "fog_raw_cache_integrity_runtime",
                        "started_at_utc": started_at,
                        "completed_at_utc": _now(),
                        "wall_seconds": elapsed,
                        "compute_cap_seconds": timeout_seconds,
                        "within_compute_cap": elapsed <= timeout_seconds,
                        "controller_count": 1,
                        "model_fit_count": 0,
                    }
                ),
            )
        _write_artifact_manifest(output_directory)
        raise


def validate_run(run_directory: Path) -> dict[str, Any]:
    """Validate the create-only evidence without network or raw data."""

    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "run directory does not exist")
    _require(not (run_directory / "failure.json").exists(), "failed audit is incomplete")
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion exists")
    manifest = _read_json(run_directory / "artifact_manifest.json")
    _verify_sealed(manifest, "artifact manifest")
    artifacts = cast(list[dict[str, Any]], manifest["artifacts"])
    declared = {str(item["path"]) for item in artifacts}
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    }
    _require(declared == actual, "artifact manifest coverage mismatch")
    _require(set(REQUIRED_RUN_FILES).issubset(declared), "required artifact missing")
    for item in artifacts:
        path = (run_directory / str(item["path"])).resolve()
        _require(_is_relative_to(path, run_directory), "artifact path escapes run")
        _require(
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size == int(item["size_bytes"])
            and sha256_file(path) == str(item["sha256"]),
            f"artifact mismatch: {item['path']}",
        )
    for name in (
        "preflight.json",
        "source_receipt.json",
        "per_session_integrity.json",
        "raw_cache_comparison.json",
        "p7_descriptor_confirmation.json",
        "result.json",
        "runtime.json",
    ):
        _verify_sealed(_read_json(run_directory / name), name)
    result = _read_json(run_directory / "result.json")
    comparison = _read_json(run_directory / "raw_cache_comparison.json")
    sessions = _read_json(run_directory / "per_session_integrity.json")
    runtime = _read_json(run_directory / "runtime.json")
    _require(
        result["implementation_verdict"] == "no_defect_detected_under_frozen_contract",
        "defect verdict cannot validate as clean",
    )
    _require(result["semantic_verdict"] == "unresolved", "semantic limitation was lost")
    _require(comparison["all_checks_passed"] is True, "raw/cache checks did not all pass")
    _require(
        all(
            item["status"] == "no_defect_detected_under_frozen_contract"
            for item in cast(list[dict[str, Any]], sessions["sessions"])
        ),
        "one or more sessions did not pass the frozen implementation checks",
    )
    _require(runtime["model_fit_count"] == 0, "model fit count changed")
    validation = _sealed(
        {
            "record_kind": "fog_raw_cache_integrity_validation",
            "status": "validated",
            "artifact_count": len(artifacts),
            "artifact_hashes_verified": True,
            "sealed_records_verified": True,
            "all_raw_cache_checks_passed": True,
            "all_sessions_accounted": sessions["session_count"] == 31,
            "implementation_verdict": result["implementation_verdict"],
            "semantic_verdict": result["semantic_verdict"],
            "network_or_raw_source_accessed": False,
            "model_fit_count": 0,
        }
    )
    _write_json_create_only(run_directory / "validation.json", validation)
    final_paths = sorted(path for path in run_directory.rglob("*") if path.is_file())
    completion = _sealed(
        {
            "record_kind": "fog_raw_cache_integrity_completion_manifest",
            "status": "complete_validated",
            "artifacts": [
                {
                    "path": path.relative_to(run_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in final_paths
            ],
            "implementation_verdict": result["implementation_verdict"],
            "semantic_verdict": result["semantic_verdict"],
            "model_fit_count": 0,
        }
    )
    _write_json_create_only(run_directory / "completion_manifest.json", completion)
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--repository-root", type=Path, required=True)
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--reference-run", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--protocol", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--code-commit", required=True)
    run.add_argument("--timeout-seconds", type=int, default=1800)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "run":
        result = run_audit(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            reference_run=arguments.reference_run,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            output_directory=arguments.output,
            code_commit=str(arguments.code_commit),
            timeout_seconds=int(arguments.timeout_seconds),
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    validation = validate_run(arguments.run_directory)
    print(json.dumps(validation, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
