"""Sealed fresh-cohort confirmation for the finalized CTGR native-nine predictor.

The lane has three deliberately separate phases:

* ``prepare`` verifies the frozen predictor and fixes the power calculation;
* ``predict`` accepts signal-only windows and seals label-blind probabilities;
* ``score`` opens separately supplied labels after the prediction seal exists.

No phase fits or selects a CTGR model.  Missing native gravity uses an exact B6
fallback and remains visible in the result.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from scipy.stats import nct, t  # type: ignore[import-untyped]
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]

from inclusive_shift_har.data.external_har import STANDARD_GRAVITY_M_S2 as STANDARD_GRAVITY_M_S2
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]

CLASS_NAMES = ("mobility", "sitting", "standing")
CORE_METHODS = ("B6", "B9", "T9", "U9")
SECONDARY_METHODS = ("HERA-v1", "RandomForest", "XGBoost", "PriorSoftRouter")
FROZEN_MODEL_UNITS = {"linear_acceleration": "g", "angular_velocity": "rad/s", "gravity": "g"}
PROHIBITED_ARCHIVE_KEYS = {
    "label",
    "labels",
    "target",
    "targets",
    "y",
    "y_true",
    "activity",
    "class",
    "classes",
}


class ConfirmationError(ValueError):
    """Raised when a frozen confirmation contract is violated."""


def _to_frozen_model_units(
    signals: FloatArray, gravity: FloatArray, units: Mapping[str, str]
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Bridge declared cohort units to the frozen g-trained model, without fitting.

    Native-gravity provenance and coordinate conventions remain separate cohort
    qualification requirements. Missing-gravity rows retain their availability mask.
    """
    si_units = {"linear_acceleration": "m/s^2", "angular_velocity": "rad/s", "gravity": "m/s^2"}
    if dict(units) == FROZEN_MODEL_UNITS:
        factor = 1.0
    elif dict(units) == si_units:
        factor = 1.0 / STANDARD_GRAVITY_M_S2
    else:
        raise ConfirmationError("cohort units cannot be converted to frozen model g/rad/s")
    converted = signals.copy()
    converted[:, :, :3] *= factor
    return (
        converted,
        np.asarray(gravity * factor, dtype=np.float64),
        {
            "input_units": dict(units),
            "model_units": dict(FROZEN_MODEL_UNITS),
            "acceleration_and_gravity_multiplier": factor,
            "gyroscope_multiplier": 1.0,
            "fitted": False,
            "axis_or_polarity_conversion": False,
        },
    )


def _validate_frozen_predictor(package: Path) -> dict[str, Any]:
    # Keep the heavyweight research stack out of power/scoring-only processes.
    from inclusive_shift_har.experiments.ctgr_source_finalization import validate_artifacts

    return validate_artifacts(package)


def _predict_native_windows(
    signals: FloatArray, native_gravity: FloatArray, predictor: dict[str, Any]
) -> dict[str, FloatArray]:
    from inclusive_shift_har.experiments.ctgr_source_finalization import predict_native_windows

    return predict_native_windows(signals, native_gravity, predictor)


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _code_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def git(*arguments: str) -> str:
        process = subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        return process.stdout.strip()

    return {
        "repository_root": str(root),
        "git_head": git("rev-parse", "HEAD"),
        "working_tree_dirty": bool(git("status", "--porcelain")),
        "runner_source": str(Path(__file__).resolve()),
        "runner_source_sha256": sha256_file(Path(__file__)),
    }


def _read_json(path: Path, *, self_hash: str | None = None) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationError(f"JSON object required: {path}")
    if self_hash is not None:
        claimed = value.get(self_hash)
        body = {key: item for key, item in value.items() if key != self_hash}
        if claimed != canonical_json_sha256(body):
            raise ConfirmationError(f"{self_hash} mismatch: {path}")
    return cast(dict[str, Any], value)


def _read_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ConfirmationError("confirmation config must be a mapping")
    required = {
        "schema_version",
        "protocol_id",
        "class_names",
        "interface",
        "predictor",
        "power",
        "comparisons",
        "gates",
        "consumed_cohorts",
    }
    if set(value) != required:
        raise ConfirmationError(
            f"confirmation config fields differ: {sorted(set(value) ^ required)}"
        )
    if tuple(value["class_names"]) != CLASS_NAMES:
        raise ConfirmationError("confirmation class order changed")
    comparisons = cast(dict[str, Any], value["comparisons"])
    if comparisons != {
        "primary": {"candidate": "T9", "control": "B9"},
        "mechanism": {"candidate": "T9", "control": "U9"},
        "sensor_plus_method": {"candidate": "T9", "control": "B6"},
        "secondary_methods": list(SECONDARY_METHODS),
    }:
        raise ConfirmationError("comparison set changed")
    return cast(dict[str, Any], value)


def _write_json(path: Path, payload: Mapping[str, Any], *, self_hash: str | None = None) -> None:
    value = dict(payload)
    if self_hash is not None:
        value[self_hash] = canonical_json_sha256(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _seal_directory(directory: Path, *, name: str = "completion_manifest.json") -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(directory).as_posix(),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.name != name
    ]
    record = {
        "schema_version": "1.0.0",
        "record_kind": "ctgr_native9_confirmation_completion_manifest",
        "created_at_utc": _utc_now(),
        "files": files,
    }
    _write_json(directory / name, record, self_hash="record_sha256")
    return record | {"record_sha256": canonical_json_sha256(record)}


def validate_run(directory: Path) -> dict[str, Any]:
    """Validate a create-only preparation, prediction, or score directory."""

    completion = _read_json(directory / "completion_manifest.json", self_hash="record_sha256")
    listed = set()
    for entry in completion.get("files", []):
        if not isinstance(entry, dict):
            raise ConfirmationError("completion manifest file entry is invalid")
        path = (directory / str(entry.get("path", ""))).resolve()
        if (
            not path.is_relative_to(directory.resolve())
            or not path.is_file()
            or sha256_file(path) != entry.get("sha256")
            or path.stat().st_size != entry.get("size_bytes")
        ):
            raise ConfirmationError(f"completion manifest mismatch: {entry.get('path')}")
        relative = path.relative_to(directory.resolve()).as_posix()
        if relative in listed:
            raise ConfirmationError("completion manifest repeats a file")
        listed.add(relative)
        if path.suffix == ".json":
            _read_json(path, self_hash="record_sha256")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != "completion_manifest.json"
    }
    if actual != listed:
        raise ConfirmationError("completion manifest is not exhaustive")
    return {
        "status": "passed",
        "directory": str(directory.resolve()),
        "checked_files": len(listed),
        "completion_manifest_record_sha256": completion["record_sha256"],
    }


def _create_output(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"create-only output already exists: {path}")
    path.mkdir(parents=True)


def _power_for_n(effect: float, standard_deviation: float, n: int, alpha: float) -> float:
    degrees = n - 1
    critical = float(t.ppf(1.0 - alpha / 2.0, degrees))
    noncentrality = effect * math.sqrt(n) / standard_deviation
    return float(
        nct.sf(critical, degrees, noncentrality) + nct.cdf(-critical, degrees, noncentrality)
    )


def power_analysis(config: Mapping[str, Any]) -> dict[str, Any]:
    """Calculate the frozen paired-participant sensitivity table."""

    power = cast(dict[str, Any], config["power"])
    deltas = np.asarray(power["planning_participant_deltas"], dtype=np.float64)
    if deltas.shape != (10,) or not np.isfinite(deltas).all():
        raise ConfirmationError("planning deltas must be ten finite development values")
    expected_hash = str(power["planning_deltas_canonical_sha256"])
    if canonical_json_sha256(deltas.tolist()) != expected_hash:
        raise ConfirmationError("planning delta hash differs")
    standard_deviation = float(np.std(deltas, ddof=1))
    alpha = float(power["two_sided_alpha"])
    target_power = float(power["target_power"])
    effects = [float(item) for item in cast(list[Any], power["sensitivity_effects"])]
    rows = []
    for effect in effects:
        minimum = next(
            (
                n
                for n in range(6, int(power["maximum_search_participants"]) + 1)
                if _power_for_n(effect, standard_deviation, n, alpha) >= target_power
            ),
            None,
        )
        rows.append(
            {
                "paired_effect": effect,
                "minimum_complete_participants": minimum,
                "attained_power": None
                if minimum is None
                else _power_for_n(effect, standard_deviation, minimum, alpha),
            }
        )
    smallest = float(power["smallest_worthwhile_effect"])
    primary = next(row for row in rows if row["paired_effect"] == smallest)
    configured = int(power["minimum_complete_participants"])
    if primary["minimum_complete_participants"] != configured:
        raise ConfirmationError(
            "configured sample size differs from reproducible power calculation"
        )
    return {
        "schema_version": "1.0.0",
        "record_kind": "ctgr_native9_confirmation_power_analysis",
        "status": "planning_sensitivity_not_outcome_evidence",
        "method": "two_sided_one_sample_noncentral_t_power_on_paired_participant_differences",
        "alpha": alpha,
        "target_power": target_power,
        "planning_participant_count": int(deltas.size),
        "planning_mean_difference": float(np.mean(deltas)),
        "planning_sample_standard_deviation": standard_deviation,
        "planning_source_path": str(power["planning_source_path"]),
        "planning_source_sha256": str(power["planning_source_sha256"]),
        "planning_deltas_canonical_sha256": expected_hash,
        "smallest_worthwhile_effect": smallest,
        "minimum_complete_participants": configured,
        "sensitivity": rows,
        "limitations": [
            "The variance comes from ten reused development participants and may not transport.",
            "The calculation fixes precision before confirmation labels open; it is not a result.",
            "The six-wearer physical information pilot is underpowered for this endpoint.",
        ],
    }


def _predictor_binding(package: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validation = _validate_frozen_predictor(package)
    result = _read_json(package / "result.json", self_hash="record_sha256")
    predictor = cast(dict[str, Any], config["predictor"])
    expected = {
        "predictor_sha256": sha256_file(package / "predictor.pkl"),
        "result_record_sha256": result["record_sha256"],
        "component_sha256": {
            name: result["component_checkpoints"][name]["sha256"] for name in ("B6", "B9", "expert")
        },
    }
    for key, item in expected.items():
        if predictor[key] != item:
            raise ConfirmationError(f"frozen predictor binding changed: {key}")
    if result["predictor_ids"] != list(CORE_METHODS) or result["confirmation_evaluated"]:
        raise ConfirmationError(
            "source package is not the expected unconfirmed four-predictor bundle"
        )
    return {
        "status": "verified",
        "package": str(package.resolve()),
        "predictor_path": str((package / "predictor.pkl").resolve()),
        **expected,
        "source_validation": validation,
    }


def prepare(
    *, config_path: Path, predictor_package: Path, cohort_directory: Path, output: Path
) -> dict[str, Any]:
    """Verify all available inputs and emit an explicit go/blocker record."""

    _create_output(output)
    started = time.monotonic()
    config = _read_config(config_path)
    power = power_analysis(config)
    binding = _predictor_binding(predictor_package, config)
    _write_json(output / "power_analysis.json", power, self_hash="record_sha256")
    _write_json(output / "predictor_binding.json", binding, self_hash="record_sha256")
    cohort_manifest = cohort_directory / "cohort_manifest.json"
    windows = cohort_directory / "windows_label_blind.npz"
    missing = [str(path.resolve()) for path in (cohort_manifest, windows) if not path.is_file()]
    source_raw = Path(str(config["predictor"]["source_raw_csv_for_secondary_controls"]))
    source_raw_valid = source_raw.is_file() and sha256_file(source_raw) == str(
        config["predictor"]["source_raw_csv_sha256"]
    )
    secondary_blockers = [] if source_raw_valid else [str(source_raw.resolve())]
    status = (
        "ready_for_label_blind_prediction"
        if not missing and not secondary_blockers
        else "incomplete"
    )
    record = {
        "schema_version": "1.0.0",
        "record_kind": "ctgr_native9_confirmation_preparation",
        "created_at_utc": _utc_now(),
        "code_provenance": _code_provenance(),
        "status": status,
        "evidence_status": "prospective_confirmation_not_yet_scored",
        "config": {
            "path": str(config_path.resolve()),
            "sha256": sha256_file(config_path),
        },
        "predictor_binding_record_sha256": canonical_json_sha256(binding),
        "power_record_sha256": canonical_json_sha256(power),
        "cohort_directory": str(cohort_directory.resolve()),
        "missing_fresh_cohort_inputs": missing,
        "missing_secondary_control_inputs": secondary_blockers,
        "blocking_conditions": [
            *(("fresh qualified native-nine cohort is absent",) if missing else ()),
            *(
                ("source raw CSV required to freeze secondary controls is absent or changed",)
                if secondary_blockers
                else ()
            ),
        ],
        "consumed_cohorts_prohibited": list(config["consumed_cohorts"]),
        "target_labels_accessed": False,
        "model_fits": 0,
        "prediction_matrices_created": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    filename = "READY.json" if status.startswith("ready") else "INCOMPLETE.json"
    _write_json(output / filename, record, self_hash="record_sha256")
    _seal_directory(output)
    return record


def _cohort_inputs(
    manifest_path: Path, archive_path: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], FloatArray, FloatArray, BoolArray, StringArray, StringArray]:
    manifest = _read_json(manifest_path, self_hash="record_sha256")
    required = {
        "schema_version",
        "record_kind",
        "cohort_id",
        "evidence_status",
        "freshness_adjudication",
        "provider_native_gravity",
        "ability_relevant",
        "participant_partition_before_windowing",
        "sampling_rate_hz",
        "window_samples",
        "stride_samples",
        "class_names",
        "channel_order",
        "units",
        "participant_roster",
        "qualification_record",
        "window_archive",
        "labels_separately_custodied",
        "labels_available_to_prediction_controller",
        "record_sha256",
    }
    if set(manifest) != required:
        raise ConfirmationError(
            f"cohort manifest fields differ: {sorted(set(manifest) ^ required)}"
        )
    interface = cast(dict[str, Any], config["interface"])
    roster = cast(list[str], manifest["participant_roster"])
    if (
        manifest["record_kind"] != "ctgr_native9_confirmation_cohort_manifest"
        or manifest["evidence_status"] != "fresh_untouched_confirmation_cohort"
        or not isinstance(manifest["freshness_adjudication"], dict)
        or manifest["freshness_adjudication"].get("passed") is not True
        or not manifest["provider_native_gravity"]
        or not manifest["ability_relevant"]
        or not manifest["participant_partition_before_windowing"]
        or manifest["sampling_rate_hz"] != interface["sampling_rate_hz"]
        or manifest["window_samples"] != interface["window_samples"]
        or manifest["stride_samples"] != interface["stride_samples"]
        or tuple(manifest["class_names"]) != CLASS_NAMES
        or manifest["channel_order"] != interface["channel_order"]
        or manifest["units"] != interface["units"]
        or not manifest["labels_separately_custodied"]
        or manifest["labels_available_to_prediction_controller"]
        or roster != sorted(set(roster))
        or len(roster) < int(config["power"]["minimum_complete_participants"])
    ):
        raise ConfirmationError("cohort does not satisfy the frozen fresh native-nine contract")
    if str(manifest["cohort_id"]) in set(config["consumed_cohorts"]):
        raise ConfirmationError("a consumed cohort cannot be reused as confirmation")
    qualification = cast(dict[str, Any], manifest["qualification_record"])
    if not qualification.get("path") or len(str(qualification.get("sha256", ""))) != 64:
        raise ConfirmationError("cohort qualification record is incomplete")
    qualification_path = (manifest_path.parent / str(qualification["path"])).resolve()
    if (
        not qualification_path.is_file()
        or sha256_file(qualification_path) != qualification["sha256"]
    ):
        raise ConfirmationError("cohort qualification record hash mismatch")
    archive_record = cast(dict[str, Any], manifest["window_archive"])
    if archive_path.resolve() != (manifest_path.parent / str(archive_record["path"])).resolve():
        raise ConfirmationError("window archive path differs from cohort manifest")
    if sha256_file(archive_path) != archive_record["sha256"]:
        raise ConfirmationError("window archive hash mismatch")
    with np.load(archive_path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if keys & PROHIBITED_ARCHIVE_KEYS:
            raise PermissionError("label-like arrays are prohibited from the prediction archive")
        expected = {
            "signals",
            "native_gravity",
            "native_gravity_available",
            "participant_ids",
            "window_ids",
        }
        if keys != expected:
            raise ConfirmationError(f"label-blind archive fields differ: {sorted(keys ^ expected)}")
        signals = np.asarray(archive["signals"], dtype=np.float64)
        gravity = np.asarray(archive["native_gravity"], dtype=np.float64)
        available = np.asarray(archive["native_gravity_available"], dtype=np.bool_)
        participants = np.asarray(archive["participant_ids"], dtype=np.str_)
        windows = np.asarray(archive["window_ids"], dtype=np.str_)
    count = signals.shape[0]
    if (
        signals.shape != (count, 128, 6)
        or gravity.shape != (count, 128, 3)
        or available.shape != (count,)
        or participants.shape != (count,)
        or windows.shape != (count,)
        or count == 0
        or not np.isfinite(signals).all()
        or not np.isfinite(gravity[available]).all()
        or any(np.any(np.char.str_len(values) == 0) for values in (participants, windows))
        or len(set(windows.tolist())) != count
        or set(participants.tolist()) != set(roster)
    ):
        raise ConfirmationError("label-blind native-nine arrays are invalid or misaligned")
    missing_fraction = float(np.mean(~available))
    if missing_fraction > float(interface["maximum_native_gravity_missing_fraction"]):
        raise ConfirmationError("native-gravity missingness exceeds the frozen limit")
    if any(not np.any(available[participants == person]) for person in roster):
        raise ConfirmationError("each participant requires at least one native-gravity window")
    return manifest, signals, gravity, available, participants, windows


def _load_secondary_probabilities(
    path: Path, *, windows: StringArray, required: bool
) -> tuple[dict[str, FloatArray], dict[str, Any] | None]:
    if not path.is_file():
        if required:
            raise FileNotFoundError(f"sealed secondary probability archive is required: {path}")
        return {}, None
    sidecar_path = path.with_suffix(".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(
            f"secondary prediction provenance sidecar is required: {sidecar_path}"
        )
    provenance = _read_json(sidecar_path, self_hash="record_sha256")
    expected_provenance_fields = {
        "schema_version",
        "record_kind",
        "status",
        "evidence_status",
        "prediction_archive_sha256",
        "methods",
        "source_only_training",
        "target_labels_accessed",
        "target_labels_used_for_fit_selection_or_calibration",
        "checkpoint_receipts",
        "record_sha256",
    }
    receipts = provenance.get("checkpoint_receipts")
    if (
        set(provenance) != expected_provenance_fields
        or provenance["record_kind"] != "ctgr_native9_secondary_prediction_seal"
        or provenance["status"] != "sealed_before_target_labels"
        or provenance["evidence_status"] != "source_only_secondary_controls"
        or provenance["prediction_archive_sha256"] != sha256_file(path)
        or provenance["methods"] != list(SECONDARY_METHODS)
        or provenance["source_only_training"] is not True
        or provenance["target_labels_accessed"] is not False
        or provenance["target_labels_used_for_fit_selection_or_calibration"] is not False
        or not isinstance(receipts, dict)
        or set(receipts) != set(SECONDARY_METHODS)
        or any(
            not isinstance(receipts[method], dict)
            or len(str(receipts[method].get("sha256", ""))) != 64
            or not str(receipts[method].get("method_contract", ""))
            for method in SECONDARY_METHODS
        )
    ):
        raise ConfirmationError("secondary prediction provenance contract is invalid")
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        expected = {"window_ids", *SECONDARY_METHODS}
        if keys != expected:
            raise ConfirmationError(
                f"secondary prediction fields differ: {sorted(keys ^ expected)}"
            )
        if not np.array_equal(np.asarray(archive["window_ids"], dtype=np.str_), windows):
            raise ConfirmationError("secondary predictions are not aligned to window ids")
        values = {
            method: np.asarray(archive[method], dtype=np.float64) for method in SECONDARY_METHODS
        }
    for method, probability in values.items():
        if (
            probability.shape != (windows.size, 3)
            or not np.isfinite(probability).all()
            or np.any(probability < 0)
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ConfirmationError(f"secondary probabilities are invalid: {method}")
    return values, provenance


def predict(
    *,
    config_path: Path,
    predictor_package: Path,
    cohort_manifest: Path,
    windows_archive: Path,
    secondary_predictions: Path,
    output: Path,
) -> dict[str, Any]:
    """Create and seal label-blind predictions for exactly one fresh cohort."""

    _create_output(output)
    started = time.monotonic()
    config = _read_config(config_path)
    binding = _predictor_binding(predictor_package, config)
    manifest, signals, gravity, available, participants, windows = _cohort_inputs(
        cohort_manifest, windows_archive, config
    )
    signals, gravity, unit_conversion = _to_frozen_model_units(signals, gravity, manifest["units"])
    with (predictor_package / "predictor.pkl").open("rb") as stream:
        bundle = cast(dict[str, Any], pickle.load(stream))
    core = {method: np.empty((signals.shape[0], 3), dtype=np.float64) for method in CORE_METHODS}
    native = _predict_native_windows(signals[available], gravity[available], bundle)
    if set(native) != set(CORE_METHODS):
        raise ConfirmationError("frozen predictor output set changed")
    for method in CORE_METHODS:
        core[method][available] = native[method]
    if np.any(~available):
        fallback = _predict_native_windows(
            signals[~available], np.zeros((int((~available).sum()), 128, 3)), bundle
        )["B6"]
        for method in CORE_METHODS:
            core[method][~available] = fallback
    for method, probability in core.items():
        if (
            probability.shape != (signals.shape[0], 3)
            or not np.isfinite(probability).all()
            or not np.allclose(probability.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ConfirmationError(f"invalid frozen probability matrix: {method}")
    if np.any(~available) and any(
        core[method][~available].tobytes() != core["B6"][~available].tobytes()
        for method in CORE_METHODS
    ):
        raise ConfirmationError("missing-gravity fallback differs from exact B6 bytes")
    secondary, secondary_provenance = _load_secondary_probabilities(
        secondary_predictions,
        windows=windows,
        required=bool(config["interface"]["require_secondary_controls"]),
    )
    if secondary_provenance is None:
        raise ConfirmationError("the frozen experiment requires secondary control provenance")
    probability_path = output / "predictions_label_blind.npz"
    prediction_payload: dict[str, NDArray[Any]] = {
        "participant_ids": participants,
        "window_ids": windows,
        "native_gravity_available": available,
    }
    prediction_payload.update(core)
    prediction_payload.update(secondary)
    with probability_path.open("xb") as stream:
        np.savez_compressed(stream, **prediction_payload)  # type: ignore[arg-type]
    seal = {
        "schema_version": "1.0.0",
        "record_kind": "ctgr_native9_label_blind_prediction_seal",
        "created_at_utc": _utc_now(),
        "code_provenance": _code_provenance(),
        "status": "sealed_before_label_opening",
        "evidence_status": "fresh_confirmation_predictions_unscored",
        "config_sha256": sha256_file(config_path),
        "cohort_manifest_record_sha256": manifest["record_sha256"],
        "window_archive_sha256": sha256_file(windows_archive),
        "model_interface_unit_conversion": unit_conversion,
        "predictor_binding": binding,
        "secondary_prediction_archive_sha256": sha256_file(secondary_predictions),
        "secondary_prediction_seal_record_sha256": secondary_provenance["record_sha256"],
        "prediction_archive": {
            "path": probability_path.name,
            "sha256": sha256_file(probability_path),
            "size_bytes": probability_path.stat().st_size,
        },
        "methods": [*CORE_METHODS, *SECONDARY_METHODS],
        "window_count": int(windows.size),
        "participant_count": len(set(participants.tolist())),
        "native_gravity_missing_window_count": int((~available).sum()),
        "native_gravity_missing_fraction": float(np.mean(~available)),
        "missing_gravity_exact_B6_fallback_verified": True,
        "target_labels_accessed": False,
        "model_fits": 0,
        "elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output / "prediction_seal.json", seal, self_hash="record_sha256")
    _seal_directory(output)
    return seal


def _f1_from_confusion(matrix: IntArray) -> FloatArray:
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return np.asarray(
        np.divide(
            2.0 * np.diag(matrix),
            denominator,
            out=np.zeros(matrix.shape[0], dtype=np.float64),
            where=denominator > 0,
        ),
        dtype=np.float64,
    )


def _participant_report(
    labels: IntArray,
    probabilities: FloatArray,
    participants: StringArray,
    missing: BoolArray,
) -> dict[str, Any]:
    predictions = probabilities.argmax(axis=1)
    rows = []
    for person in sorted(set(participants.tolist())):
        selected = participants == person
        y = labels[selected]
        p = probabilities[selected]
        predicted = predictions[selected]
        matrix = confusion_matrix(y, predicted, labels=np.arange(3)).astype(np.int64)
        support = matrix.sum(axis=1)
        recall = np.divide(
            np.diag(matrix), support, out=np.zeros(3, dtype=np.float64), where=support > 0
        )
        clipped = np.clip(p, 1e-15, 1.0)
        one_hot = np.eye(3, dtype=np.float64)[y]
        rows.append(
            {
                "participant_id": person,
                "window_count": int(selected.sum()),
                "macro_f1": float(_f1_from_confusion(matrix).mean()),
                "accuracy": float(np.mean(y == predicted)),
                "class_recall": dict(zip(CLASS_NAMES, recall.tolist(), strict=True)),
                "class_support": dict(zip(CLASS_NAMES, support.tolist(), strict=True)),
                "negative_log_likelihood": float(-np.log(clipped[np.arange(y.size), y]).mean()),
                "multiclass_brier": float(np.sum((p - one_hot) ** 2, axis=1).mean()),
                "native_gravity_missing_fraction": float(np.mean(missing[selected])),
                "confusion_matrix": matrix.tolist(),
            }
        )
    macro = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    tail_count = max(1, math.ceil(0.30 * macro.size))
    recalls = {
        name: float(np.mean([row["class_recall"][name] for row in rows])) for name in CLASS_NAMES
    }
    calibration = classification_report(
        labels, probabilities, participants.tolist(), class_names=CLASS_NAMES
    )["calibration"]
    return {
        "participant_count": len(rows),
        "primary": {
            "mean_participant_macro_f1": float(macro.mean()),
            "bottom_30_percent_participant_macro_f1": float(np.sort(macro)[:tail_count].mean()),
            "bottom_30_participant_count": tail_count,
            "worst_participant_macro_f1": float(macro.min()),
            "mean_participant_accuracy": float(np.mean([row["accuracy"] for row in rows])),
            "mean_participant_class_recall": recalls,
            "mean_participant_negative_log_likelihood": float(
                np.mean([row["negative_log_likelihood"] for row in rows])
            ),
            "mean_participant_multiclass_brier": float(
                np.mean([row["multiclass_brier"] for row in rows])
            ),
        },
        "pooled_window_diagnostics": classification_report(
            labels, probabilities, participants.tolist(), class_names=CLASS_NAMES
        )["window_level_diagnostics"],
        "pooled_window_calibration": calibration,
        "participants": rows,
    }


def _paired_comparison(
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    left = {row["participant_id"]: row for row in candidate["participants"]}
    right = {row["participant_id"]: row for row in control["participants"]}
    if set(left) != set(right):
        raise ConfirmationError("paired reports do not cover the same participants")
    people = sorted(left)
    candidate_values = np.asarray([left[p]["macro_f1"] for p in people], dtype=np.float64)
    control_values = np.asarray([right[p]["macro_f1"] for p in people], dtype=np.float64)
    difference = candidate_values - control_values
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(people), size=(replicates, len(people)))
    mean_draws = difference[indices].mean(axis=1)
    tail = max(1, math.ceil(0.30 * len(people)))
    candidate_tail = np.sort(candidate_values[indices], axis=1)[:, :tail].mean(axis=1)
    control_tail = np.sort(control_values[indices], axis=1)[:, :tail].mean(axis=1)
    class_recall_difference = {
        name: float(
            np.mean(
                [left[p]["class_recall"][name] - right[p]["class_recall"][name] for p in people]
            )
        )
        for name in CLASS_NAMES
    }
    leave_one_out = {
        person: float(np.mean(np.delete(difference, index))) for index, person in enumerate(people)
    }
    paired_rows = [
        {
            "participant_id": person,
            "candidate_macro_f1": float(left[person]["macro_f1"]),
            "control_macro_f1": float(right[person]["macro_f1"]),
            "difference": float(difference[index]),
        }
        for index, person in enumerate(people)
    ]
    return {
        "mean_participant_macro_f1_difference": float(difference.mean()),
        "paired_participant_bootstrap_95_percent_ci": np.quantile(
            mean_draws, [0.025, 0.975], method="linear"
        ).tolist(),
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "bottom_30_percent_difference": float(
            np.sort(candidate_values)[:tail].mean() - np.sort(control_values)[:tail].mean()
        ),
        "bottom_30_percent_difference_95_percent_ci": np.quantile(
            candidate_tail - control_tail, [0.025, 0.975], method="linear"
        ).tolist(),
        "worst_participant_score_difference": float(candidate_values.min() - control_values.min()),
        "minimum_paired_participant_difference": float(difference.min()),
        "maximum_paired_participant_harm": float(min(0.0, difference.min())),
        "wins": int(np.sum(difference > 1e-12)),
        "harms": int(np.sum(difference < -1e-12)),
        "ties": int(np.sum(np.abs(difference) <= 1e-12)),
        "mean_participant_class_recall_difference": class_recall_difference,
        "leave_one_participant_out_mean_differences": leave_one_out,
        "all_leave_one_participant_out_means_positive": all(
            value > 0.0 for value in leave_one_out.values()
        ),
        "participants": paired_rows,
    }


def _primary_gates(
    comparison: Mapping[str, Any], *, gates: Mapping[str, Any], missing_fraction: float
) -> dict[str, Any]:
    recalls = comparison["mean_participant_class_recall_difference"]
    checks = {
        "mean_gain_at_least_smallest_worthwhile_effect": comparison[
            "mean_participant_macro_f1_difference"
        ]
        >= gates["minimum_mean_macro_f1_difference"],
        "paired_bootstrap_lower_above_zero": comparison[
            "paired_participant_bootstrap_95_percent_ci"
        ][0]
        > 0.0,
        "bottom_30_noninferiority": comparison["bottom_30_percent_difference"]
        >= gates["minimum_bottom_30_difference"],
        "worst_score_noninferiority": comparison["worst_participant_score_difference"]
        >= gates["minimum_worst_score_difference"],
        "paired_person_harm_guard": comparison["minimum_paired_participant_difference"]
        >= gates["minimum_paired_participant_difference"],
        "mobility_recall_guard": recalls["mobility"]
        >= gates["minimum_class_recall_difference"]["mobility"],
        "sitting_recall_guard": recalls["sitting"]
        >= gates["minimum_class_recall_difference"]["sitting"],
        "standing_recall_guard": recalls["standing"]
        >= gates["minimum_class_recall_difference"]["standing"],
        "leave_one_participant_out_robustness": comparison[
            "all_leave_one_participant_out_means_positive"
        ],
        "native_gravity_missingness_within_limit": missing_fraction
        <= gates["maximum_native_gravity_missing_fraction"],
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "decision": "promote_independent_ctgr_claim" if all(checks.values()) else "do_not_promote",
    }


def score(
    *, config_path: Path, prediction_run: Path, labels_archive: Path, output: Path
) -> dict[str, Any]:
    """Score once, after a valid label-blind prediction seal exists."""

    _create_output(output)
    started = time.monotonic()
    config = _read_config(config_path)
    seal_path = prediction_run / "prediction_seal.json"
    seal = _read_json(seal_path, self_hash="record_sha256")
    if seal["status"] != "sealed_before_label_opening" or seal["target_labels_accessed"]:
        raise ConfirmationError("predictions were not validly sealed before scoring")
    prediction_path = prediction_run / str(seal["prediction_archive"]["path"])
    if sha256_file(prediction_path) != seal["prediction_archive"]["sha256"]:
        raise ConfirmationError("sealed prediction archive hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as archive:
        methods = [*CORE_METHODS, *SECONDARY_METHODS]
        expected = {
            "participant_ids",
            "window_ids",
            "native_gravity_available",
            *methods,
        }
        if set(archive.files) != expected:
            raise ConfirmationError("sealed prediction method set differs")
        participants = np.asarray(archive["participant_ids"], dtype=np.str_)
        windows = np.asarray(archive["window_ids"], dtype=np.str_)
        missing = ~np.asarray(archive["native_gravity_available"], dtype=np.bool_)
        probabilities = {
            method: np.asarray(archive[method], dtype=np.float64) for method in methods
        }
    with np.load(labels_archive, allow_pickle=False) as labels_file:
        if set(labels_file.files) != {"window_ids", "labels"}:
            raise ConfirmationError("label archive requires only window_ids and labels")
        label_windows = np.asarray(labels_file["window_ids"], dtype=np.str_)
        labels = np.asarray(labels_file["labels"], dtype=np.int64)
    if (
        not np.array_equal(label_windows, windows)
        or labels.shape != (windows.size,)
        or set(labels.tolist()) != {0, 1, 2}
    ):
        raise ConfirmationError("opened labels are invalid or not aligned to sealed predictions")
    support = {
        person: np.bincount(labels[participants == person], minlength=3).tolist()
        for person in sorted(set(participants.tolist()))
    }
    if any(0 in counts for counts in support.values()):
        raise ConfirmationError("every confirmation participant must contain all three classes")
    reports = {
        method: _participant_report(labels, probability, participants, missing)
        for method, probability in probabilities.items()
    }
    bootstrap = cast(dict[str, Any], config["gates"]["bootstrap"])
    contrast_names = {
        "primary_T9_minus_B9": ("T9", "B9"),
        "mechanism_T9_minus_U9": ("T9", "U9"),
        "sensor_plus_method_T9_minus_B6": ("T9", "B6"),
    }
    comparisons = {
        name: _paired_comparison(
            reports[candidate],
            reports[control],
            seed=int(bootstrap["seed"]),
            replicates=int(bootstrap["replicates"]),
        )
        | {"candidate": candidate, "control": control}
        for name, (candidate, control) in contrast_names.items()
    }
    comparisons["descriptive_T9_vs_secondary"] = {
        method: _paired_comparison(
            reports["T9"],
            reports[method],
            seed=int(bootstrap["seed"]),
            replicates=int(bootstrap["replicates"]),
        )
        for method in SECONDARY_METHODS
    }
    primary = comparisons["primary_T9_minus_B9"]
    gates = _primary_gates(
        primary,
        gates=cast(dict[str, Any], config["gates"]),
        missing_fraction=float(np.mean(missing)),
    )
    result = {
        "schema_version": "1.0.0",
        "record_kind": "ctgr_native9_confirmation_result",
        "created_at_utc": _utc_now(),
        "code_provenance": _code_provenance(),
        "status": "complete",
        "evidence_status": "fresh_independent_confirmation_scored_once",
        "config_sha256": sha256_file(config_path),
        "prediction_seal_record_sha256": seal["record_sha256"],
        "prediction_archive_sha256": seal["prediction_archive"]["sha256"],
        "labels_archive_sha256": sha256_file(labels_archive),
        "participant_count": len(support),
        "window_count": int(labels.size),
        "participant_class_support": support,
        "native_gravity_missing_window_count": int(missing.sum()),
        "native_gravity_missing_fraction": float(np.mean(missing)),
        "missing_gravity_exact_B6_fallback_verified": seal[
            "missing_gravity_exact_B6_fallback_verified"
        ],
        "reports": reports,
        "comparisons": comparisons,
        "primary_promotion_gate": gates,
        "multiplicity": {
            "primary_contrast": "T9_minus_B9",
            "primary_alpha": config["power"]["two_sided_alpha"],
            "secondary_contrasts": "descriptive; no multiplicity-adjusted promotion claim",
        },
        "model_fits": 0,
        "scoring_elapsed_seconds": time.monotonic() - started,
    }
    _write_json(output / "result.json", result, self_hash="record_sha256")
    report = [
        "# CTGR native-nine confirmation result",
        "",
        f"Status: **{gates['decision']}**.",
        "",
        "| Method | Mean participant macro-F1 | Bottom 30% | Worst | Sitting recall | Standing recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in [*CORE_METHODS, *SECONDARY_METHODS]:
        metric = reports[method]["primary"]
        report.append(
            f"| {method} | {metric['mean_participant_macro_f1']:.6f} | "
            f"{metric['bottom_30_percent_participant_macro_f1']:.6f} | "
            f"{metric['worst_participant_macro_f1']:.6f} | "
            f"{metric['mean_participant_class_recall']['sitting']:.6f} | "
            f"{metric['mean_participant_class_recall']['standing']:.6f} |"
        )
    report.extend(
        [
            "",
            "Primary contrast: T9 minus B9 = "
            f"{primary['mean_participant_macro_f1_difference']:.6f}, 95% participant bootstrap CI "
            f"[{primary['paired_participant_bootstrap_95_percent_ci'][0]:.6f}, "
            f"{primary['paired_participant_bootstrap_95_percent_ci'][1]:.6f}].",
            "",
            "Every participant difference and every failed gate is retained in `result.json`.",
        ]
    )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    _seal_directory(output)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--config", type=Path, required=True)
    prepare_parser.add_argument("--predictor-package", type=Path, required=True)
    prepare_parser.add_argument("--cohort-directory", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    predict_parser = commands.add_parser("predict")
    predict_parser.add_argument("--config", type=Path, required=True)
    predict_parser.add_argument("--predictor-package", type=Path, required=True)
    predict_parser.add_argument("--cohort-manifest", type=Path, required=True)
    predict_parser.add_argument("--windows-archive", type=Path, required=True)
    predict_parser.add_argument("--secondary-predictions", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    score_parser = commands.add_parser("score")
    score_parser.add_argument("--config", type=Path, required=True)
    score_parser.add_argument("--prediction-run", type=Path, required=True)
    score_parser.add_argument("--labels-archive", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    validate_parser = commands.add_parser("validate")
    validate_parser.add_argument("--run-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        result = prepare(
            config_path=args.config,
            predictor_package=args.predictor_package,
            cohort_directory=args.cohort_directory,
            output=args.output,
        )
    elif args.command == "predict":
        result = predict(
            config_path=args.config,
            predictor_package=args.predictor_package,
            cohort_manifest=args.cohort_manifest,
            windows_archive=args.windows_archive,
            secondary_predictions=args.secondary_predictions,
            output=args.output,
        )
    elif args.command == "score":
        result = score(
            config_path=args.config,
            prediction_run=args.prediction_run,
            labels_archive=args.labels_archive,
            output=args.output,
        )
    else:
        result = validate_run(args.run_directory)
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
