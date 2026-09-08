"""Create-only controller and zero-fit replay for physical signal identifiability."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from itertools import pairwise
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import numpy.typing as npt

from inclusive_shift_har.data import physical_information_signal as signal_data_module
from inclusive_shift_har.data.physical_information_pilot import (
    load_config,
    validate_collection_records,
    validate_config,
    validate_plan,
    verify_self_hash,
)
from inclusive_shift_har.data.physical_information_signal import (
    CanonicalRecording,
    DeviceSignalContract,
    SignalContractError,
    analyze_physical_information,
    build_signal_audit,
    build_window_table,
    canonical_recording_from_arrays,
    parse_device_signal_contract,
    project_bouts,
    window_table_arrays,
    window_table_from_arrays,
    window_table_hash,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}$")
SIGNAL_SOFTWARE_PREREQUISITES = (
    "device_specific_canonicalization_adapter_and_receipt",
    "canonical_continuous_recordings",
    "adjudicated_bout_annotations",
)


def _now_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SignalContractError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _sequence_of_mappings(value: object, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        raise SignalContractError(f"{name} must be a JSON array")
    return [_mapping(item, f"{name}[{index}]") for index, item in enumerate(value)]


def _sealed(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    if "record_sha256" in result:
        raise SignalContractError("payload is already sealed")
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def _verify_sealed(record: Mapping[str, Any], name: str) -> None:
    body = dict(record)
    declared = body.pop("record_sha256", None)
    if not isinstance(declared, str) or declared != canonical_json_sha256(body):
        raise SignalContractError(f"{name} self-hash mismatch")


def _write_json(path: Path, payload: Mapping[str, Any], *, root: Path) -> None:
    atomic_write_json_new(payload, path, allowed_root=root)


def _write_bytes(path: Path, payload: bytes, *, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    try:
        parent.relative_to(resolved_root)
    except ValueError as exc:
        raise SignalContractError(f"output escapes run directory: {path}") from exc
    target = parent / path.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    with target.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _copy_create_only(source: Path, target: Path, *, root: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise SignalContractError(f"canonical input is not a regular file: {source}")
    resolved_root = root.resolve(strict=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    parent = target.parent.resolve(strict=True)
    try:
        parent.relative_to(resolved_root)
    except ValueError as exc:
        raise SignalContractError(f"copy target escapes run directory: {target}") from exc
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing file: {target}")
    with source.open("rb") as input_stream, target.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            output_stream.write(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())


def _write_npz(path: Path, arrays: Mapping[str, npt.NDArray[Any]], *, root: Path) -> None:
    resolved_root = root.resolve(strict=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    parent = path.parent.resolve(strict=True)
    try:
        parent.relative_to(resolved_root)
    except ValueError as exc:
        raise SignalContractError(f"NPZ path escapes run directory: {path}") from exc
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing file: {path}")
    for name, array in arrays.items():
        if np.asarray(array).dtype.kind == "O":
            raise SignalContractError(f"object array forbidden in NPZ: {name}")
    with path.open("xb") as handle:
        np.savez_compressed(handle, **cast(dict[str, Any], dict(arrays)))
        handle.flush()
        os.fsync(handle.fileno())


def _read_npz(path: Path) -> dict[str, npt.NDArray[Any]]:
    if path.is_symlink() or not path.is_file():
        raise SignalContractError(f"NPZ is missing or symbolic: {path}")
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if any(array.dtype.kind == "O" for array in arrays.values()):
        raise SignalContractError(f"object array forbidden in NPZ: {path}")
    return arrays


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise SignalContractError(f"{name} must be a safe POSIX relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or value.endswith("/"):
        raise SignalContractError(f"{name} must be a safe POSIX relative path")
    return path


def _resolve_below(root: Path, relative: PurePosixPath) -> Path:
    result = root.joinpath(*relative.parts).resolve(strict=True)
    try:
        result.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise SignalContractError(f"input escapes canonical root: {relative}") from exc
    return result


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "clean": not status, "status_lines": status}


def _validate_source_lock(repository_root: Path, code_commit: str) -> dict[str, Any]:
    actual_root = Path(__file__).resolve(strict=True).parents[3]
    if repository_root.resolve(strict=True) != actual_root:
        raise SignalContractError("repository_root is not the executing controller source checkout")
    module_paths = [
        Path(__file__).resolve(strict=True),
        Path(signal_data_module.__file__).resolve(strict=True),
    ]
    if any(not path.is_relative_to(actual_root) for path in module_paths):
        raise SignalContractError("imported signal module is outside the executing source checkout")
    if not COMMIT_PATTERN.fullmatch(code_commit):
        raise SignalContractError("code_commit must be a full lowercase commit hash")
    state = _git_state(repository_root)
    if state["commit"] != code_commit:
        raise SignalContractError("code_commit does not match repository HEAD")
    if state["clean"] is not True:
        raise SignalContractError("signal analysis source worktree must be clean")
    state["executing_sources"] = {
        path.relative_to(actual_root).as_posix(): sha256_file(path) for path in module_paths
    }
    return state


def _load_plan_and_config(
    plan_path: Path, config_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _mapping(load_json_strict(plan_path), "plan")
    config = load_config(config_path)
    validate_config(config)
    report = validate_plan(plan, config)
    if report.get("valid") is not True:
        raise SignalContractError(f"frozen plan validation failed: {report.get('errors')}")
    if not verify_self_hash(plan):
        raise SignalContractError("frozen plan self-hash failed")
    return plan, config


def emit_blocker_record(
    *,
    repository_root: Path,
    plan_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Emit evidence that analysis was not attempted because external inputs are absent."""

    repository_root = repository_root.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    config_path = config_path.resolve(strict=True)
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_directory}")
    state = _validate_source_lock(repository_root, code_commit)
    plan, _config = _load_plan_and_config(plan_path, config_path)
    prerequisites = list(
        dict.fromkeys(
            [*cast(list[str], plan["pending_prerequisites"]), *SIGNAL_SOFTWARE_PREREQUISITES]
        )
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    try:
        blocker = _sealed(
            {
                "record_kind": "physical_information_signal_incomplete_blocker",
                "schema_version": "1.0.0",
                "study_id": plan.get("study_id"),
                "created_at_utc": _now_z(),
                "status": "incomplete_external_prerequisites_missing",
                "readiness_status": "not_ready_for_signal_analysis",
                "blocked_phase": "physical_collection_and_device_canonicalization",
                "blocking_prerequisites": prerequisites,
                "blocking_detail": {
                    "device_specific_canonicalization_adapter_and_receipt": (
                        "No qualified device adapter/receipt with explicit timestamps, channels, "
                        "units, acceleration semantics, and gravity provenance was supplied."
                    ),
                    "canonical_continuous_recordings": (
                        "No hash-bound canonical continuous attachment recordings were supplied."
                    ),
                    "adjudicated_bout_annotations": (
                        "No complete hash-bound adjudicated annotation set was supplied."
                    ),
                },
                "collection_attempted": False,
                "canonical_recordings_loaded": 0,
                "participant_outcomes_loaded": False,
                "participant_outcomes_reported": False,
                "model_fit_count": 0,
                "physical_identifiability_gate": {
                    "status": "not_evaluated",
                    "reason": "required real canonical collection was not supplied",
                },
                "claims": {
                    "collection_completed": False,
                    "device_qualified": False,
                    "physical_identifiability_demonstrated": False,
                    "population_superiority_demonstrated": False,
                    "architecture_eligible": False,
                },
                "source_provenance": {
                    "code_commit": code_commit,
                    "worktree_clean": state["clean"],
                    "executing_sources": state.get("executing_sources", {}),
                    "plan_path": str(plan_path),
                    "plan_sha256": sha256_file(plan_path),
                    "plan_record_sha256": plan.get("record_sha256"),
                    "config_path": str(config_path),
                    "config_sha256": sha256_file(config_path),
                },
            }
        )
        _write_json(output_directory / "INCOMPLETE.json", blocker, root=output_directory)
        persisted_blocker = _mapping(
            load_json_strict(output_directory / "INCOMPLETE.json"), "persisted blocker"
        )
        _verify_sealed(persisted_blocker, "persisted blocker")
        manifest = _seal_artifact_manifest(output_directory, excluded={"artifact_manifest.json"})
        _write_json(output_directory / "artifact_manifest.json", manifest, root=output_directory)
        persisted_manifest = _mapping(
            load_json_strict(output_directory / "artifact_manifest.json"),
            "persisted artifact manifest",
        )
        _verify_sealed(persisted_manifest, "persisted artifact manifest")
        _verify_artifact_manifest(output_directory)
        validation = _sealed(
            {
                "record_kind": "physical_information_signal_blocker_validation",
                "validated_at_utc": _now_z(),
                "valid": True,
                "model_fit_count": 0,
                "blocker_record_self_hash_valid": True,
                "artifact_manifest_self_hash_valid": True,
                "participant_outcomes_absent": True,
                "gate_status": "not_evaluated",
            }
        )
        _write_json(output_directory / "validation.json", validation, root=output_directory)
        completion = _sealed(
            {
                "record_kind": "physical_information_signal_blocker_completion_manifest",
                "status": "incomplete_blocker_recorded",
                "completed_at_utc": _now_z(),
                "model_fit_count": 0,
                "worker_shutdown": {"workers_started": 0, "workers_remaining": 0},
                "artifacts": _artifact_rows(
                    output_directory,
                    excluded={"completion_manifest.json", "controller_final_receipt.json"},
                ),
            }
        )
        _write_json(
            output_directory / "completion_manifest.json", completion, root=output_directory
        )
        final = _sealed(
            {
                "record_kind": "physical_information_signal_controller_final_receipt",
                "status": "incomplete_blocker_sealed",
                "finished_at_utc": _now_z(),
                "model_fit_count": 0,
                "task_owned_workers_remaining": 0,
                "completion_manifest_sha256": sha256_file(
                    output_directory / "completion_manifest.json"
                ),
            }
        )
        _write_json(
            output_directory / "controller_final_receipt.json", final, root=output_directory
        )
        _verify_completion(output_directory, require_final=True)
        return final
    except BaseException as exc:
        _record_failure(output_directory, started, exc)
        raise


def _contract_to_public_dict(contract: DeviceSignalContract) -> dict[str, Any]:
    return {
        "record_kind": contract.record_kind,
        "schema_version": contract.schema_version,
        "status": contract.status,
        "evidence_status": contract.evidence_status,
        "device_id": contract.device_id,
        "manufacturer": contract.manufacturer,
        "model": contract.model,
        "firmware": contract.firmware,
        "logger": contract.logger,
        "axis_convention": contract.axis_convention,
        "clock_domain": contract.clock_domain,
        "channel_names": contract.channel_names,
        "units": {
            "timestamp": contract.timestamp_unit,
            "acceleration": contract.acceleration_unit,
            "angular_velocity": contract.angular_velocity_unit,
            "gravity": contract.gravity_unit,
        },
        "acceleration_semantics": contract.acceleration_semantics,
        "native_gravity_available": contract.native_gravity_available,
        "processing_contract": {
            "source_rate_hz": contract.processing.source_rate_hz,
            "target_rate_hz": contract.processing.target_rate_hz,
            "gravity_source": contract.processing.gravity_source,
            "gravity_cutoff_hz": contract.processing.gravity_cutoff_hz,
            "maximum_gap_seconds": contract.processing.maximum_gap_seconds,
            "resampling_method": contract.processing.resampling_method,
            "window_samples": contract.processing.window_samples,
            "window_stride_samples": contract.processing.window_stride_samples,
        },
        "frame_conditioning_limit": contract.frame_conditioning_limit,
        "equipment_receipt_sha256": contract.equipment_receipt_sha256,
        "placement_instruction_sha256": contract.placement_instruction_sha256,
        "bench_qualification_sha256": contract.bench_qualification_sha256,
        "canonicalization_receipt_sha256": contract.canonicalization_receipt_sha256,
        "processing_frozen_at_utc": contract.processing_frozen_at_utc,
    }


def _utc(value: object, name: str) -> datetime:
    if not isinstance(value, str) or not value.endswith("Z"):
        raise SignalContractError(f"{name} must be explicit UTC")
    try:
        parsed = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError as exc:
        raise SignalContractError(f"invalid UTC: {name}") from exc
    offset = parsed.utcoffset()
    if offset is None or offset.total_seconds() != 0:
        raise SignalContractError(f"invalid UTC offset: {name}")
    return parsed


def _qualification(
    *,
    contract: DeviceSignalContract,
    plan: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest_path: Path | None,
    root: Path,
    copy_root: Path | None,
) -> dict[str, Any]:
    if contract.evidence_status == "synthetic_fixture_not_physical_evidence":
        return {
            "status": "synthetic_only",
            "physical_evidence_qualified": False,
            "study_prerequisites_verified": False,
        }
    if manifest_path is None:
        raise SignalContractError("real physical analysis requires actual qualification receipts")
    manifest = _mapping(load_json_strict(manifest_path), "qualification manifest")
    _verify_sealed(manifest, "qualification manifest")
    if (
        set(manifest)
        != {"record_kind", "schema_version", "equipment", "canonicalization", "record_sha256"}
        or manifest["record_kind"] != "physical_information_signal_qualification_manifest"
        or manifest["schema_version"] != "1.0.0"
    ):
        raise SignalContractError("qualification manifest schema mismatch")
    references = []
    loaded = {}
    for name, expected in (
        ("equipment", contract.equipment_receipt_sha256),
        ("canonicalization", contract.canonicalization_receipt_sha256),
    ):
        reference = _mapping(manifest[name], name)
        if set(reference) != {"path", "sha256"} or reference["sha256"] != expected:
            raise SignalContractError(f"{name} contract hash binding mismatch")
        relative = _safe_relative_path(reference["path"], name)
        path = _resolve_below(root, relative)
        if sha256_file(path) != expected:
            raise SignalContractError(f"{name} receipt bytes changed")
        loaded[name] = _mapping(load_json_strict(path), name)
        references.append((relative, expected))
    equipment, adapter = loaded["equipment"], loaded["canonicalization"]
    equipment_validation = validate_collection_records(
        plan=plan, config=config, equipment=equipment, recordings=[], annotations=[], raw_root=root
    )
    if equipment_validation["equipment_evidence_verified"] is not True:
        raise SignalContractError(
            f"equipment qualification invalid: {equipment_validation['errors']}"
        )
    public = _contract_to_public_dict(contract)
    for name in (
        "device_id",
        "manufacturer",
        "model",
        "firmware",
        "logger",
        "axis_convention",
        "clock_domain",
        "native_gravity_available",
        "processing_contract",
        "frame_conditioning_limit",
        "processing_frozen_at_utc",
    ):
        if equipment[name] != public[name]:
            raise SignalContractError(f"equipment/signal contract disagreement: {name}")
    for name, value in equipment["channel_map"].items():
        if contract.channel_names.get(name) != value:
            raise SignalContractError("equipment channel mapping differs from signal contract")
    for name, value in equipment["units"].items():
        if public["units"].get(name) != value:
            raise SignalContractError("equipment units differ from signal contract")
    for name, expected in (
        ("placement_instruction", contract.placement_instruction_sha256),
        ("bench_qualification", contract.bench_qualification_sha256),
    ):
        if equipment[f"{name}_sha256"] != expected:
            raise SignalContractError(f"{name} contract hash mismatch")
        references.append((_safe_relative_path(equipment[f"{name}_path"], name), expected))
    _verify_sealed(adapter, "canonicalization receipt")
    if (
        adapter.get("record_kind") != "physical_information_canonicalization_receipt"
        or adapter.get("status") != "reviewed"
        or adapter.get("device_id") != contract.device_id
        or adapter.get("equipment_receipt_sha256") != contract.equipment_receipt_sha256
        or adapter.get("processing_contract_sha256")
        != canonical_json_sha256(public["processing_contract"])
        or not isinstance(adapter.get("reviewer_id"), str)
        or not adapter["reviewer_id"]
        or not isinstance(adapter.get("adapter_code_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", adapter["adapter_code_sha256"])
    ):
        raise SignalContractError("canonicalization receipt is unreviewed or incorrectly bound")
    qualified = _utc(equipment["qualified_at_utc"], "equipment qualification")
    frozen = _utc(contract.processing_frozen_at_utc, "processing freeze")
    if not (qualified <= _utc(adapter.get("reviewed_at_utc"), "adapter review") <= frozen):
        raise SignalContractError("qualification/adapter review must precede processing freeze")
    if copy_root is not None:
        for relative, expected in references:
            source = _resolve_below(root, relative)
            target = copy_root.joinpath(*relative.parts)
            _copy_create_only(source, target, root=copy_root)
            if sha256_file(target) != expected:
                raise SignalContractError("copied qualification receipt hash mismatch")
    return {
        "status": "equipment_and_processing_provenance_verified",
        "equipment_evidence_verified": True,
        "study_prerequisites_verified": False,
        "qualification_manifest_sha256": sha256_file(manifest_path),
        "physical_evidence_qualified": False,
        "processing_frozen_at_utc": contract.processing_frozen_at_utc,
    }


def _collection_chronology(
    rows: Sequence[Mapping[str, Any]], contract: DeviceSignalContract
) -> None:
    visits: dict[tuple[str, int], list[Mapping[str, Any]]] = {}
    frozen = _utc(contract.processing_frozen_at_utc, "processing freeze")
    for row in rows:
        anchor = _utc(row.get("visit_started_at_utc"), "visit start")
        if anchor <= frozen:
            raise SignalContractError("visit must start after qualified processing freeze")
        start, end = row.get("started_at"), row.get("ended_at")
        if (
            isinstance(start, bool)
            or isinstance(end, bool)
            or not isinstance(start, (int, float))
            or not isinstance(end, (int, float))
            or not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise SignalContractError(
                "recording common-clock interval must be finite and increasing"
            )
        if not isinstance(row.get("source_clock_reset_observed"), bool):
            raise SignalContractError("source clock reset observation must be explicit")
        if row["source_clock_reset_observed"] and row.get("clock_mapping_receipt") is None:
            raise SignalContractError("clock reset requires a reviewed mapping receipt")
        visits.setdefault((str(row["wearer_id"]), int(row["visit_number"])), []).append(row)
    wearer_intervals: dict[str, list[tuple[datetime, datetime]]] = {}
    for (wearer, _visit), visit_rows in sorted(visits.items()):
        ordered = sorted(visit_rows, key=lambda row: int(row["attachment_number"]))
        anchors = {_utc(row["visit_started_at_utc"], "visit start") for row in ordered}
        if len(anchors) != 1:
            raise SignalContractError("attachments do not share one visit UTC anchor")
        for first, second in pairwise(ordered):
            if float(second["started_at"]) < float(first["ended_at"]):
                raise SignalContractError(
                    "attachment clock reset/overlap violates common visit clock"
                )
        anchor = next(iter(anchors))
        wearer_intervals.setdefault(wearer, []).append(
            (
                anchor + timedelta(seconds=float(ordered[0]["started_at"])),
                anchor + timedelta(seconds=float(ordered[-1]["ended_at"])),
            )
        )
    for intervals in wearer_intervals.values():
        for earlier, later in pairwise(intervals):
            if later[0] < earlier[1]:
                raise SignalContractError("wearer visit chronology overlaps or reverses")


def _load_collection(
    *,
    manifest: Mapping[str, Any],
    canonical_root: Path,
    contract: DeviceSignalContract,
    plan: Mapping[str, Any],
    copy_root: Path | None,
) -> tuple[list[CanonicalRecording], list[dict[str, Any]]]:
    expected_fields = {"record_kind", "schema_version", "recordings", "record_sha256"}
    if set(manifest) != expected_fields:
        raise SignalContractError("canonical collection manifest fields are not exact")
    if manifest.get("record_kind") != "physical_information_canonical_collection_manifest":
        raise SignalContractError("unexpected canonical collection manifest kind")
    if manifest.get("schema_version") != "1.0.0":
        raise SignalContractError("unexpected canonical collection schema")
    _verify_sealed(manifest, "canonical collection manifest")
    rows = _sequence_of_mappings(manifest.get("recordings"), "recordings")
    required = {
        "recording_id",
        "block_id",
        "wearer_id",
        "visit_number",
        "attachment_number",
        "device_id",
        "canonical_path",
        "canonical_sha256",
        "canonical_size_bytes",
        "visit_started_at_utc",
        "started_at",
        "ended_at",
        "source_clock_reset_observed",
        "clock_mapping_receipt",
    }
    planned_blocks = {
        str(item["block_id"]): item
        for item in cast(list[Mapping[str, Any]], plan.get("blocks", []))
    }
    if len(rows) != len(planned_blocks):
        raise SignalContractError(
            "canonical collection must contain exactly one recording per block"
        )
    _collection_chronology(rows, contract)
    seen_recordings: set[str] = set()
    seen_blocks: set[str] = set()
    recordings: list[CanonicalRecording] = []
    replay_rows: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if set(row) != required:
            raise SignalContractError(f"canonical recording fields are not exact at index {index}")
        recording_id = str(row.get("recording_id", ""))
        block_id = str(row.get("block_id", ""))
        if not recording_id or recording_id in seen_recordings:
            raise SignalContractError(f"duplicate or missing recording_id: {recording_id!r}")
        if block_id not in planned_blocks or block_id in seen_blocks:
            raise SignalContractError(f"duplicate or unplanned block_id: {block_id!r}")
        block = planned_blocks[block_id]
        for field in ("wearer_id", "visit_number", "attachment_number"):
            if row.get(field) != block.get(field):
                raise SignalContractError(f"recording {recording_id} has wrong {field}")
        if row.get("device_id") != contract.device_id:
            raise SignalContractError(f"recording {recording_id} has wrong device_id")
        relative = _safe_relative_path(row.get("canonical_path"), "canonical_path")
        source = _resolve_below(canonical_root, relative)
        if row.get("canonical_sha256") != sha256_file(source):
            raise SignalContractError(f"canonical recording hash mismatch: {recording_id}")
        size_value = row.get("canonical_size_bytes")
        if (
            isinstance(size_value, bool)
            or not isinstance(size_value, int)
            or size_value != source.stat().st_size
        ):
            raise SignalContractError(f"canonical recording size mismatch: {recording_id}")
        load_path = source
        replay_path = relative.as_posix()
        if copy_root is not None:
            copy_relative = PurePosixPath("canonical_inputs") / f"recording-{index:02d}.npz"
            load_path = copy_root.joinpath(*copy_relative.parts)
            _copy_create_only(source, load_path, root=copy_root)
            replay_path = copy_relative.as_posix()
        if (
            sha256_file(load_path) != row["canonical_sha256"]
            or load_path.stat().st_size != size_value
        ):
            raise SignalContractError(
                f"copied canonical bytes differ from source manifest: {recording_id}"
            )
        arrays = _read_npz(load_path)
        timestamps = np.asarray(arrays["timestamps"], dtype=np.float64)
        if contract.timestamp_unit == "ms":
            timestamps = timestamps / 1000.0
        finite = timestamps[np.isfinite(timestamps)]
        if (
            not finite.size
            or finite.min() < float(row["started_at"]) - 1e-9
            or finite.max() > float(row["ended_at"]) + 1e-9
        ):
            raise SignalContractError(
                "canonical timestamps are outside the declared common visit clock"
            )
        mapping_reference = row["clock_mapping_receipt"]
        replay_mapping = mapping_reference
        if mapping_reference is not None:
            reference = _mapping(mapping_reference, "clock mapping receipt")
            mapping_source = _resolve_below(
                canonical_root, _safe_relative_path(reference.get("path"), "clock mapping")
            )
            if sha256_file(mapping_source) != reference.get("sha256"):
                raise SignalContractError("clock mapping receipt hash mismatch")
            mapping = _mapping(load_json_strict(mapping_source), "clock mapping receipt")
            _verify_sealed(mapping, "clock mapping receipt")
            if (
                mapping.get("record_kind") != "physical_information_clock_mapping_receipt"
                or mapping.get("status") != "reviewed"
                or mapping.get("recording_id") != recording_id
                or mapping.get("canonical_sha256") != row["canonical_sha256"]
                or mapping.get("clock_domain") != contract.clock_domain
                or not isinstance(mapping.get("reviewer_id"), str)
                or not mapping["reviewer_id"]
                or not isinstance(mapping.get("mapping_description"), str)
                or not mapping["mapping_description"]
            ):
                raise SignalContractError(
                    "clock mapping is not reviewed and bound to canonical bytes"
                )
            _utc(mapping.get("reviewed_at_utc"), "clock mapping review")
            if copy_root is not None:
                mapping_relative = f"canonical_inputs/clock-mapping-{index:02d}.json"
                _copy_create_only(mapping_source, copy_root / mapping_relative, root=copy_root)
                if sha256_file(copy_root / mapping_relative) != reference["sha256"]:
                    raise SignalContractError("copied clock mapping receipt changed")
                replay_mapping = {"path": mapping_relative, "sha256": reference["sha256"]}
        metadata = {
            "recording_id": recording_id,
            "block_id": block_id,
            "wearer_id": row["wearer_id"],
            "visit_number": row["visit_number"],
            "attachment_number": row["attachment_number"],
        }
        recordings.append(
            canonical_recording_from_arrays(metadata=metadata, arrays=arrays, contract=contract)
        )
        replay_rows.append(
            {
                **metadata,
                "device_id": contract.device_id,
                "canonical_path": replay_path,
                "canonical_sha256": sha256_file(load_path),
                "canonical_size_bytes": load_path.stat().st_size,
                "source_canonical_path": relative.as_posix(),
                "source_canonical_sha256": str(row["canonical_sha256"]),
                "visit_started_at_utc": row["visit_started_at_utc"],
                "started_at": row["started_at"],
                "ended_at": row["ended_at"],
                "source_clock_reset_observed": row["source_clock_reset_observed"],
                "clock_mapping_receipt": replay_mapping,
            }
        )
        seen_recordings.add(recording_id)
        seen_blocks.add(block_id)
    if seen_blocks != set(planned_blocks):
        raise SignalContractError("canonical collection does not cover the exact frozen block set")
    return recordings, replay_rows


def _validate_annotations(
    annotations: Sequence[Mapping[str, Any]], plan: Mapping[str, Any]
) -> list[dict[str, Any]]:
    required = {
        "bout_id",
        "recording_id",
        "start_sample",
        "stop_sample_exclusive",
        "observed_activity",
        "observed_motion",
        "adjudication_status",
        "adjudicator_id",
        "adjudicated_at_utc",
    }
    result = [dict(item) for item in annotations]
    if any(set(item) != required for item in result):
        raise SignalContractError("annotation fields are not exact")
    planned = {str(item["bout_id"]) for item in cast(list[Mapping[str, Any]], plan["bouts"])}
    observed = [str(item.get("bout_id", "")) for item in result]
    if len(observed) != len(planned) or set(observed) != planned:
        raise SignalContractError("annotations must cover the exact frozen bout set")
    if len(observed) != len(set(observed)):
        raise SignalContractError("annotation bout identifiers are not unique")
    planned_by_bout = {
        str(item["bout_id"]): item for item in cast(list[Mapping[str, Any]], plan["bouts"])
    }
    allowed_activities = {"sitting", "standing", "mobility", "other", "unresolved"}
    allowed_motions = {
        "quiet",
        "upper_body_motion",
        "usual",
        "slow",
        "turning",
        "other",
        "unresolved",
    }
    allowed_statuses = {"adjudicated", "failed", "withdrawn", "unresolved"}
    by_recording: dict[str, list[dict[str, Any]]] = {}
    for item in result:
        bout_id = str(item["bout_id"])
        recording_id = item.get("recording_id")
        start = item.get("start_sample")
        stop = item.get("stop_sample_exclusive")
        if not isinstance(recording_id, str) or not recording_id:
            raise SignalContractError(f"annotation recording_id is missing: {bout_id}")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(stop, bool)
            or not isinstance(stop, int)
            or start < 0
            or stop <= start
        ):
            raise SignalContractError(f"annotation sample interval is invalid: {bout_id}")
        if item.get("observed_activity") not in allowed_activities:
            raise SignalContractError(f"annotation observed_activity is invalid: {bout_id}")
        if item.get("observed_motion") not in allowed_motions:
            raise SignalContractError(f"annotation observed_motion is invalid: {bout_id}")
        if item.get("adjudication_status") not in allowed_statuses:
            raise SignalContractError(f"annotation adjudication_status is invalid: {bout_id}")
        if not isinstance(item.get("adjudicator_id"), str) or not item["adjudicator_id"]:
            raise SignalContractError(f"annotation adjudicator_id is missing: {bout_id}")
        _utc(item.get("adjudicated_at_utc"), f"annotation adjudicated_at_utc: {bout_id}")
        if planned_by_bout[bout_id].get("sequence_in_block") is None:
            raise SignalContractError(f"planned bout sequence is missing: {bout_id}")
        by_recording.setdefault(recording_id, []).append(item)
    for recording_id, rows in by_recording.items():
        ordered = sorted(rows, key=lambda item: int(item["start_sample"]))
        for left, right in pairwise(ordered):
            if int(left["stop_sample_exclusive"]) > int(right["start_sample"]):
                raise SignalContractError(
                    f"annotation intervals overlap in {recording_id}: "
                    f"{left['bout_id']} and {right['bout_id']}"
                )
        sequences = [
            int(planned_by_bout[str(item["bout_id"])]["sequence_in_block"]) for item in ordered
        ]
        if sequences != sorted(sequences):
            raise SignalContractError(
                f"annotation chronology violates planned sequence: {recording_id}"
            )
    return result


def _validate_annotation_recording_bindings(
    annotations: Sequence[Mapping[str, Any]],
    recordings: Sequence[CanonicalRecording],
    plan: Mapping[str, Any],
) -> None:
    planned_by_bout = {
        str(item["bout_id"]): item for item in cast(list[Mapping[str, Any]], plan["bouts"])
    }
    recording_by_id = {item.recording_id: item for item in recordings}
    for annotation in annotations:
        bout_id = str(annotation["bout_id"])
        recording_id = str(annotation["recording_id"])
        recording = recording_by_id.get(recording_id)
        if recording is None:
            raise SignalContractError(f"annotation recording is absent: {bout_id}")
        if recording.block_id != planned_by_bout[bout_id].get("block_id"):
            raise SignalContractError(
                f"annotation is bound to the wrong attachment block: {bout_id}"
            )
        if int(annotation["stop_sample_exclusive"]) > recording.timestamps.size:
            raise SignalContractError(f"annotation exceeds canonical recording: {bout_id}")


def _artifact_rows(output_directory: Path, *, excluded: set[str]) -> list[dict[str, Any]]:
    return [
        {
            "path": path.relative_to(output_directory).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name not in excluded
    ]


def _seal_artifact_manifest(output_directory: Path, *, excluded: set[str]) -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "physical_information_signal_artifact_manifest",
            "created_at_utc": _now_z(),
            "artifacts": _artifact_rows(output_directory, excluded=excluded),
        }
    )


def _verify_artifact_manifest(run_directory: Path) -> None:
    manifest = _mapping(
        load_json_strict(run_directory / "artifact_manifest.json"), "artifact manifest"
    )
    _verify_sealed(manifest, "artifact manifest")
    rows = _sequence_of_mappings(manifest.get("artifacts"), "artifact manifest artifacts")
    names = [str(row.get("path")) for row in rows]
    if len(names) != len(set(names)):
        raise SignalContractError("artifact manifest paths are not unique")
    lifecycle = {
        "artifact_manifest.json",
        "validation.json",
        "completion_manifest.json",
        "controller_final_receipt.json",
    }
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    if set(names) != actual - lifecycle:
        raise SignalContractError("artifact manifest completeness mismatch")
    for row in rows:
        relative = _safe_relative_path(row.get("path"), "artifact path")
        path = _resolve_below(run_directory, relative)
        if row.get("size_bytes") != path.stat().st_size or row.get("sha256") != sha256_file(path):
            raise SignalContractError(f"artifact manifest mismatch: {relative}")


def _verify_completion(run_directory: Path, *, require_final: bool) -> None:
    completion_path = run_directory / "completion_manifest.json"
    final_path = run_directory / "controller_final_receipt.json"
    if not completion_path.exists():
        if require_final or final_path.exists():
            raise SignalContractError("completion manifest is missing")
        return
    completion = _mapping(load_json_strict(completion_path), "completion manifest")
    _verify_sealed(completion, "completion manifest")
    rows = _sequence_of_mappings(completion.get("artifacts"), "completion artifacts")
    names = [str(row.get("path")) for row in rows]
    actual = {
        path.relative_to(run_directory).as_posix()
        for path in run_directory.rglob("*")
        if path.is_file()
    }
    if len(names) != len(set(names)) or set(names) != actual - {
        "completion_manifest.json",
        "controller_final_receipt.json",
    }:
        raise SignalContractError("completion manifest completeness/uniqueness mismatch")
    for row in rows:
        path = _resolve_below(
            run_directory, _safe_relative_path(row.get("path"), "completion path")
        )
        if sha256_file(path) != row.get("sha256") or path.stat().st_size != row.get("size_bytes"):
            raise SignalContractError("completion artifact integrity mismatch")
    validation = _mapping(load_json_strict(run_directory / "validation.json"), "validation")
    _verify_sealed(validation, "validation")
    if validation.get("valid") is not True or validation.get("model_fit_count") != 0:
        raise SignalContractError("retained validation is not a zero-fit pass")
    if (
        completion.get("validation_record_sha256") is not None
        and completion["validation_record_sha256"] != validation["record_sha256"]
    ):
        raise SignalContractError("completion validation binding mismatch")
    if not final_path.exists():
        if require_final:
            raise SignalContractError("controller final receipt is missing")
        return
    final = _mapping(load_json_strict(final_path), "controller final receipt")
    _verify_sealed(final, "controller final receipt")
    if (
        final.get("completion_manifest_sha256") != sha256_file(completion_path)
        or final.get("model_fit_count") != 0
        or final.get("task_owned_workers_remaining") != 0
    ):
        raise SignalContractError("controller completion/shutdown binding mismatch")


def _compute_from_snapshots(run_directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = _mapping(load_json_strict(run_directory / "plan_snapshot.json"), "plan snapshot")
    config = load_config(run_directory / "config_snapshot.yaml")
    validate_config(config)
    plan_validation = validate_plan(plan, config)
    if plan_validation.get("valid") is not True:
        raise SignalContractError("plan snapshot does not match frozen config snapshot")
    contract_value = _mapping(
        load_json_strict(run_directory / "contract_snapshot.json"), "contract snapshot"
    )
    contract = parse_device_signal_contract(contract_value)
    qualification_path = run_directory / "qualification_manifest_snapshot.json"
    qualification = _qualification(
        contract=contract,
        plan=plan,
        config=config,
        manifest_path=qualification_path if qualification_path.exists() else None,
        root=run_directory / "qualification_inputs",
        copy_root=None,
    )
    stored_qualification = _mapping(
        load_json_strict(run_directory / "qualification_validation.json"),
        "qualification validation",
    )
    _verify_sealed(stored_qualification, "qualification validation")
    if {
        key: value for key, value in stored_qualification.items() if key != "record_sha256"
    } != qualification:
        raise SignalContractError("qualification evidence did not replay")
    annotations = _sequence_of_mappings(
        load_json_strict(run_directory / "annotations_snapshot.json"), "annotations"
    )
    replay_manifest = _mapping(
        load_json_strict(run_directory / "replay_collection_manifest.json"), "replay manifest"
    )
    _verify_sealed(replay_manifest, "replay collection manifest")
    # The replay rows include source lineage in addition to the input schema; remove it here.
    replay_rows = _sequence_of_mappings(replay_manifest.get("recordings"), "replay recordings")
    if any(
        row.get("canonical_sha256") != row.get("source_canonical_sha256") for row in replay_rows
    ):
        raise SignalContractError("replay canonical/source lineage hash mismatch")
    input_rows = [
        {
            key: row[key]
            for key in (
                "recording_id",
                "block_id",
                "wearer_id",
                "visit_number",
                "attachment_number",
                "device_id",
                "canonical_path",
                "canonical_sha256",
                "canonical_size_bytes",
                "visit_started_at_utc",
                "started_at",
                "ended_at",
                "source_clock_reset_observed",
                "clock_mapping_receipt",
            )
        }
        for row in replay_rows
    ]
    input_manifest = _sealed(
        {
            "record_kind": "physical_information_canonical_collection_manifest",
            "schema_version": "1.0.0",
            "recordings": input_rows,
        }
    )
    recordings, _ = _load_collection(
        manifest=input_manifest,
        canonical_root=run_directory,
        contract=contract,
        plan=plan,
        copy_root=None,
    )
    annotations = _validate_annotations(annotations, plan)
    _validate_annotation_recording_bindings(annotations, recordings, plan)
    windows = build_window_table(recordings, contract)
    projections = project_bouts(
        plan_bouts=cast(list[Mapping[str, Any]], plan["bouts"]),
        annotations=annotations,
        recordings=recordings,
        windows=windows,
        source_rate_hz=contract.processing.source_rate_hz,
    )
    analysis = analyze_physical_information(
        plan=plan, projections=projections, windows=windows, contract=contract
    )
    return analysis, {
        "windows": windows,
        "projections": projections,
        "recording_count": len(recordings),
        "signal_audit": build_signal_audit(recordings, contract, windows),
    }


def validate_signal_run(
    run_directory: Path,
    *,
    write_receipt: bool = False,
    require_final: bool = True,
) -> dict[str, Any]:
    """Replay from copied canonical inputs without fitting or trusting cached outputs."""

    started = time.perf_counter()
    run_directory = run_directory.resolve(strict=True)
    if (run_directory / "INCOMPLETE.json").exists():
        raise SignalContractError("incomplete run cannot qualify as completed signal analysis")
    _verify_artifact_manifest(run_directory)
    _verify_completion(run_directory, require_final=require_final)
    stored_analysis = _mapping(load_json_strict(run_directory / "analysis.json"), "stored analysis")
    _verify_sealed(stored_analysis, "stored analysis")
    stored_projection = _mapping(
        load_json_strict(run_directory / "bout_projection.json"), "stored bout projection"
    )
    _verify_sealed(stored_projection, "stored bout projection")
    stored_body = dict(stored_analysis)
    stored_body.pop("record_sha256")
    recomputed, detail = _compute_from_snapshots(run_directory)
    cached = window_table_from_arrays(_read_npz(run_directory / "signal_cache.npz"))
    audit = _mapping(load_json_strict(run_directory / "signal_audit.json"), "signal audit")
    _verify_sealed(audit, "signal audit")
    audit_body = dict(audit)
    audit_body.pop("record_sha256")
    result = _mapping(load_json_strict(run_directory / "result.json"), "result")
    _verify_sealed(result, "result")
    result_matches = (
        result.get("status") == "analysis_complete_replay_pending"
        and result.get("model_fit_count") == 0
        and result.get("recording_count") == detail["recording_count"]
        and result.get("planned_bout_count") == len(detail["projections"])
        and result.get("eligible_bout_count")
        == sum(row["eligible"] is True for row in detail["projections"])
        and result.get("window_table_sha256") == recomputed.get("window_table_sha256")
        and result.get("physical_identifiability_gate")
        == recomputed.get("physical_identifiability_gate")
        and result.get("claims") == recomputed.get("claims")
    )
    final_path = run_directory / "controller_final_receipt.json"
    if final_path.exists():
        final_receipt = _mapping(load_json_strict(final_path), "final receipt")
        result_matches = result_matches and (
            final_receipt.get("status") == "complete"
            and final_receipt.get("physical_identifiability_gate")
            == recomputed.get("physical_identifiability_gate")
            and final_receipt.get("claims") == recomputed.get("claims")
        )
    valid = (
        recomputed == stored_body
        and audit_body == detail["signal_audit"]
        and result_matches
        and stored_projection.get("bouts") == detail["projections"]
        and stored_projection.get("window_table_sha256") == recomputed.get("window_table_sha256")
        and stored_projection.get("model_fit_count") == 0
        and window_table_hash(cached) == recomputed.get("window_table_sha256")
    )
    record = _sealed(
        {
            "record_kind": "physical_information_signal_zero_fit_replay_validation",
            "validated_at_utc": _now_z(),
            "valid": valid,
            "model_fit_count": 0,
            "source_inputs_reloaded": True,
            "signal_preprocessing_recomputed": True,
            "bout_projection_recomputed": True,
            "stored_bout_projection_exact_match": stored_projection.get("bouts")
            == detail["projections"],
            "analysis_recomputed": True,
            "stored_analysis_record_sha256": stored_analysis["record_sha256"],
            "recomputed_analysis_sha256": canonical_json_sha256(recomputed),
            "window_table_sha256": recomputed.get("window_table_sha256"),
            "recording_count": detail["recording_count"],
            "window_count": len(detail["windows"].window_ids),
            "wall_seconds": time.perf_counter() - started,
            "task_owned_workers_remaining": 0,
        }
    )
    if write_receipt:
        _write_json(run_directory / "validation.json", record, root=run_directory)
    if not valid:
        raise SignalContractError("independent zero-fit replay differs from stored analysis/cache")
    return record


def execute_signal_run(
    *,
    repository_root: Path,
    plan_path: Path,
    config_path: Path,
    contract_path: Path,
    collection_manifest_path: Path,
    annotations_path: Path,
    canonical_root: Path,
    output_directory: Path,
    code_commit: str,
    qualification_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Execute the frozen signal analysis and seal an independently replayed package."""

    started = time.perf_counter()
    repository_root = repository_root.resolve(strict=True)
    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite output directory: {output_directory}")
    state = _validate_source_lock(repository_root, code_commit)
    plan, _config = _load_plan_and_config(
        plan_path.resolve(strict=True), config_path.resolve(strict=True)
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    try:
        contract_value = _mapping(load_json_strict(contract_path), "signal contract")
        contract = parse_device_signal_contract(contract_value)
        qualification_root = output_directory / "qualification_inputs"
        if qualification_manifest_path is not None:
            qualification_root.mkdir()
        qualification = _qualification(
            contract=contract,
            plan=plan,
            config=_config,
            manifest_path=qualification_manifest_path,
            root=canonical_root.resolve(strict=True),
            copy_root=qualification_root if qualification_manifest_path is not None else None,
        )
        _write_json(
            output_directory / "qualification_validation.json",
            _sealed(qualification),
            root=output_directory,
        )
        if qualification_manifest_path is not None:
            _write_bytes(
                output_directory / "qualification_manifest_snapshot.json",
                qualification_manifest_path.read_bytes(),
                root=output_directory,
            )
        manifest = _mapping(load_json_strict(collection_manifest_path), "collection manifest")
        annotations = _validate_annotations(
            _sequence_of_mappings(load_json_strict(annotations_path), "annotations"), plan
        )
        recordings, replay_rows = _load_collection(
            manifest=manifest,
            canonical_root=canonical_root.resolve(strict=True),
            contract=contract,
            plan=plan,
            copy_root=output_directory,
        )
        _validate_annotation_recording_bindings(annotations, recordings, plan)
        _write_json(output_directory / "plan_snapshot.json", plan, root=output_directory)
        _write_bytes(
            output_directory / "config_snapshot.yaml",
            config_path.read_bytes(),
            root=output_directory,
        )
        _write_json(
            output_directory / "contract_snapshot.json", contract_value, root=output_directory
        )
        _write_bytes(
            output_directory / "annotations_snapshot.json",
            json.dumps(annotations, allow_nan=False, sort_keys=True, separators=(",", ":")).encode()
            + b"\n",
            root=output_directory,
        )
        replay_manifest = _sealed(
            {
                "record_kind": "physical_information_signal_replay_collection_manifest",
                "schema_version": "1.0.0",
                "recordings": replay_rows,
            }
        )
        _write_json(
            output_directory / "replay_collection_manifest.json",
            replay_manifest,
            root=output_directory,
        )
        input_manifest = _sealed(
            {
                "record_kind": "physical_information_signal_input_manifest",
                "created_at_utc": _now_z(),
                "code_commit": code_commit,
                "worktree_clean": state["clean"],
                "executing_sources": state.get("executing_sources", {}),
                "plan": {"path": str(plan_path.resolve()), "sha256": sha256_file(plan_path)},
                "config": {
                    "path": str(config_path.resolve()),
                    "sha256": sha256_file(config_path),
                },
                "contract": {
                    "path": str(contract_path.resolve()),
                    "sha256": sha256_file(contract_path),
                },
                "collection_manifest": {
                    "path": str(collection_manifest_path.resolve()),
                    "sha256": sha256_file(collection_manifest_path),
                },
                "annotations": {
                    "path": str(annotations_path.resolve()),
                    "sha256": sha256_file(annotations_path),
                },
                "copied_canonical_recording_count": len(recordings),
                "model_fit_budget": 0,
            }
        )
        _write_json(output_directory / "input_manifest.json", input_manifest, root=output_directory)
        windows = build_window_table(recordings, contract)
        signal_audit = _sealed(build_signal_audit(recordings, contract, windows))
        _write_json(output_directory / "signal_audit.json", signal_audit, root=output_directory)
        projections = project_bouts(
            plan_bouts=cast(list[Mapping[str, Any]], plan["bouts"]),
            annotations=annotations,
            recordings=recordings,
            windows=windows,
            source_rate_hz=contract.processing.source_rate_hz,
        )
        analysis_body = analyze_physical_information(
            plan=plan, projections=projections, windows=windows, contract=contract
        )
        _write_npz(
            output_directory / "signal_cache.npz",
            window_table_arrays(windows),
            root=output_directory,
        )
        projection_record = _sealed(
            {
                "record_kind": "physical_information_signal_bout_projection",
                "model_fit_count": 0,
                "window_table_sha256": window_table_hash(windows),
                "bouts": projections,
            }
        )
        _write_json(
            output_directory / "bout_projection.json", projection_record, root=output_directory
        )
        analysis_record = _sealed(analysis_body)
        _write_json(output_directory / "analysis.json", analysis_record, root=output_directory)
        result = _sealed(
            {
                "record_kind": "physical_information_signal_result",
                "schema_version": "1.0.0",
                "status": "analysis_complete_replay_pending",
                "model_fit_count": 0,
                "recording_count": len(recordings),
                "planned_bout_count": len(projections),
                "eligible_bout_count": sum(item["eligible"] is True for item in projections),
                "window_table_sha256": window_table_hash(windows),
                "physical_identifiability_gate": analysis_body["physical_identifiability_gate"],
                "claims": analysis_body["claims"],
            }
        )
        _write_json(output_directory / "result.json", result, root=output_directory)
        runtime = _sealed(
            {
                "record_kind": "physical_information_signal_runtime",
                "finished_at_utc": _now_z(),
                "wall_seconds_before_replay": time.perf_counter() - started,
                "recording_count": len(recordings),
                "window_count": len(windows.window_ids),
                "planned_bout_count": len(projections),
                "eligible_bout_count": sum(item["eligible"] is True for item in projections),
                "model_fit_count": 0,
            }
        )
        _write_json(output_directory / "runtime.json", runtime, root=output_directory)
        shutdown = _sealed(
            {
                "record_kind": "physical_information_signal_worker_shutdown",
                "workers_started": 0,
                "workers_remaining": 0,
                "monitors_remaining": 0,
            }
        )
        _write_json(output_directory / "worker_shutdown.json", shutdown, root=output_directory)
        artifact_manifest = _seal_artifact_manifest(
            output_directory,
            excluded={
                "artifact_manifest.json",
                "validation.json",
                "completion_manifest.json",
                "controller_final_receipt.json",
            },
        )
        _write_json(
            output_directory / "artifact_manifest.json", artifact_manifest, root=output_directory
        )
        validation = validate_signal_run(output_directory, write_receipt=True, require_final=False)
        completion = _sealed(
            {
                "record_kind": "physical_information_signal_completion_manifest",
                "status": "complete_zero_fit_analysis_replayed",
                "completed_at_utc": _now_z(),
                "model_fit_count": 0,
                "validation_record_sha256": validation["record_sha256"],
                "task_owned_workers_remaining": 0,
                "artifacts": _artifact_rows(
                    output_directory,
                    excluded={"completion_manifest.json", "controller_final_receipt.json"},
                ),
            }
        )
        _write_json(
            output_directory / "completion_manifest.json", completion, root=output_directory
        )
        final = _sealed(
            {
                "record_kind": "physical_information_signal_controller_final_receipt",
                "status": "complete",
                "finished_at_utc": _now_z(),
                "model_fit_count": 0,
                "task_owned_workers_remaining": 0,
                "completion_manifest_sha256": sha256_file(
                    output_directory / "completion_manifest.json"
                ),
                "total_wall_seconds": time.perf_counter() - started,
                "physical_identifiability_gate": analysis_body["physical_identifiability_gate"],
                "claims": analysis_body["claims"],
            }
        )
        _write_json(
            output_directory / "controller_final_receipt.json", final, root=output_directory
        )
        _verify_completion(output_directory, require_final=True)
        return final
    except BaseException as exc:
        _record_failure(output_directory, started, exc)
        raise


def _record_failure(output: Path, started: float, exc: BaseException) -> None:
    """Retain the original failure even if an earlier blocker/completion record exists."""
    failure = _sealed(
        {
            "record_kind": "physical_information_signal_incomplete_failure",
            "failed_at_utc": _now_z(),
            "status": "incomplete_validation_or_execution_failure",
            "reason_type": type(exc).__name__,
            "reason": str(exc),
            "physical_identifiability_gate": {"status": "not_evaluated"},
            "model_fit_count": 0,
            "task_owned_workers_remaining": 0,
            "claims": {
                "physical_identifiability_demonstrated": False,
                "population_superiority_demonstrated": False,
                "architecture_eligible": False,
            },
        }
    )
    try:
        if not (output / "INCOMPLETE.json").exists():
            _write_json(output / "INCOMPLETE.json", failure, root=output)
        if not (output / "failure_reason.json").exists():
            _write_json(output / "failure_reason.json", failure, root=output)
    finally:
        _finalize_failure(output, started)


def _finalize_failure(output: Path, started: float) -> None:
    """Seal additional failure evidence without replacing any earlier lifecycle record."""
    records: dict[str, dict[str, Any]] = {
        "failure_runtime.json": {
            "wall_seconds": time.perf_counter() - started,
            "model_fit_count": 0,
        },
        "failure_shutdown.json": {
            "workers_started": 0,
            "workers_remaining": 0,
            "monitors_remaining": 0,
            "model_fit_count": 0,
        },
    }
    for name, value in records.items():
        if not (output / name).exists():
            _write_json(output / name, _sealed(value), root=output)
    manifest_path = output / "failure_manifest.json"
    if not manifest_path.exists():
        _write_json(
            manifest_path,
            _seal_artifact_manifest(
                output, excluded={"failure_manifest.json", "failure_final_receipt.json"}
            ),
            root=output,
        )
    final_path = output / "failure_final_receipt.json"
    if not final_path.exists():
        _write_json(
            final_path,
            _sealed(
                {
                    "status": "incomplete_failure_sealed",
                    "failure_manifest_sha256": sha256_file(manifest_path),
                    "model_fit_count": 0,
                    "task_owned_workers_remaining": 0,
                }
            ),
            root=output,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    blocker = subparsers.add_parser("blocker")
    blocker.add_argument("--repository-root", type=Path, default=Path.cwd())
    blocker.add_argument("--plan", type=Path, required=True)
    blocker.add_argument("--config", type=Path, required=True)
    blocker.add_argument("--output", type=Path, required=True)
    blocker.add_argument("--code-commit", required=True)
    execute = subparsers.add_parser("execute")
    execute.add_argument("--repository-root", type=Path, default=Path.cwd())
    execute.add_argument("--plan", type=Path, required=True)
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--contract", type=Path, required=True)
    execute.add_argument("--collection-manifest", type=Path, required=True)
    execute.add_argument("--annotations", type=Path, required=True)
    execute.add_argument("--canonical-root", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    execute.add_argument("--code-commit", required=True)
    execute.add_argument("--qualification-manifest", type=Path)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "blocker":
        result = emit_blocker_record(
            repository_root=arguments.repository_root,
            plan_path=arguments.plan,
            config_path=arguments.config,
            output_directory=arguments.output,
            code_commit=str(arguments.code_commit),
        )
    elif arguments.command == "execute":
        result = execute_signal_run(
            repository_root=arguments.repository_root,
            plan_path=arguments.plan,
            config_path=arguments.config,
            contract_path=arguments.contract,
            collection_manifest_path=arguments.collection_manifest,
            annotations_path=arguments.annotations,
            canonical_root=arguments.canonical_root,
            output_directory=arguments.output,
            code_commit=str(arguments.code_commit),
            qualification_manifest_path=arguments.qualification_manifest,
        )
    else:
        result = validate_signal_run(arguments.run_directory)
    print(json.dumps(result, allow_nan=False, sort_keys=True))
    return 0 if result.get("valid", True) is True else 2


if __name__ == "__main__":
    raise SystemExit(main())
