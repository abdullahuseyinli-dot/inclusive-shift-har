"""Create and validate the physical-information pilot acquisition package.

The commands generate a plan or validate later records. They do not acquire
human data, control a device, process signals before device qualification, or
train a model.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from jsonschema import Draft202012Validator

from inclusive_shift_har.data.physical_information_pilot import (
    build_plan,
    load_config,
    load_plan,
    validate_collection_records,
    validate_config,
    validate_plan,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

REQUIRED_PLAN_RUN_FILES = (
    "blocker_record.json",
    "config_snapshot.yaml",
    "equipment_receipt_template.json",
    "plan_manifest.json",
    "protocol_snapshot.md",
    "readiness.json",
    "runbook_snapshot.md",
    "schema_snapshot.json",
)
EXPECTED_INPUT_PATHS = {
    "config": Path("configs/experiments/physical_information_identifiability_v1.yaml"),
    "protocol": Path("docs/research/PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md"),
    "schema": Path("configs/schema/physical_information_pilot.schema.json"),
}


def _now_z() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return {str(key): item for key, item in value.items()}


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
        f"{name} self-hash mismatch",
    )


def _write_bytes_create_only(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _write_json_create_only(path: Path, payload: Mapping[str, Any], *, root: Path) -> None:
    atomic_write_json_new(dict(payload), path, allowed_root=root)


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


def _schema_validate(plan: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    errors = sorted(validator.iter_errors(plan), key=lambda item: list(item.absolute_path))
    if errors:
        first = errors[0]
        location = "/".join(str(item) for item in first.absolute_path)
        raise ValueError(
            f"plan schema validation failed at {location or '<root>'}: {first.message}"
        )


def _schema_errors(
    value: object,
    schema: Mapping[str, Any],
    *,
    reference: str | None,
    label: str,
) -> list[str]:
    if reference is None:
        selected_schema: Mapping[str, Any] = schema
    else:
        selected_schema = {
            "$schema": schema.get("$schema"),
            "$defs": schema.get("$defs"),
            "$ref": reference,
        }
    try:
        Draft202012Validator.check_schema(selected_schema)
    except Exception as exc:  # jsonschema exposes several schema-error subclasses
        return [f"{label} schema is invalid: {exc}"]
    validator = Draft202012Validator(
        selected_schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    )
    result: list[str] = []
    for error in sorted(validator.iter_errors(value), key=lambda item: list(item.absolute_path)):
        location = "/".join(str(item) for item in error.absolute_path)
        result.append(f"{label} schema at {location or '<root>'}: {error.message}")
    return result


def _equipment_template() -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "physical_information_equipment_receipt_template",
            "status": "pending_device_qualification",
            "receipt_id": None,
            "device_id": None,
            "manufacturer": None,
            "model": None,
            "firmware": None,
            "logger": None,
            "placement_instruction_path": None,
            "placement_instruction_sha256": None,
            "axis_convention": None,
            "channel_map": None,
            "units": None,
            "clock_domain": None,
            "nominal_rate_hz": None,
            "observed_interval_summary": None,
            "native_gravity_available": None,
            "processing_contract": None,
            "frame_conditioning_limit": None,
            "bench_qualification_path": None,
            "bench_qualification_sha256": None,
            "qualified_at_utc": None,
            "processing_frozen_at_utc": None,
            "instruction": (
                "Populate from measured hardware/bench evidence; unknown fields receive no defaults."
            ),
        }
    )


def _blocker_record(plan: Mapping[str, Any]) -> dict[str, Any]:
    return _sealed(
        {
            "record_kind": "physical_information_pilot_incomplete_blocker",
            "study_id": plan.get("study_id"),
            "status": "collection_not_started_prerequisites_unverified",
            "blocked_phase": "physical_collection",
            "blocking_prerequisites": plan.get("pending_prerequisites"),
            "reason": (
                "Lawful human-study and qualified-device evidence was not provided to or "
                "verifiable by this plan-generation run."
            ),
            "collection_attempted": False,
            "recorded_bout_count": 0,
            "model_fit_count": 0,
            "architecture_experiment_launched": False,
            "claims": plan.get("claims"),
        }
    )


def _artifact_manifest(output_directory: Path) -> dict[str, Any]:
    paths = sorted(
        path
        for path in output_directory.rglob("*")
        if path.is_file()
        and path.name
        not in {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    )
    result = _sealed(
        {
            "record_kind": "physical_information_pilot_artifact_manifest",
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
    _write_json_create_only(
        output_directory / "artifact_manifest.json",
        result,
        root=output_directory,
    )
    return result


def prepare_plan_run(
    *,
    repository_root: Path,
    evidence_root: Path,
    config_path: Path,
    protocol_path: Path,
    schema_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Create the frozen acquisition plan and explicit collection blocker."""

    preparation_started = time.perf_counter()
    repository_root = repository_root.resolve()
    evidence_root = evidence_root.resolve()
    config_path = config_path.resolve()
    protocol_path = protocol_path.resolve()
    schema_path = schema_path.resolve()
    output_directory = output_directory.resolve()
    for path in (config_path, protocol_path, schema_path):
        _require(
            path.is_file() and _is_relative_to(path, repository_root), "input outside repository"
        )
    for name, path in (
        ("config", config_path),
        ("protocol", protocol_path),
        ("schema", schema_path),
    ):
        _require(
            path == (repository_root / EXPECTED_INPUT_PATHS[name]).resolve(),
            f"unexpected {name} input path",
        )
    _require(
        output_directory.parent == evidence_root / ".audit" / "physical_information_pilot",
        "output must be one create-only run below the physical pilot evidence root",
    )
    _require(not output_directory.exists(), "output directory already exists")
    state = _git_state(repository_root)
    _require(state["clean"] is True, "plan source worktree must be clean")
    _require(state["commit"] == code_commit, "code commit does not match HEAD")
    config = load_config(config_path)
    validate_config(config)
    schema = _mapping(load_json_strict(schema_path), str(schema_path))
    runbook_path = protocol_path.with_name(
        protocol_path.name.replace("_PROTOCOL.md", "_RUNBOOK.md")
    )
    _require(runbook_path.is_file(), "pilot runbook is missing")
    output_directory.mkdir(parents=True, exist_ok=False)
    _write_bytes_create_only(output_directory / "config_snapshot.yaml", config_path.read_bytes())
    _write_bytes_create_only(output_directory / "protocol_snapshot.md", protocol_path.read_bytes())
    _write_bytes_create_only(output_directory / "runbook_snapshot.md", runbook_path.read_bytes())
    _write_bytes_create_only(output_directory / "schema_snapshot.json", schema_path.read_bytes())
    plan = build_plan(
        config,
        created_at_utc=_now_z(),
        code_commit=code_commit,
        config_reference={
            "path": config_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(config_path),
        },
        protocol_reference={
            "path": protocol_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(protocol_path),
        },
        schema_reference={
            "path": schema_path.relative_to(repository_root).as_posix(),
            "sha256": sha256_file(schema_path),
        },
    )
    plan_validation = validate_plan(plan, config)
    _require(
        plan_validation["valid"] is True, f"generated plan invalid: {plan_validation['errors']}"
    )
    _schema_validate(plan, schema)
    _write_json_create_only(
        output_directory / "plan_manifest.json",
        plan,
        root=output_directory,
    )
    blocker = _blocker_record(plan)
    _write_json_create_only(
        output_directory / "blocker_record.json",
        blocker,
        root=output_directory,
    )
    equipment_template = _equipment_template()
    _write_json_create_only(
        output_directory / "equipment_receipt_template.json",
        equipment_template,
        root=output_directory,
    )
    readiness = _sealed(
        {
            "record_kind": "physical_information_pilot_readiness",
            "study_id": config["study_id"],
            "status": "plan_valid_collection_pending",
            "plan_validation": plan_validation,
            "schema_validation": "passed",
            "planned_counts": plan["counts"],
            "pending_prerequisites": plan["pending_prerequisites"],
            "device_specific_processing_status": "not_frozen_device_not_qualified",
            "collection_attempted": False,
            "recorded_bout_count": 0,
            "model_fit_count": 0,
            "architecture_experiment_launched": False,
            "claims": plan["claims"],
            "preparation_runtime_seconds": time.perf_counter() - preparation_started,
            "source_manifest": {
                "code": {
                    "path": Path(__file__).resolve().relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(Path(__file__).resolve()),
                },
                "data_contract": {
                    "path": (
                        Path(__file__).resolve().parents[1]
                        / "data"
                        / "physical_information_pilot.py"
                    )
                    .relative_to(repository_root)
                    .as_posix(),
                    "sha256": sha256_file(
                        Path(__file__).resolve().parents[1]
                        / "data"
                        / "physical_information_pilot.py"
                    ),
                },
                "runbook": {
                    "path": runbook_path.relative_to(repository_root).as_posix(),
                    "sha256": sha256_file(runbook_path),
                },
            },
            "git": state,
        }
    )
    _write_json_create_only(
        output_directory / "readiness.json",
        readiness,
        root=output_directory,
    )
    _artifact_manifest(output_directory)
    return readiness


def validate_plan_run(run_directory: Path) -> dict[str, Any]:
    """Validate a plan package without collection or model execution."""

    validation_started = time.perf_counter()
    run_directory = run_directory.resolve()
    _require(run_directory.is_dir(), "plan run directory does not exist")
    _require(not (run_directory / "validation.json").exists(), "validation already exists")
    _require(not (run_directory / "completion_manifest.json").exists(), "completion exists")
    manifest = _mapping(
        load_json_strict(run_directory / "artifact_manifest.json"), "artifact manifest"
    )
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
    _require(set(REQUIRED_PLAN_RUN_FILES).issubset(declared), "required plan artifact missing")
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
    config = load_config(run_directory / "config_snapshot.yaml")
    schema = _mapping(load_json_strict(run_directory / "schema_snapshot.json"), "schema")
    plan = load_plan(run_directory / "plan_manifest.json")
    readiness = _mapping(load_json_strict(run_directory / "readiness.json"), "readiness")
    template = _mapping(
        load_json_strict(run_directory / "equipment_receipt_template.json"),
        "equipment template",
    )
    blocker = _mapping(load_json_strict(run_directory / "blocker_record.json"), "blocker")
    _verify_sealed(readiness, "readiness")
    _verify_sealed(template, "equipment template")
    _verify_sealed(blocker, "blocker")
    _require(template == _equipment_template(), "equipment template semantics changed")
    _require(blocker == _blocker_record(plan), "collection blocker semantics changed")
    plan_validation = validate_plan(plan, config)
    _require(
        plan_validation["valid"] is True, f"plan validation failed: {plan_validation['errors']}"
    )
    _schema_validate(plan, schema)
    lineage = _mapping(plan.get("lineage"), "plan lineage")
    for key, snapshot_name in (
        ("config", "config_snapshot.yaml"),
        ("protocol", "protocol_snapshot.md"),
        ("schema", "schema_snapshot.json"),
    ):
        reference = _mapping(lineage.get(key), f"plan {key} reference")
        _require(
            reference.get("sha256") == sha256_file(run_directory / snapshot_name),
            f"plan {key} hash is not bound to its snapshot",
        )
    _require(
        readiness.get("record_kind") == "physical_information_pilot_readiness",
        "readiness record kind changed",
    )
    _require(readiness.get("study_id") == plan.get("study_id"), "readiness study changed")
    _require(readiness["status"] == "plan_valid_collection_pending", "readiness status changed")
    _require(
        readiness.get("plan_validation") == plan_validation,
        "readiness plan validation changed",
    )
    _require(readiness.get("schema_validation") == "passed", "readiness schema status changed")
    _require(readiness.get("planned_counts") == plan.get("counts"), "readiness counts changed")
    _require(
        readiness.get("pending_prerequisites") == plan.get("pending_prerequisites"),
        "readiness prerequisites changed",
    )
    _require(
        readiness.get("device_specific_processing_status") == "not_frozen_device_not_qualified",
        "readiness device status changed",
    )
    _require(readiness["collection_attempted"] is False, "unrecorded collection claim changed")
    _require(readiness.get("recorded_bout_count") == 0, "recorded bout count changed")
    _require(readiness["model_fit_count"] == 0, "model fit count changed")
    _require(
        readiness.get("architecture_experiment_launched") is False,
        "architecture launch claim changed",
    )
    _require(readiness.get("claims") == plan.get("claims"), "readiness claims changed")
    preparation_runtime = readiness.get("preparation_runtime_seconds")
    _require(
        isinstance(preparation_runtime, (int, float))
        and not isinstance(preparation_runtime, bool)
        and math.isfinite(float(preparation_runtime))
        and float(preparation_runtime) >= 0.0,
        "readiness preparation runtime is invalid",
    )
    preparation_runtime_seconds = float(cast(int | float, preparation_runtime))
    git_state = _mapping(readiness.get("git"), "readiness git state")
    _require(git_state.get("clean") is True, "readiness source was not clean")
    _require(git_state.get("porcelain") == [], "readiness source changes were hidden")
    _require(
        git_state.get("commit") == lineage.get("code_commit"),
        "readiness commit differs from plan lineage",
    )
    source_manifest = _mapping(readiness.get("source_manifest"), "readiness source manifest")
    runbook_reference = _mapping(source_manifest.get("runbook"), "readiness runbook reference")
    _require(
        runbook_reference.get("sha256") == sha256_file(run_directory / "runbook_snapshot.md"),
        "readiness runbook hash is not bound to its snapshot",
    )
    validation_runtime = time.perf_counter() - validation_started
    runtime = _sealed(
        {
            "record_kind": "physical_information_pilot_runtime",
            "preparation_wall_seconds": preparation_runtime_seconds,
            "validation_wall_seconds": validation_runtime,
            "physical_collection_wall_seconds": 0.0,
            "analysis_compute_wall_seconds": 0.0,
            "model_fit_wall_seconds": 0.0,
            "model_fit_count": 0,
        }
    )
    _write_json_create_only(run_directory / "runtime.json", runtime, root=run_directory)
    validation = _sealed(
        {
            "record_kind": "physical_information_pilot_plan_validation",
            "status": "validated_plan_collection_pending",
            "artifact_hashes_verified": True,
            "schema_validation": "passed",
            "schedule_validation": plan_validation,
            "wearer_count": 6,
            "visit_count": 12,
            "block_count": 24,
            "bout_count": 264,
            "nominal_recording_minutes": 150,
            "collection_attempted": False,
            "model_fit_count": 0,
            "architecture_eligibility": "not_evaluated_no_physical_evidence",
            "validation_runtime_seconds": validation_runtime,
        }
    )
    _write_json_create_only(
        run_directory / "validation.json",
        validation,
        root=run_directory,
    )
    final_paths = sorted(path for path in run_directory.rglob("*") if path.is_file())
    completion = _sealed(
        {
            "record_kind": "physical_information_pilot_plan_completion",
            "status": "plan_complete_collection_pending",
            "artifacts": [
                {
                    "path": path.relative_to(run_directory).as_posix(),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in final_paths
            ],
            "collection_attempted": False,
            "model_fit_count": 0,
            "architecture_experiment_launched": False,
            "claims": plan.get("claims"),
        }
    )
    _write_json_create_only(
        run_directory / "completion_manifest.json",
        completion,
        root=run_directory,
    )
    return validation


def check_collection(
    *,
    plan_path: Path,
    config_path: Path,
    schema_path: Path,
    equipment_path: Path,
    recordings_path: Path,
    annotations_path: Path,
    raw_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Validate later collection records and publish one create-only report."""

    for path in (
        plan_path,
        config_path,
        schema_path,
        equipment_path,
        recordings_path,
        annotations_path,
    ):
        _require(path.resolve().is_file() and not path.is_symlink(), f"input file missing: {path}")
    plan = load_plan(plan_path)
    config = load_config(config_path)
    schema = _mapping(load_json_strict(schema_path), "schema")
    equipment_value = load_json_strict(equipment_path)
    recordings_value = load_json_strict(recordings_path)
    annotations_value = load_json_strict(annotations_path)
    schema_errors = _schema_errors(plan, schema, reference=None, label="plan")
    schema_errors.extend(
        _schema_errors(
            equipment_value,
            schema,
            reference="#/$defs/equipment_receipt",
            label="equipment",
        )
    )
    equipment = (
        _mapping(equipment_value, "equipment receipt")
        if isinstance(equipment_value, Mapping)
        else None
    )
    recordings: list[Mapping[str, Any]] = []
    if not isinstance(recordings_value, list):
        schema_errors.append("recordings manifest must be a list")
    else:
        for index, value in enumerate(recordings_value):
            schema_errors.extend(
                _schema_errors(
                    value,
                    schema,
                    reference="#/$defs/raw_recording",
                    label=f"recording[{index}]",
                )
            )
            if isinstance(value, Mapping):
                recordings.append(cast(Mapping[str, Any], value))
    annotations: list[Mapping[str, Any]] = []
    if not isinstance(annotations_value, list):
        schema_errors.append("annotation manifest must be a list")
    else:
        for index, value in enumerate(annotations_value):
            schema_errors.extend(
                _schema_errors(
                    value,
                    schema,
                    reference="#/$defs/annotation",
                    label=f"annotation[{index}]",
                )
            )
            if isinstance(value, Mapping):
                annotations.append(cast(Mapping[str, Any], value))
    lineage = plan.get("lineage")
    if isinstance(lineage, Mapping):
        for name, path in (("config", config_path), ("schema", schema_path)):
            reference = lineage.get(name)
            if not isinstance(reference, Mapping) or reference.get("sha256") != sha256_file(
                path.resolve()
            ):
                schema_errors.append(f"plan {name} lineage does not match supplied input")
    if schema_errors:
        blocks_value = plan.get("blocks")
        bouts_value = plan.get("bouts")
        planned_blocks = (
            {
                str(item.get("block_id"))
                for item in blocks_value
                if isinstance(item, Mapping) and isinstance(item.get("block_id"), str)
            }
            if isinstance(blocks_value, list)
            else set()
        )
        planned_bouts = (
            {
                str(item.get("bout_id"))
                for item in bouts_value
                if isinstance(item, Mapping) and isinstance(item.get("bout_id"), str)
            }
            if isinstance(bouts_value, list)
            else set()
        )
        report: dict[str, Any] = {
            "valid": False,
            "status": "collection_records_invalid",
            "errors": schema_errors,
            "equipment_evidence_verified": False,
            "study_prerequisites_verified": False,
            "remaining_prerequisites": plan.get("pending_prerequisites", []),
            "recording_count": len(recordings_value) if isinstance(recordings_value, list) else 0,
            "recorded_block_count": 0,
            "claimed_block_count": 0,
            "recording_status_counts": {},
            "annotation_count": (
                len(annotations_value) if isinstance(annotations_value, list) else 0
            ),
            "adjudicated_bout_count": 0,
            "metadata_eligible_bout_count": 0,
            "signal_eligibility_evaluated": False,
            "analysis_eligible_bout_count": 0,
            "annotation_status_counts": {},
            "planned_block_count": len(planned_blocks),
            "planned_bout_count": len(planned_bouts),
            "missing_recorded_block_ids": sorted(planned_blocks),
            "missing_annotation_bout_ids": sorted(planned_bouts),
            "metadata_ineligible_bout_ids": sorted(planned_bouts),
            "analysis_ready": False,
            "recorded_evidence_complete": False,
            "collection_complete": False,
            "collection_completed_claim": False,
            "model_fit_count": 0,
        }
    else:
        try:
            report = validate_collection_records(
                plan=plan,
                config=config,
                equipment=equipment,
                recordings=recordings,
                annotations=annotations,
                raw_root=raw_root,
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            report = {
                "valid": False,
                "status": "collection_records_invalid",
                "errors": [f"semantic validation raised: {exc}"],
                "collection_complete": False,
                "collection_completed_claim": False,
                "analysis_ready": False,
                "model_fit_count": 0,
            }
        report = dict(report)
    input_paths = {
        "plan": plan_path,
        "config": config_path,
        "schema": schema_path,
        "equipment": equipment_path,
        "recordings": recordings_path,
        "annotations": annotations_path,
    }
    input_manifest = {
        name: {
            "path": str(path.resolve()),
            "size_bytes": path.resolve().stat().st_size,
            "sha256": sha256_file(path.resolve()),
        }
        for name, path in input_paths.items()
    }
    validator_path = Path(validate_collection_records.__code__.co_filename).resolve()
    runner_path = Path(__file__).resolve()
    sealed = _sealed(
        {
            "record_kind": "physical_information_collection_validation",
            **report,
            "input_manifest": input_manifest,
            "validator_source": {
                "path": str(validator_path),
                "size_bytes": validator_path.stat().st_size,
                "sha256": sha256_file(validator_path),
            },
            "runner_source": {
                "path": str(runner_path),
                "size_bytes": runner_path.stat().st_size,
                "sha256": sha256_file(runner_path),
            },
            "analysis_or_model_run": False,
        }
    )
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_json_create_only(output_path, sealed, root=output_path.parent)
    return sealed


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--repository-root", type=Path, required=True)
    prepare.add_argument("--evidence-root", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--protocol", type=Path, required=True)
    prepare.add_argument("--schema", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--code-commit", required=True)
    validate = subparsers.add_parser("validate")
    validate.add_argument("--run-directory", type=Path, required=True)
    collection = subparsers.add_parser("check-collection")
    collection.add_argument("--plan", type=Path, required=True)
    collection.add_argument("--config", type=Path, required=True)
    collection.add_argument("--schema", type=Path, required=True)
    collection.add_argument("--equipment", type=Path, required=True)
    collection.add_argument("--recordings", type=Path, required=True)
    collection.add_argument("--annotations", type=Path, required=True)
    collection.add_argument("--raw-root", type=Path, required=True)
    collection.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "prepare":
        result = prepare_plan_run(
            repository_root=arguments.repository_root,
            evidence_root=arguments.evidence_root,
            config_path=arguments.config,
            protocol_path=arguments.protocol,
            schema_path=arguments.schema,
            output_directory=arguments.output,
            code_commit=str(arguments.code_commit),
        )
    elif arguments.command == "validate":
        result = validate_plan_run(arguments.run_directory)
    else:
        result = check_collection(
            plan_path=arguments.plan,
            config_path=arguments.config,
            schema_path=arguments.schema,
            equipment_path=arguments.equipment,
            recordings_path=arguments.recordings,
            annotations_path=arguments.annotations,
            raw_root=arguments.raw_root,
            output_path=arguments.output,
        )
    print(json.dumps(result, sort_keys=True))
    if arguments.command == "check-collection" and result.get("valid") is not True:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
