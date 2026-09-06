"""Lightweight create-only research provenance without importing model runtimes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

PUBLICATION_SOURCE_PROTOCOL_ID = "external-har-publication-v4"
PUBLICATION_PROTOCOL_PATH = "configs/protocols/external_har_publication_v4.yaml"
EVIDENCE_ROLE_LEDGER_PATH = "configs/datasets/evidence_roles_20260905_v3.yaml"
SUPERSESSION_LEDGER_PATH = (
    "results/research/cross_dataset_har_v4/context_training_population_supersession_20260905.json"
)
PARENT_SUPERSESSION_LEDGER_PATH = (
    "results/research/cross_dataset_har_v4/prewindow_provider_boundary_supersession_20260905.json"
)
SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH = (
    "results/research/cross_dataset_har_v4/"
    "sole_harmony_oracle_materialization_interruption_20260906.json"
)
RUNTIME_ENVIRONMENT_PROTOCOL_ID = "external-har-runtime-environment-v2"
PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID = "external-har-inherited-clean-launch-v1"
EXECUTED_RESEARCH_MODULE_PATH = "src/inclusive_shift_har/artifacts/research_provenance.py"
FOG_STAR_FULL_PARTICIPANT_ROSTER = tuple(
    f"fogstar:{participant:03d}" for participant in range(1, 23)
)
HAR_PMD_FULL_PARTICIPANT_ROSTER = tuple(
    f"harpmd:{participant:03d}" for participant in range(1, 121)
)
IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER = tuple(
    f"imuharil:P_{participant:02d}" for participant in range(1, 51)
)
IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER = tuple(
    f"imuharil:P_{participant:02d}"
    for participant in (3, 5, 8, 10, 11, 12, 17, 20, 21, 23, 24, 25, 26, 31, 32, 33, 42, 47, 48)
)
IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER = tuple(
    participant
    for participant in IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER
    if participant not in {"imuharil:P_18", "imuharil:P_19", "imuharil:P_46"}
)
IMU_HAR_IL_INVENTORY_PROTOCOL_ID = "imu-har-il-fixed-inventory-v1"

_IMU_HAR_IL_INCOMPLETE_INVENTORY_PARTICIPANTS = tuple(
    f"imuharil:P_{participant:02d}"
    for participant in (
        1,
        2,
        4,
        6,
        7,
        9,
        13,
        14,
        15,
        16,
        27,
        28,
        29,
        30,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        43,
        44,
        45,
        46,
        49,
        50,
    )
)
_IMU_HAR_IL_MISSING_TRIAL_INVENTORY_SHA256 = (
    "651857292b0e4a66a52a7d124016c8871c1a96aa010c71d05d476c16d0f50719"
)
_IMU_HAR_IL_COHORT_CONTRACTS: dict[str, dict[str, Any]] = {
    "complete_requested_core": {
        "protocol_id": "imu-har-il-complete-requested-core-v1",
        "participant_roster": list(IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER),
        "participant_count": 19,
        "trial_count": 228,
        "inventory_selected_source_file_count": 264,
        "source_inventory_sha256": (
            "060748891b6869712c7323ab4b0a754b2d57703b6cccb2ef574434676e70c7cc"
        ),
        "source_payload_inventory_sha256": (
            "cba6f7bb1dcaea8fc60a319d73e9123aeb478f9f09752c462a6bd94086b4d78c"
        ),
        "data_quality_affected_participants": [
            "imuharil:P_18",
            "imuharil:P_19",
            "imuharil:P_22",
        ],
        "quarantined_trial_count": 30,
        "participant_exclusion_record_count": 31,
        "participants_without_retained_windows": sorted(
            set(IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER)
            - set(IMU_HAR_IL_COMPLETE_CORE_PARTICIPANT_ROSTER)
        ),
    },
    "available_valid_trials": {
        "protocol_id": "imu-har-il-available-trials-v1",
        "participant_roster": list(IMU_HAR_IL_AVAILABLE_VALID_PARTICIPANT_ROSTER),
        "participant_count": 47,
        "trial_count": 457,
        "inventory_selected_source_file_count": 490,
        "source_inventory_sha256": (
            "b7aa481380224e756e39ad28156d94961d49d659d106bf69f8cac358b0271385"
        ),
        "source_payload_inventory_sha256": (
            "ca81875cd3bcab424aa3821b7723c8b6f42d170ea8931a507dad47a16e1b1c25"
        ),
        "data_quality_affected_participants": [
            "imuharil:P_04",
            "imuharil:P_18",
            "imuharil:P_19",
            "imuharil:P_22",
        ],
        "quarantined_trial_count": 33,
        "participant_exclusion_record_count": 32,
        "participants_without_retained_windows": [
            "imuharil:P_18",
            "imuharil:P_19",
            "imuharil:P_46",
        ],
    },
}

_EXPLICIT_SOURCE_INPUTS = {
    "AGENTS.md",
    "docs/LOCKED_PROTOCOL.md",
    PUBLICATION_PROTOCOL_PATH,
    EVIDENCE_ROLE_LEDGER_PATH,
    "pyproject.toml",
    "uv.lock",
    "requirements/external-har-research.in",
    "requirements/external-har-research.lock",
    SUPERSESSION_LEDGER_PATH,
    PARENT_SUPERSESSION_LEDGER_PATH,
    SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
}


def _is_publication_source_input_path(relative: str) -> bool:
    """Return whether one repository-relative path belongs to the frozen source scope."""

    path = PurePosixPath(relative)
    return (
        relative in _EXPLICIT_SOURCE_INPUTS
        or (path.parts[:1] == ("src",) and path.suffix == ".py")
        or (path.parts[:1] == ("configs",) and path.suffix in {".yaml", ".json"})
        or (path.parts[:2] == ("results", "protocol") and path.suffix == ".json")
        or (
            len(path.parts) == 3 and path.parts[:2] == ("docs", "research") and path.suffix == ".md"
        )
    )


def _write_json_create_only(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _typed_path_locator(path: Path, repository_root: Path) -> dict[str, str]:
    """Describe one existing path without obscuring whether it is portable."""

    resolved = path.resolve(strict=True)
    repository = repository_root.resolve(strict=True)
    try:
        relative = resolved.relative_to(repository).as_posix()
    except ValueError:
        return {"path_kind": "external_absolute", "path": str(resolved)}
    return {"path_kind": "repository_relative", "path": relative or "."}


def _write_self_hashed_json_create_only(
    path: Path,
    value: dict[str, Any],
    *,
    hash_field: str = "record_sha256",
) -> dict[str, Any]:
    """Seal and create one JSON artifact, returning its physical-file binding."""

    if hash_field in value:
        raise ValueError(f"record already contains reserved hash field: {hash_field}")
    sealed = dict(value)
    sealed[hash_field] = canonical_json_sha256(sealed)
    _write_json_create_only(path, sealed)
    return {
        "path": path.name,
        "sha256": sha256_file(path),
        hash_field: sealed[hash_field],
    }


def _assert_executed_repository_root(repository_root: Path) -> dict[str, str]:
    """Fail unless this module is the exact governed file below ``repository_root``."""

    try:
        repository = repository_root.resolve(strict=True)
        expected_path = repository.joinpath(*PurePosixPath(EXECUTED_RESEARCH_MODULE_PATH).parts)
        actual = Path(__file__).resolve(strict=True)
        expected = expected_path.resolve(strict=True)
    except OSError as error:
        raise ValueError(
            "declared repository root does not contain the executed research module"
        ) from error
    if (
        expected_path.absolute() != expected
        or expected_path.is_symlink()
        or not expected_path.is_file()
        or actual != expected
    ):
        raise ValueError(
            "executed research package does not belong to the declared repository root"
        )
    return {
        "repository_relative_path": EXECUTED_RESEARCH_MODULE_PATH,
        "resolved_path_at_runtime": str(actual),
        "sha256": sha256_file(actual),
    }


def _runtime_environment() -> dict[str, Any]:
    """Capture the interpreter, executed source and installed distribution set."""

    packages = {
        str(distribution.metadata["Name"]).lower().replace("_", "-"): distribution.version
        for distribution in importlib.metadata.distributions()
        if "Name" in distribution.metadata
    }
    executable = Path(sys.executable).resolve(strict=True)
    source = Path(__file__).resolve(strict=True)
    return {
        "protocol_id": RUNTIME_ENVIRONMENT_PROTOCOL_ID,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "python_executable": {
            "resolved_path_at_runtime": str(executable),
            "sha256": sha256_file(executable),
        },
        "executed_research_module": {
            "repository_relative_path": EXECUTED_RESEARCH_MODULE_PATH,
            "resolved_path_at_runtime": str(source),
            "sha256": sha256_file(source),
        },
        "packages": dict(sorted(packages.items())),
    }


def _publication_artifact_contract(
    source_input_manifest: dict[str, Any], data_audit_artifact: dict[str, Any]
) -> dict[str, Any]:
    """Bind one result or failure to the frozen v4 provenance chain."""

    return {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "data_audit_artifact": data_audit_artifact,
        "publication_protocol": source_input_manifest["publication_protocol"],
        "evidence_role_ledger": source_input_manifest["evidence_role_ledger"],
        "supersession_ledger": source_input_manifest["supersession_ledger"],
    }


def _har_pmd_full_cohort_observed(summary: dict[str, Any]) -> bool:
    """Require all 120 planned HAR-PMD people to contribute retained windows."""

    plan = summary.get("participant_partition_plan")
    roster = plan.get("participant_roster") if isinstance(plan, dict) else None
    observation = summary.get("participant_partition_observation")
    return bool(
        roster == list(HAR_PMD_FULL_PARTICIPANT_ROSTER)
        and summary.get("participant_count") == 120
        and isinstance(observation, dict)
        and observation.get("planned_participant_count") == 120
        and observation.get("observed_window_participant_count") == 120
        and observation.get("participants_without_retained_windows") == []
    )


def _fog_star_full_cohort_observed(summary: dict[str, Any]) -> bool:
    """Require the exact 22-person roster in the pinned FoG-STAR v3 bytes."""

    plan = summary.get("participant_partition_plan")
    roster = plan.get("participant_roster") if isinstance(plan, dict) else None
    observation = summary.get("participant_partition_observation")
    return bool(
        roster == list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
        and summary.get("participant_count") == 22
        and isinstance(observation, dict)
        and observation.get("planned_participant_count") == 22
        and observation.get("observed_window_participant_count") == 22
        and observation.get("participants_without_retained_windows") == []
    )


def _annotation_independent_boundary_observed(summary: dict[str, Any]) -> bool:
    """Recognize only the frozen label-independent repository boundary contract."""

    boundary = summary.get("boundary_provenance")
    return bool(
        isinstance(boundary, dict)
        and boundary.get("protocol_id") == "external-har-boundary-provenance-v1"
        and boundary.get("repository_signal_grid_annotation_independent") is True
        and boundary.get("provider_upstream_annotation_conditioned") is False
    )


def _imu_har_il_expected_cohort_observed(
    summary: dict[str, Any], *, selection_policy: str | None = None
) -> bool:
    """Require the hash-pinned IMU provider inventory and lane-specific cohort.

    Both lanes start from the same 50-person provider roster.  The 19-person
    complete-core lane and 47-person available-trial lane are missing-data views
    of those people; repetitions remain observations, never independent people.
    """

    plan = summary.get("participant_partition_plan")
    observation = summary.get("participant_partition_observation")
    cohort = summary.get("cohort_audit")
    if (
        not isinstance(plan, dict)
        or not isinstance(observation, dict)
        or not isinstance(cohort, dict)
    ):
        return False
    observed_policy = cohort.get("selection_policy")
    if selection_policy is not None and observed_policy != selection_policy:
        return False
    if not isinstance(observed_policy, str):
        return False
    expected = _IMU_HAR_IL_COHORT_CONTRACTS.get(observed_policy)
    if expected is None:
        return False
    retained_roster = expected["participant_roster"]
    absent = expected["participants_without_retained_windows"]
    quality_people = expected["data_quality_affected_participants"]
    expected_cohort_fields = {
        "protocol_id": expected["protocol_id"],
        "inventory_protocol_id": IMU_HAR_IL_INVENTORY_PROTOCOL_ID,
        "selection_policy": observed_policy,
        "provider_reported_participant_count": 50,
        "requested_repetition_limit": 4,
        "requested_repetitions": [
            "Repetition_1",
            "Repetition_2",
            "Repetition_3",
            "Repetition_4",
        ],
        "requested_activities": ["Walk", "Sit", "Stand"],
        "requested_participant_count": 50,
        "requested_participants": list(IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER),
        "requested_trial_count": 600,
        "inventory_selected_source_file_count": expected["inventory_selected_source_file_count"],
        "source_inventory_sha256": expected["source_inventory_sha256"],
        "source_payload_inventory_sha256": expected["source_payload_inventory_sha256"],
        "inventory_incomplete_participant_count": len(
            _IMU_HAR_IL_INCOMPLETE_INVENTORY_PARTICIPANTS
        ),
        "inventory_incomplete_participants": list(_IMU_HAR_IL_INCOMPLETE_INVENTORY_PARTICIPANTS),
        "missing_requested_trial_count": 110,
        "missing_requested_trial_folders_sha256": (_IMU_HAR_IL_MISSING_TRIAL_INVENTORY_SHA256),
        "data_quality_affected_participant_count": len(quality_people),
        "data_quality_affected_participants": quality_people,
        "quarantined_trial_count": expected["quarantined_trial_count"],
        "participant_exclusion_record_count": expected["participant_exclusion_record_count"],
        "retained_source_file_count": expected["trial_count"],
        "retained_participant_count": expected["participant_count"],
        "retained_participants": retained_roster,
        "retained_trial_count": expected["trial_count"],
        "participants_with_no_retained_windows": absent,
        "independent_unit": "participant; repetitions do not increase independent N",
        "complete_case_or_repetition1_before_after_comparison_allowed": False,
    }
    return bool(
        summary.get("dataset_id") == "imu_har_il_v1"
        and plan.get("participant_roster") == list(IMU_HAR_IL_PROVIDER_PARTICIPANT_ROSTER)
        and summary.get("participant_count") == expected["participant_count"]
        and summary.get("trial_count") == expected["trial_count"]
        and summary.get("receipt_count") == expected["inventory_selected_source_file_count"]
        and observation.get("planned_participant_count") == 50
        and observation.get("observed_window_participant_count") == expected["participant_count"]
        and observation.get("participants_without_retained_windows") == absent
        and all(cohort.get(key) == value for key, value in expected_cohort_fields.items())
    )


def _external_evidence_status(*summaries: dict[str, Any]) -> str:
    """Return the fail-closed scientific role carried by an external result."""

    if not summaries or any(not isinstance(summary, dict) for summary in summaries):
        return "provisional_unregistered_dataset"
    dataset_ids = [summary.get("dataset_id") for summary in summaries]
    registered = {"fog_star_v3", "imu_har_il_v1", "har_pmd_v1", "sole_harmony_v1"}
    if any(
        not isinstance(dataset_id, str) or not dataset_id or dataset_id not in registered
        for dataset_id in dataset_ids
    ):
        return "provisional_unregistered_dataset"

    if len(summaries) == 2:
        if dataset_ids != ["imu_har_il_v1", "fog_star_v3"]:
            return "provisional_unregistered_transfer"
        target = summaries[1]
        if not _annotation_independent_boundary_observed(target):
            return "provisional_boundary_contract_invalid_source_transfer"
        if not _fog_star_full_cohort_observed(target):
            return "provisional_subset_source_transfer"
        return "diagnostic_provider_presegmented_source_transfer"
    if len(summaries) != 1:
        return "provisional_unregistered_dataset_combination"

    summary = summaries[0]
    dataset_id = dataset_ids[0]
    if dataset_id == "imu_har_il_v1":
        # The immutable provider representation is label-presegmented even if a
        # local record omits or contradicts its boundary metadata.
        return "diagnostic_provider_presegmented_development"
    if dataset_id == "sole_harmony_v1":
        boundary = summary.get("boundary_provenance")
        if not isinstance(boundary, dict):
            return "provisional_sole_boundary_contract_missing"
        mode = boundary.get("boundary_mode")
        common_valid = bool(
            boundary.get("protocol_id") == "external-har-boundary-provenance-v1"
            and boundary.get("provider_upstream_annotation_conditioned") is False
        )
        if (
            mode == "camera_bout_oracle"
            and common_valid
            and boundary.get("repository_signal_grid_annotation_independent") is False
        ):
            return "diagnostic_oracle_boundary"
        if mode == "session_observable" and _annotation_independent_boundary_observed(summary):
            return "diagnostic_session_observable_temporal_development"
        return "provisional_sole_boundary_contract_invalid"
    if dataset_id == "har_pmd_v1":
        if not _annotation_independent_boundary_observed(summary):
            return "provisional_boundary_contract_invalid_stress"
        return (
            "validated_stress_test"
            if _har_pmd_full_cohort_observed(summary)
            else "provisional_pilot_stress"
        )
    if not _annotation_independent_boundary_observed(summary):
        return "provisional_boundary_contract_invalid_development"
    return (
        "validated_development"
        if _fog_star_full_cohort_observed(summary)
        else "provisional_subset_development"
    )


def _git_state(repository_root: Path) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status_output = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all", "-z"],
        cwd=repository_root,
        check=True,
        capture_output=True,
    ).stdout
    status = [os.fsdecode(entry) for entry in status_output.split(b"\0") if entry]
    return {"commit": commit, "worktree_dirty": bool(status), "status_entries": status}


def _stable_source_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    """Remove only capture-time fields before comparing two source manifests."""

    return {
        name: value
        for name, value in manifest.items()
        if name not in {"captured_at", "record_sha256"}
    }


def _launch_context_self_hash_valid(context: dict[str, Any]) -> bool:
    declared = context.get("record_sha256")
    unhashed = dict(context)
    unhashed.pop("record_sha256", None)
    return isinstance(declared, str) and canonical_json_sha256(unhashed) == declared


def _resolved_output_root(
    repository_root: Path,
    output_root: Path,
    manifest: dict[str, Any],
) -> tuple[Path, PurePosixPath | None]:
    """Resolve an output root and reject roots that can contain governed inputs."""

    repository = repository_root.resolve()
    output = output_root.resolve()
    try:
        relative = output.relative_to(repository)
    except ValueError:
        return output, None
    if not relative.parts:
        raise ValueError("publication output root cannot be the repository root")
    relative_posix = PurePosixPath(relative.as_posix())
    if relative_posix.parts[:1] in {("src",), ("configs",)} or relative_posix.parts[:2] in {
        ("docs", "research"),
        ("results", "protocol"),
    }:
        raise ValueError("publication output root is inside a governed source namespace")
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError("publication launch manifest files are absent")
    for governed_name in files:
        governed = PurePosixPath(str(governed_name))
        if governed == relative_posix or relative_posix in governed.parents:
            raise ValueError("publication output root may not contain governed source inputs")
    return output, relative_posix


def _status_entries_within_output_root(
    status_entries: Any,
    allowed_relative_root: PurePosixPath | None,
) -> bool:
    """Allow only untracked paths located below the inherited create-only root."""

    if not isinstance(status_entries, list):
        return False
    if not status_entries:
        return True
    if allowed_relative_root is None:
        return False
    for entry in status_entries:
        if not isinstance(entry, str) or not entry.startswith("?? "):
            return False
        raw_path = entry[3:]
        candidate = PurePosixPath(raw_path)
        if (
            candidate.is_absolute()
            or ".." in candidate.parts
            or "\\" in raw_path
            or ":" in raw_path
            or not (
                candidate == allowed_relative_root or allowed_relative_root in candidate.parents
            )
        ):
            return False
    return True


def _resolve_publication_launch_context(
    *,
    repository_root: Path,
    output_directory: Path,
    current_git_state: dict[str, Any],
    current_source_manifest: dict[str, Any],
    manifest_commit_validator: Callable[[Path, Any, dict[str, Any]], list[str]],
    inherited_launch_context: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Capture or verify one clean launch shared by nested create-only writers.

    A parent may create plan and child artifacts below its declared output root without
    causing a false dirty-tree failure.  HEAD, every governed source hash, and all dirt
    outside that root remain fail-closed.
    """

    repository = repository_root.resolve()
    output = output_directory.resolve()
    if inherited_launch_context is None:
        if output.exists():
            raise FileExistsError(output)
        if (
            current_git_state.get("worktree_dirty") is not False
            or current_git_state.get("status_entries") != []
        ):
            raise ValueError("publication evidence requires an initially clean worktree")
        commit_errors = manifest_commit_validator(
            repository,
            current_git_state.get("commit"),
            current_source_manifest,
        )
        if commit_errors:
            raise ValueError(
                "publication launch source does not match its commit: " + "; ".join(commit_errors)
            )
        allowed_root, _relative = _resolved_output_root(repository, output, current_source_manifest)
        context: dict[str, Any] = {
            "schema_version": "1.0.0",
            "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
            "allowed_create_only_output_root": {
                "kind": "external_absolute" if _relative is None else "repository_relative",
                "path": str(allowed_root) if _relative is None else _relative.as_posix(),
            },
            "git_at_launch": current_git_state,
            "source_input_manifest": current_source_manifest,
        }
        context["record_sha256"] = canonical_json_sha256(context)
        return current_git_state, current_source_manifest, context

    context = inherited_launch_context
    if context.get(
        "protocol_id"
    ) != PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID or not _launch_context_self_hash_valid(context):
        raise ValueError("inherited publication launch context is invalid")
    launch = context.get("git_at_launch")
    manifest = context.get("source_input_manifest")
    declared_output = context.get("allowed_create_only_output_root")
    if (
        not isinstance(launch, dict)
        or not isinstance(manifest, dict)
        or not isinstance(declared_output, dict)
        or declared_output.get("kind") not in {"repository_relative", "external_absolute"}
        or not isinstance(declared_output.get("path"), str)
    ):
        raise ValueError("inherited publication launch context is incomplete")
    declared_path = str(declared_output["path"])
    if declared_output["kind"] == "repository_relative":
        relative_path = PurePosixPath(declared_path)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in declared_path
            or ":" in declared_path
        ):
            raise ValueError("inherited publication output root is unsafe")
        resolved_declared = repository.joinpath(*relative_path.parts)
    else:
        resolved_declared = Path(declared_path)
        if not resolved_declared.is_absolute():
            raise ValueError("inherited external publication output root is not absolute")
    allowed_root, allowed_relative = _resolved_output_root(repository, resolved_declared, manifest)
    if (
        declared_output["kind"] == "repository_relative"
        and (allowed_relative is None or allowed_relative.as_posix() != declared_path)
    ) or (
        declared_output["kind"] == "external_absolute"
        and (allowed_relative is not None or str(allowed_root) != declared_path)
    ):
        raise ValueError("inherited publication output root is not canonical")
    try:
        output.relative_to(allowed_root)
    except ValueError as error:
        raise ValueError("child output is outside the inherited create-only root") from error
    if launch.get("worktree_dirty") is not False or launch.get("status_entries") != []:
        raise ValueError("inherited publication launch was not clean")
    commit_errors = manifest_commit_validator(repository, launch.get("commit"), manifest)
    if commit_errors:
        raise ValueError(
            "inherited publication source does not match its commit: " + "; ".join(commit_errors)
        )
    if current_git_state.get("commit") != launch.get("commit"):
        raise ValueError("repository HEAD changed after publication launch")
    current_entries = current_git_state.get("status_entries")
    if bool(current_entries) != bool(current_git_state.get("worktree_dirty")):
        raise ValueError("current Git dirt flag and status entries disagree")
    if not _status_entries_within_output_root(current_entries, allowed_relative):
        raise ValueError("repository dirt exists outside the inherited create-only output root")
    if _stable_source_manifest(current_source_manifest) != _stable_source_manifest(manifest):
        raise ValueError("governed source inputs changed after publication launch")
    return launch, manifest, context


def _publication_launch_context_binding(context: dict[str, Any]) -> dict[str, Any]:
    """Return the compact, retainable binding for an inherited launch context."""

    if not _launch_context_self_hash_valid(context):
        raise ValueError("publication launch context self-hash differs")
    return {
        "protocol_id": context["protocol_id"],
        "allowed_create_only_output_root": context["allowed_create_only_output_root"],
        "record_sha256": context["record_sha256"],
    }


def _write_launch_failure_envelope(
    *,
    repository_root: Path,
    output_directory: Path,
    launch_context: dict[str, Any],
    started_at: str,
    stage: str,
    exception: Exception,
    traceback_text: str,
) -> dict[str, Any]:
    """Preserve a pre-writer config/acquisition failure under the declared output root."""

    launch, manifest, verified_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output_directory,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=launch_context,
    )
    output_directory.mkdir(parents=True, exist_ok=False)
    artifact = _write_self_hashed_json_create_only(
        output_directory / "failure.json",
        {
            "schema_version": "1.0.0",
            "status": "FAILED_PRESERVED",
            "failure_scope": "pre_writer_configuration_or_dataset_acquisition",
            "stage": stage,
            "started_at": started_at,
            "failed_at": datetime.now(UTC).isoformat(),
            "exception_type": type(exception).__name__,
            "exception_message": str(exception),
            "traceback": traceback_text,
            "git_at_launch": launch,
            "source_input_manifest": manifest,
            "publication_launch_context": _publication_launch_context_binding(verified_context),
            "environment": _runtime_environment(),
        },
        hash_field="failure_payload_sha256_before_serialization",
    )
    # Import lazily to keep this provenance module lightweight and avoid its
    # validator's import-time dependency cycle. A failed launch is still an
    # evidence package and must carry an independently reconstructed validation.
    from inclusive_shift_har.experiments.external_evidence_validate import (
        validate_and_record_run_directory,
    )

    validate_and_record_run_directory(output_directory, repository_root)
    return artifact


def _source_manifest_commit_errors(
    repository_root: Path,
    commit: Any,
    manifest: dict[str, Any],
) -> list[str]:
    """Verify that the launch manifest is the exact governed source tree at ``commit``."""

    files = manifest.get("files")
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        return ["launch commit is not a full lowercase Git object ID"]
    if not isinstance(files, dict) or not files:
        return ["launch source manifest files are absent"]
    typed_files = {str(name): str(digest) for name, digest in files.items()}
    try:
        tree = subprocess.run(
            ["git", "ls-tree", "-r", "--name-only", commit],
            cwd=repository_root,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as error:
        return [f"launch commit source tree is unavailable: {type(error).__name__}"]
    if tree.returncode:
        return ["launch commit source tree is unavailable"]
    expected = {
        name for name in tree.stdout.splitlines() if _is_publication_source_input_path(name)
    }
    errors: list[str] = []
    if set(typed_files) != expected:
        errors.append("launch manifest is not the exact governed commit source scope")
    names = sorted(set(typed_files) & expected)
    try:
        response = subprocess.run(
            ["git", "cat-file", "--batch"],
            cwd=repository_root,
            input="".join(f"{commit}:{name}\n" for name in names).encode("utf-8"),
            capture_output=True,
            check=False,
        )
    except OSError as error:
        return [*errors, f"launch source blobs are unavailable: {type(error).__name__}"]
    if response.returncode:
        return [*errors, "launch source blobs are unavailable"]
    offset = 0
    for name in names:
        end = response.stdout.find(b"\n", offset)
        if end < 0:
            errors.append(f"launch source blob header is absent: {name}")
            break
        header = response.stdout[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            errors.append(f"launch source blob is unavailable: {name}")
            offset = end + 1
            continue
        size = int(header[2])
        blob = response.stdout[end + 1 : end + 1 + size]
        if len(blob) != size or hashlib.sha256(blob).hexdigest() != typed_files[name]:
            errors.append(f"launch source blob differs from manifest: {name}")
        offset = end + 1 + size + 1
    return errors


def _source_input_manifest(repository_root: Path) -> dict[str, Any]:
    """Hash the executed package and the unchanged declared source-input scope."""
    _assert_executed_repository_root(repository_root)
    candidates = list((repository_root / "src").rglob("*.py"))
    candidates.extend((repository_root / "configs").rglob("*.yaml"))
    candidates.extend((repository_root / "configs").rglob("*.json"))
    candidates.extend((repository_root / "docs/research").glob("*.md"))
    candidates.extend((repository_root / "results/protocol").rglob("*.json"))
    candidates.extend(
        repository_root / relative
        for relative in (
            "AGENTS.md",
            "docs/LOCKED_PROTOCOL.md",
            "configs/datasets/external_har_portfolio_v1.yaml",
            "configs/experiments/cross_dataset_har_rnd_v1.yaml",
            "docs/research/CROSS_DATASET_HAR_RND_V1_PROTOCOL.md",
            "docs/research/EXTERNAL_HAR_PHYSICAL_GRID_V2_CORRECTION.md",
            "configs/protocols/external_har_physical_grid_v2.yaml",
            PUBLICATION_PROTOCOL_PATH,
            "configs/datasets/evidence_roles_20260905_v2.yaml",
            EVIDENCE_ROLE_LEDGER_PATH,
            SUPERSESSION_LEDGER_PATH,
            PARENT_SUPERSESSION_LEDGER_PATH,
            SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
            "pyproject.toml",
            "uv.lock",
            "requirements/external-har-research.in",
            "requirements/external-har-research.lock",
        )
    )
    candidates = [
        path
        for path in candidates
        if _is_publication_source_input_path(path.relative_to(repository_root).as_posix())
    ]
    files = {
        path.relative_to(repository_root).as_posix(): sha256_file(path)
        for path in sorted(set(candidates))
        if path.is_file()
    }
    missing_required = sorted(_EXPLICIT_SOURCE_INPUTS - set(files))
    if missing_required:
        raise FileNotFoundError(
            "required publication source inputs are absent: " + ", ".join(missing_required)
        )
    supersession_path = repository_root / SUPERSESSION_LEDGER_PATH
    supersession = load_json_strict(supersession_path)
    if not isinstance(supersession, dict):
        raise ValueError("external-HAR supersession ledger is not an object")
    declared_supersession_hash = supersession.get("record_sha256")
    unhashed_supersession = dict(supersession)
    unhashed_supersession.pop("record_sha256", None)
    if (
        not isinstance(declared_supersession_hash, str)
        or canonical_json_sha256(unhashed_supersession) != declared_supersession_hash
    ):
        raise ValueError("external-HAR supersession ledger self-hash differs")
    parent_path = repository_root / PARENT_SUPERSESSION_LEDGER_PATH
    parent = load_json_strict(parent_path)
    if not isinstance(parent, dict):
        raise ValueError("external-HAR parent supersession ledger is not an object")
    parent_hash = parent.get("record_sha256")
    unhashed_parent = dict(parent)
    unhashed_parent.pop("record_sha256", None)
    if (
        not isinstance(parent_hash, str)
        or canonical_json_sha256(unhashed_parent) != parent_hash
        or supersession.get("parent_ledger")
        != {
            "path": PARENT_SUPERSESSION_LEDGER_PATH,
            "file_sha256": files[PARENT_SUPERSESSION_LEDGER_PATH],
        }
    ):
        raise ValueError("external-HAR supersession parent chain differs")
    sole_receipt_path = repository_root / SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH
    sole_receipt = load_json_strict(sole_receipt_path)
    if not isinstance(sole_receipt, dict):
        raise ValueError("Sole-HARmony interruption receipt is not an object")
    sole_receipt_hash = sole_receipt.get("record_sha256")
    unhashed_sole_receipt = dict(sole_receipt)
    unhashed_sole_receipt.pop("record_sha256", None)
    sole_claim_contract = sole_receipt.get("claim_contract")
    sole_terminal = sole_receipt.get("terminal_artifact_observation")
    if (
        not isinstance(sole_receipt_hash, str)
        or canonical_json_sha256(unhashed_sole_receipt) != sole_receipt_hash
        or sole_receipt.get("record_kind")
        != "sole_harmony_interrupted_oracle_materialization_receipt"
        or sole_receipt.get("status") != "INTERRUPTED_AFTER_DATA_AUDIT_NO_MODEL_RESULT"
        or not isinstance(sole_claim_contract, dict)
        or sole_claim_contract.get("deployable_claim_allowed") is not False
        or sole_claim_contract.get("confirmation_allowed") is not False
        or sole_claim_contract.get("numerical_before_after_comparison_allowed") is not False
        or not isinstance(sole_terminal, dict)
        or sole_terminal.get("score_available") is not False
    ):
        raise ValueError("Sole-HARmony interruption receipt or no-score contract differs")
    publication_protocol = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "path": PUBLICATION_PROTOCOL_PATH,
        "sha256": files[PUBLICATION_PROTOCOL_PATH],
    }
    evidence_role_ledger = {
        "ledger_id": "external-evidence-roles-20260905-v3",
        "path": EVIDENCE_ROLE_LEDGER_PATH,
        "sha256": files[EVIDENCE_ROLE_LEDGER_PATH],
    }
    supersession_ledger = {
        "record_kind": "external_har_create_only_supersession_ledger",
        "path": SUPERSESSION_LEDGER_PATH,
        "sha256": files[SUPERSESSION_LEDGER_PATH],
        "record_sha256": declared_supersession_hash,
    }
    sole_harmony_interruption_receipt = {
        "record_kind": sole_receipt["record_kind"],
        "path": SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
        "sha256": files[SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH],
        "record_sha256": sole_receipt_hash,
        "status": sole_receipt["status"],
    }
    manifest = {
        "schema_version": "2.0.0",
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "file_count": len(files),
        "files": files,
        "manifest_sha256": canonical_json_sha256(files),
        "publication_protocol": publication_protocol,
        "evidence_role_ledger": evidence_role_ledger,
        "supersession_ledger": supersession_ledger,
        "sole_harmony_interruption_receipt": sole_harmony_interruption_receipt,
        "captured_at": datetime.now(UTC).isoformat(),
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    return manifest
