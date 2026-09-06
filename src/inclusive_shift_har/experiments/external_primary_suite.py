"""Run the full IMU-HAR-IL development suite and zero-shot FoG transfer once."""

from __future__ import annotations

import argparse
import gc
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

from inclusive_shift_har.artifacts.research_provenance import (
    IMU_HAR_IL_INVENTORY_PROTOCOL_ID,
    PUBLICATION_SOURCE_PROTOCOL_ID,
    _imu_har_il_expected_cohort_observed,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_manifest_commit_errors,
    _write_self_hashed_json_create_only,
)
from inclusive_shift_har.data.external_har import load_fog_star, load_imu_har_il
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _read_mapping,
    _source_input_manifest,
    _write_json_create_only,
    run_and_write,
)
from inclusive_shift_har.experiments.cross_dataset_neural import run_and_write_neural
from inclusive_shift_har.experiments.cross_dataset_transfer import run_and_write_transfer
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_SCIENTIFIC_RUN_ROLE = "scientific_run"
_ORCHESTRATION_INDEX_ROLE = "non_run_create_only_orchestration_index"
_PRIMARY_SUITE_COMPLETE_STATUS = "DIAGNOSTIC_PROVIDER_PRESEGMENTED_SUITE_COMPLETE"


def _validated_sibling_suite_plan(path: Path, reference: Any) -> dict[str, Any] | None:
    """Return the exact self-hashed sibling plan referenced by a suite terminal."""

    if not isinstance(reference, dict) or reference.get("path") != "suite_plan.json":
        return None
    plan_path = path.parent / "suite_plan.json"
    if not plan_path.is_file() or plan_path.is_symlink():
        return None
    try:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        physical_sha256 = sha256_file(plan_path)
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(plan, dict):
        return None
    unhashed = dict(plan)
    declared_self_hash = unhashed.pop("record_sha256", None)
    expected = {
        "path": "suite_plan.json",
        "sha256": physical_sha256,
        "record_sha256": declared_self_hash,
    }
    if not (
        isinstance(declared_self_hash, str)
        and declared_self_hash == canonical_json_sha256(unhashed)
        and reference == expected
    ):
        return None
    return plan


def _sibling_plan_binding_valid(path: Path, reference: Any) -> bool:
    """Verify a terminal's immutable binding to its sibling suite plan."""

    return _validated_sibling_suite_plan(path, reference) is not None


def _planned_children_from_suite_plan(
    plan: dict[str, Any] | None,
) -> tuple[tuple[str, str], ...] | None:
    """Parse one direct-child topology only when it is unambiguous and safe."""

    if (
        not isinstance(plan, dict)
        or plan.get("schema_version") != "1.0.0"
        or plan.get("protocol_id") != PUBLICATION_SOURCE_PROTOCOL_ID
        or plan.get("artifact_role") != _ORCHESTRATION_INDEX_ROLE
        or plan.get("root_run_validation_applicable") is not False
    ):
        return None
    declared = plan.get("planned_children")
    if not isinstance(declared, list) or not declared:
        return None
    planned: list[tuple[str, str]] = []
    names: set[str] = set()
    for item in declared:
        if not isinstance(item, dict) or set(item) != {"relative_directory", "artifact_role"}:
            return None
        name = item.get("relative_directory")
        role = item.get("artifact_role")
        if not isinstance(name, str) or not name:
            return None
        parsed = PurePosixPath(name)
        if (
            parsed.is_absolute()
            or len(parsed.parts) != 1
            or parsed.name != name
            or name in {".", ".."}
            or "\\" in name
            or name in names
            or role not in {_SCIENTIFIC_RUN_ROLE, _ORCHESTRATION_INDEX_ROLE}
        ):
            return None
        names.add(name)
        planned.append((name, str(role)))
    return tuple(planned)


def _bindings_match_plan(
    bindings: list[dict[str, Any]], planned_children: tuple[tuple[str, str], ...]
) -> bool:
    """Return whether bindings exactly and completely cover a declared topology."""

    expected = sorted(planned_children)
    observed = sorted(
        (str(binding.get("relative_directory")), str(binding.get("artifact_role")))
        for binding in bindings
    )
    return observed == expected and all(
        binding.get("binding_complete") is True for binding in bindings
    )


def _json_artifact_binding(output_root: Path, path: Path) -> dict[str, Any]:
    """Bind one JSON artifact without copying machine-specific absolute paths."""

    regular_non_symlink = path.is_file() and not path.is_symlink()
    binding: dict[str, Any] = {
        "artifact_name": path.name,
        "path": path.relative_to(output_root).as_posix(),
        "sha256": sha256_file(path) if regular_non_symlink else None,
        "regular_non_symlink": regular_non_symlink,
    }
    if not binding["regular_non_symlink"]:
        binding.update({"json_object_valid": False, "status": "NON_REGULAR_OR_SYMLINK"})
        return binding
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        binding.update(
            {
                "json_object_valid": False,
                "status": "UNREADABLE_JSON",
                "parse_error_type": type(error).__name__,
            }
        )
        return binding
    if not isinstance(value, dict):
        binding.update({"json_object_valid": False, "status": "NON_OBJECT_JSON"})
        return binding
    status = value.get("status")
    if status is None:
        status = value.get("artifact_evidence_status", value.get("evidence_status"))
    binding.update({"json_object_valid": True, "status": status})
    if path.name == "validation.json":
        binding.update(
            {
                "integrity_passed": value.get("integrity_passed"),
                "publication_evidence_ready": value.get("publication_evidence_ready"),
                "publication_evidence_ready_for_unqualified_methods": value.get(
                    "publication_evidence_ready_for_unqualified_methods"
                ),
                "diagnostic_contract_passed": value.get("diagnostic_contract_passed"),
            }
        )
    if path.name in {"suite_complete.json", "suite_failure.json"}:
        child_bindings = value.get("child_artifact_bindings")
        plan = _validated_sibling_suite_plan(path, value.get("suite_plan_artifact"))
        planned_children = _planned_children_from_suite_plan(plan)
        nested_bindings = (
            _orchestration_child_bindings(path.parent, planned_children)
            if planned_children is not None
            else None
        )
        binding["suite_plan_binding_valid"] = plan is not None
        binding["suite_plan_topology_valid"] = planned_children is not None
        binding["orchestration_contract_declared"] = bool(
            value.get("artifact_role") == _ORCHESTRATION_INDEX_ROLE
            and value.get("root_run_validation_applicable") is False
            and isinstance(child_bindings, list)
        )
        binding["suite_completion_status_valid"] = bool(
            path.name == "suite_complete.json"
            and value.get("schema_version") == "1.0.0"
            and value.get("status") == _PRIMARY_SUITE_COMPLETE_STATUS
            and value.get("publication_evidence_ready") is False
        )
        binding["nested_child_bindings_present"] = isinstance(child_bindings, list)
        binding["nested_child_binding_count"] = (
            len(child_bindings) if isinstance(child_bindings, list) else None
        )
        binding["nested_child_bindings_match_current_tree"] = bool(
            isinstance(child_bindings, list)
            and isinstance(nested_bindings, list)
            and child_bindings == nested_bindings
        )
        binding["nested_child_bindings_complete"] = bool(
            isinstance(child_bindings, list)
            and isinstance(nested_bindings, list)
            and planned_children is not None
            and child_bindings == nested_bindings
            and _bindings_match_plan(nested_bindings, planned_children)
        )
    hash_fields = {
        "result.json": "result_payload_sha256_before_serialization",
        "failure.json": "failure_payload_sha256_before_serialization",
        "suite_complete.json": "record_sha256",
        "suite_failure.json": "failure_payload_sha256_before_serialization",
    }
    hash_field = hash_fields.get(path.name)
    if hash_field is not None:
        unhashed = dict(value)
        declared_hash = unhashed.pop(hash_field, None)
        binding["self_hash_field"] = hash_field
        binding["self_hash_valid"] = bool(
            isinstance(declared_hash, str) and declared_hash == canonical_json_sha256(unhashed)
        )
    return binding


def _orchestration_child_bindings(
    output_root: Path,
    planned_children: tuple[tuple[str, str], ...],
) -> list[dict[str, Any]]:
    """Bind every existing direct child without treating the root as a run."""

    roles = dict(planned_children)
    existing_names = {child.name for child in output_root.iterdir() if child.is_dir()}
    bindings: list[dict[str, Any]] = []
    for name in sorted(existing_names):
        directory = output_root / name
        role = roles.get(name, "unexpected_child_directory")
        directory_regular = directory.is_dir() and not directory.is_symlink()
        terminal_names = (
            ("result.json", "failure.json")
            if role == _SCIENTIFIC_RUN_ROLE
            else ("suite_complete.json", "suite_failure.json")
            if role == _ORCHESTRATION_INDEX_ROLE
            else ("result.json", "failure.json", "suite_complete.json", "suite_failure.json")
        )
        terminals = (
            [
                _json_artifact_binding(output_root, directory / terminal_name)
                for terminal_name in terminal_names
                if (directory / terminal_name).is_file()
            ]
            if directory_regular
            else []
        )
        validation_path = directory / "validation.json"
        validation = (
            _json_artifact_binding(output_root, validation_path)
            if directory_regular and validation_path.is_file()
            else None
        )
        terminal_json_valid = bool(
            len(terminals) == 1
            and terminals[0].get("regular_non_symlink") is True
            and terminals[0].get("json_object_valid") is True
            and terminals[0].get("status") is not None
        )
        validation_json_valid = bool(
            isinstance(validation, dict)
            and validation.get("regular_non_symlink") is True
            and validation.get("json_object_valid") is True
            and validation.get("status") is not None
        )
        validation_accepted = bool(
            validation_json_valid
            and validation is not None
            and validation.get("integrity_passed") is True
            and (
                (
                    validation.get("status") == "VALIDATED"
                    and validation.get("publication_evidence_ready") is True
                    and validation.get("publication_evidence_ready_for_unqualified_methods")
                    is False
                    and validation.get("diagnostic_contract_passed") is False
                )
                or (
                    validation.get("status") == "PARTIALLY_VALIDATED_METHODS"
                    and validation.get("publication_evidence_ready") is False
                    and validation.get("publication_evidence_ready_for_unqualified_methods") is True
                    and validation.get("diagnostic_contract_passed") is False
                )
                or (
                    validation.get("status") == "DIAGNOSTIC"
                    and validation.get("publication_evidence_ready") is False
                    and validation.get("publication_evidence_ready_for_unqualified_methods")
                    is False
                    and validation.get("diagnostic_contract_passed") is True
                )
            )
        )
        if role == _SCIENTIFIC_RUN_ROLE:
            completion_requirements = {
                "child_directory_regular_non_symlink": directory_regular,
                "exactly_one_result_and_no_failure": bool(
                    terminal_json_valid and terminals[0].get("artifact_name") == "result.json"
                ),
                "result_self_hash_valid": bool(
                    terminal_json_valid and terminals[0].get("self_hash_valid") is True
                ),
                "validation_json_valid": validation_json_valid,
                "validation_integrity_and_acceptance_passed": validation_accepted,
            }
        elif role == _ORCHESTRATION_INDEX_ROLE:
            completion_requirements = {
                "child_directory_regular_non_symlink": directory_regular,
                "exactly_one_suite_complete_and_no_suite_failure": bool(
                    terminal_json_valid
                    and terminals[0].get("artifact_name") == "suite_complete.json"
                ),
                "suite_complete_self_hash_valid": bool(
                    terminal_json_valid and terminals[0].get("self_hash_valid") is True
                ),
                "orchestration_contract_declared": bool(
                    terminal_json_valid
                    and terminals[0].get("orchestration_contract_declared") is True
                ),
                "suite_completion_status_valid": bool(
                    terminal_json_valid
                    and terminals[0].get("suite_completion_status_valid") is True
                ),
                "nested_child_bindings_present_and_complete": bool(
                    terminal_json_valid
                    and terminals[0].get("nested_child_bindings_complete") is True
                ),
                "suite_plan_binding_valid": bool(
                    terminal_json_valid and terminals[0].get("suite_plan_binding_valid") is True
                ),
                "suite_plan_topology_valid": bool(
                    terminal_json_valid and terminals[0].get("suite_plan_topology_valid") is True
                ),
                "nested_child_bindings_match_current_tree": bool(
                    terminal_json_valid
                    and terminals[0].get("nested_child_bindings_match_current_tree") is True
                ),
            }
        else:
            completion_requirements = {"planned_child_role_recognized": False}
        bindings.append(
            {
                "relative_directory": name,
                "artifact_role": role,
                "terminal_artifacts": terminals,
                "validation_record": validation,
                "validation_required": role == _SCIENTIFIC_RUN_ROLE,
                "completion_requirements": completion_requirements,
                "binding_complete": all(completion_requirements.values()),
            }
        )
    return bindings


def _require_complete_orchestration_children(
    bindings: list[dict[str, Any]],
    planned_children: tuple[tuple[str, str], ...],
) -> None:
    """Fail before sealing completion when any planned child binding is incomplete."""

    if not _bindings_match_plan(bindings, planned_children):
        raise ValueError("orchestration completion cannot be sealed without every child binding")


def _method_means(result: dict[str, Any]) -> dict[str, float]:
    reports = cast(dict[str, dict[str, Any]], result["primary_seed_averaged"]["methods"])
    return {name: float(report["mean_participant_macro_f1"]) for name, report in reports.items()}


def _validate_stage(
    directory: Path, repository_root: Path, *, allow_provider_segmented_diagnostic: bool = False
) -> None:
    validate_and_record_run_directory(
        directory,
        repository_root,
        diagnostic=True if allow_provider_segmented_diagnostic else False,
    )


def _imu_cohort_guard_evidence(summary: dict[str, Any], *, selection_policy: str) -> dict[str, Any]:
    """Return a compact, self-hashed witness for the pre-training cohort gate."""

    witness = {
        "protocol_id": IMU_HAR_IL_INVENTORY_PROTOCOL_ID,
        "selection_policy": selection_policy,
        "contract_passed": _imu_har_il_expected_cohort_observed(
            summary, selection_policy=selection_policy
        ),
        "dataset_id": summary.get("dataset_id"),
        "participant_count": summary.get("participant_count"),
        "trial_count": summary.get("trial_count"),
        "receipt_count": summary.get("receipt_count"),
        "participant_partition_plan": summary.get("participant_partition_plan"),
        "participant_partition_observation": summary.get("participant_partition_observation"),
        "cohort_audit": summary.get("cohort_audit"),
    }
    witness["record_sha256"] = canonical_json_sha256(witness)
    return witness


def run_primary_suite(
    *,
    output_root: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    epochs: int,
    n_jobs: int,
    selection_policy: str = "complete_requested_core",
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reuse one full source materialization across source and transfer experiments."""

    if selection_policy not in {"complete_requested_core", "available_valid_trials"}:
        raise ValueError("unknown IMU-HAR-IL suite cohort policy")
    if seeds != (11, 23, 47):
        raise ValueError("external primary suite requires frozen seeds 11, 23, and 47")
    if epochs != 40:
        raise ValueError("external primary suite requires the frozen 40-epoch neural budget")
    source_tag = (
        "imu_har_il_available_trials"
        if selection_policy == "available_valid_trials"
        else "imu_har_il_all_repetitions"
    )
    transfer_tag = (
        "imu_available_to_fog_zero_shot_3seed"
        if selection_policy == "available_valid_trials"
        else "imu_all_to_fog_zero_shot_3seed"
    )
    planned_children = (
        (f"{source_tag}_3seed", _SCIENTIFIC_RUN_ROLE),
        (f"{source_tag}_neural_3seed", _SCIENTIFIC_RUN_ROLE),
        (transfer_tag, _SCIENTIFIC_RUN_ROLE),
    )
    git_at_launch, source_input_manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output_root,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    output_root.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    suite_plan_artifact = _write_self_hashed_json_create_only(
        output_root / "suite_plan.json",
        {
            "schema_version": "1.0.0",
            "started_at": started,
            "execution_order": [
                "IMU-HAR-IL all-repetition nested invention/classical evaluation",
                "IMU-HAR-IL all-repetition neural controls",
                "IMU-HAR-IL to FoG-STAR zero-shot transfer",
            ],
            "seeds": list(seeds),
            "neural_epochs": epochs,
            "cohort_selection_policy": selection_policy,
            "raw_local_mirror": False,
            "source_materialization_reused_in_memory": True,
            "git_at_launch": git_at_launch,
            "source_input_manifest": source_input_manifest,
            "publication_launch_context": launch_context_binding,
            "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
            "environment": _runtime_environment(),
            "artifact_role": _ORCHESTRATION_INDEX_ROLE,
            "root_run_validation_applicable": False,
            "planned_children": [
                {"relative_directory": name, "artifact_role": role}
                for name, role in planned_children
            ],
        },
    )
    imu_cohort_guard: dict[str, Any] | None = None
    try:
        experiment = _read_mapping(
            repository_root / "configs/experiments/cross_dataset_har_rnd_v1.yaml"
        )
        preprocessing = cast(dict[str, Any], experiment["preprocessing"])
        derived = cast(dict[str, Any], preprocessing["derived_gravity"])
        target_rate_hz = float(preprocessing["target_sampling_rate_hz"])
        window_samples = int(preprocessing["window_samples"])
        gravity_cutoff_hz = float(derived["cutoff_hz"])
        source = load_imu_har_il(
            repetition_limit=4,
            selection_policy=selection_policy,
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        source_summary = source.summary()
        imu_cohort_guard = _imu_cohort_guard_evidence(
            source_summary, selection_policy=selection_policy
        )
        if imu_cohort_guard["contract_passed"] is not True:
            raise ValueError(
                "IMU-HAR-IL suite requires the exact frozen 50-person provider inventory "
                "and its declared lane-specific retained cohort before model fitting"
            )
        classical = run_and_write(
            data=source,
            output_directory=output_root / f"{source_tag}_3seed",
            repository_root=repository_root,
            seeds=seeds,
            n_jobs=n_jobs,
            include_classical=True,
            inherited_launch_context=launch_context,
        )
        _validate_stage(
            output_root / f"{source_tag}_3seed",
            repository_root,
            allow_provider_segmented_diagnostic=True,
        )
        _write_json_create_only(
            output_root / "stage_01_imu_classical_complete.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "dataset": source_summary,
                "imu_har_il_cohort_guard": imu_cohort_guard,
                "method_means": _method_means(classical),
            },
        )
        neural = run_and_write_neural(
            data=source,
            output_directory=output_root / f"{source_tag}_neural_3seed",
            repository_root=repository_root,
            seeds=seeds,
            epochs=epochs,
            inherited_launch_context=launch_context,
        )
        _validate_stage(
            output_root / f"{source_tag}_neural_3seed",
            repository_root,
            allow_provider_segmented_diagnostic=True,
        )
        _write_json_create_only(
            output_root / "stage_02_imu_neural_complete.json",
            {
                "completed_at": datetime.now(UTC).isoformat(),
                "method_means": _method_means(neural),
            },
        )
        gc.collect()
        target = load_fog_star(
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        transfer = run_and_write_transfer(
            source=source,
            target=target,
            output_directory=output_root / transfer_tag,
            repository_root=repository_root,
            seeds=seeds,
            n_jobs=n_jobs,
            include_classical=True,
            inherited_launch_context=launch_context,
        )
        _validate_stage(
            output_root / transfer_tag,
            repository_root,
            allow_provider_segmented_diagnostic=True,
        )
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output_root,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        child_bindings = _orchestration_child_bindings(output_root, planned_children)
        _require_complete_orchestration_children(child_bindings, planned_children)
        completed = {
            "schema_version": "1.0.0",
            "status": "DIAGNOSTIC_PROVIDER_PRESEGMENTED_SUITE_COMPLETE",
            "publication_evidence_ready": False,
            "started_at": started,
            "completed_at": datetime.now(UTC).isoformat(),
            "imu_method_means": _method_means(classical),
            "imu_neural_method_means": _method_means(neural),
            "zero_shot_method_means": _method_means(transfer),
            "raw_local_mirror": False,
            "cohort_selection_policy": selection_policy,
            "imu_har_il_cohort_guard": imu_cohort_guard,
            "publication_launch_context": launch_context_binding,
            "artifact_role": _ORCHESTRATION_INDEX_ROLE,
            "root_run_validation_applicable": False,
            "child_artifact_bindings": child_bindings,
            "suite_plan_artifact": suite_plan_artifact,
        }
        completed["git_at_launch"] = git_at_launch
        completed["source_input_manifest"] = source_input_manifest
        _write_self_hashed_json_create_only(output_root / "suite_complete.json", completed)
        return completed
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_PRESERVED",
            "started_at": started,
            "failed_at": datetime.now(UTC).isoformat(),
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "git": _git_state(repository_root),
            "git_at_launch": git_at_launch,
            "source_input_manifest": source_input_manifest,
            "publication_launch_context": launch_context_binding,
            "environment": _runtime_environment(),
            "artifact_role": _ORCHESTRATION_INDEX_ROLE,
            "root_run_validation_applicable": False,
            "child_artifact_bindings": _orchestration_child_bindings(output_root, planned_children),
            "suite_plan_artifact": suite_plan_artifact,
        }
        if imu_cohort_guard is not None:
            failure["imu_har_il_cohort_guard"] = imu_cohort_guard
        _write_self_hashed_json_create_only(
            output_root / "suite_failure.json",
            failure,
            hash_field="failure_payload_sha256_before_serialization",
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--selection-policy",
        choices=("complete_requested_core", "available_valid_trials"),
        default="complete_requested_core",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_primary_suite(
        output_root=args.output_root.resolve(),
        repository_root=args.repository_root.resolve(),
        seeds=tuple(args.seeds),
        epochs=args.epochs,
        n_jobs=args.n_jobs,
        selection_policy=args.selection_policy,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
