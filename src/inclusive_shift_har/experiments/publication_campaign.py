"""Frozen-source external campaigns with acquisition failures and acceptance gates."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inclusive_shift_har.artifacts.research_provenance import (
    PUBLICATION_SOURCE_PROTOCOL_ID,
    _fog_star_full_cohort_observed,
    _har_pmd_full_cohort_observed,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_manifest_commit_errors,
    _write_self_hashed_json_create_only,
)
from inclusive_shift_har.data.external_har import (
    load_fog_star,
    load_har_pmd_native,
    load_sole_harmony,
)
from inclusive_shift_har.experiments import (
    cross_dataset_har as cross_dataset_har,
)
from inclusive_shift_har.experiments import (
    cross_dataset_neural as cross_dataset_neural,
)
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
)
from inclusive_shift_har.experiments.external_primary_suite import (
    _ORCHESTRATION_INDEX_ROLE,
    _SCIENTIFIC_RUN_ROLE,
    _orchestration_child_bindings,
    _require_complete_orchestration_children,
    run_primary_suite,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _accept(directory: Path, root: Path, *, diagnostic: bool = False) -> None:
    validate_and_record_run_directory(directory, root, diagnostic=diagnostic)


def _campaign_child_specs(dataset: str) -> tuple[tuple[str, str], ...]:
    """Return the frozen direct-child topology for one campaign index."""

    if dataset in {"imu-har-il", "imu-har-il-available"}:
        return (("primary_suite", _ORCHESTRATION_INDEX_ROLE),)
    if dataset == "fog-star":
        return (("classical", _SCIENTIFIC_RUN_ROLE), ("neural", _SCIENTIFIC_RUN_ROLE))
    if dataset == "har-pmd":
        return (
            ("classical", _SCIENTIFIC_RUN_ROLE),
            ("neural_6ch", _SCIENTIFIC_RUN_ROLE),
            ("neural_N9", _SCIENTIFIC_RUN_ROLE),
        )
    if dataset == "sole-harmony":
        return (
            ("session_observable", _SCIENTIFIC_RUN_ROLE),
            ("camera_bout_oracle", _SCIENTIFIC_RUN_ROLE),
        )
    return ()


def run_campaign(
    *,
    dataset: str,
    output: Path,
    repository_root: Path,
    n_jobs: int = 4,
    har_pmd_source_archive: Path | None = None,
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if har_pmd_source_archive is not None and dataset != "har-pmd":
        raise ValueError("--har-pmd-source-archive is valid only for the HAR-PMD campaign")
    launch, manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    planned_children = _campaign_child_specs(dataset)
    output.mkdir(parents=True, exist_ok=False)
    plan = {
        "schema_version": "1.0.0",
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "dataset": dataset,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "seeds": [11, 23, 47],
        "neural_epochs": 40,
        "n_jobs": n_jobs,
        "git_at_launch": launch,
        "source_input_manifest": manifest,
        "publication_launch_context": launch_context_binding,
        "environment": _runtime_environment(),
        "provider_boundary_audit_protocol_id": "external-har-provider-boundary-audit-v1",
        "artifact_role": _ORCHESTRATION_INDEX_ROLE,
        "root_run_validation_applicable": False,
        "planned_children": [
            {"relative_directory": name, "artifact_role": role} for name, role in planned_children
        ],
        "evidence_role": "development_temporal_diagnostic_with_separate_oracle_lane"
        if dataset == "sole-harmony"
        else "stress_test"
        if dataset == "har-pmd"
        else "development_diagnostic_provider_presegmented"
        if dataset in {"imu-har-il", "imu-har-il-available"}
        else "development",
    }
    if dataset == "sole-harmony":
        temporal_protocol = "configs/protocols/sole_harmony_observable_session_temporal_v1.yaml"
        plan["sole_harmony_temporal_protocol"] = {
            "protocol_id": "sole-harmony-observable-session-temporal-v1",
            "path": temporal_protocol,
            "sha256": sha256_file(repository_root / temporal_protocol),
        }
        plan["ordered_lanes"] = [
            "session_observable_primary_development",
            "camera_bout_oracle_non_comparable_diagnostic",
        ]
    if dataset == "har-pmd":
        plan["har_pmd_source_archive"] = (
            None
            if har_pmd_source_archive is None
            else {
                "path": str(har_pmd_source_archive.resolve()),
                "verification": "exact pinned provider size, SHA-256 and MD5 before ZIP parsing",
            }
        )
    plan["record_sha256"] = canonical_json_sha256(plan)
    _write_json_create_only(output / "campaign_plan.json", plan)
    print(
        json.dumps(
            {"stage": "frozen_campaign_started", "dataset": dataset, "commit": launch["commit"]}
        ),
        flush=True,
    )
    try:
        if dataset in {"imu-har-il", "imu-har-il-available"}:
            run_primary_suite(
                output_root=output / "primary_suite",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                epochs=40,
                n_jobs=n_jobs,
                selection_policy="available_valid_trials"
                if dataset == "imu-har-il-available"
                else "complete_requested_core",
                inherited_launch_context=launch_context,
            )
        elif dataset == "fog-star":
            data = load_fog_star()
            if not _fog_star_full_cohort_observed(data.summary()):
                raise ValueError(
                    "full FoG-STAR campaign requires the exact 22-person provider roster "
                    "and all planned participants to contribute scored windows"
                )
            cross_dataset_har.run_and_write(
                data=data,
                output_directory=output / "classical",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                n_jobs=n_jobs,
                include_classical=True,
                inherited_launch_context=launch_context,
            )
            _accept(output / "classical", repository_root)
            print("FoG classical acceptance gate passed; starting neural controls", flush=True)
            cross_dataset_neural.run_and_write_neural(
                data=data,
                output_directory=output / "neural",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                epochs=40,
                inherited_launch_context=launch_context,
            )
            _accept(output / "neural", repository_root)
        elif dataset == "har-pmd":
            from inclusive_shift_har.experiments.har_pmd_stress import run_and_write

            data = load_har_pmd_native(source_archive=har_pmd_source_archive)
            har_summary = data.summary()
            if not _har_pmd_full_cohort_observed(har_summary):
                raise ValueError(
                    "full HAR-PMD campaign requires all 120 planned participants to contribute windows"
                )
            run_and_write(
                data=data,
                output_directory=output / "classical",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                inherited_launch_context=launch_context,
            )
            _accept(output / "classical", repository_root)
            for suffix, signals in (("6ch", data.signals), ("N9", data.nine_channel_signals)):
                cross_dataset_neural.run_and_write_neural(
                    data=data,
                    signals=signals,
                    class_names=data.class_names,
                    method_suffix=suffix,
                    experiment_id="har-pmd-native-interface-neural-controls-v1",
                    output_directory=output / f"neural_{suffix}",
                    repository_root=repository_root,
                    seeds=(11, 23, 47),
                    epochs=40,
                    inherited_launch_context=launch_context,
                )
                _accept(output / f"neural_{suffix}", repository_root)
        elif dataset == "sole-harmony":
            from inclusive_shift_har.experiments.sole_harmony_temporal import (
                run_and_write as run_temporal,
            )

            data = load_sole_harmony(
                participant_limit=12,
                sessions_per_participant=2,
                boundary_mode="session_observable",
            )
            run_temporal(
                data=data,
                output_directory=output / "session_observable",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                n_jobs=n_jobs,
                inherited_launch_context=launch_context,
            )
            _accept(output / "session_observable", repository_root, diagnostic=True)
            print(
                "Sole-HARmony session-observable acceptance gate passed; "
                "starting separate camera-bout oracle diagnostic",
                flush=True,
            )
            oracle_data = load_sole_harmony(
                participant_limit=12,
                sessions_per_participant=2,
                boundary_mode="camera_bout_oracle",
            )
            run_temporal(
                data=oracle_data,
                output_directory=output / "camera_bout_oracle",
                repository_root=repository_root,
                seeds=(11, 23, 47),
                n_jobs=n_jobs,
                inherited_launch_context=launch_context,
            )
            _accept(output / "camera_bout_oracle", repository_root, diagnostic=True)
        else:
            raise ValueError(f"dataset is not in the frozen campaign: {dataset}")
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        child_bindings = _orchestration_child_bindings(output, planned_children)
        _require_complete_orchestration_children(child_bindings, planned_children)
        completed = {
            "schema_version": "1.0.0",
            "status": "DIAGNOSTIC_COMPLETE"
            if dataset in {"sole-harmony", "imu-har-il", "imu-har-il-available"}
            else "CAMPAIGN_COMPLETE",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "plan_sha256": plan["record_sha256"],
            "publication_launch_context": launch_context_binding,
            "artifact_role": _ORCHESTRATION_INDEX_ROLE,
            "root_run_validation_applicable": False,
            "child_artifact_bindings": child_bindings,
        }
        _write_self_hashed_json_create_only(output / "campaign_complete.json", completed)
        return completed
    except Exception as error:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_PRESERVED",
            "failed_at_utc": datetime.now(UTC).isoformat(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "traceback": traceback.format_exc(),
            "git_at_launch": launch,
            "source_input_manifest": manifest,
            "publication_launch_context": launch_context_binding,
            "plan_sha256": plan["record_sha256"],
            "artifact_role": _ORCHESTRATION_INDEX_ROLE,
            "root_run_validation_applicable": False,
            "child_artifact_bindings": _orchestration_child_bindings(output, planned_children),
        }
        _write_self_hashed_json_create_only(
            output / "campaign_failure.json",
            failure,
            hash_field="failure_payload_sha256_before_serialization",
        )
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("fog-star", "imu-har-il", "imu-har-il-available", "har-pmd", "sole-harmony"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--har-pmd-source-archive", type=Path)
    args = parser.parse_args(argv)
    result = run_campaign(
        dataset=args.dataset,
        output=args.output.resolve(),
        repository_root=args.repository_root.resolve(),
        n_jobs=args.n_jobs,
        har_pmd_source_archive=args.har_pmd_source_archive,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
