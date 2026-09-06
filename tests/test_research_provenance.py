from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from inclusive_shift_har.artifacts import publication_checkpoint, research_provenance
from inclusive_shift_har.artifacts.research_provenance import (
    EXECUTED_RESEARCH_MODULE_PATH,
    FOG_STAR_FULL_PARTICIPANT_ROSTER,
    HAR_PMD_FULL_PARTICIPANT_ROSTER,
    PUBLICATION_SOURCE_PROTOCOL_ID,
    SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
    _assert_executed_repository_root,
    _external_evidence_status,
    _git_state,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_input_manifest,
    _write_json_create_only,
    _write_launch_failure_envelope,
)
from inclusive_shift_har.experiments import external_evidence_validate
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def test_command_recorder_does_not_import_torch_or_sklearn() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import inclusive_shift_har.artifacts.publication_checkpoint; "
            "assert 'torch' not in sys.modules; assert 'sklearn' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_readonly_research_audits_do_not_import_torch() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from inclusive_shift_har.experiments import external_evidence_validate, "
            "publication_table, publication_diagnostics, observable_context_parity; "
            "assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_lightweight_manifest_keeps_executed_package_binding(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="research module"):
        _source_input_manifest(tmp_path)
    root = Path(__file__).resolve().parents[1]
    manifest = _source_input_manifest(root)
    assert manifest["protocol_id"] == PUBLICATION_SOURCE_PROTOCOL_ID
    assert "src/inclusive_shift_har/artifacts/research_provenance.py" in manifest["files"]
    assert "configs/protocols/external_har_publication_v4.yaml" in manifest["files"]
    assert "AGENTS.md" in manifest["files"]
    assert "docs/LOCKED_PROTOCOL.md" in manifest["files"]
    assert "results/protocol/protocol_lock_v1_2.json" in manifest["files"]
    assert manifest["evidence_role_ledger"]["path"] in manifest["files"]
    assert manifest["supersession_ledger"]["path"] in manifest["files"]
    assert (
        "results/research/cross_dataset_har_v4/"
        "prewindow_provider_boundary_supersession_20260905.json"
    ) in manifest["files"]
    assert SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH in manifest["files"]
    assert manifest["file_count"] == len(manifest["files"])


def test_source_binding_rejects_same_layout_copy_at_a_different_root(tmp_path: Path) -> None:
    other_root = tmp_path / "same-depth-copy"
    copied_module = other_root.joinpath(*EXECUTED_RESEARCH_MODULE_PATH.split("/"))
    copied_module.parent.mkdir(parents=True)
    copied_module.write_bytes(Path(research_provenance.__file__).read_bytes())
    with pytest.raises(ValueError, match="does not belong"):
        _assert_executed_repository_root(other_root)


def test_source_binding_rejects_linked_source_ancestor(tmp_path: Path) -> None:
    linked_root = tmp_path / "linked-repository"
    linked_root.mkdir()
    source_link = linked_root / "src"
    actual_source = Path(research_provenance.__file__).resolve().parents[2]
    try:
        source_link.symlink_to(actual_source, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="does not belong"):
        _assert_executed_repository_root(linked_root)


def test_runtime_environment_records_hashed_interpreter_and_executed_module() -> None:
    environment = _runtime_environment()
    executable = environment["python_executable"]
    executed_module = environment["executed_research_module"]
    assert executable["sha256"] == sha256_file(Path(executable["resolved_path_at_runtime"]))
    assert executed_module["repository_relative_path"] == EXECUTED_RESEARCH_MODULE_PATH
    assert executed_module["sha256"] == sha256_file(
        Path(executed_module["resolved_path_at_runtime"])
    )


def test_publication_v4_protocol_composes_every_validity_correction() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs/protocols/external_har_publication_v4.yaml"
    protocol = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert protocol["protocol_id"] == PUBLICATION_SOURCE_PROTOCOL_ID
    assert {item["protocol_id"] for item in protocol["composes"].values()} == {
        "external-har-session-grid-v3",
        "external-har-observable-context-v1",
        "external-har-observable-context-training-population-v1",
        "external-har-provider-boundary-audit-v1",
        "external-har-prewindow-partitions-v1",
        "imu-har-il-fixed-inventory-v1",
    }
    assert (
        protocol["declaration_scope"]["v4_replacement_outcomes_examined_before_declaration"]
        is False
    )
    assert (
        protocol["required_run_contract"][
            "full_fog_star_provider_roster_and_scored_participant_coverage_required"
        ]
        is True
    )
    assert all(
        protocol["required_run_contract"][name] is True
        for name in (
            "full_imu_har_il_provider_roster_and_lane_specific_inventory_required",
            "imu_har_il_source_receipt_inventory_hash_reconstruction_required",
            "imu_har_il_repetitions_do_not_increase_independent_n",
        )
    )


def test_retained_sole_oracle_materialization_is_hash_bound_and_cannot_be_promoted() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt_path = root / SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH
    receipt_file_sha256 = sha256_file(receipt_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    declared_record_sha256 = receipt.pop("record_sha256")

    assert declared_record_sha256 == canonical_json_sha256(receipt)
    assert receipt["immutable"] is True
    assert receipt["status"] == "INTERRUPTED_AFTER_DATA_AUDIT_NO_MODEL_RESULT"
    assert receipt["evidence_role"] == "oracle_diagnostic"
    assert receipt["frozen_launch"]["git_commit"] == ("b2dae889d96d51a953289d00a20c68fcfec3f1fa")
    assert receipt["frozen_launch"]["git_worktree_dirty"] is False
    assert receipt["frozen_launch"]["source_manifest_sha256"] == (
        "2aac7c3e1da7269fea7dca10fd4e8f346de576f6b841ce484bd9e81c6febe99f"
    )
    artifact_bindings = receipt["retained_artifact_bindings"]
    assert artifact_bindings["campaign_plan"]["file_sha256"] == (
        "0b3db1820a17d7b9e4477061f9caf993f53e473ca092713c5f54621a9883a02e"
    )
    assert artifact_bindings["data_audit"]["file_sha256"] == (
        "081c3ddeedf998e5fabca158691aa98cdfb3ff4cf09b74a0f31d477e1cf78b2d"
    )
    assert artifact_bindings["launch_receipt"]["file_sha256"] == (
        "74f5d901469cadc2ace6363e5c38ffac094665d73168ccde3a97ad42dacc223c"
    )
    assert receipt["materialized_data_audit"]["participant_count"] == 12
    assert receipt["materialized_data_audit"]["session_count"] == 24
    assert receipt["materialized_data_audit"]["window_count"] == 137997
    assert receipt["materialized_data_audit"]["source_receipt_count"] == 48
    terminal = receipt["terminal_artifact_observation"]
    assert terminal["run_files_present"] == [
        "campaign_plan.json",
        "oracle_temporal/data_audit.json",
    ]
    assert all(
        terminal[name] is False
        for name in (
            "result_json_present",
            "predictions_present",
            "failure_json_present",
            "validation_json_present",
            "campaign_complete_json_present",
            "exit_receipt_present",
            "score_available",
        )
    )
    assert receipt["claim_contract"]["deployable_claim_allowed"] is False
    assert receipt["claim_contract"]["confirmation_allowed"] is False
    assert receipt["claim_contract"]["numerical_before_after_comparison_allowed"] is False

    manifest = _source_input_manifest(root)
    manifest_binding = manifest["sole_harmony_interruption_receipt"]
    assert manifest_binding == {
        "record_kind": receipt["record_kind"],
        "path": SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
        "sha256": receipt_file_sha256,
        "record_sha256": declared_record_sha256,
        "status": receipt["status"],
    }

    expected_binding = {
        "record_kind": "sole_harmony_interrupted_oracle_materialization_receipt",
        "path": SOLE_HARMONY_INTERRUPTED_MATERIALIZATION_RECEIPT_PATH,
        "file_sha256": receipt_file_sha256,
        "record_sha256": declared_record_sha256,
    }
    ledger = yaml.safe_load(
        (root / "configs/datasets/evidence_roles_20260905_v3.yaml").read_text(encoding="utf-8")
    )
    ledger_binding = ledger["datasets"]["sole_harmony_v1"][
        "retained_interrupted_oracle_materialization"
    ]
    assert {key: ledger_binding[key] for key in expected_binding} == expected_binding
    assert ledger_binding["deployable_claim_allowed"] is False
    assert ledger_binding["confirmation_allowed"] is False
    assert ledger_binding["numerical_before_after_comparison_allowed"] is False

    protocol = yaml.safe_load(
        (root / "configs/protocols/external_har_publication_v4.yaml").read_text(encoding="utf-8")
    )
    protocol_binding = protocol["evidence_governance"]["sole_harmony_retained_interruption"]
    assert {key: protocol_binding[key] for key in expected_binding} == expected_binding
    assert (
        protocol["required_run_contract"][
            "sole_retained_interrupted_materialization_may_not_be_promoted_to_a_result"
        ]
        is True
    )

    literature = (
        root / "docs/research/EXTERNAL_HAR_LITERATURE_AND_COMPARABILITY_AUDIT_20260905.md"
    ).read_text(encoding="utf-8")
    checkpoint_source = Path(publication_checkpoint.__file__).read_text(encoding="utf-8")
    assert "one-participant/session" not in literature
    assert "one participant/session" not in literature
    assert "one-session diagnostic" not in literature
    assert "one participant/session and camera-bout oracle boundaries" not in checkpoint_source
    assert "interrupted **oracle diagnostic materialization, not a model result**" in literature


def test_create_only_writer_cannot_replace_prior_evidence(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    _write_json_create_only(path, {"value": 1})
    with pytest.raises(FileExistsError):
        _write_json_create_only(path, {"value": 2})
    assert json.loads(path.read_text()) == {"value": 1}


def _annotation_independent_boundary() -> dict[str, Any]:
    return {
        "protocol_id": "external-har-boundary-provenance-v1",
        "repository_signal_grid_annotation_independent": True,
        "provider_upstream_annotation_conditioned": False,
    }


def test_har_pmd_pilot_status_cannot_be_upgraded_without_full_planned_roster() -> None:
    pilot = {
        "dataset_id": "har_pmd_v1",
        "boundary_provenance": _annotation_independent_boundary(),
        "participant_count": 12,
        "participant_partition_plan": {"participant_roster": list(HAR_PMD_FULL_PARTICIPANT_ROSTER)},
        "participant_partition_observation": {
            "planned_participant_count": 120,
            "observed_window_participant_count": 12,
            "participants_without_retained_windows": list(HAR_PMD_FULL_PARTICIPANT_ROSTER[12:]),
        },
    }
    full = {
        "dataset_id": "har_pmd_v1",
        "boundary_provenance": _annotation_independent_boundary(),
        "participant_count": 120,
        "participant_partition_plan": {"participant_roster": list(HAR_PMD_FULL_PARTICIPANT_ROSTER)},
        "participant_partition_observation": {
            "planned_participant_count": 120,
            "observed_window_participant_count": 120,
            "participants_without_retained_windows": [],
        },
    }
    assert _external_evidence_status(pilot) == "provisional_pilot_stress"
    assert _external_evidence_status(full) == "validated_stress_test"


def test_fog_subset_status_cannot_be_upgraded_without_exact_full_scored_roster() -> None:
    roster = list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
    full = {
        "dataset_id": "fog_star_v3",
        "boundary_provenance": _annotation_independent_boundary(),
        "participant_count": 22,
        "participant_partition_plan": {"participant_roster": roster},
        "participant_partition_observation": {
            "planned_participant_count": 22,
            "observed_window_participant_count": 22,
            "participants_without_retained_windows": [],
        },
    }
    subset = {
        **full,
        "participant_count": 21,
        "participant_partition_observation": {
            "planned_participant_count": 22,
            "observed_window_participant_count": 21,
            "participants_without_retained_windows": [roster[-1]],
        },
    }
    wrong_roster = {
        **full,
        "participant_partition_plan": {"participant_roster": [*roster[:-1], "fogstar:999"]},
    }
    assert _external_evidence_status(subset) == "provisional_subset_development"
    assert _external_evidence_status(wrong_roster) == "provisional_subset_development"
    assert _external_evidence_status(full) == "validated_development"


@pytest.mark.parametrize(
    "summaries",
    [
        (),
        ({},),
        ({"dataset_id": "unregistered_v1"},),
        ({"dataset_id": "imu_har_il_v1"}, {"dataset_id": "unregistered_v1"}),
    ],
)
def test_external_evidence_status_never_validates_unregistered_inputs(
    summaries: tuple[dict[str, Any], ...],
) -> None:
    status = _external_evidence_status(*summaries)
    assert status.startswith("provisional_")
    assert "validated" not in status


@pytest.mark.parametrize(
    "boundary",
    [
        None,
        {},
        {
            **_annotation_independent_boundary(),
            "repository_signal_grid_annotation_independent": False,
        },
        {
            **_annotation_independent_boundary(),
            "provider_upstream_annotation_conditioned": True,
        },
        {**_annotation_independent_boundary(), "protocol_id": "unknown-boundary-v1"},
    ],
)
def test_fog_full_roster_cannot_override_invalid_boundary_contract(
    boundary: dict[str, Any] | None,
) -> None:
    summary: dict[str, Any] = {
        "dataset_id": "fog_star_v3",
        "participant_count": 22,
        "participant_partition_plan": {
            "participant_roster": list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
        },
        "participant_partition_observation": {
            "planned_participant_count": 22,
            "observed_window_participant_count": 22,
            "participants_without_retained_windows": [],
        },
    }
    if boundary is not None:
        summary["boundary_provenance"] = boundary
    assert _external_evidence_status(summary) == "provisional_boundary_contract_invalid_development"


def test_only_exact_imu_to_full_valid_fog_transfer_gets_diagnostic_role() -> None:
    target = {
        "dataset_id": "fog_star_v3",
        "boundary_provenance": _annotation_independent_boundary(),
        "participant_count": 22,
        "participant_partition_plan": {
            "participant_roster": list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
        },
        "participant_partition_observation": {
            "planned_participant_count": 22,
            "observed_window_participant_count": 22,
            "participants_without_retained_windows": [],
        },
    }
    source = {"dataset_id": "imu_har_il_v1"}
    assert (
        _external_evidence_status(source, target)
        == "diagnostic_provider_presegmented_source_transfer"
    )
    assert _external_evidence_status(target, source) == "provisional_unregistered_transfer"
    target["participant_count"] = 21
    assert _external_evidence_status(source, target) == "provisional_subset_source_transfer"


def _initialize_clean_git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=path, check=True)
    (path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "frozen"], cwd=path, check=True)


def test_git_state_uses_nul_delimited_porcelain_for_untracked_paths(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    assert _git_state(repository)["status_entries"] == []
    evidence = repository / "results" / "research" / "run with spaces"
    evidence.mkdir(parents=True)
    (evidence / "plan.json").write_text("{}\n", encoding="utf-8")
    state = _git_state(repository)
    assert state["worktree_dirty"] is True
    assert state["status_entries"] == ["?? results/research/run with spaces/plan.json"]


def test_inherited_launch_allows_only_its_create_only_output_tree(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "campaign"
    manifest = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "files": {"tracked.txt": "0" * 64},
        "captured_at": "2026-09-05T00:00:00+00:00",
    }
    launch, frozen_manifest, context = _resolve_publication_launch_context(
        repository_root=repository,
        output_directory=output,
        current_git_state=_git_state(repository),
        current_source_manifest=manifest,
        manifest_commit_validator=lambda *_args: [],
    )
    assert launch["worktree_dirty"] is False
    assert frozen_manifest is manifest
    assert _publication_launch_context_binding(context)["allowed_create_only_output_root"] == {
        "kind": "repository_relative",
        "path": "results/research/campaign",
    }
    output.mkdir(parents=True)
    (output / "campaign_plan.json").write_text("{}\n", encoding="utf-8")
    child = output / "classical"
    inherited_launch, inherited_manifest, inherited_context = _resolve_publication_launch_context(
        repository_root=repository,
        output_directory=child,
        current_git_state=_git_state(repository),
        current_source_manifest={**manifest, "captured_at": "2026-09-05T00:01:00+00:00"},
        manifest_commit_validator=lambda *_args: [],
        inherited_launch_context=context,
    )
    assert inherited_launch == launch
    assert inherited_manifest == manifest
    assert inherited_context == context

    drifted_manifest = {**manifest, "files": {"tracked.txt": "1" * 64}}
    with pytest.raises(ValueError, match="governed source inputs changed"):
        _resolve_publication_launch_context(
            repository_root=repository,
            output_directory=child,
            current_git_state=_git_state(repository),
            current_source_manifest=drifted_manifest,
            manifest_commit_validator=lambda *_args: [],
            inherited_launch_context=context,
        )
    with pytest.raises(ValueError, match="outside the inherited"):
        _resolve_publication_launch_context(
            repository_root=repository,
            output_directory=repository / "other-run",
            current_git_state=_git_state(repository),
            current_source_manifest=manifest,
            manifest_commit_validator=lambda *_args: [],
            inherited_launch_context=context,
        )

    (repository / "unrelated.txt").write_text("unexpected\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dirt exists outside"):
        _resolve_publication_launch_context(
            repository_root=repository,
            output_directory=child,
            current_git_state=_git_state(repository),
            current_source_manifest=manifest,
            manifest_commit_validator=lambda *_args: [],
            inherited_launch_context=context,
        )


def test_pre_writer_failure_envelope_is_create_only_and_self_hashed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "failed-acquisition"
    manifest = {
        "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
        "files": {"tracked.txt": "0" * 64},
        "captured_at": "2026-09-05T00:00:00+00:00",
    }
    _launch, _manifest, context = _resolve_publication_launch_context(
        repository_root=repository,
        output_directory=output,
        current_git_state=_git_state(repository),
        current_source_manifest=manifest,
        manifest_commit_validator=lambda *_args: [],
    )
    monkeypatch.setattr(research_provenance, "_source_input_manifest", lambda _root: manifest)
    monkeypatch.setattr(research_provenance, "_source_manifest_commit_errors", lambda *_args: [])

    def record_validation(directory: Path, _repository: Path) -> dict[str, Any]:
        validation = {
            "status": "FAILED_RUN_PRESERVED",
            "integrity_passed": True,
            "publication_evidence_ready": False,
        }
        (directory / "validation.json").write_text(
            json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8"
        )
        return validation

    monkeypatch.setattr(
        external_evidence_validate,
        "validate_and_record_run_directory",
        record_validation,
    )
    _write_launch_failure_envelope(
        repository_root=repository,
        output_directory=output,
        launch_context=context,
        started_at="2026-09-05T00:01:00+00:00",
        stage="dataset_acquisition",
        exception=RuntimeError("provider unavailable"),
        traceback_text="synthetic traceback",
    )
    payload = json.loads((output / "failure.json").read_text(encoding="utf-8"))
    digest = payload.pop("failure_payload_sha256_before_serialization")
    assert digest == canonical_json_sha256(payload)
    assert payload["stage"] == "dataset_acquisition"
    assert payload["publication_launch_context"]["record_sha256"] == context["record_sha256"]
    assert (
        json.loads((output / "validation.json").read_text(encoding="utf-8"))["status"]
        == "FAILED_RUN_PRESERVED"
    )
    with pytest.raises(FileExistsError):
        _write_launch_failure_envelope(
            repository_root=repository,
            output_directory=output,
            launch_context=context,
            started_at="2026-09-05T00:01:00+00:00",
            stage="dataset_acquisition",
            exception=RuntimeError("again"),
            traceback_text="synthetic traceback",
        )


@pytest.mark.parametrize(
    "relative_output",
    ["src/new-run", "configs/new-run", "docs/research/new-run", "results/protocol/new-run"],
)
def test_launch_context_rejects_governed_output_namespaces(
    tmp_path: Path, relative_output: str
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    with pytest.raises(ValueError, match="governed source namespace"):
        _resolve_publication_launch_context(
            repository_root=repository,
            output_directory=repository / relative_output,
            current_git_state=_git_state(repository),
            current_source_manifest={"files": {"tracked.txt": "0" * 64}},
            manifest_commit_validator=lambda *_args: [],
        )


@pytest.mark.parametrize("drift", [False, True])
def test_quality_gate_cannot_pass_with_source_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: bool
) -> None:
    manifests = iter(
        [
            {"manifest_sha256": "a" * 64},
            {"manifest_sha256": ("b" if drift else "a") * 64},
        ]
    )
    monkeypatch.setattr(
        publication_checkpoint, "_source_input_manifest", lambda _root: next(manifests)
    )

    def command(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "exit_code": 0,
            "record_sha256": "c" * 64,
            "source_input_manifest": {"manifest_sha256": "a" * 64},
        }

    monkeypatch.setattr(publication_checkpoint, "record_command", command)
    result = publication_checkpoint.quality_gates(
        tmp_path / "gates", tmp_path, Path("uv"), workers=1
    )
    assert (result["status"] == "PASS") is not drift
    assert result["source_unchanged_across_all_gate_launches_and_completion"] is not drift
