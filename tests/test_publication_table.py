from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.artifacts.research_provenance import (
    FOG_STAR_FULL_PARTICIPANT_ROSTER,
    HAR_PMD_FULL_PARTICIPANT_ROSTER,
)
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import publication_table as table


def _boundary_provenance(*, provider_conditioned: bool) -> dict[str, Any]:
    return {
        "protocol_id": "external-har-boundary-provenance-v1",
        "source_boundary_unit": "synthetic physical recording",
        "repository_signal_grid_annotation_independent": not provider_conditioned,
        "provider_upstream_annotation_conditioned": provider_conditioned,
        "zero_lookahead_streaming_valid": False,
        "evidence_scope": "synthetic evidence-status test only",
    }


def _full_fog_dataset() -> dict[str, Any]:
    return {
        "dataset_id": "fog_star_v3",
        "channel_lane": "derived-gravity-9ch",
        "boundary_provenance": _boundary_provenance(provider_conditioned=False),
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


def _fixture(root: Path, name: str, method: str, *, correct: bool = True) -> Path:
    directory = root / name
    directory.mkdir()
    labels = np.tile(np.arange(3), 3)
    participants = np.repeat(["p1", "p2", "p3"], 3)
    probability = np.eye(3)[labels]
    if not correct:
        probability = np.roll(probability, 1, axis=1)
    primary, predictions = seed_evidence(
        ParticipantMetricInputs(
            "fog_star_v3", ("mobility", "sitting", "standing"), labels, participants
        ),
        {seed: {method: probability} for seed in (11, 23, 47)},
    )
    result: dict[str, Any] = {
        "dataset": _full_fog_dataset(),
        "artifact_evidence_status": "validated_development",
        "seeds": [11, 23, 47],
        "primary_seed_averaged": primary,
        "reports": {method: {"class_names": ["mobility", "sitting", "standing"]}},
        "source_input_manifest": {
            "protocol_id": "external-har-session-grid-v3",
            "manifest_sha256": "a" * 64,
            "files": {"src/inclusive_shift_har/data/external_har.py": "b" * 64},
        },
        "prediction_artifact": {"path": "predictions.npz"},
        "git_at_launch": {"commit": "c" * 40, "worktree_dirty": False},
    }
    audit = {
        "source_receipts": [
            {
                "dataset_id": "fog_star_v3",
                "locator": "https://example.invalid/source",
                "computed_sha256": "d" * 64,
            }
        ]
    }
    (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (directory / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    np.savez_compressed(
        directory / "predictions.npz",
        labels=labels,
        participant_ids=participants,
        session_ids=participants,
        trial_ids=participants,
        window_ids=np.array([f"w{i}" for i in range(9)]),
        **{f"probability__{key}": value for key, value in predictions.items()},  # type: ignore[arg-type]
    )
    return directory


@pytest.fixture
def validated(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unit boundary: actual artifact validation has its own adversarial suite.
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "status": "VALIDATED",
            "integrity_passed": True,
            "publication_evidence_ready": True,
            "publication_evidence_ready_for_unqualified_methods": False,
            "diagnostic_contract_passed": False,
        },
    )


@pytest.fixture
def diagnostic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "status": "DIAGNOSTIC",
            "integrity_passed": True,
            "publication_evidence_ready": False,
            "publication_evidence_ready_for_unqualified_methods": False,
            "diagnostic_contract_passed": True,
            "diagnostic_scope_reasons": ["synthetic scoped diagnostic"],
        },
    )


def _rewrite_result(directory: Path, mutation: dict[str, Any]) -> None:
    path = directory / "result.json"
    result = json.loads(path.read_text(encoding="utf-8"))
    result.update(mutation)
    path.write_text(json.dumps(result), encoding="utf-8")


def test_paired_comparisons_align_explicit_participant_ids() -> None:
    statistics = {
        "methods": {
            "RandomForest-6ch": {
                "mean_participant_macro_f1": 0.5,
                "participant_values": {"p1": 0.0, "p2": 1.0},
            },
            "candidate": {
                "mean_participant_macro_f1": 0.5,
                "participant_values": {"p2": 0.0, "p1": 1.0},
            },
        }
    }
    contract = {
        name: {
            "input_channel_count": 6,
            "inference_unit": "independent_fixed_window",
            "context_population": "window_only",
            "annotation_selected_evaluation_context": False,
        }
        for name in statistics["methods"]
    }
    result = table._strongest_control_comparisons(statistics, contract)
    pair = result["pairs"]["candidate"]
    assert pair["participant_delta_values"] == {"p1": 1.0, "p2": -1.0}
    assert pair["rescue_count"] == pair["harm_count"] == 1
    assert pair["tie_count"] == 0
    assert pair["difference_direction"] == "candidate_minus_control"


def test_sole_temporal_candidate_name_cannot_be_selected_as_its_own_control(
    tmp_path: Path, diagnostic: None
) -> None:
    control_name = "XGBoost-6ch-unsmoothed"
    candidate_name = "XGBoost-6ch-causal-probability-w5"
    run = _fixture(tmp_path, "sole", control_name, correct=False)
    prediction_path = run / "predictions.npz"
    with np.load(prediction_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    labels = np.asarray(arrays["labels"], dtype=np.int64)
    participants = np.asarray(arrays["participant_ids"], dtype=np.str_)
    control_by_seed = {
        seed: np.asarray(arrays[f"probability__seed-{seed}__{control_name}"], dtype=np.float64)
        for seed in (11, 23, 47)
    }
    candidate = np.eye(3, dtype=np.float64)[labels]
    probabilities = {
        seed: {control_name: control_by_seed[seed], candidate_name: candidate}
        for seed in (11, 23, 47)
    }
    primary, serialized = seed_evidence(
        ParticipantMetricInputs(
            "sole_harmony_v1",
            ("mobility", "sitting", "standing"),
            labels,
            participants,
        ),
        probabilities,
    )
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result.update(
        {
            "dataset": {
                "dataset_id": "sole_harmony_v1",
                "boundary_provenance": {
                    **_boundary_provenance(provider_conditioned=False),
                    "boundary_mode": "session_observable",
                },
            },
            "artifact_evidence_status": "diagnostic_session_observable_temporal_development",
            "primary_seed_averaged": primary,
            "reports": {
                control_name: {"class_names": ["mobility", "sitting", "standing"]},
                candidate_name: {"class_names": ["mobility", "sitting", "standing"]},
            },
            "advancement_gate": {"comparator": control_name},
            "temporal_contract": {"protocol_id": "sole-harmony-observable-session-temporal-v1"},
        }
    )
    result_path.write_text(json.dumps(result), encoding="utf-8")
    identity = {name: arrays[name] for name in table._IDENTITY_ARRAYS}
    np.savez_compressed(
        prediction_path,
        **identity,
        **{  # type: ignore[arg-type]
            f"probability__{name}": value for name, value in serialized.items()
        },
    )

    record = table.reconstruct_matched_table([run], tmp_path)
    comparison = record["descriptive_comparisons_vs_strongest_observed_control"]
    assert comparison["control"] == control_name
    assert comparison["pairs"][candidate_name]["mean_difference"] > 0
    candidate_group = next(
        group
        for group in record["descriptive_comparisons_vs_strongest_same_input_and_context_control"]
        if candidate_name in group["methods"]
    )
    assert candidate_group["comparisons"]["status"] == "no_applicable_control_in_table"


def test_neural_control_remains_eligible_when_classical_run_declares_its_best_control(
    tmp_path: Path, validated: None
) -> None:
    classical = _fixture(tmp_path, "classical", "XGBoost-6ch", correct=False)
    _rewrite_result(
        classical,
        {"descriptive_benchmark_comparison": {"strongest_control": "XGBoost-6ch"}},
    )
    neural = _fixture(tmp_path, "neural", "TinyHAR-6ch", correct=True)

    record = table.reconstruct_matched_table([classical, neural], tmp_path)
    assert (
        record["descriptive_comparisons_vs_strongest_observed_control"]["control"] == "TinyHAR-6ch"
    )


@pytest.mark.parametrize(
    "boundary",
    [
        None,
        _boundary_provenance(provider_conditioned=False),
        _boundary_provenance(provider_conditioned=True),
    ],
)
def test_imu_provider_presegmented_identity_can_never_be_validated(
    boundary: dict[str, Any] | None,
) -> None:
    dataset: dict[str, Any] = {"dataset_id": "imu_har_il_v1"}
    if boundary is not None:
        dataset["boundary_provenance"] = boundary
    status = table._run_evidence_status(
        {
            "dataset": dataset,
            "artifact_evidence_status": "diagnostic_provider_presegmented_development",
        },
        dataset,
    )
    assert status == "diagnostic_provider_presegmented_development"
    assert "validated" not in status


@pytest.mark.parametrize(
    "source_boundary",
    [
        None,
        _boundary_provenance(provider_conditioned=False),
        _boundary_provenance(provider_conditioned=True),
    ],
)
def test_imu_to_fog_source_identity_can_never_be_validated(
    source_boundary: dict[str, Any] | None,
) -> None:
    source: dict[str, Any] = {"dataset_id": "imu_har_il_v1"}
    if source_boundary is not None:
        source["boundary_provenance"] = source_boundary
    target = _full_fog_dataset()
    status = table._run_evidence_status(
        {
            "source_dataset": source,
            "target_dataset": target,
            "artifact_evidence_status": "diagnostic_provider_presegmented_source_transfer",
        },
        target,
    )
    assert status == "diagnostic_provider_presegmented_source_transfer"
    assert "validated" not in status


@pytest.mark.parametrize(
    ("dataset_id", "expected"),
    [
        ("fog_star_v3", "validated_development"),
        ("har_pmd_v1", "provisional_pilot_stress"),
    ],
)
def test_annotation_independent_fog_and_har_pmd_keep_their_evidence_status(
    dataset_id: str, expected: str
) -> None:
    dataset = (
        _full_fog_dataset()
        if dataset_id == "fog_star_v3"
        else {
            "dataset_id": dataset_id,
            "boundary_provenance": _boundary_provenance(provider_conditioned=False),
        }
    )
    assert (
        table._run_evidence_status(
            {"dataset": dataset, "artifact_evidence_status": expected}, dataset
        )
        == expected
    )


def test_har_pmd_status_requires_the_exact_planned_120_person_roster() -> None:
    pilot = {
        "dataset_id": "har_pmd_v1",
        "boundary_provenance": _boundary_provenance(provider_conditioned=False),
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
        "boundary_provenance": _boundary_provenance(provider_conditioned=False),
        "participant_count": 120,
        "participant_partition_plan": {"participant_roster": list(HAR_PMD_FULL_PARTICIPANT_ROSTER)},
        "participant_partition_observation": {
            "planned_participant_count": 120,
            "observed_window_participant_count": 120,
            "participants_without_retained_windows": [],
        },
    }
    assert (
        table._run_evidence_status(
            {"dataset": pilot, "artifact_evidence_status": "provisional_pilot_stress"},
            pilot,
        )
        == "provisional_pilot_stress"
    )
    assert (
        table._run_evidence_status(
            {"dataset": full, "artifact_evidence_status": "validated_stress_test"}, full
        )
        == "validated_stress_test"
    )


@pytest.mark.parametrize(
    ("boundary_mode", "expected"),
    [
        (None, "provisional_sole_boundary_contract_missing"),
        ("camera_bout_oracle", "diagnostic_oracle_boundary"),
        ("session_observable", "diagnostic_session_observable_temporal_development"),
    ],
)
def test_sole_identity_can_never_be_upgraded_to_validated(
    boundary_mode: str | None, expected: str
) -> None:
    dataset: dict[str, Any] = {"dataset_id": "sole_harmony_v1"}
    if boundary_mode is not None:
        dataset["boundary_provenance"] = {
            **_boundary_provenance(provider_conditioned=False),
            "boundary_mode": boundary_mode,
            "repository_signal_grid_annotation_independent": (
                boundary_mode == "session_observable"
            ),
        }
    status = table._run_evidence_status(
        {"dataset": dataset, "artifact_evidence_status": expected}, dataset
    )
    assert status == expected
    assert "validated" not in status


def test_table_status_derivation_is_fail_closed_for_unknown_or_forged_inputs() -> None:
    unknown = {"dataset_id": "unregistered_external_v1"}
    assert (
        table._run_evidence_status(
            {
                "dataset": unknown,
                "artifact_evidence_status": "provisional_unregistered_dataset",
            },
            unknown,
        )
        == "provisional_unregistered_dataset"
    )
    with pytest.raises(ValueError, match="differs from its frozen dataset role"):
        table._run_evidence_status(
            {"dataset": unknown, "artifact_evidence_status": "validated_development"},
            unknown,
        )


def test_table_rejects_full_fog_with_invalid_boundary_before_reconstruction(
    tmp_path: Path, validated: None
) -> None:
    run = _fixture(tmp_path, "invalid-boundary", "XGBoost-6ch")
    dataset = _full_fog_dataset()
    dataset["boundary_provenance"]["repository_signal_grid_annotation_independent"] = False
    _rewrite_result(
        run,
        {
            "dataset": dataset,
            "artifact_evidence_status": "provisional_boundary_contract_invalid_development",
        },
    )
    with pytest.raises(ValueError, match="no publication-eligible evidence role"):
        table.reconstruct_matched_table([run], tmp_path)


def test_table_rejects_validator_role_incompatible_with_canonical_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "role-mismatch", "XGBoost-6ch")
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "status": "DIAGNOSTIC",
            "integrity_passed": True,
            "publication_evidence_ready": False,
            "publication_evidence_ready_for_unqualified_methods": False,
            "diagnostic_contract_passed": True,
            "diagnostic_scope_reasons": ["forged diagnostic mode"],
        },
    )
    with pytest.raises(ValueError, match="incompatible with evidence role"):
        table.reconstruct_matched_table([run], tmp_path)


def test_imu_diagnostic_status_survives_derived_gravity_method_qualification(
    tmp_path: Path, diagnostic: None
) -> None:
    run = _fixture(tmp_path, "imu", "PB-RF-D9")
    dataset = {
        "dataset_id": "imu_har_il_v1",
        "channel_lane": "derived-gravity-9ch",
        # An inconsistent flag must not override the immutable provider evidence role.
        "boundary_provenance": _boundary_provenance(provider_conditioned=False),
    }
    _rewrite_result(
        run,
        {
            "dataset": dataset,
            "artifact_evidence_status": "diagnostic_provider_presegmented_development",
            "method_input_lanes": {"PB-RF-D9": "derived-nine-channel diagnostic"},
        },
    )
    result = table.reconstruct_matched_table([run], tmp_path)
    statuses = [result["evidence_status"], *result["method_evidence_statuses"].values()]
    assert all(status.startswith("diagnostic_provider_presegmented") for status in statuses)
    assert all("validated" not in status for status in statuses)


def test_imu_to_fog_diagnostic_status_survives_context_method_qualification(
    tmp_path: Path, diagnostic: None
) -> None:
    run = _fixture(tmp_path, "transfer", "HERA-DG-full")
    target = _full_fog_dataset()
    source = {
        "dataset_id": "imu_har_il_v1",
        # Dataset identity is fail closed even if provenance is absent or contradictory.
        "boundary_provenance": _boundary_provenance(provider_conditioned=False),
    }
    _rewrite_result(
        run,
        {
            "source_dataset": source,
            "target_dataset": target,
            "artifact_evidence_status": "diagnostic_provider_presegmented_source_transfer",
        },
    )
    result_path = run / "result.json"
    stored = json.loads(result_path.read_text(encoding="utf-8"))
    stored.pop("dataset")
    result_path.write_text(json.dumps(stored), encoding="utf-8")
    result = table.reconstruct_matched_table([run], tmp_path)
    statuses = [result["evidence_status"], *result["method_evidence_statuses"].values()]
    assert all(status.startswith("diagnostic_provider_presegmented") for status in statuses)
    assert all("validated" not in status for status in statuses)


def test_matched_table_reconstructs_seeds_and_preserves_primary_contrast(
    tmp_path: Path, validated: None
) -> None:
    base = _fixture(tmp_path, "base", "XGBoost-6ch", correct=False)
    forest = _fixture(tmp_path, "forest", "RandomForest-6ch")
    candidate = _fixture(tmp_path, "candidate", "HERA-DG-full")
    result = table.reconstruct_matched_table([base, forest, candidate], tmp_path)
    assert result["statistics"]["participant_count"] == 3
    assert result["statistics"]["methods"]["RandomForest-6ch"]["mean_participant_macro_f1"] == 1
    assert result["statistics"]["comparisons_vs_xgboost_6ch"]["HERA-DG-full"]["primary_contrast"]
    assert (
        result["descriptive_comparisons_vs_strongest_observed_control"]["control"]
        == "RandomForest-6ch"
    )
    assert "Evidence status" in table.table_markdown(result)
    assert result["reconstruction_only_no_training_or_raw_data_access"]


@pytest.mark.parametrize("mutation", ["lane", "seeds", "preprocessing", "raw_hash", "transfer"])
def test_table_rejects_noncomparable_contracts(
    tmp_path: Path, validated: None, mutation: str
) -> None:
    left = _fixture(tmp_path, "left", "XGBoost-6ch")
    right = _fixture(tmp_path, "right", "TinyHAR-6ch")
    result = json.loads((right / "result.json").read_text())
    audit = json.loads((right / "data_audit.json").read_text())
    if mutation == "lane":
        result["dataset"]["channel_lane"] = "native-gravity-9ch"
    elif mutation == "seeds":
        result["seeds"] = [11]
    elif mutation == "preprocessing":
        result["source_input_manifest"]["files"]["src/inclusive_shift_har/data/external_har.py"] = (
            "e" * 64
        )
    elif mutation == "raw_hash":
        audit["source_receipts"][0]["computed_sha256"] = "e" * 64
    else:
        result["target_dataset"] = result.pop("dataset")
    (right / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (right / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    with pytest.raises(ValueError, match=r"non-comparable|three frozen seeds|transfer table input"):
        table.reconstruct_matched_table([left, right], tmp_path)


def test_table_rejects_duplicate_methods_and_failed_validation(
    tmp_path: Path, validated: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "run", "XGBoost-6ch")
    with pytest.raises(ValueError, match="duplicate method"):
        table.reconstruct_matched_table([run, run], tmp_path)
    monkeypatch.setattr(
        table, "validate_run_directory", lambda *_: {"publication_evidence_ready": False}
    )
    with pytest.raises(ValueError, match="not validated"):
        table.reconstruct_matched_table([run], tmp_path)


def _superseded_validation(run: Path, *, integrity: bool = True) -> dict[str, Any]:
    return {
        "run_directory": str(run.resolve()),
        "status": "SUPERSEDED_PROTOCOL",
        "integrity_passed": integrity,
        "publication_evidence_ready": False,
        "publication_evidence_ready_for_unqualified_methods": False,
        "diagnostic_contract_passed": False,
    }


def test_table_rejects_superseded_protocol_by_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "historical", "XGBoost-6ch")
    monkeypatch.setattr(table, "validate_run_directory", lambda *_: _superseded_validation(run))
    with pytest.raises(ValueError, match="rejected by default"):
        table.reconstruct_matched_table([run], tmp_path)


def test_superseded_only_reconstructs_integrity_checked_history_without_promotion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "historical", "XGBoost-6ch")
    monkeypatch.setattr(table, "validate_run_directory", lambda *_: _superseded_validation(run))
    record = table.reconstruct_matched_table([run], tmp_path, superseded_only=True)

    assert record["superseded_protocol_reconstruction_only"] is True
    assert record["evidence_status"] == "superseded_protocol_reconstruction_only"
    assert record["input_validation_statuses"] == ["SUPERSEDED_PROTOCOL"]
    assert record["supersession_contract"] == {
        "all_inputs_require_exact_status": "SUPERSEDED_PROTOCOL",
        "all_inputs_require_integrity_passed_true": True,
        "historical_metrics_may_be_promoted_to_current_evidence": False,
        "replacement_result_claimed": False,
    }
    assert record["method_evidence_statuses"] == {"XGBoost-6ch": "superseded_protocol"}
    assert "validated" not in record["method_evidence_statuses"]["XGBoost-6ch"]
    assert record["sources"][0]["run_directory"] == "historical"
    assert record["sources"][0]["validation"]["run_directory"] == "historical"
    assert "Historical supersession-only reconstruction" in table.table_markdown(record)


@pytest.mark.parametrize(
    "validation",
    [
        {
            "status": "SUPERSEDED_PROTOCOL",
            "integrity_passed": False,
            "publication_evidence_ready": False,
        },
        {
            "status": "VALIDATED",
            "integrity_passed": True,
            "publication_evidence_ready": True,
        },
        {
            "status": "DIAGNOSTIC",
            "integrity_passed": True,
            "publication_evidence_ready": False,
            "diagnostic_contract_passed": True,
        },
        {
            "status": "PROVISIONAL",
            "integrity_passed": True,
            "publication_evidence_ready": False,
        },
    ],
)
def test_superseded_only_rejects_wrong_status_or_failed_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, validation: dict[str, Any]
) -> None:
    run = _fixture(tmp_path, "run", "XGBoost-6ch")
    monkeypatch.setattr(table, "validate_run_directory", lambda *_: validation)
    with pytest.raises(ValueError, match="requires every input"):
        table.reconstruct_matched_table([run], tmp_path, superseded_only=True)


def test_superseded_only_forbids_mixed_current_and_historical_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    historical = _fixture(tmp_path, "historical", "XGBoost-6ch")
    current = _fixture(tmp_path, "current", "RandomForest-6ch")

    def validation(directory: Path, _root: Path) -> dict[str, Any]:
        if directory == historical:
            return _superseded_validation(historical)
        return {
            "status": "VALIDATED",
            "integrity_passed": True,
            "publication_evidence_ready": True,
        }

    monkeypatch.setattr(table, "validate_run_directory", validation)
    with pytest.raises(ValueError, match="requires every input"):
        table.reconstruct_matched_table([historical, current], tmp_path, superseded_only=True)


def test_publication_table_main_wrong_repository_root_creates_no_output(tmp_path: Path) -> None:
    run = _fixture(tmp_path, "run", "XGBoost-6ch")
    output = tmp_path / "must-not-exist"
    with pytest.raises((FileNotFoundError, ValueError)):
        table.main(
            [
                "--repository-root",
                str(tmp_path),
                "--run-directory",
                str(run),
                "--output",
                str(output),
            ]
        )
    assert not output.exists()


def test_publication_table_main_writes_explicit_superseded_only_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "historical", "XGBoost-6ch")
    output = tmp_path / "output"
    monkeypatch.setattr(table, "_assert_executed_repository_root", lambda _root: {})
    monkeypatch.setattr(table, "validate_run_directory", lambda *_: _superseded_validation(run))

    assert (
        table.main(
            [
                "--repository-root",
                str(tmp_path),
                "--superseded-only",
                "--run-directory",
                str(run),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    written = json.loads((output / "table.json").read_text(encoding="utf-8"))
    assert written["superseded_protocol_reconstruction_only"] is True
    assert written["method_evidence_statuses"] == {"XGBoost-6ch": "superseded_protocol"}


def test_table_accepts_only_explicit_contract_validated_diagnostics_without_upgrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = _fixture(tmp_path, "imu-diagnostic", "XGBoost-6ch")
    _rewrite_result(
        run,
        {
            "dataset": {
                "dataset_id": "imu_har_il_v1",
                "channel_lane": "derived-gravity-9ch",
                "boundary_provenance": _boundary_provenance(provider_conditioned=True),
            },
            "artifact_evidence_status": "diagnostic_provider_presegmented_development",
        },
    )
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "integrity_passed": True,
            "publication_evidence_ready": False,
            "publication_evidence_ready_for_unqualified_methods": False,
            "status": "DIAGNOSTIC",
            "diagnostic_contract_passed": True,
            "diagnostic_scope_reasons": ["provider-presegmented synthetic diagnostic"],
        },
    )
    reconstructed = table.reconstruct_matched_table([run], tmp_path)
    assert reconstructed["evidence_status"].startswith("diagnostic_provider_presegmented")
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "publication_evidence_ready": False,
            "status": "DIAGNOSTIC",
            "diagnostic_contract_passed": False,
        },
    )
    with pytest.raises(ValueError, match="not validated"):
        table.reconstruct_matched_table([run], tmp_path)


def test_shared_fog_forest_reference_requires_explicit_opt_in_and_bitwise_parity(
    tmp_path: Path, validated: None
) -> None:
    first = _fixture(tmp_path, "first", "RandomForest-6ch")
    repeated = _fixture(tmp_path, "repeated", "RandomForest-6ch")
    result = table.reconstruct_matched_table(
        [first, repeated], tmp_path, identical_reference_methods=("RandomForest-6ch",)
    )
    assert list(result["statistics"]["methods"]) == ["RandomForest-6ch"]
    assert result["statistics"]["participant_count"] == 3
    assert len(result["sources"]) == 2
    assert len(result["identical_reference_checks"]) == 3
    assert all(
        item["probabilities_bitwise_identical"]
        and not item["counted_as_independent_replication_or_extra_participants"]
        for item in result["identical_reference_checks"]
    )
    changed = _fixture(tmp_path, "changed", "RandomForest-6ch", correct=False)
    with pytest.raises(ValueError, match="duplicate method"):
        table.reconstruct_matched_table(
            [first, changed], tmp_path, identical_reference_methods=("RandomForest-6ch",)
        )


def test_reference_deduplication_cannot_hide_signed_zero_probability_differences(
    tmp_path: Path, validated: None
) -> None:
    first = _fixture(tmp_path, "first", "RandomForest-6ch")
    second = _fixture(tmp_path, "second", "RandomForest-6ch")
    path = second / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["probability__seed-11__RandomForest-6ch"][0, 1] = -0.0
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="duplicate method"):
        table.reconstruct_matched_table(
            [first, second], tmp_path, identical_reference_methods=("RandomForest-6ch",)
        )


def test_no_arbitrary_method_deduplication_allowlist(tmp_path: Path, validated: None) -> None:
    run = _fixture(tmp_path, "run", "XGBoost-6ch")
    with pytest.raises(ValueError, match="only the explicit FoG"):
        table.reconstruct_matched_table(
            [run, run], tmp_path, identical_reference_methods=("XGBoost-6ch",)
        )


def test_diagnostic_control_and_failed_candidate_never_become_passed_primary(
    tmp_path: Path, validated: None
) -> None:
    diagnostic = _fixture(tmp_path, "diagnostic", "PB-RF-D9")
    candidate = _fixture(tmp_path, "candidate", "PB-HPF", correct=False)
    for directory, extra in (
        (diagnostic, {"method_input_lanes": {"PB-RF-D9": "derived-nine-channel diagnostic"}}),
        (candidate, {"advancement_gate": {"all_advancement_gates_passed": False}}),
    ):
        path = directory / "result.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        record.update(extra)
        path.write_text(json.dumps(record), encoding="utf-8")
    result = table.reconstruct_matched_table([diagnostic, candidate], tmp_path)
    assert result["method_evidence_statuses"]["PB-RF-D9"].startswith("diagnostic_")
    assert result["method_evidence_statuses"]["PB-HPF"].endswith("failed_candidate_gate")
    assert (
        result["sources"][1]["retained_advancement_gate"]["all_advancement_gates_passed"] is False
    )
    assert result["descriptive_comparisons_vs_strongest_observed_control"]["control"] == "PB-RF-D9"
    six = next(
        group
        for group in result["descriptive_comparisons_vs_strongest_same_input_and_context_control"]
        if "PB-HPF" in group["methods"]
    )
    assert six["comparisons"]["status"] == "no_applicable_control_in_table"


@pytest.mark.parametrize("name", ["labels", "window_ids"])
def test_table_rejects_rephased_or_relabelled_predictions(
    tmp_path: Path, validated: None, name: str
) -> None:
    left = _fixture(tmp_path, "left", "XGBoost-6ch")
    right = _fixture(tmp_path, "right", "TinyHAR-6ch")
    path = right / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays[name] = np.roll(arrays[name], 1)
    np.savez_compressed(path, **arrays)
    with pytest.raises(ValueError, match="prediction identity"):
        table.reconstruct_matched_table([left, right], tmp_path)


def test_table_carries_runtime_qualification_and_discloses_local_neural_variant(
    tmp_path: Path, validated: None
) -> None:
    run = _fixture(tmp_path, "run", "TinyHAR-6ch")
    (tmp_path / "runtime_termination_failure_example.json").write_text(
        '{"status":"failure"}', encoding="utf-8"
    )
    result = table.reconstruct_matched_table([run], tmp_path)
    assert result["evidence_status"].endswith("with_runtime_qualification")
    assert result["runtime_qualifications"]
    assert "TinyHAR-style-6ch" in table.table_markdown(result)


def test_nested_primary_suite_keeps_enclosing_campaign_failure(
    tmp_path: Path, validated: None
) -> None:
    campaign = tmp_path / "campaign"
    suite = campaign / "primary_suite"
    suite.mkdir(parents=True)
    run = _fixture(suite, "run", "RandomForest-6ch")
    (campaign / "campaign_plan.json").write_text("{}", encoding="utf-8")
    (campaign / "runtime_termination_failure_example.json").write_text(
        '{"status":"failure"}', encoding="utf-8"
    )
    result = table.reconstruct_matched_table([run], tmp_path)
    assert result["evidence_status"].endswith("with_runtime_qualification")
    assert len(result["runtime_qualifications"]) == 1


def test_partial_validation_keeps_context_diagnostic_and_valid_control_separate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        table,
        "validate_run_directory",
        lambda *_: {
            "status": "PARTIALLY_VALIDATED_METHODS",
            "integrity_passed": True,
            "publication_evidence_ready": False,
            "publication_evidence_ready_for_unqualified_methods": True,
            "diagnostic_contract_passed": False,
            "unqualified_method_names": ["RandomForest-6ch"],
            "annotation_selected_context_methods": ["HERA-DG-full"],
        },
    )
    control = _fixture(tmp_path, "base", "RandomForest-6ch")
    context = _fixture(tmp_path, "context", "HERA-DG-full")
    result = table.reconstruct_matched_table([control, context], tmp_path)
    assert result["evidence_status"] == "mixed_validated_and_diagnostic_methods"
    assert "diagnostic" in result["method_evidence_statuses"]["HERA-DG-full"]
    assert "diagnostic" not in result["method_evidence_statuses"]["RandomForest-6ch"]
    markdown = table.table_markdown(result)
    assert "noncausal participant batch; 9 input channels" in markdown
    assert "independent fixed window; 6 input channels" in markdown


def test_unknown_channel_group_is_not_silently_nine_channels(
    tmp_path: Path, validated: None
) -> None:
    control = _fixture(tmp_path, "base", "RandomForest-6ch")
    unknown = _fixture(tmp_path, "unknown", "unrecognized")
    result = table.reconstruct_matched_table([control, unknown], tmp_path)
    assert "unknown input channels" in table.table_markdown(result)


def test_strongest_matched_control_cannot_come_from_another_inference_budget(
    tmp_path: Path, validated: None
) -> None:
    forest = _fixture(tmp_path, "forest", "RandomForest-6ch")
    neural = _fixture(tmp_path, "neural", "TinyHAR-DG", correct=False)
    local = _fixture(tmp_path, "local", "CTGR-DG")
    context = _fixture(tmp_path, "context", "HERA-DG-full")
    result = table.reconstruct_matched_table([forest, neural, local, context], tmp_path)
    groups = result["descriptive_comparisons_vs_strongest_same_input_and_context_control"]
    six = next(group for group in groups if "RandomForest-6ch" in group["methods"])
    nine = next(group for group in groups if "CTGR-DG" in group["methods"])
    batch = next(group for group in groups if "HERA-DG-full" in group["methods"])
    assert six["comparisons"]["control"] == "RandomForest-6ch"
    assert nine["comparisons"]["control"] == "TinyHAR-DG"
    assert nine["comparisons"]["pairs"]["CTGR-DG"]["input_or_inference_contract_differences"] == []
    assert batch["comparisons"]["status"] == "no_applicable_control_in_table"
    assert "HERA-DG-full" not in six["comparisons"]["pairs"]


def _add_materialization_witness(directory: Path, *, different_source: bool) -> None:
    path = directory / "result.json"
    result = json.loads(path.read_text())
    result["dataset"]["preprocessing_audit"] = [
        {
            name: "f" * 64
            for name in (
                "signals_sha256",
                "gravity_sha256",
                "source_timestamps_sha256",
                "timestamps_sha256",
                "candidate_grid_sha256",
            )
        }
    ]
    if different_source:
        result["source_input_manifest"]["files"]["src/inclusive_shift_har/data/external_har.py"] = (
            "e" * 64
        )
    path.write_text(json.dumps(result), encoding="utf-8")


def test_complete_observed_fog_witness_allows_unrelated_loader_source_changes(
    tmp_path: Path, validated: None
) -> None:
    left = _fixture(tmp_path, "left", "RandomForest-6ch")
    right = _fixture(tmp_path, "right", "TinyHAR-6ch")
    _add_materialization_witness(left, different_source=False)
    _add_materialization_witness(right, different_source=True)
    record = table.reconstruct_matched_table([left, right], tmp_path)
    assert record["comparison_contract"]["preprocessing_equivalence"]["equivalence_basis"] == (
        "complete_observed_fog_segment_tensors_and_grids"
    )
    assert len({source["preprocessing_source_sha256"] for source in record["sources"]}) == 2


@pytest.mark.parametrize(
    "mutation",
    ["signals", "gravity", "source_timestamps", "timestamps", "candidate_grid", "missing"],
)
def test_fog_materialization_witness_cannot_conceal_input_changes(
    tmp_path: Path, validated: None, mutation: str
) -> None:
    left = _fixture(tmp_path, "left", "RandomForest-6ch")
    right = _fixture(tmp_path, "right", "TinyHAR-6ch")
    _add_materialization_witness(left, different_source=False)
    _add_materialization_witness(right, different_source=True)
    path = right / "result.json"
    result = json.loads(path.read_text())
    witness = result["dataset"]["preprocessing_audit"][0]
    if mutation == "missing":
        del witness["gravity_sha256"]
    else:
        witness[f"{mutation}_sha256"] = "a" * 64
    path.write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match="non-comparable"):
        table.reconstruct_matched_table([left, right], tmp_path)
