from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.artifacts.research_provenance import FOG_STAR_FULL_PARTICIPANT_ROSTER
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import publication_statistical_supplement as supplement
from inclusive_shift_har.experiments import publication_table
from inclusive_shift_har.manifests.canonical import sha256_file


def _probability(predicted: list[int], class_count: int) -> np.ndarray[Any, Any]:
    return np.eye(class_count, dtype=np.float64)[np.asarray(predicted, dtype=np.int64)]


def test_participant_balanced_class_metrics_and_normalized_confusion() -> None:
    labels = np.asarray([0, 1, 0, 2, 2, 2], dtype=np.int64)
    participants = np.asarray(["p1", "p1", "p2", "p2", "p2", "p2"])
    result = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("mobility", "sitting", "standing"),
        {11: {"candidate": _probability([0, 0, 2, 2, 0, 2], 3)}},
        bootstrap_replicates=200,
    )

    aggregate = result["methods"]["candidate"]["seed_averaged_participant_cluster_statistics"]
    sitting_precision = aggregate["participant_balanced_per_class"]["precision"]["sitting"]
    sitting_recall = aggregate["participant_balanced_per_class"]["recall"]["sitting"]
    standing_f1 = aggregate["participant_balanced_per_class"]["f1"]["standing"]
    assert sitting_precision["eligible_participant_n"] == 0
    assert sitting_precision["eligible_participant_sensitivity"] is None
    assert sitting_recall["eligible_participant_n"] == 1
    assert sitting_recall["ineligible_participants"] == ["p2"]
    assert standing_f1["eligible_participant_n"] == 1
    assert standing_f1["all_participants_zero_division_0"]["mean"] == pytest.approx(1 / 3)

    confusion = aggregate["participant_normalized_confusion"]
    assert confusion["mean_matrix"][0] == pytest.approx([0.5, 0.0, 0.5])
    assert confusion["eligible_participant_n_by_true_class"] == {
        "mobility": 2,
        "sitting": 1,
        "standing": 1,
    }
    assert result["class_support"]["sitting"]["participants_with_true_support"] == 1
    complete = aggregate["all_declared_classes_supported_participant_sensitivity"]
    assert complete["eligible_participant_n"] == 0
    assert complete["summary"] is None


def test_participant_mean_calibration_is_distinct_from_pooled_window_diagnostic() -> None:
    labels = np.zeros(4, dtype=np.int64)
    participants = np.asarray(["p1", "p2", "p2", "p2"])
    probability = np.asarray([[0.9, 0.1], [0.6, 0.4], [0.6, 0.4], [0.6, 0.4]], dtype=np.float64)
    result = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("still", "moving"),
        {11: {"model": probability}, 23: {"model": probability}},
        bootstrap_replicates=200,
    )
    method = result["methods"]["model"]
    aggregate = method["seed_averaged_participant_cluster_statistics"]
    participant_nll = aggregate["participant_mean_calibration"]["negative_log_likelihood"]["mean"]
    pooled_nll = method["participant_seed_details"]["11"]["pooled_window_diagnostics"][
        "calibration"
    ]["negative_log_likelihood"]
    assert participant_nll == pytest.approx((-np.log(0.9) - np.log(0.6)) / 2)
    assert pooled_nll == pytest.approx((-np.log(0.9) - 3 * np.log(0.6)) / 4)
    assert participant_nll != pytest.approx(pooled_nll)
    assert (
        "descriptive pooled-window"
        in method["participant_seed_details"]["11"]["pooled_window_diagnostics"][
            "inferential_status"
        ]
    )
    assert set(method["participant_seed_details"]) == {"11", "23"}
    assert "sample-size-sensitive" in result["estimand_definitions"]["participant_mean_calibration"]


def test_bootstrap_is_deterministic_and_invalid_probabilities_are_rejected() -> None:
    labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
    participants = np.asarray(["p1", "p1", "p2", "p2"])
    probabilities = {11: {"model": _probability([0, 1, 1, 1], 2)}}
    first = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("a", "b"),
        probabilities,
        bootstrap_replicates=300,
    )
    second = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("a", "b"),
        probabilities,
        bootstrap_replicates=300,
    )
    first_primary = first["methods"]["model"]["seed_averaged_participant_cluster_statistics"][
        "primary_fixed_class_macro_f1"
    ]
    second_primary = second["methods"]["model"]["seed_averaged_participant_cluster_statistics"][
        "primary_fixed_class_macro_f1"
    ]
    assert first_primary == second_primary

    invalid = np.asarray([[0.8, 0.8], [0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    with pytest.raises(ValueError, match="invalid probabilities"):
        supplement.compute_participant_statistical_supplement(
            labels,
            participants,
            ("a", "b"),
            {11: {"model": invalid}},
            bootstrap_replicates=10,
        )


def test_fixed_cohort_seed_mean_retains_zero_when_precision_is_undefined() -> None:
    labels = np.asarray([1, 0], dtype=np.int64)
    participants = np.asarray(["p1", "p2"])
    result = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("a", "b"),
        {
            11: {"model": _probability([1, 0], 2)},
            23: {"model": _probability([0, 0], 2)},
        },
        bootstrap_replicates=100,
    )
    precision = result["methods"]["model"]["seed_averaged_participant_cluster_statistics"][
        "participant_balanced_per_class"
    ]["precision"]["b"]
    assert precision["all_participants_zero_division_0"]["mean"] == pytest.approx(0.25)
    assert precision["eligible_participant_sensitivity"]["mean"] == pytest.approx(1.0)
    assert precision["eligible_participant_n"] == 1


def test_present_class_and_all_classes_supported_sensitivities_are_explicit() -> None:
    labels = np.asarray([0, 1, 2, 0, 1], dtype=np.int64)
    participants = np.asarray(["p1", "p1", "p1", "p2", "p2"])
    result = supplement.compute_participant_statistical_supplement(
        labels,
        participants,
        ("a", "b", "c"),
        {11: {"model": _probability(labels.tolist(), 3)}},
        bootstrap_replicates=100,
    )

    assert result["participants_with_all_declared_classes"] == 1
    assert result["participants_missing_at_least_one_declared_class"] == ["p2"]
    assert result["perfect_prediction_fixed_class_mean_ceiling"] == pytest.approx(5 / 6)
    aggregate = result["methods"]["model"]["seed_averaged_participant_cluster_statistics"]
    assert aggregate["primary_fixed_class_macro_f1"]["mean"] == pytest.approx(5 / 6)
    present = aggregate["participant_present_true_class_macro_f1_sensitivity"]
    assert present["mean"] == pytest.approx(1.0)
    assert present["participant_values"] == {"p1": 1.0, "p2": 1.0}
    complete = aggregate["all_declared_classes_supported_participant_sensitivity"]
    assert complete["eligible_participant_n"] == 1
    assert complete["eligible_participants"] == ["p1"]
    assert complete["ineligible_participants"] == ["p2"]
    assert complete["summary"]["mean"] == pytest.approx(1.0)
    assert (
        "secondary missing-class"
        in result["estimand_definitions"]["participant_present_true_class_macro_f1_sensitivity"]
    )


def test_paired_comparisons_align_explicit_participant_ids() -> None:
    replicates = 200
    seed = 20260905
    control = np.asarray([0.0, 1.0], dtype=np.float64)
    candidate = np.asarray([1.0, 0.0], dtype=np.float64)
    delta = candidate - control
    draws = np.random.default_rng(seed).integers(0, 2, size=(replicates, 2), dtype=np.int64)
    mean_ci = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975]).tolist()
    tail_ci = np.quantile(
        np.sort(candidate[draws], axis=1)[:, :1].mean(axis=1)
        - np.sort(control[draws], axis=1)[:, :1].mean(axis=1),
        [0.025, 0.975],
    ).tolist()
    statistics: dict[str, Any] = {
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "methods": {
            "control": {
                "seed_averaged_participant_cluster_statistics": {
                    "primary_fixed_class_macro_f1": {"participant_values": {"p1": 0.0, "p2": 1.0}}
                }
            },
            "candidate": {
                "seed_averaged_participant_cluster_statistics": {
                    "primary_fixed_class_macro_f1": {
                        # Reversed insertion order must not reverse scientific pairing.
                        "participant_values": {"p2": 0.0, "p1": 1.0}
                    }
                }
            },
        },
    }
    source_groups: list[dict[str, Any]] = [
        {
            "input_and_context_budget": {"input_channel_count": 6},
            "comparisons": {
                "control": "control",
                "selection": "fixture",
                "pairs": {
                    "candidate": {
                        "mean_difference": 0.0,
                        "paired_participant_bootstrap_95_percent_ci": mean_ci,
                        "bottom_30_percent_difference_95_percent_ci": tail_ci,
                        "rescue_count": 1,
                        "harm_count": 1,
                        "tie_count": 0,
                    }
                },
            },
        }
    ]

    rebuilt = supplement._paired_comparison_groups(source_groups, statistics)
    pair = rebuilt[0]["comparisons"]["pairs"]["candidate"]
    assert pair["participant_delta_values"] == {"p1": 1.0, "p2": -1.0}
    assert pair["rescue_count"] == pair["harm_count"] == 1
    assert pair["tie_count"] == 0
    assert pair["difference_direction"] == "candidate_minus_control"
    assert pair["source_comparison_cross_check_passed"] is True

    source_groups[0]["comparisons"]["pairs"]["candidate"]["tie_count"] = 2
    with pytest.raises(ValueError, match="tie_count mismatch"):
        supplement._paired_comparison_groups(source_groups, statistics)


def _run_fixture(root: Path, method: str) -> Path:
    directory = root / method.lower().replace("-", "_")
    directory.mkdir()
    labels = np.tile(np.arange(3, dtype=np.int64), 3)
    participants = np.repeat(np.asarray(["p1", "p2", "p3"]), 3)
    probability = np.eye(3, dtype=np.float64)[labels]
    primary, prediction = seed_evidence(
        ParticipantMetricInputs(
            "fog_star_v3", ("mobility", "sitting", "standing"), labels, participants
        ),
        {seed: {method: probability} for seed in (11, 23, 47)},
    )
    result: dict[str, Any] = {
        "dataset": {
            "dataset_id": "fog_star_v3",
            "channel_lane": "six-channel",
            "boundary_provenance": {
                "protocol_id": "external-har-boundary-provenance-v1",
                "repository_signal_grid_annotation_independent": True,
                "provider_upstream_annotation_conditioned": False,
            },
            "participant_count": 22,
            "participant_partition_plan": {
                "participant_roster": list(FOG_STAR_FULL_PARTICIPANT_ROSTER)
            },
            "participant_partition_observation": {
                "planned_participant_count": 22,
                "observed_window_participant_count": 22,
                "participants_without_retained_windows": [],
            },
        },
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
        window_ids=np.asarray([f"w{index}" for index in range(labels.size)]),
        **{f"probability__{key}": value for key, value in prediction.items()},  # type: ignore[arg-type]
    )
    return directory


def _table_source_fixture(run: Path, repository_root: Path) -> dict[str, Any]:
    return {
        "run_directory": publication_table._repository_relative_path(run, repository_root),
        "result_sha256": sha256_file(run / "result.json"),
        "predictions_sha256": sha256_file(run / "predictions.npz"),
        "data_audit_sha256": sha256_file(run / "data_audit.json"),
    }


def test_reconstruction_cross_checks_table_and_writes_create_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        publication_table,
        "validate_run_directory",
        lambda *_: {
            "status": "VALIDATED",
            "integrity_passed": True,
            "publication_evidence_ready": True,
            "publication_evidence_ready_for_unqualified_methods": False,
            "diagnostic_contract_passed": False,
        },
    )
    monkeypatch.setattr(
        supplement,
        "_git_state",
        lambda *_: {"commit": "e" * 40, "worktree_dirty": False, "status_entries": []},
    )
    monkeypatch.setattr(
        supplement,
        "_source_input_manifest",
        lambda *_: {"manifest_sha256": "f" * 64, "files": {}},
    )
    run = _run_fixture(tmp_path, "RandomForest-6ch")
    output = tmp_path / "supplement"
    record = supplement.write_statistical_supplement([run], tmp_path, output)

    assert (output / "statistical_supplement.json").is_file()
    assert (output / "statistical_supplement.md").is_file()
    assert record["reconstruction_only_no_training_or_raw_signal_access"]
    assert record["independent_confirmation_or_sota_claim_allowed"] is False
    assert (
        record["scientific_status_inheritance"][
            "reporting_reconstruction_can_upgrade_scientific_status"
        ]
        is False
    )
    assert record["missing_data_limitation"]["imputation_performed"] is False
    assert "do not expose" in record["compute_and_model_size_limitation"]
    comparison = record["strongest_same_input_and_context_budget_comparisons"][0]["comparisons"]
    assert comparison["control"] == "RandomForest-6ch"
    assert comparison["pairs"]["RandomForest-6ch"]["tie_count"] == 3
    assert comparison["pairs"]["RandomForest-6ch"]["source_comparison_cross_check_passed"]
    assert comparison["pairs"]["RandomForest-6ch"]["difference_direction"] == (
        "candidate_minus_control"
    )
    source = record["prediction_sources"][0]
    assert source["run_directory"] == {
        "path_kind": "repository_relative",
        "path": "randomforest_6ch",
    }
    assert source["prediction_artifact"]["path_kind"] == "run_directory_relative"
    assert source["prediction_artifact"]["path"] == "predictions.npz"
    assert source["matched_table_source_hash_cross_check_passed"] is True
    assert source["prediction_loaded_from_hashed_byte_snapshot"] is True
    assert "predictions_path" not in source
    assert record["validated_run_sources"][0]["run_directory"] == source["run_directory"]
    assert record["validated_run_sources"][0]["supplement_source_hash_cross_check_passed"] is True
    assert "Participant-balanced" in supplement.supplement_markdown(record)

    before = (output / "statistical_supplement.json").read_bytes()
    with pytest.raises(FileExistsError):
        supplement.write_statistical_supplement([run], tmp_path, output)
    assert (output / "statistical_supplement.json").read_bytes() == before


@pytest.mark.parametrize(
    ("artifact_name", "expected_message"),
    [
        ("result.json", "result SHA-256 differs"),
        ("predictions.npz", "prediction SHA-256 differs"),
        ("data_audit.json", "data-audit SHA-256 differs"),
    ],
)
def test_prediction_loader_rejects_artifacts_changed_after_table_reconstruction(
    tmp_path: Path, artifact_name: str, expected_message: str
) -> None:
    run = _run_fixture(tmp_path, "RandomForest-6ch")
    table_source = _table_source_fixture(run, tmp_path)
    (run / artifact_name).write_bytes(b"changed after matched-table reconstruction")

    with pytest.raises(ValueError, match=expected_message):
        supplement._load_prediction_evidence([run], [11, 23, 47], (), [table_source], tmp_path)


@pytest.mark.parametrize(
    "unsafe",
    ["../predictions.npz", "nested/predictions.npz", "C:/predictions.npz", "\\share\\x"],
)
def test_prediction_loader_rejects_non_direct_artifact_paths(tmp_path: Path, unsafe: str) -> None:
    run = _run_fixture(tmp_path, "RandomForest-6ch")
    result_path = run / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["prediction_artifact"]["path"] = unsafe
    result_path.write_text(json.dumps(result), encoding="utf-8")
    table_source = _table_source_fixture(run, tmp_path)

    with pytest.raises(ValueError, match="one direct run-directory file"):
        supplement._load_prediction_evidence([run], [11, 23, 47], (), [table_source], tmp_path)


def test_external_run_locator_is_explicitly_typed(tmp_path: Path) -> None:
    repository_root = tmp_path / "repository"
    repository_root.mkdir()
    run = _run_fixture(tmp_path, "RandomForest-6ch")
    table_source = _table_source_fixture(run, repository_root)

    _identity, _probabilities, sources, checked_sources = supplement._load_prediction_evidence(
        [run], [11, 23, 47], (), [table_source], repository_root
    )

    expected = {"path_kind": "external_absolute", "path": str(run.resolve())}
    assert sources[0]["run_directory"] == expected
    assert checked_sources[0]["run_directory"] == expected
