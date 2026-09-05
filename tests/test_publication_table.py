from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import publication_table as table


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
        "dataset": {"dataset_id": "fog_star_v3", "channel_lane": "derived-gravity-9ch"},
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
        table, "validate_run_directory", lambda *_: {"publication_evidence_ready": True}
    )


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
    with pytest.raises(ValueError, match=r"non-comparable|three frozen seeds"):
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
            "publication_evidence_ready": False,
            "publication_evidence_ready_for_unqualified_methods": True,
            "unqualified_method_names": ["RandomForest-6ch"],
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
