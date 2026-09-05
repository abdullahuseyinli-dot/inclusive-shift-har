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
