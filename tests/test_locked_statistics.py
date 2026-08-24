from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.locked_statistics import (
    LockedStatisticsError,
    aggregate_locked_target_statistics,
)
from inclusive_shift_har.evaluation.locked_target import LOCKED_TARGET_EVIDENCE_STATUS
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

CLASSES = ("mobility", "sitting")
SEEDS = (11, 23)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _probabilities(model_id: str) -> np.ndarray:
    if model_id == "candidate":
        predicted = np.asarray([0, 1, 0, 1, 0, 1])
    elif model_id == "reference-a":
        predicted = np.asarray([0, 1, 0, 1, 1, 0])
    else:
        predicted = np.asarray([1, 0, 0, 1, 1, 0])
    probabilities = np.full((6, 2), 0.1, dtype=np.float64)
    probabilities[np.arange(6), predicted] = 0.9
    return probabilities


def _fixture(tmp_path: Path) -> Path:
    labels = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    participants = ("11", "11", "12", "12", "13", "13")
    window_ids = tuple(f"target-window-{index}" for index in range(6))
    result_entries: list[dict[str, Any]] = []
    for model_id in ("candidate", "reference-a", "reference-b"):
        for seed in SEEDS:
            probabilities = _probabilities(model_id)
            logits = np.log(probabilities)
            predictions = probabilities.argmax(axis=1).astype(np.int64)
            array_path = tmp_path / "arrays" / f"{model_id}--seed-{seed}.npz"
            array_path.parent.mkdir(parents=True, exist_ok=True)
            with array_path.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    evidence_status=np.asarray([LOCKED_TARGET_EVIDENCE_STATUS]),
                    window_ids=np.asarray(window_ids),
                    participant_ids=np.asarray(participants),
                    true_labels=labels,
                    logits=logits,
                    uncalibrated_probabilities=probabilities,
                    calibrated_probabilities=probabilities,
                    predicted_labels=predictions,
                )
            report = classification_report(
                labels,
                probabilities,
                participants,
                class_names=CLASSES,
            )
            record: dict[str, Any] = {
                "schema_version": "1.0.0",
                "record_kind": "locked_target_per_seed_result",
                "status": "complete_create_only",
                "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
                "target_opening_number": 1,
                "model_id": model_id,
                "seed": seed,
                "class_names": list(CLASSES),
                "prediction_array": {
                    "path": array_path.relative_to(tmp_path).as_posix(),
                    "sha256": sha256_file(array_path),
                    "format": "npz",
                },
                "participant_level_report": report,
                "target_information_used_for_model_selection": False,
            }
            record["record_sha256"] = canonical_json_sha256(record)
            record_path = tmp_path / "records" / f"{model_id}--seed-{seed}.json"
            _write_json(record_path, record)
            result_entries.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "array_path": array_path.relative_to(tmp_path).as_posix(),
                    "array_sha256": sha256_file(array_path),
                    "record_path": record_path.relative_to(tmp_path).as_posix(),
                    "record_file_sha256": sha256_file(record_path),
                    "record_sha256": record["record_sha256"],
                }
            )
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "target_seal_id": "e" * 64,
        "final_freeze_inventory_sha256": "f" * 64,
        "model_seed_result_count": len(result_entries),
        "class_names": list(CLASSES),
        "results": result_entries,
        "target_information_used_for_model_selection": False,
    }
    index["record_sha256"] = canonical_json_sha256(index)
    index_path = tmp_path / "locked-index.json"
    _write_json(index_path, index)
    return index_path


def test_locked_statistics_use_participants_and_predeclared_model_family(tmp_path: Path) -> None:
    index_path = _fixture(tmp_path)
    result = aggregate_locked_target_statistics(
        index_path,
        output_root=tmp_path,
        destination="statistics.json",
        required_seed_order=SEEDS,
        candidate_model_id="candidate",
        eligible_reference_model_ids=("reference-a", "reference-b"),
        source_noninferiority_gate_passed=True,
        analysis_plan_sha256="a" * 64,
        bootstrap_resamples=100,
        bootstrap_seed=7,
    )
    assert result["hypothesis_decision"]["supported"] is True
    assert result["participant_count"] == 3
    assert len(result["candidate_comparisons"]) == 2
    assert (
        result["models"][0]["participant_seed_averaged"]["bootstrap_mean_macro_f1"]["cluster_unit"]
        == "participant"
    )
    assert (tmp_path / "statistics.json").is_file()


def test_locked_statistics_reject_array_mutation(tmp_path: Path) -> None:
    index_path = _fixture(tmp_path)
    array_path = next((tmp_path / "arrays").glob("*.npz"))
    array_path.write_bytes(array_path.read_bytes() + b"retained mutation")
    with pytest.raises(LockedStatisticsError, match="hash mismatch"):
        aggregate_locked_target_statistics(
            index_path,
            output_root=tmp_path,
            destination="statistics.json",
            required_seed_order=SEEDS,
            candidate_model_id="candidate",
            eligible_reference_model_ids=("reference-a", "reference-b"),
            source_noninferiority_gate_passed=True,
            analysis_plan_sha256="a" * 64,
            bootstrap_resamples=100,
        )
