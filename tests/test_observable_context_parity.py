from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.inference_contracts import (
    OBSERVABLE_CONTEXT_PROTOCOL,
    OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
)
from inclusive_shift_har.experiments import observable_context_parity as parity
from inclusive_shift_har.manifests.canonical import sha256_file


def _pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    def validation(directory: Path, *_args: Any) -> dict[str, Any]:
        if directory.name == "old":
            return {
                "status": "SUPERSEDED_PROTOCOL",
                "integrity_passed": True,
                "publication_evidence_ready": False,
                "publication_evidence_ready_for_unqualified_methods": False,
                "diagnostic_contract_passed": False,
            }
        return {
            "status": "VALIDATED",
            "integrity_passed": True,
            "publication_evidence_ready": True,
            "scientific_contract_passed": True,
            "diagnostic_contract_passed": False,
        }

    monkeypatch.setattr(parity, "validate_run_directory", validation)
    monkeypatch.setattr(parity, "_source_input_manifest", lambda *_: {})
    monkeypatch.setattr(parity, "_git_state", lambda *_: {})
    for name in ("old", "new"):
        directory = tmp_path / name
        directory.mkdir()
        segment: dict[str, Any] = {
            "protocol_id": "external-har-session-grid-v3",
            "signals_sha256": "4" * 64,
            "gravity_sha256": "5" * 64,
            "candidate_grid_sha256": "6" * 64,
            "candidate_start_samples": [0],
            "admitted_candidate_indices": [0],
            "excluded_candidate_indices": {},
        }
        pool_arrays = {
            "signals": "7" * 64,
            "gravity": "8" * 64,
            "participant_ids": "9" * 64,
            "session_ids": "a" * 64,
            "trial_ids": "b" * 64,
            "window_ids": "c" * 64,
        }
        dataset: dict[str, Any] = {
            "dataset_id": "fog_star_v3",
            "window_count": 3,
            "preprocessing_audit": [segment],
            "observable_candidate_pool": {
                "protocol_id": OBSERVABLE_CONTEXT_PROTOCOL,
                "annotation_fields_present": False,
                "window_count": 3,
                "participant_window_counts": {"p1": 3},
                "arrays": pool_arrays,
            },
        }
        if name == "old":
            segment["annotation_dependency"] = False
        else:
            segment["within_declared_segment_transform_annotation_dependency"] = False
            segment["segment_boundary_annotation_conditioned"] = False
            pool_arrays["nine_channel_signals"] = "d" * 64
            dataset.update(
                {
                    "boundary_provenance": {
                        "repository_signal_grid_annotation_independent": True,
                        "provider_upstream_annotation_conditioned": False,
                    },
                    "participant_partition_plan": {"protocol_id": "synthetic-plan"},
                    "participant_partition_observation": {"planned_participant_count": 1},
                    "retained_array_hashes": {
                        field: str(index) * 64
                        for index, field in enumerate(sorted(parity._RETAINED_ARRAY_NAMES), start=1)
                    },
                }
            )
        result: dict[str, Any] = {
            "experiment_id": "cross-dataset-har-rnd-v1",
            "dataset": dataset,
            "seeds": [11, 23, 47],
            "primary_seed_averaged": {
                "methods": {
                    method: {"mean_participant_macro_f1": 1.0}
                    for method in ("RandomForest-6ch", "HERA-DG-strict", "HERA-DG-full")
                }
            },
        }
        if name == "new":
            result["observable_context_protocol"] = OBSERVABLE_CONTEXT_PROTOCOL
            result["observable_context_training_protocol"] = OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (directory / "data_audit.json").write_text(
            json.dumps({"dataset": dataset}), encoding="utf-8"
        )
        np.savez_compressed(
            directory / "predictions.npz",
            labels=np.arange(3),
            participant_ids=np.array(["p1"] * 3),
            session_ids=np.array(["s"] * 3),
            trial_ids=np.array(["t"] * 3),
            window_ids=np.array(["w0", "w1", "w2"]),
            **{  # type: ignore[arg-type]
                f"probability__seed-{seed}__{method}": np.eye(3)
                for seed in (11, 23, 47)
                for method in ("RandomForest-6ch", "HERA-DG-strict", "HERA-DG-full")
            },
        )
    old = tmp_path / "old"
    old_result = json.loads((old / "result.json").read_text(encoding="utf-8"))
    old_result.update(
        {
            "observable_context_protocol": OBSERVABLE_CONTEXT_PROTOCOL,
            "result_payload_sha256_before_serialization": "1" * 64,
        }
    )
    (old / "result.json").write_text(json.dumps(old_result), encoding="utf-8")
    old_audit = {
        "dataset": old_result["dataset"],
        "git_at_launch": {
            "commit": "2" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
        "source_input_manifest": {
            "protocol_id": "external-har-session-grid-v3",
            "manifest_sha256": "3" * 64,
        },
    }
    (old / "data_audit.json").write_text(json.dumps(old_audit), encoding="utf-8")
    pin = {
        "launch_commit": "2" * 40,
        "dataset_id": "fog_star_v3",
        "observable_context_protocol": OBSERVABLE_CONTEXT_PROTOCOL,
        "source_manifest_protocol_id": "external-har-session-grid-v3",
        "source_manifest_sha256": "3" * 64,
        "result_payload_sha256_before_serialization": "1" * 64,
        "artifact_sha256": {
            name: sha256_file(old / name)
            for name in ("result.json", "predictions.npz", "data_audit.json")
        },
    }
    monkeypatch.setattr(parity, "_historical_reference_pin", lambda *_args: pin)
    return tmp_path / "old", tmp_path / "new"


@pytest.mark.parametrize("mutation", ["none", "base", "context", "identities"])
def test_parity_gates_unaffected_methods_and_preserves_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    old, new = _pair(tmp_path, monkeypatch)
    path = new / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if mutation in {"base", "context"}:
        method = "RandomForest-6ch" if mutation == "base" else "HERA-DG-full"
        arrays[f"probability__seed-11__{method}"] = np.roll(np.eye(3), 1, axis=1)
    elif mutation == "identities":
        arrays["window_ids"] = np.roll(arrays["window_ids"], 1)
    np.savez_compressed(path, **arrays)
    if mutation == "identities":
        with pytest.raises(ValueError, match="identities changed"):
            parity.compare_context_replacement(old, new, tmp_path)
    else:
        result = parity.compare_context_replacement(old, new, tmp_path)
        assert (result["status"] == "PARITY_PASSED") is (mutation != "base")
        assert result["methods"]["HERA-DG-strict"]["parity_passed"] is None
        assert result["methods"]["HERA-DG-full"]["parity_passed"] is None
        assert not result["better_score_selection_allowed"]
        assert result["sources"][0]["directory"] == {
            "path_kind": "repository_relative",
            "path": "old",
        }


@pytest.mark.parametrize("mutation", ["reference_validation", "reference_hash", "replacement"])
def test_parity_rejects_unpinned_or_not_current_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    old, new = _pair(tmp_path, monkeypatch)
    original_validate = parity.__dict__["validate_run_directory"]
    if mutation == "reference_validation":
        monkeypatch.setattr(
            parity,
            "validate_run_directory",
            lambda directory, root: (
                {"status": "VALIDATED", "integrity_passed": True}
                if directory == old
                else original_validate(directory, root)
            ),
        )
        match = "superseded protocol state"
    elif mutation == "reference_hash":
        with (old / "result.json").open("a", encoding="utf-8") as stream:
            stream.write("\n")
        match = "frozen pin"
    else:
        monkeypatch.setattr(
            parity,
            "validate_run_directory",
            lambda directory, root: (
                {
                    "status": "SUPERSEDED_PROTOCOL",
                    "integrity_passed": True,
                    "publication_evidence_ready": False,
                    "scientific_contract_passed": False,
                    "diagnostic_contract_passed": False,
                }
                if directory == new
                else original_validate(directory, root)
            ),
        )
        match = "current, fully validated"
    with pytest.raises(ValueError, match=match):
        parity.compare_context_replacement(old, new, tmp_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("candidate_signal", "signals, grid or preprocessing differs"),
        ("candidate_grid", "signals, grid or preprocessing differs"),
        ("unsafe_segment_boundary", "safe-key migration"),
        ("missing_retained_witness", "retained-array witness"),
    ),
)
def test_parity_allows_only_the_predeclared_audit_schema_migration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    old, new = _pair(tmp_path, monkeypatch)
    result = json.loads((new / "result.json").read_text(encoding="utf-8"))
    dataset = result["dataset"]
    if mutation == "candidate_signal":
        dataset["observable_candidate_pool"]["arrays"]["signals"] = "e" * 64
    elif mutation == "candidate_grid":
        dataset["preprocessing_audit"][0]["candidate_start_samples"] = [128]
    elif mutation == "unsafe_segment_boundary":
        dataset["preprocessing_audit"][0]["segment_boundary_annotation_conditioned"] = True
    else:
        dataset.pop("retained_array_hashes")
    (new / "result.json").write_text(json.dumps(result), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        parity.compare_context_replacement(old, new, tmp_path)
