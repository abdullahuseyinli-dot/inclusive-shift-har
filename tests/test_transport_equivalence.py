from __future__ import annotations

import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.artifacts.research_provenance import (
    HAR_PMD_FULL_PARTICIPANT_ROSTER,
)
from inclusive_shift_har.data import provider_copy
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments import publication_table as table
from inclusive_shift_har.experiments import transport_equivalence as transport
from inclusive_shift_har.manifests.canonical import canonical_json_sha256


@pytest.fixture
def fixture(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    # Synthetic values only. Preserve the real 120-person scientific cohort contract.
    participant_count = len(HAR_PMD_FULL_PARTICIPANT_ROSTER)
    window_count = participant_count * 5
    monkeypatch.setattr(transport, "_PEOPLE", participant_count)
    monkeypatch.setattr(transport, "_WINDOWS", window_count)
    monkeypatch.setattr(transport, "_manifest_commit_errors", lambda *_: [])
    blob = b"synthetic immutable old source; not an experimental implementation"
    monkeypatch.setattr(subprocess, "check_output", lambda *_args, **_kwargs: blob)
    identities = {
        "labels": np.tile(np.arange(5), participant_count),
        "participant_ids": np.repeat(HAR_PMD_FULL_PARTICIPANT_ROSTER, 5),
        "session_ids": np.repeat(
            [f"harpmd-session:{person:03d}" for person in range(1, participant_count + 1)],
            5,
        ),
        "trial_ids": np.array([f"t{i}" for i in range(window_count)]),
        "window_ids": np.array([f"w{i}" for i in range(window_count)]),
    }
    fields = ("signals", "gravity", *identities)
    classes = ("stationary", "walking", "walker", "crutches", "manual_wheelchair")
    people: list[dict[str, Any]] = [
        {
            "participant": str(person),
            "windows": 5,
            "bitwise_equal": True,
            "arrays_sha256": {name: "a" * 64 for name in fields},
            "source_member_sha256": {
                f"synthetic/{person}/{trial}.csv": "b" * 64 for trial in range(10)
            },
            "class_window_counts": dict.fromkeys(classes, 1),
        }
        for person in range(1, participant_count + 1)
    ]
    storage = {
        "protocol_id": provider_copy.HAR_PMD_PROVIDER_COPY_PROTOCOL,
        "dataset_id": "har_pmd_v1",
        "provider_record": 7939223,
        "provider_version": "2",
        "source_url": provider_copy.HAR_PMD_ARCHIVE_URL,
        "archive_size_bytes": provider_copy.HAR_PMD_ARCHIVE_SIZE,
        "archive_sha256": provider_copy.HAR_PMD_ARCHIVE_SHA256,
        "archive_md5": provider_copy.HAR_PMD_ARCHIVE_MD5,
        "digest_verified": True,
        "license": "CC-BY-4.0",
        "raw_local_mirror": True,
        "path": "not_opened_synthetic.zip",
    }
    files = {transport._LOADER: "c" * 64}
    witness: dict[str, Any] = {
        "status": "ALL_PARTICIPANT_TRANSPORT_PARITY_PASSED",
        "dataset_id": "har_pmd_v1",
        "archive_sha256": provider_copy.HAR_PMD_ARCHIVE_SHA256,
        "old_processing_source_commit": transport._OLD_COMMIT,
        "old_processing_source_sha256": hashlib.sha256(blob).hexdigest(),
        "new_processing_source_sha256": files[transport._LOADER],
        "script_sha256": transport._PROBE_SHA256,
        "models_fit": 0,
        "annotation_or_value_mutations": 0,
        "performance_scores_computed": False,
        "participant_count": participant_count,
        "window_count": window_count,
        "purpose": "transport_only_materialization_parity_no_model_fitting",
        "global_array_digest_convention": "C-order raw bytes concatenated in numeric participant order; dtype and global row count recorded separately; not the shape-prefixed _array_sha256 convention",
        "analysis_git": {"commit": "d" * 40, "worktree_dirty": False, "status_entries": []},
        "analysis_source_input_manifest": {
            "files": files,
            "manifest_sha256": canonical_json_sha256(files),
        },
        "source_storage_audit": storage,
        "participants": people,
        "array_dtypes": {
            "signals": "<f4",
            "gravity": "<f4",
            **{name: value.dtype.str for name, value in identities.items()},
        },
        "ordered_full_population_raw_array_byte_sha256": {
            "signals": "e" * 64,
            "gravity": "f" * 64,
            **{
                name: hashlib.sha256(value.tobytes()).hexdigest()
                for name, value in identities.items()
            },
        },
    }
    witness["record_sha256"] = canonical_json_sha256(witness)
    result = {
        "dataset": {
            "dataset_id": "har_pmd_v1",
            "participant_count": participant_count,
            "window_count": window_count,
            "class_window_counts": dict.fromkeys(classes, participant_count),
            "sampling_rate_hz": 50,
            "raw_local_mirror": False,
            "boundary_provenance": {
                "protocol_id": "external-har-boundary-provenance-v1",
                "repository_signal_grid_annotation_independent": True,
                "provider_upstream_annotation_conditioned": False,
            },
            "participant_partition_plan": {
                "participant_roster": list(HAR_PMD_FULL_PARTICIPANT_ROSTER)
            },
            "participant_partition_observation": {
                "planned_participant_count": participant_count,
                "observed_window_participant_count": participant_count,
                "participants_without_retained_windows": [],
            },
        },
        "artifact_evidence_status": "validated_stress_test",
        "source_input_manifest": {
            "files": {transport._LOADER: witness["old_processing_source_sha256"]}
        },
    }
    audit = {
        "source_receipts": [
            {"member": member, "computed_sha256": digest}
            for person in people
            for member, digest in person["source_member_sha256"].items()
        ]
    }
    return witness, result, audit, identities


def _write(path: Path, value: dict[str, Any]) -> None:
    value["record_sha256"] = canonical_json_sha256(
        {key: item for key, item in value.items() if key != "record_sha256"}
    )
    path.write_text(json.dumps(value), encoding="utf-8")


def test_witness_allows_only_storage_normalisation_after_member_and_identity_checks(
    tmp_path: Path, fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    witness, original, audit, identities = fixture
    path = tmp_path / "witness.json"
    _write(path, witness)
    checked = transport.load_transport_witness(path, tmp_path)
    mirror = deepcopy(original)
    mirror["dataset"]["raw_local_mirror"] = True
    mirror["dataset"]["source_storage_audit"] = witness["source_storage_audit"]
    mirror["source_input_manifest"]["files"][transport._LOADER] = witness[
        "new_processing_source_sha256"
    ]
    left = transport.run_transport_contract(original, audit, checked, identities)
    right = transport.run_transport_contract(mirror, audit, checked, identities)
    assert left == right
    assert original["dataset"]["raw_local_mirror"] is False
    assert mirror["dataset"]["raw_local_mirror"] is True
    assert left[0]["sampling_rate_hz"] == 50
    assert transport.witness_reference(path, checked)[
        "no_raw_provider_data_read_for_table_reconstruction"
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        "archive",
        "probe",
        "people",
        "windows",
        "dirty",
        "old_source",
        "new_source",
        "identity_witness",
        "partial",
        "not_equal",
        "integer_boolean",
        "fit",
        "member_missing",
        "convention",
    ],
)
def test_witness_rejects_scope_provenance_and_materialisation_mutations(
    tmp_path: Path,
    fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    mutation: str,
) -> None:
    witness = deepcopy(fixture[0])
    if mutation == "archive":
        witness["archive_sha256"] = "0" * 64
    elif mutation == "probe":
        witness["script_sha256"] = "0" * 64
    elif mutation == "people":
        witness["participant_count"] = 2
    elif mutation == "windows":
        witness["window_count"] = 14
    elif mutation == "dirty":
        witness["analysis_git"]["worktree_dirty"] = True
    elif mutation == "old_source":
        witness["old_processing_source_sha256"] = "0" * 64
    elif mutation == "new_source":
        witness["new_processing_source_sha256"] = "0" * 64
    elif mutation == "identity_witness":
        witness["ordered_full_population_raw_array_byte_sha256"].pop("window_ids")
    elif mutation == "partial":
        witness["participants"].pop()
    elif mutation == "not_equal":
        witness["participants"][0]["bitwise_equal"] = False
    elif mutation == "integer_boolean":
        witness["performance_scores_computed"] = 0
    elif mutation == "fit":
        witness["models_fit"] = 1
    elif mutation == "member_missing":
        witness["participants"][0]["source_member_sha256"].popitem()
    else:
        witness["global_array_digest_convention"] = "different digest convention"
    path = tmp_path / "mutated.json"
    _write(path, witness)
    with pytest.raises(ValueError, match="transport witness"):
        transport.load_transport_witness(path, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset",
        "transfer",
        "source",
        "cohort",
        "members",
        "duplicate_member",
        "labels",
        "phase",
        "dtype",
        "class_support",
    ],
)
def test_witness_cannot_conceal_changes_in_the_actual_run(
    fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]], mutation: str
) -> None:
    witness, result, audit, identities = deepcopy(fixture)
    if mutation == "dataset":
        result["dataset"]["dataset_id"] = "fog_star_v3"
    elif mutation == "transfer":
        result["target_dataset"] = result["dataset"]
    elif mutation == "source":
        result["source_input_manifest"]["files"][transport._LOADER] = "0" * 64
    elif mutation == "cohort":
        result["dataset"]["participant_count"] = 2
    elif mutation == "members":
        audit["source_receipts"][0]["computed_sha256"] = "0" * 64
    elif mutation == "duplicate_member":
        audit["source_receipts"].append(audit["source_receipts"][0])
    elif mutation == "labels":
        identities["labels"][0] = 3
    elif mutation == "phase":
        identities["window_ids"] = np.roll(identities["window_ids"], 1)
    elif mutation == "dtype":
        identities["labels"] = identities["labels"].astype(np.float32)
    else:
        result["dataset"]["class_window_counts"]["walking"] += 1
    with pytest.raises(ValueError, match="non-comparable"):
        transport.run_transport_contract(result, audit, witness, identities)


def test_witness_self_hash_is_required(
    tmp_path: Path, fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]
) -> None:
    path = tmp_path / "witness.json"
    witness = fixture[0]
    witness["record_sha256"] = "0" * 64
    path.write_text(json.dumps(witness), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        transport.load_transport_witness(path, tmp_path)


@pytest.mark.parametrize("changed_sampling", [False, True])
def test_table_transport_integration_preserves_other_comparison_requirements(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fixture: tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]],
    changed_sampling: bool,
) -> None:
    witness, original, audit, identities = fixture
    witness_path = tmp_path / "witness.json"
    _write(witness_path, witness)
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
    classes = tuple(original["dataset"]["class_window_counts"])
    for receipt in audit["source_receipts"]:
        receipt.update(dataset_id="har_pmd_v1", locator=provider_copy.HAR_PMD_ARCHIVE_URL)
    directories = []
    for local, method in ((False, "RandomForest-6ch"), (True, "TinyHAR-6ch")):
        directory = tmp_path / method
        directory.mkdir()
        directories.append(directory)
        result = deepcopy(original)
        if local:
            result["dataset"].update(
                raw_local_mirror=True, source_storage_audit=witness["source_storage_audit"]
            )
            result["source_input_manifest"]["files"][transport._LOADER] = witness[
                "new_processing_source_sha256"
            ]
        if local and changed_sampling:
            result["dataset"]["sampling_rate_hz"] = 60
        primary, predictions = seed_evidence(
            ParticipantMetricInputs(
                "har_pmd_v1", classes, identities["labels"], identities["participant_ids"]
            ),
            {seed: {method: np.eye(5)[identities["labels"]]} for seed in (11, 23, 47)},
        )
        result.update(
            seeds=[11, 23, 47],
            primary_seed_averaged=primary,
            reports={method: {"class_names": classes}},
            prediction_artifact={"path": "predictions.npz"},
            git_at_launch={"commit": "d" * 40, "worktree_dirty": False},
        )
        result["source_input_manifest"].update(
            protocol_id="external-har-session-grid-v3", manifest_sha256="a" * 64
        )
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (directory / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
        arrays = {**identities, **{f"probability__{k}": v for k, v in predictions.items()}}
        np.savez_compressed(directory / "predictions.npz", **arrays)
    # No implicit equivalence: callers must supply the separately validated witness.
    with pytest.raises(ValueError, match="non-comparable"):
        table.reconstruct_matched_table(directories, tmp_path)
    if changed_sampling:
        with pytest.raises(ValueError, match="non-comparable"):
            table.reconstruct_matched_table(directories, tmp_path, transport_parity=witness_path)
    else:
        record = table.reconstruct_matched_table(
            directories, tmp_path, transport_parity=witness_path
        )
        assert record["statistics"]["participant_count"] == 120
        assert len(record["statistics"]["methods"]) == 2
        assert (
            record["transport_equivalence_reference"]["record_sha256"] == witness["record_sha256"]
        )
        assert [source["original_raw_local_mirror"] for source in record["sources"]] == [
            False,
            True,
        ]
