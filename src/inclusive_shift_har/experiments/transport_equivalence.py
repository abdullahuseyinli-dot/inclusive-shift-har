"""Narrow, observed HAR-PMD transport equivalence for publication reconstruction."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.data.provider_copy import (
    HAR_PMD_ARCHIVE_SHA256,
    HAR_PMD_ARCHIVE_URL,
    provider_copy_storage_errors,
)
from inclusive_shift_har.experiments.external_evidence_validate import _manifest_commit_errors
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_PEOPLE = 120
_WINDOWS = 248496
_LOADER = "src/inclusive_shift_har/data/external_har.py"
_OLD_COMMIT = "5a585e9764a82d9e40bc805b684e13274551fdb0"
_PROBE_SHA256 = "8f22d7c8fbb861031911521f826737ac7acf03f029284a4979e39318de30753b"
_IDENTITIES = ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids")
_ARRAYS = ("signals", "gravity", *_IDENTITIES)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_transport_witness(path: Path, root: Path) -> dict[str, Any]:
    """Check a clean, no-fit, all-participant witness; never read its raw ZIP."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("record_sha256") != canonical_json_sha256(
        {key: item for key, item in value.items() if key != "record_sha256"}
    ):
        raise ValueError("transport witness self-hash is invalid")
    expected = {
        "status": "ALL_PARTICIPANT_TRANSPORT_PARITY_PASSED",
        "dataset_id": "har_pmd_v1",
        "archive_sha256": HAR_PMD_ARCHIVE_SHA256,
        "old_processing_source_commit": _OLD_COMMIT,
        "script_sha256": _PROBE_SHA256,
        "models_fit": 0,
        "annotation_or_value_mutations": 0,
        "performance_scores_computed": False,
        "participant_count": _PEOPLE,
        "window_count": _WINDOWS,
        "purpose": "transport_only_materialization_parity_no_model_fitting",
        "global_array_digest_convention": "C-order raw bytes concatenated in numeric participant order; dtype and global row count recorded separately; not the shape-prefixed _array_sha256 convention",
    }
    if any(
        type(value.get(key)) is not type(item) or value.get(key) != item
        for key, item in expected.items()
    ):
        raise ValueError("transport witness is not the declared full-cohort no-fit audit")
    git, manifest = value.get("analysis_git", {}), value.get("analysis_source_input_manifest", {})
    if not isinstance(git, dict) or not isinstance(manifest, dict):
        raise ValueError("transport witness execution records must be objects")
    files = manifest.get("files", {})
    if (
        git.get("worktree_dirty") is not False
        or git.get("status_entries") != []
        or not files
        or manifest.get("manifest_sha256") != canonical_json_sha256(files)
        or _manifest_commit_errors(root, git.get("commit", ""), files)
        or files.get(_LOADER) != value.get("new_processing_source_sha256")
    ):
        raise ValueError("transport witness does not bind clean committed execution")
    old_blob = subprocess.check_output(["git", "show", f"{_OLD_COMMIT}:{_LOADER}"], cwd=root)
    if hashlib.sha256(old_blob).hexdigest() != value.get("old_processing_source_sha256"):
        raise ValueError("transport witness old loader does not match its committed source")
    if provider_copy_storage_errors(
        value.get("source_storage_audit"), dataset_id="har_pmd_v1", locator=HAR_PMD_ARCHIVE_URL
    ):
        raise ValueError("transport witness provider-copy contract is invalid")
    people = value.get("participants", [])
    if (
        not isinstance(people, list)
        or not all(isinstance(person, dict) for person in people)
        or [person.get("participant") for person in people]
        != [str(i) for i in range(1, _PEOPLE + 1)]
        or any(
            person.get("bitwise_equal") is not True
            or type(person.get("windows")) is not int
            or person["windows"] <= 0
            or set(person.get("arrays_sha256", {})) != set(_ARRAYS)
            or not all(_is_sha256(digest) for digest in person["arrays_sha256"].values())
            or not person.get("source_member_sha256")
            or len(person["source_member_sha256"]) != 10
            or not all(_is_sha256(digest) for digest in person["source_member_sha256"].values())
            or sum(person.get("class_window_counts", {}).values()) != person["windows"]
            for person in people
        )
        or sum(person["windows"] for person in people) != _WINDOWS
        or set(value.get("ordered_full_population_raw_array_byte_sha256", {})) != set(_ARRAYS)
        or not all(
            _is_sha256(digest)
            for digest in value["ordered_full_population_raw_array_byte_sha256"].values()
        )
        or set(value.get("array_dtypes", {})) != set(_ARRAYS)
    ):
        raise ValueError("transport witness lacks complete per-participant materialization")
    return value


def run_transport_contract(
    result: dict[str, Any],
    audit: dict[str, Any],
    witness: dict[str, Any],
    identities: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require actual run receipts and prediction identities to match the witness."""
    dataset = result.get("dataset", {})
    source = result.get("source_input_manifest", {}).get("files", {}).get(_LOADER)
    if (
        "target_dataset" in result
        or dataset.get("dataset_id") != "har_pmd_v1"
        or dataset.get("participant_count") != _PEOPLE
        or dataset.get("window_count") != _WINDOWS
        or source
        not in {witness["old_processing_source_sha256"], witness["new_processing_source_sha256"]}
    ):
        raise ValueError("non-comparable run is outside the witnessed HAR-PMD source/cohort scope")
    members: dict[str, str] = {}
    class_counts: dict[str, int] = {}
    for person in witness["participants"]:
        for member, digest in person["source_member_sha256"].items():
            if member in members:
                raise ValueError("transport witness repeats a source member")
            members[member] = digest
        for name, count in person["class_window_counts"].items():
            class_counts[name] = class_counts.get(name, 0) + count
    receipts = audit.get("source_receipts", [])
    observed = {item.get("member"): item.get("computed_sha256") for item in receipts}
    if (
        len(receipts) != len(observed)
        or observed != members
        or dataset.get("class_window_counts") != class_counts
    ):
        raise ValueError(
            "non-comparable HAR-PMD source members or class support differ from the witness"
        )
    for name in _IDENTITIES:
        array = np.ascontiguousarray(identities[name])
        if (
            array.shape != (_WINDOWS,)
            or array.dtype.str != witness["array_dtypes"][name]
            or hashlib.sha256(array.tobytes(order="C")).hexdigest()
            != witness["ordered_full_population_raw_array_byte_sha256"][name]
        ):
            raise ValueError(
                f"non-comparable HAR-PMD prediction identity differs from witness: {name}"
            )
    normalised = {
        key: item
        for key, item in dataset.items()
        if key not in {"raw_local_mirror", "source_storage_audit"}
    }
    contract = {
        "equivalence_basis": "all_participant_observed_har_pmd_transport_parity",
        "witness_record_sha256": witness["record_sha256"],
        "observed_array_byte_sha256": witness["ordered_full_population_raw_array_byte_sha256"],
        "scope": "Pinned provider bytes and observed full cohort only; storage changes are not model improvements. All other dataset/split/window/seed/metric checks remain required.",
    }
    return normalised, contract


def witness_reference(path: Path, witness: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "file_sha256": sha256_file(path),
        "record_sha256": witness["record_sha256"],
        "no_raw_provider_data_read_for_table_reconstruction": True,
    }
