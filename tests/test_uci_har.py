from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.data.uci_har import (
    CROSS_SOURCE_ELIGIBILITY,
    UCIHARDataError,
    UCIHAREvidencePurpose,
    audit_uci_har_archive,
    load_uci_har_split,
    select_uci_native_core,
    stable_uci_window_id,
)
from inclusive_shift_har.manifests.canonical import sha256_file
from tests._uci_synthetic import materialize_synthetic_uci_archive


def test_loads_primary_six_with_subjects_and_stable_window_ids(tmp_path: Path) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    expected_hash = sha256_file(archive)

    train = load_uci_har_split(archive, split="train", expected_sha256=expected_hash)

    assert train.signals.shape == (30, 128, 6)
    assert train.signals.dtype == np.float32
    assert train.activity_ids.tolist()[:6] == [1, 2, 3, 4, 5, 6]
    assert sorted(set(train.subject_ids.tolist())) == [1, 2, 3, 4, 5]
    assert train.window_ids[0] == "uci_har_v1:train:window:000001"
    assert train.window_ids[-1] == stable_uci_window_id("train", 30)
    assert train.sample_ids == train.window_ids
    assert train.sample_id_semantics == "released_window_instance_not_recoverable_raw_sample"
    assert not train.signals.flags.writeable
    assert not train.activity_ids.flags.writeable
    assert np.allclose(train.signals[0, :3, 0], [0.0, 0.001, 0.002])
    assert np.allclose(train.signals[0, :3, 5], [5.0, 5.001, 5.002])
    with pytest.raises(ValueError, match="positive integer"):
        stable_uci_window_id("train", 0)


def test_official_test_requires_explicit_consumed_audit_purpose(tmp_path: Path) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")

    with pytest.raises(UCIHARDataError, match="development-consumed"):
        load_uci_har_split(archive, split="test")

    test = load_uci_har_split(
        archive,
        split="test",
        purpose=UCIHAREvidencePurpose.LEGACY_CONSUMED_AUDIT,
    )
    assert test.signals.shape == (12, 128, 6)


def test_uci_native_core_mapping_does_not_overstate_cross_source_eligibility(
    tmp_path: Path,
) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    train = load_uci_har_split(archive, split="train")

    core = select_uci_native_core(train)

    assert core.source_activity_ids.tolist()[:3] == [1, 4, 5]
    assert core.canonical_activity_ids.tolist()[:3] == [0, 1, 2]
    assert core.canonical_activity_names[:3] == ("walking", "sitting", "standing")
    assert CROSS_SOURCE_ELIGIBILITY[4]["status"] == "exact"
    assert CROSS_SOURCE_ELIGIBILITY[5]["status"] == "provisional"
    assert CROSS_SOURCE_ELIGIBILITY[1]["status"] == "excluded_all_cohort_exact"


def test_read_only_archive_audit_checks_crc_shapes_and_subject_exclusivity(tmp_path: Path) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    report = audit_uci_har_archive(archive, expected_sha256=sha256_file(archive))

    assert report["archive"]["zip_crc_status"] == "pass"
    assert report["released_split_audit"]["train"]["shape"] == [30, 128, 6]
    assert report["released_split_audit"]["test"]["shape"] == [12, 128, 6]
    assert report["released_split_audit"]["subject_overlap"] == []
    assert report["source_development_gate"]["status"] == "pass"


def test_manifest_preserves_both_conflicting_license_notices(repository_root: Path) -> None:
    manifest = json.loads(
        (repository_root / "manifests/datasets/uci_har_v1.json").read_text(encoding="utf-8")
    )

    assert manifest["license"]["status"] == "conflicting_notices"
    notices = manifest["license"]["notices"]
    assert {notice["identifier"] for notice in notices} == {"CC-BY-4.0", "NOASSERTION"}
    assert manifest["license"]["redistribution_policy"] == "do_not_redistribute_raw"
    assert manifest["artifacts"][0]["expected_sha256"] == (
        "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
    )
