from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest

from inclusive_shift_har.data import external_har, provider_copy
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory


def _synthetic_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "synthetic_provider.zip"
    time = np.arange(384) / 60
    frame = pd.DataFrame(
        {
            "Time": time,
            **{f"LAcc{axis}": np.sin(time + index) for index, axis in enumerate("XYZ")},
            **{f"Gyr{axis}": np.cos(time + index) for index, axis in enumerate("XYZ")},
            **{f"Gra{axis}": np.full(time.size, 9.81 if axis == "Z" else 0.0) for axis in "XYZ"},
        }
    )
    with ZipFile(path, "x") as archive:
        for activity in ("still", "walking", "crutches", "walker", "manual"):
            archive.writestr(
                f"data_publish/1/phone/1_{activity}_phone_indoor.csv", frame.to_csv(index=False)
            )
    payload = path.read_bytes()
    # Synthetic transport contract only; real loader pins are never replaced in a scientific run.
    monkeypatch.setattr(provider_copy, "HAR_PMD_ARCHIVE_SIZE", len(payload))
    monkeypatch.setattr(
        provider_copy, "HAR_PMD_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest()
    )
    monkeypatch.setattr(provider_copy, "HAR_PMD_ARCHIVE_MD5", hashlib.md5(payload).hexdigest())
    return path


def test_verified_copy_and_remote_reader_have_identical_member_processing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = _synthetic_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(external_har, "RemoteZip", lambda _url: ZipFile(path))
    remote = external_har.load_har_pmd_native(participant_limit=1, environments=("indoor",))
    local = external_har.load_har_pmd_native(
        source_archive=path, participant_limit=1, environments=("indoor",)
    )
    for name in (
        "signals",
        "gravity",
        "labels",
        "participant_ids",
        "session_ids",
        "trial_ids",
        "window_ids",
    ):
        np.testing.assert_array_equal(getattr(remote, name), getattr(local, name))
    assert remote.receipts == tuple(
        replace(receipt, raw_local_mirror=False) for receipt in local.receipts
    )
    assert remote.summary()["raw_local_mirror"] is False
    assert local.summary()["raw_local_mirror"] is True
    assert local.source_storage_audit is not None
    assert not provider_copy.provider_copy_storage_errors(
        local.source_storage_audit,
        dataset_id=local.dataset_id,
        locator=provider_copy.HAR_PMD_ARCHIVE_URL,
    )


@pytest.mark.parametrize("mutation", ["truncated", "same_size_changed"])
def test_unverified_copy_is_rejected_before_any_zip_parsing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = _synthetic_archive(tmp_path, monkeypatch)
    payload = path.read_bytes()
    path.write_bytes(
        payload[:-1] if mutation == "truncated" else bytes([payload[0] ^ 1]) + payload[1:]
    )

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("unverified bytes reached ZIP parser")

    monkeypatch.setattr(provider_copy, "ZipFile", forbidden)
    with pytest.raises(ValueError, match=r"wrong size|pinned provider bytes"):
        with provider_copy.verified_har_pmd_archive(path):
            raise AssertionError("unverified copy reached caller")


@pytest.mark.parametrize("mutation", ["missing", "hash", "license", "boolean", "record"])
def test_storage_contract_cannot_accept_an_undisclosed_or_forged_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    path = _synthetic_archive(tmp_path, monkeypatch)
    with provider_copy.verified_har_pmd_archive(path) as (_archive, audit):
        if mutation == "missing":
            audit = {}
        elif mutation == "hash":
            audit["archive_sha256"] = "f" * 64
        elif mutation == "license":
            audit["license"] = "Apache-2.0"
        elif mutation == "boolean":
            audit["digest_verified"] = 1
        else:
            audit["provider_record"] = 0
        assert provider_copy.provider_copy_storage_errors(
            audit, dataset_id="har_pmd_v1", locator=provider_copy.HAR_PMD_ARCHIVE_URL
        )


@pytest.mark.parametrize("value", [None, False, [], "verified"])
def test_non_object_storage_audit_fails_without_crashing(value: Any) -> None:
    assert provider_copy.provider_copy_storage_errors(
        value, dataset_id="har_pmd_v1", locator=provider_copy.HAR_PMD_ARCHIVE_URL
    )


@pytest.mark.parametrize("tampered", [False, True])
def test_run_validator_binds_local_receipts_to_the_matching_dataset_storage_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tampered: bool
) -> None:
    path = _synthetic_archive(tmp_path, monkeypatch)
    data = external_har.load_har_pmd_native(
        source_archive=path, participant_limit=1, environments=("indoor",)
    )
    audit: dict[str, Any] = {
        "dataset": data.summary(),
        "source_receipts": [receipt.to_dict() for receipt in data.receipts],
    }
    if tampered:
        audit["dataset"]["source_storage_audit"]["archive_sha256"] = "f" * 64
    run = tmp_path / "synthetic_failed_run"
    run.mkdir()
    (run / "data_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    (run / "failure.json").write_text('{"status":"FAILED_PRESERVED"}', encoding="utf-8")
    result = validate_run_directory(run, tmp_path)
    assert result["integrity_passed"] is not tampered
    assert result["publication_evidence_ready"] is False
