from __future__ import annotations

import hashlib
import json
import os
import stat
from io import BytesIO
from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.data.acquire_manifest import (
    ManifestAcquisitionError,
    acquire_manifest_artifacts,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

from ._synthetic import synthetic_dataset_manifest, with_second_artifact


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _make_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            path.chmod(path.stat().st_mode | stat.S_IWUSR)


def test_manifest_acquisition_is_pinned_create_only_and_self_hashed(tmp_path: Path) -> None:
    first = b"first official synthetic payload\n"
    second = b"second official synthetic payload\n"
    manifest = with_second_artifact(
        synthetic_dataset_manifest(payload=first, storage_path="synthetic/v1/first.bin")
    )
    manifest["artifacts"][1].update(
        {
            "download_url": "https://example.invalid/synthetic-har/second.bin",
            "storage_path": "synthetic/v1/second.bin",
            "expected_size_bytes": len(second),
            "expected_sha256": hashlib.sha256(second).hexdigest(),
        }
    )
    manifest_path = tmp_path / "manifest.json"
    data_root = tmp_path / "raw"
    receipt_root = tmp_path / "receipts"
    data_root.mkdir()
    receipt_root.mkdir()
    _write_manifest(manifest_path, manifest)
    payloads = {
        artifact["download_url"]: payload
        for artifact, payload in zip(manifest["artifacts"], (first, second), strict=True)
    }

    def opener(request: Any, *, timeout: float) -> BytesIO:
        assert timeout > 0
        return BytesIO(payloads[request.full_url])

    try:
        receipt = acquire_manifest_artifacts(
            manifest_path=manifest_path,
            data_root=data_root,
            receipt_output="inclusivehar.synthetic.receipt.json",
            receipt_root=receipt_root,
            repository_root=tmp_path,
            opener=opener,
        )

        assert receipt["status"] == "complete_create_only"
        assert receipt["retrieval_count"] == 2
        assert receipt["manifest_reference"] == {
            "scope": "repository_root",
            "path": "manifest.json",
            "absolute_path_recorded": False,
        }
        assert [item["destination_reference"] for item in receipt["retrievals"]] == [
            {
                "scope": "data_root",
                "path": "synthetic/v1/first.bin",
                "absolute_path_recorded": False,
            },
            {
                "scope": "data_root",
                "path": "synthetic/v1/second.bin",
                "absolute_path_recorded": False,
            },
        ]
        assert str(tmp_path) not in json.dumps(receipt)
        record_hash = receipt.pop("record_sha256")
        assert record_hash == canonical_json_sha256(receipt)
        assert (data_root / "synthetic/v1/first.bin").read_bytes() == first
        assert (data_root / "synthetic/v1/second.bin").read_bytes() == second
        assert not (data_root / "synthetic/v1/first.bin").stat().st_mode & stat.S_IWUSR
        persisted = json.loads(
            (receipt_root / "inclusivehar.synthetic.receipt.json").read_text(encoding="utf-8")
        )
        persisted_hash = persisted.pop("record_sha256")
        assert persisted_hash == canonical_json_sha256(persisted)
        with pytest.raises(FileExistsError, match="refusing to overwrite"):
            acquire_manifest_artifacts(
                manifest_path=manifest_path,
                data_root=data_root,
                receipt_output="inclusivehar.synthetic.receipt.json",
                receipt_root=receipt_root,
                repository_root=tmp_path,
                opener=opener,
            )
    finally:
        _make_writable(data_root)


def test_manifest_acquisition_preserves_partial_failure_and_completed_files(
    tmp_path: Path,
) -> None:
    first = b"first official synthetic payload\n"
    expected_second = b"expected second payload\n"
    observed_second = b"corrupt second payload!\n"
    assert len(expected_second) == len(observed_second)
    manifest = with_second_artifact(
        synthetic_dataset_manifest(payload=first, storage_path="synthetic/v1/first.bin")
    )
    manifest["artifacts"][1].update(
        {
            "download_url": "https://example.invalid/synthetic-har/second.bin",
            "storage_path": "synthetic/v1/second.bin",
            "expected_size_bytes": len(expected_second),
            "expected_sha256": hashlib.sha256(expected_second).hexdigest(),
        }
    )
    manifest_path = tmp_path / "manifest.json"
    data_root = tmp_path / "raw"
    receipt_root = tmp_path / "receipts"
    data_root.mkdir()
    receipt_root.mkdir()
    _write_manifest(manifest_path, manifest)
    payloads = {
        manifest["artifacts"][0]["download_url"]: first,
        manifest["artifacts"][1]["download_url"]: observed_second,
    }

    def opener(request: Any, *, timeout: float) -> BytesIO:
        assert timeout > 0
        return BytesIO(payloads[request.full_url])

    try:
        with pytest.raises(ManifestAcquisitionError, match="receipt preserved"):
            acquire_manifest_artifacts(
                manifest_path=manifest_path,
                data_root=data_root,
                receipt_output="failed.receipt.json",
                receipt_root=receipt_root,
                repository_root=tmp_path,
                opener=opener,
            )

        assert (data_root / "synthetic/v1/first.bin").read_bytes() == first
        assert not (data_root / "synthetic/v1/second.bin").exists()
        partials = list((data_root / "synthetic/v1").glob(".second.bin.partial.*"))
        assert len(partials) == 1
        assert partials[0].read_bytes() == observed_second
        receipt = json.loads((receipt_root / "failed.receipt.json").read_text(encoding="utf-8"))
        assert receipt["status"] == "failed_preserved_create_only"
        assert receipt["retrieval_count"] == 1
        assert receipt["completed_retrievals_preserved"] is True
        assert receipt["failure"]["partial_reference"] == {
            "scope": "data_root",
            "path": partials[0].relative_to(data_root).as_posix(),
            "absolute_path_recorded": False,
        }
        assert str(tmp_path) not in json.dumps(receipt)
        record_hash = receipt.pop("record_sha256")
        assert record_hash == canonical_json_sha256(receipt)
    finally:
        _make_writable(data_root)


def test_manifest_acquisition_rejects_receipt_escape_before_network(tmp_path: Path) -> None:
    payload = b"official synthetic payload\n"
    manifest = synthetic_dataset_manifest(payload=payload, storage_path="synthetic/v1/data.bin")
    manifest_path = tmp_path / "manifest.json"
    data_root = tmp_path / "raw"
    receipt_root = tmp_path / "receipts"
    data_root.mkdir()
    receipt_root.mkdir()
    _write_manifest(manifest_path, manifest)
    network_called = False

    def opener(_request: Any, *, timeout: float) -> BytesIO:
        nonlocal network_called
        network_called = True
        return BytesIO(payload)

    with pytest.raises(ManifestAcquisitionError, match="escapes"):
        acquire_manifest_artifacts(
            manifest_path=manifest_path,
            data_root=data_root,
            receipt_output=Path("..") / "escaped.json",
            receipt_root=receipt_root,
            repository_root=tmp_path,
            opener=opener,
        )

    assert network_called is False
    assert not os.path.lexists(tmp_path / "escaped.json")


def test_manifest_acquisition_marks_external_manifest_without_absolute_path(
    tmp_path: Path,
) -> None:
    payload = b"official synthetic payload\n"
    manifest = synthetic_dataset_manifest(payload=payload, storage_path="synthetic/v1/data.bin")
    manifest_path = tmp_path / "external-manifest.json"
    repository_root = tmp_path / "repository"
    data_root = repository_root / "data"
    receipt_root = repository_root / "receipts"
    repository_root.mkdir()
    data_root.mkdir()
    receipt_root.mkdir()
    _write_manifest(manifest_path, manifest)

    def opener(_request: Any, *, timeout: float) -> BytesIO:
        assert timeout > 0
        return BytesIO(payload)

    try:
        receipt = acquire_manifest_artifacts(
            manifest_path=manifest_path,
            data_root=data_root,
            receipt_output="receipt.json",
            receipt_root=receipt_root,
            repository_root=repository_root,
            opener=opener,
        )

        assert receipt["manifest_reference"] == {
            "scope": "external",
            "basename": "external-manifest.json",
            "absolute_path_recorded": False,
        }
        assert str(tmp_path) not in json.dumps(receipt)
    finally:
        _make_writable(data_root)
