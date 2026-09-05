"""Verified local transport for the exact public HAR-PMD provider archive."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from zipfile import ZipFile

HAR_PMD_PROVIDER_COPY_PROTOCOL = "external-har-verified-provider-copy-v1"
HAR_PMD_ARCHIVE_URL = "https://zenodo.org/api/records/7939223/files/data_publish.zip/content"
HAR_PMD_ARCHIVE_SIZE = 7288182191
HAR_PMD_ARCHIVE_MD5 = "cdc7b79aaa0450d68d94f430f2a2de63"
HAR_PMD_ARCHIVE_SHA256 = "3dd04d8239540ebc1eb50c569f0c421ce54875b6b14a69dddc6f3459b6f0d665"


@contextmanager
def verified_har_pmd_archive(path: Path) -> Iterator[tuple[ZipFile, dict[str, Any]]]:
    """Hash before ZIP parsing and read the same open file handle after verification."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size != HAR_PMD_ARCHIVE_SIZE:
        raise ValueError("local HAR-PMD provider copy has the wrong size or is not a regular file")
    sha256, md5, count = hashlib.sha256(), hashlib.md5(), 0
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            sha256.update(chunk)
            md5.update(chunk)
            count += len(chunk)
        if (count, sha256.hexdigest(), md5.hexdigest()) != (
            HAR_PMD_ARCHIVE_SIZE,
            HAR_PMD_ARCHIVE_SHA256,
            HAR_PMD_ARCHIVE_MD5,
        ):
            raise ValueError("local HAR-PMD provider copy does not match pinned provider bytes")
        audit = {
            "protocol_id": HAR_PMD_PROVIDER_COPY_PROTOCOL,
            "dataset_id": "har_pmd_v1",
            "provider_record": 7939223,
            "provider_version": "2",
            "source_url": HAR_PMD_ARCHIVE_URL,
            "path": str(path.resolve()),
            "archive_size_bytes": count,
            "archive_sha256": sha256.hexdigest(),
            "archive_md5": md5.hexdigest(),
            "digest_verified": True,
            "license": "CC-BY-4.0",
            "raw_local_mirror": True,
            "scope": "transport/storage only; member bytes, parsing, signal processing and windows unchanged",
        }
        stream.seek(0)
        with ZipFile(stream) as archive:
            yield archive, audit


def provider_copy_storage_errors(storage: Any, *, dataset_id: Any, locator: Any) -> list[str]:
    """Validate the disclosed pinned transport contract, without opening raw data."""
    if not isinstance(storage, dict):
        return ["local provider-copy storage audit is not an object"]
    expected = {
        "protocol_id": HAR_PMD_PROVIDER_COPY_PROTOCOL,
        "dataset_id": "har_pmd_v1",
        "provider_record": 7939223,
        "provider_version": "2",
        "source_url": HAR_PMD_ARCHIVE_URL,
        "archive_size_bytes": HAR_PMD_ARCHIVE_SIZE,
        "archive_sha256": HAR_PMD_ARCHIVE_SHA256,
        "archive_md5": HAR_PMD_ARCHIVE_MD5,
        "digest_verified": True,
        "license": "CC-BY-4.0",
        "raw_local_mirror": True,
    }
    if (
        dataset_id != "har_pmd_v1"
        or locator != HAR_PMD_ARCHIVE_URL
        or any(
            type(storage.get(key)) is not type(value) or storage.get(key) != value
            for key, value in expected.items()
        )
    ):
        return ["local storage does not disclose the exact verified HAR-PMD provider-copy contract"]
    return []
