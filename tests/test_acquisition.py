from __future__ import annotations

import hashlib
import stat
from io import BytesIO
from pathlib import Path

from inclusive_shift_har.data.acquisition import DownloadSpec, download_new_file


def test_read_only_publication_removes_staging_link_before_chmod(tmp_path: Path) -> None:
    payload = b"synthetic InclusiveShift-HAR acquisition fixture\n"
    destination = tmp_path / "artifact.bin"
    spec = DownloadSpec(
        artifact_id="synthetic_fixture",
        url="https://example.test/artifact.bin",
        destination=destination,
        expected_size_bytes=len(payload),
        expected_sha256=hashlib.sha256(payload).hexdigest(),
    )

    def opener(_request: object, *, timeout: float) -> BytesIO:
        assert timeout > 0
        return BytesIO(payload)

    try:
        result = download_new_file(
            spec,
            allowed_root=tmp_path,
            make_read_only=True,
            opener=opener,
        )

        assert destination.read_bytes() == payload
        assert result.sha256 == spec.expected_sha256
        assert not destination.stat().st_mode & stat.S_IWUSR
        assert not list(tmp_path.glob(".artifact.bin.partial.*"))
    finally:
        if destination.exists():
            destination.chmod(destination.stat().st_mode | stat.S_IWUSR)
