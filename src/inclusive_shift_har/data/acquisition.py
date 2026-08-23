"""Safe, provenance-aware acquisition primitives.

There is intentionally no eager downloader or import-time network activity.  A
caller must provide a destination under an existing allowed root, and unpinned
downloads require an explicit opt-in.  Successful publication never overwrites
an existing file; failed partial transfers remain uniquely named as evidence.
"""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import uuid4

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class AcquisitionError(RuntimeError):
    """Raised when an acquisition cannot safely complete."""

    def __init__(self, message: str, *, partial_path: Path | None = None) -> None:
        super().__init__(message)
        self.partial_path = partial_path


@dataclass(frozen=True)
class DownloadSpec:
    artifact_id: str
    url: str
    destination: Path
    expected_size_bytes: int | None
    expected_sha256: str | None

    @classmethod
    def from_manifest_artifact(
        cls,
        artifact: Mapping[str, Any],
        *,
        data_root: str | os.PathLike[str],
    ) -> DownloadSpec:
        resolved_root = Path(data_root).resolve(strict=True)
        return cls(
            artifact_id=str(artifact["artifact_id"]),
            url=str(artifact["download_url"]),
            destination=resolved_root / str(artifact["storage_path"]),
            expected_size_bytes=artifact.get("expected_size_bytes"),
            expected_sha256=artifact.get("expected_sha256"),
        )


@dataclass(frozen=True)
class DownloadResult:
    artifact_id: str
    destination: str
    size_bytes: int
    sha256: str
    source_url: str
    completed_at_utc: str
    provider_hash_was_pinned: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "completed_at_utc": self.completed_at_utc,
            "destination": self.destination,
            "provider_hash_was_pinned": self.provider_hash_was_pinned,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "source_url": self.source_url,
        }


def _validate_spec(spec: DownloadSpec, *, allow_unpinned: bool) -> None:
    parsed = urlparse(spec.url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username is not None:
        raise AcquisitionError("download URL must be credential-free HTTPS")
    if not spec.artifact_id:
        raise AcquisitionError("artifact_id must not be empty")
    if spec.expected_size_bytes is not None and spec.expected_size_bytes <= 0:
        raise AcquisitionError("expected_size_bytes must be positive when present")
    if spec.expected_sha256 is not None and not _SHA256_RE.fullmatch(spec.expected_sha256):
        raise AcquisitionError("expected_sha256 must be a lowercase 64-character SHA-256")
    if spec.expected_sha256 is None and not allow_unpinned:
        raise AcquisitionError(
            "provider publishes no pinned SHA-256; repeat with allow_unpinned=True only for an explicit first official retrieval"
        )


def _safe_target(destination: Path, allowed_root: Path) -> Path:
    root = allowed_root.resolve(strict=True)
    if not root.is_dir():
        raise AcquisitionError(f"allowed data root is not a directory: {root}")
    candidate = destination if destination.is_absolute() else root / destination
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(f"destination escapes allowed data root: {destination}") from exc
    if not resolved_candidate.name or resolved_candidate == root:
        raise AcquisitionError("destination must name a file below the allowed data root")
    resolved_candidate.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = resolved_candidate.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise AcquisitionError(
            f"destination parent escapes allowed data root: {resolved_parent}"
        ) from exc
    target = resolved_parent / resolved_candidate.name
    if os.path.lexists(target):
        raise AcquisitionError(f"refusing to overwrite existing destination: {target}")
    return target


def download_new_file(
    spec: DownloadSpec,
    *,
    allowed_root: str | os.PathLike[str],
    allow_unpinned: bool = False,
    timeout_seconds: float = 120.0,
    chunk_size: int = 1024 * 1024,
    make_read_only: bool = True,
    opener: Callable[..., AbstractContextManager[Any]] = urlopen,
) -> DownloadResult:
    """Download, verify, and atomically publish one new file.

    ``opener`` is injectable so synthetic tests can use an in-memory response
    without weakening the production HTTPS policy.
    """

    _validate_spec(spec, allow_unpinned=allow_unpinned)
    if timeout_seconds <= 0:
        raise AcquisitionError("timeout_seconds must be positive")
    if chunk_size <= 0:
        raise AcquisitionError("chunk_size must be positive")
    target = _safe_target(spec.destination, Path(allowed_root))
    partial = target.parent / f".{target.name}.partial.{uuid4().hex}"
    digest = hashlib.sha256()
    bytes_written = 0
    request = Request(
        spec.url,
        headers={
            "Accept": "application/octet-stream,*/*;q=0.1",
            "Accept-Encoding": "identity",
            "User-Agent": "InclusiveShift-HAR provenance downloader/0.1",
        },
        method="GET",
    )
    try:
        with partial.open("xb") as output:
            with opener(request, timeout=timeout_seconds) as response:
                status = getattr(response, "status", 200)
                if status != 200:
                    raise AcquisitionError(f"unexpected HTTP status {status}", partial_path=partial)
                while block := response.read(chunk_size):
                    if not isinstance(block, bytes):
                        raise AcquisitionError(
                            "download response returned non-byte content", partial_path=partial
                        )
                    output.write(block)
                    digest.update(block)
                    bytes_written += len(block)
            output.flush()
            os.fsync(output.fileno())

        observed_sha256 = digest.hexdigest()
        if spec.expected_size_bytes is not None and bytes_written != spec.expected_size_bytes:
            raise AcquisitionError(
                f"size mismatch for {spec.artifact_id}: expected {spec.expected_size_bytes}, observed {bytes_written}",
                partial_path=partial,
            )
        if spec.expected_sha256 is not None and observed_sha256 != spec.expected_sha256:
            raise AcquisitionError(
                f"SHA-256 mismatch for {spec.artifact_id}: expected {spec.expected_sha256}, observed {observed_sha256}",
                partial_path=partial,
            )
        if os.path.lexists(target):
            raise AcquisitionError(
                f"destination appeared during download; refusing overwrite: {target}",
                partial_path=partial,
            )
        try:
            os.link(partial, target)
        except OSError as exc:
            raise AcquisitionError(
                f"atomic create-only publication failed; partial retained at {partial}: {exc}",
                partial_path=partial,
            ) from exc
        # Remove the staging name before changing permissions.  On Windows,
        # read-only state belongs to the shared hard-linked file record, so
        # chmod-before-unlink makes the staging link impossible to remove.
        partial.unlink()
        if make_read_only:
            current_mode = target.stat().st_mode
            target.chmod(current_mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    except AcquisitionError:
        raise
    except Exception as exc:
        raise AcquisitionError(
            f"download failed; partial evidence retained at {partial}: {exc}",
            partial_path=partial if os.path.lexists(partial) else None,
        ) from exc

    completed = datetime.now(UTC).isoformat().replace("+00:00", "Z")
    return DownloadResult(
        artifact_id=spec.artifact_id,
        destination=str(target),
        size_bytes=bytes_written,
        sha256=observed_sha256,
        source_url=spec.url,
        completed_at_utc=completed,
        provider_hash_was_pinned=spec.expected_sha256 is not None,
    )
