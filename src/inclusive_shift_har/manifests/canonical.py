"""Deterministic JSON and hashing helpers for provenance records.

The helpers in this module deliberately reject duplicate JSON keys and non-finite
numbers.  Both can otherwise produce records whose interpretation depends on the
parser, which is unacceptable for immutable data and experiment manifests.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from uuid import uuid4


class CanonicalJSONError(ValueError):
    """Raised when a value cannot be represented as strict canonical JSON."""


def _reject_duplicate_keys(pairs: Iterable[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CanonicalJSONError(f"duplicate JSON object key: {key!r}")
        result[key] = value
    return result


def _reject_non_finite(token: str) -> None:
    raise CanonicalJSONError(f"non-finite JSON number is forbidden: {token}")


def canonical_json_bytes(value: Any) -> bytes:
    """Return a stable UTF-8 JSON representation of *value*.

    This is a small, documented canonicalization profile rather than an
    implementation of RFC 8785.  Repository hashes are stable because every
    producer uses these exact options.
    """

    try:
        text = json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise CanonicalJSONError(str(exc)) from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file without loading it into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    if source.is_symlink():
        raise ValueError(f"refusing to hash symbolic link: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_strict(path: str | os.PathLike[str]) -> Any:
    """Load UTF-8 JSON while rejecting duplicate keys and non-finite values."""

    source = Path(path)
    if source.is_symlink():
        raise CanonicalJSONError(f"refusing to load JSON through symbolic link: {source}")
    try:
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            return json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_non_finite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanonicalJSONError(f"cannot load strict JSON from {source}: {exc}") from exc


def atomic_write_json_new(
    value: Any,
    destination: str | os.PathLike[str],
    *,
    allowed_root: str | os.PathLike[str],
) -> Path:
    """Atomically publish canonical JSON at a path that must not already exist.

    Publication uses a hard link from a fully flushed temporary file.  Hard-link
    creation has create-if-absent semantics on supported local filesystems, so it
    cannot silently replace a file created by another process.  Unsupported
    filesystems fail closed.
    """

    root = Path(allowed_root).resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"allowed_root is not a directory: {root}")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved_parent = target.parent.resolve(strict=True)
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"destination escapes allowed_root: {target}") from exc
    target = resolved_parent / target.name
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite existing file: {target}")

    temporary = resolved_parent / f".{target.name}.partial.{uuid4().hex}"
    payload = canonical_json_bytes(value) + b"\n"
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    except Exception as exc:
        # Preserve the uniquely named partial file as failure evidence.
        raise OSError(
            f"atomic JSON publication failed; partial retained at {temporary}: {exc}"
        ) from exc
    else:
        temporary.unlink()
    return target
