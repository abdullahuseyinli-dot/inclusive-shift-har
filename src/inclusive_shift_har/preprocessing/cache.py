"""Content-addressed preprocessing cache identity and validation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from inclusive_shift_har.manifests.canonical import canonical_json_sha256


def _require_sha256(value: str, *, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def preprocessing_cache_key(
    *,
    source_artifact_sha256: str,
    preprocessing_config_sha256: str,
    split_manifest_sha256: str,
    code_version: str,
) -> str:
    """Hash every input that can change deterministic preprocessed output."""

    _require_sha256(source_artifact_sha256, name="source_artifact_sha256")
    _require_sha256(preprocessing_config_sha256, name="preprocessing_config_sha256")
    _require_sha256(split_manifest_sha256, name="split_manifest_sha256")
    if not code_version or code_version.strip() != code_version:
        raise ValueError("code_version must be a non-empty canonical token")
    return canonical_json_sha256(
        {
            "code_version": code_version,
            "preprocessing_config_sha256": preprocessing_config_sha256,
            "source_artifact_sha256": source_artifact_sha256,
            "split_manifest_sha256": split_manifest_sha256,
        }
    )


def validate_cache_metadata(
    metadata: Mapping[str, Any],
    *,
    source_artifact_sha256: str,
    preprocessing_config_sha256: str,
    split_manifest_sha256: str,
    code_version: str,
) -> bool:
    """Return whether cache metadata exactly matches the current lineage."""

    expected = preprocessing_cache_key(
        source_artifact_sha256=source_artifact_sha256,
        preprocessing_config_sha256=preprocessing_config_sha256,
        split_manifest_sha256=split_manifest_sha256,
        code_version=code_version,
    )
    return metadata.get("cache_key") == expected and metadata.get("schema_version") == "1.0.0"
