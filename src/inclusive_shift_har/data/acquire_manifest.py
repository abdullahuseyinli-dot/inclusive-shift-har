"""Create-only acquisition of pinned artifacts from one validated dataset manifest."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from inclusive_shift_har.data.acquisition import DownloadResult, DownloadSpec, download_new_file
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.manifests.validation import validate_manifest_file


class ManifestAcquisitionError(RuntimeError):
    """Raised when an official manifest acquisition cannot fail closed."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestAcquisitionError(f"{name} must be an object")
    return value


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _confined_new_file(value: str | Path, *, root: Path, name: str) -> Path:
    if root.is_symlink():
        raise ManifestAcquisitionError(f"{name} root may not be a symlink")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ManifestAcquisitionError(f"{name} root must be an existing directory")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else resolved_root / raw
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise ManifestAcquisitionError(f"{name} may not be a symlink")
    prospective = candidate.resolve(strict=False)
    try:
        prospective.relative_to(resolved_root)
    except ValueError as exc:
        raise ManifestAcquisitionError(f"{name} escapes its allowed root") from exc
    if os.path.lexists(prospective):
        raise FileExistsError(f"refusing to overwrite existing {name}: {prospective}")
    return prospective


def _selected_artifacts(
    manifest: Mapping[str, Any], artifact_ids: Sequence[str]
) -> list[Mapping[str, Any]]:
    values = manifest.get("artifacts")
    if not isinstance(values, list) or not values:
        raise ManifestAcquisitionError("validated manifest has no artifact list")
    artifacts = [_mapping(value, name="manifest artifact") for value in values]
    by_id: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        artifact_id = artifact.get("artifact_id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in by_id:
            raise ManifestAcquisitionError("manifest artifact IDs are invalid or duplicated")
        by_id[artifact_id] = artifact
    if not artifact_ids:
        return artifacts
    if len(set(artifact_ids)) != len(artifact_ids):
        raise ManifestAcquisitionError("requested artifact IDs must be unique")
    unknown = sorted(set(artifact_ids) - set(by_id))
    if unknown:
        raise ManifestAcquisitionError(f"unknown artifact IDs: {unknown}")
    return [by_id[artifact_id] for artifact_id in artifact_ids]


def _scoped_file_reference(
    path: Path,
    *,
    root: Path,
    root_scope: str,
) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        relative = resolved.relative_to(root)
    except ValueError:
        return {
            "scope": "external",
            "basename": resolved.name,
            "absolute_path_recorded": False,
        }
    return {
        "scope": root_scope,
        "path": relative.as_posix(),
        "absolute_path_recorded": False,
    }


def _portable_error_message(error: Exception, *, data_root: Path) -> str:
    message = str(error)
    for spelling in {str(data_root), data_root.as_posix()}:
        message = message.replace(spelling, "<data_root>")
    return message


def _portable_retrieval(result: DownloadResult, *, data_root: Path) -> dict[str, Any]:
    retrieval = result.to_dict()
    destination = retrieval.pop("destination")
    if not isinstance(destination, str):
        raise ManifestAcquisitionError("download result destination must be a string")
    retrieval["destination_reference"] = _scoped_file_reference(
        Path(destination), root=data_root, root_scope="data_root"
    )
    return retrieval


def _receipt_body(
    *,
    status: str,
    manifest_path: Path,
    repository_root: Path,
    data_root: Path,
    manifest: Mapping[str, Any],
    manifest_canonical_sha256: str,
    results: Sequence[DownloadResult],
    error: Exception | None,
) -> dict[str, Any]:
    release = _mapping(manifest.get("release"), name="manifest release")
    licence = _mapping(manifest.get("license"), name="manifest license")
    body: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "official_dataset_manifest_acquisition",
        "status": status,
        "created_at_utc": _utc_now(),
        "dataset_id": manifest["dataset_id"],
        "release_version": release["version"],
        "release_doi": release["doi"],
        "manifest_reference": _scoped_file_reference(
            manifest_path, root=repository_root, root_scope="repository_root"
        ),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_canonical_sha256": manifest_canonical_sha256,
        "license_status": licence["status"],
        "raw_redistribution_policy": licence["redistribution_policy"],
        "retrievals": [_portable_retrieval(result, data_root=data_root) for result in results],
        "retrieval_count": len(results),
        "publication_policy": "create_only_read_only_raw_files",
        "raw_files_must_remain_outside_git": True,
    }
    if error is not None:
        body["failure"] = {
            "type": type(error).__name__,
            "message": _portable_error_message(error, data_root=data_root),
        }
        partial_path = getattr(error, "partial_path", None)
        if isinstance(partial_path, Path):
            body["failure"]["partial_reference"] = _scoped_file_reference(
                partial_path, root=data_root, root_scope="data_root"
            )
        body["completed_retrievals_preserved"] = True
    return body


def acquire_manifest_artifacts(
    *,
    manifest_path: str | Path,
    data_root: str | Path,
    receipt_output: str | Path,
    receipt_root: str | Path,
    repository_root: str | Path = ".",
    artifact_ids: Sequence[str] = (),
    timeout_seconds: float = 120.0,
    opener: Callable[..., AbstractContextManager[Any]] | None = None,
) -> dict[str, Any]:
    """Download the selected pinned official artifacts and publish one receipt."""

    source = Path(manifest_path)
    repository_root_path = Path(repository_root)
    if repository_root_path.is_symlink():
        raise ManifestAcquisitionError("repository_root may not be a symlink")
    resolved_repository_root = repository_root_path.resolve(strict=True)
    if not resolved_repository_root.is_dir():
        raise ManifestAcquisitionError("repository_root must be an existing directory")
    validation = validate_manifest_file(source)
    if not validation.valid or validation.manifest_sha256 is None:
        raise ManifestAcquisitionError(
            f"dataset manifest validation failed: {validation.to_dict()}"
        )
    manifest = _mapping(load_json_strict(source), name="dataset manifest")
    raw_root = Path(data_root)
    if raw_root.is_symlink():
        raise ManifestAcquisitionError("data_root may not be a symlink")
    resolved_data_root = raw_root.resolve(strict=True)
    if not resolved_data_root.is_dir():
        raise ManifestAcquisitionError("data_root must be an existing directory")
    receipt_root_path = Path(receipt_root)
    receipt_target = _confined_new_file(
        receipt_output, root=receipt_root_path, name="acquisition receipt"
    )
    selected = _selected_artifacts(manifest, artifact_ids)
    specs = [
        DownloadSpec.from_manifest_artifact(artifact, data_root=resolved_data_root)
        for artifact in selected
    ]
    # Fail before network access if any canonical destination is occupied or unsafe.
    for spec in specs:
        _confined_new_file(spec.destination, root=resolved_data_root, name=spec.artifact_id)
        if spec.expected_sha256 is None:
            raise ManifestAcquisitionError(
                f"{spec.artifact_id} is unpinned; update the reviewed manifest before acquisition"
            )

    receipt_target.parent.mkdir(parents=True, exist_ok=True)
    results: list[DownloadResult] = []
    try:
        for spec in specs:
            kwargs: dict[str, Any] = {
                "allowed_root": resolved_data_root,
                "allow_unpinned": False,
                "timeout_seconds": timeout_seconds,
                "make_read_only": True,
            }
            if opener is not None:
                kwargs["opener"] = opener
            results.append(download_new_file(spec, **kwargs))
    except Exception as exc:
        failure = _receipt_body(
            status="failed_preserved_create_only",
            manifest_path=source.resolve(strict=True),
            repository_root=resolved_repository_root,
            data_root=resolved_data_root,
            manifest=manifest,
            manifest_canonical_sha256=validation.manifest_sha256,
            results=results,
            error=exc,
        )
        failure["record_sha256"] = canonical_json_sha256(failure)
        atomic_write_json_new(failure, receipt_target, allowed_root=receipt_root_path)
        raise ManifestAcquisitionError(
            f"official acquisition failed; receipt preserved at {receipt_target}: {exc}"
        ) from exc

    receipt = _receipt_body(
        status="complete_create_only",
        manifest_path=source.resolve(strict=True),
        repository_root=resolved_repository_root,
        data_root=resolved_data_root,
        manifest=manifest,
        manifest_canonical_sha256=validation.manifest_sha256,
        results=results,
        error=None,
    )
    if receipt["retrieval_count"] != len(selected):
        raise AssertionError("acquisition receipt count differs from selected artifacts")
    receipt["record_sha256"] = canonical_json_sha256(receipt)
    atomic_write_json_new(receipt, receipt_target, allowed_root=receipt_root_path)
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--receipt-output", type=Path, required=True)
    parser.add_argument("--receipt-root", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--artifact-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        receipt = acquire_manifest_artifacts(
            manifest_path=args.manifest,
            data_root=args.data_root,
            receipt_output=args.receipt_output,
            receipt_root=args.receipt_root,
            repository_root=args.repository_root,
            artifact_ids=args.artifact_id,
            timeout_seconds=args.timeout_seconds,
        )
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
