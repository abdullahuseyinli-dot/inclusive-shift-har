"""Hash-gated, domain-sealed reader for the DAGHAR v2 standardized archive."""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_AXES = ("accel-x", "accel-y", "accel-z", "gyro-x", "gyro-y", "gyro-z")
_PARTITIONS = ("train", "validation", "test")
_CODE_TO_FUNCTIONAL = {2: 0, 0: 1, 1: 2}


@dataclass(frozen=True)
class DAGHARWindows:
    signals: NDArray[np.float32]
    labels: NDArray[np.int64]
    participant_ids: NDArray[np.str_]
    domain_ids: NDArray[np.str_]
    partitions: NDArray[np.str_]
    window_ids: NDArray[np.str_]


def daghar_signal_columns(window_length: int = 60) -> tuple[str, ...]:
    if window_length < 1:
        raise ValueError("DAGHAR window length must be positive")
    return tuple(f"{axis}-{index}" for axis in _AXES for index in range(window_length))


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _read_manifest(path: Path) -> dict[str, Any]:
    return _mapping(json.loads(path.read_text(encoding="utf-8")), name="DAGHAR manifest")


def _archive_contract(manifest: dict[str, Any], archive_path: Path) -> tuple[str, int]:
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != 1:
        raise ValueError("DAGHAR manifest must pin exactly one standardized archive")
    artifact = _mapping(artifacts[0], name="DAGHAR archive artifact")
    expected_hash = str(artifact["expected_sha256"])
    expected_size = int(artifact["expected_size_bytes"])
    if archive_path.stat().st_size != expected_size or sha256_file(archive_path) != expected_hash:
        raise ValueError("DAGHAR archive size or SHA-256 does not match its manifest")
    return expected_hash, expected_size


def _load_daghar_windows(
    archive_path: Path,
    *,
    archive_sha256: str,
    domains: tuple[str, ...],
    partitions: tuple[str, ...],
) -> DAGHARWindows:
    """Read already-authorized DAGHAR members after the caller enforces the seal."""

    feature_columns = daghar_signal_columns()
    use_columns = [*feature_columns, "user", "standard activity code"]
    signals: list[NDArray[np.float32]] = []
    labels: list[NDArray[np.int64]] = []
    participant_ids: list[NDArray[np.str_]] = []
    domain_ids: list[NDArray[np.str_]] = []
    partition_ids: list[NDArray[np.str_]] = []
    window_ids: list[NDArray[np.str_]] = []
    users_by_partition: dict[tuple[str, str], set[str]] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for domain in domains:
            for partition in partitions:
                member = f"standardized_view/{domain}/{partition}.csv"
                if member not in names:
                    raise ValueError(f"DAGHAR archive lacks declared member {member}")
                with archive.open(member) as stream:
                    frame = pd.read_csv(
                        stream,
                        usecols=use_columns,
                        dtype={"user": "string"},
                    )
                raw_codes = frame["standard activity code"].to_numpy(dtype=np.int64)
                keep = np.isin(raw_codes, tuple(_CODE_TO_FUNCTIONAL))
                filtered = frame.loc[keep]
                if filtered.empty:
                    raise ValueError(f"DAGHAR member {member} has no functional-core rows")
                raw = filtered.loc[:, feature_columns].to_numpy(dtype=np.float32)
                if raw.shape[1] != 360 or not np.isfinite(raw).all():
                    raise ValueError(f"DAGHAR member {member} violates its finite signal schema")
                count = raw.shape[0]
                member_signals = raw.reshape(count, 6, 60).transpose(0, 2, 1)
                member_codes = filtered["standard activity code"].to_numpy(dtype=np.int64)
                member_labels = np.asarray(
                    [_CODE_TO_FUNCTIONAL[int(code)] for code in member_codes], dtype=np.int64
                )
                users = filtered["user"].astype(str).to_numpy(dtype=np.str_)
                qualified_users = np.asarray([f"{domain}:{user}" for user in users], dtype=np.str_)
                original_rows = filtered.index.to_numpy(dtype=np.int64)
                ids = np.asarray(
                    [
                        canonical_json_sha256(
                            {
                                "dataset": "daghar_v2",
                                "archive_sha256": archive_sha256,
                                "domain": domain,
                                "partition": partition,
                                "member": member,
                                "csv_zero_based_row": int(row),
                            }
                        )
                        for row in original_rows
                    ],
                    dtype=np.str_,
                )
                signals.append(member_signals)
                labels.append(member_labels)
                participant_ids.append(qualified_users)
                domain_ids.append(np.asarray([domain] * count, dtype=np.str_))
                partition_ids.append(np.asarray([partition] * count, dtype=np.str_))
                window_ids.append(ids)
                users_by_partition[(domain, partition)] = set(qualified_users.tolist())
    for domain in domains:
        selected_partitions = [item for item in partitions if (domain, item) in users_by_partition]
        for first_index, first in enumerate(selected_partitions):
            for second in selected_partitions[first_index + 1 :]:
                overlap = users_by_partition[(domain, first)] & users_by_partition[(domain, second)]
                if overlap:
                    raise ValueError(
                        f"DAGHAR provider partitions overlap users for {domain}: {sorted(overlap)}"
                    )
    result = DAGHARWindows(
        signals=np.concatenate(signals),
        labels=np.concatenate(labels),
        participant_ids=np.concatenate(participant_ids),
        domain_ids=np.concatenate(domain_ids),
        partitions=np.concatenate(partition_ids),
        window_ids=np.concatenate(window_ids),
    )
    if len(set(result.window_ids.tolist())) != result.window_ids.size:
        raise ValueError("DAGHAR stable window identifiers are not unique")
    return result


def _validate_partitions(partitions: tuple[str, ...]) -> None:
    if (
        not partitions
        or len(set(partitions)) != len(partitions)
        or not set(partitions) <= set(_PARTITIONS)
    ):
        raise ValueError("DAGHAR partitions must be unique train/validation/test values")


def load_daghar_development_windows(
    archive_path: Path,
    manifest_path: Path,
    *,
    domains: tuple[str, ...],
    partitions: tuple[str, ...] = _PARTITIONS,
) -> DAGHARWindows:
    """Load only predeclared development domains; sealed domains are refused."""

    manifest = _read_manifest(manifest_path)
    archive_sha256, _ = _archive_contract(manifest, archive_path)
    partition_record = _mapping(
        manifest.get("external_evaluation_partition"), name="external_evaluation_partition"
    )
    development = {
        str(item) for item in cast(list[object], partition_record["development_domains"])
    }
    sealed = {
        str(item) for item in cast(list[object], partition_record["sealed_evaluation_domains"])
    }
    requested = set(domains)
    if not domains or len(requested) != len(domains) or not requested <= development:
        forbidden = sorted(requested & sealed)
        suffix = f"; sealed domains requested: {forbidden}" if forbidden else ""
        raise PermissionError(f"DAGHAR development loader accepts development domains only{suffix}")
    _validate_partitions(partitions)
    return _load_daghar_windows(
        archive_path,
        archive_sha256=archive_sha256,
        domains=domains,
        partitions=partitions,
    )


def load_daghar_evaluation_windows(
    archive_path: Path,
    manifest_path: Path,
    *,
    domains: tuple[str, ...],
    partitions: tuple[str, ...],
    freeze_record_path: Path,
    expected_freeze_file_sha256: str,
) -> DAGHARWindows:
    """Open exactly the sealed domains after validating a create-only candidate freeze."""

    manifest = _read_manifest(manifest_path)
    archive_sha256, _ = _archive_contract(manifest, archive_path)
    partition_record = _mapping(
        manifest.get("external_evaluation_partition"), name="external_evaluation_partition"
    )
    sealed = tuple(
        str(item) for item in cast(list[object], partition_record["sealed_evaluation_domains"])
    )
    if domains != sealed:
        raise PermissionError(
            "DAGHAR evaluation loader requires every sealed domain in frozen order"
        )
    _validate_partitions(partitions)
    if sha256_file(freeze_record_path) != expected_freeze_file_sha256:
        raise ValueError("DAGHAR external candidate freeze file hash changed")
    freeze = _mapping(
        json.loads(freeze_record_path.read_text(encoding="utf-8")),
        name="DAGHAR external candidate freeze",
    )
    claimed = freeze.get("record_sha256")
    unhashed = dict(freeze)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError("DAGHAR external candidate freeze self-hash changed")
    if freeze.get("record_kind") != "daghar_external_evaluation_candidate_freeze":
        raise PermissionError("DAGHAR evaluation loader requires the candidate-freeze record kind")
    if freeze.get("status") != "frozen_before_external_evaluation_opening":
        raise PermissionError("DAGHAR evaluation candidate is not frozen")
    if freeze.get("daghar_archive_sha256") != archive_sha256:
        raise ValueError("DAGHAR evaluation freeze references a different archive")
    if tuple(freeze.get("sealed_evaluation_domains", [])) != domains:
        raise ValueError("DAGHAR evaluation freeze domain order changed")
    if (
        freeze.get("sealed_domain_labels_predictions_or_metrics_accessed_before_freeze")
        is not False
    ):
        raise PermissionError(
            "DAGHAR evaluation freeze does not attest a sealed performance boundary"
        )
    if freeze.get("schema_probe_before_partition_declaration_disclosed") is not True:
        raise PermissionError("DAGHAR evaluation freeze must disclose the earlier schema probe")
    return _load_daghar_windows(
        archive_path,
        archive_sha256=archive_sha256,
        domains=domains,
        partitions=partitions,
    )
