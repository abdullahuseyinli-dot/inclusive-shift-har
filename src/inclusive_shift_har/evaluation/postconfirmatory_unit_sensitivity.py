"""Create a cache-only SI acceleration-unit sensitivity record.

The command consumes only hash-pinned, partition-pure post-confirmatory caches.
It has no raw-data, unlock, target-opening, model, or training interface.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from inclusive_shift_har.data.inclusivehar import INCLUSIVEHAR_PRIMARY_CHANNELS
from inclusive_shift_har.evaluation._strict_config import (
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
    require_utc_timestamp,
)
from inclusive_shift_har.evaluation.unit_sensitivity import (
    build_acceleration_unit_sensitivity_record,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    PreparedPrimaryCacheEvidence,
    load_prepared_primary_cache,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    sha256_file,
)
from inclusive_shift_har.preprocessing.units import (
    STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED,
)

RECORD_KIND = "postconfirmatory_acceleration_unit_conversion_sensitivity"
ANALYSIS_ID = "inclusivehar_primary_g_to_si_zscore_equivalence_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class PostconfirmatoryUnitSensitivityError(RuntimeError):
    """Raised when the cache-only sensitivity lineage is invalid."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostconfirmatoryUnitSensitivityError(f"{name} must be an object")
    return value


def _sha256(value: str, *, name: str) -> str:
    normalized = value.casefold() if isinstance(value, str) else ""
    if _SHA256_RE.fullmatch(normalized) is None:
        raise PostconfirmatoryUnitSensitivityError(f"{name} must be a full SHA-256")
    return normalized


def _existing_file(value: str | Path, *, root: Path, name: str) -> Path:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    if candidate.is_symlink():
        raise PostconfirmatoryUnitSensitivityError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryUnitSensitivityError(f"{name} escapes artifact_root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise PostconfirmatoryUnitSensitivityError(f"{name} must be a regular file")
    return resolved


def _planned_output(value: str | Path, *, artifact_root: Path, output_root: Path) -> Path:
    try:
        output_root.relative_to(artifact_root)
    except ValueError as exc:
        raise PostconfirmatoryUnitSensitivityError(
            "output_root must remain inside artifact_root"
        ) from exc
    raw = Path(value)
    candidate = raw if raw.is_absolute() else output_root / raw
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(output_root)
    except ValueError as exc:
        raise PostconfirmatoryUnitSensitivityError("output escapes output_root") from exc
    if os.path.lexists(resolved):
        raise FileExistsError(f"refusing to overwrite existing unit-sensitivity record: {resolved}")
    return resolved


def _load_sensitivity_config(
    path_value: str | Path,
    *,
    artifact_root: Path,
    expected_file_sha256: str,
) -> tuple[Mapping[str, Any], Path, str, str]:
    path = _existing_file(path_value, root=artifact_root, name="SI sensitivity configuration")
    file_hash = _sha256(
        expected_file_sha256, name="expected SI sensitivity configuration file hash"
    )
    if sha256_file(path) != file_hash:
        raise PostconfirmatoryUnitSensitivityError("SI sensitivity configuration file hash changed")
    config = load_strict_yaml_mapping(path)
    require_exact_keys(
        config,
        {
            "schema_version",
            "preprocessing_id",
            "dataset_id",
            "status",
            "evidence_status",
            "track_role",
            "primary_claim_eligible",
            "channels",
            "units",
            "conversion",
            "window_length_samples",
            "window_stride_samples",
            "declared_sampling_rate_hz",
            "boundary_unit",
            "remainder_policy",
            "normalization",
            "interpretation",
            "forbidden_claims",
        },
        location="SI sensitivity configuration",
    )
    units = require_mapping(config.get("units"), location="SI sensitivity configuration.units")
    conversion = require_mapping(
        config.get("conversion"), location="SI sensitivity configuration.conversion"
    )
    normalization = require_mapping(
        config.get("normalization"), location="SI sensitivity configuration.normalization"
    )
    require_exact_keys(
        units,
        {
            "motionUserAcceleration_input",
            "motionUserAcceleration_output",
            "motionRotationRate",
        },
        location="SI sensitivity configuration.units",
    )
    require_exact_keys(
        conversion,
        {"acceleration_multiplier", "timing"},
        location="SI sensitivity configuration.conversion",
    )
    require_exact_keys(
        normalization,
        {
            "method",
            "fit_scope",
            "validation_in_fit",
            "target_in_fit",
            "epsilon",
        },
        location="SI sensitivity configuration.normalization",
    )
    required = {
        "schema_version": "1.0.0",
        "preprocessing_id": "inclusivehar-primary-six-si-128-sensitivity-v1",
        "dataset_id": "inclusivehar_v4",
        "status": "predeclared_post_confirmatory_preprocessing_sensitivity",
        "evidence_status": "post_confirmatory_unit_sensitivity",
        "track_role": "numerical_unit-representation_sensitivity_not_model_evaluation",
        "primary_claim_eligible": False,
        "channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
        "window_length_samples": 128,
        "window_stride_samples": 128,
        "declared_sampling_rate_hz": 50,
        "boundary_unit": "released_contiguous_subject_activity_block",
        "remainder_policy": "drop_tail_shorter_than_128_per_released_block",
    }
    mismatches = [key for key, expected in required.items() if config.get(key) != expected]
    nested_valid = (
        units
        == {
            "motionUserAcceleration_input": "g",
            "motionUserAcceleration_output": "m/s^2",
            "motionRotationRate": "rad/s",
        }
        and conversion.get("timing") == "before_training_only_standardization"
        and isinstance(conversion.get("acceleration_multiplier"), (int, float))
        and not isinstance(conversion.get("acceleration_multiplier"), bool)
        and float(conversion["acceleration_multiplier"])
        == STANDARD_GRAVITY_METRES_PER_SECOND_SQUARED
        and normalization
        == {
            "method": "per_channel_z_score",
            "fit_scope": "source_training_partition_only",
            "validation_in_fit": False,
            "target_in_fit": False,
            "epsilon": 1.0e-8,
        }
        and config.get("forbidden_claims")
        == ["confirmatory_model_result", "accuracy_improvement", "independent_experiment"]
    )
    if mismatches or not nested_valid:
        raise PostconfirmatoryUnitSensitivityError(
            f"SI sensitivity configuration contract changed: {mismatches}"
        )
    return config, path, file_hash, canonical_json_sha256(config)


def _identity_lineage(evidence: PreparedPrimaryCacheEvidence, *, partition: str) -> dict[str, Any]:
    batch = evidence.cache.batch
    metadata_lineage = _mapping(
        evidence.cache.metadata.get("lineage"), name=f"{partition} cache lineage"
    )
    identities = {
        "window_ids_sha256": canonical_json_sha256(sorted(batch.window_ids)),
        "ordered_window_ids_sha256": canonical_json_sha256(list(batch.window_ids)),
        "ordered_participant_ids_sha256": canonical_json_sha256(list(batch.participant_ids)),
        "ordered_labels_sha256": canonical_json_sha256(batch.labels.tolist()),
    }
    if metadata_lineage.get("partition") != partition:
        raise PostconfirmatoryUnitSensitivityError(
            f"{partition} cache metadata has the wrong partition"
        )
    mismatches = [
        key for key, expected in identities.items() if metadata_lineage.get(key) != expected
    ]
    if mismatches:
        raise PostconfirmatoryUnitSensitivityError(
            f"{partition} cache ordered identity does not reconstruct: {mismatches}"
        )
    return {
        "partition": partition,
        "window_count": int(batch.signals.shape[0]),
        "participant_ids": sorted(set(batch.participant_ids), key=int),
        "array_sha256": evidence.cache.array_sha256,
        "metadata_sha256": evidence.cache.metadata_sha256,
        **identities,
    }


def _cache_reference(
    evidence: PreparedPrimaryCacheEvidence,
    *,
    partition: str,
    artifact_root: Path,
) -> dict[str, Any]:
    reference = _identity_lineage(evidence, partition=partition)
    reference["array_path"] = evidence.cache.array_path.relative_to(artifact_root).as_posix()
    reference["metadata_path"] = evidence.cache.metadata_path.relative_to(artifact_root).as_posix()
    return reference


def _validate_shared_cache_lineage(
    source: PreparedPrimaryCacheEvidence,
    target: PreparedPrimaryCacheEvidence,
    *,
    expected_split_manifest_sha256: str,
    expected_source_artifact_sha256: str,
) -> None:
    if (
        source.record_path != target.record_path
        or source.record_file_sha256 != target.record_file_sha256
        or source.record_sha256 != target.record_sha256
    ):
        raise PostconfirmatoryUnitSensitivityError(
            "source and target caches do not share one externally pinned cache index"
        )
    source_lineage = _mapping(source.cache.metadata.get("lineage"), name="source cache lineage")
    target_lineage = _mapping(target.cache.metadata.get("lineage"), name="target cache lineage")
    common_fields = (
        "split_manifest_sha256",
        "split_manifest_file_sha256",
        "source_artifact_sha256",
        "preprocessing_config_sha256",
        "ontology_config_sha256",
        "ontology_class_schema_sha256",
        "opening_receipt_record_sha256",
        "opening_receipt_file_sha256",
        "locked_target_index_record_sha256",
        "locked_target_index_file_sha256",
        "target_seal_id",
        "code_commit",
    )
    differences = [
        field for field in common_fields if source_lineage.get(field) != target_lineage.get(field)
    ]
    if differences:
        raise PostconfirmatoryUnitSensitivityError(
            f"source and target cache lineage differs: {differences}"
        )
    if (
        source_lineage.get("split_manifest_sha256") != expected_split_manifest_sha256
        or source_lineage.get("source_artifact_sha256") != expected_source_artifact_sha256
    ):
        raise PostconfirmatoryUnitSensitivityError("cache split or source-artifact pin changed")


def create_postconfirmatory_unit_sensitivity_record(
    *,
    sensitivity_config_path: str | Path,
    expected_sensitivity_config_file_sha256: str,
    primary_cache_record_path: str | Path,
    expected_primary_cache_record_file_sha256: str,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: str | Path,
    expected_split_manifest_sha256: str,
    expected_source_artifact_sha256: str,
    output: str | Path,
    output_root: str | Path,
    created_at_utc: str,
    equivalence_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Validate two immutable caches and publish one preprocessing-only record."""

    root = Path(artifact_root).resolve(strict=True)
    allowed_output_root = Path(output_root).resolve(strict=True)
    destination = _planned_output(output, artifact_root=root, output_root=allowed_output_root)
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    split_hash = _sha256(expected_split_manifest_sha256, name="expected split-manifest hash")
    source_hash = _sha256(expected_source_artifact_sha256, name="expected source artifact hash")
    cache_file_hash = _sha256(
        expected_primary_cache_record_file_sha256,
        name="expected primary cache record file hash",
    )
    config, config_path, config_file_hash, config_hash = _load_sensitivity_config(
        sensitivity_config_path,
        artifact_root=root,
        expected_file_sha256=expected_sensitivity_config_file_sha256,
    )
    source = load_prepared_primary_cache(
        record_path=primary_cache_record_path,
        expected_record_file_sha256=cache_file_hash,
        partition="source_train",
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
        expected_split_manifest_sha256=split_hash,
        expected_source_artifact_sha256=source_hash,
    )
    target = load_prepared_primary_cache(
        record_path=primary_cache_record_path,
        expected_record_file_sha256=cache_file_hash,
        partition="target_sealed",
        opening_receipt_path=opening_receipt_path,
        locked_target_index_path=locked_target_index_path,
        artifact_root=root,
        expected_split_manifest_sha256=split_hash,
        expected_source_artifact_sha256=source_hash,
    )
    _validate_shared_cache_lineage(
        source,
        target,
        expected_split_manifest_sha256=split_hash,
        expected_source_artifact_sha256=source_hash,
    )
    source_reference = _cache_reference(source, partition="source_train", artifact_root=root)
    target_reference = _cache_reference(target, partition="target_sealed", artifact_root=root)
    primary_record = source.record
    opening = _mapping(primary_record.get("opening_1"), name="primary cache opening lineage")
    channels = tuple(str(value) for value in config["channels"])
    lineage: dict[str, Any] = {
        "sensitivity_config": {
            "path": config_path.relative_to(root).as_posix(),
            "file_sha256": config_file_hash,
            "canonical_config_sha256": config_hash,
        },
        "primary_cache_index": {
            "path": source.record_path.relative_to(root).as_posix(),
            "file_sha256": source.record_file_sha256,
            "record_sha256": source.record_sha256,
            "code_commit": primary_record.get("code_commit"),
        },
        "source_training_cache": source_reference,
        "evaluation_cache": target_reference,
        "consumed_opening_1": dict(opening),
        "normalization_fit_partition": "source_train",
        "evaluation_partition": "target_sealed",
    }
    core = build_acceleration_unit_sensitivity_record(
        source.cache.batch.signals,
        target.cache.batch.signals,
        source.cache.batch.participant_ids,
        declared_source_participants=set(source.cache.batch.participant_ids),
        split_manifest_sha256=split_hash,
        channel_names=channels,
        lineage=lineage,
        equivalence_tolerance=equivalence_tolerance,
    )
    record = dict(core)
    record.pop("record_sha256")
    record.update(
        {
            "record_kind": RECORD_KIND,
            "analysis_id": ANALYSIS_ID,
            "created_at_utc": timestamp,
            "primary_claim_eligible": False,
            "model_selection_use": False,
            "accuracy_or_model_metrics_computed": False,
            "training_performed": False,
            "cpu_training_performed": False,
            "gpu_training_performed": False,
            "raw_dataset_file_accessed": False,
            "existing_materialized_target_cache_accessed": True,
            "target_labels_accessed_only_for_cache_lineage_validation": True,
            "target_opening_number": 1,
            "unlock_api_called": False,
            "new_target_opening_created": False,
        }
    )
    record["record_sha256"] = canonical_json_sha256(record)
    atomic_write_json_new(record, destination, allowed_root=allowed_output_root)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sensitivity-config", type=Path, required=True)
    parser.add_argument("--expected-sensitivity-config-file-sha256", required=True)
    parser.add_argument("--primary-cache-record", type=Path, required=True)
    parser.add_argument("--expected-primary-cache-record-file-sha256", required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--expected-split-manifest-sha256", required=True)
    parser.add_argument("--expected-source-artifact-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--equivalence-tolerance", type=float, default=1e-6)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        record = create_postconfirmatory_unit_sensitivity_record(
            sensitivity_config_path=args.sensitivity_config,
            expected_sensitivity_config_file_sha256=(args.expected_sensitivity_config_file_sha256),
            primary_cache_record_path=args.primary_cache_record,
            expected_primary_cache_record_file_sha256=(
                args.expected_primary_cache_record_file_sha256
            ),
            opening_receipt_path=args.opening_receipt,
            locked_target_index_path=args.locked_target_index,
            artifact_root=args.artifact_root,
            expected_split_manifest_sha256=args.expected_split_manifest_sha256,
            expected_source_artifact_sha256=args.expected_source_artifact_sha256,
            output=args.output,
            output_root=args.output_root,
            created_at_utc=args.created_at_utc,
            equivalence_tolerance=args.equivalence_tolerance,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        print(json.dumps({"status": "fail", "message": str(exc)}, sort_keys=True))
        return 2
    destination = Path(args.output)
    if not destination.is_absolute():
        destination = Path(args.output_root) / destination
    print(
        json.dumps(
            {
                "status": record["status"],
                "record_sha256": record["record_sha256"],
                "record_file_sha256": sha256_file(destination.resolve(strict=True)),
                "training_performed": False,
                "raw_dataset_file_accessed": False,
                "new_target_opening_created": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
