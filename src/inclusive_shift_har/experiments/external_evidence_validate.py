"""Validate create-only external-HAR run evidence before any result is reported."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np
import yaml

from inclusive_shift_har.artifacts.research_provenance import (
    _EXPLICIT_SOURCE_INPUTS,
    EVIDENCE_ROLE_LEDGER_PATH,
    EXECUTED_RESEARCH_MODULE_PATH,
    PARENT_SUPERSESSION_LEDGER_PATH,
    PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
    PUBLICATION_PROTOCOL_PATH,
    PUBLICATION_SOURCE_PROTOCOL_ID,
    RUNTIME_ENVIRONMENT_PROTOCOL_ID,
    SUPERSESSION_LEDGER_PATH,
    _assert_executed_repository_root,
    _external_evidence_status,
    _fog_star_full_cohort_observed,
    _har_pmd_full_cohort_observed,
    _imu_har_il_expected_cohort_observed,
    _is_publication_source_input_path,
    _publication_artifact_contract,
    _write_json_create_only,
)
from inclusive_shift_har.data.external_har import (
    BOUNDARY_PROVENANCE_PROTOCOL,
    CORE_CLASS_NAMES,
    OBSERVABLE_SCORING_ELIGIBILITY_POLICY,
    PHYSICAL_GRID_PROTOCOL,
    SOURCE_RECEIPT_METADATA,
    _array_sha256,
)
from inclusive_shift_har.data.participant_partitions import (
    PARTITION_PROTOCOL_ID,
    build_participant_partition_plan,
)
from inclusive_shift_har.data.provider_copy import provider_copy_storage_errors
from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.evaluation.inference_contracts import (
    ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS,
    OBSERVABLE_CONTEXT_PROTOCOL,
    OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
    annotation_selected_context_methods,
    method_inference_contracts,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.publication_split_audit import (
    _absolute_direct_child_name,
    _stable_split_audit_payload,
    build_split_audit,
    write_split_audit,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

_FOG_SOURCE_URL = "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content"
_FOG_SOURCE_SIZE = 119_629_580
_FOG_SOURCE_MD5 = "952a37ab147da35e6d4e7a1e9bac44cb"
_FOG_SOURCE_SHA256 = "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
_HAR_PMD_SOURCE_URL = "https://zenodo.org/api/records/7939223/files/data_publish.zip/content"
_IMU_SOURCE_PREFIX = "https://data.csiro.au/dap/ws/v2/collections/74700/data/"
_SOLE_SOURCE_PREFIX = "https://zenodo.org/api/records/19242395/files/"

_HAR_PMD_CLASS_NAMES = (
    "stationary",
    "walking",
    "crutches",
    "walker",
    "manual_wheelchair",
)
_EXTERNAL_CLASSICAL_METHODS = (
    "RMRP-DG",
    "CTGR-DG",
    "CAGE-DG",
    "CTGR-DG-top3-equal",
    "HERA-DG-strict",
    "HERA-DG-context-safe",
    "HERA-DG-full",
    "HERA-DG-v2-weighted",
    "HERA-DG-v2-core",
    "HERA-DG-v2-dual",
    "HERA-DG-v2-full",
    "RandomForest-6ch",
    "XGBoost-6ch",
)


def _read_object(path: Path) -> dict[str, Any]:
    value = load_json_strict(path)
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def _objects(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _objects(child)
    elif isinstance(value, list):
        for child in value:
            yield from _objects(child)


def _safe_run_artifact_path(run_directory: Path, value: Any) -> Path | None:
    """Resolve one direct run artifact without permitting absolute or parent paths."""

    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 1:
        return None
    candidate = run_directory / relative.name
    try:
        if candidate.resolve(strict=False).parent != run_directory.resolve(strict=True):
            return None
    except OSError:
        return None
    return candidate


def _safe_nested_run_artifact_path(
    run_directory: Path, value: Any, *, required_parent: str
) -> Path | None:
    """Resolve a nested artifact while rejecting traversal and symlink indirection."""

    if not isinstance(value, str) or not value or "\\" in value or ":" in value:
        return None
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or len(relative.parts) != 2
        or relative.parts[0] != required_parent
    ):
        return None
    candidate = run_directory.joinpath(*relative.parts)
    try:
        resolved_run = run_directory.resolve(strict=True)
        if candidate.resolve(strict=False).parent != (resolved_run / required_parent):
            return None
        parent = resolved_run / required_parent
        if parent.is_symlink() or candidate.is_symlink():
            return None
    except OSError:
        return None
    return candidate


def _file_sha256_matches(path: Path | None, expected: Any) -> bool:
    if path is None or not isinstance(expected, str) or not path.is_file() or path.is_symlink():
        return False
    try:
        return sha256_file(path) == expected
    except (OSError, ValueError):
        return False


def _self_hash_valid(record: dict[str, Any], field: str) -> bool:
    declared = record.get(field)
    unhashed = dict(record)
    unhashed.pop(field, None)
    return isinstance(declared, str) and canonical_json_sha256(unhashed) == declared


def _stored_split_audit_matches(
    stored: dict[str, Any], reconstructed: dict[str, Any] | None
) -> bool:
    """Compare every deterministic split field, including warnings and assignments."""

    if (
        reconstructed is None
        or not _timezone_aware_iso8601(stored.get("created_at"))
        or not _self_hash_valid(stored, "record_sha256")
    ):
        return False
    return _stable_split_audit_payload(stored) == _stable_split_audit_payload(reconstructed)


def _timezone_aware_iso8601(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None


def _manifest_scope_at_commit(repository_root: Path, commit: object) -> set[str] | None:
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return None
    response = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if response.returncode:
        return None
    return {
        name for name in response.stdout.splitlines() if _is_publication_source_input_path(name)
    }


def _publication_manifest_errors(
    manifest: dict[str, Any], repository_root: Path, commit: object
) -> list[str]:
    errors: list[str] = []
    files = manifest.get("files")
    if not isinstance(files, dict):
        return ["publication source manifest files are absent"]
    typed_files = {str(name): str(digest) for name, digest in files.items()}
    if manifest.get("schema_version") != "2.0.0" or not _self_hash_valid(manifest, "record_sha256"):
        errors.append("publication source manifest schema or self-hash differs")
    if not _timezone_aware_iso8601(manifest.get("captured_at")):
        errors.append("publication source manifest capture time is absent or timezone-naive")
    if any(re.fullmatch(r"[0-9a-f]{64}", digest) is None for digest in typed_files.values()):
        errors.append("publication source manifest contains an invalid file SHA-256")
    missing_required = sorted(_EXPLICIT_SOURCE_INPUTS - set(typed_files))
    if missing_required:
        errors.append(
            "publication source manifest omits required governance inputs: "
            + ", ".join(missing_required)
        )
    expected_scope = _manifest_scope_at_commit(repository_root, commit)
    if expected_scope is None or set(typed_files) != expected_scope:
        errors.append("publication source manifest is not the exact launch-commit source scope")

    expected_references = {
        "publication_protocol": {
            "protocol_id": PUBLICATION_SOURCE_PROTOCOL_ID,
            "path": PUBLICATION_PROTOCOL_PATH,
            "sha256": typed_files.get(PUBLICATION_PROTOCOL_PATH),
        },
        "evidence_role_ledger": {
            "ledger_id": "external-evidence-roles-20260905-v3",
            "path": EVIDENCE_ROLE_LEDGER_PATH,
            "sha256": typed_files.get(EVIDENCE_ROLE_LEDGER_PATH),
        },
    }
    for name, expected in expected_references.items():
        if manifest.get(name) != expected or expected["sha256"] is None:
            errors.append(f"publication source manifest {name} linkage differs")
    supersession = manifest.get("supersession_ledger")
    if not isinstance(supersession, dict) or (
        supersession.get("record_kind") != "external_har_create_only_supersession_ledger"
        or supersession.get("path") != SUPERSESSION_LEDGER_PATH
        or supersession.get("sha256") != typed_files.get(SUPERSESSION_LEDGER_PATH)
        or re.fullmatch(r"[0-9a-f]{64}", str(supersession.get("record_sha256"))) is None
    ):
        errors.append("publication source manifest supersession linkage differs")
    elif isinstance(commit, str):
        response = subprocess.run(
            ["git", "show", f"{commit}:{SUPERSESSION_LEDGER_PATH}"],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        parent_response = subprocess.run(
            ["git", "show", f"{commit}:{PARENT_SUPERSESSION_LEDGER_PATH}"],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        protocol_response = subprocess.run(
            ["git", "show", f"{commit}:{PUBLICATION_PROTOCOL_PATH}"],
            cwd=repository_root,
            capture_output=True,
            check=False,
        )
        try:
            retained = json.loads(response.stdout.decode("utf-8"))
            retained_unhashed = dict(retained)
            retained_hash = retained_unhashed.pop("record_sha256", None)
            parent = json.loads(parent_response.stdout.decode("utf-8"))
            parent_unhashed = dict(parent)
            parent_hash = parent_unhashed.pop("record_sha256", None)
            protocol_value = yaml.safe_load(protocol_response.stdout.decode("utf-8"))
            protocol = protocol_value if isinstance(protocol_value, dict) else {}
            supersession_context = protocol.get("evidence_governance", {}).get(
                "supersession_context"
            )
            context_protocol = protocol.get("composes", {}).get(
                "observable_context_training_population"
            )
            retained_valid = (
                response.returncode == 0
                and parent_response.returncode == 0
                and protocol_response.returncode == 0
                and isinstance(retained, dict)
                and isinstance(parent, dict)
                and retained_hash == supersession.get("record_sha256")
                and canonical_json_sha256(retained_unhashed) == retained_hash
                and canonical_json_sha256(parent_unhashed) == parent_hash
                and retained.get("parent_ledger")
                == {
                    "path": PARENT_SUPERSESSION_LEDGER_PATH,
                    "file_sha256": hashlib.sha256(parent_response.stdout).hexdigest(),
                }
                and supersession_context
                == {
                    "record_kind": "external_har_create_only_supersession_ledger",
                    "path": SUPERSESSION_LEDGER_PATH,
                    "record_sha256": retained_hash,
                }
                and context_protocol
                == {
                    "protocol_id": OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
                    "path": (
                        "configs/protocols/"
                        "external_har_observable_context_training_population_v1.yaml"
                    ),
                }
            )
        except (UnicodeError, json.JSONDecodeError, TypeError, ValueError, yaml.YAMLError):
            retained_valid = False
        if not retained_valid:
            errors.append("launch-commit supersession ledger self-hash differs")
    return errors


def _runtime_path_is_absolute(value: Any) -> bool:
    return isinstance(value, str) and any(
        path_type(value).is_absolute() for path_type in (PurePosixPath, PureWindowsPath)
    )


def _environment_errors(
    environment: Any,
    source_manifest: Any = None,
) -> list[str]:
    if not isinstance(environment, dict):
        return ["runtime environment record is absent"]
    packages = environment.get("packages")
    executable = environment.get("python_executable")
    executed_module = environment.get("executed_research_module")
    errors: list[str] = []
    if (
        environment.get("protocol_id") != RUNTIME_ENVIRONMENT_PROTOCOL_ID
        or not _timezone_aware_iso8601(environment.get("captured_at_utc"))
        or any(
            not isinstance(environment.get(name), str) or not environment.get(name)
            for name in ("python_version", "python_implementation", "platform")
        )
    ):
        errors.append("runtime interpreter/platform identity is incomplete")
    if (
        not isinstance(executable, dict)
        or set(executable) != {"resolved_path_at_runtime", "sha256"}
        or not _runtime_path_is_absolute(executable.get("resolved_path_at_runtime"))
        or re.fullmatch(r"[0-9a-f]{64}", str(executable.get("sha256"))) is None
    ):
        errors.append("runtime Python executable identity is incomplete")
    if (
        not isinstance(executed_module, dict)
        or set(executed_module)
        != {"repository_relative_path", "resolved_path_at_runtime", "sha256"}
        or executed_module.get("repository_relative_path") != EXECUTED_RESEARCH_MODULE_PATH
        or not _runtime_path_is_absolute(executed_module.get("resolved_path_at_runtime"))
        or not str(executed_module.get("resolved_path_at_runtime", ""))
        .replace("\\", "/")
        .casefold()
        .endswith(f"/{EXECUTED_RESEARCH_MODULE_PATH}".casefold())
        or re.fullmatch(r"[0-9a-f]{64}", str(executed_module.get("sha256"))) is None
    ):
        errors.append("executed research module identity is incomplete")
    if (
        isinstance(source_manifest, dict)
        and source_manifest.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
    ):
        files = source_manifest.get("files")
        expected_source_hash = (
            files.get(EXECUTED_RESEARCH_MODULE_PATH) if isinstance(files, dict) else None
        )
        observed_source_hash = (
            executed_module.get("sha256") if isinstance(executed_module, dict) else None
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", str(expected_source_hash)) is None
            or observed_source_hash != expected_source_hash
        ):
            errors.append(
                "executed research module hash differs from the publication source manifest"
            )
    if (
        not isinstance(packages, dict)
        or not packages
        or any(
            not isinstance(name, str) or not name or not isinstance(version, str) or not version
            for name, version in packages.items()
        )
        or "numpy" not in packages
    ):
        errors.append("runtime dependency inventory is incomplete")
    return errors


def _receipt_contract_errors(audit: dict[str, Any], *, publication_v4: bool) -> list[str]:
    errors: list[str] = []
    if not publication_v4:
        receipts = _receipt_objects(audit)
        if not receipts:
            return ["source receipts are absent"]
        for receipt in receipts:
            digest = receipt.get("computed_sha256")
            received = receipt.get("received_size_bytes")
            if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
                errors.append("legacy source receipt SHA-256 is invalid")
            if not isinstance(received, int) or received <= 0:
                errors.append("legacy source receipt byte count is invalid")
            if receipt.get("declared_digest_verified") is False:
                errors.append("legacy source receipt declared digest differs")
            if type(receipt.get("raw_local_mirror")) is not bool:
                errors.append("legacy source receipt storage disclosure is invalid")
        for summary_name, receipts_name in (
            ("dataset", "source_receipts"),
            ("source_dataset", "source_receipts"),
            ("target_dataset", "target_receipts"),
        ):
            summary = audit.get(summary_name)
            grouped_receipts = audit.get(receipts_name)
            if not isinstance(summary, dict) or not isinstance(grouped_receipts, list):
                continue
            for receipt in grouped_receipts:
                if not isinstance(receipt, dict) or receipt.get("raw_local_mirror") is not True:
                    continue
                errors.extend(
                    provider_copy_storage_errors(
                        summary.get("source_storage_audit", {}),
                        dataset_id=summary.get("dataset_id"),
                        locator=receipt.get("locator"),
                    )
                )
        return errors
    receipt_groups = {
        "dataset": "source_receipts",
        "source_dataset": "source_receipts",
        "target_dataset": "target_receipts",
    }
    seen: set[tuple[str, str, str, str]] = set()
    for summary_name, receipts_name in receipt_groups.items():
        summary = audit.get(summary_name)
        if not isinstance(summary, dict):
            continue
        dataset_id = summary.get("dataset_id")
        grouped_receipts = audit.get(receipts_name)
        if not isinstance(grouped_receipts, list) or not grouped_receipts:
            errors.append(f"{summary_name} source receipts are absent")
            continue
        if summary.get("receipt_count") != len(grouped_receipts):
            errors.append(f"{summary_name} receipt count differs from its dataset audit")
        expected_metadata = SOURCE_RECEIPT_METADATA.get(str(dataset_id))
        for receipt in grouped_receipts:
            if not isinstance(receipt, dict):
                errors.append(f"{summary_name} contains a non-object source receipt")
                continue
            digest = receipt.get("computed_sha256")
            locator = receipt.get("locator")
            member = receipt.get("member")
            identity = (str(dataset_id), str(locator), str(member), str(digest))
            if identity in seen:
                errors.append(f"duplicate source receipt: {member or locator}")
            seen.add(identity)
            if receipt.get("dataset_id") != dataset_id:
                errors.append(f"{summary_name} receipt dataset identity differs")
            if re.fullmatch(r"[0-9a-f]{64}", str(digest)) is None:
                errors.append(f"invalid SHA-256: {member or locator}")
            received = receipt.get("received_size_bytes")
            declared = receipt.get("declared_size_bytes")
            if not isinstance(received, int) or received <= 0:
                errors.append(f"invalid byte count: {member or locator}")
            if declared is not None and declared != received:
                errors.append(f"declared/received byte count differs: {member or locator}")
            if not isinstance(locator, str) or (
                publication_v4 and not locator.startswith("https://")
            ):
                errors.append(f"invalid source URL: {member or locator}")
            elif dataset_id == "fog_star_v3" and locator != _FOG_SOURCE_URL:
                errors.append("FoG-STAR receipt does not use the frozen provider object")
            elif dataset_id == "har_pmd_v1" and locator != _HAR_PMD_SOURCE_URL:
                errors.append("HAR-PMD receipt does not use the frozen provider archive")
            elif dataset_id == "imu_har_il_v1" and not locator.startswith(_IMU_SOURCE_PREFIX):
                errors.append("IMU-HAR-IL receipt is outside the frozen provider collection")
            elif dataset_id == "sole_harmony_v1" and not locator.startswith(_SOLE_SOURCE_PREFIX):
                errors.append("Sole-HARmony receipt is outside the frozen provider record")
            if member is not None and (not isinstance(member, str) or not member):
                errors.append(f"invalid source member: {member!r}")
            if type(receipt.get("raw_local_mirror")) is not bool:
                errors.append(f"storage disclosure mismatch: {member or locator}")
            if not _timezone_aware_iso8601(receipt.get("accessed_at_utc")) and publication_v4:
                errors.append(f"source access time is absent or invalid: {member or locator}")
            if publication_v4 and any(
                not isinstance(receipt.get(name), str) or not receipt.get(name)
                for name in (
                    "record_url",
                    "dataset_version",
                    "dataset_license",
                    "permissible_redistribution",
                    "evidence_role",
                )
            ):
                errors.append(
                    f"source version/license/role metadata is incomplete: {member or locator}"
                )
            if (
                publication_v4
                and expected_metadata is not None
                and any(
                    receipt.get(name) != expected for name, expected in expected_metadata.items()
                )
            ):
                errors.append(f"frozen source version/license/role differs: {member or locator}")
            if dataset_id == "fog_star_v3" and (
                received != _FOG_SOURCE_SIZE
                or receipt.get("declared_size_bytes") != _FOG_SOURCE_SIZE
                or digest != _FOG_SOURCE_SHA256
                or receipt.get("declared_digest_algorithm") != "md5"
                or receipt.get("declared_digest") != _FOG_SOURCE_MD5
                or receipt.get("computed_declared_digest") != _FOG_SOURCE_MD5
                or receipt.get("declared_digest_verified") is not True
            ):
                errors.append("FoG-STAR receipt differs from the pinned object hash and size")
            verified = receipt.get("declared_digest_verified")
            if verified is False:
                errors.append(f"declared digest mismatch: {member or locator}")
            declared_digest = receipt.get("declared_digest")
            if declared_digest is not None and (
                verified is not True
                or receipt.get("declared_digest_algorithm") != "md5"
                or receipt.get("computed_declared_digest") != declared_digest
            ):
                errors.append(f"declared digest evidence is incomplete: {member or locator}")
            if receipt.get("raw_local_mirror") is True:
                storage: Any = summary.get("source_storage_audit", {})
                errors.extend(
                    provider_copy_storage_errors(storage, dataset_id=dataset_id, locator=locator)
                )
    return errors


def _receipt_objects(audit: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for key, value in audit.items():
        if key.endswith("receipts") and isinstance(value, list):
            receipts.extend(item for item in value if isinstance(item, dict))
    return receipts


def _imu_receipt_inventory_hashes(audit: dict[str, Any]) -> tuple[str, str] | None:
    """Reconstruct stable IMU inventory hashes independently from run receipts."""

    grouped_receipts: list[Any] | None = None
    for summary_name, receipts_name in (
        ("dataset", "source_receipts"),
        ("source_dataset", "source_receipts"),
        ("target_dataset", "target_receipts"),
    ):
        summary = audit.get(summary_name)
        receipts = audit.get(receipts_name)
        if (
            isinstance(summary, dict)
            and summary.get("dataset_id") == "imu_har_il_v1"
            and isinstance(receipts, list)
        ):
            grouped_receipts = receipts
            break
    if not grouped_receipts:
        return None
    source_records: list[dict[str, Any]] = []
    payload_records: list[dict[str, Any]] = []
    for receipt in grouped_receipts:
        if not isinstance(receipt, dict):
            return None
        member = receipt.get("member")
        locator = receipt.get("locator")
        if not isinstance(member, str) or not isinstance(locator, str):
            return None
        parts = PurePosixPath(member).parts
        if (
            len(parts) != 5
            or parts[0] != "HAR_IMU_IL"
            or parts[4] != "Body-WT.csv"
            or not parts[1].startswith("P_")
            or not parts[2].startswith("Repetition_")
            or parts[3] not in {"Walk", "Sit", "Stand"}
            or not locator.startswith(_IMU_SOURCE_PREFIX)
        ):
            return None
        try:
            file_id = int(locator.removeprefix(_IMU_SOURCE_PREFIX))
        except ValueError:
            return None
        source_records.append(
            {
                "participant": parts[1],
                "repetition": parts[2],
                "activity": parts[3],
                "file_id": file_id,
                "filename": member,
                "file_size": receipt.get("declared_size_bytes"),
                "locator": locator,
            }
        )
        payload_records.append(
            {
                "locator": locator,
                "member": member,
                "declared_size_bytes": receipt.get("declared_size_bytes"),
                "received_size_bytes": receipt.get("received_size_bytes"),
                "computed_sha256": receipt.get("computed_sha256"),
            }
        )
    source_records.sort(
        key=lambda item: (
            str(item["participant"]),
            str(item["repetition"]),
            str(item["activity"]),
        )
    )
    payload_records.sort(key=lambda item: (str(item["member"]), str(item["locator"])))
    return canonical_json_sha256(source_records), canonical_json_sha256(payload_records)


def _shared_artifact_contract_errors(
    terminal: dict[str, Any], audit: dict[str, Any], run_directory: Path
) -> list[str]:
    """Bind the self-hashed terminal artifact to its exact data audit and ledgers."""

    errors: list[str] = []
    failed_run = terminal.get("status") == "FAILED_PRESERVED"
    audit_reference = terminal.get("data_audit_artifact")
    audit_path = None
    if isinstance(audit_reference, dict):
        audit_path = _safe_run_artifact_path(run_directory, audit_reference.get("path"))
        audit_sha256 = audit_reference.get("sha256")
        audit_record_sha256 = audit_reference.get("record_sha256")
    else:
        audit_sha256 = None
        audit_record_sha256 = None
    if (
        audit_path is None
        or audit_path.name != "data_audit.json"
        or not _file_sha256_matches(audit_path, audit_sha256)
        or audit_record_sha256 != audit.get("record_sha256")
    ):
        errors.append("terminal artifact does not bind the exact self-hashed data audit")
    for name in ("dataset", "source_dataset", "target_dataset"):
        # A post-audit failure need not duplicate a potentially large summary: the
        # exact self-hashed data_audit.json is already bound above. If it does
        # duplicate one, it must still be exact. Successful results must retain the
        # redundant binding used by all claim-bearing readers.
        if name in terminal:
            if terminal.get(name) != audit.get(name):
                errors.append(f"terminal and data audit {name} differ")
        elif name in audit and not failed_run:
            errors.append(f"terminal and data audit {name} differ")
    for name in (
        "git_at_launch",
        "source_input_manifest",
        "artifact_evidence_status",
        "publication_launch_context",
    ):
        if terminal.get(name) != audit.get(name):
            errors.append(f"terminal and data audit {name} differ")
    manifest = terminal.get("source_input_manifest")
    try:
        expected_contract = (
            _publication_artifact_contract(manifest, audit_reference)
            if isinstance(manifest, dict) and isinstance(audit_reference, dict)
            else None
        )
    except KeyError:
        expected_contract = None
    if expected_contract is None or terminal.get("artifact_contract") != expected_contract:
        errors.append("terminal artifact does not bind the v4 protocol/role/supersession chain")
    summary_owner = audit if failed_run else terminal
    summaries = [
        value
        for name in ("source_dataset", "target_dataset", "dataset")
        if isinstance(value := summary_owner.get(name), dict)
    ]
    if not summaries or terminal.get("artifact_evidence_status") != _external_evidence_status(
        *summaries
    ):
        errors.append("terminal artifact evidence status differs from the frozen dataset role")
    errors.extend(_environment_errors(terminal.get("environment"), manifest))
    binding = terminal.get("publication_launch_context")
    if not isinstance(binding, dict):
        errors.append("publication launch-context binding is absent")
    else:
        expected_context = {
            "schema_version": "1.0.0",
            "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
            "allowed_create_only_output_root": binding.get("allowed_create_only_output_root"),
            "git_at_launch": terminal.get("git_at_launch"),
            "source_input_manifest": terminal.get("source_input_manifest"),
        }
        expected_binding = {
            "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
            "allowed_create_only_output_root": expected_context["allowed_create_only_output_root"],
            "record_sha256": canonical_json_sha256(expected_context),
        }
        if binding != expected_binding:
            errors.append("publication launch-context binding is invalid")
    return errors


def _participant_partition_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for index, item in enumerate(_objects(result)):
        groups = {
            name: set(cast(list[str], item[name]))
            for name in (
                "training_participants",
                "validation_participants",
                "evaluation_participants",
                "source_participants",
                "target_participants",
            )
            if isinstance(item.get(name), list)
        }
        names = sorted(groups)
        for left_index, left in enumerate(names):
            for right in names[left_index + 1 :]:
                overlap = groups[left] & groups[right]
                if overlap:
                    errors.append(f"object-{index}:{left}/{right} overlap={sorted(overlap)[:5]}")
    return errors


def _label_isolation_errors(result: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    forbidden_fragments = (
        "used_before_prediction",
        "used_before_predictions",
        "used_for_training_or_selection",
        "used_for_fit_selection_or_calibration",
        "present_in_modelling_container",
    )
    for index, item in enumerate(_objects(result)):
        for key, value in item.items():
            if any(fragment in key for fragment in forbidden_fragments) and value is not False:
                errors.append(f"object-{index}:{key}={value!r}")
    return errors


def _report_mapping(result: dict[str, Any]) -> dict[str, Any]:
    for name in ("reports", "temporal_reports"):
        value = result.get(name)
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return {}


def _expected_window_count(result: dict[str, Any], audit: dict[str, Any]) -> int | None:
    candidates = (
        result.get("dataset"),
        result.get("target_dataset"),
        audit.get("dataset"),
        audit.get("target_dataset"),
    )
    for candidate in candidates:
        if isinstance(candidate, dict) and isinstance(candidate.get("window_count"), int):
            return int(candidate["window_count"])
    return None


def _scoring_dataset_summary(result: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    """Return the audited dataset whose labels are retained in ``predictions.npz``."""

    for owner, name in (
        (audit, "target_dataset"),
        (audit, "dataset"),
        (result, "target_dataset"),
        (result, "dataset"),
    ):
        value = owner.get(name)
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    return {}


def _expected_class_names(summary: dict[str, Any]) -> tuple[str, ...] | None:
    """Resolve the immutable label-column ontology for a supported evidence dataset."""

    dataset_id = summary.get("dataset_id")
    if dataset_id == "har_pmd_v1":
        return _HAR_PMD_CLASS_NAMES
    if dataset_id in {"fog_star_v3", "imu_har_il_v1", "sole_harmony_v1"}:
        return CORE_CLASS_NAMES
    counts = summary.get("class_window_counts")
    if isinstance(counts, dict) and len(counts) == len(CORE_CLASS_NAMES):
        # Synthetic CI fixtures exercise the same frozen three-column contract.
        return CORE_CLASS_NAMES
    if isinstance(counts, dict) and len(counts) == len(_HAR_PMD_CLASS_NAMES):
        return _HAR_PMD_CLASS_NAMES
    return None


def _expected_method_set(result: dict[str, Any]) -> set[str] | None:
    experiment = result.get("experiment_id")
    if experiment in {
        "cross-dataset-har-rnd-v1",
        "imu-har-il-to-fog-star-zero-shot-v1",
    }:
        return set(_EXTERNAL_CLASSICAL_METHODS)
    if experiment == "har-pmd-native-interface-stress-v1":
        return {"RandomForest-6ch", "RandomForest-N9", "XGBoost-6ch", "XGBoost-N9"}
    if experiment == "participant-balanced-hierarchical-posture-forest-v3":
        return set(_HIERARCHY_VARIANTS)
    if experiment in {
        "cross-dataset-har-neural-controls-v1",
        "har-pmd-native-interface-neural-controls-v1",
    }:
        suffix = result.get("method_suffix")
        if suffix not in {"6ch", "N9"}:
            return set()
        return {f"DeepConvLSTM-{suffix}", f"TinyHAR-{suffix}"}
    if experiment in {
        "sole-harmony-observable-session-temporal-v1",
        "sole-harmony-camera-bout-oracle-temporal-v2",
    }:
        return {
            "XGBoost-6ch-unsmoothed",
            *(f"XGBoost-6ch-causal-probability-w{width}" for width in (3, 5, 7)),
            *(f"XGBoost-6ch-causal-majority-w{width}" for width in (3, 5, 7)),
            *(f"XGBoost-6ch-hysteresis-c{count}" for count in (2, 3)),
        }
    return None


def _top_level_report(
    *,
    labels: np.ndarray[Any, Any],
    probabilities: np.ndarray[Any, Any],
    participants: np.ndarray[Any, Any],
    class_names: tuple[str, ...],
    include_bottom_tail: bool,
) -> dict[str, Any]:
    report = classification_report(
        labels,
        probabilities,
        participants.astype(str).tolist(),
        class_names=class_names,
    )
    if include_bottom_tail:
        values = np.sort(
            np.asarray(
                [float(item["macro_f1"]) for item in report["participants"]],
                dtype=np.float64,
            )
        )
        tail_count = max(1, int(np.ceil(0.30 * values.size)))
        report["primary"]["bottom_30_percent_participant_macro_f1"] = float(
            np.mean(values[:tail_count])
        )
    return report


def _scientific_prediction_payload_errors(
    result: dict[str, Any], audit: dict[str, Any], archive: Any
) -> list[str]:
    """Bind labels, ontology, seed ensembles, and top-level reports to audited evidence."""

    errors: list[str] = []
    summary = _scoring_dataset_summary(result, audit)
    class_names = _expected_class_names(summary)
    if class_names is None:
        return ["prediction ontology is not a supported frozen dataset contract"]
    required = {"labels", "participant_ids", "session_ids", "trial_ids", "window_ids"}
    missing = sorted(required - set(archive.files))
    if missing:
        return [f"required scored identity arrays are absent: {missing}"]
    labels = np.asarray(archive["labels"])
    participants = np.asarray(archive["participant_ids"])
    sample_count = int(labels.size) if labels.ndim == 1 else -1
    identities = {name: np.asarray(archive[name]) for name in sorted(required)}
    if labels.ndim != 1 or not np.issubdtype(labels.dtype, np.integer):
        errors.append("retained labels are not a one-dimensional integer array")
        valid_labels = False
    else:
        valid_labels = bool(
            labels.size > 0 and int(labels.min()) >= 0 and int(labels.max()) < len(class_names)
        )
        if not valid_labels:
            errors.append("retained labels lie outside the frozen ontology")
    if any(values.shape != (sample_count,) for values in identities.values()):
        errors.append("scored identity arrays are not label-aligned")
    for name in ("participant_ids", "session_ids", "trial_ids", "window_ids"):
        values = identities[name]
        if values.dtype.kind not in "US" or np.any(np.char.str_len(values.astype(str)) == 0):
            errors.append(f"retained {name} are not nonempty strings")
    if len(set(identities["window_ids"].astype(str).tolist())) != sample_count:
        errors.append("retained scored window identifiers are not unique")

    retained_hashes = summary.get("retained_array_hashes")
    if not isinstance(retained_hashes, dict):
        errors.append("audited retained-array hashes are absent")
    else:
        for name, values in identities.items():
            if retained_hashes.get(name) != _array_sha256(values):
                errors.append(f"retained {name} differ from the audited dataset array hash")

    if valid_labels and participants.shape == labels.shape:
        integer_labels = labels.astype(np.int64, copy=False)
        counts = np.bincount(integer_labels, minlength=len(class_names))
        expected_counts = {name: int(counts[index]) for index, name in enumerate(class_names)}
        if summary.get("class_window_counts") != expected_counts:
            errors.append("retained label counts differ from the audited class-window counts")
        participant_names = np.unique(participants.astype(str))
        expected_participant_support = {
            name: int(
                sum(
                    np.any(integer_labels[participants.astype(str) == participant] == index)
                    for participant in participant_names
                )
            )
            for index, name in enumerate(class_names)
        }
        if summary.get("participant_class_support") != expected_participant_support:
            errors.append("retained labels differ from audited participant class support")
        expected_per_participant = {
            participant: {
                name: int(value)
                for name, value in zip(
                    class_names,
                    np.bincount(
                        integer_labels[participants.astype(str) == participant],
                        minlength=len(class_names),
                    ).tolist(),
                    strict=True,
                )
            }
            for participant in participant_names.tolist()
        }
        primary = result.get("primary_seed_averaged")
        if (
            not isinstance(primary, dict)
            or primary.get("participant_class_support") != expected_per_participant
        ):
            errors.append("primary participant-class support differs from retained labels")

    primary = result.get("primary_seed_averaged")
    methods = primary.get("methods") if isinstance(primary, dict) else None
    seeds = primary.get("seeds") if isinstance(primary, dict) else None
    expected_methods = _expected_method_set(result)
    if not isinstance(methods, dict) or not isinstance(seeds, list):
        errors.append("primary seed/method declaration is absent")
        return errors
    if seeds != [11, 23, 47] or any(isinstance(seed, bool) for seed in seeds):
        errors.append("primary seeds differ from the frozen top-level seed contract")
    if expected_methods is None:
        errors.append("experiment identity is outside the frozen publication runner allowlist")
    elif set(methods) != expected_methods:
        errors.append("reported method set differs from the frozen comparison matrix")
    for method in methods.values():
        reports_by_seed = method.get("reports_by_seed") if isinstance(method, dict) else None
        if not isinstance(reports_by_seed, dict):
            errors.append("primary per-seed reports are absent")
            continue
        for report in reports_by_seed.values():
            if not isinstance(report, dict) or tuple(report.get("class_names", ())) != class_names:
                errors.append("primary per-seed report class order differs from frozen ontology")
                break
            per_class = report.get("per_class")
            if not isinstance(per_class, dict) or set(per_class) != set(class_names):
                errors.append("primary per-seed class metrics differ from frozen ontology")
                break

    probability = {
        key.removeprefix("probability__"): np.asarray(archive[key], dtype=np.float64)
        for key in archive.files
        if key.startswith("probability__")
    }
    reports = _report_mapping(result)
    if set(reports) != set(methods):
        errors.append("top-level report methods differ from primary methods")
    include_bottom_tail = (
        result.get("experiment_id") != "participant-balanced-hierarchical-posture-forest-v3"
    )
    reconstructed_reports: dict[str, dict[str, Any]] = {}
    per_seed_probabilities: dict[int, dict[str, np.ndarray[Any, Any]]] = {}
    if valid_labels and participants.shape == labels.shape:
        per_seed_probabilities = {
            int(seed): {
                name: probability[f"seed-{int(seed)}__{name}"]
                for name in methods
                if f"seed-{int(seed)}__{name}" in probability
            }
            for seed in seeds
        }
        for name in methods:
            seed_names = [f"seed-{int(seed)}__{name}" for seed in seeds]
            if (
                any(seed_name not in probability for seed_name in seed_names)
                or name not in probability
            ):
                errors.append(f"seed or ensemble probabilities are absent for method: {name}")
                continue
            seed_values = [probability[seed_name] for seed_name in seed_names]
            if any(values.shape != (sample_count, len(class_names)) for values in seed_values):
                errors.append(f"seed probability columns differ from frozen ontology: {name}")
                continue
            ensemble = np.mean(np.stack(seed_values, axis=0), axis=0)
            if not np.array_equal(probability[name], ensemble):
                errors.append(f"ensemble probability differs from arithmetic seed mean: {name}")
            try:
                reconstructed = _top_level_report(
                    labels=labels.astype(np.int64, copy=False),
                    probabilities=ensemble,
                    participants=participants,
                    class_names=class_names,
                    include_bottom_tail=include_bottom_tail,
                )
            except (TypeError, ValueError) as error:
                errors.append(f"top-level report reconstruction failed for {name}: {error}")
                continue
            reconstructed_reports[name] = reconstructed
            if canonical_json_sha256(reconstructed) != canonical_json_sha256(reports.get(name)):
                errors.append(f"top-level report differs from retained predictions: {name}")

    experiment = result.get("experiment_id")
    if reconstructed_reports and all(
        set(seed_methods) == set(methods) for seed_methods in per_seed_probabilities.values()
    ):
        from inclusive_shift_har.experiments.cross_dataset_har import _paired_bootstrap

        if experiment == "cross-dataset-har-rnd-v1":
            reports_by_seed = {
                str(seed): {
                    method: _top_level_report(
                        labels=labels.astype(np.int64, copy=False),
                        probabilities=values,
                        participants=participants,
                        class_names=class_names,
                        include_bottom_tail=True,
                    )
                    for method, values in seed_methods.items()
                }
                for seed, seed_methods in per_seed_probabilities.items()
            }
            comparison_methods = (
                "CTGR-DG",
                "CAGE-DG",
                "CTGR-DG-top3-equal",
                "HERA-DG-strict",
                "HERA-DG-full",
                "HERA-DG-v2-dual",
                "HERA-DG-v2-full",
            )
            comparisons = {
                method: _paired_bootstrap(
                    reconstructed_reports[method],
                    reconstructed_reports["RMRP-DG"],
                    seed=20260904 + index,
                )
                for index, method in enumerate(comparison_methods)
            }
            v2 = comparisons["HERA-DG-v2-full"]
            gate = {
                "method": "HERA-DG-v2-full",
                "comparator": "RMRP-DG",
                "minimum_mean_gain": 0.020,
                "observed_mean_gain": v2["mean_difference"],
                "mean_gain_passed": float(v2["mean_difference"]) >= 0.020,
                "mean_interval_lower_bound_above_zero": float(
                    v2["mean_difference_95_percent_bootstrap_interval"][0]
                )
                > 0.0,
                "bottom_interval_lower_bound_above_zero": float(
                    v2["bottom_30_percent_difference_95_percent_bootstrap_interval"][0]
                )
                > 0.0,
                "overall_passed": bool(
                    float(v2["mean_difference"]) >= 0.020
                    and float(v2["mean_difference_95_percent_bootstrap_interval"][0]) > 0.0
                    and float(v2["bottom_30_percent_difference_95_percent_bootstrap_interval"][0])
                    > 0.0
                ),
                "interpretation": "development advancement only; never a confirmatory or SOTA gate",
            }
            invention_names = _EXTERNAL_CLASSICAL_METHODS[:-2]
            best_invention = max(
                invention_names,
                key=lambda name: float(
                    reconstructed_reports[name]["primary"]["mean_participant_macro_f1"]
                ),
            )
            control_names = _EXTERNAL_CLASSICAL_METHODS[-2:]
            strongest_control = max(
                control_names,
                key=lambda name: float(
                    reconstructed_reports[name]["primary"]["mean_participant_macro_f1"]
                ),
            )
            benchmark = {
                "best_invention": best_invention,
                "strongest_control": strongest_control,
                "paired_participant_bootstrap": _paired_bootstrap(
                    reconstructed_reports[best_invention],
                    reconstructed_reports[strongest_control],
                    seed=20261004,
                ),
                "selection_note": (
                    "descriptive post-evaluation best-method comparison; not a multiplicity-"
                    "controlled superiority test"
                ),
            }
            reconstructed_fields = {
                "reports_by_seed": reports_by_seed,
                "paired_participant_bootstrap": comparisons,
                "development_advancement_gate": gate,
                "descriptive_benchmark_comparison": benchmark,
            }
            changed = [
                name
                for name, value in reconstructed_fields.items()
                if canonical_json_sha256(value) != canonical_json_sha256(result.get(name))
            ]
            if changed:
                errors.append(
                    "classical secondary reports/comparisons differ from retained predictions: "
                    + ", ".join(changed)
                )
        elif experiment == "imu-har-il-to-fog-star-zero-shot-v1":
            invention_names = _EXTERNAL_CLASSICAL_METHODS[:-2]
            comparisons = {
                method: _paired_bootstrap(
                    reconstructed_reports[method],
                    reconstructed_reports["RMRP-DG"],
                    seed=20261104 + index,
                )
                for index, method in enumerate(
                    name for name in invention_names if name != "RMRP-DG"
                )
            }
            best_invention = max(
                invention_names,
                key=lambda name: float(
                    reconstructed_reports[name]["primary"]["mean_participant_macro_f1"]
                ),
            )
            strongest_control = max(
                _EXTERNAL_CLASSICAL_METHODS[-2:],
                key=lambda name: float(
                    reconstructed_reports[name]["primary"]["mean_participant_macro_f1"]
                ),
            )
            transfer_fields: dict[str, Any] = {
                "paired_participant_bootstrap_vs_rmrp": comparisons,
                "descriptive_best_invention": best_invention,
                "descriptive_strongest_classical_control": strongest_control,
            }
            changed = [
                name
                for name, value in transfer_fields.items()
                if canonical_json_sha256(value) != canonical_json_sha256(result.get(name))
            ]
            if changed:
                errors.append(
                    "transfer comparisons differ from retained predictions: " + ", ".join(changed)
                )
        elif experiment == "har-pmd-native-interface-stress-v1":
            sessions = identities["session_ids"].astype(str)
            environment_reports: dict[str, dict[str, Any]] = {}
            for environment in ("indoor", "outdoor"):
                selected = np.char.endswith(sessions, f":{environment}")
                environment_reports[environment] = {
                    method: _top_level_report(
                        labels=labels[selected].astype(np.int64, copy=False),
                        probabilities=probability[method][selected],
                        participants=participants[selected],
                        class_names=class_names,
                        include_bottom_tail=True,
                    )
                    for method in methods
                }
            comparisons = {
                "RandomForest-N9_vs_6ch": _paired_bootstrap(
                    reconstructed_reports["RandomForest-N9"],
                    reconstructed_reports["RandomForest-6ch"],
                    seed=20261201,
                ),
                "XGBoost-N9_vs_6ch": _paired_bootstrap(
                    reconstructed_reports["XGBoost-N9"],
                    reconstructed_reports["XGBoost-6ch"],
                    seed=20261202,
                ),
            }
            if canonical_json_sha256(environment_reports) != canonical_json_sha256(
                result.get("environment_reports")
            ) or canonical_json_sha256(comparisons) != canonical_json_sha256(
                result.get("paired_native_gravity_comparisons")
            ):
                errors.append("HAR-PMD environment reports or gravity comparisons differ")
        elif experiment in {
            "cross-dataset-har-neural-controls-v1",
            "har-pmd-native-interface-neural-controls-v1",
        }:
            suffix = str(result.get("method_suffix"))
            comparison = _paired_bootstrap(
                reconstructed_reports[f"TinyHAR-{suffix}"],
                reconstructed_reports[f"DeepConvLSTM-{suffix}"],
            )
            if canonical_json_sha256(comparison) != canonical_json_sha256(
                result.get("paired_model_comparison")
            ):
                errors.append("neural paired-model comparison differs from retained predictions")
    return errors


def _manifest_commit_errors(
    repository_root: Path,
    commit: object,
    files: dict[str, str],
) -> list[str]:
    """Compare retained launch hashes with Git blobs, never current source alone."""

    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        return ["missing or invalid launch commit"]
    if not files or any(
        PurePosixPath(name).is_absolute()
        or ".." in PurePosixPath(name).parts
        or "\n" in name
        or "\\" in name
        or ":" in name
        for name in files
    ):
        return ["missing or unsafe manifest file paths"]
    names = sorted(files)
    response = subprocess.run(
        ["git", "cat-file", "--batch"],
        cwd=repository_root,
        input="".join(f"{commit}:{name}\n" for name in names).encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if response.returncode:
        return ["launch commit objects unavailable in this repository"]
    errors: list[str] = []
    offset = 0
    for name in names:
        end = response.stdout.find(b"\n", offset)
        header = response.stdout[offset:end].split()
        if len(header) != 3 or header[1] != b"blob":
            errors.append(f"launch source blob unavailable: {name}")
            offset = end + 1
            continue
        size = int(header[2])
        blob = response.stdout[end + 1 : end + 1 + size]
        if hashlib.sha256(blob).hexdigest() != files[name]:
            errors.append(f"launch manifest differs from committed source: {name}")
        offset = end + 1 + size + 1
    return errors


def _valid_partition_plan(plan: dict[str, Any], dataset_id: str) -> bool:
    """Validate the exact deterministic plan payload emitted by the loader."""

    roster = plan.get("participant_roster")
    roster_basis = plan.get("roster_basis")
    if (
        plan.get("protocol_id") != PARTITION_PROTOCOL_ID
        or plan.get("dataset_id") != dataset_id
        or plan.get("created_before_windowing") is not True
        or not isinstance(roster, list)
        or len(roster) < 5
        or any(not isinstance(item, str) or not item for item in roster)
        or roster != sorted(set(roster))
        or not isinstance(roster_basis, str)
        or not roster_basis
    ):
        return False
    try:
        expected = build_participant_partition_plan(
            dataset_id,
            roster,
            roster_basis=roster_basis,
        ).audit()
    except ValueError:
        return False
    return plan == expected


def _hierarchy_baseline_reconstruction_errors(result: dict[str, Any]) -> list[str]:
    """Fail closed unless the retained strongest control was reproduced exactly."""

    if result.get("experiment_id") != "participant-balanced-hierarchical-posture-forest-v3":
        return []
    errors: list[str] = []
    baseline = result.get("baseline_reconstruction")
    if not isinstance(baseline, dict):
        return ["hierarchy baseline reconstruction contract is absent"]
    if baseline.get("gate_passed") is not True:
        errors.append("hierarchy baseline reconstruction gate did not pass")
    if baseline.get("scientific_input_tuple_exact") is not True:
        errors.append("hierarchy baseline scientific input tuple is not exact")
    reference_run = baseline.get("reference_run")
    if not isinstance(reference_run, dict) or set(reference_run) != {"path_kind", "path"}:
        errors.append("hierarchy baseline reference-run location is malformed")
    else:
        path_kind = reference_run.get("path_kind")
        path_value = reference_run.get("path")
        if path_kind == "repository_relative" and isinstance(path_value, str):
            parsed = PurePosixPath(path_value)
            valid_location = bool(
                path_value == "."
                or (
                    parsed.parts
                    and not parsed.is_absolute()
                    and ".." not in parsed.parts
                    and "." not in parsed.parts
                    and "\\" not in path_value
                    and ":" not in path_value
                    and parsed.as_posix() == path_value
                )
            )
        elif path_kind == "external_absolute" and isinstance(path_value, str):
            windows = PureWindowsPath(path_value)
            posix = PurePosixPath(path_value)
            valid_location = bool(
                ".." not in windows.parts
                and "." not in windows.parts
                and (
                    (windows.is_absolute() and "/" not in path_value and str(windows) == path_value)
                    or (
                        posix.is_absolute()
                        and "\\" not in path_value
                        and posix.as_posix() == path_value
                    )
                )
            )
        else:
            valid_location = False
        if not valid_location:
            errors.append("hierarchy baseline reference-run location is unsafe or noncanonical")
    seeds = result.get("seeds")
    per_seed = baseline.get("per_seed")
    expected_seeds = {str(seed) for seed in seeds} if isinstance(seeds, list) else set()
    if not expected_seeds or not isinstance(per_seed, dict) or set(per_seed) != expected_seeds:
        errors.append("hierarchy baseline reconstruction does not cover every declared seed")
        return errors
    for seed in sorted(expected_seeds):
        record = per_seed[seed]
        if not isinstance(record, dict):
            errors.append(f"hierarchy baseline reconstruction record is malformed: seed {seed}")
            continue
        if record.get("class_decisions_exact") is not True:
            errors.append(f"hierarchy baseline class decisions differ: seed {seed}")
        if record.get("probabilities_within_strict_tolerance") is not True:
            errors.append(f"hierarchy baseline probabilities differ: seed {seed}")
        tolerance = record.get("probability_absolute_tolerance")
        maximum = record.get("maximum_probability_absolute_difference")
        tolerance_value = (
            float(tolerance)
            if isinstance(tolerance, (int, float)) and not isinstance(tolerance, bool)
            else float("nan")
        )
        maximum_value = (
            float(maximum)
            if isinstance(maximum, (int, float)) and not isinstance(maximum, bool)
            else float("nan")
        )
        valid_tolerance = (
            isinstance(tolerance, (int, float))
            and not isinstance(tolerance, bool)
            and tolerance_value == 1e-12
        )
        if not valid_tolerance:
            errors.append(f"hierarchy baseline probability tolerance differs: seed {seed}")
        valid_maximum = (
            isinstance(maximum, (int, float))
            and not isinstance(maximum, bool)
            and np.isfinite(maximum_value)
            and 0.0 <= maximum_value <= 1e-12
            and (not valid_tolerance or maximum_value <= tolerance_value)
        )
        if not valid_maximum:
            errors.append(
                f"hierarchy baseline maximum probability difference exceeds tolerance: seed {seed}"
            )
    return errors


def _hierarchy_baseline_probability_record(
    reference: np.ndarray[Any, Any], reconstructed: np.ndarray[Any, Any]
) -> dict[str, Any]:
    """Recompute the exact probability-bearing hierarchy baseline comparison."""

    shape_matches = bool(
        reference.ndim == 2
        and reconstructed.ndim == 2
        and reference.shape == reconstructed.shape
        and reference.shape[1] == 3
    )
    finite = bool(np.isfinite(reference).all() and np.isfinite(reconstructed).all())
    maximum_difference = (
        float(np.max(np.abs(reference - reconstructed)))
        if shape_matches and reference.size and finite
        else float("inf")
    )
    return {
        "class_decisions_exact": bool(
            shape_matches
            and finite
            and np.array_equal(reference.argmax(axis=1), reconstructed.argmax(axis=1))
        ),
        "probabilities_within_strict_tolerance": bool(
            shape_matches and finite and np.allclose(reference, reconstructed, rtol=0.0, atol=1e-12)
        ),
        "probability_absolute_tolerance": 1e-12,
        "maximum_probability_absolute_difference": maximum_difference,
    }


def _neural_runtime_artifact_errors(
    result: dict[str, Any], audit: dict[str, Any], run_directory: Path
) -> list[str]:
    """Verify frozen CUDA execution, canonical lane identity, and all checkpoints."""

    experiment_id = result.get("experiment_id")
    if experiment_id not in {
        "cross-dataset-har-neural-controls-v1",
        "har-pmd-native-interface-neural-controls-v1",
    }:
        return []
    errors: list[str] = []
    if result.get("runtime_backend_protocol") != "external-neural-cuda-nocudnn-v2":
        errors.append("neural result does not bind the frozen CUDA backend protocol")
    expected_backend_core = {
        "device_type": "cuda",
        "amp_enabled": True,
        "autocast_device_type": "cuda",
        "autocast_dtype": "float16",
        "gradient_scaler_enabled": True,
    }
    backend = result.get("runtime_backend")
    backend_identity_valid = bool(
        isinstance(backend, dict)
        and all(backend.get(name) == value for name, value in expected_backend_core.items())
        and isinstance(backend.get("cuda_runtime_version"), str)
        and bool(backend.get("cuda_runtime_version"))
        and isinstance(backend.get("cuda_device_name"), str)
        and bool(backend.get("cuda_device_name"))
    )
    if result.get("device") != "cuda" or not backend_identity_valid:
        errors.append("neural result does not record actual CUDA float16 AMP execution")
    expected_training = {
        "epochs": 40,
        "batch_size": 128,
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "patience": 8,
        "minimum_epochs": 8,
        "mixed_precision": "float16",
        "outer_folds": 5,
        "models": ["deepconvlstm", "tinyhar"],
    }
    if result.get("training_contract") != expected_training:
        errors.append("neural training budget/configuration differs from the frozen protocol")

    summary = audit.get("dataset", result.get("dataset"))
    representation = result.get("input_representation")
    if not isinstance(summary, dict) or not isinstance(representation, dict):
        errors.append("neural canonical input-representation contract is absent")
    else:
        dataset_id = summary.get("dataset_id")
        source_field = representation.get("source_field")
        expected_experiment = (
            "har-pmd-native-interface-neural-controls-v1"
            if dataset_id == "har_pmd_v1"
            else "cross-dataset-har-neural-controls-v1"
        )
        retained_hashes = summary.get("retained_array_hashes")
        pool = summary.get("observable_candidate_pool")
        pool_hashes = pool.get("arrays") if isinstance(pool, dict) else None
        expected_channels = 6 if source_field == "signals" else 9
        expected_suffix = "6ch" if source_field == "signals" else "N9"
        expected_scored_hash = (
            retained_hashes.get(source_field) if isinstance(retained_hashes, dict) else None
        )
        expected_full_hash = (
            pool_hashes.get(source_field) if isinstance(pool_hashes, dict) else expected_scored_hash
        )
        expected_gravity = (
            summary.get("gravity_preprocessing", {}).get("source")
            if source_field == "nine_channel_signals"
            and isinstance(summary.get("gravity_preprocessing"), dict)
            else None
        )
        if (
            source_field not in {"signals", "nine_channel_signals"}
            or experiment_id != expected_experiment
            or result.get("method_suffix") != expected_suffix
            or result.get("input_channels") != expected_channels
            or representation.get("method_suffix") != expected_suffix
            or representation.get("input_channels") != expected_channels
            or representation.get("array_sha256") != expected_full_hash
            or representation.get("scored_source_array_sha256") != expected_scored_hash
            or representation.get("gravity_source") != expected_gravity
            or (
                source_field == "nine_channel_signals"
                and (
                    dataset_id != "har_pmd_v1"
                    or summary.get("channel_lane") != "native-gravity-9ch"
                )
            )
        ):
            errors.append(
                "neural lane/channels/gravity provenance/hash do not match the canonical dataset representation"
            )

    seeds = result.get("seeds")
    suffix = result.get("method_suffix")
    records = result.get("fold_records")
    expected_models = {f"DeepConvLSTM-{suffix}", f"TinyHAR-{suffix}"}
    expected_keys = {
        (seed, fold, model)
        for seed in (11, 23, 47)
        for fold in range(5)
        for model in expected_models
    }
    seen_keys: set[tuple[Any, Any, Any]] = set()
    checkpoint_paths: set[str] = set()
    if seeds != [11, 23, 47] or not isinstance(records, list) or len(records) != 30:
        errors.append("neural fold records do not cover the frozen 3x5x2 design")
        records = records if isinstance(records, list) else []
    for record in records:
        if not isinstance(record, dict):
            errors.append("neural fold record is malformed")
            continue
        key = (record.get("seed"), record.get("outer_fold"), record.get("model"))
        seen_keys.add(key)
        model_slug = (
            "deepconvlstm"
            if str(record.get("model", "")).startswith("DeepConvLSTM-")
            else "tinyhar"
        )
        seed_value = record.get("seed")
        fold_value = record.get("outer_fold")
        config_seed = (
            seed_value + 101 * fold_value
            if isinstance(seed_value, int)
            and not isinstance(seed_value, bool)
            and isinstance(fold_value, int)
            and not isinstance(fold_value, bool)
            else None
        )
        expected_config = {
            "model_name": model_slug,
            "seed": config_seed,
            "epochs": 40,
            "batch_size": 128,
            "learning_rate": 3e-4,
            "weight_decay": 1e-4,
            "patience": 8,
            "minimum_epochs": 8,
            "mixed_precision": "float16",
            "disable_cudnn": model_slug == "deepconvlstm",
        }
        backend_ok = all(
            record.get(name) == value for name, value in expected_backend_core.items()
        ) and bool(
            isinstance(backend, dict)
            and record.get("cuda_runtime_version") == backend.get("cuda_runtime_version")
            and record.get("cuda_device_name") == backend.get("cuda_device_name")
        )
        if (
            record.get("device") != "cuda"
            or not backend_ok
            or record.get("disable_cudnn") is not expected_config["disable_cudnn"]
            or record.get("mixed_precision") != "float16"
            or record.get("training_config") != expected_config
            or record.get("training_config_sha256") != canonical_json_sha256(expected_config)
        ):
            errors.append(f"neural fold CUDA/configuration evidence differs: {key}")
        checkpoint = record.get("checkpoint")
        expected_relative = (
            f"checkpoints/seed-{record.get('seed')}__fold-{record.get('outer_fold')}__"
            f"{model_slug}.pt"
        )
        relative = checkpoint.get("path") if isinstance(checkpoint, dict) else None
        path = _safe_nested_run_artifact_path(
            run_directory, relative, required_parent="checkpoints"
        )
        try:
            actual_size = path.stat().st_size if path is not None and path.is_file() else None
        except OSError:
            actual_size = None
        if (
            relative != expected_relative
            or relative in checkpoint_paths
            or path is None
            or not path.is_file()
            or not isinstance(checkpoint, dict)
            or not _file_sha256_matches(
                path, checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            )
            or not isinstance(checkpoint, dict)
            or checkpoint.get("size_bytes") != actual_size
        ):
            errors.append(
                f"neural checkpoint artifact is missing, unsafe, or hash-mismatched: {key}"
            )
        if isinstance(relative, str):
            checkpoint_paths.add(relative)
    if seen_keys != expected_keys:
        errors.append("neural seed/fold/model records are missing, duplicated, or unexpected")
    checkpoint_root = run_directory / "checkpoints"
    actual_paths = (
        {
            path.relative_to(run_directory).as_posix()
            for path in checkpoint_root.glob("*.pt")
            if path.is_file() and not path.is_symlink()
        }
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink()
        else set()
    )
    if actual_paths != checkpoint_paths or len(actual_paths) != 30:
        errors.append("neural checkpoint directory does not contain exactly the 30 bound files")
    return errors


_HIERARCHY_VARIANTS = (
    "RandomForest-6ch",
    "PB-RF-6ch",
    "PB-RF-D9",
    "HPF-unweighted",
    "PB-HPF",
    "PB-HPF-shuffled-posture",
)


def _hierarchy_artifact_errors(
    result: dict[str, Any], audit: dict[str, Any], run_directory: Path
) -> list[str]:
    """Bind every hierarchy checkpoint and fold archive without loading pickle data."""

    if result.get("experiment_id") != "participant-balanced-hierarchical-posture-forest-v3":
        return []
    errors: list[str] = []
    summary = audit.get("dataset", result.get("dataset"))
    pool = summary.get("observable_candidate_pool") if isinstance(summary, dict) else None
    contract = result.get("candidate_prediction_contract")
    prediction = result.get("prediction_artifact")
    main_path = (
        _safe_run_artifact_path(run_directory, prediction.get("path"))
        if isinstance(prediction, dict)
        else None
    )
    main_arrays: dict[str, np.ndarray[Any, Any]] = {}
    required_main = {
        "labels",
        "participant_ids",
        "session_ids",
        "trial_ids",
        "window_ids",
        "candidate_participant_ids",
        "candidate_window_ids",
        "candidate_scoring_indices",
        "candidate_scoring_eligibility",
    }
    required_main.update(
        f"probability__seed-{seed}__{method}"
        for seed in (11, 23, 47)
        for method in _HIERARCHY_VARIANTS
    )
    if main_path is None or not main_path.is_file():
        errors.append("hierarchy primary prediction archive is absent")
    else:
        try:
            with np.load(main_path, allow_pickle=False) as archive:
                missing = required_main - set(archive.files)
                if missing:
                    errors.append(
                        f"hierarchy primary candidate arrays are absent: {sorted(missing)}"
                    )
                else:
                    main_arrays = {name: np.asarray(archive[name]).copy() for name in required_main}
        except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
            errors.append(
                f"hierarchy primary candidate archive is unreadable: {type(error).__name__}: {error}"
            )
    if main_arrays:
        candidate_ids = main_arrays["candidate_window_ids"]
        candidate_people = main_arrays["candidate_participant_ids"]
        scored_ids = main_arrays["window_ids"]
        scored_people = main_arrays["participant_ids"]
        labels = main_arrays["labels"]
        indices = main_arrays["candidate_scoring_indices"]
        eligibility = main_arrays["candidate_scoring_eligibility"]
        candidate_count = int(candidate_ids.size)
        scored_count = int(scored_ids.size)
        indices_valid = bool(
            indices.ndim == 1
            and np.issubdtype(indices.dtype, np.integer)
            and indices.shape == (scored_count,)
            and (indices.size == 0 or (int(indices[0]) >= 0 and int(indices[-1]) < candidate_count))
            and (indices.size < 2 or np.all(np.diff(indices) > 0))
        )
        expected_eligibility = np.zeros(candidate_count, dtype=np.bool_)
        if indices_valid:
            expected_eligibility[indices.astype(np.int64)] = True
        if (
            candidate_ids.shape != (candidate_count,)
            or candidate_people.shape != (candidate_count,)
            or scored_people.shape != (scored_count,)
            or labels.shape != (scored_count,)
            or eligibility.dtype != np.dtype(np.bool_)
            or eligibility.shape != (candidate_count,)
            or not indices_valid
            or not np.array_equal(eligibility, expected_eligibility)
            or not np.array_equal(scored_ids, candidate_ids[indices.astype(np.int64)])
            or not np.array_equal(scored_people, candidate_people[indices.astype(np.int64)])
            or len(set(candidate_ids.astype(str).tolist())) != candidate_count
            or len(set(scored_ids.astype(str).tolist())) != scored_count
        ):
            errors.append("hierarchy primary scored/candidate identity slice is invalid")
        pool_arrays = pool.get("arrays") if isinstance(pool, dict) else None
        actual_identifier_hashes = {
            "participant_ids": _array_sha256(candidate_people),
            "window_ids": _array_sha256(candidate_ids),
        }
        if (
            not isinstance(contract, dict)
            or contract.get("candidate_window_count") != candidate_count
            or contract.get("scored_window_count") != scored_count
            or contract.get("scoring_indices_sha256") != canonical_json_sha256(indices.tolist())
            or contract.get("scoring_indices_array_sha256") != _array_sha256(indices)
            or contract.get("scoring_eligibility_array_sha256") != _array_sha256(eligibility)
            or contract.get("candidate_identifier_hashes") != actual_identifier_hashes
            or not isinstance(pool, dict)
            or pool.get("window_count") != candidate_count
            or not isinstance(pool_arrays, dict)
            or pool_arrays.get("participant_ids") != actual_identifier_hashes["participant_ids"]
            or pool_arrays.get("window_ids") != actual_identifier_hashes["window_ids"]
            or (isinstance(summary, dict) and summary.get("window_count") != scored_count)
        ):
            errors.append("hierarchy primary candidate contract differs from the dataset pool")

    baseline = result.get("baseline_reconstruction")
    witness_record = (
        baseline.get("retained_reference_witness") if isinstance(baseline, dict) else None
    )
    witness_relative = witness_record.get("path") if isinstance(witness_record, dict) else None
    witness_path = _safe_run_artifact_path(run_directory, witness_relative)
    try:
        witness_size = (
            witness_path.stat().st_size
            if witness_path is not None and witness_path.is_file()
            else None
        )
    except OSError:
        witness_size = None
    witness_descriptor_valid = bool(
        isinstance(baseline, dict)
        and isinstance(witness_record, dict)
        and witness_relative == "baseline_reference_predictions.npz"
        and witness_path is not None
        and _file_sha256_matches(witness_path, witness_record.get("sha256"))
        and witness_record.get("size_bytes") == witness_size
        and re.fullmatch(r"[0-9a-f]{64}", str(baseline.get("reference_result_sha256"))) is not None
        and re.fullmatch(r"[0-9a-f]{64}", str(baseline.get("reference_predictions_sha256")))
        is not None
    )
    if not witness_descriptor_valid:
        errors.append("hierarchy retained baseline-reference witness is missing or unbound")
    elif main_arrays:
        assert witness_path is not None
        try:
            with np.load(witness_path, allow_pickle=False) as witness:
                reference_names = {
                    f"probability__seed-{seed}__RandomForest-6ch" for seed in (11, 23, 47)
                }
                witness_required = {
                    "labels",
                    "participant_ids",
                    "session_ids",
                    "trial_ids",
                    "window_ids",
                    *reference_names,
                }
                if set(witness.files) != witness_required:
                    errors.append("hierarchy retained baseline-reference witness members differ")
                else:
                    metadata_match = all(
                        np.array_equal(witness[name], main_arrays[name])
                        for name in (
                            "labels",
                            "participant_ids",
                            "session_ids",
                            "trial_ids",
                            "window_ids",
                        )
                    )
                    recomputed = {
                        str(seed): _hierarchy_baseline_probability_record(
                            np.asarray(witness[f"probability__seed-{seed}__RandomForest-6ch"]),
                            main_arrays[f"probability__seed-{seed}__RandomForest-6ch"],
                        )
                        for seed in (11, 23, 47)
                    }
                    if (
                        not metadata_match
                        or not isinstance(baseline, dict)
                        or canonical_json_sha256(recomputed)
                        != canonical_json_sha256(baseline.get("per_seed"))
                        or not all(
                            record["class_decisions_exact"]
                            and record["probabilities_within_strict_tolerance"]
                            for record in recomputed.values()
                        )
                    ):
                        errors.append(
                            "hierarchy retained baseline witness differs from reconstructed control"
                        )
        except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
            errors.append(
                "hierarchy retained baseline-reference witness is unreadable: "
                f"{type(error).__name__}: {error}"
            )

    records = result.get("fold_records")
    expected_record_keys = {
        (seed, fold, method)
        for seed in (11, 23, 47)
        for fold in range(5)
        for method in _HIERARCHY_VARIANTS
    }
    records_by_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    if result.get("seeds") != [11, 23, 47] or not isinstance(records, list) or len(records) != 90:
        errors.append("hierarchy fold records do not cover the frozen 3x5x6 design")
        records = records if isinstance(records, list) else []
    checkpoint_paths: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            errors.append("hierarchy fold record is malformed")
            continue
        key = (record.get("seed"), record.get("outer_fold"), record.get("method"))
        if key in records_by_key:
            errors.append(f"hierarchy fold record is duplicated: {key}")
        records_by_key[key] = record
        expected_relative = f"checkpoints/seed-{key[0]}__fold-{key[1]}__{key[2]}.pickle"
        checkpoint = record.get("checkpoint")
        relative = checkpoint.get("path") if isinstance(checkpoint, dict) else None
        path = _safe_nested_run_artifact_path(
            run_directory, relative, required_parent="checkpoints"
        )
        try:
            size = path.stat().st_size if path is not None and path.is_file() else None
        except OSError:
            size = None
        if (
            relative != expected_relative
            or relative in checkpoint_paths
            or path is None
            or not _file_sha256_matches(
                path, checkpoint.get("sha256") if isinstance(checkpoint, dict) else None
            )
            or not isinstance(checkpoint, dict)
            or checkpoint.get("size_bytes") != size
            or checkpoint.get("trusted_local_pickle_only") is not True
        ):
            errors.append(f"hierarchy checkpoint is missing, unsafe, or hash-mismatched: {key}")
        if isinstance(relative, str):
            checkpoint_paths.add(relative)
    if set(records_by_key) != expected_record_keys:
        errors.append("hierarchy seed/fold/method records are missing or unexpected")
    checkpoint_root = run_directory / "checkpoints"
    actual_checkpoints = (
        {
            path.relative_to(run_directory).as_posix()
            for path in checkpoint_root.glob("*.pickle")
            if path.is_file() and not path.is_symlink()
        }
        if checkpoint_root.is_dir() and not checkpoint_root.is_symlink()
        else set()
    )
    if actual_checkpoints != checkpoint_paths or len(actual_checkpoints) != 90:
        errors.append("hierarchy checkpoint directory does not contain exactly 90 bound files")

    artifacts = result.get("fold_artifacts")
    artifacts_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    if not isinstance(artifacts, list) or len(artifacts) != 15:
        errors.append("hierarchy fold-artifact inventory does not cover the frozen 3x5 design")
        artifacts = artifacts if isinstance(artifacts, list) else []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            errors.append("hierarchy fold-artifact record is malformed")
            continue
        fold_key = (artifact.get("seed"), artifact.get("outer_fold"))
        if fold_key in artifacts_by_key:
            errors.append(f"hierarchy fold-artifact record is duplicated: {fold_key}")
        artifacts_by_key[fold_key] = artifact
        expected_archive = f"seed-{fold_key[0]}__fold-{fold_key[1]}__predictions.npz"
        expected_marker = f"seed-{fold_key[0]}__fold-{fold_key[1]}__complete.json"
        archive_record = artifact.get("prediction_archive")
        marker_record = artifact.get("completion_marker")
        archive_relative = archive_record.get("path") if isinstance(archive_record, dict) else None
        marker_relative = marker_record.get("path") if isinstance(marker_record, dict) else None
        archive_path = _safe_run_artifact_path(run_directory, archive_relative)
        marker_path = _safe_run_artifact_path(run_directory, marker_relative)
        for expected_relative, descriptor, path, kind in (
            (expected_archive, archive_record, archive_path, "prediction archive"),
            (expected_marker, marker_record, marker_path, "completion marker"),
        ):
            try:
                size = path.stat().st_size if path is not None and path.is_file() else None
            except OSError:
                size = None
            if (
                not isinstance(descriptor, dict)
                or descriptor.get("path") != expected_relative
                or not _file_sha256_matches(path, descriptor.get("sha256"))
                or descriptor.get("size_bytes") != size
            ):
                errors.append(
                    f"hierarchy {kind} is missing, unsafe, or hash-mismatched: {fold_key}"
                )
        if marker_path is not None and marker_path.is_file():
            try:
                marker = _read_object(marker_path)
                if (
                    marker.get("seed") != fold_key[0]
                    or marker.get("fold") != fold_key[1]
                    or marker.get("all_variants_completed") != list(_HIERARCHY_VARIANTS)
                    or not isinstance(archive_record, dict)
                    or marker.get("prediction_archive_sha256") != archive_record.get("sha256")
                ):
                    errors.append(f"hierarchy completion marker semantics differ: {fold_key}")
            except (OSError, UnicodeError, ValueError, TypeError) as error:
                errors.append(
                    f"hierarchy completion marker is unreadable: {fold_key}: {type(error).__name__}"
                )
        if archive_path is not None and archive_path.is_file() and main_arrays:
            try:
                with np.load(archive_path, allow_pickle=False) as archive:
                    candidate_people = np.asarray(archive["candidate_participant_ids"])
                    candidate_ids = np.asarray(archive["candidate_window_ids"])
                    eligibility = np.asarray(archive["candidate_scoring_eligibility"])
                    scored_people = np.asarray(archive["participant_ids"])
                    scored_ids = np.asarray(archive["window_ids"])
                    labels = np.asarray(archive["labels"])
                    evaluation_records = [
                        records_by_key.get((fold_key[0], fold_key[1], method))
                        for method in _HIERARCHY_VARIANTS
                    ]
                    if any(item is None for item in evaluation_records):
                        errors.append(f"hierarchy fold record linkage is incomplete: {fold_key}")
                        continue
                    linked = cast(list[dict[str, Any]], evaluation_records)
                    evaluation_people = linked[0].get("evaluation_participants")
                    if not isinstance(evaluation_people, list):
                        errors.append(
                            f"hierarchy evaluation participant list is absent: {fold_key}"
                        )
                        continue
                    candidate_mask = np.isin(
                        main_arrays["candidate_participant_ids"], evaluation_people
                    )
                    scored_mask = np.isin(main_arrays["participant_ids"], evaluation_people)
                    expected_candidate_ids = main_arrays["candidate_window_ids"][candidate_mask]
                    expected_candidate_people = main_arrays["candidate_participant_ids"][
                        candidate_mask
                    ]
                    expected_eligibility = main_arrays["candidate_scoring_eligibility"][
                        candidate_mask
                    ]
                    expected_scored_ids = main_arrays["window_ids"][scored_mask]
                    expected_scored_people = main_arrays["participant_ids"][scored_mask]
                    expected_labels = main_arrays["labels"][scored_mask]
                    if (
                        any(
                            item.get("evaluation_participants") != evaluation_people
                            for item in linked
                        )
                        or any(
                            item.get("evaluation_candidate_window_count")
                            != int(expected_candidate_ids.size)
                            or item.get("evaluation_scored_window_count")
                            != int(expected_scored_ids.size)
                            for item in linked
                        )
                        or eligibility.dtype != np.dtype(np.bool_)
                        or not np.array_equal(candidate_ids, expected_candidate_ids)
                        or not np.array_equal(candidate_people, expected_candidate_people)
                        or not np.array_equal(eligibility, expected_eligibility)
                        or not np.array_equal(scored_ids, expected_scored_ids)
                        or not np.array_equal(scored_people, expected_scored_people)
                        or not np.array_equal(labels, expected_labels)
                        or not np.array_equal(scored_ids, candidate_ids[eligibility])
                        or not np.array_equal(scored_people, candidate_people[eligibility])
                    ):
                        errors.append(f"hierarchy fold identity/scoring slice differs: {fold_key}")
                    for method in _HIERARCHY_VARIANTS:
                        scored_name = f"probability__{method}"
                        candidate_name = f"candidate_probability__{method}"
                        if scored_name not in archive.files or candidate_name not in archive.files:
                            errors.append(
                                f"hierarchy fold probabilities are absent: {fold_key}: {method}"
                            )
                            continue
                        scored_probability = np.asarray(archive[scored_name])
                        candidate_probability = np.asarray(archive[candidate_name])
                        if (
                            candidate_probability.shape != (candidate_ids.size, 3)
                            or scored_probability.shape != (scored_ids.size, 3)
                            or not np.isfinite(candidate_probability).all()
                            or not np.isfinite(scored_probability).all()
                            or np.any(candidate_probability < -1e-9)
                            or np.any(candidate_probability > 1.0 + 1e-9)
                            or np.any(scored_probability < -1e-9)
                            or np.any(scored_probability > 1.0 + 1e-9)
                            or not np.allclose(
                                candidate_probability.sum(axis=1), 1.0, atol=1e-6, rtol=0.0
                            )
                            or not np.array_equal(
                                scored_probability, candidate_probability[eligibility]
                            )
                            or not np.array_equal(
                                scored_probability,
                                main_arrays[f"probability__seed-{fold_key[0]}__{method}"][
                                    scored_mask
                                ],
                            )
                        ):
                            errors.append(
                                f"hierarchy fold probability slice differs: {fold_key}: {method}"
                            )
            except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
                errors.append(
                    f"hierarchy fold archive is unreadable: {fold_key}: {type(error).__name__}: {error}"
                )
    expected_fold_keys = {(seed, fold) for seed in (11, 23, 47) for fold in range(5)}
    if set(artifacts_by_key) != expected_fold_keys:
        errors.append("hierarchy fold archives/markers are missing or unexpected")
    expected_direct_files = {
        descriptor["path"]
        for artifact in artifacts_by_key.values()
        for name in ("prediction_archive", "completion_marker")
        if isinstance(descriptor := artifact.get(name), dict)
        and isinstance(descriptor.get("path"), str)
    }
    actual_direct_files = {
        path.name
        for pattern in ("seed-*__fold-*__predictions.npz", "seed-*__fold-*__complete.json")
        for path in run_directory.glob(pattern)
        if path.is_file() and not path.is_symlink()
    }
    if actual_direct_files != expected_direct_files or len(actual_direct_files) != 30:
        errors.append("hierarchy run does not contain exactly 15 bound archives and markers")
    return errors


def _fog_physics_reference_record_errors(
    record: dict[str, Any], expected_training_candidates: int
) -> list[str]:
    """Require auditable signal-only HERA physics-reference fit-population evidence."""

    errors: list[str] = []
    candidate_count = record.get("physics_reference_fit_candidate_window_count")
    valid_count = record.get("physics_reference_fit_signal_valid_window_count")
    threshold = record.get("physics_reference_threshold")
    if (
        record.get("physics_reference_fit_population") != "all_observable_outer_training_candidates"
        or candidate_count != expected_training_candidates
    ):
        errors.append(
            "FoG physics reference was not fit on every observable outer-training candidate"
        )
    if (
        type(valid_count) is not int
        or valid_count <= 0
        or valid_count > expected_training_candidates
    ):
        errors.append("FoG physics-reference signal-valid fit count is invalid")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not np.isfinite(threshold)
        or float(threshold) <= 0.0
    ):
        errors.append("FoG physics-reference threshold is absent or invalid")
    return errors


def _method_contract_errors(result: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    result_manifest = result.get("source_input_manifest")
    audit_manifest = audit.get("source_input_manifest")
    publication_v4 = any(
        isinstance(manifest, dict) and manifest.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
        for manifest in (result_manifest, audit_manifest)
    )
    summaries = [
        value
        for key in ("dataset", "source_dataset", "target_dataset")
        if isinstance(value := audit.get(key, result.get(key)), dict)
    ]
    if not summaries or any("dataset_id" not in item for item in summaries):
        errors.append("dataset identity and preprocessing contract unavailable")
    for summary in summaries:
        dataset_id = summary.get("dataset_id")
        if str(dataset_id) not in SOURCE_RECEIPT_METADATA:
            errors.append(f"{dataset_id} is absent from the frozen evidence-role receipt registry")
        boundary = summary.get("boundary_provenance")
        if dataset_id in {
            "fog_star_v3",
            "imu_har_il_v1",
            "har_pmd_v1",
            "sole_harmony_v1",
        }:
            if not isinstance(boundary, dict):
                errors.append(f"{dataset_id} boundary provenance is absent")
            elif boundary.get("zero_lookahead_streaming_valid") is not False:
                errors.append(f"{dataset_id} overstates zero-lookahead streaming validity")
            plan = summary.get("participant_partition_plan")
            if not isinstance(plan, dict):
                errors.append(f"{dataset_id} pre-window participant partition plan is absent")
            else:
                roster = plan.get("participant_roster")
                plan_digest = plan.get("plan_sha256")
                if not _valid_partition_plan(plan, str(dataset_id)):
                    errors.append(f"{dataset_id} pre-window participant plan is invalid")
                observation = summary.get("participant_partition_observation")
                if (
                    isinstance(roster, list)
                    and all(isinstance(item, str) and item for item in roster)
                    and isinstance(observation, dict)
                ):
                    absent = observation.get("participants_without_retained_windows")
                    observation_valid = (
                        isinstance(absent, list)
                        and all(isinstance(item, str) and item for item in absent)
                        and absent == sorted(set(absent))
                        and set(absent).issubset(set(roster))
                        and observation.get("planned_participant_count") == len(roster)
                        and observation.get("observed_window_participant_count")
                        == len(roster) - len(absent)
                        and summary.get("participant_count") == len(roster) - len(absent)
                    )
                else:
                    observation_valid = False
                if not observation_valid:
                    errors.append(f"{dataset_id} planned/observed participant audit is invalid")
                recorded_hashes = {
                    value
                    for item in _objects(result)
                    for key, value in item.items()
                    if key.endswith("participant_partition_plan_sha256")
                }
                if plan_digest not in recorded_hashes:
                    errors.append(
                        f"{dataset_id} executed folds do not bind the pre-window plan hash"
                    )
        if dataset_id == "har_pmd_v1":
            if not _har_pmd_full_cohort_observed(summary):
                errors.append(
                    "HAR-PMD publication stress evidence requires all 120 planned participants to contribute windows"
                )
        if dataset_id == "imu_har_il_v1":
            if publication_v4 and not _imu_har_il_expected_cohort_observed(summary):
                errors.append(
                    "IMU-HAR-IL publication diagnostics require the exact frozen 50-person "
                    "provider inventory and the declared 19-person complete-core or "
                    "47-person available-valid-trial missing-data lane"
                )
            if publication_v4:
                receipt_hashes = _imu_receipt_inventory_hashes(audit)
                cohort = summary.get("cohort_audit")
                if not isinstance(cohort, dict) or receipt_hashes != (
                    cohort.get("source_inventory_sha256"),
                    cohort.get("source_payload_inventory_sha256"),
                ):
                    errors.append(
                        "IMU-HAR-IL recorded cohort inventory hashes do not reproduce "
                        "from the exact source receipts"
                    )
            if not isinstance(boundary, dict) or (
                boundary.get("provider_upstream_annotation_conditioned") is not True
                or boundary.get("repository_signal_grid_annotation_independent") is not False
            ):
                errors.append("IMU-HAR-IL provider label-boundary conditioning is undisclosed")
            errors.append(
                "IMU-HAR-IL public source is provider-label-presegmented; continuous-stream publication claim blocked"
            )
        if dataset_id == "sole_harmony_v1" and (
            not isinstance(boundary, dict)
            or boundary.get("repository_signal_grid_annotation_independent") is not True
            or boundary.get("provider_upstream_annotation_conditioned") is not False
        ):
            errors.append("Sole-HARmony camera-bout preprocessing is an oracle diagnostic")
        if summary.get("dataset_id") != "fog_star_v3":
            continue
        if publication_v4 and not _fog_star_full_cohort_observed(summary):
            errors.append(
                "FoG-STAR publication evidence requires the exact 22-person provider roster "
                "and all planned participants to contribute scored windows"
            )
        fog_plan = summary.get("participant_partition_plan")
        fog_roster = fog_plan.get("participant_roster") if isinstance(fog_plan, dict) else None
        reported_methods = result.get("primary_seed_averaged", {}).get(
            "methods", result.get("reports", {})
        )
        if (
            isinstance(reported_methods, dict)
            and set(reported_methods) & ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS
            and result.get("observable_context_training_protocol")
            != OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
        ):
            errors.append(
                "FoG context-dependent methods do not bind the frozen observable-training population"
            )
        if not isinstance(boundary, dict) or (
            boundary.get("repository_signal_grid_annotation_independent") is not True
            or boundary.get("provider_upstream_annotation_conditioned") is not False
        ):
            errors.append(
                "FoG boundary provenance does not establish repository annotation isolation"
            )
        segments = summary.get("preprocessing_audit", [])
        admitted_total = 0
        if not segments:
            errors.append(
                "FoG annotation-independent resampling evidence absent; historical run superseded"
            )
        for segment in segments:
            if (
                segment.get("protocol_id") != "external-har-session-grid-v3"
                or segment.get("within_declared_segment_transform_annotation_dependency")
                is not False
                or segment.get("segment_boundary_annotation_conditioned") is not False
                or segment.get("resampling_passes") != 1
            ):
                errors.append("FoG physical segment does not satisfy the v3 signal contract")
            starts = segment.get("candidate_start_samples", [])
            admitted = segment.get("admitted_candidate_indices", [])
            rejected = segment.get("excluded_candidate_indices", {})
            accounted = list(admitted) + [
                index for indices in rejected.values() for index in indices
            ]
            if sorted(accounted) != list(range(len(starts))) or len(starts) != segment.get(
                "candidate_window_count"
            ):
                errors.append("FoG candidate admission/exclusion accounting is incomplete")
            if starts != [
                index * int(segment.get("window_samples", 0)) for index in range(len(starts))
            ]:
                errors.append(
                    "FoG candidate grid is not globally uniform within its physical segment"
                )
            admitted_total += len(admitted)
        if admitted_total != summary.get("window_count"):
            errors.append("FoG admitted candidates do not match retained windows")
        observable_protocol_declared = (
            result.get("observable_context_protocol") == OBSERVABLE_CONTEXT_PROTOCOL
        )
        if not observable_protocol_declared:
            errors.append("FoG result does not bind the annotation-independent candidate pool")
        if observable_protocol_declared:
            pool = summary.get("observable_candidate_pool", {})
            if (
                pool.get("protocol_id") != OBSERVABLE_CONTEXT_PROTOCOL
                or pool.get("annotation_fields_present") is not False
                or pool.get("window_count")
                != sum(item.get("candidate_window_count", 0) for item in segments)
                or sum(pool.get("participant_window_counts", {}).values())
                != pool.get("window_count")
                or any(
                    re.fullmatch(r"[0-9a-f]{64}", str(pool.get("arrays", {}).get(name))) is None
                    for name in (
                        "signals",
                        "gravity",
                        "participant_ids",
                        "session_ids",
                        "trial_ids",
                        "window_ids",
                    )
                )
            ):
                errors.append("FoG observable inference pool is incomplete or label-bearing")
            if "target_dataset" in result:
                records = result.get("seed_records", [])
            else:
                fold_records = result.get("fold_records", [])
                records = (
                    [fold for seed_record in fold_records for fold in seed_record.get("folds", [])]
                    if fold_records
                    and all(
                        isinstance(seed_record, dict) and "folds" in seed_record
                        for seed_record in fold_records
                    )
                    else fold_records
                )
            if not records:
                errors.append("FoG observable inference population is not recorded per fit")
            require_full_training_context = (
                "target_dataset" not in result
                and result.get("experiment_id") == "cross-dataset-har-rnd-v1"
            )
            for record in records:
                transfer = "target_dataset" in result
                names = record.get(
                    "target_participants" if transfer else "evaluation_participants", []
                )
                if not isinstance(names, list):
                    errors.append("FoG held-out participant candidate audit is malformed")
                    names = []
                expected = sum(
                    pool.get("participant_window_counts", {}).get(name, 0) for name in names
                )
                actual = record.get(
                    "target_inference_candidate_window_count"
                    if transfer
                    else "evaluation_candidate_window_count"
                )
                if expected <= 0 or actual != expected:
                    errors.append(
                        "FoG fit did not evaluate every observable candidate of its held-out participants"
                    )
                if not require_full_training_context:
                    continue
                training_names = record.get("training_participants")
                training_with_candidates = record.get("training_participants_with_candidates")
                training_with_supervision = record.get("training_participants_with_supervision")
                if not isinstance(training_names, list) or not isinstance(
                    training_with_candidates, list
                ):
                    errors.append("FoG outer training candidate participant audit is absent")
                    continue
                expected_training = sum(
                    pool.get("participant_window_counts", {}).get(name, 0)
                    for name in training_names
                )
                expected_training_people = sorted(
                    name
                    for name in training_names
                    if pool.get("participant_window_counts", {}).get(name, 0) > 0
                )
                if (
                    set(training_names) & set(names)
                    or set(training_names) | set(names) != set(fog_roster or [])
                    or training_with_candidates != expected_training_people
                    or record.get("training_candidate_window_count") != expected_training
                    or expected_training + expected != pool.get("window_count")
                ):
                    errors.append(
                        "FoG outer training/evaluation candidates do not partition the observable pool"
                    )
                if (
                    not isinstance(training_with_supervision, list)
                    or record.get("responder_training_participants") != training_with_supervision
                    or record.get("responder_context_participants_without_supervision")
                    != sorted(set(training_with_candidates) - set(training_with_supervision))
                ):
                    errors.append(
                        "FoG responder context/supervision populations are not explicitly separated"
                    )
                if (
                    record.get("channel_scale_fit_population")
                    != "all_observable_outer_training_candidates"
                    or record.get("channel_scale_fit_candidate_window_count") != expected_training
                ):
                    errors.append(
                        "FoG context-feature scale was not fit on every outer-training candidate"
                    )
                errors.extend(_fog_physics_reference_record_errors(record, expected_training))
                inner_folds = record.get("inner_folds")
                if not isinstance(inner_folds, list) or not inner_folds:
                    errors.append("FoG inner validation candidate coverage is absent")
                    continue
                inner_validation_people: list[str] = []
                inner_candidate_count = 0
                for inner in inner_folds:
                    if not isinstance(inner, dict) or not isinstance(
                        inner.get("validation_participants"), list
                    ):
                        errors.append("FoG inner validation candidate coverage is malformed")
                        continue
                    inner_names = inner["validation_participants"]
                    inner_expected = sum(
                        pool.get("participant_window_counts", {}).get(name, 0)
                        for name in inner_names
                    )
                    if inner.get("validation_candidate_window_count") != inner_expected:
                        errors.append(
                            "FoG inner fold did not predict every observable validation candidate"
                        )
                    inner_validation_people.extend(inner_names)
                    inner_candidate_count += inner_expected
                if (
                    len(inner_validation_people) != len(set(inner_validation_people))
                    or set(inner_validation_people) != set(training_names)
                    or inner_candidate_count != expected_training
                ):
                    errors.append(
                        "FoG inner validation folds do not partition outer-training candidates"
                    )
    primary_seed_evidence = result.get("primary_seed_averaged")
    if publication_v4 and (
        result.get("seeds") != [11, 23, 47]
        or any(isinstance(seed, bool) for seed in result.get("seeds", []))
        or not isinstance(primary_seed_evidence, dict)
        or primary_seed_evidence.get("seeds") != [11, 23, 47]
        or any(isinstance(seed, bool) for seed in primary_seed_evidence.get("seeds", []))
    ):
        errors.append("frozen three-seed participant-averaged evidence is missing")
    expected_methods = _expected_method_set(result)
    primary_methods = result.get("primary_seed_averaged", {}).get("methods")
    if publication_v4 and expected_methods is None:
        errors.append("experiment identity is outside the frozen publication runner allowlist")
    elif expected_methods is not None and (
        not isinstance(primary_methods, dict) or set(primary_methods) != expected_methods
    ):
        errors.append("reported method set differs from the frozen comparison matrix")
    if result.get("runtime_backend_protocol") == "external-neural-cuda-nocudnn-v2":
        records = result.get("fold_records", [])
        if not records:
            errors.append("external neural backend records are missing")
        for record in records:
            expected_disabled = str(record.get("model", "")).startswith("DeepConvLSTM-")
            if record.get("disable_cudnn") is not expected_disabled:
                errors.append("neural fold differs from the declared recurrent backend policy")
            if record.get("mixed_precision") != "float16":
                errors.append("neural fold differs from inherited float16 AMP policy")
    errors.extend(_hierarchy_baseline_reconstruction_errors(result))
    return errors


def _diagnostic_scope_reasons(result: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    summaries = [
        value
        for key in ("dataset", "source_dataset", "target_dataset")
        if isinstance(value := audit.get(key, result.get(key)), dict)
    ]
    for summary in summaries:
        dataset_id = summary.get("dataset_id")
        boundary = summary.get("boundary_provenance")
        if dataset_id == "imu_har_il_v1":
            reasons.append(
                "IMU-HAR-IL uses provider label-derived activity segments; original continuous chronology is unavailable"
            )
        if dataset_id == "sole_harmony_v1":
            if not isinstance(boundary, dict) or (
                boundary.get("repository_signal_grid_annotation_independent") is not True
            ):
                reasons.append("Sole-HARmony uses camera-annotation boundaries and is oracle-only")
            else:
                reasons.append(
                    "Sole-HARmony session-observable temporal evaluation remains consumed development diagnostic evidence"
                )
    return sorted(set(reasons))


def _sole_observable_prediction_errors(
    result: dict[str, Any], audit: dict[str, Any], archive: Any
) -> list[str]:
    """Verify Sole temporal predictions against the complete observable candidate stream."""

    summary = audit.get("dataset", result.get("dataset"))
    if not isinstance(summary, dict) or summary.get("dataset_id") != "sole_harmony_v1":
        return []
    errors: list[str] = []
    contract = result.get("prediction_contract")
    if not isinstance(contract, dict):
        return ["Sole observable prediction contract is absent"]
    required_identifiers = {
        "participant_ids": "observable_participant_ids",
        "session_ids": "observable_session_ids",
        "trial_ids": "observable_trial_ids",
        "window_ids": "observable_window_ids",
    }
    required_arrays = {
        *required_identifiers.values(),
        *required_identifiers.keys(),
        "labels",
        "observable_scoring_indices",
        "observable_scoring_eligibility",
    }
    archive_names = set(archive.files)
    missing_arrays = sorted(required_arrays - archive_names)
    if missing_arrays:
        errors.append(f"Sole observable archive arrays are absent: {missing_arrays}")
        return errors

    candidate_identifiers = {
        name: np.asarray(archive[key]) for name, key in required_identifiers.items()
    }
    scored_identifiers = {name: np.asarray(archive[name]) for name in required_identifiers}
    labels = np.asarray(archive["labels"])
    indices = np.asarray(archive["observable_scoring_indices"])
    eligibility = np.asarray(archive["observable_scoring_eligibility"])
    candidate_count = int(candidate_identifiers["window_ids"].size)
    scored_count = int(labels.size)

    if any(values.shape != (candidate_count,) for values in candidate_identifiers.values()):
        errors.append("Sole observable identifiers are not candidate-aligned")
    if any(values.shape != (scored_count,) for values in scored_identifiers.values()):
        errors.append("Sole scored identifiers are not label-aligned")
    if any(
        values.dtype.kind not in "US" or np.any(np.char.str_len(values.astype(str)) == 0)
        for values in (*candidate_identifiers.values(), *scored_identifiers.values())
    ):
        errors.append("Sole observable/scored identifiers are not nonempty strings")
    if len(
        set(candidate_identifiers["window_ids"].astype(str).tolist())
    ) != candidate_count or np.any(
        np.char.str_len(candidate_identifiers["window_ids"].astype(str)) == 0
    ):
        errors.append("Sole observable window identifiers are empty or non-unique")
    valid_indices = bool(
        indices.ndim == 1
        and np.issubdtype(indices.dtype, np.integer)
        and indices.size == scored_count
        and (indices.size == 0 or (int(indices[0]) >= 0 and int(indices[-1]) < candidate_count))
        and (indices.size < 2 or np.all(np.diff(indices) > 0))
    )
    if not valid_indices:
        errors.append("Sole scoring indices are not unique, ordered, bounded integers")
    if eligibility.dtype != np.dtype(np.bool_) or eligibility.shape != (candidate_count,):
        errors.append("Sole observable scoring eligibility is not a candidate-aligned bool mask")
        valid_eligibility = False
    else:
        expected_eligibility = np.zeros(candidate_count, dtype=np.bool_)
        if valid_indices:
            expected_eligibility[indices.astype(np.int64)] = True
        valid_eligibility = bool(
            valid_indices and np.array_equal(eligibility, expected_eligibility)
        )
        if not valid_eligibility:
            errors.append("Sole observable scoring eligibility differs from the scoring indices")
    if valid_indices:
        for name in required_identifiers:
            if not np.array_equal(
                scored_identifiers[name], candidate_identifiers[name][indices.astype(np.int64)]
            ):
                errors.append(f"Sole scored {name} differ from the observable scoring slice")

    if (
        contract.get("scored_window_count") != scored_count
        or summary.get("window_count") != scored_count
    ):
        errors.append("Sole scored-window counts differ between archive, result, and dataset")
    pool = summary.get("observable_candidate_pool")
    boundary = summary.get("boundary_provenance")
    boundary_mode = boundary.get("boundary_mode") if isinstance(boundary, dict) else None
    expected_experiment = {
        "session_observable": "sole-harmony-observable-session-temporal-v1",
        "camera_bout_oracle": "sole-harmony-camera-bout-oracle-temporal-v2",
    }.get(str(boundary_mode))
    if expected_experiment is None or result.get("experiment_id") != expected_experiment:
        errors.append("Sole experiment identity differs from its audited boundary mode")
    observable = boundary_mode == "session_observable"
    expected_annotation_application = (
        "camera intervals projected after global signal-grid construction"
        if observable
        else "camera interval defines each signal/reset boundary"
    )
    if not isinstance(boundary, dict) or (
        boundary.get("protocol_id") != BOUNDARY_PROVENANCE_PROTOCOL
        or boundary.get("repository_signal_grid_annotation_independent") is not observable
        or boundary.get("provider_upstream_annotation_conditioned") is not False
        or boundary.get("zero_lookahead_streaming_valid") is not False
        or boundary.get("annotation_application") != expected_annotation_application
    ):
        errors.append("Sole boundary provenance differs from the frozen lane contract")
    if contract.get("observable_candidate_window_count") != candidate_count:
        errors.append("Sole observable candidate count differs from the result contract")
    if isinstance(pool, dict):
        if pool.get("window_count") != candidate_count:
            errors.append("Sole observable candidate count differs from the dataset pool")
        if contract.get("observable_candidate_pool_audit_sha256") != canonical_json_sha256(pool):
            errors.append("Sole observable candidate-pool audit hash differs")
        pool_arrays = pool.get("arrays")
        if not isinstance(pool_arrays, dict):
            errors.append("Sole observable candidate-pool array hashes are absent")
        else:
            for name, values in candidate_identifiers.items():
                if pool_arrays.get(name) != _array_sha256(values):
                    errors.append(f"Sole observable {name} differ from the dataset pool hash")
        participant_counts = {
            str(participant): int(np.sum(candidate_identifiers["participant_ids"] == participant))
            for participant in np.unique(candidate_identifiers["participant_ids"])
        }
        if pool.get("participant_window_counts") != participant_counts:
            errors.append("Sole observable participant counts differ from the dataset pool")
    elif boundary_mode == "session_observable":
        errors.append("Sole session-observable dataset candidate-pool audit is absent")
    else:
        if contract.get("observable_candidate_pool_audit_sha256") is not None:
            errors.append("Sole oracle lane claims a nonexistent observable candidate-pool audit")
        if (
            candidate_count != scored_count
            or not valid_indices
            or not np.array_equal(indices, np.arange(scored_count, dtype=indices.dtype))
            or not valid_eligibility
        ):
            errors.append("Sole oracle lane does not retain its complete scored candidate stream")

    if boundary_mode in {"session_observable", "camera_bout_oracle"}:
        # Window identifiers determine every temporal reset. Bind them to the
        # loader's ordered physical-segment audit rather than merely accepting a
        # self-consistent set of mutable prefixes. The oracle branch remains a
        # diagnostic, but its reported camera-bout upper bound must also be exact.
        preprocessing = summary.get("preprocessing_audit")
        expected_identifiers: dict[str, list[str]] = {
            "participant_ids": [],
            "session_ids": [],
            "trial_ids": [],
            "window_ids": [],
        }
        segment_keys: set[tuple[str, str, str, int, int]] = set()
        malformed_segment = not isinstance(preprocessing, list) or not preprocessing
        if isinstance(preprocessing, list):
            for record in preprocessing:
                if not isinstance(record, dict):
                    malformed_segment = True
                    continue
                participant = record.get("participant_id")
                session = record.get("session_id")
                trial = record.get("trial_id")
                run_index = record.get("run_index")
                finite_index_value = record.get("finite_run_index")
                resampled_samples = record.get("resampled_samples")
                window_samples = record.get("window_samples")
                starts = record.get("candidate_start_samples")
                count = record.get("candidate_window_count")
                if (
                    record.get("protocol_id") != PHYSICAL_GRID_PROTOCOL
                    or not all(
                        isinstance(value, str) and bool(value)
                        for value in (participant, session, trial)
                    )
                    or not isinstance(run_index, int)
                    or isinstance(run_index, bool)
                    or run_index < 0
                    or not isinstance(resampled_samples, int)
                    or isinstance(resampled_samples, bool)
                    or resampled_samples < 0
                    or not isinstance(window_samples, int)
                    or isinstance(window_samples, bool)
                    or window_samples < 1
                    or not isinstance(starts, list)
                    or not isinstance(count, int)
                    or isinstance(count, bool)
                    or count < 0
                ):
                    malformed_segment = True
                    continue
                assert isinstance(participant, str)
                assert isinstance(session, str)
                assert isinstance(trial, str)
                if boundary_mode == "session_observable":
                    finite_index = 0
                    trial_valid = trial == f"{session}:session-recording"
                else:
                    finite_index = (
                        finite_index_value
                        if isinstance(finite_index_value, int)
                        and not isinstance(finite_index_value, bool)
                        and finite_index_value >= 0
                        else -1
                    )
                    trial_valid = bool(
                        trial.startswith(f"{session}:camera-bout-")
                        and re.fullmatch(r".*:camera-bout-[0-9]{6}", trial)
                    )
                expected_starts = list(
                    range(0, (resampled_samples // window_samples) * window_samples, window_samples)
                )
                key = (participant, session, trial, run_index, finite_index)
                if (
                    not trial_valid
                    or finite_index < 0
                    or key in segment_keys
                    or starts != expected_starts
                    or count != len(expected_starts)
                    or record.get("candidate_grid_sha256")
                    != _array_sha256(np.asarray(expected_starts, dtype=np.int64))
                    or record.get("within_declared_segment_transform_annotation_dependency")
                    is not False
                    or record.get("segment_boundary_annotation_conditioned")
                    is not (boundary_mode == "camera_bout_oracle")
                    or record.get("resampling_passes") != 1
                ):
                    malformed_segment = True
                segment_keys.add(key)
                for window_index in range(len(expected_starts)):
                    expected_identifiers["participant_ids"].append(participant)
                    expected_identifiers["session_ids"].append(session)
                    expected_identifiers["trial_ids"].append(trial)
                    expected_identifiers["window_ids"].append(
                        f"{participant}/{session}/{trial}/run-{run_index:04d}-"
                        f"finite-{finite_index:03d}/"
                        f"window-{window_index:06d}"
                    )
        if malformed_segment:
            errors.append("Sole temporal physical-segment audit is malformed")
        for name, expected_values in expected_identifiers.items():
            if not np.array_equal(
                candidate_identifiers[name], np.asarray(expected_values, dtype=np.str_)
            ):
                errors.append(
                    "Sole temporal reset/window sequence differs from the audited "
                    f"physical segments: {name}"
                )
    if contract.get("scoring_eligibility_policy") != OBSERVABLE_SCORING_ELIGIBILITY_POLICY:
        errors.append("Sole scoring-eligibility policy differs")
    if valid_indices and contract.get("scoring_indices_sha256") != canonical_json_sha256(
        indices.tolist()
    ):
        errors.append("Sole scoring-index canonical hash differs")
    if contract.get("scoring_indices_array_sha256") != _array_sha256(indices):
        errors.append("Sole scoring-index array hash differs")
    if contract.get("observable_scoring_eligibility_sha256") != _array_sha256(eligibility):
        errors.append("Sole scoring-eligibility array hash differs")
    actual_candidate_hashes = {
        name: _array_sha256(values) for name, values in candidate_identifiers.items()
    }
    actual_scored_hashes = {
        name: _array_sha256(values) for name, values in scored_identifiers.items()
    }
    if contract.get("observable_identifier_array_sha256") != actual_candidate_hashes:
        errors.append("Sole observable identifier hashes differ")
    if contract.get("scored_identifier_array_sha256") != actual_scored_hashes:
        errors.append("Sole scored identifier hashes differ")

    scored_probabilities = {
        key.removeprefix("probability__"): np.asarray(archive[key])
        for key in archive.files
        if key.startswith("probability__")
    }
    observable_probabilities = {
        key.removeprefix("observable_probability__"): np.asarray(archive[key])
        for key in archive.files
        if key.startswith("observable_probability__")
    }
    primary = result.get("primary_seed_averaged")
    methods = primary.get("methods") if isinstance(primary, dict) else None
    seeds = primary.get("seeds") if isinstance(primary, dict) else None
    if not isinstance(methods, dict) or not isinstance(seeds, list):
        errors.append("Sole primary method/seed declaration is absent")
        expected_probability_names: set[str] = set()
    else:
        expected_probability_names = set(methods) | {
            f"seed-{seed}__{method}" for seed in seeds for method in methods
        }
    if set(scored_probabilities) != expected_probability_names:
        errors.append("Sole scored probability names differ from every declared seed/method")
    if set(observable_probabilities) != expected_probability_names:
        errors.append("Sole observable probability names differ from every declared seed/method")
    expected_temporal_contract = {
        "protocol_id": "sole-harmony-observable-session-temporal-v1",
        "boundary_mode": boundary_mode,
        "window_order": (
            "provider participant/session, timestamp-gap or finite-run block, chronological window"
            if observable
            else "provider participant/session, camera-labelled bout, chronological window"
        ),
        "state_reset": (
            "provider session, timestamp discontinuity, or finite-sensor run only"
            if observable
            else "camera-labelled bout or finite sub-run (oracle)"
        ),
        "future_window_access": False,
        "state_updates_before_scoring_slice": True,
        "state_input_population": (
            "all annotation-independent observable candidates"
            if observable
            else "camera-bout-selected oracle candidates"
        ),
        "scoring_eligibility_applied_after_all_temporal_predictions": True,
        "widths_reported_without_outcome_selection": [3, 5, 7],
        "hysteresis_confirmations_reported_without_outcome_selection": [2, 3],
        "primary_candidate": "XGBoost-6ch-causal-probability-w5",
        "primary_control": "XGBoost-6ch-unsmoothed",
        "offline_symmetric_resampling_precludes_zero_lookahead_claim": True,
    }
    if result.get("temporal_contract") != expected_temporal_contract:
        errors.append("Sole temporal contract differs from the frozen lane semantics")
    expected_comparison_contract = {
        "within_lane_matched_methods": sorted(methods) if isinstance(methods, dict) else [],
        "camera_bout_lane_comparable_to_session_observable": False,
        "camera_bout_lane_role": "oracle-boundary upper-bound diagnostic only",
        "cross_lane_before_after_language_allowed": False,
    }
    if result.get("comparison_contract") != expected_comparison_contract:
        errors.append("Sole comparison contract permits invalid cross-lane claims")
    expected_base_contract = {
        "method": "XGBoost-6ch",
        "features": "fixed six-channel engineered-feature interface",
        "outer_folds": 5,
        "hyperparameter_trials": 0,
        "inner_fold_assignments": "retained for split audit but unused by fixed control",
        "held_out_predictions_cover_every_observable_candidate": True,
        "outer_training_participant_labels_used_for_supervised_fit": True,
        "outer_evaluation_participant_labels_used_for_fit_selection_or_calibration": False,
    }
    if result.get("base_model_contract") != expected_base_contract:
        errors.append("Sole base-model contract differs from the fixed label-free control")
    expected_claim_policy = {
        "confirmatory_claim_allowed": False,
        "state_of_the_art_claim_allowed": False,
        "clinical_claim_allowed": False,
        "zero_lookahead_streaming_claim_allowed": False,
        "camera_bout_lane_is_deployable": False,
        "alternate_width_selection_after_primary_failure_allowed": False,
        "camera_annotations_define_preprocessing_boundaries": not observable,
        "evaluation_activity_labels_are_model_features_or_fit_targets": False,
    }
    if result.get("claim_policy") != expected_claim_policy:
        errors.append("Sole claim policy differs from the frozen diagnostic limitations")
    class_counts = summary.get("class_window_counts")
    class_count = len(class_counts) if isinstance(class_counts, dict) else 0
    for name, values in observable_probabilities.items():
        if (
            values.shape != (candidate_count, class_count)
            or not np.isfinite(values).all()
            or np.any(values < -1e-9)
            or np.any(values > 1.0 + 1e-9)
            or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
        ):
            errors.append(f"invalid Sole observable probability matrix: {name}")
    if valid_indices:
        for name in sorted(set(scored_probabilities) & set(observable_probabilities)):
            if not np.array_equal(
                scored_probabilities[name], observable_probabilities[name][indices.astype(np.int64)]
            ):
                errors.append(f"Sole scored probabilities differ from observable slice: {name}")

    # Artifact hashes and matching scored slices cannot establish that a claimed
    # causal method was actually computed from the retained unsmoothed stream.
    # Re-run every fixed temporal transform from that stream, using only the
    # observable physical-run identifiers, and reconstruct both ensembles.
    if isinstance(methods, dict) and isinstance(seeds, list):
        try:
            from types import SimpleNamespace

            from inclusive_shift_har.experiments.sole_harmony_temporal import (
                BASE_METHOD,
                _temporal_block_ids,
                _temporal_methods,
            )

            block_ids = _temporal_block_ids(
                cast(Any, SimpleNamespace(window_ids=candidate_identifiers["window_ids"]))
            )
            derived_by_seed: dict[int, dict[str, np.ndarray[Any, Any]]] = {}
            for declared_seed in seeds:
                seed = int(declared_seed)
                base_name = f"seed-{seed}__{BASE_METHOD}"
                base = observable_probabilities.get(base_name)
                if base is None:
                    errors.append(f"Sole retained unsmoothed stream is absent: {base_name}")
                    continue
                derived = _temporal_methods(np.asarray(base, dtype=np.float64), block_ids)
                derived_by_seed[seed] = derived
                if set(derived) != set(methods):
                    errors.append(
                        "Sole declared temporal methods differ from the fixed transform bank"
                    )
                    continue
                for method, expected_probability in derived.items():
                    name = f"seed-{seed}__{method}"
                    actual = observable_probabilities.get(name)
                    if actual is None or not np.array_equal(actual, expected_probability):
                        errors.append(
                            "Sole observable temporal probability is not independently "
                            f"derived from the retained unsmoothed stream: {name}"
                        )

            if len(derived_by_seed) == len(seeds) and derived_by_seed:
                ordered_seeds = [int(seed) for seed in seeds]
                for method in methods:
                    expected_observable_ensemble = np.mean(
                        np.stack([derived_by_seed[seed][method] for seed in ordered_seeds], axis=0),
                        axis=0,
                    )
                    observable_ensemble = observable_probabilities.get(str(method))
                    if observable_ensemble is None or not np.array_equal(
                        observable_ensemble, expected_observable_ensemble
                    ):
                        errors.append(
                            "Sole observable ensemble differs from the arithmetic seed mean: "
                            f"{method}"
                        )
                    if valid_indices:
                        expected_scored_ensemble = expected_observable_ensemble[
                            indices.astype(np.int64)
                        ]
                        scored_ensemble = scored_probabilities.get(str(method))
                        if scored_ensemble is None or not np.array_equal(
                            scored_ensemble, expected_scored_ensemble
                        ):
                            errors.append(
                                "Sole scored ensemble differs from the independently derived "
                                f"observable seed mean: {method}"
                            )
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"Sole temporal transform reconstruction failed: {exc}")
    actual_observable_probability_hashes = {
        name: _array_sha256(values) for name, values in observable_probabilities.items()
    }
    actual_scored_probability_hashes = {
        name: _array_sha256(values) for name, values in scored_probabilities.items()
    }
    if contract.get("observable_probability_array_sha256") != actual_observable_probability_hashes:
        errors.append("Sole observable probability hashes differ")
    if contract.get("scored_probability_array_sha256") != actual_scored_probability_hashes:
        errors.append("Sole scored probability hashes differ")
    if contract.get("observable_probabilities_retained_for_every_seed_and_method") is not True:
        errors.append("Sole full-stream probability-retention declaration is absent")
    if contract.get("scored_probabilities_retained_for_every_seed_and_method") is not True:
        errors.append("Sole scored probability-retention declaration is absent")
    fold_records = result.get("fold_records")
    if not isinstance(fold_records, list) or not fold_records:
        errors.append("Sole temporal fold records are absent")
        fold_records = []
    for record in fold_records:
        if not isinstance(record, dict) or (
            record.get("temporal_state_input_window_count") != candidate_count
            or record.get("temporal_scoring_slice_window_count") != scored_count
            or record.get("temporal_state_updated_on_every_observable_candidate") is not True
            or record.get("temporal_labels_or_eligibility_used_before_state_update") is not False
            or record.get("boundary_mode") != boundary_mode
        ):
            errors.append("Sole temporal fold record does not bind the full observable slice")
            break
        folds = record.get("folds")
        if (
            not isinstance(folds, list)
            or not folds
            or any(
                not isinstance(fold, dict)
                or fold.get("temporal_state_updated_on_every_evaluation_candidate") is not True
                or fold.get("evaluation_scoring_eligibility_used_before_prediction") is not False
                for fold in folds
            )
        ):
            errors.append("Sole nested fold does not bind prediction-before-scoring semantics")
            break
    return errors


def _metric_evidence_errors(result: dict[str, Any], prediction_path: Path | None) -> list[str]:
    """Recompute the complete primary report from each retained seed, not its hash alone."""

    primary = result.get("primary_seed_averaged")
    if not isinstance(primary, dict) or prediction_path is None or not prediction_path.is_file():
        return ["primary metrics cannot be reconstructed from retained seed predictions"]
    try:
        methods = primary["methods"]
        seeds = primary["seeds"]
        dataset = result.get("target_dataset", result.get("dataset", {}))
        expected_class_names = _expected_class_names(
            cast(dict[str, Any], dataset) if isinstance(dataset, dict) else {}
        )
        if expected_class_names is None:
            return ["primary metric reconstruction lacks a frozen dataset ontology"]
        sole_inputs: (
            tuple[dict[int, np.ndarray[Any, Any]], np.ndarray[Any, Any], np.ndarray[Any, Any]]
            | None
        ) = None
        with np.load(prediction_path, allow_pickle=False) as archive:
            inputs = ParticipantMetricInputs(
                dataset_id=dataset["dataset_id"],
                class_names=expected_class_names,
                labels=archive["labels"],
                participant_ids=archive["participant_ids"],
            )
            probabilities = {
                seed: {
                    method: np.asarray(
                        archive[f"probability__seed-{seed}__{method}"], dtype=np.float64
                    )
                    for method in methods
                }
                for seed in seeds
            }
            if result.get("experiment_id") in {
                "sole-harmony-observable-session-temporal-v1",
                "sole-harmony-camera-bout-oracle-temporal-v2",
            }:
                sole_inputs = (
                    {
                        int(seed): np.asarray(
                            archive[f"observable_probability__seed-{seed}__XGBoost-6ch-unsmoothed"],
                            dtype=np.float64,
                        ).copy()
                        for seed in seeds
                    },
                    np.asarray(archive["observable_window_ids"]).copy(),
                    np.asarray(archive["observable_scoring_indices"]).copy(),
                )
        recomputed, _ = seed_evidence(
            inputs, probabilities, primary_contrast_eligible="target_dataset" not in result
        )
        if canonical_json_sha256(recomputed) != canonical_json_sha256(primary):
            return ["primary participant statistics differ from retained predictions"]
        if result.get("experiment_id") == "participant-balanced-hierarchical-posture-forest-v3":
            from inclusive_shift_har.experiments.hierarchical_posture import (
                posture_advancement_gate,
            )

            if canonical_json_sha256(posture_advancement_gate(recomputed)) != canonical_json_sha256(
                result.get("advancement_gate")
            ):
                return ["posture R&D gate differs from independently reconstructed statistics"]
        if sole_inputs is not None:
            from types import SimpleNamespace

            from inclusive_shift_har.experiments.sole_harmony_temporal import (
                _advancement_gate,
                _paired_primary_comparison,
                _participant_statistical_supplement,
                _shuffled_time_probability,
                _temporal_block_ids,
            )

            observable_base, observable_window_ids, scoring_indices = sole_inputs
            supplement = _participant_statistical_supplement(cast(Any, inputs), probabilities)
            comparison = _paired_primary_comparison(recomputed)
            block_ids = _temporal_block_ids(
                cast(Any, SimpleNamespace(window_ids=observable_window_ids))
            )
            participant_names = np.unique(inputs.participant_ids)
            class_count = len(inputs.class_names)

            def shuffled_score(per_seed: dict[int, np.ndarray[Any, Any]]) -> float:
                participant_scores: list[float] = []
                for model_seed in sorted(per_seed):
                    predicted = per_seed[model_seed].argmax(axis=1)
                    for participant in participant_names:
                        selected = inputs.participant_ids == participant
                        truth = inputs.labels[selected]
                        guess = predicted[selected]
                        matrix = np.bincount(
                            truth * class_count + guess, minlength=class_count**2
                        ).reshape(class_count, class_count)
                        true_positive = np.diag(matrix).astype(np.float64)
                        denominator = (
                            2.0 * true_positive
                            + matrix.sum(axis=0).astype(np.float64)
                            - true_positive
                            + matrix.sum(axis=1).astype(np.float64)
                            - true_positive
                        )
                        f1 = np.divide(
                            2.0 * true_positive,
                            denominator,
                            out=np.zeros(class_count, dtype=np.float64),
                            where=denominator != 0.0,
                        )
                        participant_scores.append(float(f1.mean()))
                return float(np.mean(participant_scores))

            shuffled: dict[str, Any] = {}
            for width in (3, 5, 7):
                replicate_scores: list[float] = []
                schedule: list[dict[str, Any]] = []
                for replicate in range(100):
                    shuffled_scored: dict[int, np.ndarray[Any, Any]] = {}
                    random_seeds: dict[str, int] = {}
                    for model_seed in seeds:
                        random_seed = (
                            202_609_050 + int(width) * 100_000 + int(model_seed) * 1_000 + replicate
                        )
                        random_seeds[str(model_seed)] = random_seed
                        full = _shuffled_time_probability(
                            observable_base[int(model_seed)],
                            block_ids,
                            width=width,
                            seed=random_seed,
                        )
                        shuffled_scored[int(model_seed)] = full[scoring_indices.astype(np.int64)]
                    replicate_scores.append(shuffled_score(shuffled_scored))
                    schedule.append(
                        {"replicate": replicate, "random_seeds_by_model_seed": random_seeds}
                    )
                values = np.asarray(replicate_scores, dtype=np.float64)
                causal_name = f"XGBoost-6ch-causal-probability-w{width}"
                causal_score = float(
                    recomputed["methods"][causal_name]["mean_participant_macro_f1"]
                )
                shuffled[f"width_{width}"] = {
                    "replicate_count": 100,
                    "replicate_seed_averaged_participant_macro_f1": replicate_scores,
                    "random_seed_schedule": schedule,
                    "causal_mean_participant_macro_f1": causal_score,
                    "shuffled_mean": float(values.mean()),
                    "shuffled_median": float(np.median(values)),
                    "shuffled_95_percent_interval": np.quantile(values, [0.025, 0.975]).tolist(),
                    "causal_minus_shuffled_median": causal_score - float(np.median(values)),
                    "permutation_tail_fraction_shuffled_at_least_causal": float(
                        (1 + np.sum(values >= causal_score)) / (1 + values.size)
                    ),
                    "interpretation": (
                        "predeclared within-observable-block negative control; replicates are not "
                        "independent participants and the tail fraction is not a confirmatory p-value"
                    ),
                }
            boundary = result.get("dataset", {}).get("boundary_provenance", {})
            boundary_mode = boundary.get("boundary_mode") if isinstance(boundary, dict) else None
            gate = _advancement_gate(
                boundary_mode=str(boundary_mode),
                primary=recomputed,
                supplement=supplement,
                comparison=comparison,
                shuffled=shuffled,
            )
            reconstructed_fields = {
                "participant_statistical_supplement": supplement,
                "primary_paired_participant_bootstrap": comparison,
                "shuffled_time_negative_controls": shuffled,
                "advancement_gate": gate,
            }
            mismatches = [
                name
                for name, value in reconstructed_fields.items()
                if canonical_json_sha256(value) != canonical_json_sha256(result.get(name))
            ]
            if mismatches:
                return [
                    "Sole inferential/statistical evidence differs from retained predictions: "
                    + ", ".join(mismatches)
                ]
    except (KeyError, ValueError, TypeError, StopIteration, IndexError) as error:
        return [f"primary reconstruction failed: {type(error).__name__}: {error}"]
    return []


def validate_run_directory(run_directory: Path, repository_root: Path) -> dict[str, Any]:
    """Return an evidence validation without mutating the run directory."""

    run_directory = run_directory.resolve()
    audit_path = run_directory / "data_audit.json"
    result_path = run_directory / "result.json"
    failure_path = run_directory / "failure.json"
    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, detail: Any, *, blocking: bool = True) -> None:
        checks.append(
            {
                "name": name,
                "passed": bool(passed),
                "blocking": blocking,
                "detail": detail,
            }
        )

    def invalid_package() -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "validated_at": datetime.now(UTC).isoformat(),
            "run_directory": ".",
            "status": "INVALID_EVIDENCE_PACKAGE",
            "checks": checks,
            "integrity_passed": False,
            "publication_evidence_ready": False,
        }

    result_present = result_path.exists() or result_path.is_symlink()
    failure_present = failure_path.exists() or failure_path.is_symlink()
    terminal_count = int(result_present) + int(failure_present)
    check(
        "exactly_one_terminal_artifact",
        terminal_count == 1,
        {"result": result_present, "failure": failure_present},
    )
    if terminal_count != 1:
        return invalid_package()
    terminal_path = result_path if result_present else failure_path
    terminal_regular = terminal_path.is_file() and not terminal_path.is_symlink()
    check(
        "terminal_artifact_regular_non_symlink",
        terminal_regular,
        terminal_path.name,
    )
    if not terminal_regular:
        return invalid_package()

    audit_present = audit_path.exists() or audit_path.is_symlink()
    if failure_present and not audit_present:
        check(
            "data_audit_not_applicable_before_dataset_acquisition",
            True,
            "pre-writer failure occurred before a dataset audit could be created",
        )
        try:
            failure = _read_object(failure_path)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            check("failure_strict_json", False, f"{type(error).__name__}: {error}")
            return invalid_package()
        check("failure_strict_json", True, "strict JSON object")
        manifest = failure.get("source_input_manifest")
        launch = failure.get("git_at_launch")
        binding = failure.get("publication_launch_context")
        files = (
            {str(name): str(digest) for name, digest in manifest.get("files", {}).items()}
            if isinstance(manifest, dict) and isinstance(manifest.get("files"), dict)
            else {}
        )
        commit = launch.get("commit") if isinstance(launch, dict) else None
        contract_errors: list[str] = []
        if failure.get("failure_scope") != "pre_writer_configuration_or_dataset_acquisition":
            contract_errors.append("failure scope is not the pre-writer envelope contract")
        if (
            failure.get("status") != "FAILED_PRESERVED"
            or not _timezone_aware_iso8601(failure.get("started_at"))
            or not _timezone_aware_iso8601(failure.get("failed_at"))
            or not isinstance(failure.get("stage"), str)
            or not failure.get("stage")
            or not isinstance(failure.get("exception_type"), str)
            or not failure.get("exception_type")
            or not isinstance(failure.get("exception_message"), str)
            or not isinstance(failure.get("traceback"), str)
            or not failure.get("traceback")
        ):
            contract_errors.append("pre-writer failure fields are incomplete")
        if not isinstance(launch, dict) or (
            launch.get("worktree_dirty") is not False or launch.get("status_entries") != []
        ):
            contract_errors.append("pre-writer failure did not launch from a clean worktree")
        contract_errors.extend(_manifest_commit_errors(repository_root, commit, files))
        if isinstance(manifest, dict):
            contract_errors.extend(_publication_manifest_errors(manifest, repository_root, commit))
        else:
            contract_errors.append("pre-writer publication source manifest is absent")
        contract_errors.extend(_environment_errors(failure.get("environment"), manifest))
        if isinstance(binding, dict):
            context = {
                "schema_version": "1.0.0",
                "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
                "allowed_create_only_output_root": binding.get("allowed_create_only_output_root"),
                "git_at_launch": launch,
                "source_input_manifest": manifest,
            }
            expected_binding = {
                "protocol_id": PUBLICATION_LAUNCH_CONTEXT_PROTOCOL_ID,
                "allowed_create_only_output_root": context["allowed_create_only_output_root"],
                "record_sha256": canonical_json_sha256(context),
            }
            if binding != expected_binding:
                contract_errors.append("pre-writer launch-context binding is invalid")
        else:
            contract_errors.append("pre-writer launch-context binding is absent")
        check(
            "failure_payload_self_hash",
            _self_hash_valid(failure, "failure_payload_sha256_before_serialization"),
            failure.get("failure_payload_sha256_before_serialization"),
        )
        check("pre_writer_failure_contract", not contract_errors, contract_errors)
        integrity = all(item["passed"] for item in checks if item["blocking"])
        return {
            "schema_version": "1.0.0",
            "validated_at": datetime.now(UTC).isoformat(),
            "run_directory": ".",
            "status": "FAILED_RUN_PRESERVED" if integrity else "INVALID_EVIDENCE_PACKAGE",
            "checks": checks,
            "integrity_passed": integrity,
            "publication_evidence_ready": False,
            "failure_scope": "pre_writer_configuration_or_dataset_acquisition",
        }

    audit_regular = audit_path.is_file() and not audit_path.is_symlink()
    check("data_audit_present", audit_regular, audit_path.name)
    if not audit_regular:
        return invalid_package()

    try:
        audit = _read_object(audit_path)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        check(
            "data_audit_strict_json",
            False,
            f"{type(error).__name__}: {error}",
        )
        return invalid_package()
    check("data_audit_strict_json", True, "strict JSON object")
    audit_manifest = audit.get("source_input_manifest")
    audit_publication_v4 = bool(
        isinstance(audit_manifest, dict)
        and audit_manifest.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
    )
    check(
        "data_audit_self_hash",
        _self_hash_valid(audit, "record_sha256"),
        audit.get("record_sha256"),
        blocking=audit_publication_v4,
    )
    receipts = _receipt_objects(audit)
    receipt_errors = _receipt_contract_errors(audit, publication_v4=audit_publication_v4)
    check("source_receipts_valid", bool(receipts) and not receipt_errors, receipt_errors)

    if failure_present:
        try:
            failure = _read_object(failure_path)
        except (OSError, UnicodeError, ValueError, TypeError) as error:
            check(
                "failure_strict_json",
                False,
                f"{type(error).__name__}: {error}",
            )
            return invalid_package()
        check("failure_strict_json", True, "strict JSON object")
        failure_contract_errors = (
            _shared_artifact_contract_errors(failure, audit, run_directory)
            if audit_publication_v4
            else []
        )
        if audit_publication_v4:
            failure_manifest = failure.get("source_input_manifest")
            failure_launch = failure.get("git_at_launch")
            failure_files = (
                {
                    str(name): str(digest)
                    for name, digest in failure_manifest.get("files", {}).items()
                }
                if isinstance(failure_manifest, dict)
                and isinstance(failure_manifest.get("files"), dict)
                else {}
            )
            failure_commit = (
                failure_launch.get("commit") if isinstance(failure_launch, dict) else None
            )
            if not isinstance(failure_launch, dict) or (
                failure_launch.get("worktree_dirty") is not False
                or failure_launch.get("status_entries") != []
            ):
                failure_contract_errors.append("failed run did not launch from a clean worktree")
            failure_contract_errors.extend(
                _manifest_commit_errors(repository_root, failure_commit, failure_files)
            )
            if isinstance(failure_manifest, dict):
                failure_contract_errors.extend(
                    _publication_manifest_errors(failure_manifest, repository_root, failure_commit)
                )
            else:
                failure_contract_errors.append("failed run publication source manifest is absent")
        required_failure_fields = failure.get("status") == "FAILED_PRESERVED"
        if audit_publication_v4:
            required_failure_fields = bool(
                required_failure_fields
                and _timezone_aware_iso8601(failure.get("started_at"))
                and _timezone_aware_iso8601(failure.get("failed_at", failure.get("failed_at_utc")))
                and isinstance(failure.get("exception_type"), str)
                and bool(failure.get("exception_type"))
                and isinstance(failure.get("exception_message"), str)
                and isinstance(failure.get("traceback"), str)
                and bool(failure.get("traceback"))
            )
        check(
            "failure_preserved",
            required_failure_fields,
            {
                "status": failure.get("status"),
                "exception_type": failure.get("exception_type"),
            },
        )
        check(
            "failure_payload_self_hash",
            _self_hash_valid(failure, "failure_payload_sha256_before_serialization"),
            failure.get("failure_payload_sha256_before_serialization"),
            blocking=audit_publication_v4,
        )
        check(
            "failure_artifact_contract",
            not failure_contract_errors,
            failure_contract_errors,
            blocking=audit_publication_v4,
        )
        integrity = all(item["passed"] for item in checks if item["blocking"])
        return {
            "schema_version": "1.0.0",
            "validated_at": datetime.now(UTC).isoformat(),
            "run_directory": ".",
            "status": "FAILED_RUN_PRESERVED" if integrity else "INVALID_EVIDENCE_PACKAGE",
            "checks": checks,
            "integrity_passed": integrity,
            "publication_evidence_ready": False,
        }

    try:
        result = _read_object(result_path)
    except (OSError, UnicodeError, ValueError, TypeError) as error:
        check(
            "result_strict_json",
            False,
            f"{type(error).__name__}: {error}",
        )
        return invalid_package()
    check("result_strict_json", True, "strict JSON object")
    result_manifest_value = result.get("source_input_manifest")
    result_publication_v4 = bool(
        isinstance(result_manifest_value, dict)
        and result_manifest_value.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
    )
    current_v4 = audit_publication_v4 or result_publication_v4
    if current_v4 and not audit_publication_v4:
        check(
            "current_protocol_data_audit_self_hash",
            _self_hash_valid(audit, "record_sha256"),
            audit.get("record_sha256"),
        )
        strict_receipt_errors = _receipt_contract_errors(audit, publication_v4=True)
        check(
            "current_protocol_source_receipts_valid",
            not strict_receipt_errors,
            strict_receipt_errors,
        )
    prediction = result.get("prediction_artifact")
    prediction_path: Path | None = None
    expected_prediction_sha256: str | None = None
    if isinstance(prediction, dict) and isinstance(prediction.get("path"), str):
        prediction_path = _safe_run_artifact_path(run_directory, prediction["path"])
        expected_prediction_sha256 = (
            str(prediction["sha256"]) if isinstance(prediction.get("sha256"), str) else None
        )
    check("prediction_artifact_declared", prediction_path is not None, prediction)
    if prediction_path is not None:
        digest_ok = _file_sha256_matches(prediction_path, expected_prediction_sha256)
        check("prediction_artifact_sha256", digest_ok, prediction_path.name)
    else:
        check("prediction_artifact_sha256", False, "prediction path unavailable")

    payload_digest = result.get("result_payload_sha256_before_serialization")
    unhashed = dict(result)
    unhashed.pop("result_payload_sha256_before_serialization", None)
    check(
        "result_payload_self_hash",
        isinstance(payload_digest, str)
        and re.fullmatch(r"[0-9a-f]{64}", payload_digest) is not None
        and canonical_json_sha256(unhashed) == payload_digest,
        payload_digest,
    )

    shared_contract_errors = (
        _shared_artifact_contract_errors(result, audit, run_directory) if current_v4 else []
    )
    check(
        "result_data_audit_and_ledger_binding",
        not shared_contract_errors,
        shared_contract_errors,
        blocking=current_v4,
    )
    neural_artifact_errors = (
        _neural_runtime_artifact_errors(result, audit, run_directory) if current_v4 else []
    )
    check(
        "neural_cuda_lane_and_checkpoint_contract",
        not neural_artifact_errors,
        neural_artifact_errors,
        blocking=current_v4,
    )
    hierarchy_artifact_errors = (
        _hierarchy_artifact_errors(result, audit, run_directory) if current_v4 else []
    )
    check(
        "hierarchy_checkpoint_fold_archive_contract",
        not hierarchy_artifact_errors,
        hierarchy_artifact_errors,
        blocking=current_v4,
    )

    manifest = result.get("source_input_manifest")
    manifest_files: dict[str, str] = {}
    manifest_internal_ok = False
    if isinstance(manifest, dict) and isinstance(manifest.get("files"), dict):
        manifest_files = {
            str(path): str(digest)
            for path, digest in cast(dict[str, Any], manifest["files"]).items()
        }
        manifest_internal_ok = manifest.get("file_count") == len(manifest_files) and manifest.get(
            "manifest_sha256"
        ) == canonical_json_sha256(manifest_files)
    check("source_input_manifest_internal", manifest_internal_ok, manifest)
    current_mismatches = []
    for relative, digest in manifest_files.items():
        path = PurePosixPath(relative)
        candidate = repository_root / relative
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in relative
            or ":" in relative
            or not _file_sha256_matches(candidate, digest)
        ):
            current_mismatches.append(relative)
    check(
        "source_inputs_match_current_workspace",
        manifest_internal_ok and not current_mismatches,
        current_mismatches,
        blocking=False,
    )

    partition_errors = _participant_partition_errors(result)
    check("participant_partitions_disjoint", not partition_errors, partition_errors)
    isolation_errors = _label_isolation_errors(result)
    check("held_out_label_isolation_flags", not isolation_errors, isolation_errors)

    expected_count = _expected_window_count(result, audit)
    probability_errors: list[str] = []
    probability_names: list[str] = []
    validation_summary = audit.get("dataset", result.get("dataset"))
    sole_observable_required = bool(
        current_v4
        and isinstance(validation_summary, dict)
        and validation_summary.get("dataset_id") == "sole_harmony_v1"
    )
    sole_observable_errors: list[str] = (
        [] if not sole_observable_required else ["Sole observable prediction archive is absent"]
    )
    scientific_payload_errors: list[str] = (
        [] if not current_v4 else ["current-protocol scientific prediction payload is absent"]
    )
    if prediction_path is not None and prediction_path.is_file():
        try:
            with np.load(prediction_path, allow_pickle=False) as archive:
                if "labels" not in archive or "participant_ids" not in archive:
                    probability_errors.append("labels or participant_ids are absent")
                    observed_count = None
                else:
                    observed_count = int(archive["labels"].shape[0])
                    if archive["participant_ids"].shape != (observed_count,):
                        probability_errors.append("participant identifiers are misaligned")
                    for metadata_name in ("session_ids", "trial_ids", "window_ids"):
                        if metadata_name in archive and archive[metadata_name].shape != (
                            observed_count,
                        ):
                            probability_errors.append(f"{metadata_name} are misaligned")
                    if (
                        "window_ids" in archive
                        and len(set(archive["window_ids"].astype(str).tolist())) != observed_count
                    ):
                        probability_errors.append("window identifiers are not unique")
                if expected_count is not None and observed_count != expected_count:
                    probability_errors.append(
                        f"prediction count {observed_count} != audited count {expected_count}"
                    )
                for key in archive.files:
                    if not key.startswith("probability__"):
                        continue
                    probability_names.append(key.removeprefix("probability__"))
                    values = np.asarray(archive[key], dtype=np.float64)
                    if (
                        observed_count is None
                        or values.ndim != 2
                        or values.shape[0] != observed_count
                        or not np.isfinite(values).all()
                        or np.any(values < -1e-9)
                        or np.any(values > 1.0 + 1e-9)
                        or not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0.0)
                    ):
                        probability_errors.append(f"invalid probability matrix: {key}")
                if sole_observable_required:
                    sole_observable_errors = _sole_observable_prediction_errors(
                        result, audit, archive
                    )
                if current_v4:
                    scientific_payload_errors = _scientific_prediction_payload_errors(
                        result, audit, archive
                    )
        except (OSError, EOFError, KeyError, TypeError, ValueError) as error:
            probability_errors.append(
                f"prediction archive cannot be read safely: {type(error).__name__}: {error}"
            )
            if sole_observable_required:
                sole_observable_errors = [
                    f"Sole observable prediction archive cannot be read safely: "
                    f"{type(error).__name__}: {error}"
                ]
    check(
        "prediction_arrays_aligned_finite_normalized",
        bool(probability_names) and not probability_errors,
        probability_errors,
    )
    check(
        "sole_full_observable_prediction_contract",
        not sole_observable_errors,
        sole_observable_errors if sole_observable_required else "not applicable",
        blocking=sole_observable_required,
    )
    check(
        "labels_ontology_seed_ensembles_and_reports_reconstructed",
        not scientific_payload_errors,
        scientific_payload_errors if current_v4 else "not applicable",
        blocking=current_v4,
    )
    reports = _report_mapping(result)
    missing_predictions = sorted(set(reports) - set(probability_names))
    check("reported_methods_have_predictions", not missing_predictions, missing_predictions)

    claim_policy = result.get("claim_policy")
    unsafe_claims = [
        key
        for key in ("confirmatory_claim_allowed", "state_of_the_art_claim_allowed")
        if not isinstance(claim_policy, dict) or claim_policy.get(key) is not False
    ]
    check("development_claim_boundary_retained", not unsafe_claims, unsafe_claims)

    split_audit: dict[str, Any] | None = None
    split_audit_errors: list[str] = []
    try:
        split_audit = build_split_audit(run_directory)
        if not split_audit.get("valid"):
            split_audit_errors.extend(str(item) for item in split_audit.get("errors", []))
        if split_audit.get("status") != "PASS_RECORDED_RESULT":
            split_audit_errors.append(
                "participant split evidence is not a recorded executed-result audit"
            )
    except Exception as error:  # malformed evidence must fail validation, not crash it
        split_audit_errors.append(f"split audit failed: {type(error).__name__}: {error}")
    stored_split_audit: dict[str, Any] | None = None
    split_path = run_directory / "split_audit.json"
    if current_v4:
        if not split_path.is_file() or split_path.is_symlink():
            split_audit_errors.append("create-only split_audit.json is absent")
        else:
            try:
                stored_split_audit = _read_object(split_path)
                if not _stored_split_audit_matches(stored_split_audit, split_audit):
                    split_audit_errors.append(
                        "stored split audit is not self-hashed or differs from reconstruction"
                    )
            except Exception as error:
                split_audit_errors.append(
                    f"stored split audit failed: {type(error).__name__}: {error}"
                )
    check(
        "participant_split_audit_recorded_and_valid",
        not split_audit_errors,
        split_audit_errors,
        blocking=False,
    )

    integrity = all(item["passed"] for item in checks if item["blocking"])
    launch = result.get("git_at_launch", {})
    clean_launch = (
        isinstance(launch, dict)
        and launch.get("worktree_dirty") is False
        and launch.get("status_entries") == []
    )
    commit_errors = _manifest_commit_errors(
        repository_root,
        launch.get("commit") if isinstance(launch, dict) else None,
        manifest_files,
    )
    publication_manifest_errors = (
        _publication_manifest_errors(
            manifest,
            repository_root,
            launch.get("commit") if isinstance(launch, dict) else None,
        )
        if isinstance(manifest, dict)
        and manifest.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
        else []
    )
    method_errors = _method_contract_errors(result, audit)
    diagnostic_reasons = _diagnostic_scope_reasons(result, audit)
    metric_errors = _metric_evidence_errors(result, prediction_path)
    publication_manifest = (
        isinstance(manifest, dict)
        and manifest.get("protocol_id") == PUBLICATION_SOURCE_PROTOCOL_ID
        and not publication_manifest_errors
    )
    legacy_manifest = (
        isinstance(manifest, dict) and manifest.get("protocol_id") == "external-har-session-grid-v3"
    )
    scientific_contract = (
        clean_launch
        and not commit_errors
        and not method_errors
        and not metric_errors
        and not split_audit_errors
        and publication_manifest
    )
    common_scientific_contract = scientific_contract
    qualified_methods = annotation_selected_context_methods(result)
    inference_contracts = method_inference_contracts(result)
    unqualified_methods = sorted(set(inference_contracts) - set(qualified_methods))
    scientific_contract = common_scientific_contract and not qualified_methods
    ready = integrity and scientific_contract and not diagnostic_reasons
    partially_ready = (
        not ready
        and integrity
        and common_scientific_contract
        and bool(unqualified_methods)
        and not diagnostic_reasons
    )
    permitted_diagnostic_errors = {
        "IMU-HAR-IL public source is provider-label-presegmented; continuous-stream publication claim blocked",
        "Sole-HARmony camera-bout preprocessing is an oracle diagnostic",
    }
    unexpected_diagnostic_errors = sorted(
        error for error in method_errors if error not in permitted_diagnostic_errors
    )
    diagnostic_contract = bool(
        integrity
        and diagnostic_reasons
        and clean_launch
        and not commit_errors
        and publication_manifest
        and not metric_errors
        and not split_audit_errors
        and not unexpected_diagnostic_errors
    )
    return {
        "schema_version": "1.0.0",
        "validated_at": datetime.now(UTC).isoformat(),
        "run_directory": ".",
        "status": "VALIDATED"
        if ready
        else "PARTIALLY_VALIDATED_METHODS"
        if partially_ready
        else "DIAGNOSTIC"
        if diagnostic_contract
        else "SUPERSEDED_PROTOCOL"
        if integrity and legacy_manifest
        else "PROVISIONAL"
        if integrity
        else "INVALID_EVIDENCE_PACKAGE",
        "checks": checks,
        "integrity_passed": integrity,
        "publication_evidence_ready": ready,
        "publication_evidence_ready_for_unqualified_methods": partially_ready,
        "unqualified_method_names": unqualified_methods if partially_ready else [],
        "method_inference_contracts": inference_contracts,
        "annotation_selected_context_methods": qualified_methods,
        "scientific_contract_passed": scientific_contract,
        "diagnostic_contract_passed": diagnostic_contract,
        "diagnostic_scope_reasons": diagnostic_reasons,
        "participant_split_audit": split_audit,
        "stored_participant_split_audit": (
            None
            if stored_split_audit is None
            else {
                "path": "split_audit.json",
                "sha256": sha256_file(split_path),
                "record_sha256": stored_split_audit.get("record_sha256"),
            }
        ),
        "scientific_contract_checks": {
            "clean_git_at_launch": clean_launch,
            "versioned_protocol_correction": publication_manifest,
            "source_manifest_protocol_id": (
                manifest.get("protocol_id") if isinstance(manifest, dict) else None
            ),
            "source_manifest_protocol_status": (
                "CURRENT_PUBLICATION_V4"
                if publication_manifest
                else "LEGACY_READABLE_SUPERSEDED"
                if legacy_manifest
                else "UNKNOWN_OR_ABSENT"
            ),
            "launch_manifest_commit_errors": commit_errors,
            "publication_manifest_errors": publication_manifest_errors,
            "participant_split_audit_errors": split_audit_errors,
            "method_contract_errors": method_errors,
            "unexpected_diagnostic_method_errors": unexpected_diagnostic_errors,
            "metric_reconstruction_errors": metric_errors,
            "annotation_selected_context_methods": qualified_methods,
        },
        "limitation": "Contract validation is not a general proof of scientific validity or deployability.",
        "claim_scope": (
            "diagnostic evidence only; not continuous-stream, confirmatory, or SOTA"
            if diagnostic_reasons
            else "development evidence only; not confirmatory and not SOTA"
        ),
    }


def _stable_validation_payload(validation: dict[str, Any]) -> dict[str, Any]:
    """Remove volatility and normalize only legacy absolute presentation paths."""

    stable = cast(dict[str, Any], json.loads(json.dumps(validation)))
    stable.pop("validated_at", None)
    legacy_root = stable.get("run_directory")
    split = stable.get("participant_split_audit")
    if isinstance(split, dict):
        stable["participant_split_audit"] = _stable_split_audit_payload(split)
    normalized_split = stable.get("participant_split_audit")
    prediction_name: str | None = None
    if isinstance(normalized_split, dict):
        inputs = normalized_split.get("inputs")
        predictions = inputs.get("predictions") if isinstance(inputs, dict) else None
        if isinstance(predictions, dict) and isinstance(predictions.get("path"), str):
            prediction_name = predictions["path"]
    if _runtime_path_is_absolute(legacy_root):
        for record in stable.get("checks", []):
            if not isinstance(record, dict):
                continue
            name = record.get("name")
            direct_name = _absolute_direct_child_name(record.get("detail"), legacy_root)
            if name == "data_audit_present" and direct_name == "data_audit.json":
                record["detail"] = direct_name
            elif (
                name == "prediction_artifact_sha256"
                and prediction_name is not None
                and direct_name == prediction_name
            ):
                record["detail"] = direct_name
        stable["run_directory"] = "."
    return stable


def validate_and_record_run_directory(
    run_directory: Path,
    repository_root: Path,
    *,
    diagnostic: bool | None = None,
) -> dict[str, Any]:
    """Create or verify immutable validation and enforce its evidence-role gate."""

    _assert_executed_repository_root(repository_root)
    run_directory = run_directory.resolve()
    split_path = run_directory / "split_audit.json"
    result_path = run_directory / "result.json"
    failure_path = run_directory / "failure.json"
    result_present = result_path.exists() or result_path.is_symlink()
    failure_present = failure_path.exists() or failure_path.is_symlink()
    if int(result_present) + int(failure_present) != 1:
        raise ValueError("run must contain exactly one terminal artifact before validation")
    terminal_path = result_path if result_present else failure_path
    if not terminal_path.is_file() or terminal_path.is_symlink():
        raise ValueError("run terminal artifact must be one regular non-symlink file")
    terminal_failure = failure_present
    if split_path.is_symlink():
        raise ValueError("stored split audit may not be a symbolic link")
    if not split_path.exists() and not terminal_failure:
        write_split_audit(
            run_directory,
            split_path,
            repository_root=repository_root,
        )
    validation = validate_run_directory(run_directory, repository_root)
    validation_path = run_directory / "validation.json"
    if validation_path.is_symlink():
        raise ValueError("stored validation may not be a symbolic link")
    if validation_path.exists():
        stored = _read_object(validation_path)
        stored_split = stored.get("participant_split_audit")
        if not _timezone_aware_iso8601(stored.get("validated_at")) or (
            stored_split is not None
            and (
                not isinstance(stored_split, dict)
                or not _timezone_aware_iso8601(stored_split.get("created_at"))
                or not _self_hash_valid(stored_split, "record_sha256")
            )
        ):
            raise ValueError(
                f"stored validation volatile integrity fields are invalid: {run_directory}"
            )
        if canonical_json_sha256(_stable_validation_payload(stored)) != canonical_json_sha256(
            _stable_validation_payload(validation)
        ):
            raise ValueError(
                f"stored validation differs from independent reconstruction: {run_directory}"
            )
    else:
        _write_json_create_only(validation_path, validation)
    if terminal_failure:
        if (
            validation.get("status") != "FAILED_RUN_PRESERVED"
            or validation.get("integrity_passed") is not True
            or validation.get("publication_evidence_ready") is not False
        ):
            raise ValueError(f"failed-run evidence gate failed: {run_directory}")
        return validation
    split_audit = validation.get("participant_split_audit")
    if not isinstance(split_audit, dict) or (
        split_audit.get("valid") is not True or split_audit.get("status") != "PASS_RECORDED_RESULT"
    ):
        raise ValueError(f"recorded participant-split gate failed: {run_directory}")
    require_diagnostic = (
        bool(validation.get("diagnostic_scope_reasons")) if diagnostic is None else diagnostic
    )
    accepted = (
        validation.get("status") == "DIAGNOSTIC"
        and validation.get("diagnostic_contract_passed") is True
        if require_diagnostic
        else validation.get("publication_evidence_ready") is True
    )
    if not accepted:
        role = "diagnostic" if require_diagnostic else "publication"
        raise ValueError(f"{role} evidence acceptance gate failed: {run_directory}")
    return validation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-name", default="validation.json")
    return parser


def _safe_direct_output_name(value: str) -> str:
    parsed = PurePosixPath(value)
    if (
        value != "validation.json"
        or parsed.is_absolute()
        or parsed.parts != (value,)
        or value in {".", ".."}
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"validation output name must be one direct-child filename: {value!r}")
    return value


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repository_root = args.repository_root.resolve()
    _assert_executed_repository_root(repository_root)
    output_name = _safe_direct_output_name(args.output_name)
    failed = False
    summaries = []
    for directory in args.run_directories:
        resolved_directory = directory.resolve()
        split_output = resolved_directory / "split_audit.json"
        if (resolved_directory / "result.json").is_file() and not split_output.exists():
            write_split_audit(
                resolved_directory,
                split_output,
                repository_root=repository_root,
            )
        validation = validate_run_directory(resolved_directory, repository_root)
        output = resolved_directory / output_name
        _write_json_create_only(output, validation)
        summaries.append(
            {
                "run_directory": str(directory),
                "status": validation["status"],
                "publication_evidence_ready": validation["publication_evidence_ready"],
            }
        )
        failed |= not bool(validation["integrity_passed"])
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
