"""Validate create-only external-HAR run evidence before any result is reported."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np

from inclusive_shift_har.evaluation.external_statistics import (
    ParticipantMetricInputs,
    seed_evidence,
)
from inclusive_shift_har.experiments.cross_dataset_har import _write_json_create_only
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
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


def _receipt_objects(audit: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for key, value in audit.items():
        if key.endswith("receipts") and isinstance(value, list):
            receipts.extend(item for item in value if isinstance(item, dict))
    return receipts


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


def _method_contract_errors(result: dict[str, Any], audit: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    summaries = [
        value
        for key in ("dataset", "source_dataset", "target_dataset")
        if isinstance(value := audit.get(key, result.get(key)), dict)
    ]
    if not summaries or any("dataset_id" not in item for item in summaries):
        errors.append("dataset identity and preprocessing contract unavailable")
    for summary in summaries:
        if summary.get("dataset_id") == "sole_harmony_v1":
            errors.append("Sole-HARmony camera-bout preprocessing is an oracle diagnostic")
        if summary.get("dataset_id") != "fog_star_v3":
            continue
        segments = summary.get("preprocessing_audit", [])
        admitted_total = 0
        if not segments:
            errors.append(
                "FoG annotation-independent resampling evidence absent; historical run superseded"
            )
        for segment in segments:
            if (
                segment.get("protocol_id") != "external-har-session-grid-v3"
                or segment.get("annotation_dependency") is not False
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
    if result.get("seeds") != [11, 23, 47] or not isinstance(
        result.get("primary_seed_averaged"), dict
    ):
        errors.append("frozen three-seed participant-averaged evidence is missing")
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
    return errors


def _metric_evidence_errors(result: dict[str, Any], prediction_path: Path | None) -> list[str]:
    """Recompute the complete primary report from each retained seed, not its hash alone."""

    primary = result.get("primary_seed_averaged")
    if not isinstance(primary, dict) or prediction_path is None or not prediction_path.is_file():
        return ["primary metrics cannot be reconstructed from retained seed predictions"]
    try:
        methods = primary["methods"]
        seeds = primary["seeds"]
        example = next(iter(methods.values()))["reports_by_seed"][str(seeds[0])]
        dataset = result.get("target_dataset", result.get("dataset", {}))
        with np.load(prediction_path, allow_pickle=False) as archive:
            inputs = ParticipantMetricInputs(
                dataset_id=dataset["dataset_id"],
                class_names=tuple(example["class_names"]),
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

    check("data_audit_present", audit_path.is_file(), str(audit_path))
    terminal_count = int(result_path.is_file()) + int(failure_path.is_file())
    check(
        "exactly_one_terminal_artifact",
        terminal_count == 1,
        {"result": result_path.is_file(), "failure": failure_path.is_file()},
    )
    if not audit_path.is_file() or terminal_count != 1:
        return {
            "schema_version": "1.0.0",
            "run_directory": str(run_directory),
            "status": "INVALID_EVIDENCE_PACKAGE",
            "checks": checks,
            "integrity_passed": False,
            "publication_evidence_ready": False,
        }

    audit = _read_object(audit_path)
    receipts = _receipt_objects(audit)
    receipt_errors = []
    for receipt in receipts:
        digest = receipt.get("computed_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            receipt_errors.append(
                f"invalid SHA-256: {receipt.get('member') or receipt.get('locator')}"
            )
        if (
            not isinstance(receipt.get("received_size_bytes"), int)
            or int(receipt["received_size_bytes"]) <= 0
        ):
            receipt_errors.append(f"invalid byte count: {receipt.get('member')}")
        if receipt.get("declared_digest_verified") is False:
            receipt_errors.append(f"declared digest mismatch: {receipt.get('member')}")
        if receipt.get("raw_local_mirror") is not False:
            receipt_errors.append(f"storage disclosure mismatch: {receipt.get('member')}")
    check("source_receipts_valid", bool(receipts) and not receipt_errors, receipt_errors)

    if failure_path.is_file():
        failure = _read_object(failure_path)
        check(
            "failure_preserved",
            failure.get("status") == "FAILED_PRESERVED",
            failure.get("status"),
        )
        integrity = all(item["passed"] for item in checks if item["blocking"])
        return {
            "schema_version": "1.0.0",
            "validated_at": datetime.now(UTC).isoformat(),
            "run_directory": str(run_directory),
            "status": "FAILED_RUN_PRESERVED",
            "checks": checks,
            "integrity_passed": integrity,
            "publication_evidence_ready": False,
        }

    result = _read_object(result_path)
    prediction = result.get("prediction_artifact")
    prediction_path: Path | None = None
    expected_prediction_sha256: str | None = None
    if isinstance(prediction, dict) and isinstance(prediction.get("path"), str):
        prediction_path = run_directory / str(prediction["path"])
        expected_prediction_sha256 = (
            str(prediction["sha256"]) if isinstance(prediction.get("sha256"), str) else None
        )
    check("prediction_artifact_declared", prediction_path is not None, prediction)
    if prediction_path is not None:
        digest_ok = (
            expected_prediction_sha256 is not None
            and prediction_path.is_file()
            and sha256_file(prediction_path) == expected_prediction_sha256
        )
        check("prediction_artifact_sha256", digest_ok, str(prediction_path))
    else:
        check("prediction_artifact_sha256", False, "prediction path unavailable")

    payload_digest = result.get("result_payload_sha256_before_serialization")
    unhashed = dict(result)
    unhashed.pop("result_payload_sha256_before_serialization", None)
    check(
        "result_payload_self_hash",
        isinstance(payload_digest, str) and canonical_json_sha256(unhashed) == payload_digest,
        payload_digest,
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
    current_mismatches = [
        relative
        for relative, digest in manifest_files.items()
        if not (repository_root / relative).is_file()
        or sha256_file(repository_root / relative) != digest
    ]
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
    if prediction_path is not None and prediction_path.is_file():
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
    check(
        "prediction_arrays_aligned_finite_normalized",
        bool(probability_names) and not probability_errors,
        probability_errors,
    )
    reports = _report_mapping(result)
    missing_predictions = sorted(set(reports) - set(probability_names))
    check("reported_methods_have_predictions", not missing_predictions, missing_predictions)

    claim_policy = result.get("claim_policy")
    unsafe_claims = []
    if isinstance(claim_policy, dict):
        unsafe_claims = [
            key
            for key in ("confirmatory_claim_allowed", "state_of_the_art_claim_allowed")
            if claim_policy.get(key) is True
        ]
    check("development_claim_boundary_retained", not unsafe_claims, unsafe_claims)

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
    method_errors = _method_contract_errors(result, audit)
    metric_errors = _metric_evidence_errors(result, prediction_path)
    manifest_v2 = (
        isinstance(manifest, dict) and manifest.get("protocol_id") == "external-har-session-grid-v3"
    )
    scientific_contract = (
        clean_launch
        and not commit_errors
        and not method_errors
        and not metric_errors
        and manifest_v2
    )
    ready = integrity and scientific_contract
    return {
        "schema_version": "1.0.0",
        "validated_at": datetime.now(UTC).isoformat(),
        "run_directory": str(run_directory),
        "status": "VALIDATED"
        if ready
        else "PROVISIONAL"
        if integrity
        else "INVALID_EVIDENCE_PACKAGE",
        "checks": checks,
        "integrity_passed": integrity,
        "publication_evidence_ready": ready,
        "scientific_contract_passed": scientific_contract,
        "scientific_contract_checks": {
            "clean_git_at_launch": clean_launch,
            "versioned_protocol_correction": manifest_v2,
            "launch_manifest_commit_errors": commit_errors,
            "method_contract_errors": method_errors,
            "metric_reconstruction_errors": metric_errors,
        },
        "limitation": "Contract validation is not a general proof of scientific validity or deployability.",
        "claim_scope": "development evidence only; not confirmatory and not SOTA",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directories", type=Path, nargs="+")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-name", default="validation.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    failed = False
    summaries = []
    for directory in args.run_directories:
        validation = validate_run_directory(directory, args.repository_root.resolve())
        output = directory.resolve() / args.output_name
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
