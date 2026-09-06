"""Create-only parity gate for the annotation-independent context correction."""

from __future__ import annotations

import argparse
import copy
import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from inclusive_shift_har.artifacts.research_provenance import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.evaluation.inference_contracts import (
    ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS,
    OBSERVABLE_CONTEXT_PROTOCOL,
    OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
)
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.experiments.publication_table import _receipt_contract
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

_TRAINING_POPULATION_PROTOCOL_PATH = (
    "configs/protocols/external_har_observable_context_training_population_v1.yaml"
)
_REPLACEMENT_SUMMARY_ADDITIONS = frozenset(
    {
        "boundary_provenance",
        "participant_partition_observation",
        "participant_partition_plan",
        "retained_array_hashes",
    }
)
_RETAINED_ARRAY_NAMES = frozenset(
    {
        "signals",
        "gravity",
        "nine_channel_signals",
        "labels",
        "participant_ids",
        "session_ids",
        "trial_ids",
        "window_ids",
    }
)
_HISTORICAL_POOL_ARRAY_NAMES = frozenset(
    {"signals", "gravity", "participant_ids", "session_ids", "trial_ids", "window_ids"}
)


def _run_location(directory: Path, repository_root: Path) -> dict[str, str]:
    resolved = directory.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return {"path_kind": "external_absolute", "path": str(resolved)}
    return {"path_kind": "repository_relative", "path": relative or "."}


def _historical_reference_pin(repository_root: Path) -> dict[str, Any]:
    protocol_path = repository_root / _TRAINING_POPULATION_PROTOCOL_PATH
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("observable-context training-population protocol is not a mapping")
    pin = protocol.get("historical_parity_reference")
    if not isinstance(pin, dict):
        raise ValueError("observable-context protocol lacks the historical parity reference pin")
    return pin


def _require_historical_reference(
    directory: Path,
    result: dict[str, Any],
    audit: dict[str, Any],
    validation: dict[str, Any],
    pin: dict[str, Any],
) -> None:
    expected_validation = {
        "status": "SUPERSEDED_PROTOCOL",
        "integrity_passed": True,
        "publication_evidence_ready": False,
        "publication_evidence_ready_for_unqualified_methods": False,
        "diagnostic_contract_passed": False,
    }
    if validation.get("status") != expected_validation["status"] or any(
        validation.get(key) is not value
        for key, value in expected_validation.items()
        if key != "status"
    ):
        raise ValueError("reference is not the exact integrity-valid superseded protocol state")
    expected_hashes = pin.get("artifact_sha256")
    if not isinstance(expected_hashes, dict) or set(expected_hashes) != {
        "data_audit.json",
        "predictions.npz",
        "result.json",
    }:
        raise ValueError("historical reference pin has an invalid artifact hash set")
    for name, expected in expected_hashes.items():
        if not isinstance(expected, str) or sha256_file(directory / name) != expected:
            raise ValueError(f"historical reference artifact differs from its frozen pin: {name}")
    launch = audit.get("git_at_launch")
    manifest = audit.get("source_input_manifest")
    expected_launch = pin.get("launch_commit")
    if (
        not isinstance(launch, dict)
        or launch.get("commit") != expected_launch
        or launch.get("worktree_dirty") is not False
        or launch.get("status_entries") != []
        or not isinstance(manifest, dict)
        or manifest.get("protocol_id") != pin.get("source_manifest_protocol_id")
        or manifest.get("manifest_sha256") != pin.get("source_manifest_sha256")
        or result.get("dataset", {}).get("dataset_id") != pin.get("dataset_id")
        or result.get("observable_context_protocol") != pin.get("observable_context_protocol")
        or result.get("observable_context_training_protocol")
        != pin.get("observable_context_training_protocol")
        or result.get("result_payload_sha256_before_serialization")
        != pin.get("result_payload_sha256_before_serialization")
    ):
        raise ValueError("historical reference semantic identity differs from its frozen pin")


def _require_current_replacement(validation: dict[str, Any]) -> None:
    if not (
        validation.get("status") == "VALIDATED"
        and validation.get("integrity_passed") is True
        and validation.get("publication_evidence_ready") is True
        and validation.get("scientific_contract_passed") is True
        and validation.get("diagnostic_contract_passed") is False
    ):
        raise ValueError("replacement must be current, fully validated publication evidence")


def _sha256_mapping(value: Any, *, exact_names: frozenset[str]) -> bool:
    return bool(
        isinstance(value, dict)
        and set(value) == exact_names
        and all(re.fullmatch(r"[0-9a-f]{64}", str(digest)) for digest in value.values())
    )


def _historical_parity_summary(
    summary: dict[str, Any], *, replacement_side: bool
) -> dict[str, Any]:
    """Normalize only the frozen v3-to-v4 audit-schema migration, never data."""

    normalized = copy.deepcopy(summary)
    if replacement_side:
        if not _sha256_mapping(
            normalized.get("retained_array_hashes"), exact_names=_RETAINED_ARRAY_NAMES
        ):
            raise ValueError("replacement retained-array witness is absent or malformed")
        boundary = normalized.get("boundary_provenance")
        plan = normalized.get("participant_partition_plan")
        observation = normalized.get("participant_partition_observation")
        if (
            not isinstance(boundary, dict)
            or boundary.get("repository_signal_grid_annotation_independent") is not True
            or boundary.get("provider_upstream_annotation_conditioned") is not False
            or not isinstance(plan, dict)
            or not isinstance(observation, dict)
        ):
            raise ValueError("replacement governance-only summary additions are malformed")
        for name in _REPLACEMENT_SUMMARY_ADDITIONS:
            normalized.pop(name)
    elif set(normalized) & _REPLACEMENT_SUMMARY_ADDITIONS:
        raise ValueError("historical summary unexpectedly contains v4-only audit fields")

    segments = normalized.get("preprocessing_audit")
    if not isinstance(segments, list) or not segments:
        raise ValueError("FoG preprocessing audit is absent from parity input")
    for segment in segments:
        if not isinstance(segment, dict):
            raise ValueError("FoG preprocessing audit contains a malformed segment")
        if replacement_side:
            if (
                "annotation_dependency" in segment
                or segment.get("within_declared_segment_transform_annotation_dependency")
                is not False
                or segment.get("segment_boundary_annotation_conditioned") is not False
            ):
                raise ValueError("replacement segment audit is not the frozen safe-key migration")
            segment.pop("within_declared_segment_transform_annotation_dependency")
            segment.pop("segment_boundary_annotation_conditioned")
            segment["annotation_dependency"] = False
        elif (
            segment.get("annotation_dependency") is not False
            or "within_declared_segment_transform_annotation_dependency" in segment
            or "segment_boundary_annotation_conditioned" in segment
        ):
            raise ValueError("historical segment audit differs from the pinned v3 schema")

    pool = normalized.get("observable_candidate_pool")
    arrays = pool.get("arrays") if isinstance(pool, dict) else None
    expected_names = set(_HISTORICAL_POOL_ARRAY_NAMES)
    if replacement_side:
        expected_names.add("nine_channel_signals")
    if not _sha256_mapping(arrays, exact_names=frozenset(expected_names)):
        raise ValueError("observable candidate-pool array witness is absent or malformed")
    assert isinstance(arrays, dict)
    if replacement_side:
        arrays.pop("nine_channel_signals")
    return normalized


def compare_context_replacement(reference: Path, replacement: Path, root: Path) -> dict[str, Any]:
    root = root.resolve()
    packages = []
    provenance = []
    validations = []
    for directory in (reference, replacement):
        validation = validate_run_directory(directory, root)
        result = json.loads((directory / "result.json").read_text(encoding="utf-8"))
        audit = json.loads((directory / "data_audit.json").read_text(encoding="utf-8"))
        packages.append((result, audit))
        validations.append(validation)
        provenance.append(
            {
                "directory": _run_location(directory, root),
                "result_sha256": sha256_file(directory / "result.json"),
                "predictions_sha256": sha256_file(directory / "predictions.npz"),
                "validation": validation,
            }
        )
    old, new = packages[0][0], packages[1][0]
    _require_historical_reference(
        reference,
        packages[0][0],
        packages[0][1],
        validations[0],
        _historical_reference_pin(root),
    )
    _require_current_replacement(validations[1])
    if old.get("experiment_id") != "cross-dataset-har-rnd-v1" or new.get(
        "experiment_id"
    ) != old.get("experiment_id"):
        raise ValueError("parity inputs are not the frozen FoG classical experiment")
    if (
        new.get("observable_context_protocol") != OBSERVABLE_CONTEXT_PROTOCOL
        or new.get("observable_context_training_protocol") != OBSERVABLE_CONTEXT_TRAINING_PROTOCOL
    ):
        raise ValueError("replacement does not declare both observable-context corrections")
    if any(key in old or key in new for key in ("source_dataset", "target_dataset")):
        raise ValueError("historical FoG parity does not accept transfer-shaped results")
    summaries = [
        _historical_parity_summary(result.get("dataset", {}), replacement_side=index == 1)
        for index, result in enumerate((old, new))
    ]
    if summaries[0] != summaries[1]:
        raise ValueError("scoring cohort, signals, grid or preprocessing differs: dataset")
    audit_summaries = [
        _historical_parity_summary(audit.get("dataset", {}), replacement_side=index == 1)
        for index, (_result, audit) in enumerate(packages)
    ]
    if audit_summaries != summaries or audit_summaries[0] != audit_summaries[1]:
        raise ValueError("result/data-audit scientific dataset tuple differs")
    if _receipt_contract(packages[0][1]) != _receipt_contract(packages[1][1]):
        raise ValueError("provider source receipts differ")
    if old.get("seeds") != [11, 23, 47] or new.get("seeds") != [11, 23, 47]:
        raise ValueError("parity requires the three frozen seeds")
    old_methods, new_methods = (
        set(result["primary_seed_averaged"]["methods"]) for result in (old, new)
    )
    if old_methods != new_methods:
        raise ValueError("method set changed during a context-only repair")
    comparisons: dict[str, Any] = {}
    with (
        np.load(reference / "predictions.npz", allow_pickle=False) as left,
        np.load(replacement / "predictions.npz", allow_pickle=False) as right,
    ):
        for name in ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids"):
            if not np.array_equal(left[name], right[name]):
                raise ValueError(f"scoring identities changed: {name}")
        for method in sorted(new_methods):
            per_seed = {}
            for seed in (11, 23, 47):
                key = f"probability__seed-{seed}__{method}"
                a, b = left[key], right[key]
                per_seed[str(seed)] = {
                    "decisions_exact": bool(np.array_equal(a.argmax(axis=1), b.argmax(axis=1))),
                    "probabilities_bitwise_equal": bool(np.array_equal(a, b)),
                    "maximum_absolute_probability_difference": float(np.max(np.abs(a - b))),
                    "probabilities_within_predeclared_tolerance": bool(
                        np.allclose(a, b, rtol=0, atol=1e-12)
                    ),
                }
            affected = method in ANNOTATION_INDEPENDENT_CONTEXT_FIT_METHODS
            comparisons[method] = {
                "context_affected": affected,
                "seeds": per_seed,
                "parity_required": not affected,
                "parity_passed": None
                if affected
                else all(
                    row["decisions_exact"] and row["probabilities_within_predeclared_tolerance"]
                    for row in per_seed.values()
                ),
                "old_mean_participant_macro_f1": old["primary_seed_averaged"]["methods"][method][
                    "mean_participant_macro_f1"
                ],
                "replacement_mean_participant_macro_f1": new["primary_seed_averaged"]["methods"][
                    method
                ]["mean_participant_macro_f1"],
            }
    record = {
        "protocol_id": OBSERVABLE_CONTEXT_PROTOCOL,
        "training_population_protocol_id": OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "PARITY_PASSED"
        if all(item["parity_passed"] for item in comparisons.values() if item["parity_required"])
        else "PARITY_DISCREPANCY_PRESERVED",
        "unaffected_probability_absolute_tolerance": 1e-12,
        "tolerance_protocol": _TRAINING_POPULATION_PROTOCOL_PATH,
        "methods": comparisons,
        "sources": provenance,
        "analysis_git": _git_state(root),
        "analysis_source_input_manifest": _source_input_manifest(root),
        "affected_context_before_after_is_a_validity_correction_not_a_matched_model_improvement": True,
        "better_score_selection_allowed": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--replacement", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    record = compare_context_replacement(
        args.reference, args.replacement, args.repository_root.resolve()
    )
    _write_json_create_only(args.output, record)
    print(json.dumps({"status": record["status"], "record_sha256": record["record_sha256"]}))
    return 0 if record["status"] == "PARITY_PASSED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
