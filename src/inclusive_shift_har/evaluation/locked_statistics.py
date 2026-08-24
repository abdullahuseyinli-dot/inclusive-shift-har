"""Participant-level aggregation of immutable locked-target seed results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np

from inclusive_shift_har.evaluation.locked_target import LOCKED_TARGET_EVIDENCE_STATUS
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import (
    holm_adjust,
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

LOCKED_STATISTICS_SCHEMA_VERSION = "1.0.0"


class LockedStatisticsError(RuntimeError):
    """Raised when locked result aggregation encounters incomplete or changed evidence."""


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise LockedStatisticsError(f"{name} must be an object")
    return value


def _validate_self_hash(record: Mapping[str, Any], *, field: str, role: str) -> None:
    body = dict(record)
    claimed = body.pop(field, None)
    if claimed != canonical_json_sha256(body):
        raise LockedStatisticsError(f"{role} self-hash does not validate")


def _resolve_file(value: Any, *, output_root: Path, role: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value):
        raise LockedStatisticsError(f"{role} path is missing")
    raw = Path(value)
    candidate = raw if raw.is_absolute() else output_root / raw
    root = output_root.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise LockedStatisticsError(f"{role} escapes the output root") from exc
    if resolved.is_symlink() or not resolved.is_file():
        raise LockedStatisticsError(f"{role} is missing, non-regular, or a symlink")
    return resolved


def _participant_metric(report: Mapping[str, Any]) -> dict[str, float]:
    rows = report.get("participants")
    if not isinstance(rows, list) or not rows:
        raise LockedStatisticsError("participant report is missing")
    values: dict[str, float] = {}
    for row_value in rows:
        row = _require_mapping(row_value, name="participant row")
        participant = row.get("participant_id")
        value = row.get("macro_f1")
        if (
            not isinstance(participant, str)
            or not participant
            or participant in values
            or isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
        ):
            raise LockedStatisticsError("participant macro-F1 row is invalid")
        values[participant] = float(value)
    return values


def _load_seed_result(
    index_entry: Mapping[str, Any],
    *,
    output_root: Path,
    expected_class_names: tuple[str, ...],
) -> dict[str, Any]:
    record_path = _resolve_file(
        index_entry.get("record_path"), output_root=output_root, role="seed result record"
    )
    if sha256_file(record_path) != index_entry.get("record_file_sha256"):
        raise LockedStatisticsError("seed result record file hash mismatch")
    record_value = load_json_strict(record_path)
    record = _require_mapping(record_value, name="seed result record")
    _validate_self_hash(record, field="record_sha256", role="seed result record")
    if record.get("record_sha256") != index_entry.get("record_sha256"):
        raise LockedStatisticsError("seed result record hash differs from index")
    required = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "class_names": list(expected_class_names),
        "target_information_used_for_model_selection": False,
    }
    mismatches = [key for key, expected in required.items() if record.get(key) != expected]
    if mismatches:
        raise LockedStatisticsError(f"seed result contract mismatch: {mismatches}")
    prediction = _require_mapping(record.get("prediction_array"), name="prediction array")
    array_path = _resolve_file(
        prediction.get("path"), output_root=output_root, role="seed prediction array"
    )
    indexed_array_path = _resolve_file(
        index_entry.get("array_path"), output_root=output_root, role="indexed prediction array"
    )
    if array_path != indexed_array_path:
        raise LockedStatisticsError("record and index point to different prediction arrays")
    if sha256_file(array_path) != prediction.get("sha256") or prediction.get(
        "sha256"
    ) != index_entry.get("array_sha256"):
        raise LockedStatisticsError("seed prediction array hash mismatch")
    with np.load(array_path, allow_pickle=False) as arrays:
        required_arrays = {
            "evidence_status",
            "window_ids",
            "participant_ids",
            "true_labels",
            "logits",
            "uncalibrated_probabilities",
            "calibrated_probabilities",
            "predicted_labels",
        }
        if set(arrays.files) != required_arrays:
            raise LockedStatisticsError("locked prediction keys differ from contract")
        evidence = arrays["evidence_status"].tolist()
        window_ids = tuple(str(value) for value in arrays["window_ids"].tolist())
        participant_ids = tuple(str(value) for value in arrays["participant_ids"].tolist())
        labels = np.asarray(arrays["true_labels"], dtype=np.int64)
        probabilities = np.asarray(arrays["calibrated_probabilities"], dtype=np.float64)
        predictions = np.asarray(arrays["predicted_labels"], dtype=np.int64)
    if evidence != [LOCKED_TARGET_EVIDENCE_STATUS]:
        raise LockedStatisticsError("prediction array evidence status is invalid")
    if len(window_ids) != labels.size or len(set(window_ids)) != labels.size:
        raise LockedStatisticsError("locked window IDs are not aligned and unique")
    if len(participant_ids) != labels.size or probabilities.shape != (
        labels.size,
        len(expected_class_names),
    ):
        raise LockedStatisticsError("locked predictions are not aligned")
    if not np.array_equal(predictions, probabilities.argmax(axis=1)):
        raise LockedStatisticsError("stored labels differ from calibrated probability argmax")
    recomputed = classification_report(
        labels,
        probabilities,
        participant_ids,
        class_names=expected_class_names,
    )
    report = _require_mapping(record.get("participant_level_report"), name="metric report")
    if canonical_json_sha256(recomputed) != canonical_json_sha256(report):
        raise LockedStatisticsError("stored locked metrics do not reconstruct from predictions")
    return {
        "model_id": str(record["model_id"]),
        "seed": int(record["seed"]),
        "window_ids": window_ids,
        "participant_ids": participant_ids,
        "labels": labels,
        "report": recomputed,
        "record_sha256": record["record_sha256"],
        "prediction_sha256": prediction["sha256"],
    }


def _mean(values: Sequence[float]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def aggregate_locked_target_statistics(
    index_path: str | Path,
    *,
    output_root: str | Path,
    destination: str | Path,
    required_seed_order: Sequence[int],
    candidate_model_id: str,
    eligible_reference_model_ids: Sequence[str],
    source_noninferiority_gate_passed: bool,
    analysis_plan_sha256: str,
    bootstrap_resamples: int = 10_000,
    bootstrap_seed: int = 1729,
) -> dict[str, Any]:
    """Aggregate one locked opening using participants, never windows, as replicates."""

    root = Path(output_root).resolve(strict=True)
    index_file = _resolve_file(index_path, output_root=root, role="locked target index")
    index_value = load_json_strict(index_file)
    index = _require_mapping(index_value, name="locked target index")
    _validate_self_hash(index, field="record_sha256", role="locked target index")
    required_index = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "target_opening_number": 1,
        "target_information_used_for_model_selection": False,
    }
    index_mismatches = [
        key for key, expected in required_index.items() if index.get(key) != expected
    ]
    if index_mismatches:
        raise LockedStatisticsError(f"locked target index mismatch: {index_mismatches}")
    class_names_value = index.get("class_names")
    if not isinstance(class_names_value, list) or len(class_names_value) < 2:
        raise LockedStatisticsError("locked target index lacks class names")
    class_names = tuple(str(value) for value in class_names_value)
    entries = index.get("results")
    if not isinstance(entries, list) or len(entries) != index.get("model_seed_result_count"):
        raise LockedStatisticsError("locked target result index is incomplete")
    results = [
        _load_seed_result(
            _require_mapping(entry, name="result index entry"),
            output_root=root,
            expected_class_names=class_names,
        )
        for entry in entries
    ]
    if not results:
        raise LockedStatisticsError("locked target index has no results")
    alignment = (
        results[0]["window_ids"],
        results[0]["participant_ids"],
        results[0]["labels"].tolist(),
    )
    if any(
        (result["window_ids"], result["participant_ids"], result["labels"].tolist()) != alignment
        for result in results[1:]
    ):
        raise LockedStatisticsError("model/seed target arrays do not share exact ordered labels")
    seeds = tuple(int(seed) for seed in required_seed_order)
    if not seeds or len(set(seeds)) != len(seeds):
        raise LockedStatisticsError("required seeds must be non-empty and unique")
    by_model: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_model.setdefault(str(result["model_id"]), []).append(result)
    for model_id, model_results in by_model.items():
        observed = sorted(int(result["seed"]) for result in model_results)
        if observed != sorted(seeds):
            raise LockedStatisticsError(f"{model_id} does not have the exact required seed set")
        model_results.sort(key=lambda result: seeds.index(int(result["seed"])))
    references = tuple(str(value) for value in eligible_reference_model_ids)
    if (
        candidate_model_id not in by_model
        or not references
        or len(set(references)) != len(references)
        or candidate_model_id in references
        or any(reference not in by_model for reference in references)
    ):
        raise LockedStatisticsError("candidate/reference model analysis family is invalid")

    model_summaries: list[dict[str, Any]] = []
    participant_means: dict[str, dict[str, float]] = {}
    for model_id in sorted(by_model, key=str.casefold):
        model_results = by_model[model_id]
        per_seed_primary = [
            cast(Mapping[str, Any], result["report"])["primary"] for result in model_results
        ]
        participant_by_seed = [
            _participant_metric(cast(Mapping[str, Any], result["report"]))
            for result in model_results
        ]
        participant_ids = sorted(participant_by_seed[0], key=lambda value: (len(value), value))
        if any(set(values) != set(participant_ids) for values in participant_by_seed):
            raise LockedStatisticsError(f"{model_id} participant sets differ across seeds")
        averaged = {
            participant: _mean([values[participant] for values in participant_by_seed])
            for participant in participant_ids
        }
        participant_means[model_id] = averaged
        averaged_values = np.asarray(list(averaged.values()), dtype=np.float64)
        calibration_rows = [
            cast(Mapping[str, Any], result["report"])["calibration"] for result in model_results
        ]
        model_summaries.append(
            {
                "model_id": model_id,
                "seed_order": list(seeds),
                "seed_level_primary": [
                    {"seed": seed, **dict(cast(Mapping[str, Any], primary))}
                    for seed, primary in zip(seeds, per_seed_primary, strict=True)
                ],
                "participant_seed_averaged": {
                    "mean_macro_f1": float(averaged_values.mean()),
                    "worst_macro_f1": float(averaged_values.min()),
                    "lower_decile_macro_f1": float(
                        np.quantile(averaged_values, 0.1, method="linear")
                    ),
                    "bootstrap_mean_macro_f1": participant_bootstrap_interval(
                        averaged,
                        resamples=bootstrap_resamples,
                        seed=bootstrap_seed,
                    ),
                    "participant_values": averaged,
                },
                "calibration_seed_mean": {
                    metric: _mean(
                        [float(cast(Mapping[str, Any], row)[metric]) for row in calibration_rows]
                    )
                    for metric in ("negative_log_likelihood", "multiclass_brier_score", "ece")
                },
                "source_result_record_sha256s": [
                    result["record_sha256"] for result in model_results
                ],
                "prediction_artifact_sha256s": [
                    result["prediction_sha256"] for result in model_results
                ],
            }
        )

    comparisons: list[dict[str, Any]] = []
    for reference in references:
        comparison = paired_participant_comparison(
            participant_means[reference],
            participant_means[candidate_model_id],
        )
        comparisons.append({"reference_model_id": reference, **comparison})
    permutation_adjusted = holm_adjust(
        [float(comparison["permutation"]["two_sided_p_value"]) for comparison in comparisons]
    )
    wilcoxon_adjusted = holm_adjust(
        [float(comparison["wilcoxon"]["two_sided_p_value"]) for comparison in comparisons]
    )
    for comparison, permutation_p, wilcoxon_p in zip(
        comparisons, permutation_adjusted, wilcoxon_adjusted, strict=True
    ):
        comparison["multiplicity_family"] = "candidate_vs_predeclared_eligible_baselines"
        comparison["holm_adjusted_permutation_p_value"] = permutation_p
        comparison["holm_adjusted_wilcoxon_p_value"] = wilcoxon_p

    summary_by_model = {row["model_id"]: row for row in model_summaries}
    candidate_primary = summary_by_model[candidate_model_id]["participant_seed_averaged"]
    strongest_mean_reference = max(
        references,
        key=lambda model_id: summary_by_model[model_id]["participant_seed_averaged"][
            "mean_macro_f1"
        ],
    )
    strongest_tail_reference = max(
        references,
        key=lambda model_id: (
            summary_by_model[model_id]["participant_seed_averaged"]["worst_macro_f1"],
            summary_by_model[model_id]["participant_seed_averaged"]["lower_decile_macro_f1"],
        ),
    )
    best_mean = summary_by_model[strongest_mean_reference]["participant_seed_averaged"]
    best_tail = summary_by_model[strongest_tail_reference]["participant_seed_averaged"]
    mean_improved = candidate_primary["mean_macro_f1"] > best_mean["mean_macro_f1"]
    tail_improved = (
        candidate_primary["worst_macro_f1"] > best_tail["worst_macro_f1"]
        and candidate_primary["lower_decile_macro_f1"] > best_tail["lower_decile_macro_f1"]
    )
    hypothesis_supported = bool(
        mean_improved and tail_improved and source_noninferiority_gate_passed
    )
    payload: dict[str, Any] = {
        "schema_version": LOCKED_STATISTICS_SCHEMA_VERSION,
        "record_kind": "locked_target_participant_statistics",
        "status": "complete_create_only",
        "evidence_status": LOCKED_TARGET_EVIDENCE_STATUS,
        "analysis_plan_sha256": analysis_plan_sha256,
        "locked_target_index_record_sha256": index["record_sha256"],
        "final_freeze_inventory_sha256": index["final_freeze_inventory_sha256"],
        "target_seal_id": index["target_seal_id"],
        "required_seed_order": list(seeds),
        "participant_count": len(set(alignment[1])),
        "statistical_unit": "participant",
        "seed_handling": "participant macro-F1 averaged over predeclared seeds before inference",
        "bootstrap": {
            "method": "participant_cluster_percentile",
            "resamples": bootstrap_resamples,
            "seed": bootstrap_seed,
        },
        "models": model_summaries,
        "candidate_model_id": candidate_model_id,
        "eligible_reference_model_ids": list(references),
        "candidate_comparisons": comparisons,
        "hypothesis_decision": {
            "supported": hypothesis_supported,
            "mean_target_participant_macro_f1_improved": mean_improved,
            "worst_and_lower_decile_improved": tail_improved,
            "source_noninferiority_gate_passed": source_noninferiority_gate_passed,
            "strongest_mean_reference_model_id": strongest_mean_reference,
            "strongest_tail_reference_model_id": strongest_tail_reference,
            "rule": (
                "support requires higher candidate target participant-mean macro-F1 and both "
                "higher worst-participant and lower-decile macro-F1 than every predeclared "
                "eligible baseline, plus the pre-opening source non-inferiority gate"
            ),
        },
        "independence_note": "overlapping or repeated windows are never statistical replicates",
        "target_information_used_for_model_selection": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    destination_path = Path(destination)
    if not destination_path.is_absolute():
        destination_path = root / destination_path
    atomic_write_json_new(payload, destination_path, allowed_root=root)
    return payload
