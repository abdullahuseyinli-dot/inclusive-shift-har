"""Evaluate labelled per-user posture-semantic reconciliation on frozen OOF predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import (
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import _mapping
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    method = _mapping(config.get("method"), name="method")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    budgets = method.get("anchor_budgets_per_reconcilable_class")
    if not isinstance(budgets, list) or not budgets:
        raise ValueError("semantic-anchor config requires budgets")
    parsed = [int(item) for item in budgets]
    if parsed != sorted(set(parsed)) or any(item <= 0 for item in parsed):
        raise ValueError("semantic-anchor budgets must be unique ascending positive integers")
    if float(method.get("minimum_log_bayes_factor", -1.0)) < 0:
        raise ValueError("semantic-anchor log Bayes threshold must be nonnegative")
    if method.get("evaluation_labels_used_for_decision") is not False:
        raise ValueError("evaluation labels must be forbidden from adaptation decisions")
    if method.get("anchor_windows_excluded_from_evaluation") is not True:
        raise ValueError("anchor windows must be excluded from evaluation")
    if policy.get("zero_shot_claim_allowed") is not False:
        raise ValueError("labelled semantic reconciliation cannot claim zero-shot performance")
    if policy.get("target_data_allowed") is not False:
        raise ValueError("semantic-anchor development must forbid target data")
    return config


def select_semantic_anchors(
    labels: IntArray,
    participant_ids: StringArray,
    window_ids: StringArray,
    *,
    participant_id: str,
    budget_per_class: int,
    seed: int,
) -> NDArray[np.bool_]:
    """Select a replayable stratified labelled calibration set without time/order."""

    if labels.shape != participant_ids.shape or labels.shape != window_ids.shape:
        raise ValueError("semantic-anchor arrays must align")
    if budget_per_class <= 0:
        raise ValueError("semantic-anchor budget must be positive")
    selected = np.zeros(labels.size, dtype=np.bool_)
    for class_index in (1, 2):
        candidates = np.flatnonzero(
            (participant_ids == participant_id) & (labels == class_index)
        ).tolist()
        if len(candidates) <= budget_per_class:
            raise ValueError("semantic-anchor budget leaves no posture evaluation examples")
        ranked = sorted(
            candidates,
            key=lambda index: hashlib.sha256(
                f"{seed}|{participant_id}|{window_ids[index]}".encode()
            ).hexdigest(),
        )
        selected[ranked[:budget_per_class]] = True
    return selected


def reconcile_posture_semantics(
    probabilities: FloatArray,
    labels: IntArray,
    anchor_mask: NDArray[np.bool_],
    *,
    minimum_log_bayes_factor: float,
) -> tuple[FloatArray, dict[str, Any]]:
    """Choose identity or sitting/standing swap using anchor likelihood only."""

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or labels.shape != (values.shape[0],):
        raise ValueError("semantic reconciliation requires aligned three-class probabilities")
    if anchor_mask.shape != labels.shape or not np.any(anchor_mask):
        raise ValueError("semantic reconciliation requires an aligned non-empty anchor mask")
    if set(labels[anchor_mask].tolist()) != {1, 2}:
        raise ValueError("semantic anchors must contain both posture classes and no mobility")
    if minimum_log_bayes_factor < 0:
        raise ValueError("minimum_log_bayes_factor must be nonnegative")
    rows = np.flatnonzero(anchor_mask)
    truth = labels[anchor_mask]
    identity_log_likelihood = float(np.log(np.maximum(values[rows, truth], 1e-12)).sum())
    swapped_truth = np.where(truth == 1, 2, 1)
    swapped_log_likelihood = float(np.log(np.maximum(values[rows, swapped_truth], 1e-12)).sum())
    log_bayes_factor = swapped_log_likelihood - identity_log_likelihood
    swap = log_bayes_factor >= minimum_log_bayes_factor
    reconciled = values.copy()
    if swap:
        reconciled[:, [1, 2]] = reconciled[:, [2, 1]]
    return reconciled, {
        "decision": "swap_sitting_standing" if swap else "retain_identity",
        "identity_log_likelihood": identity_log_likelihood,
        "swapped_log_likelihood": swapped_log_likelihood,
        "swap_log_bayes_factor": log_bayes_factor,
        "minimum_log_bayes_factor": minimum_log_bayes_factor,
        "anchor_count": int(anchor_mask.sum()),
    }


def _participant_values(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], report["participants"])
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_semantic_anchor_reconciliation(
    *, nested_result_path: Path, config_path: Path, output_directory: Path
) -> dict[str, Any]:
    """Evaluate every fixed labelled-calibration budget on disjoint remaining windows."""

    if output_directory.exists():
        raise FileExistsError(f"semantic-anchor output already exists: {output_directory}")
    config = _load_config(config_path)
    nested = _mapping(load_json_strict(nested_result_path), name="nested result")
    if nested.get("status") != "complete_target_sealed":
        raise PermissionError("semantic anchors require a complete target-sealed source result")
    if nested.get("target_subject_or_window_records_loaded") is not False:
        raise PermissionError("semantic-anchor runner refuses target-bearing results")
    if nested.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("semantic-anchor runner refuses target-informed results")
    prediction_record = _mapping(nested.get("predictions"), name="predictions")
    prediction_path = Path(str(prediction_record["path"]))
    if sha256_file(prediction_path) != str(prediction_record["sha256"]):
        raise ValueError("frozen prediction artifact hash mismatch")
    with np.load(prediction_path, allow_pickle=False) as payload:
        probabilities = np.asarray(payload["probabilities"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        participants = np.asarray(payload["participant_ids"], dtype=np.str_)
        windows = np.asarray(payload["window_ids"], dtype=np.str_)
    class_names = tuple(str(item) for item in nested["aggregate_report"]["class_names"])
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("semantic reconciliation requires functional-core probabilities")
    method = _mapping(config["method"], name="method")
    threshold = float(method["minimum_log_bayes_factor"])
    seed = int(method["seed"])
    budgets = [int(item) for item in method["anchor_budgets_per_reconcilable_class"]]
    budget_records: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True)
    for budget in budgets:
        evaluation_mask = np.ones(labels.size, dtype=np.bool_)
        adapted = probabilities.copy()
        decisions: list[dict[str, Any]] = []
        anchor_ids: list[str] = []
        for participant_id in sorted(set(participants.tolist()), key=int):
            anchor_mask = select_semantic_anchors(
                labels,
                participants,
                windows,
                participant_id=participant_id,
                budget_per_class=budget,
                seed=seed,
            )
            participant_mask = participants == participant_id
            reconciled, decision = reconcile_posture_semantics(
                probabilities[participant_mask],
                labels[participant_mask],
                anchor_mask[participant_mask],
                minimum_log_bayes_factor=threshold,
            )
            adapted[participant_mask] = reconciled
            evaluation_mask[anchor_mask] = False
            selected_ids = windows[anchor_mask].tolist()
            anchor_ids.extend(selected_ids)
            decisions.append(
                {
                    "participant_id": participant_id,
                    **decision,
                    "anchor_window_ids_sha256": canonical_json_sha256(sorted(selected_ids)),
                    "evaluation_window_count": int((participant_mask & ~anchor_mask).sum()),
                }
            )
        baseline_report = classification_report(
            labels[evaluation_mask],
            probabilities[evaluation_mask],
            participants[evaluation_mask].tolist(),
            class_names=class_names,
        )
        adapted_report = classification_report(
            labels[evaluation_mask],
            adapted[evaluation_mask],
            participants[evaluation_mask].tolist(),
            class_names=class_names,
        )
        baseline_values = _participant_values(baseline_report)
        adapted_values = _participant_values(adapted_report)
        deltas = {
            participant: adapted_values[participant] - baseline_values[participant]
            for participant in baseline_values
        }
        budget_records.append(
            {
                "budget_per_posture_class": budget,
                "total_anchor_count": len(anchor_ids),
                "anchor_window_ids_sha256": canonical_json_sha256(sorted(anchor_ids)),
                "evaluation_window_count": int(evaluation_mask.sum()),
                "decision_counts": {
                    decision: sum(item["decision"] == decision for item in decisions)
                    for decision in ("retain_identity", "swap_sitting_standing")
                },
                "participant_decisions": decisions,
                "same_remaining_windows_zero_shot_report": baseline_report,
                "adapted_report": adapted_report,
                "paired_participant_comparison": paired_participant_comparison(
                    baseline_values, adapted_values
                ),
                "paired_delta_bootstrap": participant_bootstrap_interval(deltas),
            }
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "semantic_anchor_reconciliation_source_personalization",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_labelled_personalization_not_zero_shot_not_confirmatory",
        "claim_scope": "labelled per-user posture calibration only",
        "all_budgets_reported_without_selection": True,
        "nested_result": {
            "path": nested_result_path.as_posix(),
            "sha256": sha256_file(nested_result_path),
            "canonical_record_sha256": nested["record_sha256"],
        },
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "budgets": budget_records,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_semantic_anchor_reconciliation(
        nested_result_path=args.nested_result,
        config_path=args.config,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "claim_scope": result["claim_scope"],
                "budgets": [
                    {
                        "budget_per_posture_class": item["budget_per_posture_class"],
                        "zero_shot_mean": item["same_remaining_windows_zero_shot_report"][
                            "primary"
                        ]["mean_participant_macro_f1"],
                        "adapted_mean": item["adapted_report"]["primary"][
                            "mean_participant_macro_f1"
                        ],
                        "adapted_lower_decile": item["adapted_report"]["primary"][
                            "lower_decile_participant_macro_f1"
                        ],
                        "adapted_worst": item["adapted_report"]["primary"][
                            "worst_participant_macro_f1"
                        ],
                        "decision_counts": item["decision_counts"],
                    }
                    for item in result["budgets"]
                ],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
