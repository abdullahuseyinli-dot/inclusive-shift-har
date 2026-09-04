"""Active labelled-query resolution of participant-specific posture semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_nested import _lower_fraction_mean
from inclusive_shift_har.experiments.microstate_posture_graph_nested import _read_hashed_record
from inclusive_shift_har.experiments.semantic_anchor_reconciliation import (
    reconcile_posture_semantics,
    select_semantic_anchors,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
QueryStrategy = Literal["active_information", "random_hash"]

_CLASS_NAMES = ("mobility", "sitting", "standing")


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    method = _mapping(config.get("method"), name="method")
    if tuple(int(item) for item in method.get("maximum_queries_per_participant", [])) != (
        1,
        2,
        3,
    ):
        raise ValueError("semantic sentinel query budgets changed")
    if float(method.get("minimum_absolute_log_bayes_factor", -1.0)) != np.log(3.0):
        raise ValueError("semantic sentinel Bayes threshold changed")
    if float(method.get("minimum_predicted_stationary_probability", -1.0)) != 0.5:
        raise ValueError("semantic sentinel stationary threshold changed")
    if method.get("query_windows_excluded_from_evaluation") is not True:
        raise ValueError("semantic sentinel queries must be excluded from evaluation")
    if tuple(config.get("base_models", {})) != ("gsp", "rmrp"):
        raise ValueError("semantic sentinel base-model order changed")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("zero_shot_claim_allowed") is not False
        or policy.get("labelled_personalization_only") is not True
        or policy.get("target_data_allowed") is not False
    ):
        raise ValueError("semantic sentinel claim boundary changed")
    return config


def _query_order(
    probabilities: FloatArray,
    window_ids: StringArray,
    *,
    strategy: QueryStrategy,
    seed: int,
    minimum_stationary_probability: float,
) -> list[int]:
    stationary = probabilities[:, 1] + probabilities[:, 2]
    conditional_sitting = probabilities[:, 1] / np.maximum(stationary, 1e-12)
    information = stationary * np.abs(
        np.log(np.maximum(conditional_sitting, 1e-12))
        - np.log(np.maximum(1.0 - conditional_sitting, 1e-12))
    )
    eligible = np.flatnonzero(stationary >= minimum_stationary_probability).tolist()
    fallback = np.flatnonzero(stationary < minimum_stationary_probability).tolist()
    if strategy == "active_information":

        def key(index: int) -> tuple[float | str, ...]:
            return (-float(information[index]), str(window_ids[index]))

    elif strategy == "random_hash":

        def key(index: int) -> tuple[float | str, ...]:
            return (
                hashlib.sha256(f"{seed}|{window_ids[index]}".encode()).hexdigest(),
                str(window_ids[index]),
            )

    else:
        raise ValueError(f"unknown query strategy {strategy!r}")
    return sorted(eligible, key=key) + sorted(fallback, key=key)


def run_semantic_gauge_sentinel_for_participant(
    probabilities: NDArray[np.floating[Any]],
    labels: NDArray[np.integer[Any]],
    window_ids: NDArray[np.str_],
    *,
    maximum_queries: int,
    minimum_absolute_log_bayes_factor: float,
    minimum_stationary_probability: float,
    strategy: QueryStrategy,
    seed: int,
) -> tuple[FloatArray, BoolArray, dict[str, Any]]:
    """Query without label peeking, then retain or swap sitting/standing semantics."""

    values = np.asarray(probabilities, dtype=np.float64)
    truth = np.asarray(labels, dtype=np.int64)
    windows = np.asarray(window_ids, dtype=np.str_)
    if (
        values.ndim != 2
        or values.shape[1] != 3
        or truth.shape != (values.shape[0],)
        or windows.shape != truth.shape
        or values.shape[0] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("semantic sentinel inputs must be aligned finite arrays")
    if maximum_queries < 1 or minimum_absolute_log_bayes_factor < 0:
        raise ValueError("semantic sentinel budget and threshold are invalid")
    order = _query_order(
        values,
        windows,
        strategy=strategy,
        seed=seed,
        minimum_stationary_probability=minimum_stationary_probability,
    )
    queried = np.zeros(truth.size, dtype=np.bool_)
    log_bayes_swap_over_identity = 0.0
    query_records: list[dict[str, Any]] = []
    for index in order[:maximum_queries]:
        queried[index] = True
        observed = int(truth[index])
        increment = 0.0
        if observed in {1, 2}:
            other = 2 if observed == 1 else 1
            increment = float(
                np.log(max(float(values[index, other]), 1e-12))
                - np.log(max(float(values[index, observed]), 1e-12))
            )
            log_bayes_swap_over_identity += increment
        query_records.append(
            {
                "window_id": str(windows[index]),
                "observed_label": observed,
                "log_bayes_increment": increment,
                "cumulative_log_bayes_swap_over_identity": log_bayes_swap_over_identity,
            }
        )
        if abs(log_bayes_swap_over_identity) >= minimum_absolute_log_bayes_factor:
            break
    swap = log_bayes_swap_over_identity >= minimum_absolute_log_bayes_factor
    adjusted = values.copy()
    if swap:
        adjusted[:, [1, 2]] = adjusted[:, [2, 1]]
    return (
        adjusted,
        queried,
        {
            "strategy": strategy,
            "maximum_queries": maximum_queries,
            "query_count": int(queried.sum()),
            "decision": "swap_sitting_standing" if swap else "retain_identity",
            "threshold_reached": abs(log_bayes_swap_over_identity)
            >= minimum_absolute_log_bayes_factor,
            "log_bayes_swap_over_identity": log_bayes_swap_over_identity,
            "queries": query_records,
        },
    )


def _add_bottom_tail(report: dict[str, Any]) -> None:
    values = [
        float(item["macro_f1"]) for item in cast(list[dict[str, Any]], report["participants"])
    ]
    report["primary"]["bottom_30_percent_participant_macro_f1"] = _lower_fraction_mean(values)


def _report(
    labels: IntArray,
    probabilities: FloatArray,
    participants: StringArray,
    evaluation_mask: BoolArray,
) -> dict[str, Any]:
    report = classification_report(
        labels[evaluation_mask],
        probabilities[evaluation_mask],
        participants[evaluation_mask].tolist(),
        class_names=_CLASS_NAMES,
    )
    _add_bottom_tail(report)
    return report


def _load_base(path: Path) -> tuple[dict[str, Any], FloatArray, IntArray, StringArray, StringArray]:
    record = _read_hashed_record(path)
    if (
        record.get("status") != "complete_target_sealed"
        or record.get("target_subject_or_window_records_loaded") is not False
        or record.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("semantic sentinel requires target-sealed source OOF predictions")
    prediction_record = _mapping(record.get("predictions"), name="prediction record")
    prediction_path = Path(str(prediction_record["path"]))
    if sha256_file(prediction_path) != str(prediction_record["sha256"]):
        raise ValueError("semantic sentinel base predictions changed")
    with np.load(prediction_path, allow_pickle=False) as loaded:
        probability = np.asarray(loaded["probabilities"], dtype=np.float64)
        labels = np.asarray(loaded["labels"], dtype=np.int64)
        participants = np.asarray(loaded["participant_ids"], dtype=np.str_)
        windows = np.asarray(loaded["window_ids"], dtype=np.str_)
    if (
        probability.shape != (labels.size, 3)
        or participants.shape != labels.shape
        or windows.shape != labels.shape
    ):
        raise ValueError("semantic sentinel base prediction arrays are misaligned")
    return record, probability, labels, participants, windows


def _run_strategy(
    probability: FloatArray,
    labels: IntArray,
    participants: StringArray,
    windows: StringArray,
    *,
    maximum_queries: int,
    threshold: float,
    minimum_stationary_probability: float,
    strategy: QueryStrategy,
    seed: int,
) -> tuple[FloatArray, BoolArray, list[dict[str, Any]]]:
    adjusted = probability.copy()
    queried = np.zeros(labels.size, dtype=np.bool_)
    records: list[dict[str, Any]] = []
    for participant in sorted(set(participants.tolist()), key=int):
        mask = participants == participant
        participant_adjusted, participant_queried, decision = (
            run_semantic_gauge_sentinel_for_participant(
                probability[mask],
                labels[mask],
                windows[mask],
                maximum_queries=maximum_queries,
                minimum_absolute_log_bayes_factor=threshold,
                minimum_stationary_probability=minimum_stationary_probability,
                strategy=strategy,
                seed=seed + int(participant) * 1009,
            )
        )
        adjusted[mask] = participant_adjusted
        queried[np.flatnonzero(mask)] = participant_queried
        decision["participant_id"] = participant
        records.append(decision)
    return adjusted, queried, records


def _run_fixed_sar(
    probability: FloatArray,
    labels: IntArray,
    participants: StringArray,
    windows: StringArray,
    *,
    threshold: float,
    seed: int,
) -> tuple[FloatArray, BoolArray, list[dict[str, Any]]]:
    adjusted = probability.copy()
    queried = np.zeros(labels.size, dtype=np.bool_)
    decisions: list[dict[str, Any]] = []
    for participant in sorted(set(participants.tolist()), key=int):
        mask = participants == participant
        anchors = select_semantic_anchors(
            labels,
            participants,
            windows,
            participant_id=participant,
            budget_per_class=1,
            seed=seed,
        )
        reconciled, decision = reconcile_posture_semantics(
            probability[mask],
            labels[mask],
            anchors[mask],
            minimum_log_bayes_factor=threshold,
        )
        adjusted[mask] = reconciled
        queried |= anchors
        decision["participant_id"] = participant
        decisions.append(decision)
    return adjusted, queried, decisions


def run_active_semantic_gauge_sentinel(
    *, config_path: Path, output_directory: Path, code_commit: str
) -> dict[str, Any]:
    """Evaluate active, random, fixed-SAR, and no-adaptation labelled-personalization paths."""

    if output_directory.exists():
        raise FileExistsError(f"semantic sentinel output already exists: {output_directory}")
    config = _load_config(config_path)
    method = cast(dict[str, Any], config["method"])
    threshold = float(method["minimum_absolute_log_bayes_factor"])
    minimum_stationary = float(method["minimum_predicted_stationary_probability"])
    seed = int(method["seed"])
    output_directory.mkdir(parents=True)
    base_records: dict[str, Any] = {}
    prediction_payload: dict[str, NDArray[Any]] = {}
    for base_name, raw_reference in cast(dict[str, Any], config["base_models"]).items():
        reference = _mapping(raw_reference, name=f"base {base_name}")
        path = Path(str(reference["result_path"]))
        base, probability, labels, participants, windows = _load_base(path)
        budgets: list[dict[str, Any]] = []
        for budget in cast(list[int], method["maximum_queries_per_participant"]):
            active_probability, active_queries, active_decisions = _run_strategy(
                probability,
                labels,
                participants,
                windows,
                maximum_queries=int(budget),
                threshold=threshold,
                minimum_stationary_probability=minimum_stationary,
                strategy="active_information",
                seed=seed,
            )
            random_probability, random_queries, random_decisions = _run_strategy(
                probability,
                labels,
                participants,
                windows,
                maximum_queries=int(budget),
                threshold=threshold,
                minimum_stationary_probability=minimum_stationary,
                strategy="random_hash",
                seed=seed,
            )
            active_evaluation = ~active_queries
            random_evaluation = ~random_queries
            budgets.append(
                {
                    "maximum_queries_per_participant": int(budget),
                    "active": {
                        "total_query_count": int(active_queries.sum()),
                        "decision_counts": {
                            decision: sum(item["decision"] == decision for item in active_decisions)
                            for decision in ("retain_identity", "swap_sitting_standing")
                        },
                        "decisions": active_decisions,
                        "same_remaining_window_no_adaptation_report": _report(
                            labels,
                            probability,
                            participants,
                            active_evaluation,
                        ),
                        "adapted_report": _report(
                            labels,
                            active_probability,
                            participants,
                            active_evaluation,
                        ),
                    },
                    "random": {
                        "total_query_count": int(random_queries.sum()),
                        "decision_counts": {
                            decision: sum(item["decision"] == decision for item in random_decisions)
                            for decision in ("retain_identity", "swap_sitting_standing")
                        },
                        "decisions": random_decisions,
                        "same_remaining_window_no_adaptation_report": _report(
                            labels,
                            probability,
                            participants,
                            random_evaluation,
                        ),
                        "adapted_report": _report(
                            labels,
                            random_probability,
                            participants,
                            random_evaluation,
                        ),
                    },
                }
            )
            prefix = f"{base_name}__budget_{budget}"
            prediction_payload[f"{prefix}__active_probabilities"] = active_probability
            prediction_payload[f"{prefix}__active_query_mask"] = active_queries
            prediction_payload[f"{prefix}__random_probabilities"] = random_probability
            prediction_payload[f"{prefix}__random_query_mask"] = random_queries
        sar_probability, sar_queries, sar_decisions = _run_fixed_sar(
            probability,
            labels,
            participants,
            windows,
            threshold=threshold,
            seed=int(method["fixed_sar_seed"]),
        )
        sar_evaluation = ~sar_queries
        base_records[base_name] = {
            "base_result": {
                "path": path.as_posix(),
                "sha256": sha256_file(path),
                "record_sha256": base["record_sha256"],
            },
            "budgets": budgets,
            "fixed_sar_1_plus_1": {
                "total_query_count": int(sar_queries.sum()),
                "decisions": sar_decisions,
                "same_remaining_window_no_adaptation_report": _report(
                    labels, probability, participants, sar_evaluation
                ),
                "adapted_report": _report(labels, sar_probability, participants, sar_evaluation),
            },
        }
        prediction_payload[f"{base_name}__base_probabilities"] = probability
        prediction_payload[f"{base_name}__labels"] = labels
        prediction_payload[f"{base_name}__participant_ids"] = participants
        prediction_payload[f"{base_name}__window_ids"] = windows
        prediction_payload[f"{base_name}__sar_probabilities"] = sar_probability
        prediction_payload[f"{base_name}__sar_query_mask"] = sar_queries
    prediction_path = output_directory / "personalization_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "active_semantic_gauge_sentinel_source_personalization",
        "status": "complete_labelled_personalization_not_zero_shot",
        "evidence_status": "post_zero_shot_failure_source_personalization_not_confirmatory",
        "code_commit": code_commit,
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "base_models": base_records,
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "query_labels_used_only_after_query_selection": True,
        "query_windows_excluded_from_evaluation": True,
        "zero_shot_claim_allowed": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "confirmatory_claim_allowed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output_directory / "result.json").open("xb") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_active_semantic_gauge_sentinel(
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    summary: dict[str, Any] = {}
    for base_name, base in result["base_models"].items():
        summary[base_name] = {
            str(item["maximum_queries_per_participant"]): {
                "active_mean": item["active"]["adapted_report"]["primary"][
                    "mean_participant_macro_f1"
                ],
                "active_query_count": item["active"]["total_query_count"],
                "random_mean": item["random"]["adapted_report"]["primary"][
                    "mean_participant_macro_f1"
                ],
                "random_query_count": item["random"]["total_query_count"],
            }
            for item in base["budgets"]
        }
        summary[base_name]["fixed_sar_1_plus_1"] = {
            "mean": base["fixed_sar_1_plus_1"]["adapted_report"]["primary"][
                "mean_participant_macro_f1"
            ],
            "query_count": base["fixed_sar_1_plus_1"]["total_query_count"],
        }
    print(
        json.dumps(
            {
                "status": result["status"],
                "summary": summary,
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
