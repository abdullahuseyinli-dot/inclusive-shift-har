"""Apply a fixed labelled semantic gauge to frozen HERA source predictions.

The runner is intentionally prediction-only: it cannot load raw or target data and it
does not fit a new classifier. Query selection is label-free; labels are consumed only
after a selected window is removed from evaluation. Both hard and fixed soft gauges are
reported with active and random controls.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
QueryStrategy = Literal["active_information", "random_hash"]

CLASS_NAMES = ("mobility", "sitting", "standing")
PARTICIPANTS = tuple(str(index) for index in range(1, 11))
QUERY_BUDGETS = (1, 2, 3)
THRESHOLD = math.log(3.0)
MIN_STATIONARY = 0.5
ACTIVE_SEED = 271828
SAR_SEED = 314159
BOOTSTRAP_SEED = 1729
BOOTSTRAP_RESAMPLES = 10_000


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    if config.get("experiment_id") != "inclusivehar-hera-posture-gauge-v1":
        raise ValueError("unexpected HERA posture-gauge experiment ID")
    method = _mapping(config.get("method"), name="method")
    if (
        tuple(int(item) for item in method.get("maximum_queries_per_participant", []))
        != QUERY_BUDGETS
    ):
        raise ValueError("query budget drift")
    if not math.isclose(float(method.get("minimum_absolute_log_bayes_factor", -1.0)), THRESHOLD):
        raise ValueError("Bayes threshold drift")
    if float(method.get("minimum_predicted_stationary_probability", -1.0)) != MIN_STATIONARY:
        raise ValueError("stationary filter drift")
    if method.get("query_windows_excluded_from_evaluation") is not True:
        raise ValueError("queried windows must be excluded")
    if int(method.get("active_random_seed", -1)) != ACTIVE_SEED:
        raise ValueError("active seed drift")
    if int(method.get("fixed_sar_seed", -1)) != SAR_SEED:
        raise ValueError("SAR seed drift")
    soft = _mapping(method.get("soft_gauge"), name="soft gauge")
    if (
        float(soft.get("prior_swap_probability", -1.0)) != 0.5
        or float(soft.get("maximum_swap_weight", -1.0)) != 1.0
        or float(soft.get("evidence_temperature", -1.0)) != 1.0
    ):
        raise ValueError("soft gauge constants drift")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("zero_shot_claim_allowed") is not False
        or policy.get("labelled_personalization_only") is not True
        or policy.get("target_data_allowed") is not False
        or policy.get("confirmatory_claim_allowed") is not False
    ):
        raise ValueError("claim boundary drift")
    return config


def _load_prediction_artifact(
    path: Path, *, probability_key: str
) -> tuple[FloatArray, IntArray, StringArray, StringArray]:
    with np.load(path, allow_pickle=False) as archive:
        required = {probability_key, "labels", "participant_ids", "window_ids"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"prediction artifact is missing {sorted(required - set(archive.files))}"
            )
        probabilities = np.asarray(archive[probability_key], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        participants = np.asarray(archive["participant_ids"], dtype=np.str_)
        windows = np.asarray(archive["window_ids"], dtype=np.str_)
    if probabilities.ndim != 2 or probabilities.shape[1] != 3:
        raise ValueError("HERA probabilities must have shape [n,3]")
    if (
        labels.shape != (probabilities.shape[0],)
        or participants.shape != labels.shape
        or windows.shape != labels.shape
    ):
        raise ValueError("HERA prediction arrays are not aligned")
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0.0):
        raise ValueError("HERA probabilities must be finite and non-negative")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError("HERA probability rows must sum to one")
    if labels.size == 0 or np.any(~np.isin(labels, [0, 1, 2])):
        raise ValueError("HERA labels must use class indices 0,1,2")
    if tuple(sorted(set(participants.tolist()), key=int)) != PARTICIPANTS:
        raise ValueError("HERA artifact must contain exactly source participants 1--10")
    if len(set(windows.tolist())) != windows.size or np.any(np.char.str_len(windows) == 0):
        raise ValueError("HERA window IDs must be unique and non-empty")
    return probabilities, labels, participants, windows


def _query_order(
    probabilities: FloatArray,
    windows: StringArray,
    *,
    strategy: QueryStrategy,
    seed: int,
) -> list[int]:
    stationary = probabilities[:, 1] + probabilities[:, 2]
    sitting = probabilities[:, 1] / np.maximum(stationary, 1e-12)
    utility = stationary * np.abs(
        np.log(np.maximum(sitting, 1e-12)) - np.log(np.maximum(1.0 - sitting, 1e-12))
    )
    eligible = np.flatnonzero(stationary >= MIN_STATIONARY).tolist()
    fallback = np.flatnonzero(stationary < MIN_STATIONARY).tolist()
    if strategy == "active_information":

        def active_key(index: int) -> tuple[float, str]:
            return (-float(utility[index]), str(windows[index]))

        return sorted(eligible, key=active_key) + sorted(fallback, key=active_key)
    elif strategy == "random_hash":

        def random_key(index: int) -> tuple[str, str]:
            return (
                hashlib.sha256(f"{seed}|{windows[index]}".encode()).hexdigest(),
                str(windows[index]),
            )

        return sorted(eligible, key=random_key) + sorted(fallback, key=random_key)
    else:
        raise ValueError(f"unknown query strategy {strategy!r}")


def _soft_adjust(probabilities: FloatArray, swap_posterior: float) -> FloatArray:
    swapped = probabilities.copy()
    swapped[:, [1, 2]] = swapped[:, [2, 1]]
    adjusted = (1.0 - swap_posterior) * probabilities + swap_posterior * swapped
    adjusted[:, 0] = probabilities[:, 0]
    stationary = 1.0 - probabilities[:, 0]
    adjusted[:, 1:] *= stationary[:, None] / np.maximum(adjusted[:, 1:].sum(axis=1)[:, None], 1e-12)
    return np.asarray(adjusted, dtype=np.float64)


def _participant_gauge(
    probabilities: FloatArray,
    labels: IntArray,
    windows: StringArray,
    *,
    maximum_queries: int,
    strategy: QueryStrategy,
    seed: int,
) -> tuple[FloatArray, FloatArray, BoolArray, dict[str, Any]]:
    if maximum_queries not in QUERY_BUDGETS:
        raise ValueError("unsupported query budget")
    order = _query_order(probabilities, windows, strategy=strategy, seed=seed)
    queried = np.zeros(labels.size, dtype=np.bool_)
    log_bayes = 0.0
    records: list[dict[str, Any]] = []
    for index in order[:maximum_queries]:
        queried[index] = True
        observed = int(labels[index])
        increment = 0.0
        if observed in (1, 2):
            other = 2 if observed == 1 else 1
            increment = float(
                np.log(max(probabilities[index, other], 1e-12))
                - np.log(max(probabilities[index, observed], 1e-12))
            )
            log_bayes += increment
        records.append(
            {
                "window_id": str(windows[index]),
                "observed_label": observed,
                "log_bayes_increment": increment,
                "cumulative_log_bayes_swap_over_identity": log_bayes,
            }
        )
        if abs(log_bayes) >= THRESHOLD:
            break
    hard_swap = log_bayes >= THRESHOLD
    hard = probabilities.copy()
    if hard_swap:
        hard[:, [1, 2]] = hard[:, [2, 1]]
    soft_posterior = float(1.0 / (1.0 + np.exp(-np.clip(log_bayes, -40.0, 40.0))))
    soft = _soft_adjust(probabilities, soft_posterior)
    return (
        hard,
        soft,
        queried,
        {
            "strategy": strategy,
            "maximum_queries": maximum_queries,
            "query_count": int(queried.sum()),
            "decision": "swap_sitting_standing" if hard_swap else "retain_identity",
            "threshold_reached": abs(log_bayes) >= THRESHOLD,
            "log_bayes_swap_over_identity": log_bayes,
            "soft_posterior_swap_probability": soft_posterior,
            "queries": records,
        },
    )


def _apply_strategy(
    probabilities: FloatArray,
    labels: IntArray,
    participants: StringArray,
    windows: StringArray,
    *,
    maximum_queries: int,
    strategy: QueryStrategy,
    seed: int,
) -> tuple[FloatArray, FloatArray, BoolArray, list[dict[str, Any]]]:
    hard = probabilities.copy()
    soft = probabilities.copy()
    queried = np.zeros(labels.size, dtype=np.bool_)
    decisions: list[dict[str, Any]] = []
    for participant in PARTICIPANTS:
        mask = participants == participant
        hard_p, soft_p, participant_queried, decision = _participant_gauge(
            probabilities[mask],
            labels[mask],
            windows[mask],
            maximum_queries=maximum_queries,
            strategy=strategy,
            seed=seed + int(participant) * 1009,
        )
        hard[mask] = hard_p
        soft[mask] = soft_p
        indices = np.flatnonzero(mask)
        queried[indices] = participant_queried
        decision["participant_id"] = participant
        decisions.append(decision)
    return hard, soft, queried, decisions


def _select_sar_anchors(
    labels: IntArray, participants: StringArray, windows: StringArray
) -> BoolArray:
    selected = np.zeros(labels.size, dtype=np.bool_)
    for participant in PARTICIPANTS:
        for class_index in (1, 2):
            candidates = np.flatnonzero(
                (participants == participant) & (labels == class_index)
            ).tolist()
            if len(candidates) <= 1:
                raise ValueError("SAR anchor budget leaves no posture evaluation rows")
            ranked = sorted(
                candidates,
                key=lambda index: hashlib.sha256(
                    f"{SAR_SEED}|{participant}|{windows[index]}".encode()
                ).hexdigest(),
            )
            selected[ranked[0]] = True
    return selected


def _apply_sar(
    probabilities: FloatArray, labels: IntArray, participants: StringArray, windows: StringArray
) -> tuple[FloatArray, BoolArray, list[dict[str, Any]]]:
    adjusted = probabilities.copy()
    selected = _select_sar_anchors(labels, participants, windows)
    decisions: list[dict[str, Any]] = []
    for participant in PARTICIPANTS:
        mask = participants == participant
        anchors = mask & selected
        rows = np.flatnonzero(anchors)
        truth = labels[rows]
        identity = float(np.log(np.maximum(probabilities[rows, truth], 1e-12)).sum())
        swapped_truth = np.where(truth == 1, 2, 1)
        swapped = float(np.log(np.maximum(probabilities[rows, swapped_truth], 1e-12)).sum())
        log_bayes = swapped - identity
        if log_bayes >= THRESHOLD:
            participant_rows = np.flatnonzero(mask)
            posture_columns = np.asarray([1, 2], dtype=np.intp)
            swapped_columns = np.asarray([2, 1], dtype=np.intp)
            adjusted[np.ix_(participant_rows, posture_columns)] = adjusted[
                np.ix_(participant_rows, swapped_columns)
            ]
        decisions.append(
            {
                "participant_id": participant,
                "anchor_count": int(rows.size),
                "anchor_window_ids": windows[rows].tolist(),
                "decision": "swap_sitting_standing"
                if log_bayes >= THRESHOLD
                else "retain_identity",
                "swap_log_bayes_factor": log_bayes,
            }
        )
    return adjusted, selected, decisions


def _participant_values(report: dict[str, Any]) -> dict[str, float]:
    return {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], report["participants"])
    }


def _bootstrap(delta: dict[str, float]) -> dict[str, Any]:
    names = sorted(delta, key=int)
    values = np.asarray([delta[name] for name in names], dtype=np.float64)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.integers(0, values.size, size=(BOOTSTRAP_RESAMPLES, values.size))
    means = values[draws].mean(axis=1)
    return {
        "mean_difference": float(values.mean()),
        "lower": float(np.quantile(means, 0.025)),
        "upper": float(np.quantile(means, 0.975)),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
    }


def _report(
    labels: IntArray, probabilities: FloatArray, participants: StringArray, mask: BoolArray
) -> dict[str, Any]:
    return classification_report(
        labels[mask], probabilities[mask], participants[mask].tolist(), class_names=CLASS_NAMES
    )


def _comparison(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    left = _participant_values(candidate)
    right = _participant_values(baseline)
    delta = {name: left[name] - right[name] for name in left}
    return {
        "paired_participant_macro_f1_difference": delta,
        "participant_wins": int(sum(value > 0.0 for value in delta.values())),
        "participant_harms": int(sum(value < 0.0 for value in delta.values())),
        "participant_ties": int(sum(value == 0.0 for value in delta.values())),
        "paired_bootstrap": _bootstrap(delta),
    }


def run_hera_posture_gauge(*, config_path: Path, output_directory: Path) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"create-only output already exists: {output_directory}")
    config = _load_config(config_path)
    input_cfg = _mapping(config.get("input"), name="input")
    prediction_path = Path(str(input_cfg["prediction_artifact"])).resolve()
    probabilities, labels, participants, windows = _load_prediction_artifact(
        prediction_path, probability_key=str(input_cfg["probability_key"])
    )
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    full_report = _report(labels, probabilities, participants, np.ones(labels.size, dtype=np.bool_))
    prediction_payload: dict[str, NDArray[Any]] = {
        "labels": labels,
        "participant_ids": participants,
        "window_ids": windows,
        "hera_v1_strict_probabilities": probabilities,
    }
    budget_records: list[dict[str, Any]] = []
    for budget in QUERY_BUDGETS:
        active_hard, active_soft, active_mask, active_decisions = _apply_strategy(
            probabilities,
            labels,
            participants,
            windows,
            maximum_queries=budget,
            strategy="active_information",
            seed=ACTIVE_SEED,
        )
        random_hard, random_soft, random_mask, random_decisions = _apply_strategy(
            probabilities,
            labels,
            participants,
            windows,
            maximum_queries=budget,
            strategy="random_hash",
            seed=ACTIVE_SEED,
        )
        active_base = _report(labels, probabilities, participants, ~active_mask)
        random_base = _report(labels, probabilities, participants, ~random_mask)
        active_hard_report = _report(labels, active_hard, participants, ~active_mask)
        active_soft_report = _report(labels, active_soft, participants, ~active_mask)
        random_hard_report = _report(labels, random_hard, participants, ~random_mask)
        random_soft_report = _report(labels, random_soft, participants, ~random_mask)
        budget_records.append(
            {
                "maximum_queries_per_participant": budget,
                "active": {
                    "total_query_count": int(active_mask.sum()),
                    "decision_counts": {
                        name: sum(item["decision"] == name for item in active_decisions)
                        for name in ("retain_identity", "swap_sitting_standing")
                    },
                    "decisions": active_decisions,
                    "same_remaining_window_report": active_base,
                    "hard_report": active_hard_report,
                    "soft_report": active_soft_report,
                    "hard_comparison": _comparison(active_hard_report, active_base),
                    "soft_comparison": _comparison(active_soft_report, active_base),
                },
                "random": {
                    "total_query_count": int(random_mask.sum()),
                    "decision_counts": {
                        name: sum(item["decision"] == name for item in random_decisions)
                        for name in ("retain_identity", "swap_sitting_standing")
                    },
                    "decisions": random_decisions,
                    "same_remaining_window_report": random_base,
                    "hard_report": random_hard_report,
                    "soft_report": random_soft_report,
                    "hard_comparison": _comparison(random_hard_report, random_base),
                    "soft_comparison": _comparison(random_soft_report, random_base),
                },
            }
        )
        prefix = f"budget_{budget}"
        prediction_payload.update(
            {
                f"active_hard_{prefix}": active_hard,
                f"active_soft_{prefix}": active_soft,
                f"active_query_mask_{prefix}": active_mask,
                f"random_hard_{prefix}": random_hard,
                f"random_soft_{prefix}": random_soft,
                f"random_query_mask_{prefix}": random_mask,
            }
        )
    sar, sar_mask, sar_decisions = _apply_sar(probabilities, labels, participants, windows)
    sar_base = _report(labels, probabilities, participants, ~sar_mask)
    sar_report = _report(labels, sar, participants, ~sar_mask)
    prediction_payload.update({"sar_probabilities": sar, "sar_query_mask": sar_mask})
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "hera_posture_gauge_source_development_result",
        "experiment_id": "inclusivehar-hera-posture-gauge-v1",
        "status": "complete_labelled_personalization_source_development",
        "evidence_status": "reused_source_development_not_independent",
        "human_performance_claim": False,
        "claim_scope": "labelled participant-specific sitting/standing semantic calibration",
        "input": {
            "path": prediction_path.as_posix(),
            "sha256": sha256_file(prediction_path),
            "probability_key": str(input_cfg["probability_key"]),
        },
        "config": {"path": config_path.resolve().as_posix(), "sha256": sha256_file(config_path)},
        "implementation": {
            "path": Path(__file__).resolve().as_posix(),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "full_frozen_hera_report": full_report,
        "budgets": budget_records,
        "fixed_sar_1_plus_1": {
            "total_query_count": int(sar_mask.sum()),
            "decision_counts": {
                name: sum(item["decision"] == name for item in sar_decisions)
                for name in ("retain_identity", "swap_sitting_standing")
            },
            "decisions": sar_decisions,
            "same_remaining_window_report": sar_base,
            "hard_report": sar_report,
            "comparison": _comparison(sar_report, sar_base),
        },
        "temporal_stage": {
            "status": "blocked",
            "reason": "authoritative contiguous source session/trial order is unavailable in the released prediction artifact",
            "no_temporal_fit_launched": True,
        },
        "query_labels_used_only_after_query_selection": True,
        "query_windows_excluded_from_evaluation": True,
        "zero_shot_claim_allowed": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "runtime_seconds": time.perf_counter() - started,
        "software": {"numpy": importlib.metadata.version("numpy")},
    }
    prediction_path_out = output_directory / "predictions.npz"
    with prediction_path_out.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], prediction_payload))
    result["predictions"] = {
        "path": prediction_path_out.as_posix(),
        "sha256": sha256_file(prediction_path_out),
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output_directory / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    manifest: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "hera_posture_gauge_manifest",
        "artifacts": [
            {
                "path": "result.json",
                "sha256": sha256_file(output_directory / "result.json"),
                "size_bytes": (output_directory / "result.json").stat().st_size,
            },
            {
                "path": "predictions.npz",
                "sha256": sha256_file(prediction_path_out),
                "size_bytes": prediction_path_out.stat().st_size,
            },
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    with (output_directory / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    result = run_hera_posture_gauge(
        config_path=args.config.resolve(), output_directory=args.output_directory.resolve()
    )
    summary = {
        "status": result["status"],
        "full_mean": result["full_frozen_hera_report"]["primary"]["mean_participant_macro_f1"],
        "active": [
            {
                "budget": item["maximum_queries_per_participant"],
                "queries": item["active"]["total_query_count"],
                "hard_delta": item["active"]["hard_comparison"]["paired_bootstrap"][
                    "mean_difference"
                ],
                "soft_delta": item["active"]["soft_comparison"]["paired_bootstrap"][
                    "mean_difference"
                ],
            }
            for item in result["budgets"]
        ],
        "record_sha256": result["record_sha256"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
