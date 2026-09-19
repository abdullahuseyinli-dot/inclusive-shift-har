"""Retrospective proxy for the same-attachment reference architecture.

This module deliberately uses only the preserved CTGR source feature package.  It is
useful for checking whether the nine heads can consume an existing participant-held-out
feature table, but it is *not* the prospective same-attachment experiment: the source
package has no attachment IDs, continuous timestamps, bout boundaries, or raw gravity
streams.  Support rows are selected deterministically from each participant's release
order and the result is labelled ``retrospective_proxy_existing_source_features``.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.same_attachment_reference_training import (
    apply_scaler,
    calibrate_common_concentration,
    calibrate_variable_concentration,
    compose_probabilities,
    directional_features_batch,
    fit_binary_logistic,
    fit_constrained_posture,
    fit_map_prior,
    fit_multinomial_zero_sum,
    fit_weighted_scaler,
    kappa_from_spread,
    map_sitting_probability,
    point_reference_features,
    predict_binary,
    predict_multinomial,
    update_map_posterior,
)

FloatArray = NDArray[np.float64]

ARM_IDS = ("Z", "A", "B", "C", "D", "D-additive", "D-constant", "K", "MAP")
PHYSICS_INDEX = {
    "gravity_x_mean": 0,
    "gravity_x_std": 1,
    "gravity_y_mean": 15,
    "gravity_y_std": 16,
    "gravity_z_mean": 30,
    "gravity_z_std": 31,
    "acceleration_rms": 250,
    "gyroscope_rms": 265,
}


def _array_sha256(array: NDArray[Any]) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode("utf-8") + b"\0" + value.tobytes()).hexdigest()


def _unit_rows(values: NDArray[Any]) -> FloatArray:
    rows = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(rows, axis=1)
    if rows.ndim != 2 or rows.shape[1] != 3 or np.any(~np.isfinite(rows)):
        raise ValueError("gravity direction rows are invalid")
    if np.any(norms <= 1.0e-12):
        raise ValueError("gravity direction row is degenerate")
    return np.asarray(rows / norms[:, None], dtype=np.float64)


def load_source_feature_proxy(source_features_path: Path) -> dict[str, Any]:
    """Load only the documented source feature table and derive proxy inputs."""

    with np.load(source_features_path, allow_pickle=False) as archive:
        required = {"physics", "labels", "participants", "windows"}
        if not required.issubset(archive.files):
            raise ValueError(f"source feature archive is missing {sorted(required - set(archive.files))}")
        physics = np.asarray(archive["physics"], dtype=np.float64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        participants = np.asarray(archive["participants"], dtype=np.str_)
        windows = np.asarray(archive["windows"], dtype=np.str_)
    if physics.ndim != 2 or physics.shape[1] <= max(PHYSICS_INDEX.values()):
        raise ValueError("physics feature table has an unexpected shape")
    rows = physics.shape[0]
    if labels.shape != (rows,) or participants.shape != (rows,) or windows.shape != (rows,):
        raise ValueError("source feature metadata is not row aligned")
    if not np.all(np.isin(labels, [0, 1, 2])):
        raise ValueError("source labels must use the locked three-class order")
    if len(set(windows.tolist())) != rows:
        raise ValueError("source window identifiers are not unique")
    gravity = _unit_rows(
        physics[:, [PHYSICS_INDEX["gravity_x_mean"], PHYSICS_INDEX["gravity_y_mean"], PHYSICS_INDEX["gravity_z_mean"]]]
    )
    # These are provider-derived source summaries.  Units and continuous chronology are
    # intentionally not promoted to the physical protocol; the values are a proxy input.
    x = np.column_stack(
        (
            np.log1p(np.maximum(physics[:, PHYSICS_INDEX["acceleration_rms"]], 0.0)),
            np.log1p(np.maximum(physics[:, PHYSICS_INDEX["gyroscope_rms"]], 0.0)),
        )
    )
    sigma = np.linalg.norm(
        physics[:, [PHYSICS_INDEX["gravity_x_std"], PHYSICS_INDEX["gravity_y_std"], PHYSICS_INDEX["gravity_z_std"]]],
        axis=1,
    )
    sigma = np.clip(sigma, 1.0e-4, math.pi - 1.0e-4)
    if not np.isfinite(x).all() or not np.isfinite(sigma).all():
        raise ValueError("derived proxy inputs are nonfinite")
    return {
        "x": np.asarray(x, dtype=np.float64),
        "direction": gravity,
        "sigma": np.asarray(sigma, dtype=np.float64),
        "labels": labels,
        "participants": participants,
        "windows": windows,
    }


def _support_for_participant(data: dict[str, Any], participant: str) -> tuple[np.ndarray, FloatArray, FloatArray]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    participants = np.asarray(data["participants"], dtype=np.str_)
    direction = np.asarray(data["direction"], dtype=np.float64)
    sigma = np.asarray(data["sigma"], dtype=np.float64)
    selected: list[np.ndarray] = []
    for class_index in (1, 2):
        available = np.flatnonzero((participants == participant) & (labels == class_index))
        if available.size < 2:
            raise ValueError(f"participant {participant} lacks two class-{class_index} support rows")
        selected.append(np.asarray(available[:2], dtype=np.int64))
    support_indices = np.concatenate(selected)
    support_directions = np.stack((direction[selected[0]], direction[selected[1]]), axis=0)
    support_spread = np.stack((sigma[selected[0]], sigma[selected[1]]), axis=0)
    return support_indices, support_directions, support_spread


def _reference_arrays(
    query_directions: FloatArray, support_directions: FloatArray, support_spread: FloatArray
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    rows = query_directions.shape[0]
    support = np.broadcast_to(support_directions[None, ...], (rows, 2, 2, 3)).copy()
    spread = np.broadcast_to(support_spread[None, ...], (rows, 2, 2)).copy()
    point_core = point_reference_features(query_directions, support_directions[0], support_directions[1])
    point = np.column_stack((point_core, spread.mean(axis=(1, 2))))
    return support, spread, point, np.asarray(spread.mean(axis=(1, 2)), dtype=np.float64)


def _motion_design(name: str, x: FloatArray, reference: FloatArray) -> FloatArray:
    if name == "factorized":
        return x
    if name in {"point_joint", "directional_joint", "directional_constant"}:
        return np.column_stack((x, reference[:, 1], reference[:, 2], reference[:, 1] * x[:, 0], reference[:, 1] * x[:, 1]))
    if name == "directional_additive":
        return np.column_stack((x, reference[:, 1], reference[:, 2]))
    if name == "polynomial_capacity":
        return np.column_stack((x, x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2, x[:, 0] ** 3 + x[:, 1] ** 3))
    raise ValueError(f"unknown motion design: {name}")


def _equal_participant_class_weights(participants: NDArray[np.str_], labels: NDArray[np.int64]) -> FloatArray:
    unique = sorted(set(participants.tolist()))
    weights = np.zeros(labels.size, dtype=np.float64)
    for participant in unique:
        for class_index in range(3):
            selected = np.flatnonzero((participants == participant) & (labels == class_index))
            if selected.size == 0:
                raise ValueError("training participant/class cell is empty")
            weights[selected] = 1.0 / (len(unique) * 3 * selected.size)
    return weights


def _macro_f1(labels: NDArray[np.int64], predictions: NDArray[np.int64]) -> tuple[float, dict[str, float]]:
    names = ("mobility", "sitting", "standing")
    values: list[float] = []
    recalls: dict[str, float] = {}
    for class_index, name in enumerate(names):
        tp = int(np.sum((labels == class_index) & (predictions == class_index)))
        fp = int(np.sum((labels != class_index) & (predictions == class_index)))
        fn = int(np.sum((labels == class_index) & (predictions != class_index)))
        denominator = 2 * tp + fp + fn
        values.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        positives = tp + fn
        recalls[name] = 0.0 if positives == 0 else tp / positives
    return float(np.mean(values)), recalls


def _bootstrap_interval(values: FloatArray, *, seed: int = 1729, draws: int = 10_000) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    samples = values[generator.integers(0, values.size, size=(draws, values.size))].mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _fit_outer_fold(data: dict[str, Any], held_out: str, *, bench_delta: float = 0.05) -> dict[str, Any]:
    labels = np.asarray(data["labels"], dtype=np.int64)
    participants = np.asarray(data["participants"], dtype=np.str_)
    x = np.asarray(data["x"], dtype=np.float64)
    direction = np.asarray(data["direction"], dtype=np.float64)
    unique = sorted(set(participants.tolist()))
    train_participants = [value for value in unique if value != held_out]
    train_indices: list[int] = []
    train_support: dict[str, tuple[np.ndarray, FloatArray, FloatArray]] = {}
    for participant in train_participants:
        support_indices, support_directions, support_spread = _support_for_participant(data, participant)
        train_support[participant] = (support_indices, support_directions, support_spread)
        train_indices.extend(
            index
            for index in np.flatnonzero(participants == participant).tolist()
            if index not in set(support_indices.tolist())
        )
    train_idx = np.asarray(sorted(train_indices), dtype=np.int64)
    test_support_indices, test_support_directions, test_support_spread = _support_for_participant(data, held_out)
    test_indices = np.asarray(
        [index for index in np.flatnonzero(participants == held_out).tolist() if index not in set(test_support_indices.tolist())],
        dtype=np.int64,
    )
    train_labels = labels[train_idx]
    train_participant_values = participants[train_idx]
    weights = _equal_participant_class_weights(train_participant_values, train_labels)
    train_direction = direction[train_idx]
    train_support_arrays: list[FloatArray] = []
    train_spread_arrays: list[FloatArray] = []
    for participant in train_participant_values.tolist():
        _, support_directions, support_spread = train_support[participant]
        train_support_arrays.append(support_directions)
        train_spread_arrays.append(support_spread)
    support_train = np.asarray(train_support_arrays, dtype=np.float64)
    spread_train = np.asarray(train_spread_arrays, dtype=np.float64)
    support_test, spread_test, point_test, _ = _reference_arrays(direction[test_indices], test_support_directions, test_support_spread)
    # Build each training row's point reference from its participant's own support.
    point_train_rows: list[FloatArray] = []
    for index, participant in zip(train_idx.tolist(), train_participant_values.tolist(), strict=True):
        _, support_directions, _ = train_support[participant]
        point_train_rows.append(point_reference_features(direction[index : index + 1], support_directions[0], support_directions[1])[0])
    point_train = np.column_stack((np.asarray(point_train_rows), spread_train.mean(axis=(1, 2))))
    stationary = np.isin(train_labels, [1, 2])
    stationary_weights = weights[stationary]
    variable = calibrate_variable_concentration(
        train_direction[stationary], support_train[stationary], spread_train[stationary],
        (train_labels[stationary] == 2).astype(np.int64), stationary_weights, bench_delta=bench_delta,
    )
    common = calibrate_common_concentration(
        train_direction[stationary], support_train[stationary],
        (train_labels[stationary] == 2).astype(np.int64), stationary_weights,
    )
    variable_train = kappa_from_spread(spread_train, variable.value, bench_delta)
    variable_test = kappa_from_spread(spread_test, variable.value, bench_delta)
    common_train = np.full(spread_train.shape, common.value, dtype=np.float64)
    common_test = np.full(spread_test.shape, common.value, dtype=np.float64)
    direction_train = directional_features_batch(train_direction, support_train, variable_train, spread_train)
    direction_test = directional_features_batch(direction[test_indices], support_test, variable_test, spread_test)
    constant_train = directional_features_batch(train_direction, support_train, common_train, spread_train)
    constant_test = directional_features_batch(direction[test_indices], support_test, common_test, spread_test)
    x_scaler = fit_weighted_scaler(x[train_idx], weights)
    x_train = apply_scaler(x[train_idx], x_scaler)
    x_test = apply_scaler(x[test_indices], x_scaler)
    motion_target = (train_labels == 0).astype(np.int64)
    motion_objects: dict[str, tuple[Any, Any]] = {}
    motion_features: dict[str, tuple[FloatArray, FloatArray]] = {}
    for name, train_reference, test_reference in (
        ("factorized", direction_train, direction_test),
        ("point_joint", point_train, point_test),
        ("directional_joint", direction_train, direction_test),
        ("directional_additive", direction_train, direction_test),
        ("directional_constant", constant_train, constant_test),
        ("polynomial_capacity", direction_train, direction_test),
    ):
        train_design = _motion_design(name, x_train, train_reference)
        test_design = _motion_design(name, x_test, test_reference)
        scaler = fit_weighted_scaler(train_design, weights)
        fit = fit_binary_logistic(apply_scaler(train_design, scaler), motion_target, weights)
        motion_objects[name] = (scaler, fit)
        motion_features[name] = (train_design, test_design)
    posture_labels = (train_labels[stationary] == 1).astype(np.int64)
    posture_weights = stationary_weights
    posture_objects: dict[str, tuple[float, Any]] = {}
    for name, values in (("point", point_train), ("directional", direction_train), ("directional_constant", constant_train)):
        ratio = values[stationary, 0]
        rms = float(np.sqrt(np.sum(posture_weights * ratio**2) / posture_weights.sum()))
        if rms <= 1.0e-12:
            raise ValueError(f"proxy posture ratio is degenerate for {name}")
        posture_objects[name] = (rms, fit_constrained_posture(ratio / rms, values[stationary, 2] / math.pi, posture_labels, posture_weights))
    z_train = np.column_stack((x[train_idx], direction[train_idx]))
    z_test = np.column_stack((x[test_indices], direction[test_indices]))
    z_scaler = fit_weighted_scaler(z_train, weights)
    z_fit = fit_multinomial_zero_sum(apply_scaler(z_train, z_scaler), train_labels, weights)
    probabilities: dict[str, FloatArray] = {"Z": predict_multinomial(apply_scaler(z_test, z_scaler), z_fit)}

    def product(motion_name: str, posture_name: str, reference_train: FloatArray, reference_test: FloatArray) -> FloatArray:
        motion_scaler, motion_fit = motion_objects[motion_name]
        motion_design_test = motion_features[motion_name][1]
        mobility = predict_binary(apply_scaler(motion_design_test, motion_scaler), motion_fit)
        rms, posture_fit = posture_objects[posture_name]
        ratio = reference_test[:, 0] / rms
        h = reference_test[:, 2] / math.pi
        sitting = predict_binary(np.column_stack((ratio * (1.0 - h), ratio * h)), posture_fit, include_intercept=False)
        del reference_train
        return compose_probabilities(mobility, sitting)

    probabilities.update(
        {
            "A": product("factorized", "point", point_train, point_test),
            "B": product("factorized", "directional", direction_train, direction_test),
            "C": product("point_joint", "point", point_train, point_test),
            "D": product("directional_joint", "directional", direction_train, direction_test),
            "D-additive": product("directional_additive", "directional", direction_train, direction_test),
            "D-constant": product("directional_constant", "directional_constant", constant_train, constant_test),
            "K": product("polynomial_capacity", "directional", direction_train, direction_test),
        }
    )
    phi_train = np.column_stack((x[train_idx][stationary], train_direction[stationary]))
    prior = fit_map_prior(phi_train, (train_labels[stationary] == 2).astype(np.int64), train_participant_values[stationary], stationary_weights)
    map_support_sit = np.column_stack((x[test_support_indices[:2]], direction[test_support_indices[:2]]))
    map_support_stand = np.column_stack((x[test_support_indices[2:]], direction[test_support_indices[2:]]))
    posterior = update_map_posterior(prior, map_support_sit, map_support_stand)
    map_probability = map_sitting_probability(np.column_stack((x[test_indices], direction[test_indices])), prior, posterior)
    motion_scaler, motion_fit = motion_objects["factorized"]
    map_mobility = predict_binary(apply_scaler(motion_features["factorized"][1], motion_scaler), motion_fit)
    probabilities["MAP"] = compose_probabilities(map_mobility, map_probability)
    return {
        "held_out_participant": held_out,
        "support_window_ids": [str(value) for value in np.asarray(data["windows"])[test_support_indices].tolist()],
        "query_indices": test_indices,
        "query_window_ids": np.asarray(data["windows"])[test_indices],
        "labels": labels[test_indices],
        "probabilities": probabilities,
        "support_count": int(test_support_indices.size),
        "query_count": int(test_indices.size),
        "variable_concentration": float(variable.value),
        "common_concentration": float(common.value),
    }


def run_existing_data_proxy(source_features_path: Path, output_directory: Path) -> dict[str, Any]:
    """Run ten participant-held-out proxy folds and write create-only evidence."""

    if output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {output_directory}")
    started = time.perf_counter()
    data = load_source_feature_proxy(source_features_path)
    participants = sorted(set(np.asarray(data["participants"]).tolist()))
    fold_results = [_fit_outer_fold(data, participant) for participant in participants]
    support_ids_by_participant = {
        fold["held_out_participant"]: [str(value) for value in fold["support_window_ids"]]
        for fold in fold_results
    }
    query_ids_by_participant = {
        fold["held_out_participant"]: [str(value) for value in fold["query_window_ids"]]
        for fold in fold_results
    }
    rows: dict[str, list[dict[str, Any]]] = {arm: [] for arm in ARM_IDS}
    prediction_payload: dict[str, Any] = {}
    all_labels: list[NDArray[np.int64]] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    for fold in fold_results:
        labels = np.asarray(fold["labels"], dtype=np.int64)
        pid = np.full(labels.shape, fold["held_out_participant"], dtype=np.str_)
        all_labels.append(labels)
        all_participants.append(pid)
        all_windows.append(np.asarray(fold["query_window_ids"], dtype=np.str_))
        for arm in ARM_IDS:
            probabilities = np.asarray(fold["probabilities"][arm], dtype=np.float64)
            predictions = np.argmax(probabilities, axis=1).astype(np.int64)
            score, recalls = _macro_f1(labels, predictions)
            rows[arm].append({"participant_id": fold["held_out_participant"], "macro_f1": score, "class_recalls": recalls, "query_count": int(labels.size)})
            prediction_payload[f"probabilities__{arm}"] = prediction_payload.get(f"probabilities__{arm}", [])
            prediction_payload[f"probabilities__{arm}"].append(probabilities)
    baseline = {row["participant_id"]: float(row["macro_f1"]) for row in rows["Z"]}
    summaries: dict[str, Any] = {}
    for arm in ARM_IDS:
        arm_scores = {row["participant_id"]: float(row["macro_f1"]) for row in rows[arm]}
        differences = np.asarray(
            [arm_scores[participant] - baseline[participant] for participant in participants],
            dtype=np.float64,
        )
        summaries[arm] = {
            "mean_participant_macro_f1": float(np.mean(list(arm_scores.values()))),
            "participant_wins_vs_Z": int(np.sum(differences > 0.0)),
            "participant_ties_vs_Z": int(np.sum(differences == 0.0)),
            "participant_harms_vs_Z": int(np.sum(differences < 0.0)),
            "mean_delta_vs_Z": float(differences.mean()),
            "paired_bootstrap_95_percent_interval_vs_Z": list(_bootstrap_interval(differences)),
            "minimum_delta_vs_Z": float(differences.min()),
            "participant_deltas_vs_Z": differences.tolist(),
            "participant_rows": rows[arm],
        }
    output_directory.mkdir(parents=True)
    for arm in ARM_IDS:
        prediction_payload[f"probabilities__{arm}"] = np.concatenate(prediction_payload[f"probabilities__{arm}"], axis=0)
    prediction_payload["labels"] = np.concatenate(all_labels)
    prediction_payload["participant_ids"] = np.concatenate(all_participants)
    prediction_payload["window_ids"] = np.concatenate(all_windows)
    prediction_path = output_directory / "predictions.npz"
    np.savez_compressed(prediction_path, **prediction_payload)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    implementation_path = Path(__file__).resolve()
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "same_attachment_reference_retrospective_proxy_result",
        "status": "complete_retrospective_proxy",
        "evidence_status": "retrospective_proxy_existing_source_features",
        "human_performance_claim": False,
        "physical_admission": "blocked_missing_same_attachment_provenance",
        "source_features_path": str(source_features_path).replace("\\", "/"),
        "source_features_sha256": sha256_file(source_features_path),
        "source_window_count": int(np.asarray(data["labels"]).size),
        "participant_count": len(participants),
        "participants": participants,
        "support_selection": "first_two_rows_per_sitting_and_standing_class_in_preserved_release_order_per_participant",
        "support_rows_per_participant": 4,
        "support_window_ids_by_participant": support_ids_by_participant,
        "query_window_ids_by_participant": query_ids_by_participant,
        "query_window_count": int(sum(fold["query_count"] for fold in fold_results)),
        "support_and_query_disjoint": True,
        "missing_physical_fields": ["attachment_id", "continuous_timestamp", "bout_id", "attachment_continuity", "raw_gravity_stream", "measured_bench_uncertainty", "quiet_or_upper_body_motion_stratum"],
        "proxy_units_and_spread": "provider-derived feature summaries; log1p RMS inputs and gravity-component-std norm used as explicit proxy; bench_delta=0.05 proxy value",
        "outer_evaluation": "ten participant-held-out folds; held-out support rows excluded from query metrics",
        "arms": list(ARM_IDS),
        "summaries": summaries,
        "fold_accounting": {"folds": len(fold_results), "arm_fold_outputs": len(ARM_IDS) * len(fold_results), "human_model_fits": 0},
        "prediction_path": str(prediction_path).replace("\\", "/"),
        "prediction_sha256": sha256_file(prediction_path),
        "runtime_seconds": time.perf_counter() - started,
        "code_commit": head,
        "implementation_path": str(implementation_path).replace("\\", "/"),
        "implementation_sha256": sha256_file(implementation_path),
        "target_accessed": False,
        "query_labels_used_only_for_scoring": True,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    result_path = output_directory / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": "1.0.0",
        "record_kind": "same_attachment_reference_retrospective_proxy_manifest",
        "artifacts": [
            {"path": "result.json", "sha256": sha256_file(result_path), "size_bytes": result_path.stat().st_size},
            {"path": "predictions.npz", "sha256": sha256_file(prediction_path), "size_bytes": prediction_path.stat().st_size},
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    (output_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    validation: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "same_attachment_reference_retrospective_proxy_validation",
        "status": "passed",
        "checks": {
            "result_hash_valid": canonical_json_sha256({key: value for key, value in result.items() if key != "record_sha256"}) == result["record_sha256"],
            "manifest_hash_valid": canonical_json_sha256({key: value for key, value in manifest.items() if key != "record_sha256"}) == manifest["record_sha256"],
            "all_arms_present": all(arm in summaries for arm in ARM_IDS),
            "all_fold_outputs_present": all(len(rows[arm]) == len(participants) for arm in ARM_IDS),
            "support_counts_correct": all(fold["support_count"] == 4 for fold in fold_results),
            "support_query_ids_disjoint": all(
                set(fold["support_window_ids"]).isdisjoint(set(fold["query_window_ids"]))
                for fold in fold_results
            ),
            "prediction_probability_simplex": all(
                np.isfinite(np.asarray(prediction_payload[f"probabilities__{arm}"], dtype=np.float64)).all()
                and np.allclose(
                    np.asarray(prediction_payload[f"probabilities__{arm}"], dtype=np.float64).sum(axis=1),
                    1.0,
                    rtol=0.0,
                    atol=1.0e-12,
                )
                for arm in ARM_IDS
            ),
            "target_not_accessed": result["target_accessed"] is False,
            "support_query_disjoint": result["support_and_query_disjoint"] is True,
            "human_claim_false": result["human_performance_claim"] is False,
        },
    }
    validation["status"] = "passed" if all(validation["checks"].values()) else "failed"
    (output_directory / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_existing_data_proxy(arguments.source_features.resolve(), arguments.output.resolve())
    print(json.dumps({"status": result["status"], "runtime_seconds": result["runtime_seconds"], "participants": result["participant_count"], "arms": result["arms"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
