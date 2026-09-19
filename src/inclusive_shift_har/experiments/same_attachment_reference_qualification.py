"""Create-only synthetic dry run for the nine-arm reference implementation.

This qualification deliberately performs optimizer fits on constructed arrays. They are
counted as synthetic fits and cannot satisfy physical admission or support a performance claim.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.experiments.same_attachment_reference_comparison import (
    ARM_IDS,
    FIT_SCHEDULE,
    schedule_receipt,
    validate_supervised_admission,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.same_attachment_reference_training import (
    ConcentrationCalibration,
    MapPosterior,
    MapPrior,
    OptimizerFit,
    WeightedScaler,
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


def _array_sha256(array: NDArray[Any]) -> str:
    value = np.ascontiguousarray(array)
    header = json.dumps({"dtype": value.dtype.str, "shape": list(value.shape)}, sort_keys=True)
    return hashlib.sha256(header.encode("utf-8") + b"\0" + value.tobytes()).hexdigest()


def _fit_record(
    fit: OptimizerFit, *, fold: int, object_kind: str, object_id: str
) -> dict[str, Any]:
    return {
        "fold_index": fold,
        "object_kind": object_kind,
        "object_id": object_id,
        "converged": fit.converged,
        "objective": fit.objective,
        "iterations": fit.iterations,
        "evaluations": fit.evaluations,
        "coefficient_shape": list(fit.coefficients.shape),
        "coefficient_sha256": _array_sha256(fit.coefficients),
        "message": fit.message,
    }


def _calibration_record(value: ConcentrationCalibration, *, fold: int) -> dict[str, Any]:
    return {
        "fold_index": fold,
        "object_kind": "scalar_concentration",
        "object_id": value.kind,
        "converged": value.converged,
        "value": value.value,
        "objective": value.objective,
        "evaluations": [list(row) for row in value.evaluations],
        "cap_fraction": value.cap_fraction,
        "message": value.message,
    }


def _support(rows: int) -> tuple[FloatArray, FloatArray]:
    sitting = np.array([[1.0, 0.0, 0.0], [0.98, 0.2, 0.0]], dtype=np.float64)
    standing = np.array([[0.0, 1.0, 0.0], [0.2, 0.98, 0.0]], dtype=np.float64)
    sitting /= np.linalg.norm(sitting, axis=1)[:, None]
    standing /= np.linalg.norm(standing, axis=1)[:, None]
    support = np.tile(np.stack((sitting, standing))[None, ...], (rows, 1, 1, 1))
    spread = np.tile(np.array([[[0.10, 0.16], [0.12, 0.18]]]), (rows, 1, 1))
    return support, spread


def _synthetic_packet(fold: int, rows: int, *, training: bool) -> dict[str, Any]:
    # Deterministic construction; the integer changes geometry between folds, not a seed sweep.
    index = np.arange(rows, dtype=np.float64)
    labels = ((index.astype(np.int64) + fold) % 3).astype(np.int64)
    phase = 0.17 * index + 0.11 * fold + (0.03 if training else 0.07)
    x = np.column_stack(
        (
            0.3 + 1.6 * (labels == 0) + 0.12 * np.sin(phase),
            0.2 + 1.3 * (labels == 0) + 0.10 * np.cos(phase),
        )
    )
    sitting = labels == 1
    standing = labels == 2
    direction = np.column_stack(
        (
            0.9 * sitting + 0.35 * (labels == 0) + 0.05 * np.sin(phase),
            0.9 * standing + 0.35 * (labels == 0) + 0.05 * np.cos(phase),
            0.25 + 0.04 * np.sin(2 * phase),
        )
    )
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    support, spread = _support(rows)
    return {
        "x": np.asarray(x, dtype=np.float64),
        "direction": np.asarray(direction),
        "support": support,
        "spread": spread,
        "labels": labels,
    }


def _motion_design(name: str, x: FloatArray, reference: FloatArray) -> FloatArray:
    _, e, h_raw = reference.T
    h = h_raw / math.pi
    if name == "factorized":
        return x
    if name in {"point_joint", "directional_joint", "directional_constant"}:
        return np.column_stack((x, e, h, e * x[:, 0], e * x[:, 1]))
    if name == "directional_additive":
        return np.column_stack((x, e, h))
    if name == "polynomial_capacity":
        return np.column_stack(
            (x, x[:, 0] ** 2, x[:, 0] * x[:, 1], x[:, 1] ** 2, x[:, 0] ** 3 + x[:, 1] ** 3)
        )
    raise ValueError(f"unknown motion design: {name}")


def _point_features(packet: Mapping[str, Any]) -> FloatArray:
    query = np.asarray(packet["direction"], dtype=np.float64)
    support = np.asarray(packet["support"], dtype=np.float64)
    rows = [
        point_reference_features(query[index : index + 1], support[index, 0], support[index, 1])[0]
        for index in range(query.shape[0])
    ]
    dispersion = np.asarray(packet["spread"], dtype=np.float64).mean(axis=(1, 2))
    return np.column_stack((np.asarray(rows), dispersion))


def _directional_features(packet: Mapping[str, Any], kappa: FloatArray) -> FloatArray:
    return directional_features_batch(
        packet["direction"], packet["support"], kappa, packet["spread"]
    )


def _fit_scaled_binary(
    features: FloatArray, labels: FloatArray, weights: FloatArray
) -> tuple[WeightedScaler, OptimizerFit]:
    scaler = fit_weighted_scaler(features, weights)
    return scaler, fit_binary_logistic(apply_scaler(features, scaler), labels, weights)


def _predict_scaled_binary(
    features: FloatArray, scaler: WeightedScaler, fit: OptimizerFit
) -> FloatArray:
    return predict_binary(apply_scaler(features, scaler), fit)


def _map_objects(
    train: Mapping[str, Any], test: Mapping[str, Any]
) -> tuple[MapPrior, MapPosterior, FloatArray]:
    x = np.asarray(train["x"], dtype=np.float64)
    direction = np.asarray(train["direction"], dtype=np.float64)
    labels = np.asarray(train["labels"], dtype=np.int64)
    stationary = labels != 0
    phi = np.column_stack((x[stationary], direction[stationary]))
    posture = (labels[stationary] == 2).astype(np.int64)
    wearers = np.asarray([f"train:{index % 5}" for index in range(phi.shape[0])])
    prior = fit_map_prior(phi, posture, wearers, np.full(phi.shape[0], 1 / phi.shape[0]))
    support = np.asarray(test["support"], dtype=np.float64)[0]
    support_x = np.asarray(test["x"], dtype=np.float64)
    sit = np.column_stack((support_x[:2], support[0]))
    stand = np.column_stack((support_x[2:4], support[1]))
    posterior = update_map_posterior(prior, sit, stand)
    test_phi = np.column_stack((test["x"], test["direction"]))
    return prior, posterior, map_sitting_probability(test_phi, prior, posterior)


def _fit_fold(fold: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    train = _synthetic_packet(fold, 180, training=True)
    test = _synthetic_packet(fold, 54, training=False)
    train_labels = np.asarray(train["labels"], dtype=np.int64)
    test_labels = np.asarray(test["labels"], dtype=np.int64)
    weights = np.full(train_labels.size, 1 / train_labels.size, dtype=np.float64)
    stationary = train_labels != 0
    posture_labels = (train_labels[stationary] == 1).astype(np.int64)
    posture_weights = weights[stationary]
    support = np.asarray(train["support"])[stationary]
    spread = np.asarray(train["spread"])[stationary]
    direction = np.asarray(train["direction"])[stationary]
    concentration_labels = (train_labels[stationary] == 2).astype(np.int64)
    variable = calibrate_variable_concentration(
        direction, support, spread, concentration_labels, posture_weights, bench_delta=0.05
    )
    common = calibrate_common_concentration(
        direction, support, concentration_labels, posture_weights
    )
    variable_train = kappa_from_spread(np.asarray(train["spread"]), variable.value, 0.05)
    variable_test = kappa_from_spread(np.asarray(test["spread"]), variable.value, 0.05)
    common_train = np.full((train_labels.size, 2, 2), common.value)
    common_test = np.full((test_labels.size, 2, 2), common.value)
    point_train, point_test = _point_features(train), _point_features(test)
    direction_train = _directional_features(train, variable_train)
    direction_test = _directional_features(test, variable_test)
    constant_train = _directional_features(train, common_train)
    constant_test = _directional_features(test, common_test)
    x_scaler = fit_weighted_scaler(train["x"], weights)
    train_x = apply_scaler(train["x"], x_scaler)
    test_x = apply_scaler(test["x"], x_scaler)
    motion_target = (train_labels == 0).astype(np.int64)

    motion_objects: dict[str, tuple[WeightedScaler, OptimizerFit]] = {}
    motion_features: dict[str, tuple[FloatArray, FloatArray]] = {}
    for name, train_reference, test_reference in (
        ("factorized", direction_train, direction_test),
        ("point_joint", point_train, point_test),
        ("directional_joint", direction_train, direction_test),
        ("directional_additive", direction_train, direction_test),
        ("directional_constant", constant_train, constant_test),
        ("polynomial_capacity", direction_train, direction_test),
    ):
        train_design = _motion_design(name, train_x, train_reference)
        test_design = _motion_design(name, test_x, test_reference)
        motion_features[name] = (train_design, test_design)
        motion_objects[name] = _fit_scaled_binary(train_design, motion_target, weights)

    posture_objects: dict[str, tuple[float, OptimizerFit]] = {}
    for name, values in (
        ("point", point_train),
        ("directional", direction_train),
        ("directional_constant", constant_train),
    ):
        ratio = values[stationary, 0]
        rms = float(np.sqrt(np.sum(posture_weights * ratio**2) / posture_weights.sum()))
        fit = fit_constrained_posture(
            ratio / rms, values[stationary, 2] / math.pi, posture_labels, posture_weights
        )
        posture_objects[name] = (rms, fit)

    z_train = np.column_stack((train["x"], train["direction"]))
    z_test = np.column_stack((test["x"], test["direction"]))
    z_scaler = fit_weighted_scaler(z_train, weights)
    z_fit = fit_multinomial_zero_sum(apply_scaler(z_train, z_scaler), train_labels, weights)
    probabilities: dict[str, FloatArray] = {
        "Z": predict_multinomial(apply_scaler(z_test, z_scaler), z_fit)
    }

    def product(motion_name: str, posture_name: str, reference: FloatArray) -> FloatArray:
        motion_scaler, motion_fit = motion_objects[motion_name]
        m = _predict_scaled_binary(motion_features[motion_name][1], motion_scaler, motion_fit)
        rms, posture_fit = posture_objects[posture_name]
        r = reference[:, 0] / rms
        h = reference[:, 2] / math.pi
        posture_design = np.column_stack((r * (1 - h), r * h))
        q = predict_binary(posture_design, posture_fit, include_intercept=False)
        return compose_probabilities(m, q)

    probabilities.update(
        {
            "A": product("factorized", "point", point_test),
            "B": product("factorized", "directional", direction_test),
            "C": product("point_joint", "point", point_test),
            "D": product("directional_joint", "directional", direction_test),
            "D-additive": product("directional_additive", "directional", direction_test),
            "D-constant": product("directional_constant", "directional_constant", constant_test),
            "K": product("polynomial_capacity", "directional", direction_test),
        }
    )
    prior, posterior, map_q = _map_objects(train, test)
    factor_scaler, factor_fit = motion_objects["factorized"]
    map_m = _predict_scaled_binary(motion_features["factorized"][1], factor_scaler, factor_fit)
    probabilities["MAP"] = compose_probabilities(map_m, map_q)

    records = [
        _fit_record(fit, fold=fold, object_kind="binary_motion", object_id=name)
        for name, (_, fit) in motion_objects.items()
    ]
    records += [
        _fit_record(fit, fold=fold, object_kind="binary_posture", object_id=name)
        for name, (_, fit) in posture_objects.items()
    ]
    records.append(_fit_record(z_fit, fold=fold, object_kind="multinomial", object_id="Z"))
    records += [_calibration_record(variable, fold=fold), _calibration_record(common, fold=fold)]
    outputs = []
    for arm in ARM_IDS:
        value = probabilities[arm]
        if value.shape != (54, 3) or not np.allclose(value.sum(axis=1), 1.0):
            raise RuntimeError(f"synthetic arm output invalid: {arm}")
        outputs.append(
            {
                "fold_index": fold,
                "arm_id": arm,
                "row_count": 54,
                "probability_sha256": _array_sha256(value),
            }
        )
    extras = {
        "x_scaler_sha256": _array_sha256(np.concatenate((x_scaler.mean, x_scaler.scale))),
        "map_prior_sha256": _array_sha256(np.column_stack((prior.means, prior.variances))),
        "map_posterior_sha256": _array_sha256(
            np.column_stack((posterior.means, posterior.variances))
        ),
        "held_out_labels_used_by_prediction_api": "labels" in inspect.signature(product).parameters,
        "test_label_sha256_reporting_only": _array_sha256(test_labels),
    }
    return records, outputs, extras


def run_synthetic_qualification(repository_root: Path, output_directory: Path) -> dict[str, Any]:
    """Execute one finite create-only synthetic implementation qualification."""
    if output_directory.exists():
        raise ValueError("create-only qualification directory exists")
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    fold_extras: list[dict[str, Any]] = []
    for fold in range(6):
        fold_records, fold_outputs, extras = _fit_fold(fold)
        records.extend(fold_records)
        outputs.extend(fold_outputs)
        fold_extras.append({"fold_index": fold, **extras})
    expected = [(row.fold_index, row.object_kind, row.object_id) for row in FIT_SCHEDULE]
    observed = [(row["fold_index"], row["object_kind"], row["object_id"]) for row in records]
    if observed != expected:
        raise RuntimeError("synthetic fit ledger differs from finite schedule")
    blocked = validate_supervised_admission(
        {"evidence_status": "synthetic_fixture_not_physical_evidence"}
    )
    if blocked["fit_authorized"] is not False:
        raise RuntimeError("synthetic evidence bypassed supervised admission")
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "same_attachment_reference_synthetic_implementation_qualification",
        "status": "pass",
        "evidence_status": "synthetic_fixture_not_physical_evidence",
        "human_performance_result": False,
        "physical_admission": blocked,
        "fit_ledger": records,
        "arm_fold_outputs": outputs,
        "fold_receipts": fold_extras,
        "accounting": {
            **schedule_receipt(),
            "synthetic_optimizer_fits_executed": 72,
            "human_model_fits_executed": 0,
            "human_outcomes_loaded": False,
        },
        "runtime_seconds": time.perf_counter() - started,
        "source_files": [],
    }
    for relative in (
        "src/inclusive_shift_har/models/same_attachment_reference_training.py",
        "src/inclusive_shift_har/experiments/same_attachment_reference_comparison.py",
        "src/inclusive_shift_har/experiments/same_attachment_reference_qualification.py",
        "configs/experiments/same_attachment_reference_comparison_v1.yaml",
        "docs/research/SAME_ATTACHMENT_REFERENCE_COMPARISON_V1_PROTOCOL.md",
    ):
        path = repository_root / relative
        result["source_files"].append(
            {"path": relative, "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        )
    result["record_sha256"] = canonical_json_sha256(result)
    result_path = output_directory / "result.json"
    with result_path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema_version": "1.0.0",
        "record_kind": "synthetic_qualification_manifest",
        "artifacts": [
            {
                "path": "result.json",
                "sha256": sha256_file(result_path),
                "size_bytes": result_path.stat().st_size,
            }
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    with (output_directory / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_synthetic_qualification(args.repository_root.resolve(), args.output.resolve())
    print(
        json.dumps(
            {
                "status": result["status"],
                "runtime_seconds": result["runtime_seconds"],
                "synthetic_optimizer_fits": 72,
                "human_model_fits": 0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
