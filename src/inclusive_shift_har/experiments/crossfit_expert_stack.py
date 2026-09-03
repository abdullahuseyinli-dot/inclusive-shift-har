"""Cross-fitted stacking of frozen SpectralShape and neural committee outputs."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression  # type: ignore[import-untyped]
from sklearn.pipeline import make_pipeline  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import (
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EXPERT_NAMES = ("spectral_shape", "compact_residual_dann", "inception_erm")
METHOD_NAMES = (
    "spectral_shape",
    "neural_committee_mean",
    "equal_three_expert_mean",
    "multinomial_stack",
    "binary_expert_hard",
    "binary_expert_soft",
    "correctness_expert_hard",
)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_record(path: Path) -> dict[str, Any]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    claimed = payload.get("record_sha256")
    unhashed = dict(payload)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError(f"source result self-hash changed: {path}")
    if (
        payload.get("target_subject_or_window_records_loaded") is not False
        or payload.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("cross-fitted stack refuses target-informed inputs")
    return payload


def _uncertainty(probabilities: FloatArray) -> FloatArray:
    ordered = np.sort(probabilities, axis=1)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    return np.stack(
        (
            probabilities.max(axis=1),
            ordered[:, -1] - ordered[:, -2],
            -np.sum(clipped * np.log(clipped), axis=1) / math.log(probabilities.shape[1]),
        ),
        axis=1,
    )


def stack_features(experts: tuple[FloatArray, FloatArray, FloatArray]) -> FloatArray:
    """Build fixed meta-features without identity or cohort variables."""

    if any(item.ndim != 2 or item.shape[1] != 3 for item in experts) or any(
        item.shape != experts[0].shape for item in experts
    ):
        raise ValueError("stack experts must be aligned [window,3] matrices")
    if any(
        not np.isfinite(item).all()
        or np.any(item < 0)
        or not np.allclose(item.sum(axis=1), 1.0, atol=1e-6)
        for item in experts
    ):
        raise ValueError("stack expert probabilities are invalid")
    components: list[FloatArray] = [*experts, *(_uncertainty(item) for item in experts)]
    for first in range(len(experts)):
        for second in range(first + 1, len(experts)):
            components.append(np.abs(experts[first] - experts[second]))
            components.append(
                (experts[first].argmax(axis=1) != experts[second].argmax(axis=1)).astype(
                    np.float64
                )[:, None]
            )
    result = np.concatenate(components, axis=1)
    if result.shape[1] != 30 or not np.isfinite(result).all():
        raise AssertionError("cross-fitted stack feature contract changed")
    return np.asarray(result, dtype=np.float64)


def _fit_binary_router(
    features: FloatArray,
    labels: IntArray,
    first: FloatArray,
    second: FloatArray,
    *,
    seed: int,
    regularization_c: float,
) -> tuple[Any | None, int | None, dict[str, Any]]:
    first_correct = first.argmax(axis=1) == labels
    second_correct = second.argmax(axis=1) == labels
    decisive = (first.argmax(axis=1) != second.argmax(axis=1)) & (first_correct ^ second_correct)
    targets = first_correct[decisive].astype(np.int64)
    model: Any | None = None
    constant: int | None = None
    if targets.size == 0:
        constant = 1
        status = "constant_spectral_no_decisive_rows"
    elif np.unique(targets).size == 1:
        constant = int(targets[0])
        status = "constant_single_expert_preference"
    else:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed,
            ),
        )
        model.fit(features[decisive], targets)
        status = "fitted_binary_reliability_router"
    return (
        model,
        constant,
        {
            "status": status,
            "all_meta_train_rows": int(labels.size),
            "decisive_rows": int(decisive.sum()),
            "spectral_wins": int(targets.sum()),
            "neural_mean_wins": int(targets.size - targets.sum()),
            "participant_identity_feature_used": False,
        },
    )


def _positive_probability(
    model: Any | None, constant: int | None, features: FloatArray
) -> FloatArray:
    if model is None:
        if constant not in {0, 1}:
            raise ValueError("constant meta-model choice is invalid")
        return np.full(features.shape[0], float(constant), dtype=np.float64)
    probabilities = np.asarray(model.predict_proba(features), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    positive = np.flatnonzero(classes == 1)
    if positive.size != 1:
        raise ValueError("binary meta-model lacks its positive class")
    return np.asarray(probabilities[:, int(positive[0])], dtype=np.float64)


def _fit_correctness_models(
    features: FloatArray,
    labels: IntArray,
    experts: tuple[FloatArray, FloatArray, FloatArray],
    *,
    seed: int,
    regularization_c: float,
) -> list[tuple[Any | None, int | None]]:
    result: list[tuple[Any | None, int | None]] = []
    for index, probabilities in enumerate(experts):
        correct = (probabilities.argmax(axis=1) == labels).astype(np.int64)
        if np.unique(correct).size == 1:
            result.append((None, int(correct[0])))
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=2_000,
                random_state=seed + index,
            ),
        )
        model.fit(features, correct)
        result.append((model, None))
    return result


def _load_fold_predictions(
    spectral_root: Path,
    committee_root: Path,
    fold_id: str,
) -> dict[str, Any]:
    spectral_record = _load_record(spectral_root / fold_id / "result.json")
    committee_record = _load_record(committee_root / fold_id / "result.json")
    spectral_reference = _mapping(spectral_record["predictions"], name="spectral predictions")
    committee_reference = _mapping(committee_record["predictions"], name="committee predictions")
    spectral_path = Path(str(spectral_reference["path"]))
    committee_path = Path(str(committee_reference["path"]))
    if (
        sha256_file(spectral_path) != spectral_reference["sha256"]
        or sha256_file(committee_path) != committee_reference["sha256"]
    ):
        raise ValueError("cross-stack input prediction hash changed")
    with np.load(spectral_path, allow_pickle=False) as arrays:
        labels = np.asarray(arrays["labels"], dtype=np.int64)
        participants = np.asarray(arrays["participant_ids"], dtype=np.str_)
        windows = np.asarray(arrays["window_ids"], dtype=np.str_)
        spectral = np.asarray(arrays["probabilities"], dtype=np.float64)
    with np.load(committee_path, allow_pickle=False) as arrays:
        if (
            not np.array_equal(labels, arrays["labels"])
            or not np.array_equal(participants, arrays["participant_ids"])
            or not np.array_equal(windows, arrays["window_ids"])
        ):
            raise ValueError("spectral and neural committee rows are not aligned")
        dann = np.asarray(arrays["compact_residual_dann_probabilities"], dtype=np.float64)
        inception = np.asarray(arrays["inception_erm_probabilities"], dtype=np.float64)
    return {
        "labels": labels,
        "participants": participants,
        "windows": windows,
        "experts": (spectral, dann, inception),
        "input_records": {
            "spectral": {
                "path": (spectral_root / fold_id / "result.json").as_posix(),
                "sha256": sha256_file(spectral_root / fold_id / "result.json"),
            },
            "committee": {
                "path": (committee_root / fold_id / "result.json").as_posix(),
                "sha256": sha256_file(committee_root / fold_id / "result.json"),
            },
        },
    }


def run_crossfit_expert_stack(
    *,
    spectral_root: Path,
    committee_root: Path,
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Fit each meta-model on four source folds and evaluate the excluded fifth."""

    if output_directory.exists():
        raise FileExistsError(f"cross-stack output directory already exists: {output_directory}")
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), name="config")
    fold_ids = [str(item) for item in cast(list[object], config["meta_folds"])]
    if fold_ids != [f"source_cv_{index:02d}" for index in range(1, 6)]:
        raise ValueError("cross-stack requires the five locked source folds in order")
    regularization_c = float(config["regularization_c"])
    if regularization_c <= 0:
        raise ValueError("cross-stack regularization must be positive")
    fold_data = {
        fold_id: _load_fold_predictions(spectral_root, committee_root, fold_id)
        for fold_id in fold_ids
    }
    output_directory.mkdir(parents=True)
    method_outputs: dict[str, list[FloatArray]] = {name: [] for name in METHOD_NAMES}
    all_labels: list[IntArray] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    fold_records: list[dict[str, Any]] = []
    for fold_index, fold_id in enumerate(fold_ids):
        train_records = [fold_data[item] for item in fold_ids if item != fold_id]
        evaluation = fold_data[fold_id]
        train_labels = np.concatenate([item["labels"] for item in train_records])
        train_experts = cast(
            tuple[FloatArray, FloatArray, FloatArray],
            tuple(
                np.concatenate([item["experts"][index] for item in train_records])
                for index in range(3)
            ),
        )
        evaluation_experts = cast(tuple[FloatArray, FloatArray, FloatArray], evaluation["experts"])
        train_features = stack_features(train_experts)
        evaluation_features = stack_features(evaluation_experts)
        direct = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=regularization_c,
                class_weight="balanced",
                max_iter=2_000,
                random_state=11 + fold_index,
            ),
        )
        direct.fit(train_features, train_labels)
        direct_probability = np.asarray(direct.predict_proba(evaluation_features), dtype=np.float64)
        direct_classes = np.asarray(direct.classes_, dtype=np.int64)
        if not np.array_equal(direct_classes, np.arange(3)):
            raise ValueError("multinomial stack class order changed")

        train_neural_mean = 0.5 * (train_experts[1] + train_experts[2])
        evaluation_neural_mean = 0.5 * (evaluation_experts[1] + evaluation_experts[2])
        binary, constant, binary_record = _fit_binary_router(
            train_features,
            train_labels,
            train_experts[0],
            train_neural_mean,
            seed=101 + fold_index,
            regularization_c=regularization_c,
        )
        spectral_weight = _positive_probability(binary, constant, evaluation_features)
        binary_hard = np.where(
            (spectral_weight >= 0.5)[:, None],
            evaluation_experts[0],
            evaluation_neural_mean,
        )
        binary_soft = (
            spectral_weight[:, None] * evaluation_experts[0]
            + (1.0 - spectral_weight[:, None]) * evaluation_neural_mean
        )

        correctness_models = _fit_correctness_models(
            train_features,
            train_labels,
            train_experts,
            seed=211 + fold_index * 10,
            regularization_c=regularization_c,
        )
        predicted_correctness = np.stack(
            [
                _positive_probability(model, constant_value, evaluation_features)
                for model, constant_value in correctness_models
            ],
            axis=1,
        )
        selected_expert = predicted_correctness.argmax(axis=1)
        correctness_hard = np.stack(evaluation_experts, axis=1)[
            np.arange(selected_expert.size), selected_expert
        ]
        methods = {
            "spectral_shape": evaluation_experts[0],
            "neural_committee_mean": evaluation_neural_mean,
            "equal_three_expert_mean": sum(evaluation_experts) / 3.0,
            "multinomial_stack": direct_probability,
            "binary_expert_hard": binary_hard,
            "binary_expert_soft": binary_soft,
            "correctness_expert_hard": correctness_hard,
        }
        labels = cast(IntArray, evaluation["labels"])
        participants = cast(NDArray[np.str_], evaluation["participants"])
        reports = {
            name: classification_report(
                labels,
                probability,
                participants.tolist(),
                class_names=("mobility", "sitting", "standing"),
            )
            for name, probability in methods.items()
        }
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "meta_models.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(
                {
                    "direct": direct,
                    "binary": binary,
                    "binary_constant": constant,
                    "correctness_models": correctness_models,
                },
                stream,
                protocol=5,
            )
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                labels=labels,
                participant_ids=participants,
                window_ids=evaluation["windows"],
                **{f"{name}_probabilities": value for name, value in methods.items()},
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "crossfit_expert_stack_fold",
            "fold_id": fold_id,
            "meta_train_folds": [item for item in fold_ids if item != fold_id],
            "meta_evaluation_fold": fold_id,
            "meta_evaluation_participants_in_training": False,
            "binary_router_training": binary_record,
            "reports": reports,
            "models": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "input_records": evaluation["input_records"],
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        with fold_path.open("xb") as stream:
            stream.write(json.dumps(fold_record, indent=2, sort_keys=True).encode("utf-8"))
            stream.write(b"\n")
        fold_records.append(
            {
                "fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "reports": reports,
            }
        )
        for name, probability in methods.items():
            method_outputs[name].append(np.asarray(probability, dtype=np.float64))
        all_labels.append(labels)
        all_participants.append(participants)
        all_windows.append(cast(NDArray[np.str_], evaluation["windows"]))

    labels = np.concatenate(all_labels)
    participants = np.concatenate(all_participants)
    windows = np.concatenate(all_windows)
    if len(set(windows.tolist())) != windows.size:
        raise ValueError("cross-stack evaluated a source window more than once")
    aggregate_probabilities = {
        name: np.concatenate(values) for name, values in method_outputs.items()
    }
    aggregate_reports = {
        name: classification_report(
            labels,
            probability,
            participants.tolist(),
            class_names=("mobility", "sitting", "standing"),
        )
        for name, probability in aggregate_probabilities.items()
    }
    participant_values = {
        name: {
            str(row["participant_id"]): float(row["macro_f1"])
            for row in cast(list[dict[str, Any]], report["participants"])
        }
        for name, report in aggregate_reports.items()
    }
    uncertainty = {
        name: participant_bootstrap_interval(values) for name, values in participant_values.items()
    }
    comparisons = {
        name: paired_participant_comparison(
            participant_values["spectral_shape"], participant_values[name]
        )
        for name in METHOD_NAMES
        if name != "spectral_shape"
    }
    prediction_path = output_directory / "all_crossfit_predictions.npz"
    with prediction_path.open("xb") as stream:
        arrays: dict[str, Any] = {
            "labels": labels,
            "participant_ids": participants,
            "window_ids": windows,
            **{f"{name}_probabilities": value for name, value in aggregate_probabilities.items()},
        }
        np.savez_compressed(stream, **arrays)
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "crossfit_expert_stack_source_aggregate",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_cross_fitted_source_development_not_independent",
        "method_origin": config["method_origin"],
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "folds": fold_records,
        "aggregate_reports": aggregate_reports,
        "participant_bootstrap": uncertainty,
        "paired_comparisons_vs_spectral_shape": comparisons,
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "each_meta_evaluation_pair_excluded_from_meta_training": True,
            "participant_identity_feature_used": False,
            "method_family_inspired_by_prior_source_results": True,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    with (output_directory / "result.json").open("xb") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectral-root", type=Path, required=True)
    parser.add_argument("--committee-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_crossfit_expert_stack(
        spectral_root=args.spectral_root,
        committee_root=args.committee_root,
        config_path=args.config,
        output_directory=args.output_directory,
    )
    compact = {
        "status": result["status"],
        "evidence_status": result["evidence_status"],
        "primary": {
            name: report["primary"] for name, report in result["aggregate_reports"].items()
        },
        "record_sha256": result["record_sha256"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
