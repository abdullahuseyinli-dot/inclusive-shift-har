"""Fixed modern time-series controls on target-sealed InclusiveHAR source folds."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.experiments.multirocket_source import _aligned_probabilities
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.preprocessing.normalization import ChannelStandardizer

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


class _Classifier(Protocol):
    classes_: NDArray[np.integer[Any]]

    def fit(self, x: NDArray[np.floating[Any]], y: IntArray) -> _Classifier: ...

    def predict(self, x: NDArray[np.floating[Any]]) -> NDArray[np.integer[Any]]: ...

    def predict_proba(self, x: NDArray[np.floating[Any]]) -> NDArray[np.floating[Any]]: ...


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_control(config_path: Path, algorithm: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), name="config")
    algorithms = _mapping(config.get("algorithms"), name="algorithms")
    if algorithm not in algorithms:
        raise ValueError(f"algorithm {algorithm!r} is not predeclared")
    parameters = _mapping(algorithms[algorithm], name=f"algorithms.{algorithm}")
    if int(config.get("seed", -1)) < 0 or int(config.get("n_jobs", 0)) < 1:
        raise ValueError("invalid fixed-control seed or job count")
    return config, parameters


def _build_classifier(
    algorithm: str,
    parameters: dict[str, Any],
    *,
    seed: int,
    n_jobs: int,
) -> _Classifier:
    try:
        from aeon.classification.convolution_based import (  # type: ignore[import-untyped]
            HydraClassifier,
            MultiRocketHydraClassifier,
        )
        from aeon.classification.hybrid import RISTClassifier  # type: ignore[import-untyped]
        from aeon.classification.interval_based import (  # type: ignore[import-untyped]
            QUANTClassifier,
        )
    except ImportError as exc:  # pragma: no cover - installation gate
        raise RuntimeError("aeon controls require the research dependency group") from exc
    if algorithm == "quant":
        classifier: Any = QUANTClassifier(
            interval_depth=int(parameters["interval_depth"]),
            quantile_divisor=int(parameters["quantile_divisor"]),
            random_state=seed,
        )
    elif algorithm == "hydra":
        classifier = HydraClassifier(
            n_kernels=int(parameters["n_kernels"]),
            n_groups=int(parameters["n_groups"]),
            class_weight=str(parameters["class_weight"]),
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif algorithm == "multirocket_hydra":
        classifier = MultiRocketHydraClassifier(
            n_kernels=int(parameters["n_kernels"]),
            n_groups=int(parameters["n_groups"]),
            class_weight=str(parameters["class_weight"]),
            n_jobs=n_jobs,
            random_state=seed,
        )
    elif algorithm == "rist_budgeted":
        classifier = RISTClassifier(
            n_intervals=int(parameters["n_intervals"]),
            n_shapelets=int(parameters["n_shapelets"]),
            n_jobs=n_jobs,
            random_state=seed,
        )
    else:  # pragma: no cover - guarded by config plus exhaustive tests
        raise ValueError(f"unsupported predeclared aeon control {algorithm!r}")
    return cast(_Classifier, classifier)


def _write_json_create_only(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_aeon_source_control(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    config_path: Path,
    algorithm: str,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Run one fixed algorithm over all five source folds without target access."""

    if output_directory.exists():
        raise FileExistsError(f"aeon-control output already exists: {output_directory}")
    config, parameters = _load_control(config_path, algorithm)
    seed = int(config["seed"])
    n_jobs = int(config["n_jobs"])
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("aeon controls require the functional-core ontology")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(batch.signals, dtype=np.float32)
    labels = np.asarray(batch.labels, dtype=np.int64)
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    window_ids = np.asarray(batch.window_ids, dtype=np.str_)
    output_directory.mkdir(parents=True)
    fold_records: list[dict[str, Any]] = []
    all_labels: list[IntArray] = []
    all_probabilities: list[FloatArray] = []
    all_participants: list[NDArray[np.str_]] = []
    all_windows: list[NDArray[np.str_]] = []
    explicit_configuration = {
        "algorithm": algorithm,
        "parameters": parameters,
        "seed": seed,
        "n_jobs": n_jobs,
        "preprocessing": config["preprocessing"],
        "input_dtype": (
            "float64_aeon_numba_compatibility" if algorithm == "rist_budgeted" else "float32"
        ),
    }
    for fold in cast(list[dict[str, Any]], manifest["source_cv_folds"]):
        fold_id = str(fold["fold_id"])
        train_subjects = [str(item) for item in fold["train_subjects"]]
        evaluation_subjects = [str(item) for item in fold["validation_subjects"]]
        train_mask = np.asarray(np.isin(participants, train_subjects), dtype=np.bool_)
        evaluation_mask = np.asarray(np.isin(participants, evaluation_subjects), dtype=np.bool_)
        if (
            np.any(train_mask & evaluation_mask)
            or not train_mask.any()
            or not evaluation_mask.any()
        ):
            raise PermissionError(f"invalid participant-exclusive masks in {fold_id}")
        standardizer = ChannelStandardizer.fit(
            signals[train_mask],
            participants[train_mask],
            declared_training_participants=set(train_subjects),
            split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
            channel_names=PRIMARY_CHANNELS,
        )
        train_x = np.transpose(standardizer.transform(signals[train_mask]), (0, 2, 1))
        evaluation_x = np.transpose(standardizer.transform(signals[evaluation_mask]), (0, 2, 1))
        if algorithm == "rist_budgeted":
            # aeon 1.5.0's dilated-shapelet Numba kernel cannot unify its
            # internal float64 normalization with float32 collection input.
            train_x = np.asarray(train_x, dtype=np.float64)
            evaluation_x = np.asarray(evaluation_x, dtype=np.float64)
        classifier = _build_classifier(
            algorithm,
            parameters,
            seed=seed,
            n_jobs=n_jobs,
        )
        classifier.fit(train_x, labels[train_mask])
        probabilities, calibration_status = _aligned_probabilities(
            classifier,
            evaluation_x,
            num_classes=len(class_names),
        )
        report = classification_report(
            labels[evaluation_mask],
            probabilities,
            participants[evaluation_mask].tolist(),
            class_names=class_names,
        )
        report["calibration"]["status"] = calibration_status
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(classifier, stream, protocol=5)
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                probabilities=probabilities,
                labels=labels[evaluation_mask],
                participant_ids=participants[evaluation_mask],
                window_ids=window_ids[evaluation_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "aeon_fixed_source_control_fold",
            "status": "complete_target_sealed",
            "fold_id": fold_id,
            "train_participants": train_subjects,
            "evaluation_participants": evaluation_subjects,
            "configuration": explicit_configuration,
            "configuration_sha256": canonical_json_sha256(explicit_configuration),
            "preprocessor": standardizer.to_dict(),
            "evaluation_report": report,
            "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "outer_evaluation_access_count_during_training": 0,
            "outer_evaluation_access_count_after_training": 1,
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json_create_only(fold_path, fold_record)
        fold_records.append(
            {
                "fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "evaluation_report": report,
            }
        )
        all_labels.append(labels[evaluation_mask])
        all_probabilities.append(probabilities)
        all_participants.append(participants[evaluation_mask])
        all_windows.append(window_ids[evaluation_mask])

    aggregate_labels = np.concatenate(all_labels)
    aggregate_probabilities = np.concatenate(all_probabilities)
    aggregate_participants = np.concatenate(all_participants)
    aggregate_windows = np.concatenate(all_windows)
    if (
        set(aggregate_windows.tolist()) != set(window_ids.tolist())
        or len(set(aggregate_windows.tolist())) != aggregate_windows.size
    ):
        raise ValueError("aeon folds did not cover each source window exactly once")
    aggregate_report = classification_report(
        aggregate_labels,
        aggregate_probabilities,
        aggregate_participants.tolist(),
        class_names=class_names,
    )
    participant_values = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], aggregate_report["participants"])
    }
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=aggregate_probabilities,
            labels=aggregate_labels,
            participant_ids=aggregate_participants,
            window_ids=aggregate_windows,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "aeon_fixed_grouped_source_control",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_fixed_source_control_not_confirmatory",
        "configuration": explicit_configuration,
        "configuration_sha256": canonical_json_sha256(explicit_configuration),
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "code_commit": code_commit,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("aeon", "numba", "numpy", "scikit-learn")
        },
        "folds": fold_records,
        "aggregate_report": aggregate_report,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "method": "none_fixed_control",
            "outer_results_used_for_hyperparameter_selection": False,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json_create_only(output_directory / "result.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--algorithm", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_aeon_source_control(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        config_path=args.config,
        algorithm=args.algorithm,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "evidence_status": result["evidence_status"],
                "algorithm": result["configuration"]["algorithm"],
                "aggregate_primary": result["aggregate_report"]["primary"],
                "class_recall": result["aggregate_report"]["window_level_diagnostics"][
                    "per_class_recall"
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
