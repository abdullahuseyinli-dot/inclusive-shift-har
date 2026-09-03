"""Nested source evaluation with development-only DAGHAR support data."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]

from inclusive_shift_har.data.daghar import load_daghar_development_windows
from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _candidate_summary,
    _HierarchicalForest,
    _probabilities,
    _selection_key,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    candidates = config.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("DAGHAR augmentation config requires candidates")
    ids: list[str] = []
    ranks: list[int] = []
    for index, raw in enumerate(candidates):
        candidate = _mapping(raw, name=f"candidates[{index}]")
        ids.append(str(candidate.get("id", "")))
        ranks.append(int(candidate.get("complexity_rank", -1)))
    if (
        len(set(ids)) != len(ids)
        or any(not item for item in ids)
        or sorted(ranks) != list(range(1, len(ranks) + 1))
    ):
        raise ValueError("DAGHAR augmentation candidate ids/ranks are invalid")
    return config


def _balanced_domain_weights(
    labels: IntArray,
    participants: NDArray[np.str_],
    domains: NDArray[np.str_],
) -> FloatArray:
    """Give every domain equal mass, then every participant-class cell equal mass."""

    if labels.shape != participants.shape or labels.shape != domains.shape or labels.size == 0:
        raise ValueError("domain weights require aligned non-empty vectors")
    result = np.empty(labels.size, dtype=np.float64)
    unique_domains = sorted(set(domains.tolist()))
    for domain in unique_domains:
        domain_mask = domains == domain
        cells = sorted(
            set(zip(participants[domain_mask].tolist(), labels[domain_mask].tolist(), strict=True))
        )
        for participant, label in cells:
            mask = domain_mask & (participants == participant) & (labels == label)
            result[mask] = 1.0 / (len(unique_domains) * len(cells) * int(mask.sum()))
    result *= labels.size / result.sum()
    return result


def _deterministic_cell_cap(
    labels: IntArray,
    participants: NDArray[np.str_],
    domains: NDArray[np.str_],
    window_ids: NDArray[np.str_],
    *,
    maximum: int,
) -> NDArray[np.bool_]:
    if maximum < 1:
        raise ValueError("external cell cap must be positive")
    selected = np.zeros(labels.size, dtype=np.bool_)
    cells = sorted(set(zip(domains.tolist(), participants.tolist(), labels.tolist(), strict=True)))
    for domain, participant, label in cells:
        indices = np.flatnonzero(
            (domains == domain) & (participants == participant) & (labels == label)
        )
        order = indices[np.argsort(window_ids[indices], kind="stable")]
        selected[order[:maximum]] = True
    return selected


def _build_candidate(candidate: dict[str, Any], config: dict[str, Any]) -> Any:
    identifier = str(candidate["id"])
    forest = _mapping(config["forest"], name="forest")
    n_estimators = int(forest["n_estimators"])
    seed, n_jobs = int(config["seed"]), int(config["n_jobs"])
    if bool(candidate["hierarchy"]):
        return _HierarchicalForest(seed=seed, n_jobs=n_jobs, n_estimators=n_estimators)
    if "leaf3" in identifier:
        leaf = 3
    elif "leaf1" in identifier:
        leaf = 1
    else:
        raise ValueError(f"unrecognized DAGHAR augmentation candidate {identifier!r}")
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        max_features=str(forest["max_features"]),
        min_samples_leaf=leaf,
        class_weight=None,
        n_jobs=n_jobs,
        random_state=seed,
    )


def _fit_candidate(
    estimator: Any,
    features: FloatArray,
    labels: IntArray,
    participants: NDArray[np.str_],
    domains: NDArray[np.str_],
) -> Any:
    return estimator.fit(
        features,
        labels,
        sample_weight=_balanced_domain_weights(labels, participants, domains),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def run_daghar_augmented_nested_cv(
    *,
    source_manifest_path: Path,
    raw_csv_path: Path,
    dataset_manifest_path: Path,
    daghar_archive_path: Path,
    daghar_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Select external support use only on inner source participants."""

    if output_directory.exists():
        raise FileExistsError(f"DAGHAR augmentation output already exists: {output_directory}")
    config = _load_config(config_path)
    daghar_config = _mapping(config["daghar"], name="daghar")
    if sha256_file(daghar_archive_path) != str(daghar_config["archive_sha256"]):
        raise ValueError("DAGHAR archive does not match the experiment pin")
    domains = tuple(str(item) for item in cast(list[object], daghar_config["development_domains"]))
    partitions = tuple(
        str(item) for item in cast(list[object], daghar_config["allowed_partitions"])
    )
    if partitions != ("train",) or daghar_config["sealed_evaluation_domains_allowed"] is not False:
        raise PermissionError("DAGHAR augmentation may load development training partitions only")
    external = load_daghar_development_windows(
        daghar_archive_path,
        daghar_manifest_path,
        domains=domains,
        partitions=partitions,
    )
    cap = _deterministic_cell_cap(
        external.labels,
        external.participant_ids,
        external.domain_ids,
        external.window_ids,
        maximum=int(daghar_config["deterministic_max_windows_per_domain_participant_class"]),
    )
    external_features = extract_geometric_spectral_pyramid_features(
        external.signals[cap], sampling_rate_hz=float(daghar_config["sampling_rate_hz"])
    )
    external_labels = external.labels[cap]
    external_participants = external.participant_ids[cap]
    external_domains = external.domain_ids[cap]
    external_windows = external.window_ids[cap]

    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("DAGHAR augmentation requires the functional-core ontology")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    source = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    source_signals = np.asarray(source.signals, dtype=np.float32).copy()
    source_signals[:, :, :3] *= float(
        _mapping(config["inclusivehar"], name="inclusivehar")["acceleration_conversion_to_m_per_s2"]
    )
    source_features = extract_geometric_spectral_pyramid_features(
        source_signals,
        sampling_rate_hz=float(config["inclusivehar"]["sampling_rate_hz"]),
    )
    source_labels = np.asarray(source.labels, dtype=np.int64)
    source_participants = np.asarray(source.participant_ids, dtype=np.str_)
    source_windows = np.asarray(source.window_ids, dtype=np.str_)
    source_domains = np.asarray(["InclusiveHAR"] * source_labels.size, dtype=np.str_)
    candidates = cast(list[dict[str, Any]], config["candidates"])
    output_directory.mkdir(parents=True)
    fold_records: list[dict[str, Any]] = []
    aggregate_labels: list[IntArray] = []
    aggregate_probabilities: list[FloatArray] = []
    aggregate_participants: list[NDArray[np.str_]] = []
    aggregate_windows: list[NDArray[np.str_]] = []
    for outer in cast(list[dict[str, Any]], manifest["source_nested_cv"]):
        fold_id = str(outer["outer_fold_id"])
        outer_subjects = [str(item) for item in outer["outer_test_subjects"]]
        outer_mask = _mask_for_subjects(source_participants, outer_subjects)
        summaries: list[dict[str, Any]] = []
        candidate_lookup = {str(item["id"]): item for item in candidates}
        for candidate in candidates:
            reports: list[dict[str, Any]] = []
            for inner in cast(list[dict[str, Any]], outer["inner_folds"]):
                train_mask = _mask_for_subjects(
                    source_participants, [str(item) for item in inner["train_subjects"]]
                )
                validation_mask = _mask_for_subjects(
                    source_participants, [str(item) for item in inner["validation_subjects"]]
                )
                if np.any(train_mask & validation_mask) or np.any(validation_mask & outer_mask):
                    raise PermissionError("DAGHAR-augmented nested partition leakage")
                train_features = source_features[train_mask]
                train_labels = source_labels[train_mask]
                train_participants = source_participants[train_mask]
                train_domains = source_domains[train_mask]
                if bool(candidate["external_training"]):
                    train_features = np.concatenate((train_features, external_features))
                    train_labels = np.concatenate((train_labels, external_labels))
                    train_participants = np.concatenate((train_participants, external_participants))
                    train_domains = np.concatenate((train_domains, external_domains))
                estimator = _build_candidate(candidate, config)
                _fit_candidate(
                    estimator,
                    train_features,
                    train_labels,
                    train_participants,
                    train_domains,
                )
                reports.append(
                    classification_report(
                        source_labels[validation_mask],
                        _probabilities(estimator, source_features[validation_mask]),
                        source_participants[validation_mask].tolist(),
                        class_names=class_names,
                    )
                )
            summaries.append(_candidate_summary(candidate, reports))
        ranking = sorted(summaries, key=_selection_key)
        selected_id = str(ranking[0]["candidate_id"])
        selected_candidate = candidate_lookup[selected_id]
        train_mask = ~outer_mask
        train_features = source_features[train_mask]
        train_labels = source_labels[train_mask]
        train_participants = source_participants[train_mask]
        train_domains = source_domains[train_mask]
        if bool(selected_candidate["external_training"]):
            train_features = np.concatenate((train_features, external_features))
            train_labels = np.concatenate((train_labels, external_labels))
            train_participants = np.concatenate((train_participants, external_participants))
            train_domains = np.concatenate((train_domains, external_domains))
        selected = _build_candidate(selected_candidate, config)
        _fit_candidate(
            selected,
            train_features,
            train_labels,
            train_participants,
            train_domains,
        )
        probability = _probabilities(selected, source_features[outer_mask])
        report = classification_report(
            source_labels[outer_mask],
            probability,
            source_participants[outer_mask].tolist(),
            class_names=class_names,
        )
        fold_directory = output_directory / fold_id
        fold_directory.mkdir()
        model_path = fold_directory / "selected_model.pkl"
        with model_path.open("xb") as stream:
            pickle.dump(selected, stream, protocol=5)
        prediction_path = fold_directory / "predictions.npz"
        with prediction_path.open("xb") as stream:
            np.savez_compressed(
                stream,
                probabilities=probability,
                labels=source_labels[outer_mask],
                participant_ids=source_participants[outer_mask],
                window_ids=source_windows[outer_mask],
            )
        fold_record: dict[str, Any] = {
            "schema_version": "1.0.0",
            "record_kind": "daghar_augmented_geometric_nested_outer_result",
            "status": "complete_target_and_external_evaluation_domains_sealed",
            "outer_fold_id": fold_id,
            "inner_candidate_ranking": ranking,
            "selection": {
                "candidate_id": selected_id,
                "external_training": bool(selected_candidate["external_training"]),
                "outer_labels_used": False,
            },
            "outer_report": report,
            "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
            "predictions": {
                "path": prediction_path.as_posix(),
                "sha256": sha256_file(prediction_path),
            },
            "target_subject_or_window_records_loaded": False,
            "target_performance_or_prediction_accessed": False,
            "daghar_sealed_domain_labels_or_metrics_accessed": False,
        }
        fold_record["record_sha256"] = canonical_json_sha256(fold_record)
        fold_path = fold_directory / "result.json"
        _write_json(fold_path, fold_record)
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "path": fold_path.as_posix(),
                "sha256": sha256_file(fold_path),
                "selected_candidate_id": selected_id,
                "outer_report": report,
            }
        )
        aggregate_labels.append(source_labels[outer_mask])
        aggregate_probabilities.append(probability)
        aggregate_participants.append(source_participants[outer_mask])
        aggregate_windows.append(source_windows[outer_mask])
    labels = np.concatenate(aggregate_labels)
    probabilities = np.concatenate(aggregate_probabilities)
    participants = np.concatenate(aggregate_participants)
    windows = np.concatenate(aggregate_windows)
    if set(windows.tolist()) != set(source_windows.tolist()) or len(set(windows.tolist())) != len(
        windows
    ):
        raise ValueError("DAGHAR-augmented outer folds do not cover source exactly once")
    report = classification_report(
        labels,
        probabilities,
        participants.tolist(),
        class_names=class_names,
    )
    participant_values = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], report["participants"])
    }
    prediction_path = output_directory / "all_outer_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=probabilities,
            labels=labels,
            participant_ids=participants,
            window_ids=windows,
        )
    external_selection_record = {
        "uncapped_train_window_count": int(external.labels.size),
        "selected_train_window_count": int(cap.sum()),
        "selected_window_ids_sha256": canonical_json_sha256(sorted(external_windows.tolist())),
        "domains": sorted(set(external_domains.tolist())),
        "participants": len(set(external_participants.tolist())),
        "class_counts": np.bincount(external_labels, minlength=3).tolist(),
    }
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "daghar_augmented_geometric_nested_source_aggregate",
        "status": "complete_target_and_external_evaluation_domains_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "dataset_manifest": {
            "path": dataset_manifest_path.as_posix(),
            "sha256": sha256_file(dataset_manifest_path),
        },
        "daghar_manifest": {
            "path": daghar_manifest_path.as_posix(),
            "sha256": sha256_file(daghar_manifest_path),
        },
        "daghar_archive_sha256": sha256_file(daghar_archive_path),
        "daghar_training_selection": external_selection_record,
        "code_commit": code_commit,
        "feature_count": source_features.shape[1],
        "feature_names_sha256": canonical_json_sha256(
            list(geometric_spectral_pyramid_feature_names())
        ),
        "folds": fold_records,
        "aggregate_report": report,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "outer_labels_used_for_candidate_selection": False,
            "external_evaluation_domain_labels_or_metrics_used": False,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "daghar_sealed_domain_labels_or_metrics_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    _write_json(output_directory / "result.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--daghar-archive", type=Path, required=True)
    parser.add_argument("--daghar-manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_daghar_augmented_nested_cv(
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        dataset_manifest_path=args.dataset_manifest,
        daghar_archive_path=args.daghar_archive,
        daghar_manifest_path=args.daghar_manifest,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_by_fold": {
                    item["outer_fold_id"]: item["selected_candidate_id"] for item in result["folds"]
                },
                "external_training": result["daghar_training_selection"],
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
