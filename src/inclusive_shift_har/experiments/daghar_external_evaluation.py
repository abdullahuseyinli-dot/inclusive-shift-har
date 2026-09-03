"""Freeze and open one held-out DAGHAR evaluation for a source-trained candidate."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.data.daghar import load_daghar_evaluation_windows
from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import participant_bootstrap_interval
from inclusive_shift_har.experiments.fuse_reframe_source import _load_source_manifest
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _build_estimator,
    _fit_estimator,
    _mapping,
    _probabilities,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
    geometric_spectral_pyramid_feature_names,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_signal_views,
)

FloatArray = NDArray[np.float64]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")


def _verify_record(record: dict[str, Any], *, name: str) -> None:
    claimed = record.get("record_sha256")
    unhashed = dict(record)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError(f"{name} self-hash changed")


def _load_config(path: Path) -> dict[str, Any]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    disclosures = _mapping(config.get("disclosures"), name="disclosures")
    candidate = _mapping(config.get("candidate"), name="candidate")
    bridge = _mapping(config.get("unit_bridge"), name="unit bridge")
    evaluation = _mapping(config.get("external_evaluation"), name="external evaluation")
    selection = _mapping(config.get("selection"), name="selection")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        disclosures.get(
            "schema_headers_and_first_raw_row_probed_before_domain_partition_declaration"
        )
        is not True
    ):
        raise ValueError("DAGHAR external protocol must disclose the earlier schema probe")
    if (
        disclosures.get("sealed_domain_labels_predictions_or_metrics_accessed_before_freeze")
        is not False
    ):
        raise PermissionError("DAGHAR sealed-domain performance was not preserved")
    if (
        candidate.get("required_unanimous_selected_view") != "denoised"
        or candidate.get("required_unanimous_estimator") != "extra_trees_leaf1"
    ):
        raise ValueError("DAGHAR external candidate must be the frozen denoised leaf-1 forest")
    if candidate.get("target_data_used") is not False:
        raise PermissionError("DAGHAR external candidate cannot use target data")
    if float(bridge.get("inclusivehar_acceleration_multiplier", 0.0)) != 9.80665:
        raise ValueError("DAGHAR external SI-unit bridge changed")
    if evaluation.get("domains_in_frozen_order") != ["MotionSense", "KuHar", "WISDM"]:
        raise ValueError("DAGHAR sealed-domain order changed")
    if evaluation.get("provider_partitions") != ["train", "validation", "test"]:
        raise ValueError("DAGHAR provider partition coverage changed")
    if (
        evaluation.get("train_on_evaluation_domains") is not False
        or evaluation.get("one_opening_only") is not True
    ):
        raise PermissionError("DAGHAR external protocol is not a one-opening evaluation")
    if (
        selection.get("evaluation_metrics_may_change_candidate") is not False
        or selection.get("target_data_allowed") is not False
    ):
        raise PermissionError("DAGHAR evaluation metrics or target data cannot select the method")
    if policy.get("confirmatory_claim_allowed") is not False:
        raise ValueError("DAGHAR held-out-after-schema-probe evidence is not confirmatory")
    return config


def _selected_candidate_contract(nested: dict[str, Any]) -> tuple[str, str]:
    folds = cast(list[dict[str, Any]], nested.get("folds"))
    candidates = {str(item["selected_candidate_id"]) for item in folds}
    views = {str(item["selected_feature_view"]) for item in folds}
    if candidates != {"denoised_extra_trees_leaf1"} or views != {"denoised"}:
        raise ValueError("external freeze requires unanimous denoised leaf-1 outer selections")
    config_reference = _mapping(nested.get("config"), name="nested config")
    nested_config_path = Path(str(config_reference["path"]))
    if sha256_file(nested_config_path) != str(config_reference["sha256"]):
        raise ValueError("nested experiment config hash changed")
    nested_config = _mapping(
        yaml.safe_load(nested_config_path.read_text(encoding="utf-8")), name="nested config"
    )
    selected = [
        _mapping(item, name="nested candidate")
        for item in cast(list[object], nested_config["candidates"])
        if _mapping(item, name="nested candidate").get("id") == "denoised_extra_trees_leaf1"
    ]
    if (
        len(selected) != 1
        or selected[0].get("view") != "denoised"
        or selected[0].get("estimator") != "extra_trees_leaf1"
    ):
        raise ValueError("nested denoised candidate definition changed")
    return "denoised_extra_trees_leaf1", "extra_trees_leaf1"


def freeze_daghar_external_candidate(
    *,
    nested_result_path: Path,
    raw_csv_path: Path,
    daghar_archive_path: Path,
    daghar_manifest_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Fit and hash the one final source model before held-out-domain access."""

    if output_directory.exists():
        raise FileExistsError(f"DAGHAR candidate-freeze output already exists: {output_directory}")
    config = _load_config(config_path)
    nested = _mapping(load_json_strict(nested_result_path), name="nested result")
    _verify_record(nested, name="nested result")
    if nested.get("record_kind") != "robust_multiscale_residual_nested_source_aggregate":
        raise ValueError("DAGHAR candidate freeze requires robust multiscale nested evidence")
    if (
        nested.get("status") != "complete_target_sealed"
        or nested.get("target_subject_or_window_records_loaded") is not False
        or nested.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("DAGHAR candidate freeze refuses target-bearing evidence")
    if nested.get("code_commit") != code_commit:
        raise ValueError("nested result was not produced by the declared candidate code commit")
    candidate_id, estimator_id = _selected_candidate_contract(nested)
    daghar_config = _mapping(config["external_evaluation"], name="external evaluation")
    if sha256_file(daghar_archive_path) != str(daghar_config["archive_sha256"]):
        raise ValueError("DAGHAR archive hash differs from the external-evaluation protocol")
    if Path(str(daghar_config["manifest"])) != daghar_manifest_path:
        raise ValueError("DAGHAR manifest path differs from the external-evaluation protocol")

    source_reference = _mapping(nested.get("source_manifest"), name="source manifest")
    source_manifest_path = Path(str(source_reference["path"]))
    if sha256_file(source_manifest_path) != str(source_reference["sha256"]):
        raise ValueError("source manifest hash changed")
    manifest = _load_source_manifest(source_manifest_path)
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("DAGHAR external candidate requires functional core")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    source = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(source.signals, dtype=np.float32).copy()
    signals[:, :, :3] *= float(config["unit_bridge"]["inclusivehar_acceleration_multiplier"])
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=50.0)["denoised"]
    features = extract_geometric_spectral_pyramid_features(denoised, sampling_rate_hz=50.0)
    labels = np.asarray(source.labels, dtype=np.int64)
    participants = np.asarray(source.participant_ids, dtype=np.str_)
    candidate_config = _mapping(config["candidate"], name="candidate")
    estimator = _build_estimator(estimator_id, seed=int(candidate_config["seed"]), n_jobs=4)
    _fit_estimator(estimator, estimator_id, features, labels, participants)

    output_directory.mkdir(parents=True)
    model_path = output_directory / "frozen_source_model.pkl"
    with model_path.open("xb") as stream:
        pickle.dump(
            {
                "candidate_id": candidate_id,
                "feature_view": "denoised",
                "estimator_id": estimator_id,
                "source_acceleration_multiplier": 9.80665,
                "estimator": estimator,
            },
            stream,
            protocol=5,
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "daghar_external_evaluation_candidate_freeze",
        "status": "frozen_before_external_evaluation_opening",
        "evidence_status": "source_developed_heldout_performance_after_schema_probe",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "source_nested_result": {
            "path": nested_result_path.as_posix(),
            "sha256": sha256_file(nested_result_path),
            "record_sha256": nested["record_sha256"],
        },
        "source_manifest": source_reference,
        "raw_source": {"path": raw_csv_path.as_posix(), "sha256": sha256_file(raw_csv_path)},
        "code_commit": code_commit,
        "candidate": {
            "candidate_id": candidate_id,
            "feature_view": "denoised",
            "estimator_id": estimator_id,
            "seed": int(candidate_config["seed"]),
            "training_participants": sorted(set(participants.tolist()), key=int),
            "training_window_count": int(labels.size),
            "source_acceleration_multiplier": 9.80665,
            "feature_count": int(features.shape[1]),
            "feature_names_sha256": canonical_json_sha256(
                list(geometric_spectral_pyramid_feature_names())
            ),
        },
        "model": {"path": model_path.as_posix(), "sha256": sha256_file(model_path)},
        "daghar_manifest": {
            "path": daghar_manifest_path.as_posix(),
            "sha256": sha256_file(daghar_manifest_path),
        },
        "daghar_archive_sha256": sha256_file(daghar_archive_path),
        "sealed_evaluation_domains": list(daghar_config["domains_in_frozen_order"]),
        "provider_partitions": list(daghar_config["provider_partitions"]),
        "schema_probe_before_partition_declaration_disclosed": True,
        "sealed_domain_labels_predictions_or_metrics_accessed_before_freeze": False,
        "evaluation_metrics_may_change_candidate": False,
        "confirmatory_claim_allowed": False,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "freeze.json", result)
    return result


def _predict_external_in_chunks(
    estimator: Any,
    signals: NDArray[np.float32],
    *,
    sampling_rate_hz: float,
    chunk_size: int = 512,
) -> FloatArray:
    rows: list[FloatArray] = []
    for start in range(0, signals.shape[0], chunk_size):
        batch = signals[start : start + chunk_size]
        denoised = robust_multiscale_signal_views(batch, sampling_rate_hz=sampling_rate_hz)[
            "denoised"
        ]
        features = extract_geometric_spectral_pyramid_features(
            denoised, sampling_rate_hz=sampling_rate_hz
        )
        rows.append(_probabilities(estimator, features))
    return np.concatenate(rows)


def external_domain_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate frozen domain reports without weighting large domains more heavily."""

    if not reports:
        raise ValueError("external domain summary requires at least one report")
    means = {
        domain: float(report["primary"]["mean_participant_macro_f1"])
        for domain, report in reports.items()
    }
    if not np.isfinite(list(means.values())).all():
        raise ValueError("external domain means must be finite")
    return {
        "equal_domain_mean_of_participant_macro_f1": float(np.mean(list(means.values()))),
        "worst_domain_mean_participant_macro_f1": min(means.values()),
        "domain_mean_participant_macro_f1": means,
        "domain_count": len(means),
    }


def evaluate_daghar_external_candidate(
    *,
    freeze_record_path: Path,
    daghar_archive_path: Path,
    daghar_manifest_path: Path,
    config_path: Path,
    opening_receipt_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Consume the one held-out DAGHAR performance opening for the frozen model."""

    if output_directory.exists():
        raise FileExistsError(f"DAGHAR external-evaluation output exists: {output_directory}")
    if opening_receipt_path.exists():
        raise FileExistsError(
            f"DAGHAR external performance opening is consumed: {opening_receipt_path}"
        )
    config = _load_config(config_path)
    freeze = _mapping(load_json_strict(freeze_record_path), name="candidate freeze")
    _verify_record(freeze, name="candidate freeze")
    if (
        freeze.get("record_kind") != "daghar_external_evaluation_candidate_freeze"
        or freeze.get("status") != "frozen_before_external_evaluation_opening"
    ):
        raise PermissionError("DAGHAR external evaluation requires a completed candidate freeze")
    config_reference = _mapping(freeze.get("config"), name="frozen config")
    if sha256_file(config_path) != str(config_reference["sha256"]):
        raise ValueError("DAGHAR external evaluation config changed after freeze")
    manifest_reference = _mapping(freeze.get("daghar_manifest"), name="DAGHAR manifest")
    if daghar_manifest_path.as_posix() != str(manifest_reference["path"]) or sha256_file(
        daghar_manifest_path
    ) != str(manifest_reference["sha256"]):
        raise ValueError("DAGHAR manifest changed after freeze")
    archive_sha256 = sha256_file(daghar_archive_path)
    if archive_sha256 != str(freeze["daghar_archive_sha256"]):
        raise ValueError("DAGHAR archive changed after freeze")
    model_reference = _mapping(freeze.get("model"), name="frozen model")
    model_path = Path(str(model_reference["path"]))
    if sha256_file(model_path) != str(model_reference["sha256"]):
        raise ValueError("DAGHAR frozen source model hash changed")
    with model_path.open("rb") as stream:
        bundle = _mapping(pickle.load(stream), name="frozen source model")
    if (
        bundle.get("feature_view") != "denoised"
        or bundle.get("estimator_id") != "extra_trees_leaf1"
    ):
        raise ValueError("DAGHAR frozen source model contract changed")

    evaluation = _mapping(config["external_evaluation"], name="external evaluation")
    domains = tuple(str(item) for item in evaluation["domains_in_frozen_order"])
    partitions = tuple(str(item) for item in evaluation["provider_partitions"])
    output_directory.mkdir(parents=True)
    opening_receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "daghar_external_evaluation_opening_receipt",
        "status": "performance_opening_consumed_before_domain_read",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "candidate_freeze": {
            "path": freeze_record_path.as_posix(),
            "sha256": sha256_file(freeze_record_path),
            "record_sha256": freeze["record_sha256"],
        },
        "daghar_archive_sha256": archive_sha256,
        "domains": list(domains),
        "partitions": list(partitions),
        "schema_probe_before_partition_declaration_disclosed": True,
        "sealed_domain_labels_predictions_or_metrics_accessed_before_receipt": False,
        "evidence_status": "heldout_performance_after_schema_probe_not_pristine_confirmatory",
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt)
    _write_json(opening_receipt_path, receipt)

    external = load_daghar_evaluation_windows(
        daghar_archive_path,
        daghar_manifest_path,
        domains=domains,
        partitions=partitions,
        freeze_record_path=freeze_record_path,
        expected_freeze_file_sha256=sha256_file(freeze_record_path),
    )
    probabilities = _predict_external_in_chunks(
        bundle["estimator"],
        external.signals,
        sampling_rate_hz=float(evaluation["sampling_rate_hz"]),
    )
    class_names = ("mobility", "sitting", "standing")
    all_report = classification_report(
        external.labels,
        probabilities,
        external.participant_ids.tolist(),
        class_names=class_names,
    )
    domain_reports: dict[str, dict[str, Any]] = {}
    for domain in domains:
        mask = external.domain_ids == domain
        domain_reports[domain] = classification_report(
            external.labels[mask],
            probabilities[mask],
            external.participant_ids[mask].tolist(),
            class_names=class_names,
        )
    domain_summary = external_domain_summary(domain_reports)
    participant_values = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in cast(list[dict[str, Any]], all_report["participants"])
    }
    prediction_path = output_directory / "heldout_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            probabilities=probabilities,
            labels=external.labels,
            participant_ids=external.participant_ids,
            domain_ids=external.domain_ids,
            partitions=external.partitions,
            window_ids=external.window_ids,
        )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "daghar_external_heldout_performance_evaluation",
        "status": "complete_heldout_performance_after_schema_probe",
        "evidence_status": "heldout_performance_after_schema_probe_not_pristine_confirmatory",
        "candidate_freeze": receipt["candidate_freeze"],
        "opening_receipt": {
            "path": opening_receipt_path.as_posix(),
            "sha256": sha256_file(opening_receipt_path),
            "record_sha256": receipt["record_sha256"],
        },
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "daghar_manifest": manifest_reference,
        "daghar_archive_sha256": archive_sha256,
        "library_versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "scipy")
        },
        "evaluation": {
            "domains": list(domains),
            "partitions": list(partitions),
            "window_count": int(external.labels.size),
            "participant_count": len(set(external.participant_ids.tolist())),
            "class_counts": np.bincount(external.labels, minlength=3).tolist(),
            "domain_partition_window_counts": {
                f"{domain}:{partition}": int(
                    np.sum((external.domain_ids == domain) & (external.partitions == partition))
                )
                for domain in domains
                for partition in partitions
            },
        },
        "primary": domain_summary,
        "all_participant_report": all_report,
        "domain_reports": domain_reports,
        "participant_bootstrap": participant_bootstrap_interval(participant_values),
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "evaluation_metrics_may_change_candidate": False,
            "evaluation_domains_used_for_training": False,
            "target_data_used": False,
            "confirmatory_claim_allowed": False,
        },
        "daghar_sealed_domain_labels_predictions_or_metrics_accessed": True,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    _write_json(output_directory / "result.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze")
    freeze.add_argument("--nested-result", type=Path, required=True)
    freeze.add_argument("--raw-csv", type=Path, required=True)
    freeze.add_argument("--daghar-archive", type=Path, required=True)
    freeze.add_argument("--daghar-manifest", type=Path, required=True)
    freeze.add_argument("--config", type=Path, required=True)
    freeze.add_argument("--output-directory", type=Path, required=True)
    freeze.add_argument("--code-commit", required=True)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--freeze-record", type=Path, required=True)
    evaluate.add_argument("--daghar-archive", type=Path, required=True)
    evaluate.add_argument("--daghar-manifest", type=Path, required=True)
    evaluate.add_argument("--config", type=Path, required=True)
    evaluate.add_argument("--opening-receipt", type=Path, required=True)
    evaluate.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze":
        result = freeze_daghar_external_candidate(
            nested_result_path=args.nested_result,
            raw_csv_path=args.raw_csv,
            daghar_archive_path=args.daghar_archive,
            daghar_manifest_path=args.daghar_manifest,
            config_path=args.config,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
        )
        payload = {
            "status": result["status"],
            "candidate": result["candidate"],
            "model": result["model"],
            "record_sha256": result["record_sha256"],
        }
    else:
        result = evaluate_daghar_external_candidate(
            freeze_record_path=args.freeze_record,
            daghar_archive_path=args.daghar_archive,
            daghar_manifest_path=args.daghar_manifest,
            config_path=args.config,
            opening_receipt_path=args.opening_receipt,
            output_directory=args.output_directory,
        )
        payload = {
            "status": result["status"],
            "evidence_status": result["evidence_status"],
            "evaluation": result["evaluation"],
            "primary": result["primary"],
            "all_participant_primary": result["all_participant_report"]["primary"],
            "class_recall": result["all_participant_report"]["window_level_diagnostics"][
                "per_class_recall"
            ],
            "record_sha256": result["record_sha256"],
        }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
