"""Frozen-source zero-shot benchmark on the AICOS-HAR provider test fold."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.aicos_har import (
    AICOS_ARCHIVE_MD5,
    AICOS_ARCHIVE_SIZE_BYTES,
    INCLUSIVEHAR_SIGNAL_UNITS,
    aicos_in_inclusivehar_units,
    audit_aicos_archive,
    load_aicos_har,
    verify_aicos_archive,
)
from inclusive_shift_har.data.external_har import (
    CORE_CLASS_NAMES,
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _feature_views,
    _materialize_gravity,
)
from inclusive_shift_har.experiments.cross_dataset_har import _paired_bootstrap
from inclusive_shift_har.experiments.cross_dataset_transfer import _fit_apply_seed
from inclusive_shift_har.experiments.ctgr_source_finalization import (
    _predict_features,
)
from inclusive_shift_har.experiments.ctgr_source_finalization import (
    validate_artifacts as validate_frozen_predictor,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import _materialize_source
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]

SOURCE_MANIFEST = Path(
    "results/protocol/source_windows/inclusivehar_v4_source_development_v1_2.json"
)
EXPECTED_SOURCE_MANIFEST_SHA256 = "1d49435ad9371a9701198e15ec30a35a5c3c38ee5458f6f1980de7d4306e76e2"
EXPECTED_SOURCE_CSV_SHA256 = "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34"
SEED = 11


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def _source_windows(repository_root: Path, raw_csv: Path) -> ExternalHARWindows:
    manifest_path = repository_root / SOURCE_MANIFEST
    if sha256_file(manifest_path) != EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ValueError("InclusiveHAR source manifest differs from the frozen source recipe")
    raw_sha256 = sha256_file(raw_csv)
    if raw_sha256 != EXPECTED_SOURCE_CSV_SHA256:
        raise ValueError("InclusiveHAR raw CSV differs from the frozen source recipe")
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=manifest_path,
        raw_csv_path=raw_csv,
    )
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=raw_csv,
        expected_participants=participants,
        expected_windows=windows,
    )
    if signals.shape != (725, 128, 6) or gravity.shape != (725, 128, 3):
        raise ValueError("InclusiveHAR source windows differ from the finalized 725-window recipe")
    prefixed = np.asarray([f"inclusivehar:P{item}" for item in participants], dtype=np.str_)
    sessions = np.asarray([f"inclusivehar-session:P{item}" for item in participants], dtype=np.str_)
    trials = np.asarray([f"inclusivehar-window:{item}" for item in windows], dtype=np.str_)
    identifiers = np.asarray([f"inclusivehar/{item}" for item in windows], dtype=np.str_)
    signal32 = np.asarray(signals, dtype=np.float32)
    gravity32 = np.asarray(gravity, dtype=np.float32)
    receipt = SourceReceipt(
        dataset_id="inclusivehar_v4_source",
        locator=str(raw_csv.resolve()),
        member=None,
        declared_size_bytes=raw_csv.stat().st_size,
        received_size_bytes=raw_csv.stat().st_size,
        computed_sha256=raw_sha256,
        raw_local_mirror=True,
        evidence_role="frozen_source_training",
    )
    pool = ObservableWindowPool(
        signals=signal32,
        gravity=gravity32,
        participant_ids=prefixed,
        session_ids=sessions,
        trial_ids=trials,
        window_ids=identifiers,
    )
    plan = build_participant_partition_plan(
        "inclusivehar_v4_source",
        np.unique(prefixed).tolist(),
        roster_basis="frozen source P1-P10 roster before materialization",
    )
    result = ExternalHARWindows(
        dataset_id="inclusivehar_v4_source",
        channel_lane="native-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=signal32,
        gravity=gravity32,
        labels=labels,
        participant_ids=prefixed,
        session_ids=sessions,
        trial_ids=trials,
        window_ids=identifiers,
        receipts=(receipt,),
        gravity_source="provider-native motionGravity",
        gravity_cutoff_hz=None,
        observable_candidates=pool,
        participant_partition_plan=plan,
        cohort_audit={
            "signal_units": dict(INCLUSIVEHAR_SIGNAL_UNITS),
            "source_participants": np.unique(prefixed).tolist(),
            "frozen_source_window_count": int(labels.size),
            "raw_csv_sha256": raw_sha256,
            "source_manifest_sha256": EXPECTED_SOURCE_MANIFEST_SHA256,
        },
    )
    result.validate()
    return result


def _frozen_predictor_probabilities(
    target: ExternalHARWindows, predictor_path: Path
) -> dict[str, FloatArray]:
    if (target.cohort_audit or {}).get("signal_units") != INCLUSIVEHAR_SIGNAL_UNITS:
        raise ValueError("frozen InclusiveHAR predictor requires acceleration/gravity in g")
    with predictor_path.open("rb") as stream:
        predictor = cast(dict[str, Any], pickle.load(stream))
    base, views = _feature_views(
        np.asarray(target.signals, dtype=np.float64),
        np.asarray(target.gravity, dtype=np.float64),
        sampling_rate_hz=target.sampling_rate_hz,
    )
    values = _predict_features(
        base,
        views,
        cast(dict[str, Any], predictor["models"]),
        cast(dict[str, Any], predictor["selected_candidate"]),
    )
    names = {
        "B6": "Frozen-B6-RMRP",
        "B9": "Frozen-B9-PhysicsRMRP",
        "T9": "Frozen-T9-CTGR",
        "U9": "Frozen-U9-Unconditional",
    }
    return {names[key]: value for key, value in values.items()}


def _complete_participant_mask(target: ExternalHARWindows) -> NDArray[np.bool_]:
    complete = {
        str(participant)
        for participant in np.unique(target.participant_ids)
        if set(target.labels[target.participant_ids == participant].tolist()) == {0, 1, 2}
    }
    if not complete:
        raise ValueError("AICOS-HAR qualified provider test fold has no complete participant")
    return np.asarray(np.isin(target.participant_ids, sorted(complete)), dtype=np.bool_)


def _report(
    target: ExternalHARWindows, probability: FloatArray, mask: NDArray[np.bool_]
) -> dict[str, Any]:
    report = classification_report(
        target.labels[mask],
        probability[mask],
        target.participant_ids[mask].tolist(),
        class_names=CORE_CLASS_NAMES,
    )
    values = sorted(float(item["macro_f1"]) for item in report["participants"])
    count = max(1, int(np.ceil(0.30 * len(values))))
    report["primary"]["bottom_30_percent_participant_macro_f1"] = float(np.mean(values[:count]))
    return report


def _strata(
    target: ExternalHARWindows, probability: FloatArray, mask: NDArray[np.bool_]
) -> dict[str, Any]:
    prediction = np.argmax(probability, axis=1)
    output: dict[str, Any] = {}
    for kind in ("device", "position"):
        values = []
        for session in target.session_ids:
            device_position = str(session).removeprefix("aicos-device:")
            device, position = device_position.rsplit("_", 1)
            values.append(device if kind == "device" else position)
        strata = np.asarray(values, dtype=np.str_)
        records: list[dict[str, Any]] = []
        for value in np.unique(strata[mask]):
            selected = mask & (strata == value)
            truth = target.labels[selected]
            guessed = prediction[selected]
            recall = {}
            for index, name in enumerate(CORE_CLASS_NAMES):
                rows = truth == index
                recall[name] = None if not rows.any() else float(np.mean(guessed[rows] == index))
            records.append(
                {
                    kind: str(value),
                    "window_count": int(selected.sum()),
                    "participant_count": int(np.unique(target.participant_ids[selected]).size),
                    "accuracy": float(np.mean(truth == guessed)),
                    "per_class_recall": recall,
                }
            )
        output[kind] = records
    return output


def _participant_differences(report: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    left = {str(item["participant_id"]): float(item["macro_f1"]) for item in report["participants"]}
    right = {
        str(item["participant_id"]): float(item["macro_f1"]) for item in reference["participants"]
    }
    differences = {participant: left[participant] - right[participant] for participant in left}
    return {
        "differences": differences,
        "wins": sum(value > 0.0 for value in differences.values()),
        "harms": sum(value < 0.0 for value in differences.values()),
        "ties": sum(value == 0.0 for value in differences.values()),
    }


def run_benchmark(
    *,
    repository_root: Path,
    archive_path: Path,
    source_csv: Path,
    frozen_predictor_directory: Path,
    output_directory: Path,
    n_jobs: int,
) -> dict[str, Any]:
    """Run the one-opening AICOS provider-test comparison and seal its evidence."""

    if output_directory.exists():
        raise FileExistsError(f"create-only output already exists: {output_directory}")
    output_directory.mkdir(parents=True)
    started = datetime.now(UTC).isoformat()
    overall_started = time.perf_counter()
    stage_seconds: dict[str, float] = {}
    try:
        stage = time.perf_counter()
        verification = verify_aicos_archive(archive_path)
        stage_seconds["archive_verification"] = time.perf_counter() - stage

        stage = time.perf_counter()
        development_audit = audit_aicos_archive(archive_path, folds=("1", "2", "3", "4", "5"))
        development_audit["archive_verification"] = {
            "size_bytes": verification.size_bytes,
            "md5": verification.md5,
            "sha256": verification.sha256,
            "provider_md5_verified": verification.provider_md5_verified,
        }
        development_audit["record_sha256"] = canonical_json_sha256(development_audit)
        _write_json(output_directory / "development_signal_qualification.json", development_audit)
        stage_seconds["development_signal_qualification"] = time.perf_counter() - stage

        if development_audit["status_counts"].get("qualified", 0) == 0:
            raise RuntimeError(
                "AICOS-HAR development signal qualification produced no valid acquisition"
            )

        stage = time.perf_counter()
        source = _source_windows(repository_root, source_csv)
        predictor_validation = validate_frozen_predictor(frozen_predictor_directory)
        predictor_path = frozen_predictor_directory / "predictor.pkl"
        predictor_hash = sha256_file(predictor_path)
        stage_seconds["source_and_predictor_validation"] = time.perf_counter() - stage

        stage = time.perf_counter()
        target = load_aicos_har(
            archive_path,
            verification=verification,
            folds=("test",),
            target_rate_hz=50.0,
            window_samples=128,
            gravity_cutoff_hz=0.30,
        )
        target = aicos_in_inclusivehar_units(target)
        stage_seconds["provider_test_materialization"] = time.perf_counter() - stage

        stage = time.perf_counter()
        suite_probability, suite_record = _fit_apply_seed(
            source,
            target,
            seed=SEED,
            repository_root=repository_root,
            n_jobs=n_jobs,
            include_classical=True,
            expected_dataset_pair=("inclusivehar_v4_source", "aicos_har_v1"),
            allow_cross_gravity_interface=True,
        )
        frozen_probability = _frozen_predictor_probabilities(target, predictor_path)
        probabilities = {**suite_probability, **frozen_probability}
        stage_seconds["model_fit_and_inference"] = time.perf_counter() - stage

        primary_mask = _complete_participant_mask(target)
        all_mask = np.ones(target.labels.size, dtype=np.bool_)
        primary_reports = {
            method: _report(target, probability, primary_mask)
            for method, probability in probabilities.items()
        }
        coverage_reports = {
            method: _report(target, probability, all_mask)
            for method, probability in probabilities.items()
        }
        reference_name = "RMRP-DG"
        comparisons = {
            method: {
                "paired_participant_bootstrap_vs_rmrp": _paired_bootstrap(
                    report,
                    primary_reports[reference_name],
                    seed=20260919 + index,
                ),
                "participant_effects_vs_rmrp": _participant_differences(
                    report, primary_reports[reference_name]
                ),
            }
            for index, (method, report) in enumerate(primary_reports.items())
            if method != reference_name
        }
        strongest = max(
            primary_reports,
            key=lambda name: float(primary_reports[name]["primary"]["mean_participant_macro_f1"]),
        )
        stratum_reports = {
            "reference": {
                "method": reference_name,
                **_strata(target, probabilities[reference_name], primary_mask),
            },
            "descriptive_best": {
                "method": strongest,
                **_strata(target, probabilities[strongest], primary_mask),
            },
        }
        primary_participants = sorted(np.unique(target.participant_ids[primary_mask]).tolist())
        prediction_path = output_directory / "predictions.npz"
        prediction_payload: dict[str, Any] = {
            "labels": target.labels,
            "participant_ids": target.participant_ids,
            "session_ids": target.session_ids,
            "trial_ids": target.trial_ids,
            "window_ids": target.window_ids,
            "primary_complete_participant_mask": primary_mask,
            **{f"probability__{name}": value for name, value in probabilities.items()},
        }
        with prediction_path.open("xb") as stream:
            np.savez_compressed(stream, **prediction_payload)
        cohort_audit = cast(dict[str, Any], target.cohort_audit)
        result: dict[str, Any] = {
            "schema_version": "1.0.0",
            "experiment_id": "inclusivehar-to-aicos-zero-shot-v1",
            "status": "complete",
            "evidence_status": "EXTERNAL_ZERO_SHOT_HEALTHY_CROSS_DEVICE_POSITION_NOT_ABILITY_CONFIRMATION",
            "started_at": started,
            "completed_at": datetime.now(UTC).isoformat(),
            "seed": SEED,
            "source_dataset": source.summary(),
            "target_dataset": target.summary(),
            "primary_participants": primary_participants,
            "primary_participant_count": len(primary_participants),
            "primary_endpoint": "provider-test complete-three-class participants",
            "coverage_endpoint": "all unit-qualified provider-test participants",
            "primary_reports": primary_reports,
            "coverage_reports": coverage_reports,
            "paired_comparisons": comparisons,
            "descriptive_best_method": strongest,
            "device_position_strata": stratum_reports,
            "suite_fit_record": suite_record,
            "unit_qualification_summary": {
                "development_status_counts": development_audit["status_counts"],
                "development_unit_rule_counts": development_audit["unit_rule_counts"],
                "test_qualified_acquisitions": cohort_audit["qualified_acquisition_count"],
                "test_quarantined_acquisitions": cohort_audit["quarantined_acquisition_count"],
            },
            "frozen_predictor": {
                "directory": str(frozen_predictor_directory.resolve()),
                "predictor_sha256": predictor_hash,
                "validation": predictor_validation,
                "validated_before_provider_test_model_inference": True,
            },
            "archive": {
                "path": str(archive_path.resolve()),
                "expected_size_bytes": AICOS_ARCHIVE_SIZE_BYTES,
                "size_bytes": verification.size_bytes,
                "expected_md5": AICOS_ARCHIVE_MD5,
                "md5": verification.md5,
                "sha256": verification.sha256,
                "extracted": False,
            },
            "runtime_seconds": {
                **stage_seconds,
                "total": time.perf_counter() - overall_started,
            },
            "artifacts": {
                "development_signal_qualification": "development_signal_qualification.json",
                "predictions": prediction_path.name,
                "predictions_sha256": sha256_file(prediction_path),
            },
            "claim_policy": {
                "target_labels_used_for_fit_selection_or_calibration": False,
                "target_scores_used_to_select_method": False,
                "descriptive_best_is_post_evaluation_only": True,
                "ability_generalization_claim_allowed": False,
                "state_of_the_art_claim_allowed_without_protocol_matched_literature": False,
                "derived_and_native_gravity_results_pooled": False,
            },
        }
        result["record_sha256"] = canonical_json_sha256(result)
        _write_json(output_directory / "result.json", result)
        manifest = {
            "schema_version": "1.0.0",
            "files": [
                {
                    "path": path.name,
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
                for path in sorted(output_directory.iterdir())
                if path.is_file()
            ],
        }
        manifest["record_sha256"] = canonical_json_sha256(manifest)
        _write_json(output_directory / "manifest.json", manifest)
        return result
    except BaseException as error:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAILED_PRESERVED",
            "started_at": started,
            "failed_at": datetime.now(UTC).isoformat(),
            "exception_type": type(error).__name__,
            "exception_message": str(error),
            "runtime_seconds": time.perf_counter() - overall_started,
            "completed_stages": stage_seconds,
        }
        failure["record_sha256"] = canonical_json_sha256(failure)
        _write_json(output_directory / "failure.json", failure)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--frozen-predictor-directory", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=4)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run_benchmark(
        repository_root=args.repository_root.resolve(),
        archive_path=args.archive.resolve(),
        source_csv=args.source_csv.resolve(),
        frozen_predictor_directory=args.frozen_predictor_directory.resolve(),
        output_directory=args.output_directory.resolve(),
        n_jobs=args.n_jobs,
    )
    summary = {
        name: {
            "accuracy": report["window_level_diagnostics"]["accuracy"],
            "macro_f1": report["primary"]["mean_participant_macro_f1"],
            "sitting_recall": report["window_level_diagnostics"]["per_class_recall"]["sitting"],
            "standing_recall": report["window_level_diagnostics"]["per_class_recall"]["standing"],
        }
        for name, report in result["primary_reports"].items()
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
