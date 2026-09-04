"""Replay frozen RMRP with explicit-mask reconstruction on a new corruption suite."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.provenance_corruptions import (
    ProvenanceCorruptionSpec,
    apply_provenance_corruption,
    interpolate_from_validity_mask,
)
from inclusive_shift_har.experiments.fuse_reframe_nested import (
    _lower_fraction_mean,
    _mask_for_subjects,
)
from inclusive_shift_har.experiments.geometric_pyramid_corruptions import (
    train_only_corruption_scales,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import _probabilities
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _materialize_source,
    _read_hashed_record,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.geometric_spectral_pyramid import (
    extract_geometric_spectral_pyramid_features,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_signal_views,
)

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

_CLASS_NAMES = ("mobility", "sitting", "standing")
_METHODS = ("numeric_zero_or_corrupted_rmrp", "explicit_mask_reconstruction_rmrp")
_EXPECTED_CASES = (
    ("clean", 0.0),
    ("axis_dropout", 1 / 3),
    ("axis_dropout", 2 / 3),
    ("modality_dropout", 1.0),
    ("contiguous_gap", 0.125),
    ("contiguous_gap", 0.25),
    ("stuck_at", 0.25),
    ("stuck_at", 0.50),
    ("saturation", 0.25),
    ("saturation", 0.50),
    ("bias_drift", 0.20),
    ("bias_drift", 0.50),
    ("scale_drift", 0.20),
    ("scale_drift", 0.50),
    ("gaussian_noise", 0.10),
    ("gaussian_noise", 0.30),
    ("constrained_rotation", 0.25),
    ("constrained_rotation", 0.50),
)


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_config(path: Path) -> tuple[dict[str, Any], tuple[ProvenanceCorruptionSpec, ...]]:
    config = _mapping(yaml.safe_load(path.read_text(encoding="utf-8")), name="config")
    raw_suite = config.get("suite")
    if not isinstance(raw_suite, list):
        raise ValueError("provenance-mask config requires a suite")
    observed = tuple(
        (str(_mapping(item, name="suite item")["name"]), float(item["severity"]))
        for item in raw_suite
    )
    if observed != _EXPECTED_CASES:
        raise ValueError("provenance-mask corruption suite changed")
    if int(config.get("seed", -1)) != 20260904:
        raise ValueError("provenance-mask corruption seed changed")
    if config.get("selection_use_prohibited") is not True:
        raise ValueError("provenance-mask results must be prohibited from selection")
    policy = _mapping(config.get("claim_policy"), name="claim policy")
    if (
        policy.get("target_data_allowed") is not False
        or policy.get("clean_model_advancement_claim_allowed") is not False
    ):
        raise ValueError("provenance-mask claim boundary changed")
    specs = tuple(
        ProvenanceCorruptionSpec(cast(Any, name), severity) for name, severity in observed
    )
    return config, specs


def _case_id(spec: ProvenanceCorruptionSpec) -> str:
    return f"{spec.name}__{spec.severity:.6f}".replace(".", "p")


def _add_bottom_tail(report: dict[str, Any]) -> None:
    values = [
        float(item["macro_f1"]) for item in cast(list[dict[str, Any]], report["participants"])
    ]
    report["primary"]["bottom_30_percent_participant_macro_f1"] = _lower_fraction_mean(values)


def _features(signals: FloatArray, *, sampling_rate_hz: float) -> FloatArray:
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=sampling_rate_hz)[
        "denoised"
    ]
    return extract_geometric_spectral_pyramid_features(denoised, sampling_rate_hz=sampling_rate_hz)


def run_provenance_mask_robustness(
    *,
    nested_result_path: Path,
    source_manifest_path: Path,
    raw_csv_path: Path,
    config_path: Path,
    output_directory: Path,
    code_commit: str,
) -> dict[str, Any]:
    """Evaluate zero/corrupted input and explicit-mask reconstruction on identical cases."""

    if output_directory.exists():
        raise FileExistsError(f"provenance-mask output already exists: {output_directory}")
    config, specs = _load_config(config_path)
    nested = _read_hashed_record(nested_result_path)
    if (
        nested.get("status") != "complete_target_sealed"
        or nested.get("target_subject_or_window_records_loaded") is not False
        or nested.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("provenance-mask runner requires a target-sealed source result")
    if nested["source_manifest"]["sha256"] != sha256_file(source_manifest_path):
        raise ValueError("provenance-mask source manifest does not match frozen RMRP")
    manifest, signals, labels, participants, windows = _materialize_source(
        source_manifest_path=source_manifest_path, raw_csv_path=raw_csv_path
    )
    sampling_rate = float(config["sampling_rate_hz"])
    seed = int(config["seed"])
    clipping_quantile = float(config["saturation_training_quantile"])
    outer_by_id = {
        str(item["outer_fold_id"]): item
        for item in cast(list[dict[str, Any]], manifest["source_nested_cv"])
    }
    accumulator: dict[str, dict[str, dict[str, list[NDArray[Any]]]]] = {
        _case_id(spec): {
            method: {"labels": [], "probabilities": [], "participants": [], "windows": []}
            for method in _METHODS
        }
        for spec in specs
    }
    affected: dict[str, list[tuple[float, float, int]]] = {_case_id(spec): [] for spec in specs}
    fold_records: list[dict[str, Any]] = []
    output_directory.mkdir(parents=True)
    for fold_index, frozen_reference in enumerate(cast(list[dict[str, Any]], nested["folds"])):
        fold_id = str(frozen_reference["outer_fold_id"])
        outer = outer_by_id[fold_id]
        evaluation_mask = _mask_for_subjects(
            participants, [str(item) for item in outer["outer_test_subjects"]]
        )
        training_mask = ~evaluation_mask
        fold_path = Path(str(frozen_reference["path"]))
        if sha256_file(fold_path) != str(frozen_reference["sha256"]):
            raise ValueError(f"frozen RMRP fold hash changed: {fold_id}")
        fold = _read_hashed_record(fold_path)
        model_record = _mapping(fold.get("model"), name=f"model {fold_id}")
        model_path = Path(str(model_record["path"]))
        if sha256_file(model_path) != str(model_record["sha256"]):
            raise ValueError(f"frozen RMRP model hash changed: {fold_id}")
        with model_path.open("rb") as model_stream:
            bundle = _mapping(pickle.load(model_stream), name=f"model bundle {fold_id}")
        if bundle.get("feature_view") != "denoised":
            raise ValueError("mask reconstruction runner requires frozen denoised RMRP folds")
        estimator = bundle["estimator"]
        saved_record = _mapping(fold.get("predictions"), name=f"predictions {fold_id}")
        saved_path = Path(str(saved_record["path"]))
        if sha256_file(saved_path) != str(saved_record["sha256"]):
            raise ValueError(f"frozen RMRP predictions changed: {fold_id}")
        with np.load(saved_path, allow_pickle=False) as saved:
            saved_probability = np.asarray(saved["probabilities"], dtype=np.float64)
            np.testing.assert_array_equal(saved["labels"], labels[evaluation_mask])
            np.testing.assert_array_equal(saved["participant_ids"], participants[evaluation_mask])
            np.testing.assert_array_equal(saved["window_ids"], windows[evaluation_mask])
        channel_scale, saturation_threshold = train_only_corruption_scales(
            signals[training_mask], clipping_quantile=clipping_quantile
        )
        training_median = np.median(signals[training_mask], axis=(0, 1))
        fold_cases: list[dict[str, Any]] = []
        for case_index, spec in enumerate(specs):
            corrupted = apply_provenance_corruption(
                signals[evaluation_mask],
                spec,
                channel_scale=channel_scale,
                saturation_threshold=saturation_threshold,
                seed=seed + fold_index * 1_000_003 + case_index * 100_003,
            )
            reconstructed = interpolate_from_validity_mask(
                corrupted.signals,
                corrupted.validity_mask,
                training_channel_median=training_median,
            )
            probabilities = {
                "numeric_zero_or_corrupted_rmrp": _probabilities(
                    estimator, _features(corrupted.signals, sampling_rate_hz=sampling_rate)
                ),
                "explicit_mask_reconstruction_rmrp": _probabilities(
                    estimator, _features(reconstructed, sampling_rate_hz=sampling_rate)
                ),
            }
            if spec.name == "clean":
                for probability in probabilities.values():
                    np.testing.assert_allclose(
                        probability, saved_probability, rtol=1e-12, atol=1e-15
                    )
            fold_case_reports: dict[str, dict[str, Any]] = {}
            case_id = _case_id(spec)
            for method, probability in probabilities.items():
                report = classification_report(
                    labels[evaluation_mask],
                    probability,
                    participants[evaluation_mask].tolist(),
                    class_names=_CLASS_NAMES,
                )
                _add_bottom_tail(report)
                fold_case_reports[method] = report
                accumulator[case_id][method]["labels"].append(labels[evaluation_mask])
                accumulator[case_id][method]["probabilities"].append(probability)
                accumulator[case_id][method]["participants"].append(participants[evaluation_mask])
                accumulator[case_id][method]["windows"].append(windows[evaluation_mask])
            count = int(evaluation_mask.sum())
            affected[case_id].append(
                (corrupted.affected_fraction, corrupted.invalid_fraction, count)
            )
            fold_cases.append(
                {
                    "case_id": case_id,
                    "corruption": spec.name,
                    "severity": spec.severity,
                    "affected_fraction": corrupted.affected_fraction,
                    "invalid_fraction": corrupted.invalid_fraction,
                    "validity_mask_sha256": canonical_json_sha256(
                        corrupted.validity_mask.astype(np.uint8).tolist()
                    ),
                    "reports": fold_case_reports,
                }
            )
        fold_records.append(
            {
                "outer_fold_id": fold_id,
                "frozen_model_sha256": model_record["sha256"],
                "training_scale_participants": sorted(set(participants[training_mask].tolist())),
                "cases": fold_cases,
            }
        )
    cases: list[dict[str, Any]] = []
    prediction_payload: dict[str, NDArray[Any]] = {}
    for spec in specs:
        case_id = _case_id(spec)
        aggregate_case_reports: dict[str, dict[str, Any]] = {}
        for method in _METHODS:
            block = accumulator[case_id][method]
            aggregate_labels = np.concatenate(block["labels"])
            probability = np.concatenate(block["probabilities"])
            aggregate_participants = np.concatenate(block["participants"])
            aggregate_windows = np.concatenate(block["windows"])
            if len(set(aggregate_windows.tolist())) != windows.size or set(
                aggregate_windows.tolist()
            ) != set(windows.tolist()):
                raise ValueError(f"provenance case does not cover source exactly once: {case_id}")
            report = classification_report(
                aggregate_labels,
                probability,
                aggregate_participants.tolist(),
                class_names=_CLASS_NAMES,
            )
            _add_bottom_tail(report)
            aggregate_case_reports[method] = report
            prediction_payload[f"{case_id}__{method}__probabilities"] = probability
        rows = affected[case_id]
        counts = np.asarray([item[2] for item in rows], dtype=np.float64)
        cases.append(
            {
                "case_id": case_id,
                "corruption": spec.name,
                "severity": spec.severity,
                "affected_fraction": float(np.average([item[0] for item in rows], weights=counts)),
                "invalid_fraction": float(np.average([item[1] for item in rows], weights=counts)),
                "reports": aggregate_case_reports,
            }
        )
    clean = next(item for item in cases if item["corruption"] == "clean")
    frozen_mean = float(nested["aggregate_report"]["primary"]["mean_participant_macro_f1"])
    for method in _METHODS:
        if clean["reports"][method]["primary"]["mean_participant_macro_f1"] != frozen_mean:
            raise ValueError("clean mask-aware replay does not match frozen RMRP")
    prediction_payload.update(
        {"labels": labels, "participant_ids": participants, "window_ids": windows}
    )
    prediction_path = output_directory / "all_provenance_corruption_predictions.npz"
    with prediction_path.open("xb") as prediction_stream:
        np.savez_compressed(prediction_stream, **cast(dict[str, Any], prediction_payload))
    corrupted_cases = [item for item in cases if item["corruption"] != "clean"]
    missing_cases = [item for item in corrupted_cases if item["invalid_fraction"] > 0]

    def mean_score(rows: list[dict[str, Any]], method: str) -> float:
        return float(
            np.mean(
                [item["reports"][method]["primary"]["mean_participant_macro_f1"] for item in rows]
            )
        )

    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "provenance_mask_rmrp_source_robustness_diagnostic",
        "status": "complete_target_sealed",
        "evidence_status": "new_seed_source_corruption_diagnostic_not_clean_advancement",
        "code_commit": code_commit,
        "selection_use_prohibited": True,
        "nested_result": {
            "path": nested_result_path.as_posix(),
            "sha256": sha256_file(nested_result_path),
            "record_sha256": nested["record_sha256"],
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
        },
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "folds": fold_records,
        "suite": {
            "case_count": len(cases),
            "clean_mean_participant_macro_f1": frozen_mean,
            "mean_all_corrupted": {
                method: mean_score(corrupted_cases, method) for method in _METHODS
            },
            "mean_explicitly_invalid_cases": {
                method: mean_score(missing_cases, method) for method in _METHODS
            },
            "cases": cases,
        },
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "numeric_zeros_used_to_infer_missingness": False,
        "same_explicit_mask_provided_to_all_comparators": True,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "clean_model_advancement_claim_allowed": False,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output_directory / "result.json").open("xb") as result_stream:
        result_stream.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8"))
        result_stream.write(b"\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nested-result", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_provenance_mask_robustness(
        nested_result_path=args.nested_result,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        config_path=args.config,
        output_directory=args.output_directory,
        code_commit=args.code_commit,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "clean_mean": result["suite"]["clean_mean_participant_macro_f1"],
                "mean_all_corrupted": result["suite"]["mean_all_corrupted"],
                "mean_explicitly_invalid_cases": result["suite"]["mean_explicitly_invalid_cases"],
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
