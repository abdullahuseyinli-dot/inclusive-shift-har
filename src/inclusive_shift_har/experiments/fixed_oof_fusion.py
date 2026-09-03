"""Evaluate predeclared fixed pooling of aligned frozen source OOF predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.statistics import (
    paired_participant_comparison,
    participant_bootstrap_interval,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _record(path: Path) -> dict[str, Any]:
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    claimed = payload.get("record_sha256")
    unhashed = dict(payload)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise ValueError(f"frozen input record self-hash changed: {path}")
    if (
        payload.get("target_subject_or_window_records_loaded") is not False
        or payload.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("fixed fusion refuses target-informed inputs")
    return payload


def _arrays(record: dict[str, Any]) -> dict[str, NDArray[Any]]:
    reference = _mapping(record["predictions"], name="predictions")
    path = Path(str(reference["path"]))
    if sha256_file(path) != reference["sha256"]:
        raise ValueError("frozen prediction file hash changed")
    with np.load(path, allow_pickle=False) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def fixed_probability_pools(first: FloatArray, second: FloatArray) -> dict[str, FloatArray]:
    """Return fixed, symmetric probability pools with no fitted quantities."""

    if first.shape != second.shape or first.ndim != 2 or first.shape[1] != 3:
        raise ValueError("fixed fusion requires aligned [window,3] experts")
    if any(
        not np.isfinite(item).all()
        or np.any(item < 0)
        or not np.allclose(item.sum(axis=1), 1.0, atol=1e-6)
        for item in (first, second)
    ):
        raise ValueError("fixed fusion received invalid probabilities")
    arithmetic = 0.5 * (first + second)
    logarithmic = np.sqrt(np.clip(first, 1e-12, 1.0) * np.clip(second, 1e-12, 1.0))
    logarithmic /= logarithmic.sum(axis=1, keepdims=True)
    return {
        "equal_arithmetic_pool": arithmetic,
        "equal_logarithmic_pool": logarithmic,
    }


def run_fixed_oof_fusion(
    *,
    first_result_path: Path,
    second_result_path: Path,
    config_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    if output_directory.exists():
        raise FileExistsError(f"fixed-fusion output already exists: {output_directory}")
    config = _mapping(yaml.safe_load(config_path.read_text(encoding="utf-8")), name="config")
    if config.get("methods") != ["equal_arithmetic_pool", "equal_logarithmic_pool"]:
        raise ValueError("fixed-fusion method contract changed")
    first_record, second_record = _record(first_result_path), _record(second_result_path)
    first, second = _arrays(first_record), _arrays(second_record)
    for field in ("labels", "participant_ids", "window_ids"):
        if not np.array_equal(first[field], second[field]):
            raise ValueError(f"fixed fusion inputs are not aligned on {field}")
    first_probabilities = np.asarray(first["probabilities"], dtype=np.float64)
    second_probabilities = np.asarray(second["probabilities"], dtype=np.float64)
    probabilities = {
        "spectral_shape": first_probabilities,
        "rist_budgeted": second_probabilities,
        **fixed_probability_pools(first_probabilities, second_probabilities),
    }
    labels = np.asarray(first["labels"], dtype=np.int64)
    participants = np.asarray(first["participant_ids"], dtype=np.str_)
    windows = np.asarray(first["window_ids"], dtype=np.str_)
    reports = {
        name: classification_report(
            labels,
            values,
            participants.tolist(),
            class_names=("mobility", "sitting", "standing"),
        )
        for name, values in probabilities.items()
    }
    participant_values = {
        name: {
            str(row["participant_id"]): float(row["macro_f1"])
            for row in cast(list[dict[str, Any]], report["participants"])
        }
        for name, report in reports.items()
    }
    output_directory.mkdir(parents=True)
    prediction_path = output_directory / "predictions.npz"
    arrays: dict[str, Any] = {
        "labels": labels,
        "participant_ids": participants,
        "window_ids": windows,
        **{f"{name}_probabilities": value for name, value in probabilities.items()},
    }
    with prediction_path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fixed_frozen_oof_probability_fusion",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "config": {"path": config_path.as_posix(), "sha256": sha256_file(config_path)},
        "inputs": {
            "spectral_shape": {
                "path": first_result_path.as_posix(),
                "sha256": sha256_file(first_result_path),
            },
            "rist_budgeted": {
                "path": second_result_path.as_posix(),
                "sha256": sha256_file(second_result_path),
            },
        },
        "reports": reports,
        "participant_bootstrap": {
            name: participant_bootstrap_interval(values)
            for name, values in participant_values.items()
        },
        "paired_comparisons_vs_spectral_shape": {
            name: paired_participant_comparison(participant_values["spectral_shape"], values)
            for name, values in participant_values.items()
            if name != "spectral_shape"
        },
        "predictions": {"path": prediction_path.as_posix(), "sha256": sha256_file(prediction_path)},
        "selection": {
            "learned_or_tuned_weights": False,
            "outer_labels_used_to_choose_weights": False,
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
    parser.add_argument("--first-result", type=Path, required=True)
    parser.add_argument("--second-result", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_fixed_oof_fusion(
        first_result_path=args.first_result,
        second_result_path=args.second_result,
        config_path=args.config,
        output_directory=args.output_directory,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "primary": {name: report["primary"] for name, report in result["reports"].items()},
                "record_sha256": result["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
