"""Fixed, no-retuning integration of the compact gravity branch into HERA-v1.

HERA-v1 remains frozen. The compact branch was trained in the preceding bounded
follow-up, and this module tests one predeclared 25% probability contribution.
The fixed blend measures complementarity without selecting a weight from outcomes.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
CLASS_NAMES = ("mobility", "sitting", "standing")
ALPHA = 0.25


def _load(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _validate_probability(values: FloatArray, name: str) -> FloatArray:
    result = np.asarray(values, dtype=np.float64)
    if result.ndim != 2 or result.shape[1] != 3:
        raise ValueError(f"{name} must be [sample,3]")
    if not np.isfinite(result).all() or np.any(result < 0.0):
        raise ValueError(f"{name} must be finite and non-negative")
    if not np.allclose(result.sum(axis=1), 1.0, atol=1e-7, rtol=1e-7):
        raise ValueError(f"{name} rows must sum to one")
    return result


def _bootstrap(candidate: dict[str, float], reference: dict[str, float]) -> dict[str, Any]:
    people = sorted(candidate)
    if people != sorted(reference) or len(people) != 10:
        raise ValueError("HERA integration requires the ten fixed source participants")
    left = np.asarray([candidate[item] for item in people], dtype=np.float64)
    right = np.asarray([reference[item] for item in people], dtype=np.float64)
    rng = np.random.default_rng(1729)
    indices = rng.integers(0, len(people), size=(10_000, len(people)))
    values = (left[indices] - right[indices]).mean(axis=1)
    return {
        "mean_difference": float((left - right).mean()),
        "lower": float(np.quantile(values, 0.025)),
        "upper": float(np.quantile(values, 0.975)),
        "resamples": 10_000,
        "seed": 1729,
    }


def run_integration(
    *, hera_predictions: Path, compact_predictions: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"create-only output already exists: {output}")
    hera = _load(hera_predictions)
    compact = _load(compact_predictions)
    required_hera = {"labels", "participant_ids", "window_ids", "hera_v1_strict_probabilities"}
    required_compact = {
        "labels",
        "participant_ids",
        "window_ids",
        "gravity_invariant_probabilities",
    }
    if not required_hera.issubset(hera) or not required_compact.issubset(compact):
        raise ValueError("integration inputs are missing required arrays")
    for key in ("labels", "participant_ids", "window_ids"):
        if not np.array_equal(hera[key], compact[key]):
            raise ValueError(f"integration input {key} differs; ID alignment is mandatory")
    labels = np.asarray(hera["labels"], dtype=np.int64)
    participants = np.asarray(hera["participant_ids"], dtype=np.str_)
    windows = np.asarray(hera["window_ids"], dtype=np.str_)
    hera_probability = _validate_probability(
        np.asarray(hera["hera_v1_strict_probabilities"], dtype=np.float64), "HERA probabilities"
    )
    compact_probability = _validate_probability(
        np.asarray(compact["gravity_invariant_probabilities"], dtype=np.float64),
        "compact probabilities",
    )
    integrated = np.asarray((1.0 - ALPHA) * hera_probability + ALPHA * compact_probability)
    started = time.perf_counter()
    hera_report = classification_report(
        labels, hera_probability, participants.tolist(), class_names=CLASS_NAMES
    )
    compact_report = classification_report(
        labels, compact_probability, participants.tolist(), class_names=CLASS_NAMES
    )
    integrated_report = classification_report(
        labels, integrated, participants.tolist(), class_names=CLASS_NAMES
    )
    hera_people = {
        str(row["participant_id"]): float(row["macro_f1"]) for row in hera_report["participants"]
    }
    integrated_people = {
        str(row["participant_id"]): float(row["macro_f1"])
        for row in integrated_report["participants"]
    }
    differences = {
        person: integrated_people[person] - hera_people[person] for person in hera_people
    }
    output.mkdir(parents=True)
    np.savez_compressed(
        output / "predictions.npz",
        labels=labels,
        participant_ids=participants,
        window_ids=windows,
        hera_v1_strict_probabilities=hera_probability,
        compact_probabilities=compact_probability,
        integrated_probabilities=integrated,
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "hera_compact_evidence_integration_result",
        "experiment_id": "hera-compact-evidence-integration-v1",
        "status": "complete_exploratory_source_development",
        "evidence_status": "reused_source_development_not_independent",
        "human_performance_claim": False,
        "configuration": {
            "reference": "frozen HERA-v1 strict",
            "compact_branch": "gravity-invariant-followup-v1",
            "compact_probability_weight": ALPHA,
            "hera_probability_weight": 1.0 - ALPHA,
            "weight_selection": "fixed before integration outcomes; no tuning",
            "new_hera_fits": 0,
            "upstream_compact_fits": 5,
        },
        "hera_v1_strict": hera_report,
        "compact_branch": compact_report,
        "integrated": integrated_report,
        "paired_participant_macro_f1_difference": differences,
        "paired_bootstrap_vs_hera": _bootstrap(integrated_people, hera_people),
        "participant_wins": int(sum(value > 0.0 for value in differences.values())),
        "participant_harms": int(sum(value < 0.0 for value in differences.values())),
        "participant_ties": int(sum(value == 0.0 for value in differences.values())),
        "runtime_seconds": time.perf_counter() - started,
        "input_hashes": {
            "hera_predictions": sha256_file(hera_predictions),
            "compact_predictions": sha256_file(compact_predictions),
        },
        "software": {"numpy": importlib.metadata.version("numpy")},
        "interpretation": "A fixed compact-evidence branch tests complementarity with HERA without changing HERA training or routing. Any gain remains reused-source exploratory evidence.",
    }
    result["record_sha256"] = canonical_json_sha256(result)
    with (output / "result.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    manifest = {
        "schema_version": "1.0.0",
        "record_kind": "hera_compact_evidence_integration_manifest",
        "artifacts": [
            {
                "path": "result.json",
                "sha256": sha256_file(output / "result.json"),
                "size_bytes": (output / "result.json").stat().st_size,
            },
            {
                "path": "predictions.npz",
                "sha256": sha256_file(output / "predictions.npz"),
                "size_bytes": (output / "predictions.npz").stat().st_size,
            },
        ],
    }
    manifest["record_sha256"] = canonical_json_sha256(manifest)
    with (output / "manifest.json").open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hera-predictions", type=Path, required=True)
    parser.add_argument("--compact-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_integration(
        hera_predictions=args.hera_predictions.resolve(),
        compact_predictions=args.compact_predictions.resolve(),
        output=args.output.resolve(),
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "mean_difference": result["paired_bootstrap_vs_hera"]["mean_difference"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
