"""Ordered-session Sole-HARmony evaluation with causal and shuffled-time controls."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import ExternalHARWindows, load_sole_harmony
from inclusive_shift_har.experiments.cage_har import _report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _paired_bootstrap,
    _source_input_manifest,
    _write_json_create_only,
    evaluate_external_development,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.hera_ctgr_v2 import accumulate_bout_intervention_evidence

FloatArray = NDArray[np.float64]

_WIDTHS = (3, 5, 7)
_SHUFFLE_REPLICATES = 100


def _temporal_block_ids(data: ExternalHARWindows) -> NDArray[np.str_]:
    identifiers = np.asarray(
        [value.rsplit("/window-", 1)[0] for value in data.window_ids.tolist()],
        dtype=np.str_,
    )
    if np.any(np.char.str_len(identifiers) == 0):
        raise ValueError("Sole-HARmony window identifiers lack temporal block provenance")
    return identifiers


def _causal_probability(
    probabilities: FloatArray,
    bout_ids: NDArray[np.str_],
    *,
    width: int,
) -> FloatArray:
    smoothed = np.column_stack(
        [
            accumulate_bout_intervention_evidence(
                probabilities[:, class_index], bout_ids, width=width
            )
            for class_index in range(probabilities.shape[1])
        ]
    )
    return np.asarray(smoothed / smoothed.sum(axis=1, keepdims=True), dtype=np.float64)


def _bout_slices(bout_ids: NDArray[np.str_]) -> tuple[NDArray[np.int64], ...]:
    if bout_ids.ndim != 1 or bout_ids.size == 0:
        raise ValueError("Sole-HARmony bout identifiers must be a non-empty vector")
    boundaries = np.flatnonzero(np.concatenate(([True], bout_ids[1:] != bout_ids[:-1])))
    stops = np.concatenate((boundaries[1:], np.array([bout_ids.size], dtype=np.int64)))
    slices = tuple(
        np.arange(start, stop, dtype=np.int64)
        for start, stop in zip(boundaries, stops, strict=True)
    )
    if len({str(bout_ids[index[0]]) for index in slices}) != len(slices):
        raise ValueError("a Sole-HARmony bout appears in multiple non-contiguous blocks")
    return slices


def _shuffled_time_probability(
    probabilities: FloatArray,
    bout_ids: NDArray[np.str_],
    *,
    width: int,
    seed: int,
) -> FloatArray:
    generator = np.random.default_rng(seed)
    result = np.empty_like(probabilities)
    for indices in _bout_slices(bout_ids):
        order = generator.permutation(indices.size)
        shuffled = probabilities[indices][order]
        shuffled_ids = np.full(indices.size, str(bout_ids[indices[0]]), dtype=np.str_)
        smoothed = _causal_probability(shuffled, shuffled_ids, width=width)
        result[indices[order]] = smoothed
    return result


def evaluate_temporal_lane(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
    repository_root: Path,
    n_jobs: int,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    data.validate()
    if data.dataset_id != "sole_harmony_v1":
        raise ValueError("temporal lane requires Sole-HARmony")
    if np.unique(data.participant_ids).size < 12:
        raise ValueError("temporal protocol requires at least 12 participants")
    if (
        min(
            np.unique(data.session_ids[data.participant_ids == participant]).size
            for participant in np.unique(data.participant_ids)
        )
        < 2
    ):
        raise ValueError("temporal protocol requires at least two sessions per participant")
    base_result, base_predictions = evaluate_external_development(
        data,
        seeds=seeds,
        repository_root=repository_root,
        n_jobs=n_jobs,
        include_classical=False,
    )
    raw = base_predictions["HERA-DG-v2-full"]
    temporal_blocks = _temporal_block_ids(data)
    outputs: dict[str, FloatArray] = {"HERA-DG-v2-full-unsmoothed": raw}
    reports: dict[str, dict[str, Any]] = {
        "HERA-DG-v2-full-unsmoothed": _report(data.labels, raw, data.participant_ids)
    }
    shuffled_controls: dict[str, Any] = {}
    for width in _WIDTHS:
        name = f"HERA-DG-v2-full-causal-bout-w{width}"
        causal = _causal_probability(raw, temporal_blocks, width=width)
        outputs[name] = causal
        reports[name] = _report(data.labels, causal, data.participant_ids)
        shuffled_scores: list[float] = []
        representative: FloatArray | None = None
        for replicate in range(_SHUFFLE_REPLICATES):
            shuffled = _shuffled_time_probability(
                raw,
                temporal_blocks,
                width=width,
                seed=20260904 + 1_000 * width + replicate,
            )
            if representative is None:
                representative = shuffled
            shuffled_report = _report(data.labels, shuffled, data.participant_ids)
            shuffled_scores.append(float(shuffled_report["primary"]["mean_participant_macro_f1"]))
        if representative is None:
            raise AssertionError("shuffled-time control produced no replicate")
        representative_name = f"shuffled-time-control-w{width}-rep000"
        outputs[representative_name] = representative
        causal_score = float(reports[name]["primary"]["mean_participant_macro_f1"])
        values = np.asarray(shuffled_scores, dtype=np.float64)
        shuffled_controls[f"width_{width}"] = {
            "replicate_count": _SHUFFLE_REPLICATES,
            "causal_mean_participant_macro_f1": causal_score,
            "shuffled_mean": float(values.mean()),
            "shuffled_median": float(np.median(values)),
            "shuffled_95_percent_interval": [
                float(np.quantile(values, 0.025)),
                float(np.quantile(values, 0.975)),
            ],
            "causal_minus_shuffled_median": causal_score - float(np.median(values)),
            "permutation_tail_fraction_shuffled_at_least_causal": float(
                (1 + np.sum(values >= causal_score)) / (1 + values.size)
            ),
            "interpretation": (
                "negative-control diagnostic only; shuffle replicates are not independent "
                "participants and this is not a confirmatory p-value"
            ),
        }
    comparisons = {
        name: _paired_bootstrap(
            report,
            reports["HERA-DG-v2-full-unsmoothed"],
            seed=20261300 + index,
        )
        for index, (name, report) in enumerate(reports.items())
        if name != "HERA-DG-v2-full-unsmoothed"
    }
    result = {
        "schema_version": "1.0.0",
        "experiment_id": "sole-harmony-causal-bout-v1",
        "evidence_status": "EXTERNAL_TEMPORAL_DEVELOPMENT_NOT_CONFIRMATORY",
        "dataset": data.summary(),
        "seeds": list(seeds),
        "base_nested_result": base_result,
        "temporal_reports": reports,
        "paired_participant_bootstrap_vs_unsmoothed": comparisons,
        "shuffled_time_negative_controls": shuffled_controls,
        "temporal_contract": {
            "widths_reported_without_target_selection": list(_WIDTHS),
            "window_order": "provider session, camera interval, chronological window",
            "state_reset": "every camera-annotated bout and timestamp-contiguous segment",
            "future_window_access": False,
            "cross_bout_smoothing": False,
            "best_width_claim_allowed": False,
        },
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "shuffled_control_is_independent_inference": False,
        },
    }
    return result, outputs


def run_and_write(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    n_jobs: int,
) -> dict[str, Any]:
    git_at_launch = _git_state(repository_root)
    source_input_manifest = _source_input_manifest(repository_root)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    _write_json_create_only(
        output_directory / "data_audit.json",
        {
            "schema_version": "1.0.0",
            "created_at": started,
            "dataset": data.summary(),
            "source_receipts": [receipt.to_dict() for receipt in data.receipts],
            "raw_local_mirror": False,
            "source_input_manifest": source_input_manifest,
            "git_at_launch": git_at_launch,
        },
    )
    try:
        result, predictions = evaluate_temporal_lane(
            data, seeds=seeds, repository_root=repository_root, n_jobs=n_jobs
        )
        prediction_path = output_directory / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            labels=data.labels,
            participant_ids=data.participant_ids,
            session_ids=data.session_ids,
            trial_ids=data.trial_ids,
            window_ids=data.window_ids,
            **{  # type: ignore[arg-type]
                f"probability__{name}": value for name, value in predictions.items()
            },
        )
        result["started_at"] = started
        result["created_at"] = datetime.now(UTC).isoformat()
        result["git"] = _git_state(repository_root)
        result["git_at_launch"] = git_at_launch
        result["source_input_manifest"] = source_input_manifest
        result["inputs"] = {
            name: {"path": path, "sha256": sha256_file(repository_root / path)}
            for name, path in {
                "portfolio_config": "configs/datasets/external_har_portfolio_v1.yaml",
                "experiment_config": "configs/experiments/cross_dataset_har_rnd_v1.yaml",
            }.items()
        }
        result["prediction_artifact"] = {
            "path": prediction_path.name,
            "sha256": sha256_file(prediction_path),
        }
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output_directory / "result.json", result)
        return result
    except Exception as exc:
        _write_json_create_only(
            output_directory / "failure.json",
            {
                "schema_version": "1.0.0",
                "status": "FAILED_PRESERVED",
                "started_at": started,
                "failed_at": datetime.now(UTC).isoformat(),
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
                "traceback": traceback.format_exc(),
                "git": _git_state(repository_root),
                "git_at_launch": git_at_launch,
            },
        )
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--participants", type=int, default=12)
    parser.add_argument("--sessions-per-participant", type=int, default=2)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--audit-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output_directory = args.output_directory.resolve()
    if args.audit_only:
        git_at_launch = _git_state(args.repository_root.resolve())
        source_input_manifest = _source_input_manifest(args.repository_root.resolve())
        output_directory.mkdir(parents=True, exist_ok=False)
        started = datetime.now(UTC).isoformat()
        try:
            data = load_sole_harmony(
                participant_limit=args.participants,
                sessions_per_participant=args.sessions_per_participant,
            )
            audit = {
                "schema_version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "started_at": started,
                "status": "DATA_AUDIT_COMPLETE_NO_MODELS_RUN",
                "dataset": data.summary(),
                "source_receipts": [receipt.to_dict() for receipt in data.receipts],
                "raw_local_mirror": False,
                "git": _git_state(args.repository_root.resolve()),
                "git_at_launch": git_at_launch,
                "source_input_manifest": source_input_manifest,
            }
            _write_json_create_only(output_directory / "data_audit.json", audit)
            print(json.dumps(data.summary(), indent=2, sort_keys=True))
            return 0
        except Exception as exc:
            _write_json_create_only(
                output_directory / "failure.json",
                {
                    "schema_version": "1.0.0",
                    "status": "FAILED_PRESERVED",
                    "started_at": started,
                    "failed_at": datetime.now(UTC).isoformat(),
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                    "traceback": traceback.format_exc(),
                    "git": _git_state(args.repository_root.resolve()),
                    "git_at_launch": git_at_launch,
                },
            )
            raise
    data = load_sole_harmony(
        participant_limit=args.participants,
        sessions_per_participant=args.sessions_per_participant,
    )
    result = run_and_write(
        data=data,
        output_directory=output_directory,
        repository_root=args.repository_root.resolve(),
        seeds=tuple(args.seeds),
        n_jobs=args.n_jobs,
    )
    print(
        json.dumps(
            {
                name: report["primary"]["mean_participant_macro_f1"]
                for name, report in result["temporal_reports"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
