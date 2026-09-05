"""Participant-exclusive HAR-PMD native-interface mobility-mode stress evaluation."""

from __future__ import annotations

import argparse
import json
import platform
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    load_har_pmd_native,
    participant_fold_assignment,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _paired_bootstrap,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import (
    ClassicalConfig,
    fit_classical_model,
    predict_classical_probabilities,
)

FloatArray = NDArray[np.float64]

_SIX_CHANNEL_NAMES = (
    "lin_acc_x",
    "lin_acc_y",
    "lin_acc_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
)
_NINE_CHANNEL_NAMES = (*_SIX_CHANNEL_NAMES, "gravity_x", "gravity_y", "gravity_z")
_METHODS = (
    "RandomForest-6ch",
    "RandomForest-N9",
    "XGBoost-6ch",
    "XGBoost-N9",
)


def _report(
    labels: NDArray[np.int64],
    probabilities: FloatArray,
    participants: NDArray[np.str_],
    *,
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    result = classification_report(
        labels,
        probabilities,
        participants.tolist(),
        class_names=class_names,
    )
    values = sorted(float(item["macro_f1"]) for item in result["participants"])
    count = max(1, int(np.ceil(0.30 * len(values))))
    result["primary"]["bottom_30_percent_participant_macro_f1"] = float(np.mean(values[:count]))
    return result


def _evaluate_seed(data: ExternalHARWindows, *, seed: int) -> dict[str, FloatArray]:
    participants = data.participant_ids
    assignment = participant_fold_assignment(
        np.unique(participants).tolist(), fold_count=5, seed=seed
    )
    probabilities = {
        method: np.full((data.labels.size, len(data.class_names)), np.nan, dtype=np.float64)
        for method in _METHODS
    }
    six = data.signals
    nine = data.nine_channel_signals
    for fold in range(5):
        evaluation_ids = {
            participant for participant, assigned in assignment.items() if assigned == fold
        }
        evaluation = np.isin(participants, sorted(evaluation_ids))
        training = ~evaluation
        expected = set(range(len(data.class_names)))
        if set(np.unique(data.labels[training]).tolist()) != expected:
            raise ValueError("HAR-PMD outer training fold lacks a declared class")
        for model_name, prefix in (
            ("random_forest", "RandomForest"),
            ("xgboost", "XGBoost"),
        ):
            for windows, channel_names, suffix in (
                (six, _SIX_CHANNEL_NAMES, "6ch"),
                (nine, _NINE_CHANNEL_NAMES, "N9"),
            ):
                fitted = fit_classical_model(
                    windows[training],
                    data.labels[training],
                    participants[training].tolist(),
                    config=ClassicalConfig(
                        model_name=model_name,
                        num_classes=len(data.class_names),
                        seed=seed + 101 * fold,
                    ),
                    channel_names=channel_names,
                    lineage={
                        "dataset_id": data.dataset_id,
                        "endpoint": "native-interface mobility-mode stress",
                        "participant_exclusive": True,
                        "outer_fold": fold,
                    },
                )
                _, fold_probability = predict_classical_probabilities(
                    fitted,
                    windows[evaluation],
                )
                probabilities[f"{prefix}-{suffix}"][evaluation] = fold_probability
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise ValueError("HAR-PMD outer predictions are incomplete")
    return probabilities


def evaluate_har_pmd_stress(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    data.validate()
    if data.dataset_id != "har_pmd_v1" or data.channel_lane != "native-gravity-9ch":
        raise ValueError("native stress runner requires HAR-PMD's native lane")
    if np.unique(data.participant_ids).size < 12:
        raise ValueError("HAR-PMD stress evaluation requires at least 12 participants")
    per_seed = {seed: _evaluate_seed(data, seed=seed) for seed in seeds}
    ensemble = {
        method: np.mean(np.stack([per_seed[seed][method] for seed in seeds], axis=0), axis=0)
        for method in _METHODS
    }
    reports = {
        method: _report(
            data.labels,
            probability,
            data.participant_ids,
            class_names=data.class_names,
        )
        for method, probability in ensemble.items()
    }
    environment_reports: dict[str, dict[str, Any]] = {}
    for environment in ("indoor", "outdoor"):
        selected = np.char.endswith(data.session_ids, f":{environment}")
        environment_reports[environment] = {
            method: _report(
                data.labels[selected],
                probability[selected],
                data.participant_ids[selected],
                class_names=data.class_names,
            )
            for method, probability in ensemble.items()
        }
    comparisons = {
        "RandomForest-N9_vs_6ch": _paired_bootstrap(
            reports["RandomForest-N9"], reports["RandomForest-6ch"], seed=20261201
        ),
        "XGBoost-N9_vs_6ch": _paired_bootstrap(
            reports["XGBoost-N9"], reports["XGBoost-6ch"], seed=20261202
        ),
    }
    result = {
        "schema_version": "1.0.0",
        "experiment_id": "har-pmd-native-interface-stress-v1",
        "evidence_status": "EXTERNAL_DEVELOPMENT_STRESS_NOT_CONFIRMATORY",
        "dataset": data.summary(),
        "seeds": list(seeds),
        "reports": reports,
        "environment_reports": environment_reports,
        "paired_native_gravity_comparisons": comparisons,
        "endpoint_restriction": (
            "The source 'still' class merges sitting and standing. This five-class "
            "mobility-mode endpoint cannot validate the three-class posture invention."
        ),
        "claim_policy": {
            "pooled_with_three_class_results": False,
            "proposed_method_superiority_test": False,
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
        },
    }
    return result, ensemble


def run_and_write(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
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
        result, predictions = evaluate_har_pmd_stress(data, seeds=seeds)
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
        result["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
        }
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
    parser.add_argument("--participant-limit", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--with-neural", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    data = load_har_pmd_native(participant_limit=args.participant_limit)
    result = run_and_write(
        data=data,
        output_directory=args.output_directory.resolve(),
        repository_root=args.repository_root.resolve(),
        seeds=tuple(args.seeds),
    )
    neural_summary = None
    if args.with_neural:
        from inclusive_shift_har.experiments.cross_dataset_neural import (
            run_and_write_neural,
        )

        neural = run_and_write_neural(
            data=data,
            signals=data.nine_channel_signals,
            class_names=data.class_names,
            method_suffix="N9",
            experiment_id="har-pmd-native-interface-neural-controls-v1",
            output_directory=(args.output_directory.resolve() / "neural"),
            repository_root=args.repository_root.resolve(),
            seeds=tuple(args.seeds),
            epochs=args.neural_epochs,
        )
        neural_summary = {
            name: report["primary"]["mean_participant_macro_f1"]
            for name, report in neural["reports"].items()
        }
    print(
        json.dumps(
            {
                "classical": {
                    name: report["primary"]["mean_participant_macro_f1"]
                    for name, report in result["reports"].items()
                },
                "neural": neural_summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
