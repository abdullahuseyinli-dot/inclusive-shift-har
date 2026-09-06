"""Participant-exclusive HAR-PMD native-interface mobility-mode stress evaluation."""

from __future__ import annotations

import argparse
import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.artifacts.research_provenance import (
    _external_evidence_status,
    _publication_artifact_contract,
    _publication_launch_context_binding,
    _resolve_publication_launch_context,
    _runtime_environment,
    _source_manifest_commit_errors,
    _write_launch_failure_envelope,
    _write_self_hashed_json_create_only,
)
from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    load_har_pmd_native,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _paired_bootstrap,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
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


def _evaluate_seed(
    data: ExternalHARWindows, *, seed: int
) -> tuple[dict[str, FloatArray], list[dict[str, Any]]]:
    if data.participant_partition_plan is None:
        raise PermissionError("HAR-PMD evaluation requires a pre-window participant plan")
    partition_plan = data.participant_partition_plan
    participants = data.participant_ids
    assignment = partition_plan.resolve(
        partition_plan.participant_roster,
        fold_count=5,
        seed=seed,
        role="outer",
    )
    probabilities = {
        method: np.full((data.labels.size, len(data.class_names)), np.nan, dtype=np.float64)
        for method in _METHODS
    }
    six = data.signals
    nine = data.nine_channel_signals
    fold_records: list[dict[str, Any]] = []
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
        fold_records.append(
            {
                "seed": seed,
                "outer_fold": fold,
                "training_participants": sorted(set(assignment) - evaluation_ids),
                "evaluation_participants": sorted(evaluation_ids),
                "participant_partition_plan_sha256": partition_plan.audit()["plan_sha256"],
                "outer_evaluation_labels_used_for_training_or_selection": False,
            }
        )
    if any(not np.isfinite(value).all() for value in probabilities.values()):
        raise ValueError("HAR-PMD outer predictions are incomplete")
    return probabilities, fold_records


def evaluate_har_pmd_stress(
    data: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    data.validate()
    if data.dataset_id != "har_pmd_v1" or data.channel_lane != "native-gravity-9ch":
        raise ValueError("native stress runner requires HAR-PMD's native lane")
    if data.participant_partition_plan is None:
        raise PermissionError("HAR-PMD evaluation requires a pre-window participant plan")
    data.participant_partition_plan.validate()
    if len(data.participant_partition_plan.participant_roster) < 12:
        raise ValueError("HAR-PMD stress evaluation requires at least 12 participants")
    evaluated = {seed: _evaluate_seed(data, seed=seed) for seed in seeds}
    per_seed = {seed: values[0] for seed, values in evaluated.items()}
    fold_records = [record for values in evaluated.values() for record in values[1]]
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
        "fold_records": fold_records,
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
    from inclusive_shift_har.evaluation.external_statistics import seed_evidence

    result["primary_seed_averaged"], seed_predictions = seed_evidence(data, per_seed)
    return result, {**ensemble, **seed_predictions}


def run_and_write(
    *,
    data: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if seeds != (11, 23, 47):
        raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
    git_at_launch, source_input_manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output_directory,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
    output_directory.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    dataset_summary = data.summary()
    evidence_status = _external_evidence_status(dataset_summary)
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "dataset": dataset_summary,
        "artifact_evidence_status": evidence_status,
        "source_receipts": [receipt.to_dict() for receipt in data.receipts],
        "raw_local_mirror": any(receipt.raw_local_mirror for receipt in data.receipts),
        "source_input_manifest": source_input_manifest,
        "git_at_launch": git_at_launch,
        "publication_launch_context": launch_context_binding,
    }
    data_audit_artifact = _write_self_hashed_json_create_only(
        output_directory / "data_audit.json", audit
    )
    artifact_contract = _publication_artifact_contract(source_input_manifest, data_audit_artifact)
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
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output_directory,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        result["started_at"] = started
        result["created_at"] = datetime.now(UTC).isoformat()
        result["git"] = _git_state(repository_root)
        result["git_at_launch"] = git_at_launch
        result["source_input_manifest"] = source_input_manifest
        result["publication_launch_context"] = launch_context_binding
        result["environment"] = _runtime_environment()
        result["artifact_evidence_status"] = evidence_status
        result["data_audit_artifact"] = data_audit_artifact
        result["artifact_contract"] = artifact_contract
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
    except Exception as exc:
        _write_self_hashed_json_create_only(
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
                "source_input_manifest": source_input_manifest,
                "publication_launch_context": launch_context_binding,
                "environment": _runtime_environment(),
                "artifact_evidence_status": evidence_status,
                "data_audit_artifact": data_audit_artifact,
                "artifact_contract": artifact_contract,
            },
            hash_field="failure_payload_sha256_before_serialization",
        )
        validate_and_record_run_directory(output_directory, repository_root)
        raise
    validate_and_record_run_directory(output_directory, repository_root)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--participant-limit", type=int)
    parser.add_argument("--har-pmd-source-archive", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--with-neural", action="store_true")
    parser.add_argument("--neural-epochs", type=int, default=40)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    output = args.output_directory.resolve()
    _launch, _manifest, launch_context = _resolve_publication_launch_context(
        repository_root=root,
        output_directory=output,
        current_git_state=_git_state(root),
        current_source_manifest=_source_input_manifest(root),
        manifest_commit_validator=_source_manifest_commit_errors,
    )
    started = datetime.now(UTC).isoformat()
    stage = "configuration"
    try:
        if tuple(args.seeds) != (11, 23, 47):
            raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
        if args.with_neural and args.neural_epochs != 40:
            raise ValueError("external publication neural evidence requires the frozen 40 epochs")
        stage = "dataset_acquisition"
        data = load_har_pmd_native(
            participant_limit=args.participant_limit,
            source_archive=args.har_pmd_source_archive,
        )
        stage = "classical_experiment_writer"
        result = run_and_write(
            data=data,
            output_directory=output,
            repository_root=root,
            seeds=tuple(args.seeds),
            inherited_launch_context=launch_context,
        )
        neural_summary = None
        if args.with_neural:
            from inclusive_shift_har.experiments.cross_dataset_neural import (
                run_and_write_neural,
            )

            neural_summary = {}
            stage = "neural_experiment_writers"
            for suffix, signals in (("6ch", data.signals), ("N9", data.nine_channel_signals)):
                neural = run_and_write_neural(
                    data=data,
                    signals=signals,
                    class_names=data.class_names,
                    method_suffix=suffix,
                    experiment_id="har-pmd-native-interface-neural-controls-v1",
                    output_directory=output / f"neural_{suffix}",
                    repository_root=root,
                    seeds=tuple(args.seeds),
                    epochs=args.neural_epochs,
                    inherited_launch_context=launch_context,
                )
                neural_summary.update(
                    {
                        name: report["primary"]["mean_participant_macro_f1"]
                        for name, report in neural["reports"].items()
                    }
                )
    except Exception as error:
        if not output.exists():
            _write_launch_failure_envelope(
                repository_root=root,
                output_directory=output,
                launch_context=launch_context,
                started_at=started,
                stage=stage,
                exception=error,
                traceback_text=traceback.format_exc(),
            )
        raise
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
