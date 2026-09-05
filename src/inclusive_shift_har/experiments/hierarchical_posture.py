"""One prospective, fixed-budget posture-forest experiment on consumed development data."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.data.external_har import (
    ExternalHARWindows,
    load_fog_star,
    participant_fold_assignment,
)
from inclusive_shift_har.evaluation.external_statistics import seed_evidence
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    _manifest_commit_errors,
    validate_run_directory,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.hierarchical_posture import (
    VARIANTS,
    FloatArray,
    fit_posture_forest,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features

PROTOCOL_ID = "participant-balanced-hierarchical-posture-forest-v3"
PROTOCOL_PATH = "configs/protocols/participant_balanced_hierarchical_posture_forest_v3.json"
SIX_NAMES = ("lin_acc_x", "lin_acc_y", "lin_acc_z", "gyro_x", "gyro_y", "gyro_z")


def posture_advancement_gate(statistics: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct the fixed-sequence gate, retaining all failed/descriptive contrasts."""
    methods = statistics["methods"]
    candidate = methods["PB-HPF"]
    people = sorted(candidate["participant_values"])
    values = np.array([candidate["participant_values"][person] for person in people])
    draws = np.random.default_rng(20260905).integers(0, len(people), (10000, len(people)))
    tail = max(1, int(np.ceil(0.30 * len(people))))
    comparisons = {}
    for name in VARIANTS:
        if name == "PB-HPF":
            continue
        base = np.array([methods[name]["participant_values"][person] for person in people])
        delta = values - base
        mean_interval = np.quantile(delta[draws].mean(axis=1), [0.025, 0.975])
        tail_delta = np.sort(values[draws], axis=1)[:, :tail].mean(axis=1) - np.sort(
            base[draws], axis=1
        )[:, :tail].mean(axis=1)
        comparisons[name] = {
            "mean_difference": float(delta.mean()),
            "paired_participant_bootstrap_95_percent_ci": mean_interval.tolist(),
            "bottom_30_percent_difference_95_percent_ci": np.quantile(
                tail_delta, [0.025, 0.975]
            ).tolist(),
            "rescue_count": int(np.sum(delta > 1e-12)),
            "harm_count": int(np.sum(delta < -1e-12)),
            "tie_count": int(np.sum(np.abs(delta) <= 1e-12)),
            "input_tuple_matched": name != "PB-RF-D9",
            "role": "primary" if name == "RandomForest-6ch" else "secondary_or_descriptive",
        }

    def averaged(method: str, group: str, key: str, subkey: str | None = None) -> float:
        reports = methods[method]["reports_by_seed"].values()
        return float(
            np.mean(
                [
                    report[group][key] if subkey is None else report[group][key][subkey]
                    for report in reports
                ]
            )
        )

    base_name = "RandomForest-6ch"
    primary = comparisons[base_name]
    criteria = {
        "primary_mean_superiority": primary["paired_participant_bootstrap_95_percent_ci"][0] > 0,
        "primary_lower_tail_noninferiority": primary["bottom_30_percent_difference_95_percent_ci"][
            0
        ]
        >= 0,
        "weighting_only_control_superiority": comparisons["PB-RF-6ch"][
            "paired_participant_bootstrap_95_percent_ci"
        ][0]
        > 0,
        "unweighted_hierarchy_superiority": comparisons["HPF-unweighted"][
            "paired_participant_bootstrap_95_percent_ci"
        ][0]
        > 0,
        "sitting_recall_improves": averaged("PB-HPF", "per_class", "sitting", "recall")
        > averaged(base_name, "per_class", "sitting", "recall"),
        "mobility_recall_not_worse": averaged("PB-HPF", "per_class", "mobility", "recall")
        >= averaged(base_name, "per_class", "mobility", "recall"),
        "negative_log_likelihood_not_worse": averaged(
            "PB-HPF", "calibration", "negative_log_likelihood"
        )
        <= averaged(base_name, "calibration", "negative_log_likelihood"),
        "true_specialist_outperforms_shuffled": comparisons["PB-HPF-shuffled-posture"][
            "mean_difference"
        ]
        > 0,
    }
    return {
        "protocol_id": PROTOCOL_ID,
        "primary_candidate": "PB-HPF",
        "primary_control": base_name,
        "criteria": {name: bool(value) for name, value in criteria.items()},
        "all_advancement_gates_passed": all(criteria.values()),
        "decision": "ELIGIBLE_FOR_FIXED_DEVELOPMENT_REPLICATION"
        if all(criteria.values())
        else "STOP_CANDIDATE_EXPANSION_RETAIN_NEGATIVE_OR_TRADEOFF_RESULT",
        "comparisons": comparisons,
        "multiplicity": "Fixed-sequence intersection gate. Secondary superiority is inferential only if the preceding primary mean and tail gates pass; no replacement winner is selected.",
    }


def evaluate_posture_forests(
    data: ExternalHARWindows, *, output: Path, seeds: tuple[int, ...] = (11, 23, 47)
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    data.validate()
    if data.class_names != ("mobility", "sitting", "standing"):
        raise ValueError("posture hierarchy requires the fixed three-class ontology")
    six = extract_engineered_features(data.signals, channel_names=SIX_NAMES)
    nine = extract_engineered_features(
        data.nine_channel_signals, channel_names=(*SIX_NAMES, "gravity_x", "gravity_y", "gravity_z")
    )
    per_seed: dict[int, dict[str, FloatArray]] = {}
    records = []
    for seed in seeds:
        assignment = participant_fold_assignment(
            np.unique(data.participant_ids).tolist(), fold_count=5, seed=seed
        )
        probabilities = {name: np.full((len(data.labels), 3), np.nan) for name in VARIANTS}
        for fold in range(5):
            evaluation_people = sorted(
                person for person, value in assignment.items() if value == fold
            )
            evaluation = np.isin(data.participant_ids, evaluation_people)
            training = ~evaluation
            for name in VARIANTS:
                features = nine if name == "PB-RF-D9" else six
                started = time.perf_counter()
                fitted = fit_posture_forest(
                    features.values[training],
                    data.labels[training],
                    data.participant_ids[training],
                    variant=name,
                    seed=seed + fold,
                )
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                probability = fitted.predict(features.values[evaluation])
                predict_seconds = time.perf_counter() - started
                probabilities[name][evaluation] = probability
                checkpoint = output / "checkpoints" / f"seed-{seed}__fold-{fold}__{name}.pickle"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("xb") as stream:
                    pickle.dump(fitted, stream, protocol=5)
                records.append(
                    {
                        "seed": seed,
                        "outer_fold": fold,
                        "method": name,
                        "training_participants": list(fitted.training_participants),
                        "evaluation_participants": evaluation_people,
                        "outer_labels_used_for_training_or_selection": False,
                        "feature_names": list(features.names),
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": predict_seconds,
                        "timing_scope": "local wall time, may include concurrent system work; not portable device latency",
                        "model_size": fitted.size_summary(),
                        "checkpoint": {
                            "path": checkpoint.relative_to(output).as_posix(),
                            "sha256": sha256_file(checkpoint),
                            "size_bytes": checkpoint.stat().st_size,
                            "trusted_local_pickle_only": True,
                        },
                        "mechanism": fitted.mechanism(features.values[evaluation]),
                    }
                )
            fold_archive = output / f"seed-{seed}__fold-{fold}__predictions.npz"
            with fold_archive.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    labels=data.labels[evaluation],
                    participant_ids=data.participant_ids[evaluation],
                    window_ids=data.window_ids[evaluation],
                    **{  # type: ignore[arg-type]
                        f"probability__{name}": values[evaluation]
                        for name, values in probabilities.items()
                    },
                )
            _write_json_create_only(
                output / f"seed-{seed}__fold-{fold}__complete.json",
                {
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "seed": seed,
                    "fold": fold,
                    "all_variants_completed": list(VARIANTS),
                    "prediction_archive_sha256": sha256_file(fold_archive),
                },
            )
            print(
                json.dumps({"stage": "posture_forest_fold_complete", "seed": seed, "fold": fold}),
                flush=True,
            )
        if any(not np.isfinite(values).all() for values in probabilities.values()):
            raise ValueError("posture forest predictions are incomplete")
        per_seed[seed] = probabilities
    statistics, archive = seed_evidence(data, per_seed)
    ensemble = {
        name: np.mean([per_seed[seed][name] for seed in seeds], axis=0) for name in VARIANTS
    }
    result = {
        "schema_version": "1.0.0",
        "experiment_id": PROTOCOL_ID,
        "dataset": data.summary(),
        "seeds": list(seeds),
        "evidence_status": "PROSPECTIVE_DEVELOPMENT_RND_NOT_CONFIRMATORY",
        "reports": {
            name: classification_report(
                data.labels, values, data.participant_ids.tolist(), class_names=data.class_names
            )
            for name, values in ensemble.items()
        },
        "primary_seed_averaged": statistics,
        "advancement_gate": posture_advancement_gate(statistics),
        "fold_records": records,
        "method_input_lanes": {
            name: "derived-nine-channel diagnostic"
            if name == "PB-RF-D9"
            else "six-channel matched primary lane"
            for name in VARIANTS
        },
        "claim_policy": {
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "generic_hierarchy_novelty_claim_allowed": False,
        },
    }
    return result, {**ensemble, **archive}


def run_and_write(
    *, data: ExternalHARWindows, output: Path, repository_root: Path, reference_run: Path
) -> dict[str, Any]:
    launch, manifest = _git_state(repository_root), _source_input_manifest(repository_root)
    if launch["worktree_dirty"] or _manifest_commit_errors(
        repository_root, launch["commit"], manifest["files"]
    ):
        raise ValueError("R&D requires clean committed source before outcomes")
    protocol_path = repository_root / PROTOCOL_PATH
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    unhashed = {key: value for key, value in protocol.items() if key != "record_sha256"}
    if (
        protocol["protocol_id"] != PROTOCOL_ID
        or canonical_json_sha256(unhashed) != protocol["record_sha256"]
    ):
        raise ValueError("frozen R&D declaration hash/identifier differs")
    reference_validation = validate_run_directory(reference_run, repository_root)
    if not reference_validation["publication_evidence_ready"] and not (
        reference_validation.get("publication_evidence_ready_for_unqualified_methods", False)
        and "RandomForest-6ch" in reference_validation.get("unqualified_method_names", [])
    ):
        raise ValueError("baseline reference must be a validated corrected result package")
    reference_result = json.loads((reference_run / "result.json").read_text(encoding="utf-8"))
    reference_path = reference_run / reference_result["prediction_artifact"]["path"]
    with np.load(reference_path, allow_pickle=False) as archive:
        for name in ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids"):
            if not np.array_equal(archive[name], getattr(data, name)):
                raise ValueError(f"baseline reference has a different evaluation tuple: {name}")
        reference_probabilities = {
            seed: archive[f"probability__seed-{seed}__RandomForest-6ch"].copy()
            for seed in (11, 23, 47)
        }
    output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(UTC).isoformat()
    _write_json_create_only(
        output / "data_audit.json",
        {
            "schema_version": "1.0.0",
            "created_at": started,
            "dataset": data.summary(),
            "source_receipts": [receipt.to_dict() for receipt in data.receipts],
            "source_input_manifest": manifest,
            "git_at_launch": launch,
            "protocol": protocol,
        },
    )
    try:
        result, probabilities = evaluate_posture_forests(data, output=output)
        baseline_checks = {
            str(seed): {
                "class_decisions_exact": bool(
                    np.array_equal(
                        values.argmax(axis=1),
                        probabilities[f"seed-{seed}__RandomForest-6ch"].argmax(axis=1),
                    )
                ),
                "maximum_probability_absolute_difference": float(
                    np.max(np.abs(values - probabilities[f"seed-{seed}__RandomForest-6ch"]))
                ),
            }
            for seed, values in reference_probabilities.items()
        }
        result["baseline_reconstruction"] = {
            "reference_run": str(reference_run.resolve()),
            "reference_result_sha256": sha256_file(reference_run / "result.json"),
            "reference_predictions_sha256": sha256_file(reference_path),
            "per_seed": baseline_checks,
            "gate_passed": all(item["class_decisions_exact"] for item in baseline_checks.values()),
        }
        if not result["baseline_reconstruction"]["gate_passed"]:
            _write_json_create_only(
                output / "baseline_reconstruction_failure.json", result["baseline_reconstruction"]
            )
            raise ValueError("matched baseline class decisions did not reproduce")
        prediction = output / "predictions.npz"
        with prediction.open("xb") as stream:
            np.savez_compressed(
                stream,
                labels=data.labels,
                participant_ids=data.participant_ids,
                session_ids=data.session_ids,
                trial_ids=data.trial_ids,
                window_ids=data.window_ids,
                **{  # type: ignore[arg-type]
                    f"probability__{name}": value for name, value in probabilities.items()
                },
            )
        result.update(
            {
                "started_at": started,
                "created_at": datetime.now(UTC).isoformat(),
                "git_at_launch": launch,
                "git": _git_state(repository_root),
                "source_input_manifest": manifest,
                "protocol": {
                    "path": PROTOCOL_PATH,
                    "sha256": sha256_file(protocol_path),
                    "record_sha256": protocol["record_sha256"],
                },
                "prediction_artifact": {"path": prediction.name, "sha256": sha256_file(prediction)},
                "environment": {
                    dist.metadata["Name"]: dist.version
                    for dist in importlib.metadata.distributions()
                    if "Name" in dist.metadata
                },
            }
        )
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output / "result.json", result)
    except Exception as error:
        _write_json_create_only(
            output / "failure.json",
            {
                "status": "FAILED_PRESERVED",
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "git_at_launch": launch,
                "source_input_manifest": manifest,
            },
        )
        raise
    validation = validate_run_directory(output, repository_root)
    _write_json_create_only(output / "validation.json", validation)
    if not validation["publication_evidence_ready"]:
        raise ValueError("R&D evidence failed the independent reconstruction gate")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-run", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.repository_root.resolve()
    launch = _git_state(root)
    manifest = _source_input_manifest(root)
    if launch["worktree_dirty"] or _manifest_commit_errors(
        root, launch["commit"], manifest["files"]
    ):
        raise ValueError("R&D requires clean source before dataset acquisition")
    result = run_and_write(
        data=load_fog_star(),
        output=args.output.resolve(),
        repository_root=root,
        reference_run=args.reference_run.resolve(),
    )
    print(json.dumps(result["advancement_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
