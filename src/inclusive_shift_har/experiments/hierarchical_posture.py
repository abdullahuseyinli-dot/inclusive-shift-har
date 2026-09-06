"""One prospective, fixed-budget posture-forest experiment on consumed development data."""

from __future__ import annotations

import argparse
import json
import pickle
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.artifacts.research_provenance import (
    _external_evidence_status,
    _fog_star_full_cohort_observed,
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
    _array_sha256,
    load_fog_star,
    observable_modelling_pool,
)
from inclusive_shift_har.evaluation.external_statistics import seed_evidence
from inclusive_shift_har.evaluation.inference_contracts import OBSERVABLE_CONTEXT_PROTOCOL
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.cross_dataset_har import (
    _git_state,
    _source_input_manifest,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
    validate_run_directory,
)
from inclusive_shift_har.experiments.publication_split_audit import write_split_audit
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
BASELINE_PROBABILITY_ATOL = 1e-12


def _reference_run_location(reference_run: Path, repository_root: Path) -> dict[str, str]:
    resolved = reference_run.resolve()
    try:
        relative = resolved.relative_to(repository_root.resolve()).as_posix()
    except ValueError:
        return {"path_kind": "external_absolute", "path": str(resolved)}
    return {"path_kind": "repository_relative", "path": relative or "."}


def _stable_receipt_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    """Remove only the retrieval timestamp from an otherwise exact source receipt."""

    return {name: value for name, value in receipt.items() if name != "accessed_at_utc"}


def _matched_reference_input_errors(
    data: ExternalHARWindows,
    reference_result: dict[str, Any],
    reference_audit: dict[str, Any],
) -> list[str]:
    """Require a baseline produced from the exact current scientific input tuple."""

    errors: list[str] = []
    summary = data.summary()
    if reference_result.get("dataset") != summary:
        errors.append("result dataset summary differs")
    if reference_audit.get("dataset") != summary:
        errors.append("data-audit dataset summary differs")
    expected_receipts = [_stable_receipt_identity(receipt.to_dict()) for receipt in data.receipts]
    actual_receipts = reference_audit.get("source_receipts")
    if (
        not isinstance(actual_receipts, list)
        or [
            _stable_receipt_identity(receipt) if isinstance(receipt, dict) else {}
            for receipt in actual_receipts
        ]
        != expected_receipts
    ):
        errors.append("source receipt identity/version/hash differs")
    return errors


def _baseline_probability_check(reference: FloatArray, reconstructed: FloatArray) -> dict[str, Any]:
    """Check a calibration-bearing comparator, not only its class decisions."""

    shape_matches = bool(
        reference.ndim == 2
        and reconstructed.ndim == 2
        and reference.shape == reconstructed.shape
        and reference.shape[1] == 3
    )
    finite = bool(np.isfinite(reference).all() and np.isfinite(reconstructed).all())
    maximum_difference = (
        float(np.max(np.abs(reference - reconstructed)))
        if shape_matches and reference.size and finite
        else float("inf")
    )
    return {
        "class_decisions_exact": bool(
            shape_matches
            and finite
            and np.array_equal(reference.argmax(axis=1), reconstructed.argmax(axis=1))
        ),
        "probabilities_within_strict_tolerance": bool(
            shape_matches
            and finite
            and np.allclose(
                reference,
                reconstructed,
                rtol=0.0,
                atol=BASELINE_PROBABILITY_ATOL,
            )
        ),
        "probability_absolute_tolerance": BASELINE_PROBABILITY_ATOL,
        "maximum_probability_absolute_difference": maximum_difference,
    }


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
    scored_data = data
    has_observable_pool = data.observable_candidates is not None
    if data.participant_partition_plan is None:
        raise PermissionError("posture evaluation requires a pre-window participant plan")
    partition_plan = data.participant_partition_plan
    partition_plan.validate()
    if data.class_names != ("mobility", "sitting", "standing"):
        raise ValueError("posture hierarchy requires the fixed three-class ontology")
    data, scoring_indices, supervised_eligibility = observable_modelling_pool(
        data, include_supervised_labels=True
    )
    if (
        scoring_indices.ndim != 1
        or scoring_indices.size != scored_data.labels.size
        or np.any(np.diff(scoring_indices) <= 0)
        or not np.array_equal(data.window_ids[scoring_indices], scored_data.window_ids)
    ):
        raise ValueError("scored windows are not an ordered subset of the observable pool")
    six = extract_engineered_features(data.signals, channel_names=SIX_NAMES)
    nine = extract_engineered_features(
        data.nine_channel_signals, channel_names=(*SIX_NAMES, "gravity_x", "gravity_y", "gravity_z")
    )
    per_seed: dict[int, dict[str, FloatArray]] = {}
    records = []
    fold_artifacts: list[dict[str, Any]] = []
    for seed in seeds:
        assignment = partition_plan.resolve(
            partition_plan.participant_roster,
            fold_count=5,
            seed=seed,
            role="outer",
        )
        candidate_probabilities = {
            name: np.full((len(data.labels), 3), np.nan) for name in VARIANTS
        }
        for fold in range(5):
            evaluation_people = sorted(
                person for person, value in assignment.items() if value == fold
            )
            assigned_training_people = sorted(set(assignment) - set(evaluation_people))
            evaluation = np.isin(data.participant_ids, evaluation_people)
            training_candidates = ~evaluation
            training = training_candidates & supervised_eligibility
            scored_evaluation = evaluation & supervised_eligibility
            if not evaluation.any():
                raise ValueError("an outer fold has no observable inference candidates")
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
                candidate_probabilities[name][evaluation] = probability
                checkpoint = output / "checkpoints" / f"seed-{seed}__fold-{fold}__{name}.pickle"
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                with checkpoint.open("xb") as stream:
                    pickle.dump(fitted, stream, protocol=5)
                records.append(
                    {
                        "seed": seed,
                        "outer_fold": fold,
                        "method": name,
                        "training_participants": assigned_training_people,
                        "training_participants_with_supervision": list(
                            fitted.training_participants
                        ),
                        "training_participants_with_candidates": sorted(
                            np.unique(data.participant_ids[training_candidates]).tolist()
                        ),
                        "evaluation_participants": evaluation_people,
                        "evaluation_participants_with_candidates": sorted(
                            np.unique(data.participant_ids[evaluation]).tolist()
                        ),
                        "evaluation_participants_with_scoring": sorted(
                            np.unique(data.participant_ids[scored_evaluation]).tolist()
                        ),
                        "participant_partition_plan_sha256": partition_plan.audit()["plan_sha256"],
                        "training_candidate_window_count": int(training_candidates.sum()),
                        "training_scored_window_count": int(training.sum()),
                        "evaluation_candidate_window_count": int(evaluation.sum()),
                        "evaluation_scored_window_count": int(scored_evaluation.sum()),
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
                    labels=data.labels[scored_evaluation],
                    participant_ids=data.participant_ids[scored_evaluation],
                    window_ids=data.window_ids[scored_evaluation],
                    candidate_participant_ids=data.participant_ids[evaluation],
                    candidate_window_ids=data.window_ids[evaluation],
                    candidate_scoring_eligibility=supervised_eligibility[evaluation],
                    **{  # type: ignore[arg-type]
                        f"probability__{name}": values[scored_evaluation]
                        for name, values in candidate_probabilities.items()
                    },
                    **{  # type: ignore[arg-type]
                        f"candidate_probability__{name}": values[evaluation]
                        for name, values in candidate_probabilities.items()
                    },
                )
            completion = output / f"seed-{seed}__fold-{fold}__complete.json"
            _write_json_create_only(
                completion,
                {
                    "completed_at_utc": datetime.now(UTC).isoformat(),
                    "seed": seed,
                    "fold": fold,
                    "all_variants_completed": list(VARIANTS),
                    "prediction_archive_sha256": sha256_file(fold_archive),
                },
            )
            fold_artifacts.append(
                {
                    "seed": seed,
                    "outer_fold": fold,
                    "prediction_archive": {
                        "path": fold_archive.relative_to(output).as_posix(),
                        "sha256": sha256_file(fold_archive),
                        "size_bytes": fold_archive.stat().st_size,
                    },
                    "completion_marker": {
                        "path": completion.relative_to(output).as_posix(),
                        "sha256": sha256_file(completion),
                        "size_bytes": completion.stat().st_size,
                    },
                }
            )
            print(
                json.dumps({"stage": "posture_forest_fold_complete", "seed": seed, "fold": fold}),
                flush=True,
            )
        if any(not np.isfinite(values).all() for values in candidate_probabilities.values()):
            raise ValueError("posture forest predictions are incomplete")
        per_seed[seed] = {
            name: values[scoring_indices] for name, values in candidate_probabilities.items()
        }
    statistics, archive = seed_evidence(scored_data, per_seed)
    ensemble = {
        name: np.mean([per_seed[seed][name] for seed in seeds], axis=0) for name in VARIANTS
    }
    result = {
        "schema_version": "1.0.0",
        "experiment_id": PROTOCOL_ID,
        "observable_context_protocol": OBSERVABLE_CONTEXT_PROTOCOL if has_observable_pool else None,
        "dataset": scored_data.summary(),
        "seeds": list(seeds),
        "evidence_status": "PROSPECTIVE_DEVELOPMENT_RND_NOT_CONFIRMATORY",
        "reports": {
            name: classification_report(
                scored_data.labels,
                values,
                scored_data.participant_ids.tolist(),
                class_names=scored_data.class_names,
            )
            for name, values in ensemble.items()
        },
        "primary_seed_averaged": statistics,
        "advancement_gate": posture_advancement_gate(statistics),
        "fold_records": records,
        "fold_artifacts": fold_artifacts,
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
    *,
    data: ExternalHARWindows,
    output: Path,
    repository_root: Path,
    reference_run: Path,
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    launch, manifest, launch_context = _resolve_publication_launch_context(
        repository_root=repository_root,
        output_directory=output,
        current_git_state=_git_state(repository_root),
        current_source_manifest=_source_input_manifest(repository_root),
        manifest_commit_validator=_source_manifest_commit_errors,
        inherited_launch_context=inherited_launch_context,
    )
    launch_context_binding = _publication_launch_context_binding(launch_context)
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
    reference_audit = json.loads((reference_run / "data_audit.json").read_text(encoding="utf-8"))
    if data.observable_candidates is None:
        raise ValueError("posture R&D requires the complete observable candidate pool")
    reference_errors = _matched_reference_input_errors(data, reference_result, reference_audit)
    if reference_errors:
        raise ValueError(
            "baseline reference has a different scientific input tuple: "
            + "; ".join(reference_errors)
        )
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
    dataset_summary = data.summary()
    evidence_status = _external_evidence_status(dataset_summary)
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "dataset": dataset_summary,
        "artifact_evidence_status": evidence_status,
        "source_receipts": [receipt.to_dict() for receipt in data.receipts],
        "source_input_manifest": manifest,
        "git_at_launch": launch,
        "publication_launch_context": launch_context_binding,
        "protocol": protocol,
    }
    data_audit_artifact = _write_self_hashed_json_create_only(output / "data_audit.json", audit)
    artifact_contract = _publication_artifact_contract(manifest, data_audit_artifact)
    try:
        reference_witness = output / "baseline_reference_predictions.npz"
        with reference_witness.open("xb") as stream:
            np.savez_compressed(
                stream,
                labels=data.labels,
                participant_ids=data.participant_ids,
                session_ids=data.session_ids,
                trial_ids=data.trial_ids,
                window_ids=data.window_ids,
                **{
                    f"probability__seed-{seed}__RandomForest-6ch": values
                    for seed, values in reference_probabilities.items()
                },
            )
        reference_witness_artifact = {
            "path": reference_witness.relative_to(output).as_posix(),
            "sha256": sha256_file(reference_witness),
            "size_bytes": reference_witness.stat().st_size,
        }
        result, probabilities = evaluate_posture_forests(data, output=output)
        baseline_checks = {
            str(seed): _baseline_probability_check(
                values, probabilities[f"seed-{seed}__RandomForest-6ch"]
            )
            for seed, values in reference_probabilities.items()
        }
        result["baseline_reconstruction"] = {
            "reference_run": _reference_run_location(reference_run, repository_root),
            "reference_result_sha256": sha256_file(reference_run / "result.json"),
            "reference_predictions_sha256": sha256_file(reference_path),
            "retained_reference_witness": reference_witness_artifact,
            "per_seed": baseline_checks,
            "scientific_input_tuple_exact": True,
            "gate_passed": all(
                item["class_decisions_exact"] and item["probabilities_within_strict_tolerance"]
                for item in baseline_checks.values()
            ),
        }
        if not result["baseline_reconstruction"]["gate_passed"]:
            _write_json_create_only(
                output / "baseline_reconstruction_failure.json", result["baseline_reconstruction"]
            )
            raise ValueError("matched baseline class decisions did not reproduce")
        modelling, scoring_indices, scoring_eligibility = observable_modelling_pool(
            data, include_supervised_labels=True
        )
        candidate_identifiers = {
            "participant_ids": modelling.participant_ids,
            "window_ids": modelling.window_ids,
        }
        result["candidate_prediction_contract"] = {
            "candidate_window_count": int(modelling.window_ids.size),
            "scored_window_count": int(data.window_ids.size),
            "scoring_indices_sha256": canonical_json_sha256(scoring_indices.tolist()),
            "scoring_indices_array_sha256": _array_sha256(scoring_indices),
            "scoring_eligibility_array_sha256": _array_sha256(scoring_eligibility),
            "candidate_identifier_hashes": {
                name: _array_sha256(values) for name, values in candidate_identifiers.items()
            },
        }
        prediction = output / "predictions.npz"
        with prediction.open("xb") as stream:
            np.savez_compressed(
                stream,
                labels=data.labels,
                participant_ids=data.participant_ids,
                session_ids=data.session_ids,
                trial_ids=data.trial_ids,
                window_ids=data.window_ids,
                candidate_participant_ids=modelling.participant_ids,
                candidate_window_ids=modelling.window_ids,
                candidate_scoring_indices=scoring_indices,
                candidate_scoring_eligibility=scoring_eligibility,
                **{  # type: ignore[arg-type]
                    f"probability__{name}": value for name, value in probabilities.items()
                },
            )
        _resolve_publication_launch_context(
            repository_root=repository_root,
            output_directory=output,
            current_git_state=_git_state(repository_root),
            current_source_manifest=_source_input_manifest(repository_root),
            manifest_commit_validator=_source_manifest_commit_errors,
            inherited_launch_context=launch_context,
        )
        result.update(
            {
                "started_at": started,
                "created_at": datetime.now(UTC).isoformat(),
                "git_at_launch": launch,
                "git": _git_state(repository_root),
                "source_input_manifest": manifest,
                "publication_launch_context": launch_context_binding,
                "protocol": {
                    "path": PROTOCOL_PATH,
                    "sha256": sha256_file(protocol_path),
                    "record_sha256": protocol["record_sha256"],
                },
                "prediction_artifact": {"path": prediction.name, "sha256": sha256_file(prediction)},
                "environment": _runtime_environment(),
                "artifact_evidence_status": evidence_status,
                "data_audit_artifact": data_audit_artifact,
                "artifact_contract": artifact_contract,
            }
        )
        result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
        _write_json_create_only(output / "result.json", result)
    except Exception as error:
        _write_self_hashed_json_create_only(
            output / "failure.json",
            {
                "status": "FAILED_PRESERVED",
                "started_at": started,
                "failed_at_utc": datetime.now(UTC).isoformat(),
                "exception_type": type(error).__name__,
                "exception_message": str(error),
                "traceback": traceback.format_exc(),
                "git_at_launch": launch,
                "source_input_manifest": manifest,
                "publication_launch_context": launch_context_binding,
                "environment": _runtime_environment(),
                "artifact_evidence_status": evidence_status,
                "data_audit_artifact": data_audit_artifact,
                "artifact_contract": artifact_contract,
            },
            hash_field="failure_payload_sha256_before_serialization",
        )
        validate_and_record_run_directory(output, repository_root)
        raise
    split_audit = write_split_audit(
        output,
        output / "split_audit.json",
        repository_root=repository_root,
    )
    validation = validate_run_directory(output, repository_root)
    _write_json_create_only(output / "validation.json", validation)
    if not split_audit["valid"] or split_audit["status"] != "PASS_RECORDED_RESULT":
        raise ValueError("R&D evidence failed the recorded participant-split gate")
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
    _launch, _manifest, launch_context = _resolve_publication_launch_context(
        repository_root=root,
        output_directory=args.output.resolve(),
        current_git_state=_git_state(root),
        current_source_manifest=_source_input_manifest(root),
        manifest_commit_validator=_source_manifest_commit_errors,
    )
    output = args.output.resolve()
    started = datetime.now(UTC).isoformat()
    stage = "dataset_acquisition"
    try:
        data = load_fog_star()
        if not _fog_star_full_cohort_observed(data.summary()):
            raise ValueError(
                "posture R&D requires the exact 22-person FoG-STAR provider roster "
                "and all planned participants to contribute scored windows"
            )
        stage = "reference_validation_and_experiment_writer"
        result = run_and_write(
            data=data,
            output=output,
            repository_root=root,
            reference_run=args.reference_run.resolve(),
            inherited_launch_context=launch_context,
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
    print(json.dumps(result["advancement_gate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
