"""Zero-shot IMU-HAR-IL to FoG-STAR evaluation for the external HAR portfolio.

Every selection, calibration, route, and physics threshold is learned from participant-
exclusive IMU-HAR-IL out-of-fold predictions.  FoG-STAR labels are replaced by dummy
values in the modelling container and are read only after every target probability matrix
has been fixed.
"""

from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

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
    CORE_CLASS_NAMES,
    ExternalHARWindows,
    concatenate_external_windows,
    load_fog_star,
    load_imu_har_il,
    observable_modelling_pool,
)
from inclusive_shift_har.evaluation.inference_contracts import (
    OBSERVABLE_CONTEXT_PROTOCOL,
    OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
)
from inclusive_shift_har.experiments.cage_har import _report
from inclusive_shift_har.experiments.cage_har_retrospective import (
    _outer_context,
    evaluate_cage_outer_arrays,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates,
    _feature_views,
    _fit_base,
    _fit_expert,
    _gravity_probability,
)
from inclusive_shift_har.experiments.cross_dataset_har import (
    _CLASSICAL_METHODS,
    _INVENTION_METHODS,
    _candidate_probability,
    _git_state,
    _inner_oof_and_selection,
    _paired_bootstrap,
    _read_mapping,
    _source_input_manifest,
    _three_probabilities,
    _write_json_create_only,
)
from inclusive_shift_har.experiments.external_evidence_validate import (
    validate_and_record_run_directory,
)
from inclusive_shift_har.experiments.hera_ctgr_retrospective import (
    _apply_calibration,
    _controller,
    _meta_training_controller,
    _select_posture_calibration,
    _slice_context,
    _utility_targets,
)
from inclusive_shift_har.experiments.hera_ctgr_v2_retrospective import (
    _apply_dual_candidate,
    _fit_route_lane,
    _marginalize,
    _select_dual_candidate,
    _select_offset,
    _select_temperature,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _positive_probability,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.classical import ClassicalConfig, fit_classical_model
from inclusive_shift_har.models.hera_ctgr import (
    apply_physics_reference,
    apply_physics_veto,
    build_responder_signatures,
    extract_gravity_kinematic_context,
    fit_physics_reference,
)
from inclusive_shift_har.models.hera_ctgr_v2 import (
    apply_rank_locked_posture_offset,
    build_rescue_harm_features,
    extract_dual_frame_posture_features,
    temperature_scale_probabilities,
)
from inclusive_shift_har.models.microstate_posture_graph import (
    compose_mobility_posture_probabilities,
)
from inclusive_shift_har.models.robust_multiscale_residual_pyramid import (
    robust_multiscale_signal_views,
)
from inclusive_shift_har.preprocessing.features import extract_engineered_features

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]


def _classical_zero_shot(
    source: ExternalHARWindows,
    target: ExternalHARWindows,
    *,
    seed: int,
) -> dict[str, FloatArray]:
    output: dict[str, FloatArray] = {}
    channel_names = (
        "lin_acc_x",
        "lin_acc_y",
        "lin_acc_z",
        "gyro_x",
        "gyro_y",
        "gyro_z",
    )
    for model_name, result_name in (
        ("random_forest", "RandomForest-6ch"),
        ("xgboost", "XGBoost-6ch"),
    ):
        fitted = fit_classical_model(
            source.signals,
            source.labels,
            source.participant_ids.tolist(),
            config=ClassicalConfig(model_name=model_name, num_classes=3, seed=seed),
            channel_names=channel_names,
            lineage={
                "evidence_status": "zero_shot_external_development_only",
                "source_dataset": source.dataset_id,
                "target_dataset": target.dataset_id,
                "target_labels_used_for_fit_or_selection": False,
            },
        )
        target_features = extract_engineered_features(target.signals, channel_names=channel_names)
        if target_features.names != fitted.feature_names:
            raise ValueError("zero-shot engineered feature schema differs between datasets")
        output[result_name] = _three_probabilities(fitted.estimator, target_features.values)
    return output


def _fit_apply_seed(
    source: ExternalHARWindows,
    target: ExternalHARWindows,
    *,
    seed: int,
    repository_root: Path,
    n_jobs: int,
    include_classical: bool,
) -> tuple[dict[str, FloatArray], dict[str, Any]]:
    source.validate()
    target.validate()
    if source.dataset_id != "imu_har_il_v1" or target.dataset_id != "fog_star_v3":
        raise ValueError("zero-shot runner is bound to IMU-HAR-IL source and FoG-STAR target")
    if source.participant_partition_plan is None:
        raise PermissionError("zero-shot source requires a pre-window participant plan")
    source_partition_plan = source.participant_partition_plan
    source_partition_plan.validate()
    if target.participant_partition_plan is None:
        raise PermissionError("zero-shot target requires a pre-window participant plan")
    target_partition_plan = target.participant_partition_plan
    target_partition_plan.validate()
    if set(source_partition_plan.participant_roster) & set(
        target_partition_plan.participant_roster
    ):
        raise PermissionError("zero-shot source and target participant rosters overlap")
    scored_target = target
    target, scoring_indices, _target_supervised_eligibility = observable_modelling_pool(
        target, include_supervised_labels=False
    )
    combined = concatenate_external_windows(
        (source, target),
        dataset_id=f"{source.dataset_id}_to_{target.dataset_id}",
        require_all_classes_per_dataset=False,
    )
    source_count = source.labels.size
    # This replacement makes target-label leakage through any imported fitting helper
    # impossible. The real target labels remain only in `scored_target`, which is not
    # passed to any fitting helper.
    modelling = replace(
        combined,
        labels=np.concatenate(
            (source.labels, np.zeros(target.labels.size, dtype=np.int64)), axis=0
        ),
    )
    training = np.zeros(modelling.labels.size, dtype=np.bool_)
    training[:source_count] = True
    evaluation = ~training
    labels = modelling.labels
    participants = modelling.participant_ids
    signals = np.asarray(modelling.signals, dtype=np.float64)
    gravity = np.asarray(modelling.gravity, dtype=np.float64)
    rmrp, views = _feature_views(signals, gravity, sampling_rate_hz=modelling.sampling_rate_hz)
    denoised = robust_multiscale_signal_views(signals, sampling_rate_hz=modelling.sampling_rate_hz)[
        "denoised"
    ]
    dual = extract_dual_frame_posture_features(
        denoised, gravity, sampling_rate_hz=modelling.sampling_rate_hz
    )
    kinematic = extract_gravity_kinematic_context(
        denoised, gravity, sampling_rate_hz=modelling.sampling_rate_hz
    )
    ctgr_config = _read_mapping(
        repository_root / "configs/experiments/confidence_triggered_gravity_residual_v1.yaml"
    )
    candidates = _candidates(ctgr_config)
    cage_config = _read_mapping(repository_root / "configs/experiments/cage_har_v1.yaml")
    v1_fixed = cast(
        dict[str, Any],
        _read_mapping(repository_root / "configs/experiments/hera_ctgr_retrospective_v1.yaml")[
            "fixed_method"
        ],
    )
    v2_fixed = cast(
        dict[str, Any],
        _read_mapping(repository_root / "configs/experiments/hera_ctgr_v2_retrospective_v1.yaml")[
            "fixed_method"
        ],
    )
    inner = _inner_oof_and_selection(
        rmrp=rmrp,
        views=views,
        dual_features=dual.features,
        labels=labels,
        participants=participants,
        outer_training=training,
        outer_training_candidates=training,
        candidates=candidates,
        outer_index=0,
        seed=seed,
        inner_fold_count=4,
        n_jobs=n_jobs,
        partition_plan=source_partition_plan,
        prewindow_training_ids=source_partition_plan.participant_roster,
        assignment_role="transfer_inner",
    )
    base_model = _fit_base(
        rmrp,
        labels,
        participants,
        training,
        seed=seed + 500,
        n_jobs=n_jobs,
    )
    target_base = _three_probabilities(base_model, rmrp[evaluation])
    needed = {
        (str(item["expert_view"]), str(item["posture_estimator"]))
        for item in (inner.selected_expert_candidate, *inner.top_three_expert_candidates)
    }
    if str(inner.selected_candidate["id"]) != "base_no_route":
        needed.add(
            (
                str(inner.selected_candidate["expert_view"]),
                str(inner.selected_candidate["posture_estimator"]),
            )
        )
    target_expert_by_key: dict[tuple[str, str], FloatArray] = {}
    for key_index, (view, estimator) in enumerate(sorted(needed)):
        model = _fit_expert(
            estimator,
            views[view],
            labels,
            participants,
            training,
            seed=seed + 9_001 + key_index,
            n_jobs=n_jobs,
        )
        target_expert_by_key[(view, estimator)] = _gravity_probability(
            target_base, model, views[view][evaluation]
        )
    selected_key = (
        str(inner.selected_expert_candidate["expert_view"]),
        str(inner.selected_expert_candidate["posture_estimator"]),
    )
    source_expert = inner.expert_by_key[selected_key][training]
    target_expert = target_expert_by_key[selected_key]
    source_ctgr = _candidate_probability(
        inner.base[training],
        {key: value[training] for key, value in inner.expert_by_key.items()},
        inner.selected_candidate,
    )
    target_ctgr = _candidate_probability(
        target_base, target_expert_by_key, inner.selected_candidate
    )
    source_stack = np.stack(
        [
            _candidate_probability(
                inner.base[training],
                {key: value[training] for key, value in inner.expert_by_key.items()},
                candidate,
            )
            for candidate in inner.top_three_expert_candidates
        ],
        axis=0,
    )
    target_stack = np.stack(
        [
            _candidate_probability(target_base, target_expert_by_key, candidate)
            for candidate in inner.top_three_expert_candidates
        ],
        axis=0,
    )
    source_equal = np.mean(source_stack, axis=0)
    target_equal = np.mean(target_stack, axis=0)
    context, cage_reliability, channel_scale = _outer_context(
        signals,
        gravity,
        training,
        sampling_rate_hz=modelling.sampling_rate_hz,
    )
    all_base = np.concatenate((inner.base[training], target_base), axis=0)
    all_expert = np.concatenate((source_expert, target_expert), axis=0)
    all_ctgr = np.concatenate((source_ctgr, target_ctgr), axis=0)
    all_context = np.concatenate((context[training], context[evaluation]), axis=0)
    all_reliability = np.concatenate(
        (cage_reliability[training], cage_reliability[evaluation]), axis=0
    )
    cage_methods, cage_diagnostics, cage_arrays = evaluate_cage_outer_arrays(
        training_base_probability=inner.base[training],
        training_expert_probability=source_expert,
        training_labels=labels[training],
        training_participants=participants[training],
        training_context=context[training],
        training_reliability=cage_reliability[training],
        evaluation_base_probability=all_base,
        evaluation_expert_probability=all_expert,
        evaluation_ctgr_probability=all_ctgr,
        evaluation_context=all_context,
        evaluation_reliability=all_reliability,
        expert_name=str(inner.selected_expert_candidate["id"]),
        cage_config=cage_config,
    )
    source_training_count = int(training.sum())
    source_broad = cage_methods["global_trust_blend"][:source_training_count]
    target_broad = cage_methods["global_trust_blend"][source_training_count:]
    target_cage = cage_methods["cage_har"][source_training_count:]
    physics_reference = fit_physics_reference(
        _slice_context(kinematic, training),
        veto_quantile=float(v1_fixed["physics_veto_quantile"]),
        minimum_valid_pair_fraction=float(v1_fixed["physics_minimum_valid_pair_fraction"]),
    )
    source_physics = apply_physics_reference(_slice_context(kinematic, training), physics_reference)
    target_physics = apply_physics_reference(
        _slice_context(kinematic, evaluation), physics_reference
    )
    source_reliability = np.minimum(cage_reliability[training], source_physics.reliability)
    target_reliability = np.minimum(cage_reliability[evaluation], target_physics.reliability)
    strict_selection, strict_candidates = _select_posture_calibration(
        source_equal, labels[training], participants[training], v1_fixed
    )
    target_strict_pre_veto = _apply_calibration(target_equal, strict_selection)
    target_strict, strict_vetoed = apply_physics_veto(
        target_base, target_strict_pre_veto, target_physics.trusted
    )
    source_signatures = build_responder_signatures(
        inner.base[training],
        source_expert,
        source_ctgr,
        source_broad,
        participants[training],
        gravity_reliability=source_reliability,
        kinematic_context=_slice_context(kinematic, training),
        physics_trusted=source_physics.trusted,
        low_confidence_threshold=float(v1_fixed["responder_low_confidence_threshold"]),
    )
    target_signatures = build_responder_signatures(
        target_base,
        target_expert,
        target_ctgr,
        target_broad,
        participants[evaluation],
        gravity_reliability=target_reliability,
        kinematic_context=_slice_context(kinematic, evaluation),
        physics_trusted=target_physics.trusted,
        low_confidence_threshold=float(v1_fixed["responder_low_confidence_threshold"]),
    )
    base_utility, broad_utility = _utility_targets(
        labels[training],
        participants[training],
        inner.base[training],
        source_ctgr,
        source_broad,
        source_signatures.participant_ids,
    )
    safe_route = _controller(
        training_features=source_signatures.features,
        training_ids=source_signatures.participant_ids,
        base_utility=base_utility,
        broad_utility=broad_utility,
        evaluation_features=target_signatures.features,
        evaluation_signature_ids=target_signatures.participant_ids,
        evaluation_base=target_base,
        evaluation_pulse=target_ctgr,
        evaluation_broad=target_broad,
        evaluation_participants=participants[evaluation],
        physics_trusted=target_physics.trusted,
        fixed=v1_fixed,
        lower_bound_z=float(v1_fixed["responder_safe_lower_bound_z"]),
    )
    meta_probability, meta_state, meta_veto = _meta_training_controller(
        signatures=source_signatures.features,
        signature_ids=source_signatures.participant_ids,
        base_utility=base_utility,
        broad_utility=broad_utility,
        base=inner.base[training],
        pulse=source_ctgr,
        broad=source_broad,
        participants=participants[training],
        physics_trusted=source_physics.trusted,
        fixed=v1_fixed,
    )
    meta_restore = (meta_state == 0) | meta_veto
    full_selection, full_candidates = _select_posture_calibration(
        meta_probability,
        labels[training],
        participants[training],
        v1_fixed,
        base_probability=inner.base[training],
        restore_mask=meta_restore,
    )
    target_restore = (safe_route.state_index == 0) | safe_route.physics_vetoed
    target_v1_full = _apply_calibration(
        safe_route.probabilities,
        full_selection,
        base_probability=target_base,
        restore_mask=target_restore,
    )
    _source_equal_v2, source_weighted, source_disagreement = _marginalize(source_stack, v2_fixed)
    _target_equal_v2, target_weighted, target_disagreement = _marginalize(target_stack, v2_fixed)
    temperature, temperature_candidates = _select_temperature(
        source_weighted, labels[training], participants[training], v2_fixed
    )
    source_temperature = temperature_scale_probabilities(source_weighted, temperature=temperature)
    target_temperature = temperature_scale_probabilities(target_weighted, temperature=temperature)
    offset, offset_candidates = _select_offset(
        source_temperature, labels[training], participants[training], v2_fixed
    )
    source_core, _ = apply_rank_locked_posture_offset(
        source_temperature, posture_logit_offset=offset
    )
    target_core, _ = apply_rank_locked_posture_offset(
        target_temperature, posture_logit_offset=offset
    )
    dual_model = _fit_expert(
        "extra_trees_leaf3",
        dual.features,
        labels,
        participants,
        training,
        seed=seed + 17_001,
        n_jobs=n_jobs,
    )
    target_dual_expert = compose_mobility_posture_probabilities(
        target_base[:, 0],
        _positive_probability(dual_model, dual.features[evaluation]),
    )
    dual_selection, dual_candidates = _select_dual_candidate(
        source_core,
        inner.dual_expert[training],
        labels[training],
        participants[training],
        v2_fixed,
    )
    source_dual = _apply_dual_candidate(source_core, inner.dual_expert[training], dual_selection)
    target_dual = _apply_dual_candidate(target_core, target_dual_expert, dual_selection)
    source_route_features, route_names = build_rescue_harm_features(
        source_core,
        source_dual,
        candidate_disagreement=source_disagreement,
        physics_reliability=source_reliability,
        dual_frame_context=dual.compact_context[training],
    )
    target_route_features, target_route_names = build_rescue_harm_features(
        target_core,
        target_dual,
        candidate_disagreement=target_disagreement,
        physics_reliability=target_reliability,
        dual_frame_context=dual.compact_context[evaluation],
    )
    if route_names != target_route_names:
        raise AssertionError("zero-shot HERA-DG-v2 route feature order changed")
    target_v2_route, v2_selection, semantic_disagreements = _fit_route_lane(
        training_core=source_core,
        training_candidate=source_dual,
        training_features=source_route_features,
        training_labels=labels[training],
        training_participants=participants[training],
        training_reliability=source_reliability,
        training_trusted=source_physics.trusted,
        evaluation_core=target_core,
        evaluation_candidate=target_dual,
        evaluation_features=target_route_features,
        evaluation_reliability=target_reliability,
        evaluation_trusted=target_physics.trusted,
        fixed=v2_fixed,
        use_physics=True,
        use_support=True,
    )
    probabilities: dict[str, FloatArray] = {
        "RMRP-DG": target_base,
        "CTGR-DG": target_ctgr,
        "CAGE-DG": target_cage,
        "CTGR-DG-top3-equal": target_equal,
        "HERA-DG-strict": target_strict,
        "HERA-DG-context-safe": safe_route.probabilities,
        "HERA-DG-full": target_v1_full,
        "HERA-DG-v2-weighted": target_weighted,
        "HERA-DG-v2-core": target_core,
        "HERA-DG-v2-dual": target_dual,
        "HERA-DG-v2-full": target_v2_route.probabilities,
    }
    if include_classical:
        probabilities.update(_classical_zero_shot(source, target, seed=seed))
    record = {
        "seed": seed,
        "source_participants": list(source_partition_plan.participant_roster),
        "source_participants_with_supervision": sorted(np.unique(participants[training]).tolist()),
        "target_participants": list(target_partition_plan.participant_roster),
        "target_participants_with_candidates": sorted(np.unique(participants[evaluation]).tolist()),
        "target_participants_with_scoring": sorted(
            np.unique(scored_target.participant_ids).tolist()
        ),
        "target_inference_candidate_window_count": int(target.labels.size),
        "target_scored_window_count": int(scoring_indices.size),
        "participant_partition_plan_sha256": source_partition_plan.audit()["plan_sha256"],
        "source_participant_partition_plan_sha256": source_partition_plan.audit()["plan_sha256"],
        "target_participant_partition_plan_sha256": target_partition_plan.audit()["plan_sha256"],
        "context_event_population": "all supplied observable candidates of target participants",
        "selected_ctgr_candidate": str(inner.selected_candidate["id"]),
        "selected_cage_expert": str(inner.selected_expert_candidate["id"]),
        "top_three_hera_candidates": [
            str(item["id"]) for item in inner.top_three_expert_candidates
        ],
        "eligible_ctgr_candidate_count": len(inner.eligible_ids),
        "candidate_ranking": inner.candidate_ranking,
        "inner_folds": list(inner.folds),
        "cage_diagnostics": cage_diagnostics,
        "cage_target_route_count": int(
            np.sum(cage_arrays["cage_har_route_mask"][source_training_count:])
        ),
        "channel_scale_fit_on_source": channel_scale.tolist(),
        "physics_reference_threshold": physics_reference.normalized_residual_p90_threshold,
        "physics_target_trusted_fraction": float(target_physics.trusted.mean()),
        "hera_v1_strict_calibration": strict_selection,
        "hera_v1_strict_calibration_candidates": len(strict_candidates),
        "hera_v1_full_calibration": full_selection,
        "hera_v1_full_calibration_candidates": len(full_candidates),
        "hera_v2_temperature": temperature,
        "hera_v2_temperature_candidates": len(temperature_candidates),
        "hera_v2_posture_offset": offset,
        "hera_v2_posture_offset_candidates": len(offset_candidates),
        "hera_v2_dual_selection": dual_selection,
        "hera_v2_dual_candidates": len(dual_candidates),
        "hera_v2_route_selection": v2_selection,
        "hera_v2_semantic_source_disagreements": semantic_disagreements,
        "hera_v1_strict_target_veto_count": int(strict_vetoed.sum()),
        "real_target_labels_present_in_modelling_container": False,
        "target_labels_used_for_fit_selection_or_calibration": False,
    }
    return {method: values[scoring_indices] for method, values in probabilities.items()}, record


def evaluate_zero_shot_transfer(
    source: ExternalHARWindows,
    target: ExternalHARWindows,
    *,
    seeds: tuple[int, ...],
    repository_root: Path,
    n_jobs: int = -1,
    include_classical: bool = True,
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    source.validate()
    target.validate()
    if source.dataset_id != "imu_har_il_v1" or target.dataset_id != "fog_star_v3":
        raise ValueError("zero-shot runner is bound to IMU-HAR-IL source and FoG-STAR target")
    if source.participant_partition_plan is None or target.participant_partition_plan is None:
        raise PermissionError("zero-shot transfer requires source and target pre-window plans")
    if set(source.participant_partition_plan.participant_roster) & set(
        target.participant_partition_plan.participant_roster
    ):
        raise PermissionError("zero-shot source and target participant rosters overlap")
    if (
        source.channel_lane != "derived-gravity-9ch"
        or target.channel_lane != source.channel_lane
        or source.class_names != CORE_CLASS_NAMES
        or target.class_names != CORE_CLASS_NAMES
    ):
        raise ValueError("zero-shot transfer requires matching derived-gravity lanes")
    if (
        source.sampling_rate_hz != target.sampling_rate_hz
        or source.signals.shape[1:] != target.signals.shape[1:]
    ):
        raise ValueError("zero-shot source and target interfaces are not aligned")
    per_seed: dict[int, dict[str, FloatArray]] = {}
    seed_records: list[dict[str, Any]] = []
    for seed in seeds:
        probabilities, record = _fit_apply_seed(
            source,
            target,
            seed=seed,
            repository_root=repository_root,
            n_jobs=n_jobs,
            include_classical=include_classical,
        )
        per_seed[seed] = probabilities
        seed_records.append(record)
    ensemble = {
        method: np.mean(np.stack([per_seed[seed][method] for seed in seeds], axis=0), axis=0)
        for method in per_seed[seeds[0]]
    }
    # Numerical scoring begins here. Annotation admission was established by the
    # loader, but the full observable target pool is predicted before that subset
    # is selected; admission does not determine the participant context population.
    reports = {
        method: _report(target.labels, values, target.participant_ids)
        for method, values in ensemble.items()
    }
    comparisons = {
        method: _paired_bootstrap(reports[method], reports["RMRP-DG"], seed=20261104 + index)
        for index, method in enumerate(name for name in _INVENTION_METHODS if name != "RMRP-DG")
    }
    best_invention = max(
        _INVENTION_METHODS,
        key=lambda name: float(reports[name]["primary"]["mean_participant_macro_f1"]),
    )
    controls = [name for name in _CLASSICAL_METHODS if name in reports]
    strongest_control = (
        max(
            controls,
            key=lambda name: float(reports[name]["primary"]["mean_participant_macro_f1"]),
        )
        if controls
        else None
    )
    result = {
        "schema_version": "1.0.0",
        "experiment_id": "imu-har-il-to-fog-star-zero-shot-v1",
        "observable_context_protocol": OBSERVABLE_CONTEXT_PROTOCOL,
        "observable_context_training_protocol": OBSERVABLE_CONTEXT_TRAINING_PROTOCOL,
        "evidence_status": "EXTERNAL_ZERO_SHOT_DEVELOPMENT_NOT_CONFIRMATORY",
        "source_dataset": source.summary(),
        "target_dataset": target.summary(),
        "seeds": list(seeds),
        "reports": reports,
        "paired_participant_bootstrap_vs_rmrp": comparisons,
        "seed_records": seed_records,
        "descriptive_best_invention": best_invention,
        "descriptive_strongest_classical_control": strongest_control,
        "claim_policy": {
            "target_labels_used_for_fit_selection_or_calibration": False,
            "target_labels_read_only_after_all_probabilities_fixed": False,
            "target_annotations_preloaded_for_scoring_eligibility": True,
            "target_numeric_scoring_after_full_candidate_prediction": True,
            "participant_context_is_noncausal_and_not_matched_window_inference": True,
            "confirmatory_claim_allowed": False,
            "state_of_the_art_claim_allowed": False,
            "wear_gait_opened": False,
        },
    }
    from inclusive_shift_har.evaluation.external_statistics import seed_evidence

    result["primary_seed_averaged"], seed_predictions = seed_evidence(
        target, per_seed, primary_contrast_eligible=False
    )
    return result, {**ensemble, **seed_predictions}


def run_and_write_transfer(
    *,
    source: ExternalHARWindows,
    target: ExternalHARWindows,
    output_directory: Path,
    repository_root: Path,
    seeds: tuple[int, ...],
    n_jobs: int,
    include_classical: bool,
    inherited_launch_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if seeds != (11, 23, 47):
        raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
    if not include_classical:
        raise ValueError("publication transfer evidence requires the complete classical controls")
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
    source_summary = source.summary()
    target_summary = target.summary()
    evidence_status = _external_evidence_status(source_summary, target_summary)
    audit = {
        "schema_version": "1.0.0",
        "created_at": started,
        "source_dataset": source_summary,
        "target_dataset": target_summary,
        "artifact_evidence_status": evidence_status,
        "source_receipts": [item.to_dict() for item in source.receipts],
        "target_receipts": [item.to_dict() for item in target.receipts],
        "storage_disclosure": {
            "raw_local_mirror": False,
            "processing": "streamed provider bytes and in-memory materialization",
        },
        "source_input_manifest": source_input_manifest,
        "git_at_launch": git_at_launch,
        "publication_launch_context": launch_context_binding,
    }
    data_audit_artifact = _write_self_hashed_json_create_only(
        output_directory / "data_audit.json", audit
    )
    artifact_contract = _publication_artifact_contract(source_input_manifest, data_audit_artifact)
    try:
        result, predictions = evaluate_zero_shot_transfer(
            source,
            target,
            seeds=seeds,
            repository_root=repository_root,
            n_jobs=n_jobs,
            include_classical=include_classical,
        )
        prediction_path = output_directory / "predictions.npz"
        np.savez_compressed(
            prediction_path,
            labels=target.labels,
            participant_ids=target.participant_ids,
            session_ids=target.session_ids,
            trial_ids=target.trial_ids,
            window_ids=target.window_ids,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--source-participant-limit", type=int)
    parser.add_argument("--source-repetition-limit", type=int, default=4)
    parser.add_argument(
        "--source-selection-policy",
        choices=("complete_requested_core", "available_valid_trials"),
        default="complete_requested_core",
    )
    parser.add_argument("--target-participant-limit", type=int)
    parser.add_argument(
        "--gravity-cutoff-hz",
        type=float,
        help="Declared derived-gravity sensitivity value; defaults to the primary cutoff.",
    )
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--skip-classical", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repository_root.resolve()
    output_directory = args.output_directory.resolve()
    _launch, _manifest, launch_context = _resolve_publication_launch_context(
        repository_root=root,
        output_directory=output_directory,
        current_git_state=_git_state(root),
        current_source_manifest=_source_input_manifest(root),
        manifest_commit_validator=_source_manifest_commit_errors,
    )
    started = datetime.now(UTC).isoformat()
    stage = "configuration"
    try:
        if tuple(args.seeds) != (11, 23, 47):
            raise ValueError("external publication evidence requires frozen seeds 11, 23, and 47")
        experiment = _read_mapping(root / "configs/experiments/cross_dataset_har_rnd_v1.yaml")
        preprocessing = cast(dict[str, Any], experiment["preprocessing"])
        derived = cast(dict[str, Any], preprocessing["derived_gravity"])
        target_rate_hz = float(preprocessing["target_sampling_rate_hz"])
        window_samples = int(preprocessing["window_samples"])
        declared_cutoffs = tuple(float(value) for value in derived["cutoff_sensitivity_hz"])
        gravity_cutoff_hz = (
            float(derived["cutoff_hz"])
            if args.gravity_cutoff_hz is None
            else float(args.gravity_cutoff_hz)
        )
        if not any(np.isclose(gravity_cutoff_hz, value) for value in declared_cutoffs):
            raise ValueError(
                f"gravity cutoff {gravity_cutoff_hz} is outside the declared sensitivity set "
                f"{declared_cutoffs}"
            )
        stage = "source_dataset_acquisition"
        source = load_imu_har_il(
            participant_limit=args.source_participant_limit,
            repetition_limit=args.source_repetition_limit,
            selection_policy=args.source_selection_policy,
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        stage = "target_dataset_acquisition"
        target = load_fog_star(
            participant_limit=args.target_participant_limit,
            target_rate_hz=target_rate_hz,
            window_samples=window_samples,
            gravity_cutoff_hz=gravity_cutoff_hz,
        )
        stage = "experiment_writer"
        result = run_and_write_transfer(
            source=source,
            target=target,
            output_directory=output_directory,
            repository_root=root,
            seeds=tuple(args.seeds),
            n_jobs=args.n_jobs,
            include_classical=not args.skip_classical,
            inherited_launch_context=launch_context,
        )
    except Exception as error:
        if not output_directory.exists():
            _write_launch_failure_envelope(
                repository_root=root,
                output_directory=output_directory,
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
                name: report["mean_participant_macro_f1"]
                for name, report in result["primary_seed_averaged"]["methods"].items()
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
