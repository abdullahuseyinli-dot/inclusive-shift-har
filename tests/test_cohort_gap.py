from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.cohort_gap import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    FINAL_MODEL_IDS,
    SOURCE_PARTICIPANT_IDS,
    TARGET_PARTICIPANT_IDS,
    TARGET_SEED_ORDER,
    CohortGapError,
    build_cross_cohort_gap_report,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(values: dict[str, float], *, source: bool) -> dict[str, Any]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    if source:
        return {
            "mean": float(np.mean(array)),
            "worst": float(np.min(array)),
            "lower_decile": float(np.quantile(array, 0.1, method="linear")),
        }
    return {
        "mean_macro_f1": float(np.mean(array)),
        "worst_macro_f1": float(np.min(array)),
        "lower_decile_macro_f1": float(np.quantile(array, 0.1, method="linear")),
        "participant_values": values,
    }


def _rehash(payload: dict[str, Any], field: str) -> None:
    body = dict(payload)
    body.pop(field, None)
    payload[field] = canonical_json_sha256(body)


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, str]]:
    mapping = {
        model_id: f"configuration-{index:02d}" for index, model_id in enumerate(FINAL_MODEL_IDS)
    }
    source_models: list[dict[str, Any]] = []
    target_models: list[dict[str, Any]] = []
    for model_index, model_id in enumerate(FINAL_MODEL_IDS):
        source_values = {
            participant_id: 0.55 + participant_index * 0.02 + model_index * 0.001
            for participant_index, participant_id in enumerate(SOURCE_PARTICIPANT_IDS)
        }
        target_values = {
            participant_id: 0.35 + participant_index * 0.02 + model_index * 0.001
            for participant_index, participant_id in enumerate(TARGET_PARTICIPANT_IDS)
        }
        configuration_id = mapping[model_id]
        runner_name = f"runner-{model_index:02d}"
        source_models.append(
            {
                "configuration_id": configuration_id,
                "configuration_sha256": canonical_json_sha256({"id": configuration_id}),
                "configuration": {
                    "model_name": runner_name,
                    "seed": 11,
                    "learning_rate": 0.001,
                    "weight_decay": 0.0001,
                    "use_augmentation": False,
                    "use_content_objective": False,
                    "use_realization_factorization": False,
                    "use_group_dro": False,
                    "coral_weight": 0.0,
                    "dann_domain_loss_weight": 0.0,
                    "dann_grl_max_strength": 1.0,
                    "dann_grl_warmup_epochs": 10,
                    "zero_channel_indices": [],
                },
                "best_epoch": {"median": 5.0},
                "model_variant_id": f"variant-{model_index:02d}",
                "participant_count": len(SOURCE_PARTICIPANT_IDS),
                "participant_metrics": {"macro_f1": _summary(source_values, source=True)},
                "participants": [
                    {
                        "participant_id": participant_id,
                        "macro_f1": metric,
                        "balanced_accuracy": metric,
                        "fold_id": f"fold-{participant_index:02d}",
                        "window_count": 3,
                    }
                    for participant_index, (participant_id, metric) in enumerate(
                        source_values.items()
                    )
                ],
                "seed": 11,
            }
        )
        target_models.append(
            {
                "model_id": model_id,
                "participant_seed_averaged": _summary(target_values, source=False),
                "seed_level_primary": [
                    {
                        "seed": seed,
                        "mean_participant_macro_f1": float(np.mean(list(target_values.values()))),
                    }
                    for seed in TARGET_SEED_ORDER
                ],
                "seed_order": list(TARGET_SEED_ORDER),
                "source_result_record_sha256s": [
                    canonical_json_sha256({"model_id": model_id, "seed": seed})
                    for seed in TARGET_SEED_ORDER
                ],
            }
        )

    source: dict[str, Any] = {
        "schema_version": "1.0.0",
        "artifact_type": "source_only_grouped_cv_aggregate",
        "status": "complete_source_development_target_sealed",
        "evidence_status": "source_development_not_confirmatory",
        "selection_status": "no_model_ranking_selection_or_freeze_performed",
        "configuration_count": len(source_models),
        "quarantined_configuration_count": 0,
        "source_manifest": {"source_window_manifest_sha256": "a" * 64},
        "source_only_guarantees": {
            "aggregation_unit": "participant",
            "each_participant_held_out_once_per_aggregated_configuration": True,
            "expected_source_participants": list(SOURCE_PARTICIPANT_IDS),
            "raw_signals_or_prediction_arrays_read_by_aggregator": False,
            "target_performance_or_prediction_accessed": False,
            "target_subject_or_window_records_loaded": False,
        },
        "models": source_models,
    }
    _rehash(source, "aggregate_record_sha256")
    source_path = tmp_path / "inputs" / "source.json"
    _write_json(source_path, source)

    selection_plan: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "final_source_selection_plan",
        "status": "predeclared_source_only_target_sealed",
        "source_selection_aggregate_path": source_path.relative_to(tmp_path).as_posix(),
        "source_selection_aggregate_record_sha256": source["aggregate_record_sha256"],
        "required_seed_order": list(TARGET_SEED_ORDER),
        "final_epoch_rule": "lower integer median of five grouped source-fold best epochs",
        "common_neural_runner_arguments": {
            "weight_decay": 0.0001,
            "use_augmentation": False,
            "use_content_objective": False,
            "use_realization_factorization": False,
            "use_group_dro": False,
            "coral_weight": 0.0,
            "dann_domain_loss_weight": 0.0,
            "dann_grl_max_strength": 1.0,
            "dann_grl_warmup_epochs": 10,
            "zero_channel_indices": [],
        },
        "models": [
            {
                "model_id": model_id,
                "runner_model_name": f"runner-{model_index:02d}",
                "training_regime": "fixed_epoch_neural",
                "runner_arguments": {"epochs": 5, "learning_rate": 0.001},
                "configuration_expectations": {"model_name": f"runner-{model_index:02d}"},
            }
            for model_index, model_id in enumerate(FINAL_MODEL_IDS)
        ],
        "target_subject_or_window_records_used": False,
        "target_predictions_or_performance_accessed": False,
    }
    _rehash(selection_plan, "selection_plan_sha256")
    selection_plan_path = tmp_path / "inputs" / "selection.json"
    _write_json(selection_plan_path, selection_plan)

    target: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_participant_statistics",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "participant_count": len(TARGET_PARTICIPANT_IDS),
        "statistical_unit": "participant",
        "required_seed_order": list(TARGET_SEED_ORDER),
        "locked_target_index_record_sha256": "d" * 64,
        "final_freeze_inventory_sha256": "c" * 64,
        "target_seal_id": "e" * 64,
        "analysis_plan_sha256": "b" * 64,
        "target_information_used_for_model_selection": False,
        "models": target_models,
    }
    _rehash(target, "record_sha256")
    target_path = tmp_path / "inputs" / "target.json"
    _write_json(target_path, target)

    specification: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "cross_cohort_gap_analysis_specification",
        "status": "locked_post_confirmatory_descriptive_analysis",
        "analysis_id": "test_cross_cohort_gap_v1",
        "inputs": {
            "source": {
                "path": source_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_file(source_path),
                "aggregate_record_sha256": source["aggregate_record_sha256"],
            },
            "target": {
                "path": target_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_file(target_path),
                "record_sha256": target["record_sha256"],
            },
        },
        "lineage": {
            "source_window_manifest_sha256": "a" * 64,
            "analysis_plan_sha256": "b" * 64,
            "final_freeze_inventory_sha256": "c" * 64,
            "locked_target_index_record_sha256": "d" * 64,
            "target_seal_id": "e" * 64,
            "mapping_basis": {
                "path": selection_plan_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_file(selection_plan_path),
                "selection_plan_sha256": selection_plan["selection_plan_sha256"],
            },
        },
        "source_participant_ids": list(SOURCE_PARTICIPANT_IDS),
        "target_participant_ids": list(TARGET_PARTICIPANT_IDS),
        "source_seed": 11,
        "target_seed_order": list(TARGET_SEED_ORDER),
        "bootstrap": {
            "method": "independent_participant_cluster_percentile",
            "confidence": 0.95,
            "resamples": BOOTSTRAP_RESAMPLES,
            "seed": BOOTSTRAP_SEED,
            "independent_cohort_draws": True,
            "shared_indices_across_models": True,
        },
        "model_to_source_configuration": mapping,
        "limitations": [
            "Source selection is consumed.",
            "Source and target seed and training regimes differ.",
        ],
        "target_information_used_to_choose_mapping": False,
    }
    _rehash(specification, "specification_sha256")
    specification_path = tmp_path / "inputs" / "mapping.json"
    _write_json(specification_path, specification)
    return source_path, target_path, specification_path, mapping


def _build(
    tmp_path: Path, source_path: Path, target_path: Path, specification_path: Path
) -> dict[str, Any]:
    return build_cross_cohort_gap_report(
        source_path,
        target_path,
        specification_path,
        output_root=tmp_path,
        destination="outputs/gaps.json",
        csv_destination="outputs/gaps.csv",
        markdown_destination="outputs/gaps.md",
    )


def test_cross_cohort_report_is_deterministic_hashed_and_descriptive(tmp_path: Path) -> None:
    source_path, target_path, specification_path, mapping = _fixture(tmp_path)

    result = _build(tmp_path, source_path, target_path, specification_path)

    assert result["model_count"] == 20
    assert result["evidence_status"] == "descriptive_post_confirmatory_cross_cohort_comparison"
    assert result["interpretation_status"] == "descriptive_only_not_matched_confirmatory_or_causal"
    assert result["raw_data_or_prediction_arrays_read"] is False
    assert result["model_to_source_configuration"] == mapping
    first = result["models"][0]
    assert first["gaps"]["source_mean_minus_target_mean_macro_f1"] == pytest.approx(0.2)

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    source_indices = rng.integers(0, 10, size=(BOOTSTRAP_RESAMPLES, 10))
    target_indices = rng.integers(0, 10, size=(BOOTSTRAP_RESAMPLES, 10))
    source_values = np.asarray(
        list(first["source_participant_macro_f1"]["participant_values"].values())
    )
    target_values = np.asarray(
        list(first["target_participant_macro_f1"]["participant_values"].values())
    )
    expected = source_values[source_indices].mean(axis=1) - target_values[target_indices].mean(
        axis=1
    )
    interval = first["gaps"]["mean_gap_bootstrap_interval"]
    assert interval["lower"] == pytest.approx(np.quantile(expected, 0.025, method="linear"))
    assert interval["upper"] == pytest.approx(np.quantile(expected, 0.975, method="linear"))

    published = load_json_strict(tmp_path / "outputs" / "gaps.json")
    assert isinstance(published, dict)
    unhashed = dict(published)
    recorded = unhashed.pop("record_sha256")
    assert recorded == canonical_json_sha256(unhashed)
    for role, suffix in (("csv", "csv"), ("markdown", "md")):
        assert result["output_artifacts"][role]["sha256"] == sha256_file(
            tmp_path / "outputs" / f"gaps.{suffix}"
        )
    markdown = (tmp_path / "outputs" / "gaps.md").read_text(encoding="utf-8")
    assert "descriptive post-confirmatory" in markdown
    assert "does not make the regimes comparable" in markdown


def test_cross_cohort_report_refuses_all_overwrites(tmp_path: Path) -> None:
    source_path, target_path, specification_path, _ = _fixture(tmp_path)
    _build(tmp_path, source_path, target_path, specification_path)

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _build(tmp_path, source_path, target_path, specification_path)


@pytest.mark.parametrize(
    ("input_index", "hash_field", "message"),
    [
        (0, "aggregate_record_sha256", "source aggregate self-hash"),
        (1, "record_sha256", "target statistics self-hash"),
    ],
)
def test_cross_cohort_report_rejects_mutated_input_self_hashes(
    tmp_path: Path, input_index: int, hash_field: str, message: str
) -> None:
    source_path, target_path, specification_path, _ = _fixture(tmp_path)
    chosen = (source_path, target_path)[input_index]
    record = json.loads(chosen.read_text(encoding="utf-8"))
    record[hash_field] = "0" * 64
    _write_json(chosen, record)

    with pytest.raises(CohortGapError, match=message):
        _build(tmp_path, source_path, target_path, specification_path)


def test_cross_cohort_report_requires_exact_mapping_and_participant_sets(tmp_path: Path) -> None:
    source_path, target_path, specification_path, _ = _fixture(tmp_path)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    specification["model_to_source_configuration"].pop(FINAL_MODEL_IDS[-1])
    _rehash(specification, "specification_sha256")
    _write_json(specification_path, specification)

    with pytest.raises(CohortGapError, match="exact 20 final model IDs"):
        _build(tmp_path, source_path, target_path, specification_path)

    source_path, target_path, specification_path, _ = _fixture(tmp_path / "participants")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["source_only_guarantees"]["expected_source_participants"][-1] = "99"
    _rehash(source, "aggregate_record_sha256")
    _write_json(source_path, source)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    specification["inputs"]["source"]["file_sha256"] = sha256_file(source_path)
    specification["inputs"]["source"]["aggregate_record_sha256"] = source["aggregate_record_sha256"]
    _rehash(specification, "specification_sha256")
    _write_json(specification_path, specification)

    with pytest.raises(
        CohortGapError,
        match=r"selection plan does not bind|exact source participant set",
    ):
        _build(tmp_path / "participants", source_path, target_path, specification_path)


def test_cross_cohort_report_rejects_mapping_permutation_even_when_rehashed(
    tmp_path: Path,
) -> None:
    source_path, target_path, specification_path, _ = _fixture(tmp_path)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    first, second = FINAL_MODEL_IDS[:2]
    mapping = specification["model_to_source_configuration"]
    mapping[first], mapping[second] = mapping[second], mapping[first]
    _rehash(specification, "specification_sha256")
    _write_json(specification_path, specification)

    with pytest.raises(CohortGapError, match="differs from the source-only selection plan"):
        _build(tmp_path, source_path, target_path, specification_path)


def test_cross_cohort_report_validates_target_lineage(tmp_path: Path) -> None:
    source_path, target_path, specification_path, _ = _fixture(tmp_path)
    target = json.loads(target_path.read_text(encoding="utf-8"))
    target["analysis_plan_sha256"] = "9" * 64
    _rehash(target, "record_sha256")
    _write_json(target_path, target)
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    specification["inputs"]["target"]["file_sha256"] = sha256_file(target_path)
    specification["inputs"]["target"]["record_sha256"] = target["record_sha256"]
    _rehash(specification, "specification_sha256")
    _write_json(specification_path, specification)

    with pytest.raises(CohortGapError, match="analysis_plan_sha256 lineage changed"):
        _build(tmp_path, source_path, target_path, specification_path)
