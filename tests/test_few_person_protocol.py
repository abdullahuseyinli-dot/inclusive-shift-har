from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from inclusive_shift_har.protocols.config import load_protocol_spec
from inclusive_shift_har.protocols.few_person import (
    FEW_PERSON_EVIDENCE_STATUS,
    FewPersonProtocolError,
    build_few_person_manifest,
    write_few_person_manifest_new,
)


def _write_hashed(path: Path, payload: dict[str, Any], field: str) -> Path:
    payload[field] = canonical_json_sha256(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def materialize_few_person_inputs(
    tmp_path: Path, config_path: Path
) -> tuple[Path, Path, Path, Path]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_folds: list[dict[str, Any]] = []
    for fold in config["outer_folds"]:
        scenarios = []
        evaluation = list(fold["evaluation_subjects"])
        for k in (1, 2, 4):
            inclusion = list(fold["inclusion_order"][:k])
            unused = sorted(
                set(str(value) for value in range(11, 21)) - set(evaluation) - set(inclusion),
                key=int,
            )
            scenarios.append(
                {
                    "k": k,
                    "evaluation_subjects": evaluation,
                    "evaluation_window_count": 2,
                    "evaluation_window_ids_sha256": canonical_json_sha256(
                        [f"eval-{value}" for value in evaluation]
                    ),
                    "normalization_fit_subjects": [
                        "1",
                        "2",
                        "3",
                        "4",
                        "5",
                        "6",
                        "7",
                        "9",
                    ],
                    "target_inclusion_subjects": inclusion,
                    "target_inclusion_window_count": k,
                    "target_inclusion_window_ids_sha256": canonical_json_sha256(
                        [f"train-{value}" for value in inclusion]
                    ),
                    "unused_target_subjects": unused,
                }
            )
        split_folds.append(
            {
                "fold_id": fold["fold_id"],
                "inclusion_order": fold["inclusion_order"],
                "scenarios": scenarios,
            }
        )
    split = {
        "schema_version": "1.0.0",
        "target_seal": {"seal_id": "e" * 64},
        "few_person_outer_folds": split_folds,
        "source_artifact_sha256": "a" * 64,
        "windows": [],
    }
    split_path = _write_hashed(tmp_path / "split.json", split, "split_manifest_sha256")
    receipt = {
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "target_seal_id": "e" * 64,
    }
    receipt_path = _write_hashed(tmp_path / "receipt.json", receipt, "record_sha256")
    zero_shot = {
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "target_information_used_for_model_selection": False,
        "split_manifest_sha256": split["split_manifest_sha256"],
        "target_seal_id": "e" * 64,
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opaque_result_not_for_design": {"metric_sentinel": 0.987654321},
    }
    zero_path = _write_hashed(tmp_path / "zero.json", zero_shot, "record_sha256")
    training_configuration = {
        "model_name": "compact_residual_96",
        "num_classes": 3,
        "seed": 11,
        "epochs": 2,
        "batch_size": 4,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "patience": 1,
        "minimum_epochs": 1,
        "gradient_clip_norm": 5.0,
        "mixed_precision": "float16",
        "data_loader_workers": 0,
        "checkpoint_interval": 2,
        "checkpoint_selection_rule": "fixed_last_epoch",
        "disable_cudnn": False,
        "use_augmentation": False,
        "use_content_objective": False,
        "use_realization_factorization": False,
        "use_group_dro": False,
        "coral_weight": 0.0,
        "zero_channel_indices": [],
    }
    inventory = {
        "record_kind": "final_source_artifact_freeze",
        "status": "frozen_before_target_unlock",
        "evidence_status": "source_only_final_models_target_sealed",
        "split_manifest_sha256": split["split_manifest_sha256"],
        "models": [
            {
                "model_id": "compact-erm",
                "seed": 11,
                "selected_epoch": 2,
                "training_regime": "fixed_epoch_neural",
                "training_configuration": training_configuration,
                "training_configuration_sha256": canonical_json_sha256(training_configuration),
                "checkpoint": {"path": "checkpoint.pt", "sha256": "b" * 64},
                "calibrator": {"path": "calibrator.json", "sha256": "c" * 64},
                "calibrator_record_sha256": "d" * 64,
            },
            {
                "model_id": "classical-excluded",
                "seed": 11,
                "training_regime": "deterministic_classical",
            },
        ],
    }
    inventory_path = _write_hashed(tmp_path / "freeze.json", inventory, "inventory_sha256")
    return split_path, receipt_path, zero_path, inventory_path


def test_build_is_deterministic_nested_disjoint_and_metric_blind(
    tmp_path: Path, repository_root: Path
) -> None:
    config_path = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    split, receipt, zero, inventory = materialize_few_person_inputs(tmp_path, config_path)

    first = build_few_person_manifest(
        split_manifest_path=split,
        config_path=config_path,
        opening_receipt_path=receipt,
        zero_shot_index_path=zero,
        final_freeze_inventory_path=inventory,
    )
    second = build_few_person_manifest(
        split_manifest_path=split,
        config_path=config_path,
        opening_receipt_path=receipt,
        zero_shot_index_path=zero,
        final_freeze_inventory_path=inventory,
    )

    assert first == second
    assert first["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
    assert first["evidence_status"] == FEW_PERSON_EVIDENCE_STATUS
    assert len(first["scenarios"]) == 15
    assert [model["model_id"] for model in first["models"]] == ["compact-erm"]
    assert "metric_sentinel" not in json.dumps(first)
    for fold_id in {scenario["fold_id"] for scenario in first["scenarios"]}:
        fold = [scenario for scenario in first["scenarios"] if scenario["fold_id"] == fold_id]
        by_k = {scenario["k"]: scenario for scenario in fold}
        assert set(by_k[1]["target_inclusion_subjects"]) <= set(
            by_k[2]["target_inclusion_subjects"]
        )
        assert set(by_k[2]["target_inclusion_subjects"]) <= set(
            by_k[4]["target_inclusion_subjects"]
        )
        for scenario in fold:
            assert set(scenario["target_inclusion_subjects"]).isdisjoint(
                scenario["evaluation_subjects"]
            )
            assert scenario["validation_subjects"] == []
            assert scenario["calibration_subjects"] == []


def test_dedicated_config_exactly_reuses_locked_v1_2_outer_folds(
    repository_root: Path,
) -> None:
    config = yaml.safe_load(
        (repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    parent = load_protocol_spec(
        repository_root / "configs/protocols/inclusivehar_released_block_v1_2.yaml"
    )

    assert config["k_values"] == list(parent.few_person_k_values)
    assert [fold["fold_id"] for fold in config["outer_folds"]] == [
        fold.fold_id for fold in parent.few_person_outer_folds
    ]
    for configured, locked in zip(
        config["outer_folds"], parent.few_person_outer_folds, strict=True
    ):
        assert configured["evaluation_subjects"] == list(locked.evaluation_subjects)
        assert configured["inclusion_order"] == list(locked.inclusion_order)


def test_manifest_publication_is_create_only(tmp_path: Path, repository_root: Path) -> None:
    config_path = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    split, receipt, zero, inventory = materialize_few_person_inputs(tmp_path, config_path)
    manifest = build_few_person_manifest(
        split_manifest_path=split,
        config_path=config_path,
        opening_receipt_path=receipt,
        zero_shot_index_path=zero,
        final_freeze_inventory_path=inventory,
    )
    output = tmp_path / "few-person.json"

    write_few_person_manifest_new(manifest, output, allowed_root=tmp_path)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        write_few_person_manifest_new(manifest, output, allowed_root=tmp_path)


def test_tampered_zero_shot_completion_is_rejected(tmp_path: Path, repository_root: Path) -> None:
    config_path = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    split, receipt, zero, inventory = materialize_few_person_inputs(tmp_path, config_path)
    payload = json.loads(zero.read_text(encoding="utf-8"))
    payload["status"] = "incomplete"
    zero.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(FewPersonProtocolError, match="self-hash"):
        build_few_person_manifest(
            split_manifest_path=split,
            config_path=config_path,
            opening_receipt_path=receipt,
            zero_shot_index_path=zero,
            final_freeze_inventory_path=inventory,
        )
