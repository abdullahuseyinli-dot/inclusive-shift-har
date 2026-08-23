from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.artifacts.final_freeze import (
    FrozenModelInput,
    build_final_freeze_inventory,
    record_target_opening_once,
    validate_exact_target_unlock,
    validate_final_freeze_inventory_file,
    write_final_freeze_inventory_new,
)
from inclusive_shift_har.evaluation.source_calibration import (
    build_source_temperature_calibrator_record,
    write_source_temperature_calibrator_new,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.protocols.seal import TargetSealError


def _target_manifest(*, split_hash_placeholder: str = "placeholder") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "1.0.0",
        "manifest_kind": "released_block_split",
        "target_performance_or_prediction_accessed": False,
        "target_seal": {
            "maximum_confirmatory_openings": 1,
            "performance_inspection": "forbidden",
            "seal_id": "e" * 64,
            "status": "sealed",
            "subject_ids": ["synthetic-target"],
            "unlock_record": None,
        },
        "windows": [{"partition": "target_sealed", "window_id": "metadata-only"}],
        "split_manifest_sha256": split_hash_placeholder,
    }
    payload.pop("split_manifest_sha256")
    payload["split_manifest_sha256"] = canonical_json_sha256(payload)
    return payload


def _freeze_fixture(tmp_path: Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    checkpoint = tmp_path / "runs" / "model-seed11.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic immutable checkpoint bytes\n")
    configuration = {
        "model_name": "synthetic_neural",
        "seed": 11,
        "epochs": 7,
        "checkpoint_selection_rule": "fixed_last_epoch",
    }
    target_manifest = _target_manifest()
    calibration = build_source_temperature_calibrator_record(
        np.asarray([[2.0, 0.0], [0.0, 2.0]], dtype=np.float64),
        np.asarray([0, 1], dtype=np.int64),
        ["source-validation-1", "source-validation-2"],
        fit_partition="source_validation",
        checkpoint_sha256=sha256_file(checkpoint),
        training_configuration_sha256=canonical_json_sha256(configuration),
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        class_names=("mobility", "sitting"),
    )
    calibrator_path = tmp_path / "runs" / "model-seed11.calibrator.json"
    write_source_temperature_calibrator_new(
        calibration,
        calibrator_path,
        allowed_root=tmp_path,
    )
    inventory = build_final_freeze_inventory(
        [
            FrozenModelInput(
                model_id="synthetic_neural",
                seed=11,
                training_regime="fixed_epoch_neural",
                training_configuration=configuration,
                checkpoint_path=checkpoint,
                calibrator_path=calibrator_path,
                selected_epoch=7,
            )
        ],
        artifact_root=tmp_path,
        created_at_utc="2099-01-01T00:00:00Z",
        code_commit="synthetic-code-commit",
        dataset_manifest_sha256="a" * 64,
        source_window_manifest_sha256="b" * 64,
        split_manifest_sha256=target_manifest["split_manifest_sha256"],
        protocol_lock_sha256="d" * 64,
        target_seal_id="e" * 64,
        required_seed_order=[11],
    )
    inventory_path = tmp_path / "freeze.json"
    write_final_freeze_inventory_new(inventory, inventory_path, allowed_root=tmp_path)
    return inventory, inventory_path, target_manifest


def _unlock(inventory: dict[str, Any], inventory_path: Path) -> dict[str, Any]:
    return {
        "approved_at_utc": "2099-01-01T01:00:00Z",
        "code_commit": "synthetic-code-commit",
        "final_gates": {
            "artifact_validation_passed": True,
            "configuration_validation_passed": True,
            "manifest_validation_passed": True,
            "protocol_lock_present": True,
            "split_audit_passed": True,
            "tests_passed": True,
            "type_checks_passed": True,
            "working_tree_clean_or_documented": True,
        },
        "gate": "final_evaluation_unlock",
        "protocol_lock_sha256": "d" * 64,
        "reason": "locked_confirmatory_evaluation",
        "schema_version": "1.0.0",
        "split_manifest_sha256": inventory["split_manifest_sha256"],
        "status": "approved",
        "target_opening_number": 1,
        "target_performance_previously_accessed": False,
        "target_seal_id": "e" * 64,
        "final_freeze_inventory_sha256": inventory["inventory_sha256"],
        "final_freeze_file_sha256": sha256_file(inventory_path),
        "frozen_artifact_set_sha256": inventory["frozen_artifact_set_sha256"],
        "source_models_and_calibrators_frozen": True,
    }


def test_exact_unlock_refuses_target_manifest_until_freeze_is_bound(tmp_path: Path) -> None:
    inventory, inventory_path, target_manifest = _freeze_fixture(tmp_path)
    report = validate_final_freeze_inventory_file(inventory_path, artifact_root=tmp_path)
    assert report.valid, report.to_dict()
    with pytest.raises(TargetSealError, match="remains sealed"):
        validate_exact_target_unlock(
            target_manifest,
            None,
            final_freeze_inventory_path=inventory_path,
            artifact_root=tmp_path,
        )
    incomplete = _unlock(inventory, inventory_path)
    incomplete.pop("frozen_artifact_set_sha256")
    with pytest.raises(TargetSealError, match="exact frozen artifacts"):
        validate_exact_target_unlock(
            target_manifest,
            incomplete,
            final_freeze_inventory_path=inventory_path,
            artifact_root=tmp_path,
        )
    authorized = validate_exact_target_unlock(
        target_manifest,
        _unlock(inventory, inventory_path),
        final_freeze_inventory_path=inventory_path,
        artifact_root=tmp_path,
    )
    assert authorized["target_opening_number"] == 1


def test_freeze_detects_checkpoint_mutation(tmp_path: Path) -> None:
    _, inventory_path, _ = _freeze_fixture(tmp_path)
    checkpoint = tmp_path / "runs" / "model-seed11.pt"
    checkpoint.write_bytes(b"mutated checkpoint bytes retained for test\n")
    report = validate_final_freeze_inventory_file(inventory_path, artifact_root=tmp_path)
    assert report.valid is False
    assert "metadata mismatch" in report.errors[0].message


def test_exact_unlock_can_be_consumed_only_once(tmp_path: Path) -> None:
    inventory, inventory_path, target_manifest = _freeze_fixture(tmp_path)
    unlock = _unlock(inventory, inventory_path)
    receipt_path = tmp_path / "opening-receipt.json"
    receipt = record_target_opening_once(
        target_manifest,
        unlock,
        final_freeze_inventory_path=inventory_path,
        artifact_root=tmp_path,
        opening_receipt_path=receipt_path,
        receipt_root=tmp_path,
        opened_at_utc="2099-01-01T02:00:00Z",
        machine_record_sha256="f" * 64,
    )
    assert receipt["target_signals_materialized_at_receipt_time"] is False
    with pytest.raises(TargetSealError, match="already has an opening receipt"):
        record_target_opening_once(
            target_manifest,
            unlock,
            final_freeze_inventory_path=inventory_path,
            artifact_root=tmp_path,
            opening_receipt_path=receipt_path,
            receipt_root=tmp_path,
            opened_at_utc="2099-01-01T02:00:01Z",
            machine_record_sha256="f" * 64,
        )


def test_freeze_rejects_non_last_epoch_configuration(tmp_path: Path) -> None:
    inventory, _, target_manifest = _freeze_fixture(tmp_path)
    bad = deepcopy(inventory["models"][0]["training_configuration"])
    bad["checkpoint_selection_rule"] = "source_validation_best"
    checkpoint = tmp_path / inventory["models"][0]["checkpoint"]["path"]
    calibrator = tmp_path / inventory["models"][0]["calibrator"]["path"]
    with pytest.raises(ValueError, match="fixed_last_epoch"):
        build_final_freeze_inventory(
            [
                FrozenModelInput(
                    model_id="synthetic_neural",
                    seed=11,
                    training_regime="fixed_epoch_neural",
                    training_configuration=bad,
                    checkpoint_path=checkpoint,
                    calibrator_path=calibrator,
                    selected_epoch=7,
                )
            ],
            artifact_root=tmp_path,
            created_at_utc="2099-01-01T00:00:00Z",
            code_commit="synthetic-code-commit",
            dataset_manifest_sha256="a" * 64,
            source_window_manifest_sha256="b" * 64,
            split_manifest_sha256=target_manifest["split_manifest_sha256"],
            protocol_lock_sha256="d" * 64,
            target_seal_id="e" * 64,
            required_seed_order=[11],
        )
