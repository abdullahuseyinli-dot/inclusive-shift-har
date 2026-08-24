from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
import torch

import inclusive_shift_har.experiments.postconfirmatory_sensor_stress as stress_runner
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    PostconfirmatoryInputError,
    load_consumed_target_context,
    load_materialized_cache_evidence,
    load_target_clean_reference,
)
from inclusive_shift_har.evaluation.sensor_reliability import (
    load_sensor_reliability_config,
)
from inclusive_shift_har.experiments.postconfirmatory_efficiency import (
    PostconfirmatoryEfficiencyError,
    run_frozen_efficiency_profiles,
)
from inclusive_shift_har.experiments.postconfirmatory_efficiency import (
    build_parser as build_efficiency_parser,
)
from inclusive_shift_har.experiments.postconfirmatory_sensor_stress import (
    PostconfirmatoryStressError,
    run_postconfirmatory_sensor_stress,
)
from inclusive_shift_har.experiments.postconfirmatory_sensor_stress import (
    build_parser as build_stress_parser,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

CLASSES = ("mobility", "sitting", "standing")


def _write_record(path: Path, record: dict[str, Any], *, field: str = "record_sha256") -> None:
    record[field] = canonical_json_sha256(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, sort_keys=True) + "\n", encoding="utf-8")


def _consumed_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    split_hash = "a" * 64
    seal_id = "b" * 64
    freeze_hash = "c" * 64
    receipt_path = tmp_path / "protocol" / "confirmatory_target_opening_1.json"
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
        "split_manifest_sha256": split_hash,
        "target_seal_id": seal_id,
    }
    _write_record(receipt_path, receipt)

    window_ids = ("target-window-0", "target-window-1", "target-window-2")
    participant_ids = ("11", "12", "12")
    labels = np.asarray([0, 1, 2], dtype=np.int64)
    probabilities = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]], dtype=np.float64
    )
    logits = np.log(probabilities)
    prediction_path = tmp_path / "confirmatory" / "compact--seed-11.predictions.npz"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            evidence_status=np.asarray(["locked_confirmatory_target_opening_1"]),
            window_ids=np.asarray(window_ids),
            participant_ids=np.asarray(participant_ids),
            true_labels=labels,
            logits=logits,
            uncalibrated_probabilities=probabilities,
            calibrated_probabilities=probabilities,
            predicted_labels=labels,
        )
    result_path = tmp_path / "confirmatory" / "compact--seed-11.result.json"
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_per_seed_result",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "model_id": "compact",
        "seed": 11,
        "class_names": list(CLASSES),
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "final_freeze_inventory_sha256": freeze_hash,
        "prediction_array": {
            "path": prediction_path.relative_to(tmp_path).as_posix(),
            "sha256": sha256_file(prediction_path),
            "format": "npz",
        },
        "participant_level_report": classification_report(
            labels, probabilities, participant_ids, class_names=CLASSES
        ),
        "target_information_used_for_model_selection": False,
    }
    _write_record(result_path, result)
    index_path = tmp_path / "confirmatory" / "locked_target_evaluation_index.json"
    index: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "locked_target_evaluation_index",
        "status": "complete_create_only",
        "evidence_status": "locked_confirmatory_target_opening_1",
        "target_opening_number": 1,
        "split_manifest_sha256": split_hash,
        "target_seal_id": seal_id,
        "final_freeze_inventory_sha256": freeze_hash,
        "opening_receipt_record_sha256": receipt["record_sha256"],
        "opening_receipt_file_sha256": sha256_file(receipt_path),
        "class_names": list(CLASSES),
        "model_seed_result_count": 1,
        "results": [
            {
                "model_id": "compact",
                "seed": 11,
                "array_path": prediction_path.relative_to(tmp_path).as_posix(),
                "array_sha256": sha256_file(prediction_path),
                "record_path": result_path.relative_to(tmp_path).as_posix(),
                "record_file_sha256": sha256_file(result_path),
                "record_sha256": result["record_sha256"],
            }
        ],
        "target_information_used_for_model_selection": False,
    }
    _write_record(index_path, index)
    target_cache = tmp_path / "cache" / "target.npz"
    target_cache.parent.mkdir(parents=True, exist_ok=True)
    signals = np.zeros((3, 128, 6), dtype=np.float32)
    with target_cache.open("xb") as stream:
        np.savez_compressed(
            stream,
            signals=signals,
            labels=labels,
            window_ids=np.asarray(window_ids),
            participant_ids=np.asarray(participant_ids),
            released_labels=np.asarray(["Walking", "Sitting", "Standing"]),
            partitions=np.asarray(["target_sealed"] * 3),
        )
    target_metadata = tmp_path / "cache" / "target.json"
    target_metadata.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "cache_key": "d" * 64,
                "array_sha256": sha256_file(target_cache),
                "window_count": 3,
                "shape": [3, 128, 6],
                "class_names": list(CLASSES),
                "ontology_track": "functional_core",
                "lineage": {
                    "split_manifest_sha256": split_hash,
                    "target_seal_id": seal_id,
                    "opening_receipt_record_sha256": receipt["record_sha256"],
                },
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return receipt_path, index_path, target_cache, target_metadata


def test_consumed_target_cache_matches_opening_one_without_unlock(tmp_path: Path) -> None:
    receipt, index, cache, metadata = _consumed_fixture(tmp_path)
    context = load_consumed_target_context(
        opening_receipt_path=receipt,
        locked_target_index_path=index,
        artifact_root=tmp_path,
    )
    evidence = load_materialized_cache_evidence(
        array_path=cache,
        metadata_path=metadata,
        expected_array_sha256=sha256_file(cache),
        expected_metadata_sha256=sha256_file(metadata),
        artifact_root=tmp_path,
        expected_partition="target_sealed",
        expected_split_manifest_sha256="a" * 64,
        expected_class_names=CLASSES,
        target_context=context,
    )
    assert evidence.batch.window_ids == (
        "target-window-0",
        "target-window-1",
        "target-window-2",
    )
    reference = load_target_clean_reference(context, model_id="compact", seed=11)
    assert (
        reference.prediction_path
        == (tmp_path / "confirmatory/compact--seed-11.predictions.npz").resolve()
    )
    assert reference.prediction_sha256 == sha256_file(
        tmp_path / "confirmatory/compact--seed-11.predictions.npz"
    )


def test_target_cache_alignment_change_is_rejected(tmp_path: Path) -> None:
    receipt, index, cache, metadata = _consumed_fixture(tmp_path)
    context = load_consumed_target_context(
        opening_receipt_path=receipt,
        locked_target_index_path=index,
        artifact_root=tmp_path,
    )
    with np.load(cache, allow_pickle=False) as arrays:
        payload = {key: np.asarray(arrays[key]) for key in arrays.files}
    payload["window_ids"] = np.asarray(["changed", "target-window-1", "target-window-2"])
    changed = tmp_path / "cache" / "changed.npz"
    with changed.open("xb") as stream:
        np.savez_compressed(stream, **cast(dict[str, Any], payload))
    metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
    metadata_value["array_sha256"] = sha256_file(changed)
    changed_metadata = tmp_path / "cache" / "changed.json"
    changed_metadata.write_text(json.dumps(metadata_value, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(PostconfirmatoryInputError, match="identity differs"):
        load_materialized_cache_evidence(
            array_path=changed,
            metadata_path=changed_metadata,
            expected_array_sha256=sha256_file(changed),
            expected_metadata_sha256=sha256_file(changed_metadata),
            artifact_root=tmp_path,
            expected_partition="target_sealed",
            expected_split_manifest_sha256="a" * 64,
            expected_class_names=CLASSES,
            target_context=context,
        )


def test_operational_runners_refuse_before_opening_any_path_without_cuda(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    missing = tmp_path / "missing"
    with pytest.raises(PostconfirmatoryEfficiencyError, match="before any frozen checkpoint"):
        run_frozen_efficiency_profiles(
            profile_config_path=missing,
            final_freeze_inventory_path=missing,
            opening_receipt_path=missing,
            locked_target_index_path=missing,
            artifact_root=missing,
            output_directory=missing,
            output_root=missing,
            created_at_utc="2099-01-01T00:00:00Z",
            device=torch.device("cuda"),
        )
    with pytest.raises(PostconfirmatoryStressError, match="before any cache"):
        run_postconfirmatory_sensor_stress(
            stress_config_path=missing,
            final_freeze_inventory_path=missing,
            opening_receipt_path=missing,
            locked_target_index_path=missing,
            primary_cache_record_path=missing,
            expected_primary_cache_record_file_sha256="a" * 64,
            artifact_root=missing,
            output_directory=missing,
            output_root=missing,
            created_at_utc="2099-01-01T00:00:00Z",
            device=torch.device("cuda"),
        )


def test_stress_runner_preserves_failure_record_and_failed_index_after_output_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, repository_root: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_context_load(**_: Any) -> None:
        raise RuntimeError("synthetic consumed-context failure")

    monkeypatch.setattr(stress_runner, "load_consumed_target_context", fail_context_load)
    output = tmp_path / "stress"
    with pytest.raises(PostconfirmatoryStressError, match="create-only failure evidence preserved"):
        run_postconfirmatory_sensor_stress(
            stress_config_path=(
                repository_root / "configs/experiments/sensor_reliability_stress_v1.yaml"
            ),
            final_freeze_inventory_path=tmp_path / "missing-freeze.json",
            opening_receipt_path=tmp_path / "missing-receipt.json",
            locked_target_index_path=tmp_path / "missing-index.json",
            primary_cache_record_path=tmp_path / "missing-cache.json",
            expected_primary_cache_record_file_sha256="a" * 64,
            artifact_root=tmp_path,
            output_directory=output,
            output_root=tmp_path,
            created_at_utc="2099-01-01T00:00:00Z",
            device=torch.device("cuda"),
        )

    failure_path = output / "failure.json"
    index_path = output / "sensor_reliability_stress_index.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    index = json.loads(index_path.read_text(encoding="utf-8"))
    failure_body = dict(failure)
    failure_hash = failure_body.pop("record_sha256")
    index_body = dict(index)
    index_hash = index_body.pop("record_sha256")

    assert failure_hash == canonical_json_sha256(failure_body)
    assert index_hash == canonical_json_sha256(index_body)
    assert failure["status"] == "failed_preserved_create_only"
    assert failure["execution_stage"] == "consumed_opening_context_validation"
    assert failure["partial_outputs_preserved"] is True
    assert failure["opening_or_unlock_invoked"] is False
    assert index["status"] == "failed_preserved_create_only"
    assert index["stress_result_count"] == 0
    assert index["failure"]["record_sha256"] == failure["record_sha256"]
    assert index["failure"]["file_sha256"] == sha256_file(failure_path)

    failure_file_hash = sha256_file(failure_path)
    index_file_hash = sha256_file(index_path)
    config = load_sensor_reliability_config(
        repository_root / "configs/experiments/sensor_reliability_stress_v1.yaml"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite stress failure evidence"):
        stress_runner._publish_stress_failure_artifacts(
            output=output,
            output_root=tmp_path,
            artifact_root=tmp_path,
            timestamp="2099-01-01T00:00:00Z",
            config=config,
            state={},
            error=RuntimeError("second failure"),
        )
    assert sha256_file(failure_path) == failure_file_hash
    assert sha256_file(index_path) == index_file_hash


def test_stress_main_returns_nonzero_for_controlled_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail_run(**_: Any) -> None:
        raise PostconfirmatoryStressError("synthetic controlled failure")

    monkeypatch.setattr(stress_runner, "run_postconfirmatory_sensor_stress", fail_run)
    exit_code = stress_runner.main(
        [
            "--stress-config",
            str(tmp_path / "config.yaml"),
            "--final-freeze-inventory",
            str(tmp_path / "freeze.json"),
            "--opening-receipt",
            str(tmp_path / "receipt.json"),
            "--locked-target-index",
            str(tmp_path / "index.json"),
            "--primary-cache-record",
            str(tmp_path / "cache.json"),
            "--expected-primary-cache-record-file-sha256",
            "a" * 64,
            "--artifact-root",
            str(tmp_path),
            "--output-directory",
            str(tmp_path / "stress"),
            "--output-root",
            str(tmp_path),
            "--created-at-utc",
            "2099-01-01T00:00:00Z",
        ]
    )

    assert exit_code == 2
    assert "synthetic controlled failure" in capsys.readouterr().err


def test_module_clis_expose_no_raw_unlock_or_opening_acknowledgement() -> None:
    for parser in (build_efficiency_parser(), build_stress_parser()):
        help_text = parser.format_help()
        assert "--raw-csv" not in help_text
        assert "--unlock-record" not in help_text
        assert "acknowledge" not in help_text.casefold()
    stress_help = build_stress_parser().format_help()
    assert "--primary-cache-record" in stress_help
    assert "--expected-primary-cache-record-file-sha256" in stress_help
    assert "--source-cache-array" not in stress_help
    assert "--target-cache-array" not in stress_help


def test_stress_cache_loader_uses_one_external_pin_for_exact_partitions(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_path = tmp_path / "primary-cache-index.json"
    record_path.write_text(json.dumps({"raw_sensor_csv": {"sha256": "b" * 64}}), encoding="utf-8")
    file_hash = sha256_file(record_path)
    calls: list[dict[str, Any]] = []

    def fake_load(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(
            record_path=record_path.resolve(strict=True),
            record_file_sha256=file_hash,
            record_sha256="c" * 64,
        )

    monkeypatch.setattr(stress_runner, "load_prepared_primary_cache", fake_load)
    source, target = stress_runner._load_pinned_primary_stress_caches(
        primary_cache_record_path=record_path,
        expected_primary_cache_record_file_sha256=file_hash,
        opening_receipt_path=tmp_path / "receipt.json",
        locked_target_index_path=tmp_path / "target-index.json",
        artifact_root=tmp_path,
        expected_split_manifest_sha256="a" * 64,
        source_partition="source_validation",
        target_partition="target_sealed",
    )

    assert source.record_sha256 == target.record_sha256 == "c" * 64
    assert [call["partition"] for call in calls] == ["source_validation", "target_sealed"]
    assert all(call["expected_record_file_sha256"] == file_hash for call in calls)
    assert all(call["expected_source_artifact_sha256"] == "b" * 64 for call in calls)


def test_stress_cache_loader_rejects_changed_external_pin_before_cache_load(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    record_path = tmp_path / "primary-cache-index.json"
    record_path.write_text(json.dumps({"raw_sensor_csv": {"sha256": "b" * 64}}), encoding="utf-8")

    def forbidden_load(**_: Any) -> None:
        raise AssertionError("cache loader must not run after the external pin fails")

    monkeypatch.setattr(stress_runner, "load_prepared_primary_cache", forbidden_load)
    with pytest.raises(PostconfirmatoryStressError, match="file hash changed"):
        stress_runner._load_pinned_primary_stress_caches(
            primary_cache_record_path=record_path,
            expected_primary_cache_record_file_sha256="d" * 64,
            opening_receipt_path=tmp_path / "receipt.json",
            locked_target_index_path=tmp_path / "target-index.json",
            artifact_root=tmp_path,
            expected_split_manifest_sha256="a" * 64,
            source_partition="source_validation",
            target_partition="target_sealed",
        )


def test_stress_operation_locks_one_compact_family(repository_root: Path) -> None:
    config = load_sensor_reliability_config(
        repository_root / "configs/experiments/sensor_reliability_stress_v1.yaml"
    )
    assert config.model_family_id == "frozen-compact-residual-96-v1"
    assert config.eligible_training_model_names == ("compact_residual_96",)
    assert config.source_partition == "source_validation"
    assert config.target_partition == "target_sealed"
