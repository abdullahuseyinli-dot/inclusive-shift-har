from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import inclusive_shift_har.experiments.uci_source as uci_runner
from inclusive_shift_har.data.uci_har import load_uci_har_split
from inclusive_shift_har.experiments.uci_source import (
    UCISourceRunError,
    _portable_output_reference,
    _validate_repository_head,
    _validate_retry_chain,
    load_uci_reproduction_config,
    prepare_uci_source_fold,
    run_uci_source_fold,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.protocols.uci_source import build_uci_source_protocol_manifest
from tests._uci_synthetic import materialize_synthetic_uci_archive

EXPERIMENT_CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "experiments"
    / "uci_har_corrected_reproduction_v1_1.yaml"
)
EXPERIMENT_CONFIG_FILE_SHA256 = sha256_file(EXPERIMENT_CONFIG_PATH)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    dataset_manifest = {
        "schema_version": "1.0.0",
        "dataset_id": "uci_har_v1",
        "expected_data": {
            "official_download_wrapper": {
                "embedded_processed_archive_sha256": sha256_file(archive),
            }
        },
    }
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(json.dumps(dataset_manifest), encoding="utf-8")
    train = load_uci_har_split(archive, split="train")
    protocol = build_uci_source_protocol_manifest(
        train,
        dataset_manifest_sha256=canonical_json_sha256(dataset_manifest),
        n_folds=5,
        seed=5062,
    )
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    return archive, dataset_path, protocol_path


def test_preparation_uses_grouped_official_train_and_training_only_normalization(
    tmp_path: Path,
) -> None:
    archive, dataset_path, protocol_path = _inputs(tmp_path)

    prepared = prepare_uci_source_fold(
        archive_path=archive,
        dataset_manifest_path=dataset_path,
        protocol_path=protocol_path,
        fold_id="uci_source_cv_01",
    )

    assert prepared.windows.released_split == "train"
    assert set(prepared.train_participant_ids).isdisjoint(prepared.validation_participant_ids)
    assert prepared.train_windows.shape[1:] == (128, 6)
    assert prepared.validation_windows.shape[1:] == (128, 6)
    assert np.allclose(prepared.train_windows.mean(axis=(0, 1)), 0.0, atol=1e-5)
    assert prepared.standardizer.training_participants == tuple(
        sorted(set(prepared.train_participant_ids))
    )
    assert prepared.train_labels.min() == 0
    assert prepared.train_labels.max() == 5


def test_protocol_tampering_fails_before_training(tmp_path: Path) -> None:
    archive, dataset_path, protocol_path = _inputs(tmp_path)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol["fold_assignment"]["seed"] = 1
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")

    with pytest.raises(UCISourceRunError, match="self-hash"):
        prepare_uci_source_fold(
            archive_path=archive,
            dataset_manifest_path=dataset_path,
            protocol_path=protocol_path,
            fold_id="uci_source_cv_01",
        )


def test_runner_refuses_to_fall_back_to_cpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, dataset_path, protocol_path = _inputs(tmp_path)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(UCISourceRunError, match="CUDA is required"):
        run_uci_source_fold(
            archive_path=archive,
            dataset_manifest_path=dataset_path,
            protocol_path=protocol_path,
            model_name="legacy_cnn1d_h128",
            fold_id="uci_source_cv_01",
            seed=42,
            attempt=1,
            code_commit="a" * 40,
            repository_root=tmp_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            run_directory=tmp_path / "outputs" / "run",
            summary_path=tmp_path / "outputs" / "summary.json",
            allowed_output_root=tmp_path,
        )
    assert not (tmp_path / "outputs").exists()


def test_prediction_reference_is_output_root_relative_and_confined(tmp_path: Path) -> None:
    output_root = tmp_path / "outputs"
    prediction = output_root / "run" / "predictions.npz"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"synthetic")

    assert (
        _portable_output_reference(prediction, allowed_root=output_root, kind="prediction artifact")
        == "run/predictions.npz"
    )
    outside = tmp_path / "outside.npz"
    outside.write_bytes(b"synthetic")
    with pytest.raises(UCISourceRunError, match="escapes"):
        _portable_output_reference(outside, allowed_root=output_root, kind="prediction artifact")


def test_experiment_config_is_hash_pinned_and_locks_recurrent_cudnn_policy() -> None:
    config = load_uci_reproduction_config(
        EXPERIMENT_CONFIG_PATH,
        expected_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
    )

    assert config.models == (
        "legacy_cnn1d_h128",
        "legacy_bilstm_h192",
        "legacy_joint_bilstm256_cnn128",
    )
    assert config.disable_cudnn_by_model == {
        "legacy_cnn1d_h128": False,
        "legacy_bilstm_h192": True,
        "legacy_joint_bilstm256_cnn128": True,
    }
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_uci_reproduction_config(
            EXPERIMENT_CONFIG_PATH,
            expected_file_sha256="0" * 64,
        )


def test_repository_head_binding_rejects_supplied_commit_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "inclusive_shift_har.experiments.uci_source.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout=f"{'b' * 40}\n"),
    )

    with pytest.raises(UCISourceRunError, match="does not match executing HEAD"):
        _validate_repository_head(tmp_path, expected_commit="a" * 40)


def test_retry_requires_contiguous_self_hashed_failures(tmp_path: Path) -> None:
    config = load_uci_reproduction_config(
        EXPERIMENT_CONFIG_PATH,
        expected_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
    )
    result_root = tmp_path / "matrix"
    run_path = result_root / "runs" / "cell" / "attempt-002"
    summary_path = result_root / "records" / "cell" / "attempt-002.json"
    failure_path = result_root / "runs" / "cell" / "attempt-001" / "failure.json"
    failure_path.parent.mkdir(parents=True)
    failure = {
        "schema_version": "1.0.0",
        "status": "corrected_uci_source_fold_failed_preserved",
        "model_name": "legacy_cnn1d_h128",
        "fold_id": "uci_source_cv_01",
        "seed": 42,
        "attempt": 1,
        "code_commit": "a" * 40,
        "experiment_config": {
            "experiment_id": config.experiment_id,
            "file_sha256": config.file_sha256,
            "canonical_sha256": config.canonical_sha256,
        },
        "official_test_member_opened": False,
        "official_test_performance_or_prediction_accessed": False,
        "inclusivehar_data_or_target_accessed": False,
    }
    failure["record_sha256"] = canonical_json_sha256(failure)
    failure_path.write_text(json.dumps(failure), encoding="utf-8")

    references = _validate_retry_chain(
        run_path=run_path,
        summary_path=summary_path,
        result_root=result_root,
        model_name="legacy_cnn1d_h128",
        fold_id="uci_source_cv_01",
        seed=42,
        attempt=2,
        code_commit="a" * 40,
        experiment_config=config,
    )
    assert references[0]["record_sha256"] == failure["record_sha256"]
    failure["seed"] = 43
    failure_path.write_text(json.dumps(failure), encoding="utf-8")
    with pytest.raises(UCISourceRunError, match="failure lineage is invalid"):
        _validate_retry_chain(
            run_path=run_path,
            summary_path=summary_path,
            result_root=result_root,
            model_name="legacy_cnn1d_h128",
            fold_id="uci_source_cv_01",
            seed=42,
            attempt=2,
            code_commit="a" * 40,
            experiment_config=config,
        )


def test_preparation_failure_is_preserved_inside_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(uci_runner, "_validate_repository_head", lambda *args, **kwargs: "a" * 40)
    monkeypatch.setattr(uci_runner, "_validate_matrix_paths", lambda **kwargs: None)

    def fail_prepare(**_: object) -> None:
        raise RuntimeError("synthetic preparation failure")

    monkeypatch.setattr(uci_runner, "prepare_uci_source_fold", fail_prepare)
    archive_path = tmp_path / "archive.zip"
    dataset_path = tmp_path / "dataset.json"
    protocol_path = tmp_path / "protocol.json"
    archive_path.write_bytes(b"synthetic")
    dataset_path.write_text("{}", encoding="utf-8")
    protocol_path.write_text("{}", encoding="utf-8")
    run_path = tmp_path / "matrix" / "runs" / "cell" / "attempt-001"
    summary_path = tmp_path / "matrix" / "records" / "cell" / "attempt-001.json"
    with pytest.raises(UCISourceRunError, match="evidence preserved"):
        run_uci_source_fold(
            archive_path=archive_path,
            dataset_manifest_path=dataset_path,
            protocol_path=protocol_path,
            model_name="legacy_cnn1d_h128",
            fold_id="uci_source_cv_01",
            seed=42,
            attempt=1,
            code_commit="a" * 40,
            repository_root=tmp_path,
            experiment_config_path=EXPERIMENT_CONFIG_PATH,
            expected_experiment_config_file_sha256=EXPERIMENT_CONFIG_FILE_SHA256,
            run_directory=run_path,
            summary_path=summary_path,
            allowed_output_root=tmp_path,
        )
    failure = json.loads((run_path / "failure.json").read_text(encoding="utf-8"))
    body = dict(failure)
    assert body.pop("record_sha256") == canonical_json_sha256(body)
    assert failure["execution_stage"] == "official_train_fold_preparation"
    assert failure["attempt"] == 1
