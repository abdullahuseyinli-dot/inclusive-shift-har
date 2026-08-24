from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from inclusive_shift_har.data.uci_har import load_uci_har_split
from inclusive_shift_har.experiments.uci_source import (
    UCISourceRunError,
    prepare_uci_source_fold,
    run_uci_source_fold,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.protocols.uci_source import build_uci_source_protocol_manifest
from tests._uci_synthetic import materialize_synthetic_uci_archive


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
            code_commit="a" * 40,
            run_directory=tmp_path / "outputs" / "run",
            summary_path=tmp_path / "outputs" / "summary.json",
            allowed_output_root=tmp_path,
            epochs=1,
            batch_size=2,
            learning_rate=3e-4,
            weight_decay=1e-4,
            patience=1,
            minimum_epochs=1,
            checkpoint_selection_rule="fixed_last_epoch",
        )
    assert not (tmp_path / "outputs").exists()
