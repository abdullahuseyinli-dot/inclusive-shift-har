from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    _accumulate_training_losses,
    _advance_scheduler_after_optimizer_updates,
    reconstruct_checkpoint,
    train_source_model,
)
from inclusive_shift_har.training.final_checkpoint import validate_fixed_epoch_checkpoint


def _synthetic_windows(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 3
    windows = generator.normal(size=(count, 128, 6)).astype(np.float32)
    windows[:, :, 0] += labels[:, None] * 0.5
    return windows, labels


def test_source_training_checkpoint_is_complete_and_reconstructable(tmp_path: Path) -> None:
    train_windows, train_labels = _synthetic_windows(24, seed=1)
    validation_windows, validation_labels = _synthetic_windows(12, seed=2)
    lineage = TrainingLineage(
        dataset_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        preprocessing_config_sha256="c" * 64,
        ontology_sha256="d" * 64,
        code_commit="test-tree",
        evidence_status="synthetic_smoke_not_scientific_evidence",
        label_schema=("mobility", "sitting", "standing"),
        normalization={"fit_scope": "synthetic_training_only"},
    )
    config = TrainingConfig(
        model_name="compact_residual_32",
        num_classes=3,
        seed=19,
        epochs=2,
        batch_size=8,
        patience=2,
        minimum_epochs=1,
        mixed_precision="disabled",
        checkpoint_interval=1,
    )
    result = train_source_model(
        train_windows,
        train_labels,
        ["train-1"] * 12 + ["train-2"] * 12,
        validation_windows,
        validation_labels,
        ["validation-1"] * 6 + ["validation-2"] * 6,
        config=config,
        lineage=lineage,
        output_directory=tmp_path / "run",
        device=torch.device("cpu"),
    )
    checkpoint_path = Path(result["checkpoint_path"])
    assert checkpoint_path.is_file()
    assert len(result["checkpoint_sha256"]) == 64
    reconstructed, payload = reconstruct_checkpoint(checkpoint_path, device=torch.device("cpu"))
    assert payload["normalization"]["fit_scope"] == "synthetic_training_only"
    assert payload["lineage"]["split_manifest_sha256"] == "b" * 64
    assert payload["configuration_sha256"] == result["configuration_sha256"]
    assert reconstructed(torch.from_numpy(validation_windows[:2])).logits.shape == (2, 3)
    assert all(item["optimizer_update_count"] == 3 for item in result["history"])
    assert all(item["amp_skipped_step_count"] == 0 for item in result["history"])
    assert all(item["scheduler_advanced"] is True for item in result["history"])


def test_fixed_epoch_training_selects_last_epoch_without_early_stopping(tmp_path: Path) -> None:
    train_windows, train_labels = _synthetic_windows(18, seed=11)
    validation_windows, validation_labels = _synthetic_windows(9, seed=12)
    lineage = TrainingLineage(
        dataset_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        preprocessing_config_sha256="c" * 64,
        ontology_sha256="d" * 64,
        code_commit="test-fixed-tree",
        evidence_status="synthetic_source_final_target_sealed",
        label_schema=("mobility", "sitting", "standing"),
        normalization={"fit_scope": "synthetic_training_only"},
    )
    config = TrainingConfig(
        model_name="compact_residual_32",
        num_classes=3,
        seed=29,
        epochs=3,
        batch_size=9,
        patience=1,
        minimum_epochs=1,
        mixed_precision="disabled",
        checkpoint_interval=3,
        checkpoint_selection_rule="fixed_last_epoch",
    )
    result = train_source_model(
        train_windows,
        train_labels,
        ["train-1"] * 9 + ["train-2"] * 9,
        validation_windows,
        validation_labels,
        ["validation-1"] * 9,
        config=config,
        lineage=lineage,
        output_directory=tmp_path / "fixed",
        device=torch.device("cpu"),
    )
    assert result["epochs_completed"] == 3
    assert result["selected_epoch"] == 3
    assert result["stopped_early"] is False
    record = validate_fixed_epoch_checkpoint(
        result["checkpoint_path"],
        expected_configuration_sha256=result["configuration_sha256"],
        expected_split_manifest_sha256="b" * 64,
        expected_code_commit="test-fixed-tree",
    )
    assert record["selected_epoch"] == 3
    assert record["target_information_used_for_selection"] is False


def test_training_config_records_explicit_cudnn_backend_choice() -> None:
    config = TrainingConfig(
        model_name="legacy_bilstm_h192",
        num_classes=3,
        seed=11,
        disable_cudnn=True,
    )
    assert config.disable_cudnn is True


def test_scheduler_does_not_advance_when_grad_scaler_skips_every_update() -> None:
    class CountingScheduler:
        def __init__(self) -> None:
            self.calls = 0

        def step(self) -> None:
            self.calls += 1

    scheduler = CountingScheduler()
    assert _advance_scheduler_after_optimizer_updates(scheduler, 0) is False
    assert scheduler.calls == 0
    assert _advance_scheduler_after_optimizer_updates(scheduler, 1) is True
    assert scheduler.calls == 1


def test_training_loss_accumulator_does_not_double_count_named_total() -> None:
    totals: dict[str, float] = {}
    total = torch.tensor(1.25)
    _accumulate_training_losses(
        totals,
        total_loss=total,
        component_losses={"total": total, "classification": torch.tensor(0.75)},
        batch_size=4,
    )
    assert totals == {"total": 5.0, "classification": 3.0}
