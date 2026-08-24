from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inclusive_shift_har.evaluation.source_cv import derive_model_identity
from inclusive_shift_har.models import DANNCompactResidualHAR, gradient_reverse
from inclusive_shift_har.training.engine import (
    TrainingConfig,
    TrainingLineage,
    _dann_grl_strength,
    reconstruct_checkpoint,
    train_source_model,
)


def _lineage() -> TrainingLineage:
    return TrainingLineage(
        dataset_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        preprocessing_config_sha256="c" * 64,
        ontology_sha256="d" * 64,
        code_commit="synthetic-dann-test",
        evidence_status="synthetic_smoke_not_scientific_evidence",
        label_schema=("mobility", "sitting", "standing"),
        normalization={"fit_scope": "synthetic_training_only"},
    )


def _config(*, epochs: int = 2) -> TrainingConfig:
    return TrainingConfig(
        model_name="dann_compact_residual_96",
        num_classes=3,
        seed=41,
        epochs=epochs,
        batch_size=6,
        patience=epochs,
        minimum_epochs=1,
        mixed_precision="disabled",
        checkpoint_interval=1,
        dann_domain_loss_weight=0.1,
        dann_grl_max_strength=1.0,
        dann_grl_warmup_epochs=2,
    )


def _windows(count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
    generator = np.random.default_rng(seed)
    labels = np.arange(count, dtype=np.int64) % 3
    windows = generator.normal(size=(count, 128, 6)).astype(np.float32)
    windows[:, :, 0] += labels[:, None]
    return windows, labels


def test_gradient_reversal_changes_only_the_backward_sign_and_scale() -> None:
    features = torch.tensor([[1.0, -2.0]], requires_grad=True)
    gradient_reverse(features, strength=0.25).sum().backward()  # type: ignore[no-untyped-call]

    assert torch.equal(features.detach(), torch.tensor([[1.0, -2.0]]))
    assert features.grad is not None
    assert torch.equal(features.grad, torch.full_like(features, -0.25))


def test_dann_model_exposes_activity_and_checkpointed_domain_heads() -> None:
    model = DANNCompactResidualHAR(num_classes=3, num_source_domains=4)
    output = model(torch.randn(2, 128, 6), grl_strength=0.5)

    assert model.established_baseline is True
    assert output.logits.shape == (2, 3)
    assert output.content is not None and output.content.shape == (2, 96)
    assert output.domain_logits is not None and output.domain_logits.shape == (2, 4)
    assert any(name.startswith("domain_head.") for name in model.state_dict())


def test_dann_config_and_semantic_source_cv_identity_are_explicit() -> None:
    config = _config()
    identity = derive_model_identity(
        {
            "model_name": config.model_name,
            "dann_domain_loss_weight": config.dann_domain_loss_weight,
            "dann_grl_max_strength": config.dann_grl_max_strength,
            "dann_grl_warmup_epochs": config.dann_grl_warmup_epochs,
            "coral_weight": 0.0,
            "zero_channel_indices": [],
        }
    )

    assert identity["model_family_id"] == "dann_compact_residual_96"
    assert identity["variant_name"] == "established_lambda_0p1_grl_1p0_warmup_2"
    with pytest.raises(ValueError, match="positive domain-loss weight"):
        TrainingConfig(
            model_name="dann_compact_residual_96",
            num_classes=3,
            seed=1,
        )


def test_dann_grl_schedule_is_bounded_and_reaches_the_predeclared_maximum() -> None:
    config = _config()

    assert _dann_grl_strength(config, epoch=1, batch_index=0, batch_count=2) == 0.25
    assert _dann_grl_strength(config, epoch=2, batch_index=1, batch_count=2) == 1.0
    assert _dann_grl_strength(config, epoch=10, batch_index=0, batch_count=2) == 1.0


def test_dann_checkpoint_reconstructs_and_resumes_with_exact_source_domains(
    tmp_path: Path,
) -> None:
    train_windows, train_labels = _windows(12, seed=1)
    validation_windows, validation_labels = _windows(6, seed=2)
    train_participants = ["source-1"] * 6 + ["source-2"] * 6
    validation_participants = ["source-validation"] * 6
    config = _config()
    result = train_source_model(
        train_windows,
        train_labels,
        train_participants,
        validation_windows,
        validation_labels,
        validation_participants,
        config=config,
        lineage=_lineage(),
        output_directory=tmp_path / "initial",
        device=torch.device("cpu"),
    )

    checkpoint_path = Path(result["checkpoint_path"])
    reconstructed, payload = reconstruct_checkpoint(
        checkpoint_path,
        device=torch.device("cpu"),
    )
    assert isinstance(reconstructed, DANNCompactResidualHAR)
    reconstructed.eval()
    inference = reconstructed(torch.from_numpy(validation_windows[:2]))
    assert inference.logits.shape == (2, 3)
    assert payload["domain_adversarial"] == {
        "baseline_status": "established_baseline_not_proposed_contribution",
        "domain_definition": "source_training_participant_id_only",
        "participant_or_disability_metadata_required_at_inference": False,
        "num_source_domains": 2,
        "participant_domain_map": {"source-1": 0, "source-2": 1},
        "domain_loss_weight": 0.1,
        "grl_max_strength": 1.0,
        "grl_warmup_epochs": 2,
        "domain_head_checkpointed_in_model_state": True,
    }
    assert all(
        "domain_adversarial" in epoch["training_losses"]
        and "domain_accuracy" in epoch["training_losses"]
        and "grl_strength_mean" in epoch["training_losses"]
        for epoch in result["history"]
    )

    resumed = train_source_model(
        train_windows,
        train_labels,
        train_participants,
        validation_windows,
        validation_labels,
        validation_participants,
        config=config,
        lineage=_lineage(),
        output_directory=tmp_path / "resumed",
        device=torch.device("cpu"),
        resume_checkpoint=tmp_path / "initial" / "epoch_001.pt",
    )
    assert resumed["epochs_completed"] == 2
    assert resumed["history"][0] == result["history"][0]
