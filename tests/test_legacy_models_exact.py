from __future__ import annotations

import torch

from inclusive_shift_har.models.legacy_models import (
    LEGACY_MODEL_SOURCE_CELLS,
    LegacyJointLogitFusion,
    build_exact_legacy_model,
)
from inclusive_shift_har.models.legacy_specs import corrected_legacy_architectures


def _parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def test_runnable_exact_graphs_match_framework_independent_parameter_counts() -> None:
    expected = {
        spec.architecture_id: spec.parameter_count() for spec in corrected_legacy_architectures()
    }

    for variant, parameter_count in expected.items():
        model = build_exact_legacy_model(variant)
        assert _parameter_count(model) == parameter_count


def test_exact_models_accept_batch_time_channel_and_return_six_logits() -> None:
    inputs = torch.randn(2, 128, 6)
    variants = (
        "legacy_bilstm_h192",
        "legacy_bilstm_h192_temporal",
        "legacy_cnn1d_h128",
        "legacy_joint_bilstm256_cnn128",
    )

    for variant in variants:
        model = build_exact_legacy_model(variant).eval()
        with torch.no_grad():
            output = model(inputs)
        assert output.logits.shape == (2, 6)
        assert output.content.shape[0] == 2


def test_joint_model_is_exact_logit_average_and_trains_both_branches() -> None:
    model = build_exact_legacy_model("legacy_joint_bilstm256_cnn128")
    assert isinstance(model, LegacyJointLogitFusion)
    inputs = torch.randn(2, 128, 6)
    model.eval()

    recurrent = model.bilstm(inputs)
    convolutional = model.cnn1d(inputs)
    output = model(inputs)

    assert torch.equal(output.logits, 0.5 * (recurrent.logits + convolutional.logits))
    output.logits.sum().backward()
    assert all(parameter.grad is not None for parameter in model.bilstm.parameters())
    assert all(parameter.grad is not None for parameter in model.cnn1d.parameters())
    assert LEGACY_MODEL_SOURCE_CELLS["joint_logit_fusion"] == (
        "10/11/4929a1ae",
        "12/13/cc39cfc2",
        "14/15/8fcda24e",
        "26/27/12004a09",
    )
