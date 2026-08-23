from __future__ import annotations

import pytest
import torch

from inclusive_shift_har.models import MoReHAR, build_baseline, trainable_parameter_count


@pytest.mark.parametrize(
    "name",
    [
        "cnn1d",
        "bilstm",
        "joint_cnn_bilstm",
        "deepconvlstm",
        "compact_residual_96",
        "compact_residual_64",
        "compact_residual_32",
        "static_dual_branch",
        "static_dual_branch_matched",
    ],
)
def test_baseline_tensor_contract(name: str) -> None:
    model = build_baseline(name, num_classes=3)
    output = model(torch.randn(4, 128, 6))
    assert output.logits.shape == (4, 3)
    assert output.content is not None
    assert torch.isfinite(output.logits).all()


def test_more_har_factorization_contract_and_parameter_limit() -> None:
    model = MoReHAR(num_classes=3)
    output = model(torch.randn(4, 128, 6))
    assert output.logits.shape == (4, 3)
    assert output.content is not None and output.content.shape == (4, 96)
    assert output.realization is not None and output.realization.shape == (4, 48)
    assert output.descriptor_prediction is not None
    assert output.descriptor_prediction.shape == (4, 8)
    assert trainable_parameter_count(model) < 2_000_000


def test_static_dual_branch_is_parameter_matched_to_more_har() -> None:
    proposed = MoReHAR(num_classes=3)
    matched = build_baseline("static_dual_branch_matched", num_classes=3)
    ratio = trainable_parameter_count(matched) / trainable_parameter_count(proposed)
    assert 0.95 <= ratio <= 1.05


def test_model_rejects_wrong_channel_interface() -> None:
    model = MoReHAR(num_classes=3)
    with pytest.raises(ValueError, match="expected 6 channels"):
        model(torch.randn(2, 128, 9))


def test_model_registry_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown baseline"):
        build_baseline("unrecorded-model", num_classes=3)
