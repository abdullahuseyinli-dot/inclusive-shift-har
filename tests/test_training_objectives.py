from __future__ import annotations

import torch

from inclusive_shift_har.models import MoReHAR
from inclusive_shift_har.training import (
    GroupDROState,
    MoReObjectiveWeights,
    more_har_objective,
    physically_plausible_augmentation,
    signal_descriptors,
    supervised_contrastive_loss,
)


def test_augmentation_is_deterministic_for_identical_generator_state() -> None:
    signals = torch.randn(5, 128, 6)
    first_generator = torch.Generator().manual_seed(91)
    second_generator = torch.Generator().manual_seed(91)
    first = physically_plausible_augmentation(signals, generator=first_generator)
    second = physically_plausible_augmentation(signals, generator=second_generator)
    assert torch.equal(first.signals, second.signals)
    assert torch.equal(first.parameters, second.parameters)
    assert first.parameters.shape == (5, 5)


def test_signal_descriptors_are_finite_and_non_sensitive() -> None:
    descriptors = signal_descriptors(torch.randn(5, 128, 6))
    assert descriptors.shape == (5, 8)
    assert torch.isfinite(descriptors).all()


def test_more_har_minimal_objective_backpropagates() -> None:
    model = MoReHAR(num_classes=3)
    clean_signals = torch.randn(6, 128, 6)
    generator = torch.Generator().manual_seed(7)
    augmented_signals = physically_plausible_augmentation(
        clean_signals, generator=generator
    ).signals
    clean = model(clean_signals)
    augmented = model(augmented_signals)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])
    losses = more_har_objective(
        clean,
        augmented,
        labels,
        signal_descriptors(augmented_signals),
        weights=MoReObjectiveWeights(),
    )
    assert set(losses) == {
        "total",
        "classification",
        "content",
        "consistency",
        "realization",
        "factor",
    }
    assert all(torch.isfinite(loss) for loss in losses.values())
    losses["total"].backward()  # type: ignore[no-untyped-call]
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_supervised_contrastive_singletons_return_differentiable_zero() -> None:
    embedding = torch.randn(3, 4, requires_grad=True)
    loss = supervised_contrastive_loss(embedding, torch.tensor([0, 1, 2]))
    assert loss.item() == 0.0
    loss.backward()  # type: ignore[no-untyped-call]
    assert embedding.grad is not None


def test_group_dro_updates_normalized_weights() -> None:
    state = GroupDROState.initialize(3, device=torch.device("cpu"), step_size=0.1)
    aggregate = state.aggregate(
        torch.tensor([0.2, 0.4, 1.0, 1.2], requires_grad=True),
        torch.tensor([0, 0, 1, 1]),
    )
    assert aggregate.ndim == 0
    assert torch.isclose(state.weights.sum(), torch.tensor(1.0))
    assert state.weights[1] > state.weights[0]
