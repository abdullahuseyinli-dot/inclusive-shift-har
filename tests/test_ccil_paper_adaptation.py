from __future__ import annotations

import pytest
import torch

from inclusive_shift_har.training.ccil_paper import (
    CCILPaperConceptMeanLoss,
    concept_matrices,
)


def test_concept_matrix_matches_paper_factorization() -> None:
    features = torch.tensor([[2.0, 3.0]])
    classifier_weight = torch.tensor([[1.0, 10.0], [2.0, 20.0]])

    observed = concept_matrices(features, classifier_weight)

    expected = torch.tensor([[[2.0, 4.0], [30.0, 60.0]]])
    torch.testing.assert_close(observed, expected)


def test_identical_within_class_concepts_initialize_to_zero_loss() -> None:
    objective = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.1)
    features = torch.tensor([[1.0, 2.0], [1.0, 2.0]])
    labels = torch.tensor([0, 0])
    classifier_weight = torch.eye(2)

    loss = objective(features, labels, classifier_weight, update=True)

    torch.testing.assert_close(loss, torch.tensor(0.0))
    assert objective.initialized.tolist() == [True, False]
    assert objective.update_counts.tolist() == [1, 0]


def test_ema_state_is_checkpointable_and_uses_explicit_update_weight() -> None:
    objective = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.25)
    classifier_weight = torch.eye(2)
    labels = torch.tensor([0])
    objective(torch.tensor([[1.0, 2.0]]), labels, classifier_weight, update=True)
    first = objective.class_means[0].clone()

    loss = objective(torch.tensor([[3.0, 4.0]]), labels, classifier_weight, update=True)

    second_batch = concept_matrices(torch.tensor([[3.0, 4.0]]), classifier_weight)[0]
    torch.testing.assert_close(objective.class_means[0], 0.75 * first + 0.25 * second_batch)
    assert float(loss) > 0.0
    assert objective.update_counts.tolist() == [2, 0]

    restored = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.25)
    restored.load_state_dict(objective.state_dict())
    torch.testing.assert_close(restored.class_means, objective.class_means)
    assert restored.initialized.tolist() == objective.initialized.tolist()
    assert restored.update_counts.tolist() == objective.update_counts.tolist()


def test_frozen_state_propagates_gradients_to_features_and_classifier() -> None:
    objective = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.1)
    labels = torch.tensor([0])
    classifier_weight = torch.eye(2, requires_grad=True)
    objective(torch.tensor([[1.0, 2.0]]), labels, classifier_weight, update=True)

    features = torch.tensor([[2.0, 3.0]], requires_grad=True)
    loss = objective(features, labels, classifier_weight, update=False)
    loss.backward()

    assert features.grad is not None
    assert classifier_weight.grad is not None
    assert bool((features.grad != 0).any())
    assert bool((classifier_weight.grad != 0).any())


def test_frozen_state_rejects_uninitialized_activity_class() -> None:
    objective = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.1)

    with pytest.raises(RuntimeError, match="uninitialized"):
        objective(
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([1]),
            torch.eye(2),
            update=False,
        )


@pytest.mark.parametrize(
    ("labels", "error_type"),
    [
        (torch.tensor([-1]), ValueError),
        (torch.tensor([2]), ValueError),
        (torch.tensor([0.0]), TypeError),
    ],
)
def test_invalid_activity_labels_are_rejected(
    labels: torch.Tensor, error_type: type[Exception]
) -> None:
    objective = CCILPaperConceptMeanLoss(2, 2, ema_update_weight=0.1)

    with pytest.raises(error_type):
        objective(torch.tensor([[1.0, 2.0]]), labels, torch.eye(2), update=True)
