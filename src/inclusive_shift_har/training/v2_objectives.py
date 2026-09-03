"""Losses for exact, partial-label, reconstruction, and participant-tail v2 training."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def exact_nll(log_probabilities: Tensor, labels: Tensor) -> Tensor:
    """Return one exact-label negative log-likelihood value per example."""

    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 3:
        raise ValueError("exact NLL requires [batch,3] log probabilities")
    if labels.ndim != 1 or labels.shape[0] != log_probabilities.shape[0]:
        raise ValueError("exact NLL labels are not aligned")
    return F.nll_loss(log_probabilities, labels, reduction="none")


def set_valued_nll(log_probabilities: Tensor, allowed_classes: Tensor) -> Tensor:
    """Return partial-label NLL without inventing a leaf label within the allowed set."""

    if log_probabilities.ndim != 2 or log_probabilities.shape[1] != 3:
        raise ValueError("set-valued NLL requires [batch,3] log probabilities")
    if allowed_classes.shape != log_probabilities.shape or allowed_classes.dtype != torch.bool:
        raise ValueError("allowed-class mask must be boolean and match log probabilities")
    if not torch.all(allowed_classes.any(dim=1)):
        raise ValueError("every partial-label example must allow at least one class")
    masked = log_probabilities.masked_fill(~allowed_classes, float("-inf"))
    return -torch.logsumexp(masked, dim=1)


def participant_cvar(
    per_example_losses: Tensor,
    participant_indices: Tensor,
    *,
    tail_fraction: float = 0.3,
) -> Tensor:
    """Average the highest-loss participant means over a smooth, declared tail fraction."""

    if per_example_losses.ndim != 1 or participant_indices.shape != per_example_losses.shape:
        raise ValueError("participant CVaR inputs must be aligned vectors")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must lie in (0, 1]")
    unique = torch.unique(participant_indices, sorted=True)
    if unique.numel() == 0:
        raise ValueError("participant CVaR requires at least one participant")
    participant_means = torch.stack(
        [per_example_losses[participant_indices == participant].mean() for participant in unique]
    )
    tail_count = max(1, math.ceil(float(unique.numel()) * tail_fraction))
    return torch.topk(participant_means, k=tail_count, largest=True).values.mean()


def masked_reconstruction_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor,
) -> Tensor:
    """Mean squared reconstruction error over an explicit boolean time-channel mask."""

    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("reconstruction prediction, target, and mask must have equal shape")
    if mask.dtype != torch.bool or not torch.any(mask):
        raise ValueError("reconstruction mask must be boolean and non-empty")
    return F.mse_loss(prediction[mask], target[mask])


def symmetric_probability_kl(
    first_log_probabilities: Tensor, second_log_probabilities: Tensor
) -> Tensor:
    """Symmetric KL consistency between two normalized three-class predictions."""

    if first_log_probabilities.shape != second_log_probabilities.shape:
        raise ValueError("consistency predictions must have equal shape")
    first = first_log_probabilities.exp()
    second = second_log_probabilities.exp()
    return 0.5 * (
        F.kl_div(first_log_probabilities, second, reduction="batchmean")
        + F.kl_div(second_log_probabilities, first, reduction="batchmean")
    )


def reliability_gate_target(
    raw_log_probabilities: Tensor,
    invariant_log_probabilities: Tensor,
    allowed_classes: Tensor,
    *,
    temperature: float = 0.25,
) -> Tensor:
    """Return a detached soft oracle indicating when the raw branch is more reliable."""

    if temperature <= 0:
        raise ValueError("gate-target temperature must be positive")
    raw_loss = set_valued_nll(raw_log_probabilities, allowed_classes)
    invariant_loss = set_valued_nll(invariant_log_probabilities, allowed_classes)
    return torch.sigmoid((invariant_loss - raw_loss) / temperature).detach()


def vicreg_loss(
    first: Tensor,
    second: Tensor,
    *,
    invariance_weight: float = 25.0,
    variance_weight: float = 25.0,
    covariance_weight: float = 1.0,
    variance_target: float = 1.0,
) -> Tensor:
    """VICReg objective for paired clean/physical views without negative examples."""

    if first.shape != second.shape or first.ndim != 2:
        raise ValueError("VICReg inputs must be aligned [batch,feature] tensors")
    if min(invariance_weight, variance_weight, covariance_weight) < 0:
        raise ValueError("VICReg component weights must be non-negative")
    if variance_target <= 0:
        raise ValueError("VICReg variance target must be positive")
    invariance = F.mse_loss(first, second)
    if first.shape[0] < 2:
        variance = first.sum() * 0.0
        covariance = first.sum() * 0.0
    else:
        first_centered = first - first.mean(dim=0)
        second_centered = second - second.mean(dim=0)
        first_standard_deviation = torch.sqrt(first_centered.var(dim=0, unbiased=True) + 1e-4)
        second_standard_deviation = torch.sqrt(second_centered.var(dim=0, unbiased=True) + 1e-4)
        variance = 0.5 * (
            F.relu(variance_target - first_standard_deviation).mean()
            + F.relu(variance_target - second_standard_deviation).mean()
        )
        denominator = first.shape[0] - 1
        first_covariance = first_centered.T @ first_centered / denominator
        second_covariance = second_centered.T @ second_centered / denominator
        feature_count = first.shape[1]
        diagonal = torch.eye(feature_count, dtype=torch.bool, device=first.device)
        covariance = 0.5 * (
            first_covariance.masked_select(~diagonal).square().sum() / feature_count
            + second_covariance.masked_select(~diagonal).square().sum() / feature_count
        )
    return (
        invariance_weight * invariance + variance_weight * variance + covariance_weight * covariance
    )
