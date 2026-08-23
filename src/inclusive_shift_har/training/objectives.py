"""Losses used by ERM, domain-generalization baselines, and MoRe-HAR ablations."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

from inclusive_shift_har.models.common import HAROutput


def _differentiable_zero(reference: Tensor) -> Tensor:
    return reference.sum() * 0.0


def supervised_contrastive_loss(
    embeddings: Tensor,
    labels: Tensor,
    *,
    temperature: float = 0.1,
) -> Tensor:
    """Class-conditional supervised contrastive loss with safe singleton handling."""

    if embeddings.ndim != 2 or labels.ndim != 1 or embeddings.shape[0] != labels.shape[0]:
        raise ValueError("expected embeddings [batch, features] and aligned labels [batch]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch = embeddings.shape[0]
    if batch < 2:
        return _differentiable_zero(embeddings)
    normalized = F.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.transpose(0, 1) / temperature
    diagonal = torch.eye(batch, dtype=torch.bool, device=embeddings.device)
    positive = labels[:, None].eq(labels[None, :]) & ~diagonal
    valid = positive.any(dim=1)
    if not bool(valid.any()):
        return _differentiable_zero(embeddings)
    masked_logits = similarities.masked_fill(diagonal, float("-inf"))
    log_probabilities = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    mean_log_probability = (
        log_probabilities.masked_fill(~positive, 0.0).sum(dim=1)
    ) / positive_count
    return -mean_log_probability[valid].mean()


def symmetric_consistency_loss(clean_logits: Tensor, augmented_logits: Tensor) -> Tensor:
    """Symmetric KL divergence between clean and label-preserving augmented predictions."""

    if clean_logits.shape != augmented_logits.shape:
        raise ValueError("clean and augmented logits must have identical shapes")
    clean_log = F.log_softmax(clean_logits, dim=1)
    augmented_log = F.log_softmax(augmented_logits, dim=1)
    clean_probability = clean_log.exp()
    augmented_probability = augmented_log.exp()
    forward = F.kl_div(clean_log, augmented_probability, reduction="batchmean")
    reverse = F.kl_div(augmented_log, clean_probability, reduction="batchmean")
    return 0.5 * (forward + reverse)


def cross_covariance_loss(content: Tensor, realization: Tensor) -> Tensor:
    """Squared cross-covariance penalty between factorized representations."""

    if content.ndim != 2 or realization.ndim != 2 or content.shape[0] != realization.shape[0]:
        raise ValueError("content and realization must be aligned rank-two tensors")
    if content.shape[0] < 2:
        return _differentiable_zero(content)
    centered_content = content - content.mean(dim=0, keepdim=True)
    centered_realization = realization - realization.mean(dim=0, keepdim=True)
    covariance = centered_content.transpose(0, 1) @ centered_realization
    covariance = covariance / (content.shape[0] - 1)
    return covariance.square().mean()


def coral_loss(features: Tensor, domain_ids: Tensor) -> Tensor:
    """Average pairwise CORAL covariance discrepancy across observed source domains."""

    if features.ndim != 2 or domain_ids.ndim != 1 or features.shape[0] != domain_ids.shape[0]:
        raise ValueError("expected aligned features [batch, dim] and domain ids [batch]")
    covariance_by_domain: list[Tensor] = []
    for domain_id in torch.unique(domain_ids, sorted=True):
        domain_features = features[domain_ids == domain_id]
        if domain_features.shape[0] < 2:
            continue
        centered = domain_features - domain_features.mean(dim=0, keepdim=True)
        covariance_by_domain.append(
            centered.transpose(0, 1) @ centered / (domain_features.shape[0] - 1)
        )
    if len(covariance_by_domain) < 2:
        return _differentiable_zero(features)
    discrepancies = [
        (covariance_by_domain[left] - covariance_by_domain[right]).square().mean()
        for left in range(len(covariance_by_domain))
        for right in range(left + 1, len(covariance_by_domain))
    ]
    return torch.stack(discrepancies).mean()


@dataclass(frozen=True)
class MoReObjectiveWeights:
    """Predeclared coefficients for the minimal MoRe-HAR objective."""

    content: float = 0.1
    consistency: float = 0.2
    realization: float = 0.1
    factor: float = 0.01

    def __post_init__(self) -> None:
        if min(self.content, self.consistency, self.realization, self.factor) < 0:
            raise ValueError("objective weights must be non-negative")


def more_har_objective(
    clean: HAROutput,
    augmented: HAROutput,
    labels: Tensor,
    descriptor_targets: Tensor,
    *,
    weights: MoReObjectiveWeights,
) -> dict[str, Tensor]:
    """Evaluate the locked minimal factorization objective and its auditable components."""

    if clean.content is None or clean.realization is None:
        raise ValueError("clean MoRe-HAR output lacks factorized representations")
    if augmented.content is None or augmented.realization is None:
        raise ValueError("augmented MoRe-HAR output lacks factorized representations")
    if augmented.descriptor_prediction is None:
        raise ValueError("augmented MoRe-HAR output lacks descriptor predictions")
    classification = 0.5 * (
        F.cross_entropy(clean.logits, labels) + F.cross_entropy(augmented.logits, labels)
    )
    content = supervised_contrastive_loss(
        torch.cat((clean.content, augmented.content), dim=0),
        torch.cat((labels, labels), dim=0),
    )
    consistency = symmetric_consistency_loss(clean.logits, augmented.logits)
    realization = F.smooth_l1_loss(augmented.descriptor_prediction, descriptor_targets)
    factor = 0.5 * (
        cross_covariance_loss(clean.content, clean.realization)
        + cross_covariance_loss(augmented.content, augmented.realization)
    )
    total = (
        classification
        + weights.content * content
        + weights.consistency * consistency
        + weights.realization * realization
        + weights.factor * factor
    )
    return {
        "total": total,
        "classification": classification,
        "content": content,
        "consistency": consistency,
        "realization": realization,
        "factor": factor,
    }


@dataclass
class GroupDROState:
    """Serializable exponentiated-gradient weights over source participant groups."""

    weights: Tensor
    step_size: float = 0.01

    @classmethod
    def initialize(
        cls, group_count: int, *, device: torch.device, step_size: float = 0.01
    ) -> GroupDROState:
        if group_count < 1 or step_size <= 0:
            raise ValueError("GroupDRO requires at least one group and a positive step size")
        return cls(torch.full((group_count,), 1.0 / group_count, device=device), step_size)

    def aggregate(self, per_example_losses: Tensor, group_ids: Tensor) -> Tensor:
        if per_example_losses.ndim != 1 or group_ids.shape != per_example_losses.shape:
            raise ValueError("GroupDRO expects aligned one-dimensional loss and group tensors")
        group_losses: list[Tensor] = []
        observed: list[int] = []
        for group_index in range(self.weights.numel()):
            selected = per_example_losses[group_ids == group_index]
            if selected.numel():
                group_losses.append(selected.mean())
                observed.append(group_index)
        if not group_losses:
            raise ValueError("GroupDRO batch contains no valid groups")
        stacked = torch.stack(group_losses)
        indices = torch.tensor(observed, device=self.weights.device)
        with torch.no_grad():
            self.weights[indices] *= torch.exp(self.step_size * stacked.detach())
            self.weights /= self.weights.sum()
        active_weights = self.weights[indices]
        active_weights = active_weights / active_weights.sum()
        return torch.sum(active_weights * stacked)
