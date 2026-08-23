"""Paper-derived CCIL concept-mean similarity objective.

This is a local implementation of equations 2-6 in the CCIL paper, not code
released or endorsed by its authors.  No official implementation or software
license was located during the 2026-08-23 audit.  The paper leaves operational
details such as first-observation initialization and update timing underspecified;
the explicit choices made here are documented on :class:`CCILPaperConceptMeanLoss`.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def concept_matrices(features: Tensor, classifier_weight: Tensor) -> Tensor:
    """Construct per-example CCIL concept matrices.

    Args:
        features: Penultimate features with shape ``[batch, feature_dim]``.
        classifier_weight: Linear-classifier weight with PyTorch layout
            ``[num_classes, feature_dim]``.

    Returns:
        Tensor with shape ``[batch, feature_dim, num_classes]`` whose element
        ``(i, d, c)`` is ``features[i, d] * classifier_weight[c, d]``.
    """

    if features.ndim != 2:
        raise ValueError("CCIL features must have shape [batch, feature_dim]")
    if classifier_weight.ndim != 2:
        raise ValueError("CCIL classifier weight must have shape [num_classes, feature_dim]")
    if features.shape[0] < 1:
        raise ValueError("CCIL requires a non-empty batch")
    if features.shape[1] != classifier_weight.shape[1]:
        raise ValueError("CCIL feature dimension must match the classifier weight")
    if features.device != classifier_weight.device:
        raise ValueError("CCIL features and classifier weight must be on the same device")
    if not torch.is_floating_point(features) or not torch.is_floating_point(classifier_weight):
        raise TypeError("CCIL features and classifier weight must be floating-point tensors")
    return features.unsqueeze(-1) * classifier_weight.transpose(0, 1).unsqueeze(0)


class CCILPaperConceptMeanLoss(nn.Module):
    """Stateful paper-derived concept-mean similarity loss.

    The paper's EMA coefficient :math:`lambda` is represented by
    ``ema_update_weight``: ``new = (1-lambda) * old + lambda * batch_mean``.
    The first observation of each activity class initializes its mean directly
    from that source-training batch.  During an updating call, the current
    batch is incorporated before its loss is evaluated, matching the ordering
    in the published equations.  EMA references are detached from autograd.

    This module has no domain, disability, assistive-device, or participant
    inputs.  Protocol enforcement remains the caller's responsibility: state
    may be updated only from the authorized source-training partition and must
    be frozen for validation and evaluation.
    """

    class_means: Tensor
    initialized: Tensor
    update_counts: Tensor

    def __init__(
        self,
        num_classes: int,
        feature_dim: int,
        *,
        ema_update_weight: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_classes < 1:
            raise ValueError("CCIL num_classes must be positive")
        if feature_dim < 1:
            raise ValueError("CCIL feature_dim must be positive")
        if not 0.0 < ema_update_weight <= 1.0:
            raise ValueError("CCIL ema_update_weight must be in (0, 1]")
        state_dtype = torch.get_default_dtype() if dtype is None else dtype
        if not state_dtype.is_floating_point:
            raise TypeError("CCIL state dtype must be floating point")

        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.ema_update_weight = float(ema_update_weight)
        self.register_buffer(
            "class_means",
            torch.zeros(
                num_classes,
                feature_dim,
                num_classes,
                device=device,
                dtype=state_dtype,
            ),
        )
        self.register_buffer(
            "initialized",
            torch.zeros(num_classes, device=device, dtype=torch.bool),
        )
        self.register_buffer(
            "update_counts",
            torch.zeros(num_classes, device=device, dtype=torch.long),
        )

    def _validate_labels(self, labels: Tensor, *, batch_size: int, device: torch.device) -> None:
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError("CCIL labels must have shape [batch] and align with features")
        if labels.device != device:
            raise ValueError("CCIL labels and features must be on the same device")
        if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise TypeError("CCIL labels must use an integer dtype")
        if bool((labels < 0).any()) or bool((labels >= self.num_classes).any()):
            raise ValueError("CCIL labels must be in [0, num_classes)")

    def _references_and_optional_update(
        self,
        concepts: Tensor,
        labels: Tensor,
        *,
        update: bool,
    ) -> Tensor:
        if concepts.device != self.class_means.device or concepts.dtype != self.class_means.dtype:
            raise ValueError(
                "CCIL state must match the concept tensor device and dtype; move the loss module "
                "with .to(device=..., dtype=...)"
            )

        # Normalize integer label widths before using labels as indices.  In
        # particular, PyTorch treats uint8 tensors as masks rather than indices.
        observed = torch.unique(labels.to(dtype=torch.long), sorted=True)
        if not update and not bool(self.initialized[observed].all()):
            missing = observed[~self.initialized[observed]].tolist()
            raise RuntimeError(
                f"CCIL evaluation encountered uninitialized activity-class means: {missing}"
            )

        references = self.class_means.detach().clone()
        if not update:
            return references

        with torch.no_grad():
            for class_index_tensor in observed:
                class_index = int(class_index_tensor.item())
                batch_mean = concepts.detach()[labels == class_index].mean(dim=0)
                if bool(self.initialized[class_index]):
                    weight = self.ema_update_weight
                    updated = (1.0 - weight) * self.class_means[class_index] + weight * batch_mean
                else:
                    updated = batch_mean
                references[class_index].copy_(updated)
                self.class_means[class_index].copy_(updated)
                self.initialized[class_index] = True
                self.update_counts[class_index] += 1
        return references

    def forward(
        self,
        features: Tensor,
        labels: Tensor,
        classifier_weight: Tensor,
        *,
        update: bool,
    ) -> Tensor:
        """Compute mean squared Frobenius distance to each class concept mean."""

        concepts = concept_matrices(features, classifier_weight)
        if classifier_weight.shape[0] != self.num_classes:
            raise ValueError("CCIL classifier class count must match the loss state")
        if features.shape[1] != self.feature_dim:
            raise ValueError("CCIL feature dimension must match the loss state")
        self._validate_labels(labels, batch_size=features.shape[0], device=features.device)
        references = self._references_and_optional_update(concepts, labels, update=update)
        residual = concepts - references[labels.to(dtype=torch.long)]
        return residual.square().sum(dim=(1, 2)).mean()

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, feature_dim={self.feature_dim}, "
            f"ema_update_weight={self.ema_update_weight}"
        )
