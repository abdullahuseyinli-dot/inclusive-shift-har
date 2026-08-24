"""Qualified paper-derived adapters for post-confirmatory HAR comparisons.

These modules deliberately do not reproduce third-party training code.  The
CCIL comparison uses the locally implemented published concept regularizer on
the repository's compact residual classifier.  The BPD comparison re-expresses
the disclosed activity/redundant factorization behind the same compact temporal
generator, while leaving boundary handling and evaluation to repository code.
Every result must retain those qualifiers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inclusive_shift_har.models.baselines import CompactResidualHAR
from inclusive_shift_har.models.common import HAROutput, require_window_tensor


@dataclass(frozen=True, slots=True)
class BPDAdaptationOutput:
    """All outputs needed by the explicitly qualified BPD training adapter."""

    backbone_features: Tensor
    activity_features: Tensor
    redundant_features: Tensor
    activity_logits: Tensor
    redundant_activity_logits: Tensor
    reconstructed_backbone_features: Tensor


class BPDMINEstimator(nn.Module):
    """Small Donsker--Varadhan critic matching BPD's disclosed MINE role."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        if latent_dim < 2:
            raise ValueError("BPD MINE latent_dim must be at least two")
        hidden = max(latent_dim // 2, 2)
        self.activity = nn.Linear(latent_dim, hidden)
        self.redundant = nn.Linear(latent_dim, hidden)
        self.score = nn.Linear(hidden, 1)

    def forward(self, activity: Tensor, redundant: Tensor) -> Tensor:
        if activity.ndim != 2 or redundant.shape != activity.shape:
            raise ValueError("BPD MINE inputs must be aligned rank-two tensors")
        return cast(
            Tensor,
            self.score(F.leaky_relu(self.activity(activity) + self.redundant(redundant), 0.2)),
        )


class BPDBoundarySafeCompactAdapter(nn.Module):
    """BPD-inspired factorization over the local compact residual generator.

    This is a boundary-safe *protocol adaptation*, not the official BPD model or
    trainer.  Its temporal generator is exactly the feature-producing part of
    ``compact_residual_96``.  The one-quarter-width activity/redundant heads,
    activity prediction, reconstruction, class confusion, and MINE roles are
    paper-derived; LayerNorm and the optimization schedule are local stability
    choices documented by the experiment configuration.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        input_channels: int = 6,
        backbone_width: int = 96,
        backbone_blocks: int = 5,
        latent_dim: int = 24,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("BPD adaptation requires at least two classes")
        if latent_dim < 2 or latent_dim * 4 != backbone_width:
            raise ValueError("BPD adaptation locks latent_dim to one quarter of backbone_width")
        self.num_classes = num_classes
        self.input_channels = input_channels
        self.backbone_width = backbone_width
        self.latent_dim = latent_dim
        generator = CompactResidualHAR(
            num_classes,
            input_channels=input_channels,
            width=backbone_width,
            blocks=backbone_blocks,
            dropout=dropout,
        )
        # The local ERM classifier is replaced, leaving its stem, residual
        # blocks, and pooling byte-for-byte identical in construction.
        generator.classifier = cast(nn.Linear, nn.Identity())
        self.generator = generator
        self.activity_disentangler = nn.Sequential(
            nn.Linear(backbone_width, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.redundant_disentangler = nn.Sequential(
            nn.Linear(backbone_width, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.ReLU(),
        )
        self.activity_classifier = nn.Linear(latent_dim, num_classes)
        self.redundant_activity_classifier = nn.Linear(latent_dim, num_classes)
        self.reconstructor = nn.Linear(latent_dim * 2, backbone_width)
        self.mine = BPDMINEstimator(latent_dim)

    def decompose(self, x: Tensor) -> BPDAdaptationOutput:
        """Return the complete local factorization for source-only training."""

        require_window_tensor(x, channels=self.input_channels)
        generated = cast(HAROutput, self.generator(x))
        if generated.content is None or generated.content.shape[1] != self.backbone_width:
            raise RuntimeError("compact generator did not expose the locked backbone features")
        backbone = generated.content
        activity = cast(Tensor, self.activity_disentangler(backbone))
        redundant = cast(Tensor, self.redundant_disentangler(backbone))
        return BPDAdaptationOutput(
            backbone_features=backbone,
            activity_features=activity,
            redundant_features=redundant,
            activity_logits=self.activity_classifier(activity),
            redundant_activity_logits=self.redundant_activity_classifier(redundant),
            reconstructed_backbone_features=self.reconstructor(
                torch.cat((activity, redundant), dim=1)
            ),
        )

    def forward(self, x: Tensor) -> HAROutput:
        """Inference consumes only the activity-related representation."""

        output = self.decompose(x)
        return HAROutput(
            logits=output.activity_logits,
            content=output.activity_features,
            realization=output.redundant_features,
        )


def mine_dv_lower_bound(
    estimator: BPDMINEstimator,
    activity: Tensor,
    redundant: Tensor,
    *,
    permutation: Tensor,
) -> Tensor:
    """Compute the finite-batch Donsker--Varadhan MINE lower bound.

    The permutation is supplied by the caller's checkpointed RNG so the
    dependence objective is deterministic and reconstructable.
    """

    if activity.ndim != 2 or redundant.shape != activity.shape:
        raise ValueError("MINE features must be aligned rank-two tensors")
    if activity.shape[0] < 2:
        raise ValueError("MINE requires at least two source-training examples")
    if permutation.ndim != 1 or permutation.shape[0] != activity.shape[0]:
        raise ValueError("MINE permutation must align with the batch")
    if permutation.device != activity.device or permutation.dtype != torch.long:
        raise ValueError("MINE permutation must be int64 on the feature device")
    if not torch.equal(
        torch.sort(permutation).values, torch.arange(activity.shape[0], device=activity.device)
    ):
        raise ValueError("MINE permutation must contain every batch index exactly once")
    joint = estimator(activity, redundant).float().mean()
    marginal_scores = estimator(activity, redundant[permutation]).float().reshape(-1)
    log_mean_exp = torch.logsumexp(marginal_scores, dim=0) - math.log(marginal_scores.numel())
    return cast(Tensor, joint - log_mean_exp)


def redundant_class_confusion_loss(logits: Tensor) -> Tensor:
    """Negative categorical entropy; minimizing it removes activity evidence."""

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("class-confusion logits must have shape [batch, classes]")
    log_probability = F.log_softmax(logits, dim=1)
    probability = log_probability.exp()
    return (probability * log_probability).sum(dim=1).mean()
