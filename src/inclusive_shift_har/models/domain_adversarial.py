"""Established domain-adversarial HAR baseline.

The domain task is defined only by source-training participant identifiers.  Participant,
disability, and assistive-device metadata are never inputs to the activity model.
"""

from __future__ import annotations

from typing import Any, cast

from torch import Tensor, nn
from torch.autograd import Function

from inclusive_shift_har.models.baselines import CompactResidualHAR
from inclusive_shift_har.models.common import HAROutput


class _GradientReversal(Function):
    """Identity in the forward pass and a scaled sign reversal in backpropagation."""

    @staticmethod
    def forward(ctx: Any, features: Tensor, strength: float) -> Tensor:
        ctx.strength = float(strength)
        return features.view_as(features)

    @staticmethod
    def backward(ctx: Any, gradient: Tensor) -> tuple[Tensor, None]:
        return -float(ctx.strength) * gradient, None


def gradient_reverse(features: Tensor, *, strength: float) -> Tensor:
    """Apply a validated gradient-reversal operation."""

    if not 0.0 <= strength <= 1.0:
        raise ValueError("gradient-reversal strength must lie in [0, 1]")
    return cast(
        Tensor,
        _GradientReversal.apply(features, float(strength)),  # type: ignore[no-untyped-call]
    )


class DANNCompactResidualHAR(nn.Module):
    """DANN with the compact-residual-96 activity backbone.

    DANN is an established baseline, not a contribution of this repository.  The domain
    head predicts source participant domains during training and is checkpointed with the
    backbone.  Inference still consumes only the six inertial channels.
    """

    established_baseline = True

    def __init__(
        self,
        num_classes: int,
        num_source_domains: int,
        *,
        input_channels: int = 6,
        domain_hidden_width: int = 96,
    ) -> None:
        super().__init__()
        if num_source_domains < 2:
            raise ValueError("DANN requires at least two source participant domains")
        if domain_hidden_width < 1:
            raise ValueError("DANN domain hidden width must be positive")
        self.num_source_domains = num_source_domains
        self.activity_backbone = CompactResidualHAR(
            num_classes,
            input_channels=input_channels,
            width=96,
            blocks=5,
        )
        self.domain_head = nn.Sequential(
            nn.Linear(96, domain_hidden_width),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(domain_hidden_width, num_source_domains),
        )

    def forward(self, x: Tensor, *, grl_strength: float = 0.0) -> HAROutput:
        activity = self.activity_backbone(x)
        if activity.content is None:
            raise RuntimeError("DANN activity backbone did not expose its embedding")
        reversed_embedding = gradient_reverse(activity.content, strength=grl_strength)
        return HAROutput(
            logits=activity.logits,
            content=activity.content,
            domain_logits=self.domain_head(reversed_embedding),
        )
