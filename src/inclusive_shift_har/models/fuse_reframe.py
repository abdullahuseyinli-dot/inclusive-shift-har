"""FuSE-ReFrame: ontology-preserving functional support and reference-frame fusion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from inclusive_shift_har.models.common import ConvNormAct as BatchConvNormAct
from inclusive_shift_har.models.common import (
    ResidualTemporalBlock as BatchResidualTemporalBlock,
)
from inclusive_shift_har.models.common import require_window_tensor, trainable_parameter_count
from inclusive_shift_har.models.domain_adversarial import gradient_reverse

FusionMode = Literal["raw", "invariant", "static", "gated"]
NormalizationMode = Literal["batch", "group", "layer"]
BackboneMode = Literal["multiscale", "compact_residual", "inception", "tinyhar"]


def _group_count(channels: int, maximum: int = 8) -> int:
    for groups in range(min(maximum, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ChannelLayerNorm(nn.Module):
    """LayerNorm over channels at each time step for channel-first sequences."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(channels)

    def forward(self, signals: Tensor) -> Tensor:
        return cast(Tensor, self.normalization(signals.transpose(1, 2)).transpose(1, 2))


def _normalization(channels: int, mode: NormalizationMode) -> nn.Module:
    if mode == "batch":
        return nn.BatchNorm1d(channels)
    if mode == "group":
        return nn.GroupNorm(_group_count(channels), channels)
    if mode == "layer":
        return ChannelLayerNorm(channels)
    raise ValueError(f"unknown normalization mode {mode!r}")


class ConvNormAct(nn.Sequential):
    """Length-preserving convolution with a small-batch-safe normalization."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        dilation: int = 1,
        groups: int = 1,
        stride: int = 1,
        normalization: NormalizationMode = "group",
    ) -> None:
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            _normalization(out_channels, normalization),
            nn.GELU(),
        )


class V2ResidualBlock(nn.Module):
    """Depthwise-separable residual temporal block."""

    def __init__(
        self,
        channels: int,
        *,
        dilation: int,
        dropout: float,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(
                channels,
                channels,
                5,
                dilation=dilation,
                groups=channels,
                normalization=normalization,
            ),
            nn.Conv1d(channels, channels, 1, bias=False),
            _normalization(channels, normalization),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals: Tensor) -> Tensor:
        return cast(Tensor, signals + self.block(signals))


class CompactMultiscaleEncoder(nn.Module):
    """Compact multiscale encoder used independently for raw and invariant streams."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        branch_channels = max(hidden_channels // 2, 8)
        self.branches = nn.ModuleList(
            ConvNormAct(
                in_channels,
                branch_channels,
                kernel,
                normalization=normalization,
            )
            for kernel in (3, 7, 15)
        )
        self.fuse = ConvNormAct(
            branch_channels * 3,
            hidden_channels,
            1,
            normalization=normalization,
        )
        self.temporal = nn.Sequential(
            V2ResidualBlock(
                hidden_channels, dilation=1, dropout=dropout, normalization=normalization
            ),
            V2ResidualBlock(
                hidden_channels, dilation=2, dropout=dropout, normalization=normalization
            ),
            V2ResidualBlock(
                hidden_channels, dilation=4, dropout=dropout, normalization=normalization
            ),
            V2ResidualBlock(
                hidden_channels, dilation=8, dropout=dropout, normalization=normalization
            ),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_channels, max(hidden_channels // 2, 8), 1),
            nn.Tanh(),
            nn.Conv1d(max(hidden_channels // 2, 8), 1, 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals: Tensor) -> tuple[Tensor, Tensor]:
        features = self.fuse(torch.cat([branch(signals) for branch in self.branches], dim=1))
        features = self.temporal(features)
        weights = torch.softmax(self.attention(features), dim=2)
        mean = torch.sum(weights * features, dim=2)
        variance = torch.clamp(
            torch.sum(weights * features.square(), dim=2) - mean.square(),
            min=1e-6,
        )
        embedding = self.projection(torch.cat((mean, torch.sqrt(variance)), dim=1))
        return features, embedding


class CompactResidualEncoder(nn.Module):
    """Exact compact-residual control when BatchNorm and equal dimensions are selected."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        if normalization == "batch":
            self.stem: nn.Module = BatchConvNormAct(in_channels, hidden_channels, 7)
            self.blocks = nn.Sequential(
                *(
                    BatchResidualTemporalBlock(
                        hidden_channels,
                        dilation=2**index,
                        dropout=dropout,
                    )
                    for index in range(5)
                )
            )
        else:
            self.stem = ConvNormAct(
                in_channels,
                hidden_channels,
                7,
                normalization=normalization,
            )
            self.blocks = nn.Sequential(
                *(
                    V2ResidualBlock(
                        hidden_channels,
                        dilation=2**index,
                        dropout=dropout,
                        normalization=normalization,
                    )
                    for index in range(5)
                )
            )
        self.projection: nn.Module
        if embedding_dim == hidden_channels:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Sequential(
                nn.Linear(hidden_channels, embedding_dim),
                nn.LayerNorm(embedding_dim),
                nn.GELU(),
            )

    def forward(self, signals: Tensor) -> tuple[Tensor, Tensor]:
        features = self.blocks(self.stem(signals))
        embedding = self.projection(features.mean(dim=2))
        return features, embedding


class InceptionModule1d(nn.Module):
    """Clean-room InceptionTime-style multi-receptive-field temporal module."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        if out_channels % 4:
            raise ValueError("Inception encoder width must be divisible by four")
        branch_channels = out_channels // 4
        bottleneck_channels = max(out_channels // 4, 8)
        self.bottleneck = nn.Conv1d(in_channels, bottleneck_channels, 1, bias=False)
        self.temporal_branches = nn.ModuleList(
            nn.Conv1d(
                bottleneck_channels,
                branch_channels,
                kernel,
                padding=(kernel - 1) // 2,
                bias=False,
            )
            for kernel in (9, 19, 39)
        )
        self.pool_branch = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_channels, branch_channels, 1, bias=False),
        )
        self.normalization = _normalization(out_channels, normalization)
        self.activation = nn.GELU()

    def forward(self, signals: Tensor) -> Tensor:
        bottleneck = self.bottleneck(signals)
        features = torch.cat(
            [branch(bottleneck) for branch in self.temporal_branches] + [self.pool_branch(signals)],
            dim=1,
        )
        return cast(Tensor, self.activation(self.normalization(features)))


class InceptionResidualUnit(nn.Module):
    """Three Inception modules with the residual cadence described by InceptionTime."""

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        *,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        self.inception_stack = nn.Sequential(
            InceptionModule1d(
                in_channels,
                hidden_channels,
                normalization=normalization,
            ),
            InceptionModule1d(
                hidden_channels,
                hidden_channels,
                normalization=normalization,
            ),
            InceptionModule1d(
                hidden_channels,
                hidden_channels,
                normalization=normalization,
            ),
        )
        self.residual = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1, bias=False),
            _normalization(hidden_channels, normalization),
        )
        self.activation = nn.GELU()

    def forward(self, signals: Tensor) -> Tensor:
        return cast(
            Tensor,
            self.activation(self.inception_stack(signals) + self.residual(signals)),
        )


class InceptionTimeEncoder(nn.Module):
    """Compact two-unit InceptionTime control implemented from the paper description."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        self.units = nn.Sequential(
            InceptionResidualUnit(
                in_channels,
                hidden_channels,
                normalization=normalization,
            ),
            InceptionResidualUnit(
                hidden_channels,
                hidden_channels,
                normalization=normalization,
            ),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals: Tensor) -> tuple[Tensor, Tensor]:
        features = self.units(signals)
        mean = features.mean(dim=2)
        standard_deviation = torch.sqrt(features.var(dim=2, unbiased=False).clamp_min(1e-6))
        return features, self.projection(torch.cat((mean, standard_deviation), dim=1))


def _attention_head_count(channels: int) -> int:
    for count in (4, 3, 2):
        if channels % count == 0:
            return count
    return 1


class TinyHAREncoder(nn.Module):
    """Clean-room TinyHAR-style channel/temporal interaction encoder."""

    def __init__(
        self,
        in_channels: int,
        *,
        hidden_channels: int,
        embedding_dim: int,
        dropout: float,
        normalization: NormalizationMode,
    ) -> None:
        super().__init__()
        if hidden_channels % 2:
            raise ValueError("TinyHAR encoder width must be even")
        channel_width = max(hidden_channels // 2, 8)
        self.input_channels = in_channels
        self.channel_convolutions = nn.Sequential(
            ConvNormAct(1, channel_width, 5, stride=2, normalization=normalization),
            ConvNormAct(
                channel_width,
                channel_width,
                5,
                stride=2,
                normalization=normalization,
            ),
            ConvNormAct(
                channel_width,
                channel_width,
                5,
                stride=2,
                normalization=normalization,
            ),
            ConvNormAct(
                channel_width,
                hidden_channels,
                5,
                stride=2,
                normalization=normalization,
            ),
        )
        interaction_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=_attention_head_count(hidden_channels),
            dim_feedforward=hidden_channels * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.channel_interaction = nn.TransformerEncoder(
            interaction_layer,
            num_layers=1,
            enable_nested_tensor=False,
        )
        self.channel_attention = nn.Linear(hidden_channels, 1)
        self.temporal_interaction = nn.LSTM(
            hidden_channels,
            hidden_channels // 2,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_attention = nn.Sequential(
            nn.Linear(hidden_channels, max(hidden_channels // 2, 8)),
            nn.Tanh(),
            nn.Linear(max(hidden_channels // 2, 8), 1),
        )
        self.projection = nn.Sequential(
            nn.Linear(hidden_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, signals: Tensor) -> tuple[Tensor, Tensor]:
        batch, channels, time = signals.shape
        if channels != self.input_channels:
            raise ValueError("TinyHAR encoder received an unexpected channel count")
        channel_features = self.channel_convolutions(signals.reshape(batch * channels, 1, time))
        reduced_time = channel_features.shape[2]
        channel_features = channel_features.reshape(
            batch,
            channels,
            -1,
            reduced_time,
        ).permute(0, 3, 1, 2)
        interacted = self.channel_interaction(
            channel_features.reshape(batch * reduced_time, channels, -1)
        ).reshape(batch, reduced_time, channels, -1)
        channel_weights = torch.softmax(self.channel_attention(interacted), dim=2)
        fused = torch.sum(channel_weights * interacted, dim=2)
        temporal, _ = self.temporal_interaction(fused)
        temporal_weights = torch.softmax(self.temporal_attention(temporal), dim=1)
        embedding = self.projection(torch.sum(temporal_weights * temporal, dim=1))
        sequence = F.interpolate(
            temporal.transpose(1, 2),
            size=time,
            mode="linear",
            align_corners=False,
        )
        return sequence, embedding


def _build_encoder(
    backbone: BackboneMode,
    in_channels: int,
    *,
    hidden_channels: int,
    embedding_dim: int,
    dropout: float,
    normalization: NormalizationMode,
) -> nn.Module:
    encoder_types: dict[BackboneMode, type[nn.Module]] = {
        "multiscale": CompactMultiscaleEncoder,
        "compact_residual": CompactResidualEncoder,
        "inception": InceptionTimeEncoder,
        "tinyhar": TinyHAREncoder,
    }
    try:
        encoder_type = encoder_types[backbone]
    except KeyError as exc:
        raise ValueError(f"unknown FuSE-ReFrame backbone {backbone!r}") from exc
    return encoder_type(
        in_channels,
        hidden_channels=hidden_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
        normalization=normalization,
    )


class FunctionalHierarchy(nn.Module):
    """Mobility/stationary hierarchy with sitting/standing conditional leaves."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.nodes = nn.Linear(embedding_dim, 2)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        nodes = self.nodes(embedding)
        mobility_logit = nodes[:, 0]
        sitting_given_stationary_logit = nodes[:, 1]
        log_probabilities = torch.stack(
            (
                F.logsigmoid(mobility_logit),
                F.logsigmoid(-mobility_logit) + F.logsigmoid(sitting_given_stationary_logit),
                F.logsigmoid(-mobility_logit) + F.logsigmoid(-sitting_given_stationary_logit),
            ),
            dim=1,
        )
        return log_probabilities, nodes


class FlatHead(nn.Module):
    """Flat three-class control for the functional hierarchy ablation."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, 3)

    def forward(self, embedding: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.classifier(embedding)
        return F.log_softmax(logits, dim=1), logits


def invariant_features_torch(signals: Tensor) -> Tensor:
    """Torch equivalent of the eight native-unit invariant feature streams."""

    require_window_tensor(signals)
    acceleration = signals[:, :, :3]
    gyroscope = signals[:, :, 3:]
    acceleration_norm = torch.linalg.vector_norm(acceleration, dim=2)
    gyroscope_norm = torch.linalg.vector_norm(gyroscope, dim=2)
    acceleration_delta = torch.diff(acceleration, dim=1, prepend=acceleration[:, :1])
    gyroscope_delta = torch.diff(gyroscope, dim=1, prepend=gyroscope[:, :1])
    acceleration_delta_norm = torch.linalg.vector_norm(acceleration_delta, dim=2)
    gyroscope_delta_norm = torch.linalg.vector_norm(gyroscope_delta, dim=2)
    denominator = torch.clamp(acceleration_norm * gyroscope_norm, min=1e-8)
    normalized_cross_sensor_dot = torch.sum(acceleration * gyroscope, dim=2) / denominator
    cross_sensor_cross_norm = torch.linalg.vector_norm(
        torch.cross(acceleration, gyroscope, dim=2),
        dim=2,
    )
    previous_acceleration = torch.cat((acceleration[:, :1], acceleration[:, :-1]), dim=1)
    previous_gyroscope = torch.cat((gyroscope[:, :1], gyroscope[:, :-1]), dim=1)
    acceleration_lag_dot = torch.sum(acceleration * previous_acceleration, dim=2)
    gyroscope_lag_dot = torch.sum(gyroscope * previous_gyroscope, dim=2)
    return torch.stack(
        (
            acceleration_norm,
            gyroscope_norm,
            acceleration_delta_norm,
            gyroscope_delta_norm,
            normalized_cross_sensor_dot,
            cross_sensor_cross_norm,
            acceleration_lag_dot,
            gyroscope_lag_dot,
        ),
        dim=2,
    )


@dataclass(frozen=True)
class FuSEReFrameOutput:
    """Outputs required for exact/partial labels, reconstruction, and ablations."""

    logits: Tensor
    raw_logits: Tensor
    invariant_logits: Tensor
    raw_nodes: Tensor
    invariant_nodes: Tensor
    gate: Tensor
    content: Tensor
    reconstruction: Tensor
    domain_logits: Tensor | None


class FuSEReFrameHAR(nn.Module):
    """Functional hierarchy with reliability-gated raw/invariant evidence."""

    raw_mean: Tensor
    raw_scale: Tensor
    invariant_mean: Tensor
    invariant_scale: Tensor
    clipping_thresholds: Tensor

    def __init__(
        self,
        *,
        raw_mean: Tensor,
        raw_scale: Tensor,
        invariant_mean: Tensor,
        invariant_scale: Tensor,
        clipping_thresholds: Tensor,
        hidden_channels: int = 64,
        embedding_dim: int = 96,
        dropout: float = 0.1,
        hierarchical: bool = True,
        fusion_mode: FusionMode = "gated",
        normalization: NormalizationMode = "group",
        backbone: BackboneMode = "multiscale",
        num_source_domains: int | None = None,
        parameter_limit: int = 3_000_000,
    ) -> None:
        super().__init__()
        if raw_mean.shape != (6,) or raw_scale.shape != (6,):
            raise ValueError("raw normalization tensors must each contain six values")
        if invariant_mean.shape != (8,) or invariant_scale.shape != (8,):
            raise ValueError("invariant normalization tensors must each contain eight values")
        if clipping_thresholds.shape != (6,):
            raise ValueError("clipping thresholds must contain six values")
        if torch.any(raw_scale <= 0) or torch.any(invariant_scale <= 0):
            raise ValueError("normalization scales must be positive")
        if torch.any(clipping_thresholds <= 0):
            raise ValueError("clipping thresholds must be positive")
        if fusion_mode not in {"raw", "invariant", "static", "gated"}:
            raise ValueError(f"unknown fusion mode {fusion_mode!r}")
        if normalization not in {"batch", "group", "layer"}:
            raise ValueError(f"unknown normalization mode {normalization!r}")
        if backbone not in {"multiscale", "compact_residual", "inception", "tinyhar"}:
            raise ValueError(f"unknown backbone {backbone!r}")
        if num_source_domains is not None and num_source_domains < 2:
            raise ValueError("a domain head requires at least two source domains")
        self.fusion_mode: FusionMode = fusion_mode
        self.hierarchical = hierarchical
        self.backbone: BackboneMode = backbone
        self.register_buffer("raw_mean", raw_mean.detach().float().reshape(1, 1, 6))
        self.register_buffer("raw_scale", raw_scale.detach().float().reshape(1, 1, 6))
        self.register_buffer("invariant_mean", invariant_mean.detach().float().reshape(1, 1, 8))
        self.register_buffer("invariant_scale", invariant_scale.detach().float().reshape(1, 1, 8))
        self.register_buffer(
            "clipping_thresholds", clipping_thresholds.detach().float().reshape(1, 1, 6)
        )
        head_type: type[FunctionalHierarchy] | type[FlatHead]
        head_type = FunctionalHierarchy if hierarchical else FlatHead
        self.raw_encoder = _build_encoder(
            backbone,
            6,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            normalization=normalization,
        )
        self.raw_head = head_type(embedding_dim)
        self.invariant_encoder = _build_encoder(
            backbone,
            8,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            normalization=normalization,
        )
        self.invariant_head = head_type(embedding_dim)
        self.reconstruction_head = nn.Sequential(
            ConvNormAct(
                hidden_channels * 2,
                hidden_channels,
                3,
                normalization=normalization,
            ),
            nn.Conv1d(hidden_channels, 6, 1),
        )
        self.gate = nn.Sequential(
            nn.Linear(7, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.domain_head = (
            nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embedding_dim, num_source_domains),
            )
            if num_source_domains is not None
            else None
        )
        parameters = trainable_parameter_count(self)
        if parameters >= parameter_limit:
            raise ValueError(
                f"FuSE-ReFrame has {parameters:,} parameters, violating limit {parameter_limit:,}"
            )

    def _health_features(
        self,
        native_signals: Tensor,
        raw_log_probabilities: Tensor,
        invariant_log_probabilities: Tensor,
    ) -> Tensor:
        absolute = native_signals.abs()
        zero_fraction = torch.mean((absolute < 1e-12).float(), dim=(1, 2))
        clipping_fraction = torch.mean(
            (absolute >= self.clipping_thresholds).float(),
            dim=(1, 2),
        )
        quarter = max(native_signals.shape[1] // 4, 1)
        acceleration_norm = torch.linalg.vector_norm(native_signals[:, :, :3], dim=2)
        gyroscope_norm = torch.linalg.vector_norm(native_signals[:, :, 3:], dim=2)
        acceleration_drift = (
            acceleration_norm[:, -quarter:].mean(dim=1) - acceleration_norm[:, :quarter].mean(dim=1)
        ).abs()
        gyroscope_drift = (
            gyroscope_norm[:, -quarter:].mean(dim=1) - gyroscope_norm[:, :quarter].mean(dim=1)
        ).abs()
        channel_std = native_signals.std(dim=1, unbiased=False)
        minimum_channel_std = channel_std.min(dim=1).values
        maximum_channel_std = channel_std.max(dim=1).values
        disagreement = torch.mean(
            torch.abs(raw_log_probabilities.exp() - invariant_log_probabilities.exp()),
            dim=1,
        )
        return torch.stack(
            (
                zero_fraction,
                clipping_fraction,
                acceleration_drift,
                gyroscope_drift,
                minimum_channel_std,
                maximum_channel_std,
                disagreement,
            ),
            dim=1,
        )

    def forward(
        self,
        native_signals: Tensor,
        *,
        grl_strength: float = 0.0,
    ) -> FuSEReFrameOutput:
        require_window_tensor(native_signals)
        raw_standardized = (native_signals - self.raw_mean) / self.raw_scale
        invariant = invariant_features_torch(native_signals)
        invariant_standardized = (invariant - self.invariant_mean) / self.invariant_scale
        raw_sequence, raw_embedding = self.raw_encoder(raw_standardized.transpose(1, 2))
        invariant_sequence, invariant_embedding = self.invariant_encoder(
            invariant_standardized.transpose(1, 2)
        )
        raw_log_probabilities, raw_nodes = self.raw_head(raw_embedding)
        invariant_log_probabilities, invariant_nodes = self.invariant_head(invariant_embedding)
        if self.fusion_mode == "raw":
            gate = torch.ones((native_signals.shape[0], 1), device=native_signals.device)
            combined = raw_log_probabilities
        elif self.fusion_mode == "invariant":
            gate = torch.zeros((native_signals.shape[0], 1), device=native_signals.device)
            combined = invariant_log_probabilities
        elif self.fusion_mode == "static":
            gate = torch.full(
                (native_signals.shape[0], 1),
                0.5,
                dtype=native_signals.dtype,
                device=native_signals.device,
            )
            combined = torch.logaddexp(
                raw_log_probabilities + torch.log(gate),
                invariant_log_probabilities + torch.log1p(-gate),
            )
        else:
            health = self._health_features(
                native_signals,
                raw_log_probabilities,
                invariant_log_probabilities,
            )
            gate_logit = self.gate(health)
            gate = torch.sigmoid(gate_logit)
            combined = torch.logaddexp(
                raw_log_probabilities + F.logsigmoid(gate_logit),
                invariant_log_probabilities + F.logsigmoid(-gate_logit),
            )
        content = gate * raw_embedding + (1.0 - gate) * invariant_embedding
        reconstruction = self.reconstruction_head(
            torch.cat((raw_sequence, invariant_sequence), dim=1)
        ).transpose(1, 2)
        domain_logits = (
            self.domain_head(gradient_reverse(content, strength=grl_strength))
            if self.domain_head is not None
            else None
        )
        return FuSEReFrameOutput(
            logits=combined,
            raw_logits=raw_log_probabilities,
            invariant_logits=invariant_log_probabilities,
            raw_nodes=raw_nodes,
            invariant_nodes=invariant_nodes,
            gate=gate,
            content=content,
            reconstruction=reconstruction,
            domain_logits=domain_logits,
        )
