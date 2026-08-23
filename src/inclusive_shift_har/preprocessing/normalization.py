"""Training-partition-only channel normalization with explicit lineage."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ChannelStandardizer:
    """Immutable per-channel moments fitted exclusively to declared training participants."""

    mean: NDArray[np.float64]
    scale: NDArray[np.float64]
    training_participants: tuple[str, ...]
    split_manifest_sha256: str
    channel_names: tuple[str, ...]
    fitted_value_count_per_channel: int

    @classmethod
    def fit(
        cls,
        windows: NDArray[np.floating[Any]],
        participant_ids: list[str] | NDArray[np.str_],
        *,
        declared_training_participants: set[str],
        split_manifest_sha256: str,
        channel_names: tuple[str, ...],
        minimum_scale: float = 1e-8,
    ) -> ChannelStandardizer:
        array = np.asarray(windows)
        participants = np.asarray(participant_ids, dtype=np.str_)
        if array.ndim != 3 or participants.ndim != 1 or array.shape[0] != participants.size:
            raise ValueError("normalization expects aligned [window,time,channel] data and ids")
        if array.shape[0] == 0 or not np.isfinite(array).all():
            raise ValueError("normalization training data must be non-empty and finite")
        if array.shape[2] != len(channel_names) or len(set(channel_names)) != len(channel_names):
            raise ValueError("channel names must uniquely align with the tensor")
        observed = set(participants.tolist())
        if observed != declared_training_participants:
            raise ValueError(
                "normalization participants differ from the declared training partition: "
                f"observed={sorted(observed)}, declared={sorted(declared_training_participants)}"
            )
        if len(split_manifest_sha256) != 64:
            raise ValueError("normalization requires a full split-manifest SHA-256")
        if minimum_scale <= 0:
            raise ValueError("minimum_scale must be positive")
        flattened = np.asarray(array, dtype=np.float64).reshape(-1, array.shape[2])
        mean = flattened.mean(axis=0)
        scale = flattened.std(axis=0, ddof=0)
        scale = np.maximum(scale, minimum_scale)
        mean.setflags(write=False)
        scale.setflags(write=False)
        return cls(
            mean=mean,
            scale=scale,
            training_participants=tuple(sorted(observed)),
            split_manifest_sha256=split_manifest_sha256,
            channel_names=channel_names,
            fitted_value_count_per_channel=flattened.shape[0],
        )

    def transform(self, windows: NDArray[np.floating[Any]]) -> NDArray[np.float32]:
        array = np.asarray(windows)
        if array.ndim != 3 or array.shape[2] != self.mean.size:
            raise ValueError("normalization input does not match the fitted channel dimension")
        if not np.isfinite(array).all():
            raise ValueError("normalization input must be finite")
        return np.asarray((array - self.mean) / self.scale, dtype=np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0.0",
            "method": "per_channel_population_standardization",
            "mean": self.mean.tolist(),
            "scale": self.scale.tolist(),
            "training_participants": list(self.training_participants),
            "split_manifest_sha256": self.split_manifest_sha256,
            "channel_names": list(self.channel_names),
            "fitted_value_count_per_channel": self.fitted_value_count_per_channel,
            "fit_scope": "training_partition_only",
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ChannelStandardizer:
        mean = np.asarray(payload["mean"], dtype=np.float64)
        scale = np.asarray(payload["scale"], dtype=np.float64)
        if mean.ndim != 1 or scale.shape != mean.shape or np.any(scale <= 0):
            raise ValueError("invalid serialized normalization moments")
        mean.setflags(write=False)
        scale.setflags(write=False)
        return cls(
            mean=mean,
            scale=scale,
            training_participants=tuple(str(item) for item in payload["training_participants"]),
            split_manifest_sha256=str(payload["split_manifest_sha256"]),
            channel_names=tuple(str(item) for item in payload["channel_names"]),
            fitted_value_count_per_channel=int(payload["fitted_value_count_per_channel"]),
        )
