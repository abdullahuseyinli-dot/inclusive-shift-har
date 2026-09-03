"""Deterministic participant-by-class balanced sampling."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class ParticipantClassSampler(Sampler[int]):
    """Cycle equally across observed participant/class cells for one fixed-size epoch."""

    def __init__(
        self,
        participant_ids: Sequence[str],
        labels: Sequence[int],
        *,
        seed: int,
        sample_count: int | None = None,
    ) -> None:
        if len(participant_ids) != len(labels) or not labels:
            raise ValueError("participant/class sampler inputs must be non-empty and aligned")
        if seed < 0:
            raise ValueError("sampler seed must be non-negative")
        groups: dict[tuple[str, int], list[int]] = defaultdict(list)
        for index, (participant, label) in enumerate(zip(participant_ids, labels, strict=True)):
            groups[(str(participant), int(label))].append(index)
        self._groups = {key: tuple(indices) for key, indices in sorted(groups.items())}
        self._seed = seed
        self._epoch = 0
        self._sample_count = len(labels) if sample_count is None else sample_count
        if self._sample_count < 1:
            raise ValueError("sample_count must be positive")

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("sampler epoch must be non-negative")
        self._epoch = epoch

    def __len__(self) -> int:
        return self._sample_count

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self._seed + self._epoch * 1_000_003)
        keys = list(self._groups)
        rng.shuffle(keys)
        shuffled = {key: rng.permutation(self._groups[key]).tolist() for key in keys}
        offsets = {key: 0 for key in keys}
        for position in range(self._sample_count):
            key = keys[position % len(keys)]
            members = shuffled[key]
            offset = offsets[key]
            if offset >= len(members):
                members = rng.permutation(self._groups[key]).tolist()
                shuffled[key] = members
                offset = 0
            yield int(members[offset])
            offsets[key] = offset + 1
