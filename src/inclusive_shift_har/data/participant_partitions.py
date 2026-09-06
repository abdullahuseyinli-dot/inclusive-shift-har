"""Pre-window participant partition plans for external HAR evidence.

The plan is deliberately a data-loader artifact.  It freezes every fold assignment
from an inference-observable participant inventory before any signal segmentation or
window extraction occurs.  Experiment runners may resolve a declared assignment, but
they may not construct a new one from the post-window/scored population.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np

DEFAULT_EXTERNAL_SEEDS = (11, 23, 47)
PARTITION_PROTOCOL_ID = "external-har-prewindow-partitions-v1"


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _assignment(
    participant_ids: tuple[str, ...], *, fold_count: int, seed: int
) -> tuple[tuple[str, int], ...]:
    if fold_count < 2 or len(participant_ids) < fold_count or seed < 0:
        raise ValueError("participant fold assignment requires enough groups and a valid seed")
    order = np.random.default_rng(seed).permutation(len(participant_ids))
    values = {
        participant_ids[int(index)]: position % fold_count for position, index in enumerate(order)
    }
    return tuple(sorted(values.items()))


@dataclass(frozen=True, slots=True)
class PlannedFoldAssignment:
    """One immutable assignment over an exact predeclared participant subset."""

    role: str
    seed: int
    fold_count: int
    participants: tuple[str, ...]
    assignment: tuple[tuple[str, int], ...]

    def validate(self) -> None:
        if not self.role or self.seed < 0 or self.fold_count < 2:
            raise ValueError("planned participant assignment metadata is invalid")
        if self.participants != tuple(sorted(set(self.participants))):
            raise ValueError("planned participants must be unique and sorted")
        mapping = dict(self.assignment)
        if (
            tuple(sorted(mapping)) != self.participants
            or len(mapping) != len(self.assignment)
            or set(mapping.values()) != set(range(self.fold_count))
            or any(value < 0 or value >= self.fold_count for value in mapping.values())
        ):
            raise ValueError("planned participant assignment is incomplete or invalid")
        if self.assignment != _assignment(
            self.participants, fold_count=self.fold_count, seed=self.seed
        ):
            raise ValueError("planned participant assignment differs from deterministic rule")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "role": self.role,
            "seed": self.seed,
            "fold_count": self.fold_count,
            "participants": list(self.participants),
            "assignment": {participant: fold for participant, fold in self.assignment},
        }


@dataclass(frozen=True, slots=True)
class ParticipantPartitionPlan:
    """All outer/inner assignments frozen from a pre-window observable roster."""

    dataset_id: str
    participant_roster: tuple[str, ...]
    records: tuple[PlannedFoldAssignment, ...]
    roster_basis: str
    created_before_windowing: bool = True
    protocol_id: str = PARTITION_PROTOCOL_ID

    def validate(self) -> None:
        if (
            self.protocol_id != PARTITION_PROTOCOL_ID
            or not self.created_before_windowing
            or not self.dataset_id
            or not self.roster_basis
            or self.participant_roster != tuple(sorted(set(self.participant_roster)))
            or any(not participant for participant in self.participant_roster)
            or len(self.participant_roster) < 5
            or not self.records
        ):
            raise ValueError("pre-window participant partition plan metadata is invalid")
        seen: set[tuple[str, int, int, tuple[str, ...]]] = set()
        roster = set(self.participant_roster)
        for record in self.records:
            record.validate()
            key = (record.role, record.seed, record.fold_count, record.participants)
            if key in seen or not set(record.participants).issubset(roster):
                raise ValueError(
                    "pre-window participant partition plan has duplicate/alien records"
                )
            seen.add(key)

    def resolve(
        self,
        participant_ids: Iterable[str],
        *,
        fold_count: int,
        seed: int,
        role: str,
    ) -> dict[str, int]:
        """Resolve one exact declared map; never derive from post-window observations."""

        self.validate()
        participants = tuple(sorted({str(item) for item in participant_ids}))
        matches = [
            record
            for record in self.records
            if record.role == role
            and record.seed == seed
            and record.fold_count == fold_count
            and record.participants == participants
        ]
        if len(matches) != 1:
            raise PermissionError(
                "participant assignment was not frozen before windowing: "
                f"role={role}, seed={seed}, folds={fold_count}, participants={participants}"
            )
        return dict(matches[0].assignment)

    def audit(self) -> dict[str, Any]:
        self.validate()
        payload: dict[str, Any] = {
            "protocol_id": self.protocol_id,
            "dataset_id": self.dataset_id,
            "created_before_windowing": self.created_before_windowing,
            "roster_basis": self.roster_basis,
            "participant_roster": list(self.participant_roster),
            "participant_roster_sha256": _canonical_sha256(list(self.participant_roster)),
            "records": [record.to_dict() for record in self.records],
        }
        payload["plan_sha256"] = _canonical_sha256(payload)
        return payload


def build_participant_partition_plan(
    dataset_id: str,
    participant_ids: Iterable[str],
    *,
    roster_basis: str,
    seeds: tuple[int, ...] = DEFAULT_EXTERNAL_SEEDS,
    outer_fold_count: int = 5,
    inner_fold_count: int = 4,
) -> ParticipantPartitionPlan:
    """Freeze all assignments required by the external three-seed protocol."""

    participants = tuple(sorted({str(item) for item in participant_ids}))
    if (
        not dataset_id
        or not roster_basis
        or len(participants) < outer_fold_count
        or any(not participant for participant in participants)
        or len(set(seeds)) != len(seeds)
        or not seeds
        or any(seed < 0 for seed in seeds)
    ):
        raise ValueError("cannot build pre-window participant partition plan")
    records: list[PlannedFoldAssignment] = []
    for seed in seeds:
        outer_values = _assignment(participants, fold_count=outer_fold_count, seed=seed)
        outer = dict(outer_values)
        records.append(
            PlannedFoldAssignment(
                role="outer",
                seed=seed,
                fold_count=outer_fold_count,
                participants=participants,
                assignment=outer_values,
            )
        )
        for outer_index in range(outer_fold_count):
            training = tuple(
                participant for participant in participants if outer[participant] != outer_index
            )
            records.append(
                PlannedFoldAssignment(
                    role=f"classical_inner_outer_{outer_index}",
                    seed=seed + 10_000 + outer_index,
                    fold_count=inner_fold_count,
                    participants=training,
                    assignment=_assignment(
                        training,
                        fold_count=inner_fold_count,
                        seed=seed + 10_000 + outer_index,
                    ),
                )
            )
            records.append(
                PlannedFoldAssignment(
                    role=f"neural_validation_outer_{outer_index}",
                    seed=seed + 20_000 + outer_index,
                    fold_count=inner_fold_count,
                    participants=training,
                    assignment=_assignment(
                        training,
                        fold_count=inner_fold_count,
                        seed=seed + 20_000 + outer_index,
                    ),
                )
            )
        records.append(
            PlannedFoldAssignment(
                role="transfer_inner",
                seed=seed + 10_000,
                fold_count=inner_fold_count,
                participants=participants,
                assignment=_assignment(
                    participants,
                    fold_count=inner_fold_count,
                    seed=seed + 10_000,
                ),
            )
        )
    plan = ParticipantPartitionPlan(
        dataset_id=dataset_id,
        participant_roster=participants,
        records=tuple(records),
        roster_basis=roster_basis,
    )
    plan.validate()
    return plan
