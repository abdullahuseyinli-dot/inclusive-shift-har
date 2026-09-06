from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from inclusive_shift_har.data.participant_partitions import (
    PARTITION_PROTOCOL_ID,
    build_participant_partition_plan,
)


def _participants(count: int = 12) -> list[str]:
    return [f"person-{index:03d}" for index in range(count)]


def test_plan_freezes_outer_and_both_inner_protocols() -> None:
    participants = _participants()
    plan = build_participant_partition_plan(
        "synthetic",
        participants,
        roster_basis="synthetic provider inventory",
    )

    outer = plan.resolve(participants, fold_count=5, seed=11, role="outer")
    assert set(outer) == set(participants)
    assert set(outer.values()) == set(range(5))
    outer_training = [person for person in participants if outer[person] != 2]
    classical = plan.resolve(
        outer_training,
        fold_count=4,
        seed=10_013,
        role="classical_inner_outer_2",
    )
    neural = plan.resolve(
        outer_training,
        fold_count=4,
        seed=20_013,
        role="neural_validation_outer_2",
    )
    assert set(classical) == set(outer_training)
    assert set(neural) == set(outer_training)


def test_plan_is_order_invariant_and_self_hashed() -> None:
    first = build_participant_partition_plan(
        "synthetic", _participants(), roster_basis="provider inventory"
    )
    second = build_participant_partition_plan(
        "synthetic", list(reversed(_participants())), roster_basis="provider inventory"
    )
    assert first == second
    audit = first.audit()
    assert audit["protocol_id"] == PARTITION_PROTOCOL_ID
    assert audit["created_before_windowing"] is True
    assert len(audit["participant_roster_sha256"]) == 64
    assert len(audit["plan_sha256"]) == 64


def test_plan_refuses_post_window_roster_or_undeclared_seed() -> None:
    participants = _participants()
    plan = build_participant_partition_plan(
        "synthetic", participants, roster_basis="provider inventory"
    )
    with pytest.raises(PermissionError, match="not frozen before windowing"):
        plan.resolve(participants[:-1], fold_count=5, seed=11, role="outer")
    with pytest.raises(PermissionError, match="not frozen before windowing"):
        plan.resolve(participants, fold_count=5, seed=89, role="outer")


def test_plan_rejects_an_empty_participant_identifier() -> None:
    with pytest.raises(ValueError, match="cannot build"):
        build_participant_partition_plan(
            "synthetic",
            ["", *_participants()],
            roster_basis="provider inventory",
        )


def test_plan_validation_detects_assignment_tampering() -> None:
    plan = build_participant_partition_plan(
        "synthetic", _participants(), roster_basis="provider inventory"
    )
    record = plan.records[0]
    tampered = replace(
        plan,
        records=(replace(record, assignment=((record.participants[0], 99),)), *plan.records[1:]),
    )
    with pytest.raises(ValueError, match="incomplete or invalid"):
        tampered.validate()


def test_transfer_inner_is_frozen_over_complete_source_roster() -> None:
    participants = np.asarray(_participants(), dtype=np.str_)
    plan = build_participant_partition_plan(
        "synthetic", participants, roster_basis="provider inventory"
    )
    assignment = plan.resolve(
        participants,
        fold_count=4,
        seed=10_011,
        role="transfer_inner",
    )
    assert set(assignment) == set(participants.tolist())
    assert set(assignment.values()) == set(range(4))
