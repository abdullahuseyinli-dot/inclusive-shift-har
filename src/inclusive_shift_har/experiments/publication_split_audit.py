"""Create-only participant-split audits for external-HAR result packages.

The audit binds an immutable result and prediction package to a compact participant
manifest, then verifies every recorded outer/inner partition.  HAR-PMD classical
results produced before fold records were added can only receive an explicitly
post-hoc reconstruction status.  This module does not claim that participant
assignment happened before signal materialization; that execution-order property
requires a prospective pre-materialization manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, cast

import numpy as np

from inclusive_shift_har.artifacts.research_provenance import (
    _assert_executed_repository_root,
)
from inclusive_shift_har.data.external_har import participant_fold_assignment
from inclusive_shift_har.data.participant_partitions import (
    PARTITION_PROTOCOL_ID,
    PlannedFoldAssignment,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

_OUTER_FOLDS = 5
_INNER_FOLDS = 4
_IDENTITY_ARRAYS = ("labels", "participant_ids", "session_ids", "trial_ids", "window_ids")
_ASSIGNMENT_PROTOCOL = "sorted-unique-numpy-default-rng-round-robin-v1"


def _absolute_direct_child_name(value: Any, root: Any) -> str | None:
    """Return a legacy absolute artifact basename only when it is below ``root``."""

    if not isinstance(value, str) or not isinstance(root, str):
        return None
    for path_type in (PurePosixPath, PureWindowsPath):
        candidate = path_type(value)
        parent = path_type(root)
        if candidate.is_absolute() and parent.is_absolute() and candidate.parent == parent:
            return candidate.name
    return None


def _stable_split_audit_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Drop volatile fields and normalize only the legacy absolute input presentation."""

    stable = cast(dict[str, Any], json.loads(json.dumps(report)))
    stable.pop("created_at", None)
    stable.pop("record_sha256", None)
    inputs = stable.get("inputs")
    if not isinstance(inputs, dict):
        return stable
    legacy_root = inputs.get("run_directory")
    result = inputs.get("result")
    predictions = inputs.get("predictions")
    result_name = (
        _absolute_direct_child_name(result.get("path"), legacy_root)
        if isinstance(result, dict)
        else None
    )
    prediction_name = (
        _absolute_direct_child_name(predictions.get("path"), legacy_root)
        if isinstance(predictions, dict)
        else None
    )
    # Normalize the three path fields as one unit. A mismatched/tampered legacy
    # path remains visible and therefore cannot compare equal to reconstruction.
    if (
        isinstance(result, dict)
        and isinstance(predictions, dict)
        and result_name == "result.json"
        and prediction_name
    ):
        inputs["run_directory"] = "."
        result["path"] = result_name
        predictions["path"] = prediction_name
    return stable


@dataclass(frozen=True, slots=True)
class _PredictionIdentity:
    participant_ids: tuple[str, ...]
    session_ids: tuple[str, ...]
    trial_ids: tuple[str, ...]
    window_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _PartitionPlanIdentity:
    """Validated participant-plan identity embedded in a dataset summary."""

    dataset_id: str
    roster: tuple[str, ...]
    observed_participants: tuple[str, ...]
    plan_sha256: str
    records: tuple[PlannedFoldAssignment, ...]

    def assignment(
        self,
        *,
        role: str,
        seed: int,
        fold_count: int,
        participants: set[str],
    ) -> dict[str, int] | None:
        ordered = tuple(sorted(participants))
        matches = [
            record
            for record in self.records
            if record.role == role
            and record.seed == seed
            and record.fold_count == fold_count
            and record.participants == ordered
        ]
        return dict(matches[0].assignment) if len(matches) == 1 else None


@dataclass(slots=True)
class _Checks:
    """Collect named checks without hiding multiple violations behind one exception."""

    active: dict[str, bool] = field(default_factory=dict)
    failures: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def enable(self, name: str, *, blocking: bool = True) -> None:
        previous = self.active.setdefault(name, blocking)
        if previous != blocking:
            raise ValueError(f"check {name!r} was registered with conflicting severity")

    def require(
        self,
        name: str,
        condition: bool,
        detail: str,
        *,
        blocking: bool = True,
    ) -> None:
        self.enable(name, blocking=blocking)
        if not condition:
            self.failures[name].append(detail)

    def records(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "blocking": blocking,
                "passed": not self.failures.get(name),
                "details": self.failures.get(name, []),
            }
            for name, blocking in sorted(self.active.items())
        ]

    def blocking_passed(self) -> bool:
        return all(
            not self.failures.get(name) for name, blocking in self.active.items() if blocking
        )


def _object(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return cast(dict[str, Any], value)


def _objects(value: Any, *, name: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"{name} must be a list of JSON objects")
    return cast(list[dict[str, Any]], value)


def _safe_artifact_path(run_directory: Path, relative: Any) -> Path:
    if not isinstance(relative, str):
        raise ValueError("prediction artifact path is absent")
    parsed = PurePosixPath(relative)
    if (
        parsed.is_absolute()
        or not parsed.parts
        or ".." in parsed.parts
        or "\\" in relative
        or ":" in relative
    ):
        raise ValueError(f"prediction artifact path is unsafe: {relative!r}")
    candidate = run_directory.joinpath(*parsed.parts)
    resolved_root = run_directory.resolve(strict=True)
    resolved = candidate.resolve(strict=True)
    if candidate.absolute() != resolved:
        raise ValueError("prediction artifact path may not contain symbolic-link indirection")
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("prediction artifact escapes the run directory") from exc
    return resolved


def _string_vector(archive: Any, name: str, *, count: int | None = None) -> tuple[str, ...]:
    if name not in archive:
        raise ValueError(f"prediction artifact lacks {name}")
    values = np.asarray(archive[name])
    if values.ndim != 1 or (count is not None and values.size != count):
        raise ValueError(f"prediction identity array {name} is misaligned")
    output = tuple(str(item) for item in values.tolist())
    if any(not item for item in output):
        raise ValueError(f"prediction identity array {name} contains an empty value")
    return output


def _prediction_identity(path: Path, checks: _Checks) -> _PredictionIdentity:
    with np.load(path, allow_pickle=False) as archive:
        absent = sorted(set(_IDENTITY_ARRAYS) - set(archive.files))
        checks.require(
            "prediction_identity_arrays_present",
            not absent,
            f"missing={absent}",
        )
        if absent:
            raise ValueError(f"prediction artifact lacks identity arrays: {absent}")
        labels = np.asarray(archive["labels"])
        if labels.ndim != 1 or labels.size == 0:
            raise ValueError("prediction labels must be a non-empty vector")
        count = int(labels.size)
        participants = _string_vector(archive, "participant_ids", count=count)
        sessions = _string_vector(archive, "session_ids", count=count)
        trials = _string_vector(archive, "trial_ids", count=count)
        windows = _string_vector(archive, "window_ids", count=count)
    checks.require(
        "prediction_window_ids_unique",
        len(set(windows)) == len(windows),
        "window identifiers are duplicated",
    )
    session_owners: dict[str, str] = {}
    trial_owners: dict[str, tuple[str, str]] = {}
    hierarchy_errors: list[str] = []
    for participant, session, trial in zip(participants, sessions, trials, strict=True):
        existing_session = session_owners.setdefault(session, participant)
        if existing_session != participant:
            hierarchy_errors.append(f"session {session!r} spans participants")
        existing_trial = trial_owners.setdefault(trial, (participant, session))
        if existing_trial != (participant, session):
            hierarchy_errors.append(f"trial {trial!r} spans participant/session groups")
    checks.require(
        "prediction_metadata_hierarchy",
        not hierarchy_errors,
        "; ".join(hierarchy_errors[:10]),
    )
    return _PredictionIdentity(participants, sessions, trials, windows)


def _participant_manifest(identity: _PredictionIdentity) -> tuple[list[str], list[dict[str, Any]]]:
    participants = sorted(set(identity.participant_ids))
    counts = Counter(identity.participant_ids)
    positions: dict[str, list[str]] = defaultdict(list)
    for participant, window in zip(identity.participant_ids, identity.window_ids, strict=True):
        positions[participant].append(window)
    records = [
        {
            "participant_id": participant,
            "window_count": counts[participant],
            "window_ids_sha256": canonical_json_sha256(sorted(positions[participant])),
        }
        for participant in participants
    ]
    return participants, records


def _partition_plan_identity(
    summary: Any,
    *,
    prediction_participants: set[str] | None,
    checks: _Checks,
    context: str,
) -> _PartitionPlanIdentity | None:
    """Validate one embedded pre-window plan without conflating it with scored rows."""

    value = _object(summary, name=f"{context} dataset summary")
    raw_plan = value.get("participant_partition_plan")
    if raw_plan is None:
        checks.require(
            "prewindow_partition_plan_present",
            False,
            f"{context}: participant partition plan is absent",
            blocking=False,
        )
        return None
    checks.require(
        "prewindow_partition_plan_present",
        isinstance(raw_plan, dict),
        f"{context}: participant partition plan is not an object",
    )
    if not isinstance(raw_plan, dict):
        return None
    plan = cast(dict[str, Any], raw_plan)
    dataset_id = value.get("dataset_id")
    roster_value = plan.get("participant_roster")
    roster_valid = (
        isinstance(roster_value, list)
        and len(roster_value) >= _OUTER_FOLDS
        and all(isinstance(item, str) and item for item in roster_value)
        and roster_value == sorted(set(roster_value))
    )
    checks.require(
        "prewindow_plan_roster_valid",
        roster_valid,
        f"{context}: invalid participant roster",
    )
    roster = tuple(cast(list[str], roster_value)) if roster_valid else ()
    checks.require(
        "prewindow_plan_dataset_binding",
        isinstance(dataset_id, str) and bool(dataset_id) and plan.get("dataset_id") == dataset_id,
        (f"{context}: summary dataset={dataset_id!r}, plan dataset={plan.get('dataset_id')!r}"),
    )
    checks.require(
        "prewindow_plan_protocol",
        plan.get("protocol_id") == PARTITION_PROTOCOL_ID
        and plan.get("created_before_windowing") is True
        and isinstance(plan.get("roster_basis"), str)
        and bool(plan.get("roster_basis")),
        f"{context}: protocol/timing/roster-basis metadata is invalid",
    )
    checks.require(
        "prewindow_plan_roster_hash",
        roster_valid
        and plan.get("participant_roster_sha256") == canonical_json_sha256(list(roster)),
        f"{context}: participant roster hash differs",
    )
    declared_digest = plan.get("plan_sha256")
    unhashed_plan = dict(plan)
    unhashed_plan.pop("plan_sha256", None)
    checks.require(
        "prewindow_plan_self_hash",
        isinstance(declared_digest, str)
        and len(declared_digest) == 64
        and canonical_json_sha256(unhashed_plan) == declared_digest,
        f"{context}: plan self-hash is absent or differs",
    )

    record_errors: list[str] = []
    parsed_records: list[PlannedFoldAssignment] = []
    raw_records = plan.get("records")
    if not isinstance(raw_records, list) or not raw_records:
        record_errors.append("records are absent or not a non-empty list")
    else:
        for index, raw_record in enumerate(raw_records):
            if not isinstance(raw_record, dict):
                record_errors.append(f"record {index} is not an object")
                continue
            participants = raw_record.get("participants")
            assignment = raw_record.get("assignment")
            seed = raw_record.get("seed")
            fold_count = raw_record.get("fold_count")
            role = raw_record.get("role")
            if (
                not isinstance(role, str)
                or not role
                or not isinstance(seed, int)
                or isinstance(seed, bool)
                or not isinstance(fold_count, int)
                or isinstance(fold_count, bool)
                or not isinstance(participants, list)
                or not all(isinstance(item, str) and item for item in participants)
                or not isinstance(assignment, dict)
                or not all(
                    isinstance(key, str)
                    and key
                    and isinstance(fold, int)
                    and not isinstance(fold, bool)
                    for key, fold in assignment.items()
                )
            ):
                record_errors.append(f"record {index} has invalid fields")
                continue
            parsed = PlannedFoldAssignment(
                role=role,
                seed=seed,
                fold_count=fold_count,
                participants=tuple(cast(list[str], participants)),
                assignment=tuple(sorted(cast(dict[str, int], assignment).items())),
            )
            try:
                parsed.validate()
            except ValueError as exc:
                record_errors.append(f"record {index}: {exc}")
                continue
            if not set(parsed.participants).issubset(roster):
                record_errors.append(f"record {index} contains a participant outside the roster")
                continue
            parsed_records.append(parsed)
    record_keys = [
        (item.role, item.seed, item.fold_count, item.participants) for item in parsed_records
    ]
    if len(record_keys) != len(set(record_keys)):
        record_errors.append("assignment records contain duplicate identities")
    checks.require(
        "prewindow_plan_records_valid",
        not record_errors,
        f"{context}: {'; '.join(record_errors[:10])}",
    )

    observation = value.get("participant_partition_observation")
    observed: set[str] = set()
    observation_valid = isinstance(observation, dict)
    if isinstance(observation, dict):
        missing = observation.get("participants_without_retained_windows")
        missing_valid = (
            isinstance(missing, list)
            and all(isinstance(item, str) and item for item in missing)
            and missing == sorted(set(missing))
            and set(missing).issubset(roster)
        )
        if missing_valid:
            observed = set(roster) - set(cast(list[str], missing))
        observation_valid = bool(
            missing_valid
            and observation.get("planned_participant_count") == len(roster)
            and observation.get("observed_window_participant_count") == len(observed)
            and value.get("participant_count") == len(observed)
        )
    checks.require(
        "participant_partition_observation_valid",
        observation_valid,
        f"{context}: planned/observed participant accounting is absent or inconsistent",
    )
    if prediction_participants is not None:
        checks.require(
            "prediction_participants_within_plan",
            prediction_participants <= set(roster),
            f"{context}: prediction participant lies outside the plan roster",
        )
        checks.require(
            "prediction_participants_match_observed",
            prediction_participants == observed,
            (
                f"{context}: predictions={sorted(prediction_participants)}, "
                f"observed={sorted(observed)}"
            ),
        )
    return _PartitionPlanIdentity(
        dataset_id=str(dataset_id),
        roster=roster,
        observed_participants=tuple(sorted(observed)),
        plan_sha256=declared_digest if isinstance(declared_digest, str) else "",
        records=tuple(parsed_records),
    )


def _participant_set(value: Any, *, context: str, checks: _Checks) -> set[str]:
    valid = (
        isinstance(value, list)
        and all(isinstance(item, str) and item for item in value)
        and len(value) == len(set(value))
    )
    checks.require("participant_lists_valid", valid, context)
    if not valid:
        return set()
    return set(cast(list[str], value))


def _seed_list(value: Any, checks: _Checks) -> list[int]:
    valid = (
        isinstance(value, list)
        and bool(value)
        and all(
            isinstance(seed, int) and not isinstance(seed, bool) and seed >= 0 for seed in value
        )
        and len(value) == len(set(value))
    )
    checks.require("seeds_valid", valid, f"seeds={value!r}")
    return cast(list[int], value) if valid else []


def _fold_number(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _expected_group(participants: set[str], *, count: int, seed: int, fold: int) -> set[str]:
    assignment = participant_fold_assignment(sorted(participants), fold_count=count, seed=seed)
    return {participant for participant, assigned in assignment.items() if assigned == fold}


def _planned_group(
    participants: set[str],
    *,
    count: int,
    seed: int,
    fold: int,
    role: str,
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
    context: str,
) -> set[str]:
    if plan is None:
        return _expected_group(participants, count=count, seed=seed, fold=fold)
    assignment = plan.assignment(
        role=role,
        seed=seed,
        fold_count=count,
        participants=participants,
    )
    checks.require(
        "executed_assignment_declared_in_prewindow_plan",
        assignment is not None,
        (f"{context}: no unique plan record for role={role!r}, seed={seed}, fold_count={count}"),
    )
    if assignment is None:
        return set()
    return {participant for participant, assigned in assignment.items() if assigned == fold}


def _check_plan_hash(
    record: dict[str, Any],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
    *,
    context: str,
    field: str = "participant_partition_plan_sha256",
) -> None:
    if plan is None:
        return
    checks.require(
        "executed_partition_records_bind_plan_hash",
        record.get(field) == plan.plan_sha256,
        (f"{context}: {field}={record.get(field)!r}, expected={plan.plan_sha256!r}"),
    )


def _check_observed_subset(
    record: dict[str, Any],
    *,
    field: str,
    assigned: set[str],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
    context: str,
    check_name: str,
    required: bool,
) -> set[str] | None:
    """Bind an executed supervision/scoring roster to the plan's observed subset."""

    if plan is None:
        return None
    present = field in record
    checks.require(
        f"{check_name}_present",
        present or not required,
        f"{context}: {field} is absent",
    )
    if not present:
        return None
    actual = _participant_set(
        record.get(field),
        context=f"{context}: {field}",
        checks=checks,
    )
    expected = assigned & set(plan.observed_participants)
    checks.require(
        check_name,
        actual == expected,
        f"{context}: {field}={sorted(actual)}, expected={sorted(expected)}",
    )
    return actual


def _check_dataset_summary(
    summary: Any,
    *,
    participants: set[str],
    window_count: int,
    checks: _Checks,
) -> None:
    value = _object(summary, name="dataset summary")
    checks.require(
        "dataset_summary_matches_predictions",
        value.get("participant_count") == len(participants)
        and value.get("window_count") == window_count,
        (
            f"summary participants/windows={value.get('participant_count')}/"
            f"{value.get('window_count')}, predictions={len(participants)}/{window_count}"
        ),
    )


def _nested_partitions(
    result: dict[str, Any],
    *,
    participants: set[str],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
) -> list[dict[str, Any]]:
    seeds = _seed_list(result.get("seeds"), checks)
    records = _objects(result.get("fold_records"), name="fold_records")
    checks.require("partition_records_present", bool(records), "nested fold records are absent")
    by_seed: dict[int, dict[str, Any]] = {}
    duplicate_seeds: list[int] = []
    for record in records:
        seed = _fold_number(record.get("seed"))
        if seed is None:
            checks.require("partition_record_keys_valid", False, "nested record has invalid seed")
            continue
        if seed in by_seed:
            duplicate_seeds.append(seed)
        by_seed[seed] = record
    checks.require(
        "partition_record_keys_valid",
        not duplicate_seeds and set(by_seed) == set(seeds),
        f"record seeds={sorted(by_seed)}, expected={sorted(seeds)}, duplicates={duplicate_seeds}",
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        selected_seed_record = by_seed.get(seed)
        if selected_seed_record is None:
            continue
        folds = _objects(selected_seed_record.get("folds"), name=f"fold_records[{seed}].folds")
        by_outer: dict[int, dict[str, Any]] = {}
        duplicate_outer: list[int] = []
        for fold_record in folds:
            fold = _fold_number(fold_record.get("outer_fold"))
            if fold is None:
                checks.require("outer_fold_ids_exact", False, f"seed {seed}: invalid fold id")
                continue
            if fold in by_outer:
                duplicate_outer.append(fold)
            by_outer[fold] = fold_record
        checks.require(
            "outer_fold_ids_exact",
            not duplicate_outer and set(by_outer) == set(range(_OUTER_FOLDS)),
            (
                f"seed {seed}: folds={sorted(by_outer)}, expected={list(range(_OUTER_FOLDS))}, "
                f"duplicates={duplicate_outer}"
            ),
        )
        evaluation_groups: list[set[str]] = []
        outer_output: list[dict[str, Any]] = []
        for outer_fold in range(_OUTER_FOLDS):
            selected_outer_record = by_outer.get(outer_fold)
            if selected_outer_record is None:
                continue
            context = f"seed {seed}, outer {outer_fold}"
            _check_plan_hash(selected_outer_record, plan, checks, context=context)
            training = _participant_set(
                selected_outer_record.get("training_participants"),
                context=f"{context}: training participants",
                checks=checks,
            )
            evaluation = _participant_set(
                selected_outer_record.get("evaluation_participants"),
                context=f"{context}: evaluation participants",
                checks=checks,
            )
            evaluation_groups.append(evaluation)
            checks.require(
                "outer_partitions_disjoint",
                not training & evaluation,
                f"{context}: overlap={sorted(training & evaluation)}",
            )
            checks.require(
                "outer_training_complement",
                training == participants - evaluation,
                f"{context}: training is not the cohort complement",
            )
            expected_outer_training = participants - evaluation
            checks.require(
                "deterministic_outer_assignment",
                evaluation
                == _planned_group(
                    participants,
                    count=_OUTER_FOLDS,
                    seed=seed,
                    fold=outer_fold,
                    role="outer",
                    plan=plan,
                    checks=checks,
                    context=context,
                ),
                f"{context}: evaluation group differs from seeded assignment",
            )
            _check_observed_subset(
                selected_outer_record,
                field="training_participants_with_supervision",
                assigned=training,
                plan=plan,
                checks=checks,
                context=context,
                check_name="recorded_supervision_rosters_match_plan_observation",
                required=True,
            )
            _check_observed_subset(
                selected_outer_record,
                field="evaluation_participants_with_scoring",
                assigned=evaluation,
                plan=plan,
                checks=checks,
                context=context,
                check_name="recorded_scoring_rosters_match_plan_observation",
                required=False,
            )
            inner_records = _objects(
                selected_outer_record.get("inner_folds"), name=f"{context}.inner_folds"
            )
            by_inner: dict[int, dict[str, Any]] = {}
            duplicate_inner: list[int] = []
            for inner_record in inner_records:
                inner = _fold_number(inner_record.get("inner_fold"))
                if inner is None:
                    checks.require("inner_fold_ids_exact", False, f"{context}: invalid inner id")
                    continue
                if inner in by_inner:
                    duplicate_inner.append(inner)
                by_inner[inner] = inner_record
            checks.require(
                "inner_fold_ids_exact",
                not duplicate_inner and set(by_inner) == set(range(_INNER_FOLDS)),
                (
                    f"{context}: folds={sorted(by_inner)}, expected={list(range(_INNER_FOLDS))}, "
                    f"duplicates={duplicate_inner}"
                ),
            )
            validation_groups: list[set[str]] = []
            inner_output: list[dict[str, Any]] = []
            for inner_fold in range(_INNER_FOLDS):
                selected_inner_record = by_inner.get(inner_fold)
                if selected_inner_record is None:
                    continue
                inner_context = f"{context}, inner {inner_fold}"
                _check_plan_hash(selected_inner_record, plan, checks, context=inner_context)
                inner_training = _participant_set(
                    selected_inner_record.get("training_participants"),
                    context=f"{inner_context}: training participants",
                    checks=checks,
                )
                validation = _participant_set(
                    selected_inner_record.get("validation_participants"),
                    context=f"{inner_context}: validation participants",
                    checks=checks,
                )
                validation_groups.append(validation)
                checks.require(
                    "inner_contained_in_outer_training",
                    (inner_training | validation) <= expected_outer_training,
                    f"{inner_context}: participant outside outer training",
                )
                checks.require(
                    "inner_partitions_disjoint",
                    not inner_training & validation,
                    f"{inner_context}: overlap={sorted(inner_training & validation)}",
                )
                checks.require(
                    "inner_training_complement",
                    inner_training == expected_outer_training - validation,
                    f"{inner_context}: training is not the inner complement",
                )
                checks.require(
                    "deterministic_inner_assignment",
                    validation
                    == _planned_group(
                        expected_outer_training,
                        count=_INNER_FOLDS,
                        seed=seed + 10_000 + outer_fold,
                        fold=inner_fold,
                        role=f"classical_inner_outer_{outer_fold}",
                        plan=plan,
                        checks=checks,
                        context=inner_context,
                    ),
                    f"{inner_context}: validation group differs from seeded assignment",
                )
                _check_observed_subset(
                    selected_inner_record,
                    field="training_participants_with_supervision",
                    assigned=inner_training,
                    plan=plan,
                    checks=checks,
                    context=inner_context,
                    check_name="recorded_supervision_rosters_match_plan_observation",
                    required=False,
                )
                _check_observed_subset(
                    selected_inner_record,
                    field="validation_participants_with_supervision",
                    assigned=validation,
                    plan=plan,
                    checks=checks,
                    context=inner_context,
                    check_name="recorded_supervision_rosters_match_plan_observation",
                    required=False,
                )
                inner_output.append(
                    {
                        "inner_fold": inner_fold,
                        "training_participants": sorted(inner_training),
                        "validation_participants": sorted(validation),
                    }
                )
            validation_union = set().union(*validation_groups) if validation_groups else set()
            validation_total = sum(len(group) for group in validation_groups)
            checks.require(
                "inner_partitions_exhaustive",
                validation_union == expected_outer_training
                and validation_total == len(validation_union),
                f"{context}: inner validation folds do not cover outer training exactly once",
            )
            outer_output.append(
                {
                    "outer_fold": outer_fold,
                    "training_participants": sorted(training),
                    "evaluation_participants": sorted(evaluation),
                    "inner_folds": inner_output,
                }
            )
        evaluation_union = set().union(*evaluation_groups) if evaluation_groups else set()
        evaluation_total = sum(len(group) for group in evaluation_groups)
        checks.require(
            "outer_partitions_exhaustive",
            evaluation_union == participants and evaluation_total == len(evaluation_union),
            f"seed {seed}: outer evaluation folds do not cover the cohort exactly once",
        )
        output.append({"seed": seed, "outer_folds": outer_output})
    return output


def _flat_partitions(
    result: dict[str, Any],
    *,
    participants: set[str],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
) -> tuple[list[dict[str, Any]], str]:
    seeds = _seed_list(result.get("seeds"), checks)
    records = _objects(result.get("fold_records"), name="fold_records")
    checks.require("partition_records_present", bool(records), "flat fold records are absent")
    group_key = "model" if any("model" in record for record in records) else "method"
    report_names = (
        set(cast(dict[str, Any], result.get("reports", {})))
        if isinstance(result.get("reports"), dict)
        else set()
    )
    groups = sorted(
        {
            str(record[group_key])
            for record in records
            if isinstance(record.get(group_key), str) and record[group_key]
        }
    )
    checks.require(
        "record_groups_match_reports",
        bool(groups) and (not report_names or set(groups) == report_names),
        f"record groups={groups}, reported groups={sorted(report_names)}",
    )
    keyed: dict[tuple[int, str, int], dict[str, Any]] = {}
    duplicate_keys: list[tuple[int, str, int]] = []
    for record in records:
        seed = _fold_number(record.get("seed"))
        fold = _fold_number(record.get("outer_fold"))
        group = record.get(group_key)
        if seed is None or fold is None or not isinstance(group, str) or not group:
            checks.require("partition_record_keys_valid", False, "flat record key is invalid")
            continue
        key = (seed, group, fold)
        if key in keyed:
            duplicate_keys.append(key)
        keyed[key] = record
    expected_keys = {
        (seed, group, fold) for seed in seeds for group in groups for fold in range(_OUTER_FOLDS)
    }
    checks.require(
        "partition_record_keys_valid",
        not duplicate_keys and set(keyed) == expected_keys,
        f"missing={sorted(expected_keys - set(keyed))}, extra={sorted(set(keyed) - expected_keys)}",
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        canonical_by_fold: dict[int, tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]] = {}
        evaluation_groups: list[set[str]] = []
        outer_output: list[dict[str, Any]] = []
        for outer_fold in range(_OUTER_FOLDS):
            representative: tuple[set[str], set[str], set[str]] | None = None
            for group in groups:
                selected_record = keyed.get((seed, group, outer_fold))
                if selected_record is None:
                    continue
                context = f"seed {seed}, outer {outer_fold}, {group_key} {group}"
                _check_plan_hash(selected_record, plan, checks, context=context)
                training = _participant_set(
                    selected_record.get("training_participants"),
                    context=f"{context}: training participants",
                    checks=checks,
                )
                evaluation = _participant_set(
                    selected_record.get("evaluation_participants"),
                    context=f"{context}: evaluation participants",
                    checks=checks,
                )
                has_validation = "validation_participants" in selected_record
                validation = (
                    _participant_set(
                        selected_record.get("validation_participants"),
                        context=f"{context}: validation participants",
                        checks=checks,
                    )
                    if has_validation
                    else set()
                )
                checks.require(
                    "outer_partitions_disjoint",
                    not (training & evaluation or training & validation or validation & evaluation),
                    f"{context}: train/validation/evaluation overlap",
                )
                checks.require(
                    "outer_training_complement",
                    training | validation == participants - evaluation,
                    f"{context}: training plus validation is not the outer complement",
                )
                expected_evaluation = _planned_group(
                    participants,
                    count=_OUTER_FOLDS,
                    seed=seed,
                    fold=outer_fold,
                    role="outer",
                    plan=plan,
                    checks=checks,
                    context=context,
                )
                checks.require(
                    "deterministic_outer_assignment",
                    evaluation == expected_evaluation,
                    f"{context}: evaluation group differs from seeded assignment",
                )
                if has_validation:
                    expected_validation = _planned_group(
                        participants - evaluation,
                        count=_INNER_FOLDS,
                        seed=seed + 20_000 + outer_fold,
                        fold=0,
                        role=f"neural_validation_outer_{outer_fold}",
                        plan=plan,
                        checks=checks,
                        context=context,
                    )
                    checks.require(
                        "deterministic_validation_assignment",
                        validation == expected_validation,
                        f"{context}: validation group differs from seeded assignment",
                    )
                    checks.require(
                        "validation_contained_in_outer_training",
                        validation <= participants - evaluation,
                        f"{context}: validation participant lies outside outer training",
                    )
                _check_observed_subset(
                    selected_record,
                    field="training_participants_with_supervision",
                    assigned=training,
                    plan=plan,
                    checks=checks,
                    context=context,
                    check_name="recorded_supervision_rosters_match_plan_observation",
                    required=True,
                )
                if has_validation:
                    _check_observed_subset(
                        selected_record,
                        field="validation_participants_with_supervision",
                        assigned=validation,
                        plan=plan,
                        checks=checks,
                        context=context,
                        check_name="recorded_supervision_rosters_match_plan_observation",
                        required=True,
                    )
                scoring = _check_observed_subset(
                    selected_record,
                    field="evaluation_participants_with_scoring",
                    assigned=evaluation,
                    plan=plan,
                    checks=checks,
                    context=context,
                    check_name="recorded_scoring_rosters_match_plan_observation",
                    required=True,
                )
                if group_key == "model":
                    candidates = _participant_set(
                        selected_record.get("evaluation_participants_with_candidates"),
                        context=f"{context}: evaluation candidate participants",
                        checks=checks,
                    )
                    checks.require(
                        "recorded_candidate_rosters_within_assigned_folds",
                        candidates <= evaluation and (scoring is None or scoring <= candidates),
                        f"{context}: candidate/scoring rosters are not nested in evaluation",
                    )
                current = (
                    tuple(sorted(training)),
                    tuple(sorted(validation)),
                    tuple(sorted(evaluation)),
                )
                if outer_fold in canonical_by_fold:
                    checks.require(
                        "cross_method_partition_consistency",
                        canonical_by_fold[outer_fold] == current,
                        f"seed {seed}, outer {outer_fold}: {group} differs from peer methods",
                    )
                else:
                    canonical_by_fold[outer_fold] = current
                if representative is None:
                    representative = (training, validation, evaluation)
            if representative is None:
                continue
            training, validation, evaluation = representative
            evaluation_groups.append(evaluation)
            outer_record: dict[str, Any] = {
                "outer_fold": outer_fold,
                "training_participants": sorted(training),
                "evaluation_participants": sorted(evaluation),
            }
            if validation:
                outer_record["validation_participants"] = sorted(validation)
                outer_record["validation_protocol"] = (
                    "fold-0 of a four-way participant assignment; not rotating inner CV"
                )
            outer_output.append(outer_record)
        evaluation_union = set().union(*evaluation_groups) if evaluation_groups else set()
        evaluation_total = sum(len(group) for group in evaluation_groups)
        checks.require(
            "outer_partitions_exhaustive",
            evaluation_union == participants and evaluation_total == len(evaluation_union),
            f"seed {seed}: outer evaluation folds do not cover the cohort exactly once",
        )
        output.append({"seed": seed, "outer_folds": outer_output})
    return output, group_key


def _shared_outer_partitions(
    result: dict[str, Any],
    *,
    participants: set[str],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
) -> list[dict[str, Any]]:
    """Audit one shared seed/fold record used by every reported HAR-PMD method."""

    seeds = _seed_list(result.get("seeds"), checks)
    records = _objects(result.get("fold_records"), name="fold_records")
    reports = result.get("reports")
    checks.require(
        "shared_partition_applies_to_reported_methods",
        isinstance(reports, dict) and bool(reports),
        "shared fold records have no reported methods",
    )
    keyed: dict[tuple[int, int], dict[str, Any]] = {}
    duplicates: list[tuple[int, int]] = []
    for record in records:
        seed = _fold_number(record.get("seed"))
        fold = _fold_number(record.get("outer_fold"))
        if seed is None or fold is None:
            checks.require(
                "partition_record_keys_valid", False, "shared outer record key is invalid"
            )
            continue
        key = (seed, fold)
        if key in keyed:
            duplicates.append(key)
        keyed[key] = record
    expected_keys = {(seed, fold) for seed in seeds for fold in range(_OUTER_FOLDS)}
    checks.require(
        "partition_record_keys_valid",
        not duplicates and set(keyed) == expected_keys,
        f"missing={sorted(expected_keys - set(keyed))}, extra={sorted(set(keyed) - expected_keys)}",
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        evaluation_groups: list[set[str]] = []
        folds: list[dict[str, Any]] = []
        for fold in range(_OUTER_FOLDS):
            selected = keyed.get((seed, fold))
            if selected is None:
                continue
            context = f"seed {seed}, outer {fold}, shared methods"
            _check_plan_hash(selected, plan, checks, context=context)
            training = _participant_set(
                selected.get("training_participants"),
                context=f"{context}: training participants",
                checks=checks,
            )
            evaluation = _participant_set(
                selected.get("evaluation_participants"),
                context=f"{context}: evaluation participants",
                checks=checks,
            )
            evaluation_groups.append(evaluation)
            checks.require(
                "outer_partitions_disjoint",
                not training & evaluation,
                f"{context}: overlap={sorted(training & evaluation)}",
            )
            checks.require(
                "outer_training_complement",
                training == participants - evaluation,
                f"{context}: training is not the cohort complement",
            )
            checks.require(
                "deterministic_outer_assignment",
                evaluation
                == _planned_group(
                    participants,
                    count=_OUTER_FOLDS,
                    seed=seed,
                    fold=fold,
                    role="outer",
                    plan=plan,
                    checks=checks,
                    context=context,
                ),
                f"{context}: evaluation group differs from seeded assignment",
            )
            folds.append(
                {
                    "outer_fold": fold,
                    "training_participants": sorted(training),
                    "evaluation_participants": sorted(evaluation),
                }
            )
        union = set().union(*evaluation_groups) if evaluation_groups else set()
        total = sum(len(group) for group in evaluation_groups)
        checks.require(
            "outer_partitions_exhaustive",
            union == participants and total == len(union),
            f"seed {seed}: outer evaluation folds do not cover the cohort exactly once",
        )
        output.append({"seed": seed, "outer_folds": folds})
    return output


def _transfer_partitions(
    result: dict[str, Any],
    *,
    target_prediction_participants: set[str],
    source_plan: _PartitionPlanIdentity | None,
    target_plan: _PartitionPlanIdentity | None,
    checks: _Checks,
) -> tuple[list[dict[str, Any]], set[str]]:
    source_summary = _object(result.get("source_dataset"), name="source dataset summary")
    target_summary = _object(result.get("target_dataset"), name="target dataset summary")
    checks.require(
        "transfer_dataset_identity",
        source_summary.get("dataset_id") == "imu_har_il_v1"
        and target_summary.get("dataset_id") == "fog_star_v3",
        (
            f"source={source_summary.get('dataset_id')!r}, "
            f"target={target_summary.get('dataset_id')!r}"
        ),
    )
    checks.require(
        "transfer_prewindow_plans_present",
        source_plan is not None and target_plan is not None,
        "zero-shot transfer lacks a source or target pre-window participant plan",
    )
    if source_plan is not None and target_plan is not None:
        checks.require(
            "transfer_plan_rosters_disjoint",
            not set(source_plan.roster) & set(target_plan.roster),
            "source and target pre-window plan rosters overlap",
        )
    seeds = _seed_list(result.get("seeds"), checks)
    records = _objects(result.get("seed_records"), name="seed_records")
    checks.require("partition_records_present", bool(records), "transfer seed records are absent")
    by_seed: dict[int, dict[str, Any]] = {}
    duplicate_seeds: list[int] = []
    for record in records:
        seed = _fold_number(record.get("seed"))
        if seed is None:
            checks.require("partition_record_keys_valid", False, "transfer record seed is invalid")
            continue
        if seed in by_seed:
            duplicate_seeds.append(seed)
        by_seed[seed] = record
    checks.require(
        "partition_record_keys_valid",
        not duplicate_seeds and set(by_seed) == set(seeds),
        f"record seeds={sorted(by_seed)}, expected={sorted(seeds)}, duplicates={duplicate_seeds}",
    )
    source_reference: set[str] | None = None
    target_reference: set[str] | None = None
    output: list[dict[str, Any]] = []
    for seed in seeds:
        selected_seed_record = by_seed.get(seed)
        if selected_seed_record is None:
            continue
        source = _participant_set(
            selected_seed_record.get("source_participants"),
            context=f"transfer seed {seed}: source participants",
            checks=checks,
        )
        target = _participant_set(
            selected_seed_record.get("target_participants"),
            context=f"transfer seed {seed}: target participants",
            checks=checks,
        )
        _check_plan_hash(
            selected_seed_record,
            source_plan,
            checks,
            context=f"transfer seed {seed}: source plan",
            field="source_participant_partition_plan_sha256",
        )
        _check_plan_hash(
            selected_seed_record,
            target_plan,
            checks,
            context=f"transfer seed {seed}: target plan",
            field="target_participant_partition_plan_sha256",
        )
        if source_plan is not None:
            checks.require(
                "transfer_source_matches_plan_roster",
                source == set(source_plan.roster),
                f"seed {seed}: recorded source differs from its plan roster",
            )
            source_with_supervision = _participant_set(
                selected_seed_record.get("source_participants_with_supervision"),
                context=f"transfer seed {seed}: supervised source participants",
                checks=checks,
            )
            checks.require(
                "transfer_source_observation_matches_plan",
                source_with_supervision == set(source_plan.observed_participants),
                f"seed {seed}: supervised source differs from plan observation",
            )
        if target_plan is not None:
            checks.require(
                "transfer_target_matches_plan_roster",
                target == set(target_plan.roster),
                f"seed {seed}: recorded target differs from its plan roster",
            )
            target_with_scoring = _participant_set(
                selected_seed_record.get("target_participants_with_scoring"),
                context=f"transfer seed {seed}: scored target participants",
                checks=checks,
            )
            checks.require(
                "transfer_target_scoring_matches_predictions",
                target_with_scoring == target_prediction_participants,
                f"seed {seed}: scored target roster differs from predictions",
            )
            target_with_candidates = _participant_set(
                selected_seed_record.get("target_participants_with_candidates"),
                context=f"transfer seed {seed}: candidate target participants",
                checks=checks,
            )
            checks.require(
                "transfer_target_candidate_roster_valid",
                target_with_scoring <= target_with_candidates <= target,
                f"seed {seed}: target scored/candidate/plan sets are not nested",
            )
        checks.require(
            "source_target_disjoint",
            not source & target,
            f"seed {seed}: overlap={sorted(source & target)}",
        )
        checks.require(
            "transfer_target_matches_predictions",
            target_prediction_participants <= target
            if target_plan is not None
            else target_prediction_participants == target,
            f"seed {seed}: prediction participant lies outside target roster",
        )
        if source_reference is None:
            source_reference = source
            target_reference = target
        else:
            checks.require(
                "source_target_consistent_across_seeds",
                source == source_reference and target == target_reference,
                f"seed {seed}: source/target roster differs from the first seed",
            )
        inner_records = _objects(
            selected_seed_record.get("inner_folds"), name=f"seed_records[{seed}].inner"
        )
        by_inner: dict[int, dict[str, Any]] = {}
        duplicate_inner: list[int] = []
        for inner_record in inner_records:
            inner = _fold_number(inner_record.get("inner_fold"))
            if inner is None:
                checks.require("inner_fold_ids_exact", False, f"seed {seed}: invalid inner id")
                continue
            if inner in by_inner:
                duplicate_inner.append(inner)
            by_inner[inner] = inner_record
        checks.require(
            "inner_fold_ids_exact",
            not duplicate_inner and set(by_inner) == set(range(_INNER_FOLDS)),
            f"seed {seed}: folds={sorted(by_inner)}, duplicates={duplicate_inner}",
        )
        validation_groups: list[set[str]] = []
        inner_output: list[dict[str, Any]] = []
        for inner_fold in range(_INNER_FOLDS):
            selected_inner_record = by_inner.get(inner_fold)
            if selected_inner_record is None:
                continue
            context = f"transfer seed {seed}, inner {inner_fold}"
            _check_plan_hash(selected_inner_record, source_plan, checks, context=context)
            training = _participant_set(
                selected_inner_record.get("training_participants"),
                context=f"{context}: training participants",
                checks=checks,
            )
            validation = _participant_set(
                selected_inner_record.get("validation_participants"),
                context=f"{context}: validation participants",
                checks=checks,
            )
            validation_groups.append(validation)
            checks.require(
                "inner_contained_in_source",
                (training | validation) <= source,
                f"{context}: participant outside source roster",
            )
            checks.require(
                "inner_partitions_disjoint",
                not training & validation,
                f"{context}: overlap={sorted(training & validation)}",
            )
            checks.require(
                "inner_training_complement",
                training == source - validation,
                f"{context}: training is not the source complement",
            )
            checks.require(
                "deterministic_inner_assignment",
                validation
                == _planned_group(
                    source,
                    count=_INNER_FOLDS,
                    seed=seed + 10_000,
                    fold=inner_fold,
                    role="transfer_inner",
                    plan=source_plan,
                    checks=checks,
                    context=context,
                ),
                f"{context}: validation group differs from seeded assignment",
            )
            _check_observed_subset(
                selected_inner_record,
                field="training_participants_with_supervision",
                assigned=training,
                plan=source_plan,
                checks=checks,
                context=context,
                check_name="recorded_supervision_rosters_match_plan_observation",
                required=False,
            )
            _check_observed_subset(
                selected_inner_record,
                field="validation_participants_with_supervision",
                assigned=validation,
                plan=source_plan,
                checks=checks,
                context=context,
                check_name="recorded_supervision_rosters_match_plan_observation",
                required=False,
            )
            inner_output.append(
                {
                    "inner_fold": inner_fold,
                    "training_participants": sorted(training),
                    "validation_participants": sorted(validation),
                }
            )
        validation_union = set().union(*validation_groups) if validation_groups else set()
        validation_total = sum(len(group) for group in validation_groups)
        checks.require(
            "inner_partitions_exhaustive",
            validation_union == source and validation_total == len(validation_union),
            f"seed {seed}: source inner folds do not cover the source roster exactly once",
        )
        output.append({"seed": seed, "source_inner_folds": inner_output})
    source_participants = source_reference or set()
    expected_observed_source_count = (
        len(source_plan.observed_participants)
        if source_plan is not None
        else len(source_participants)
    )
    checks.require(
        "source_dataset_summary_matches_records",
        source_summary.get("participant_count") == expected_observed_source_count,
        (
            f"source summary participant_count={source_summary.get('participant_count')}, "
            f"observed={expected_observed_source_count}"
        ),
    )
    return output, source_participants


def _posthoc_har_pmd_partitions(
    result: dict[str, Any],
    *,
    participants: set[str],
    plan: _PartitionPlanIdentity | None,
    checks: _Checks,
) -> list[dict[str, Any]]:
    seeds = _seed_list(result.get("seeds"), checks)
    checks.require(
        "recorded_assignment_available",
        False,
        "HAR-PMD classical result contains no executed fold records",
        blocking=False,
    )
    output: list[dict[str, Any]] = []
    for seed in seeds:
        folds = []
        evaluation_groups: list[set[str]] = []
        for outer_fold in range(_OUTER_FOLDS):
            evaluation = _planned_group(
                participants,
                count=_OUTER_FOLDS,
                seed=seed,
                fold=outer_fold,
                role="outer",
                plan=plan,
                checks=checks,
                context=f"posthoc seed {seed}, outer {outer_fold}",
            )
            training = participants - evaluation
            evaluation_groups.append(evaluation)
            folds.append(
                {
                    "outer_fold": outer_fold,
                    "training_participants": sorted(training),
                    "evaluation_participants": sorted(evaluation),
                    "evidence_origin": "posthoc deterministic reconstruction",
                }
            )
        union = set().union(*evaluation_groups) if evaluation_groups else set()
        checks.require(
            "posthoc_outer_reconstruction_exhaustive",
            union == participants and sum(map(len, evaluation_groups)) == len(union),
            f"seed {seed}: reconstructed folds are not exhaustive",
        )
        output.append({"seed": seed, "outer_folds": folds})
    return output


def build_split_audit(run_directory: Path) -> dict[str, Any]:
    """Build, but do not write, a participant-split audit for one run directory."""

    run_directory = run_directory.resolve(strict=True)
    result_path = run_directory / "result.json"
    result = _object(load_json_strict(result_path), name="result.json")
    checks = _Checks()
    declared_result_hash = result.get("result_payload_sha256_before_serialization")
    unhashed = dict(result)
    unhashed.pop("result_payload_sha256_before_serialization", None)
    checks.require(
        "result_payload_self_hash",
        isinstance(declared_result_hash, str)
        and canonical_json_sha256(unhashed) == declared_result_hash,
        "result payload self-hash is absent or differs",
    )
    prediction = _object(result.get("prediction_artifact"), name="prediction_artifact")
    prediction_path = _safe_artifact_path(run_directory, prediction.get("path"))
    actual_prediction_hash = sha256_file(prediction_path)
    checks.require(
        "prediction_artifact_hash",
        prediction.get("sha256") == actual_prediction_hash,
        (f"declared={prediction.get('sha256')!r}, actual={actual_prediction_hash}"),
    )
    identity = _prediction_identity(prediction_path, checks)
    participant_list, participant_records = _participant_manifest(identity)
    prediction_participants = set(participant_list)
    summary = result.get("target_dataset", result.get("dataset"))
    _check_dataset_summary(
        summary,
        participants=prediction_participants,
        window_count=len(identity.window_ids),
        checks=checks,
    )
    target_plan = _partition_plan_identity(
        summary,
        prediction_participants=prediction_participants,
        checks=checks,
        context="target" if "target_dataset" in result else "dataset",
    )
    participants = set(target_plan.roster) if target_plan is not None else prediction_participants
    source_plan = (
        _partition_plan_identity(
            result.get("source_dataset"),
            prediction_participants=None,
            checks=checks,
            context="source",
        )
        if "target_dataset" in result
        else None
    )

    warnings = [
        (
            "This audit validates any embedded pre-window plan record and its binding to "
            "executed folds; it cannot independently observe loader call order."
        ),
        (
            "Raw-sample non-overlap is outside this artifact's scope and must be supported by "
            "the dataset preprocessing/boundary audit."
        ),
    ]
    evidence_origin = "recorded_in_result"
    source_participants: list[str] | None = None
    base_nested = result.get("base_nested_result")
    if isinstance(base_nested, dict):
        nested = cast(dict[str, Any], base_nested)
        checks.require(
            "sole_nested_seeds_match",
            result.get("seeds") == nested.get("seeds"),
            "top-level and base nested seeds differ",
        )
        seed_partitions = _nested_partitions(
            nested, participants=participants, plan=target_plan, checks=checks
        )
        run_kind = "sole_harmony_nested_classical"
        warnings.append(
            "The split audit does not qualify Sole-HARmony's camera-bout reset as deployable."
        )
    elif "seed_records" in result:
        seed_partitions, source = _transfer_partitions(
            result,
            target_prediction_participants=prediction_participants,
            source_plan=source_plan,
            target_plan=target_plan,
            checks=checks,
        )
        source_participants = sorted(source)
        run_kind = "cross_dataset_transfer"
    elif isinstance(result.get("fold_records"), list) and result["fold_records"]:
        fold_records = cast(list[Any], result["fold_records"])
        if all(isinstance(item, dict) and "folds" in item for item in fold_records):
            seed_partitions = _nested_partitions(
                result, participants=participants, plan=target_plan, checks=checks
            )
            run_kind = "classical_nested"
        elif result.get("experiment_id") == "har-pmd-native-interface-stress-v1" and all(
            isinstance(item, dict) and "model" not in item and "method" not in item
            for item in fold_records
        ):
            seed_partitions = _shared_outer_partitions(
                result, participants=participants, plan=target_plan, checks=checks
            )
            run_kind = "har_pmd_shared_outer_folds"
        else:
            seed_partitions, group_key = _flat_partitions(
                result, participants=participants, plan=target_plan, checks=checks
            )
            run_kind = f"flat_{group_key}_folds"
    elif (
        result.get("experiment_id") == "har-pmd-native-interface-stress-v1"
        or _object(summary, name="dataset summary").get("dataset_id") == "har_pmd_v1"
    ):
        seed_partitions = _posthoc_har_pmd_partitions(
            result, participants=participants, plan=target_plan, checks=checks
        )
        run_kind = "har_pmd_classical_posthoc"
        evidence_origin = "posthoc_deterministic_reconstruction"
        warnings.append(
            "The executed HAR-PMD fold membership was not retained; reconstructed folds are "
            "the protocol-prescribed intent, not independent evidence of execution."
        )
    else:
        checks.require("partition_records_present", False, "unrecognized or absent split records")
        seed_partitions = []
        run_kind = "unrecognized"

    if evidence_origin == "recorded_in_result":
        checks.require(
            "recorded_split_has_prewindow_plan",
            target_plan is not None and ("target_dataset" not in result or source_plan is not None),
            "recorded split lacks a required target/source pre-window participant plan",
        )

    check_records = checks.records()
    valid = checks.blocking_passed()
    if not valid:
        status = "FAIL"
    elif evidence_origin == "posthoc_deterministic_reconstruction":
        status = "PASS_RECONSTRUCTED_POSTHOC"
    else:
        status = "PASS_RECORDED_RESULT"
    report: dict[str, Any] = {
        "schema_version": "1.0.0",
        "audit_kind": "external_har_participant_split_audit",
        "created_at": datetime.now(UTC).isoformat(),
        "status": status,
        "valid": valid,
        "run_kind": run_kind,
        "evidence_origin": evidence_origin,
        "assignment_protocol": {
            "id": _ASSIGNMENT_PROTOCOL,
            "outer_fold_count": _OUTER_FOLDS,
            "inner_fold_count": _INNER_FOLDS,
            "numpy_version": np.__version__,
        },
        "inputs": {
            "run_directory": ".",
            "result": {
                "path": result_path.name,
                "sha256": sha256_file(result_path),
            },
            "predictions": {
                "path": prediction_path.name,
                "sha256": actual_prediction_hash,
            },
            "launch_commit": (
                result.get("git_at_launch", {}).get("commit")
                if isinstance(result.get("git_at_launch"), dict)
                else None
            ),
            "source_input_manifest_sha256": (
                result.get("source_input_manifest", {}).get("manifest_sha256")
                if isinstance(result.get("source_input_manifest"), dict)
                else None
            ),
        },
        "prediction_cohort": {
            "participant_count": len(participant_list),
            "window_count": len(identity.window_ids),
            "participant_ids": participant_list,
            "participant_ids_sha256": canonical_json_sha256(participant_list),
            "participants": participant_records,
        },
        "partition_cohort": {
            "participant_count": len(participants),
            "participant_ids": sorted(participants),
            "participant_ids_sha256": canonical_json_sha256(sorted(participants)),
            "prediction_participants_are_subset": prediction_participants <= participants,
            "participants_without_retained_windows": sorted(participants - prediction_participants),
            "participant_partition_plan_sha256": (
                target_plan.plan_sha256 if target_plan is not None else None
            ),
        },
        "source_participant_partition_plan_sha256": (
            source_plan.plan_sha256 if source_plan is not None else None
        ),
        "source_participants": source_participants,
        "seed_partitions": seed_partitions,
        "seed_partitions_sha256": canonical_json_sha256(seed_partitions),
        "checks": check_records,
        "errors": [
            f"{item['name']}: {detail}"
            for item in check_records
            if item["blocking"] and not item["passed"]
            for detail in item["details"]
        ],
        "warnings": warnings,
        "scope": {
            "recorded_executed_partition_membership_verified": (
                valid and evidence_origin == "recorded_in_result"
            ),
            "deterministic_protocol_reconstruction_valid": valid,
            "pre_window_partition_plan_record_verified": bool(
                valid
                and target_plan is not None
                and ("target_dataset" not in result or source_plan)
            ),
            "pre_window_partition_execution_order_verified": False,
            "raw_sample_non_overlap_verified": False,
            "performance_metrics_inspected": False,
        },
    }
    report["record_sha256"] = canonical_json_sha256(report)
    return report


def write_split_audit(
    run_directory: Path,
    output_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    """Build and atomically publish one create-only split-audit record."""

    _assert_executed_repository_root(repository_root)
    report = build_split_audit(run_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json_new(report, output_path, allowed_root=output_path.parent)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = write_split_audit(
        args.run_directory,
        args.output,
        repository_root=args.repository_root,
    )
    print(
        json.dumps(
            {
                "output": args.output.as_posix(),
                "status": report["status"],
                "valid": report["valid"],
                "record_sha256": report["record_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
