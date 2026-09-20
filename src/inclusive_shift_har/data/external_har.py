"""Boundary-safe streaming adapters for the external HAR research portfolio.

The adapters intentionally keep raw source artifacts out of the repository.  Public
artifacts are streamed from immutable provider locators and every consumed object receives
a cryptographic receipt.  Controlled-access datasets are not handled here: their absence
must remain an explicit blocked gate rather than being replaced by synthetic data.
"""

from __future__ import annotations

import hashlib
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from fractions import Fraction
from pathlib import Path, PurePosixPath
from typing import Any, cast
from urllib.parse import urlencode
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import requests
from numpy.typing import NDArray
from remotezip import RemoteZip  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

from inclusive_shift_har.data.participant_partitions import (
    ParticipantPartitionPlan,
    build_participant_partition_plan,
)
from inclusive_shift_har.data.provider_copy import verified_har_pmd_archive
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

STANDARD_GRAVITY_M_S2 = 9.80665
CORE_CLASS_NAMES = ("mobility", "sitting", "standing")
CORE_CLASS_INDEX = {name: index for index, name in enumerate(CORE_CLASS_NAMES)}
PHYSICAL_GRID_PROTOCOL = "external-har-session-grid-v3"
BOUNDARY_PROVENANCE_PROTOCOL = "external-har-boundary-provenance-v1"
OBSERVABLE_SCORING_ELIGIBILITY_POLICY = (
    "boolean mask true exactly at the strictly increasing scored-window indices"
)
IMU_HAR_IL_COHORT_INVENTORY_PROTOCOL = "imu-har-il-fixed-inventory-v1"

SOURCE_RECEIPT_METADATA: dict[str, dict[str, str]] = {
    "fog_star_v3": {
        "record_url": "https://zenodo.org/records/17838806",
        "dataset_version": "Zenodo record 17838806 v3",
        "dataset_license": "CC-BY-4.0",
        "permissible_redistribution": "raw data excluded from Git; attribution required",
        "evidence_role": "development",
    },
    "imu_har_il_v1": {
        "record_url": "https://data.csiro.au/collection/csiro:74700",
        "dataset_version": ("CSIRO collection 74700 dataVersion 1, DOI 10.25919/d7xf-n080"),
        "dataset_license": (
            "CC-BY-NC-4.0; provider re-verification remains required before redistribution"
        ),
        "permissible_redistribution": (
            "noncommercial restrictions apply to data and derived artifacts; Apache "
            "repository licensing does not relicense them"
        ),
        "evidence_role": "development",
    },
    "har_pmd_v1": {
        "record_url": "https://zenodo.org/records/7939223",
        "dataset_version": "provider Zenodo v2; repository identifier retained for compatibility",
        "dataset_license": "CC-BY-4.0",
        "permissible_redistribution": "raw data excluded from Git; attribution required",
        "evidence_role": "stress_test",
    },
    "sole_harmony_v1": {
        "record_url": "https://zenodo.org/records/19242395",
        "dataset_version": "Zenodo record 19242395",
        "dataset_license": "CC-BY-4.0",
        "permissible_redistribution": "raw data excluded from Git; attribution required",
        "evidence_role": "oracle_diagnostic",
    },
    "aicos_har_v1": {
        "record_url": "https://zenodo.org/records/19452049",
        "dataset_version": "Zenodo record 19452049 v1",
        "dataset_license": "CC-BY-4.0",
        "permissible_redistribution": "raw data excluded from Git; attribution required",
        "evidence_role": "external_zero_shot_evaluation",
    },
}


@dataclass(frozen=True, slots=True)
class SourceReceipt:
    """Verifiable record for one streamed source object or archive member."""

    dataset_id: str
    locator: str
    member: str | None
    declared_size_bytes: int | None
    received_size_bytes: int
    computed_sha256: str
    declared_digest_algorithm: str | None = None
    declared_digest: str | None = None
    computed_declared_digest: str | None = None
    declared_digest_verified: bool | None = None
    archive_crc32: str | None = None
    raw_local_mirror: bool = False
    accessed_at_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    record_url: str | None = None
    dataset_version: str | None = None
    dataset_license: str | None = None
    permissible_redistribution: str | None = None
    evidence_role: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable receipt."""

        value = asdict(self)
        metadata = SOURCE_RECEIPT_METADATA.get(self.dataset_id, {})
        for name, expected in metadata.items():
            if value.get(name) is None:
                value[name] = expected
        return value

    def validate(self) -> None:
        """Reject incomplete hashes, source identity, or frozen dataset metadata."""

        value = self.to_dict()
        try:
            accessed = datetime.fromisoformat(str(value["accessed_at_utc"]))
        except ValueError as exc:
            raise ValueError("source receipt access timestamp is invalid") from exc
        expected = SOURCE_RECEIPT_METADATA.get(self.dataset_id)
        if (
            not self.dataset_id
            or not self.locator
            or (expected is not None and not self.locator.startswith("https://"))
            or self.received_size_bytes <= 0
            or (
                self.declared_size_bytes is not None
                and self.declared_size_bytes != self.received_size_bytes
            )
            or len(self.computed_sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.computed_sha256)
            or accessed.tzinfo is None
        ):
            raise ValueError("source receipt identity, size, hash, or access time is invalid")
        if expected is not None and any(value.get(name) != item for name, item in expected.items()):
            raise ValueError("source receipt differs from frozen dataset metadata")


@dataclass(frozen=True, slots=True)
class ObservableWindowPool:
    """All signal-only candidate windows; deliberately contains no annotations."""

    signals: Float32Array
    gravity: Float32Array
    participant_ids: StringArray
    session_ids: StringArray
    trial_ids: StringArray
    window_ids: StringArray

    def validate(self) -> None:
        count = self.window_ids.size
        if (
            self.signals.ndim != 3
            or self.signals.shape[0] != count
            or self.signals.shape[2] != 6
            or self.gravity.shape != (*self.signals.shape[:2], 3)
            or not np.isfinite(self.signals).all()
            or not np.isfinite(self.gravity).all()
            or any(
                values.shape != (count,) or np.any(np.char.str_len(values) == 0)
                for values in (
                    self.participant_ids,
                    self.session_ids,
                    self.trial_ids,
                    self.window_ids,
                )
            )
            or len(set(self.window_ids.tolist())) != count
        ):
            raise ValueError(
                "observable candidates must be finite, aligned and uniquely identified"
            )

    def audit(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_id": "external-har-observable-context-v1",
            "window_count": int(self.window_ids.size),
            "participant_window_counts": {
                str(participant): int(np.sum(self.participant_ids == participant))
                for participant in np.unique(self.participant_ids)
            },
            "annotation_fields_present": False,
            "construction": "all fixed physical-segment candidates before annotation admission",
            "arrays": {
                name: _array_sha256(getattr(self, name))
                for name in (
                    "signals",
                    "gravity",
                    "participant_ids",
                    "session_ids",
                    "trial_ids",
                    "window_ids",
                )
            }
            | {
                "nine_channel_signals": _array_sha256(
                    np.asarray(
                        np.concatenate((self.signals, self.gravity), axis=2), dtype=np.float32
                    )
                )
            },
        }


@dataclass(frozen=True, slots=True)
class ExternalHARWindows:
    """Aligned core-HAR windows plus non-feature grouping/provenance metadata."""

    dataset_id: str
    channel_lane: str
    sampling_rate_hz: float
    signals: Float32Array
    gravity: Float32Array
    labels: IntArray
    participant_ids: StringArray
    session_ids: StringArray
    trial_ids: StringArray
    window_ids: StringArray
    receipts: tuple[SourceReceipt, ...]
    exclusions: tuple[dict[str, Any], ...] = ()
    source_issues: tuple[dict[str, Any], ...] = ()
    class_names: tuple[str, ...] = CORE_CLASS_NAMES
    gravity_source: str = "unspecified"
    gravity_cutoff_hz: float | None = None
    preprocessing_audit: tuple[dict[str, Any], ...] = ()
    cohort_audit: dict[str, Any] | None = None
    observable_candidates: ObservableWindowPool | None = None
    source_storage_audit: dict[str, Any] | None = None
    boundary_provenance: dict[str, Any] | None = None
    participant_partition_plan: ParticipantPartitionPlan | None = None

    def validate(self, *, require_all_classes: bool = True) -> None:
        """Fail closed on alignment, units, identifiers, or lane ambiguity."""

        count = self.labels.size
        if self.dataset_id == "" or self.channel_lane not in {
            "derived-gravity-9ch",
            "native-gravity-9ch",
        }:
            raise ValueError("external HAR dataset/lane identity is invalid")
        if not np.isfinite(self.sampling_rate_hz) or self.sampling_rate_hz <= 0.0:
            raise ValueError("external HAR sampling rate must be finite and positive")
        if self.signals.ndim != 3 or self.signals.shape[0] != count or self.signals.shape[2] != 6:
            raise ValueError("external HAR primary signals must have shape [window,time,6]")
        if self.gravity.shape != (*self.signals.shape[:2], 3):
            raise ValueError("external HAR gravity must align as [window,time,3]")
        if self.signals.shape[1] < 32 or count == 0:
            raise ValueError("external HAR windows must be non-empty and at least 32 samples")
        metadata = (
            self.participant_ids,
            self.session_ids,
            self.trial_ids,
            self.window_ids,
        )
        if any(item.shape != (count,) for item in metadata):
            raise ValueError("external HAR metadata arrays are not aligned")
        if not np.isfinite(self.signals).all() or not np.isfinite(self.gravity).all():
            raise ValueError("external HAR features must be finite")
        if self.gravity_source == "":
            raise ValueError("external HAR gravity source must be non-empty")
        if self.gravity_cutoff_hz is not None and (
            not np.isfinite(self.gravity_cutoff_hz)
            or not 0.0 < self.gravity_cutoff_hz < self.sampling_rate_hz / 2.0
        ):
            raise ValueError("external HAR gravity cutoff must be valid when declared")
        if len(self.class_names) < 2 or len(set(self.class_names)) != len(self.class_names):
            raise ValueError("external HAR class names are invalid")
        observed = set(self.labels.tolist())
        expected = set(range(len(self.class_names)))
        if not observed.issubset(expected) or (require_all_classes and observed != expected):
            raise ValueError("external HAR labels do not match the declared ontology")
        if len(set(self.window_ids.tolist())) != count:
            raise ValueError("external HAR window identifiers must be unique")
        if any(np.any(np.char.str_len(item) == 0) for item in metadata):
            raise ValueError("external HAR identifiers must be non-empty")
        if not self.receipts:
            raise ValueError("external HAR data require at least one source receipt")
        for receipt in self.receipts:
            receipt.validate()
        if self.boundary_provenance is not None:
            required_boundary_fields = {
                "protocol_id",
                "source_boundary_unit",
                "repository_signal_grid_annotation_independent",
                "provider_upstream_annotation_conditioned",
                "zero_lookahead_streaming_valid",
                "evidence_scope",
            }
            if (
                not required_boundary_fields.issubset(self.boundary_provenance)
                or self.boundary_provenance["protocol_id"] != BOUNDARY_PROVENANCE_PROTOCOL
                or not isinstance(self.boundary_provenance["source_boundary_unit"], str)
                or not self.boundary_provenance["source_boundary_unit"]
                or not isinstance(self.boundary_provenance["evidence_scope"], str)
                or not self.boundary_provenance["evidence_scope"]
                or any(
                    type(self.boundary_provenance[name]) is not bool
                    for name in (
                        "repository_signal_grid_annotation_independent",
                        "provider_upstream_annotation_conditioned",
                        "zero_lookahead_streaming_valid",
                    )
                )
            ):
                raise ValueError("external HAR boundary provenance is invalid")
        if self.participant_partition_plan is not None:
            self.participant_partition_plan.validate()
            roster = set(self.participant_partition_plan.participant_roster)
            observed_participants = set(self.participant_ids.tolist())
            if (
                self.participant_partition_plan.dataset_id != self.dataset_id
                or not observed_participants.issubset(roster)
            ):
                raise ValueError("external HAR participant partition plan does not match data")
        if self.observable_candidates is not None:
            pool = self.observable_candidates
            pool.validate()
            if self.participant_partition_plan is not None and not set(
                pool.participant_ids.tolist()
            ).issubset(set(self.participant_partition_plan.participant_roster)):
                raise ValueError(
                    "observable candidates contain a participant outside the pre-window plan"
                )
            positions = {name: index for index, name in enumerate(pool.window_ids.tolist())}
            if any(name not in positions for name in self.window_ids.tolist()):
                raise ValueError("scored windows are not a subset of the observable candidate pool")
            indices = np.asarray([positions[name] for name in self.window_ids.tolist()])
            for name in ("signals", "gravity", "participant_ids", "session_ids", "trial_ids"):
                if not np.array_equal(getattr(self, name), getattr(pool, name)[indices]):
                    raise ValueError(f"scored and observable candidate values differ: {name}")

    @property
    def nine_channel_signals(self) -> Float32Array:
        """Return the explicit six-primary plus three-gravity representation."""

        return np.asarray(np.concatenate((self.signals, self.gravity), axis=2), dtype=np.float32)

    def summary(self) -> dict[str, Any]:
        """Return a compact non-performance audit summary."""

        counts = np.bincount(self.labels, minlength=len(self.class_names))
        participants = np.unique(self.participant_ids)
        participant_support = {
            name: int(
                sum(
                    np.any(self.labels[self.participant_ids == participant] == index)
                    for participant in participants
                )
            )
            for index, name in enumerate(self.class_names)
        }
        present_class_counts = [
            len(set(self.labels[self.participant_ids == participant].tolist()))
            for participant in participants
        ]
        summary = {
            "dataset_id": self.dataset_id,
            "channel_lane": self.channel_lane,
            "sampling_rate_hz": self.sampling_rate_hz,
            "window_samples": int(self.signals.shape[1]),
            "window_count": int(self.labels.size),
            "participant_count": int(participants.size),
            "session_count": int(np.unique(self.session_ids).size),
            "trial_count": int(np.unique(self.trial_ids).size),
            "class_window_counts": {
                name: int(counts[index]) for index, name in enumerate(self.class_names)
            },
            "participant_class_support": participant_support,
            "participants_with_all_classes": int(
                sum(value == len(self.class_names) for value in present_class_counts)
            ),
            "perfect_prediction_mean_participant_macro_f1_ceiling": float(
                np.mean(present_class_counts) / len(self.class_names)
            ),
            "receipt_count": len(self.receipts),
            "exclusion_count": len(self.exclusions),
            "exclusions": list(self.exclusions),
            "source_issue_count": len(self.source_issues),
            "source_issues": list(self.source_issues),
            "gravity_preprocessing": {
                "source": self.gravity_source,
                "causal_lowpass_cutoff_hz": self.gravity_cutoff_hz,
            },
            "retained_array_hashes": {
                "signals": _array_sha256(self.signals),
                "gravity": _array_sha256(self.gravity),
                "nine_channel_signals": _array_sha256(self.nine_channel_signals),
                "labels": _array_sha256(self.labels),
                "participant_ids": _array_sha256(self.participant_ids),
                "session_ids": _array_sha256(self.session_ids),
                "trial_ids": _array_sha256(self.trial_ids),
                "window_ids": _array_sha256(self.window_ids),
            },
            "preprocessing_audit": list(self.preprocessing_audit),
            "raw_local_mirror": any(receipt.raw_local_mirror for receipt in self.receipts),
        }
        if self.cohort_audit is not None:
            summary["cohort_audit"] = self.cohort_audit
        if self.observable_candidates is not None:
            summary["observable_candidate_pool"] = self.observable_candidates.audit()
        if self.source_storage_audit is not None:
            summary["source_storage_audit"] = self.source_storage_audit
        if self.boundary_provenance is not None:
            summary["boundary_provenance"] = self.boundary_provenance
        if self.participant_partition_plan is not None:
            summary["participant_partition_plan"] = self.participant_partition_plan.audit()
            planned = set(self.participant_partition_plan.participant_roster)
            observed = set(self.participant_ids.tolist())
            summary["participant_partition_observation"] = {
                "planned_participant_count": len(planned),
                "observed_window_participant_count": len(observed),
                "participants_without_retained_windows": sorted(planned - observed),
            }
        return summary


def observable_modelling_pool(
    data: ExternalHARWindows, *, include_supervised_labels: bool
) -> tuple[ExternalHARWindows, IntArray, NDArray[np.bool_]]:
    """Separate inference candidates, scoring positions and supervised eligibility.

    Placeholder labels in the modelling container are never supervision: callers
    must intersect every training mask with the returned eligibility mask. Target
    mode installs no real labels at all. Scripted-trial loaders without an explicit
    continuous candidate pool retain their declared, limited input population.
    """
    pool = data.observable_candidates
    if pool is None:
        indices = np.arange(data.labels.size, dtype=np.int64)
        return (
            data if include_supervised_labels else replace(data, labels=np.zeros_like(data.labels)),
            indices,
            np.ones(data.labels.size, dtype=np.bool_),
        )
    data.validate(require_all_classes=False)
    # Keep the complete pre-annotation population.  In particular, a participant with
    # zero homogeneous/labelled windows must remain in the candidate roster and frozen
    # partition plan; annotations may change eligibility, never the inference population.
    identifiers = pool.window_ids
    positions = {name: index for index, name in enumerate(identifiers.tolist())}
    indices = np.asarray([positions[name] for name in data.window_ids.tolist()], dtype=np.int64)
    eligibility = np.zeros(identifiers.size, dtype=np.bool_)
    eligibility[indices] = True
    labels = np.zeros(identifiers.size, dtype=np.int64)
    if include_supervised_labels:
        labels[indices] = data.labels
    modelling = replace(
        data,
        signals=pool.signals,
        gravity=pool.gravity,
        labels=labels,
        participant_ids=pool.participant_ids,
        session_ids=pool.session_ids,
        trial_ids=pool.trial_ids,
        window_ids=identifiers,
        observable_candidates=None,
    )
    return modelling, indices, eligibility


@dataclass(frozen=True, slots=True)
class _IMUSourceFile:
    participant: str
    repetition: str
    activity: str
    file_id: int
    filename: str
    file_size: int
    locator: str


def _imu_source_inventory_sha256(sources: tuple[_IMUSourceFile, ...]) -> str:
    """Hash the exact provider identities selected before payload download."""

    records = [asdict(source) for source in sources]
    records.sort(
        key=lambda item: (
            str(item["participant"]),
            str(item["repetition"]),
            str(item["activity"]),
        )
    )
    return canonical_json_sha256(records)


def _imu_source_payload_inventory_sha256(receipts: list[SourceReceipt]) -> str:
    """Hash stable identities and bytes for every downloaded IMU source object."""

    records = [
        {
            "locator": receipt.locator,
            "member": receipt.member,
            "declared_size_bytes": receipt.declared_size_bytes,
            "received_size_bytes": receipt.received_size_bytes,
            "computed_sha256": receipt.computed_sha256,
        }
        for receipt in receipts
    ]
    records.sort(key=lambda item: (str(item["member"]), str(item["locator"])))
    return canonical_json_sha256(records)


def _imu_missing_trial_inventory_sha256(exclusions: tuple[dict[str, Any], ...]) -> str:
    """Hash the provider paths missing from the requested IMU trial matrix."""

    folders = sorted(
        str(folder)
        for exclusion in exclusions
        for folder in cast(list[Any], exclusion.get("missing_folders", []))
    )
    return canonical_json_sha256(folders)


@dataclass(slots=True)
class _WindowAccumulator:
    signals: list[Float32Array]
    gravity: list[Float32Array]
    labels: list[int]
    participants: list[str]
    sessions: list[str]
    trials: list[str]
    windows: list[str]
    preprocessing_audit: list[dict[str, Any]] = field(default_factory=list)
    exclusions: list[dict[str, Any]] = field(default_factory=list)
    candidate_signals: list[Float32Array] = field(default_factory=list)
    candidate_gravity: list[Float32Array] = field(default_factory=list)
    candidate_participants: list[str] = field(default_factory=list)
    candidate_sessions: list[str] = field(default_factory=list)
    candidate_trials: list[str] = field(default_factory=list)
    candidate_windows: list[str] = field(default_factory=list)

    @classmethod
    def empty(cls) -> _WindowAccumulator:
        return cls([], [], [], [], [], [], [])

    def add_physical_candidates(
        self,
        segment: UniformPhysicalSegment,
        *,
        participant: str,
        session: str,
        trial: str,
        run_index: int,
    ) -> None:
        """Retain inference candidates through a label-free interface."""
        for window_index, start in enumerate(segment.candidate_starts):
            stop = int(start) + segment.window_samples
            self.candidate_signals.append(np.asarray(segment.signals[start:stop], dtype=np.float32))
            self.candidate_gravity.append(np.asarray(segment.gravity[start:stop], dtype=np.float32))
            self.candidate_participants.append(participant)
            self.candidate_sessions.append(session)
            self.candidate_trials.append(trial)
            self.candidate_windows.append(
                f"{participant}/{session}/{trial}/run-{run_index:04d}-finite-000/"
                f"window-{window_index:06d}"
            )

    def observable_pool(self) -> ObservableWindowPool | None:
        if not self.candidate_signals:
            return None
        return ObservableWindowPool(
            signals=np.stack(self.candidate_signals),
            gravity=np.stack(self.candidate_gravity),
            participant_ids=np.asarray(self.candidate_participants, dtype=np.str_),
            session_ids=np.asarray(self.candidate_sessions, dtype=np.str_),
            trial_ids=np.asarray(self.candidate_trials, dtype=np.str_),
            window_ids=np.asarray(self.candidate_windows, dtype=np.str_),
        )

    def add_annotated_segment(
        self,
        segment: UniformPhysicalSegment,
        *,
        source_labels: FloatArray,
        label_map: dict[int, int],
        participant: str,
        session: str,
        trial: str,
        run_index: int,
    ) -> None:
        """Assign labels and apply homogeneous admission to an already fixed grid.

        Source samples in the complete half-open window time interval are checked
        as well as projected annotations, so downsampling cannot hide a short
        transition or a missing annotation. No rejected candidate shifts the grid.
        """

        self.add_physical_candidates(
            segment, participant=participant, session=session, trial=trial, run_index=run_index
        )
        if source_labels.shape != segment.source_timestamps.shape:
            raise ValueError("annotations must align with the physical source segment")
        source_indices = (
            np.searchsorted(segment.source_timestamps, segment.timestamps, side="right") - 1
        )
        projected = source_labels[np.clip(source_indices, 0, source_labels.size - 1)]
        admitted: list[int] = []
        rejected: dict[str, list[int]] = {
            "missing_or_unsupported_annotation": [],
            "mixed_annotation": [],
        }
        for window_index, start in enumerate(segment.candidate_starts):
            stop = int(start) + segment.window_samples
            start_time = segment.timestamps[start]
            stop_time = start_time + segment.window_samples / segment.target_rate_hz
            left = max(
                0, int(np.searchsorted(segment.source_timestamps, start_time, side="right")) - 1
            )
            right = int(np.searchsorted(segment.source_timestamps, stop_time, side="left"))
            annotations = np.concatenate((projected[start:stop], source_labels[left:right]))
            if not np.isfinite(annotations).all() or any(
                value not in label_map for value in np.unique(annotations)
            ):
                rejected["missing_or_unsupported_annotation"].append(window_index)
                continue
            if np.any(annotations != annotations[0]):
                rejected["mixed_annotation"].append(window_index)
                continue
            identifier = (
                f"{participant}/{session}/{trial}/run-{run_index:04d}-finite-000/"
                f"window-{window_index:06d}"
            )
            self.signals.append(np.asarray(segment.signals[start:stop], dtype=np.float32))
            self.gravity.append(np.asarray(segment.gravity[start:stop], dtype=np.float32))
            self.labels.append(label_map[int(annotations[0])])
            self.participants.append(participant)
            self.sessions.append(session)
            self.trials.append(trial)
            self.windows.append(identifier)
            admitted.append(window_index)
        audit = {
            **segment.audit(),
            "participant_id": participant,
            "session_id": session,
            "trial_id": trial,
            "run_index": run_index,
            "segment_boundary_annotation_conditioned": False,
            "admitted_candidate_indices": admitted,
            "excluded_candidate_indices": rejected,
        }
        self.preprocessing_audit.append(audit)
        for reason, indices in rejected.items():
            if indices:
                self.exclusions.append(
                    {
                        "trial_id": trial,
                        "run_index": run_index,
                        "reason": reason,
                        "candidate_indices": indices,
                        "window_count": len(indices),
                    }
                )

    def add_uniform_trial(
        self,
        *,
        total_acceleration: FloatArray,
        gyroscope: FloatArray,
        gravity: FloatArray | None,
        source_rate_hz: float,
        target_rate_hz: float,
        gravity_cutoff_hz: float,
        label: int,
        participant: str,
        session: str,
        trial: str,
        run_index: int,
        window_samples: int,
        segment_boundary_annotation_conditioned: bool,
    ) -> None:
        """Resample and window one already boundary-isolated provider trial.

        Provider trial boundaries can still be annotation-conditioned and are disclosed
        separately. Within each finite run, however, all signal and gravity channels are
        transformed together in exactly one repository resampling pass.
        """

        total = np.asarray(total_acceleration, dtype=np.float64)
        gyro = np.asarray(gyroscope, dtype=np.float64)
        if total.ndim != 2 or total.shape[1] != 3 or gyro.shape != total.shape:
            raise ValueError("trial accelerometer/gyroscope arrays must align as [time,3]")
        finite = np.asarray(
            np.isfinite(total).all(axis=1) & np.isfinite(gyro).all(axis=1), dtype=np.bool_
        )
        if gravity is not None:
            gravity_values = np.asarray(gravity, dtype=np.float64)
            if gravity_values.shape != total.shape:
                raise ValueError("native gravity must align with the primary trial")
            finite &= np.isfinite(gravity_values).all(axis=1)
        else:
            gravity_values = None
        nonfinite_rows = int((~finite).sum())
        if nonfinite_rows:
            self.exclusions.append(
                {
                    "trial_id": trial,
                    "run_index": run_index,
                    "reason": "nonfinite_sensor",
                    "source_rows": nonfinite_rows,
                }
            )
        for finite_index, (start, stop) in enumerate(_true_runs(finite)):
            if stop - start < 3:
                self.exclusions.append(
                    {
                        "trial_id": trial,
                        "run_index": run_index,
                        "finite_run_index": finite_index,
                        "reason": "short_finite_signal_run",
                        "source_rows": stop - start,
                    }
                )
                continue
            source_timestamps = np.arange(stop - start, dtype=np.float64) / source_rate_hz
            segment = resample_physical_segment(
                timestamps=source_timestamps,
                acceleration=total[start:stop],
                gyroscope=gyro[start:stop],
                gravity=None if gravity_values is None else gravity_values[start:stop],
                source_rate_hz=source_rate_hz,
                target_rate_hz=target_rate_hz,
                gravity_cutoff_hz=gravity_cutoff_hz,
                window_samples=window_samples,
            )
            for window_index, window_start in enumerate(segment.candidate_starts.tolist()):
                window_stop = window_start + window_samples
                identifier = (
                    f"{participant}/{session}/{trial}/run-{run_index:04d}-"
                    f"finite-{finite_index:03d}/window-{window_index:06d}"
                )
                self.signals.append(
                    np.asarray(segment.signals[window_start:window_stop], dtype=np.float32)
                )
                self.gravity.append(
                    np.asarray(segment.gravity[window_start:window_stop], dtype=np.float32)
                )
                self.labels.append(label)
                self.participants.append(participant)
                self.sessions.append(session)
                self.trials.append(trial)
                self.windows.append(identifier)
            segment_audit = segment.audit()
            self.preprocessing_audit.append(
                {
                    **segment_audit,
                    "participant_id": participant,
                    "session_id": session,
                    "trial_id": trial,
                    "run_index": run_index,
                    "finite_run_index": finite_index,
                    "segment_boundary_annotation_conditioned": (
                        segment_boundary_annotation_conditioned
                    ),
                    "admitted_candidate_indices": list(range(segment.candidate_starts.size)),
                    "excluded_candidate_indices": {},
                }
            )
            if segment_audit["dropped_tail_samples"]:
                self.exclusions.append(
                    {
                        "trial_id": trial,
                        "run_index": run_index,
                        "finite_run_index": finite_index,
                        "reason": "incomplete_resampled_tail",
                        "target_samples": segment_audit["dropped_tail_samples"],
                    }
                )

    def finish(
        self,
        *,
        dataset_id: str,
        channel_lane: str,
        sampling_rate_hz: float,
        receipts: list[SourceReceipt],
        gravity_source: str = "unspecified",
        gravity_cutoff_hz: float | None = None,
        require_all_classes: bool = True,
        boundary_provenance: dict[str, Any] | None = None,
        participant_partition_plan: ParticipantPartitionPlan | None = None,
    ) -> ExternalHARWindows:
        if not self.signals:
            raise ValueError(f"{dataset_id} produced no complete core windows")
        result = ExternalHARWindows(
            dataset_id=dataset_id,
            channel_lane=channel_lane,
            sampling_rate_hz=sampling_rate_hz,
            signals=np.stack(self.signals).astype(np.float32, copy=False),
            gravity=np.stack(self.gravity).astype(np.float32, copy=False),
            labels=np.asarray(self.labels, dtype=np.int64),
            participant_ids=np.asarray(self.participants, dtype=np.str_),
            session_ids=np.asarray(self.sessions, dtype=np.str_),
            trial_ids=np.asarray(self.trials, dtype=np.str_),
            window_ids=np.asarray(self.windows, dtype=np.str_),
            receipts=tuple(receipts),
            exclusions=tuple(self.exclusions),
            source_issues=(),
            gravity_source=gravity_source,
            gravity_cutoff_hz=gravity_cutoff_hz,
            preprocessing_audit=tuple(self.preprocessing_audit),
            observable_candidates=self.observable_pool(),
            boundary_provenance=boundary_provenance,
            participant_partition_plan=participant_partition_plan,
        )
        result.validate(require_all_classes=require_all_classes)
        return result


def causal_gravity_lowpass(
    total_acceleration: NDArray[np.floating[Any]],
    *,
    sampling_rate_hz: float,
    cutoff_hz: float,
) -> FloatArray:
    """Estimate gravity with a causal first-order low-pass inside one trial only."""

    values = np.asarray(total_acceleration, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] != 3:
        raise ValueError("gravity derivation requires non-empty [time,3] acceleration")
    if not np.isfinite(values).all():
        raise ValueError("gravity derivation input must be finite")
    if (
        not np.isfinite(sampling_rate_hz)
        or not np.isfinite(cutoff_hz)
        or sampling_rate_hz <= 0.0
        or not 0.0 < cutoff_hz < sampling_rate_hz / 2.0
    ):
        raise ValueError("gravity filter rate/cutoff is invalid")
    alpha = 1.0 - np.exp(-2.0 * np.pi * cutoff_hz / sampling_rate_hz)
    gravity = np.empty_like(values)
    gravity[0] = values[0]
    for index in range(1, values.shape[0]):
        gravity[index] = gravity[index - 1] + alpha * (values[index] - gravity[index - 1])
    return np.asarray(gravity, dtype=np.float64)


def _resampled_sample_count(
    sample_count: int, *, source_rate_hz: float, target_rate_hz: float
) -> int:
    """Return SciPy polyphase output length without transforming the signal."""

    if sample_count < 2:
        raise ValueError("uniform resampling requires at least two samples")
    if (
        not np.isfinite([source_rate_hz, target_rate_hz]).all()
        or min(source_rate_hz, target_rate_hz) <= 0.0
    ):
        raise ValueError("resampling rates must be positive")
    ratio = Fraction(str(target_rate_hz / source_rate_hz)).limit_denominator(10_000)
    return (sample_count * ratio.numerator + ratio.denominator - 1) // ratio.denominator


def resample_uniform(
    values: NDArray[np.floating[Any]], *, source_rate_hz: float, target_rate_hz: float
) -> FloatArray:
    """Polyphase-resample one boundary-isolated uniformly sampled sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("uniform resampling requires a finite [time,channel] sequence")
    if (
        not np.isfinite([source_rate_hz, target_rate_hz]).all()
        or min(source_rate_hz, target_rate_hz) <= 0.0
    ):
        raise ValueError("resampling rates must be positive")
    expected_count = _resampled_sample_count(
        array.shape[0], source_rate_hz=source_rate_hz, target_rate_hz=target_rate_hz
    )
    ratio = Fraction(str(target_rate_hz / source_rate_hz)).limit_denominator(10_000)
    result = resample_poly(array, ratio.numerator, ratio.denominator, axis=0)
    if result.shape[0] != expected_count:
        raise RuntimeError("polyphase resampling length departed from the frozen contract")
    return np.asarray(result, dtype=np.float64)


def _array_sha256(values: NDArray[Any]) -> str:
    """Hash shape, dtype and C-order bytes; retain no raw signal values in JSON."""

    digest = hashlib.sha256()
    digest.update(str((values.shape, values.dtype.str)).encode("ascii"))
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class UniformPhysicalSegment:
    """Pre-annotation signal grid. This interface deliberately has no label field."""

    signals: FloatArray
    gravity: FloatArray
    source_timestamps: FloatArray
    timestamps: FloatArray
    candidate_starts: IntArray
    source_rate_hz: float
    target_rate_hz: float
    window_samples: int

    def audit(self) -> dict[str, Any]:
        return {
            "protocol_id": PHYSICAL_GRID_PROTOCOL,
            "source_samples": int(self.source_timestamps.size),
            "resampled_samples": int(self.timestamps.size),
            "source_rate_hz": self.source_rate_hz,
            "target_rate_hz": self.target_rate_hz,
            "window_samples": self.window_samples,
            "candidate_start_samples": self.candidate_starts.tolist(),
            "candidate_window_count": int(self.candidate_starts.size),
            "dropped_tail_samples": int(self.timestamps.size % self.window_samples),
            "start_timestamp": float(self.timestamps[0]),
            "last_timestamp": float(self.timestamps[-1]),
            "source_timestamps_sha256": _array_sha256(self.source_timestamps),
            "timestamps_sha256": _array_sha256(self.timestamps),
            "signals_sha256": _array_sha256(self.signals),
            "gravity_sha256": _array_sha256(self.gravity),
            "candidate_grid_sha256": _array_sha256(self.candidate_starts),
            "resampling_passes": 1,
            "within_declared_segment_transform_annotation_dependency": False,
        }


def resample_physical_segment(
    *,
    timestamps: FloatArray,
    acceleration: FloatArray,
    gyroscope: FloatArray,
    gravity: FloatArray | None,
    source_rate_hz: float,
    target_rate_hz: float,
    gravity_cutoff_hz: float,
    window_samples: int,
) -> UniformPhysicalSegment:
    """Filter one finite observable segment and resample its nine channels once.

    Uniform sampling is provider-declared. Polyphase resampling is an offline,
    symmetric FIR operation, not a zero-lookahead streaming implementation. Its
    boundaries and padding depend only on the physical segment, never annotations.
    When gravity is supplied, acceleration already denotes linear acceleration.
    """

    if (
        timestamps.ndim != 1
        or timestamps.size < 3
        or acceleration.shape != (timestamps.size, 3)
        or gyroscope.shape != acceleration.shape
        or not np.isfinite(timestamps).all()
        or np.any(np.diff(timestamps) <= 0.0)
        or not np.isfinite(acceleration).all()
        or not np.isfinite(gyroscope).all()
        or window_samples < 1
    ):
        raise ValueError("physical resampling requires aligned finite monotonic sensor data")
    derived = (
        causal_gravity_lowpass(
            acceleration, sampling_rate_hz=source_rate_hz, cutoff_hz=gravity_cutoff_hz
        )
        if gravity is None
        else np.asarray(gravity, dtype=np.float64)
    )
    if derived.shape != acceleration.shape or not np.isfinite(derived).all():
        raise ValueError("physical gravity must align with the sensor segment")
    linear = acceleration - derived if gravity is None else acceleration
    resampled = resample_uniform(
        np.column_stack((linear, gyroscope, derived)),
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
    )
    grid = timestamps[0] + np.arange(resampled.shape[0], dtype=np.float64) / target_rate_hz
    # A source sample represents one nominal sample interval. Never extrapolate
    # annotations beyond the final interval, even if rounded rates/jitter disagree.
    in_recording = grid < timestamps[-1] + 1.0 / source_rate_hz
    grid, resampled = grid[in_recording], resampled[in_recording]
    count = grid.size // window_samples
    return UniformPhysicalSegment(
        signals=np.asarray(resampled[:, :6], dtype=np.float64),
        gravity=np.asarray(resampled[:, 6:], dtype=np.float64),
        source_timestamps=timestamps.copy(),
        timestamps=grid,
        candidate_starts=np.arange(count, dtype=np.int64) * window_samples,
        source_rate_hz=source_rate_hz,
        target_rate_hz=target_rate_hz,
        window_samples=window_samples,
    )


def participant_fold_assignment(
    participant_ids: list[str] | tuple[str, ...] | StringArray,
    *,
    fold_count: int,
    seed: int,
) -> dict[str, int]:
    """Assign participants before windowing using a deterministic seeded permutation."""

    participants = sorted({str(item) for item in participant_ids})
    if fold_count < 2 or len(participants) < fold_count or seed < 0:
        raise ValueError("participant fold assignment requires enough groups and a valid seed")
    order = np.random.default_rng(seed).permutation(len(participants))
    return {participants[int(index)]: position % fold_count for position, index in enumerate(order)}


def _prewindow_partition_plan(
    dataset_id: str,
    participant_ids: list[str] | tuple[str, ...] | StringArray,
    *,
    roster_basis: str,
) -> ParticipantPartitionPlan | None:
    """Freeze the publication fold plan when the requested diagnostic has enough people.

    Small synthetic/unit diagnostics remain loadable, but cannot be passed to hardened
    publication runners because they carry no plan.  Every full external protocol has at
    least five provider-roster participants and therefore receives a plan here.
    """

    participants = tuple(sorted({str(item) for item in participant_ids}))
    if len(participants) < 5:
        return None
    return build_participant_partition_plan(
        dataset_id,
        participants,
        roster_basis=roster_basis,
    )


def concatenate_external_windows(
    datasets: tuple[ExternalHARWindows, ...],
    *,
    dataset_id: str,
    require_all_classes_per_dataset: bool = True,
) -> ExternalHARWindows:
    """Concatenate only measurement-compatible evidence-lane datasets."""

    if not datasets:
        raise ValueError("at least one external HAR dataset is required")
    for item in datasets:
        item.validate(require_all_classes=require_all_classes_per_dataset)
    lanes = {item.channel_lane for item in datasets}
    rates = {item.sampling_rate_hz for item in datasets}
    lengths = {item.signals.shape[1] for item in datasets}
    ontologies = {item.class_names for item in datasets}
    gravity_sources = {item.gravity_source for item in datasets}
    gravity_cutoffs = {item.gravity_cutoff_hz for item in datasets}
    if (
        len(lanes) != 1
        or len(rates) != 1
        or len(lengths) != 1
        or len(ontologies) != 1
        or len(gravity_sources) != 1
        or len(gravity_cutoffs) != 1
    ):
        raise ValueError("external HAR datasets may be combined only within one aligned lane")
    result = ExternalHARWindows(
        dataset_id=dataset_id,
        channel_lane=datasets[0].channel_lane,
        sampling_rate_hz=datasets[0].sampling_rate_hz,
        signals=np.concatenate([item.signals for item in datasets], axis=0),
        gravity=np.concatenate([item.gravity for item in datasets], axis=0),
        labels=np.concatenate([item.labels for item in datasets], axis=0),
        participant_ids=np.concatenate([item.participant_ids for item in datasets], axis=0),
        session_ids=np.concatenate([item.session_ids for item in datasets], axis=0),
        trial_ids=np.concatenate([item.trial_ids for item in datasets], axis=0),
        window_ids=np.concatenate([item.window_ids for item in datasets], axis=0),
        receipts=tuple(receipt for item in datasets for receipt in item.receipts),
        exclusions=tuple(exclusion for item in datasets for exclusion in item.exclusions),
        source_issues=tuple(issue for item in datasets for issue in item.source_issues),
        class_names=datasets[0].class_names,
        gravity_source=datasets[0].gravity_source,
        gravity_cutoff_hz=datasets[0].gravity_cutoff_hz,
        preprocessing_audit=tuple(audit for item in datasets for audit in item.preprocessing_audit),
    )
    result.validate()
    return result


def _true_runs(mask: NDArray[np.bool_]) -> list[tuple[int, int]]:
    padded = np.concatenate((np.array([False]), mask, np.array([False]))).astype(np.int8)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    stops = np.flatnonzero(changes == -1)
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


def _request_bytes(url: str, *, timeout_seconds: float = 180.0) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.get(url, timeout=timeout_seconds)
            response.raise_for_status()
            return response.content
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(0.5 * 2**attempt)
    raise RuntimeError(f"failed to stream source after retries: {url}") from last_error


def _request_json(url: str, *, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    query = url if parameters is None else f"{url}?{urlencode(parameters)}"
    payload = _request_bytes(query, timeout_seconds=60.0)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"source API did not return an object: {query}")
    return cast(dict[str, Any], value)


def _receipt(
    *,
    dataset_id: str,
    locator: str,
    payload: bytes,
    member: str | None = None,
    declared_size: int | None = None,
    declared_md5: str | None = None,
    archive_crc32: int | None = None,
) -> SourceReceipt:
    actual_size = len(payload)
    if declared_size is not None and actual_size != declared_size:
        raise ValueError(
            f"source size mismatch for {member or locator}: expected {declared_size}, got {actual_size}"
        )
    computed_md5 = hashlib.md5(payload).hexdigest() if declared_md5 is not None else None
    verified = computed_md5 == declared_md5 if declared_md5 is not None else None
    if verified is False:
        raise ValueError(f"source MD5 mismatch for {member or locator}")
    return SourceReceipt(
        dataset_id=dataset_id,
        locator=locator,
        member=member,
        declared_size_bytes=declared_size,
        received_size_bytes=actual_size,
        computed_sha256=hashlib.sha256(payload).hexdigest(),
        declared_digest_algorithm="md5" if declared_md5 is not None else None,
        declared_digest=declared_md5,
        computed_declared_digest=computed_md5,
        declared_digest_verified=verified,
        archive_crc32=None if archive_crc32 is None else f"{archive_crc32:08x}",
    )


def _contiguous_signal_runs(
    timestamps: FloatArray,
    *signals: FloatArray,
    nominal_rate_hz: float,
    maximum_gap_factor: float,
) -> list[tuple[int, int]]:
    """Return finite monotonic signal segments without consulting activity labels."""

    if timestamps.ndim != 1 or any(item.shape[0] != timestamps.size for item in signals):
        raise ValueError("timestamp and signal arrays must align")
    finite = np.isfinite(timestamps)
    for item in signals:
        if item.ndim == 1:
            finite &= np.isfinite(item)
        else:
            finite &= np.isfinite(item).all(axis=tuple(range(1, item.ndim)))
    output: list[tuple[int, int]] = []
    for finite_start, finite_stop in _true_runs(finite):
        values = timestamps[finite_start:finite_stop]
        if values.size == 0:
            continue
        boundaries = np.ones(values.size, dtype=np.bool_)
        if values.size > 1:
            delta = np.diff(values)
            boundaries[1:] = (delta <= 0.0) | (delta > maximum_gap_factor / nominal_rate_hz)
        starts = np.flatnonzero(boundaries)
        stops = np.concatenate((starts[1:], np.array([values.size], dtype=np.int64)))
        output.extend(
            (finite_start + int(start), finite_start + int(stop))
            for start, stop in zip(starts, stops, strict=True)
        )
    return output


def load_fog_star(
    *,
    participant_limit: int | None = None,
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    gravity_cutoff_hz: float = 0.30,
    maximum_gap_factor: float = 3.0,
) -> ExternalHARWindows:
    """Stream and materialize the FoG-STAR back-IMU three-class core."""

    dataset_id = "fog_star_v3"
    url = "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content"
    payload = _request_bytes(url)
    receipt = _receipt(
        dataset_id=dataset_id,
        locator=url,
        payload=payload,
        declared_size=119_629_580,
        declared_md5="952a37ab147da35e6d4e7a1e9bac44cb",
    )
    columns = [
        "timestamp",
        "back_acc_x",
        "back_acc_y",
        "back_acc_z",
        "back_gyro_x",
        "back_gyro_y",
        "back_gyro_z",
        "activity",
        "subjectID",
        "sessionID",
    ]
    frame = pd.read_csv(io.BytesIO(payload), usecols=columns)
    participants = sorted(frame["subjectID"].dropna().astype(int).unique().tolist())
    if participant_limit is not None:
        if participant_limit < 1:
            raise ValueError("participant limit must be positive")
        participants = participants[:participant_limit]
    selected = frame[frame["subjectID"].isin(participants)]
    participant_plan = _prewindow_partition_plan(
        dataset_id,
        [f"fogstar:{participant:03d}" for participant in participants],
        roster_basis=(
            "all provider subjectID values selected before session segmentation; "
            "independent of activity annotations and scored-window eligibility"
        ),
    )
    label_map = {1: 0, 2: 1, 3: 2, 6: 0, 7: 0}
    accumulator = _WindowAccumulator.empty()
    # taskID is an annotated task code, not a verified acquisition-start event.
    # Even changing or removing it must not change the signal/grid construction.
    grouping = ["subjectID", "sessionID"]
    for keys, trial_frame in selected.groupby(grouping, sort=True, dropna=False):
        subject, session_value = cast(tuple[Any, Any], keys)
        timestamps = trial_frame["timestamp"].to_numpy(dtype=np.float64)
        activities = trial_frame["activity"].to_numpy(dtype=np.float64)
        total = (
            trial_frame[["back_acc_x", "back_acc_y", "back_acc_z"]].to_numpy(dtype=np.float64)
            * STANDARD_GRAVITY_M_S2
        )
        gyro = (
            trial_frame[["back_gyro_x", "back_gyro_y", "back_gyro_z"]].to_numpy(dtype=np.float64)
            * np.pi
            / 180.0
        )
        participant = f"fogstar:{int(subject):03d}"
        session = f"{participant}:session-{int(session_value):03d}"
        trial = f"{session}:recording"
        physical_runs = _contiguous_signal_runs(
            timestamps,
            total,
            gyro,
            nominal_rate_hz=60.0,
            maximum_gap_factor=maximum_gap_factor,
        )
        omitted_rows = timestamps.size - sum(stop - start for start, stop in physical_runs)
        if omitted_rows:
            accumulator.exclusions.append(
                {
                    "trial_id": trial,
                    "reason": "nonfinite_sensor_or_timestamp",
                    "source_rows": omitted_rows,
                }
            )
        for run_index, (physical_start, physical_stop) in enumerate(physical_runs):
            if physical_stop - physical_start < 3:
                accumulator.exclusions.append(
                    {
                        "trial_id": trial,
                        "run_index": run_index,
                        "reason": "short_physical_segment",
                        "source_rows": physical_stop - physical_start,
                    }
                )
                continue
            segment = resample_physical_segment(
                timestamps=timestamps[physical_start:physical_stop],
                acceleration=total[physical_start:physical_stop],
                gyroscope=gyro[physical_start:physical_stop],
                gravity=None,
                source_rate_hz=60.0,
                target_rate_hz=target_rate_hz,
                gravity_cutoff_hz=gravity_cutoff_hz,
                window_samples=window_samples,
            )
            accumulator.add_annotated_segment(
                segment,
                source_labels=activities[physical_start:physical_stop],
                label_map=label_map,
                participant=participant,
                session=session,
                trial=trial,
                run_index=run_index,
            )
    return accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=[receipt],
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=gravity_cutoff_hz,
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": (
                "provider participant/session recording, then timestamp-discontinuity "
                "and finite-sensor runs"
            ),
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": (
                "development evidence on an annotation-independent repository grid; "
                "offline polyphase resampling is not a zero-lookahead streaming implementation"
            ),
            "annotation_application": (
                "activity annotations are projected only after signal/gravity resampling and "
                "global candidate-window construction"
            ),
            "released_timestamp_unit_documentation": "milliseconds",
            "fixed_timestamp_interpretation": (
                "seconds, based on the pinned file's approximately 1/60 increments and "
                "provider-declared 60 Hz sampling; the provider documentation discrepancy "
                "remains disclosed"
            ),
        },
        participant_partition_plan=participant_plan,
    )


def _find_har_imu_root(folders: dict[str, Any]) -> dict[str, Any]:
    queue = [folders]
    while queue:
        current = queue.pop()
        if current.get("name") == "HAR_IMU_IL":
            return current
        children = current.get("folders", [])
        if isinstance(children, list):
            queue.extend(item for item in children if isinstance(item, dict))
    raise ValueError("CSIRO folder tree does not contain HAR_IMU_IL")


def _imu_folder_paths(
    folder_payload: dict[str, Any],
    *,
    participant_limit: int | None,
    repetition_limit: int | None,
    activities: tuple[str, ...],
) -> list[tuple[str, str, str, str]]:
    root = _find_har_imu_root(folder_payload)
    participant_nodes = [
        cast(dict[str, Any], item)
        for item in cast(list[Any], root.get("folders", []))
        if isinstance(item, dict) and str(item.get("name", "")).startswith("P_")
    ]
    participant_nodes.sort(key=lambda item: str(item["name"]))
    if participant_limit is not None:
        if participant_limit < 1:
            raise ValueError("participant limit must be positive")
        participant_nodes = participant_nodes[:participant_limit]
    paths: list[tuple[str, str, str, str]] = []
    for participant_node in participant_nodes:
        participant = str(participant_node["name"])
        repetition_nodes = [
            cast(dict[str, Any], item)
            for item in cast(list[Any], participant_node.get("folders", []))
            if isinstance(item, dict) and str(item.get("name", "")).startswith("Repetition_")
        ]
        repetition_nodes.sort(key=lambda item: str(item["name"]))
        if repetition_limit is not None:
            if repetition_limit < 1:
                raise ValueError("repetition limit must be positive")
            repetition_names = [f"Repetition_{index}" for index in range(1, repetition_limit + 1)]
        else:
            repetition_names = [str(item["name"]) for item in repetition_nodes]
        for repetition in repetition_names:
            for activity in activities:
                # Resolve every requested activity path, including an absent folder.  The
                # inventory gate can then exclude the participant explicitly instead of
                # silently reducing their class support.
                path = f"HAR_IMU_IL/{participant}/{repetition}/{activity}"
                paths.append((participant, repetition, activity, path))
    return paths


def _resolve_imu_file(specification: tuple[str, str, str, str]) -> _IMUSourceFile:
    participant, repetition, activity, folder = specification
    url = "https://data.csiro.au/dap/ws/v2/collections/74700/files"
    payload = _request_json(url, parameters={"folder": folder, "size": 100, "page": 1})
    files = cast(list[Any], payload.get("file", []))
    matches = [
        cast(dict[str, Any], item)
        for item in files
        if isinstance(item, dict) and str(item.get("filename", "")).endswith("/Body-WT.csv")
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one Body-WT.csv in {folder}, found {len(matches)}")
    selected = matches[0]
    link = cast(dict[str, Any], selected["link"])
    return _IMUSourceFile(
        participant=participant,
        repetition=repetition,
        activity=activity,
        file_id=int(selected["id"]),
        filename=str(selected["filename"]),
        file_size=int(selected["fileSize"]),
        locator=str(link["href"]),
    )


def _resolve_imu_file_optional(
    specification: tuple[str, str, str, str],
) -> tuple[tuple[str, str, str, str], _IMUSourceFile | None]:
    try:
        return specification, _resolve_imu_file(specification)
    except ValueError as exc:
        if "found 0" not in str(exc):
            raise
        return specification, None


def _imu_har_il_inventory(
    *,
    participant_limit: int | None,
    repetition_limit: int | None,
    activities: tuple[str, ...],
    workers: int,
    require_complete_core: bool = True,
) -> tuple[tuple[_IMUSourceFile, ...], tuple[dict[str, Any], ...]]:
    folders = _request_json("https://data.csiro.au/dap/ws/v2/collections/74700/folders")
    specifications = _imu_folder_paths(
        folders,
        participant_limit=participant_limit,
        repetition_limit=repetition_limit,
        activities=activities,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        resolved = list(pool.map(_resolve_imu_file_optional, specifications))
    incomplete_participants = {
        specification[0] for specification, source in resolved if source is None
    }
    files = tuple(
        sorted(
            (
                source
                for specification, source in resolved
                if source is not None
                and (not require_complete_core or specification[0] not in incomplete_participants)
            ),
            key=lambda item: (item.participant, item.repetition, item.activity),
        )
    )
    exclusions: list[dict[str, Any]] = []
    for participant in sorted(incomplete_participants):
        missing = [
            specification[3]
            for specification, source in resolved
            if specification[0] == participant and source is None
        ]
        exclusions.append(
            {
                "participant_id": f"imuharil:{participant}",
                "reason": "missing Body-WT.csv in at least one requested core trial",
                "action": "participant excluded before windowing to retain complete core support"
                if require_complete_core
                else "missing trial omitted; other available trials retained",
                "missing_folders": missing,
            }
        )
    return files, tuple(exclusions)


def discover_imu_har_il_files(
    *,
    participant_limit: int | None = None,
    repetition_limit: int | None = None,
    activities: tuple[str, ...] = ("Walk", "Sit", "Stand"),
    workers: int = 12,
) -> tuple[_IMUSourceFile, ...]:
    """Resolve waist-IMU file identities before any trial is windowed."""

    if set(activities) - {"Walk", "Sit", "Stand", "Stairs", "Sit_stand", "Lay_sit"}:
        raise ValueError("unsupported IMU-HAR-IL activity requested")
    folders = _request_json("https://data.csiro.au/dap/ws/v2/collections/74700/folders")
    specifications = _imu_folder_paths(
        folders,
        participant_limit=participant_limit,
        repetition_limit=repetition_limit,
        activities=activities,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(_resolve_imu_file, specifications))
    files.sort(key=lambda item: (item.participant, item.repetition, item.activity))
    return tuple(files)


def _download_imu_payload(source: _IMUSourceFile) -> tuple[_IMUSourceFile, bytes]:
    return source, _request_bytes(source.locator)


def load_imu_har_il(
    *,
    participant_limit: int | None = None,
    repetition_limit: int | None = None,
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    gravity_cutoff_hz: float = 0.30,
    discovery_workers: int = 12,
    download_workers: int = 8,
    selection_policy: str = "complete_requested_core",
) -> ExternalHARWindows:
    """Stream the IMU-HAR-IL Body-WT core without storing third-party raw files."""

    if selection_policy not in {"complete_requested_core", "available_valid_trials"}:
        raise ValueError("unknown IMU-HAR-IL cohort selection policy")
    complete_core = selection_policy == "complete_requested_core"
    dataset_id = "imu_har_il_v1"
    files, exclusions = _imu_har_il_inventory(
        participant_limit=participant_limit,
        repetition_limit=repetition_limit,
        activities=("Walk", "Sit", "Stand"),
        workers=discovery_workers,
        require_complete_core=complete_core,
    )
    participant_plan = _prewindow_partition_plan(
        dataset_id,
        [
            *[f"imuharil:{source.participant}" for source in files],
            *[str(item["participant_id"]) for item in exclusions],
        ],
        roster_basis=(
            "all requested/provider-inventory participants, including people with missing "
            "activity files, frozen before download, sensor parsing, finite-run segmentation, "
            "resampling, and window extraction; local Activity_label values are audit-only"
        ),
    )
    label_map = {"Walk": 0, "Sit": 1, "Stand": 2}
    provider_label_map = {"Walk": 5, "Sit": 1, "Stand": 2}
    accumulator = _WindowAccumulator.empty()
    receipts: list[SourceReceipt] = []
    issues: list[dict[str, Any]] = []
    signal_columns = ["Acc_X", "Acc_Y", "Acc_Z", "Gyr_X", "Gyr_Y", "Gyr_Z"]
    downloaded: list[tuple[_IMUSourceFile, pd.DataFrame]] = []
    invalid_trials: dict[str, list[dict[str, Any]]] = {}
    with ThreadPoolExecutor(max_workers=download_workers) as pool:
        for source, payload in pool.map(_download_imu_payload, files):
            receipts.append(
                _receipt(
                    dataset_id=dataset_id,
                    locator=source.locator,
                    member=source.filename,
                    payload=payload,
                    declared_size=source.file_size,
                )
            )
            available_columns = pd.read_csv(io.BytesIO(payload), nrows=0).columns.tolist()
            missing_signals = sorted(set(signal_columns) - set(available_columns))
            if missing_signals:
                invalid_trials.setdefault(source.participant, []).append(
                    {
                        "member": source.filename,
                        "reason": "required sensor columns are absent",
                        "missing_signal_columns": missing_signals,
                    }
                )
                continue
            label_column = next(
                (
                    name
                    for name in ("Activity_label", "Activity Label")
                    if name in available_columns
                ),
                None,
            )
            selected_columns = signal_columns + ([label_column] if label_column else [])
            frame = pd.read_csv(io.BytesIO(payload), usecols=selected_columns)
            if label_column is not None:
                source_labels = pd.to_numeric(frame[label_column], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                if label_column != "Activity_label":
                    issues.append(
                        {
                            "member": source.filename,
                            "issue": "activity-label column uses the documented schema variant",
                            "observed_column": label_column,
                            "canonical_column": "Activity_label",
                            "action": (
                                "numeric annotation audited only; the provider activity-folder "
                                "identity remains the released ontology source"
                            ),
                        }
                    )
                if source_labels.size == 0 or not np.isfinite(source_labels).all():
                    issues.append(
                        {
                            "member": source.filename,
                            "issue": "provider activity-label column is empty or non-finite",
                            "label_column": label_column,
                            "action": (
                                "annotation limitation retained; sensor transforms, window grid, "
                                "and participant eligibility use the activity-folder identity only"
                            ),
                        }
                    )
                elif set(np.unique(source_labels).tolist()) != {
                    provider_label_map[source.activity]
                }:
                    issues.append(
                        {
                            "member": source.filename,
                            "issue": "provider activity-label column conflicts with folder label",
                            "label_column": label_column,
                            "observed_labels": sorted(np.unique(source_labels).tolist()),
                            "expected_label": provider_label_map[source.activity],
                            "action": (
                                "annotation conflict retained; sensor transforms, window grid, "
                                "and participant eligibility use the activity-folder identity only"
                            ),
                        }
                    )
            else:
                issues.append(
                    {
                        "member": source.filename,
                        "issue": "provider activity-label column absent",
                        "action": (
                            "annotation limitation retained; sensor transforms, window grid, and "
                            "participant eligibility use the activity-folder identity only"
                        ),
                    }
                )
            signal_values = frame[signal_columns].to_numpy(dtype=np.float64)
            finite = np.isfinite(signal_values).all(axis=1)
            complete_finite_segment = any(
                _resampled_sample_count(
                    stop - start,
                    source_rate_hz=60.0,
                    target_rate_hz=target_rate_hz,
                )
                >= window_samples
                for start, stop in _true_runs(finite)
                if stop - start >= 3
            )
            if not complete_finite_segment:
                invalid_trials.setdefault(source.participant, []).append(
                    {
                        "member": source.filename,
                        "reason": "no finite segment can produce one complete analysis window",
                        "row_count": int(frame.shape[0]),
                        "finite_sensor_row_count": int(finite.sum()),
                    }
                )
            downloaded.append((source, frame))

    invalid_participants = set(invalid_trials)
    invalid_members = {str(item["member"]) for items in invalid_trials.values() for item in items}
    data_quality_exclusions = tuple(
        {
            "participant_id": f"imuharil:{participant}",
            "reason": "one or more requested core trials failed the pre-window data-quality gate"
            if complete_core
            else "one or more requested core trials quarantined by the trial-quality gate",
            "action": "participant excluded before any of their trials were windowed"
            if complete_core
            else "affected trials excluded; other valid trials retained",
            "invalid_trials": invalid_trials[participant],
        }
        for participant in sorted(invalid_participants)
    )
    retained_sources = [
        (source, frame)
        for source, frame in downloaded
        if not (
            (complete_core and source.participant in invalid_participants)
            or (not complete_core and source.filename in invalid_members)
        )
    ]
    for source, frame in retained_sources:
        total = frame[["Acc_X", "Acc_Y", "Acc_Z"]].to_numpy(dtype=np.float64)
        gyro = frame[["Gyr_X", "Gyr_Y", "Gyr_Z"]].to_numpy(dtype=np.float64) * np.pi / 180.0
        participant = f"imuharil:{source.participant}"
        session = f"{participant}:collection-visit"
        trial = f"{participant}:{source.repetition}:{source.activity}"
        accumulator.add_uniform_trial(
            total_acceleration=total,
            gyroscope=gyro,
            gravity=None,
            source_rate_hz=60.0,
            target_rate_hz=target_rate_hz,
            gravity_cutoff_hz=gravity_cutoff_hz,
            label=label_map[source.activity],
            participant=participant,
            session=session,
            trial=trial,
            run_index=0,
            window_samples=window_samples,
            segment_boundary_annotation_conditioned=True,
        )
    result = accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=receipts,
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=gravity_cutoff_hz,
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": (
                "provider posthoc activity-segmented Body-WT file; original continuous "
                "timestamps and removed breaks are unavailable"
            ),
            "repository_signal_grid_annotation_independent": False,
            "provider_upstream_annotation_conditioned": True,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": (
                "provider-presegmented scripted-activity development diagnostic only; "
                "not continuous or streaming-valid HAR evidence"
            ),
            "local_numeric_annotation_affects_signal_grid": False,
            "folder_identity_is_released_ontology_source": True,
            "continuous_source_reconstruction_possible": False,
        },
        participant_partition_plan=participant_plan,
    )
    requested_people = {f"imuharil:{item.participant}" for item in files} | {
        str(item["participant_id"]) for item in exclusions
    }
    requested_repetitions = (
        [f"Repetition_{index}" for index in range(1, repetition_limit + 1)]
        if repetition_limit is not None
        else sorted({source.repetition for source in files})
    )
    observed_people = sorted(set(result.participant_ids.tolist()))
    incomplete_people = sorted(str(item["participant_id"]) for item in exclusions)
    quality_people = sorted(f"imuharil:{participant}" for participant in invalid_participants)
    cohort_audit = {
        "protocol_id": (
            "imu-har-il-complete-requested-core-v1"
            if complete_core
            else "imu-har-il-available-trials-v1"
        ),
        "inventory_protocol_id": IMU_HAR_IL_COHORT_INVENTORY_PROTOCOL,
        "selection_policy": selection_policy,
        "provider_reported_participant_count": 50,
        "requested_repetition_limit": repetition_limit,
        "requested_repetitions": requested_repetitions,
        "requested_activities": ["Walk", "Sit", "Stand"],
        "requested_participant_count": len(requested_people),
        "requested_participants": sorted(requested_people),
        "requested_trial_count": (
            len(requested_people) * len(requested_repetitions) * len(label_map)
        ),
        "inventory_selected_source_file_count": len(files),
        "source_inventory_sha256": _imu_source_inventory_sha256(files),
        "source_payload_inventory_sha256": _imu_source_payload_inventory_sha256(receipts),
        "inventory_incomplete_participant_count": len(incomplete_people),
        "inventory_incomplete_participants": incomplete_people,
        "missing_requested_trial_count": sum(len(item["missing_folders"]) for item in exclusions),
        "missing_requested_trial_folders_sha256": _imu_missing_trial_inventory_sha256(exclusions),
        "data_quality_affected_participant_count": len(quality_people),
        "data_quality_affected_participants": quality_people,
        "quarantined_trial_count": len(invalid_members),
        "participant_exclusion_record_count": len(exclusions) + len(data_quality_exclusions),
        "retained_source_file_count": len(retained_sources),
        "retained_participant_count": len(observed_people),
        "retained_participants": observed_people,
        "retained_trial_count": int(np.unique(result.trial_ids).size),
        "participants_with_no_retained_windows": sorted(requested_people - set(observed_people)),
        "independent_unit": "participant; repetitions do not increase independent N",
        "complete_case_or_repetition1_before_after_comparison_allowed": False,
    }
    return replace(
        result,
        exclusions=tuple((*result.exclusions, *exclusions, *data_quality_exclusions)),
        source_issues=tuple(issues),
        cohort_audit=cohort_audit,
    )


def _irregular_segments(
    timestamps: FloatArray, *, maximum_gap_factor: float
) -> list[tuple[int, int]]:
    results, _audit = _irregular_segments_with_audit(
        timestamps, maximum_gap_factor=maximum_gap_factor
    )
    return results


def _irregular_segments_with_audit(
    timestamps: FloatArray, *, maximum_gap_factor: float
) -> tuple[list[tuple[int, int]], dict[str, int]]:
    """Return observable timestamp runs plus exhaustive boundary/drop accounting."""

    if timestamps.ndim != 1 or not np.isfinite(maximum_gap_factor) or maximum_gap_factor <= 0.0:
        raise ValueError("irregular timestamps must be a vector with a positive gap factor")
    finite = np.isfinite(timestamps)
    results: list[tuple[int, int]] = []
    finite_runs = _true_runs(finite)
    short_run_count = 0
    short_run_rows = 0
    no_positive_count = 0
    no_positive_rows = 0
    gap_count = 0
    nonincreasing_count = 0
    for start, stop in finite_runs:
        values = timestamps[start:stop]
        if values.size < 3:
            short_run_count += 1
            short_run_rows += int(values.size)
            continue
        positive = np.diff(values)
        nominal_values = positive[positive > 0.0]
        if nominal_values.size == 0:
            no_positive_count += 1
            no_positive_rows += int(values.size)
            continue
        nominal = float(np.median(nominal_values))
        nonincreasing = positive <= 0.0
        gaps = positive > maximum_gap_factor * nominal
        nonincreasing_count += int(nonincreasing.sum())
        gap_count += int(gaps.sum())
        boundary = np.ones(values.size, dtype=np.bool_)
        boundary[1:] = nonincreasing | gaps
        starts = np.flatnonzero(boundary)
        stops = np.concatenate((starts[1:], np.array([values.size], dtype=np.int64)))
        results.extend(
            (start + int(left), start + int(right))
            for left, right in zip(starts, stops, strict=True)
        )
    audit = {
        "source_rows": int(timestamps.size),
        "nonfinite_timestamp_rows": int((~finite).sum()),
        "finite_timestamp_run_count": len(finite_runs),
        "short_timestamp_run_count": short_run_count,
        "short_timestamp_run_rows": short_run_rows,
        "timestamp_runs_without_positive_interval_count": no_positive_count,
        "timestamp_runs_without_positive_interval_rows": no_positive_rows,
        "timestamp_gap_boundary_count": gap_count,
        "timestamp_nonincreasing_boundary_count": nonincreasing_count,
        "observable_timestamp_segment_count": len(results),
    }
    represented_rows = sum(stop - start for start, stop in results)
    accounted_rows = (
        audit["nonfinite_timestamp_rows"] + short_run_rows + no_positive_rows + represented_rows
    )
    if accounted_rows != timestamps.size:
        raise AssertionError("irregular timestamp accounting is not exhaustive")
    return results, audit


def _interpolate_to_rate(
    timestamps: FloatArray, values: FloatArray, *, target_rate_hz: float
) -> FloatArray:
    if timestamps.ndim != 1 or values.shape[0] != timestamps.size or values.ndim != 2:
        raise ValueError("timestamp interpolation arrays do not align")
    target = np.arange(timestamps[0], timestamps[-1] + 1e-12, 1.0 / target_rate_hz)
    if target.size < 2:
        return np.empty((0, values.shape[1]), dtype=np.float64)
    return np.column_stack(
        [np.interp(target, timestamps, values[:, column]) for column in range(values.shape[1])]
    )


def load_har_pmd_native(
    *,
    source_archive: Path | None = None,
    participant_limit: int | None = None,
    environments: tuple[str, ...] = ("indoor", "outdoor"),
    activities: tuple[str, ...] = ("still", "walking", "crutches", "walker", "manual"),
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    maximum_gap_factor: float = 3.0,
) -> ExternalHARWindows:
    """Stream HAR-PMD's native nine-channel phone interface.

    This adapter uses a five-class stress ontology.  Its output is represented by the same
    container for alignment, but callers must not pass it to the three-class validator or
    pool its scores with the derived-gravity core.
    """

    dataset_id = "har_pmd_v1"
    url = "https://zenodo.org/api/records/7939223/files/data_publish.zip/content"
    allowed_activities = ("still", "walking", "crutches", "walker", "manual", "electric")
    if set(activities) - set(allowed_activities):
        raise ValueError("unsupported HAR-PMD activity requested")
    if set(environments) - {"indoor", "outdoor"}:
        raise ValueError("unsupported HAR-PMD environment requested")
    label_map = {name: index for index, name in enumerate(activities)}
    display_name = {
        "still": "stationary",
        "walking": "walking",
        "crutches": "crutches",
        "walker": "walker",
        "manual": "manual_wheelchair",
        "electric": "electric_wheelchair",
    }
    accumulator = _WindowAccumulator.empty()
    receipts: list[SourceReceipt] = []
    columns = [
        "Time",
        "LAccX",
        "LAccY",
        "LAccZ",
        "GyrX",
        "GyrY",
        "GyrZ",
        "GraX",
        "GraY",
        "GraZ",
    ]
    storage_audit: dict[str, Any] | None = None
    # Both readers expose the same ZIP member interface. Local mode requires the
    # exact pinned full archive before parsing any member; it is never an unchecked cache.
    reader = RemoteZip(url) if source_archive is None else verified_har_pmd_archive(source_archive)
    archive: ZipFile
    with reader as opened:
        if source_archive is None:
            archive = cast(ZipFile, opened)
        else:
            archive, storage_audit = cast(tuple[ZipFile, dict[str, Any]], opened)
        names = sorted(
            name for name in archive.namelist() if name.endswith(".csv") and "/phone/" in name
        )
        participants = sorted(
            {PurePosixPath(name).parts[1] for name in names}, key=lambda item: int(item)
        )
        if participant_limit is not None:
            if participant_limit < 1:
                raise ValueError("participant limit must be positive")
            participants = participants[:participant_limit]
        participant_plan = _prewindow_partition_plan(
            dataset_id,
            [f"harpmd:{int(participant):03d}" for participant in participants],
            roster_basis=(
                "all selected participant directories in the verified provider archive, "
                "frozen before activity-file parsing and window extraction"
            ),
        )
        selected_names = []
        for participant in participants:
            for activity in activities:
                for environment in environments:
                    expected = (
                        f"data_publish/{participant}/phone/"
                        f"{participant}_{activity}_phone_{environment}.csv"
                    )
                    if expected not in names:
                        raise ValueError(f"HAR-PMD archive lacks expected member {expected}")
                    selected_names.append(expected)
        for member in selected_names:
            info = archive.getinfo(member)
            payload = archive.read(member)
            receipts.append(
                replace(
                    _receipt(
                        dataset_id=dataset_id,
                        locator=url,
                        member=member,
                        payload=payload,
                        declared_size=int(info.file_size),
                        archive_crc32=int(info.CRC),
                    ),
                    raw_local_mirror=storage_audit is not None,
                )
            )
            frame = pd.read_csv(io.BytesIO(payload), usecols=columns)
            path = PurePosixPath(member)
            participant_value = path.parts[1]
            stem_parts = path.stem.split("_")
            environment = stem_parts[-1]
            activity = "_".join(stem_parts[1:-2])
            timestamps = frame["Time"].to_numpy(dtype=np.float64)
            linear = frame[["LAccX", "LAccY", "LAccZ"]].to_numpy(dtype=np.float64)
            gyro = frame[["GyrX", "GyrY", "GyrZ"]].to_numpy(dtype=np.float64)
            gravity = frame[["GraX", "GraY", "GraZ"]].to_numpy(dtype=np.float64)
            combined = np.column_stack((linear, gyro, gravity))
            participant = f"harpmd:{int(participant_value):03d}"
            session = f"{participant}:{environment}"
            trial = f"{session}:{activity}"
            timestamp_segments, timestamp_audit = _irregular_segments_with_audit(
                timestamps, maximum_gap_factor=maximum_gap_factor
            )
            nonfinite_sensor_rows = 0
            finite_signal_run_count = 0
            short_finite_run_count = 0
            short_finite_source_rows = 0
            interpolated_source_rows = 0
            interpolated_target_samples = 0
            retained_target_samples = 0
            dropped_tail_target_samples = 0
            candidate_window_count = 0
            for run_index, (start, stop) in enumerate(timestamp_segments):
                finite = np.isfinite(combined[start:stop]).all(axis=1)
                nonfinite_sensor_rows += int((~finite).sum())
                for finite_index, (left, right) in enumerate(_true_runs(finite)):
                    finite_signal_run_count += 1
                    left += start
                    right += start
                    if right - left < 3:
                        short_finite_run_count += 1
                        short_finite_source_rows += right - left
                        continue
                    interpolated = _interpolate_to_rate(
                        timestamps[left:right], combined[left:right], target_rate_hz=target_rate_hz
                    )
                    interpolated_source_rows += right - left
                    interpolated_target_samples += int(interpolated.shape[0])
                    usable = interpolated.shape[0] // window_samples * window_samples
                    retained_target_samples += usable
                    dropped_tail_target_samples += int(interpolated.shape[0] - usable)
                    candidate_window_count += usable // window_samples
                    for window_index in range(usable // window_samples):
                        window_start = window_index * window_samples
                        window_stop = window_start + window_samples
                        primary = interpolated[window_start:window_stop, :6]
                        gravity_window = interpolated[window_start:window_stop, 6:9]
                        identifier = (
                            f"{trial}/run-{run_index:04d}-finite-{finite_index:03d}/"
                            f"window-{window_index:06d}"
                        )
                        accumulator.signals.append(np.asarray(primary, dtype=np.float32))
                        accumulator.gravity.append(np.asarray(gravity_window, dtype=np.float32))
                        accumulator.labels.append(label_map[activity])
                        accumulator.participants.append(participant)
                        accumulator.sessions.append(session)
                        accumulator.trials.append(trial)
                        accumulator.windows.append(identifier)
            timestamp_unusable_rows = (
                timestamp_audit["short_timestamp_run_rows"]
                + timestamp_audit["timestamp_runs_without_positive_interval_rows"]
            )
            accounted_source_rows = (
                timestamp_audit["nonfinite_timestamp_rows"]
                + timestamp_unusable_rows
                + nonfinite_sensor_rows
                + short_finite_source_rows
                + interpolated_source_rows
            )
            if accounted_source_rows != timestamps.size:
                raise AssertionError("HAR-PMD source-row accounting is not exhaustive")
            if retained_target_samples + dropped_tail_target_samples != interpolated_target_samples:
                raise AssertionError("HAR-PMD target-sample accounting is not exhaustive")
            accumulator.preprocessing_audit.append(
                {
                    "protocol_id": "har-pmd-scripted-recording-grid-v1",
                    "member": member,
                    "participant_id": participant,
                    "session_id": session,
                    "trial_id": trial,
                    "source_rows": int(timestamps.size),
                    "target_rate_hz": target_rate_hz,
                    "window_samples": window_samples,
                    "maximum_gap_factor": maximum_gap_factor,
                    "timestamps_sha256": _array_sha256(timestamps),
                    "native_nine_channel_source_sha256": _array_sha256(combined),
                    **timestamp_audit,
                    "nonfinite_sensor_rows": nonfinite_sensor_rows,
                    "finite_signal_run_count": finite_signal_run_count,
                    "short_finite_signal_run_count": short_finite_run_count,
                    "short_finite_signal_run_rows": short_finite_source_rows,
                    "interpolated_source_rows": interpolated_source_rows,
                    "interpolated_target_samples": interpolated_target_samples,
                    "candidate_window_count": candidate_window_count,
                    "retained_target_samples": retained_target_samples,
                    "dropped_tail_target_samples": dropped_tail_target_samples,
                    "source_rows_accounted": accounted_source_rows,
                    "target_samples_accounted": (
                        retained_target_samples + dropped_tail_target_samples
                    ),
                    "annotation_dependency_within_recording": False,
                    "interpolation_passes_per_finite_run": 1,
                }
            )
            for reason, count, unit in (
                (
                    "nonfinite_timestamp",
                    timestamp_audit["nonfinite_timestamp_rows"],
                    "source_rows",
                ),
                ("unusable_timestamp_run", timestamp_unusable_rows, "source_rows"),
                ("nonfinite_sensor", nonfinite_sensor_rows, "source_rows"),
                ("short_finite_signal_run", short_finite_source_rows, "source_rows"),
                ("incomplete_resampled_tail", dropped_tail_target_samples, "target_samples"),
            ):
                if count:
                    accumulator.exclusions.append(
                        {
                            "trial_id": trial,
                            "member": member,
                            "reason": reason,
                            unit: count,
                        }
                    )
    # The container validator's core class constraint is deliberately bypassed for this
    # declared five-class stress endpoint; all alignment and finiteness checks follow here.
    if not accumulator.signals:
        raise ValueError("HAR-PMD produced no complete native-interface windows")
    result = ExternalHARWindows(
        dataset_id=dataset_id,
        channel_lane="native-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        signals=np.stack(accumulator.signals).astype(np.float32, copy=False),
        gravity=np.stack(accumulator.gravity).astype(np.float32, copy=False),
        labels=np.asarray(accumulator.labels, dtype=np.int64),
        participant_ids=np.asarray(accumulator.participants, dtype=np.str_),
        session_ids=np.asarray(accumulator.sessions, dtype=np.str_),
        trial_ids=np.asarray(accumulator.trials, dtype=np.str_),
        window_ids=np.asarray(accumulator.windows, dtype=np.str_),
        receipts=tuple(receipts),
        exclusions=tuple(accumulator.exclusions),
        source_issues=(),
        class_names=tuple(display_name[name] for name in activities),
        gravity_source="provider_native_gravity_vector",
        gravity_cutoff_hz=None,
        preprocessing_audit=tuple(accumulator.preprocessing_audit),
        source_storage_audit=storage_audit,
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": (
                "provider scripted participant/activity/environment phone recording file, "
                "then timestamp-discontinuity and finite-sensor runs"
            ),
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": (
                "scripted activity stress test only; not free-living continuous or clinical "
                "ability validation"
            ),
            "activity_semantics_source": "provider recording filename and scripted protocol",
            "timestamped_source": True,
            "provider_scripted_activity_conditioned": True,
        },
        participant_partition_plan=participant_plan,
    )
    count = result.labels.size
    if (
        result.signals.shape != (count, window_samples, 6)
        or result.gravity.shape != (count, window_samples, 3)
        or len(set(result.window_ids.tolist())) != count
        or not np.isfinite(result.nine_channel_signals).all()
        or set(result.labels.tolist()) != set(range(len(activities)))
    ):
        raise ValueError("HAR-PMD native-interface materialization failed validation")
    result.validate()
    return result


def _sole_session_arrays(
    payload: bytes,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
    """Read only the right-insole IMU and camera labels from one MATLAB 7.3 object."""

    try:
        import h5py  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised by optional-dependency installs
        raise RuntimeError("Sole-HARmony requires the research dependency h5py") from exc
    with h5py.File(io.BytesIO(payload), "r") as handle:
        root_names = [name for name in handle if not str(name).startswith("#")]
        if root_names != ["DataStruct"]:
            raise ValueError(f"unexpected Sole-HARmony HDF5 roots: {root_names}")
        root = handle["DataStruct"]
        right = root["InsoleR"]

        def vector(name: str) -> FloatArray:
            values = np.asarray(right[name], dtype=np.float64).squeeze()
            if values.ndim != 1 or values.size == 0:
                raise ValueError(f"Sole-HARmony field {name} is not a non-empty vector")
            return np.asarray(values, dtype=np.float64)

        timestamps = vector("t_ms")
        linear = np.column_stack((vector("lin_acc_x"), vector("lin_acc_y"), vector("lin_acc_z")))
        gyroscope = np.column_stack((vector("gyr_x"), vector("gyr_y"), vector("gyr_z")))
        raw = np.column_stack((vector("raw_acc_x"), vector("raw_acc_y"), vector("raw_acc_z")))
        labels = np.asarray(root["labelsCam"], dtype=np.float64).squeeze()
    if labels.ndim != 2:
        raise ValueError("Sole-HARmony camera labels are not a matrix")
    if labels.shape[0] == 3 and labels.shape[1] != 3:
        labels = labels.T
    if labels.shape[1] != 3:
        raise ValueError("Sole-HARmony labels must contain label/start/end columns")
    if linear.shape != raw.shape or gyroscope.shape != raw.shape:
        raise ValueError("Sole-HARmony right-insole vectors do not align")
    if timestamps.size != raw.shape[0]:
        raise ValueError("Sole-HARmony timestamps do not align with IMU vectors")
    return timestamps, linear, gyroscope, raw, np.asarray(labels, dtype=np.float64)


def _validated_sole_camera_intervals(
    labels: FloatArray,
    *,
    member: str,
    annotation_issues: list[dict[str, Any]],
) -> tuple[tuple[int, int, float, float], ...]:
    """Validate camera annotations without exposing them to signal segmentation."""

    ordered = labels[np.argsort(labels[:, 1], kind="stable")]
    previous_stop = float("-inf")
    intervals: list[tuple[int, int, float, float]] = []
    for interval_index, row in enumerate(ordered):
        if not np.isfinite(row).all() or row[0] != np.rint(row[0]):
            raise ValueError(f"invalid camera annotation code in {member}")
        source_label = int(row[0])
        start_time = float(row[1])
        stop_time = float(row[2])
        if source_label == -1 and start_time == stop_time:
            annotation_issues.append(
                {
                    "protocol_id": "sole-harmony-zero-duration-unknown-v1",
                    "member": member,
                    "stable_sorted_interval_index": interval_index,
                    "start_time": start_time,
                    "stop_time": stop_time,
                    "source_label": source_label,
                    "excluded_duration_ms": 0.0,
                    "reason": "zero-duration unknown camera marker",
                    "action": "empty marker omitted; positive-duration bouts unchanged",
                }
            )
            continue
        if stop_time <= start_time:
            raise ValueError(f"invalid camera interval in {member}")
        if start_time < previous_stop:
            raise ValueError(f"overlapping camera intervals in {member}")
        previous_stop = stop_time
        intervals.append((interval_index, source_label, start_time, stop_time))
    return tuple(intervals)


def _sole_sample_annotations(
    timestamps_ms: FloatArray,
    intervals: tuple[tuple[int, int, float, float], ...],
    *,
    label_map: dict[int, int],
) -> FloatArray:
    """Project validated interval codes to source rows only after fixing the signal grid."""

    annotations = np.full(timestamps_ms.size, np.nan, dtype=np.float64)
    for _interval_index, source_label, start_time, stop_time in intervals:
        if source_label not in label_map:
            continue
        within = (
            np.isfinite(timestamps_ms) & (timestamps_ms >= start_time) & (timestamps_ms < stop_time)
        )
        annotations[within] = float(source_label)
    return annotations


def load_sole_harmony(
    *,
    participant_limit: int = 12,
    sessions_per_participant: int = 2,
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    maximum_gap_factor: float = 3.0,
    boundary_mode: str = "camera_bout_oracle",
) -> ExternalHARWindows:
    """Stream ordered Sole-HARmony sessions for temporal development.

    The historical default deliberately preserves camera-bout resets as an oracle
    diagnostic.  ``session_observable`` instead fixes timestamp/gap/finite-data segments,
    resampling and a global window grid before camera annotations are projected post hoc.
    Both modes use offline polyphase resampling and therefore are not zero-lookahead.
    """

    if not 1 <= participant_limit <= 13:
        raise ValueError("Sole-HARmony participant limit must lie in [1,13]")
    if not 1 <= sessions_per_participant <= 5:
        raise ValueError("Sole-HARmony session limit must lie in [1,5]")
    if boundary_mode not in {"camera_bout_oracle", "session_observable"}:
        raise ValueError("unknown Sole-HARmony boundary mode")
    dataset_id = "sole_harmony_v1"
    record = _request_json("https://zenodo.org/api/records/19242395")
    files = record.get("files", [])
    entries = files.get("entries") if isinstance(files, dict) else files
    if not isinstance(entries, list):
        # Zenodo currently returns a list directly under `files`; retain compatibility
        # with the alternate response shape used by older deployments.
        entries = files
    if not isinstance(entries, list):
        raise ValueError("Zenodo Sole-HARmony record lacks a file listing")
    by_name = {
        str(item["key"]): cast(dict[str, Any], item)
        for item in entries
        if isinstance(item, dict) and "key" in item
    }
    participant_plan = _prewindow_partition_plan(
        dataset_id,
        [f"sole:C{index:03d}" for index in range(1, participant_limit + 1)],
        roster_basis=(
            "requested provider participant archives frozen before session member parsing "
            "and window extraction"
        ),
    )
    accumulator = _WindowAccumulator.empty()
    receipts: list[SourceReceipt] = []
    annotation_issues: list[dict[str, Any]] = []
    label_map = {0: 1, 1: 2, 2: 0, 3: 0, 4: 0}
    for participant_index in range(1, participant_limit + 1):
        participant_code = f"C{participant_index:03d}"
        archive_name = f"{participant_code}.zip"
        if archive_name not in by_name:
            raise ValueError(f"Sole-HARmony record lacks {archive_name}")
        links = cast(dict[str, Any], by_name[archive_name].get("links", {}))
        url = str(links.get("content", links.get("self", "")))
        if not url:
            raise ValueError(f"Sole-HARmony record lacks a content URL for {archive_name}")
        with RemoteZip(url) as archive:
            mat_names = sorted(
                name for name in archive.namelist() if name.endswith("DataStruct.mat")
            )
            selected_mats = mat_names[:sessions_per_participant]
            if len(selected_mats) != sessions_per_participant:
                raise ValueError(
                    f"{participant_code} has {len(selected_mats)} available sessions, "
                    f"expected {sessions_per_participant}"
                )
            for mat_name in selected_mats:
                session_directory = str(PurePosixPath(mat_name).parent)
                meta_name = f"{session_directory}/meta.json"
                if meta_name not in archive.namelist():
                    raise ValueError(f"Sole-HARmony archive lacks {meta_name}")
                meta_info = archive.getinfo(meta_name)
                meta_payload = cast(bytes, archive.read(meta_name))
                receipts.append(
                    _receipt(
                        dataset_id=dataset_id,
                        locator=url,
                        member=meta_name,
                        payload=meta_payload,
                        declared_size=int(meta_info.file_size),
                        archive_crc32=int(meta_info.CRC),
                    )
                )
                metadata = json.loads(meta_payload)
                if not isinstance(metadata, dict):
                    raise ValueError(f"Sole-HARmony metadata is invalid: {meta_name}")
                recording = cast(dict[str, Any], metadata.get("recording", {}))
                units = cast(dict[str, Any], metadata.get("units", {}))
                if (
                    float(recording.get("sampling_hz", -1.0)) != 270.0
                    or units.get("acc") != "m/s^2"
                    or units.get("gyr") != "rad/s"
                ):
                    raise ValueError(f"Sole-HARmony unit/rate contract changed: {meta_name}")
                mat_info = archive.getinfo(mat_name)
                payload = cast(bytes, archive.read(mat_name))
                receipts.append(
                    _receipt(
                        dataset_id=dataset_id,
                        locator=url,
                        member=mat_name,
                        payload=payload,
                        declared_size=int(mat_info.file_size),
                        archive_crc32=int(mat_info.CRC),
                    )
                )
                timestamps, linear, gyroscope, raw, labels = _sole_session_arrays(payload)
                if boundary_mode == "camera_bout_oracle" and np.any(np.diff(timestamps) < 0.0):
                    raise ValueError(
                        f"Sole-HARmony timestamps regress within a session: {mat_name}"
                    )
                gravity = raw - linear
                participant = f"sole:{participant_code}"
                session_value = str(metadata.get("session_id", PurePosixPath(mat_name).parent.name))
                session = f"{participant}:session-{session_value}"
                if boundary_mode == "session_observable":
                    # Build every signal segment and its global candidate grid before
                    # validating or projecting camera annotations.
                    timestamps_seconds = timestamps / 1000.0
                    physical_runs = _contiguous_signal_runs(
                        timestamps_seconds,
                        linear,
                        gyroscope,
                        gravity,
                        nominal_rate_hz=270.0,
                        maximum_gap_factor=maximum_gap_factor,
                    )
                    omitted_rows = timestamps.size - sum(
                        stop - start for start, stop in physical_runs
                    )
                    trial = f"{session}:session-recording"
                    if omitted_rows:
                        accumulator.exclusions.append(
                            {
                                "trial_id": trial,
                                "reason": "nonfinite_sensor_or_timestamp",
                                "source_rows": omitted_rows,
                            }
                        )
                    prepared_segments: list[tuple[int, int, int, UniformPhysicalSegment]] = []
                    for run_index, (start, stop) in enumerate(physical_runs):
                        if stop - start < 3:
                            accumulator.exclusions.append(
                                {
                                    "trial_id": trial,
                                    "run_index": run_index,
                                    "reason": "short_physical_segment",
                                    "source_rows": stop - start,
                                }
                            )
                            continue
                        segment = resample_physical_segment(
                            timestamps=timestamps_seconds[start:stop],
                            acceleration=linear[start:stop],
                            gyroscope=gyroscope[start:stop],
                            gravity=gravity[start:stop],
                            source_rate_hz=270.0,
                            target_rate_hz=target_rate_hz,
                            gravity_cutoff_hz=0.30,
                            window_samples=window_samples,
                        )
                        prepared_segments.append((run_index, start, stop, segment))
                    intervals = _validated_sole_camera_intervals(
                        labels,
                        member=mat_name,
                        annotation_issues=annotation_issues,
                    )
                    source_annotations = _sole_sample_annotations(
                        timestamps,
                        intervals,
                        label_map=label_map,
                    )
                    for run_index, start, stop, segment in prepared_segments:
                        accumulator.add_annotated_segment(
                            segment,
                            source_labels=source_annotations[start:stop],
                            label_map=label_map,
                            participant=participant,
                            session=session,
                            trial=trial,
                            run_index=run_index,
                        )
                else:
                    intervals = _validated_sole_camera_intervals(
                        labels,
                        member=mat_name,
                        annotation_issues=annotation_issues,
                    )
                    for interval_index, source_label, start_time, stop_time in intervals:
                        if source_label not in label_map:
                            continue
                        start = int(np.searchsorted(timestamps, start_time, side="left"))
                        stop = int(np.searchsorted(timestamps, stop_time, side="left"))
                        trial = f"{session}:camera-bout-{interval_index:06d}"
                        interval_timestamps = timestamps[start:stop]
                        for segment_index, (left, right) in enumerate(
                            _irregular_segments(
                                interval_timestamps, maximum_gap_factor=maximum_gap_factor
                            )
                        ):
                            accumulator.add_uniform_trial(
                                total_acceleration=linear[start + left : start + right],
                                gyroscope=gyroscope[start + left : start + right],
                                gravity=gravity[start + left : start + right],
                                source_rate_hz=270.0,
                                target_rate_hz=target_rate_hz,
                                gravity_cutoff_hz=0.30,
                                label=label_map[source_label],
                                participant=participant,
                                session=session,
                                trial=trial,
                                run_index=segment_index,
                                window_samples=window_samples,
                                segment_boundary_annotation_conditioned=True,
                            )
                del payload, timestamps, linear, gyroscope, raw, gravity, labels
    result = accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=receipts,
        gravity_source="provider_raw_acceleration_minus_linear_acceleration",
        gravity_cutoff_hz=None,
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": (
                "camera annotation bout (oracle boundary)"
                if boundary_mode == "camera_bout_oracle"
                else (
                    "provider participant/session recording, then timestamp-discontinuity "
                    "and finite-sensor runs"
                )
            ),
            "repository_signal_grid_annotation_independent": (
                boundary_mode == "session_observable"
            ),
            "provider_upstream_annotation_conditioned": False,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": (
                "camera-bout oracle-boundary diagnostic only; not deployable"
                if boundary_mode == "camera_bout_oracle"
                else (
                    "session-observable offline temporal development diagnostic; annotations "
                    "affect only post hoc homogeneous-window eligibility"
                )
            ),
            "boundary_mode": boundary_mode,
            "annotation_application": (
                "camera interval defines each signal/reset boundary"
                if boundary_mode == "camera_bout_oracle"
                else "camera intervals projected after global signal-grid construction"
            ),
        },
        participant_partition_plan=participant_plan,
    )
    return replace(result, source_issues=tuple(annotation_issues))
