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
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import PurePosixPath
from typing import Any, cast
from urllib.parse import urlencode

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import requests
from numpy.typing import NDArray
from remotezip import RemoteZip  # type: ignore[import-untyped]
from scipy.signal import resample_poly  # type: ignore[import-untyped]

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]

STANDARD_GRAVITY_M_S2 = 9.80665
CORE_CLASS_NAMES = ("mobility", "sitting", "standing")
CORE_CLASS_INDEX = {name: index for index, name in enumerate(CORE_CLASS_NAMES)}


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

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable receipt."""

        return asdict(self)


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
        return {
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
            "raw_local_mirror": False,
        }


@dataclass(frozen=True, slots=True)
class _IMUSourceFile:
    participant: str
    repetition: str
    activity: str
    file_id: int
    filename: str
    file_size: int
    locator: str


@dataclass(slots=True)
class _WindowAccumulator:
    signals: list[Float32Array]
    gravity: list[Float32Array]
    labels: list[int]
    participants: list[str]
    sessions: list[str]
    trials: list[str]
    windows: list[str]

    @classmethod
    def empty(cls) -> _WindowAccumulator:
        return cls([], [], [], [], [], [], [])

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
    ) -> None:
        """Resample and window exactly one already boundary-isolated trial run."""

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
        for finite_index, (start, stop) in enumerate(_true_runs(finite)):
            if stop - start < 3:
                continue
            segment_total = total[start:stop]
            segment_gravity = (
                causal_gravity_lowpass(
                    segment_total,
                    sampling_rate_hz=source_rate_hz,
                    cutoff_hz=gravity_cutoff_hz,
                )
                if gravity_values is None
                else gravity_values[start:stop]
            )
            segment_gyro = gyro[start:stop]
            linear = segment_total - segment_gravity if gravity is None else segment_total
            primary = np.column_stack((linear, segment_gyro))
            primary_resampled = resample_uniform(
                primary, source_rate_hz=source_rate_hz, target_rate_hz=target_rate_hz
            )
            gravity_resampled = resample_uniform(
                segment_gravity,
                source_rate_hz=source_rate_hz,
                target_rate_hz=target_rate_hz,
            )
            usable = min(primary_resampled.shape[0], gravity_resampled.shape[0])
            window_count = usable // window_samples
            for window_index in range(window_count):
                window_start = window_index * window_samples
                window_stop = window_start + window_samples
                identifier = (
                    f"{participant}/{session}/{trial}/run-{run_index:04d}-"
                    f"finite-{finite_index:03d}/window-{window_index:06d}"
                )
                self.signals.append(
                    np.asarray(primary_resampled[window_start:window_stop], dtype=np.float32)
                )
                self.gravity.append(
                    np.asarray(gravity_resampled[window_start:window_stop], dtype=np.float32)
                )
                self.labels.append(label)
                self.participants.append(participant)
                self.sessions.append(session)
                self.trials.append(trial)
                self.windows.append(identifier)

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
            exclusions=(),
            source_issues=(),
            gravity_source=gravity_source,
            gravity_cutoff_hz=gravity_cutoff_hz,
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


def resample_uniform(
    values: NDArray[np.floating[Any]], *, source_rate_hz: float, target_rate_hz: float
) -> FloatArray:
    """Polyphase-resample one boundary-isolated uniformly sampled sequence."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 2 or not np.isfinite(array).all():
        raise ValueError("uniform resampling requires a finite [time,channel] sequence")
    if min(source_rate_hz, target_rate_hz) <= 0.0:
        raise ValueError("resampling rates must be positive")
    ratio = Fraction(str(target_rate_hz / source_rate_hz)).limit_denominator(10_000)
    result = resample_poly(array, ratio.numerator, ratio.denominator, axis=0)
    return np.asarray(result, dtype=np.float64)


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


def concatenate_external_windows(
    datasets: tuple[ExternalHARWindows, ...], *, dataset_id: str
) -> ExternalHARWindows:
    """Concatenate only measurement-compatible evidence-lane datasets."""

    if not datasets:
        raise ValueError("at least one external HAR dataset is required")
    for item in datasets:
        item.validate()
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


def _contiguous_label_runs(
    timestamps: FloatArray,
    labels: IntArray,
    *,
    nominal_rate_hz: float,
    maximum_gap_factor: float,
) -> list[tuple[int, int]]:
    if timestamps.shape != labels.shape or timestamps.ndim != 1:
        raise ValueError("timestamp and label vectors must align")
    boundaries = np.ones(labels.size, dtype=np.bool_)
    if labels.size > 1:
        delta = np.diff(timestamps)
        boundaries[1:] = (
            (labels[1:] != labels[:-1])
            | ~np.isfinite(delta)
            | (delta <= 0.0)
            | (delta > maximum_gap_factor / nominal_rate_hz)
        )
    starts = np.flatnonzero(boundaries)
    stops = np.concatenate((starts[1:], np.array([labels.size], dtype=np.int64)))
    return list(zip(starts.tolist(), stops.tolist(), strict=True))


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
        "taskID",
    ]
    frame = pd.read_csv(io.BytesIO(payload), usecols=columns)
    participants = sorted(frame["subjectID"].dropna().astype(int).unique().tolist())
    if participant_limit is not None:
        if participant_limit < 1:
            raise ValueError("participant limit must be positive")
        participants = participants[:participant_limit]
    selected = frame[frame["subjectID"].isin(participants)]
    label_map = {1: 0, 2: 1, 3: 2, 6: 0, 7: 0}
    accumulator = _WindowAccumulator.empty()
    grouping = ["subjectID", "sessionID", "taskID"]
    for keys, trial_frame in selected.groupby(grouping, sort=True, dropna=False):
        subject, session_value, task_value = cast(tuple[Any, Any, Any], keys)
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
        trial = f"{session}:task-{int(task_value):03d}"
        physical_runs = _contiguous_signal_runs(
            timestamps,
            total,
            gyro,
            nominal_rate_hz=60.0,
            maximum_gap_factor=maximum_gap_factor,
        )
        run_index = 0
        for physical_start, physical_stop in physical_runs:
            physical_total = total[physical_start:physical_stop]
            physical_gravity = causal_gravity_lowpass(
                physical_total,
                sampling_rate_hz=60.0,
                cutoff_hz=gravity_cutoff_hz,
            )
            physical_linear = physical_total - physical_gravity
            physical_timestamps = timestamps[physical_start:physical_stop]
            physical_activities = np.where(
                np.isfinite(activities[physical_start:physical_stop]),
                activities[physical_start:physical_stop],
                -1_000_000,
            ).astype(np.int64)
            label_runs = _contiguous_label_runs(
                physical_timestamps,
                physical_activities,
                nominal_rate_hz=60.0,
                maximum_gap_factor=maximum_gap_factor,
            )
            for label_start, label_stop in label_runs:
                activity = int(physical_activities[label_start])
                if activity in label_map:
                    accumulator.add_uniform_trial(
                        total_acceleration=physical_linear[label_start:label_stop],
                        gyroscope=gyro[physical_start + label_start : physical_start + label_stop],
                        gravity=physical_gravity[label_start:label_stop],
                        source_rate_hz=60.0,
                        target_rate_hz=target_rate_hz,
                        gravity_cutoff_hz=gravity_cutoff_hz,
                        label=label_map[activity],
                        participant=participant,
                        session=session,
                        trial=trial,
                        run_index=run_index,
                        window_samples=window_samples,
                    )
                run_index += 1
    return accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=[receipt],
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=gravity_cutoff_hz,
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
                if source is not None and specification[0] not in incomplete_participants
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
                "action": "participant excluded before windowing to retain complete core support",
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
) -> ExternalHARWindows:
    """Stream the IMU-HAR-IL Body-WT core without storing third-party raw files."""

    dataset_id = "imu_har_il_v1"
    files, exclusions = _imu_har_il_inventory(
        participant_limit=participant_limit,
        repetition_limit=repetition_limit,
        activities=("Walk", "Sit", "Stand"),
        workers=discovery_workers,
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
                source_labels = frame[label_column].to_numpy(dtype=np.float64)
                if source_labels.size == 0 or not np.isfinite(source_labels).all():
                    invalid_trials.setdefault(source.participant, []).append(
                        {
                            "member": source.filename,
                            "reason": "provider activity-label column is empty or non-finite",
                            "label_column": label_column,
                        }
                    )
                elif set(np.unique(source_labels).tolist()) != {
                    provider_label_map[source.activity]
                }:
                    invalid_trials.setdefault(source.participant, []).append(
                        {
                            "member": source.filename,
                            "reason": "provider activity-label column conflicts with folder label",
                            "label_column": label_column,
                            "observed_labels": sorted(np.unique(source_labels).tolist()),
                            "expected_label": provider_label_map[source.activity],
                        }
                    )
                elif label_column != "Activity_label":
                    issues.append(
                        {
                            "member": source.filename,
                            "issue": "activity-label column uses the documented schema variant",
                            "observed_column": label_column,
                            "canonical_column": "Activity_label",
                            "action": "provider numeric label was validated against the activity folder",
                        }
                    )
            else:
                issues.append(
                    {
                        "member": source.filename,
                        "issue": "provider activity-label column absent",
                        "action": (
                            "trial retained using the provider activity-folder label; all six "
                            "required sensor channels were present"
                        ),
                    }
                )
            signal_values = frame[signal_columns].to_numpy(dtype=np.float64)
            finite = np.isfinite(signal_values).all(axis=1)
            complete_finite_segment = any(
                resample_uniform(
                    signal_values[start:stop],
                    source_rate_hz=60.0,
                    target_rate_hz=target_rate_hz,
                ).shape[0]
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
    data_quality_exclusions = tuple(
        {
            "participant_id": f"imuharil:{participant}",
            "reason": "one or more requested core trials failed the pre-window data-quality gate",
            "action": "participant excluded before any of their trials were windowed",
            "invalid_trials": invalid_trials[participant],
        }
        for participant in sorted(invalid_participants)
    )
    for source, frame in downloaded:
        if source.participant in invalid_participants:
            continue
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
        )
    result = accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=receipts,
        gravity_source="causal_lowpass_from_total_acceleration",
        gravity_cutoff_hz=gravity_cutoff_hz,
    )
    return ExternalHARWindows(
        dataset_id=result.dataset_id,
        channel_lane=result.channel_lane,
        sampling_rate_hz=result.sampling_rate_hz,
        signals=result.signals,
        gravity=result.gravity,
        labels=result.labels,
        participant_ids=result.participant_ids,
        session_ids=result.session_ids,
        trial_ids=result.trial_ids,
        window_ids=result.window_ids,
        receipts=result.receipts,
        exclusions=tuple((*exclusions, *data_quality_exclusions)),
        source_issues=tuple(issues),
        gravity_source=result.gravity_source,
        gravity_cutoff_hz=result.gravity_cutoff_hz,
    )


def _irregular_segments(
    timestamps: FloatArray, *, maximum_gap_factor: float
) -> list[tuple[int, int]]:
    finite = np.isfinite(timestamps)
    results: list[tuple[int, int]] = []
    for start, stop in _true_runs(finite):
        values = timestamps[start:stop]
        if values.size < 3:
            continue
        positive = np.diff(values)
        nominal_values = positive[positive > 0.0]
        if nominal_values.size == 0:
            continue
        nominal = float(np.median(nominal_values))
        boundary = np.ones(values.size, dtype=np.bool_)
        boundary[1:] = (positive <= 0.0) | (positive > maximum_gap_factor * nominal)
        starts = np.flatnonzero(boundary)
        stops = np.concatenate((starts[1:], np.array([values.size], dtype=np.int64)))
        results.extend(
            (start + int(left), start + int(right))
            for left, right in zip(starts, stops, strict=True)
        )
    return results


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
    with RemoteZip(url) as archive:
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
            payload = cast(bytes, archive.read(member))
            receipts.append(
                _receipt(
                    dataset_id=dataset_id,
                    locator=url,
                    member=member,
                    payload=payload,
                    declared_size=int(info.file_size),
                    archive_crc32=int(info.CRC),
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
            for run_index, (start, stop) in enumerate(
                _irregular_segments(timestamps, maximum_gap_factor=maximum_gap_factor)
            ):
                finite = np.isfinite(combined[start:stop]).all(axis=1)
                for finite_index, (left, right) in enumerate(_true_runs(finite)):
                    left += start
                    right += start
                    if right - left < 3:
                        continue
                    interpolated = _interpolate_to_rate(
                        timestamps[left:right], combined[left:right], target_rate_hz=target_rate_hz
                    )
                    usable = interpolated.shape[0] // window_samples * window_samples
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
        exclusions=(),
        source_issues=(),
        class_names=tuple(display_name[name] for name in activities),
        gravity_source="provider_native_gravity_vector",
        gravity_cutoff_hz=None,
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


def load_sole_harmony(
    *,
    participant_limit: int = 12,
    sessions_per_participant: int = 2,
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    maximum_gap_factor: float = 3.0,
) -> ExternalHARWindows:
    """Stream ordered Sole-HARmony sessions for temporal development.

    Camera annotation intervals are the bout boundaries.  Windows are non-overlapping and
    retained only when wholly contained in one annotated interval.  No filtering or
    interpolation crosses a camera, session, or participant boundary.
    """

    if not 1 <= participant_limit <= 13:
        raise ValueError("Sole-HARmony participant limit must lie in [1,13]")
    if not 1 <= sessions_per_participant <= 5:
        raise ValueError("Sole-HARmony session limit must lie in [1,5]")
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
    accumulator = _WindowAccumulator.empty()
    receipts: list[SourceReceipt] = []
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
                if np.any(np.diff(timestamps) < 0.0):
                    raise ValueError(
                        f"Sole-HARmony timestamps regress within a session: {mat_name}"
                    )
                labels = labels[np.argsort(labels[:, 1], kind="stable")]
                gravity = raw - linear
                participant = f"sole:{participant_code}"
                session_value = str(metadata.get("session_id", PurePosixPath(mat_name).parent.name))
                session = f"{participant}:session-{session_value}"
                previous_stop = float("-inf")
                for interval_index, row in enumerate(labels):
                    source_label = int(row[0])
                    start_time = float(row[1])
                    stop_time = float(row[2])
                    if not np.isfinite([start_time, stop_time]).all() or stop_time <= start_time:
                        raise ValueError(f"invalid camera interval in {mat_name}")
                    if start_time < previous_stop:
                        raise ValueError(f"overlapping camera intervals in {mat_name}")
                    previous_stop = stop_time
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
                        )
                del payload, timestamps, linear, gyroscope, raw, gravity, labels
    return accumulator.finish(
        dataset_id=dataset_id,
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        receipts=receipts,
        gravity_source="provider_raw_acceleration_minus_linear_acceleration",
        gravity_cutoff_hz=None,
    )
