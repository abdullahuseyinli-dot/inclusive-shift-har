"""Verified loader and audit helpers for the processed UCI-HAR v1 release.

The official dataset 240 release contains already-windowed inertial signals.  It
does not expose experiment/trial identifiers or raw-sample indices, so the
stable identifiers created here identify *released window instances*, not
recoverable raw samples.  The official test split is permanently marked as
legacy/development-consumed and cannot be opened through the ordinary source
development path.
"""

from __future__ import annotations

import hashlib
import io
import math
import os
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

UCI_HAR_DATASET_ID = "uci_har_v1"
UCI_HAR_ARCHIVE_MEMBER_ROOT = "UCI HAR Dataset"
UCI_HAR_WINDOW_LENGTH = 128
UCI_HAR_SAMPLE_RATE_HZ = 50
UCI_HAR_RELEASED_STRIDE = 64
UCI_HAR_AUDIT_ALGORITHM_VERSION = "1.0.0"

UCI_HAR_CHANNELS: tuple[str, ...] = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
)

UCI_HAR_ACTIVITY_NAMES: dict[int, str] = {
    1: "WALKING",
    2: "WALKING_UPSTAIRS",
    3: "WALKING_DOWNSTAIRS",
    4: "SITTING",
    5: "STANDING",
    6: "LAYING",
}

# These are exact UCI-native semantic names.  Cross-source eligibility is
# deliberately separate because InclusiveHAR's released Walking label includes
# manual wheelchair propulsion.
UCI_NATIVE_CORE_CANONICAL_NAMES: dict[int, str] = {
    1: "walking",
    4: "sitting",
    5: "standing",
}
UCI_NATIVE_CORE_CLASS_IDS: dict[int, int] = {1: 0, 4: 1, 5: 2}
UCI_NATIVE_CORE_CLASS_NAMES: tuple[str, ...] = ("walking", "sitting", "standing")

CROSS_SOURCE_ELIGIBILITY: dict[int, dict[str, str]] = {
    1: {
        "canonical_name": "walking",
        "status": "excluded_all_cohort_exact",
        "permitted_track": "ambulatory_walking_sensitivity_only_after_eligibility_lock",
        "reason": "InclusiveHAR Walking includes manual wheelchair propulsion",
    },
    4: {
        "canonical_name": "sitting",
        "status": "exact",
        "permitted_track": "cross_source_core",
        "reason": "locked ontology exact class",
    },
    5: {
        "canonical_name": "standing",
        "status": "provisional",
        "permitted_track": "blocked_pending_realization_documentation",
        "reason": "standing realization is not documented for every target participant",
    },
}

_SPLITS = frozenset({"train", "test"})
_MAX_TEXT_MEMBER_BYTES = 80 * 1024 * 1024


class UCIHARDataError(RuntimeError):
    """Raised when UCI-HAR provenance or released arrays fail validation."""


class UCIHAREvidencePurpose(StrEnum):
    """Allowed reasons for opening a released split."""

    SOURCE_DEVELOPMENT = "source_development"
    LEGACY_CONSUMED_AUDIT = "legacy_exploratory_development_consumed_audit"


@dataclass(frozen=True)
class UCIHARWindows:
    """One validated released UCI-HAR split in ``[window, time, channel]`` form."""

    signals: NDArray[np.float32]
    activity_ids: NDArray[np.int64]
    subject_ids: NDArray[np.int64]
    window_ids: tuple[str, ...]
    released_split: Literal["train", "test"]
    archive_sha256: str
    evidence_purpose: UCIHAREvidencePurpose

    @property
    def activity_names(self) -> tuple[str, ...]:
        return tuple(UCI_HAR_ACTIVITY_NAMES[int(value)] for value in self.activity_ids)

    @property
    def sample_ids(self) -> tuple[str, ...]:
        """Stable dataset-instance IDs; each instance is a released window."""

        return self.window_ids

    @property
    def sample_id_semantics(self) -> str:
        return "released_window_instance_not_recoverable_raw_sample"


@dataclass(frozen=True)
class UCINativeCoreWindows:
    """UCI-native walking/sitting/standing subset with fixed canonical IDs."""

    signals: NDArray[np.float32]
    canonical_activity_ids: NDArray[np.int64]
    canonical_activity_names: tuple[str, ...]
    source_activity_ids: NDArray[np.int64]
    subject_ids: NDArray[np.int64]
    window_ids: tuple[str, ...]
    released_split: Literal["train", "test"]


def stable_uci_window_id(split: Literal["train", "test"], row_index_one_based: int) -> str:
    """Return the stable identity of one row in a released window matrix."""

    if split not in _SPLITS:
        raise ValueError(f"unsupported released split: {split!r}")
    if (
        isinstance(row_index_one_based, bool)
        or not isinstance(row_index_one_based, int)
        or row_index_one_based <= 0
    ):
        raise ValueError("row_index_one_based must be a positive integer")
    return f"{UCI_HAR_DATASET_ID}:{split}:window:{row_index_one_based:06d}"


def _safe_archive(path: str | os.PathLike[str]) -> Path:
    source = Path(path)
    if source.is_symlink():
        raise UCIHARDataError(f"refusing to read archive through a symbolic link: {source}")
    try:
        source = source.resolve(strict=True)
    except OSError as exc:
        raise UCIHARDataError(f"UCI-HAR archive does not exist: {source}") from exc
    if not source.is_file():
        raise UCIHARDataError(f"UCI-HAR archive is not a regular file: {source}")
    return source


def _is_zip_symlink(info: zipfile.ZipInfo) -> bool:
    unix_mode = (info.external_attr >> 16) & 0xFFFF
    return stat.S_IFMT(unix_mode) == stat.S_IFLNK


def _validate_member_table(archive: zipfile.ZipFile) -> None:
    exact_names: set[str] = set()
    casefold_names: set[str] = set()
    for info in archive.infolist():
        name = info.filename
        if not name or "\\" in name or "\x00" in name:
            raise UCIHARDataError(f"unsafe ZIP member name: {name!r}")
        pure = PurePosixPath(name)
        if pure.is_absolute() or ".." in pure.parts or ":" in pure.parts[0]:
            raise UCIHARDataError(f"unsafe ZIP member path: {name!r}")
        if name in exact_names or name.casefold() in casefold_names:
            raise UCIHARDataError(f"duplicate or case-colliding ZIP member: {name!r}")
        if _is_zip_symlink(info):
            raise UCIHARDataError(f"symbolic-link ZIP member is forbidden: {name!r}")
        exact_names.add(name)
        casefold_names.add(name.casefold())


def _read_member(archive: zipfile.ZipFile, name: str) -> bytes:
    try:
        info = archive.getinfo(name)
    except KeyError as exc:
        raise UCIHARDataError(f"required UCI-HAR member is missing: {name}") from exc
    if info.is_dir() or info.file_size <= 0:
        raise UCIHARDataError(f"required UCI-HAR member is not a non-empty file: {name}")
    if info.file_size > _MAX_TEXT_MEMBER_BYTES:
        raise UCIHARDataError(f"required UCI-HAR member exceeds safety limit: {name}")
    payload = archive.read(info)
    if len(payload) != info.file_size:
        raise UCIHARDataError(f"short read for UCI-HAR member: {name}")
    return payload


def _load_float_matrix(archive: zipfile.ZipFile, name: str) -> NDArray[np.float32]:
    payload = _read_member(archive, name)
    try:
        parsed = np.loadtxt(io.BytesIO(payload), dtype=np.float32, ndmin=2)
    except (UnicodeError, ValueError) as exc:
        raise UCIHARDataError(f"cannot parse floating-point matrix {name}: {exc}") from exc
    matrix = np.asarray(parsed, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != UCI_HAR_WINDOW_LENGTH:
        raise UCIHARDataError(
            f"{name} must have shape [N,{UCI_HAR_WINDOW_LENGTH}], observed {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise UCIHARDataError(f"non-finite value in UCI-HAR matrix: {name}")
    return matrix


def _load_int_vector(archive: zipfile.ZipFile, name: str) -> NDArray[np.int64]:
    payload = _read_member(archive, name)
    try:
        parsed = np.loadtxt(io.BytesIO(payload), dtype=np.int64, ndmin=1)
    except (UnicodeError, ValueError) as exc:
        raise UCIHARDataError(f"cannot parse integer vector {name}: {exc}") from exc
    vector = np.asarray(parsed, dtype=np.int64).reshape(-1)
    return vector


def _validate_activity_table(archive: zipfile.ZipFile) -> None:
    name = f"{UCI_HAR_ARCHIVE_MEMBER_ROOT}/activity_labels.txt"
    try:
        text = _read_member(archive, name).decode("ascii")
    except UnicodeDecodeError as exc:
        raise UCIHARDataError("activity label table is not ASCII") from exc
    observed: dict[int, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        fields = raw_line.split()
        if len(fields) != 2:
            raise UCIHARDataError(f"malformed activity label line {line_number}")
        try:
            label_id = int(fields[0])
        except ValueError as exc:
            raise UCIHARDataError(f"invalid activity ID on line {line_number}") from exc
        if label_id in observed:
            raise UCIHARDataError(f"duplicate activity ID: {label_id}")
        observed[label_id] = fields[1]
    if observed != UCI_HAR_ACTIVITY_NAMES:
        raise UCIHARDataError(
            f"released activity ontology changed: expected {UCI_HAR_ACTIVITY_NAMES}, observed {observed}"
        )


def load_uci_har_split(
    archive_path: str | os.PathLike[str],
    *,
    split: Literal["train", "test"],
    purpose: UCIHAREvidencePurpose = UCIHAREvidencePurpose.SOURCE_DEVELOPMENT,
    expected_sha256: str | None = None,
) -> UCIHARWindows:
    """Load one released split after enforcing hash, schema, and evidence policy.

    ``test`` is refused unless ``purpose`` explicitly records that the official
    test is already legacy/development-consumed.  It must never be used as a new
    confirmatory endpoint.
    """

    if split not in _SPLITS:
        raise UCIHARDataError(f"unsupported released split: {split!r}")
    if split == "test" and purpose is not UCIHAREvidencePurpose.LEGACY_CONSUMED_AUDIT:
        raise UCIHARDataError(
            "official UCI-HAR test is development-consumed; explicit legacy audit purpose required"
        )
    source = _safe_archive(archive_path)
    archive_sha256 = sha256_file(source)
    if expected_sha256 is not None and archive_sha256 != expected_sha256.casefold():
        raise UCIHARDataError(
            f"archive SHA-256 mismatch: expected {expected_sha256.casefold()}, observed {archive_sha256}"
        )

    try:
        archive_context = zipfile.ZipFile(source)
    except zipfile.BadZipFile as exc:
        raise UCIHARDataError(f"invalid UCI-HAR ZIP archive: {source}") from exc

    with archive_context as archive:
        _validate_member_table(archive)
        _validate_activity_table(archive)
        prefix = f"{UCI_HAR_ARCHIVE_MEMBER_ROOT}/{split}"
        signal_matrices = [
            _load_float_matrix(
                archive,
                f"{prefix}/Inertial Signals/{channel}_{split}.txt",
            )
            for channel in UCI_HAR_CHANNELS
        ]
        activity_ids = _load_int_vector(archive, f"{prefix}/y_{split}.txt")
        subject_ids = _load_int_vector(archive, f"{prefix}/subject_{split}.txt")

    first_shape = signal_matrices[0].shape
    if any(matrix.shape != first_shape for matrix in signal_matrices[1:]):
        shapes = [matrix.shape for matrix in signal_matrices]
        raise UCIHARDataError(f"primary inertial signal shapes disagree: {shapes}")
    window_count = first_shape[0]
    if activity_ids.shape != (window_count,) or subject_ids.shape != (window_count,):
        raise UCIHARDataError(
            "signal, activity, and subject row counts disagree: "
            f"signals={window_count}, activities={activity_ids.shape}, subjects={subject_ids.shape}"
        )
    unknown_labels = sorted(
        set(int(value) for value in activity_ids) - UCI_HAR_ACTIVITY_NAMES.keys()
    )
    if unknown_labels:
        raise UCIHARDataError(f"unknown UCI-HAR activity IDs: {unknown_labels}")
    if window_count == 0 or np.any(subject_ids <= 0):
        raise UCIHARDataError("UCI-HAR split has no windows or contains invalid subject IDs")

    signals = np.stack(signal_matrices, axis=-1).astype(np.float32, copy=False)
    if signals.shape != (window_count, UCI_HAR_WINDOW_LENGTH, len(UCI_HAR_CHANNELS)):
        raise UCIHARDataError(f"unexpected stacked signal shape: {signals.shape}")
    signals.setflags(write=False)
    activity_ids.setflags(write=False)
    subject_ids.setflags(write=False)
    window_ids = tuple(stable_uci_window_id(split, index) for index in range(1, window_count + 1))
    return UCIHARWindows(
        signals=signals,
        activity_ids=activity_ids,
        subject_ids=subject_ids,
        window_ids=window_ids,
        released_split=split,
        archive_sha256=archive_sha256,
        evidence_purpose=purpose,
    )


def select_uci_native_core(windows: UCIHARWindows) -> UCINativeCoreWindows:
    """Select UCI-native walking/sitting/standing without changing target eligibility."""

    keep = np.isin(windows.activity_ids, tuple(UCI_NATIVE_CORE_CLASS_IDS))
    indices = np.flatnonzero(keep)
    selected_source_ids = np.asarray(windows.activity_ids[indices], dtype=np.int64)
    canonical_ids = np.asarray(
        [UCI_NATIVE_CORE_CLASS_IDS[int(value)] for value in selected_source_ids],
        dtype=np.int64,
    )
    selected_signals = np.asarray(windows.signals[indices], dtype=np.float32)
    selected_subjects = np.asarray(windows.subject_ids[indices], dtype=np.int64)
    selected_signals.setflags(write=False)
    selected_source_ids.setflags(write=False)
    canonical_ids.setflags(write=False)
    selected_subjects.setflags(write=False)
    return UCINativeCoreWindows(
        signals=selected_signals,
        canonical_activity_ids=canonical_ids,
        canonical_activity_names=tuple(
            UCI_NATIVE_CORE_CANONICAL_NAMES[int(value)] for value in selected_source_ids
        ),
        source_activity_ids=selected_source_ids,
        subject_ids=selected_subjects,
        window_ids=tuple(windows.window_ids[int(index)] for index in indices),
        released_split=windows.released_split,
    )


def _label_counts(values: NDArray[np.int64]) -> dict[str, int]:
    counts = Counter(int(value) for value in values)
    return {
        UCI_HAR_ACTIVITY_NAMES[label_id]: counts.get(label_id, 0)
        for label_id in sorted(UCI_HAR_ACTIVITY_NAMES)
    }


def _channel_profiles(signals: NDArray[np.float32]) -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for channel_index, channel in enumerate(UCI_HAR_CHANNELS):
        values = signals[:, :, channel_index]
        profiles.append(
            {
                "channel": channel,
                "finite_count": int(np.isfinite(values).sum()),
                "maximum": float(np.max(values)),
                "mean": float(np.mean(values, dtype=np.float64)),
                "minimum": float(np.min(values)),
                "non_finite_count": int((~np.isfinite(values)).sum()),
                "sample_standard_deviation": float(np.std(values, dtype=np.float64, ddof=1)),
            }
        )
    return profiles


def audit_uci_har_archive(
    archive_path: str | os.PathLike[str],
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Return a machine-readable, read-only audit of the processed v1 archive."""

    source = _safe_archive(archive_path)
    archive_sha256 = sha256_file(source)
    if expected_sha256 is not None and archive_sha256 != expected_sha256.casefold():
        raise UCIHARDataError(
            f"archive SHA-256 mismatch: expected {expected_sha256.casefold()}, observed {archive_sha256}"
        )
    with zipfile.ZipFile(source) as archive:
        _validate_member_table(archive)
        bad_crc_member = archive.testzip()
        members = archive.infolist()
        non_directory_members = [info for info in members if not info.is_dir()]
        required_metadata = (
            f"{UCI_HAR_ARCHIVE_MEMBER_ROOT}/README.txt",
            f"{UCI_HAR_ARCHIVE_MEMBER_ROOT}/activity_labels.txt",
        )
        metadata_records = [
            {
                "member": name,
                "sha256": hashlib.sha256(_read_member(archive, name)).hexdigest(),
                "size_bytes": archive.getinfo(name).file_size,
            }
            for name in required_metadata
        ]

    train = load_uci_har_split(
        source,
        split="train",
        purpose=UCIHAREvidencePurpose.SOURCE_DEVELOPMENT,
        expected_sha256=archive_sha256,
    )
    test = load_uci_har_split(
        source,
        split="test",
        purpose=UCIHAREvidencePurpose.LEGACY_CONSUMED_AUDIT,
        expected_sha256=archive_sha256,
    )
    train_subjects = sorted(set(int(value) for value in train.subject_ids))
    test_subjects = sorted(set(int(value) for value in test.subject_ids))
    overlap = sorted(set(train_subjects) & set(test_subjects))
    train_core = select_uci_native_core(train)
    test_core = select_uci_native_core(test)
    payload: dict[str, Any] = {
        "archive": {
            "bad_crc_member": bad_crc_member,
            "file_member_count": len(non_directory_members),
            "member_count_including_directories": len(members),
            "metadata_members": metadata_records,
            "sha256": archive_sha256,
            "size_bytes": source.stat().st_size,
            "zip_crc_status": "pass" if bad_crc_member is None else "fail",
        },
        "audit_algorithm_version": UCI_HAR_AUDIT_ALGORITHM_VERSION,
        "audit_kind": "processed_uci_har_v1_read_only",
        "dataset_id": UCI_HAR_DATASET_ID,
        "evidence_status": "source_development_and_legacy_consumed_test_audit",
        "label_ontology": {
            "cross_source_eligibility": CROSS_SOURCE_ELIGIBILITY,
            "released_activity_names": UCI_HAR_ACTIVITY_NAMES,
            "uci_native_core": UCI_NATIVE_CORE_CANONICAL_NAMES,
        },
        "limitations": {
            "official_test_status": "legacy_exploratory_development_consumed",
            "raw_sample_index_status": "absent_from_processed_v1_windows",
            "released_window_overlap": "50_percent_reported_by_provider",
            "trial_identifier_status": "absent_from_processed_v1_windows",
        },
        "primary_channels": list(UCI_HAR_CHANNELS),
        "released_split_audit": {
            "subject_overlap": overlap,
            "subject_overlap_status": "pass" if not overlap else "fail",
            "test": {
                "channel_profiles": _channel_profiles(test.signals),
                "label_counts": _label_counts(test.activity_ids),
                "native_core_window_count": len(test_core.window_ids),
                "shape": list(test.signals.shape),
                "subject_count": len(test_subjects),
                "subject_ids": test_subjects,
                "window_count": len(test.window_ids),
                "window_id_count": len(set(test.window_ids)),
            },
            "train": {
                "channel_profiles": _channel_profiles(train.signals),
                "label_counts": _label_counts(train.activity_ids),
                "native_core_window_count": len(train_core.window_ids),
                "shape": list(train.signals.shape),
                "subject_count": len(train_subjects),
                "subject_ids": train_subjects,
                "window_count": len(train.window_ids),
                "window_id_count": len(set(train.window_ids)),
            },
        },
        "source_development_gate": {
            "official_test_allowed_for_model_selection": False,
            "status": "pass" if not overlap and bad_crc_member is None else "fail",
            "training_scope": "official_train_subjects_only",
        },
        "window_definition": {
            "duration_seconds_at_reported_rate": UCI_HAR_WINDOW_LENGTH / UCI_HAR_SAMPLE_RATE_HZ,
            "instance_id_semantics": "released_window_instance_not_recoverable_raw_sample",
            "reported_sample_rate_hz": UCI_HAR_SAMPLE_RATE_HZ,
            "released_stride_samples": UCI_HAR_RELEASED_STRIDE,
            "shape": [UCI_HAR_WINDOW_LENGTH, len(UCI_HAR_CHANNELS)],
        },
    }
    if not math.isclose(payload["window_definition"]["duration_seconds_at_reported_rate"], 2.56):
        raise AssertionError("UCI-HAR duration invariant failed")
    payload["report_sha256"] = canonical_json_sha256(payload)
    return payload
