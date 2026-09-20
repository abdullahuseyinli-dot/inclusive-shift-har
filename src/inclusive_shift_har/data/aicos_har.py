"""Audited local-ZIP adapter for the AICOS-HAR external benchmark.

The provider archive is retained as one immutable raw object.  This module never
extracts it.  Accelerometer and gyroscope members are read acquisition by
acquisition, aligned on their overlapping physical interval, and windowed only
inside the provider acquisition boundary.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import (
    BOUNDARY_PROVENANCE_PROTOCOL,
    CORE_CLASS_INDEX,
    CORE_CLASS_NAMES,
    ExternalHARWindows,
    ObservableWindowPool,
    SourceReceipt,
    causal_gravity_lowpass,
)
from inclusive_shift_har.data.external_har import (
    STANDARD_GRAVITY_M_S2 as STANDARD_GRAVITY_M_S2,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan

FloatArray = NDArray[np.float64]

AICOS_ARCHIVE_SIZE_BYTES = 2_539_861_502
AICOS_ARCHIVE_MD5 = "faa895e7b203d664525b0a1e74542793"
AICOS_CONTENT_URL = "https://zenodo.org/api/records/19452049/files/AICOS-HAR.zip/content"
AICOS_ROOT = "AICOS-HAR"
AICOS_SIGNAL_UNITS = {"linear_acceleration": "m/s^2", "gyroscope": "rad/s", "gravity": "m/s^2"}
INCLUSIVEHAR_SIGNAL_UNITS = {"linear_acceleration": "g", "gyroscope": "rad/s", "gravity": "g"}

_ACTIVITY_MAP = {
    "Walking": "mobility",
    "Running": "mobility",
    "Upstairs": "mobility",
    "Downstairs": "mobility",
    "RampUp": "mobility",
    "RampDown": "mobility",
    "Sitting": "sitting",
    "Standing": "standing",
}
_PHONE_PATTERN = re.compile(r"^W\d+_")


@dataclass(frozen=True, slots=True)
class AICOSArchiveVerification:
    """One-pass verification of the preserved provider archive."""

    archive_path: str
    size_bytes: int
    md5: str
    sha256: str
    expected_size_verified: bool
    provider_md5_verified: bool

    def source_receipt(self) -> SourceReceipt:
        return SourceReceipt(
            dataset_id="aicos_har_v1",
            locator=AICOS_CONTENT_URL,
            member=None,
            declared_size_bytes=AICOS_ARCHIVE_SIZE_BYTES,
            received_size_bytes=self.size_bytes,
            computed_sha256=self.sha256,
            declared_digest_algorithm="md5",
            declared_digest=AICOS_ARCHIVE_MD5,
            computed_declared_digest=self.md5,
            declared_digest_verified=self.provider_md5_verified,
            raw_local_mirror=True,
        )


@dataclass(frozen=True, slots=True)
class _AcquisitionAudit:
    acquisition_id: str
    participant_id: str
    provider_fold: str
    activity: str
    device_position: str
    device: str
    position: str
    status: str
    acceleration_unit_rule: str | None
    acceleration_scale_to_m_s2: float | None
    acceleration_norm_median: float | None
    acceleration_rate_hz: float | None
    gyroscope_rate_hz: float | None
    overlap_seconds: float | None
    source_acceleration_rows: int | None
    source_gyroscope_rows: int | None
    reason: str | None


def verify_aicos_archive(path: Path) -> AICOSArchiveVerification:
    """Verify provider size and MD5 and retain an independent SHA-256."""

    resolved = path.resolve()
    size = resolved.stat().st_size
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with resolved.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            md5.update(block)
            sha256.update(block)
    result = AICOSArchiveVerification(
        archive_path=str(resolved),
        size_bytes=size,
        md5=md5.hexdigest(),
        sha256=sha256.hexdigest(),
        expected_size_verified=size == AICOS_ARCHIVE_SIZE_BYTES,
        provider_md5_verified=md5.hexdigest() == AICOS_ARCHIVE_MD5,
    )
    if not result.expected_size_verified or not result.provider_md5_verified:
        raise ValueError("AICOS-HAR archive size or provider MD5 does not match Zenodo v1")
    return result


def _read_metadata(archive: ZipFile) -> pd.DataFrame:
    payload = archive.read(f"{AICOS_ROOT}/metadata.csv")
    frame = pd.read_csv(io.BytesIO(payload), dtype=str, keep_default_na=False)
    expected = ["AcquisitionID", "Age", "Gender", "Position_FullName", "Fold"]
    if frame.columns.tolist() != expected or frame.empty:
        raise ValueError("AICOS-HAR metadata schema differs from the qualified release")
    if frame["AcquisitionID"].duplicated().any():
        raise ValueError("AICOS-HAR metadata contains duplicate acquisition identifiers")
    return frame


def _identity(acquisition_id: str) -> tuple[str, str, str, str, str]:
    parts = PurePosixPath(acquisition_id).parts
    if len(parts) != 3:
        raise ValueError("AICOS-HAR acquisition identifier must have three path components")
    participant, activity_trial, device_position = parts
    if "_" not in activity_trial or "_" not in device_position:
        raise ValueError("AICOS-HAR acquisition identifier lacks trial/device-position fields")
    activity = activity_trial.rsplit("_", 1)[0]
    device, position = device_position.rsplit("_", 1)
    return participant, activity, device_position, device, position


def _eligible_metadata(
    frame: pd.DataFrame,
    member_names: set[str],
    *,
    folds: frozenset[str],
    participants: frozenset[str] | None,
) -> tuple[list[dict[str, str]], list[dict[str, Any]], tuple[str, ...]]:
    selected: list[dict[str, str]] = []
    exclusions: list[dict[str, Any]] = []
    prewindow_roster: set[str] = set()
    for row in frame.to_dict(orient="records"):
        acquisition_id = str(row["AcquisitionID"])
        participant, activity, device_position, device, position = _identity(acquisition_id)
        fold = str(row["Fold"])
        if fold not in folds or (participants is not None and participant not in participants):
            continue
        if activity not in _ACTIVITY_MAP:
            exclusions.append(
                {"acquisition_id": acquisition_id, "reason": "outside_three_class_endpoint"}
            )
            continue
        if _PHONE_PATTERN.match(device_position):
            exclusions.append({"acquisition_id": acquisition_id, "reason": "wearable_excluded"})
            continue
        acceleration_member = f"{AICOS_ROOT}/{acquisition_id}/Accelerometer.txt"
        gyroscope_member = f"{AICOS_ROOT}/{acquisition_id}/Gyroscope.txt"
        if acceleration_member not in member_names or gyroscope_member not in member_names:
            exclusions.append(
                {"acquisition_id": acquisition_id, "reason": "missing_accelerometer_or_gyroscope"}
            )
            continue
        prewindow_roster.add(f"aicos:{participant}")
        selected.append(
            {
                "acquisition_id": acquisition_id,
                "participant": participant,
                "fold": fold,
                "activity": activity,
                "device_position": device_position,
                "device": device,
                "position": position,
                "acceleration_member": acceleration_member,
                "gyroscope_member": gyroscope_member,
            }
        )
    return selected, exclusions, tuple(sorted(prewindow_roster))


def _sensor(payload: bytes, *, name: str) -> FloatArray:
    try:
        values = np.loadtxt(io.BytesIO(payload), delimiter=",", dtype=np.float64)
    except ValueError as exc:
        raise ValueError(f"cannot parse AICOS-HAR {name}") from exc
    if values.ndim == 1:
        values = values[np.newaxis, :]
    if values.ndim != 2 or values.shape[1] != 4 or values.shape[0] < 3:
        raise ValueError(f"AICOS-HAR {name} must be a non-empty timestamp+x+y+z matrix")
    if not np.isfinite(values).all() or np.any(np.diff(values[:, 0]) <= 0.0):
        raise ValueError(f"AICOS-HAR {name} has nonfinite or nonmonotonic rows")
    return values


def _rate(values: FloatArray) -> float:
    median_delta_seconds = float(np.median(np.diff(values[:, 0])) * 1e-9)
    if not 0.002 <= median_delta_seconds <= 0.2:
        raise ValueError("AICOS-HAR sensor rate lies outside the qualified 5--500 Hz range")
    return 1.0 / median_delta_seconds


def _unit_rule(acceleration: FloatArray) -> tuple[str, float, float]:
    median_norm = float(np.median(np.linalg.norm(acceleration[:, 1:4], axis=1)))
    if 0.75 <= median_norm <= 1.25:
        return "physical_g_band_0.75_to_1.25", STANDARD_GRAVITY_M_S2, median_norm
    if 6.0 <= median_norm <= 16.0:
        return "physical_m_s2_band_6_to_16", 1.0, median_norm
    raise ValueError("acceleration median norm is outside both predeclared physical unit bands")


def _audit_acquisition(
    archive: ZipFile, row: dict[str, str]
) -> tuple[_AcquisitionAudit, FloatArray | None, FloatArray | None]:
    try:
        acceleration = _sensor(archive.read(row["acceleration_member"]), name="accelerometer")
        gyroscope = _sensor(archive.read(row["gyroscope_member"]), name="gyroscope")
        acceleration_rate = _rate(acceleration)
        gyroscope_rate = _rate(gyroscope)
        rule, scale, norm = _unit_rule(acceleration)
        acceleration_seconds = acceleration[:, 0] * 1e-9
        gyroscope_seconds = gyroscope[:, 0] * 1e-9
        overlap = min(acceleration_seconds[-1], gyroscope_seconds[-1]) - max(
            acceleration_seconds[0], gyroscope_seconds[0]
        )
        if overlap < 128 / 50.0:
            raise ValueError("overlapping sensor interval is shorter than one target window")
    except (KeyError, OSError, ValueError) as exc:
        return (
            _AcquisitionAudit(
                acquisition_id=row["acquisition_id"],
                participant_id=row["participant"],
                provider_fold=row["fold"],
                activity=row["activity"],
                device_position=row["device_position"],
                device=row["device"],
                position=row["position"],
                status="quarantined",
                acceleration_unit_rule=None,
                acceleration_scale_to_m_s2=None,
                acceleration_norm_median=None,
                acceleration_rate_hz=None,
                gyroscope_rate_hz=None,
                overlap_seconds=None,
                source_acceleration_rows=None,
                source_gyroscope_rows=None,
                reason=str(exc),
            ),
            None,
            None,
        )
    return (
        _AcquisitionAudit(
            acquisition_id=row["acquisition_id"],
            participant_id=row["participant"],
            provider_fold=row["fold"],
            activity=row["activity"],
            device_position=row["device_position"],
            device=row["device"],
            position=row["position"],
            status="qualified",
            acceleration_unit_rule=rule,
            acceleration_scale_to_m_s2=scale,
            acceleration_norm_median=norm,
            acceleration_rate_hz=acceleration_rate,
            gyroscope_rate_hz=gyroscope_rate,
            overlap_seconds=float(overlap),
            source_acceleration_rows=int(acceleration.shape[0]),
            source_gyroscope_rows=int(gyroscope.shape[0]),
            reason=None,
        ),
        acceleration,
        gyroscope,
    )


def audit_aicos_archive(
    archive_path: Path,
    *,
    folds: Iterable[str] = ("1", "2", "3", "4", "5"),
    participants: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Audit units, rates and sensor eligibility without calculating class scores."""

    selected_folds = frozenset(str(item) for item in folds)
    selected_participants = (
        None if participants is None else frozenset(str(item) for item in participants)
    )
    with ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        rows, exclusions, roster = _eligible_metadata(
            _read_metadata(archive),
            names,
            folds=selected_folds,
            participants=selected_participants,
        )
        audits = [_audit_acquisition(archive, row)[0] for row in rows]
    counts = Counter(item.status for item in audits)
    unit_counts = Counter(
        item.acceleration_unit_rule for item in audits if item.acceleration_unit_rule is not None
    )
    return {
        "schema_version": "1.0.0",
        "record_kind": "aicos_har_signal_qualification",
        "folds": sorted(selected_folds),
        "class_scores_calculated": False,
        "participant_roster": list(roster),
        "eligible_metadata_acquisition_count": len(rows),
        "metadata_exclusion_count": len(exclusions),
        "status_counts": dict(sorted(counts.items())),
        "unit_rule_counts": {str(key): value for key, value in sorted(unit_counts.items())},
        "unit_policy": {
            "g_median_norm_inclusive": [0.75, 1.25],
            "m_s2_median_norm_inclusive": [6.0, 16.0],
            "ambiguous_policy": "quarantine; never infer from activity scores",
        },
        "acquisitions": [asdict(item) for item in audits],
        "metadata_exclusions": exclusions,
    }


def _aligned_target_grid(
    acceleration: FloatArray,
    gyroscope: FloatArray,
    *,
    acceleration_scale: float,
    target_rate_hz: float,
    gravity_cutoff_hz: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    acc_time = acceleration[:, 0] * 1e-9
    gyro_time = gyroscope[:, 0] * 1e-9
    start = max(float(acc_time[0]), float(gyro_time[0]))
    stop = min(float(acc_time[-1]), float(gyro_time[-1]))
    count = int(np.floor((stop - start) * target_rate_hz)) + 1
    if count < 3:
        raise ValueError("AICOS-HAR aligned interval is empty")
    grid = start + np.arange(count, dtype=np.float64) / target_rate_hz
    grid = grid[grid <= stop]
    total = (
        np.column_stack([np.interp(grid, acc_time, acceleration[:, axis]) for axis in range(1, 4)])
        * acceleration_scale
    )
    gyro = np.column_stack([np.interp(grid, gyro_time, gyroscope[:, axis]) for axis in range(1, 4)])
    gravity = causal_gravity_lowpass(
        total, sampling_rate_hz=target_rate_hz, cutoff_hz=gravity_cutoff_hz
    )
    linear = total - gravity
    return np.asarray(linear), np.asarray(gyro), np.asarray(gravity)


def load_aicos_har(
    archive_path: Path,
    *,
    verification: AICOSArchiveVerification,
    folds: Iterable[str] = ("test",),
    participants: Iterable[str] | None = None,
    target_rate_hz: float = 50.0,
    window_samples: int = 128,
    gravity_cutoff_hz: float = 0.30,
) -> ExternalHARWindows:
    """Materialize the qualified AICOS phone accelerometer+gyroscope lane."""

    if Path(verification.archive_path).resolve() != archive_path.resolve():
        raise ValueError("AICOS-HAR verification is bound to a different archive")
    verification.source_receipt().validate()
    selected_folds = frozenset(str(item) for item in folds)
    selected_participants = (
        None if participants is None else frozenset(str(item) for item in participants)
    )
    signals: list[NDArray[np.float32]] = []
    gravity_windows: list[NDArray[np.float32]] = []
    labels: list[int] = []
    participant_ids: list[str] = []
    session_ids: list[str] = []
    trial_ids: list[str] = []
    window_ids: list[str] = []
    preprocessing: list[dict[str, Any]] = []
    with ZipFile(archive_path) as archive:
        rows, exclusions, roster = _eligible_metadata(
            _read_metadata(archive),
            set(archive.namelist()),
            folds=selected_folds,
            participants=selected_participants,
        )
        for row in rows:
            audit, acceleration, gyroscope = _audit_acquisition(archive, row)
            if acceleration is None or gyroscope is None:
                exclusions.append(
                    {
                        "acquisition_id": row["acquisition_id"],
                        "reason": "signal_qualification_quarantine",
                        "detail": audit.reason,
                    }
                )
                preprocessing.append(asdict(audit))
                continue
            if audit.acceleration_scale_to_m_s2 is None:
                raise AssertionError("qualified AICOS acquisition lacks an acceleration scale")
            linear, gyro, gravity = _aligned_target_grid(
                acceleration,
                gyroscope,
                acceleration_scale=audit.acceleration_scale_to_m_s2,
                target_rate_hz=target_rate_hz,
                gravity_cutoff_hz=gravity_cutoff_hz,
            )
            count = linear.shape[0] // window_samples
            if count == 0:
                exclusions.append(
                    {"acquisition_id": row["acquisition_id"], "reason": "no_complete_window"}
                )
                preprocessing.append(asdict(audit))
                continue
            participant = f"aicos:{row['participant']}"
            session = f"aicos-device:{row['device_position']}"
            trial = f"aicos-acquisition:{row['acquisition_id']}"
            class_index = CORE_CLASS_INDEX[_ACTIVITY_MAP[row["activity"]]]
            for index in range(count):
                start = index * window_samples
                stop = start + window_samples
                signals.append(
                    np.asarray(
                        np.column_stack((linear[start:stop], gyro[start:stop])), dtype=np.float32
                    )
                )
                gravity_windows.append(np.asarray(gravity[start:stop], dtype=np.float32))
                labels.append(class_index)
                participant_ids.append(participant)
                session_ids.append(session)
                trial_ids.append(trial)
                window_ids.append(f"aicos/{row['acquisition_id']}/window-{index:06d}")
            preprocessing.append(
                {
                    **asdict(audit),
                    "target_rate_hz": target_rate_hz,
                    "window_samples": window_samples,
                    "window_count": count,
                    "dropped_tail_samples": int(linear.shape[0] - count * window_samples),
                    "alignment": "linear interpolation on overlapping 50 Hz grid",
                    "gravity_derivation": "causal first-order low-pass after unit harmonization",
                }
            )
    if not signals:
        raise ValueError("AICOS-HAR selection produced no qualified complete windows")
    signal_array = np.stack(signals)
    gravity_array = np.stack(gravity_windows)
    participant_array = np.asarray(participant_ids, dtype=np.str_)
    session_array = np.asarray(session_ids, dtype=np.str_)
    trial_array = np.asarray(trial_ids, dtype=np.str_)
    window_array = np.asarray(window_ids, dtype=np.str_)
    label_array = np.asarray(labels, dtype=np.int64)
    observed_participants = np.unique(participant_array)
    class_support: dict[str, list[str]] = defaultdict(list)
    for participant in observed_participants:
        present = set(label_array[participant_array == participant].tolist())
        for index, name in enumerate(CORE_CLASS_NAMES):
            if index in present:
                class_support[name].append(str(participant))
    complete = [
        str(participant)
        for participant in observed_participants
        if set(label_array[participant_array == participant].tolist()) == {0, 1, 2}
    ]
    plan = (
        build_participant_partition_plan(
            "aicos_har_v1",
            roster,
            roster_basis="provider-fold phone accelerometer+gyroscope roster before unit audit/windowing",
        )
        if len(roster) >= 5
        else None
    )
    pool = ObservableWindowPool(
        signals=signal_array,
        gravity=gravity_array,
        participant_ids=participant_array,
        session_ids=session_array,
        trial_ids=trial_array,
        window_ids=window_array,
    )
    result = ExternalHARWindows(
        dataset_id="aicos_har_v1",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=target_rate_hz,
        signals=signal_array,
        gravity=gravity_array,
        labels=label_array,
        participant_ids=participant_array,
        session_ids=session_array,
        trial_ids=trial_array,
        window_ids=window_array,
        receipts=(verification.source_receipt(),),
        exclusions=tuple(exclusions),
        source_issues=(
            {
                "issue": "provider README declares m/s^2 but released accelerometer magnitudes are heterogeneous",
                "resolution": "fixed physical-band unit audit with ambiguous acquisitions quarantined",
            },
            {
                "issue": "metadata names lifts while archive folders use ElevatorUp/ElevatorDown",
                "resolution": "both are outside the frozen three-class endpoint",
            },
        ),
        gravity_source="causal-derived-from-unit-qualified-total-acceleration",
        gravity_cutoff_hz=gravity_cutoff_hz,
        preprocessing_audit=tuple(preprocessing),
        cohort_audit={
            "signal_units": dict(AICOS_SIGNAL_UNITS),
            "provider_folds": sorted(selected_folds),
            "prewindow_phone_sensor_roster": list(roster),
            "observed_participants": observed_participants.tolist(),
            "complete_three_class_participants": complete,
            "class_participant_support": dict(class_support),
            "qualified_acquisition_count": sum(
                item.get("status") == "qualified" and "window_count" in item
                for item in preprocessing
            ),
            "quarantined_acquisition_count": sum(
                item.get("status") == "quarantined" for item in preprocessing
            ),
        },
        observable_candidates=pool,
        source_storage_audit={
            "archive_path": verification.archive_path,
            "archive_size_bytes": verification.size_bytes,
            "archive_md5": verification.md5,
            "archive_sha256": verification.sha256,
            "archive_extracted": False,
        },
        boundary_provenance={
            "protocol_id": BOUNDARY_PROVENANCE_PROTOCOL,
            "source_boundary_unit": "provider activity acquisition folder",
            "repository_signal_grid_annotation_independent": True,
            "provider_upstream_annotation_conditioned": True,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": "offline zero-shot evaluation within provider acquisition boundaries",
        },
        participant_partition_plan=plan,
    )
    result.validate()
    return result


def aicos_in_inclusivehar_units(data: ExternalHARWindows) -> ExternalHARWindows:
    """Convert the qualified SI adapter output to the frozen source's g contract.

    This is a fixed dimensional conversion, not a fitted normalization. It does
    not address coordinate conventions or native-versus-derived gravity. Reject
    missing units and double application; do not infer scale from target labels.
    """
    audit = data.cohort_audit or {}
    if data.dataset_id != "aicos_har_v1" or audit.get("signal_units") != AICOS_SIGNAL_UNITS:
        raise ValueError("AICOS-to-InclusiveHAR conversion requires declared SI AICOS input")

    def converted_signals(values: NDArray[np.float32]) -> NDArray[np.float32]:
        result = values.copy()
        result[:, :, :3] /= STANDARD_GRAVITY_M_S2
        return result

    pool = data.observable_candidates
    converted_pool = (
        None
        if pool is None
        else replace(
            pool,
            signals=converted_signals(pool.signals),
            gravity=np.asarray(pool.gravity / STANDARD_GRAVITY_M_S2, dtype=np.float32),
        )
    )
    result = replace(
        data,
        signals=converted_signals(data.signals),
        gravity=np.asarray(data.gravity / STANDARD_GRAVITY_M_S2, dtype=np.float32),
        observable_candidates=converted_pool,
        cohort_audit={
            **audit,
            "signal_units": dict(INCLUSIVEHAR_SIGNAL_UNITS),
            "model_interface_unit_conversion": {
                "input_units": dict(AICOS_SIGNAL_UNITS),
                "output_units": dict(INCLUSIVEHAR_SIGNAL_UNITS),
                "acceleration_and_gravity_multiplier": 1.0 / STANDARD_GRAVITY_M_S2,
                "gyroscope_multiplier": 1.0,
                "fitted": False,
                "axis_or_polarity_conversion": False,
            },
        },
    )
    result.validate(require_all_classes=False)
    return result
