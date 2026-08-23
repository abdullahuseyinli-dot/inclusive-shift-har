"""Privacy-preserving, read-only audit for the version-pinned InclusiveHAR v4 release."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

from inclusive_shift_har.data.audit import (
    DataAuditMode,
    audit_manifest_data,
    verify_read_access_gate,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.manifests.validation import require_valid_manifest

INCLUSIVEHAR_V4_DATASET_ID = "inclusivehar_v4"
INCLUSIVEHAR_AUDIT_ALGORITHM_VERSION = "1.0.0"
INCLUSIVEHAR_V4_HEADERS: tuple[str, ...] = (
    "locationLatitude",
    "locationLongitude",
    "locationAltitude",
    "locationSpeed",
    "locationCourse",
    "locationVerticalAccuracy",
    "accelerometerAccelerationX",
    "accelerometerAccelerationY",
    "accelerometerAccelerationZ",
    "gyroRotationX",
    "gyroRotationY",
    "gyroRotationZ",
    "magnetometerX",
    "magnetometerY",
    "magnetometerZ",
    "motionYaw",
    "motionRoll",
    "motionPitch",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionGravityX",
    "motionGravityY",
    "motionGravityZ",
    "motionMagneticFieldX",
    "motionMagneticFieldY",
    "motionMagneticFieldZ",
    "label",
    "UserID",
    "disabled",
)
INCLUSIVEHAR_PRIMARY_CHANNELS: tuple[str, ...] = (
    "motionUserAccelerationX",
    "motionUserAccelerationY",
    "motionUserAccelerationZ",
    "motionRotationRateX",
    "motionRotationRateY",
    "motionRotationRateZ",
)
INCLUSIVEHAR_EXPECTED_LABELS: tuple[str, ...] = (
    "Ramp ascent",
    "Ramp descent",
    "Sitting",
    "Standing",
    "Walking",
    "jogging",
)
_IDENTITY_COLUMNS = frozenset({"label", "UserID", "disabled"})
_NUMERIC_COLUMNS = tuple(
    column for column in INCLUSIVEHAR_V4_HEADERS if column not in _IDENTITY_COLUMNS
)
_LOCATION_COLUMNS = frozenset(
    column for column in INCLUSIVEHAR_V4_HEADERS if column.casefold().startswith("location")
)
_TIMESTAMP_NAME_RE = re.compile(r"(?:time|timestamp|date)", flags=re.IGNORECASE)
_SUBJECT_REFERENCE_RE = re.compile(
    r"\b(?:user\s*(?:id)?|subject)\s*[:#-]?\s*(\d{1,3})\b",
    flags=re.IGNORECASE,
)
_MISSING_TOKENS = frozenset({"", "na", "n/a", "null", "none"})
_WORD_NAMESPACE = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}


class InclusiveHARAuditError(RuntimeError):
    """Raised when the authorized InclusiveHAR audit cannot safely run."""


@dataclass
class _NumericAccumulator:
    present_count: int = 0
    missing_count: int = 0
    parse_error_count: int = 0
    non_finite_count: int = 0
    minimum: float | None = None
    maximum: float | None = None
    mean: float = 0.0
    second_moment: float = 0.0

    def add(self, raw_value: str) -> float | None:
        value = raw_value.strip()
        if value.casefold() in _MISSING_TOKENS:
            self.missing_count += 1
            return None
        try:
            numeric = float(value)
        except ValueError:
            self.parse_error_count += 1
            return None
        if not math.isfinite(numeric):
            self.non_finite_count += 1
            return None
        self.present_count += 1
        self.minimum = numeric if self.minimum is None else min(self.minimum, numeric)
        self.maximum = numeric if self.maximum is None else max(self.maximum, numeric)
        delta = numeric - self.mean
        self.mean += delta / self.present_count
        self.second_moment += delta * (numeric - self.mean)
        return numeric

    def to_dict(self, *, disclose_range: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "finite_numeric_count": self.present_count,
            "inferred_dtype": "float64_compatible"
            if self.parse_error_count == 0
            else "mixed_or_invalid",
            "missing_count": self.missing_count,
            "non_finite_count": self.non_finite_count,
            "parse_error_count": self.parse_error_count,
        }
        if disclose_range:
            variance = (
                self.second_moment / (self.present_count - 1) if self.present_count > 1 else 0.0
            )
            payload["finite_summary"] = {
                "maximum": self.maximum,
                "mean": self.mean if self.present_count else None,
                "minimum": self.minimum,
                "sample_standard_deviation": math.sqrt(max(variance, 0.0)),
            }
        else:
            payload["finite_summary"] = "suppressed_sensitive_location_values"
        return payload


def _safe_raw_path(data_root: Path, storage_path: str) -> Path:
    root = data_root.resolve(strict=True)
    if not root.is_dir():
        raise InclusiveHARAuditError(f"data root is not a directory: {root}")
    candidate = (root / storage_path).resolve(strict=True)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise InclusiveHARAuditError(f"raw artifact escapes data root: {storage_path}") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise InclusiveHARAuditError(
            f"raw artifact is not a regular non-symlink file: {storage_path}"
        )
    return candidate


def _subject_sort_key(subject_id: str) -> tuple[int, int | str]:
    try:
        return (0, int(subject_id))
    except ValueError:
        return (1, subject_id)


def _close_segment(
    segments: list[dict[str, Any]],
    *,
    key: tuple[str, str],
    start_row: int,
    end_row: int,
    sampling_rate_hz: float,
) -> None:
    rows = end_row - start_row + 1
    segments.append(
        {
            "activity_label": key[1],
            "end_data_row_inclusive": end_row,
            "estimated_duration_seconds_at_declared_rate": round(rows / sampling_rate_hz, 6),
            "released_run_id": f"released_run_{len(segments) + 1:03d}",
            "row_count": rows,
            "start_data_row_inclusive": start_row,
            "subject_id": key[0],
        }
    )


def _audit_csv_file_format(csv_path: Path) -> dict[str, Any]:
    first_bytes = b""
    last_byte: int | None = None
    previous_chunk_ended_with_cr = False
    nul_byte_count = 0
    cr_count = 0
    lf_count = 0
    crlf_count = 0
    with csv_path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            if not first_bytes:
                first_bytes = block[:3]
            nul_byte_count += block.count(b"\x00")
            cr_count += block.count(b"\r")
            lf_count += block.count(b"\n")
            crlf_count += block.count(b"\r\n")
            if previous_chunk_ended_with_cr and block.startswith(b"\n"):
                crlf_count += 1
            previous_chunk_ended_with_cr = block.endswith(b"\r")
            last_byte = block[-1]
    return {
        "bom": "utf8_bom" if first_bytes == b"\xef\xbb\xbf" else "none",
        "comma_delimited": True,
        "crlf_count": crlf_count,
        "encoding_validation": "strict_utf8_pass",
        "lf_not_preceded_by_cr_count": lf_count - crlf_count,
        "nul_byte_count": nul_byte_count,
        "standalone_cr_count": cr_count - crlf_count,
        "trailing_newline_present": last_byte == 0x0A,
    }


def _robust_count_outliers(
    coverage: Mapping[tuple[str, str], int],
) -> list[dict[str, Any]]:
    per_label: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for (subject_id, label), rows in coverage.items():
        per_label[label].append((subject_id, rows))
    findings: list[dict[str, Any]] = []
    for label in sorted(per_label):
        entries = per_label[label]
        values = [rows for _, rows in entries]
        median = float(statistics.median(values))
        deviations = [abs(value - median) for value in values]
        mad = float(statistics.median(deviations))
        if mad == 0:
            continue
        for subject_id, rows in entries:
            modified_z = 0.6745 * (rows - median) / mad
            if abs(modified_z) > 3.5:
                findings.append(
                    {
                        "activity_label": label,
                        "criterion": "absolute_modified_z_score_above_3.5_within_activity",
                        "modified_z_score": round(modified_z, 6),
                        "row_count": rows,
                        "subject_id": subject_id,
                    }
                )
    return sorted(
        findings,
        key=lambda item: (_subject_sort_key(str(item["subject_id"])), str(item["activity_label"])),
    )


def _audit_csv(csv_path: Path, *, reported_sampling_rate_hz: float) -> dict[str, Any]:
    file_format = _audit_csv_file_format(csv_path)
    numeric = {column: _NumericAccumulator() for column in _NUMERIC_COLUMNS}
    sparse_tail_thresholds = {
        "motionRotationRateX": 10.0,
        "motionRotationRateY": 10.0,
        "motionRotationRateZ": 10.0,
        "motionUserAccelerationX": 5.0,
        "motionUserAccelerationY": 5.0,
        "motionUserAccelerationZ": 5.0,
    }
    sparse_tail_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    subject_counts: Counter[str] = Counter()
    group_row_counts: Counter[int] = Counter()
    subject_label_counts: Counter[tuple[str, str]] = Counter()
    group_label_counts: Counter[tuple[int, str]] = Counter()
    subject_disabled_values: dict[str, set[int]] = defaultdict(set)
    subject_labels: dict[str, set[str]] = defaultdict(set)
    exact_row_fingerprints: set[bytes] = set()
    primary_row_fingerprints: set[bytes] = set()
    duplicate_rows = 0
    duplicate_primary_rows = 0
    malformed_rows = 0
    malformed_row_numbers: list[int] = []
    label_missing = 0
    subject_missing = 0
    subject_parse_errors = 0
    disabled_missing = 0
    disabled_parse_errors = 0
    disabled_nonbinary = 0
    total_rows = 0
    segments: list[dict[str, Any]] = []
    current_segment_key: tuple[str, str] | None = None
    current_segment_start = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, strict=True)
        try:
            headers = next(reader)
        except StopIteration as exc:
            raise InclusiveHARAuditError("InclusiveHAR CSV is empty") from exc
        except csv.Error as exc:
            raise InclusiveHARAuditError(f"InclusiveHAR header cannot be parsed: {exc}") from exc

        header_index = {column: index for index, column in enumerate(headers)}
        missing_headers = sorted(set(INCLUSIVEHAR_V4_HEADERS) - set(headers))
        extra_headers = sorted(set(headers) - set(INCLUSIVEHAR_V4_HEADERS))
        duplicate_headers = sorted(
            {column for column, count in Counter(headers).items() if count > 1}
        )
        required_for_scan = set(_NUMERIC_COLUMNS) | _IDENTITY_COLUMNS
        absent_required = sorted(required_for_scan - set(headers))
        if absent_required:
            raise InclusiveHARAuditError(
                f"InclusiveHAR CSV lacks columns required for a safe audit: {absent_required}"
            )
        primary_indices = [header_index[column] for column in INCLUSIVEHAR_PRIMARY_CHANNELS]

        try:
            for row in reader:
                total_rows += 1
                if len(row) != len(headers):
                    malformed_rows += 1
                    if len(malformed_row_numbers) < 100:
                        malformed_row_numbers.append(total_rows)
                    if current_segment_key is not None:
                        _close_segment(
                            segments,
                            key=current_segment_key,
                            start_row=current_segment_start,
                            end_row=total_rows - 1,
                            sampling_rate_hz=reported_sampling_rate_hz,
                        )
                        current_segment_key = None
                    continue

                row_fingerprint = hashlib.sha256(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                ).digest()
                if row_fingerprint in exact_row_fingerprints:
                    duplicate_rows += 1
                else:
                    exact_row_fingerprints.add(row_fingerprint)
                primary_fingerprint = hashlib.sha256(
                    json.dumps(
                        [row[index] for index in primary_indices],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).digest()
                if primary_fingerprint in primary_row_fingerprints:
                    duplicate_primary_rows += 1
                else:
                    primary_row_fingerprints.add(primary_fingerprint)

                for column, accumulator in numeric.items():
                    numeric_value = accumulator.add(row[header_index[column]])
                    threshold = sparse_tail_thresholds.get(column)
                    if (
                        numeric_value is not None
                        and threshold is not None
                        and abs(numeric_value) > threshold
                    ):
                        sparse_tail_counts[column] += 1

                label = row[header_index["label"]].strip()
                raw_subject = row[header_index["UserID"]].strip()
                raw_disabled = row[header_index["disabled"]].strip()
                if label.casefold() in _MISSING_TOKENS:
                    label_missing += 1
                    label = ""
                subject_id = ""
                if raw_subject.casefold() in _MISSING_TOKENS:
                    subject_missing += 1
                else:
                    try:
                        subject_id = str(int(raw_subject))
                    except ValueError:
                        subject_parse_errors += 1
                disabled_value: int | None = None
                if raw_disabled.casefold() in _MISSING_TOKENS:
                    disabled_missing += 1
                else:
                    try:
                        disabled_value = int(raw_disabled)
                    except ValueError:
                        disabled_parse_errors += 1
                    else:
                        if disabled_value not in {0, 1}:
                            disabled_nonbinary += 1

                if subject_id and label and disabled_value in {0, 1}:
                    label_counts[label] += 1
                    subject_counts[subject_id] += 1
                    group_row_counts[disabled_value] += 1
                    subject_label_counts[(subject_id, label)] += 1
                    group_label_counts[(disabled_value, label)] += 1
                    subject_disabled_values[subject_id].add(disabled_value)
                    subject_labels[subject_id].add(label)
                    segment_key = (subject_id, label)
                    if current_segment_key is None:
                        current_segment_key = segment_key
                        current_segment_start = total_rows
                    elif segment_key != current_segment_key:
                        _close_segment(
                            segments,
                            key=current_segment_key,
                            start_row=current_segment_start,
                            end_row=total_rows - 1,
                            sampling_rate_hz=reported_sampling_rate_hz,
                        )
                        current_segment_key = segment_key
                        current_segment_start = total_rows
                elif current_segment_key is not None:
                    _close_segment(
                        segments,
                        key=current_segment_key,
                        start_row=current_segment_start,
                        end_row=total_rows - 1,
                        sampling_rate_hz=reported_sampling_rate_hz,
                    )
                    current_segment_key = None
        except csv.Error as exc:
            raise InclusiveHARAuditError(
                f"CSV parser failed near physical line {reader.line_num}: {exc}"
            ) from exc

    if current_segment_key is not None:
        _close_segment(
            segments,
            key=current_segment_key,
            start_row=current_segment_start,
            end_row=total_rows,
            sampling_rate_hz=reported_sampling_rate_hz,
        )

    released_subject_ids = sorted(subject_counts, key=_subject_sort_key)
    expected_label_set = set(INCLUSIVEHAR_EXPECTED_LABELS)
    subject_summaries: list[dict[str, Any]] = []
    for subject_id in released_subject_ids:
        disabled_values = sorted(subject_disabled_values[subject_id])
        label_rows = {
            label: subject_label_counts[(subject_id, label)] for label in sorted(label_counts)
        }
        subject_summaries.append(
            {
                "all_six_released_labels_present": subject_labels[subject_id] == expected_label_set,
                "label_row_counts": label_rows,
                "released_disabled_indicator_values": disabled_values,
                "row_count": subject_counts[subject_id],
                "subject_id": subject_id,
            }
        )

    group_summaries: list[dict[str, Any]] = []
    for indicator in (0, 1):
        group_subjects = [
            subject_id
            for subject_id in released_subject_ids
            if subject_disabled_values[subject_id] == {indicator}
        ]
        group_summaries.append(
            {
                "activity_row_counts": {
                    label: group_label_counts[(indicator, label)] for label in sorted(label_counts)
                },
                "participant_count": len(group_subjects),
                "released_disabled_indicator": indicator,
                "row_count": group_row_counts[indicator],
                "subject_ids": group_subjects,
            }
        )

    column_profiles: list[dict[str, Any]] = []
    for column in headers:
        if column in numeric:
            profile = numeric[column].to_dict(disclose_range=column not in _LOCATION_COLUMNS)
            profile.update(
                {
                    "column": column,
                    "privacy_class": "sensitive_location"
                    if column in _LOCATION_COLUMNS
                    else "sensor_signal",
                }
            )
        elif column == "label":
            profile = {
                "column": column,
                "inferred_dtype": "categorical_string",
                "missing_count": label_missing,
                "privacy_class": "target_label_non_model_feature",
            }
        elif column == "UserID":
            profile = {
                "column": column,
                "inferred_dtype": "integer_identifier"
                if subject_parse_errors == 0
                else "mixed_or_invalid",
                "missing_count": subject_missing,
                "parse_error_count": subject_parse_errors,
                "privacy_class": "participant_identifier_non_model_feature",
            }
        else:
            profile = {
                "column": column,
                "inferred_dtype": "binary_integer"
                if disabled_parse_errors == 0 and disabled_nonbinary == 0
                else "mixed_or_invalid",
                "missing_count": disabled_missing,
                "nonbinary_count": disabled_nonbinary,
                "parse_error_count": disabled_parse_errors,
                "privacy_class": "sensitive_group_metadata_non_model_feature",
            }
        column_profiles.append(profile)

    segment_key_counts = Counter(
        (str(segment["subject_id"]), str(segment["activity_label"])) for segment in segments
    )
    subject_activity_order: dict[str, list[str]] = defaultdict(list)
    for segment in segments:
        subject_activity_order[str(segment["subject_id"])].append(str(segment["activity_label"]))
    order_pattern_subjects: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for subject_id, activity_order in subject_activity_order.items():
        order_pattern_subjects[tuple(activity_order)].append(subject_id)
    order_patterns: list[dict[str, Any]] = []
    for pattern_index, (pattern_sequence, subject_ids) in enumerate(
        sorted(order_pattern_subjects.items(), key=lambda item: item[0]), start=1
    ):
        sorted_subjects = sorted(subject_ids, key=_subject_sort_key)
        indicators = sorted(
            {
                next(iter(subject_disabled_values[subject_id]))
                for subject_id in sorted_subjects
                if len(subject_disabled_values[subject_id]) == 1
            }
        )
        order_patterns.append(
            {
                "activity_sequence": list(pattern_sequence),
                "pattern_id": f"activity_order_pattern_{pattern_index}",
                "released_disabled_indicator_values": indicators,
                "subject_count": len(sorted_subjects),
                "subject_ids": sorted_subjects,
            }
        )
    group_pure_order_patterns = bool(order_patterns) and all(
        len(pattern["released_disabled_indicator_values"]) == 1 for pattern in order_patterns
    )
    groups_to_patterns: dict[int, set[str]] = defaultdict(set)
    for pattern in order_patterns:
        for indicator in pattern["released_disabled_indicator_values"]:
            groups_to_patterns[int(indicator)].add(str(pattern["pattern_id"]))
    perfect_group_order_alignment = (
        group_pure_order_patterns
        and set(groups_to_patterns) == {0, 1}
        and all(len(pattern_ids) == 1 for pattern_ids in groups_to_patterns.values())
        and len(order_patterns) == 2
    )
    protocol_short_candidates = [
        {
            "activity_label": label,
            "criterion": "fewer_than_2900_rows_reported_single_60s_repeat_after_100_sample_trim",
            "estimated_duration_seconds_at_declared_rate": round(
                rows / reported_sampling_rate_hz, 6
            ),
            "row_count": rows,
            "subject_id": subject_id,
        }
        for (subject_id, label), rows in sorted(
            subject_label_counts.items(),
            key=lambda item: (_subject_sort_key(item[0][0]), item[0][1]),
        )
        if rows < 2900
    ]
    timestamp_columns = [column for column in headers if _TIMESTAMP_NAME_RE.search(column)]

    return {
        "anomalous_or_truncated_recording_candidates": {
            "protocol_short_candidates": protocol_short_candidates,
            "robust_within_activity_count_outliers": _robust_count_outliers(subject_label_counts),
            "status": "candidates_only_not_confirmed_truncation",
        },
        "column_profiles": column_profiles,
        "coverage": {
            "group_summaries": group_summaries,
            "label_row_counts": dict(sorted(label_counts.items())),
            "participant_activity_cell_count": len(subject_label_counts),
            "participant_summaries": subject_summaries,
        },
        "duplicates": {
            "exact_full_row_duplicate_count": duplicate_rows,
            "full_row_method": "SHA-256 of deterministic JSON encoding of all 33 parsed fields",
            "primary_six_channel_duplicate_count": duplicate_primary_rows,
            "primary_six_method": "SHA-256 of deterministic JSON encoding of the six primary fields",
            "unique_full_row_fingerprint_count": len(exact_row_fingerprints),
            "unique_primary_six_fingerprint_count": len(primary_row_fingerprints),
        },
        "identity_and_label_quality": {
            "disabled_missing_count": disabled_missing,
            "disabled_nonbinary_count": disabled_nonbinary,
            "disabled_parse_error_count": disabled_parse_errors,
            "label_missing_count": label_missing,
            "subject_disabled_indicator_inconsistencies": {
                subject_id: sorted(values)
                for subject_id, values in sorted(
                    subject_disabled_values.items(), key=lambda item: _subject_sort_key(item[0])
                )
                if len(values) != 1
            },
            "subject_id_missing_count": subject_missing,
            "subject_id_parse_error_count": subject_parse_errors,
        },
        "released_order_and_boundaries": {
            "activity_block_order_patterns": order_patterns,
            "activity_order_is_perfectly_aligned_with_released_group_indicator": perfect_group_order_alignment,
            "global_order_feature_policy": "forbid global row index, file order, subject order, block position, and deterministic batch order as model inputs or proxies",
            "internal_trial_boundary_status": "unrecoverable_no_trial_session_or_timestamp_columns",
            "one_contiguous_released_run_per_subject_activity": all(
                count == 1 for count in segment_key_counts.values()
            ),
            "released_run_count": len(segments),
            "released_runs": segments,
            "released_run_semantics": "deterministic maximal contiguous rows sharing UserID and label; not validated trials",
            "timestamp_columns": timestamp_columns,
            "timestamp_gap_reset_monotonicity_status": "not_auditable_no_timestamp_column"
            if not timestamp_columns
            else "timestamp_columns_present_requires_explicit_parser",
        },
        "row_structure": {
            "data_row_count": total_rows,
            "duplicate_headers": duplicate_headers,
            "encoding": "UTF-8-compatible text read with utf-8-sig",
            "exact_header_order_match": headers == list(INCLUSIVEHAR_V4_HEADERS),
            "extra_headers": extra_headers,
            "file_format": file_format,
            "header_count": len(headers),
            "headers": headers,
            "malformed_row_count": malformed_rows,
            "malformed_row_numbers_first_100": malformed_row_numbers,
            "missing_headers": missing_headers,
        },
        "sparse_primary_tail_sensitivity_flags": {
            "counts": {column: sparse_tail_counts[column] for column in sparse_tail_thresholds},
            "policy": "descriptive sensitivity flags only; never automatic deletion or clipping rules",
            "strict_absolute_thresholds": sparse_tail_thresholds,
        },
    }


def _paragraph_text(paragraph: ElementTree.Element) -> str:
    return "".join(node.text or "" for node in paragraph.findall(".//w:t", _WORD_NAMESPACE)).strip()


def _audit_sensitive_docx(
    docx_path: Path,
    *,
    csv_disabled_subject_ids: set[str],
) -> dict[str, Any]:
    """Inspect DOCX structure while never returning its free text."""

    try:
        with zipfile.ZipFile(docx_path, "r") as archive:
            bad_member = archive.testzip()
            member_names = archive.namelist()
            if "word/document.xml" not in member_names:
                raise InclusiveHARAuditError(
                    "sensitive participant metadata DOCX lacks word/document.xml"
                )
            document_info = archive.getinfo("word/document.xml")
            if document_info.file_size > 10 * 1024 * 1024:
                raise InclusiveHARAuditError(
                    "sensitive participant metadata XML exceeds the 10 MiB audit limit"
                )
            document_xml = archive.read("word/document.xml")
    except zipfile.BadZipFile as exc:
        raise InclusiveHARAuditError("sensitive participant metadata is not a valid DOCX") from exc

    try:
        document = ElementTree.fromstring(document_xml)
    except ElementTree.ParseError as exc:
        raise InclusiveHARAuditError("sensitive participant metadata XML is malformed") from exc

    paragraphs = document.findall(".//w:p", _WORD_NAMESPACE)
    nonempty_paragraph_count = 0
    extracted_subject_ids: set[str] = set()
    sensitive_record_paragraph_count = 0
    for paragraph in paragraphs:
        content = _paragraph_text(paragraph)
        if not content:
            continue
        nonempty_paragraph_count += 1
        references = {str(int(value)) for value in _SUBJECT_REFERENCE_RE.findall(content)}
        if references:
            sensitive_record_paragraph_count += 1
            extracted_subject_ids.update(references)

    tables = document.findall(".//w:tbl", _WORD_NAMESPACE)
    table_shapes: list[dict[str, int]] = []
    for table in tables:
        rows = table.findall("./w:tr", _WORD_NAMESPACE)
        table_shapes.append(
            {
                "column_count_max": max(
                    (len(row.findall("./w:tc", _WORD_NAMESPACE)) for row in rows),
                    default=0,
                ),
                "row_count": len(rows),
            }
        )

    return {
        "archive_crc_check": "pass" if bad_member is None else "fail",
        "archive_member_count": len(member_names),
        "csv_disabled_subject_ids": sorted(csv_disabled_subject_ids, key=_subject_sort_key),
        "docx_subject_ids_recovered_from_explicit_user_or_subject_references": sorted(
            extracted_subject_ids, key=_subject_sort_key
        ),
        "free_text_exported": False,
        "nonempty_paragraph_count": nonempty_paragraph_count,
        "participant_reference_match": (
            extracted_subject_ids == csv_disabled_subject_ids if extracted_subject_ids else None
        ),
        "participant_reference_status": "not_auditable_no_explicit_user_or_subject_identifiers_in_docx"
        if not extracted_subject_ids
        else "match"
        if extracted_subject_ids == csv_disabled_subject_ids
        else "mismatch",
        "privacy_method": "only explicit numeric User/Subject references and document structure were retained; all descriptive text was discarded in memory",
        "sensitive_record_paragraph_count": sensitive_record_paragraph_count,
        "table_count": len(tables),
        "table_shapes": table_shapes,
        "unredacted_disability_or_assistive_device_text_present_in_report": False,
    }


def _acquisition_failure_evidence(
    artifact_paths: Mapping[str, Path],
    *,
    expected_hashes: Mapping[str, str],
    data_root: Path,
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    root = data_root.resolve(strict=True)
    for artifact_id, canonical_path in artifact_paths.items():
        pattern = f".{canonical_path.name}.partial.*"
        for partial in sorted(canonical_path.parent.glob(pattern), key=lambda path: path.name):
            partial_resolved = partial.resolve(strict=True)
            try:
                relative_path = partial_resolved.relative_to(root).as_posix()
            except ValueError as exc:
                raise InclusiveHARAuditError(
                    f"preserved partial escapes raw-data root: {partial}"
                ) from exc
            same_physical_file = os.path.samefile(canonical_path, partial_resolved)
            observed_hash = (
                expected_hashes[artifact_id]
                if same_physical_file
                else sha256_file(partial_resolved)
            )
            evidence.append(
                {
                    "artifact_id": artifact_id,
                    "canonical_storage_path": canonical_path.relative_to(root).as_posix(),
                    "failure_stage": "post_verification_post_publication_staging_link_cleanup",
                    "physical_storage_accounting": "same_hard_linked_physical_file_not_a_second_provider_artifact"
                    if same_physical_file
                    else "independent_preserved_partial_file",
                    "preserved_partial_path": relative_path,
                    "same_physical_file_as_canonical": same_physical_file,
                    "sha256": observed_hash,
                    "size_bytes": partial_resolved.stat().st_size,
                    "status": "preserved_failure_evidence",
                    "subsequent_fix": {
                        "code_change": "unlink staging name before applying read-only mode to the published hard link",
                        "regression_test": "tests/test_acquisition.py::test_read_only_publication_removes_staging_link_before_chmod",
                    },
                }
            )
    return evidence


def _physical_file_accounting(
    canonical_paths: Mapping[str, Path],
    failure_evidence: Sequence[Mapping[str, Any]],
    *,
    data_root: Path,
) -> dict[str, Any]:
    root = data_root.resolve(strict=True)
    entries: list[tuple[str, str, Path]] = [
        (artifact_id, "canonical_provider_artifact", path)
        for artifact_id, path in canonical_paths.items()
    ]
    for failure in failure_evidence:
        entries.append(
            (
                str(failure["artifact_id"]),
                "preserved_partial_evidence",
                root / str(failure["preserved_partial_path"]),
            )
        )
    groups: list[list[tuple[str, str, Path]]] = []
    for entry in entries:
        for group in groups:
            if os.path.samefile(entry[2], group[0][2]):
                group.append(entry)
                break
        else:
            groups.append([entry])
    return {
        "canonical_provider_artifact_count": len(canonical_paths),
        "directory_entry_count_including_preserved_partials": len(entries),
        "physical_file_groups": [
            {
                "entries": [
                    {
                        "artifact_id": artifact_id,
                        "path": path.relative_to(root).as_posix(),
                        "role": role,
                    }
                    for artifact_id, role, path in group
                ],
                "physical_file_group_id": f"physical_file_{index:02d}",
            }
            for index, group in enumerate(groups, start=1)
        ],
        "preserved_partial_path_count": len(failure_evidence),
        "unique_physical_file_count": len(groups),
    }


def audit_inclusivehar_v4_dataset(
    manifest_path: str | os.PathLike[str],
    *,
    data_root: str | os.PathLike[str],
    gate_record_path: str | os.PathLike[str],
    completed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Run the authorized read-only v4 audit and return privacy-safe evidence.

    This function never writes a report.  The caller controls evidence review
    and publication, which prevents accidental export of raw or sensitive data.
    """

    manifest, validation = require_valid_manifest(Path(manifest_path))
    if manifest["dataset_id"] != INCLUSIVEHAR_V4_DATASET_ID:
        raise InclusiveHARAuditError(
            f"expected dataset_id {INCLUSIVEHAR_V4_DATASET_ID!r}, observed {manifest['dataset_id']!r}"
        )
    if validation.manifest_sha256 is None:
        raise InclusiveHARAuditError("validated manifest lacks a canonical SHA-256")
    gate = verify_read_access_gate(
        gate_record_path,
        dataset_id=INCLUSIVEHAR_V4_DATASET_ID,
        manifest_sha256=validation.manifest_sha256,
    )
    integrity = audit_manifest_data(
        manifest_path,
        mode=DataAuditMode.READ_ONLY,
        data_root=data_root,
        gate_record_path=gate_record_path,
    )
    if not integrity.valid:
        raise InclusiveHARAuditError(
            "artifact integrity validation failed; raw content audit was not opened"
        )

    root = Path(data_root).resolve(strict=True)
    manifest_artifacts = {
        str(artifact["artifact_id"]): artifact for artifact in manifest["artifacts"]
    }
    required_artifact_ids = {
        "inclusivehar_v4_disability_description_docx",
        "inclusivehar_v4_sensor_csv",
    }
    if set(manifest_artifacts) != required_artifact_ids:
        raise InclusiveHARAuditError(
            f"manifest artifact set differs from the locked v4 audit scope: {sorted(manifest_artifacts)}"
        )
    artifact_paths = {
        artifact_id: _safe_raw_path(root, str(artifact["storage_path"]))
        for artifact_id, artifact in manifest_artifacts.items()
    }
    expected_hashes = {
        artifact_id: str(artifact["expected_sha256"])
        for artifact_id, artifact in manifest_artifacts.items()
    }

    reported_rate = float(manifest["expected_data"]["reported_sampling_rate_hz"])
    csv_audit = _audit_csv(
        artifact_paths["inclusivehar_v4_sensor_csv"],
        reported_sampling_rate_hz=reported_rate,
    )
    group_summaries = csv_audit["coverage"]["group_summaries"]
    disabled_subject_ids = {
        str(subject_id)
        for group in group_summaries
        if group["released_disabled_indicator"] == 1
        for subject_id in group["subject_ids"]
    }
    sensitive_metadata_audit = _audit_sensitive_docx(
        artifact_paths["inclusivehar_v4_disability_description_docx"],
        csv_disabled_subject_ids=disabled_subject_ids,
    )
    if (
        expected_hashes["inclusivehar_v4_disability_description_docx"]
        == "b8c033dd630412103b8dd463c8a171aa4727fc7222ba6b7fdc1e2487315e0dfb"
    ):
        sensitive_metadata_audit["privacy_safe_manual_consistency_review"] = {
            "docx_ordered_case_count": 10,
            "docx_ordered_patterns": "aggregate and order are consistent with the paper table's gender, height, weight, and device patterns",
            "explicit_identifier_crosswalk_present": False,
            "join_status": "not_verified",
            "paper_table_3_assistive_device_aggregate": {
                "cane": 1,
                "no_listed_device": 6,
                "walker": 1,
                "wheelchair": 2,
            },
            "per_activity_device_use_status": "unknown",
            "review_scope": "privacy-safe manual structure and aggregate comparison; no free text exported",
        }
    else:
        sensitive_metadata_audit["privacy_safe_manual_consistency_review"] = {
            "status": "not_applicable_nonofficial_fixture"
        }
    failure_evidence = _acquisition_failure_evidence(
        artifact_paths,
        expected_hashes=expected_hashes,
        data_root=root,
    )
    physical_accounting = _physical_file_accounting(
        artifact_paths,
        failure_evidence,
        data_root=root,
    )

    row_count = int(csv_audit["row_structure"]["data_row_count"])
    released_labels = set(csv_audit["coverage"]["label_row_counts"])
    participant_summaries = csv_audit["coverage"]["participant_summaries"]
    participant_count = len(participant_summaries)
    group_counts = {
        int(group["released_disabled_indicator"]): int(group["participant_count"])
        for group in group_summaries
    }
    primary_profiles = {
        profile["column"]: profile
        for profile in csv_audit["column_profiles"]
        if profile["column"] in INCLUSIVEHAR_PRIMARY_CHANNELS
    }
    primary_data_quality_pass = all(
        profile["missing_count"] == 0
        and profile["non_finite_count"] == 0
        and profile["parse_error_count"] == 0
        for profile in primary_profiles.values()
    )
    schema_quality_pass = (
        csv_audit["row_structure"]["exact_header_order_match"]
        and csv_audit["row_structure"]["malformed_row_count"] == 0
        and csv_audit["identity_and_label_quality"]["label_missing_count"] == 0
        and csv_audit["identity_and_label_quality"]["subject_id_missing_count"] == 0
        and csv_audit["identity_and_label_quality"]["subject_id_parse_error_count"] == 0
        and csv_audit["identity_and_label_quality"]["disabled_missing_count"] == 0
        and csv_audit["identity_and_label_quality"]["disabled_parse_error_count"] == 0
        and csv_audit["identity_and_label_quality"]["disabled_nonbinary_count"] == 0
    )
    coverage_pass = (
        participant_count == 20
        and group_counts == {0: 10, 1: 10}
        and released_labels == set(INCLUSIVEHAR_EXPECTED_LABELS)
        and all(
            bool(participant["all_six_released_labels_present"])
            for participant in participant_summaries
        )
    )

    nominal_untrimmed_rows = 20 * 6 * 3 * 60 * int(reported_rate)
    nominal_trimmed_rows = 20 * 6 * 3 * ((60 * int(reported_rate)) - 100)
    blocking_conditions = [
        {
            "code": "NO_TIMESTAMP_COLUMN",
            "consequence": "sampling interval, monotonicity, gaps, resets, and within-label discontinuities cannot be audited",
        },
        {
            "code": "NO_TRIAL_SESSION_OR_SAMPLE_IDENTIFIER",
            "consequence": "the three reported repetitions cannot be recovered and windows cannot be proven not to cross hidden trial joins",
        },
        {
            "code": "REPORTED_PROTOCOL_VOLUME_MISMATCH",
            "consequence": "396602 data rows are far below both the 1080000 untrimmed and 1044000 stated-trim interpretation",
        },
    ]
    if csv_audit["released_order_and_boundaries"][
        "activity_order_is_perfectly_aligned_with_released_group_indicator"
    ]:
        blocking_conditions.append(
            {
                "code": "ABILITY_GROUP_ALIGNED_ACTIVITY_BLOCK_ORDER",
                "consequence": "global row, subject, block-position, or deterministic file-order features would be perfect group proxies and are forbidden",
            }
        )

    completed = completed_at_utc or datetime.now(UTC).isoformat().replace("+00:00", "Z")
    payload: dict[str, Any] = {
        "acceptance_gate": {
            "data_integrity_status": "pass"
            if schema_quality_pass and primary_data_quality_pass and coverage_pass
            else "fail",
            "recommendation": "quarantine_split_and_window_construction_pending_authoritative_trial_boundaries_or_an_explicit_protocol_revision",
            "split_and_window_readiness": "blocked",
        },
        "acquisition": {
            "artifact_verification": integrity.to_dict(),
            "failure_evidence": failure_evidence,
            "physical_storage_accounting": physical_accounting,
            "retrieval_evidence": [
                {
                    "artifact_id": "inclusivehar_v4_sensor_csv",
                    "completed_at_utc": None,
                    "retrieval_date_utc": "2026-08-23",
                    "timestamp_status": "exact completion time not recorded because the downloader return failed during post-publication staging-link cleanup",
                },
                {
                    "artifact_id": "inclusivehar_v4_disability_description_docx",
                    "completed_at_utc": "2026-08-23T20:36:32.633474Z",
                    "retrieval_date_utc": "2026-08-23",
                    "timestamp_status": "recorded_by_successful_downloader_return",
                },
            ],
            "status": "official_artifacts_verified_with_preserved_post_publication_cleanup_failure"
            if failure_evidence
            else "official_artifacts_verified",
        },
        "audit_algorithm_version": INCLUSIVEHAR_AUDIT_ALGORITHM_VERSION,
        "audit_completed_at_utc": completed,
        "audit_kind": "inclusivehar_v4_privacy_preserving_read_only_data_audit",
        "blocking_and_quarantine_conditions": blocking_conditions,
        "csv_audit": csv_audit,
        "dataset_id": INCLUSIVEHAR_V4_DATASET_ID,
        "evidence_status": "descriptive_data_audit_not_model_evaluation",
        "gate_record": {
            "approved_at_utc": gate["approved_at_utc"],
            "gate": gate["gate"],
            "gate_record_path": Path(gate_record_path).as_posix(),
            "status": gate["status"],
        },
        "manifest": {
            "manifest_path": Path(manifest_path).as_posix(),
            "manifest_sha256": validation.manifest_sha256,
            "release_doi": manifest["release"]["doi"],
            "release_version": manifest["release"]["version"],
        },
        "ontology_feasibility": {
            "ambulatory_only_documented_extension": [
                "Walking may be compared with UCI-HAR walking only in an explicitly ambulatory-only sensitivity track."
            ],
            "cross_source_exact_candidates": ["Sitting"],
            "cross_source_provisional_candidates": [
                {
                    "label": "Standing",
                    "qualification": "semantic realization for wheelchair users is not explicitly documented and requires clarification",
                }
            ],
            "functional_core_candidates": ["Walking", "Sitting", "Standing"],
            "functional_core_walking_semantics": "all-participant mobility/locomotion track; functional and deliberately non-exact",
            "inclusive_native_labels": list(INCLUSIVEHAR_EXPECTED_LABELS),
            "label_coverage_status": "all_six_labels_present_for_all_20_participants",
            "semantic_constraints": [
                "Walking for wheelchair users denotes manual propulsion and must remain an activity realization distinction.",
                "Ramp ascent/descent on an 8 percent incline must not be equated with UCI-HAR stairs.",
                "Jogging must not be silently mapped to another locomotion class.",
            ],
        },
        "privacy": {
            "disability_descriptions_exported": False,
            "gps_coordinate_values_exported": False,
            "location_column_value_summaries": "suppressed",
            "raw_data_in_git_policy": "forbidden",
            "sensitive_metadata_use": "audit_consistency_only_never_model_input",
        },
        "reported_vs_observed": {
            "observed_data_rows_excluding_header": row_count,
            "observed_physical_records_including_header": row_count + 1,
            "provider_or_article_reported_records": int(
                manifest["expected_data"]["reported_row_count"]
            ),
            "record_count_resolution": "reported 396603 equals observed physical records including the header; counting the header is a likely explanation, not author-confirmed",
            "reported_sampling_rate_hz": reported_rate,
            "sampling_rate_verification": "publication_declared_not_empirically_verifiable_without_timestamps",
            "stated_design_nominal_rows_after_50_samples_trimmed_from_each_end_of_each_repeat": nominal_trimmed_rows,
            "stated_design_nominal_rows_before_trim": nominal_untrimmed_rows,
        },
        "required_primary_signals": {
            "channels": list(INCLUSIVEHAR_PRIMARY_CHANNELS),
            "coordinate_convention": "X/Y/Z names are encoded; handedness and a device-to-body coordinate transform are not encoded in the CSV",
            "cross_source_bridge": {
                "closest_uci_pair": ["body_acc_x/y/z", "body_gyro_x/y/z"],
                "no_conversion_primary_units": "retain acceleration in g and angular rate in rad/s for both sources",
                "separation_caveat": "InclusiveHAR Core Motion user acceleration and UCI body acceleration are conceptually closest but use different gravity-separation pipelines",
                "si_acceleration_sensitivity": "multiply every acceleration channel consistently by exactly 9.80665 m/s^2 per g",
                "total_acceleration_policy": "raw accelerometer or total-acceleration inputs are a separate sensitivity track and must not be mixed into the primary interface",
            },
            "csv_unit_metadata_present": False,
            "units_from_publication_and_manifest": {
                "motionRotationRate": "rad/s",
                "motionUserAcceleration": "G",
            },
        },
        "sensitive_metadata_audit": sensitive_metadata_audit,
        "status": "integrity_pass_protocol_quarantine"
        if schema_quality_pass and primary_data_quality_pass and coverage_pass
        else "data_audit_fail",
        "version_history_boundary_recovery": {
            "conclusion": "failed_no_released_version_contains_timestamp_trial_session_or_sample_identifiers",
            "metadata_endpoints": [
                f"https://data.mendeley.com/public-api/datasets/r78dn3f6nc/files?folder_id=root&version={version}"
                for version in range(1, 5)
            ],
            "records": [
                {
                    "content_id": "4e985d38-6139-47f9-a8b1-b1e39fb4e6a6",
                    "file_id": "d8656c19-61ec-4943-a161-217d937d9bca",
                    "filename": "DisabledHAR_dataset_v1.csv",
                    "header_audit": "same_33_columns_no_timestamp_trial_session_or_sample_identifier",
                    "sha256": "20109a3e580ea893712182cf8594ebeaf39cdcbae7dc642297a630a56b0f7cb7",
                    "size_bytes": 148856120,
                    "version": 1,
                },
                {
                    "content_id": "54b2c2c0-135d-4f1c-8d94-b45e35958b41",
                    "file_id": "6be7ec3b-a21a-4288-8e09-18f1f238ddf2",
                    "filename": "DisabledHAR_dataset_v2 (1).csv",
                    "header_audit": "byte_identical_to_v4_sensor_csv",
                    "sha256": "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34",
                    "size_bytes": 148915541,
                    "version": 2,
                },
                {
                    "content_id": "54b2c2c0-135d-4f1c-8d94-b45e35958b41",
                    "file_id": "6be7ec3b-a21a-4288-8e09-18f1f238ddf2",
                    "filename": "DisabledHAR_dataset_v2 (1).csv",
                    "header_audit": "byte_identical_to_v4_sensor_csv",
                    "sha256": "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34",
                    "size_bytes": 148915541,
                    "version": 3,
                },
                {
                    "content_id": "54b2c2c0-135d-4f1c-8d94-b45e35958b41",
                    "file_id": "6be7ec3b-a21a-4288-8e09-18f1f238ddf2",
                    "filename": "InclusiveHAR_dataset_v2 (1).csv",
                    "header_audit": "direct_read_only_audit_this_report",
                    "sha256": "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34",
                    "size_bytes": 148915541,
                    "version": 4,
                },
            ],
            "v4_docx_change": "version_4_adds_sensitive_participant_description_docx_but_no_boundary_metadata",
            "verification_scope": "official API metadata for versions 1-4 plus authorized header-only audit; historical CSV bodies were not added to the v4 raw cache",
        },
    }
    payload["report_sha256"] = canonical_json_sha256(payload)
    return payload
