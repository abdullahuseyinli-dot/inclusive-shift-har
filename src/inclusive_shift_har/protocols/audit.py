"""Independent structural audit for released-block split manifests."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.protocols.splits import build_released_block_split_manifest


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    return value


def _issue(code: str, message: str, location: str) -> dict[str, str]:
    return {"code": code, "location": location, "message": message}


def audit_released_block_split_manifest(
    manifest: Mapping[str, Any],
    *,
    expected_manifest: Mapping[str, Any] | None = None,
    split_file_sha256: str | None = None,
) -> dict[str, Any]:
    """Audit hashes and leakage invariants without reading sensor signals."""

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    claimed_hash = manifest.get("split_manifest_sha256")
    without_hash = dict(manifest)
    without_hash.pop("split_manifest_sha256", None)
    observed_hash = canonical_json_sha256(without_hash)
    if claimed_hash != observed_hash:
        errors.append(
            _issue(
                "SELF_HASH_MISMATCH",
                f"claimed {claimed_hash!r}, recomputed {observed_hash}",
                "$.split_manifest_sha256",
            )
        )
    if expected_manifest is not None:
        expected_hash = expected_manifest.get("split_manifest_sha256")
        if claimed_hash != expected_hash:
            errors.append(
                _issue(
                    "DETERMINISTIC_REBUILD_MISMATCH",
                    f"stored {claimed_hash!r}, rebuilt {expected_hash!r}",
                    "$.split_manifest_sha256",
                )
            )
        if canonical_json_sha256(manifest) != canonical_json_sha256(expected_manifest):
            errors.append(
                _issue(
                    "MANIFEST_CONTENT_REBUILD_MISMATCH",
                    "stored split content is not byte-canonical-equivalent to a fresh rebuild",
                    "$",
                )
            )

    if manifest.get("schema_version") != "1.0.0":
        errors.append(_issue("SCHEMA_VERSION", "schema_version must equal 1.0.0", "$"))
    if manifest.get("manifest_kind") != "released_block_split":
        errors.append(
            _issue(
                "MANIFEST_KIND",
                "manifest_kind must equal released_block_split",
                "$.manifest_kind",
            )
        )
    if manifest.get("subject_assignment_before_windowing") is not True:
        errors.append(
            _issue(
                "PARTITION_ORDER",
                "participants must be assigned before windowing",
                "$.subject_assignment_before_windowing",
            )
        )
    if manifest.get("target_performance_or_prediction_accessed") is not False:
        errors.append(
            _issue(
                "TARGET_ACCESS",
                "target predictions or performance must remain unopened",
                "$.target_performance_or_prediction_accessed",
            )
        )
    if manifest.get("trial_boundary_status") != "unrecoverable":
        errors.append(
            _issue(
                "TRIAL_BOUNDARY_STATUS",
                "hidden trial boundaries must remain recorded as unrecoverable",
                "$.trial_boundary_status",
            )
        )
    model_policy = manifest.get("model_input_policy")
    exact_channels = [
        "motionUserAccelerationX",
        "motionUserAccelerationY",
        "motionUserAccelerationZ",
        "motionRotationRateX",
        "motionRotationRateY",
        "motionRotationRateZ",
    ]
    required_forbidden = {
        "UserID",
        "disabled",
        "file_order",
        "global_row_index",
        "gps_or_location",
        "label",
        "released_block_position",
        "window_offset",
    }
    if not isinstance(model_policy, Mapping) or (
        model_policy.get("allowed_channels_exact_order") != exact_channels
        or model_policy.get("global_rows_and_window_offsets_are_provenance_only") is not True
        or not isinstance(model_policy.get("forbidden_inputs"), list)
        or not required_forbidden <= set(model_policy["forbidden_inputs"])
    ):
        errors.append(
            _issue(
                "MODEL_INPUT_POLICY",
                "model inputs must use only the exact six-channel allowlist and exclude identity/order/location proxies",
                "$.model_input_policy",
            )
        )

    try:
        windows = [
            _mapping(value, name=f"windows[{index}]")
            for index, value in enumerate(_list(manifest.get("windows"), name="windows"))
        ]
        blocks = [
            _mapping(value, name=f"block_summaries[{index}]")
            for index, value in enumerate(
                _list(manifest.get("block_summaries"), name="block_summaries")
            )
        ]
    except ValueError as exc:
        errors.append(_issue("STRUCTURE", str(exc), "$"))
        windows = []
        blocks = []

    if manifest.get("window_length_samples") != 128:
        errors.append(_issue("WINDOW_LENGTH", "window length must equal 128", "$"))
    if manifest.get("window_stride_samples") != 128:
        errors.append(_issue("WINDOW_STRIDE", "window stride must equal 128", "$"))
    if manifest.get("window_count") != len(windows):
        errors.append(
            _issue(
                "WINDOW_COUNT",
                f"declared {manifest.get('window_count')!r}, observed {len(windows)}",
                "$.window_count",
            )
        )

    seen_window_ids: set[str] = set()
    seen_interval_ids: set[str] = set()
    subject_partitions: dict[str, set[str]] = defaultdict(set)
    windows_by_block: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    partition_window_ids: dict[str, list[str]] = defaultdict(list)
    intervals: list[tuple[int, int, str, str]] = []
    for index, window in enumerate(windows):
        location = f"$.windows[{index}]"
        window_id = str(window.get("window_id", ""))
        raw_interval_id = str(window.get("raw_interval_id", ""))
        if not window_id or window_id in seen_window_ids:
            errors.append(_issue("WINDOW_ID_UNIQUE", "window_id is empty or duplicated", location))
        seen_window_ids.add(window_id)
        if not raw_interval_id or raw_interval_id in seen_interval_ids:
            errors.append(
                _issue("RAW_INTERVAL_ID_UNIQUE", "raw_interval_id is empty or duplicated", location)
            )
        seen_interval_ids.add(raw_interval_id)
        try:
            start = int(window["start_row_inclusive"])
            end = int(window["end_row_inclusive"])
            length = int(window["length_samples"])
            stride = int(window["stride_samples"])
            ordinal = int(window["window_ordinal"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(_issue("WINDOW_FIELDS", str(exc), location))
            continue
        if length != 128 or stride != 128 or end - start + 1 != 128:
            errors.append(
                _issue(
                    "WINDOW_GEOMETRY",
                    f"expected length=stride=128 and inclusive interval 128; got {length}, {stride}, [{start}, {end}]",
                    location,
                )
            )
        if ordinal < 0:
            errors.append(_issue("WINDOW_ORDINAL", "window ordinal must be non-negative", location))
        if window.get("trial_id") is not None or window.get("trial_status") != "unrecoverable":
            errors.append(
                _issue(
                    "TRIAL_PROVENANCE",
                    "trial_id must remain null and trial_status unrecoverable",
                    location,
                )
            )
        subject = str(window.get("subject_id", ""))
        partition = str(window.get("partition", ""))
        block_id = str(window.get("released_run_id", ""))
        subject_partitions[subject].add(partition)
        windows_by_block[block_id].append(window)
        partition_window_ids[partition].append(window_id)
        intervals.append((start, end, partition, window_id))

    for subject, assigned_partitions in sorted(
        subject_partitions.items(), key=lambda item: item[0]
    ):
        if len(assigned_partitions) != 1:
            errors.append(
                _issue(
                    "SUBJECT_PARTITION_OVERLAP",
                    f"subject {subject} occurs in partitions {sorted(assigned_partitions)}",
                    "$.windows",
                )
            )

    intervals.sort()
    for previous, current in pairwise(intervals):
        if current[0] <= previous[1]:
            errors.append(
                _issue(
                    "RAW_SAMPLE_OVERLAP",
                    f"{previous[3]} [{previous[0]}, {previous[1]}] overlaps {current[3]} [{current[0]}, {current[1]}]",
                    "$.windows",
                )
            )

    block_ids: set[str] = set()
    total_source_rows = 0
    total_used_rows = 0
    total_dropped_rows = 0
    total_conditional_bound = 0
    for index, block in enumerate(blocks):
        location = f"$.block_summaries[{index}]"
        block_id = str(block.get("released_run_id", ""))
        if not block_id or block_id in block_ids:
            errors.append(
                _issue("BLOCK_ID_UNIQUE", "released run ID is empty or duplicated", location)
            )
        block_ids.add(block_id)
        try:
            start = int(block["start_row_inclusive"])
            end = int(block["end_row_inclusive"])
            row_count = int(block["row_count"])
            window_count = int(block["window_count"])
            used_rows = int(block["used_rows"])
            dropped_rows = int(block["dropped_tail_rows"])
            conditional_bound = int(block["conditional_hidden_join_crossing_window_bound"])
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(_issue("BLOCK_FIELDS", str(exc), location))
            continue
        if end - start + 1 != row_count:
            errors.append(
                _issue("BLOCK_INTERVAL", "block interval does not match row_count", location)
            )
        if window_count != row_count // 128:
            errors.append(
                _issue("BLOCK_WINDOW_COUNT", "block window count is not floor(rows/128)", location)
            )
        if used_rows != window_count * 128 or row_count != used_rows + dropped_rows:
            errors.append(
                _issue("REMAINDER_ACCOUNTING", "block row accounting is inconsistent", location)
            )
        if dropped_rows < 0 or dropped_rows >= 128:
            errors.append(
                _issue("REMAINDER_RANGE", "dropped remainder must be in [0, 127]", location)
            )
        if conditional_bound != min(2, window_count):
            errors.append(
                _issue(
                    "HIDDEN_JOIN_BOUND",
                    "conditional block bound must equal min(2, windows)",
                    location,
                )
            )
        block_windows = sorted(
            windows_by_block.get(block_id, []), key=lambda item: int(item["window_ordinal"])
        )
        if len(block_windows) != window_count:
            errors.append(
                _issue("BLOCK_WINDOW_COVERAGE", "block window membership count differs", location)
            )
        for ordinal, window in enumerate(block_windows):
            expected_start = start + ordinal * 128
            if (
                int(window["window_ordinal"]) != ordinal
                or int(window["start_row_inclusive"]) != expected_start
                or int(window["end_row_inclusive"]) != expected_start + 127
                or int(window["end_row_inclusive"]) > end
            ):
                errors.append(
                    _issue(
                        "BLOCK_BOUNDARY_OR_ORDINAL",
                        "window is not the expected non-overlapping released-block interval",
                        location,
                    )
                )
                break
        total_source_rows += row_count
        total_used_rows += used_rows
        total_dropped_rows += dropped_rows
        total_conditional_bound += conditional_bound
    unexpected_block_memberships = sorted(set(windows_by_block) - block_ids)
    if unexpected_block_memberships:
        errors.append(
            _issue(
                "UNKNOWN_BLOCK",
                f"windows reference unknown released blocks: {unexpected_block_memberships[:5]}",
                "$.windows",
            )
        )

    partitions_value = manifest.get("partitions")
    try:
        partition_records = [
            _mapping(value, name=f"partitions[{index}]")
            for index, value in enumerate(_list(partitions_value, name="partitions"))
        ]
    except ValueError as exc:
        errors.append(_issue("PARTITIONS", str(exc), "$.partitions"))
        partition_records = []
    expected_partition_names = {"source_train", "source_validation", "target_sealed"}
    observed_partition_names = {str(item.get("partition", "")) for item in partition_records}
    if observed_partition_names != expected_partition_names:
        errors.append(
            _issue(
                "PARTITION_SET",
                f"expected {sorted(expected_partition_names)}, observed {sorted(observed_partition_names)}",
                "$.partitions",
            )
        )
    if set(partition_window_ids) != expected_partition_names:
        errors.append(_issue("WINDOW_PARTITION_SET", "window partition set differs", "$.windows"))
    for partition_record in partition_records:
        name = str(partition_record.get("partition", ""))
        ids = sorted(partition_window_ids.get(name, []))
        if partition_record.get("window_count") != len(ids):
            errors.append(_issue("PARTITION_COUNT", f"{name} window count differs", "$.partitions"))
        if partition_record.get("window_ids_sha256") != canonical_json_sha256(ids):
            errors.append(
                _issue("PARTITION_DIGEST", f"{name} window digest differs", "$.partitions")
            )
        subject_ids = partition_record.get("subject_ids")
        if not isinstance(subject_ids, list) or partition_record.get("subject_count") != len(
            subject_ids
        ):
            errors.append(
                _issue("PARTITION_SUBJECTS", f"{name} subject metadata differs", "$.partitions")
            )

    normalization = manifest.get("normalization_policy")
    if not isinstance(normalization, Mapping) or dict(normalization) != {
        "few_person_reuses_source_only_statistics": True,
        "fit_partition": "source_train",
        "target_excluded": True,
        "validation_excluded": True,
    }:
        errors.append(
            _issue(
                "NORMALIZATION_SCOPE",
                "normalization must be fit on source_train only",
                "$.normalization_policy",
            )
        )

    try:
        source_cv = _list(manifest.get("source_cv_folds"), name="source_cv_folds")
    except ValueError as exc:
        errors.append(_issue("SOURCE_CV", str(exc), "$.source_cv_folds"))
        source_cv = []
    all_source_subjects = {
        subject
        for subject, partition_set in subject_partitions.items()
        if partition_set <= {"source_train", "source_validation"}
    }
    cv_validation_counts: Counter[str] = Counter()
    for index, value in enumerate(source_cv):
        fold = _mapping(value, name=f"source_cv_folds[{index}]")
        train_subjects = set(str(item) for item in _list(fold.get("train_subjects"), name="train"))
        validation_subjects = set(
            str(item) for item in _list(fold.get("validation_subjects"), name="validation")
        )
        normalization_subjects = set(
            str(item)
            for item in _list(fold.get("normalization_fit_subjects"), name="normalization")
        )
        if (
            train_subjects & validation_subjects
            or train_subjects | validation_subjects != all_source_subjects
        ):
            errors.append(
                _issue(
                    "SOURCE_CV_SUBJECTS",
                    "source fold does not partition source",
                    f"$.source_cv_folds[{index}]",
                )
            )
        if normalization_subjects != train_subjects:
            errors.append(
                _issue(
                    "SOURCE_CV_NORMALIZATION",
                    "fold normalization scope differs from train",
                    f"$.source_cv_folds[{index}]",
                )
            )
        cv_validation_counts.update(validation_subjects)
    if cv_validation_counts != Counter({subject: 1 for subject in all_source_subjects}):
        errors.append(
            _issue(
                "SOURCE_CV_COVERAGE",
                "source validation folds do not cover each source once",
                "$.source_cv_folds",
            )
        )

    protocol_record = manifest.get("protocol")
    nested_required = isinstance(protocol_record, Mapping) and protocol_record.get(
        "protocol_id"
    ) in {"inclusivehar-released-block-v1.1", "inclusivehar-released-block-v1.2"}
    explicit_schema_required = (
        isinstance(protocol_record, Mapping)
        and protocol_record.get("protocol_id") == "inclusivehar-released-block-v1.2"
    )
    if explicit_schema_required:
        expected_orders = {
            "functional_core": ["mobility", "sitting", "standing"],
            "inclusive_native": [
                "jogging",
                "ramp_ascent",
                "ramp_descent",
                "sitting",
                "standing",
                "walking",
            ],
        }
        expected_schemas: dict[str, dict[str, Any]] = {}
        for track, order in expected_orders.items():
            schema: dict[str, Any] = {
                "class_count": len(order),
                "class_order": order,
                "index_by_class": {label: index for index, label in enumerate(order)},
                "track": track,
            }
            schema["class_schema_sha256"] = canonical_json_sha256(schema)
            expected_schemas[track] = schema
        ontology_record = manifest.get("ontology")
        if (
            not isinstance(ontology_record, Mapping)
            or ontology_record.get("runnable_track_schemas") != expected_schemas
        ):
            errors.append(
                _issue(
                    "NUMERIC_CLASS_SCHEMA",
                    "runnable tracks lack the locked explicit numeric class order and hash",
                    "$.ontology.runnable_track_schemas",
                )
            )
    nested_value = manifest.get("source_nested_cv")
    if nested_required and not isinstance(nested_value, list):
        errors.append(
            _issue(
                "SOURCE_NESTED_CV_REQUIRED",
                "protocol v1.1 requires nested source cross-validation records",
                "$.source_nested_cv",
            )
        )
        nested_value = []
    if isinstance(nested_value, list):
        nested_outer_counts: Counter[str] = Counter()
        for outer_index, outer_value in enumerate(nested_value):
            outer = _mapping(outer_value, name="nested outer fold")
            outer_test = set(
                str(item) for item in _list(outer.get("outer_test_subjects"), name="outer test")
            )
            nested_outer_counts.update(outer_test)
            if outer.get("outer_test_allowed_for_selection") is not False:
                errors.append(
                    _issue(
                        "SOURCE_NESTED_TEST_SELECTION",
                        "nested outer test may not be used for model selection",
                        f"$.source_nested_cv[{outer_index}]",
                    )
                )
            inner_values = _list(outer.get("inner_folds"), name="inner folds")
            inner_validation_counts: Counter[str] = Counter()
            if len(inner_values) != 4:
                errors.append(
                    _issue(
                        "SOURCE_NESTED_INNER_COUNT",
                        "each source outer fold must contain four inner folds",
                        f"$.source_nested_cv[{outer_index}]",
                    )
                )
            for inner_value in inner_values:
                inner = _mapping(inner_value, name="nested inner fold")
                train = set(
                    str(item) for item in _list(inner.get("train_subjects"), name="inner train")
                )
                validation = set(
                    str(item)
                    for item in _list(inner.get("validation_subjects"), name="inner validation")
                )
                fit = set(
                    str(item)
                    for item in _list(
                        inner.get("normalization_fit_subjects"), name="inner normalization"
                    )
                )
                if (
                    train & validation
                    or train & outer_test
                    or validation & outer_test
                    or train | validation | outer_test != all_source_subjects
                ):
                    errors.append(
                        _issue(
                            "SOURCE_NESTED_SUBJECTS",
                            "inner train, validation, and outer test must be disjoint and cover source",
                            f"$.source_nested_cv[{outer_index}]",
                        )
                    )
                if fit != train:
                    errors.append(
                        _issue(
                            "SOURCE_NESTED_NORMALIZATION",
                            "nested normalization subjects must equal inner training subjects",
                            f"$.source_nested_cv[{outer_index}]",
                        )
                    )
                inner_validation_counts.update(validation)
            expected_inner = Counter({subject: 1 for subject in all_source_subjects - outer_test})
            if inner_validation_counts != expected_inner:
                errors.append(
                    _issue(
                        "SOURCE_NESTED_INNER_COVERAGE",
                        "inner validation folds must cover every non-outer source once",
                        f"$.source_nested_cv[{outer_index}]",
                    )
                )
        if nested_outer_counts != Counter({subject: 1 for subject in all_source_subjects}):
            errors.append(
                _issue(
                    "SOURCE_NESTED_OUTER_COVERAGE",
                    "nested outer tests must cover every source participant once",
                    "$.source_nested_cv",
                )
            )

    try:
        target_folds = _list(manifest.get("few_person_outer_folds"), name="few_person_outer_folds")
    except ValueError as exc:
        errors.append(_issue("FEW_PERSON", str(exc), "$.few_person_outer_folds"))
        target_folds = []
    target_subjects = {
        subject
        for subject, partitions_set in subject_partitions.items()
        if partitions_set == {"target_sealed"}
    }
    target_eval_counts: Counter[str] = Counter()
    for fold_index, value in enumerate(target_folds):
        fold = _mapping(value, name=f"few_person_outer_folds[{fold_index}]")
        inclusion_order = tuple(
            str(item) for item in _list(fold.get("inclusion_order"), name="inclusion_order")
        )
        scenarios = _list(fold.get("scenarios"), name="scenarios")
        previous_inclusion: set[str] = set()
        for scenario_index, scenario_value in enumerate(scenarios):
            scenario = _mapping(scenario_value, name="scenario")
            k = int(scenario.get("k", -1))
            evaluation = set(
                str(item) for item in _list(scenario.get("evaluation_subjects"), name="evaluation")
            )
            inclusion = tuple(
                str(item)
                for item in _list(
                    scenario.get("target_inclusion_subjects"), name="target_inclusion"
                )
            )
            unused = set(
                str(item) for item in _list(scenario.get("unused_target_subjects"), name="unused")
            )
            if scenario_index == 0:
                target_eval_counts.update(evaluation)
            if k not in {1, 2, 4} or inclusion != inclusion_order[:k]:
                errors.append(
                    _issue(
                        "FEW_PERSON_PREFIX",
                        "scenario is not the locked nested prefix",
                        f"$.few_person_outer_folds[{fold_index}]",
                    )
                )
            if previous_inclusion and not previous_inclusion < set(inclusion):
                errors.append(
                    _issue(
                        "FEW_PERSON_NESTING",
                        "inclusion sets are not strictly nested",
                        f"$.few_person_outer_folds[{fold_index}]",
                    )
                )
            previous_inclusion = set(inclusion)
            if evaluation & set(inclusion) or evaluation & unused or set(inclusion) & unused:
                errors.append(
                    _issue(
                        "FEW_PERSON_OVERLAP",
                        "evaluation, inclusion, and unused target sets overlap",
                        f"$.few_person_outer_folds[{fold_index}]",
                    )
                )
            if evaluation | set(inclusion) | unused != target_subjects:
                errors.append(
                    _issue(
                        "FEW_PERSON_COVERAGE",
                        "scenario does not cover target cohort",
                        f"$.few_person_outer_folds[{fold_index}]",
                    )
                )
            fit_subjects = set(
                str(item)
                for item in _list(
                    scenario.get("normalization_fit_subjects"), name="normalization_fit"
                )
            )
            if fit_subjects & target_subjects:
                errors.append(
                    _issue(
                        "FEW_PERSON_NORMALIZATION",
                        "target participant entered normalization",
                        f"$.few_person_outer_folds[{fold_index}]",
                    )
                )
    if target_eval_counts != Counter({subject: 1 for subject in target_subjects}):
        errors.append(
            _issue(
                "FEW_PERSON_EVAL_COVERAGE",
                "outer evaluation folds do not cover each target once",
                "$.few_person_outer_folds",
            )
        )

    target_seal = manifest.get("target_seal")
    if not isinstance(target_seal, Mapping) or (
        target_seal.get("status") != "sealed"
        or target_seal.get("unlock_record") is not None
        or target_seal.get("performance_inspection") != "forbidden"
        or target_seal.get("maximum_confirmatory_openings") != 1
        or not isinstance(target_seal.get("seal_id"), str)
        or len(str(target_seal.get("seal_id"))) != 64
    ):
        errors.append(_issue("TARGET_SEAL", "target seal is absent or open", "$.target_seal"))

    hidden_join = manifest.get("hidden_join_risk")
    if not isinstance(hidden_join, Mapping):
        errors.append(_issue("HIDDEN_JOIN_RISK", "hidden join risk must be an object", "$"))
    else:
        expected_fraction = total_conditional_bound / len(windows) if windows else 0.0
        if hidden_join.get("conditional_crossing_window_bound") != total_conditional_bound:
            errors.append(
                _issue(
                    "HIDDEN_JOIN_TOTAL",
                    "conditional crossing-window total differs",
                    "$.hidden_join_risk",
                )
            )
        observed_fraction = hidden_join.get("conditional_crossing_window_fraction_bound")
        if (
            not isinstance(observed_fraction, (int, float))
            or abs(observed_fraction - expected_fraction) > 1e-15
        ):
            errors.append(
                _issue(
                    "HIDDEN_JOIN_FRACTION",
                    "conditional crossing-window fraction differs",
                    "$.hidden_join_risk",
                )
            )
        if hidden_join.get("unconditional_crossing_window_fraction_bound") != 1.0:
            errors.append(
                _issue(
                    "HIDDEN_JOIN_UNCONDITIONAL",
                    "unconditional bound must remain 1.0",
                    "$.hidden_join_risk",
                )
            )

    remainder = manifest.get("remainder_accounting")
    if not isinstance(remainder, Mapping) or (
        remainder.get("source_data_rows") != total_source_rows
        or remainder.get("used_window_rows") != total_used_rows
        or remainder.get("dropped_tail_rows") != total_dropped_rows
        or total_source_rows != total_used_rows + total_dropped_rows
    ):
        errors.append(
            _issue(
                "REMAINDER_TOTAL",
                "aggregate remainder accounting differs",
                "$.remainder_accounting",
            )
        )

    expected_id_digest = canonical_json_sha256(sorted(seen_window_ids))
    if manifest.get("window_id_set_sha256") != expected_id_digest:
        errors.append(
            _issue("WINDOW_ID_DIGEST", "window ID set digest differs", "$.window_id_set_sha256")
        )

    warnings.extend(
        [
            _issue(
                "TRIAL_BOUNDARIES_UNRECOVERABLE",
                "released blocks may contain hidden joins; this protocol is not trial-safe",
                "$.hidden_join_risk",
            ),
            _issue(
                "CONDITIONAL_BOUND_ONLY",
                "the two-join bound depends on an unverified three-contiguous-repetition assumption; the unconditional bound is 100 percent",
                "$.hidden_join_risk",
            ),
            _issue(
                "DECLARED_RATE_UNVERIFIED",
                "2.56-second duration is conditional on the provider-declared 50 Hz rate",
                "$.preprocessing",
            ),
        ]
    )
    report: dict[str, Any] = {
        "audit_kind": "released_block_split_structural_audit",
        "checks": {
            "block_count": len(blocks),
            "conditional_hidden_join_crossing_window_bound": total_conditional_bound,
            "dropped_tail_rows": total_dropped_rows,
            "raw_interval_count": len(seen_interval_ids),
            "source_data_rows": total_source_rows,
            "subject_count": len(subject_partitions),
            "used_window_rows": total_used_rows,
            "window_count": len(windows),
        },
        "errors": errors,
        "evidence_language": "participant-exclusive released-block protocol; not trial-safe",
        "schema_version": "1.0.0",
        "split_file_sha256": split_file_sha256,
        "split_manifest_sha256": claimed_hash,
        "status": "pass_conditional_released_block" if not errors else "fail",
        "target_performance_or_prediction_accessed": False,
        "valid": not errors,
        "warnings": warnings,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def audit_split_manifest_file(
    *,
    split_manifest_path: str | Path,
    audit_report_path: str | Path,
    protocol_config_path: str | Path,
    ontology_config_path: str | Path,
    preprocessing_config_path: str | Path,
    authorization_path: str | Path,
) -> dict[str, Any]:
    """Load, deterministically rebuild, and audit a split manifest."""

    parsed = load_json_strict(split_manifest_path)
    manifest = _mapping(parsed, name="split manifest")
    expected = build_released_block_split_manifest(
        audit_report_path=audit_report_path,
        protocol_config_path=protocol_config_path,
        ontology_config_path=ontology_config_path,
        preprocessing_config_path=preprocessing_config_path,
        authorization_path=authorization_path,
    )
    return audit_released_block_split_manifest(
        manifest,
        expected_manifest=expected,
        split_file_sha256=sha256_file(split_manifest_path),
    )


def write_split_audit_new(
    report: Mapping[str, Any],
    destination: str | Path,
    *,
    allowed_root: str | Path,
) -> Path:
    """Publish an immutable split-audit report without replacement."""

    return atomic_write_json_new(report, destination, allowed_root=allowed_root)
