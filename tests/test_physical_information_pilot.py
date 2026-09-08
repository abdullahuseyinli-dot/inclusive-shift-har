from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
import yaml
from jsonschema import Draft202012Validator

import inclusive_shift_har.experiments.physical_information_pilot as runner
from inclusive_shift_har.data.physical_information_pilot import (
    PENDING_PREREQUISITES,
    build_plan,
    validate_collection_records,
    validate_config,
    validate_plan,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs/experiments/physical_information_identifiability_v1.yaml"
PROTOCOL_PATH = ROOT / "docs/research/PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md"
SCHEMA_PATH = ROOT / "configs/schema/physical_information_pilot.schema.json"


def _config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _plan() -> dict[str, Any]:
    return build_plan(
        _config(),
        created_at_utc="2026-09-07T15:00:00.000000Z",
        code_commit="a" * 40,
        config_reference={
            "path": "configs/experiments/physical_information_identifiability_v1.yaml",
            "sha256": "b" * 64,
        },
        protocol_reference={
            "path": "docs/research/PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md",
            "sha256": "c" * 64,
        },
        schema_reference={
            "path": "configs/schema/physical_information_pilot.schema.json",
            "sha256": "d" * 64,
        },
    )


def _equipment(root: Path | None = None) -> dict[str, Any]:
    placement_hash = "d" * 64
    bench_hash = "e" * 64
    if root is not None:
        protocol = root / "protocol"
        protocol.mkdir(parents=True, exist_ok=True)
        placement = protocol / "placement.md"
        bench = protocol / "bench.json"
        placement.write_text("measured placement instruction\n", encoding="utf-8")
        bench.write_text('{"status":"measured"}\n', encoding="utf-8")
        placement_hash = sha256_file(placement)
        bench_hash = sha256_file(bench)
    return {
        "record_kind": "physical_information_equipment_receipt",
        "receipt_id": "equipment-001",
        "device_id": "device-001",
        "manufacturer": "measured-manufacturer",
        "model": "measured-model",
        "firmware": "measured-firmware",
        "logger": "measured-logger",
        "placement_instruction_path": "protocol/placement.md",
        "placement_instruction_sha256": placement_hash,
        "axis_convention": "documented-right-handed-frame",
        "channel_map": {
            "ax": "column-1",
            "ay": "column-2",
            "az": "column-3",
            "gx": "column-4",
            "gy": "column-5",
            "gz": "column-6",
        },
        "units": {"acceleration": "m/s2", "angular_velocity": "rad/s"},
        "clock_domain": "device-monotonic",
        "nominal_rate_hz": 50.0,
        "observed_interval_summary": {
            "sample_count": 1000,
            "median_seconds": 0.02,
            "p05_seconds": 0.019,
            "p95_seconds": 0.021,
            "nonpositive_interval_count": 0,
            "gap_count": 0,
        },
        "native_gravity_available": False,
        "processing_contract": {
            "source_rate_hz": 50.0,
            "target_rate_hz": 50.0,
            "gravity_source": "derived_causal_lowpass",
            "gravity_cutoff_hz": 0.30,
            "maximum_gap_seconds": 0.06,
            "resampling_method": "none_equal_rate",
            "window_samples": 128,
            "window_stride_samples": 64,
        },
        "frame_conditioning_limit": None,
        "bench_qualification_path": "protocol/bench.json",
        "bench_qualification_sha256": bench_hash,
        "qualified_at_utc": "2026-09-08T08:00:00Z",
        "processing_frozen_at_utc": "2026-09-08T08:30:00Z",
        "status": "qualified",
    }


def _collection_manifests(
    plan: dict[str, Any],
    root: Path,
    *,
    recording_status: str,
    adjudication_status: str,
    materialize_raw: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recordings: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    planned_bouts = {item["bout_id"]: item for item in plan["bouts"]}
    for block_index, block in enumerate(plan["blocks"]):
        recording_id = f"recording-{block_index + 1:02d}"
        relative_path = f"raw/{recording_id}.bin"
        raw = root / relative_path
        if materialize_raw:
            raw.parent.mkdir(parents=True, exist_ok=True)
            raw.write_bytes(f"measured-{recording_id}".encode())
            raw_hash = sha256_file(raw)
            size_bytes = raw.stat().st_size
        else:
            raw_hash = f"{block_index:064x}"
            size_bytes = 1
        visit_number = int(block["visit_number"])
        attachment_number = int(block["attachment_number"])
        recordings.append(
            {
                "recording_id": recording_id,
                "relative_path": relative_path,
                "sha256": raw_hash,
                "size_bytes": size_bytes,
                "sample_count": 30000,
                "device_id": "device-001",
                "wearer_id": block["wearer_id"],
                "visit_number": visit_number,
                "attachment_number": attachment_number,
                "clock_domain": "device-monotonic",
                "visit_started_at_utc": (
                    "2026-09-09T09:00:00Z" if visit_number == 1 else "2026-09-16T09:00:00Z"
                ),
                "started_at": 0.0 if attachment_number == 1 else 700.0,
                "ended_at": 600.0 if attachment_number == 1 else 1300.0,
                "status": recording_status,
            }
        )
        for sequence_index, bout_id in enumerate(block["bout_ids"]):
            planned = planned_bouts[bout_id]
            start = sequence_index * 2500
            duration = int(planned["expected_duration_seconds"]) * 50
            adjudicated = adjudication_status == "adjudicated"
            annotations.append(
                {
                    "bout_id": bout_id,
                    "recording_id": recording_id,
                    "start_sample": start,
                    "stop_sample_exclusive": start + duration,
                    "observed_activity": planned["activity"] if adjudicated else "unresolved",
                    "observed_motion": planned["motion"] if adjudicated else "unresolved",
                    "adjudication_status": adjudication_status,
                    "adjudicator_id": "adjudicator-001",
                    "adjudicated_at_utc": "2026-09-20T10:00:00Z",
                }
            )
    return recordings, annotations


def test_frozen_config_rejects_size_and_automatic_model_drift() -> None:
    config = _config()
    validate_config(config)
    changed = copy.deepcopy(config)
    changed["schedule"]["expected_bouts"] = 263
    with pytest.raises(ValueError, match="schedule"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["resources"]["automatic_architecture_experiment_allowed"] = True
    with pytest.raises(ValueError, match="resource"):
        validate_config(changed)
    changed = copy.deepcopy(config)
    changed["analysis_contract"]["estimands"]["signed_query_margin"] = "outcome_tuned"
    with pytest.raises(ValueError, match="analysis contract"):
        validate_config(changed)


def test_plan_has_exact_people_visits_blocks_bouts_time_and_chronology() -> None:
    plan = _plan()
    report = validate_plan(plan, _config())
    assert report["valid"] is True
    assert plan["counts"] == {
        "wearers": 6,
        "visits": 12,
        "blocks": 24,
        "bouts": 264,
        "support_bouts": 96,
        "query_bouts": 168,
        "nominal_recording_seconds": 9000,
        "fresh_query_blocks": 24,
        "attachment_stale_query_blocks": 12,
        "visit_stale_query_blocks": 12,
    }
    assert len({item["bout_id"] for item in plan["bouts"]}) == 264
    for block in plan["blocks"]:
        rows = [item for item in plan["bouts"] if item["block_id"] == block["block_id"]]
        rows.sort(key=lambda item: item["sequence_in_block"])
        assert [item["role"] for item in rows] == ["support"] * 4 + ["query"] * 7
    assert plan["pending_prerequisites"] == list(PENDING_PREREQUISITES)
    assert all(value is False for value in plan["claims"].values())


def test_counterbalance_preserves_condition_inventory_without_fixed_first_prompt() -> None:
    plan = _plan()
    first_supports: list[tuple[str, str]] = []
    first_by_cell: dict[tuple[int, int], set[tuple[str, str]]] = {}
    first_query_by_cell: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for block in plan["blocks"]:
        rows = [
            item
            for item in plan["bouts"]
            if item["block_id"] == block["block_id"] and item["role"] == "support"
        ]
        first = min(rows, key=lambda item: item["sequence_in_block"])
        first_condition = (first["activity"], first["motion"])
        first_supports.append(first_condition)
        cell = (block["visit_number"], block["attachment_number"])
        first_by_cell.setdefault(cell, set()).add(first_condition)
        query_rows = [
            item
            for item in plan["bouts"]
            if item["block_id"] == block["block_id"] and item["role"] == "query"
        ]
        first_query = min(query_rows, key=lambda item: item["sequence_in_block"])
        first_query_by_cell.setdefault(cell, set()).add(
            (first_query["activity"], first_query["motion"])
        )
        assert {(item["activity"], item["motion"]) for item in rows} == {
            ("sitting", "quiet"),
            ("sitting", "upper_body_motion"),
            ("standing", "quiet"),
            ("standing", "upper_body_motion"),
        }
    assert len(set(first_supports)) == 4
    assert all(len(conditions) == 4 for conditions in first_by_cell.values())
    assert all(len(conditions) == 6 for conditions in first_query_by_cell.values())


def test_stale_comparisons_use_identical_queries_and_only_earlier_support() -> None:
    plan = _plan()
    block_order = {item["block_id"]: item["global_block_index"] for item in plan["blocks"]}
    bout_block = {item["bout_id"]: item["block_id"] for item in plan["bouts"]}
    for comparison in plan["comparisons"]:
        query_block = comparison["query_block_id"]
        query_ids = comparison["query_bout_ids"]
        assert all(bout_block[item] == query_block for item in query_ids)
        stale = comparison["stale_support_bout_ids"]
        if stale is None:
            assert comparison["comparison_type"] == "fresh"
        else:
            assert all(block_order[bout_block[item]] < block_order[query_block] for item in stale)
            assert comparison["validity_mask_policy"] == (
                "same_query_ids_and_intersection_validity_mask"
            )


def test_mutated_future_support_and_self_hash_are_rejected() -> None:
    plan = _plan()
    plan["comparisons"][24]["stale_support_bout_ids"] = plan["comparisons"][-1][
        "fresh_support_bout_ids"
    ]
    plan["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "record_sha256"}
    )
    report = validate_plan(plan, _config())
    assert report["valid"] is False
    assert any("comparisons" in item or "not earlier" in item for item in report["errors"])


def test_plan_validates_against_checked_in_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
    validator.validate(_plan())
    poisoned = _plan()
    poisoned["counts"]["bouts"] = 263
    assert list(validator.iter_errors(poisoned))


def test_schema_defines_strict_equipment_raw_and_annotation_records() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    equipment_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/equipment_receipt",
    }
    Draft202012Validator(equipment_schema).validate(_equipment())
    inconsistent_native = _equipment()
    inconsistent_native["processing_contract"]["gravity_source"] = "provider_native_gravity"
    inconsistent_native["processing_contract"]["gravity_cutoff_hz"] = None
    assert list(Draft202012Validator(equipment_schema).iter_errors(inconsistent_native))
    raw = {
        "recording_id": "recording-001",
        "relative_path": "raw/recording.bin",
        "sha256": "a" * 64,
        "size_bytes": 100,
        "sample_count": 1000,
        "device_id": "device-001",
        "wearer_id": "pilot:001",
        "visit_number": 1,
        "attachment_number": 1,
        "clock_domain": "device-monotonic",
        "visit_started_at_utc": "2026-09-08T09:00:00Z",
        "started_at": 0.0,
        "ended_at": 20.0,
        "status": "recorded",
    }
    raw_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/raw_recording",
    }
    Draft202012Validator(
        raw_schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(raw)
    annotation = {
        "bout_id": "planned-bout",
        "recording_id": "recording-001",
        "start_sample": 0,
        "stop_sample_exclusive": 100,
        "observed_activity": "sitting",
        "observed_motion": "quiet",
        "adjudication_status": "adjudicated",
        "adjudicator_id": "adjudicator-001",
        "adjudicated_at_utc": "2026-09-08T10:00:00Z",
    }
    annotation_schema = {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": "#/$defs/annotation",
    }
    Draft202012Validator(
        annotation_schema,
        format_checker=Draft202012Validator.FORMAT_CHECKER,
    ).validate(annotation)
    unsafe = dict(raw, relative_path="../outside.bin")
    assert list(Draft202012Validator(raw_schema).iter_errors(unsafe))


def test_collection_without_equipment_is_explicitly_not_ready(tmp_path: Path) -> None:
    report = validate_collection_records(
        plan=_plan(),
        config=_config(),
        equipment=None,
        recordings=[],
        annotations=[],
        raw_root=tmp_path,
    )
    assert report["valid"] is False
    assert report["status"] == "collection_not_ready"
    assert report["recording_count"] == 0
    assert len(report["missing_annotation_bout_ids"]) == 264
    assert report["model_fit_count"] == 0


def test_qualified_equipment_without_recordings_keeps_study_prerequisites_pending(
    tmp_path: Path,
) -> None:
    report = validate_collection_records(
        plan=_plan(),
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=[],
        annotations=[],
        raw_root=tmp_path,
    )
    assert report["valid"] is True
    assert report["status"] == "equipment_evidence_verified_collection_prerequisites_unverified"
    assert report["annotation_count"] == 0
    assert report["equipment_evidence_verified"] is True
    assert report["study_prerequisites_verified"] is False
    assert report["collection_complete"] is False


def test_equipment_qualification_requires_real_hashed_evidence(tmp_path: Path) -> None:
    report = validate_collection_records(
        plan=_plan(),
        config=_config(),
        equipment=_equipment(),
        recordings=[],
        annotations=[],
        raw_root=tmp_path,
    )
    assert report["valid"] is False
    assert report["status"] == "collection_records_invalid"
    assert report["equipment_evidence_verified"] is False
    assert any("equipment evidence file missing" in item for item in report["errors"])


def test_native_gravity_processing_requires_available_mapped_channels(tmp_path: Path) -> None:
    equipment = _equipment(tmp_path)
    equipment["processing_contract"]["gravity_source"] = "provider_native_gravity"
    equipment["processing_contract"]["gravity_cutoff_hz"] = None
    report = validate_collection_records(
        plan=_plan(),
        config=_config(),
        equipment=equipment,
        recordings=[],
        annotations=[],
        raw_root=tmp_path,
    )
    assert report["valid"] is False
    assert any(
        "native gravity was selected but is unavailable" in item for item in report["errors"]
    )
    assert any("native gravity channels are not mapped" in item for item in report["errors"])


def test_planned_and_unresolved_rosters_cannot_claim_complete_evidence(tmp_path: Path) -> None:
    plan = _plan()
    recordings, annotations = _collection_manifests(
        plan,
        tmp_path,
        recording_status="planned",
        adjudication_status="unresolved",
        materialize_raw=False,
    )
    report = validate_collection_records(
        plan=plan,
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=recordings,
        annotations=annotations,
        raw_root=tmp_path,
    )
    assert report["valid"] is True
    assert report["status"] == "partial_recorded_evidence"
    assert report["recording_count"] == 24
    assert report["recorded_block_count"] == 0
    assert report["annotation_count"] == 264
    assert report["adjudicated_bout_count"] == 0
    assert report["metadata_eligible_bout_count"] == 0
    assert report["analysis_eligible_bout_count"] == 0
    assert report["recorded_evidence_complete"] is False
    assert report["collection_complete"] is False


def test_complete_recorded_and_adjudicated_roster_is_evidence_complete_only(
    tmp_path: Path,
) -> None:
    plan = _plan()
    recordings, annotations = _collection_manifests(
        plan,
        tmp_path,
        recording_status="recorded",
        adjudication_status="adjudicated",
        materialize_raw=True,
    )
    report = validate_collection_records(
        plan=plan,
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=recordings,
        annotations=annotations,
        raw_root=tmp_path,
    )
    assert report["valid"] is True
    assert report["status"] == ("recorded_adjudicated_evidence_complete_signal_eligibility_pending")
    assert report["recorded_block_count"] == 24
    assert report["adjudicated_bout_count"] == 264
    assert report["metadata_eligible_bout_count"] == 264
    assert report["signal_eligibility_evaluated"] is False
    assert report["analysis_eligible_bout_count"] == 0
    assert report["analysis_ready"] is False
    assert report["recorded_evidence_complete"] is True
    assert report["study_prerequisites_verified"] is False
    assert report["collection_complete"] is False


def test_device_clock_and_attachment_chronology_are_bound(tmp_path: Path) -> None:
    plan = _plan()
    recordings, annotations = _collection_manifests(
        plan,
        tmp_path,
        recording_status="recorded",
        adjudication_status="adjudicated",
        materialize_raw=True,
    )
    recordings[0]["device_id"] = "wrong-device"
    recordings[1]["started_at"] = 500.0
    recordings[2]["started_at"] = -1.0
    recordings[3]["started_at"] = 1.0e15
    recordings[3]["ended_at"] = 1.0e15 + 1000.0
    report = validate_collection_records(
        plan=plan,
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=recordings,
        annotations=annotations,
        raw_root=tmp_path,
    )
    assert report["valid"] is False
    assert any("device differs" in item for item in report["errors"])
    assert any("attachment chronology violated" in item for item in report["errors"])
    assert any("time interval is invalid" in item for item in report["errors"])
    assert any("overflows UTC range" in item for item in report["errors"])


def test_partial_raw_record_and_nonoverlapping_bouts_validate_structurally(
    tmp_path: Path,
) -> None:
    plan = _plan()
    block = plan["blocks"][0]
    raw = tmp_path / "raw" / "recording.bin"
    raw.parent.mkdir()
    raw.write_bytes(b"measured bytes")
    recording = {
        "recording_id": "recording-001",
        "relative_path": "raw/recording.bin",
        "sha256": sha256_file(raw),
        "size_bytes": raw.stat().st_size,
        "sample_count": 2000,
        "device_id": "device-001",
        "wearer_id": block["wearer_id"],
        "visit_number": block["visit_number"],
        "attachment_number": block["attachment_number"],
        "clock_domain": "device-monotonic",
        "visit_started_at_utc": "2026-09-08T09:00:00Z",
        "started_at": 0.0,
        "ended_at": 40.0,
        "status": "recorded",
    }
    annotations = []
    for index, bout_id in enumerate(block["bout_ids"]):
        annotations.append(
            {
                "bout_id": bout_id,
                "recording_id": "recording-001",
                "start_sample": index * 100,
                "stop_sample_exclusive": index * 100 + 90,
                "observed_activity": "unresolved",
                "observed_motion": "unresolved",
                "adjudication_status": "unresolved",
                "adjudicator_id": "adjudicator-001",
                "adjudicated_at_utc": "2026-09-07T15:00:00Z",
            }
        )
    report = validate_collection_records(
        plan=plan,
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=[recording],
        annotations=annotations,
        raw_root=tmp_path,
    )
    assert report["valid"] is True
    assert report["status"] == "partial_recorded_evidence"
    assert report["recording_count"] == 1
    assert report["annotation_count"] == 11


def test_unsafe_path_hash_mismatch_and_role_overlap_are_rejected(tmp_path: Path) -> None:
    plan = _plan()
    block = plan["blocks"][0]
    recording = {
        "recording_id": "recording-001",
        "relative_path": "../outside.bin",
        "sha256": "0" * 64,
        "size_bytes": 1,
        "sample_count": 1000,
        "device_id": "device-001",
        "wearer_id": block["wearer_id"],
        "visit_number": block["visit_number"],
        "attachment_number": block["attachment_number"],
        "clock_domain": "device-monotonic",
        "visit_started_at_utc": "2026-09-08T09:00:00Z",
        "started_at": 0.0,
        "ended_at": 20.0,
        "status": "recorded",
    }
    annotations = [
        {
            "bout_id": block["bout_ids"][0],
            "recording_id": "recording-001",
            "start_sample": 100,
            "stop_sample_exclusive": 300,
            "observed_activity": "unresolved",
            "observed_motion": "unresolved",
            "adjudication_status": "unresolved",
            "adjudicator_id": "adjudicator-001",
            "adjudicated_at_utc": "2026-09-08T10:00:00Z",
        },
        {
            "bout_id": block["bout_ids"][4],
            "recording_id": "recording-001",
            "start_sample": 200,
            "stop_sample_exclusive": 400,
            "observed_activity": "unresolved",
            "observed_motion": "unresolved",
            "adjudication_status": "unresolved",
            "adjudicator_id": "adjudicator-001",
            "adjudicated_at_utc": "2026-09-08T10:00:00Z",
        },
    ]
    report = validate_collection_records(
        plan=plan,
        config=_config(),
        equipment=_equipment(tmp_path),
        recordings=[recording],
        annotations=annotations,
        raw_root=tmp_path,
    )
    assert report["valid"] is False
    assert any("unsafe raw path" in item for item in report["errors"])
    assert any("sample overlap" in item for item in report["errors"])
    assert any("chronology violated" in item for item in report["errors"])


def test_collection_cli_applies_record_schemas_binds_inputs_and_exits_nonzero(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    plan_path = tmp_path / "plan.json"
    equipment_path = tmp_path / "equipment.json"
    recordings_path = tmp_path / "recordings.json"
    annotations_path = tmp_path / "annotations.json"
    plan_path.write_text(json.dumps(_plan()), encoding="utf-8")
    malformed_equipment = _equipment(tmp_path)
    malformed_equipment["processing_contract"]["gravity_source"] = []
    equipment_path.write_text(json.dumps(malformed_equipment), encoding="utf-8")
    recordings_path.write_text(json.dumps([{"status": []}]), encoding="utf-8")
    annotations_path.write_text("[]", encoding="utf-8")
    output_path = tmp_path / "collection-validation.json"
    exit_code = runner.main(
        [
            "check-collection",
            "--plan",
            str(plan_path),
            "--config",
            str(CONFIG_PATH),
            "--schema",
            str(SCHEMA_PATH),
            "--equipment",
            str(equipment_path),
            "--recordings",
            str(recordings_path),
            "--annotations",
            str(annotations_path),
            "--raw-root",
            str(tmp_path),
            "--output",
            str(output_path),
        ]
    )
    capsys.readouterr()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert report["valid"] is False
    assert report["status"] == "collection_records_invalid"
    assert any("equipment schema" in item for item in report["errors"])
    assert any("recording[0] schema" in item for item in report["errors"])
    assert report["input_manifest"]["plan"]["sha256"] == sha256_file(plan_path)
    assert report["input_manifest"]["schema"]["sha256"] == sha256_file(SCHEMA_PATH)
    assert report["validator_source"]["sha256"] == sha256_file(
        ROOT / "src/inclusive_shift_har/data/physical_information_pilot.py"
    )
    assert report["analysis_or_model_run"] is False


def test_prepare_and_offline_validate_write_create_only_pending_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "f" * 40
    monkeypatch.setattr(
        runner,
        "_git_state",
        lambda _: {
            "branch": "test",
            "commit": commit,
            "clean": True,
            "porcelain": [],
        },
    )
    output = tmp_path / ".audit" / "physical_information_pilot" / "synthetic-plan"
    readiness = runner.prepare_plan_run(
        repository_root=ROOT,
        evidence_root=tmp_path,
        config_path=CONFIG_PATH,
        protocol_path=PROTOCOL_PATH,
        schema_path=SCHEMA_PATH,
        output_directory=output,
        code_commit=commit,
    )
    assert readiness["status"] == "plan_valid_collection_pending"
    assert readiness["collection_attempted"] is False
    assert readiness["model_fit_count"] == 0
    validation = runner.validate_plan_run(output)
    assert validation["status"] == "validated_plan_collection_pending"
    assert validation["bout_count"] == 264
    blocker = json.loads((output / "blocker_record.json").read_text(encoding="utf-8"))
    assert blocker["status"] == "collection_not_started_prerequisites_unverified"
    assert blocker["blocking_prerequisites"] == list(PENDING_PREREQUISITES)
    assert blocker["collection_attempted"] is False
    runtime = json.loads((output / "runtime.json").read_text(encoding="utf-8"))
    assert runtime["physical_collection_wall_seconds"] == 0.0
    assert runtime["analysis_compute_wall_seconds"] == 0.0
    assert runtime["model_fit_count"] == 0
    assert (output / "completion_manifest.json").is_file()
    with pytest.raises(ValueError, match="already exists"):
        runner.prepare_plan_run(
            repository_root=ROOT,
            evidence_root=tmp_path,
            config_path=CONFIG_PATH,
            protocol_path=PROTOCOL_PATH,
            schema_path=SCHEMA_PATH,
            output_directory=output,
            code_commit=commit,
        )


def test_plan_validation_cross_binds_lineage_hashes_to_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    commit = "f" * 40
    monkeypatch.setattr(
        runner,
        "_git_state",
        lambda _: {
            "branch": "test",
            "commit": commit,
            "clean": True,
            "porcelain": [],
        },
    )
    output = tmp_path / ".audit" / "physical_information_pilot" / "tampered-lineage"
    runner.prepare_plan_run(
        repository_root=ROOT,
        evidence_root=tmp_path,
        config_path=CONFIG_PATH,
        protocol_path=PROTOCOL_PATH,
        schema_path=SCHEMA_PATH,
        output_directory=output,
        code_commit=commit,
    )
    plan_path = output / "plan_manifest.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["lineage"]["config"]["sha256"] = "0" * 64
    plan["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in plan.items() if key != "record_sha256"}
    )
    plan_path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    artifact_path = output / "artifact_manifest.json"
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    plan_entry = next(
        item for item in artifact["artifacts"] if item["path"] == "plan_manifest.json"
    )
    plan_entry["size_bytes"] = plan_path.stat().st_size
    plan_entry["sha256"] = sha256_file(plan_path)
    artifact["record_sha256"] = canonical_json_sha256(
        {key: value for key, value in artifact.items() if key != "record_sha256"}
    )
    artifact_path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="config hash is not bound"):
        runner.validate_plan_run(output)
