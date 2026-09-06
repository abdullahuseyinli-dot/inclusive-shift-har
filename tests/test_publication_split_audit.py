from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.data.external_har import participant_fold_assignment
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.experiments.publication_split_audit import (
    _safe_artifact_path,
    build_split_audit,
    write_split_audit,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

SEEDS = (11, 23, 47)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _summary(
    dataset_id: str,
    roster: list[str],
    *,
    observed: list[str] | None = None,
    window_count: int | None = None,
) -> dict[str, Any]:
    observed_people = roster if observed is None else observed
    plan = build_participant_partition_plan(
        dataset_id,
        roster,
        roster_basis="synthetic provider inventory before windowing",
    ).audit()
    return {
        "dataset_id": dataset_id,
        "participant_count": len(set(observed_people)),
        "window_count": len(observed_people) if window_count is None else window_count,
        "participant_partition_plan": plan,
        "participant_partition_observation": {
            "planned_participant_count": len(set(roster)),
            "observed_window_participant_count": len(set(observed_people)),
            "participants_without_retained_windows": sorted(set(roster) - set(observed_people)),
        },
    }


def _group(participants: list[str], *, count: int, seed: int, fold: int) -> list[str]:
    assignment = participant_fold_assignment(participants, fold_count=count, seed=seed)
    return sorted(person for person, assigned in assignment.items() if assigned == fold)


def _nested_result(participants: list[str], *, observed: list[str] | None = None) -> dict[str, Any]:
    observed_people = set(participants if observed is None else observed)
    summary = _summary("synthetic", participants, observed=observed)
    plan_sha256 = summary["participant_partition_plan"]["plan_sha256"]
    seed_records = []
    for seed in SEEDS:
        outer_records = []
        for outer_fold in range(5):
            evaluation = _group(participants, count=5, seed=seed, fold=outer_fold)
            outer_training = sorted(set(participants) - set(evaluation))
            inner_records = []
            for inner_fold in range(4):
                validation = _group(
                    outer_training,
                    count=4,
                    seed=seed + 10_000 + outer_fold,
                    fold=inner_fold,
                )
                inner_records.append(
                    {
                        "inner_fold": inner_fold,
                        "training_participants": sorted(set(outer_training) - set(validation)),
                        "training_participants_with_supervision": sorted(
                            (set(outer_training) - set(validation)) & observed_people
                        ),
                        "validation_participants": validation,
                        "validation_participants_with_supervision": sorted(
                            set(validation) & observed_people
                        ),
                        "participant_partition_plan_sha256": plan_sha256,
                    }
                )
            outer_records.append(
                {
                    "outer_fold": outer_fold,
                    "training_participants": outer_training,
                    "training_participants_with_supervision": sorted(
                        set(outer_training) & observed_people
                    ),
                    "evaluation_participants": evaluation,
                    "participant_partition_plan_sha256": plan_sha256,
                    "inner_folds": inner_records,
                }
            )
        seed_records.append({"seed": seed, "folds": outer_records})
    return {
        "experiment_id": "synthetic-classical",
        "seeds": list(SEEDS),
        "fold_records": seed_records,
        "reports": {"method": {}},
        "dataset": summary,
    }


def _flat_result(
    participants: list[str], *, group_key: str, groups: tuple[str, ...], validation: bool
) -> dict[str, Any]:
    summary = _summary("synthetic", participants)
    plan_sha256 = summary["participant_partition_plan"]["plan_sha256"]
    records = []
    for seed in SEEDS:
        for outer_fold in range(5):
            evaluation = _group(participants, count=5, seed=seed, fold=outer_fold)
            outer_training = sorted(set(participants) - set(evaluation))
            selected_validation = (
                _group(
                    outer_training,
                    count=4,
                    seed=seed + 20_000 + outer_fold,
                    fold=0,
                )
                if validation
                else []
            )
            training = sorted(set(outer_training) - set(selected_validation))
            for group in groups:
                record: dict[str, Any] = {
                    "seed": seed,
                    "outer_fold": outer_fold,
                    group_key: group,
                    "training_participants": list(training),
                    "training_participants_with_supervision": list(training),
                    "evaluation_participants": list(evaluation),
                    "evaluation_participants_with_scoring": list(evaluation),
                    "participant_partition_plan_sha256": plan_sha256,
                }
                if validation:
                    record["validation_participants"] = list(selected_validation)
                    record["validation_participants_with_supervision"] = list(selected_validation)
                    record["evaluation_participants_with_candidates"] = list(evaluation)
                records.append(record)
    return {
        "experiment_id": "synthetic-flat",
        "seeds": list(SEEDS),
        "fold_records": records,
        "reports": {group: {} for group in groups},
        "dataset": summary,
    }


def _transfer_result(target: list[str], source: list[str]) -> dict[str, Any]:
    source_summary = _summary("imu_har_il_v1", source)
    target_summary = _summary("fog_star_v3", target)
    source_plan_sha256 = source_summary["participant_partition_plan"]["plan_sha256"]
    target_plan_sha256 = target_summary["participant_partition_plan"]["plan_sha256"]
    records = []
    for seed in SEEDS:
        inner = []
        for fold in range(4):
            validation = _group(source, count=4, seed=seed + 10_000, fold=fold)
            inner.append(
                {
                    "inner_fold": fold,
                    "training_participants": sorted(set(source) - set(validation)),
                    "training_participants_with_supervision": sorted(set(source) - set(validation)),
                    "validation_participants": validation,
                    "validation_participants_with_supervision": validation,
                    "participant_partition_plan_sha256": source_plan_sha256,
                }
            )
        records.append(
            {
                "seed": seed,
                "source_participants": source,
                "source_participants_with_supervision": source,
                "target_participants": target,
                "target_participants_with_candidates": target,
                "target_participants_with_scoring": target,
                "source_participant_partition_plan_sha256": source_plan_sha256,
                "target_participant_partition_plan_sha256": target_plan_sha256,
                "inner_folds": inner,
            }
        )
    return {
        "experiment_id": "imu-har-il-to-fog-star-zero-shot-v1",
        "seeds": list(SEEDS),
        "seed_records": records,
        "source_dataset": source_summary,
        "target_dataset": target_summary,
    }


def _shared_har_pmd_result(participants: list[str]) -> dict[str, Any]:
    summary = _summary("har_pmd_v1", participants)
    plan_sha256 = summary["participant_partition_plan"]["plan_sha256"]
    records = []
    for seed in SEEDS:
        for fold in range(5):
            evaluation = _group(participants, count=5, seed=seed, fold=fold)
            records.append(
                {
                    "seed": seed,
                    "outer_fold": fold,
                    "training_participants": sorted(set(participants) - set(evaluation)),
                    "evaluation_participants": evaluation,
                    "participant_partition_plan_sha256": plan_sha256,
                }
            )
    return {
        "experiment_id": "har-pmd-native-interface-stress-v1",
        "seeds": list(SEEDS),
        "fold_records": records,
        "reports": {
            name: {}
            for name in (
                "RandomForest-6ch",
                "RandomForest-N9",
                "XGBoost-6ch",
                "XGBoost-N9",
            )
        },
        "dataset": summary,
    }


def _write_run(root: Path, result: dict[str, Any], participants: list[str]) -> Path:
    root.mkdir()
    count = len(participants)
    prediction = root / "predictions.npz"
    np.savez_compressed(
        prediction,
        labels=np.arange(count, dtype=np.int64) % 3,
        participant_ids=np.asarray(participants),
        session_ids=np.asarray([f"{person}:session" for person in participants]),
        trial_ids=np.asarray([f"{person}:trial" for person in participants]),
        window_ids=np.asarray(
            [f"{person}:window-{index}" for index, person in enumerate(participants)]
        ),
        probability__method=np.full((count, 3), 1.0 / 3.0),
    )
    if "target_dataset" not in result and "dataset" not in result:
        result["dataset"] = {
            "dataset_id": "synthetic",
            "participant_count": len(set(participants)),
            "window_count": count,
        }
    result["prediction_artifact"] = {
        "path": prediction.name,
        "sha256": sha256_file(prediction),
    }
    result["result_payload_sha256_before_serialization"] = canonical_json_sha256(result)
    (root / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return root


def _check(report: dict[str, Any], name: str) -> dict[str, Any]:
    return next(item for item in report["checks"] if item["name"] == name)


def test_auditor_accepts_all_recorded_result_shapes(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    source = [f"source:p{index:02d}" for index in range(12)]
    cases = {
        "classical": _nested_result(participants),
        "neural": _flat_result(
            participants,
            group_key="model",
            groups=("DeepConvLSTM-6ch", "TinyHAR-6ch"),
            validation=True,
        ),
        "posture": _flat_result(
            participants,
            group_key="method",
            groups=("PB-RF-6ch", "PB-HPF-6ch"),
            validation=False,
        ),
        "transfer": _transfer_result(participants, source),
        "pmd_shared": _shared_har_pmd_result(participants),
    }
    expected_kinds = {
        "classical": "classical_nested",
        "neural": "flat_model_folds",
        "posture": "flat_method_folds",
        "transfer": "cross_dataset_transfer",
        "pmd_shared": "har_pmd_shared_outer_folds",
    }
    for name, result in cases.items():
        run = _write_run(tmp_path / name, result, participants)
        report = build_split_audit(run)
        assert report["valid"] is True
        assert report["status"] == "PASS_RECORDED_RESULT"
        assert report["run_kind"] == expected_kinds[name]
        assert report["scope"]["pre_window_partition_execution_order_verified"] is False
        assert report["scope"]["pre_window_partition_plan_record_verified"] is True
        assert (
            canonical_json_sha256(
                {key: value for key, value in report.items() if key != "record_sha256"}
            )
            == report["record_sha256"]
        )
    neural = build_split_audit(tmp_path / "neural")
    assert _check(neural, "cross_method_partition_consistency")["passed"] is True
    assert _check(neural, "deterministic_validation_assignment")["passed"] is True


def test_sole_nested_base_partitions_are_audited_without_qualifying_oracle_reset(
    tmp_path: Path,
) -> None:
    participants = [f"sole:p{index:02d}" for index in range(12)]
    base = _nested_result(participants)
    result: dict[str, Any] = {
        "experiment_id": "sole-harmony-causal-bout-v1",
        "seeds": list(SEEDS),
        "dataset": base["dataset"],
        "base_nested_result": base,
    }
    report = build_split_audit(_write_run(tmp_path / "sole", result, participants))
    assert report["valid"] is True
    assert report["run_kind"] == "sole_harmony_nested_classical"
    assert _check(report, "sole_nested_seeds_match")["passed"] is True
    assert any("camera-bout" in warning for warning in report["warnings"])


def test_har_pmd_without_records_is_explicitly_posthoc(tmp_path: Path) -> None:
    participants = [f"harpmd:{index:03d}" for index in range(1, 13)]
    result: dict[str, Any] = {
        "experiment_id": "har-pmd-native-interface-stress-v1",
        "seeds": list(SEEDS),
    }
    run = _write_run(tmp_path / "pmd", result, participants)
    stored = json.loads((run / "result.json").read_text(encoding="utf-8"))
    stored["dataset"]["dataset_id"] = "har_pmd_v1"
    stored.pop("result_payload_sha256_before_serialization")
    stored["result_payload_sha256_before_serialization"] = canonical_json_sha256(stored)
    (run / "result.json").write_text(
        json.dumps(stored, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_split_audit(run)
    assert report["valid"] is True
    assert report["status"] == "PASS_RECONSTRUCTED_POSTHOC"
    assert report["evidence_origin"] == "posthoc_deterministic_reconstruction"
    assert report["scope"]["recorded_executed_partition_membership_verified"] is False
    assert report["scope"]["deterministic_protocol_reconstruction_valid"] is True
    recorded = _check(report, "recorded_assignment_available")
    assert recorded["passed"] is False and recorded["blocking"] is False
    assert len(report["seed_partitions"]) == 3
    assert all(len(seed["outer_folds"]) == 5 for seed in report["seed_partitions"])


def test_zero_window_plan_member_remains_in_fold_universe(tmp_path: Path) -> None:
    roster = [f"p{index:02d}" for index in range(12)]
    scored = roster[:-1]
    result = _nested_result(roster, observed=scored)
    report = build_split_audit(_write_run(tmp_path / "zero_window", result, scored))
    assert report["valid"] is True
    assert report["prediction_cohort"]["participant_count"] == 11
    assert report["partition_cohort"]["participant_count"] == 12
    assert report["partition_cohort"]["participants_without_retained_windows"] == [roster[-1]]
    assert report["scope"]["pre_window_partition_plan_record_verified"] is True


def test_plan_and_executed_record_hash_tampering_are_blocking(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    result = _nested_result(participants)
    plan = result["dataset"]["participant_partition_plan"]
    first_assignment = plan["records"][0]["assignment"]
    left = participants[0]
    right = next(
        person for person in participants if first_assignment[person] != first_assignment[left]
    )
    first_assignment[left], first_assignment[right] = (
        first_assignment[right],
        first_assignment[left],
    )
    unhashed = {key: value for key, value in plan.items() if key != "plan_sha256"}
    plan["plan_sha256"] = canonical_json_sha256(unhashed)
    result["fold_records"][0]["folds"][0]["participant_partition_plan_sha256"] = "f" * 64
    report = build_split_audit(_write_run(tmp_path / "plan_tampering", result, participants))
    assert report["valid"] is False
    assert _check(report, "prewindow_plan_records_valid")["passed"] is False
    assert _check(report, "executed_partition_records_bind_plan_hash")["passed"] is False


def test_observed_supervision_roster_tampering_is_blocking(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    result = _flat_result(
        participants,
        group_key="model",
        groups=("DeepConvLSTM-6ch", "TinyHAR-6ch"),
        validation=True,
    )
    result["fold_records"][0]["training_participants_with_supervision"].pop()
    report = build_split_audit(_write_run(tmp_path / "subset_tampering", result, participants))
    assert report["valid"] is False
    assert _check(report, "recorded_supervision_rosters_match_plan_observation")["passed"] is False


def test_nested_tampering_detects_coverage_complements_and_inner_containment(
    tmp_path: Path,
) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    result = _nested_result(participants)
    first_outer = result["fold_records"][0]["folds"][0]
    evaluation_person = first_outer["evaluation_participants"][0]
    first_outer["training_participants"].append(evaluation_person)
    first_outer["inner_folds"].pop()
    first_outer["inner_folds"][0]["validation_participants"].append(evaluation_person)
    report = build_split_audit(_write_run(tmp_path / "tampered_nested", result, participants))
    assert report["valid"] is False and report["status"] == "FAIL"
    assert _check(report, "outer_partitions_disjoint")["passed"] is False
    assert _check(report, "outer_training_complement")["passed"] is False
    assert _check(report, "inner_fold_ids_exact")["passed"] is False
    assert _check(report, "inner_contained_in_outer_training")["passed"] is False
    assert _check(report, "inner_partitions_exhaustive")["passed"] is False


def test_flat_tampering_detects_deterministic_and_cross_method_mismatch(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    result = _flat_result(
        participants,
        group_key="model",
        groups=("DeepConvLSTM-6ch", "TinyHAR-6ch"),
        validation=True,
    )
    selected = [
        record
        for record in result["fold_records"]
        if record["seed"] == 11 and record["model"] == "TinyHAR-6ch"
    ]
    left, right = selected[0], selected[1]
    for key in (
        "training_participants",
        "validation_participants",
        "evaluation_participants",
    ):
        left[key], right[key] = right[key], left[key]
    report = build_split_audit(_write_run(tmp_path / "tampered_flat", result, participants))
    assert report["valid"] is False
    assert _check(report, "deterministic_outer_assignment")["passed"] is False
    assert _check(report, "cross_method_partition_consistency")["passed"] is False


def test_transfer_tampering_detects_source_target_overlap(tmp_path: Path) -> None:
    target = [f"target:p{index:02d}" for index in range(12)]
    source = [f"source:p{index:02d}" for index in range(12)]
    result = _transfer_result(target, source)
    result["seed_records"][0]["source_participants"].append(target[0])
    report = build_split_audit(_write_run(tmp_path / "transfer_overlap", result, target))
    assert report["valid"] is False
    assert _check(report, "source_target_disjoint")["passed"] is False
    assert _check(report, "inner_training_complement")["passed"] is False


def test_transfer_plan_hashes_cannot_be_swapped(tmp_path: Path) -> None:
    target = [f"target:p{index:02d}" for index in range(12)]
    source = [f"source:p{index:02d}" for index in range(12)]
    result = _transfer_result(target, source)
    record = result["seed_records"][0]
    (
        record["source_participant_partition_plan_sha256"],
        record["target_participant_partition_plan_sha256"],
    ) = (
        record["target_participant_partition_plan_sha256"],
        record["source_participant_partition_plan_sha256"],
    )
    report = build_split_audit(_write_run(tmp_path / "transfer_plan_swap", result, target))
    assert report["valid"] is False
    assert _check(report, "executed_partition_records_bind_plan_hash")["passed"] is False


def test_transfer_dataset_identity_is_fixed(tmp_path: Path) -> None:
    target = [f"target:p{index:02d}" for index in range(12)]
    source = [f"source:p{index:02d}" for index in range(12)]
    result = _transfer_result(target, source)
    result["source_dataset"]["dataset_id"] = "other_source_v1"
    report = build_split_audit(_write_run(tmp_path / "transfer_identity", result, target))
    assert report["valid"] is False
    assert _check(report, "transfer_dataset_identity")["passed"] is False


def test_create_only_writer_binds_inputs_and_refuses_overwrite(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    run = _write_run(tmp_path / "run", _nested_result(participants), participants)
    output = tmp_path / "audit" / "split_audit.json"
    report = write_split_audit(run, output, repository_root=REPOSITORY_ROOT)
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored == report
    assert stored["inputs"]["run_directory"] == "."
    assert stored["inputs"]["result"]["path"] == "result.json"
    assert stored["inputs"]["predictions"]["path"] == "predictions.npz"
    assert stored["inputs"]["result"]["sha256"] == sha256_file(run / "result.json")
    assert stored["inputs"]["predictions"]["sha256"] == sha256_file(run / "predictions.npz")
    assert (
        canonical_json_sha256(
            {key: value for key, value in stored.items() if key != "record_sha256"}
        )
        == stored["record_sha256"]
    )
    with pytest.raises(FileExistsError):
        write_split_audit(run, output, repository_root=REPOSITORY_ROOT)


def test_split_audit_rejects_in_run_prediction_symlink(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    run = _write_run(tmp_path / "run", _nested_result(participants), participants)
    prediction = run / "predictions.npz"
    physical = run / "physical_predictions.npz"
    prediction.replace(physical)
    try:
        prediction.symlink_to(physical.name)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")
    output = tmp_path / "audit" / "split_audit.json"
    with pytest.raises(ValueError, match="symbolic-link"):
        write_split_audit(run, output, repository_root=REPOSITORY_ROOT)
    assert not output.exists()


def test_split_audit_rejects_prediction_path_with_linked_parent(tmp_path: Path) -> None:
    run = tmp_path / "run"
    physical = run / "physical"
    physical.mkdir(parents=True)
    np.savez_compressed(physical / "predictions.npz", labels=np.asarray([0]))
    alias = run / "alias"
    try:
        alias.symlink_to(physical.name, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    with pytest.raises(ValueError, match="symbolic-link"):
        _safe_artifact_path(run, "alias/predictions.npz")


def test_create_only_split_audit_remains_identical_after_run_relocation(
    tmp_path: Path,
) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    run = _write_run(tmp_path / "original", _nested_result(participants), participants)
    output = run / "split_audit.json"
    original = write_split_audit(run, output, repository_root=REPOSITORY_ROOT)
    relocated = tmp_path / "relocated"
    shutil.copytree(run, relocated)
    reconstructed = build_split_audit(relocated)
    assert reconstructed["inputs"] == original["inputs"]
    assert {
        key: value
        for key, value in reconstructed.items()
        if key not in {"created_at", "record_sha256"}
    } == {
        key: value for key, value in original.items() if key not in {"created_at", "record_sha256"}
    }


def test_split_audit_writer_rejects_wrong_source_root_before_mutation(tmp_path: Path) -> None:
    output = tmp_path / "not-created" / "split_audit.json"
    with pytest.raises(ValueError, match="executed research module"):
        write_split_audit(
            tmp_path / "unread-run",
            output,
            repository_root=tmp_path,
        )
    assert not output.parent.exists()


def test_result_self_hash_failure_is_blocking(tmp_path: Path) -> None:
    participants = [f"p{index:02d}" for index in range(12)]
    run = _write_run(tmp_path / "run", _nested_result(participants), participants)
    result = json.loads((run / "result.json").read_text(encoding="utf-8"))
    result["experiment_id"] = "tampered-after-hash"
    (run / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = build_split_audit(run)
    assert report["valid"] is False
    assert _check(report, "result_payload_self_hash")["passed"] is False
