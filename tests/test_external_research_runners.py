"""Synthetic end-to-end runner and inference-isolation regressions.

Training seams are replaced only where the full nested research budget would be
inappropriate for CI; production reports, archive serialization and failures run.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from inclusive_shift_har.data.external_har import (
    SOURCE_RECEIPT_METADATA,
    ExternalHARWindows,
    SourceReceipt,
)
from inclusive_shift_har.data.participant_partitions import build_participant_partition_plan
from inclusive_shift_har.experiments import (
    cross_dataset_har,
    cross_dataset_neural,
    cross_dataset_transfer,
    external_primary_suite,
    har_pmd_stress,
    publication_campaign,
)
from inclusive_shift_har.experiments.external_evidence_validate import validate_run_directory
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _write_synthetic_json_with_hash(
    path: Path,
    value: dict[str, Any],
    *,
    hash_field: str,
) -> None:
    sealed = dict(value)
    sealed[hash_field] = canonical_json_sha256(sealed)
    path.write_text(json.dumps(sealed, sort_keys=True) + "\n", encoding="utf-8")


def _write_synthetic_scientific_child(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    _write_synthetic_json_with_hash(
        directory / "result.json",
        {"status": "SYNTHETIC_RESULT"},
        hash_field="result_payload_sha256_before_serialization",
    )


def _write_synthetic_validation(directory: Path) -> None:
    (directory / "validation.json").write_text(
        json.dumps(
            {
                "status": "DIAGNOSTIC",
                "integrity_passed": True,
                "publication_evidence_ready": False,
                "publication_evidence_ready_for_unqualified_methods": False,
                "diagnostic_contract_passed": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_synthetic_suite_plan(
    directory: Path,
    planned_children: tuple[tuple[str, str], ...] = (("synthetic-child", "scientific_run"),),
) -> dict[str, str]:
    plan_path = directory / "suite_plan.json"
    _write_synthetic_json_with_hash(
        plan_path,
        {
            "schema_version": "1.0.0",
            "protocol_id": "external-har-publication-v4",
            "status": "SYNTHETIC_SUITE_PLAN",
            "artifact_role": "non_run_create_only_orchestration_index",
            "root_run_validation_applicable": False,
            "planned_children": [
                {"relative_directory": name, "artifact_role": role}
                for name, role in planned_children
            ],
        },
        hash_field="record_sha256",
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    return {
        "path": plan_path.name,
        "sha256": sha256_file(plan_path),
        "record_sha256": plan["record_sha256"],
    }


def _write_synthetic_nested_index(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    child = directory / "synthetic-child"
    _write_synthetic_scientific_child(child)
    _write_synthetic_validation(child)
    planned = (("synthetic-child", external_primary_suite._SCIENTIFIC_RUN_ROLE),)
    plan_artifact = _write_synthetic_suite_plan(directory, planned)
    child_bindings = external_primary_suite._orchestration_child_bindings(directory, planned)
    _write_synthetic_json_with_hash(
        directory / "suite_complete.json",
        {
            "schema_version": "1.0.0",
            "status": "DIAGNOSTIC_PROVIDER_PRESEGMENTED_SUITE_COMPLETE",
            "publication_evidence_ready": False,
            "artifact_role": "non_run_create_only_orchestration_index",
            "root_run_validation_applicable": False,
            "child_artifact_bindings": child_bindings,
            "suite_plan_artifact": plan_artifact,
        },
        hash_field="record_sha256",
    )


def _initialize_clean_git_repository(path: Path) -> None:
    path.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=path, check=True)
    (path / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "frozen"], cwd=path, check=True)


def _data(dataset_id: str, classes: int = 3) -> ExternalHARWindows:
    labels = np.tile(np.arange(classes), 12)
    signals = np.random.default_rng(122).normal(size=(len(labels), 128, 6)).astype(np.float32)
    signals[:, :, 0] += labels[:, None] * 3.0
    persons = np.repeat([f"{dataset_id}:p{i:02d}" for i in range(12)], classes)
    participant_roster = sorted(set(persons.tolist()))
    repository_grid_independent = dataset_id != "imu_har_il_v1"
    receipt_metadata: dict[str, Any] = dict(
        SOURCE_RECEIPT_METADATA.get(
            dataset_id,
            {
                "record_url": "https://example.invalid/synthetic-record",
                "dataset_version": "synthetic-test-fixture-v1",
                "dataset_license": "test-only-no-redistribution",
                "permissible_redistribution": "synthetic test fixture only",
                "evidence_role": "test_fixture",
            },
        )
    )
    receipt_identity: dict[str, Any] = {
        "locator": "https://example.invalid/synthetic",
        "member": None,
        "declared_size_bytes": 1,
        "received_size_bytes": 1,
        "computed_sha256": "0" * 64,
    }
    if dataset_id == "fog_star_v3":
        receipt_identity.update(
            {
                "locator": (
                    "https://zenodo.org/api/records/17838806/files/sensor_data.csv/content"
                ),
                "declared_size_bytes": 119_629_580,
                "received_size_bytes": 119_629_580,
                "computed_sha256": (
                    "888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477"
                ),
                "declared_digest_algorithm": "md5",
                "declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
                "computed_declared_digest": "952a37ab147da35e6d4e7a1e9bac44cb",
                "declared_digest_verified": True,
            }
        )
    elif dataset_id == "imu_har_il_v1":
        receipt_identity["locator"] = "https://data.csiro.au/dap/ws/v2/collections/74700/data/1"
    elif dataset_id == "har_pmd_v1":
        receipt_identity["locator"] = (
            "https://zenodo.org/api/records/7939223/files/data_publish.zip/content"
        )
    return ExternalHARWindows(
        dataset_id=dataset_id,
        channel_lane="native-gravity-9ch" if classes == 5 else "derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=signals,
        gravity=np.ones((len(labels), 128, 3), dtype=np.float32),
        labels=labels,
        participant_ids=persons,
        session_ids=np.char.add(persons, ":indoor"),
        trial_ids=np.array([f"{dataset_id}:t{i}" for i in range(len(labels))]),
        window_ids=np.array([f"{dataset_id}:w{i}" for i in range(len(labels))]),
        class_names=("mobility", "sitting", "standing")
        if classes == 3
        else ("stationary", "walking", "crutches", "walker", "manual_wheelchair"),
        receipts=(
            SourceReceipt(
                dataset_id=dataset_id,
                **receipt_identity,
                **receipt_metadata,
            ),
        ),
        boundary_provenance={
            "protocol_id": "external-har-boundary-provenance-v1",
            "source_boundary_unit": "synthetic participant recording",
            "repository_signal_grid_annotation_independent": repository_grid_independent,
            "provider_upstream_annotation_conditioned": not repository_grid_independent,
            "zero_lookahead_streaming_valid": False,
            "evidence_scope": "synthetic test fixture only",
        },
        participant_partition_plan=build_participant_partition_plan(
            dataset_id,
            participant_roster,
            roster_basis="synthetic participant roster frozen before fixture window creation",
        ),
    )


@pytest.mark.parametrize(
    ("module", "extra_args"),
    (
        (cross_dataset_har, ["--dataset", "fog-star"]),
        (cross_dataset_neural, ["--dataset", "fog-star"]),
        (cross_dataset_transfer, []),
    ),
)
def test_direct_mains_capture_launch_before_config_and_preserve_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    extra_args: list[str],
) -> None:
    events: list[str] = []
    output = tmp_path / module.__name__.rsplit(".", 1)[-1]

    def resolve(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        events.append("launch")
        return ({"commit": "0" * 40}, {"files": {}}, {"record_sha256": "a" * 64})

    def read(_path: Path) -> dict[str, Any]:
        events.append("configuration")
        raise RuntimeError("synthetic configuration failure")

    def preserve(**kwargs: Any) -> dict[str, Any]:
        events.append("failure")
        destination = Path(kwargs["output_directory"])
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "failure.json").write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(module, "_resolve_publication_launch_context", resolve)
    monkeypatch.setattr(module, "_git_state", lambda _root: {})
    monkeypatch.setattr(module, "_source_input_manifest", lambda _root: {})
    monkeypatch.setattr(module, "_read_mapping", read)
    monkeypatch.setattr(module, "_write_launch_failure_envelope", preserve)
    arguments = [
        "--repository-root",
        str(tmp_path),
        "--output-directory",
        str(output),
        *extra_args,
    ]
    with pytest.raises(RuntimeError, match="synthetic configuration failure"):
        module.main(arguments)
    assert events == ["launch", "configuration", "failure"]
    assert (output / "failure.json").is_file()


@pytest.mark.parametrize(
    ("module", "loader_name"),
    ((har_pmd_stress, "load_har_pmd_native"),),
)
def test_direct_dataset_main_preserves_pre_writer_acquisition_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: Any,
    loader_name: str,
) -> None:
    events: list[str] = []
    output = tmp_path / "failed-provider"

    def resolve(**_kwargs: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        events.append("launch")
        return ({}, {}, {"record_sha256": "a" * 64})

    def load(**_kwargs: Any) -> None:
        events.append("acquisition")
        raise OSError("provider unavailable")

    def preserve(**kwargs: Any) -> dict[str, Any]:
        events.append("failure")
        destination = Path(kwargs["output_directory"])
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "failure.json").write_text("{}\n", encoding="utf-8")
        return {}

    monkeypatch.setattr(module, "_resolve_publication_launch_context", resolve)
    monkeypatch.setattr(module, "_git_state", lambda _root: {})
    monkeypatch.setattr(module, "_source_input_manifest", lambda _root: {})
    monkeypatch.setattr(module, loader_name, load)
    monkeypatch.setattr(module, "_write_launch_failure_envelope", preserve)
    with pytest.raises(OSError, match="provider unavailable"):
        module.main(
            [
                "--repository-root",
                str(tmp_path),
                "--output-directory",
                str(output),
            ]
        )
    assert events == ["launch", "acquisition", "failure"]
    assert (output / "failure.json").is_file()


def test_neural_runner_serializes_all_seeds_and_uses_label_free_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("synthetic-neural")
    plan = data.participant_partition_plan
    assert plan is not None
    planned = [*plan.participant_roster, "synthetic-neural:p99"]
    data = replace(
        data,
        participant_partition_plan=build_participant_partition_plan(
            data.dataset_id,
            planned,
            roster_basis="synthetic roster including one participant without retained windows",
        ),
    )
    visits: list[set[str]] = []

    def training(
        data: ExternalHARWindows, signals: NDArray[np.float32], **kwargs: Any
    ) -> tuple[NDArray[np.float64], dict[str, Any]]:
        train = set(data.participant_ids[kwargs["training_indices"]])
        valid = set(data.participant_ids[kwargs["validation_indices"]])
        test = set(data.participant_ids[kwargs["evaluation_indices"]])
        assert not train & valid and not train & test and not valid & test
        config = kwargs["config"]
        assert config.disable_cudnn == (config.model_name == "deepconvlstm")
        assert config.mixed_precision == "float16"
        visits.append(test)
        output_path = kwargs["output_path"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"synthetic checkpoint")
        values = signals[kwargs["evaluation_indices"]]
        predicted = np.clip(np.rint(values[:, :, 0].mean(axis=1) / 3.0), 0, 2).astype(int)
        return np.eye(3)[predicted], {
            "synthetic_training_seam": True,
            "training_config": asdict(config),
            "training_config_sha256": canonical_json_sha256(asdict(config)),
            "disable_cudnn": config.disable_cudnn,
            "mixed_precision": config.mixed_precision,
            "device": "cuda",
            "device_type": "cuda",
            "cuda_runtime_version": "synthetic-cuda",
            "cuda_device_name": "synthetic-device",
            "amp_enabled": True,
            "autocast_device_type": "cuda",
            "autocast_dtype": "float16",
            "gradient_scaler_enabled": True,
            "checkpoint": {
                "path": output_path.name,
                "sha256": sha256_file(output_path),
                "size_bytes": output_path.stat().st_size,
            },
        }

    monkeypatch.setattr(cross_dataset_neural, "_train_fold", training)
    monkeypatch.setattr(cross_dataset_neural.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        cross_dataset_neural.torch.cuda, "get_device_name", lambda _device: "synthetic-device"
    )
    monkeypatch.setattr(cross_dataset_neural.torch.version, "cuda", "synthetic-cuda")
    monkeypatch.setattr(
        cross_dataset_neural,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(cross_dataset_neural, "_source_manifest_commit_errors", lambda *_args: [])
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match="publication evidence acceptance gate failed"):
        cross_dataset_neural.run_and_write_neural(
            data=data,
            output_directory=tmp_path / "neural",
            repository_root=root,
            seeds=(11, 23, 47),
            epochs=40,
        )
    result = json.loads((tmp_path / "neural/result.json").read_text(encoding="utf-8"))
    assert len(visits) == 30
    assert len(result["fold_records"]) == 30
    for record in result["fold_records"]:
        assert (
            set(record["training_participants"])
            | set(record["validation_participants"])
            | set(record["evaluation_participants"])
        ) == set(planned)
        assert "synthetic-neural:p99" not in record["training_participants_with_supervision"]
        assert "synthetic-neural:p99" not in record["validation_participants_with_supervision"]
        assert "synthetic-neural:p99" not in record["evaluation_participants_with_candidates"]
        assert "synthetic-neural:p99" not in record["evaluation_participants_with_scoring"]
    with np.load(tmp_path / "neural/predictions.npz") as archive:
        assert "probability__seed-47__TinyHAR-6ch" in archive
    assert (tmp_path / "neural/split_audit.json").is_file()
    assert (tmp_path / "neural/validation.json").is_file()
    validation = validate_run_directory(tmp_path / "neural", root)
    assert validation["integrity_passed"] is True
    with pytest.raises(FileExistsError):
        cross_dataset_neural.run_and_write_neural(
            data=data,
            output_directory=tmp_path / "neural",
            repository_root=root,
            seeds=(11, 23, 47),
            epochs=40,
        )


def test_neural_protocol_fails_before_training_without_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cross_dataset_neural.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        cross_dataset_neural,
        "_train_fold",
        lambda *_args, **_kwargs: pytest.fail("training started without CUDA"),
    )
    with pytest.raises(RuntimeError, match="requires an available CUDA device"):
        cross_dataset_neural.evaluate_neural_controls(
            _data("synthetic-neural"),
            seeds=(11, 23, 47),
            output_directory=tmp_path,
            epochs=40,
        )


@pytest.mark.parametrize("mutation", ("signals", "suffix", "experiment", "epochs"))
def test_neural_protocol_rejects_spoofed_lane_or_budget_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    data = _data("synthetic-neural")
    monkeypatch.setattr(cross_dataset_neural.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        cross_dataset_neural.torch.cuda, "get_device_name", lambda _device: "synthetic-device"
    )
    monkeypatch.setattr(cross_dataset_neural.torch.version, "cuda", "synthetic-cuda")
    monkeypatch.setattr(
        cross_dataset_neural,
        "_train_fold",
        lambda *_args, **_kwargs: pytest.fail("training started with a forged neural lane"),
    )
    kwargs: dict[str, Any] = {
        "signals": data.signals,
        "method_suffix": "6ch",
        "experiment_id": "cross-dataset-har-neural-controls-v1",
        "epochs": 40,
    }
    if mutation == "signals":
        changed = data.signals.copy()
        changed[0, 0, 0] += 1.0
        kwargs["signals"] = changed
    elif mutation == "suffix":
        kwargs["method_suffix"] = "N9"
    elif mutation == "experiment":
        kwargs["experiment_id"] = "forged-neural-result"
    else:
        kwargs["epochs"] = 39
    with pytest.raises((ValueError, RuntimeError)):
        cross_dataset_neural.evaluate_neural_controls(
            data,
            seeds=(11, 23, 47),
            output_directory=tmp_path,
            **kwargs,
        )


def test_zero_shot_runner_keeps_target_labels_out_of_fit_and_preserves_each_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, target = _data("imu_har_il_v1"), _data("fog_star_v3")

    def fit(
        source: ExternalHARWindows, target: ExternalHARWindows, **kwargs: Any
    ) -> tuple[dict[str, NDArray[np.float64]], dict[str, Any]]:
        assert not set(source.participant_ids) & set(target.participant_ids)
        predicted = np.clip(np.rint(target.signals[:, :, 0].mean(axis=1) / 3.0), 0, 2).astype(int)
        probability = np.eye(3)[predicted]
        names = cross_dataset_har._METHODS
        return {name: probability for name in names}, {"synthetic_training_seam": True}

    monkeypatch.setattr(cross_dataset_transfer, "_fit_apply_seed", fit)
    monkeypatch.setattr(
        cross_dataset_transfer,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(cross_dataset_transfer, "_source_manifest_commit_errors", lambda *_args: [])
    with pytest.raises(ValueError, match="recorded participant-split gate failed"):
        cross_dataset_transfer.run_and_write_transfer(
            source=source,
            target=target,
            output_directory=tmp_path / "transfer",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
            n_jobs=1,
            include_classical=True,
        )
    result = json.loads((tmp_path / "transfer/result.json").read_text(encoding="utf-8"))
    assert result["primary_seed_averaged"]["participant_count"] == 12
    assert result["claim_policy"]["target_labels_read_only_after_all_probabilities_fixed"] is False
    assert result["claim_policy"]["target_annotations_preloaded_for_scoring_eligibility"] is True
    assert result["claim_policy"]["target_numeric_scoring_after_full_candidate_prediction"] is True
    assert (tmp_path / "transfer/result.json").is_file()
    assert (tmp_path / "transfer/split_audit.json").is_file()
    assert (tmp_path / "transfer/validation.json").is_file()
    validation = validate_run_directory(tmp_path / "transfer", Path(__file__).resolve().parents[1])
    assert validation["integrity_passed"] is True
    assert (
        validation["publication_evidence_ready"] is False
    )  # Synthetic FoG has no real grid audit.


def test_zero_shot_runner_rejects_wrong_dataset_roles_and_overlapping_rosters(
    tmp_path: Path,
) -> None:
    target = _data("fog_star_v3")
    with pytest.raises(ValueError, match="bound to IMU-HAR-IL source"):
        cross_dataset_transfer.evaluate_zero_shot_transfer(
            _data("wrong_source"),
            target,
            seeds=(11, 23, 47),
            repository_root=tmp_path,
        )

    source = _data("imu_har_il_v1")
    shared = source.participant_ids.copy()
    target = replace(
        target,
        participant_ids=shared,
        participant_partition_plan=build_participant_partition_plan(
            "fog_star_v3",
            shared,
            roster_basis="synthetic deliberately overlapping target roster",
        ),
    )
    with pytest.raises(PermissionError, match="participant rosters overlap"):
        cross_dataset_transfer.evaluate_zero_shot_transfer(
            source,
            target,
            seeds=(11, 23, 47),
            repository_root=tmp_path,
        )


def test_primary_suite_preserves_acquisition_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(**kwargs: Any) -> ExternalHARWindows:
        raise ConnectionError("synthetic access failure")

    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", unavailable)
    monkeypatch.setattr(
        external_primary_suite,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(external_primary_suite, "_source_manifest_commit_errors", lambda *_args: [])
    with pytest.raises(ConnectionError):
        external_primary_suite.run_primary_suite(
            output_root=tmp_path / "suite",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
            epochs=40,
            n_jobs=1,
        )
    failure = json.loads((tmp_path / "suite/suite_failure.json").read_text())
    assert failure["status"] == "FAILED_PRESERVED"
    assert failure["artifact_role"] == "non_run_create_only_orchestration_index"
    assert failure["root_run_validation_applicable"] is False
    assert failure["child_artifact_bindings"] == []
    assert failure["source_input_manifest"]["protocol_id"] == "external-har-publication-v4"
    assert not (tmp_path / "suite/suite_complete.json").exists()


@pytest.mark.parametrize(
    "acceptance",
    (
        {"status": "VALIDATED", "publication_evidence_ready": True},
        {
            "status": "PARTIALLY_VALIDATED_METHODS",
            "publication_evidence_ready_for_unqualified_methods": True,
        },
        {"status": "DIAGNOSTIC", "diagnostic_contract_passed": True},
    ),
)
def test_scientific_child_binding_accepts_only_strong_validation_contracts(
    tmp_path: Path,
    acceptance: dict[str, Any],
) -> None:
    root = tmp_path / "suite"
    child = root / "child"
    root.mkdir()
    _write_synthetic_scientific_child(child)
    validation = {
        "integrity_passed": True,
        "publication_evidence_ready": False,
        "publication_evidence_ready_for_unqualified_methods": False,
        "diagnostic_contract_passed": False,
        **acceptance,
    }
    (child / "validation.json").write_text(
        json.dumps(validation, sort_keys=True) + "\n", encoding="utf-8"
    )
    planned = (("child", external_primary_suite._SCIENTIFIC_RUN_ROLE),)
    bindings = external_primary_suite._orchestration_child_bindings(root, planned)
    assert len(bindings) == 1
    assert bindings[0]["binding_complete"] is True
    external_primary_suite._require_complete_orchestration_children(bindings, planned)


@pytest.mark.parametrize(
    ("mutation", "failed_requirement"),
    (
        ("failure_terminal", "exactly_one_result_and_no_failure"),
        ("tampered_result", "result_self_hash_valid"),
        ("weak_validation", "validation_integrity_and_acceptance_passed"),
        ("contradictory_validation", "validation_integrity_and_acceptance_passed"),
    ),
)
def test_scientific_child_binding_rejects_failed_tampered_or_weak_children(
    tmp_path: Path,
    mutation: str,
    failed_requirement: str,
) -> None:
    root = tmp_path / "suite"
    child = root / "child"
    root.mkdir()
    child.mkdir()
    if mutation == "failure_terminal":
        _write_synthetic_json_with_hash(
            child / "failure.json",
            {"status": "FAILED_PRESERVED"},
            hash_field="failure_payload_sha256_before_serialization",
        )
        _write_synthetic_validation(child)
    else:
        _write_synthetic_json_with_hash(
            child / "result.json",
            {"status": "SYNTHETIC_RESULT"},
            hash_field="result_payload_sha256_before_serialization",
        )
        if mutation == "tampered_result":
            retained = json.loads((child / "result.json").read_text(encoding="utf-8"))
            retained["status"] = "TAMPERED_AFTER_HASH"
            (child / "result.json").write_text(
                json.dumps(retained, sort_keys=True) + "\n", encoding="utf-8"
            )
            _write_synthetic_validation(child)
        elif mutation == "weak_validation":
            (child / "validation.json").write_text(
                json.dumps(
                    {
                        "status": "DIAGNOSTIC",
                        "integrity_passed": False,
                        "diagnostic_contract_passed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
        else:
            (child / "validation.json").write_text(
                json.dumps(
                    {
                        "status": "VALIDATED",
                        "integrity_passed": True,
                        "publication_evidence_ready": True,
                        "publication_evidence_ready_for_unqualified_methods": False,
                        "diagnostic_contract_passed": True,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
    planned = (("child", external_primary_suite._SCIENTIFIC_RUN_ROLE),)
    bindings = external_primary_suite._orchestration_child_bindings(root, planned)
    assert bindings[0]["binding_complete"] is False
    assert bindings[0]["completion_requirements"][failed_requirement] is False
    with pytest.raises(ValueError, match="without every child binding"):
        external_primary_suite._require_complete_orchestration_children(bindings, planned)


@pytest.mark.parametrize(
    "mutation",
    (
        "suite_failure",
        "tampered_suite",
        "absent_nested",
        "non_list_nested",
        "incomplete_nested",
        "missing_plan_binding",
        "tampered_plan",
        "swapped_plan_binding",
        "tampered_nested_result",
        "rehashed_nested_result",
        "deleted_nested_validation",
        "plan_topology_mismatch",
        "forged_nested_binding",
        "contradictory_completion_status",
    ),
)
def test_nested_orchestration_binding_requires_valid_complete_suite_tree(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = tmp_path / "campaign"
    nested = root / "primary_suite"
    nested.mkdir(parents=True)
    child = nested / "child"
    _write_synthetic_scientific_child(child)
    _write_synthetic_validation(child)
    nested_planned = (("child", external_primary_suite._SCIENTIFIC_RUN_ROLE),)
    plan_artifact = _write_synthetic_suite_plan(nested, nested_planned)
    child_bindings = external_primary_suite._orchestration_child_bindings(nested, nested_planned)
    value: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "DIAGNOSTIC_PROVIDER_PRESEGMENTED_SUITE_COMPLETE",
        "publication_evidence_ready": False,
        "artifact_role": "non_run_create_only_orchestration_index",
        "root_run_validation_applicable": False,
        "child_artifact_bindings": child_bindings,
        "suite_plan_artifact": plan_artifact,
    }
    filename = "suite_complete.json"
    hash_field = "record_sha256"
    if mutation == "suite_failure":
        filename = "suite_failure.json"
        hash_field = "failure_payload_sha256_before_serialization"
        value["status"] = "FAILED_PRESERVED"
    elif mutation == "absent_nested":
        value.pop("child_artifact_bindings")
    elif mutation == "non_list_nested":
        value["child_artifact_bindings"] = {"binding_complete": True}
    elif mutation == "incomplete_nested":
        value["child_artifact_bindings"][0]["binding_complete"] = False
    elif mutation == "missing_plan_binding":
        value.pop("suite_plan_artifact")
    elif mutation == "swapped_plan_binding":
        value["suite_plan_artifact"] = {**plan_artifact, "sha256": "f" * 64}
    elif mutation == "plan_topology_mismatch":
        plan_path = nested / "suite_plan.json"
        changed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        changed_plan.pop("record_sha256")
        changed_plan["planned_children"] = [
            {
                "relative_directory": "different-child",
                "artifact_role": external_primary_suite._SCIENTIFIC_RUN_ROLE,
            }
        ]
        _write_synthetic_json_with_hash(plan_path, changed_plan, hash_field="record_sha256")
        changed_plan = json.loads(plan_path.read_text(encoding="utf-8"))
        value["suite_plan_artifact"] = {
            "path": "suite_plan.json",
            "sha256": sha256_file(plan_path),
            "record_sha256": changed_plan["record_sha256"],
        }
    elif mutation == "forged_nested_binding":
        value["child_artifact_bindings"] = [
            {"relative_directory": "phantom", "binding_complete": True}
        ]
    elif mutation == "contradictory_completion_status":
        value["status"] = "FAILED_PRESERVED"
    _write_synthetic_json_with_hash(nested / filename, value, hash_field=hash_field)
    if mutation == "tampered_suite":
        retained = json.loads((nested / filename).read_text(encoding="utf-8"))
        retained["status"] = "TAMPERED_AFTER_HASH"
        (nested / filename).write_text(
            json.dumps(retained, sort_keys=True) + "\n", encoding="utf-8"
        )
    if mutation == "tampered_plan":
        retained_plan = json.loads((nested / "suite_plan.json").read_text(encoding="utf-8"))
        retained_plan["status"] = "TAMPERED_AFTER_HASH"
        (nested / "suite_plan.json").write_text(
            json.dumps(retained_plan, sort_keys=True) + "\n", encoding="utf-8"
        )
    if mutation in {"tampered_nested_result", "rehashed_nested_result"}:
        result_path = child / "result.json"
        retained_result = json.loads(result_path.read_text(encoding="utf-8"))
        retained_result["status"] = "CHANGED_AFTER_NESTED_COMPLETION"
        if mutation == "rehashed_nested_result":
            retained_result.pop("result_payload_sha256_before_serialization")
            _write_synthetic_json_with_hash(
                result_path,
                retained_result,
                hash_field="result_payload_sha256_before_serialization",
            )
        else:
            result_path.write_text(
                json.dumps(retained_result, sort_keys=True) + "\n", encoding="utf-8"
            )
    if mutation == "deleted_nested_validation":
        (child / "validation.json").unlink()
    planned = (("primary_suite", external_primary_suite._ORCHESTRATION_INDEX_ROLE),)
    bindings = external_primary_suite._orchestration_child_bindings(root, planned)
    assert bindings[0]["binding_complete"] is False
    with pytest.raises(ValueError, match="without every child binding"):
        external_primary_suite._require_complete_orchestration_children(bindings, planned)


@pytest.mark.parametrize("policy", ["complete_requested_core", "available_valid_trials"])
def test_primary_suite_routes_cohort_policy_and_orders_acceptance_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: str
) -> None:
    events: list[str] = []
    source, target = _data("imu_har_il_v1"), _data("fog_star_v3")

    def acquire(**kwargs: Any) -> ExternalHARWindows:
        assert kwargs["selection_policy"] == policy
        assert kwargs["repetition_limit"] == 4
        events.append("source")
        return source

    def acquire_target(**kwargs: Any) -> ExternalHARWindows:
        events.append("target_after_source_gates")
        return target

    def runner(stage: str) -> Any:
        def run(**kwargs: Any) -> dict[str, Any]:
            assert kwargs["seeds"] == (11, 23, 47)
            if stage == "transfer":
                assert kwargs["source"] is source and kwargs["target"] is target
            else:
                assert kwargs["data"] is source
            _write_synthetic_scientific_child(kwargs["output_directory"])
            events.append(stage)
            return {"reports": {}, "primary_seed_averaged": {"methods": {}}}

        return run

    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", acquire)
    monkeypatch.setattr(external_primary_suite, "load_fog_star", acquire_target)
    monkeypatch.setattr(
        external_primary_suite,
        "_imu_har_il_expected_cohort_observed",
        lambda _summary, *, selection_policy: selection_policy == policy,
    )
    monkeypatch.setattr(external_primary_suite, "run_and_write", runner("classical"))
    monkeypatch.setattr(external_primary_suite, "run_and_write_neural", runner("neural"))
    monkeypatch.setattr(external_primary_suite, "run_and_write_transfer", runner("transfer"))

    def validate_stage(directory: Path, *_args: Any, **_kwargs: Any) -> None:
        _write_synthetic_validation(directory)
        events.append("gate")

    monkeypatch.setattr(external_primary_suite, "_validate_stage", validate_stage)
    monkeypatch.setattr(
        external_primary_suite,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(external_primary_suite, "_source_manifest_commit_errors", lambda *_args: [])
    result = external_primary_suite.run_primary_suite(
        output_root=tmp_path / "suite",
        repository_root=Path(__file__).resolve().parents[1],
        seeds=(11, 23, 47),
        epochs=40,
        n_jobs=1,
        selection_policy=policy,
    )
    assert result["cohort_selection_policy"] == policy
    assert events == [
        "source",
        "classical",
        "gate",
        "neural",
        "gate",
        "target_after_source_gates",
        "transfer",
        "gate",
    ]
    plan = json.loads((tmp_path / "suite/suite_plan.json").read_text())
    assert plan["cohort_selection_policy"] == policy
    assert plan["artifact_role"] == "non_run_create_only_orchestration_index"
    assert plan["root_run_validation_applicable"] is False
    tag = "available_trials" if policy == "available_valid_trials" else "all_repetitions"
    assert (tmp_path / f"suite/imu_har_il_{tag}_3seed").is_dir()
    complete = json.loads((tmp_path / "suite/suite_complete.json").read_text())
    assert complete["imu_har_il_cohort_guard"]["contract_passed"] is True
    assert complete["imu_har_il_cohort_guard"]["selection_policy"] == policy
    assert complete["suite_plan_artifact"]["sha256"] == sha256_file(
        tmp_path / "suite/suite_plan.json"
    )
    assert complete["suite_plan_artifact"]["record_sha256"] == plan["record_sha256"]
    assert len(complete["child_artifact_bindings"]) == 3
    for binding in complete["child_artifact_bindings"]:
        assert binding["artifact_role"] == "scientific_run"
        assert binding["binding_complete"] is True
        assert binding["terminal_artifacts"][0]["sha256"] == sha256_file(
            tmp_path / "suite" / binding["terminal_artifacts"][0]["path"]
        )
        assert binding["validation_record"]["sha256"] == sha256_file(
            tmp_path / "suite" / binding["validation_record"]["path"]
        )


def test_primary_suite_rejects_unpinned_imu_cohort_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _data("imu_har_il_v1")
    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", lambda **_kwargs: source)
    monkeypatch.setattr(
        external_primary_suite,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(external_primary_suite, "_source_manifest_commit_errors", lambda *_: [])

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("model fitting must not start before the IMU cohort gate")

    monkeypatch.setattr(external_primary_suite, "run_and_write", forbidden)
    with pytest.raises(ValueError, match="exact frozen 50-person provider inventory"):
        external_primary_suite.run_primary_suite(
            output_root=tmp_path / "suite",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
            epochs=40,
            n_jobs=1,
        )
    failure = json.loads((tmp_path / "suite/suite_failure.json").read_text(encoding="utf-8"))
    assert failure["imu_har_il_cohort_guard"]["contract_passed"] is False
    assert failure["imu_har_il_cohort_guard"]["selection_policy"] == ("complete_requested_core")
    assert not (tmp_path / "suite/imu_har_il_all_repetitions_3seed").exists()


def test_primary_suite_inherits_one_real_clean_launch_after_writing_its_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "suite"
    source, target = _data("imu_har_il_v1"), _data("fog_star_v3")
    manifest = {"protocol_id": "external-har-publication-v4", "files": {}}
    contexts: list[dict[str, Any]] = []

    monkeypatch.setattr(external_primary_suite, "_source_input_manifest", lambda _root: manifest)
    monkeypatch.setattr(external_primary_suite, "_source_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(
        external_primary_suite,
        "_read_mapping",
        lambda _path: {
            "preprocessing": {
                "target_sampling_rate_hz": 32.0,
                "window_samples": 128,
                "derived_gravity": {"cutoff_hz": 0.3},
            }
        },
    )
    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", lambda **_kwargs: source)
    monkeypatch.setattr(external_primary_suite, "load_fog_star", lambda **_kwargs: target)
    monkeypatch.setattr(
        external_primary_suite,
        "_imu_har_il_expected_cohort_observed",
        lambda _summary, *, selection_policy: selection_policy == "complete_requested_core",
    )

    def no_compute_writer(**kwargs: Any) -> dict[str, Any]:
        contexts.append(kwargs["inherited_launch_context"])
        directory = Path(kwargs["output_directory"])
        _write_synthetic_scientific_child(directory)
        return {"primary_seed_averaged": {"methods": {}}}

    def validate_stage(directory: Path, *_args: Any, **_kwargs: Any) -> None:
        _write_synthetic_validation(directory)

    monkeypatch.setattr(external_primary_suite, "run_and_write", no_compute_writer)
    monkeypatch.setattr(external_primary_suite, "run_and_write_neural", no_compute_writer)
    monkeypatch.setattr(external_primary_suite, "run_and_write_transfer", no_compute_writer)
    monkeypatch.setattr(external_primary_suite, "_validate_stage", validate_stage)

    external_primary_suite.run_primary_suite(
        output_root=output,
        repository_root=repository,
        seeds=(11, 23, 47),
        epochs=40,
        n_jobs=1,
    )
    assert len(contexts) == 3
    assert len({context["record_sha256"] for context in contexts}) == 1
    plan = json.loads((output / "suite_plan.json").read_text(encoding="utf-8"))
    assert plan["publication_launch_context"]["record_sha256"] == contexts[0]["record_sha256"]


def test_primary_suite_failure_index_binds_every_created_partial_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _data("imu_har_il_v1")
    monkeypatch.setattr(external_primary_suite, "load_imu_har_il", lambda **_kwargs: source)
    monkeypatch.setattr(
        external_primary_suite,
        "_imu_har_il_expected_cohort_observed",
        lambda _summary, *, selection_policy: selection_policy == "complete_requested_core",
    )
    monkeypatch.setattr(
        external_primary_suite,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(external_primary_suite, "_source_manifest_commit_errors", lambda *_args: [])

    def classical(**kwargs: Any) -> dict[str, Any]:
        _write_synthetic_scientific_child(Path(kwargs["output_directory"]))
        return {"primary_seed_averaged": {"methods": {}}}

    def neural_failure(**kwargs: Any) -> dict[str, Any]:
        directory = Path(kwargs["output_directory"])
        directory.mkdir(parents=True)
        _write_synthetic_json_with_hash(
            directory / "failure.json",
            {"status": "FAILED_PRESERVED"},
            hash_field="failure_payload_sha256_before_serialization",
        )
        raise RuntimeError("synthetic neural failure")

    def validate_stage(directory: Path, *_args: Any, **_kwargs: Any) -> None:
        _write_synthetic_validation(directory)

    monkeypatch.setattr(external_primary_suite, "run_and_write", classical)
    monkeypatch.setattr(external_primary_suite, "run_and_write_neural", neural_failure)
    monkeypatch.setattr(external_primary_suite, "_validate_stage", validate_stage)
    with pytest.raises(RuntimeError, match="synthetic neural failure"):
        external_primary_suite.run_primary_suite(
            output_root=tmp_path / "suite",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
            epochs=40,
            n_jobs=1,
        )
    failure = json.loads((tmp_path / "suite/suite_failure.json").read_text(encoding="utf-8"))
    assert failure["suite_plan_artifact"]["sha256"] == sha256_file(
        tmp_path / "suite/suite_plan.json"
    )
    bindings = failure["child_artifact_bindings"]
    assert [item["relative_directory"] for item in bindings] == [
        "imu_har_il_all_repetitions_3seed",
        "imu_har_il_all_repetitions_neural_3seed",
    ]
    assert bindings[0]["binding_complete"] is True
    assert bindings[1]["terminal_artifacts"][0]["status"] == "FAILED_PRESERVED"
    assert bindings[1]["validation_record"] is None
    assert bindings[1]["binding_complete"] is False
    declared = failure.pop("failure_payload_sha256_before_serialization")
    assert declared == canonical_json_sha256(failure)


@pytest.mark.parametrize("diagnostic_contract", [False, True])
def test_primary_suite_accepts_only_contract_validated_provider_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    diagnostic_contract: bool,
) -> None:
    validation = {
        "publication_evidence_ready": False,
        "integrity_passed": True,
        "status": "DIAGNOSTIC",
        "diagnostic_contract_passed": diagnostic_contract,
        "diagnostic_scope_reasons": ["IMU-HAR-IL uses provider label-derived activity segments"],
        "participant_split_audit": {
            "valid": True,
            "status": "PASS_RECORDED_RESULT",
        },
    }

    def validate(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        if not diagnostic_contract:
            raise ValueError("diagnostic evidence acceptance gate failed")
        return validation

    monkeypatch.setattr(external_primary_suite, "validate_and_record_run_directory", validate)
    if diagnostic_contract:
        external_primary_suite._validate_stage(
            tmp_path,
            tmp_path,
            allow_provider_segmented_diagnostic=True,
        )
    else:
        with pytest.raises(ValueError, match="acceptance gate failed"):
            external_primary_suite._validate_stage(
                tmp_path,
                tmp_path,
                allow_provider_segmented_diagnostic=True,
            )


def test_available_campaign_cli_dispatches_explicit_selection_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        publication_campaign,
        "_git_state",
        lambda _root: {"worktree_dirty": False, "commit": "synthetic", "status_entries": []},
    )
    monkeypatch.setattr(publication_campaign, "_source_input_manifest", lambda _root: {"files": {}})
    monkeypatch.setattr(publication_campaign, "_source_manifest_commit_errors", lambda *_args: [])

    def run_suite(**kwargs: Any) -> None:
        calls.append(kwargs)
        _write_synthetic_nested_index(Path(kwargs["output_root"]))

    monkeypatch.setattr(publication_campaign, "run_primary_suite", run_suite)
    assert (
        publication_campaign.main(
            [
                "--dataset",
                "imu-har-il-available",
                "--output",
                str(tmp_path / "campaign"),
                "--repository-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    assert len(calls) == 1 and calls[0]["selection_policy"] == "available_valid_trials"
    assert calls[0]["seeds"] == (11, 23, 47) and calls[0]["epochs"] == 40
    complete_path = tmp_path / "campaign/campaign_complete.json"
    assert complete_path.is_file()
    complete = json.loads(complete_path.read_text(encoding="utf-8"))
    nested = complete["child_artifact_bindings"]
    assert len(nested) == 1 and nested[0]["relative_directory"] == "primary_suite"
    assert nested[0]["artifact_role"] == "non_run_create_only_orchestration_index"
    assert nested[0]["binding_complete"] is True


def test_campaign_children_inherit_launch_despite_create_only_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "fog-campaign"
    data = _data("fog_star_v3")
    manifest = {"protocol_id": "external-har-publication-v4", "files": {}}
    contexts: list[dict[str, Any]] = []
    monkeypatch.setattr(publication_campaign, "_source_input_manifest", lambda _root: manifest)
    monkeypatch.setattr(publication_campaign, "_source_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(publication_campaign, "load_fog_star", lambda: data)
    monkeypatch.setattr(publication_campaign, "_fog_star_full_cohort_observed", lambda _x: True)

    def no_compute_writer(**kwargs: Any) -> None:
        contexts.append(kwargs["inherited_launch_context"])
        directory = Path(kwargs["output_directory"])
        _write_synthetic_scientific_child(directory)

    def accept(directory: Path, *_args: Any, **_kwargs: Any) -> None:
        _write_synthetic_validation(directory)

    monkeypatch.setattr(publication_campaign.cross_dataset_har, "run_and_write", no_compute_writer)
    monkeypatch.setattr(
        publication_campaign.cross_dataset_neural, "run_and_write_neural", no_compute_writer
    )
    monkeypatch.setattr(publication_campaign, "_accept", accept)
    publication_campaign.run_campaign(
        dataset="fog-star", output=output, repository_root=repository, n_jobs=1
    )
    assert len(contexts) == 2
    assert contexts[0]["record_sha256"] == contexts[1]["record_sha256"]
    complete = json.loads((output / "campaign_complete.json").read_text(encoding="utf-8"))
    plan = json.loads((output / "campaign_plan.json").read_text(encoding="utf-8"))
    assert plan["schema_version"] == complete["schema_version"] == "1.0.0"
    assert complete["artifact_role"] == "non_run_create_only_orchestration_index"
    assert complete["root_run_validation_applicable"] is False
    assert [item["relative_directory"] for item in complete["child_artifact_bindings"]] == [
        "classical",
        "neural",
    ]
    assert all(item["binding_complete"] for item in complete["child_artifact_bindings"])


def test_fog_campaign_rejects_a_partial_scored_cohort_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "fog-partial"
    data = _data("fog_star_v3")
    manifest = {"protocol_id": "external-har-publication-v4", "files": {}}
    monkeypatch.setattr(publication_campaign, "_source_input_manifest", lambda _root: manifest)
    monkeypatch.setattr(publication_campaign, "_source_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(publication_campaign, "load_fog_star", lambda: data)

    def unexpected_writer(**_kwargs: Any) -> None:
        raise AssertionError("training must not start for a partial FoG cohort")

    monkeypatch.setattr(publication_campaign.cross_dataset_har, "run_and_write", unexpected_writer)
    with pytest.raises(ValueError, match="exact 22-person provider roster"):
        publication_campaign.run_campaign(
            dataset="fog-star", output=output, repository_root=repository, n_jobs=1
        )
    assert (output / "campaign_plan.json").is_file()
    assert not (output / "classical").exists()


def test_har_pmd_composite_cli_reuses_launch_for_nested_neural_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    _initialize_clean_git_repository(repository)
    output = repository / "results" / "research" / "har-pmd"
    data = _data("har_pmd_v1", 5)
    manifest = {"protocol_id": "external-har-publication-v4", "files": {}}
    contexts: list[dict[str, Any]] = []
    monkeypatch.setattr(har_pmd_stress, "_source_input_manifest", lambda _root: manifest)
    monkeypatch.setattr(har_pmd_stress, "_source_manifest_commit_errors", lambda *_args: [])
    monkeypatch.setattr(har_pmd_stress, "load_har_pmd_native", lambda **_kwargs: data)

    def classical(**kwargs: Any) -> dict[str, Any]:
        contexts.append(kwargs["inherited_launch_context"])
        directory = Path(kwargs["output_directory"])
        directory.mkdir(parents=True)
        (directory / "result.json").write_text("{}\n", encoding="utf-8")
        return {"reports": {}}

    def neural(**kwargs: Any) -> dict[str, Any]:
        contexts.append(kwargs["inherited_launch_context"])
        directory = Path(kwargs["output_directory"])
        directory.mkdir(parents=True)
        (directory / "result.json").write_text("{}\n", encoding="utf-8")
        return {"reports": {}}

    monkeypatch.setattr(har_pmd_stress, "run_and_write", classical)
    monkeypatch.setattr(cross_dataset_neural, "run_and_write_neural", neural)
    assert (
        har_pmd_stress.main(
            [
                "--output-directory",
                str(output),
                "--repository-root",
                str(repository),
                "--with-neural",
            ]
        )
        == 0
    )
    assert len(contexts) == 3
    assert len({context["record_sha256"] for context in contexts}) == 1


def test_har_pmd_runner_has_separate_native_and_six_channel_predictions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = _data("har_pmd_v1", 5)
    # Two observed environments keep the actual environment report path exercised.
    data = replace(
        data,
        session_ids=np.array(
            [
                person + (":indoor" if index % 2 else ":outdoor")
                for index, person in enumerate(data.participant_ids)
            ]
        ),
    )
    calls: list[int] = []

    def fit(
        windows: NDArray[np.float32],
        labels: NDArray[np.int64],
        participants: list[str],
        **kwargs: Any,
    ) -> int:
        calls.append(windows.shape[2])
        return int(windows.shape[2])

    def predict(
        fitted: int, windows: NDArray[np.float32]
    ) -> tuple[NDArray[np.int64], NDArray[np.float64]]:
        assert fitted == windows.shape[2]
        predicted = np.clip(np.rint(windows[:, :, 0].mean(axis=1) / 3.0), 0, 4).astype(np.int64)
        return predicted, np.eye(5)[predicted]

    monkeypatch.setattr(har_pmd_stress, "fit_classical_model", fit)
    monkeypatch.setattr(har_pmd_stress, "predict_classical_probabilities", predict)
    monkeypatch.setattr(
        har_pmd_stress,
        "_git_state",
        lambda _root: {
            "commit": "0" * 40,
            "worktree_dirty": False,
            "status_entries": [],
        },
    )
    monkeypatch.setattr(har_pmd_stress, "_source_manifest_commit_errors", lambda *_args: [])
    with pytest.raises(ValueError, match="publication evidence acceptance gate failed"):
        har_pmd_stress.run_and_write(
            data=data,
            output_directory=tmp_path / "pmd",
            repository_root=Path(__file__).resolve().parents[1],
            seeds=(11, 23, 47),
        )
    result = json.loads((tmp_path / "pmd/result.json").read_text(encoding="utf-8"))
    assert calls.count(6) == calls.count(9) == 30
    assert result["dataset"]["participant_count"] == 12
    assert len(result["reports"]) == 4
    assert len(result["fold_records"]) == 15
    assert "primary_seed_averaged" in result
    assert (tmp_path / "pmd/split_audit.json").is_file()
    assert (tmp_path / "pmd/validation.json").is_file()
