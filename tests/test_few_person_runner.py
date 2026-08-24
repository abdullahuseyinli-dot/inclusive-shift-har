from __future__ import annotations

from pathlib import Path

import pytest

from inclusive_shift_har.experiments.few_person import (
    FewPersonRunError,
    run_few_person_scenario,
)
from inclusive_shift_har.protocols.few_person import (
    build_few_person_manifest,
    write_few_person_manifest_new,
)
from tests.test_few_person_protocol import materialize_few_person_inputs


def test_neural_runner_refuses_cpu_before_raw_or_checkpoint_access(
    tmp_path: Path,
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    split, receipt, zero, inventory = materialize_few_person_inputs(tmp_path, config_path)
    manifest = build_few_person_manifest(
        split_manifest_path=split,
        config_path=config_path,
        opening_receipt_path=receipt,
        zero_shot_index_path=zero,
        final_freeze_inventory_path=inventory,
    )
    manifest_path = tmp_path / "few-person.json"
    write_few_person_manifest_new(manifest, manifest_path, allowed_root=tmp_path)
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    output = tmp_path / "outputs" / "run"

    with pytest.raises(FewPersonRunError, match="CPU fallback is forbidden"):
        run_few_person_scenario(
            manifest_path=manifest_path,
            split_manifest_path=split,
            opening_receipt_path=receipt,
            zero_shot_index_path=zero,
            final_freeze_inventory_path=inventory,
            raw_csv_path=tmp_path / "must-not-be-read.csv",
            artifact_root=tmp_path,
            fold_id="target_outer_01",
            k=1,
            model_id="compact-erm",
            seed=11,
            code_commit="a" * 40,
            output_directory=output,
            output_root=tmp_path,
        )
    assert not output.exists()


def test_unplanned_k_or_participant_assignment_is_rejected(
    tmp_path: Path, repository_root: Path
) -> None:
    config_path = repository_root / "configs/protocols/few_person_inclusion_curve_v1.yaml"
    split, receipt, zero, inventory = materialize_few_person_inputs(tmp_path, config_path)
    manifest = build_few_person_manifest(
        split_manifest_path=split,
        config_path=config_path,
        opening_receipt_path=receipt,
        zero_shot_index_path=zero,
        final_freeze_inventory_path=inventory,
    )
    scenario = manifest["scenarios"][0]
    scenario["evaluation_subjects"] = scenario["target_inclusion_subjects"]
    manifest.pop("manifest_sha256")
    from inclusive_shift_har.manifests.canonical import canonical_json_sha256

    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "tampered-plan.json"
    write_few_person_manifest_new(manifest, manifest_path, allowed_root=tmp_path)

    with pytest.raises(FewPersonRunError, match="participant sets overlap"):
        run_few_person_scenario(
            manifest_path=manifest_path,
            split_manifest_path=split,
            opening_receipt_path=receipt,
            zero_shot_index_path=zero,
            final_freeze_inventory_path=inventory,
            raw_csv_path=tmp_path / "must-not-be-read.csv",
            artifact_root=tmp_path,
            fold_id="target_outer_01",
            k=1,
            model_id="compact-erm",
            seed=11,
            code_commit="a" * 40,
            output_directory=tmp_path / "outputs" / "run",
            output_root=tmp_path,
        )
