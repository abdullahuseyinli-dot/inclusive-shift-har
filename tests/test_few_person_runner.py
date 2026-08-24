from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.experiments.few_person import (
    FewPersonRunError,
    run_few_person_scenario,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from tests.test_few_person_v1_1 import _materialize_v1_1_plan


def test_neural_runner_refuses_cpu_before_raw_or_checkpoint_access(
    tmp_path: Path,
    repository_root: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, manifest_path, _ = _materialize_v1_1_plan(tmp_path, repository_root)
    split = tmp_path / "split.json"
    receipt = tmp_path / "receipt.json"
    zero = tmp_path / "zero.json"
    inventory = tmp_path / "freeze.json"
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
    _, manifest_path, manifest = _materialize_v1_1_plan(tmp_path, repository_root)
    split = tmp_path / "split.json"
    receipt = tmp_path / "receipt.json"
    zero = tmp_path / "zero.json"
    inventory = tmp_path / "freeze.json"
    scenario = manifest["scenarios"][0]
    scenario["evaluation_subjects"] = scenario["target_inclusion_subjects"]
    manifest.pop("manifest_sha256")
    manifest["manifest_sha256"] = canonical_json_sha256(manifest)
    manifest_path = tmp_path / "tampered-plan.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FewPersonRunError, match="differs from pre-opening split"):
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
