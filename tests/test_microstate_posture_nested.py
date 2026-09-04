from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.experiments.microstate_posture_graph_ablations import (
    _METHODS as ABLATION_METHODS,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import (
    _load_config,
    _read_hashed_record,
    _selection_order,
)
from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def _summary(candidate_id: str, mean: float, lower: float) -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "mean_participant_macro_f1": mean,
        "lower_30_percent_participant_macro_f1": lower,
    }


def _candidate(candidate_id: str, clusters: int, estimator_rank: int) -> dict[str, object]:
    return {
        "id": candidate_id,
        "cluster_count": clusters,
        "estimator_complexity_rank": estimator_rank,
        "complexity_rank": estimator_rank,
    }


def test_mpg_config_is_exactly_locked_to_eighteen_candidates() -> None:
    config = _load_config(Path("configs/experiments/microstate_posture_graph_nested_v1.yaml"))
    assert len(config["candidates"]) == 18
    assert config["selection"]["seed"] == 11
    assert config["fixed_evaluation_seeds"] == [11, 23, 47, 89, 131]
    assert list(config["controls"]) == [
        "flat_rmrp",
        "hierarchical_rmrp",
        "hierarchical_gsp",
        "rmrp_mobility_rist_posture",
    ]
    assert config["claim_policy"]["confirmatory_claim_allowed"] is False
    assert tuple(config["ablations_after_freeze_without_retuning"]) == ABLATION_METHODS[1:]


def test_mpg_selection_applies_mean_tolerance_then_tail_and_simplicity() -> None:
    candidates = [
        _candidate("mean_leader", 8, 3),
        _candidate("tail_winner", 6, 2),
        _candidate("outside_tolerance", 4, 1),
    ]
    summaries = [
        _summary("mean_leader", 0.8000, 0.70),
        _summary("tail_winner", 0.7950, 0.75),
        _summary("outside_tolerance", 0.7949, 0.90),
    ]
    ranking, eligible = _selection_order(summaries, candidates, tolerance=0.005)
    assert ranking[0]["candidate_id"] == "tail_winner"
    assert eligible == ["mean_leader", "tail_winner"]

    equally_strong = [
        _summary("mean_leader", 0.8, 0.75),
        _summary("tail_winner", 0.8, 0.75),
    ]
    ranking, _ = _selection_order(equally_strong, candidates[:2], tolerance=0.005)
    assert ranking[0]["candidate_id"] == "tail_winner"


def test_mpg_protocol_lock_self_hashes_and_binds_files(repository_root: Path) -> None:
    path = repository_root / "results/protocol/microstate_posture_graph_v1.json"
    record = load_json_strict(path)
    claimed = record.pop("record_sha256")
    assert claimed == canonical_json_sha256(record)
    for field in (
        "protocol_document",
        "experiment_config",
        "source_window_manifest",
        "dataset_manifest",
    ):
        reference = record[field]
        assert reference["sha256"] == sha256_file(repository_root / reference["path"])
    assert record["evidence_boundary"]["participants_11_through_20_reopened"] is False
    assert record["evidence_boundary"]["confirmatory_claim_allowed"] is False


def test_hashed_selection_reader_rejects_tampering(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    record = {"status": "selection_frozen_before_outer_evaluation"}
    record["record_sha256"] = canonical_json_sha256(record)
    path.write_text(json.dumps(record), encoding="utf-8")
    assert _read_hashed_record(path)["status"] == "selection_frozen_before_outer_evaluation"
    record["status"] = "tampered"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(ValueError, match="self-hash"):
        _read_hashed_record(path)
