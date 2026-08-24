from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

import inclusive_shift_har.experiments.ccil_bpd_postconfirmatory as runner
from inclusive_shift_har.evaluation._strict_config import StrictConfigError
from inclusive_shift_har.evaluation.paper_adaptation_reporting import (
    PaperAdaptationAggregationError,
    _apply_target_multiplicity,
    _average_seed_reports,
    _comparison,
    _resolve_file,
)
from inclusive_shift_har.experiments.ccil_bpd_postconfirmatory import (
    COMPARATOR_IDS,
    METHOD_IDS,
    SEED_ORDER,
    AdaptationCandidate,
    PaperAdaptationRunError,
    load_paper_adaptation_config,
    reconstruct_paper_adaptation_checkpoint,
    run_paper_adaptation_extension,
    select_source_candidate,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.baselines import CompactResidualHAR
from inclusive_shift_har.models.common import HAROutput
from inclusive_shift_har.models.paper_adaptations import (
    BPDBoundarySafeCompactAdapter,
    BPDMINEstimator,
    mine_dv_lower_bound,
    redundant_class_confusion_loss,
)

CONFIG_RELATIVE = Path("configs/experiments/ccil_bpd_postconfirmatory_v1.yaml")


def test_aggregator_accepts_cli_path_objects_on_windows(tmp_path: Path) -> None:
    index = tmp_path / "nested" / "index.json"
    index.parent.mkdir()
    index.write_text("{}\n", encoding="utf-8")

    resolved = _resolve_file(
        Path("nested") / "index.json",
        root=tmp_path.resolve(),
        name="synthetic index",
        allow_absolute=True,
    )

    assert resolved == index.resolve()


def test_config_locks_source_only_adapters_comparators_and_four_test_family(
    repository_root: Path,
) -> None:
    config = load_paper_adaptation_config(repository_root / CONFIG_RELATIVE)

    assert tuple(config.candidates) == METHOD_IDS
    assert all(len(config.candidates[method_id]) == 4 for method_id in METHOD_IDS)
    assert tuple(config.comparator["model_ids"]) == COMPARATOR_IDS
    assert config.execution["required_device"] == "cuda"
    assert tuple(config.execution["required_seed_order"]) == SEED_ORDER
    assert config.selection["fit_partition"] == "source_train"
    assert config.selection["scoring_partition"] == "source_validation"
    assert config.selection["target_signals_labels_predictions_or_metrics_allowed"] is False
    assert config.statistics["comparison_references"] == list(COMPARATOR_IDS)
    assert (
        config.statistics["multiple_comparison_correction"]
        == "holm_across_four_adapter_vs_locked_comparator_target_comparisons"
    )
    assert all(method["official_code_used"] is False for method in config.methods)
    assert config.methods[1]["third_party_code_copied"] is False


def test_config_duplicate_or_qualifier_drift_is_rejected(
    repository_root: Path, tmp_path: Path
) -> None:
    source = (repository_root / CONFIG_RELATIVE).read_text(encoding="utf-8")
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(source + '\nstatus: "changed"\n', encoding="utf-8")
    with pytest.raises(StrictConfigError, match="duplicate key"):
        load_paper_adaptation_config(duplicate)

    changed = tmp_path / "changed.yaml"
    changed.write_text(
        source.replace(
            "not official-faithful BPD",
            "locally reproduced BPD",
            1,
        ),
        encoding="utf-8",
    )
    with pytest.raises(StrictConfigError, match="required qualifier"):
        load_paper_adaptation_config(changed)


def test_bpd_adapter_uses_exact_compact_generator_features_and_content_only_logits() -> None:
    torch.manual_seed(19)
    compact = CompactResidualHAR(3, width=96, blocks=5).eval()
    torch.manual_seed(19)
    adaptation = BPDBoundarySafeCompactAdapter(3, backbone_width=96, backbone_blocks=5).eval()
    signals = torch.randn(3, 128, 6)

    with torch.no_grad():
        compact_output = compact(signals)
        decomposed = adaptation.decompose(signals)
        inference = adaptation(signals)

    assert isinstance(compact_output, HAROutput)
    assert compact_output.content is not None
    assert torch.equal(compact_output.content, decomposed.backbone_features)
    assert decomposed.activity_features.shape == (3, 24)
    assert decomposed.redundant_features.shape == (3, 24)
    assert decomposed.reconstructed_backbone_features.shape == (3, 96)
    assert torch.equal(inference.logits, decomposed.activity_logits)
    assert inference.content is not None
    assert torch.equal(inference.content, decomposed.activity_features)


def test_bpd_mine_and_confusion_objectives_are_differentiable_and_fail_closed() -> None:
    estimator = BPDMINEstimator(4)
    activity = torch.randn(5, 4, requires_grad=True)
    redundant = torch.randn(5, 4, requires_grad=True)
    permutation = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
    bound = mine_dv_lower_bound(estimator, activity, redundant, permutation=permutation)
    bound.backward()  # type: ignore[no-untyped-call]
    assert activity.grad is not None
    assert redundant.grad is not None
    assert all(parameter.grad is not None for parameter in estimator.parameters())

    logits = torch.randn(5, 3, requires_grad=True)
    confusion = redundant_class_confusion_loss(logits)
    confusion.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    with pytest.raises(ValueError, match="every batch index"):
        mine_dv_lower_bound(
            estimator,
            activity.detach(),
            redundant.detach(),
            permutation=torch.tensor([0, 0, 1, 2, 3], dtype=torch.long),
        )


def test_source_candidate_selection_uses_mean_worst_then_lexicographic() -> None:
    candidates = tuple(
        AdaptationCandidate(METHOD_IDS[0], candidate_id, {"alpha": 0.1})
        for candidate_id in ("zeta", "alpha", "middle")
    )
    reports = {
        "zeta": {
            "primary": {
                "mean_participant_macro_f1": 0.7,
                "worst_participant_macro_f1": 0.5,
            }
        },
        "alpha": {
            "primary": {
                "mean_participant_macro_f1": 0.7,
                "worst_participant_macro_f1": 0.6,
            }
        },
        "middle": {
            "primary": {
                "mean_participant_macro_f1": 0.7,
                "worst_participant_macro_f1": 0.6,
            }
        },
    }
    assert select_source_candidate(candidates, reports).candidate_id == "alpha"


def test_cuda_gate_precedes_config_cache_git_and_output_access(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    missing = tmp_path / "missing"
    with pytest.raises(PaperAdaptationRunError, match="before configuration or cache access"):
        run_paper_adaptation_extension(
            config_path=missing,
            repository_root=missing,
            primary_cache_record_path=missing,
            expected_primary_cache_record_file_sha256="a" * 64,
            expected_code_commit="b" * 40,
            created_at_utc="2099-01-01T00:00:00Z",
            device=torch.device("cuda"),
        )
    assert not list(tmp_path.iterdir())


def test_source_lock_is_validated_before_target_cache_loader(
    monkeypatch: pytest.MonkeyPatch, repository_root: Path, tmp_path: Path
) -> None:
    config = load_paper_adaptation_config(repository_root / CONFIG_RELATIVE)
    lock_path = tmp_path / "source-stage-lock.json"
    payload: dict[str, Any] = {
        "target_signal_arrays_loaded": True,
        "target_labels_predictions_or_metrics_used_for_selection": False,
    }
    payload["record_sha256"] = canonical_json_sha256(payload)
    lock_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")

    def forbidden_cache_load(**_: Any) -> None:
        raise AssertionError("target cache must not load after source-lock validation fails")

    monkeypatch.setattr(runner, "load_prepared_primary_cache", forbidden_cache_load)
    with pytest.raises(PaperAdaptationRunError, match="source-stage lock does not validate"):
        runner._load_target_cache_after_source_lock(
            source_lock={
                "path": lock_path.relative_to(tmp_path).as_posix(),
                "file_sha256": sha256_file(lock_path),
                "record_sha256": payload["record_sha256"],
            },
            config=config,
            root=tmp_path,
            cache_record_path=tmp_path / "not-accessed.json",
            expected_cache_record_file_sha256="a" * 64,
        )


def test_bpd_checkpoint_reconstruction_is_hash_and_configuration_bound(
    repository_root: Path, tmp_path: Path
) -> None:
    config = load_paper_adaptation_config(repository_root / CONFIG_RELATIVE)
    candidate = config.candidates[METHOD_IDS[1]][0]
    configuration = runner._training_configuration(
        config,
        candidate,
        seed=SEED_ORDER[0],
        role="synthetic_test",
        primary_cache_record_sha256="a" * 64,
        primary_cache_record_file_sha256="b" * 64,
    )
    configuration_sha = canonical_json_sha256(configuration)
    model = BPDBoundarySafeCompactAdapter(3, backbone_width=96, backbone_blocks=5)
    payload = {
        "schema_version": runner.SCHEMA_VERSION,
        "record_kind": "postconfirmatory_paper_adaptation_checkpoint",
        "status": "complete_create_only",
        "evidence_status": runner.EVIDENCE_STATUS,
        "method_id": METHOD_IDS[1],
        "seed": SEED_ORDER[0],
        "configuration": configuration,
        "configuration_sha256": configuration_sha,
        "code_commit": "c" * 40,
        "label_schema": list(runner.CLASS_NAMES),
        "split_manifest_sha256": runner.EXPECTED_SPLIT_SHA256,
        "source_artifact_sha256": runner.EXPECTED_SOURCE_SHA256,
        "primary_cache_record_sha256": "a" * 64,
        "primary_cache_record_file_sha256": "b" * 64,
        "target_information_used": False,
        "checkpoint_selection_rule": "fixed_last_epoch",
        "model_state": model.state_dict(),
    }
    checkpoint = tmp_path / "checkpoint.pt"
    with checkpoint.open("xb") as stream:
        torch.save(payload, stream)
    rebuilt, loaded = reconstruct_paper_adaptation_checkpoint(
        checkpoint,
        expected_sha256=sha256_file(checkpoint),
        expected_method_id=METHOD_IDS[1],
        expected_seed=SEED_ORDER[0],
        expected_configuration_sha256=configuration_sha,
        expected_code_commit="c" * 40,
        device=torch.device("cpu"),
    )
    assert isinstance(rebuilt, BPDBoundarySafeCompactAdapter)
    assert loaded["target_information_used"] is False
    with pytest.raises(PaperAdaptationRunError, match="configuration"):
        reconstruct_paper_adaptation_checkpoint(
            checkpoint,
            expected_sha256=sha256_file(checkpoint),
            expected_method_id=METHOD_IDS[1],
            expected_seed=SEED_ORDER[0],
            expected_configuration_sha256="d" * 64,
            expected_code_commit="c" * 40,
            device=torch.device("cpu"),
        )


def _participant_report(seed_offset: float) -> dict[str, Any]:
    return {
        "participant_count": 2,
        "participants": [
            {
                "participant_id": "8",
                "macro_f1": 0.4 + seed_offset,
                "balanced_accuracy": 0.5 + seed_offset,
            },
            {
                "participant_id": "10",
                "macro_f1": 0.6 + seed_offset,
                "balanced_accuracy": 0.7 + seed_offset,
            },
        ],
        "calibration": {
            "negative_log_likelihood": 0.9,
            "multiclass_brier_score": 0.3,
            "ece": 0.1,
        },
    }


def test_statistics_average_within_participant_then_compare_to_each_locked_reference() -> None:
    reports = {seed: _participant_report(index / 100.0) for index, seed in enumerate(SEED_ORDER)}
    compact = _average_seed_reports(reports, expected_participants=frozenset({"8", "10"}))
    more_reports = {
        seed: _participant_report(0.05 + index / 100.0) for index, seed in enumerate(SEED_ORDER)
    }
    more = _average_seed_reports(more_reports, expected_participants=frozenset({"8", "10"}))
    candidate_reports = {
        seed: _participant_report(0.08 + index / 100.0) for index, seed in enumerate(SEED_ORDER)
    }
    candidate = _average_seed_reports(
        candidate_reports, expected_participants=frozenset({"8", "10"})
    )

    compact_comparison = _comparison(compact, candidate, reference_id="compact-erm")
    more_comparison = _comparison(more, candidate, reference_id="more-har-full")
    assert compact_comparison["reference_model_id"] == "compact-erm"
    assert more_comparison["reference_model_id"] == "more-har-full"
    assert (
        compact_comparison["candidate_minus_reference"]["candidate_minus_reference_mean"]
        > more_comparison["candidate_minus_reference"]["candidate_minus_reference_mean"]
        > 0
    )
    comparison_matrix = {
        method_id: {
            "compact-erm": _comparison(compact, candidate, reference_id="compact-erm"),
            "more-har-full": _comparison(more, candidate, reference_id="more-har-full"),
        }
        for method_id in METHOD_IDS
    }
    order = _apply_target_multiplicity(comparison_matrix)
    assert order == tuple(
        (method_id, reference_id) for method_id in METHOD_IDS for reference_id in COMPARATOR_IDS
    )
    for method_id, reference_id in order:
        entry = comparison_matrix[method_id][reference_id]
        assert entry["holm_family_size"] == 4
        assert 0 <= entry["holm_adjusted_permutation_p_value"] <= 1
        assert 0 <= entry["holm_adjusted_wilcoxon_p_value"] <= 1
    incomplete = deepcopy(comparison_matrix)
    incomplete[METHOD_IDS[0]].pop("more-har-full")
    with pytest.raises(PaperAdaptationAggregationError, match="exact comparator family"):
        _apply_target_multiplicity(incomplete)
    unexpected = deepcopy(comparison_matrix)
    unexpected["unplanned-adapter"] = deepcopy(unexpected[METHOD_IDS[0]])
    with pytest.raises(PaperAdaptationAggregationError, match="exact adaptation family"):
        _apply_target_multiplicity(unexpected)


def test_cli_has_no_raw_unlock_or_new_opening_interface() -> None:
    help_text = runner.build_parser().format_help().casefold()
    assert "--primary-cache-record" in help_text
    assert "--expected-primary-cache-record-file-sha256" in help_text
    assert "--raw-csv" not in help_text
    assert "--unlock-record" not in help_text
    assert "acknowledge" not in help_text
