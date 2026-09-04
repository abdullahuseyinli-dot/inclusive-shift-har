from __future__ import annotations

from pathlib import Path

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)


def test_v2_research_summary_is_self_hashed_and_claim_bounded(
    repository_root: Path,
) -> None:
    path = repository_root / "results/development/fuse_reframe_v2/research_summary_v1.json"
    record = load_json_strict(path)
    claimed_hash = record.pop("record_sha256")
    assert claimed_hash == canonical_json_sha256(record)

    assert record["repository_scope"].startswith("inclusive_shift_har_smartphone_wearable_imu")
    assert record["protocol"]["inclusivehar_target_participants_reopened"] is False
    assert (
        record["robust_multiscale_residual_pyramid"]["mean_participant_macro_f1"]
        == 0.8379032940215063
    )
    assert (
        record["robust_multiscale_residual_pyramid"][
            "bottom_30_percent_participant_macro_f1"
        ]
        == 0.6942572354745962
    )
    assert (
        record["paired_gsp_to_rmrp"]["bottom_30_percent_participant_macro_f1_delta"]
        == 0.017127251515686948
    )
    assert (
        record["robust_multiscale_residual_pyramid"]["publication_method_description"]
        == "denoised_geometric_spectral_pyramid_selected_from_rmrp_family"
    )
    assert record["paired_gsp_to_rmrp"]["engineering_advance_gate_passed"] is False
    assert record["claim_policy"]["breakthrough_confirmed"] is False
    assert record["claim_policy"]["new_zero_shot_target_claim_allowed"] is False
    assert record["daghar_external"]["frozen_candidate_changed_after_evaluation"] is False
    assert record["har_pmd"]["status"] == "not_downloaded_not_evaluated"


def test_v2_publication_narrative_files_are_present(repository_root: Path) -> None:
    assert (repository_root / "docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md").is_file()
    assert (repository_root / "paper/FUSE_REFRAME_V2_ADDENDUM.md").is_file()


def test_v2_research_evidence_references_match_local_audit(repository_root: Path) -> None:
    summary = load_json_strict(
        repository_root / "results/development/fuse_reframe_v2/research_summary_v1.json"
    )
    report = (
        repository_root / "docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md"
    ).read_text(encoding="utf-8")
    paths = {
        "gsp_nested": ".audit/v2/geometric-pyramid/nested-001/result.json",
        "rmrp_nested": ".audit/v2/robust-multiscale/publication-001/result.json",
        "gsp_corruptions": ".audit/v2/geometric-pyramid/corruptions-002/result.json",
        "rmrp_corruptions": ".audit/v2/robust-multiscale/publication-corruptions-001/result.json",
        "gsp_semantic_anchors": ".audit/v2/adaptation/semantic-anchor-001/result.json",
        "rmrp_semantic_anchors": ".audit/v2/adaptation/semantic-anchor-robust-001/result.json",
        "daghar_freeze": ".audit/v2/external/daghar-freeze-001/freeze.json",
        "daghar_opening": ".audit/v2/external/daghar_external_opening_v1.json",
        "daghar_heldout": ".audit/v2/external/daghar-heldout-001/result.json",
    }
    local_audit_available = (repository_root / ".audit/v2").is_dir()
    for name, relative_path in paths.items():
        reference = summary["audit_artifacts"][name]
        assert reference["record_sha256"] in report
        assert reference["file_sha256"] in report
        path = repository_root / relative_path
        if not local_audit_available:
            continue
        assert path.is_file(), f"local audit package is missing {relative_path}"
        record = load_json_strict(path)
        claimed_hash = record.pop("record_sha256")
        assert claimed_hash == canonical_json_sha256(record)
        assert reference["record_sha256"] == claimed_hash
        assert reference["file_sha256"] == sha256_file(path)
