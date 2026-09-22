"""Protect published metric identity and aggregation during figure reproduction."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from inclusive_shift_har.artifacts.research_figures import load_evidence, verify_metrics
from inclusive_shift_har.manifests.canonical import canonical_json_sha256

ROOT = Path(__file__).resolve().parents[1]


def test_tracked_research_records_reconstruct() -> None:
    lineage, _ = load_evidence(ROOT)
    assert len(lineage["methods"]) == 10


def test_pooled_f1_cannot_replace_participant_endpoint() -> None:
    lineage, _ = load_evidence(ROOT)
    metrics = dict(lineage["methods"][0]["metrics"])
    metrics["mean_participant_macro_f1"] = metrics["pooled_macro_f1"]
    with pytest.raises(ValueError, match="mean_participant_macro_f1"):
        verify_metrics(metrics, windows=725, participants=10)


@pytest.mark.parametrize("corruption", ["reorder", "duplicate_seed", "unhashed_edit"])
def test_aggregate_identity_and_integrity(tmp_path: Path, corruption: str) -> None:
    lineage, current = load_evidence(ROOT)
    destination = tmp_path / "results/research"
    destination.mkdir(parents=True)
    (destination / "source_development_lineage_v1.json").write_text(json.dumps(lineage))
    current["source_five_seed"]["results"].reverse()
    if corruption == "duplicate_seed":
        rows = current["source_five_seed"]["results"][0]["by_seed"]
        rows[0]["seed"] = rows[1]["seed"]
    if corruption != "unhashed_edit":
        current.pop("record_sha256")
        current["record_sha256"] = canonical_json_sha256(current)
    (destination / "reported_metrics_audit_v1.json").write_text(json.dumps(current))
    if corruption == "reorder":
        load_evidence(tmp_path)
    else:
        with pytest.raises(ValueError, match=r"seeds|hash mismatch"):
            load_evidence(tmp_path)
