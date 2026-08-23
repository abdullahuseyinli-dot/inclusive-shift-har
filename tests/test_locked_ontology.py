"""Tests for locked label tracks and semantic exclusions."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from inclusive_shift_har.protocols.ontology import load_locked_ontology


def test_locked_ontology_keeps_realization_sensitive_mappings_explicit(
    repository_root: Path,
) -> None:
    ontology = load_locked_ontology(repository_root / "configs/ontologies/inclusivehar_v1_1.yaml")

    walking = ontology.labels_for_released_label("Walking")
    assert walking["inclusive_native"] == "walking"
    assert walking["functional_core"] == "mobility"
    assert "cross_source_core_exact" not in walking
    assert ontology.labels_for_released_label("Sitting")["cross_source_core_exact"] == "sitting"
    functional_schema = ontology.runnable_track_schema("functional_core")
    assert functional_schema["class_order"] == ["mobility", "sitting", "standing"]
    assert functional_schema["index_by_class"] == {"mobility": 0, "sitting": 1, "standing": 2}
    assert functional_schema["class_count"] == 3
    assert len(functional_schema["class_schema_sha256"]) == 64
    assert ontology.runnable_track_schema("inclusive_native")["class_count"] == 6


def test_ontology_rejects_walking_as_all_participant_exact_cross_source_class(
    tmp_path: Path, repository_root: Path
) -> None:
    source = repository_root / "configs/ontologies/inclusivehar_v1_1.yaml"
    parsed = yaml.safe_load(source.read_text(encoding="utf-8"))
    parsed["tracks"]["cross_source_core"]["exact_classes"]["walking"] = "walking"
    mutated = tmp_path / "ontology.yaml"
    mutated.write_text(yaml.safe_dump(parsed, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="exact labels changed"):
        load_locked_ontology(mutated)
