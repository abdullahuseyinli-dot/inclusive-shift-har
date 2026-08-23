from __future__ import annotations

from collections import Counter
from pathlib import Path

from inclusive_shift_har.data.uci_har import load_uci_har_split
from inclusive_shift_har.manifests.canonical import canonical_json_sha256
from inclusive_shift_har.protocols.uci_source import (
    audit_grouped_source_folds,
    build_grouped_source_folds,
    build_uci_source_protocol_manifest,
)
from tests._uci_synthetic import materialize_synthetic_uci_archive


def test_grouped_folds_are_deterministic_subject_exclusive_and_complete(tmp_path: Path) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    train = load_uci_har_split(archive, split="train")

    first = build_grouped_source_folds(train, n_folds=5, seed=5062)
    second = build_grouped_source_folds(train, n_folds=5, seed=5062)
    report = audit_grouped_source_folds(train, first)

    assert [fold.validation_subject_ids for fold in first] == [
        fold.validation_subject_ids for fold in second
    ]
    assert [fold.validation_window_ids_sha256 for fold in first] == [
        fold.validation_window_ids_sha256 for fold in second
    ]
    validation_counts: Counter[int] = Counter()
    for fold in first:
        assert set(fold.train_subject_ids).isdisjoint(fold.validation_subject_ids)
        assert set(fold.train_indices.tolist()).isdisjoint(fold.validation_indices.tolist())
        validation_counts.update(fold.validation_subject_ids)
    assert validation_counts == Counter({1: 1, 2: 1, 3: 1, 4: 1, 5: 1})
    assert report["status"] == "pass"


def test_protocol_manifest_is_self_hashed_and_excludes_official_test(tmp_path: Path) -> None:
    archive = materialize_synthetic_uci_archive(tmp_path / "uci.zip")
    train = load_uci_har_split(archive, split="train")

    manifest = build_uci_source_protocol_manifest(
        train,
        dataset_manifest_sha256="a" * 64,
        n_folds=5,
        seed=5062,
    )
    recorded_hash = manifest.pop("protocol_sha256")

    assert recorded_hash == canonical_json_sha256(manifest)
    assert manifest["leakage_controls"]["official_test_opened_for_protocol_construction"] is False
    assert manifest["source_pretraining_policy"]["official_test_allowed_for_new_claims"] is False
    assert manifest["class_count_policy"].startswith("fixed_locked")
