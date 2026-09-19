from __future__ import annotations

import numpy as np

from inclusive_shift_har.experiments.harth_published_replication import (
    MERGED_INDEX,
    PUBLISHED_LABELS,
    _metrics,
    _official_features,
)


def test_official_features_have_fixed_161_dimensions() -> None:
    rng = np.random.default_rng(11)
    windows = rng.normal(size=(2, 250, 6))
    result = _official_features(windows)
    assert result.shape == (2, 161)
    assert np.isfinite(result).all()


def test_merge_has_all_twelve_provider_labels() -> None:
    assert set(MERGED_INDEX) == set(PUBLISHED_LABELS)


def test_metrics_keep_fixed_class_support() -> None:
    confusion = np.eye(3, dtype=np.int64)
    report = _metrics(confusion, ("a", "b", "c"))
    assert report["accuracy"] == 1.0
    assert report["macro_f1"] == 1.0
    assert report["per_class"]["b"]["support"] == 1
