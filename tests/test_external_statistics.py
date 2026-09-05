from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.data.external_har import ExternalHARWindows, SourceReceipt
from inclusive_shift_har.evaluation.external_statistics import seed_evidence


def test_seed_average_is_not_probability_ensemble_and_missing_classes_remain_visible() -> None:
    data = ExternalHARWindows(
        dataset_id="fog_star_v3",
        channel_lane="derived-gravity-9ch",
        sampling_rate_hz=50.0,
        signals=np.zeros((3, 128, 6), dtype=np.float32),
        gravity=np.zeros((3, 128, 3), dtype=np.float32),
        labels=np.array([0, 1, 2]),
        participant_ids=np.array(["p1", "p2", "p3"]),
        session_ids=np.array(["s1", "s2", "s3"]),
        trial_ids=np.array(["t1", "t2", "t3"]),
        window_ids=np.array(["w1", "w2", "w3"]),
        receipts=(SourceReceipt("synthetic", "https://example.invalid", None, 1, 1, "0" * 64),),
    )
    correct = np.eye(3, dtype=np.float64)
    incorrect = np.roll(correct, 1, axis=1)
    report, archive = seed_evidence(
        data,
        {
            11: {"XGBoost-6ch": correct, "HERA-DG-full": correct},
            23: {"XGBoost-6ch": incorrect, "HERA-DG-full": correct},
            47: {"XGBoost-6ch": incorrect, "HERA-DG-full": correct},
        },
    )
    assert report["perfect_prediction_fixed_class_mean_ceiling"] == pytest.approx(1 / 3)
    base = report["methods"]["XGBoost-6ch"]
    assert base["mean_participant_macro_f1"] == pytest.approx(1 / 9)
    assert base["present_class_sensitivity_mean"] == pytest.approx(1 / 3)
    assert base["complete_class_participant_count"] == 0
    assert base["complete_class_participant_sensitivity_mean"] is None
    contrast = report["comparisons_vs_xgboost_6ch"]["HERA-DG-full"]
    assert contrast["rescue_count"] == 3
    assert contrast["harm_count"] == contrast["tie_count"] == 0
    assert contrast["primary_contrast"] is True
    assert len(archive) == 6
    transfer, _ = seed_evidence(
        data,
        {11: {"XGBoost-6ch": incorrect, "HERA-DG-full": correct}},
        primary_contrast_eligible=False,
    )
    assert transfer["comparisons_vs_xgboost_6ch"]["HERA-DG-full"]["primary_contrast"] is False
