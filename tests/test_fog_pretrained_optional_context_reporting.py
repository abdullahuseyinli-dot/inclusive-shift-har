from __future__ import annotations

import numpy as np
import pytest

from inclusive_shift_har.experiments.fog_pretrained_optional_context_reporting import (
    reporting_strata,
)


def test_reporting_strata_keep_short_missing_and_q_zero_distinct() -> None:
    current = np.asarray([True, True, True, False, False], dtype=np.bool_)
    full = np.asarray([True, False, True, False, False], dtype=np.bool_)
    q_zero = np.asarray([True, False, False, True, False], dtype=np.bool_)
    masks = reporting_strata(current=current, full=full, q_zero=q_zero)
    assert {name: int(mask.sum()) for name, mask in masks.items()} == {
        "all_scored": 5,
        "current_ankle": 3,
        "full_history": 2,
        "short_history_current": 1,
        "missing_current_ankle": 2,
        "q_zero_all_scored": 2,
        "current_q_zero": 1,
        "current_q_nonzero_editable": 2,
        "full_history_q_zero": 1,
        "full_history_q_nonzero": 1,
    }


def test_reporting_strata_reject_full_history_without_current_ankle() -> None:
    with pytest.raises(ValueError, match="full history"):
        reporting_strata(
            current=np.asarray([False], dtype=np.bool_),
            full=np.asarray([True], dtype=np.bool_),
            q_zero=np.asarray([False], dtype=np.bool_),
        )
