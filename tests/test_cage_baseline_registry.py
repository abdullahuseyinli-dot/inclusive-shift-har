from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml


def test_cage_baseline_registry_separates_information_tracks_and_keeps_gaps_visible() -> None:
    registry = cast(
        dict[str, Any],
        yaml.safe_load(
            Path("configs/experiments/cage_har_baseline_registry_v1.yaml").read_text(
                encoding="utf-8"
            )
        ),
    )
    policy = registry["comparison_policy"]
    assert policy["separate_six_and_nine_channel_leaderboards"] is True
    assert policy["target_or_confirmatory_tuning_allowed"] is False
    assert policy["failed_and_unavailable_baselines_remain_visible"] is True
    tracks = registry["tracks"]
    assert tracks["six_channel"]["channel_count"] == 6
    assert tracks["nine_channel"]["channel_count"] == 9
    six = {item["id"]: item for item in tracks["six_channel"]["methods"]}
    nine = {item["id"]: item for item in tracks["nine_channel"]["methods"]}
    modern = {item["id"]: item for item in tracks["modern_representation"]["methods"]}
    assert {"frozen_rmrp", "random_forest", "compact_dann", "deepconvlstm"} <= set(six)
    assert {"frozen_ctgr", "gravity_posture_expert", "cage_har"} <= set(nine)
    assert {
        "reconstruction_ssl_representative",
        "contrastive_ssl_representative",
        "hybrid_reconstruction_contrastive",
        "wearable_foundation_checkpoint",
    } == set(modern)
    all_methods = [*six.values(), *nine.values(), *modern.values()]
    assert all(
        item["run_status"] == "blocked"
        for item in all_methods
        if item["licence_status"] == "pending_review"
    )
    assert registry["claim_boundary"]["generic_har_state_of_the_art_claim_allowed"] is False
