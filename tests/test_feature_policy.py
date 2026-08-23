"""Tests for the six-channel, non-sensitive model-input contract."""

from __future__ import annotations

from inclusive_shift_har.manifests.validation import validate_manifest

from ._synthetic import PRIMARY_SIX_CHANNELS, synthetic_dataset_manifest


def test_primary_model_tensor_is_exactly_128_by_six_channels() -> None:
    manifest = synthetic_dataset_manifest()
    assert manifest["feature_policy"]["primary_six_channel_allowlist"] == (PRIMARY_SIX_CHANNELS)
    assert manifest["expected_data"]["model_tensor_shape"] == ["batch", 128, 6]
    assert validate_manifest(manifest).valid


def test_sensitive_attribute_cannot_replace_a_signal_channel() -> None:
    manifest = synthetic_dataset_manifest()
    manifest["feature_policy"]["primary_six_channel_allowlist"][-1] = "disabled"
    result = validate_manifest(manifest)
    assert not result.valid
    assert any(
        issue.code in {"ALLOWLIST_EXCLUSION_OVERLAP", "SENSITIVE_FEATURE"}
        for issue in result.errors
    )


def test_gps_or_location_channel_cannot_enter_model_tensor() -> None:
    manifest = synthetic_dataset_manifest()
    manifest["feature_policy"]["primary_six_channel_allowlist"][-1] = "gpsLatitude"
    result = validate_manifest(manifest)
    assert not result.valid, "GPS/location values must never become model features"
    assert any(
        issue.code
        in {
            "ALLOWLIST_EXCLUSION_OVERLAP",
            "ALLOWLIST_PREFIX_EXCLUSION",
            "GPS_FEATURE",
            "PRIMARY_CHANNEL_PROFILE",
            "SENSITIVE_FEATURE",
        }
        for issue in result.errors
    )


def test_six_arbitrary_non_sensitive_names_do_not_satisfy_exact_allowlist() -> None:
    manifest = synthetic_dataset_manifest()
    manifest["feature_policy"]["primary_six_channel_allowlist"] = [
        f"arbitrary_signal_{index}" for index in range(6)
    ]
    result = validate_manifest(manifest)
    assert not result.valid, "exact_allowlist_only must enforce the six named inertial channels"
    assert any(
        "ALLOWLIST" in issue.code or issue.code == "PRIMARY_CHANNEL_PROFILE"
        for issue in result.errors
    )


def test_required_sensitive_and_location_exclusions_cannot_be_removed() -> None:
    manifest = synthetic_dataset_manifest()
    exclusions = manifest["feature_policy"]["sensitive_and_non_model_exclusions"]
    exclusions["exact_names"].remove("assistive_device")
    exclusions["casefold_prefixes"].remove("gps")
    result = validate_manifest(manifest)
    assert not result.valid
    codes = {issue.code for issue in result.errors}
    assert {"SENSITIVE_EXCLUSIONS", "GPS_EXCLUSIONS"} <= codes


def test_observed_lowercase_label_and_legacy_spelling_are_both_excluded() -> None:
    manifest = synthetic_dataset_manifest()
    exact_names = manifest["feature_policy"]["sensitive_and_non_model_exclusions"]["exact_names"]
    exact_names.remove("label")

    result = validate_manifest(manifest)

    assert not result.valid
    assert any(issue.code == "RAW_LABEL_EXCLUSIONS" for issue in result.errors)
