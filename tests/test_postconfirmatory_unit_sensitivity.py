from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inclusive_shift_har.evaluation.postconfirmatory_unit_sensitivity import (
    PostconfirmatoryUnitSensitivityError,
    build_parser,
    create_postconfirmatory_unit_sensitivity_record,
)
from inclusive_shift_har.experiments.postconfirmatory_cache import (
    prepare_postconfirmatory_primary_caches,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from tests.test_postconfirmatory_cache import _fixture


def _prepared_inputs(tmp_path: Path) -> dict[str, Any]:
    paths = _fixture(tmp_path)
    cache_record = prepare_postconfirmatory_primary_caches(
        split_manifest_path=paths["split"],
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        raw_csv_path=paths["raw"],
        artifact_root=tmp_path,
        cache_directory="primary-v1",
        cache_root=paths["cache_root"],
        record_output="primary-cache.json",
        record_root=paths["record_root"],
        code_commit="a" * 40,
        created_at_utc="2099-01-01T00:00:00Z",
    )
    cache_record_path = paths["record_root"] / "primary-cache.json"
    source_config = (
        Path(__file__).parents[1] / "configs" / "preprocessing" / "inclusivehar_primary_si_128.yaml"
    )
    config_path = tmp_path / "configs" / "inclusivehar_primary_si_128.yaml"
    config_path.write_text(source_config.read_text(encoding="utf-8"), encoding="utf-8")
    output_root = tmp_path / "results" / "unit-sensitivity"
    output_root.mkdir(parents=True)
    return {
        **paths,
        "cache_record": cache_record,
        "cache_record_path": cache_record_path,
        "config": config_path,
        "output_root": output_root,
    }


def _run(
    paths: dict[str, Any], *, output: str, cache_file_hash: str | None = None
) -> dict[str, Any]:
    record = paths["cache_record"]
    return create_postconfirmatory_unit_sensitivity_record(
        sensitivity_config_path=paths["config"],
        expected_sensitivity_config_file_sha256=sha256_file(paths["config"]),
        primary_cache_record_path=paths["cache_record_path"],
        expected_primary_cache_record_file_sha256=(
            cache_file_hash or sha256_file(paths["cache_record_path"])
        ),
        opening_receipt_path=paths["receipt"],
        locked_target_index_path=paths["index"],
        artifact_root=paths["cache_record_path"].parents[2],
        expected_split_manifest_sha256=record["split_manifest"]["record_sha256"],
        expected_source_artifact_sha256=record["raw_sensor_csv"]["sha256"],
        output=output,
        output_root=paths["output_root"],
        created_at_utc="2099-01-02T03:04:05Z",
    )


def test_cache_only_runner_writes_self_hashed_record_last_and_create_only(
    tmp_path: Path,
) -> None:
    paths = _prepared_inputs(tmp_path)

    record = _run(paths, output="unit-sensitivity.json")

    output = paths["output_root"] / "unit-sensitivity.json"
    assert output.is_file()
    unhashed = dict(record)
    claimed = unhashed.pop("record_sha256")
    assert claimed == canonical_json_sha256(unhashed)
    assert record["status"] == "numerically_equivalent_under_training_only_zscore"
    assert record["source_window_count"] == 582
    assert record["evaluation_window_count"] == 807
    assert record["training_performed"] is False
    assert record["cpu_training_performed"] is False
    assert record["raw_dataset_file_accessed"] is False
    assert record["new_target_opening_created"] is False
    assert record["lineage"]["source_training_cache"]["partition"] == "source_train"
    assert record["lineage"]["evaluation_cache"]["partition"] == "target_sealed"
    assert record["lineage"]["primary_cache_index"]["file_sha256"] == sha256_file(
        paths["cache_record_path"]
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _run(paths, output="unit-sensitivity.json")


def test_runner_fails_before_publication_when_external_hash_pin_is_wrong(
    tmp_path: Path,
) -> None:
    paths = _prepared_inputs(tmp_path)

    with pytest.raises(PostconfirmatoryUnitSensitivityError, match="configuration file hash"):
        create_postconfirmatory_unit_sensitivity_record(
            sensitivity_config_path=paths["config"],
            expected_sensitivity_config_file_sha256="f" * 64,
            primary_cache_record_path=paths["cache_record_path"],
            expected_primary_cache_record_file_sha256=sha256_file(paths["cache_record_path"]),
            opening_receipt_path=paths["receipt"],
            locked_target_index_path=paths["index"],
            artifact_root=tmp_path,
            expected_split_manifest_sha256=paths["cache_record"]["split_manifest"]["record_sha256"],
            expected_source_artifact_sha256=paths["cache_record"]["raw_sensor_csv"]["sha256"],
            output="bad-config.json",
            output_root=paths["output_root"],
            created_at_utc="2099-01-02T03:04:05Z",
        )
    assert not (paths["output_root"] / "bad-config.json").exists()

    with pytest.raises(RuntimeError, match="file hash changed"):
        _run(paths, output="bad-cache.json", cache_file_hash="e" * 64)
    assert not (paths["output_root"] / "bad-cache.json").exists()


def test_parser_has_no_raw_data_unlock_opening_or_training_interface() -> None:
    help_text = build_parser().format_help().casefold()

    assert "--primary-cache-record" in help_text
    assert "--raw-csv" not in help_text
    assert "--unlock-record" not in help_text
    assert "--opening-acknowledgement" not in help_text
    assert "--device" not in help_text
    assert "--train" not in help_text
