from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from inclusive_shift_har.evaluation.efficiency import (
    EfficiencyProfileError,
    _checkpoint_details,
    _conv1d_macs,
    _linear_macs,
    _lstm_macs,
    _parameter_counts,
    _state_tensor_bytes,
    efficiency_semantic_contract,
    load_efficiency_profile_config,
    profile_neural_model,
    write_efficiency_profile_new,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file


def _config_path(repository_root: Path) -> Path:
    return repository_root / "configs/experiments/neural_efficiency_profile_v1.yaml"


def test_efficiency_config_is_self_hashed_and_cuda_only(repository_root: Path) -> None:
    config = load_efficiency_profile_config(_config_path(repository_root))
    assert config.required_device == "cuda"
    assert (
        config.file_sha256
        == sha256_file(_config_path(repository_root))
        == ("179881e7771ee8e8ae2d6bdf43f68b2bc452c57b82dbf51296874b3274d83f31")
    )
    assert config.config_sha256 == (
        "59db7c71322161344f669887a757dd55b1db887a5848b4e3b719147e4939bd78"
    )
    assert config.batch_sizes == (1, 64)
    assert config.precisions == ("float32", "float16_autocast")
    assert config.warmup_iterations == 20
    assert config.measured_iterations == 100
    assert config.window_length_samples == 128
    assert config.channel_count == 6
    assert config.latency_percentiles == (50.0, 95.0)
    assert config.flop_per_mac == 2
    assert config.contention_monitor_command == "nvidia-smi"
    assert config.contention_sample_before_run is True
    assert config.contention_sample_before_and_after_each_profile is True
    assert config.contention_allowed_ambient_process_names == ("dwm.exe", "explorer.exe")
    assert config.contention_fail_on_unapproved_process is True
    assert efficiency_semantic_contract()["vram"]["cache_cleared_before_profile"] is True


@pytest.mark.parametrize(
    ("section", "field", "replacement", "message"),
    [
        ("input", "window_length_samples", 64, "128x6"),
        ("input", "channel_count", 9, "128x6"),
        ("latency", "warmup_iterations", 19, "20 warmups and 100 measures"),
        ("latency", "measured_iterations", 99, "20 warmups and 100 measures"),
        ("latency", "percentiles", [50.0, 90.0], "p50 and p95"),
    ],
)
def test_efficiency_config_semantics_are_locked_even_with_valid_self_hash(
    repository_root: Path,
    tmp_path: Path,
    section: str,
    field: str,
    replacement: object,
    message: str,
) -> None:
    value = yaml.safe_load(_config_path(repository_root).read_text(encoding="utf-8"))
    value[section][field] = replacement
    value.pop("config_sha256")
    value["config_sha256"] = canonical_json_sha256(value)
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")

    with pytest.raises(EfficiencyProfileError, match=message):
        load_efficiency_profile_config(changed)


def test_efficiency_config_byte_hash_is_external_and_required(
    repository_root: Path,
) -> None:
    path = _config_path(repository_root)
    with pytest.raises(EfficiencyProfileError, match="file SHA-256 mismatch"):
        load_efficiency_profile_config(
            path,
            allowed_root=repository_root,
            expected_file_sha256="0" * 64,
        )


def test_supported_operator_mac_formulas_are_exact() -> None:
    convolution = nn.Conv1d(2, 4, kernel_size=3, bias=False)
    convolution_output = torch.zeros((5, 4, 8), dtype=torch.float32)
    assert _conv1d_macs(convolution, convolution_output) == 5 * 4 * 8 * 2 * 3

    linear = nn.Linear(4, 7, bias=False)
    linear_output = torch.zeros((5, 7), dtype=torch.float32)
    assert _linear_macs(linear, linear_output) == 5 * 7 * 4

    lstm = nn.LSTM(input_size=3, hidden_size=4, batch_first=True, bias=True)
    lstm_input = torch.zeros((2, 5, 3), dtype=torch.float32)
    assert _lstm_macs(lstm, lstm_input) == 2 * 5 * 4 * 4 * (3 + 4)


def test_parameter_and_state_size_counts_are_explicit() -> None:
    model = nn.Linear(4, 3, bias=True)
    assert _parameter_counts(model) == (15, 15)
    assert _state_tensor_bytes(model) == 15 * 4
    model.bias.requires_grad_(False)
    assert _parameter_counts(model) == (15, 12)


def test_checkpoint_hash_and_file_size_are_verified(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    expected = sha256_file(checkpoint)
    observed, size = _checkpoint_details(checkpoint, expected)
    assert observed == expected
    assert size == len(b"synthetic checkpoint bytes")
    with pytest.raises(EfficiencyProfileError, match="hash mismatch"):
        _checkpoint_details(checkpoint, "0" * 64)


def test_profile_fails_before_inference_without_cuda(
    repository_root: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = load_efficiency_profile_config(_config_path(repository_root))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic")
    called = False

    class Sentinel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            nonlocal called
            called = True
            return value * self.weight

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(EfficiencyProfileError, match="CUDA is required"):
        profile_neural_model(
            Sentinel(),
            torch.zeros((1, 128, 6)),
            config=config,
            precision="float32",
            model_id="synthetic",
            seed=0,
            checkpoint_path=checkpoint,
            expected_checkpoint_sha256=sha256_file(checkpoint),
            checkpoint_validation_record_sha256="a" * 64,
        )
    assert called is False


def test_efficiency_record_writer_is_self_hashed_and_create_only(tmp_path: Path) -> None:
    record = {
        "schema_version": "1.0.0",
        "record_kind": "cuda_neural_efficiency_profile",
        "status": "profile_complete",
        "required_device": "cuda",
        "execution_device_type": "cuda",
        "model_selection_use": False,
        "profiler_code_commit": "9" * 40,
        "gpu_contention_samples": {
            "before_record_sha256": "a" * 64,
            "after_record_sha256": "b" * 64,
        },
        "parameters": {"total": 15},
    }
    record["record_sha256"] = canonical_json_sha256(record)
    destination = tmp_path / "profile.json"
    write_efficiency_profile_new(record, destination, allowed_root=tmp_path)
    stored = json.loads(destination.read_text(encoding="utf-8"))
    claimed = stored.pop("record_sha256")
    assert claimed == canonical_json_sha256(stored)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_efficiency_profile_new(record, destination, allowed_root=tmp_path)


def test_efficiency_record_writer_rejects_selection_use(tmp_path: Path) -> None:
    record = {
        "schema_version": "1.0.0",
        "record_kind": "cuda_neural_efficiency_profile",
        "status": "profile_complete",
        "required_device": "cuda",
        "execution_device_type": "cuda",
        "model_selection_use": True,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    with pytest.raises(EfficiencyProfileError, match="model-selection"):
        write_efficiency_profile_new(record, tmp_path / "profile.json", allowed_root=tmp_path)
