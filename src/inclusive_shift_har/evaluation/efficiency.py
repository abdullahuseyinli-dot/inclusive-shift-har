"""CUDA-only neural efficiency profiling with explicit counting conventions."""

from __future__ import annotations

import math
import os
import statistics
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
from torch import Tensor, nn

from inclusive_shift_har.evaluation._strict_config import (
    StrictConfigError,
    load_strict_yaml_mapping,
    require_exact_keys,
    require_mapping,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    sha256_file,
)

Precision = Literal["float32", "float16_autocast"]

FROZEN_NEURAL_MODEL_IDS = (
    "compact-coral",
    "compact-dann",
    "compact-erm",
    "deepconvlstm",
    "legacy-bilstm",
    "legacy-cnn1d",
    "legacy-joint-cnn-bilstm",
    "more-har-augmentation",
    "more-har-backbone",
    "more-har-content",
    "more-har-factorized",
    "more-har-full",
    "more-har-full-no-accelerometer",
    "more-har-full-no-gyroscope",
    "more-har-groupdro",
    "static-dual-branch-matched",
)
FROZEN_NEURAL_SEEDS = (11, 23, 47, 89, 131)
EFFICIENCY_BATCH_SIZES = (1, 64)
EFFICIENCY_PRECISIONS: tuple[Precision, ...] = ("float32", "float16_autocast")
RECURRENT_CUDNN_DISABLED_MODEL_IDS = frozenset(
    {"deepconvlstm", "legacy-bilstm", "legacy-joint-cnn-bilstm"}
)


class EfficiencyProfileError(ValueError):
    """Raised when a neural profile would be incomplete, ambiguous, or non-CUDA."""


@dataclass(frozen=True, slots=True)
class EfficiencyProfileConfig:
    schema_version: str
    profile_id: str
    config_sha256: str
    status: str
    required_device: str
    window_length_samples: int
    channel_count: int
    synthetic_input_values: str
    batch_sizes: tuple[int, ...]
    precisions: tuple[Precision, ...]
    warmup_iterations: int
    measured_iterations: int
    latency_percentiles: tuple[float, ...]
    synchronize_each_iteration: bool
    flop_per_mac: int


_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile_id",
    "config_sha256",
    "status",
    "required_device",
    "input",
    "precision_modes",
    "latency",
    "complexity",
    "output_policy",
}


def _integer(value: Any, *, location: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise EfficiencyProfileError(f"{location} must be an integer >= {minimum}")
    return int(value)


def _float(value: Any, *, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EfficiencyProfileError(f"{location} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise EfficiencyProfileError(f"{location} must be finite")
    return result


def _mapping(value: Any, *, location: str) -> dict[str, Any]:
    try:
        return require_mapping(value, location=location)
    except StrictConfigError as exc:
        raise EfficiencyProfileError(str(exc)) from exc


def load_efficiency_profile_config(path: str | Path) -> EfficiencyProfileConfig:
    """Load and verify the self-hashed, CUDA-only profiling configuration."""

    try:
        parsed = load_strict_yaml_mapping(path)
        require_exact_keys(parsed, _TOP_LEVEL_KEYS, location="efficiency profile")
    except StrictConfigError as exc:
        raise EfficiencyProfileError(str(exc)) from exc
    claimed_hash = parsed["config_sha256"]
    if not isinstance(claimed_hash, str):
        raise EfficiencyProfileError("config_sha256 must be a SHA-256 string")
    unhashed = dict(parsed)
    unhashed.pop("config_sha256")
    observed_hash = canonical_json_sha256(unhashed)
    if claimed_hash != observed_hash:
        raise EfficiencyProfileError(
            f"efficiency configuration self-hash mismatch: claimed={claimed_hash}, "
            f"observed={observed_hash}"
        )
    required_scalars = {
        "schema_version": "1.0.0",
        "status": "configured_not_run",
        "required_device": "cuda",
    }
    for key, expected in required_scalars.items():
        if parsed[key] != expected:
            raise EfficiencyProfileError(f"{key} must be {expected!r}")
    profile_id = parsed["profile_id"]
    if not isinstance(profile_id, str) or not profile_id.strip():
        raise EfficiencyProfileError("profile_id must be non-empty")

    input_config = _mapping(parsed["input"], location="input")
    latency = _mapping(parsed["latency"], location="latency")
    complexity = _mapping(parsed["complexity"], location="complexity")
    output_policy = _mapping(parsed["output_policy"], location="output_policy")
    try:
        require_exact_keys(
            input_config,
            {"window_length_samples", "channel_count", "batch_sizes", "synthetic_values"},
            location="input",
        )
        require_exact_keys(
            latency,
            {
                "warmup_iterations",
                "measured_iterations",
                "percentiles",
                "synchronize_each_iteration",
            },
            location="latency",
        )
        require_exact_keys(
            complexity,
            {"mac_definition", "flop_per_mac", "counted_modules", "excluded_operations"},
            location="complexity",
        )
        require_exact_keys(
            output_policy,
            {
                "create_only",
                "checkpoint_sha256_required",
                "checkpoint_validation_record_sha256_required",
            },
            location="output_policy",
        )
    except StrictConfigError as exc:
        raise EfficiencyProfileError(str(exc)) from exc
    if output_policy != {
        "create_only": True,
        "checkpoint_sha256_required": True,
        "checkpoint_validation_record_sha256_required": True,
    }:
        raise EfficiencyProfileError("efficiency output policy contract was changed")
    if complexity["mac_definition"] != "one_scalar_multiply_accumulate":
        raise EfficiencyProfileError("MAC definition contract was changed")
    if complexity["counted_modules"] != ["Conv1d", "Linear", "LSTM"]:
        raise EfficiencyProfileError("counted module set was changed")
    if not isinstance(complexity["excluded_operations"], list) or not all(
        isinstance(item, str) for item in complexity["excluded_operations"]
    ):
        raise EfficiencyProfileError("excluded_operations must be a list of strings")
    flop_per_mac = _integer(complexity["flop_per_mac"], location="flop_per_mac")
    if flop_per_mac != 2:
        raise EfficiencyProfileError("flop_per_mac must remain 2")

    raw_batches = input_config["batch_sizes"]
    if not isinstance(raw_batches, list) or not raw_batches:
        raise EfficiencyProfileError("input.batch_sizes must be non-empty")
    batches = tuple(_integer(item, location="batch_sizes") for item in raw_batches)
    if tuple(sorted(set(batches))) != batches:
        raise EfficiencyProfileError("batch_sizes must be sorted and unique")
    if batches != EFFICIENCY_BATCH_SIZES:
        raise EfficiencyProfileError(
            f"batch_sizes must remain the locked matrix {list(EFFICIENCY_BATCH_SIZES)}"
        )
    if input_config["synthetic_values"] != "zeros":
        raise EfficiencyProfileError("profiling input values must remain deterministic zeros")
    raw_precisions = parsed["precision_modes"]
    if not isinstance(raw_precisions, list) or not raw_precisions:
        raise EfficiencyProfileError("precision_modes must be non-empty")
    allowed = {"float32", "float16_autocast"}
    if any(not isinstance(item, str) or item not in allowed for item in raw_precisions):
        raise EfficiencyProfileError("precision_modes contains an unsupported mode")
    precisions = cast(tuple[Precision, ...], tuple(raw_precisions))
    if precisions != EFFICIENCY_PRECISIONS:
        raise EfficiencyProfileError(
            "precision_modes must remain ['float32', 'float16_autocast'] in that order"
        )
    raw_percentiles = latency["percentiles"]
    if not isinstance(raw_percentiles, list) or not raw_percentiles:
        raise EfficiencyProfileError("latency.percentiles must be non-empty")
    percentiles = tuple(_float(item, location="percentiles") for item in raw_percentiles)
    if tuple(sorted(set(percentiles))) != percentiles or any(
        item <= 0.0 or item >= 100.0 for item in percentiles
    ):
        raise EfficiencyProfileError("percentiles must be sorted, unique, and inside (0,100)")
    synchronize = latency["synchronize_each_iteration"]
    if synchronize is not True:
        raise EfficiencyProfileError("CUDA timing must synchronize every measured iteration")
    return EfficiencyProfileConfig(
        schema_version="1.0.0",
        profile_id=profile_id,
        config_sha256=claimed_hash,
        status="configured_not_run",
        required_device="cuda",
        window_length_samples=_integer(
            input_config["window_length_samples"], location="window_length_samples"
        ),
        channel_count=_integer(input_config["channel_count"], location="channel_count"),
        synthetic_input_values="zeros",
        batch_sizes=batches,
        precisions=precisions,
        warmup_iterations=_integer(latency["warmup_iterations"], location="warmup_iterations"),
        measured_iterations=_integer(
            latency["measured_iterations"], location="measured_iterations"
        ),
        latency_percentiles=percentiles,
        synchronize_each_iteration=True,
        flop_per_mac=flop_per_mac,
    )


def _conv1d_macs(module: nn.Conv1d, output: Tensor) -> int:
    kernel_multiplications = (module.in_channels // module.groups) * module.kernel_size[0]
    return int(output.numel() * kernel_multiplications)


def _linear_macs(module: nn.Linear, output: Tensor) -> int:
    return int(output.numel() * module.in_features)


def _lstm_macs(module: nn.LSTM, input_tensor: Tensor) -> int:
    if input_tensor.ndim != 3:
        raise EfficiencyProfileError("LSTM MAC counting requires an unpacked rank-three input")
    batch = int(input_tensor.shape[0] if module.batch_first else input_tensor.shape[1])
    steps = int(input_tensor.shape[1] if module.batch_first else input_tensor.shape[0])
    directions = 2 if module.bidirectional else 1
    recurrent_width = module.proj_size if module.proj_size > 0 else module.hidden_size
    total = 0
    for layer in range(module.num_layers):
        layer_input = module.input_size if layer == 0 else directions * recurrent_width
        per_step_direction = 4 * module.hidden_size * (layer_input + recurrent_width)
        if module.proj_size > 0:
            per_step_direction += module.hidden_size * module.proj_size
        total += batch * steps * directions * per_step_direction
    return int(total)


def _state_tensor_bytes(model: nn.Module) -> int:
    return int(
        sum(tensor.numel() * tensor.element_size() for tensor in model.state_dict().values())
    )


def _parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = int(sum(parameter.numel() for parameter in model.parameters()))
    trainable = int(
        sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    )
    return total, trainable


def _checkpoint_details(path: Path, expected_sha256: str) -> tuple[str, int]:
    expected = expected_sha256.casefold()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise EfficiencyProfileError("expected checkpoint SHA-256 is invalid")
    if path.is_symlink() or not path.is_file():
        raise EfficiencyProfileError("checkpoint must be an existing regular non-symlink file")
    observed = sha256_file(path)
    if observed != expected:
        raise EfficiencyProfileError(
            f"checkpoint hash mismatch: expected={expected}, observed={observed}"
        )
    return observed, path.stat().st_size


def _valid_sha256(value: str, *, description: str) -> str:
    normalized = value.casefold()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise EfficiencyProfileError(f"{description} must be a full SHA-256")
    return normalized


def _model_device(model: nn.Module) -> torch.device:
    devices = {tensor.device for tensor in (*tuple(model.parameters()), *tuple(model.buffers()))}
    if not devices:
        raise EfficiencyProfileError("neural profiling requires a parameterized model")
    if len(devices) != 1:
        raise EfficiencyProfileError("all model parameters and buffers must share one device")
    return next(iter(devices))


def _analytical_macs(model: nn.Module, example_input: Tensor, context: Any) -> dict[str, Any]:
    total = 0
    calls: dict[str, int] = {"Conv1d": 0, "Linear": 0, "LSTM": 0}
    handles: list[Any] = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal total
        if isinstance(module, nn.Conv1d):
            if not isinstance(output, Tensor):
                raise EfficiencyProfileError("Conv1d produced a non-tensor output")
            total += _conv1d_macs(module, output)
            calls["Conv1d"] += 1
        elif isinstance(module, nn.Linear):
            if not isinstance(output, Tensor):
                raise EfficiencyProfileError("Linear produced a non-tensor output")
            total += _linear_macs(module, output)
            calls["Linear"] += 1
        elif isinstance(module, nn.LSTM):
            if not inputs or not isinstance(inputs[0], Tensor):
                raise EfficiencyProfileError("LSTM MAC counting requires a tensor input")
            total += _lstm_macs(module, inputs[0])
            calls["LSTM"] += 1

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear, nn.LSTM)):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.inference_mode(), context:
            model(example_input)
    finally:
        for handle in handles:
            handle.remove()
    return {
        "supported_operator_macs": total,
        "module_call_counts": calls,
        "counted_modules": ["Conv1d", "Linear", "LSTM"],
        "definition": "one_scalar_multiply_accumulate",
        "coverage_status": "supported_operator_subset_only",
        "excluded_operations": [
            "bias_additions",
            "normalization",
            "activations",
            "pooling",
            "softmax",
            "attention_or_custom_functional_kernels_not_exposed_as_counted_modules",
        ],
    }


def profile_neural_model(
    model: nn.Module,
    example_input: Tensor,
    *,
    config: EfficiencyProfileConfig,
    precision: Precision,
    model_id: str,
    seed: int,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    checkpoint_validation_record_sha256: str,
) -> dict[str, Any]:
    """Profile one already-loaded frozen neural model; CUDA is mandatory.

    This routine never loads or mutates a checkpoint.  It hashes the supplied
    frozen file and requires the caller to link its prior validation record.
    """

    if not torch.cuda.is_available():
        raise EfficiencyProfileError("CUDA is required for neural efficiency profiling")
    if not model_id.strip():
        raise EfficiencyProfileError("model_id must be non-empty")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise EfficiencyProfileError("seed must be a non-negative integer")
    if precision not in config.precisions:
        raise EfficiencyProfileError("precision is not predeclared in the profile config")
    if example_input.device.type != "cuda":
        raise EfficiencyProfileError("profiling input must be on CUDA")
    device = _model_device(model)
    if device.type != "cuda" or device != example_input.device:
        raise EfficiencyProfileError("model and input must share the same CUDA device")
    if example_input.ndim != 3 or example_input.shape[1:] != (
        config.window_length_samples,
        config.channel_count,
    ):
        raise EfficiencyProfileError("example input must use locked [B,T,C] dimensions")
    batch_size = int(example_input.shape[0])
    if batch_size not in config.batch_sizes:
        raise EfficiencyProfileError("input batch size is not predeclared")
    if not example_input.is_floating_point() or not torch.isfinite(example_input).all().item():
        raise EfficiencyProfileError("example input must be finite floating point")
    checkpoint_hash, checkpoint_bytes = _checkpoint_details(
        checkpoint_path, expected_checkpoint_sha256
    )
    validation_hash = _valid_sha256(
        checkpoint_validation_record_sha256,
        description="checkpoint validation record hash",
    )
    parameter_count, trainable_count = _parameter_counts(model)
    if parameter_count == 0:
        raise EfficiencyProfileError("neural profiling requires at least one parameter")

    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.float16)
        if precision == "float16_autocast"
        else nullcontext()
    )
    training_states = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        complexity = _analytical_macs(model, example_input, autocast_context)
        with torch.inference_mode():
            for _ in range(config.warmup_iterations):
                with (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if precision == "float16_autocast"
                    else nullcontext()
                ):
                    model(example_input)
        torch.cuda.synchronize(device)
        baseline_allocated = int(torch.cuda.memory_allocated(device))
        baseline_reserved = int(torch.cuda.memory_reserved(device))
        torch.cuda.reset_peak_memory_stats(device)
        elapsed_ms: list[float] = []
        for _ in range(config.measured_iterations):
            start = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            end = torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
            start.record()
            with (
                torch.inference_mode(),
                (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if precision == "float16_autocast"
                    else nullcontext()
                ),
            ):
                model(example_input)
            end.record()
            torch.cuda.synchronize(device)
            elapsed_ms.append(float(start.elapsed_time(end)))
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
    finally:
        for module, training_state in training_states:
            module.training = training_state

    latency = np.asarray(elapsed_ms, dtype=np.float64)
    properties = torch.cuda.get_device_properties(device)
    supported_macs = int(complexity["supported_operator_macs"])
    if supported_macs % batch_size != 0:
        raise EfficiencyProfileError("supported MAC count is not divisible by batch size")
    supported_macs_per_window = supported_macs // batch_size
    profile: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "cuda_neural_efficiency_profile",
        "status": "profile_complete",
        "evidence_status": "measured_cuda_efficiency_not_model_selection_evidence",
        "required_device": "cuda",
        "execution_device_type": device.type,
        "profile_id": config.profile_id,
        "profile_config_sha256": config.config_sha256,
        "model_id": model_id,
        "seed": seed,
        "precision": precision,
        "input_shape": list(example_input.shape),
        "input_dtype": str(example_input.dtype),
        "parameters": {
            "total": parameter_count,
            "trainable": trainable_count,
            "state_tensor_bytes": _state_tensor_bytes(model),
        },
        "complexity": {
            **complexity,
            "supported_operator_macs_scope": "one_profiled_batch",
            "supported_operator_macs_per_window": supported_macs_per_window,
            "estimated_flops_from_supported_macs_per_batch": supported_macs * config.flop_per_mac,
            "estimated_flops_from_supported_macs_per_window": supported_macs_per_window
            * config.flop_per_mac,
            "flop_conversion": f"{config.flop_per_mac}_flops_per_mac",
        },
        "latency_ms_per_batch": {
            "warmup_iterations": config.warmup_iterations,
            "measured_iterations": config.measured_iterations,
            "mean": float(latency.mean()),
            "median": float(np.median(latency)),
            "minimum": float(latency.min()),
            "maximum": float(latency.max()),
            "sample_standard_deviation": float(statistics.stdev(elapsed_ms))
            if len(elapsed_ms) > 1
            else 0.0,
            "percentiles": {
                str(value): float(np.percentile(latency, value))
                for value in config.latency_percentiles
            },
            "mean_per_window": float(latency.mean() / batch_size),
            "timing_method": "torch_cuda_events_with_per_iteration_synchronization",
            "scope": "forward_pass_only_device_resident_input_no_host_to_device_transfer",
        },
        "vram_bytes": {
            "baseline_allocated": baseline_allocated,
            "baseline_reserved": baseline_reserved,
            "peak_allocated": peak_allocated,
            "peak_reserved": peak_reserved,
            "incremental_peak_allocated": max(0, peak_allocated - baseline_allocated),
            "incremental_peak_reserved": max(0, peak_reserved - baseline_reserved),
        },
        "model_size": {
            "checkpoint_file_bytes": checkpoint_bytes,
            "checkpoint_sha256": checkpoint_hash,
            "checkpoint_validation_record_sha256": validation_hash,
        },
        "environment": {
            "torch_version": str(torch.__version__),
            "torch_cuda_version": torch.version.cuda,
            "device_type": device.type,
            "device": str(device),
            "device_name": properties.name,
            "device_total_memory_bytes": int(properties.total_memory),
            "device_compute_capability": list(torch.cuda.get_device_capability(device)),
        },
        "model_selection_use": False,
    }
    profile["record_sha256"] = canonical_json_sha256(profile)
    return profile


def write_efficiency_profile_new(
    profile: Mapping[str, Any],
    destination: Path,
    *,
    allowed_root: Path,
) -> Path:
    """Publish a self-hashed efficiency record without replacing prior evidence."""

    value = dict(profile)
    claimed = value.get("record_sha256")
    unhashed = dict(value)
    unhashed.pop("record_sha256", None)
    if claimed != canonical_json_sha256(unhashed):
        raise EfficiencyProfileError("efficiency profile self-hash does not validate")
    if value.get("record_kind") != "cuda_neural_efficiency_profile":
        raise EfficiencyProfileError("record is not a CUDA neural efficiency profile")
    if value.get("status") != "profile_complete":
        raise EfficiencyProfileError("efficiency profile status is not complete")
    if value.get("required_device") != "cuda" or value.get("execution_device_type") != "cuda":
        raise EfficiencyProfileError("efficiency profile does not prove CUDA execution")
    if value.get("model_selection_use") is not False:
        raise EfficiencyProfileError("efficiency record cannot be model-selection evidence")
    root = allowed_root.resolve(strict=True)
    target = destination.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise EfficiencyProfileError("efficiency destination escapes allowed_root") from exc
    if os.path.lexists(target):
        raise FileExistsError(f"refusing to overwrite efficiency profile: {target}")
    return atomic_write_json_new(value, target, allowed_root=root)
