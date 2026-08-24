"""Sequential CUDA profiling of the exact frozen neural model/seed inventory.

This post-confirmatory operation validates the already-consumed opening index for
lineage only. It never reads materialized signals or target metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn

from inclusive_shift_har.artifacts.final_freeze import (
    validate_final_freeze_inventory_file,
)
from inclusive_shift_har.evaluation._strict_config import require_utc_timestamp
from inclusive_shift_har.evaluation.efficiency import (
    EFFICIENCY_BATCH_SIZES,
    EFFICIENCY_PRECISIONS,
    FROZEN_NEURAL_MODEL_IDS,
    FROZEN_NEURAL_SEEDS,
    RECURRENT_CUDNN_DISABLED_MODEL_IDS,
    EfficiencyProfileConfig,
    load_efficiency_profile_config,
    profile_neural_model,
    write_efficiency_profile_new,
)
from inclusive_shift_har.evaluation.postconfirmatory_inputs import (
    ConsumedTargetContext,
    load_consumed_target_context,
)
from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.training.engine import reconstruct_checkpoint
from inclusive_shift_har.training.final_checkpoint import validate_fixed_epoch_checkpoint


class PostconfirmatoryEfficiencyError(RuntimeError):
    """Raised when frozen-model profiling cannot preserve exact lineage."""


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PostconfirmatoryEfficiencyError(f"{name} must be an object")
    return value


def _sha256(value: Any, *, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PostconfirmatoryEfficiencyError(f"{name} must be a lowercase SHA-256")
    return value


def _full_commit(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{40}", value.casefold()) is None:
        raise PostconfirmatoryEfficiencyError(f"{name} must be a full Git commit")
    return value.casefold()


def _repository_head(repository_root: Path) -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PostconfirmatoryEfficiencyError(
            "artifact root is not a readable Git checkout"
        ) from exc
    return _full_commit(completed.stdout.strip(), name="repository HEAD")


def _require_profiler_head(repository_root: Path, supplied_commit: str) -> str:
    commit = _full_commit(supplied_commit, name="profiler code commit")
    observed = _repository_head(repository_root)
    if observed != commit:
        raise PostconfirmatoryEfficiencyError(
            f"profiler code commit {commit} differs from repository HEAD {observed}"
        )
    return commit


def _now_utc() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _process_basename(value: str) -> str:
    return re.split(r"[\\/]", value.strip())[-1].casefold()


def _resolve_windows_process_basename(pid: int) -> tuple[str | None, str | None]:
    """Resolve one protected Windows PID without trusting an unknown process name."""

    if os.name != "nt":
        return None, "protected process-name resolution is available only on Windows"
    command = [
        "powershell.exe",
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        f"(Get-Process -Id {pid} -ErrorAction Stop).ProcessName",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return None, f"{type(exc).__name__}: {exc}"
    if completed.returncode != 0:
        error = completed.stderr.strip()[:500] or f"Get-Process exited {completed.returncode}"
        return None, error
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        return None, "Get-Process did not return exactly one process name"
    basename = _process_basename(lines[0])
    if not basename:
        return None, "Get-Process returned an empty process name"
    if "." not in basename:
        basename = f"{basename}.exe"
    return basename, None


def _capture_gpu_contention_snapshot(
    *,
    config: EfficiencyProfileConfig,
    device: torch.device,
    profiler_code_commit: str,
    phase: str,
    profile_identity: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Capture one self-hashed, selected-device compute-process gate sample."""

    device_index = torch.cuda.current_device() if device.index is None else device.index
    command = [
        config.contention_monitor_command,
        "--id",
        str(device_index),
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    return_code: int | None = None
    stdout = ""
    error: str | None = None
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=15,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        if return_code != 0:
            error = completed.stderr.strip()[:500] or f"nvidia-smi exited {return_code}"
    except (OSError, subprocess.TimeoutExpired) as exc:
        error = f"{type(exc).__name__}: {exc}"

    allowed = {value.casefold() for value in config.contention_allowed_ambient_process_names}
    observed: list[dict[str, Any]] = []
    if error is None:
        try:
            for row in csv.reader(stdout.splitlines()):
                if not row or not any(value.strip() for value in row):
                    continue
                if len(row) != 2:
                    raise ValueError("unexpected nvidia-smi compute-process column count")
                pid = int(row[0].strip())
                reported_process_name = row[1].strip()
                basename: str | None = _process_basename(reported_process_name)
                resolution = "nvidia_smi_reported_process_name"
                resolution_error: str | None = None
                if (
                    pid != os.getpid()
                    and reported_process_name.casefold() == "[insufficient permissions]"
                ):
                    basename, resolution_error = _resolve_windows_process_basename(pid)
                    resolution = (
                        "windows_get_process_exact_pid"
                        if basename is not None
                        else "unresolved_protected_process"
                    )
                if pid == os.getpid() and config.contention_allow_current_process:
                    classification = "current_profiler_process"
                elif basename is not None and basename in allowed:
                    classification = "allowlisted_ambient_process"
                else:
                    classification = "unapproved_competing_process"
                observed.append(
                    {
                        "pid": pid,
                        "nvidia_smi_reported_process_name": reported_process_name,
                        "process_name_basename": basename,
                        "process_name_resolution": resolution,
                        "process_name_resolution_error": resolution_error,
                        "classification": classification,
                    }
                )
        except (TypeError, ValueError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            observed = []
    unapproved = [
        value for value in observed if value["classification"] == "unapproved_competing_process"
    ]
    snapshot: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "gpu_contention_snapshot",
        "captured_at_utc": _now_utc(),
        "phase": phase,
        "profile_identity": None if profile_identity is None else dict(profile_identity),
        "profiler_code_commit": profiler_code_commit,
        "device_index": device_index,
        "monitor": "nvidia_smi_selected_device_compute_apps",
        "monitor_return_code": return_code,
        "monitor_error": error,
        "allowed_ambient_process_names": list(config.contention_allowed_ambient_process_names),
        "observed_compute_processes": observed,
        "unapproved_competing_process_count": len(unapproved),
        "status": (
            "pass_no_unapproved_compute_processes"
            if error is None and not unapproved
            else "fail_contention_gate"
        ),
    }
    snapshot["record_sha256"] = canonical_json_sha256(snapshot)
    return snapshot


def _require_contention_pass(snapshot: Mapping[str, Any]) -> None:
    if snapshot.get("status") != "pass_no_unapproved_compute_processes":
        raise PostconfirmatoryEfficiencyError(
            "GPU contention gate failed; timing cannot be treated as valid"
        )


def _contention_attestation(
    *,
    timestamp: str,
    config: EfficiencyProfileConfig,
    profiler_code_commit: str,
    snapshots: list[dict[str, Any]],
) -> dict[str, Any]:
    expected = 1 + 2 * len(FROZEN_NEURAL_MODEL_IDS) * len(FROZEN_NEURAL_SEEDS) * len(
        EFFICIENCY_BATCH_SIZES
    ) * len(EFFICIENCY_PRECISIONS)
    if len(snapshots) != expected or any(
        value.get("status") != "pass_no_unapproved_compute_processes" for value in snapshots
    ):
        raise PostconfirmatoryEfficiencyError("GPU contention attestation coverage is incomplete")
    ambient = sorted(
        {
            str(process["process_name_basename"])
            for snapshot in snapshots
            for process in snapshot["observed_compute_processes"]
            if process["classification"] == "allowlisted_ambient_process"
        }
    )
    attestation: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "gpu_contention_attestation",
        "status": "pass_no_unapproved_compute_processes_sampled",
        "created_at_utc": timestamp,
        "profiler_code_commit": profiler_code_commit,
        "profile_config_sha256": config.config_sha256,
        "monitor": "nvidia_smi_selected_device_compute_apps",
        "sampling_policy": "before_run_and_immediately_before_and_after_every_profile",
        "sampling_limit": "sampled_process_gate_not_continuous_utilization_monitoring",
        "allowed_ambient_process_names": list(config.contention_allowed_ambient_process_names),
        "observed_allowlisted_ambient_process_names": ambient,
        "timing_validity": (
            "valid_exclusive_compute_process_samples"
            if not ambient
            else "valid_with_declared_allowlisted_ambient_system_processes"
        ),
        "expected_snapshot_count": expected,
        "observed_snapshot_count": len(snapshots),
        "unapproved_competing_process_count": 0,
        "snapshots": snapshots,
    }
    attestation["record_sha256"] = canonical_json_sha256(attestation)
    return attestation


def _resolve_frozen_file(record: Mapping[str, Any], *, artifact_root: Path, name: str) -> Path:
    relative = record.get("path")
    if (
        not isinstance(relative, str)
        or not relative
        or "\\" in relative
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise PostconfirmatoryEfficiencyError(f"{name} has an unsafe path")
    candidate = artifact_root.joinpath(*relative.split("/"))
    if candidate.is_symlink():
        raise PostconfirmatoryEfficiencyError(f"{name} may not be a symlink")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(artifact_root)
    except ValueError as exc:
        raise PostconfirmatoryEfficiencyError(f"{name} escapes artifact_root") from exc
    if not resolved.is_file():
        raise PostconfirmatoryEfficiencyError(f"{name} is not a regular file")
    return resolved


def _output_directory(path: Path, *, output_root: Path) -> Path:
    """Prove confinement before creating any output directory."""

    if output_root.is_symlink():
        raise PostconfirmatoryEfficiencyError("output_root may not be a symlink")
    root = output_root.resolve(strict=True)
    if not root.is_dir():
        raise PostconfirmatoryEfficiencyError("output_root must be an existing directory")
    candidate = path if path.is_absolute() else root / path
    if os.path.lexists(candidate) and candidate.is_symlink():
        raise PostconfirmatoryEfficiencyError("output directory may not be a symlink")
    prospective = candidate.resolve(strict=False)
    try:
        prospective.relative_to(root)
    except ValueError as exc:
        raise PostconfirmatoryEfficiencyError("output directory escapes output_root") from exc
    prospective.mkdir(parents=True, exist_ok=True)
    resolved = prospective.resolve(strict=True)
    try:
        resolved.relative_to(root)
    except ValueError as exc:  # pragma: no cover - race-resistant recheck
        raise PostconfirmatoryEfficiencyError("output directory escapes output_root") from exc
    if resolved.is_symlink() or not resolved.is_dir():
        raise PostconfirmatoryEfficiencyError("output directory is unsafe")
    return resolved


def _expected_model_seed_identities() -> set[tuple[str, int]]:
    return {
        (model_id, seed) for model_id in FROZEN_NEURAL_MODEL_IDS for seed in FROZEN_NEURAL_SEEDS
    }


def _validate_neural_inventory(inventory: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the exact 16-model by five-seed frozen neural matrix."""

    if inventory.get("required_seed_order") != list(FROZEN_NEURAL_SEEDS):
        raise PostconfirmatoryEfficiencyError("frozen inventory seed order is not the locked set")
    models = inventory.get("models")
    if not isinstance(models, list):
        raise PostconfirmatoryEfficiencyError("final freeze inventory lacks model entries")
    neural = [
        _mapping(value, name="frozen neural entry")
        for value in models
        if isinstance(value, Mapping) and value.get("training_regime") == "fixed_epoch_neural"
    ]
    identities: set[tuple[str, int]] = set()
    for entry in neural:
        model_id = entry.get("model_id")
        seed = entry.get("seed")
        if (
            not isinstance(model_id, str)
            or model_id not in FROZEN_NEURAL_MODEL_IDS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
            or (model_id, seed) in identities
        ):
            raise PostconfirmatoryEfficiencyError("frozen neural identity is invalid or duplicated")
        identities.add((model_id, seed))
        configuration = _mapping(
            entry.get("training_configuration"), name="frozen training configuration"
        )
        configuration_hash = _sha256(
            entry.get("training_configuration_sha256"),
            name="training configuration hash",
        )
        if canonical_json_sha256(configuration) != configuration_hash:
            raise PostconfirmatoryEfficiencyError(
                f"{model_id} seed {seed} training configuration hash changed"
            )
        if configuration.get("seed") != seed:
            raise PostconfirmatoryEfficiencyError(
                f"{model_id} seed {seed} training configuration seed differs"
            )
        disable_cudnn = configuration.get("disable_cudnn")
        expected_disable = model_id in RECURRENT_CUDNN_DISABLED_MODEL_IDS
        if not isinstance(disable_cudnn, bool) or disable_cudnn is not expected_disable:
            raise PostconfirmatoryEfficiencyError(
                f"{model_id} seed {seed} recurrent cuDNN policy differs from the frozen policy"
            )
        checkpoint = _mapping(entry.get("checkpoint"), name="frozen checkpoint")
        _sha256(checkpoint.get("sha256"), name="frozen checkpoint hash")
        size = checkpoint.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise PostconfirmatoryEfficiencyError("frozen checkpoint size is invalid")
    expected = _expected_model_seed_identities()
    if identities != expected or len(neural) != len(expected):
        missing = sorted(expected - identities)
        unexpected = sorted(identities - expected)
        raise PostconfirmatoryEfficiencyError(
            "frozen neural inventory is not the exact 16x5 matrix; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return sorted(
        neural,
        key=lambda value: (
            FROZEN_NEURAL_MODEL_IDS.index(str(value["model_id"])),
            FROZEN_NEURAL_SEEDS.index(int(value["seed"])),
        ),
    )


def _validate_recurrent_cudnn_policy(
    *, model_id: str, configuration: Mapping[str, Any], model: nn.Module
) -> bool:
    disable_cudnn = configuration.get("disable_cudnn")
    expected_disable = model_id in RECURRENT_CUDNN_DISABLED_MODEL_IDS
    contains_lstm = any(isinstance(module, nn.LSTM) for module in model.modules())
    if not isinstance(disable_cudnn, bool) or disable_cudnn is not expected_disable:
        raise PostconfirmatoryEfficiencyError(
            f"{model_id} recurrent cuDNN policy differs from the locked configuration"
        )
    if contains_lstm is not expected_disable:
        raise PostconfirmatoryEfficiencyError(
            f"{model_id} reconstructed recurrent architecture differs from its cuDNN policy"
        )
    return disable_cudnn


def _validate_checkpoint_validation(
    validation: Mapping[str, Any],
    *,
    model_id: str,
    model_name: Any,
    seed: int,
    checkpoint_sha256: str,
    configuration_sha256: str,
    split_manifest_sha256: str,
    code_commit: str,
) -> None:
    expected = {
        "schema_version": "1.0.0",
        "record_kind": "fixed_epoch_checkpoint_validation",
        "status": "pass_source_only",
        "checkpoint_sha256": checkpoint_sha256,
        "training_configuration_sha256": configuration_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "code_commit": code_commit,
        "model_name": model_name,
        "seed": seed,
        "target_information_used_for_selection": False,
    }
    mismatches = [key for key, value in expected.items() if validation.get(key) != value]
    if mismatches:
        raise PostconfirmatoryEfficiencyError(
            f"{model_id} seed {seed} checkpoint validation mismatch: {mismatches}"
        )
    claimed = _sha256(validation.get("record_sha256"), name="checkpoint validation record hash")
    body = dict(validation)
    body.pop("record_sha256", None)
    if claimed != canonical_json_sha256(body):
        raise PostconfirmatoryEfficiencyError("checkpoint validation self-hash changed")


def _profile_with_cudnn_policy(
    model: nn.Module,
    example: torch.Tensor,
    *,
    config: EfficiencyProfileConfig,
    precision: str,
    model_id: str,
    seed: int,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_validation_record_sha256: str,
    disable_cudnn: bool,
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Apply the checkpoint policy for one profile and restore global state."""

    before = bool(torch.backends.cudnn.enabled)
    enabled_during = not disable_cudnn
    try:
        torch.backends.cudnn.enabled = enabled_during
        if bool(torch.backends.cudnn.enabled) is not enabled_during:
            raise PostconfirmatoryEfficiencyError("failed to apply checkpoint cuDNN policy")
        profile = profile_neural_model(
            model,
            example,
            config=config,
            precision=precision,  # type: ignore[arg-type]
            model_id=model_id,
            seed=seed,
            checkpoint_path=checkpoint_path,
            expected_checkpoint_sha256=checkpoint_sha256,
            checkpoint_validation_record_sha256=checkpoint_validation_record_sha256,
        )
    finally:
        torch.backends.cudnn.enabled = before
    after = bool(torch.backends.cudnn.enabled)
    if after is not before:
        raise PostconfirmatoryEfficiencyError("failed to restore process cuDNN state")
    return profile, {
        "checkpoint_configuration_disable_cudnn": disable_cudnn,
        "backend_enabled_before_profile": before,
        "backend_enabled_during_profile": enabled_during,
        "backend_enabled_after_profile": after,
        "original_backend_state_restored": True,
    }


def _index_contract(
    *,
    timestamp: str,
    profile_config_sha256: str | None,
    inventory: Mapping[str, Any] | None,
    context: ConsumedTargetContext | None,
    profile_entries: list[dict[str, Any]],
    validation_entries: list[dict[str, Any]],
    profiler_code_commit: str,
    contention_attestation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_index",
        "created_at_utc": timestamp,
        "execution": "sequential_cuda_one_frozen_model_seed_at_a_time",
        "required_device": "cuda",
        "cuda_available_at_start": True,
        "profile_config_sha256": profile_config_sha256,
        "final_freeze_inventory_sha256": None
        if inventory is None
        else inventory.get("inventory_sha256"),
        "frozen_artifact_set_sha256": None
        if inventory is None
        else inventory.get("frozen_artifact_set_sha256"),
        "split_manifest_sha256": None
        if inventory is None
        else inventory.get("split_manifest_sha256"),
        "frozen_code_commit": None if inventory is None else inventory.get("code_commit"),
        "profiler_code_commit": profiler_code_commit,
        "gpu_contention_attestation": None
        if contention_attestation is None
        else dict(contention_attestation),
        "locked_target_index_record_sha256": None
        if context is None
        else context.index.get("record_sha256"),
        "opening_receipt_record_sha256": None
        if context is None
        else context.receipt.get("record_sha256"),
        "expected_model_ids": list(FROZEN_NEURAL_MODEL_IDS),
        "required_seed_order": list(FROZEN_NEURAL_SEEDS),
        "required_batch_sizes": list(EFFICIENCY_BATCH_SIZES),
        "required_precisions": list(EFFICIENCY_PRECISIONS),
        "neural_model_count": len(FROZEN_NEURAL_MODEL_IDS),
        "neural_model_seed_count": len(FROZEN_NEURAL_MODEL_IDS) * len(FROZEN_NEURAL_SEEDS),
        "profiles_per_model_seed": len(EFFICIENCY_BATCH_SIZES) * len(EFFICIENCY_PRECISIONS),
        "expected_profile_count": len(FROZEN_NEURAL_MODEL_IDS)
        * len(FROZEN_NEURAL_SEEDS)
        * len(EFFICIENCY_BATCH_SIZES)
        * len(EFFICIENCY_PRECISIONS),
        "checkpoint_validation_count": len(validation_entries),
        "profile_count": len(profile_entries),
        "checkpoint_validations": validation_entries,
        "profiles": profile_entries,
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }


def _publish_failure_artifacts(
    *,
    output: Path,
    output_root: Path,
    timestamp: str,
    profile_config_sha256: str | None,
    inventory: Mapping[str, Any] | None,
    context: ConsumedTargetContext | None,
    profile_entries: list[dict[str, Any]],
    validation_entries: list[dict[str, Any]],
    profiler_code_commit: str,
    contention_snapshots: list[dict[str, Any]],
    error: Exception,
) -> None:
    failure_path = output / "failure.json"
    index_path = output / "neural_efficiency_profile_index.json"
    if os.path.lexists(index_path):
        # A prior complete or failed run owns this directory; never add to or
        # reinterpret its create-only evidence during an accidental rerun.
        return
    failure: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "frozen_neural_efficiency_profile_failure",
        "status": "failed_preserved_create_only",
        "created_at_utc": timestamp,
        "error": {"type": type(error).__name__, "message": str(error)},
        "completed_checkpoint_validation_count": len(validation_entries),
        "completed_profile_count": len(profile_entries),
        "profiler_code_commit": profiler_code_commit,
        "gpu_contention": {
            "snapshot_count": len(contention_snapshots),
            "snapshots": contention_snapshots,
        },
        "partial_outputs_preserved": True,
        "target_signals_or_metrics_accessed": False,
        "model_selection_use": False,
    }
    failure["record_sha256"] = canonical_json_sha256(failure)
    if not os.path.lexists(failure_path):
        atomic_write_json_new(failure, failure_path, allowed_root=output_root)
        failure_record_sha256 = str(failure["record_sha256"])
    else:
        existing_failure = _mapping(
            load_json_strict(failure_path), name="existing efficiency failure"
        )
        failure_record_sha256 = _sha256(
            existing_failure.get("record_sha256"), name="existing failure record hash"
        )
        existing_body = dict(existing_failure)
        existing_body.pop("record_sha256", None)
        if canonical_json_sha256(existing_body) != failure_record_sha256:
            raise PostconfirmatoryEfficiencyError("existing failure self-hash changed")
    failure_file_sha256 = sha256_file(failure_path)
    index = _index_contract(
        timestamp=timestamp,
        profile_config_sha256=profile_config_sha256,
        inventory=inventory,
        context=context,
        profile_entries=profile_entries,
        validation_entries=validation_entries,
        profiler_code_commit=profiler_code_commit,
        contention_attestation=None,
    )
    index.update(
        {
            "status": "failed_preserved_create_only",
            "failure": {
                "path": failure_path.relative_to(output_root).as_posix(),
                "file_sha256": failure_file_sha256,
                "record_sha256": failure_record_sha256,
            },
        }
    )
    index["record_sha256"] = canonical_json_sha256(index)
    if not os.path.lexists(index_path):
        atomic_write_json_new(index, index_path, allowed_root=output_root)


def run_frozen_efficiency_profiles(
    *,
    profile_config_path: str | Path,
    final_freeze_inventory_path: str | Path,
    opening_receipt_path: str | Path,
    locked_target_index_path: str | Path,
    artifact_root: str | Path,
    output_directory: str | Path,
    output_root: str | Path,
    created_at_utc: str,
    profiler_code_commit: str,
    device: torch.device,
) -> dict[str, Any]:
    """Profile all 80 frozen neural entries and publish exactly 320 profiles."""

    if device.type != "cuda" or not torch.cuda.is_available():
        raise PostconfirmatoryEfficiencyError(
            "CUDA is required before any frozen checkpoint or consumed index is opened"
        )
    timestamp = require_utc_timestamp(created_at_utc, location="created_at_utc")
    config = load_efficiency_profile_config(profile_config_path)
    if config.batch_sizes != EFFICIENCY_BATCH_SIZES or config.precisions != EFFICIENCY_PRECISIONS:
        raise PostconfirmatoryEfficiencyError("efficiency profile matrix differs from the lock")
    root_candidate = Path(artifact_root)
    if root_candidate.is_symlink():
        raise PostconfirmatoryEfficiencyError("artifact_root may not be a symlink")
    root = root_candidate.resolve(strict=True)
    profiler_commit = _require_profiler_head(root, profiler_code_commit)
    outputs_root = Path(output_root)
    output = _output_directory(Path(output_directory), output_root=outputs_root)
    outputs_root = outputs_root.resolve(strict=True)

    context: ConsumedTargetContext | None = None
    inventory: Mapping[str, Any] | None = None
    profile_entries: list[dict[str, Any]] = []
    validation_entries: list[dict[str, Any]] = []
    contention_snapshots: list[dict[str, Any]] = []
    try:
        initial_contention = _capture_gpu_contention_snapshot(
            config=config,
            device=device,
            profiler_code_commit=profiler_commit,
            phase="before_run",
            profile_identity=None,
        )
        contention_snapshots.append(initial_contention)
        _require_contention_pass(initial_contention)
        context = load_consumed_target_context(
            opening_receipt_path=opening_receipt_path,
            locked_target_index_path=locked_target_index_path,
            artifact_root=root,
        )
        inventory_candidate = Path(final_freeze_inventory_path)
        if not inventory_candidate.is_absolute():
            inventory_candidate = root / inventory_candidate
        if inventory_candidate.is_symlink():
            raise PostconfirmatoryEfficiencyError("final freeze inventory may not be a symlink")
        inventory_path = inventory_candidate.resolve(strict=True)
        try:
            inventory_path.relative_to(root)
        except ValueError as exc:
            raise PostconfirmatoryEfficiencyError(
                "final freeze inventory escapes artifact_root"
            ) from exc
        report = validate_final_freeze_inventory_file(inventory_path, artifact_root=root)
        if not report.valid:
            raise PostconfirmatoryEfficiencyError(
                f"final freeze inventory is invalid: {report.to_dict()}"
            )
        inventory = _mapping(load_json_strict(inventory_path), name="final freeze inventory")
        inventory_hash = _sha256(
            inventory.get("inventory_sha256"), name="final freeze inventory hash"
        )
        if inventory_hash != context.index.get("final_freeze_inventory_sha256"):
            raise PostconfirmatoryEfficiencyError(
                "consumed target index and final freeze inventory differ"
            )
        split_hash = _sha256(inventory.get("split_manifest_sha256"), name="split hash")
        frozen_set_hash = _sha256(
            inventory.get("frozen_artifact_set_sha256"), name="frozen artifact set hash"
        )
        code_commit = inventory.get("code_commit")
        if not isinstance(code_commit, str) or not code_commit:
            raise PostconfirmatoryEfficiencyError("final freeze code commit is invalid")
        neural_entries = _validate_neural_inventory(inventory)

        planned: list[Path] = []
        for model_id in FROZEN_NEURAL_MODEL_IDS:
            for seed in FROZEN_NEURAL_SEEDS:
                stem = f"{model_id}--seed-{seed}"
                planned.append(output / f"{stem}.checkpoint-validation.json")
                for batch_size in EFFICIENCY_BATCH_SIZES:
                    for precision in EFFICIENCY_PRECISIONS:
                        planned.append(
                            output / f"{stem}--batch-{batch_size}--{precision}.profile.json"
                        )
        index_path = output / "neural_efficiency_profile_index.json"
        failure_path = output / "failure.json"
        contention_path = output / "gpu_contention_attestation.json"
        for path in [*planned, contention_path, index_path, failure_path]:
            if os.path.lexists(path):
                raise FileExistsError(f"refusing to overwrite efficiency evidence: {path}")

        for entry in neural_entries:
            model_id = str(entry["model_id"])
            seed = int(entry["seed"])
            configuration = _mapping(
                entry["training_configuration"], name="frozen training configuration"
            )
            configuration_hash = _sha256(
                entry["training_configuration_sha256"], name="training configuration hash"
            )
            checkpoint_record = _mapping(entry.get("checkpoint"), name="frozen checkpoint")
            checkpoint_path = _resolve_frozen_file(
                checkpoint_record, artifact_root=root, name=f"{model_id} checkpoint"
            )
            checkpoint_hash = _sha256(
                checkpoint_record.get("sha256"), name="frozen checkpoint hash"
            )
            if sha256_file(checkpoint_path) != checkpoint_hash:
                raise PostconfirmatoryEfficiencyError(f"{model_id} checkpoint hash changed")
            if checkpoint_path.stat().st_size != checkpoint_record.get("size_bytes"):
                raise PostconfirmatoryEfficiencyError(f"{model_id} checkpoint size changed")
            validation = validate_fixed_epoch_checkpoint(
                checkpoint_path,
                expected_configuration_sha256=configuration_hash,
                expected_split_manifest_sha256=split_hash,
                expected_code_commit=code_commit,
            )
            _validate_checkpoint_validation(
                validation,
                model_id=model_id,
                model_name=configuration.get("model_name"),
                seed=seed,
                checkpoint_sha256=checkpoint_hash,
                configuration_sha256=configuration_hash,
                split_manifest_sha256=split_hash,
                code_commit=code_commit,
            )
            stem = f"{model_id}--seed-{seed}"
            validation_path = output / f"{stem}.checkpoint-validation.json"
            atomic_write_json_new(validation, validation_path, allowed_root=outputs_root)
            validation_entry = {
                "model_id": model_id,
                "model_name": configuration["model_name"],
                "seed": seed,
                "training_configuration_sha256": configuration_hash,
                "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
                "checkpoint_sha256": checkpoint_hash,
                "disable_cudnn": configuration["disable_cudnn"],
                "path": validation_path.relative_to(outputs_root).as_posix(),
                "file_sha256": sha256_file(validation_path),
                "record_sha256": validation["record_sha256"],
            }
            validation_entries.append(validation_entry)

            model, payload = reconstruct_checkpoint(checkpoint_path, device=device)
            try:
                payload_configuration = _mapping(
                    payload.get("configuration"), name="reconstructed configuration"
                )
                if (
                    canonical_json_sha256(payload_configuration) != configuration_hash
                    or payload_configuration != configuration
                ):
                    raise PostconfirmatoryEfficiencyError(
                        f"{model_id} reconstructed configuration differs from inventory"
                    )
                disable_cudnn = _validate_recurrent_cudnn_policy(
                    model_id=model_id, configuration=configuration, model=model
                )
                for batch_size in EFFICIENCY_BATCH_SIZES:
                    example = torch.zeros(
                        (batch_size, config.window_length_samples, config.channel_count),
                        dtype=torch.float32,
                        device=device,
                    )
                    for precision in EFFICIENCY_PRECISIONS:
                        profile_identity = {
                            "model_id": model_id,
                            "seed": seed,
                            "batch_size": batch_size,
                            "precision": precision,
                        }
                        contention_before = _capture_gpu_contention_snapshot(
                            config=config,
                            device=device,
                            profiler_code_commit=profiler_commit,
                            phase="before_profile",
                            profile_identity=profile_identity,
                        )
                        contention_snapshots.append(contention_before)
                        _require_contention_pass(contention_before)
                        profile, cudnn_policy = _profile_with_cudnn_policy(
                            model,
                            example,
                            config=config,
                            precision=precision,
                            model_id=model_id,
                            seed=seed,
                            checkpoint_path=checkpoint_path,
                            checkpoint_sha256=checkpoint_hash,
                            checkpoint_validation_record_sha256=str(validation["record_sha256"]),
                            disable_cudnn=disable_cudnn,
                        )
                        contention_after = _capture_gpu_contention_snapshot(
                            config=config,
                            device=device,
                            profiler_code_commit=profiler_commit,
                            phase="after_profile",
                            profile_identity=profile_identity,
                        )
                        contention_snapshots.append(contention_after)
                        _require_contention_pass(contention_after)
                        profile_body = dict(profile)
                        profile_body.pop("record_sha256")
                        profile_body.update(
                            {
                                "created_at_utc": timestamp,
                                "training_configuration_sha256": configuration_hash,
                                "split_manifest_sha256": split_hash,
                                "frozen_code_commit": code_commit,
                                "profiler_code_commit": profiler_commit,
                                "final_freeze_inventory_sha256": inventory_hash,
                                "frozen_artifact_set_sha256": frozen_set_hash,
                                "locked_target_index_record_sha256": context.index["record_sha256"],
                                "opening_receipt_record_sha256": context.receipt["record_sha256"],
                                "checkpoint_path": checkpoint_path.relative_to(root).as_posix(),
                                "checkpoint_validation": {
                                    "path": validation_entry["path"],
                                    "file_sha256": validation_entry["file_sha256"],
                                    "record_sha256": validation_entry["record_sha256"],
                                },
                                "cudnn_policy": cudnn_policy,
                                "cuda_available_at_profile": True,
                                "gpu_contention_samples": {
                                    "before_record_sha256": contention_before["record_sha256"],
                                    "after_record_sha256": contention_after["record_sha256"],
                                },
                            }
                        )
                        profile_body["record_sha256"] = canonical_json_sha256(profile_body)
                        profile_path = (
                            output / f"{stem}--batch-{batch_size}--{precision}.profile.json"
                        )
                        write_efficiency_profile_new(
                            profile_body, profile_path, allowed_root=outputs_root
                        )
                        profile_entries.append(
                            {
                                "model_id": model_id,
                                "seed": seed,
                                "batch_size": batch_size,
                                "precision": precision,
                                "training_configuration_sha256": configuration_hash,
                                "checkpoint_sha256": checkpoint_hash,
                                "checkpoint_validation_path": validation_entry["path"],
                                "checkpoint_validation_file_sha256": validation_entry[
                                    "file_sha256"
                                ],
                                "checkpoint_validation_record_sha256": validation_entry[
                                    "record_sha256"
                                ],
                                "disable_cudnn": disable_cudnn,
                                "path": profile_path.relative_to(outputs_root).as_posix(),
                                "file_sha256": sha256_file(profile_path),
                                "record_sha256": profile_body["record_sha256"],
                            }
                        )
            finally:
                del model
                torch.cuda.synchronize(device)
                torch.cuda.empty_cache()

        contention = _contention_attestation(
            timestamp=timestamp,
            config=config,
            profiler_code_commit=profiler_commit,
            snapshots=contention_snapshots,
        )
        atomic_write_json_new(contention, contention_path, allowed_root=outputs_root)
        contention_reference = {
            "path": contention_path.relative_to(outputs_root).as_posix(),
            "file_sha256": sha256_file(contention_path),
            "record_sha256": contention["record_sha256"],
            "status": contention["status"],
            "timing_validity": contention["timing_validity"],
        }
        index = _index_contract(
            timestamp=timestamp,
            profile_config_sha256=config.config_sha256,
            inventory=inventory,
            context=context,
            profile_entries=profile_entries,
            validation_entries=validation_entries,
            profiler_code_commit=profiler_commit,
            contention_attestation=contention_reference,
        )
        if (
            index["checkpoint_validation_count"] != index["neural_model_seed_count"]
            or index["profile_count"] != index["expected_profile_count"]
        ):
            raise PostconfirmatoryEfficiencyError("efficiency output combinatorics are incomplete")
        index["status"] = "complete_create_only"
        index["record_sha256"] = canonical_json_sha256(index)
        atomic_write_json_new(index, index_path, allowed_root=outputs_root)
        return {
            "index_path": str(index_path),
            "index_file_sha256": sha256_file(index_path),
            "index_record_sha256": index["record_sha256"],
            "profile_count": len(profile_entries),
        }
    except Exception as exc:
        _publish_failure_artifacts(
            output=output,
            output_root=outputs_root,
            timestamp=timestamp,
            profile_config_sha256=config.config_sha256,
            inventory=inventory,
            context=context,
            profile_entries=profile_entries,
            validation_entries=validation_entries,
            profiler_code_commit=profiler_commit,
            contention_snapshots=contention_snapshots,
            error=exc,
        )
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-config", type=Path, required=True)
    parser.add_argument("--final-freeze-inventory", type=Path, required=True)
    parser.add_argument("--opening-receipt", type=Path, required=True)
    parser.add_argument("--locked-target-index", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--created-at-utc", required=True)
    parser.add_argument("--profiler-code-commit", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_frozen_efficiency_profiles(
            profile_config_path=args.profile_config,
            final_freeze_inventory_path=args.final_freeze_inventory,
            opening_receipt_path=args.opening_receipt,
            locked_target_index_path=args.locked_target_index,
            artifact_root=args.artifact_root,
            output_directory=args.output_directory,
            output_root=args.output_root,
            created_at_utc=args.created_at_utc,
            profiler_code_commit=args.profiler_code_commit,
            device=torch.device("cuda"),
        )
    except Exception as exc:
        print(
            json.dumps(
                {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
