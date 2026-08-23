"""Read-only validation of predeclared fixed-epoch final neural checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.training.engine import training_config_from_dict


def _state_dicts_equal(first: Mapping[str, Any], second: Mapping[str, Any]) -> bool:
    if set(first) != set(second):
        return False
    return all(
        isinstance(first[key], Tensor)
        and isinstance(second[key], Tensor)
        and torch.equal(first[key].detach().cpu(), second[key].detach().cpu())
        for key in first
    )


def validate_fixed_epoch_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_configuration_sha256: str | None = None,
    expected_split_manifest_sha256: str | None = None,
    expected_code_commit: str | None = None,
) -> dict[str, Any]:
    """Prove that a trusted local neural checkpoint is the predeclared last epoch."""

    source = Path(checkpoint_path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("fixed-epoch checkpoint must be a regular non-symlink file")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError("checkpoint root must be a mapping")
    configuration = payload.get("configuration")
    lineage = payload.get("lineage")
    if not isinstance(configuration, dict) or not isinstance(lineage, Mapping):
        raise ValueError("checkpoint lacks configuration or lineage")
    config = training_config_from_dict(configuration)
    configuration_sha256 = canonical_json_sha256(asdict(config))
    if payload.get("configuration_sha256") != configuration_sha256:
        raise ValueError("checkpoint configuration self-hash does not validate")
    if (
        expected_configuration_sha256 is not None
        and configuration_sha256 != expected_configuration_sha256
    ):
        raise ValueError("checkpoint differs from the expected frozen configuration")
    if config.checkpoint_selection_rule != "fixed_last_epoch":
        raise ValueError("checkpoint did not use the fixed_last_epoch selection rule")
    if payload.get("checkpoint_selection_rule") != "fixed_last_epoch":
        raise ValueError("checkpoint payload selection rule differs from configuration")
    if payload.get("checkpoint_role") != "source_fixed_epoch_last":
        raise ValueError("checkpoint role is not the final fixed-epoch source role")
    selected_epoch = payload.get("selected_epoch")
    if selected_epoch != config.epochs or payload.get("epoch") != config.epochs:
        raise ValueError("checkpoint is not the predeclared final epoch")
    if payload.get("stopped_early") is not False:
        raise ValueError("fixed-epoch checkpoint may not be early-stopped")
    if payload.get("target_information_used_for_selection") is not False:
        raise ValueError("checkpoint does not affirm source-only selection")
    split_hash = lineage.get("split_manifest_sha256")
    if not isinstance(split_hash, str) or len(split_hash) != 64:
        raise ValueError("checkpoint lacks a full split-manifest hash")
    if expected_split_manifest_sha256 is not None and split_hash != expected_split_manifest_sha256:
        raise ValueError("checkpoint split lineage differs from the frozen split")
    code_commit = lineage.get("code_commit")
    if not isinstance(code_commit, str) or not code_commit:
        raise ValueError("checkpoint lacks code-commit lineage")
    if expected_code_commit is not None and code_commit != expected_code_commit:
        raise ValueError("checkpoint code commit differs from the final freeze")
    model_state = payload.get("model_state")
    selected_state = payload.get("selected_model_state")
    if not isinstance(model_state, Mapping) or not isinstance(selected_state, Mapping):
        raise ValueError("checkpoint lacks selected model state")
    if not _state_dicts_equal(model_state, selected_state):
        raise ValueError("selected model state is not the saved last-epoch state")
    for component in (
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "normalization",
        "label_schema",
        "rng_states",
        "environment",
    ):
        if component not in payload:
            raise ValueError(f"checkpoint lacks required reconstruction component: {component}")
    record: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fixed_epoch_checkpoint_validation",
        "status": "pass_source_only",
        "checkpoint_sha256": sha256_file(source),
        "training_configuration_sha256": configuration_sha256,
        "split_manifest_sha256": split_hash,
        "code_commit": code_commit,
        "model_name": config.model_name,
        "seed": config.seed,
        "predeclared_epochs": config.epochs,
        "selected_epoch": selected_epoch,
        "checkpoint_selection_rule": config.checkpoint_selection_rule,
        "target_information_used_for_selection": False,
    }
    record["record_sha256"] = canonical_json_sha256(record)
    return record
