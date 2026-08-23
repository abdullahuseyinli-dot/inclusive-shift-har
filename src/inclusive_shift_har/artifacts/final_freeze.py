"""Source-only final-model freeze inventory and exact target-unlock authorization."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)
from inclusive_shift_har.protocols.seal import TargetSealError, validate_target_unlock_record

FINAL_FREEZE_SCHEMA_VERSION = "1.0.0"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class FinalFreezeError(ValueError):
    """Raised when the final source-only artifact freeze fails closed."""


@dataclass(frozen=True)
class FrozenModelInput:
    """One checkpoint/calibrator pair proposed for the final frozen inventory."""

    model_id: str
    seed: int
    training_regime: str
    training_configuration: Mapping[str, Any]
    checkpoint_path: Path
    calibrator_path: Path
    selected_epoch: int | None


@dataclass(frozen=True)
class FreezeValidationIssue:
    code: str
    message: str
    location: str

    def to_dict(self) -> dict[str, str]:
        return {"code": self.code, "location": self.location, "message": self.message}


@dataclass
class FinalFreezeValidationReport:
    inventory_path: str
    inventory_sha256: str | None = None
    frozen_artifact_set_sha256: str | None = None
    checked_files: int = 0
    errors: list[FreezeValidationIssue] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "inventory_path": self.inventory_path,
            "inventory_sha256": self.inventory_sha256,
            "frozen_artifact_set_sha256": self.frozen_artifact_set_sha256,
            "checked_files": self.checked_files,
            "errors": [issue.to_dict() for issue in self.errors],
            "valid": self.valid,
        }
        payload["report_sha256"] = canonical_json_sha256(payload)
        return payload


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _require_sha256(value: str, *, name: str) -> None:
    if not _is_sha256(value):
        raise FinalFreezeError(f"{name} must be a lowercase full SHA-256")


def _safe_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    candidate = PurePosixPath(value)
    return (
        not candidate.is_absolute()
        and candidate.as_posix() == value
        and ".." not in candidate.parts
        and "." not in candidate.parts
        and ":" not in candidate.parts[0]
    )


def _file_beneath_root(path: Path, *, artifact_root: Path) -> tuple[Path, str]:
    root = artifact_root.resolve(strict=True)
    if not root.is_dir():
        raise FinalFreezeError("artifact_root must be an existing directory")
    if path.is_symlink():
        raise FinalFreezeError(f"frozen artifacts may not be symlinks: {path}")
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise FinalFreezeError(f"frozen artifact is not a regular file: {path}")
    try:
        relative = resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise FinalFreezeError(f"frozen artifact escapes artifact_root: {path}") from exc
    return resolved, relative


def _file_record(path: Path, *, artifact_root: Path) -> dict[str, Any]:
    resolved, relative = _file_beneath_root(path, artifact_root=artifact_root)
    return {
        "path": relative,
        "size_bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def _validate_calibrator_payload(
    payload: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    configuration_sha256: str,
    split_manifest_sha256: str,
) -> None:
    required = {
        "schema_version": "1.0.0",
        "record_kind": "source_temperature_calibrator",
        "status": "frozen_source_validation",
        "fit_partition": "source_validation",
        "checkpoint_sha256": checkpoint_sha256,
        "training_configuration_sha256": configuration_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "target_subject_or_window_records_used": False,
        "target_labels_or_performance_used": False,
    }
    mismatches = [key for key, expected in required.items() if payload.get(key) != expected]
    if mismatches:
        raise FinalFreezeError(f"calibrator differs from frozen source contract: {mismatches}")
    body = dict(payload)
    claimed_hash = body.pop("record_sha256", None)
    if claimed_hash != canonical_json_sha256(body):
        raise FinalFreezeError("calibrator self-hash does not validate")


def _validate_training_rule(
    *,
    training_regime: str,
    configuration: Mapping[str, Any],
    selected_epoch: Any,
    seed: int,
) -> str:
    if configuration.get("seed") != seed:
        raise FinalFreezeError("training configuration seed differs from inventory seed")
    if training_regime == "fixed_epoch_neural":
        if configuration.get("checkpoint_selection_rule") != "fixed_last_epoch":
            raise FinalFreezeError("final neural checkpoint must use fixed_last_epoch")
        epochs = configuration.get("epochs")
        if (
            isinstance(epochs, bool)
            or not isinstance(epochs, int)
            or epochs < 1
            or selected_epoch != epochs
        ):
            raise FinalFreezeError("selected neural checkpoint must be the predeclared last epoch")
        return "fixed_last_epoch"
    if training_regime == "deterministic_classical":
        if selected_epoch is not None:
            raise FinalFreezeError("classical final fits must not declare a selected epoch")
        return "source_only_fit_no_epoch_selection"
    raise FinalFreezeError(f"unsupported final training regime: {training_regime!r}")


def build_final_freeze_inventory(
    model_inputs: Sequence[FrozenModelInput],
    *,
    artifact_root: Path,
    created_at_utc: str,
    code_commit: str,
    dataset_manifest_sha256: str,
    source_window_manifest_sha256: str,
    split_manifest_sha256: str,
    protocol_lock_sha256: str,
    target_seal_id: str,
    required_seed_order: Sequence[int],
) -> dict[str, Any]:
    """Build an inventory that can be published only while the target remains sealed."""

    for name, value in (
        ("dataset_manifest_sha256", dataset_manifest_sha256),
        ("source_window_manifest_sha256", source_window_manifest_sha256),
        ("split_manifest_sha256", split_manifest_sha256),
        ("protocol_lock_sha256", protocol_lock_sha256),
        ("target_seal_id", target_seal_id),
    ):
        _require_sha256(value, name=name)
    if not created_at_utc or not code_commit:
        raise FinalFreezeError("freeze inventory requires timestamp and code commit")
    seeds = tuple(int(seed) for seed in required_seed_order)
    if not seeds or len(set(seeds)) != len(seeds) or any(seed < 0 for seed in seeds):
        raise FinalFreezeError("required seeds must be a non-empty unique non-negative sequence")
    if not model_inputs:
        raise FinalFreezeError("freeze inventory requires at least one final model")

    entries: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in model_inputs:
        if not item.model_id or (item.model_id, item.seed) in seen:
            raise FinalFreezeError("model_id/seed pairs must be non-empty and unique")
        seen.add((item.model_id, item.seed))
        configuration = dict(item.training_configuration)
        configuration_sha256 = canonical_json_sha256(configuration)
        selection_rule = _validate_training_rule(
            training_regime=item.training_regime,
            configuration=configuration,
            selected_epoch=item.selected_epoch,
            seed=item.seed,
        )
        checkpoint = _file_record(item.checkpoint_path, artifact_root=artifact_root)
        calibrator = _file_record(item.calibrator_path, artifact_root=artifact_root)
        calibrator_payload = load_json_strict(item.calibrator_path)
        if not isinstance(calibrator_payload, Mapping):
            raise FinalFreezeError("calibrator root must be an object")
        _validate_calibrator_payload(
            calibrator_payload,
            checkpoint_sha256=str(checkpoint["sha256"]),
            configuration_sha256=configuration_sha256,
            split_manifest_sha256=split_manifest_sha256,
        )
        entries.append(
            {
                "model_id": item.model_id,
                "seed": item.seed,
                "training_regime": item.training_regime,
                "checkpoint_selection_rule": selection_rule,
                "selected_epoch": item.selected_epoch,
                "training_configuration": configuration,
                "training_configuration_sha256": configuration_sha256,
                "checkpoint": checkpoint,
                "calibrator": calibrator,
                "calibrator_record_sha256": calibrator_payload["record_sha256"],
            }
        )
    entries.sort(key=lambda value: (str(value["model_id"]).casefold(), int(value["seed"])))
    by_model: dict[str, list[int]] = {}
    for entry in entries:
        by_model.setdefault(str(entry["model_id"]), []).append(int(entry["seed"]))
    expected_seeds = Counter(seeds)
    incomplete = {
        model_id: observed
        for model_id, observed in by_model.items()
        if Counter(observed) != expected_seeds
    }
    if incomplete:
        raise FinalFreezeError(
            f"every final model must have the exact required seeds: {incomplete}"
        )

    artifact_identity = [
        {
            "model_id": entry["model_id"],
            "seed": entry["seed"],
            "training_configuration_sha256": entry["training_configuration_sha256"],
            "checkpoint_sha256": entry["checkpoint"]["sha256"],
            "calibrator_sha256": entry["calibrator"]["sha256"],
            "calibrator_record_sha256": entry["calibrator_record_sha256"],
        }
        for entry in entries
    ]
    payload: dict[str, Any] = {
        "schema_version": FINAL_FREEZE_SCHEMA_VERSION,
        "record_kind": "final_source_artifact_freeze",
        "status": "frozen_before_target_unlock",
        "evidence_status": "source_only_final_models_target_sealed",
        "created_at_utc": created_at_utc,
        "code_commit": code_commit,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "source_window_manifest_sha256": source_window_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "protocol_lock_sha256": protocol_lock_sha256,
        "target_seal_id": target_seal_id,
        "required_seed_order": list(seeds),
        "target_state": {
            "unlock_record": None,
            "target_subject_or_window_records_loaded": False,
            "target_predictions_or_performance_accessed": False,
        },
        "models": entries,
        "frozen_artifact_set_sha256": canonical_json_sha256(artifact_identity),
    }
    payload["inventory_sha256"] = canonical_json_sha256(payload)
    return payload


def write_final_freeze_inventory_new(
    inventory: Mapping[str, Any],
    destination: str | Path,
    *,
    allowed_root: str | Path,
) -> dict[str, str]:
    """Publish a validated source-only freeze once without replacing prior evidence."""

    body = dict(inventory)
    claimed_hash = body.pop("inventory_sha256", None)
    if claimed_hash != canonical_json_sha256(body):
        raise FinalFreezeError("freeze inventory self-hash does not validate")
    path = atomic_write_json_new(dict(inventory), destination, allowed_root=allowed_root)
    return {
        "path": str(path),
        "file_sha256": sha256_file(path),
        "inventory_sha256": str(claimed_hash),
    }


def _resolve_inventory_file(
    entry: Mapping[str, Any],
    *,
    artifact_root: Path,
    location: str,
) -> Path:
    relative = entry.get("path")
    if not _safe_relative_path(relative):
        raise FinalFreezeError(f"{location} has an unsafe relative path")
    root = artifact_root.resolve(strict=True)
    candidate = (root / str(relative)).resolve(strict=False)
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise FinalFreezeError(f"{location} escapes artifact_root") from exc
    if candidate.is_symlink() or not candidate.is_file():
        raise FinalFreezeError(f"{location} is missing, non-regular, or a symlink")
    return candidate


def validate_final_freeze_inventory_file(
    inventory_path: str | Path,
    *,
    artifact_root: str | Path,
    expected_split_manifest_sha256: str | None = None,
    expected_protocol_lock_sha256: str | None = None,
    expected_target_seal_id: str | None = None,
) -> FinalFreezeValidationReport:
    """Read-only validation of every frozen config/checkpoint/calibrator byte."""

    source = Path(inventory_path)
    report = FinalFreezeValidationReport(inventory_path=str(source))
    try:
        payload = load_json_strict(source)
        if not isinstance(payload, Mapping):
            raise FinalFreezeError("freeze inventory root must be an object")
        required = {
            "schema_version": FINAL_FREEZE_SCHEMA_VERSION,
            "record_kind": "final_source_artifact_freeze",
            "status": "frozen_before_target_unlock",
            "evidence_status": "source_only_final_models_target_sealed",
        }
        mismatches = [key for key, expected in required.items() if payload.get(key) != expected]
        if mismatches:
            raise FinalFreezeError(f"freeze inventory contract mismatch: {mismatches}")
        body = dict(payload)
        claimed_hash = body.pop("inventory_sha256", None)
        if claimed_hash != canonical_json_sha256(body):
            raise FinalFreezeError("freeze inventory self-hash does not validate")
        report.inventory_sha256 = str(claimed_hash)
        report.frozen_artifact_set_sha256 = str(payload.get("frozen_artifact_set_sha256"))
        expected_lineage = {
            "split_manifest_sha256": expected_split_manifest_sha256,
            "protocol_lock_sha256": expected_protocol_lock_sha256,
            "target_seal_id": expected_target_seal_id,
        }
        for name in (
            "dataset_manifest_sha256",
            "source_window_manifest_sha256",
            "split_manifest_sha256",
            "protocol_lock_sha256",
            "target_seal_id",
            "frozen_artifact_set_sha256",
        ):
            value = payload.get(name)
            if not _is_sha256(value):
                raise FinalFreezeError(f"freeze inventory lacks valid {name}")
            expected = expected_lineage.get(name)
            if expected is not None and value != expected:
                raise FinalFreezeError(f"freeze inventory {name} differs from expected lineage")
        target_state = payload.get("target_state")
        if target_state != {
            "unlock_record": None,
            "target_subject_or_window_records_loaded": False,
            "target_predictions_or_performance_accessed": False,
        }:
            raise FinalFreezeError("freeze inventory was not created with the target sealed")
        seeds_raw = payload.get("required_seed_order")
        if not isinstance(seeds_raw, list) or not seeds_raw:
            raise FinalFreezeError("freeze inventory lacks required seeds")
        seeds = tuple(seeds_raw)
        if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
            raise FinalFreezeError("freeze inventory contains invalid seeds")
        if len(set(seeds)) != len(seeds):
            raise FinalFreezeError("freeze inventory required seeds are duplicated")
        models = payload.get("models")
        if not isinstance(models, list) or not models:
            raise FinalFreezeError("freeze inventory lacks model artifacts")
        seen: set[tuple[str, int]] = set()
        by_model: dict[str, list[int]] = {}
        identities: list[dict[str, Any]] = []
        for index, raw_entry in enumerate(models):
            if not isinstance(raw_entry, Mapping):
                raise FinalFreezeError(f"models[{index}] must be an object")
            model_id = raw_entry.get("model_id")
            seed = raw_entry.get("seed")
            if not isinstance(model_id, str) or not model_id:
                raise FinalFreezeError(f"models[{index}] lacks model_id")
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise FinalFreezeError(f"models[{index}] has invalid seed")
            if (model_id, seed) in seen:
                raise FinalFreezeError("freeze inventory repeats a model_id/seed pair")
            seen.add((model_id, seed))
            by_model.setdefault(model_id, []).append(seed)
            configuration = raw_entry.get("training_configuration")
            if not isinstance(configuration, Mapping):
                raise FinalFreezeError(f"models[{index}] lacks training configuration")
            configuration_sha256 = canonical_json_sha256(configuration)
            if raw_entry.get("training_configuration_sha256") != configuration_sha256:
                raise FinalFreezeError(f"models[{index}] configuration hash mismatch")
            selection_rule = _validate_training_rule(
                training_regime=str(raw_entry.get("training_regime")),
                configuration=configuration,
                selected_epoch=raw_entry.get("selected_epoch"),
                seed=seed,
            )
            if raw_entry.get("checkpoint_selection_rule") != selection_rule:
                raise FinalFreezeError(f"models[{index}] selection rule mismatch")
            files: dict[str, Mapping[str, Any]] = {}
            for role in ("checkpoint", "calibrator"):
                file_entry = raw_entry.get(role)
                if not isinstance(file_entry, Mapping):
                    raise FinalFreezeError(f"models[{index}].{role} must be an object")
                file_path = _resolve_inventory_file(
                    file_entry,
                    artifact_root=Path(artifact_root),
                    location=f"models[{index}].{role}",
                )
                expected_size = file_entry.get("size_bytes")
                expected_hash = file_entry.get("sha256")
                if expected_size != file_path.stat().st_size or not _is_sha256(expected_hash):
                    raise FinalFreezeError(f"models[{index}].{role} metadata mismatch")
                if sha256_file(file_path) != expected_hash:
                    raise FinalFreezeError(f"models[{index}].{role} content hash mismatch")
                report.checked_files += 1
                files[role] = file_entry
            calibrator_path = _resolve_inventory_file(
                files["calibrator"],
                artifact_root=Path(artifact_root),
                location=f"models[{index}].calibrator",
            )
            calibrator_payload = load_json_strict(calibrator_path)
            if not isinstance(calibrator_payload, Mapping):
                raise FinalFreezeError(f"models[{index}] calibrator root is not an object")
            _validate_calibrator_payload(
                calibrator_payload,
                checkpoint_sha256=str(files["checkpoint"]["sha256"]),
                configuration_sha256=configuration_sha256,
                split_manifest_sha256=str(payload["split_manifest_sha256"]),
            )
            if raw_entry.get("calibrator_record_sha256") != calibrator_payload.get("record_sha256"):
                raise FinalFreezeError(f"models[{index}] calibrator record hash mismatch")
            identities.append(
                {
                    "model_id": model_id,
                    "seed": seed,
                    "training_configuration_sha256": configuration_sha256,
                    "checkpoint_sha256": files["checkpoint"]["sha256"],
                    "calibrator_sha256": files["calibrator"]["sha256"],
                    "calibrator_record_sha256": raw_entry["calibrator_record_sha256"],
                }
            )
        expected_seeds = Counter(seeds)
        if any(Counter(observed) != expected_seeds for observed in by_model.values()):
            raise FinalFreezeError("a final model does not have the exact required seed set")
        identities.sort(key=lambda value: (str(value["model_id"]).casefold(), int(value["seed"])))
        if canonical_json_sha256(identities) != payload.get("frozen_artifact_set_sha256"):
            raise FinalFreezeError("frozen artifact-set hash does not validate")
    except Exception as exc:
        report.errors.append(FreezeValidationIssue("FINAL_FREEZE_INVALID", str(exc), str(source)))
    return report


def _validate_target_manifest_contract(target_manifest: Mapping[str, Any]) -> tuple[str, str]:
    body = dict(target_manifest)
    claimed_hash = body.pop("split_manifest_sha256", None)
    if not _is_sha256(claimed_hash) or claimed_hash != canonical_json_sha256(body):
        raise TargetSealError("target split manifest self-hash does not validate")
    target_seal = target_manifest.get("target_seal")
    if not isinstance(target_seal, Mapping):
        raise TargetSealError("target split manifest lacks a target seal")
    seal_id = target_seal.get("seal_id")
    if (
        not _is_sha256(seal_id)
        or target_seal.get("status") != "sealed"
        or target_seal.get("unlock_record") is not None
        or target_manifest.get("target_performance_or_prediction_accessed") is not False
    ):
        raise TargetSealError("target split manifest is not in the exact unopened state")
    windows = target_manifest.get("windows")
    if not isinstance(windows, list) or not any(
        isinstance(window, Mapping) and window.get("partition") == "target_sealed"
        for window in windows
    ):
        raise TargetSealError("target split manifest lacks sealed target records")
    return str(claimed_hash), str(seal_id)


def validate_exact_target_unlock(
    target_manifest: Mapping[str, Any],
    unlock_record: Mapping[str, Any] | None,
    *,
    final_freeze_inventory_path: str | Path,
    artifact_root: str | Path,
) -> Mapping[str, Any]:
    """Authorize metadata access only when the unlock binds the exact frozen artifacts."""

    split_hash, seal_id = _validate_target_manifest_contract(target_manifest)
    basic = validate_target_unlock_record(
        unlock_record,
        split_manifest_sha256=split_hash,
        target_seal_id=seal_id,
    )
    freeze_report = validate_final_freeze_inventory_file(
        final_freeze_inventory_path,
        artifact_root=artifact_root,
        expected_split_manifest_sha256=split_hash,
        expected_protocol_lock_sha256=str(basic["protocol_lock_sha256"]),
        expected_target_seal_id=seal_id,
    )
    if not freeze_report.valid:
        raise TargetSealError(f"final source artifact freeze is invalid: {freeze_report.to_dict()}")
    freeze_file_hash = sha256_file(final_freeze_inventory_path)
    required_freeze_binding = {
        "final_freeze_inventory_sha256": freeze_report.inventory_sha256,
        "final_freeze_file_sha256": freeze_file_hash,
        "frozen_artifact_set_sha256": freeze_report.frozen_artifact_set_sha256,
        "source_models_and_calibrators_frozen": True,
    }
    mismatches = [
        key for key, expected in required_freeze_binding.items() if basic.get(key) != expected
    ]
    if mismatches:
        raise TargetSealError(
            f"target unlock does not bind the exact frozen artifacts: {mismatches}"
        )
    inventory = load_json_strict(final_freeze_inventory_path)
    if not isinstance(inventory, Mapping):
        raise TargetSealError("final freeze inventory root is not an object")
    if basic.get("code_commit") != inventory.get("code_commit"):
        raise TargetSealError("unlock code commit differs from the frozen model inventory")
    return basic


def record_target_opening_once(
    target_manifest: Mapping[str, Any],
    unlock_record: Mapping[str, Any] | None,
    *,
    final_freeze_inventory_path: str | Path,
    artifact_root: str | Path,
    opening_receipt_path: str | Path,
    receipt_root: str | Path,
    opened_at_utc: str,
    machine_record_sha256: str,
) -> dict[str, Any]:
    """Atomically consume an exact unlock before any target signal materialization."""

    _require_sha256(machine_record_sha256, name="machine_record_sha256")
    if not opened_at_utc:
        raise TargetSealError("target opening receipt requires opened_at_utc")
    authorized = validate_exact_target_unlock(
        target_manifest,
        unlock_record,
        final_freeze_inventory_path=final_freeze_inventory_path,
        artifact_root=artifact_root,
    )
    receipt: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "confirmatory_target_opening_receipt",
        "status": "unlock_consumed_before_materialization",
        "target_opening_number": 1,
        "opened_at_utc": opened_at_utc,
        "machine_record_sha256": machine_record_sha256,
        "split_manifest_sha256": authorized["split_manifest_sha256"],
        "target_seal_id": authorized["target_seal_id"],
        "unlock_record_sha256": canonical_json_sha256(authorized),
        "final_freeze_inventory_sha256": authorized["final_freeze_inventory_sha256"],
        "frozen_artifact_set_sha256": authorized["frozen_artifact_set_sha256"],
        "target_signals_materialized_at_receipt_time": False,
        "target_predictions_or_performance_accessed_at_receipt_time": False,
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt)
    try:
        atomic_write_json_new(receipt, opening_receipt_path, allowed_root=receipt_root)
    except FileExistsError as exc:
        raise TargetSealError("the one-time target unlock already has an opening receipt") from exc
    return receipt
