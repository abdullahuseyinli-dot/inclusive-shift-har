"""Evaluate a selected target-sealed FuSE-ReFrame checkpoint on source corruptions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.v2_corruptions import (
    default_corruption_suite,
    evaluate_corruption_suite,
)
from inclusive_shift_har.experiments.fuse_reframe_source import (
    _load_source_manifest,
    _partition_masks,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.fuse_reframe import FuSEReFrameHAR
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.v2_engine import v2_training_config_from_dict


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def run_source_corruption_suite(
    *,
    checkpoint_path: Path,
    source_manifest_path: Path,
    raw_csv_path: Path,
    output_path: Path,
    fold_id: str,
    device_name: str,
) -> dict[str, Any]:
    """Load one local checkpoint and write a create-only source diagnostic record."""

    if output_path.exists():
        raise FileExistsError(f"corruption output already exists: {output_path}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(device_name)
    raw_checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint = _mapping(raw_checkpoint, name="checkpoint")
    lineage = _mapping(checkpoint.get("lineage"), name="checkpoint lineage")
    if lineage.get("target_subject_or_window_records_loaded") is not False:
        raise PermissionError("corruption runner refuses a target-bearing checkpoint")
    if lineage.get("target_performance_or_prediction_accessed") is not False:
        raise PermissionError("corruption runner refuses a target-informed checkpoint")
    if lineage.get("source_fold_id") != fold_id:
        raise ValueError("requested corruption fold does not match the checkpoint selection fold")

    config = v2_training_config_from_dict(
        _mapping(checkpoint.get("configuration"), name="checkpoint configuration")
    )
    preprocessor = V2PhysicalPreprocessor.from_dict(
        _mapping(checkpoint.get("preprocessor"), name="checkpoint preprocessor")
    )
    participant_map = _mapping(
        checkpoint.get("participant_index_map"),
        name="participant index map",
    )
    model = FuSEReFrameHAR(
        raw_mean=torch.from_numpy(preprocessor.raw.mean.copy()),
        raw_scale=torch.from_numpy(preprocessor.raw.scale.copy()),
        invariant_mean=torch.from_numpy(preprocessor.invariant_mean.copy()),
        invariant_scale=torch.from_numpy(preprocessor.invariant_scale.copy()),
        clipping_thresholds=torch.from_numpy(preprocessor.clipping_thresholds.copy()),
        hidden_channels=config.hidden_channels,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        hierarchical=config.hierarchical,
        fusion_mode=config.fusion_mode,
        normalization=config.normalization,
        backbone=config.backbone,
        num_source_domains=(len(participant_map) if config.domain_adversarial_weight else None),
    )
    model.load_state_dict(_mapping(checkpoint.get("model_state"), name="model state"))
    model.to(device)

    manifest = _load_source_manifest(source_manifest_path)
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    class_names = tuple(str(item) for item in checkpoint["class_names"])
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("corruption runner requires the functional three-class checkpoint")
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    partitions = np.asarray(batch.partitions, dtype=np.str_)
    _, evaluation_mask = _partition_masks(
        participants,
        partitions,
        manifest,
        fold_id=fold_id,
    )
    suite = evaluate_corruption_suite(
        model,
        np.asarray(batch.signals[evaluation_mask], dtype=np.float32),
        np.asarray(batch.labels[evaluation_mask], dtype=np.int64),
        participants[evaluation_mask].tolist(),
        class_names=class_names,
        channel_scale=preprocessor.raw.scale,
        clipping_thresholds=preprocessor.clipping_thresholds,
        specs=default_corruption_suite(),
        batch_size=config.batch_size,
        seed=config.seed + 70_001,
        device=device,
    )
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_source_corruption_diagnostic",
        "status": "complete_target_sealed",
        "evidence_status": "source_selection_fold_diagnostic_not_confirmatory",
        "checkpoint": {
            "path": checkpoint_path.as_posix(),
            "sha256": sha256_file(checkpoint_path),
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
            "canonical_record_sha256": manifest["source_window_manifest_sha256"],
        },
        "fold_id": fold_id,
        "suite": suite,
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
        "selection_use_prohibited": True,
    }
    result["record_sha256"] = canonical_json_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("xb") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_source_corruption_suite(
        checkpoint_path=args.checkpoint,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        output_path=args.output,
        fold_id=args.fold_id,
        device_name=args.device,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
