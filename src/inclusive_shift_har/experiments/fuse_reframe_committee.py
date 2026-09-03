"""Post-analysis cross-fold checkpoint committee for FuSE-ReFrame bases.

The committee gives equal weight to the four inner checkpoints that were
selected without their outer pair. It is a source-development reanalysis,
because the idea was introduced after observing fixed-refit instability; it is
never relabelled as preregistered or confirmatory evidence.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray

from inclusive_shift_har.data.materialize import materialize_inclusivehar_windows
from inclusive_shift_har.data.windowing import WindowRecord
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.fuse_reframe_nested import _mask_for_subjects
from inclusive_shift_har.experiments.fuse_reframe_router import (
    FloatArray,
    _gate_probability,
    router_features,
)
from inclusive_shift_har.experiments.fuse_reframe_source import (
    PRIMARY_CHANNELS,
    _load_source_manifest,
)
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file
from inclusive_shift_har.models.fuse_reframe import FuSEReFrameHAR
from inclusive_shift_har.preprocessing.v2 import V2PhysicalPreprocessor
from inclusive_shift_har.training.v2_engine import v2_training_config_from_dict


def _mapping(value: object, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be an object with string keys")
    return cast(dict[str, Any], value)


def _load_json_hashed(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"hashed JSON artifact changed: {path}")
    payload = _mapping(json.loads(path.read_text(encoding="utf-8")), name=str(path))
    claimed = payload.get("record_sha256")
    if claimed is not None:
        unhashed = dict(payload)
        unhashed.pop("record_sha256")
        if claimed != canonical_json_sha256(unhashed):
            raise ValueError(f"JSON record self-hash changed: {path}")
    return payload


def _load_model(
    checkpoint_path: Path,
    *,
    expected_sha256: str,
    expected_outer_fold_id: str,
    device: torch.device,
) -> tuple[FuSEReFrameHAR, int, bool]:
    if sha256_file(checkpoint_path) != expected_sha256:
        raise ValueError("committee checkpoint hash changed")
    checkpoint = _mapping(
        torch.load(checkpoint_path, map_location="cpu", weights_only=True),
        name="committee checkpoint",
    )
    lineage = _mapping(checkpoint.get("lineage"), name="checkpoint lineage")
    if lineage.get("outer_fold_id") != expected_outer_fold_id:
        raise ValueError("committee checkpoint belongs to another outer fold")
    if (
        lineage.get("target_subject_or_window_records_loaded") is not False
        or lineage.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("committee refuses a target-informed checkpoint")
    config = v2_training_config_from_dict(
        _mapping(checkpoint.get("configuration"), name="checkpoint configuration")
    )
    preprocessor = V2PhysicalPreprocessor.from_dict(
        _mapping(checkpoint.get("preprocessor"), name="checkpoint preprocessor")
    )
    participant_map = _mapping(
        checkpoint.get("participant_index_map"), name="checkpoint participant map"
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
    model.load_state_dict(_mapping(checkpoint.get("model_state"), name="checkpoint model state"))
    model.to(device)
    model.eval()
    return model, config.batch_size, config.mixed_precision


def _predict_probabilities(
    model: FuSEReFrameHAR,
    signals: NDArray[np.float32],
    *,
    batch_size: int,
    mixed_precision: bool,
    device: torch.device,
) -> FloatArray:
    chunks: list[FloatArray] = []
    with torch.no_grad():
        for start in range(0, signals.shape[0], batch_size):
            inputs = torch.from_numpy(np.ascontiguousarray(signals[start : start + batch_size])).to(
                device
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16 if device.type == "cuda" else torch.bfloat16,
                enabled=mixed_precision and device.type == "cuda",
            ):
                output = model(inputs)
            chunks.append(
                np.asarray(output.logits.exp().detach().float().cpu().numpy(), dtype=np.float64)
            )
    probabilities = np.concatenate(chunks)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    return probabilities


def equal_weight_committee(members: list[FloatArray]) -> FloatArray:
    """Average aligned probability matrices with no fitted committee weight."""

    if not members or any(item.shape != members[0].shape for item in members):
        raise ValueError("committee members must be non-empty aligned matrices")
    values = np.stack(members, axis=0)
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("committee probabilities must be finite and non-negative")
    result = values.mean(axis=0)
    result /= result.sum(axis=1, keepdims=True)
    return np.asarray(result, dtype=np.float64)


def run_cross_fold_committee(
    *,
    router_result_directory: Path,
    source_manifest_path: Path,
    raw_csv_path: Path,
    output_directory: Path,
    device_name: str,
) -> dict[str, Any]:
    """Evaluate equal inner-checkpoint committees on one already-open source outer pair."""

    if output_directory.exists():
        raise FileExistsError(f"committee output directory already exists: {output_directory}")
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    router_result_path = router_result_directory / "result.json"
    router_result = _load_json_hashed(router_result_path)
    if (
        router_result.get("target_subject_or_window_records_loaded") is not False
        or router_result.get("target_performance_or_prediction_accessed") is not False
    ):
        raise PermissionError("committee refuses a target-informed router run")
    outer_fold_id = str(router_result["outer_fold_id"])
    base_ids = [str(item) for item in cast(list[object], router_result["base_ids"])]
    if base_ids != ["compact_residual_dann", "inception_erm"]:
        raise ValueError("committee requires the declared heterogeneous base order")
    manifest = _load_source_manifest(source_manifest_path)
    source_reference = _mapping(router_result["source_manifest"], name="source reference")
    if source_reference.get("canonical_record_sha256") != manifest["source_window_manifest_sha256"]:
        raise ValueError("committee source manifest differs from the router run")
    outer = next(
        (
            item
            for item in cast(list[dict[str, Any]], manifest["source_nested_cv"])
            if str(item["outer_fold_id"]) == outer_fold_id
        ),
        None,
    )
    if outer is None:
        raise ValueError("committee outer fold is absent from the source manifest")
    class_names = tuple(
        str(item)
        for item in manifest["ontology"]["runnable_track_schemas"]["functional_core"]["class_order"]
    )
    if class_names != ("mobility", "sitting", "standing"):
        raise ValueError("committee requires the functional three-class ontology")
    records = tuple(WindowRecord(**record) for record in manifest["windows"])
    batch = materialize_inclusivehar_windows(
        raw_csv_path,
        records,
        expected_source_sha256=str(manifest["source_artifact_sha256"]),
        ontology_track="functional_core",
        class_names=class_names,
        allowed_partitions={"source_train", "source_validation"},
    )
    signals = np.asarray(batch.signals, dtype=np.float32)
    labels = np.asarray(batch.labels, dtype=np.int64)
    participants = np.asarray(batch.participant_ids, dtype=np.str_)
    window_ids = np.asarray(batch.window_ids, dtype=np.str_)
    outer_test_subjects = [str(item) for item in outer["outer_test_subjects"]]
    outer_train_subjects = sorted(set(participants.tolist()) - set(outer_test_subjects), key=int)
    outer_test_mask = _mask_for_subjects(participants, outer_test_subjects)
    outer_train_mask = _mask_for_subjects(participants, outer_train_subjects)
    meta_preprocessor = V2PhysicalPreprocessor.fit(
        signals[outer_train_mask],
        participants[outer_train_mask].tolist(),
        declared_training_participants=set(outer_train_subjects),
        split_manifest_sha256=str(manifest["source_split_manifest_sha256"]),
        channel_names=PRIMARY_CHANNELS,
    )
    device = torch.device(device_name)
    summaries = _mapping(
        router_result["inner_candidate_summaries"], name="inner candidate summaries"
    )
    member_references: dict[str, list[dict[str, str]]] = {}
    committee_probabilities: dict[str, FloatArray] = {}
    for base_id in base_ids:
        inner_summary = _mapping(summaries[base_id], name=f"{base_id} inner summary")
        result_references = cast(list[dict[str, str]], inner_summary["inner_result_files"])
        if len(result_references) != 4:
            raise ValueError("committee requires exactly four inner checkpoints per base")
        members: list[FloatArray] = []
        member_references[base_id] = []
        for reference in result_references:
            inner_result_path = Path(reference["path"])
            inner_result = _load_json_hashed(inner_result_path, reference["sha256"])
            checkpoint = _mapping(inner_result["checkpoint"], name="inner checkpoint reference")
            checkpoint_path = Path(str(checkpoint["path"]))
            model, batch_size, mixed_precision = _load_model(
                checkpoint_path,
                expected_sha256=str(checkpoint["sha256"]),
                expected_outer_fold_id=outer_fold_id,
                device=device,
            )
            members.append(
                _predict_probabilities(
                    model,
                    signals[outer_test_mask],
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    device=device,
                )
            )
            member_references[base_id].append(
                {"path": checkpoint_path.as_posix(), "sha256": str(checkpoint["sha256"])}
            )
        committee_probabilities[base_id] = equal_weight_committee(members)

    gate_reference = _mapping(router_result["gate"], name="router gate reference")
    gate_path = Path(str(gate_reference["path"]))
    if sha256_file(gate_path) != gate_reference["sha256"]:
        raise ValueError("router gate hash changed")
    with gate_path.open("rb") as stream:
        gate_payload = _mapping(pickle.load(stream), name="router gate payload")
    outer_a = committee_probabilities[base_ids[0]]
    outer_b = committee_probabilities[base_ids[1]]
    features = router_features(
        outer_a,
        outer_b,
        signals[outer_test_mask],
        meta_preprocessor.clipping_thresholds,
    )
    gate_probability = _gate_probability(
        gate_payload.get("model"),
        cast(int | None, gate_payload.get("constant_choice")),
        features,
    )
    choose_a = gate_probability >= 0.5
    confidence_a = outer_a.max(axis=1) >= outer_b.max(axis=1)
    methods = {
        base_ids[0]: outer_a,
        base_ids[1]: outer_b,
        "fixed_mean": 0.5 * (outer_a + outer_b),
        "max_confidence": np.where(confidence_a[:, None], outer_a, outer_b),
        "learned_hard": np.where(choose_a[:, None], outer_a, outer_b),
        "learned_soft": gate_probability[:, None] * outer_a
        + (1.0 - gate_probability[:, None]) * outer_b,
    }
    reports = {
        method: classification_report(
            labels[outer_test_mask],
            probabilities,
            participants[outer_test_mask].tolist(),
            class_names=class_names,
        )
        for method, probabilities in methods.items()
    }
    output_directory.mkdir(parents=True)
    prediction_path = output_directory / "committee_predictions.npz"
    with prediction_path.open("xb") as stream:
        np.savez_compressed(
            stream,
            labels=labels[outer_test_mask],
            participant_ids=participants[outer_test_mask],
            window_ids=window_ids[outer_test_mask],
            compact_residual_dann_probabilities=methods[base_ids[0]],
            inception_erm_probabilities=methods[base_ids[1]],
            fixed_mean_probabilities=methods["fixed_mean"],
            max_confidence_probabilities=methods["max_confidence"],
            learned_hard_probabilities=methods["learned_hard"],
            learned_soft_probabilities=methods["learned_soft"],
            gate_probability_choose_base_a=gate_probability,
        )
    summary: dict[str, Any] = {
        "schema_version": "1.0.0",
        "record_kind": "fuse_reframe_cross_fold_committee_reanalysis",
        "status": "complete_target_sealed",
        "evidence_status": "post_analysis_source_development_not_independent",
        "outer_fold_id": outer_fold_id,
        "seed": router_result["seed"],
        "method_origin": (
            "introduced_after_observing_fixed_refit_instability; requires independent validation"
        ),
        "committee_rule": "equal_probability_mean_of_four_inner_selected_checkpoints",
        "member_references": member_references,
        "router_result": {
            "path": router_result_path.as_posix(),
            "sha256": sha256_file(router_result_path),
        },
        "source_manifest": {
            "path": source_manifest_path.as_posix(),
            "sha256": sha256_file(source_manifest_path),
            "canonical_record_sha256": manifest["source_window_manifest_sha256"],
        },
        "reports": reports,
        "predictions": {
            "path": prediction_path.as_posix(),
            "sha256": sha256_file(prediction_path),
        },
        "selection": {
            "committee_weights_fitted": False,
            "outer_labels_used_for_member_training_or_checkpoint_selection": False,
            "outer_results_inspired_method_family": True,
            "confirmatory_claim_allowed": False,
        },
        "target_subject_or_window_records_loaded": False,
        "target_performance_or_prediction_accessed": False,
    }
    summary["record_sha256"] = canonical_json_sha256(summary)
    result_path = output_directory / "result.json"
    with result_path.open("xb") as stream:
        stream.write(json.dumps(summary, indent=2, sort_keys=True).encode("utf-8"))
        stream.write(b"\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router-result-directory", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--raw-csv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_cross_fold_committee(
        router_result_directory=args.router_result_directory,
        source_manifest_path=args.source_manifest,
        raw_csv_path=args.raw_csv,
        output_directory=args.output_directory,
        device_name=args.device,
    )
    compact = {
        "status": result["status"],
        "evidence_status": result["evidence_status"],
        "outer_fold_id": result["outer_fold_id"],
        "primary": {method: report["primary"] for method, report in result["reports"].items()},
        "record_sha256": result["record_sha256"],
    }
    print(json.dumps(compact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
