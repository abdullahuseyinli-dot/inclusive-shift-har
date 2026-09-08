"""Bounded source-only CTGR finalization; no confirmation evaluator is exposed."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import pickle
import subprocess
import threading
import time
from collections.abc import Callable
from copy import copy
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from joblib import parallel_backend  # type: ignore[import-untyped]
from numpy.typing import NDArray
from sklearn.ensemble import ExtraTreesClassifier  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _candidates,
    _feature_views,
    _fit_base,
    _fit_expert,
    _gravity_probability,
    _materialize_gravity,
    _selection_order,
    apply_confidence_triggered_gravity_residual,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _load_config as _load_ctgr_config,
)
from inclusive_shift_har.experiments.geometric_spectral_pyramid_nested import (
    _candidate_summary,
    _probabilities,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import _materialize_source
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
StringArray = NDArray[np.str_]
BoolArray = NDArray[np.bool_]
PARTICIPANTS = tuple(str(i) for i in range(1, 11))
PAIRS = (("1", "4"), ("6", "7"), ("2", "3"), ("5", "9"), ("8", "10"))
VIEWS = ("physics", "physics_plus_rmrp", "total_gsp", "dual_gsp")
ESTIMATORS = ("extra_trees_leaf3", "extra_trees_leaf1")
CLASS_NAMES = ("mobility", "sitting", "standing")
EVIDENCE_STATUS = "new_source_only_final_predictor_recipe_not_confirmation"
INPUTS = {
    "source_manifest": (
        "results/protocol/source_windows/inclusivehar_v4_source_development_v1_2.json",
        "1d49435ad9371a9701198e15ec30a35a5c3c38ee5458f6f1980de7d4306e76e2",
    ),
    "dataset_manifest": (
        "manifests/datasets/inclusivehar_v4.json",
        "52de5370682f13fbd9a4e9affe805b4f5ee0f28901d137743884b77471b24d29",
    ),
    "raw_csv": (
        "data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv",
        "0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34",
    ),
    "ctgr_config": (
        "configs/experiments/confidence_triggered_gravity_residual_v1.yaml",
        "50492a761da31b3ea142b48f4f88fbe1a497c18c994f2b65b3ca8fc0944aacfa",
    ),
    "round_a_config": (
        ".audit/round_a_flat_gravity/round-a-seed11-20260906-002/"
        "snapshots/round_a_flat_gravity_v1.yaml",
        "58723fb931bd79451149ca243c718705cb46d5d7c67a5b439622f688f9a4bcf4",
    ),
}


def expected_config() -> dict[str, Any]:
    """Return the complete finite executable contract; reject undeclared extensions."""
    return {
        "schema_version": "1.0.0",
        "experiment_id": "ctgr-source-finalization-v1",
        "evidence_status": EVIDENCE_STATUS,
        "seed": 11,
        "n_jobs": 4,
        "inference_n_jobs": 1,
        "sampling_rate_hz": 50.0,
        "window_samples": 128,
        "source_window_count": 725,
        "participants": list(PARTICIPANTS),
        "selection_pairs": [list(pair) for pair in PAIRS],
        "selection_fit_cap": 45,
        "final_fit_cap": 3,
        "total_fit_cap": 48,
        "candidate_count": 121,
        "mean_tie_tolerance": 0.005,
        "expert_estimators": list(ESTIMATORS),
        "base_wins_policy": "T9_equals_B6_U9_and_trigger_contrast_not_applicable",
        "B9": {
            "feature_view": "physics_plus_rmrp",
            "n_estimators": 500,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
            "class_weight": None,
            "weighting": "participant_first_present_classes_mean_one",
        },
        "target_access_allowed": False,
        "confirmation_evaluation_allowed": False,
        "extra_seeds_allowed": False,
    }


def load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config != expected_config():
        raise ValueError("source-finalization configuration differs from its finite contract")
    return expected_config()


def write_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["record_sha256"] = canonical_json_sha256(record)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(record, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return record


def read_record(path: Path) -> dict[str, Any]:
    record = json.loads(path.read_text(encoding="utf-8"))
    claimed = record.pop("record_sha256", None)
    if claimed != canonical_json_sha256(record):
        raise ValueError(f"record self-hash mismatch: {path.name}")
    return dict(record, record_sha256=claimed)


def save_arrays(path: Path, **arrays: Any) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


@dataclass(frozen=True)
class SourceData:
    base: FloatArray
    views: dict[str, FloatArray]
    labels: IntArray
    participants: StringArray
    windows: StringArray


def validate_source(data: SourceData, *, expected_rows: int | None = None) -> None:
    n = data.labels.size
    if expected_rows is not None and n != expected_rows:
        raise ValueError("source row count differs from the frozen functional-core population")
    if data.labels.shape != (n,) or set(data.labels.tolist()) != {0, 1, 2}:
        raise ValueError("source labels must use the fixed three-class ontology")
    if data.participants.shape != (n,) or set(data.participants.tolist()) != set(PARTICIPANTS):
        raise PermissionError("source-only finalization requires exactly P1--P10")
    if data.windows.shape != (n,) or len(set(data.windows.tolist())) != n:
        raise ValueError("source window identifiers must be aligned and unique")
    if tuple(data.views) != VIEWS:
        raise ValueError("expert feature views differ from original CTGR")
    for features in (data.base, *data.views.values()):
        if features.ndim != 2 or features.shape[0] != n or not np.isfinite(features).all():
            raise ValueError("source features must be aligned finite matrices")
    expected_b9 = np.concatenate((data.views["physics"], data.base), axis=1)
    if not np.array_equal(expected_b9, data.views["physics_plus_rmrp"]):
        raise ValueError("B9 must exactly retain Round A A4's physics-then-RMRP order")
    coverage = np.zeros(n, dtype=np.int64)
    for pair in PAIRS:
        train, evaluation = fold_masks(data, pair)
        if set(data.labels[train].tolist()) != {0, 1, 2}:
            raise ValueError("each source training fold must contain all three classes")
        coverage += evaluation
    if not np.all(coverage == 1):
        raise ValueError("selection folds must evaluate each source row exactly once")


def fold_masks(data: SourceData, pair: tuple[str, str]) -> tuple[BoolArray, BoolArray]:
    if pair not in PAIRS:
        raise PermissionError("selection pair is not one of the five original source pairs")
    evaluation = np.isin(data.participants, pair)
    train = ~evaluation
    if (
        set(data.participants[evaluation].tolist()) != set(pair)
        or set(data.participants[train].tolist()) != set(PARTICIPANTS) - set(pair)
        or np.any(train & evaluation)
    ):
        raise PermissionError("participant-exclusive source selection partition is invalid")
    return np.asarray(train, dtype=np.bool_), np.asarray(evaluation, dtype=np.bool_)


def participant_first_weights(labels: IntArray, participants: StringArray) -> FloatArray:
    if labels.shape != participants.shape or labels.size == 0:
        raise ValueError("B9 weights require nonempty aligned labels and participants")
    weights = np.empty(labels.size, dtype=np.float64)
    for person in np.unique(participants):
        person_mask = participants == person
        classes = np.unique(labels[person_mask])
        for label in classes:
            mask = person_mask & (labels == label)
            weights[mask] = 1.0 / (len(classes) * int(mask.sum()))
    return np.asarray(weights / weights.mean(), dtype=np.float64)


def fit_b9(data: SourceData, mask: BoolArray) -> Any:
    model = ExtraTreesClassifier(
        n_estimators=500,
        max_features="sqrt",
        min_samples_leaf=1,
        class_weight=None,
        random_state=11,
        n_jobs=4,
    )
    return model.fit(
        data.views["physics_plus_rmrp"][mask],
        data.labels[mask],
        sample_weight=participant_first_weights(data.labels[mask], data.participants[mask]),
    )


class FitLedger:
    """Consume attempts before fitting, including failed attempts; never retry."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.attempts: list[dict[str, Any]] = []
        self.phase_counts = {"selection": 0, "final": 0}
        self.failed = False
        (output / "fits").mkdir()

    def fit(self, phase: str, name: str, fit: Callable[[], Any], **metadata: Any) -> Any:
        caps = {"selection": 45, "final": 3}
        if phase not in caps or self.phase_counts[phase] >= caps[phase] or len(self.attempts) >= 48:
            raise RuntimeError("hard source-finalization fit cap reached")
        if self.failed:
            raise RuntimeError("a failed fit closes this ledger; retry is forbidden")
        if phase == "final" and not (self.output / "selection.json").is_file():
            raise PermissionError("selection must be frozen before a final fit")
        self.phase_counts[phase] += 1
        attempt = {"attempt": len(self.attempts) + 1, "phase": phase, "name": name, **metadata}
        self.attempts.append(attempt)
        stem = f"{len(self.attempts):02d}-{name}"
        write_record(self.output / "fits" / f"{stem}-started.json", attempt)
        started = time.monotonic()
        try:
            model = fit()
            model_path = self.output / "fits" / f"{stem}.pkl"
            with model_path.open("xb") as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
            write_record(
                self.output / "fits" / f"{stem}-completed.json",
                {
                    **attempt,
                    "seconds": time.monotonic() - started,
                    "model_path": model_path.relative_to(self.output).as_posix(),
                    "model_sha256": sha256_file(model_path),
                    "effective_parameters": model.get_params(deep=False),
                },
            )
            return model
        except BaseException as error:
            self.failed = True
            write_record(
                self.output / "fits" / f"{stem}-failed.json",
                {
                    **attempt,
                    "seconds": time.monotonic() - started,
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise


def candidate_predictions(
    base: FloatArray,
    experts: dict[tuple[str, str], FloatArray],
    candidates: list[dict[str, Any]],
) -> tuple[FloatArray, BoolArray]:
    probabilities = np.empty((len(candidates), base.shape[0], 3), dtype=np.float64)
    triggers = np.zeros((len(candidates), base.shape[0]), dtype=np.bool_)
    for index, candidate in enumerate(candidates):
        if candidate["id"] == "base_no_route":
            probabilities[index] = base
        else:
            probabilities[index], triggers[index] = apply_confidence_triggered_gravity_residual(
                base,
                experts[(candidate["expert_view"], candidate["posture_estimator"])],
                confidence_threshold=candidate["confidence_threshold"],
                blend_weight=candidate["blend_weight"],
            )
    return probabilities, triggers


def rank_oof(
    data: SourceData,
    candidates: list[dict[str, Any]],
    probabilities: FloatArray,
    triggers: BoolArray,
) -> tuple[list[dict[str, Any]], list[str], dict[str, Any]]:
    if (
        probabilities.shape != (121, data.labels.size, 3)
        or triggers.shape != probabilities.shape[:2]
    ):
        raise ValueError("OOF candidate arrays are misaligned")
    if (
        not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or not np.allclose(
            probabilities.sum(axis=2),
            1.0,
            atol=1e-12,
            rtol=0,
        )
    ):
        raise ValueError("OOF candidate probabilities are invalid")
    summaries, reports = [], {}
    for index, candidate in enumerate(candidates):
        report = classification_report(
            data.labels,
            probabilities[index],
            data.participants.tolist(),
            class_names=CLASS_NAMES,
        )
        reports[candidate["id"]] = report
        summary = _candidate_summary(candidate, [report])
        summary["inner_fold_count"] = 5
        summary["mean_trigger_fraction"] = float(
            np.mean([triggers[index, np.isin(data.participants, pair)].mean() for pair in PAIRS])
        )
        summaries.append(summary)
    ranking, eligible = _selection_order(summaries, candidates, tolerance=0.005)
    return ranking, eligible, reports


def final_probabilities(
    data: SourceData,
    models: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, FloatArray]:
    return _predict_features(data.base, data.views, models, selected)


def _serial_probabilities(model: Any, features: FloatArray) -> FloatArray:
    """Preserve fitted bytes while accumulating tree probabilities in fixed order."""
    inference_model = copy(model)
    inference_model.n_jobs = 1
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=1):
        return _probabilities(inference_model, features)


def _serial_gravity_probability(
    base_probability: FloatArray, model: Any, features: FloatArray
) -> FloatArray:
    inference_model = copy(model)
    inference_model.n_jobs = 1
    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=1):
        return _gravity_probability(base_probability, inference_model, features)


def predict_native_windows(
    signals: FloatArray, native_gravity: FloatArray, predictor: dict[str, Any]
) -> dict[str, FloatArray]:
    """Predict qualified native windows without labels or participant identities.

    Interface qualification and confirmation-label custody remain external contracts.
    Invalid inputs are rejected; this function never invents missing native gravity.
    """
    if (
        signals.ndim != 3
        or signals.shape[1:] != (128, 6)
        or native_gravity.shape != (*signals.shape[:2], 3)
        or not np.isfinite(signals).all()
        or not np.isfinite(native_gravity).all()
        or signals.shape[0] == 0
    ):
        raise ValueError(
            "native predictor requires finite aligned 128x6 IMU and 128x3 gravity windows"
        )
    if predictor["config"] != expected_config() or tuple(predictor["class_names"]) != CLASS_NAMES:
        raise ValueError("predictor interface differs from the frozen source contract")
    base, views = _feature_views(signals, native_gravity, sampling_rate_hz=50.0)
    return _predict_features(base, views, predictor["models"], predictor["selected_candidate"])


def _predict_features(
    base_features: FloatArray,
    views: dict[str, FloatArray],
    models: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, FloatArray]:
    base = _serial_probabilities(models["B6"], base_features)
    outputs = {"B6": base, "B9": _serial_probabilities(models["B9"], views["physics_plus_rmrp"])}
    if selected["id"] == "base_no_route":
        outputs["T9"] = base.copy()
        return outputs
    expert = _serial_gravity_probability(base, models["expert"], views[selected["expert_view"]])
    outputs["T9"], _ = apply_confidence_triggered_gravity_residual(
        base,
        expert,
        confidence_threshold=selected["confidence_threshold"],
        blend_weight=selected["blend_weight"],
    )
    weight = float(selected["blend_weight"])
    outputs["U9"] = (1.0 - weight) * base + weight * expert
    outputs["U9"] /= outputs["U9"].sum(axis=1, keepdims=True)
    return outputs


def _fit_recipe(
    data: SourceData, candidates: list[dict[str, Any]], output: Path, ledger: FitLedger
) -> dict[str, Any]:
    """Internal injectable seam for mocked contract tests; production validates real source first."""
    validate_source(data)
    oof = np.full((121, data.labels.size, 3), np.nan, dtype=np.float64)
    triggers = np.zeros((121, data.labels.size), dtype=np.bool_)
    save_arrays(
        output / "source_features.npz",
        base=data.base,
        labels=data.labels,
        participants=data.participants,
        windows=data.windows,
        **data.views,
    )
    for fold, pair in enumerate(PAIRS, 1):
        train, evaluation = fold_masks(data, pair)
        metadata = {
            "seed": 11,
            "train_participants": sorted(set(data.participants[train].tolist())),
            "evaluation_participants": list(pair),
            "training_windows_sha256": canonical_json_sha256(data.windows[train].tolist()),
            "evaluation_windows_sha256": canonical_json_sha256(data.windows[evaluation].tolist()),
        }
        base = ledger.fit(
            "selection",
            f"fold{fold}-B6",
            partial(_fit_base, data.base, data.labels, data.participants, train, seed=11, n_jobs=4),
            **metadata,
        )
        base_probability = _serial_probabilities(base, data.base[evaluation])
        experts = {}
        for view in VIEWS:
            for estimator in ESTIMATORS:
                model = ledger.fit(
                    "selection",
                    f"fold{fold}-{view}-{estimator}",
                    partial(
                        _fit_expert,
                        estimator,
                        data.views[view],
                        data.labels,
                        data.participants,
                        train,
                        seed=11,
                        n_jobs=4,
                    ),
                    **metadata,
                )
                experts[(view, estimator)] = _serial_gravity_probability(
                    base_probability, model, data.views[view][evaluation]
                )
        oof[:, evaluation], triggers[:, evaluation] = candidate_predictions(
            base_probability, experts, candidates
        )
    ranking, eligible, reports = rank_oof(data, candidates, oof, triggers)
    save_arrays(
        output / "selection_oof.npz",
        probabilities=oof,
        triggers=triggers,
        labels=data.labels,
        participant_ids=data.participants,
        window_ids=data.windows,
        candidate_ids=np.asarray([item["id"] for item in candidates], dtype=np.str_),
    )
    write_record(output / "selection_reports.json", {"reports": reports})
    selected = next(item for item in candidates if item["id"] == ranking[0]["candidate_id"])
    write_record(
        output / "selection.json",
        {
            "record_kind": "ctgr_source_finalization_selection_freeze",
            "evidence_status": EVIDENCE_STATUS,
            "candidates": candidates,
            "ranking": ranking,
            "eligible": eligible,
            "selected_candidate": selected,
            "selection_fit_attempts": ledger.phase_counts["selection"],
            "historical_oof_score_replaced": False,
            "confirmation_labels_accessed": False,
            "oof_sha256": sha256_file(output / "selection_oof.npz"),
        },
    )
    full = np.ones(data.labels.size, dtype=np.bool_)
    final_metadata = {
        "seed": 11,
        "train_participants": list(PARTICIPANTS),
        "training_windows_sha256": canonical_json_sha256(data.windows.tolist()),
    }
    models = {
        "B6": ledger.fit(
            "final",
            "B6",
            lambda: _fit_base(data.base, data.labels, data.participants, full, seed=11, n_jobs=4),
            **final_metadata,
        ),
        "B9": ledger.fit("final", "B9", lambda: fit_b9(data, full), **final_metadata),
    }
    if selected["id"] != "base_no_route":
        models["expert"] = ledger.fit(
            "final",
            "expert",
            lambda: _fit_expert(
                selected["posture_estimator"],
                data.views[selected["expert_view"]],
                data.labels,
                data.participants,
                full,
                seed=11,
                n_jobs=4,
            ),
            **final_metadata,
        )
    component_checkpoints = {}
    for receipt_path in sorted((output / "fits").glob("*-completed.json")):
        receipt = read_record(receipt_path)
        if receipt["phase"] == "final":
            component_checkpoints[receipt["name"]] = {
                "path": receipt["model_path"],
                "sha256": receipt["model_sha256"],
                "receipt_path": receipt_path.relative_to(output).as_posix(),
                "receipt_record_sha256": receipt["record_sha256"],
            }
    if set(component_checkpoints) != set(models):
        raise ValueError("final component checkpoint bindings are incomplete")
    bundle = {
        "models": models,
        "component_checkpoints": component_checkpoints,
        "selected_candidate": selected,
        "class_names": CLASS_NAMES,
        "config": expected_config(),
        "evidence_status": EVIDENCE_STATUS,
    }
    with (output / "predictor.pkl").open("xb") as stream:
        pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    expected = final_probabilities(data, models, selected)
    with (output / "predictor.pkl").open("rb") as stream:
        restored = pickle.load(stream)
    replay = final_probabilities(data, restored["models"], restored["selected_candidate"])
    if any(not np.array_equal(expected[name], replay[name]) for name in expected):
        raise ValueError("serialized final predictor did not replay exactly")
    save_arrays(output / "inference_replay.npz", **expected)
    return {
        "status": "complete",
        "evidence_status": EVIDENCE_STATUS,
        "selected_candidate_id": selected["id"],
        "predictor_ids": list(expected),
        "U9_trigger_contrast_status": "not_applicable_base_selected"
        if len(models) == 2
        else "defined",
        "fit_attempts": len(ledger.attempts),
        "fit_phase_counts": ledger.phase_counts,
        "source_resubstitution_performance_reported": False,
        "predictor_replay_exact": True,
        "target_accessed": False,
        "confirmation_evaluated": False,
        "historical_oof_score_replaced": False,
        "component_checkpoints": component_checkpoints,
    }


def _load_data(paths: dict[str, Path]) -> SourceData:
    manifest, signals, labels, people, windows = _materialize_source(
        source_manifest_path=paths["source_manifest"], raw_csv_path=paths["raw_csv"]
    )
    pairs = tuple(
        tuple(str(x) for x in fold["outer_test_subjects"]) for fold in manifest["source_nested_cv"]
    )
    if pairs != PAIRS:
        raise PermissionError("source manifest original participant pairs changed")
    if signals.shape != (725, 128, 6):
        raise ValueError("materialized source signal shape changed")
    gravity = _materialize_gravity(
        manifest=manifest,
        raw_csv_path=paths["raw_csv"],
        expected_participants=people,
        expected_windows=windows,
    )
    base, views = _feature_views(signals, gravity, sampling_rate_hz=50.0)
    data = SourceData(base, views, labels, people, windows)
    validate_source(data, expected_rows=725)
    return data


def seal_artifacts(output: Path, filename: str = "artifact_manifest.json") -> dict[str, Any]:
    return write_record(
        output / filename,
        {
            "files": [
                {
                    "path": path.relative_to(output).as_posix(),
                    "sha256": sha256_file(path),
                    "size_bytes": path.stat().st_size,
                }
                for path in sorted(output.rglob("*"))
                if path.is_file()
            ],
        },
    )


def validate_artifacts(output: Path) -> dict[str, Any]:
    validation_started = time.monotonic()
    validation_threads = set(threading.enumerate())
    manifest = read_record(output / "artifact_manifest.json")
    listed = set()
    for entry in manifest["files"]:
        path = (output / entry["path"]).resolve()
        if not path.is_relative_to(output.resolve()) or path == output.resolve():
            raise ValueError("artifact path escapes run directory")
        if (
            not path.is_file()
            or sha256_file(path) != entry["sha256"]
            or path.stat().st_size != entry["size_bytes"]
        ):
            raise ValueError(f"artifact integrity mismatch: {entry['path']}")
        listed.add(entry["path"])
        if path.suffix == ".json":
            read_record(path)
    actual = {p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file()}
    lifecycle = {"artifact_manifest.json", "validation.json", "completion_manifest.json"}
    if actual - lifecycle != listed or len(listed) != len(manifest["files"]):
        raise ValueError("artifact manifest completeness or uniqueness failed")
    completion_path = output / "completion_manifest.json"
    if completion_path.exists():
        completion = read_record(completion_path)
        completed_paths = set()
        for entry in completion["files"]:
            path = (output / entry["path"]).resolve()
            if not path.is_relative_to(output.resolve()) or not path.is_file():
                raise ValueError("completion manifest path invalid")
            if sha256_file(path) != entry["sha256"] or path.stat().st_size != entry["size_bytes"]:
                raise ValueError("completion manifest integrity failed")
            completed_paths.add(entry["path"])
        if completed_paths != actual - {"completion_manifest.json"}:
            raise ValueError("completion manifest completeness failed")
        if read_record(output / "validation.json")["status"] != "passed":
            raise ValueError("retained post-run validation failed")
    result = read_record(output / "result.json")
    selection = read_record(output / "selection.json")
    with np.load(output / "source_features.npz", allow_pickle=False) as feature:
        data = SourceData(
            feature["base"],
            {v: feature[v] for v in VIEWS},
            feature["labels"],
            feature["participants"],
            feature["windows"],
        )
    validate_source(data)
    with np.load(output / "selection_oof.npz", allow_pickle=False) as archive:
        for key, expected in (
            ("labels", data.labels),
            ("participant_ids", data.participants),
            ("window_ids", data.windows),
        ):
            if not np.array_equal(archive[key], expected):
                raise ValueError(f"OOF alignment mismatch: {key}")
        if archive["candidate_ids"].tolist() != [c["id"] for c in selection["candidates"]]:
            raise ValueError("candidate IDs are not aligned")
        ranking, eligible, reports = rank_oof(
            data, selection["candidates"], archive["probabilities"], archive["triggers"]
        )
        saved_oof = archive["probabilities"].copy()
        saved_triggers = archive["triggers"].copy()
    if ranking != selection["ranking"] or eligible != selection["eligible"]:
        raise ValueError("selection ranking did not replay")
    if (
        selection["selected_candidate"]["id"] != ranking[0]["candidate_id"]
        or result["selected_candidate_id"] != selection["selected_candidate"]["id"]
    ):
        raise ValueError("selected predictor differs from the frozen ranking winner")
    if reports != read_record(output / "selection_reports.json")["reports"]:
        raise ValueError("selection reports did not replay")
    starts = list((output / "fits").glob("*-started.json"))
    completed = list((output / "fits").glob("*-completed.json"))
    expected_fits = 47 if selection["selected_candidate"]["id"] == "base_no_route" else 48
    if (
        len(starts) != len(completed)
        or len(starts) != result["fit_attempts"]
        or len(starts) != expected_fits
    ):
        raise ValueError("fit attempts/checkpoints do not match the frozen recipe")
    if result["fit_phase_counts"] != {"selection": 45, "final": expected_fits - 45}:
        raise ValueError("phase fit counts violate the recipe")
    checkpoints = {}
    expected_final_names = {"B6", "B9"}
    if selection["selected_candidate"]["id"] != "base_no_route":
        expected_final_names.add("expert")
    component_checkpoints = {}
    for path in completed:
        receipt = read_record(path)
        model_path = (output / receipt["model_path"]).resolve()
        if (
            not model_path.is_relative_to(output.resolve())
            or sha256_file(model_path) != receipt["model_sha256"]
            or receipt["seed"] != 11
        ):
            raise ValueError("checkpoint receipt provenance mismatch")
        if receipt["name"] in checkpoints:
            raise ValueError("duplicate fit name")
        if receipt["name"] in expected_final_names:
            if receipt["phase"] != "final":
                raise ValueError("final component receipt has wrong fit phase")
            component_checkpoints[receipt["name"]] = {
                "path": receipt["model_path"],
                "sha256": receipt["model_sha256"],
                "receipt_path": path.relative_to(output).as_posix(),
                "receipt_record_sha256": receipt["record_sha256"],
            }
        elif receipt["phase"] != "selection":
            raise ValueError("unexpected final component receipt")
        checkpoints[receipt["name"]] = (model_path, receipt)
    expected_selection_names = {
        name
        for fold in range(1, 6)
        for name in [
            f"fold{fold}-B6",
            *[f"fold{fold}-{view}-{estimator}" for view in VIEWS for estimator in ESTIMATORS],
        ]
    }
    if (
        set(checkpoints) != expected_selection_names | expected_final_names
        or set(component_checkpoints) != expected_final_names
        or result.get("component_checkpoints") != component_checkpoints
    ):
        raise ValueError("final component checkpoint count/names/bindings mismatch")

    def restore(name: str, mask: BoolArray) -> Any:
        model_path, receipt = checkpoints[name]
        if set(receipt["train_participants"]) != set(data.participants[mask].tolist()) or receipt[
            "training_windows_sha256"
        ] != canonical_json_sha256(data.windows[mask].tolist()):
            raise ValueError("checkpoint training partition mismatch")
        # Open model bytes only after enclosing manifests and receipt hashes pass.
        with model_path.open("rb") as stream:
            return pickle.load(stream)

    with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=4):
        for fold, pair in enumerate(PAIRS, 1):
            train, evaluation = fold_masks(data, pair)
            base = _serial_probabilities(restore(f"fold{fold}-B6", train), data.base[evaluation])
            experts = {
                (view, estimator): _serial_gravity_probability(
                    base,
                    restore(f"fold{fold}-{view}-{estimator}", train),
                    data.views[view][evaluation],
                )
                for view in VIEWS
                for estimator in ESTIMATORS
            }
            probability, trigger = candidate_predictions(base, experts, selection["candidates"])
            if not np.array_equal(probability, saved_oof[:, evaluation]) or not np.array_equal(
                trigger, saved_triggers[:, evaluation]
            ):
                raise ValueError("selection checkpoints did not replay OOF predictions exactly")
        with (output / "predictor.pkl").open("rb") as stream:
            bundle = pickle.load(stream)
        if (
            bundle["selected_candidate"] != selection["selected_candidate"]
            or bundle["config"] != expected_config()
            or tuple(bundle["class_names"]) != CLASS_NAMES
            or bundle["evidence_status"] != EVIDENCE_STATUS
            or set(bundle["models"]) != expected_final_names
            or bundle.get("component_checkpoints") != component_checkpoints
        ):
            raise ValueError("predictor component/config/evidence/selection binding mismatch")
        full = np.ones(data.labels.size, dtype=np.bool_)
        restored_final = {name: restore(name, full) for name in sorted(expected_final_names)}
        checkpoint_replay = final_probabilities(
            data, restored_final, selection["selected_candidate"]
        )
        replay = final_probabilities(data, bundle["models"], bundle["selected_candidate"])
        if set(checkpoint_replay) != set(replay) or any(
            checkpoint_replay[name].dtype != replay[name].dtype
            or checkpoint_replay[name].shape != replay[name].shape
            or checkpoint_replay[name].tobytes() != replay[name].tobytes()
            for name in replay
        ):
            raise ValueError("assembled predictor differs from final-fit checkpoints")
        with np.load(output / "inference_replay.npz", allow_pickle=False) as archive:
            if set(archive.files) != set(replay) or any(
                archive[name].dtype != replay[name].dtype
                or archive[name].shape != replay[name].shape
                or archive[name].tobytes() != replay[name].tobytes()
                for name in replay
            ):
                raise ValueError("final predictor inference did not replay exactly")
    if read_record(output / "worker_shutdown.json")["remaining_task_threads"]:
        raise ValueError("task threads remain")
    remaining = [
        t.name for t in threading.enumerate() if t not in validation_threads and t.is_alive()
    ]
    if remaining:
        raise ValueError("validation task threads remain")
    return {
        "status": "passed",
        "model_fits": 0,
        "checked_files": len(listed),
        "selection_recomputed": True,
        "selection_checkpoint_replays": 45,
        "final_predictor_replayed": True,
        "final_checkpoint_replays": len(expected_final_names),
        "final_training_partition_verified": "all_source_P1_P10_rows",
        "final_component_checkpoint_bindings_verified": True,
        "validation_elapsed_seconds": time.monotonic() - validation_started,
        "remaining_validation_task_threads": remaining,
        "fit_attempts": len(starts),
    }


def run_finalization(
    *, evidence_root: Path, config_path: Path, output_directory: Path, code_commit: str
) -> dict[str, Any]:
    """Run one explicitly commissioned source-only fit workload; never retry an output."""
    output_directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    baseline_threads = set(threading.enumerate())
    ledger = FitLedger(output_directory)
    result: dict[str, Any] | None = None
    try:
        config = load_config(config_path)
        package_root = Path(__file__).resolve().parents[3]
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=package_root, text=True
        ).strip()
        if code_commit != actual_commit:
            raise ValueError("declared source commit is not the executing worktree commit")
        if subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=package_root,
            text=True,
        ).strip():
            raise ValueError("finalization requires a clean executing source worktree")
        paths = {name: evidence_root / relative for name, (relative, _) in INPUTS.items()}
        for name, path in paths.items():
            if (
                not path.resolve().is_relative_to(evidence_root.resolve())
                or sha256_file(path) != INPUTS[name][1]
            ):
                raise ValueError(f"frozen input hash/path mismatch: {name}")
        write_record(
            output_directory / "config_snapshot.json",
            {"config": config, "config_file_sha256": sha256_file(config_path)},
        )
        write_record(
            output_directory / "provenance.json",
            {
                "code_commit": actual_commit,
                "evidence_status": EVIDENCE_STATUS,
                "input_files": {
                    name: {"path": str(path.resolve()), "sha256": sha256_file(path)}
                    for name, path in paths.items()
                },
                "source_files": {
                    p.relative_to(package_root).as_posix(): sha256_file(p)
                    for p in sorted((package_root / "src").rglob("*.py"))
                },
                "implementation_contract_files": {
                    name: sha256_file(package_root / name)
                    for name in (
                        "pyproject.toml",
                        "uv.lock",
                        "docs/research/CTGR_SOURCE_FINALIZATION_V1_PROTOCOL.md",
                        "configs/experiments/ctgr_source_finalization_v1.yaml",
                    )
                },
                "dependencies": {
                    name: importlib.metadata.version(name)
                    for name in (
                        "numpy",
                        "scipy",
                        "scikit-learn",
                        "joblib",
                        "PyYAML",
                        "threadpoolctl",
                    )
                },
                "B6_expert_weighting": "historical_participant_class_cell_plus_balanced_class_weight",
                "B9_weighting": "participant_first_present_class_mean_one_no_class_weight",
                "prediction_workers": 1,
                "prediction_order": "serial_tree_order_via_shallow_estimator_view",
                "historical_score_replaced": False,
            },
        )
        candidates = _candidates(_load_ctgr_config(paths["ctgr_config"]))
        data = _load_data(paths)
        with threadpool_limits(limits=1), parallel_backend("threading", n_jobs=4):
            result = _fit_recipe(data, candidates, output_directory, ledger)
        write_record(output_directory / "result.json", result)
    except BaseException as error:
        write_record(
            output_directory / "INCOMPLETE.json",
            {
                "status": "incomplete",
                "error_type": type(error).__name__,
                "error": str(error),
                "fit_attempts": len(ledger.attempts),
                "phase_counts": ledger.phase_counts,
                "evidence_status": EVIDENCE_STATUS,
                "retry_allowed": False,
            },
        )
        raise
    finally:
        remaining = [
            t.name for t in threading.enumerate() if t not in baseline_threads and t.is_alive()
        ]
        write_record(
            output_directory / "runtime.json",
            {
                "elapsed_seconds": time.monotonic() - started,
                "fit_attempts": len(ledger.attempts),
                "phase_counts": ledger.phase_counts,
            },
        )
        write_record(
            output_directory / "worker_shutdown.json",
            {
                "remaining_task_threads": remaining,
                "experiment_subprocesses_launched": 0,
                "execution": "sequential_synchronous_fits_with_scoped_joblib_thread_backend",
            },
        )
        seal_artifacts(output_directory)
    try:
        validation = validate_artifacts(output_directory)
    except BaseException as error:
        write_record(
            output_directory / "validation.json",
            {
                "status": "failed",
                "model_fits": 0,
                "error_type": type(error).__name__,
                "error": str(error),
                "scientific_result_qualified": False,
            },
        )
        seal_artifacts(output_directory, "completion_manifest.json")
        raise
    write_record(output_directory / "validation.json", validation)
    seal_artifacts(output_directory, "completion_manifest.json")
    return {**(result or {}), "validation": validation}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--evidence-root", type=Path, required=True)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output-directory", type=Path, required=True)
    run.add_argument("--code-commit", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args(argv)
    result = (
        validate_artifacts(args.output_directory)
        if args.command == "validate"
        else run_finalization(
            evidence_root=args.evidence_root,
            config_path=args.config,
            output_directory=args.output_directory,
            code_commit=args.code_commit,
        )
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
