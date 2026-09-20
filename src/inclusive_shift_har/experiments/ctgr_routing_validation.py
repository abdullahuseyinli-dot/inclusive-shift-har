"""One finite source/AICOS-development validation of always-consulted posture."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import confusion_matrix  # type: ignore[import-untyped]
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]

from inclusive_shift_har.data.aicos_har import (
    _read_metadata,
    aicos_in_inclusivehar_units,
    load_aicos_har,
    verify_aicos_archive,
)
from inclusive_shift_har.evaluation.metrics import classification_report
from inclusive_shift_har.experiments.aicos_external_benchmark import (
    EXPECTED_SOURCE_CSV_SHA256,
    EXPECTED_SOURCE_MANIFEST_SHA256,
    SOURCE_MANIFEST,
)
from inclusive_shift_har.experiments.confidence_triggered_gravity_residual import (
    _feature_views,
    _materialize_gravity,
    _selected_by_fold,
    apply_confidence_triggered_gravity_residual,
)
from inclusive_shift_har.experiments.ctgr_source_finalization import (
    SourceData,
    _predict_features,
    _serial_gravity_probability,
    _serial_probabilities,
    fit_b9,
    validate_artifacts,
    validate_source,
)
from inclusive_shift_har.experiments.microstate_posture_graph_nested import _materialize_source
from inclusive_shift_har.manifests.canonical import canonical_json_sha256, sha256_file

FloatArray = NDArray[np.float64]
CLASS_NAMES = ("mobility", "sitting", "standing")
METHODS = ("B6", "B9", "T9", "U9")
LEGACY = Path(".audit/v3/max-rnd/ctgr-evaluation-seed-11-001")
SELECTION = Path(".audit/v3/max-rnd/ctgr-selection-001/selection.json")
FINAL = Path(".audit/ctgr_source_finalization/ctgr-source-finalization-seed11-20260908-001")
HERA = Path(".audit/v5/hera-ctgr/retrospective-source-v1-001/cross_fitted_predictions.npz")
FROZEN_SHA = "623b7636222f0b52861709672395a7174b027ec09a3b63b50ab2e4969127ac3c"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def hashed_json(path: Path) -> dict[str, Any]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    body = {k: v for k, v in value.items() if k != "record_sha256"}
    if value.get("record_sha256") != canonical_json_sha256(body):
        raise ValueError(f"record checksum differs: {path}")
    return value


def alignment(reference: NDArray[Any], observed: NDArray[Any]) -> NDArray[np.int64]:
    """Require exact identity coverage, not positional coincidence."""
    if len(set(reference.tolist())) != len(reference) or len(set(observed.tolist())) != len(
        observed
    ):
        raise ValueError("duplicate window identifiers")
    if set(reference.tolist()) != set(observed.tolist()):
        raise ValueError("window coverage differs")
    index = {str(value): i for i, value in enumerate(observed)}
    return np.asarray([index[str(value)] for value in reference], dtype=np.int64)


def unconditional(base: FloatArray, expert: FloatArray, candidate: dict[str, Any]) -> FloatArray:
    """Change consultation only; retain the selected fold's original blend weight."""
    if candidate["id"] == "base_no_route":
        return base.copy()
    weight = float(candidate["blend_weight"])
    result = (1.0 - weight) * base + weight * expert
    return np.asarray(result / result.sum(axis=1, keepdims=True), dtype=np.float64)


def report(y: NDArray[Any], p: FloatArray, people: NDArray[Any]) -> dict[str, Any]:
    if p.shape != (y.size, 3) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("invalid probability shape or values")
    if not np.allclose(p.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError("probabilities do not sum to one")
    result = classification_report(y, p, people.tolist(), class_names=CLASS_NAMES)
    prediction = p.argmax(axis=1)
    class_recalls = []
    for item in result["participants"]:
        mask = people == item["participant_id"]
        cm = confusion_matrix(y[mask], prediction[mask], labels=[0, 1, 2])
        support = cm.sum(axis=1)
        recall = np.divide(np.diag(cm), support, out=np.zeros(3), where=support > 0)
        item["confusion"] = cm.tolist()
        item["class_support"] = support.tolist()
        item["class_recall"] = dict(zip(CLASS_NAMES, recall.tolist(), strict=True))
        item["accuracy"] = float(np.mean(y[mask] == prediction[mask]))
        item["nll"] = float(
            -np.log(np.clip(p[mask], 1e-15, 1.0)[np.arange(mask.sum()), y[mask]]).mean()
        )
        item["brier"] = float(np.mean(np.sum((p[mask] - np.eye(3)[y[mask]]) ** 2, axis=1)))
        class_recalls.append(recall)
    scores = sorted(float(v["macro_f1"]) for v in result["participants"])
    result["primary"]["bottom_30_percent_participant_macro_f1"] = float(
        np.mean(scores[: max(1, int(np.ceil(0.30 * len(scores))))])
    )
    result["primary"]["mean_participant_class_recall"] = dict(
        zip(CLASS_NAMES, np.mean(class_recalls, axis=0).tolist(), strict=True)
    )
    result["primary"]["mean_participant_nll"] = float(
        np.mean([v["nll"] for v in result["participants"]])
    )
    result["primary"]["mean_participant_brier"] = float(
        np.mean([v["brier"] for v in result["participants"]])
    )
    result["confusion"] = confusion_matrix(y, prediction, labels=[0, 1, 2]).tolist()
    return result


def comparison(candidate: dict[str, Any], control: dict[str, Any]) -> dict[str, Any]:
    a = {v["participant_id"]: v["macro_f1"] for v in candidate["participants"]}
    b = {v["participant_id"]: v["macro_f1"] for v in control["participants"]}
    if set(a) != set(b) or not a:
        raise ValueError("participant comparison is not paired")
    delta = np.asarray([a[p] - b[p] for p in sorted(a)])
    rng = np.random.default_rng(20260919)
    ix = rng.integers(0, delta.size, size=(10000, delta.size))
    recalls = {}
    for label, key in (("pooled", "window_level_diagnostics"), ("participant_mean", "primary")):
        field = "per_class_recall" if label == "pooled" else "mean_participant_class_recall"
        recalls[label] = {
            c: float(candidate[key][field][c] - control[key][field][c]) for c in CLASS_NAMES
        }
    return {
        "mean_delta": float(delta.mean()),
        "paired_95_percent_interval": np.quantile(delta[ix].mean(axis=1), [0.025, 0.975]).tolist(),
        "participant_deltas": dict(zip(sorted(a), delta.tolist(), strict=True)),
        "wins_harms_ties": [
            int(np.sum(delta > 1e-12)),
            int(np.sum(delta < -1e-12)),
            int(np.sum(np.abs(delta) <= 1e-12)),
        ],
        "worst_paired_harm": float(delta.min()),
        "bottom_tail_delta": float(
            candidate["primary"]["bottom_30_percent_participant_macro_f1"]
            - control["primary"]["bottom_30_percent_participant_macro_f1"]
        ),
        "recall_deltas": recalls,
    }


def numerical_gate(effect: dict[str, Any], *, external: bool) -> dict[str, Any]:
    checks = {
        "mean_gain": effect["mean_delta"] >= (0.02 if external else 0.0),
        "bottom_tail_non_regression": effect["bottom_tail_delta"] >= 0.0,
        "no_participant_harm_beyond_3pp": effect["worst_paired_harm"] >= -0.03,
    }
    if external:
        checks["positive_paired_interval"] = effect["paired_95_percent_interval"][0] > 0.0
    for aggregation, recalls in effect["recall_deltas"].items():
        for c, delta in recalls.items():
            checks[f"{aggregation}_{c}_recall"] = delta >= (-0.01 if c == "mobility" else 0.0)
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failed_checks": [k for k, v in checks.items() if not v],
    }


def save_predictions(
    path: Path,
    labels: NDArray[Any],
    people: NDArray[Any],
    windows: NDArray[Any],
    probabilities: dict[str, FloatArray],
    **extra: Any,
) -> None:
    payload: dict[str, Any] = {
        "labels": labels,
        "participant_ids": people,
        "window_ids": windows,
        **{f"probability__{k}": v for k, v in probabilities.items()},
        **extra,
    }
    with path.open("xb") as stream:
        np.savez_compressed(stream, **payload)


def _check_deadline(start: float) -> None:
    if time.monotonic() - start >= 7200:
        raise TimeoutError("two-hour compute limit reached; no extension")


def run(root: Path, evidence_root: Path, output: Path) -> dict[str, Any]:
    if not (output / "PROTOCOL.md").is_file() or (output / "execution_freeze.json").exists():
        raise FileExistsError("require predeclared protocol and a never-started output directory")
    start = time.monotonic()
    stage = "input_freeze"
    try:
        source_csv = evidence_root / "data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv"
        legacy = hashed_json(evidence_root / LEGACY / "result.json")
        selection_path = evidence_root / SELECTION
        selection = hashed_json(selection_path)
        if sha256_file(selection_path) != legacy["selection_record"]["sha256"]:
            raise ValueError("nested selection freeze lineage differs")
        if selection["outer_labels_used_for_selection"] or selection["outer_evaluation_performed"]:
            raise ValueError("selection freeze does not exclude outer evaluation")
        if (
            sha256_file(source_csv) != EXPECTED_SOURCE_CSV_SHA256
            or sha256_file(root / SOURCE_MANIFEST) != EXPECTED_SOURCE_MANIFEST_SHA256
        ):
            raise ValueError("frozen source input differs")
        predictor_path = root / FINAL / "predictor.pkl"
        if sha256_file(predictor_path) != FROZEN_SHA:
            raise ValueError("frozen final predictor differs")
        validate_artifacts(root / FINAL)
        hera_summary = json.loads(
            (root / "results/development/hera_ctgr_retrospective_v1_summary.json").read_text()
        )
        if sha256_file(evidence_root / HERA) != hera_summary["run"]["predictions_sha256"]:
            raise ValueError("matched HERA evidence changed")
        bound_inputs = {
            str(selection_path): sha256_file(selection_path),
            str(source_csv): sha256_file(source_csv),
            str(predictor_path): FROZEN_SHA,
            str(evidence_root / HERA): sha256_file(evidence_root / HERA),
        }
        for fold in legacy["folds"]:
            folder = evidence_root / LEGACY / fold["outer_fold_id"]
            rec = hashed_json(folder / "result.json")
            if sha256_file(folder / "result.json") != fold["sha256"]:
                raise ValueError("outer fold record changed")
            for name, field in (("models.pkl", "models"), ("predictions.npz", "predictions")):
                if sha256_file(folder / name) != rec[field]["sha256"]:
                    raise ValueError("original fold checkpoint or predictions changed")
                bound_inputs[str(folder / name)] = rec[field]["sha256"]
        freeze = {
            "utc": datetime.now(UTC).isoformat(),
            "seed": 11,
            "max_new_fits": 5,
            "external_fits": 0,
            "batch_windows": 256,
            "compute_limit_seconds": 7200,
            "methods": list(METHODS),
            "source_additional_reference": "HERA-v1-strict",
            "protocol_sha256": sha256_file(output / "PROTOCOL.md"),
            "interface_qualification_sha256": sha256_file(
                output / "preflight/interface_qualification.json"
            ),
            "input_sha256": bound_inputs,
            "git_head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=root, text=True
            ).strip(),
            "git_branch": subprocess.check_output(
                ["git", "branch", "--show-current"], cwd=root, text=True
            ).strip(),
            "git_status_before": subprocess.check_output(
                ["git", "status", "--short"], cwd=root, text=True
            ),
            "implementation_sha256": {
                str(p.relative_to(root)): sha256_file(p)
                for p in sorted((root / "src").rglob("*.py"))
            },
        }
        write_json(output / "execution_freeze.json", freeze)
        snapshots = output / "snapshots"
        snapshots.mkdir()
        for name in (
            "ctgr_routing_validation.py",
            "confidence_triggered_gravity_residual.py",
            "ctgr_source_finalization.py",
            "ctgr_native9_confirmation.py",
        ):
            (snapshots / name).write_bytes(
                (root / "src/inclusive_shift_har/experiments" / name).read_bytes()
            )
        (snapshots / "aicos_har.py").write_bytes(
            (root / "src/inclusive_shift_har/data/aicos_har.py").read_bytes()
        )
        stage = "source_materialization"
        manifest, signals, y, people, windows = _materialize_source(
            source_manifest_path=root / SOURCE_MANIFEST, raw_csv_path=source_csv
        )
        gravity = _materialize_gravity(
            manifest=manifest,
            raw_csv_path=source_csv,
            expected_participants=people,
            expected_windows=windows,
        )
        base, views = _feature_views(signals, gravity, sampling_rate_hz=50.0)
        data = SourceData(base=base, views=views, labels=y, participants=people, windows=windows)
        validate_source(data, expected_rows=725)
        probabilities = {m: np.full((725, 3), np.nan) for m in METHODS}
        selected = _selected_by_fold(selection)
        covered = np.zeros(725, dtype=np.int64)
        fold_reports = []
        maximum_replay_error = 0.0
        print("Frozen inputs verified. Replaying five source folds; five B9 fits only.", flush=True)
        for number, outer in enumerate(manifest["source_nested_cv"], 1):
            _check_deadline(start)
            fold_id = str(outer["outer_fold_id"])
            stage = f"source_{fold_id}"
            eval_mask = np.isin(people, [str(p) for p in outer["outer_test_subjects"]])
            train_mask = ~eval_mask
            folder = evidence_root / LEGACY / fold_id
            with (folder / "models.pkl").open("rb") as stream:
                original = pickle.load(stream)
            candidate = original["candidate"]
            if candidate["id"] != selected[fold_id]:
                raise ValueError("fold selection differs from original nested freeze")
            p_base = _serial_probabilities(original["base"], base[eval_mask])
            expert = _serial_gravity_probability(
                p_base,
                original["gravity_posture_expert"],
                views[candidate["expert_view"] or "physics"][eval_mask],
            )
            if candidate["id"] == "base_no_route":
                p_t9 = p_base.copy()
            else:
                p_t9, _ = apply_confidence_triggered_gravity_residual(
                    p_base,
                    expert,
                    confidence_threshold=candidate["confidence_threshold"],
                    blend_weight=candidate["blend_weight"],
                )
            with np.load(folder / "predictions.npz", allow_pickle=False) as saved:
                order = alignment(windows[eval_mask], saved["window_ids"])
                if not np.array_equal(y[eval_mask], saved["labels"][order]) or not np.array_equal(
                    people[eval_mask], saved["participant_ids"][order]
                ):
                    raise ValueError("source fold labels or participants are misaligned")
                for p, key in (
                    (p_base, "flat_rmrp_probabilities"),
                    (p_t9, "ctgr_probabilities"),
                    (expert, "gravity_posture_expert_probabilities"),
                ):
                    error = float(np.max(np.abs(p - saved[key][order])))
                    maximum_replay_error = max(maximum_replay_error, error)
                    if error > 1e-6:
                        raise ValueError(
                            f"source checkpoint replay mismatch: {fold_id}, {key}, {error}"
                        )
            dest = output / "source" / fold_id
            dest.mkdir(parents=True)
            record = {
                "fold": fold_id,
                "seed": 11,
                "source_train_people": sorted(set(people[train_mask].tolist())),
                "source_evaluation_people": sorted(set(people[eval_mask].tolist())),
                "selected_candidate": candidate,
                "new_fit": "B9 only",
                "new_fit_index": number,
                "training_window_ids_sha256": canonical_json_sha256(windows[train_mask].tolist()),
                "evaluation_window_ids_sha256": canonical_json_sha256(windows[eval_mask].tolist()),
            }
            write_json(dest / "fit_started.json", record)
            fit_started = time.monotonic()
            b9 = fit_b9(data, train_mask)
            with (dest / "B9.pkl").open("xb") as stream:
                pickle.dump(b9, stream, protocol=5)
            p_b9 = _serial_probabilities(b9, views["physics_plus_rmrp"][eval_mask])
            pred = {
                "B6": p_base,
                "B9": p_b9,
                "T9": p_t9,
                "U9": unconditional(p_base, expert, candidate),
            }
            for method in METHODS:
                probabilities[method][eval_mask] = pred[method]
            covered += eval_mask
            save_predictions(
                dest / "predictions.npz", y[eval_mask], people[eval_mask], windows[eval_mask], pred
            )
            record.update(
                {
                    "status": "complete",
                    "fit_and_inference_seconds": time.monotonic() - fit_started,
                    "model_sha256": sha256_file(dest / "B9.pkl"),
                    "reports": {
                        m: report(y[eval_mask], p, people[eval_mask]) for m, p in pred.items()
                    },
                }
            )
            write_json(dest / "result.json", record)
            fold_reports.append(record)
            print(f"Source fold {number}/5 complete; original nested recipe preserved.", flush=True)
        if not np.all(covered == 1):
            raise ValueError("source coverage is not exactly once")
        with np.load(evidence_root / HERA, allow_pickle=False) as h:
            order = alignment(windows, h["window_ids"])
            if not np.array_equal(y, h["labels"][order]) or not np.array_equal(
                people, h["participant_ids"][order]
            ):
                raise ValueError("HERA source comparison is not aligned")
            if not np.allclose(
                probabilities["T9"],
                h["seed_11__frozen_ctgr_probabilities"][order],
                atol=1e-6,
                rtol=0,
            ):
                raise ValueError("HERA's CTGR control differs from reproduced source T9")
            probabilities["HERA-v1-strict"] = np.asarray(
                h["seed_11__hera_ctgr_strict_probabilities"][order], dtype=np.float64
            )
        save_predictions(output / "source/predictions.npz", y, people, windows, probabilities)
        source_reports = {m: report(y, p, people) for m, p in probabilities.items()}
        source_effects = {
            m: comparison(source_reports["U9"], source_reports[m])
            for m in probabilities
            if m != "U9"
        }
        source_record: dict[str, Any] = {
            "status": "complete",
            "reports": source_reports,
            "U9_minus_controls": source_effects,
            "gate": numerical_gate(source_effects["T9"], external=False),
            "folds": fold_reports,
            "replay_max_absolute_error": maximum_replay_error,
            "new_fits": 5,
            "window_count": 725,
            "participant_count": 10,
            "source_runtime_seconds": time.monotonic() - start,
        }
        write_json(output / "source/result.json", source_record)
        print("Source comparison sealed. Loading AICOS provider development folds 1-5.", flush=True)
        stage = "external_materialization"
        _check_deadline(start)
        archive_path = root / "data/raw/external/aicos_har_v1/AICOS-HAR.zip"
        verification = verify_aicos_archive(archive_path)
        with ZipFile(archive_path) as archive:
            metadata = _read_metadata(archive)
        identity = metadata.AcquisitionID.str.split("/").str[0]
        provider_folds: dict[str, list[str]] = {}
        for fold in ("1", "2", "3", "4", "5", "test"):
            provider_folds[fold] = sorted(set(identity[metadata.Fold == fold]))
        for i, first in enumerate(provider_folds):
            for second in list(provider_folds)[i + 1 :]:
                if set(provider_folds[first]) & set(provider_folds[second]):
                    raise ValueError("provider participant folds overlap")
        write_json(
            output / "external/prewindow_partition.json",
            {
                "provider_fold_participants": provider_folds,
                "all_folds_participant_disjoint": True,
                "test_signals_or_predictions_loaded": False,
                "archive_sha256": verification.sha256,
            },
        )
        raw = load_aicos_har(
            archive_path, verification=verification, folds=("1", "2", "3", "4", "5")
        )
        target = aicos_in_inclusivehar_units(raw)
        del raw
        write_json(output / "external/data_qualification.json", target.summary())
        with predictor_path.open("rb") as stream:
            predictor = pickle.load(stream)
        probability = {m: np.empty((target.labels.size, 3), dtype=np.float64) for m in METHODS}
        stage = "external_inference"
        for begin in range(0, target.labels.size, 256):
            _check_deadline(start)
            stop = min(begin + 256, target.labels.size)
            b, v = _feature_views(
                np.asarray(target.signals[begin:stop], dtype=np.float64),
                np.asarray(target.gravity[begin:stop], dtype=np.float64),
                sampling_rate_hz=50.0,
            )
            pred = _predict_features(b, v, predictor["models"], predictor["selected_candidate"])
            for method in METHODS:
                probability[method][begin:stop] = pred[method]
            if begin % 4096 == 0:
                print(
                    f"External inference {stop}/{target.labels.size} windows; no target fits.",
                    flush=True,
                )
        complete = [
            str(p)
            for p in np.unique(target.participant_ids)
            if set(target.labels[target.participant_ids == p].tolist()) == {0, 1, 2}
        ]
        mask = np.isin(target.participant_ids, complete)
        if not complete:
            raise ValueError("no complete external development participants")
        save_predictions(
            output / "external/predictions.npz",
            target.labels,
            target.participant_ids,
            target.window_ids,
            probability,
            primary_complete_participant_mask=mask,
            session_ids=target.session_ids,
            trial_ids=target.trial_ids,
        )
        write_json(
            output / "external/prediction_seal.json",
            {
                "sha256": sha256_file(output / "external/predictions.npz"),
                "frozen_predictor_sha256": FROZEN_SHA,
                "utc": datetime.now(UTC).isoformat(),
                "source_only_models": True,
                "target_labels_used_for_fit_selection_or_calibration": False,
                "methods": list(METHODS),
                "model_fits": 0,
            },
        )
        stage = "external_scoring"
        external_reports = {
            m: report(target.labels[mask], p[mask], target.participant_ids[mask])
            for m, p in probability.items()
        }
        external_effects = {
            m: comparison(external_reports["U9"], external_reports[m]) for m in METHODS if m != "U9"
        }
        external_record: dict[str, Any] = {
            "status": "complete_conditional_interface",
            "reports": external_reports,
            "coverage_reports": {
                m: report(target.labels, p, target.participant_ids) for m, p in probability.items()
            },
            "U9_minus_controls": external_effects,
            "gate": numerical_gate(external_effects["T9"], external=True),
            "primary_participants": complete,
            "primary_window_count": int(mask.sum()),
            "all_participant_count": int(np.unique(target.participant_ids).size),
            "all_window_count": int(target.labels.size),
            "new_fits": 0,
            "native_and_derived_gravity_pooled": False,
        }
        write_json(output / "external/result.json", external_record)
        interface = json.loads((output / "preflight/interface_qualification.json").read_text())
        gates = {
            "source_non_regression": source_record["gate"]["passed"],
            "external_practical_gain": external_record["gate"]["passed"],
            "qualified_interface": bool(interface["portable_candidate_promotion_allowed"]),
        }
        result = {
            "status": "complete_with_explicit_interface_limitation",
            "evidence_status": "ADAPTIVE_SOURCE_AND_EXTERNAL_DEVELOPMENT_NOT_CONFIRMATION",
            "seed": 11,
            "source": source_record,
            "external": external_record,
            "nomination_gate": {"passed": all(gates.values()), "checks": gates},
            "decision": "nominate_for_independent_confirmation"
            if all(gates.values())
            else "retain_original_HERA_CTGR_close_this_routing_recipe",
            "runtime_seconds": time.monotonic() - start,
            "new_model_fits": 5,
            "new_external_fits": 0,
            "new_seeds": False,
            "provider_test_evaluation_performed": False,
            "target_P11_P20_accessed": False,
        }
        for path, digest in bound_inputs.items():
            if sha256_file(Path(path)) != digest:
                raise ValueError("preserved input changed during experiment")
        write_json(output / "result.json", result)
        write_json(
            output / "completion_manifest.json",
            {
                "status": "complete",
                "files": [
                    {
                        "path": p.relative_to(output).as_posix(),
                        "sha256": sha256_file(p),
                        "size_bytes": p.stat().st_size,
                    }
                    for p in sorted(output.rglob("*"))
                    if p.is_file()
                ],
            },
        )
        print(
            json.dumps(
                {
                    "decision": result["decision"],
                    "gates": gates,
                    "source_U9_minus_T9": source_effects["T9"],
                    "external_U9_minus_T9": external_effects["T9"],
                    "runtime_seconds": result["runtime_seconds"],
                },
                indent=2,
            ),
            flush=True,
        )
        return result
    except BaseException as error:
        write_json(
            output / "INCOMPLETE.json",
            {
                "status": "FAILED_PRESERVED",
                "stage": stage,
                "error": repr(error),
                "runtime_seconds": time.monotonic() - start,
                "utc": datetime.now(UTC).isoformat(),
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=4):
        run(args.repository_root.resolve(), args.evidence_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
