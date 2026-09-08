"""Portable, zero-fit replay of two separate retained development evidence lanes."""

from __future__ import annotations

import argparse
import math
import platform
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from inclusive_shift_har.manifests.canonical import (
    atomic_write_json_new,
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

CLASSES = ["mobility", "sitting", "standing"]
CTGR_METHODS = ["flat_rmrp", "gravity_posture_expert", "ctgr"]
L9V_METHODS = ["l9v", "lv", "bv", "b0", "f3"]
Array = NDArray[Any]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sealed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["record_sha256"] = canonical_json_sha256(result)
    return result


def verify_record(value: Any) -> dict[str, Any]:
    require(isinstance(value, dict), "JSON record must be an object")
    result = dict(value)
    digest = result.pop("record_sha256", None)
    require(digest == canonical_json_sha256(result), "JSON self-hash mismatch")
    return dict(value)


def relative_input(root: Path, relative: str) -> Path:
    path = Path(relative)
    require(not path.is_absolute() and ".." not in path.parts, "input must be relative")
    require("raw" not in {part.casefold() for part in path.parts}, "raw data access is forbidden")
    current = root
    for part in path.parts:
        current = current / part
        require(not current.is_symlink(), "symlink input is forbidden")
    resolved = current.resolve(strict=True)
    require(resolved.is_relative_to(root), "input escapes evidence root")
    require(resolved.is_file(), "input must be a regular file")
    return resolved


class Inputs:
    """Read only explicitly pinned small prediction/report artifacts."""

    def __init__(self, root: Path) -> None:
        self.root = root.resolve(strict=True)
        self.records: list[dict[str, Any]] = []

    def read(self, spec: Mapping[str, Any], filename: str) -> Path:
        relative = str(spec["path"])
        require(Path(relative).name == filename, f"unexpected input filename: {relative}")
        path = relative_input(self.root, relative)
        digest = sha256_file(path)
        matched = digest == spec["sha256"]
        self.records.append(
            {
                "path": relative,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
                "expected_sha256": spec["sha256"],
                "hash_matched": matched,
            }
        )
        require(matched, f"pinned hash mismatch: {relative}")
        return path

    def json(self, spec: Mapping[str, Any], filename: str) -> dict[str, Any]:
        return verify_record(load_json_strict(self.read(spec, filename)))


def ctgr_provenance(inputs: Inputs, result: dict[str, Any]) -> dict[str, Any]:
    """Hash model bundles and small manifests without deserializing model/raw bytes."""
    folds = result["folds"]
    expected_ids = [f"source_cv_{index:02d}" for index in range(1, 6)]
    require(
        len(folds) == 5 and sorted(row["outer_fold_id"] for row in folds) == expected_ids,
        "CTGR must bind all five fold receipts",
    )
    model_paths: set[str] = set()
    models = []
    for reference in folds:
        fold = inputs.json(reference, "result.json")
        require(fold["record_sha256"] == reference["record_sha256"], "fold self-hash binding")
        require(
            fold["record_kind"] == "confidence_triggered_gravity_residual_outer_fold"
            and fold["outer_fold_id"] == reference["outer_fold_id"]
            and fold["seed"] == 11,
            "fold identity/seed binding",
        )
        require(
            fold["code_commit"] == result["code_commit"], "fold/aggregate source commit binding"
        )
        model = fold["models"]
        require(
            Path(model["path"]).parent == Path(reference["path"]).parent,
            "fold model path must share fold directory",
        )
        require(model["path"] not in model_paths, "duplicate model bundle")
        model_paths.add(model["path"])
        inputs.read(model, "models.pkl")
        models.append({"outer_fold_id": fold["outer_fold_id"], **model})
    source_spec = result["source_manifest"]
    dataset_spec = result["dataset_manifest"]
    source = load_json_strict(inputs.read(source_spec, Path(source_spec["path"]).name))
    dataset = load_json_strict(inputs.read(dataset_spec, Path(dataset_spec["path"]).name))
    require(
        source["manifest_kind"] == "source_development_windows"
        and source["target_subject_or_window_records_included"] is False
        and source["target_performance_or_prediction_accessed"] is False,
        "source-only window manifest scope",
    )
    require(
        dataset["manifest_kind"] == "dataset" and dataset["dataset_id"] == source["dataset_id"],
        "dataset/source manifest identity",
    )
    raw = [row for row in dataset["artifacts"] if row["role"] == "raw_sensor_table"]
    require(
        len(raw) == 1 and raw[0]["expected_sha256"] == source["source_artifact_sha256"],
        "indirect raw source hash binding",
    )
    windows = source["windows"]
    require(
        len(windows) == source["source_window_count"]
        and len({row["window_id"] for row in windows}) == len(windows),
        "source manifest window IDs",
    )
    return {
        "model_bundles_byte_hashed": models,
        "model_bundle_count": len(models),
        "models_deserialized": False,
        "source_window_manifest": source_spec,
        "dataset_manifest": dataset_spec,
        "source_windows": windows,
        "raw_source_declared_sha256": raw[0]["expected_sha256"],
        "raw_source_declared_size_bytes": raw[0]["expected_size_bytes"],
        "raw_bytes_rehashed": False,
        "raw_provenance_scope": "indirect_manifest_declaration",
    }


def l9v_provenance(
    inputs: Inputs, config: dict[str, Any], entries: dict[str, Any]
) -> dict[str, Any]:
    root = Path(config["completion"]["path"]).parent
    models = sorted(name for name in entries if Path(name).suffix in {".pkl", ".joblib"})
    require(
        models == [f"checkpoints/l9v--fold-{index}.pkl" for index in range(5)],
        "L9v must bind all five model checkpoints",
    )

    def spec(name: str) -> dict[str, Any]:
        relative = Path(name)
        require(
            not relative.is_absolute() and ".." not in relative.parts, "manifest path traversal"
        )
        return {"path": (root / relative).as_posix(), "sha256": entries[name]["sha256"]}

    model_records = []
    for name in models:
        reference = spec(name)
        path = inputs.read(reference, Path(name).name)
        require(path.stat().st_size == entries[name]["size_bytes"], "model checkpoint byte size")
        model_records.append(reference)
    source = inputs.json(spec("source_manifest.json"), "source_manifest.json")
    data = inputs.json(spec("input_manifest.json"), "input_manifest.json")
    require(
        source["record_kind"] == "fog_l9v_source_manifest"
        and data["record_kind"] == "fog_l9v_input_manifest",
        "L9v provenance schema",
    )
    raw = data["raw_source"]
    return {
        "model_checkpoints_byte_hashed": model_records,
        "model_checkpoint_count": len(models),
        "models_deserialized": False,
        "source_manifest_record_sha256": source["record_sha256"],
        "source_file_hash_declarations": source["files"],
        "input_manifest_record_sha256": data["record_sha256"],
        "raw_source_declared_sha256": raw["sha256"],
        "raw_source_declared_size_bytes": raw["size_bytes"],
        "raw_bytes_rehashed": False,
        "raw_provenance_scope": "indirect_manifest_declaration",
    }


def load_arrays(path: Path, keys: set[str]) -> dict[str, Array]:
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == keys, "prediction archive schema mismatch")
        return {key: archive[key] for key in archive.files}


def probabilities(value: Array, rows: int) -> None:
    require(value.dtype == np.float64 and value.shape == (rows, 3), "probability schema")
    require(bool(np.isfinite(value).all()), "nonfinite probabilities")
    require(bool(((value >= 0) & (value <= 1)).all()), "probability range")
    require(bool(np.allclose(value.sum(axis=1), 1, atol=1e-12, rtol=0)), "simplex")


def ids(values: Array, rows: int, *, unique: bool = False) -> None:
    require(values.ndim == 1 and values.size == rows and values.dtype.kind == "U", "ID schema")
    require(bool((np.char.str_len(values) > 0).all()), "empty ID")
    if unique:
        require(len(set(values.tolist())) == rows, "duplicate window ID")


def metrics(y: Array, p: Array, people: Array) -> dict[str, Any]:
    """Compute fixed-three-class metrics independently of historical evaluators."""
    require(y.dtype == np.int64 and y.ndim == 1 and y.size > 0, "label schema")
    require(bool(np.isin(y, [0, 1, 2]).all()), "label range")
    probabilities(p, y.size)
    ids(people, y.size)

    def summarize(labels: Array, probability: Array) -> dict[str, Any]:
        prediction = probability.argmax(axis=1)
        confusion = np.bincount(3 * labels + prediction, minlength=9).reshape(3, 3)
        diagonal = np.diag(confusion).astype(np.float64)
        support = confusion.sum(axis=1)
        denominator = support + confusion.sum(axis=0)
        recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
        f1 = np.divide(2 * diagonal, denominator, out=np.zeros(3), where=denominator != 0)
        return {
            "window_count": int(labels.size),
            "confusion_matrix": confusion.tolist(),
            "class_support": support.tolist(),
            "class_recall": recall.tolist(),
            "class_f1": f1.tolist(),
            "macro_f1": float(f1.mean()),
            "accuracy": float((prediction == labels).mean()),
            "nll": float(
                -np.log(np.maximum(probability[np.arange(labels.size), labels], 1e-12)).mean()
            ),
            "multiclass_brier": float(np.square(probability - np.eye(3)[labels]).sum(1).mean()),
        }

    participants = []
    for person in sorted(set(people.tolist())):
        mask = people == person
        participants.append({"participant_id": person, **summarize(y[mask], p[mask])})
    values = np.array([row["macro_f1"] for row in participants])
    primary = {
        "mean_participant_macro_f1": float(values.mean()),
        "bottom_30_percent_participant_macro_f1": float(
            np.sort(values)[: math.ceil(0.3 * len(values))].mean()
        ),
        "bottom_30_participant_count": math.ceil(0.3 * len(values)),
        "worst_participant_macro_f1": float(values.min()),
        "lower_decile_participant_macro_f1": float(np.quantile(values, 0.1)),
        "participant_macro_class_recall": dict(
            zip(
                CLASSES,
                np.mean([row["class_recall"] for row in participants], axis=0).tolist(),
                strict=True,
            )
        ),
        "mean_participant_nll": float(np.mean([row["nll"] for row in participants])),
        "mean_participant_multiclass_brier": float(
            np.mean([row["multiclass_brier"] for row in participants])
        ),
    }
    return {
        "class_names": CLASSES,
        "fixed_class_rule": "zero_division_zero",
        "sample_count": int(y.size),
        "participant_count": len(participants),
        "participants": participants,
        "primary": primary,
        "pooled": summarize(y, p),
    }


def close(actual: Any, expected: Any, context: str) -> None:
    require(bool(np.allclose(actual, expected, atol=1e-12, rtol=0)), f"metric mismatch: {context}")


def compare_report(actual: dict[str, Any], expected: dict[str, Any], *, ctgr: bool) -> None:
    require(expected["class_names"] == CLASSES, "class order mismatch")
    require(actual["sample_count"] == expected["sample_count"], "sample count mismatch")
    for key in (
        "mean_participant_macro_f1",
        "bottom_30_percent_participant_macro_f1",
        "worst_participant_macro_f1",
    ):
        close(actual["primary"][key], expected["primary"][key], key)
    old_people = {row["participant_id"]: row for row in expected["participants"]}
    require(
        set(old_people) == {row["participant_id"] for row in actual["participants"]},
        "report participant roster mismatch",
    )
    for row in actual["participants"]:
        reference = old_people[row["participant_id"]]
        require(row["window_count"] == reference["window_count"], "participant support mismatch")
        close(row["macro_f1"], reference["macro_f1"], "participant F1")
        if not ctgr:
            for key in ("confusion_matrix", "class_recall", "nll", "multiclass_brier"):
                close(row[key], reference[key], key)
    if ctgr:
        old = expected["window_level_diagnostics"]
        close(actual["pooled"]["confusion_matrix"], old["confusion_matrix"], "pooled confusion")
        close(
            actual["pooled"]["nll"],
            expected["calibration"]["negative_log_likelihood"],
            "pooled NLL",
        )
        close(
            actual["pooled"]["multiclass_brier"],
            expected["calibration"]["multiclass_brier_score"],
            "pooled Brier",
        )
    else:
        for key in ("mean_participant_nll", "mean_participant_multiclass_brier"):
            close(actual["primary"][key], expected["primary"][key], key)
        for name in CLASSES:
            close(
                actual["primary"]["participant_macro_class_recall"][name],
                expected["primary"]["participant_macro_class_recall"][name],
                name,
            )


def replay_ctgr(inputs: Inputs, config: dict[str, Any]) -> dict[str, Any]:
    result = inputs.json(config["result"], "result.json")
    require(
        result["record_kind"] == "confidence_triggered_gravity_residual_fixed_seed_aggregate",
        "CTGR record kind",
    )
    require(
        result["seed"] == 11 and result["confirmatory_claim_allowed"] is False,
        "CTGR evidence scope",
    )
    require(result["target_performance_or_prediction_accessed"] is False, "CTGR target access")
    provenance = ctgr_provenance(inputs, result)
    path = inputs.read(config["predictions"], "all_outer_predictions.npz")
    require(
        result["predictions"]["sha256"] == config["predictions"]["sha256"],
        "CTGR prediction binding",
    )
    arrays = load_arrays(
        path,
        {"labels", "participant_ids", "window_ids"}
        | {f"{name}_probabilities" for name in CTGR_METHODS},
    )
    rows = int(config["scored_rows"])
    ids(arrays["window_ids"], rows, unique=True)
    ids(arrays["participant_ids"], rows)
    require(
        sorted(set(arrays["participant_ids"].tolist())) == sorted(config["participant_ids"]),
        "CTGR participant roster",
    )
    source_windows = {row["window_id"]: row for row in provenance.pop("source_windows")}
    for window, person, label in zip(
        arrays["window_ids"],
        arrays["participant_ids"],
        arrays["labels"],
        strict=True,
    ):
        require(
            window in source_windows and source_windows[window]["subject_id"] == person,
            "OOF window/participant source provenance",
        )
        source_label = source_windows[window].get("canonical_labels", {}).get("functional_core")
        require(source_label in CLASSES, "OOF window is not functional-core eligible")
        require(CLASSES.index(source_label) == label, "OOF label/source ontology binding")
    reports = {}
    for name in CTGR_METHODS:
        report = metrics(
            arrays["labels"], arrays[f"{name}_probabilities"], arrays["participant_ids"]
        )
        compare_report(report, result["aggregate_reports"][name], ctgr=True)
        reports[name] = report
    return {
        "lane": "ctgr_source_seed11",
        "evidence_status": result["evidence_status"],
        "source_code_commit": result["code_commit"],
        "reports": reports,
        "historical_calibration_aggregation": "pooled_windows",
        "new_equal_participant_probability_metrics": "descriptive_reaggregation_only",
        "byte_and_source_provenance": provenance,
    }


def replay_l9v(inputs: Inputs, config: dict[str, Any]) -> dict[str, Any]:
    result = inputs.json(config["result"], "result.json")
    analysis = inputs.json(config["analysis"], "analysis.json")
    completion = inputs.json(config["completion"], "completion_manifest.json")
    erratum = inputs.json(config["erratum"], "ERRATUM.json")
    require(result["InclusiveHAR_P11_P20_loaded"] is False, "L9v target access")
    require(
        result["completed_fit_count"] == 5 and result["advancement"]["status"] == "fail",
        "L9v run scope",
    )
    entries = {item["path"]: item for item in completion["artifacts"]}
    require(len(entries) == len(completion["artifacts"]), "duplicate completion manifest entry")
    provenance = l9v_provenance(inputs, config, entries)
    for key in ("result", "analysis", "predictions"):
        filename = Path(config[key]["path"]).name
        require(entries[filename]["sha256"] == config[key]["sha256"], "completion input binding")
    require(
        erratum["predictions_file_sha256"] == config["predictions"]["sha256"],
        "erratum prediction binding",
    )
    require(
        erratum["scientific_effect"]["predictions_affected"] is False,
        "erratum numerical qualification",
    )
    arrays = load_arrays(
        inputs.read(config["predictions"], "predictions.npz"),
        {
            "method_ids",
            "observable_probabilities",
            "scored_probabilities",
            "l9v_available_probabilities",
            "availability_mask",
            "available_indices",
            "scoring_indices",
            "scored_labels",
            "observable_window_ids",
            "observable_participant_ids",
            "observable_fold_index",
            "scored_participant_ids",
            "scored_window_ids",
            "scored_fold_index",
            "observable_decisions",
            "scored_decisions",
            "fallback_source_method_index",
            "fallback_source_method_id",
        },
    )
    require(arrays["method_ids"].tolist() == L9V_METHODS, "L9v method order")
    observable = int(config["observable_rows"])
    scored = int(config["scored_rows"])
    for prefix, rows in (("observable", observable), ("scored", scored)):
        ids(arrays[f"{prefix}_window_ids"], rows, unique=True)
        ids(arrays[f"{prefix}_participant_ids"], rows)
        require(
            sorted(set(arrays[f"{prefix}_participant_ids"].tolist())) == config["participant_ids"],
            "L9v roster",
        )
        require(arrays[f"{prefix}_probabilities"].shape == (5, rows, 3), "L9v probability stack")
        folds = arrays[f"{prefix}_fold_index"]
        require(folds.dtype == np.int64 and folds.shape == (rows,), "fold index schema")
        require(bool(np.isin(folds, [0, 1, 2, 3, 4]).all()), "fold index range")
        for person in config["participant_ids"]:
            require(
                len(set(folds[arrays[f"{prefix}_participant_ids"] == person].tolist())) == 1,
                "participant spans evaluation folds",
            )
        for probability in arrays[f"{prefix}_probabilities"]:
            probabilities(probability, rows)
        require(
            np.array_equal(
                arrays[f"{prefix}_decisions"], arrays[f"{prefix}_probabilities"].argmax(2)
            ),
            "saved decisions",
        )
    score_indices = arrays["scoring_indices"]
    require(
        score_indices.dtype == np.int64 and score_indices.shape == (scored,), "scoring index schema"
    )
    require(
        bool(((score_indices >= 0) & (score_indices < observable)).all())
        and len(set(score_indices.tolist())) == scored,
        "scoring index range/uniqueness",
    )
    for suffix in ("window_ids", "participant_ids", "fold_index"):
        require(
            np.array_equal(
                arrays[f"scored_{suffix}"], arrays[f"observable_{suffix}"][score_indices]
            ),
            "scored alignment",
        )
    require(
        np.array_equal(
            arrays["scored_probabilities"], arrays["observable_probabilities"][:, score_indices]
        ),
        "scored probability alignment",
    )
    mask = arrays["availability_mask"]
    require(mask.dtype == np.bool_ and mask.shape == (observable,), "availability schema")
    require(
        np.array_equal(arrays["available_indices"], np.flatnonzero(mask)), "available alignment"
    )
    require(
        np.array_equal(
            arrays["l9v_available_probabilities"], arrays["observable_probabilities"][0, mask]
        ),
        "available probability alignment",
    )
    fallback_count = int((~mask).sum())
    require(fallback_count == config["observable_fallback_rows"], "fallback count")
    require(
        int((~mask[score_indices]).sum()) == config["scored_fallback_rows"], "scored fallback count"
    )
    indices = arrays["fallback_source_method_index"]
    require(
        indices.dtype == np.int64
        and indices.shape == (fallback_count,)
        and bool((indices == 3).all()),
        "fallback numeric source",
    )
    require(arrays["method_ids"][3] == "b0", "fallback resolves to B0")
    require(
        arrays["observable_probabilities"][0, ~mask].tobytes()
        == arrays["observable_probabilities"][3, ~mask].tobytes(),
        "fallback probabilities must be byte-exact B0",
    )
    stored = arrays["fallback_source_method_id"]
    require(
        stored.shape == (fallback_count,)
        and str(stored.dtype) == "<U1"
        and set(stored.tolist()) == {"b"},
        "unexpected fallback string defect",
    )
    require(
        erratum["defect"]["stored_unique_values"] == ["b"]
        and erratum["gate_interpretation"]["complete_redundant_provenance_fields_exact"] is False,
        "mandatory erratum qualification",
    )
    reports = {}
    for index, name in enumerate(L9V_METHODS):
        report = metrics(
            arrays["scored_labels"],
            arrays["scored_probabilities"][index],
            arrays["scored_participant_ids"],
        )
        compare_report(report, analysis["reports"][name], ctgr=False)
        close(
            report["primary"]["mean_participant_macro_f1"],
            result["method_summary"][name]["mean_participant_macro_f1"],
            "L9v result binding",
        )
        reports[name] = report
    return {
        "lane": "l9v_corrected_fog",
        "evidence_status": result["evidence_status"],
        "source_code_commit": result["code_commit"],
        "reports": reports,
        "advancement": "fail_unchanged",
        "fallback_numeric_source_and_probability_exact": True,
        "fallback_string_provenance_complete": False,
        "mandatory_erratum": erratum,
        "observable_fallback_rows": fallback_count,
        "scored_fallback_rows": int((~mask[score_indices]).sum()),
        "completion_scope": "pinned_manifest_prediction_reports_all_models_and_source_input_manifests",
        "byte_and_source_provenance": provenance,
    }


def run(evidence_root: Path, output: Path, config_path: Path) -> dict[str, Any]:
    """Create an evidence package without loading raw data or any fitted model."""
    output.mkdir(parents=True, exist_ok=False)
    output = output.resolve(strict=True)
    started = time.perf_counter()
    inputs: Inputs | None = None
    failures: list[dict[str, str]] = []
    completed: list[str] = []
    active_lane = "preflight"
    interruption: BaseException | None = None

    def write(name: str, value: dict[str, Any]) -> None:
        atomic_write_json_new(sealed(value), output / name, allowed_root=output)

    try:
        write(
            "implementation_receipt.json",
            {
                "module": "src/inclusive_shift_har/experiments/portable_evidence_replay.py",
                "module_sha256": sha256_file(Path(__file__)),
                "canonical_utility_sha256": sha256_file(
                    Path(__file__).parents[1] / "manifests/canonical.py"
                ),
                "python_version": platform.python_version(),
                "numpy_version": np.__version__,
                "pyyaml_version": yaml.__version__,
                "model_fits": 0,
            },
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        require(config["experiment_id"] == "portable-evidence-replay-v1", "config experiment")
        require(
            config["model_fits"] == 0 and config["cross_lane_comparison"] is False,
            "zero-fit lane contract",
        )
        require(set(config["lanes"]) == {"ctgr", "l9v"}, "exact two lanes required")
        require(
            config["provenance"]
            == {
                "model_byte_hashes": True,
                "manifest_byte_hashes": True,
                "model_deserialization": False,
                "raw_bytes_rehashed": False,
            },
            "mandatory byte-hash-only provenance contract",
        )
        write(
            "config_snapshot.json",
            {"config": config, "source_file_sha256": sha256_file(config_path)},
        )
        inputs = Inputs(evidence_root)
        for lane, operation in (("ctgr", replay_ctgr), ("l9v", replay_l9v)):
            active_lane = lane
            try:
                result = operation(inputs, config["lanes"][lane])
                write(
                    f"{lane}_replay.json",
                    {
                        **result,
                        "status": "pass",
                        "model_fits": 0,
                        "raw_data_loaded": False,
                        "checkpoint_inference_replayed": False,
                        "independent_confirmation": False,
                    },
                )
                completed.append(lane)
            except Exception as exc:
                failure = {"lane": lane, "error_type": type(exc).__name__, "message": str(exc)}
                failures.append(failure)
                write(f"{lane}_failure.json", {**failure, "status": "fail", "model_fits": 0})
    except BaseException as exc:
        failure = {"lane": active_lane, "error_type": type(exc).__name__, "message": str(exc)}
        failures.append(failure)
        if not isinstance(exc, Exception):
            interruption = exc
            if (
                active_lane in {"ctgr", "l9v"}
                and not (output / f"{active_lane}_failure.json").exists()
            ):
                write(f"{active_lane}_failure.json", {**failure, "status": "fail", "model_fits": 0})
    finally:
        write(
            "input_manifest.json",
            {
                "inputs": [] if inputs is None else inputs.records,
                "model_fits": 0,
                "raw_data_copied": False,
            },
        )
        write(
            "shutdown_receipt.json",
            {
                "task_owned_workers_started": 0,
                "task_owned_workers_remaining": 0,
                "task_owned_monitors_started": 0,
                "task_owned_monitors_remaining": 0,
                "model_fits": 0,
                "execution": "synchronous_single_process",
            },
        )
        validation = {
            "status": "pass" if not failures else "fail",
            "completed_lanes": completed,
            "failures": failures,
            "model_fits": 0,
            "cross_lane_comparison": False,
            "elapsed_seconds": time.perf_counter() - started,
        }
        write("validation.json", validation)
        if failures:
            write("failure.json", validation)
        entries: list[dict[str, Any]] = [
            {
                "path": path.name,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "partial_file": ".partial." in path.name,
            }
            for path in sorted(output.iterdir())
            if path.is_file()
        ]
        for entry in entries:
            if not entry["partial_file"]:
                verify_record(load_json_strict(output / entry["path"]))
        write(
            "completion_manifest.json",
            {
                "status": validation["status"],
                "artifacts": entries,
                "self_excluded": True,
                "model_fits": 0,
            },
        )
    if interruption is not None:
        raise interruption
    return validation


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.evidence_root, args.output, args.config)
    return 0 if result["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
