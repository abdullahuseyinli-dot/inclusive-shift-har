"""Frozen-encoder helpers for the corrected-FoG optional-context experiment.

The signal adapter in this module is deliberately annotation blind.  It reconstructs
native finite sensor runs and only then projects the already frozen observable queries.
Labels are accepted by a separate oracle audit solely to prove that a prospective gate
is reachable; they never affect signal support, features, or fitted predictions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, TypeAlias, cast

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from torch import Tensor, nn

from inclusive_shift_har.data.external_har import _contiguous_signal_runs
from inclusive_shift_har.experiments.fog_motion_factorization import _window_coordinates
from inclusive_shift_har.experiments.fog_rf_feature_weight_factorial import (
    _array_sha256,
    _require,
)

FloatArray: TypeAlias = NDArray[np.float64]
Float32Array: TypeAlias = NDArray[np.float32]
IntArray: TypeAlias = NDArray[np.int64]
BoolArray: TypeAlias = NDArray[np.bool_]
StringArray: TypeAlias = NDArray[np.str_]
SessionKey: TypeAlias = tuple[str, str]

SOURCE_RATE_HZ = 60.0
HARNET_RATE_HZ = 30.0
HARNET_SECONDS = 10.0
HARNET_SAMPLES = 300
HARNET_CHANNELS = 3
HARNET_EMBEDDING_DIMENSION = 1_024
HARNET_PARAMETER_COUNT = 10_457_408
CLIP_G = 3.0
MAXIMUM_GAP_FACTOR = 3.0


@dataclass(frozen=True, slots=True)
class NativeSensorRun:
    """One finite monotonic native-rate run; no annotations are represented."""

    timestamps: FloatArray
    total_acceleration_g: FloatArray


@dataclass(frozen=True, slots=True)
class NativeHarnetContexts:
    """Right-aligned HARNET histories in observable-candidate order."""

    histories: Float32Array
    target_timestamps: FloatArray
    full_context_mask: BoolArray
    observable_window_ids: StringArray
    receipts: tuple[dict[str, Any], ...]
    parsed_columns: tuple[str, ...]

    def validate(self, expected_full_context_mask: BoolArray) -> None:
        count = self.observable_window_ids.size
        _require(
            self.histories.shape == (count, HARNET_CHANNELS, HARNET_SAMPLES),
            "native HARNET history shape changed",
        )
        _require(
            self.target_timestamps.shape == (count, HARNET_SAMPLES),
            "native HARNET timestamp shape changed",
        )
        _require(
            self.full_context_mask.shape == (count,)
            and np.array_equal(self.full_context_mask, expected_full_context_mask),
            "native HARNET support differs from the frozen physical-history mask",
        )
        _require(
            len(self.receipts) == count
            and bool(np.isfinite(self.histories).all())
            and bool(np.isfinite(self.target_timestamps).all()),
            "native HARNET context receipt or finite-value contract changed",
        )
        _require(
            bool(np.all(self.histories[~self.full_context_mask] == 0.0))
            and bool(np.all(self.target_timestamps[~self.full_context_mask] == 0.0)),
            "unsupported HARNET histories must remain exact zero",
        )
        if bool(self.full_context_mask.any()):
            supported = self.histories[self.full_context_mask]
            _require(
                bool(np.all((supported >= -CLIP_G) & (supported <= CLIP_G))),
                "HARNET total acceleration was not clipped to the official range",
            )
            times = self.target_timestamps[self.full_context_mask]
            expected_step = 1.0 / HARNET_RATE_HZ
            _require(
                bool(np.allclose(np.diff(times, axis=1), expected_step, rtol=0.0, atol=2e-10)),
                "HARNET timestamps are not a linear 30-Hz grid",
            )


@dataclass(frozen=True, slots=True)
class HarnetExtractor:
    """Qualified frozen feature extractor and its provenance."""

    model: nn.Module
    state_sha256: str
    parameter_count: int
    state_key_count: int
    checkpoint_feature_key_count: int | None


def _sensor_columns(sensor: str) -> list[str]:
    return [f"{sensor}_{modality}_{axis}" for modality in ("acc", "gyro") for axis in "xyz"]


def _session_identity(subject: Any, session: Any) -> SessionKey:
    participant = f"fogstar:{int(subject):03d}"
    return participant, f"{participant}:session-{int(session):03d}"


def _materialize_native_runs(
    timestamps: FloatArray,
    sensor_values: FloatArray,
) -> dict[int, NativeSensorRun]:
    total_g = np.asarray(sensor_values[:, :3], dtype=np.float64)
    gyro = np.asarray(sensor_values[:, 3:], dtype=np.float64)
    boundaries = _contiguous_signal_runs(
        np.asarray(timestamps, dtype=np.float64),
        total_g,
        gyro,
        nominal_rate_hz=SOURCE_RATE_HZ,
        maximum_gap_factor=MAXIMUM_GAP_FACTOR,
    )
    return {
        index: NativeSensorRun(
            timestamps=np.asarray(timestamps[left:right], dtype=np.float64),
            total_acceleration_g=np.asarray(total_g[left:right], dtype=np.float64),
        )
        for index, (left, right) in enumerate(boundaries)
        if right - left >= 3
    }


def _native_history_grid(end_time_exclusive: float) -> FloatArray:
    _require(np.isfinite(end_time_exclusive), "query end time is nonfinite")
    result = (
        float(end_time_exclusive)
        - HARNET_SECONDS
        + np.arange(HARNET_SAMPLES, dtype=np.float64) / HARNET_RATE_HZ
    )
    _require(
        result.shape == (HARNET_SAMPLES,) and float(result[-1]) < float(end_time_exclusive),
        "HARNET right-aligned grid construction failed",
    )
    return result


def _run_supports_grid(run: NativeSensorRun, grid: FloatArray) -> bool:
    tolerance = 1e-10
    return bool(
        grid[0] >= run.timestamps[0] - tolerance and grid[-1] <= run.timestamps[-1] + tolerance
    )


def _interpolate_history(run: NativeSensorRun, grid: FloatArray) -> FloatArray:
    _require(_run_supports_grid(run, grid), "native history interpolation would extrapolate")
    result = np.column_stack(
        [np.interp(grid, run.timestamps, run.total_acceleration_g[:, axis]) for axis in range(3)]
    )
    _require(
        result.shape == (HARNET_SAMPLES, HARNET_CHANNELS) and bool(np.isfinite(result).all()),
        "native HARNET interpolation failed",
    )
    return np.asarray(result, dtype=np.float64)


def reconstruct_native_harnet_contexts(
    *,
    raw_path: Path,
    coverage_rows: Sequence[Mapping[str, Any]],
    current_availability_mask: BoolArray,
    expected_full_context_mask: BoolArray,
) -> NativeHarnetContexts:
    """Build source-faithful HARNET10 input without reading any annotation column."""

    count = len(coverage_rows)
    available = np.asarray(current_availability_mask, dtype=np.bool_)
    expected_full = np.asarray(expected_full_context_mask, dtype=np.bool_)
    _require(
        available.shape == expected_full.shape == (count,),
        "native adapter support arrays are misaligned",
    )
    columns = ["timestamp", "subjectID", "sessionID"]
    for sensor in ("back", "ankleL"):
        columns.extend(_sensor_columns(sensor))
    frame = pd.read_csv(raw_path, usecols=columns)
    _require(set(frame.columns) == set(columns), "native adapter parsed an unexpected column set")
    back: dict[SessionKey, dict[int, NativeSensorRun]] = {}
    ankle: dict[SessionKey, dict[int, NativeSensorRun]] = {}
    for keys, group in frame.groupby(["subjectID", "sessionID"], sort=True, dropna=False):
        subject, session = cast(tuple[Any, Any], keys)
        key = _session_identity(subject, session)
        timestamps = group["timestamp"].to_numpy(dtype=np.float64)
        back[key] = _materialize_native_runs(
            timestamps, group[_sensor_columns("back")].to_numpy(dtype=np.float64)
        )
        ankle[key] = _materialize_native_runs(
            timestamps, group[_sensor_columns("ankleL")].to_numpy(dtype=np.float64)
        )

    histories = np.zeros((count, HARNET_CHANNELS, HARNET_SAMPLES), dtype=np.float32)
    targets = np.zeros((count, HARNET_SAMPLES), dtype=np.float64)
    supported = np.zeros(count, dtype=np.bool_)
    identifiers = np.asarray([str(row["window_id"]) for row in coverage_rows], dtype=np.str_)
    receipts: list[dict[str, Any]] = []
    for index, row in enumerate(coverage_rows):
        window_id = str(row["window_id"])
        participant = str(row["participant_id"])
        session = str(row["session_id"])
        receipt: dict[str, Any] = {
            "observable_index": index,
            "window_id": window_id,
            "participant_id": participant,
            "session_id": session,
            "current_ankle_available": bool(available[index]),
            "annotation_columns_accessed": [],
        }
        if not available[index]:
            receipt.update({"status": "current_ankle_unavailable", "full_context": False})
            receipts.append(receipt)
            continue
        matches = tuple(
            int(value) for value in cast(Sequence[Any], row["ankle_matching_grid_run_indices"])
        )
        _require(len(matches) == 1, f"qualified ankle run changed: {window_id}")
        back_run_index, _window_index = _window_coordinates(window_id)
        key = (participant, session)
        _require(
            key in back
            and key in ankle
            and back_run_index in back[key]
            and matches[0] in ankle[key],
            f"native physical run is missing: {window_id}",
        )
        back_run = back[key][back_run_index]
        ankle_run = ankle[key][matches[0]]
        end_time = float(row["end_time_exclusive"])
        grid = _native_history_grid(end_time)
        physical_support = _run_supports_grid(back_run, grid) and _run_supports_grid(
            ankle_run, grid
        )
        receipt.update(
            {
                "back_run_index": back_run_index,
                "ankle_run_index": matches[0],
                "query_end_time_exclusive": end_time,
                "target_start_time": float(grid[0]),
                "target_last_time": float(grid[-1]),
                "back_run_start_time": float(back_run.timestamps[0]),
                "back_run_last_time": float(back_run.timestamps[-1]),
                "ankle_run_start_time": float(ankle_run.timestamps[0]),
                "ankle_run_last_time": float(ankle_run.timestamps[-1]),
                "full_context": physical_support,
            }
        )
        if not physical_support:
            receipt["status"] = "insufficient_same_run_history"
            receipts.append(receipt)
            continue
        interpolated = _interpolate_history(ankle_run, grid)
        clipped = np.clip(interpolated, -CLIP_G, CLIP_G)
        histories[index] = clipped.T.astype(np.float32, copy=False)
        targets[index] = grid
        supported[index] = True
        receipt.update(
            {
                "status": "complete_native_harnet_history",
                "unclipped_value_count": int(interpolated.size),
                "clipped_low_count": int(np.sum(interpolated < -CLIP_G)),
                "clipped_high_count": int(np.sum(interpolated > CLIP_G)),
                "timestamp_sha256": _array_sha256(grid),
                "history_sha256": _array_sha256(histories[index]),
            }
        )
        receipts.append(receipt)
    result = NativeHarnetContexts(
        histories=histories,
        target_timestamps=targets,
        full_context_mask=supported,
        observable_window_ids=identifiers,
        receipts=tuple(receipts),
        parsed_columns=tuple(columns),
    )
    result.validate(expected_full)
    return result


def tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str((value.shape, value.dtype.str)).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _load_harnet_module(source_root: Path) -> ModuleType:
    source = source_root / "sslearning/models/accNet.py"
    _require(source.is_file(), "pinned HARNET implementation is missing")
    module_name = f"_qualified_harnet_{hashlib.sha256(str(source).encode()).hexdigest()[:12]}"
    specification = importlib.util.spec_from_file_location(module_name, source)
    if specification is None or specification.loader is None:
        raise ValueError("cannot construct HARNET module specification")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def instantiate_harnet_extractor(
    *,
    source_root: Path,
    seed: int,
    checkpoint_path: Path | None,
) -> HarnetExtractor:
    """Instantiate a random or strictly loaded frozen HARNET10 feature extractor."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    module = _load_harnet_module(source_root)
    constructor = cast(Any, module.Resnet)
    complete = cast(
        nn.Module,
        constructor(output_size=2, is_eva=True, resnet_version=1, epoch_len=10),
    )
    extractor = cast(nn.Module, cast(Any, complete).feature_extractor)
    checkpoint_count: int | None = None
    if checkpoint_path is not None:
        loaded = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _require(isinstance(loaded, Mapping), "HARNET checkpoint is not a tensor mapping")
        prefix = "module.feature_extractor."
        feature_state = {
            str(name)[len(prefix) :]: value
            for name, value in loaded.items()
            if str(name).startswith(prefix)
        }
        expected = extractor.state_dict()
        _require(
            set(feature_state) == set(expected),
            "HARNET checkpoint feature keys are incomplete or unexpected",
        )
        for name, tensor in feature_state.items():
            _require(
                isinstance(tensor, Tensor)
                and tuple(tensor.shape) == tuple(expected[name].shape)
                and bool(torch.isfinite(tensor).all()),
                f"HARNET checkpoint tensor changed: {name}",
            )
        extractor.load_state_dict(feature_state, strict=True)
        checkpoint_count = len(feature_state)
    extractor.eval()
    for parameter in extractor.parameters():
        parameter.requires_grad_(False)
    parameters = sum(parameter.numel() for parameter in extractor.parameters())
    _require(parameters == HARNET_PARAMETER_COUNT, "HARNET feature parameter count changed")
    state = extractor.state_dict()
    _require(len(state) == 131, "HARNET feature state-key count changed")
    return HarnetExtractor(
        model=extractor,
        state_sha256=tensor_state_sha256(state),
        parameter_count=parameters,
        state_key_count=len(state),
        checkpoint_feature_key_count=checkpoint_count,
    )


def extract_harnet_embeddings(
    *,
    extractor: HarnetExtractor,
    contexts: NativeHarnetContexts,
    device: torch.device,
    batch_size: int = 32,
    repeat_tolerance: float = 0.0,
    batch_single_tolerance: float = 1e-5,
) -> tuple[Float32Array, dict[str, Any]]:
    """Extract only genuine histories and leave unavailable rows as exact zero."""

    _require(batch_size > 0, "HARNET extraction batch size must be positive")
    _require(
        np.isfinite(repeat_tolerance)
        and np.isfinite(batch_single_tolerance)
        and repeat_tolerance >= 0.0
        and batch_single_tolerance >= 0.0,
        "HARNET numeric tolerances are invalid",
    )
    indices = np.flatnonzero(contexts.full_context_mask).astype(np.int64, copy=False)
    _require(indices.size > 0, "no genuine HARNET history is available")
    model = extractor.model.to(device).eval()
    before = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = np.zeros(
        (contexts.observable_window_ids.size, HARNET_EMBEDDING_DIMENSION), dtype=np.float32
    )
    with torch.inference_mode():
        first_indices = indices[: min(batch_size, indices.size)]
        first_tensor = torch.from_numpy(contexts.histories[first_indices]).to(device=device)
        first_a = model(first_tensor).reshape(first_tensor.shape[0], -1)
        first_b = model(first_tensor).reshape(first_tensor.shape[0], -1)
        repeat_difference = float(torch.max(torch.abs(first_a - first_b)).item())
        single = model(first_tensor[:1]).reshape(1, -1)
        batch_single_difference = float(torch.max(torch.abs(first_a[:1] - single)).item())
        _require(
            first_a.shape[1] == HARNET_EMBEDDING_DIMENSION
            and repeat_difference <= repeat_tolerance
            and batch_single_difference <= batch_single_tolerance,
            "HARNET deterministic or batch-equivalence smoke test failed",
        )
        for left in range(0, indices.size, batch_size):
            selected = indices[left : left + batch_size]
            tensor = torch.from_numpy(contexts.histories[selected]).to(device=device)
            embedding = model(tensor).reshape(tensor.shape[0], -1)
            _require(
                embedding.shape[1] == HARNET_EMBEDDING_DIMENSION
                and bool(torch.isfinite(embedding).all()),
                "HARNET embedding shape or finite-value check failed",
            )
            output[selected] = embedding.detach().cpu().numpy().astype(np.float32, copy=False)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
        reserved = int(torch.cuda.max_memory_reserved(device))
    else:
        peak = 0
        reserved = 0
    elapsed = time.perf_counter() - started
    after = model.state_dict()
    _require(
        all(torch.equal(before[name], after[name].detach().cpu()) for name in before),
        "HARNET parameters or BatchNorm state changed during frozen extraction",
    )
    _require(
        bool(np.isfinite(output).all())
        and bool(np.all(output[~contexts.full_context_mask] == 0.0)),
        "HARNET embedding fallback representation changed",
    )
    model.cpu()
    return output, {
        "genuine_history_rows": int(indices.size),
        "batch_size": batch_size,
        "embedding_dimension": HARNET_EMBEDDING_DIMENSION,
        "repeat_max_absolute_tolerance": repeat_tolerance,
        "batch_single_max_absolute_tolerance": batch_single_tolerance,
        "repeat_max_absolute_difference": repeat_difference,
        "batch_vs_single_max_absolute_difference": batch_single_difference,
        "extract_seconds": elapsed,
        "peak_cuda_allocated_bytes": peak,
        "peak_cuda_reserved_bytes": reserved,
        "embedding_sha256": _array_sha256(output),
        "encoder_state_unchanged": True,
        "unsupported_embeddings_exact_zero": True,
    }


def fit_standardizer(values: FloatArray) -> tuple[FloatArray, FloatArray]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 2 and array.shape[0] > 0, "standardizer input is empty")
    mean = np.asarray(array.mean(axis=0), dtype=np.float64)
    scale = np.asarray(array.std(axis=0, ddof=0), dtype=np.float64)
    scale[scale == 0.0] = 1.0
    _require(bool(np.isfinite(mean).all()) and bool(np.isfinite(scale).all()), "bad standardizer")
    return mean, scale


def standardized_optional_embedding(
    embeddings: Float32Array,
    full_context_mask: BoolArray,
    training_mask: BoolArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    values = np.asarray(embeddings, dtype=np.float64)
    available = np.asarray(full_context_mask, dtype=np.bool_)
    training = np.asarray(training_mask, dtype=np.bool_)
    _require(
        values.ndim == 2
        and available.shape == training.shape == (values.shape[0],)
        and bool((available & training).any()),
        "optional embedding arrays are misaligned",
    )
    mean, scale = fit_standardizer(values[available & training])
    result = np.zeros_like(values)
    result[available] = (values[available] - mean) / scale
    _require(
        bool(np.isfinite(result).all()) and bool(np.all(result[~available] == 0.0)),
        "optional embedding normalization failed",
    )
    return result, mean, scale


def _confusion(labels: IntArray, predictions: IntArray) -> IntArray:
    counts = np.zeros((3, 3), dtype=np.int64)
    np.add.at(counts, (labels, predictions), 1)
    return counts


def _macro_f1(counts: IntArray) -> float:
    denominator = counts.sum(axis=0) + counts.sum(axis=1)
    return float(
        np.divide(
            2 * counts.diagonal(),
            denominator,
            out=np.zeros(3, dtype=np.float64),
            where=denominator > 0,
        ).mean()
    )


def prospective_motion_headroom(
    *,
    labels: IntArray,
    participant_ids: StringArray,
    l9v_probabilities: FloatArray,
    editable_mask: BoolArray,
    roster: Sequence[str],
) -> dict[str, Any]:
    """Exact label-oracle gate-feasibility audit; output is never a fitted feature."""

    y = np.asarray(labels, dtype=np.int64)
    people = np.asarray(participant_ids, dtype=np.str_)
    probability = np.asarray(l9v_probabilities, dtype=np.float64)
    editable = np.asarray(editable_mask, dtype=np.bool_)
    _require(
        probability.shape == (y.size, 3) and people.shape == editable.shape == y.shape,
        "headroom arrays are misaligned",
    )
    baseline = probability.argmax(axis=1).astype(np.int64)
    posture = 1 + probability[:, 1:].argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for participant in roster:
        selected = people == participant
        local_y = y[selected]
        local_base = baseline[selected]
        local_posture = posture[selected]
        local_edit = editable[selected]
        fixed = _confusion(local_y[~local_edit], local_base[~local_edit])
        reachable = local_edit & ((local_y == 0) | (local_posture == local_y))
        fixed += _confusion(local_y[reachable], local_y[reachable])
        wrong_sit = int(np.sum(local_edit & (local_y == 1) & (local_posture == 2)))
        wrong_stand = int(np.sum(local_edit & (local_y == 2) & (local_posture == 1)))
        best = -1.0
        for sit_motion in range(wrong_sit + 1):
            for stand_motion in range(wrong_stand + 1):
                counts = fixed.copy()
                counts[1, 0] += sit_motion
                counts[1, 2] += wrong_sit - sit_motion
                counts[2, 0] += stand_motion
                counts[2, 1] += wrong_stand - stand_motion
                best = max(best, _macro_f1(counts))
        base_score = _macro_f1(_confusion(local_y, local_base))
        rows.append(
            {
                "participant_id": participant,
                "baseline_f1": base_score,
                "motion_only_maximum_f1": best,
                "strict_improvement_possible": best > base_score + 1e-12,
            }
        )
    return {
        "diagnostic_label_oracle_only": True,
        "not_model_performance": True,
        "editable_rows": int(editable.sum()),
        "maximum_possible_strict_participant_wins": sum(
            bool(row["strict_improvement_possible"]) for row in rows
        ),
        "mean_motion_only_ceiling": float(
            np.mean([float(row["motion_only_maximum_f1"]) for row in rows])
        ),
        "rows": rows,
    }
