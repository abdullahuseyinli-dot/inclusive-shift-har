from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from inclusive_shift_har.evaluation.inference_contracts import OBSERVABLE_CONTEXT_PROTOCOL
from inclusive_shift_har.experiments import observable_context_parity as parity


def _pair(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    monkeypatch.setattr(
        parity, "validate_run_directory", lambda *_: {"publication_evidence_ready": True}
    )
    monkeypatch.setattr(parity, "_source_input_manifest", lambda *_: {})
    monkeypatch.setattr(parity, "_git_state", lambda *_: {})
    for name in ("old", "new"):
        directory = tmp_path / name
        directory.mkdir()
        result: dict[str, Any] = {
            "dataset": {"dataset_id": "fog_star_v3"},
            "seeds": [11, 23, 47],
            "primary_seed_averaged": {
                "methods": {
                    method: {"mean_participant_macro_f1": 1.0}
                    for method in ("RandomForest-6ch", "HERA-DG-full")
                }
            },
        }
        if name == "new":
            result["observable_context_protocol"] = OBSERVABLE_CONTEXT_PROTOCOL
        (directory / "result.json").write_text(json.dumps(result), encoding="utf-8")
        (directory / "data_audit.json").write_text("{}", encoding="utf-8")
        np.savez_compressed(
            directory / "predictions.npz",
            labels=np.arange(3),
            participant_ids=np.array(["p1"] * 3),
            session_ids=np.array(["s"] * 3),
            trial_ids=np.array(["t"] * 3),
            window_ids=np.array(["w0", "w1", "w2"]),
            **{  # type: ignore[arg-type]
                f"probability__seed-{seed}__{method}": np.eye(3)
                for seed in (11, 23, 47)
                for method in ("RandomForest-6ch", "HERA-DG-full")
            },
        )
    return tmp_path / "old", tmp_path / "new"


@pytest.mark.parametrize("mutation", ["none", "base", "context", "identities"])
def test_parity_gates_unaffected_methods_and_preserves_context_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    old, new = _pair(tmp_path, monkeypatch)
    path = new / "predictions.npz"
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    if mutation in {"base", "context"}:
        method = "RandomForest-6ch" if mutation == "base" else "HERA-DG-full"
        arrays[f"probability__seed-11__{method}"] = np.roll(np.eye(3), 1, axis=1)
    elif mutation == "identities":
        arrays["window_ids"] = np.roll(arrays["window_ids"], 1)
    np.savez_compressed(path, **arrays)
    if mutation == "identities":
        with pytest.raises(ValueError, match="identities changed"):
            parity.compare_context_replacement(old, new, tmp_path)
    else:
        result = parity.compare_context_replacement(old, new, tmp_path)
        assert (result["status"] == "PARITY_PASSED") is (mutation != "base")
        assert result["methods"]["HERA-DG-full"]["parity_passed"] is None
        assert not result["better_score_selection_allowed"]
