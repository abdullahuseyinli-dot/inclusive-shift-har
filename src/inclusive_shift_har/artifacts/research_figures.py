"""Render research tables and figures from tracked aggregate evidence, without fitting models."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from inclusive_shift_har.manifests.canonical import (
    canonical_json_sha256,
    load_json_strict,
    sha256_file,
)

_SOURCE_METHODS = (
    ("base_uncalibrated", "rmrp", "RMRP"),
    ("frozen_ctgr", "ctgr", "CTGR"),
    ("hera_ctgr_strict", "hera_strict", "Strict HERA-v1"),
)


def _ordered_source(current: dict[str, Any]) -> list[dict[str, Any]]:
    rows = current["source_five_seed"]["results"]
    by_name = {row["method"]: row for row in rows}
    if len(rows) != len(by_name) or set(by_name) != {item[0] for item in _SOURCE_METHODS}:
        raise ValueError("unexpected or duplicated five-seed method identities")
    return [by_name[item[0]] for item in _SOURCE_METHODS]


def _record(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = load_json_strict(path)
    body = dict(payload)
    expected = body.pop("record_sha256", None)
    if canonical_json_sha256(body) != expected:
        raise ValueError(f"record hash mismatch: {path}")
    return payload


def verify_metrics(metrics: dict[str, Any], *, windows: int, participants: int) -> None:
    """Check aggregate arithmetic while keeping pooled and participant F1 distinct."""
    matrix = np.asarray(metrics["confusion_matrix"], dtype=np.float64)
    scores = np.asarray(metrics["participant_macro_f1_values_sorted"], dtype=np.float64)
    if (
        matrix.shape != (3, 3)
        or not np.isfinite(matrix).all()
        or np.any(matrix < 0)
        or not np.equal(matrix, np.floor(matrix)).all()
        or matrix.sum() != windows
    ):
        raise ValueError("invalid three-class confusion matrix or window denominator")
    if (
        scores.shape != (participants,)
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("invalid participant scores or participant denominator")
    expected = {
        "accuracy": float(np.trace(matrix) / windows),
        "mean_participant_macro_f1": float(scores.mean()),
    }
    for name, value in expected.items():
        if not math.isclose(float(metrics[name]), value, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"{name} does not reproduce from aggregate evidence")


def load_evidence(repository_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read and validate only the two public aggregate records."""
    evidence = repository_root / "results/research"
    lineage = _record(evidence / "source_development_lineage_v1.json")
    current = _record(evidence / "reported_metrics_audit_v1.json")
    source = current["source_five_seed"]
    if (source["participants"], source["windows_per_seed"], source["outer_folds"]) != (10, 725, 5):
        raise ValueError("unexpected source evaluation contract")
    expected_seeds = {11, 23, 47, 89, 131}
    if len(source["seeds"]) != 5 or set(source["seeds"]) != expected_seeds:
        raise ValueError("unexpected source seeds")
    for method in lineage["methods"]:
        verify_metrics(method["metrics"], windows=725, participants=10)
    for method in _ordered_source(current):
        rows = method["by_seed"]
        if len(rows) != 5 or {row["seed"] for row in rows} != expected_seeds:
            raise ValueError("missing, duplicated or unexpected method seeds")
        for row in rows:
            verify_metrics(row, windows=725, participants=10)
        for summary, column in (
            ("mean_window_accuracy_across_seeds", "accuracy"),
            ("mean_participant_macro_f1_across_seeds", "mean_participant_macro_f1"),
        ):
            observed = sum(float(row[column]) for row in rows) / len(rows)
            if not math.isclose(method[summary], observed, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"five-seed {summary} does not reproduce")
    matched = {method["method_id"]: method["metrics"] for method in lineage["methods"]}
    if len(matched) != len(lineage["methods"]):
        raise ValueError("duplicated lineage method identity")
    for method, (_, name, _) in zip(_ordered_source(current), _SOURCE_METHODS, strict=True):
        row = next(row for row in method["by_seed"] if row["seed"] == 11)
        for key in ("accuracy", "mean_participant_macro_f1"):
            if not math.isclose(matched[name][key], row[key], rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"seed-11 disagreement between evidence records: {name}")
    return lineage, current


def _tables(lineage: dict[str, Any], current: dict[str, Any]) -> str:
    rows = [
        "# Source-development tables",
        "",
        "Seed 11; 725 windows; ten source participants; five participant-exclusive outer folds.",
        "Fixed external controls and nested-selected project methods have different selection budgets.",
        "",
        "| Method | Channels | Window accuracy | Participant macro-F1 |",
        "|---|---:|---:|---:|",
    ]
    for method in lineage["methods"]:
        values = method["metrics"]
        rows.append(
            f"| {method['display_name']} | {method['input_channels']} | "
            f"{100 * values['accuracy']:.3f}% | "
            f"{100 * values['mean_participant_macro_f1']:.3f}% |"
        )
    rows += [
        "",
        "Five-seed means; same source participants and windows; seeds 11, 23, 47, 89, 131.",
        "CTGR/HERA add native gravity to the six-channel RMRP input.",
        "",
        "| Method | Mean window accuracy | Mean participant macro-F1 |",
        "|---|---:|---:|",
    ]
    for method, (_, _, name) in zip(_ordered_source(current), _SOURCE_METHODS, strict=True):
        rows.append(
            f"| {name} | {100 * method['mean_window_accuracy_across_seeds']:.3f}% | "
            f"{100 * method['mean_participant_macro_f1_across_seeds']:.3f}% |"
        )
    return "\n".join(rows) + "\n"


def render(repository_root: Path, output: Path) -> dict[str, Any]:
    """Create a new figure directory; never overwrite existing outputs or evidence."""
    lineage, current = load_evidence(repository_root)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "svg.hashsalt": "inclusive-shift-har-research-figures-v1",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    blue, grey, gold = "#176B87", "#718096", "#A85C16"

    def save(figure: Any, name: str) -> None:
        figure.savefig(output / f"{name}.svg", metadata={"Date": None}, facecolor="white")
        svg = output / f"{name}.svg"
        svg.write_text(
            "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())
            + "\n",
            encoding="utf-8",
        )
        figure.savefig(output / f"{name}.png", dpi=160, facecolor="white")
        plt.close(figure)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.4), gridspec_kw={"width_ratios": [1.25, 1]})
    fig.subplots_adjust(left=0.155, right=0.96, top=0.79, bottom=0.19, wspace=0.50)
    source = [method for method in lineage["methods"] if method["input_channels"] == 6]
    labels = {
        "multirocket_hydra": "MultiRocket + HYDRA",
        "rist_budgeted": "RIST (budgeted)",
        "rmrp": "RMRP / denoised GSP",
    }
    values = [100 * row["metrics"]["mean_participant_macro_f1"] for row in source]
    colors = [
        blue if row["method_id"] in {"spectral_shape", "gsp", "rmrp"} else grey for row in source
    ]
    axes[0].barh(range(len(source)), values, color=colors, height=0.66)
    axes[0].set_yticks(
        range(len(source)), [labels.get(row["method_id"], row["display_name"]) for row in source]
    )
    axes[0].invert_yaxis()
    axes[0].set_title(
        "A. Six-channel development\nSeed 11; identical outer evaluation", loc="left", pad=14
    )
    for i, value in enumerate(values):
        axes[0].text(value + 1, i, f"{value:.3f}", va="center", fontsize=9)
    means = [
        100 * row["mean_participant_macro_f1_across_seeds"] for row in _ordered_source(current)
    ]
    axes[1].barh(range(3), means, color=[blue, gold, gold], height=0.48)
    axes[1].set_yticks(range(3), ["RMRP (6 ch)", "CTGR (9 ch)", "Strict HERA (9 ch)"])
    axes[1].invert_yaxis()
    axes[1].set_ylim(2.9, -0.8)
    axes[1].set_title("B. Later matched extension\nFive-seed means", loc="left", pad=14)
    for i, value in enumerate(means):
        axes[1].text(value + 1, i, f"{value:.3f}", va="center", fontsize=9)
    for axis in axes:
        axis.set_xlim(0, 103)
        axis.set_xticks([0, 20, 40, 60, 80, 100])
        axis.set_xlabel("Mean participant macro-F1 (%)")
        axis.xaxis.grid(True, color="#E3E8ED", linewidth=0.7)
        axis.set_axisbelow(True)
    fig.suptitle(
        "Source development: representation, denoising and selective gravity",
        fontsize=16,
        x=0.06,
        ha="left",
        y=0.965,
    )
    fig.text(
        0.06,
        0.9,
        "InclusiveHAR P1-P10 · 725 windows · three classes · five participant-exclusive outer folds",
        color="#46566A",
    )
    fig.legend(
        handles=[
            Patch(color=grey, label="Executed external control"),
            Patch(color=blue, label="Project six-channel pipeline"),
            Patch(color=gold, label="Project native-gravity extension"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9,
    )
    fig.text(
        0.06,
        0.037,
        "A: control selection budgets differ. B: gravity adds information; paired gain intervals include zero. Reused-source development, not independent confirmation.",
        fontsize=8.5,
    )
    save(fig, "source_development_comparison")

    fig, axis = plt.subplots(figsize=(13, 6.6))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.05)
    axis.set(xlim=(0, 13), ylim=(0, 6.4))
    axis.axis("off")

    def box(x: float, y: float, width: float, title: str, body: str, color: str) -> None:
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                1.25,
                boxstyle="round,pad=0.10",
                facecolor="#F5F8FA",
                edgecolor=color,
                linewidth=1.8,
            )
        )
        axis.text(x + 0.15, y + 0.92, title, weight="bold", color=color, fontsize=12)
        axis.text(x + 0.15, y + 0.65, body, va="top", fontsize=9.5, linespacing=1.5)

    axis.text(0.2, 6.02, "What the project developed", fontsize=19, weight="bold")
    axis.text(
        0.2,
        5.63,
        "Arrows show method development; separate studies do not form one numerical improvement curve.",
        fontsize=10,
        color="#46566A",
    )
    box(
        0.2,
        3.6,
        3.6,
        "SpectralShape → GSP",
        "Geometric and spectral features\nStandard learner; inner-fold selection\nSix-channel source development",
        blue,
    )
    box(
        4.55,
        3.6,
        3.6,
        "RMRP → denoised GSP",
        "Raw, denoised and residual candidates\nDenoised view selected in every fold\nNoise gains; temporal-gap harms",
        blue,
    )
    box(
        8.85,
        3.6,
        3.6,
        "CTGR → strict HERA",
        "Add native gravity for posture\nCorrect uncertain predictions selectively\nThen ensemble and calibrate",
        gold,
    )
    for start, end in ((3.92, 4.42), (8.27, 8.72)):
        axis.add_patch(
            FancyArrowPatch(
                (start, 4.22),
                (end, 4.22),
                arrowstyle="-|>",
                mutation_scale=16,
                linewidth=1.5,
                color="#46566A",
            )
        )
    axis.text(0.2, 2.85, "Separate evidence streams", fontsize=13, weight="bold")
    box(
        0.2,
        1.15,
        3.6,
        "Locked target benchmark",
        "Different target participants and protocol\nMoRe-HAR hypothesis unsupported\nPreserved; not reopened for successors",
        grey,
    )
    box(
        4.55,
        1.15,
        3.6,
        "Labelled personalization",
        "SAR / ASGS / semantic-gauge studies\nQuery labels counted and excluded\nSeparate from zero-query recognition",
        grey,
    )
    box(
        8.85,
        1.15,
        3.6,
        "External diagnostics",
        "HARTH: placement and reference methods\nAICOS: frozen sensor-interface transfer\nExplicit limitations and negative results",
        grey,
    )
    axis.text(
        0.2,
        0.45,
        "Project contributions are representations and composite pipelines. Extra Trees, Random Forest and filtering primitives retain their established attribution.",
        fontsize=9,
    )
    save(fig, "method_development_map")
    (output / "source_development_tables.md").write_text(
        _tables(lineage, current), encoding="utf-8"
    )
    receipt: dict[str, Any] = {
        "record_kind": "research_figures_reproduction",
        "schema_version": "1.0.0",
        "new_model_fits": 0,
        "raw_datasets_loaded": False,
        "prediction_archives_loaded": False,
        "generator_sha256": sha256_file(Path(__file__)),
        "libraries": {"matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "source_record_sha256": [lineage["record_sha256"], current["record_sha256"]],
        "files": {path.name: sha256_file(path) for path in sorted(output.iterdir())},
    }
    receipt["record_sha256"] = canonical_json_sha256(receipt)
    (output / "reproduction_receipt.json").write_text(
        json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
    )
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        load_evidence(args.repository_root)
        print(
            "PASS: aggregate hashes, confusion matrices, participant means and matched seed-11 records"
        )
    elif args.output is None:
        parser.error("--output is required unless --verify-only is used")
    else:
        print(json.dumps(render(args.repository_root, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
