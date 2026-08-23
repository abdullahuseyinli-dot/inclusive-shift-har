# InclusiveShift-HAR

**Provisional title:** *InclusiveShift-HAR: A Leakage-Safe Benchmark for Physical-Ability Generalization in Smartphone Activity Recognition*

InclusiveShift-HAR is a research-software project for studying how reliably smartphone inertial human activity recognition (HAR) systems trained on conventional participant populations generalize to people whose activities may be physically realized differently, including some users of assistive devices.

The proposed model name, **MoRe-HAR (Motion-Realization Factorized Human Activity Recognition)**, is provisional and secondary to the benchmark. This repository currently contains foundation and legacy-audit evidence, not a validated benchmark result or model claim.

## Evidence classes

Results must be presented under exactly one of these evidence classes. They are not interchangeable.

| Evidence class | Meaning | Current status |
|---|---|---|
| **Legacy coursework** | Saved UCI-HAR coursework outputs with random window-level validation and repeated official-test evaluation. | Audited; permanently **exploratory/development-consumed**. |
| **Corrected reproduction / development** | Subject-grouped reproduction, baseline development, ablations, debugging, and tuning conducted under explicit development protocols. | Not yet reported. These results will not be called confirmatory. |
| **Locked confirmatory** | A preregistered configuration evaluated once on a sealed target cohort only after manifests, splits, code, and artifacts pass all gates. | **No locked confirmatory result exists.** |

The legacy audit is in [`docs/LEGACY_AUDIT.md`](docs/LEGACY_AUDIT.md). The current InclusiveHAR audit, data card, and hardware record are in [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md), [`docs/data/INCLUSIVEHAR_V4.md`](docs/data/INCLUSIVEHAR_V4.md), and [`docs/HARDWARE.md`](docs/HARDWARE.md). Machine-readable legacy evidence is under [`legacy/`](legacy/README.md).

## Research question

> How reliably do smartphone HAR systems trained on conventional participant populations generalize to people whose activities are physically realized differently, including users of assistive devices?

This is an ability-associated population-shift question. Disability labels and assistive-device status are not intended as inference inputs, and observational group differences must not be described as causal disability effects.

## Claim limits

This repository does **not** currently claim:

- publishability or state-of-the-art performance;
- fairness, clinical validity, medical utility, or safety certification;
- the first disability-related or fairness-oriented HAR study;
- the first disentangled HAR architecture;
- that a generic architecture swap is a scientific contribution;
- that legacy UCI-HAR test metrics are fresh confirmatory evidence;
- that wheelchair propulsion is ordinary gait, ramps are stairs, or semantically different activities are interchangeable.

Negative results, failed runs, out-of-memory events, quarantines, manual interventions, protocol deviations, and conditional passes are evidence and must remain visible.

## Current repository contents

```text
README.md
LICENSE
CITATION.cff
.zenodo.json
pyproject.toml
uv.lock
src/inclusive_shift_har/
    __init__.py
    cli.py
    artifacts/
    data/
    manifests/
    py.typed
manifests/datasets/
results/
    data_audit/
    gates/
docs/
    DATA_AUDIT.md
    HARDWARE.md
    LEGACY_AUDIT.md
    LITERATURE_MATRIX.md
    PROJECT_STATUS.md
    data/INCLUSIVEHAR_V4.md
legacy/
    archive_manifest.json
    verification_results.json
    legacy_metrics.locked.json
    metric_reconstruction.json
```

The repository now includes immutable dataset manifests, safe acquisition/audit primitives, artifact validation, and a privacy-preserving InclusiveHAR v4 audit. Split construction, model training, evaluation, and calibration are not implemented or authorized; their visible placeholders remain fail-closed.

## Environment

The foundation targets CPython 3.11 and uses [`uv`](https://docs.astral.sh/uv/) for deterministic dependency resolution.

```powershell
uv sync
uv run inclusive-shift-har --version
uv run pytest
uv run ruff check .
uv run mypy src
```

The console entry point exposes validated provenance and audit commands:

- `validate-manifests`
- `audit-data` (no-access dry run by default; raw reads require a matching gate record)
- `validate-artifacts`

The full privacy-safe InclusiveHAR audit is explicit and gated:

```powershell
uv run inclusive-shift-har audit-data --manifest manifests/history/inclusivehar_v4.e3fc4e0f21cdd580b930a1a694923a43e222957335fde8eb08960e007f4930bb.json --profile inclusivehar-v4 --read-only --data-root data/raw --gate-record results/gates/raw_data_read_access_inclusivehar_v4.json --json
```

`build-splits`, `audit-splits`, `train`, and `evaluate` are visible fail-closed gates. They exit non-zero with machine-readable `gated_not_implemented` status until their implementations and prerequisite evidence exist.

`uv.lock` records the software environment only. Dataset versions and immutable source hashes belong in separate data manifests because a Python dependency lock cannot establish data provenance.

## Data and evidence policy

- Download datasets only from official, versioned sources after recording URL/DOI, version, retrieval date, size, SHA-256, licence, citation, and expected schema.
- Keep raw data immutable and outside normal Git history.
- Partition participants before windowing; never window across participants, trials, discontinuities, or activity boundaries.
- Fit normalization on training data only and tune/calibrate on validation only.
- Do not derive class mappings or output dimensions from confirmatory test labels.
- Preserve manifests, configurations, predictions, failures, checkpoints, and protocol-opening records.
- Never commit secrets, the legacy source ZIP, third-party raw data, or large checkpoints.

The local `.audit/` directory contains preserved Stage 0 extraction evidence. It is ignored by Git but is not a cleanup target.

## Reproducibility status

The Stage 0 archive-integrity and legacy-artifact gate passes. It validates the checksum, complete archive inventory, and internal consistency of saved predictions—not the scientific validity of the old protocol.

The two official InclusiveHAR v4 artifacts have been acquired into immutable, Git-ignored local raw storage and verified against the versioned provider sizes and SHA-256 values. Stage 3 passes artifact, schema, numeric-integrity, participant, group, and label-coverage checks, but remains **protocol quarantined**: the release has no timestamp, trial, session, recording, or raw-sample identifier, so trial-safe windowing cannot be demonstrated. Stage 4 and all split/training/evaluation work are blocked pending authoritative boundaries or an explicit locked protocol revision.

No corrected reproduction, development model result, or locked confirmatory result is reported.

## Licensing

Repository-authored code is licensed under the [Apache License 2.0](LICENSE). That licence does **not** relicense third-party datasets, coursework artifacts, notebooks, checkpoints, papers, figures, or other external materials. Each external asset retains its own terms and must be cited and distributed according to its source licence. Dataset access instructions and manifests do not grant redistribution rights.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json). These files prepare future release metadata; they do not indicate that a DOI has been minted or that a public stable release exists.
