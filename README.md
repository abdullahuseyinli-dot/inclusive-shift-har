# InclusiveShift-HAR

Auditable research software for participant-exclusive inertial human activity
recognition under population and sensor-interface shift.

[![CI](https://github.com/abdullahuseyinli-dot/inclusive-shift-har/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/abdullahuseyinli-dot/inclusive-shift-har/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

## Research overview

The project develops selective gravity-based correction of posture errors and
tests population transfer and generalization across sensor interfaces. The
current method combines a denoised geometric/spectral base with a
confidence-triggered sitting/standing expert.

**CTGR improved source participant macro-F1 from 83.953% to 86.540%** across five
fixed seeds, passing the prespecified development advancement checks. Strict
HERA-v1 reached the highest matched mean, **86.849% macro-F1 and 87.228%
accuracy**. A separate HARTH placement study reached **97.261% participant
macro-F1 and 99.723% accuracy** for binary posture recognition from the right
thigh. The tables below give the controls and scope of each result.

Read the **[research report](docs/RESEARCH_REPORT.md)** for the study design,
architecture, matched results, ablations, and limitations. The
**[experiment map](docs/research/README.md)** organizes the full research sequence
by question and points to the original records.

The `main` branch contains the current research-software and evidence package.
The retained method is Confidence-Triggered Gravity Residual (CTGR), with strict
HERA-CTGR v1 reported as its highest source point-estimate extension. Research
disposition and publication boundaries are defined in the
[evidence index](docs/EVIDENCE_INDEX.md),
[project status](docs/PROJECT_STATUS.md), and
[publication checklist](docs/PUBLICATION_CHECKLIST.md).

CTGR and HERA are project-developed feature, expert-routing, and calibration
pipelines built around standard Extra Trees classifiers. The HARTH studies use
a separate engineered-feature pipeline with standard Random Forest classifiers,
trained within HARTH. Their scores describe different methods and tasks.

## Results at a glance

### Matched source development

Ten source participants, 725 windows, five participant-exclusive outer folds,
and five fixed seeds; three classes: mobility, sitting, and standing. Accuracy
is pooled across windows within each seed, then averaged over seeds. Macro-F1
is calculated per participant, then averaged over participants and seeds.

| Method | Channels | Accuracy | Participant macro-F1 | Matched interpretation |
|---|---:|---:|---:|---|
| RMRP: denoised GSP + Extra Trees | 6 | 84.579% | 83.953% | Matched base |
| CTGR: base + confidence-triggered gravity expert | 9 | 86.979% | 86.540% | +2.586 macro-F1 points over RMRP; development gates passed |
| Strict HERA-v1: CTGR candidate ensemble, calibration and consistency veto | 9 | **87.228%** | **86.849%** | +0.309 points over CTGR; incremental advancement gates failed |

CTGR adds native gravity, so its comparison with RMRP changes the sensor budget.
These ten people were reused for development; training seeds are not additional
independent participants. Both paired gain intervals cross zero. These are
five-seed means, rather than the best individual seed. See the
[matched evidence](docs/EVIDENCE_INDEX.md) for uncertainty and participant harms.

A separate seed-11 ablation also favored structured gravity use: CTGR reached
86.474% participant macro-F1 versus 78.653% for flat nine-channel Extra Trees,
74.576% for Random Forest, and 75.579% for XGBoost. Those fixed flat models had a
different selection budget from nested CTGR; this supports the tested pipeline,
without establishing an algorithm-wide ranking. The
[complete eight-method table](docs/RESEARCH_REPORT.md#43-fixed-source-ablation)
includes all six attempted configurations and both cached matched controls.

### External diagnostics

**HARTH sensor placement: binary sitting/standing.** The same Random Forest
recipe was evaluated on 22 participants, five participant-exclusive folds and
13,715 non-overlapping five-second windows.

| Project rich-feature Random Forest | Window accuracy | Participant macro-F1 |
|---|---:|---:|
| Lower back | 81.291% | 59.037% |
| Right thigh | **99.723%** | **97.261%** |
| Back + thigh | 99.672% | 97.198% |

The thigh improved macro-F1 by 38.224 points over the back control (95% paired
bootstrap interval: +30.320 to +46.353); 21 participants improved and one tied.
Thigh sitting and standing recall were 99.913% and 98.707%. Windows were confined
to annotated activity bouts, so this is a placement diagnostic under pure-bout
segmentation. It does not evaluate continuous activity detection or frozen
CTGR/HERA transfer. Fusion did not improve the thigh-only point estimate.

**HARTH multiclass comparison.** All rows below were executed on the same
22-participant leave-one-participant-out replay, with majority-label windows and
sample-level scoring. These are merged nine-class results.

| Executed method | Sample accuracy | Pooled macro-F1 |
|---|---:|---:|
| Project fused rich-feature Random Forest | 93.56% | 85.37% |
| Reference-style SVM | 93.79% | 86.02% |
| Reference-style Random Forest | 94.06% | 86.40% |
| Reference-style XGBoost | **94.22%** | **87.82%** |

The project pipeline is competitive but trails the executed reference methods
on this endpoint. These scores were produced locally; differences in filtering
and aggregation preclude treating them as an exact rerun of the paper's scores.

**AICOS frozen transfer: three classes.** The same 38 complete development
participants and 40,588 windows were scored without fitting on AICOS.

| Source-trained method | Window accuracy | Participant macro-F1 |
|---|---:|---:|
| Six-channel denoised GSP base (B6) | **76.28%** | 67.424% |
| Flat nine-channel Extra Trees (B9) | 68.10% | 64.638% |
| Confidence-triggered CTGR (T9) | 76.27% | **68.972%** |

CTGR gained 1.549 macro-F1 points over B6 with essentially unchanged accuracy.
This remains a conditional transfer diagnostic: gravity is derived and the
acquisition logger's axis/polarity convention is unresolved.

The [research report](docs/RESEARCH_REPORT.md) includes all HARTH multiclass
arms, twelve-class results and rejected extensions. The
[metric audit](docs/research/REPORTED_METRICS_AUDIT_20260920.md) records the
recomputation, denominators and artifact hashes behind these tables.

### Original locked target

On the distinct ten-person target cohort and original six-channel protocol,
Compact DANN reached **68.084%** mean participant macro-F1, effectively tied with
Compact CORAL. The MoRe-HAR hypothesis was unsupported. This one-time result is
preserved in the [target report](results/confirmatory/zero_shot_v1/publication_report_v1.json).
It is not a target evaluation of CTGR or HERA.

## What the repository contributes

- A participant-exclusive, released-block protocol with explicit residual risk
  where the dataset does not provide trial or timestamp boundaries.
- A one-time, evidence-gated target evaluation with participant-level uncertainty,
  lower-tail outcomes, calibration, and preserved negative results.
- RMRP, CTGR, CAGE-HAR, and HERA-CTGR research implementations with matched
  controls and explicit promotion gates.
- Reproducible audits for boundary leakage, unit conversion, annotation-dependent
  preprocessing, external transfer, robustness, and participant harms.
- A publication-oriented evidence policy that separates locked confirmation,
  source development, post-confirmatory analysis, diagnostics, failures, and
  superseded results.

The project does not establish state-of-the-art performance, clinical validity,
universal disability generalization, or independent superiority of CTGR/HERA.
The MoRe-HAR preregistered target hypothesis was not supported and remains part
of the preserved evidence record.

## Quick start

Requirements: Python 3.11 and
[`uv`](https://docs.astral.sh/uv/). Raw datasets and large checkpoints are not
distributed through Git.

```powershell
git clone --config core.longpaths=true --branch main https://github.com/abdullahuseyinli-dot/inclusive-shift-har.git
Set-Location inclusive-shift-har
uv sync --locked --extra training-cpu --group research
uv pip install --python .venv --require-hashes -r requirements/external-har-research.lock
uv run --no-sync inclusive-shift-har --version
```

Run the offline structural gates:

```powershell
uv run --no-sync inclusive-shift-har validate-manifests --json
uv run --no-sync inclusive-shift-har audit-splits `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --json
uv run --no-sync inclusive-shift-har validate-artifacts `
  --artifact-root results --require-artifacts --json
```

The [reproducibility guide](docs/REPRODUCIBILITY.md) distinguishes checks that
work from a Git clone from reconstructions that require separately restored,
hash-verified predictions or provider data.

## Repository map

| Path | Purpose |
|---|---|
| `src/inclusive_shift_har/` | Dataset adapters, models, experiments, evaluation, and artifact validation |
| `configs/` | Versioned datasets, protocols, models, experiments, schemas, and release policy |
| `results/` | Tracked aggregate evidence, locked reports, protocol records, and failure records |
| `docs/` | Research report, experiment map, benchmark/data/model cards, evidence indexes, and runbooks |
| `paper/` | Historical paper outline and addendum; see its README before reuse |
| `manifests/` | Dataset provenance and immutable source identities |
| `tests/` | Synthetic contract, leakage, statistics, artifact, and release tests |
| `.audit/` | Local create-only evidence and large run artifacts; intentionally excluded from Git |

Start with the [documentation map](docs/README.md). Historical files remain in
place because paths and byte hashes are part of the evidence chain. The
[supersession map](docs/EVIDENCE_SUPERSESSION.md) identifies the current
interpretation without rewriting those records.

## Data and ethical scope

The primary dataset is InclusiveHAR v4
([DOI 10.17632/r78dn3f6nc.4](https://doi.org/10.17632/r78dn3f6nc.4)).
The benchmark uses a functional three-class endpoint: mobility, sitting, and
standing. For wheelchair users, the released `Walking` label can represent manual
propulsion and must not be described as ordinary gait.

Participant identity, disability/device metadata, timestamps, location, labels,
and row-order proxies are prohibited model inputs. Performance differences
describe limitations of models, data, and sensing interfaces; they are not
measures of a participant's capability. This software is not intended for medical
diagnosis, rehabilitation decisions, or safety-critical control.

Each third-party dataset retains its own licence and citation requirements.
Apache-2.0 covers repository-authored code and documentation only. Raw data,
third-party implementations, papers, predictions, and checkpoints are excluded
unless a separate distribution right is documented.

## Reuse and citation

Use [CITATION.cff](CITATION.cff) for the software citation and cite every dataset
and external method separately. Candidate Zenodo metadata is in
[.zenodo.json](.zenodo.json); it does not assert that a deposit or DOI already
exists.

Contributions must follow [CONTRIBUTING.md](CONTRIBUTING.md). Security and private
disclosure guidance is in [SECURITY.md](SECURITY.md). Release history is summarized
in [CHANGELOG.md](CHANGELOG.md).
