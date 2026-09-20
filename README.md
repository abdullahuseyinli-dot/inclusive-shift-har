# InclusiveShift-HAR

Auditable research software for participant-exclusive inertial human activity
recognition under population and sensor-interface shift.

[![CI](https://github.com/abdullahuseyinli-dot/inclusive-shift-har/actions/workflows/ci.yml/badge.svg?branch=research%2Fhar-substantiation-pilot-pipeline-20260908)](https://github.com/abdullahuseyinli-dot/inclusive-shift-har/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB.svg)](pyproject.toml)

## Research overview

The project examines three questions: population transfer under a locked
participant split, selective gravity-based correction of posture errors, and
generalization across sensor interfaces. The current method combines a denoised
geometric/spectral base with a confidence-triggered sitting/standing expert.

Read the **[research report](docs/RESEARCH_REPORT.md)** for the study design,
architecture, matched results, ablations, and limitations. The
**[experiment map](docs/research/README.md)** organizes the full research sequence
by question and points to the original records.

The research phase is frozen for repository finalization. The retained
source-development package is Confidence-Triggered Gravity Residual (CTGR), with
strict HERA-CTGR v1 reported as the highest source point estimate. The latest
unconditional posture-routing candidate failed its source non-regression,
external-gain, and interface-qualification gates. No further architecture search
on the repeatedly used source participants is pending.

This repository is being prepared as an exploratory research-software and
evidence package. It has no published Zenodo DOI at this revision. The current
claim and publication boundaries are defined in the
[evidence index](docs/EVIDENCE_INDEX.md),
[project status](docs/PROJECT_STATUS.md), and
[publication checklist](docs/PUBLICATION_CHECKLIST.md).

## Results at a glance

### Matched source development

Ten source participants, 725 windows, five participant-exclusive outer folds,
and five fixed seeds. The endpoint is mean participant macro-F1.

| Method | Channels | Macro-F1 | Matched interpretation |
|---|---:|---:|---|
| RMRP | 6 | 83.953% | Baseline |
| CTGR | 9 | 86.540% | +2.586 percentage points over RMRP; development gates passed |
| Strict HERA-v1 | 9 | **86.849%** | +0.309 points over CTGR; interval crosses zero and advancement gates failed |

CTGR adds native gravity, so its comparison with RMRP changes the sensor budget.
These ten people were reused for development; training seeds are not additional
independent participants. See the [matched evidence](docs/EVIDENCE_INDEX.md).

### Original locked target

On the distinct ten-person target cohort and original six-channel protocol,
Compact DANN reached **68.084%** mean participant macro-F1, effectively tied with
Compact CORAL. The MoRe-HAR hypothesis was unsupported. This one-time result is
preserved in the [target report](results/confirmatory/zero_shot_v1/publication_report_v1.json).
It is not a target evaluation of CTGR or HERA.

### External diagnostics

Frozen CTGR reached **68.972%** mean participant macro-F1 on 38 complete AICOS
development participants; logger axes remain unresolved. On HARTH's separate
merged nine-class replay, project fused RF reached **85.37% pooled macro-F1**,
below the executed published-style XGBoost at **87.82%**. The
[research report](docs/RESEARCH_REPORT.md) gives each protocol, its matched
controls, accuracy, and limitations.

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
git clone --config core.longpaths=true --branch research/har-substantiation-pilot-pipeline-20260908 https://github.com/abdullahuseyinli-dot/inclusive-shift-har.git
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
