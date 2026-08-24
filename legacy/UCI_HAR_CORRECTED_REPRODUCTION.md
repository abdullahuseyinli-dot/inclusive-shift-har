# Corrected UCI-HAR legacy reproduction track

## Evidence boundary

This track reconstructs the coursework architectures and replaces its random
window validation with participant-exclusive validation. The corrected
official-train-only reproduction is complete: 75 of 75 predeclared CUDA cells
(three models, five grouped folds, five seeds) completed on their first attempt.
Existing saved coursework test metrics remain legacy
exploratory/development-consumed and are not re-labelled. The executable scope
is locked in `configs/experiments/uci_har_corrected_reproduction_v1_1.yaml`; it
supersedes the earlier planning YAML that also listed an unimplemented temporal
BiLSTM variant.

The official UCI test is excluded from:

- hyperparameter selection;
- checkpoint selection and early stopping;
- normalization;
- calibration and threshold choice;
- class-count or ontology derivation;
- source pretraining; and
- new confirmatory claims.

Opening it through the adapter requires the explicit purpose
`legacy_exploratory_development_consumed_audit`.

## Corrected data path

The old notebook expected six custom-named files under an unarchived layout and
did not load participant IDs. The corrected adapter reads the official Version
1.0 ZIP directly:

- `body_acc_{x,y,z}_<split>.txt`;
- `body_gyro_{x,y,z}_<split>.txt`;
- `subject_<split>.txt`; and
- `y_<split>.txt`.

It preserves fixed activity IDs 1–6, emits `[window,128,6]` float32 tensors,
validates finiteness and row alignment, and assigns stable released-window IDs.
The source archive and dataset manifest hashes travel with the protocol.

## Subject-grouped source development

Only the 7,352 windows and 21 participants in the official train split enter
development. Five deterministic folds are formed by SHA-256 ranking under
namespace `inclusive-shift-har|uci_har_v1|source_grouped_cv_v1`, seed 5062,
followed by round-robin assignment.

| Fold | Validation subjects | Validation windows |
|---|---|---:|
| `uci_source_cv_01` | 1, 11, 21, 25, 30 | 1,863 |
| `uci_source_cv_02` | 3, 15, 19, 28 | 1,411 |
| `uci_source_cv_03` | 14, 16, 22, 23 | 1,382 |
| `uci_source_cv_04` | 8, 26, 27, 29 | 1,393 |
| `uci_source_cv_05` | 5, 6, 7, 17 | 1,303 |

Each participant appears in validation exactly once. Every released window
belonging to a participant stays with that participant, so the
provider-reported overlapping windows cannot cross a fold boundary.
Normalization is fit on each fold's training participants only. The official
processed release lacks trial/raw-sample identifiers, but participant
exclusivity is sufficient to rule out the coursework's same-participant
overlapping-window leakage path.

Model selection uses a common participant-level validation macro-F1, with
worst-participant macro-F1 and then smaller parameter count as tie-breakers. It
does not compare unlike weighted, label-smoothed, and unweighted validation
losses.

## Architecture reconstruction

The framework-independent specifications and runnable PyTorch graphs are tied to
notebook SHA-256
`c25d78ad1a21e485a7bd0d9dd5cc74bba4d3872465be300d7be56d8b829f636`
and exact cell references.

| Architecture | Exact source cells | Parameters for `[128,6]`, six classes | Status |
|---|---|---:|---|
| BiLSTM, hidden 192, two layers | 10/11/`4929a1ae`; 26/27/`12004a09` | 1,198,086 | Reconstructed and evaluated |
| BiLSTM + temporal max head | 10/11/`4929a1ae`; 26/27/`12004a09` | 1,201,158 | Reconstructed, not trained |
| BiLSTM-256 + temporal max head | 10/11/`4929a1ae`; 26/27/`12004a09` | 2,125,830 | Reconstructed, not trained |
| CNN1D, widths 128/256/512 | 12/13/`cc39cfc2`; 26/27/`12004a09` | 568,198 | Reconstructed and evaluated |
| Joint BiLSTM-256/CNN-128 | plus 14/15/`8fcda24e` | 2,694,028 | Reconstructed and evaluated |
| Joint BiLSTM-512/CNN-256 | plus 14/15/`8fcda24e` | 10,696,460 | Reconstructed, not trained |

The last two are jointly optimized branches whose logits are averaged. They are
not ensembles of independently trained estimators. Architecture records are in
`legacy/corrected_architecture_specs.json`, runnable graphs are in
`inclusive_shift_har.models.legacy_models`, and PyTorch 2.12.0 is supplied
through the repository's separately locked CPU/CUDA training profiles.
Synthetic forward, parameter-count, exact-logit-fusion, and dual-branch gradient
tests pass. Only the three v1.1-configured architectures above were trained in
this reproduction.

## Executed training correction

The configuration retained the legacy optimizer-scale defaults for
reproduction—AdamW, learning rate `3e-4`, weight decay `1e-4`, at most 40
epochs, three warm-up epochs, and patience eight—while changing the scientific
controls:

- five seeds: 42, 1337, 2025, 31415, and 271828;
- subject-grouped folds;
- fixed six-class ontology from release metadata;
- training-only normalization;
- a common participant-level selection metric;
- resumable, lineage-complete checkpoints; and
- preservation of failed, negative, and out-of-memory runs.

## Corrected official-train development results

The statistical unit is the participant. Each participant's out-of-fold value
is averaged across five seeds before the 21-participant summary. Intervals use
10,000 participant-clustered bootstrap resamples. Window metrics and calibration
metrics are descriptive; the official test set was not opened.

| Model | Mean participant macro-F1 (95% CI) | Worst | Lower decile | Balanced accuracy | NLL | Brier | ECE |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint BiLSTM-256/CNN-128 | 0.9118 [0.8830, 0.9360] | 0.7290 | 0.8631 | 0.9148 | 0.2794 | 0.1437 | 0.0194 |
| CNN1D | 0.8852 [0.8511, 0.9126] | 0.6328 | 0.8439 | 0.8893 | 0.3828 | 0.1955 | 0.0628 |
| BiLSTM-192 | 0.7084 [0.6828, 0.7300] | 0.5243 | 0.6495 | 0.7186 | 0.7031 | 0.3653 | 0.0297 |

The jointly trained two-branch model exceeded CNN1D by 0.0266 mean participant
macro-F1 (Holm-adjusted Wilcoxon `p=9.54e-06`; Holm-adjusted Monte Carlo
sign-flip `p=3.00e-05`). BiLSTM-192 was 0.1768 below CNN1D. These are corrected
source-development comparisons, not fresh test or confirmatory results, and do
not validate the coursework's repeatedly consumed test estimates.

The self-hashed machine report is
`results/legacy_reproduction/uci_har_source_grouped_v1/uci_source_grouped_report_v1.json`
(record SHA-256
`c2e0191259bd2f51cb58a91e51ebc27acfa84f47e1bf654dc04cf9ecaaafa1f6`).
Its 75 records attest `official_test_member_opened=false` and
`inclusivehar_data_or_target_accessed=false`; large checkpoints and prediction
arrays remain local ignored evidence rather than Git objects.

## Source-pretraining policy

After source-only model selection, a source encoder may be refit on all
official-train participants. That cross-source extension was not implemented or
run. The official test remains excluded. Cross-source label use follows the
locked ontology: sitting is exact, standing is provisional, and walking is
excluded from the all-participant exact track. No InclusiveHAR target evaluation
was run as part of this work.

## Current gate status

The acquisition, adapter audit, deterministic split construction, CUDA training,
artifact reconstruction, and participant-level aggregation pass. All 75 cells
were executed at commit
`092544ddfd6572a909332ffa5c05d5cae5a2c079`; aggregation was recorded at commit
`ad657d3aec3e2476ce7a1e8226bb2fd3a99bd721`. The machine protocol hash is in
`results/protocol/uci_har_source_grouped_v1.json`. The official test remains
consumed legacy evidence and unopened by the corrected route.
