# Corrected UCI-HAR legacy reproduction track

## Evidence boundary

This track reconstructs the coursework architectures and replaces its random window validation with participant-exclusive validation. At this document snapshot it contains **no corrected model result yet**. Existing saved test metrics remain legacy exploratory/development-consumed and are not re-labelled. The executable three-model/five-seed scope is locked in `configs/experiments/uci_har_corrected_reproduction_v1_1.yaml`; it supersedes the earlier planning YAML that also listed an unimplemented temporal BiLSTM variant.

The official UCI test is excluded from:

- hyperparameter selection;
- checkpoint selection and early stopping;
- normalization;
- calibration and threshold choice;
- class-count or ontology derivation;
- source pretraining; and
- new confirmatory claims.

Opening it through the adapter requires the explicit purpose `legacy_exploratory_development_consumed_audit`.

## Corrected data path

The old notebook expected six custom-named files under an unarchived layout and did not load participant IDs. The corrected adapter reads the official Version 1.0 ZIP directly:

- `body_acc_{x,y,z}_<split>.txt`;
- `body_gyro_{x,y,z}_<split>.txt`;
- `subject_<split>.txt`; and
- `y_<split>.txt`.

It preserves fixed activity IDs 1–6, emits `[window,128,6]` float32 tensors, validates finiteness and row alignment, and assigns stable released-window IDs. The source archive and dataset manifest hashes travel with the protocol.

## Subject-grouped source development

Only the 7,352 windows and 21 participants in the official train split enter development. Five deterministic folds are formed by SHA-256 ranking under namespace `inclusive-shift-har|uci_har_v1|source_grouped_cv_v1`, seed 5062, followed by round-robin assignment.

| Fold | Validation subjects | Validation windows |
|---|---|---:|
| `uci_source_cv_01` | 1, 11, 21, 25, 30 | 1,863 |
| `uci_source_cv_02` | 3, 15, 19, 28 | 1,411 |
| `uci_source_cv_03` | 14, 16, 22, 23 | 1,382 |
| `uci_source_cv_04` | 8, 26, 27, 29 | 1,393 |
| `uci_source_cv_05` | 5, 6, 7, 17 | 1,303 |

Each participant appears in validation exactly once. Every released window belonging to a participant stays with that participant, so the provider-reported overlapping windows cannot cross a fold boundary. Normalization is fit on each fold's training participants only. The official processed release lacks trial/raw-sample identifiers, but participant exclusivity is sufficient to rule out the coursework's same-participant overlapping-window leakage path.

Model selection uses a common participant-level validation macro-F1, with worst-participant macro-F1 and then smaller parameter count as tie-breakers. It does not compare unlike weighted, label-smoothed, and unweighted validation losses.

## Architecture reconstruction

The framework-independent specifications and runnable PyTorch graphs are tied to notebook SHA-256 `c25d78ad1a21e485a7bd0d9dd5cc74bba4d3872465be300d7be56d8b829f636` and exact cell references.

| Architecture | Exact source cells | Parameters for `[128,6]`, six classes | Status |
|---|---|---:|---|
| BiLSTM, hidden 192, two layers | 10/11/`4929a1ae`; 26/27/`12004a09` | 1,198,086 | Reconstructed, not trained |
| BiLSTM + temporal max head | 10/11/`4929a1ae`; 26/27/`12004a09` | 1,201,158 | Reconstructed, not trained |
| BiLSTM-256 + temporal max head | 10/11/`4929a1ae`; 26/27/`12004a09` | 2,125,830 | Reconstructed, not trained |
| CNN1D, widths 128/256/512 | 12/13/`cc39cfc2`; 26/27/`12004a09` | 568,198 | Reconstructed, not trained |
| Joint BiLSTM-256/CNN-128 | plus 14/15/`8fcda24e` | 2,694,028 | Reconstructed, not trained |
| Joint BiLSTM-512/CNN-256 | plus 14/15/`8fcda24e` | 10,696,460 | Reconstructed, not trained |

The last two are jointly optimized branches whose logits are averaged. They are not ensembles of independently trained estimators. Architecture records are in `legacy/corrected_architecture_specs.json`, runnable graphs are in `inclusive_shift_har.models.legacy_models`, and PyTorch 2.12.0 is supplied through the repository's separately locked CPU/CUDA training profiles. Synthetic forward, parameter-count, exact-logit-fusion, and dual-branch gradient tests pass; no dataset training has been run.

## Predeclared training correction

The configuration retains the legacy optimizer-scale defaults for reproduction—AdamW, learning rate `3e-4`, weight decay `1e-4`, at most 40 epochs, three warm-up epochs, and patience eight—while changing the scientific controls:

- five seeds: 42, 1337, 2025, 31415, and 271828;
- subject-grouped folds;
- fixed six-class ontology from release metadata;
- training-only normalization;
- a common participant-level selection metric;
- resumable, lineage-complete checkpoints when training is implemented; and
- preservation of failed, negative, and out-of-memory runs.

No accuracy, F1, calibration, latency, or efficiency value in this document is a corrected result.

## Source-pretraining policy

After source-only model selection, a source encoder may be refit on all official-train participants. The official test remains excluded. Cross-source label use follows the locked ontology: sitting is exact, standing is provisional, and walking is excluded from the all-participant exact track. No InclusiveHAR target evaluation was run as part of this work.

## Current gate status

The acquisition, adapter audit, deterministic split construction, and synthetic tests pass. Model training/evaluation remains `configured_not_run`; it requires the repository training gate and complete artifact lineage. The machine protocol hash is recorded in `results/protocol/uci_har_source_grouped_v1.json`.
