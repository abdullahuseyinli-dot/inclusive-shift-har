# InclusiveHAR to AICOS-HAR zero-shot benchmark

> **2026-09-19 correction:** the historical tables below used AICOS acceleration
> and gravity in m/s² with InclusiveHAR models trained in g. Their numerical
> replay passed, but the cross-source unit contract was wrong. Preserve these
> tables as the original run, not as the current correctly aligned comparison.
> The complete 19-method correction, participant harms, routing analysis and
> knowledge-tree update are in [AICOS_POSTURE_REVIEW_20260919.md](AICOS_POSTURE_REVIEW_20260919.md).
> This defect does not alter the original InclusiveHAR source-development results.

**Execution date:** 2026-09-19  
**Status:** complete and independently replay-validated  
**Evidence scope:** external healthy-cohort cross-device and cross-position transfer;
not an ability-cohort confirmation and not a state-of-the-art claim

The source was the frozen 725-window InclusiveHAR P1--P10 population. The
validated seed-11 CTGR final predictor was reused byte-for-byte, and the matched
RMRP, CTGR, CAGE, HERA, Random Forest and XGBoost transfer suite used the same
source and target windows. No AICOS label entered fitting, selection,
calibration, normalization or routing. The AICOS provider `test` split was
opened only after archive verification and folds 1--5 signal qualification.

The primary endpoint contains the eight provider-test participants with all
three endpoint classes: S59, S68, S71, S75, S93, S99, S100 and S104. It contains
9,625 of the 19,856 qualified test windows. The all-participant coverage result
is retained separately because 13 of 21 eligible test participants lack at
least one endpoint class.

## Primary comparison

| Method | Accuracy (%) | Mean participant macro-F1 (%) | Bottom-30% (%) | Worst participant (%) | Mobility recall (%) | Sitting recall (%) | Standing recall (%) | Wins/harms/ties vs RMRP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Frozen-T9-CTGR | 85.51 | 69.13 | 53.91 | 38.23 | 100.00 | 29.68 | 65.20 | 4/4/0 |
| HERA-DG-strict | 85.18 | 68.80 | 57.59 | 47.94 | 100.00 | 29.28 | 64.08 | 4/4/0 |
| CTGR-DG-top3-equal | 85.13 | 68.79 | 57.60 | 47.94 | 100.00 | 29.14 | 63.92 | 4/4/0 |
| HERA-DG-v2-weighted | 85.13 | 68.79 | 57.60 | 47.94 | 100.00 | 29.14 | 63.92 | 4/4/0 |
| Frozen-U9-Unconditional | 85.04 | 68.48 | 56.16 | 45.22 | 100.00 | 32.49 | 62.56 | 5/3/0 |
| Frozen-B9-PhysicsRMRP | 87.58 | 68.22 | 60.50 | 58.73 | 100.00 | 25.13 | 74.57 | 4/4/0 |
| CTGR-DG | 85.26 | 68.20 | 57.76 | 47.94 | 100.00 | 29.14 | 64.40 | 5/3/0 |
| HERA-DG-context-safe | 85.27 | 68.18 | 57.73 | 47.94 | 100.00 | 29.28 | 64.40 | 5/3/0 |
| HERA-DG-v2-dual | 85.50 | 68.03 | 56.65 | 47.94 | 100.00 | 25.67 | 66.36 | 4/3/1 |
| HERA-DG-v2-core | 85.49 | 67.98 | 57.49 | 47.94 | 100.00 | 26.47 | 66.08 | 4/4/0 |
| HERA-DG-v2-full | 85.49 | 67.98 | 57.49 | 47.94 | 100.00 | 26.47 | 66.08 | 4/4/0 |
| HERA-DG-full | 85.76 | 67.85 | 57.09 | 47.94 | 100.00 | 25.80 | 67.32 | 5/3/0 |
| CAGE-DG | 82.40 | 65.99 | 54.64 | 44.10 | 100.00 | 42.38 | 49.42 | 5/3/0 |
| Frozen-B6-RMRP | 84.09 | 65.66 | 53.50 | 39.98 | 100.00 | 33.56 | 58.59 | 3/5/0 |
| RMRP-DG | 84.06 | 64.96 | 53.31 | 42.03 | 100.00 | 34.36 | 58.23 | reference |
| XGBoost-6ch | 78.58 | 62.63 | 53.97 | 49.96 | 100.00 | 38.77 | 35.76 | 3/5/0 |
| RandomForest-6ch | 75.27 | 57.61 | 47.37 | 37.89 | 100.00 | 47.06 | 20.54 | 2/6/0 |

Frozen T9/CTGR is the descriptive mean leader. Its gain over RMRP-DG is 4.17
macro-F1 points, with a participant bootstrap interval of -0.87 to 9.93 points;
four participants improve and four are harmed. The interval crosses zero, so
this result does not establish superiority over RMRP. Against XGBoost, the mean
gain is 6.51 points, interval 0.38 to 11.62, with seven participant gains and one
harm. Against Random Forest, the mean gain is 11.52 points, interval 6.70 to
16.85, with gains for all eight participants.

## Neural CPU replication controls

The machine exposed CPU PyTorch only. DeepConvLSTM and TinyHAR were therefore
run as a separate float32 replication lane rather than being mislabeled as the
frozen CUDA/float16 protocol. The fixed source-validation participants P8 and
P10 selected the epoch; each selected architecture was then reinitialized and
refit on all ten source participants before target inference. AICOS labels were
not used for epoch selection, normalization or fitting.

| Method | Accuracy (%) | Mean participant macro-F1 (%) | Bottom-30% (%) | Worst participant (%) | Mobility recall (%) | Sitting recall (%) | Standing recall (%) |
|---|---:|---:|---:|---:|---:|---:|---:|
| TinyHAR-6ch-CPU | 67.52 | 58.22 | 49.76 | 46.67 | 84.83 | 57.09 | 26.43 |
| DeepConvLSTM-6ch-CPU | 72.17 | 54.92 | 46.69 | 40.78 | 99.98 | 58.02 | 5.33 |

DeepConvLSTM selected epoch 7 and TinyHAR selected epoch 10. Both underperform
the source-frozen feature architectures. DeepConvLSTM is 10.04 macro-F1 points
below RMRP-DG with a participant bootstrap interval of -16.86 to -3.13 points.
TinyHAR is 6.74 points below RMRP-DG, interval -15.94 to 2.20. This lane adds a
useful negative result: generic temporal capacity does not solve the external
standing failure under the small source population.

Frozen B9 has the best pooled accuracy, standing recall, bottom-30% and worst
participant score, but it obtains that profile by lowering sitting recall. CAGE
and Random Forest move in the opposite direction: they improve sitting recall
while severely reducing standing recall. Strict HERA and equal-top-three CTGR
give the strongest robust tail among the routed methods, while frozen T9 gives
the highest mean. The external result therefore reproduces the project's
central posture tradeoff rather than resolving it.

## Data qualification and provenance

The 2,539,861,502-byte compressed archive passed the provider MD5
`faa895e7b203d664525b0a1e74542793` and independent SHA-256
`bd8c9a8b1999b95db587d8e482e01dff1509079294fb9f6a40574e6bac0b1544`.
It was never extracted. A label-independent physical unit audit retained 2,361
of 2,965 development-fold acquisitions and quarantined 604; it retained 809 of
877 provider-test acquisitions and quarantined 68. The test materialization has
13,899 mobility, 1,819 sitting and 4,138 standing windows across 21
participants. Total runtime was 169.3 seconds.

The complete evidence is under
`.audit/aicos_external_benchmark/aicos-provider-test-seed11-20260919-001`:

- `result.json`: reports, uncertainty, participant effects, strata and runtime;
- `predictions.npz`: aligned labels, IDs, primary mask and all probability matrices;
- `development_signal_qualification.json`: acquisition-level unit/rate audit;
- `execution_provenance.json`: branch, commit, command and implementation hashes;
- `validation.json`: replay and integrity checks;
- `completion_manifest.json`: final file hashes; and
- `REPORT.md`: human-readable run report.

The separate CPU neural evidence is under
`.audit/aicos_external_benchmark/aicos-neural-controls-cpu-seed11-20260919-001`.

The smallest justified next development experiment is a source-selected posture
mixture that preserves frozen B9's standing evidence and invokes the
sitting-favouring branch only when source-OOF physics and disagreement features
predict benefit. Selection must remain on source OOF or a newly declared
development domain. AICOS is now consumed and cannot be reused as an independent
confirmation set for that change.
