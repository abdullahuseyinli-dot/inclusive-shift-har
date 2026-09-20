# Reported metrics: verification and presentation audit

**Date:** 20 September 2026

**Status:** completed analysis of preserved evidence; no new model fits or dataset access.

## Purpose

The front-page HARTH binary table omitted an available accuracy and incorrectly
described the experiment as leave-one-participant-out. Its correct result is
**99.723% window accuracy and 97.261% mean participant macro-F1**, using five
participant-exclusive folds. The separate HARTH multiclass replay uses
22 leave-one-participant-out folds. This correction changes the presentation,
not the historical predictions or study contracts.

The review also adds accuracy to the five-seed source comparison, restores the
complete fixed source ablation to the reading sequence, and places executed
external controls beside the project pipelines. Source-development results,
placement diagnostics and frozen transfer remain separate comparison groups.

## Recomputed evidence

The [machine-readable record](../../results/research/reported_metrics_audit_v1.json)
contains full-precision metrics, aggregate confusion matrices, sorted participant
F1 values, denominators, per-seed source scores and input-file SHA-256 bindings.
All **796 numerical and structural checks passed**, using an absolute tolerance
of 1e-12 for metric agreement. The target cohort P11-P20 was not accessed.

| Comparison | Verification source | Evaluation contract |
|---|---|---|
| RMRP, CTGR, strict HERA | Canonically hash-bound cross-fitted probabilities; primary F1 reproduced exactly | 725 windows, ten people, five outer folds, seeds 11/23/47/89/131 |
| Six Round A cells and two controls | Complete matrix of saved probabilities; identities and control probabilities aligned with canonical seed 11 | Same 725 windows, ten people and five folds; fixed cells versus cached nested controls |
| HARTH binary placement | All four arms' saved predictions and probabilities; fold records checked for disjoint and complete participant coverage | 13,715 windows, 22 people, five outer folds; sitting/standing |
| HARTH multiclass replay | Sample metrics recomputed from saved confusion matrices; window confusion matrices verified against saved predictions | 6,461,328 samples, 22 leave-one-participant-out folds; twelve and merged nine classes |
| AICOS source-frozen transfer | Saved probabilities restricted to the recorded complete-participant mask | 40,588 shared windows, 38 people, three classes; no external fitting |

Historical artifact paths in the JSON are relative to the indicated current or
historical HAR worktree. They are restoration identities, not prerequisites for
the repository's synthetic CI checks. Original run records and previous evidence
indexes remain unchanged.

## Accuracy and macro-F1

For a confusion matrix with true classes in rows and predicted classes in
columns, accuracy is its trace divided by its total count. Class F1 is twice
the class diagonal divided by the sum of its row and column totals, with zero
for a zero denominator. Macro-F1 averages those class scores.

The source, binary HARTH and AICOS primary endpoint computes that macro-F1
separately for each person and averages people equally. The source table then
averages the five seed-specific metrics. It does not average the probabilities
before classifying and does not select the best seed.

| Five-seed source method | Mean window accuracy (%) | Mean participant macro-F1 (%) |
|---|---:|---:|
| RMRP | 84.579 | 83.953 |
| CTGR | 86.979 | 86.540 |
| Strict HERA-v1 | 87.228 | 86.849 |

For HARTH thigh, the confusion matrix is `[[11540, 10], [28, 2137]]` in
`[sitting, standing]` order: 13,677 correct windows and 38 errors. Accuracy is
99.723%, pooled window macro-F1 is 99.477%, and equal-person macro-F1 is 97.261%.
The different values reflect participant and class weighting. Windows are
confined to physical and annotated activity runs; these results do not measure
continuous recognition without those boundaries.

The multiclass replay instead pools sample confusion matrices. It must not be
compared directly with the original paper's average across participant folds.
No sample-level prediction regeneration or new raw-data analysis was performed
in this audit.

## Results selected for the main presentation

- **CTGR is the retained source improvement:** +2.586 participant macro-F1 points
  over its six-channel base across five seeds, with eight participant wins and
  two harms. The method also adds native gravity; it is not an equal-input
  architecture-only comparison, and its descriptive interval crosses zero.
- **Strict HERA is the highest matched source mean:** its +0.309-point increment
  has an interval crossing zero and failed advancement gates. Isolated seed
  peaks do not replace the five-seed mean.
- **The fixed source ablation supports selective gravity use:** the tested flat
  nine-channel Extra Trees, RF and XGBoost configurations all underperformed
  CTGR. The selection budgets differ, so the result does not rank all possible
  configurations of those learner families.
- **HARTH thigh is a strong placement result:** the same RF recipe improved
  21 participants and harmed none relative to back. It is a separately trained
  accelerometer pipeline, not a frozen CTGR/HERA transfer result. Adding back
  to thigh did not improve the binary point estimate.
- **HARTH fusion is useful for the multiclass task:** fused project RF exceeds
  its single-placement versions, but trails all three executed reference-style
  methods on merged nine-class macro-F1. All six arms and both class mappings
  are retained in the research report.
- **AICOS shows a conditional benefit:** CTGR exceeds B6's participant macro-F1
  by 1.549 points at essentially unchanged accuracy; unresolved logger axes and
  derived gravity prevent confirmation of the original sensor interface.

Labelled personalization, label-informed oracles, legacy UCI scores, locked
target results and rejected successors remain in their own evidence groups.
They are not combined into a single improvement curve. These findings support
implemented research methods and useful mechanism evidence; they do not prove
state-of-the-art performance or global algorithmic novelty.

## Rechecking the displayed arithmetic

From a repository checkout, the standard-library example below checks the
record hash and the source accuracy and F1 averages. It requires no provider data.
The same trace/count calculation applies to each external confusion matrix.

```python
import hashlib
import json
from pathlib import Path
from statistics import fmean

record = json.loads(Path("results/research/reported_metrics_audit_v1.json").read_text())
expected_hash = record.pop("record_sha256")
canonical = json.dumps(record, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False, allow_nan=False).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == expected_hash
for method in record["source_five_seed"]["results"]:
    accuracies, participant_f1 = [], []
    for seed in method["by_seed"]:
        matrix = seed["confusion_matrix"]
        accuracy = sum(matrix[i][i] for i in range(3)) / sum(map(sum, matrix))
        assert abs(accuracy - seed["accuracy"]) < 1e-12
        accuracies.append(accuracy)
        participant_f1.append(fmean(seed["participant_macro_f1_values_sorted"]))
    assert abs(fmean(accuracies) - method["mean_window_accuracy_across_seeds"]) < 1e-12
    assert abs(fmean(participant_f1) - method["mean_participant_macro_f1_across_seeds"]) < 1e-12
```

This verifies aggregate arithmetic. Independently reconstructing participant
scores and fold predictions requires restoration of the hash-bound archives
listed in `input_files`, as described in the
[reproducibility guide](../REPRODUCIBILITY.md).
