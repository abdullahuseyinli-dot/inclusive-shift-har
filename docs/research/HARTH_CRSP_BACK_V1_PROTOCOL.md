# HARTH CRSP-Back v1 bounded protocol

This is an exploratory external-development experiment. It does not alter the locked
InclusiveHAR/HERA protocol and makes no population, deployment, or confirmation claim.

## Question

Can lower-back sitting/standing recognition improve when the existing 2.56-second
control is given longer context, a fixed richer orientation/temporal feature map, and
training-only paired thigh supervision? The paired arm must perform inference from the
lower-back stream alone.

## Data and split

- Official UCI HARTH archive, 22 participants, right-thigh and lower-back Axivity
  accelerometers, nominal 50 Hz, accelerometer channels only.
- Provider labels 7 and 6 are mapped to `[sitting, standing]`.
- Physical timestamp gaps greater than 0.06 seconds, non-finite rows, label changes,
  and participant boundaries terminate a segment.
- Non-overlapping windows are made only after participant and segment partitioning.
- Window sizes are 128 samples (2.56 seconds) for the exact control and 250 samples
  (5 seconds) for all proposed arms.
- Five participant-exclusive folds use the retained seed-11 participant permutation.

## Fixed arms

| Arm | Window | Representation | Readout | Purpose |
|---|---:|---|---|---|
| A | 128 | Existing 24 back features | 300-tree RF | Exact run-004 control replay |
| B | 250 | Existing 24 back features | 300-tree RF | Context-only ablation |
| C | 250 | Fixed 161-feature back family | 300-tree RF | Orientation/feature ablation |
| D | 250 | Fixed four-subwindow temporal map | 300-tree RF | Temporal representation ablation |
| E | 250 | Temporal map + confidence-gated physics | 300-tree RF | Selective physics ablation |
| F | 250 | E's back map + back-derived paired reconstruction | 300-tree RF | Primary cross-sensor transfer test |

The 161-feature family contains fixed axis statistics, norm statistics, correlations,
covariance/eigenvalues, spectral summaries, four subwindows, and lag/difference
features. It is materialized without labels, IDs, timestamps, or target information.
The physics map contains a low-pass mean-gravity direction, dynamic residual energy,
directional stability, covariance eigenvalues, and a deterministic stationary confidence.

For F, a multi-output ridge map is fitted independently inside each outer training fold
from back features to simultaneously paired thigh features. Its back-derived predicted
thigh representation is appended at both training and inference. Held-out thigh data
are recorded only as a reconstruction diagnostic and never enter the classifier.

## Fit and metrics

All arms use `RandomForestClassifier(n_estimators=300, max_features="sqrt",
min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=4)` with seed `11 + fold`.
No early stopping, feature selection, class-weight search, or threshold tuning is used.

Primary endpoint is the mean of participant-level binary macro-F1. Secondary outputs are
per-class precision/recall/F1, pooled accuracy/micro-F1, confusion matrices, NLL, Brier,
worst participant, lower decile, paired participant bootstrap (10,000 resamples, seed
1729), and wins/harms/ties against A.

Promotion is exploratory only and requires: at least +5 percentage points participant
macro-F1 versus A, positive paired bootstrap lower bound, standing recall gain of at
least 10 points, sitting recall loss no greater than 2 points, at least 16/22 participant
wins, and no participant loss below 5 points. A candidate failing these gates is retained
as a diagnostic result and does not trigger more seeds or external stages.

## Provenance and boundaries

The archive is downloaded to a temporary path, hashed, and removed after the run. New
outputs are create-only under `.audit/harth_crsp_back_20260918/`. Raw evidence and all
prior runs remain untouched. Results are not compared as a leaderboard against multiclass
or dual-sensor papers, and they do not reopen InclusiveHAR P11-P20.
