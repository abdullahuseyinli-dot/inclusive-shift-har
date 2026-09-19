# HARTH published-protocol replication v1

This is a create-only diagnostic lane for an apples-to-apples comparison with
Logacjov et al. (Sensors 2021, 21, 7853). It does not modify the locked
InclusiveHAR/HERA protocol and does not make a state-of-the-art or deployment
claim.

## Frozen contract

- Source: the official UCI HARTH v1.2 archive, 22 participant CSV files.
- Signals: lower-back and right-thigh triaxial accelerometers, six channels.
- Sampling: 50 Hz.
- Labels: all 12 provider labels, in the paper's order
  `[walking, running, shuffling, stairs ascending, stairs descending,
  standing, sitting, lying, cycling sit, cycling stand, cycling sit inactive,
  cycling stand inactive]`.
- Preprocessing: fourth-order 20 Hz Butterworth low-pass; five-second,
  non-overlapping windows (250 samples); majority label per window.
- Features: the official 161-dimensional gravity/movement, time-domain,
  cross-axis, cross-sensor, and frequency-domain feature family. Gravity is
  estimated with the official fourth-order 1 Hz Butterworth filter.
- Evaluation: leave-one-subject-out (22 folds), with no target-participant
  information used for fitting or model selection.
- Primary paper-compatible metrics: pooled sample-level confusion matrix after
  repeating each window prediction over its samples, then macro precision,
  macro recall, macro-F1, and accuracy. A nine-physical-activity merge follows
  the paper: shuffling and standing-transport merge into standing; sitting-
  transport merges into sitting.

## Models

The published controls use the fixed hyperparameters reported by the reference
repository: RBF SVM (`C=10`, `gamma=scale`), RF (`80` trees,
`min_samples_split=10`, `max_features=sqrt`, balanced class weight), and XGB
(`1024` histogram trees, depth 3, learning rate 0.1, `reg_lambda=1`). The SVM
uses a train-only min–max scaler; the RF and XGB lanes retain the reference
repository's unscaled feature contract.

The three project lanes use the existing fixed 161-feature lower-back and
right-thigh rich views, and their 322-feature concatenation, with the retained
300-tree participant-aware RF readout. They are evaluated under the same 22
LOSO folds and windows; they are architecture comparisons, not replacements for
the published baselines.

## Evidence boundary

The source code and data are hashed in the run receipt. Raw source archives are
temporary and are not retained in the repository. Metrics are diagnostic because
the paper's sample-unfolding and majority-window protocol is intentionally
reproduced; the repository's boundary-safe research lanes remain the evidence
for current development decisions.
