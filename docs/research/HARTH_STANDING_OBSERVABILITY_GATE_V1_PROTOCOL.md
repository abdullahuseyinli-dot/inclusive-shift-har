# HARTH standing observability gate v1

## Question

Does the existing right-thigh sensor supply standing information that is absent
from the lower-back stream, and can a train-only confidence gate add that evidence
without sacrificing the retained sitting performance?

## Fixed contract

- Official HARTH archive, 22 participants, lower-back and right-thigh accelerometers,
  nominal 50 Hz.
- Provider labels 7 → sitting and 6 → standing.
- Non-overlapping 250-sample windows confined to physical and label runs.
- Five participant-exclusive folds using retained seed 11 assignment.
- Existing 161-feature map from `harth_crsp_back.py` for each sensor.
- Random Forest: 300 trees, `sqrt` features, leaf size 2,
  `balanced_subsample`, seeds 11 + fold, four fit workers.

## Four fixed arms

| Arm | Input | Purpose |
|---|---|---|
| B_back_rich | lower-back rich features | retained deployable back-only control |
| T_thigh_rich | right-thigh rich features | standing observability positive control |
| F_fused_rich | concatenated back + thigh rich features | measured-information upper bound |
| G_confidence_gated | back prediction with thigh override | test whether reliable thigh evidence preserves sitting while rescuing standing |

The gate is selected separately in each outer fold using only outer-training
participants. Four inner participant folds generate out-of-fold back/thigh
probabilities. A threshold is selected to maximize standing recall while limiting
sitting-recall loss relative to the inner back control to 2 percentage points. The
held-out participant labels never select the threshold.

## Metrics and promotion gate

Report participant macro-F1, accuracy, balanced accuracy, sitting/standing
precision, recall and F1, confusion matrices, prediction rates, participant wins,
harms, ties, worst harm, and a 10,000-draw paired participant bootstrap.

Promotion requires at least +5 percentage points participant macro-F1, +10 points
standing recall and standing F1, sitting-recall loss no greater than 2 points, at
least 16/22 participant wins, no participant harm below −5 points, and a positive
paired bootstrap lower bound. Thigh and fused gains are sensor-information results;
they are not back-only deployment claims.

No new seeds, synthetic gyroscope, external dataset, phone data, or InclusiveHAR
target access is allowed in this experiment.
