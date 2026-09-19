# HARTH gravity hierarchy v1 bounded protocol

This is a four-cell exploratory HARTH development experiment. It does not alter the
locked InclusiveHAR/HERA evidence and does not make a confirmation or deployment claim.

## Question

Does lower-back gravity geometry add information for sitting/standing, or do the recent
standing gains only reflect an operating-point shift? If geometry is useful, does a
training-only temporal state constraint preserve sitting while recovering standing?

## Fixed contract

- Official HARTH, 22 participants, lower-back and right-thigh Axivity accelerometers,
  nominal 50 Hz; only lower-back data enter these arms.
- Provider labels 7 and 6 map to sitting and standing.
- Non-overlapping 250-sample windows remain inside physical and label runs.
- Five participant-exclusive folds use the retained seed-11 assignment.
- The rich RF control uses the existing 161-feature map and the existing 300-tree
  `balanced_subsample` Random Forest contract.
- Gravity geometry is computed from a 25-sample edge-padded moving average of the
  lower-back accelerometer. No gyroscope, native gravity channel, thigh input, or
  synthetic channel is used.
- Calibration logit correction uses only the outer-training empirical standing prior.
- Transition probabilities and the two-window minimum dwell decoder use only outer-
  training contiguous windows. No held-out labels or threshold selection are used.

## Four fixed cells

| Cell | Representation | Post-processing | Purpose |
|---|---|---|---|
| C_rich_control | Existing rich 161 back features | None | Exact retained control |
| G_gravity_geometry | Rich + 30 gravity/dynamic features | None | Test new lower-back information |
| G_calibrated | Same G representation | Outer-training prior correction | Diagnose operating-point error |
| G_state_decoder | Same G representation | Outer-training two-state Viterbi, minimum dwell 2 windows | Test temporal consistency |

All cells use `RandomForestClassifier(n_estimators=300, max_features="sqrt",
min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=4)` with seed `11 + fold`.

## Promotion gate

A candidate must beat the rich control by at least 5 percentage points participant
macro-F1, 10 points standing recall, and 10 points standing F1; lose no more than 2
points sitting recall; win at least 16/22 participants; have no participant harm below
5 points; and have a positive paired participant-bootstrap lower bound. A decoder-only
gain is reported as temporal smoothing, not new sensor information.

If all three candidate cells fail, stop the back-only architecture sweep and use the
qualified same-attachment reference/additional-sensor pilot as the next information
experiment.
