# IMU-HAR-IL available-trial coverage amendment

Protocol: `imu-har-il-available-trials-v1`. The immutable declaration is
`configs/protocols/imu_har_il_available_trials_v1.json`, timestamped
2026-09-05T04:21:09.063773+00:00, before inspection of the complete-case model
outcomes or any available-trial model outcomes. Its canonical record hash is
`7df5f88cb0a32d8e1897097474267f59a320ec269a62e219352f919e5b2f4053`.

## Reason and evidence role

The frozen complete-requested-core acquisition retained 19 of 50 provider
participants: 28 were excluded for at least one unavailable requested waist
file and three for invalid requested trials. The dataset permits up to four
voluntary repetitions; requiring all four is consequently a complete-case
selection, not an evaluation of all available participants. This finding came
from the data audit, not from choosing favorable model scores.

The original run and default loader mode remain unchanged and retained. The
new analysis is a separately declared development missing-data sensitivity.
It is not confirmation and cannot be reported as a before/after model gain
over repetition 1 or the complete-case cohort. Repetitions add observations,
not independent participants.

## Fixed selection and signal rules

Request all provider participants, repetitions 1 through 4, and the Walk, Sit,
and Stand Body-WT waist recordings. Retain each available valid physical trial.
Missing files, annotation conflicts, absent sensor schemas, and trials without
finite complete windows are recorded; an invalid trial does not remove valid
peer trials. A participant needs at least one eligible window. Missing classes
remain visible in the fixed-class ceiling and sensitivity reports. The
provider's unambiguous physical-trial activity name remains the annotation
source where its optional redundant Activity_label column is absent; no
sensor-derived or model-derived activity is substituted.

Retained trials use the unchanged units, finite-run boundaries, 0.30 Hz causal
gravity estimate, 60-to-50 Hz conversion, and 128-sample non-overlapping window
grid. The new cohort audit records requested and retained participants and
trials, missing files, invalid files, and people without retained windows.
Tests require retained sensor, gravity, and window arrays to match their
counterparts before unrelated trial removal. FoG remains on the independently
corrected session-grid-v3 preprocessing contract.

## Budget, inference, and reporting

Use the inherited classical/invention and neural controls without search:
seeds 11, 23, and 47; five participant-exclusive outer folds; four inner folds
where required; the same 40-epoch maximum and checkpoint-selection rules.
Run one full source suite and source-only transfer into corrected FoG after a
clean implementation/protocol commit. Record all failures and all exclusions.

The primary summary remains the mean of seed-specific, equally weighted
participant fixed-class macro-F1, with 10,000 participant-cluster bootstrap
draws and the existing support, tail, calibration, and class analyses. This
sensitivity introduces no additional primary superiority test or outcome-based
cohort selection. Complete-case, repetition-1, and available-trial evidence
must occupy distinct comparison-tuple rows.

## Access and preservation

The provider dataset is [IMU-HAR-IL, CSIRO collection 74700](https://data.csiro.au/collection/csiro:74700),
licensed CC BY-NC 4.0. Repository Apache licensing does not override its
noncommercial restriction. Source receipts and hashes accompany the run;
raw third-party files are not committed. Prior runs, failures, declarations,
and checkpoint records are never rewritten by this amendment.
