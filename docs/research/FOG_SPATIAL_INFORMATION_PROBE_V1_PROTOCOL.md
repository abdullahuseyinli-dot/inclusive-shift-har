# Corrected FoG spatial-information and availability probe v1

Status: frozen before any outcome-producing fit. This protocol implements
`.audit/fog_decision_team_review_20260907-001/NEXT_EXPERIMENT_SPECIFICATION.md`.
It is exploratory development on the consumed FoG-STAR cohort.

## Scientific question

Test whether a qualified fixed left-ankle stream improves activity recognition in a
back-supported system. The existing corrected FoG models use only the back IMU even
though the pinned source contains synchronized ankle channels. This study tests a
measurement and estimator package. Concatenation is not claimed as an invention.

The label-independent coverage audit disqualified a universal-availability design.
It reproduced all 1,939 original back candidates and found 1,817 that have a containing
finite ankle run and an exact independently processed 128-timestamp grid match. Of
1,213 scored rows, 1,154 qualify and 59 require fallback. Preserve the full roster and
endpoint with fold-specific B0 back probabilities for every unavailable candidate.
Do not impute, interpolate, phase-correct, delete rows, change tolerance, or select a
sensor after seeing outcomes.

## Fixed cells and controls

Run five cells across the original five participant folds, exactly 25 maximum fit
attempts. B0 uses all eligible training rows and back80 features. Bv uses back80 but
only ankle-available eligible training rows. Lv substitutes left-ankle80 on those same
training/evaluation rows. BLv concatenates back80 and left-ankle80. BBv concatenates
back80 with an exact duplicate back80 block. Bv, Lv, BLv, and BBv use the identical
availability mask, training IDs and participant-first weights; they copy B0 outputs
on unavailable evaluation candidates.

Bv controls for training-support loss. BBv diagnoses feature dimension and forest
subsampling, but duplicated predictors are not a pure information-theoretic control.
Lv remains a two-sensor availability system because its fallback needs back data. A
negative ankle80 result cannot exclude static ankle posture information because these
six channels remove gravity.

The exact RF has 500 trees, sqrt feature sampling, leaf size two, bootstrap, no class
weight, random state 11 plus outer fold, four fit workers and one prediction worker.
Every cell recomputes mean-one weights `1/(m_i*n_ic)` on its actual training rows.
Use one BLAS/OpenMP thread. One controller fits cells sequentially. No tuning or inner
selection occurs.

## Data boundary and preprocessing

Use only pinned FoG-STAR SHA256
`888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477`.
This FoG roster legitimately includes `fogstar:011` through `fogstar:020`; the sealed
P11-P20 prohibition applies to the distinct InclusiveHAR target and remains obeyed.

Freeze `ankleL`. Never inspect right ankle, wrist, clinical affected side, task ID or
outcomes to change that choice. Participant/session IDs and timestamps establish
physical boundaries and alignment only. Labels enter after the candidate grid and
availability mask are frozen. No identity, time, task, clinical field or label becomes
a model feature.

For each sensor, segment only on participant/session, finite signal and observable
timestamp discontinuity. Convert acceleration from g to m/s2 and gyroscope from
degree/s to rad/s first. Then derive causal 0.30 Hz gravity continuously inside that
sensor's physical segment, subtract it from acceleration, jointly polyphase-resample
the six signal and three gravity channels from 60 to 50 Hz once, and take
128-sample non-overlapping windows. Do not reset
at labels or windows. The processing remains offline and is not a zero-lookahead
streaming claim. Extract the existing 80 deterministic summaries independently from
linear acceleration xyz and gyroscope xyz. Do not add gravity, denoising, GSP,
cross-sensor geometry, learned normalization or another representation.

## Ordering, replay and stopping

Before fitting, bind raw data, coverage audit, mask, source/config/protocol, original
candidate/scoring arrays, partitions, feature schemas, environments and reference
artifacts by hash. All five restricted training partitions must retain all classes and
exclude evaluation people.

Fit B0 first. It must reproduce the archived F1/back feature matrix and all-candidate
probabilities byte-for-byte against aligned GSP A, factorial F1, and decision-rule D0
references. Any mismatch stops before the remaining twenty fits, and the attempt stays
recorded. Then execute Bv, Lv, BLv and BBv without retries. Every failed attempt counts.

Maximum complete compute is 3,600 seconds, including materialization, fits, scoring,
validation and shutdown. Do not extend the cap after partial results. No extra seed,
sensor, representation, external dataset or follow-on is automatic.

## Metrics and decisions

Primary reporting retains all 22 people and 1,213 scored rows: fixed-three-class
macro-F1 within person then equal-person mean. Also report accuracy, NLL, Brier,
class support/precision/recall/F1, participant confusion matrices, present-class
sensitivity, worst, bottom seven, quartiles, per-person changes, rescues, harms, ties,
available/fallback strata, leave-person and leave-fold diagnostics, and paired 10,000
participant bootstrap draws with seed 1729. The available-only endpoint is descriptive.

Each required advancement contrast must gain at least 0.015 mean F1, win at least
14/22 participants, lose at most 0.01 bottom-seven, 0.03 between worst minima, 0.01
mobility recall and 0.02 for each posture recall, and retain positive mean gain after
omitting every participant and every complete outer fold. These are statistical
resource gates, not physical safety bounds. Report maximum individual harm separately.

Lv must pass against Bv, B0 and archived F3. BLv must pass against Bv, Lv, BBv, B0 and
F3. Against F3 the required mean is at least 59.726500%; this is a threshold, not a
forecast. If neither passes, close this finite package. If Lv passes alone, retain the
ankle-when-aligned/back-fallback result. If BLv passes, propose a separately authorized
replication. Never launch that replication automatically.

## Evidence

Use create-only output. Retain all 25 attempt records/checkpoints, raw/cache/reference
receipts, availability and partitions, per-sensor preprocessing/feature hashes, branch
and final probabilities, fallback provenance, hard decisions, all reports and
comparisons, timings, failures, independent checkpoint replay, completion manifest and
worker shutdown. No InclusiveHAR target access, publication work, push, tag or release.
