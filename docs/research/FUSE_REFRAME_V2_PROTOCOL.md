# FuSE-ReFrame v2 research protocol

This is the prospective implementation companion to
`configs/experiments/fuse_reframe_v2.yaml`. It distinguishes the smartphone
InclusiveShift-HAR project from both the image-based DINO/ConvNeXt HAR project
and the robot thesis repository.

## Scientific boundary

The locked v1 target was opened once on 24 August 2026 and is permanently
consumed. V2 development may use only InclusiveHAR participants 1-10 and
licence-cleared external data. The existing participants 11-20, their signals,
labels, predictions, and aggregate results cannot select architecture,
augmentation, seeds, checkpoints, calibration, fusion weights, or thresholds.

The initial five-epoch fold-1 run under `.audit/v2/smoke/` is explicitly an
engineering smoke test. Its score is not part of model selection or a research
claim because it predates this protocol and used the held-out fold during early
stopping.

## Candidate synthesis

FuSE is a functional hierarchy: first mobility versus stationary, then sitting
versus standing conditional on stationary. Exact InclusiveHAR leaves use normal
likelihood; truly ambiguous external labels use the sum of allowed leaf
probabilities. External native labels remain intact in dataset-specific heads.

ReFrame contains a raw six-channel stream and a structural stream made from
proper-rotation invariants. Its learned gate may use only signal-health
variables: zero/missing fraction, clipping, drift, channel spread, denoising
residual when available, and branch disagreement. It may not use identity,
disability, diagnosis, assistive-device label, time/order, or location.

Training transformations occur in native units before source-fitted
normalization and apply one proper SO(3) rotation to both sensor triads. The
training screen includes participant/class-balanced sampling, small batches,
masked reconstruction, paired-view consistency, participant-tail loss, best
checkpoint, EMA, SWA, and source-validation-bounded SWAD.

## Nested evaluation

Each of five outer folds holds out two complete source participants. Model and
training choices are made only across the four participant-exclusive inner
folds supplied by the source manifest. The selected configuration is refit on
all eight outer-training participants for an epoch/averaging schedule derived
from inner folds; only then is the outer pair evaluated once. Aggregate
uncertainty resamples participants, never windows.

Components advance only when the predeclared mean, lower-tail, class-harm,
fold/seed consistency, and external-proxy gates pass. A gate that merely matches
static fusion is removed. External data that merely improves one source fold or
causes posture harm is removed.

## Evidence levels

1. Unit/synthetic checks establish implementation behavior only.
2. Inner development runs select settings but do not estimate generalization.
3. Outer source folds estimate transfer among source participants.
4. Leave-one-dataset-out external proxies test device/placement robustness.
5. An optional frozen pass on the old target is post-confirmatory historical
   evidence only.
6. A preregistered, access-controlled new target cohort is required for a new
   confirmatory claim.

No architecture name, novelty statement, or desired score overrides these
evidence levels. Failed and quarantined runs remain visible.
