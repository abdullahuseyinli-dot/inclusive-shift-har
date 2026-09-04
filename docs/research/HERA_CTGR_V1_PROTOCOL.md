# HERA-CTGR v1 protocol

Status: implementation protocol declared 2026-09-04 after CTGR and CAGE-HAR source results and
before any HERA-CTGR outcome-producing run.

## Evidence boundary

HERA-CTGR is a new method. It does not alter the frozen CTGR or CAGE-HAR implementations, tags,
predictions, or reports. Its design is explicitly motivated by post-run inspection of InclusiveHAR
participants 1--10, including the systematic CTGR harm for participants 4 and 8 and CAGE-HAR's
failed window-level router. Those participants are therefore exhausted hypothesis-generation
evidence. Participants 11--20 and consumed DAGHAR domains may not be loaded.

Any later evaluation on participants 1--10 is retrospective development only. A new development
cohort and a separately sealed ability-relevant confirmation cohort remain mandatory for a
superiority or state-of-the-art claim.

## Method

Hierarchical Evidence-Regularized Adaptive CTGR (HERA-CTGR) retains three fixed prediction paths:

1. **OFF:** unchanged RMRP;
2. **PULSE:** frozen CTGR;
3. **BROAD:** the fixed global CAGE trust-region blend.

The participant/session controller defaults to PULSE. It can select OFF or BROAD only when a
participant-jackknife lower bound predicts at least the configured macro-F1 advantage and the
unlabelled responder signature lies inside training support. Participant identifiers, disability,
assistive-device metadata, timestamps, site, trial and location are grouping or audit fields, never
model features.

### Gravity--gyroscope consistency

For a stationary inertial gravity vector represented in the rotating device frame, HERA evaluates

`d(g_hat)/dt + omega cross g_hat`.

The residual, its normalized form, direction speed and angular-velocity-implied speed are summarized
per window. A 0.99 training-only quantile and minimum valid-pair fraction define a deterministic
physical veto. An untrusted gravity intervention falls back exactly to RMRP. Absolute gravity
coordinates and slow posture information remain available; full rotation invariance and aggressive
detrending are prohibited because MPG-RMRP falsified that representation for sitting versus standing.

### Unlabelled responder signature

Fifteen robust participant/session aggregates summarize CTGR coverage, broad-route hard changes,
base confidence, stationary mass, base/expert disagreement, Jensen--Shannon divergence, posture-logit
residuals, gravity reliability, kinematic consistency and trusted-window fraction. The controller is
a strongly regularized participant-level ridge utility model. It predicts OFF-minus-PULSE and
BROAD-minus-PULSE utility from participant-exclusive OOF data; evaluation labels are unavailable.

### Selection uncertainty and calibration

The selection-uncertainty ablation equally averages the first three frozen CTGR candidates in each
fold's already-recorded eligible ranking. It never reweights candidates from HERA outcomes.

Aggregate calibration operates after a probability path is composed. It adjusts only conditional
sitting/standing log odds and preserves mobility mass exactly. For full HERA, the calibrator is fit
on leave-one-participant-controller OOF predictions inside the outer training partition.

## Required lanes

- **Strict independent-window:** calibration, physical veto and frozen candidate averaging, without
  participant context.
- **Unlabelled participant context:** all windows belonging to an evaluation participant supply a
  label-free signature. This is transductive context, not strict zero-shot DG.
- **One-query personalization:** remains a separate future lane and may never be merged into either
  zero-shot table.

The released InclusiveHAR data do not expose recoverable session/trial boundaries for a clean
calibration-session versus evaluation-session experiment. Retrospective evaluation can therefore
test participant-context feasibility only.

## Advancement gates

The full new-development method must improve frozen CTGR mean participant macro-F1 by at least 0.010
and bottom-30% performance by at least 0.015, achieve at least 0.80 decisive intervention precision,
keep the harmful changed-decision fraction at or below 0.15, preserve worst-participant and posture
recalls within -0.010, and not worsen NLL or Brier score. A breakthrough requires at least +0.020
mean improvement with participant-level uncertainty bounds above zero. These are targets, not
promised results.

## Claim policy

The historical 0.7514 few-person score, six-channel RMRP, and nine-channel HERA/CTGR results remain
different protocols. Seeds do not increase the participant sample size. Strict window, transductive
unlabelled context and labelled personalization results must appear in separate tables.

