# CAGE-HAR v1 prospective development protocol

Status: implementation complete; real evaluation blocked pending a genuinely new development
cohort. Declared 2026-09-04 before any CAGE-HAR human-cohort result.

## Evidence boundary

CAGE-HAR is a new nine-channel research lane. It does not modify or supersede the locked CTGR
artifacts. InclusiveHAR participants 1--10 are exhausted for invention selection; participants
11--20 and the previously opened DAGHAR domains are also forbidden. The executable bundle gate
rejects those dataset and participant identifiers and accepts only participant-exclusive
out-of-fold predictions associated with a self-hashed new-cohort contract.

Synthetic smoke results validate software behavior only. They cannot be reported as HAR
performance, robustness evidence, model selection, or support for a state-of-the-art claim.

## Research hypothesis

CTGR improved proper probability scores while harming the hard decisions of two participants.
Its gravity branch therefore appears informative but its confidence threshold is not a reliable
estimate of whether intervention will help. CAGE-HAR tests whether predicting the counterfactual
advantage of each expert, and constraining every intervention, can retain CTGR's posture rescues
while reducing routed harm and negative responders.

## Architecture

### Posture-preserving uncertain gravity gauge

The physical context retains both kinds of evidence:

- shared-rotation-invariant vertical and horizontal acceleration/gyroscope projections;
- absolute central gravity coordinates that may encode posture;
- native-gravity coverage, norm dispersion, direction dispersion and angular speed;
- disagreement between instantaneous and central-gauge projections.

The method does not detrend away slow/DC gravity information and does not impose full rotational
invariance. Missing gravity produces finite fallback features but a zero reliability score.

### Sensor-health context and expert bank

Explicit validity, gap length, time since observation, flatline, saturation, scaled drift and
high-frequency symptoms are exposed per modality. Only direct provenance failures such as
missingness and saturation affect the hard reliability gate within one window. Potentially valid
stationary or slowly changing signals remain router features rather than automatic failures.

The future OOF bundle may contain a full nine-channel expert, gravity/posture expert,
six-channel expert, acceleration-only expert and gyroscope-only expert. Each expert declares
whether it is posture-only. Entirely absent modalities must use a dedicated observed-evidence
expert rather than being represented as confidently reconstructed measurements.

### Decision repair

Within each outer training partition only, the fixed grid evaluates temperatures
`[0.80, 1.00, 1.25]` and antisymmetric sitting/standing offsets `[-0.15, 0.00, 0.15]`.
Candidates within 0.005 participant macro-F1 of the training ceiling are resolved by lower NLL,
closeness to the identity transformation, lower-tail score, and deterministic numeric order.
The held participant's labels are never used.

### Counterfactual-advantage router

For each expert and training window, the target is:

`log P_expert(true class) - log P_base(true class)`

Router inputs contain only prediction confidence, entropy, margins, sitting/standing evidence,
base/expert disagreement, expert reliability and label-free physical context. Participant ID and
protected metadata are not inputs. A participant-balanced ridge model is fitted repeatedly while
leaving out each training participant. The mean prediction minus one jackknife standard deviation
is a conservative stability bound, not a conformal coverage guarantee.

Baseline-correct/expert-wrong events receive twice the weight. Windows from the lowest 30% of
training participants under the base receive twice the weight. This is a fixed participant-tail
and harm-weighted surrogate, not evidence of fairness.

### Trust-region intervention

An expert is eligible only when reliability is at least 0.50 and its predicted lower advantage
bound exceeds 0.010. Mixing is capped at 0.75 and scaled by predicted advantage. The resulting
posterior must remain within KL divergence 0.05 of the base. Posture-only experts preserve the
base mobility probability exactly and may redistribute only sitting/standing mass. With no safe
expert, the output is bitwise the base probability.

### Optional labelled semantic gauge

A label-free suspicion score combines stationary probability, base/gravity disagreement and
gauge reliability. Only a suspected participant supplies one selected label. The observation
updates an identity-versus-swap posterior and softly blends the sitting/standing columns while
preserving mobility mass. Queried windows must be excluded from evaluation, and this lane must be
reported separately from zero-shot CAGE-HAR.

## Required ablations

The real new-development runner produces:

1. uncalibrated base;
2. decision-repaired base;
3. global trust-region blend;
4. confidence-triggered trust-region blend;
5. disagreement-triggered trust-region blend;
6. full counterfactual advantage-gated CAGE-HAR.

Additional expert-generation ablations required when the new raw cohort exists are raw gravity
versus engineered gauge, absolute versus partial-invariant versus dual gauge, mask versus inferred
health versus both, full versus single-modality experts, and clean versus physically corrupted
training.

The machine-readable comparison inventory is
`configs/experiments/cage_har_baseline_registry_v1.yaml`. It keeps six-channel, nine-channel and
modern-representation tracks separate and leaves unavailable or licence-pending adapters visible
rather than silently omitting them.

## Physical-fault suite

The implemented secondary fault families are constant bias, random-walk drift, gain error,
colored noise, contiguous gaps, axis loss, accelerometer/gyroscope/gravity loss, stuck-at,
saturation, orientation jumps, signed axis permutations, timestamp jitter, quantization and a
combined bias/noise/gap condition. Every operation remains within an existing window and returns
an explicit validity mask. Severity is currently dimensionless and uncalibrated; real-fault
episodes are required before a deployment-robustness claim.

The fixed uncalibrated engineering matrix is recorded in
`configs/experiments/cage_har_corruptions_v1.yaml`.

## Advancement gates

The configuration freezes these proposed development thresholds:

- mean participant macro-F1 improvement at least 0.020 over the repaired base;
- nonnegative bottom-30% improvement;
- intervention precision at least 0.80;
- harmful changed-decision fraction no more than 0.15;
- health branch clean regression no more than 0.005;
- corruption-curve improvement at least 0.015.

The engineering score target is 0.875 and stretch target 0.884, motivated by the frozen CTGR
router-oracle diagnostic. These are targets, not promised outcomes or acceptance evidence.

## New data required

The cohort manifest requires native user acceleration, rotation rate and gravity; channel units
and coordinate frames; true sampling intervals; monotonic timestamp, participant, site, session,
trial and repetition keys; device, OS/app, placement/orientation and sensor-provenance fields; and
at least two sessions per participant. It also enforces partition-before-windowing, boundary-safe
windows, raw-evidence preservation and exclusion of identity/protected metadata from model input.

After selection on a genuinely new development cohort, code, configuration, checkpoints,
candidate map and analysis must be frozen before one independent confirmatory opening. Six- and
nine-channel leaderboards, zero-shot, online adaptation and labelled personalization remain
separate.
