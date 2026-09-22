# CTGR and HERA-CTGR model card

## Model identity

- **CTGR:** Confidence-Triggered Gravity Residual.
- **HERA-CTGR v1:** Hierarchical Evidence-Regularized Adaptive CTGR.
- **Current decision:** retain CTGR as the robust source-development control;
  report strict HERA-v1 as the highest source point estimate. Neither is an
  independently confirmed successor to the locked six-channel benchmark.
- **Software version:** 0.1.7a0 publication candidate.
- **Evidence date:** 2026-09-22.

These are composite classification methods: feature extraction, standard Extra
Trees learners, expert routing and calibration. They build on the project's own
SpectralShape, GSP and RMRP feature-development sequence. The contribution
includes representation design, denoising/view selection and selective posture
correction; Extra Trees and the signal filters are established components. The
separate HARTH rich-feature Random Forest models are not frozen CTGR/HERA
evaluations.

## Development lineage

| Stage | Project contribution | Seed-11 source participant macro-F1 |
|---|---|---:|
| SpectralShape | Initial deterministic signal-feature pipeline | 78.884% |
| GSP | Geometric and spectral representation | 82.916% |
| RMRP | Raw/denoised/residual candidate family; denoised GSP selected | 83.790% |
| CTGR | Native-gravity posture expert and confidence-triggered combination | 86.474% |
| Strict HERA-v1 | CTGR candidate marginalization, calibration and consistency veto | 86.755% |

These stages share the 725 source windows and five participant-exclusive outer
folds. The first three use six channels; CTGR/HERA add three native gravity
channels. The tested budgeted RIST external control reached 77.195% under this
source evaluation, with a fixed rather than nested selection recipe. RMRP's
6.595-point mean advantage over that control accompanied three participant harms
and a lower worst-participant score. The [source-lineage audit](research/SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md)
and [research report](RESEARCH_REPORT.md#421-seed-11-development-history) give
accuracy, controls, participant tradeoffs and provenance.

RMRP is therefore an earlier project contribution used as the control for later
extensions. The five-seed comparison below measures those later extensions; it
does not describe the project's total improvement from its initial methods.

## Intended task

The models classify non-overlapping 128-sample inertial windows into the ordered
functional-core classes:

1. `mobility`
2. `sitting`
3. `standing`

The source interface is nominally 50 Hz and uses nine channels:

| Channels | Unit in the frozen source models | Role |
|---|---|---|
| User/linear acceleration X/Y/Z | g | Dynamic motion |
| Rotation rate X/Y/Z | rad/s | Angular motion |
| Native gravity X/Y/Z | g | Device orientation and posture residual |

The source data originate from Core Motion. Axis names exist, but the dataset
does not publish a complete device-to-body transform. An external acquisition
must declare units and coordinate conventions. The confirmation adapter accepts
SI acceleration/gravity and converts them to g while preserving rad/s gyro; this
conversion does not resolve an unknown axis or polarity mapping.

## Architecture

### RMRP base

The six-channel Robust Multiscale Residual Pyramid (RMRP) provides the default
three-class probabilities. The candidate family evaluated raw, denoised,
trend-residual, and concatenated geometric/spectral views. All five original
outer folds selected denoised GSP with participant/class-weighted Extra Trees;
residual concatenation is not part of that retained feature path. The exact
configuration is versioned in the source protocols and experiment files.

### CTGR residual

CTGR trains a posture expert on sitting and standing examples within each
participant-exclusive training partition. Its gravity-aware views include
normalized gravity direction, axis-absolute and sorted coordinates, acceleration
and gyro projections relative to gravity, reconstructed total acceleration,
gravity-direction change, and robust temporal/spectral summaries.

The RMRP mobility probability is preserved. The posture expert may redistribute
only the remaining sitting/standing mass. It is consulted when the maximum base
probability is below a fold-selected threshold, and its output is blended using
a fold-selected fixed weight. Selection is nested inside source participants;
outer-fold labels never enter fitting or selection.

### Strict HERA-v1

The strict window-only HERA lane adds frozen top-three CTGR candidate
marginalization, posture calibration, and a gravity–gyroscope consistency veto.
It does not use evaluation-participant aggregates. The physical veto falls back
to the base path when gravity behavior lies outside training support.

HERA's separate participant-context controller and label-informed oracle are not
the retained zero-query model. The context controller failed its advancement
gates, and the oracle uses evaluation labels and is diagnostic only.

## Training and evaluation data

The canonical comparison uses InclusiveHAR v4 source participants P1-P10:

- 725 functional-core windows;
- five participant-exclusive outer folds;
- four inner folds within each outer training partition;
- seeds 11, 23, 47, 89, and 131;
- mean participant macro-F1 as the primary endpoint.

These ten people were reused during method development. Nested folds protect
each executed comparison from direct outer-label fitting, but they do not restore
independence after repeated cohort-level hypothesis generation. P11-P20 were not
reopened for HERA/CTGR development.

## Performance

| Model | Window accuracy | Mean participant macro-F1 | Bottom 30% | Worst participant | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|
| RMRP: selected denoised GSP | 84.579% | 83.953% | 69.941% | 54.767% | 0.36889 | 0.22512 |
| CTGR | 86.979% | 86.540% | **73.990%** | 57.512% | 0.35426 | 0.21165 |
| Strict HERA-v1 | **87.228%** | **86.849%** | 73.933% | **58.028%** | **0.34038** | **0.20303** |

Every column is averaged over the same five seeds. Accuracy pools windows within
each seed; the primary macro-F1 gives each participant equal weight. The
[metric audit](research/REPORTED_METRICS_AUDIT_20260920.md) recomputed both from
the preserved source predictions and reproduced the canonical macro-F1 values.

CTGR improved mean participant macro-F1 by 2.586 points over RMRP and passed all
nine predeclared source-development advancement checks. It also added three
sensor channels, so this is a sensor-sufficiency result rather than a fixed-input
architecture-only effect.

Strict HERA improved CTGR by 0.309 points. Its descriptive participant-bootstrap
interval was [-0.095, +0.713] points, and its gain and bottom-tail gates failed.
The point estimate therefore does not establish superiority over CTGR.

## External evidence

On 38 AICOS development participants with all three classes, frozen CTGR reached
68.97% mean participant macro-F1. The unconditional U9 routing candidate reached
68.93% and harmed sitting recall while improving standing recall. AICOS uses
derived rather than native gravity, and its logger axis/polarity provenance is
incomplete. That lane is a conditional transfer diagnostic.

HARTH findings are placement diagnostics, not evaluations of these frozen models.
They show that right-thigh acceleration can make sitting versus standing highly
observable under a matched protocol, while lower-back data were much weaker.

## Fallback and interface behavior

- Normal inference requires the declared nine-channel order and finite values.
- The fresh-cohort confirmation runner records all unit conversion explicitly.
- If native gravity is missing within the confirmation protocol's allowed
  fraction, all four core outputs use the exact frozen B6 fallback for the
  affected windows.
- Unsupported units, excess missing gravity, probability misalignment, duplicate
  identities, or missing control provenance fail closed.
- Participant identifiers and sensitive metadata are grouping/audit fields only.

The confirmation code is implemented and tested, but no qualified fresh cohort
has been scored. A new `prepare` receipt must bind the current runner before a
future confirmation execution.

## Appropriate use

- Reproducing source-development comparisons and ablations.
- Studying whether an independently measured gravity channel improves posture
  classification under participant-exclusive validation.
- Evaluating participant harms, calibration, and conservative fallback behavior.
- Generating frozen, label-blind probabilities for a prospectively qualified
  native-nine cohort.

## Out-of-scope use

- Medical diagnosis, rehabilitation decisions, or safety-critical control.
- Claims about all disabled people, devices, placements, or phone operating systems.
- Treating wheelchair propulsion as ordinary walking.
- Tuning on InclusiveHAR P11-P20 or reopening its consumed target result.
- Selecting external unit, axis, sign, or threshold transforms using target F1.
- Calling HERA/CTGR state of the art or independently confirmed.

## Limitations and risks

- Only ten repeatedly used people support the canonical source comparison.
- InclusiveHAR has no recoverable trial, session, or timestamp boundaries; the
  released-block protocol is participant-exclusive but not trial-safe.
- Gravity makes the input budget different from the original six-channel target
  benchmark and may not be exposed consistently on another logger.
- Sitting and standing errors can trade against one another and differ sharply
  by participant; mean performance alone is insufficient.
- External transfer may combine different devices, placements, native versus
  derived gravity, and coordinate conventions.

## Canonical artifacts

- [Source-development lineage audit](research/SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md)
- [GSP/RMRP development and robustness report](research/FUSE_REFRAME_V2_RESEARCH_REPORT.md)
- [CTGR protocol](research/CONFIDENCE_TRIGGERED_GRAVITY_RESIDUAL_PROTOCOL.md)
- [CTGR result summary](../results/development/max_rnd_secondary_v1_summary.json)
- [HERA-v1 protocol](research/HERA_CTGR_V1_PROTOCOL.md)
- [HERA-v1 result summary](../results/development/hera_ctgr_retrospective_v1_summary.json)
- [Canonical method status](research/CANONICAL_HERA_CTGR_METHOD_STATUS.md)
- [Fixed routing validation](research/CTGR_ROUTING_VALIDATION_20260919.md)
- [Native-nine confirmation protocol](research/CTGR_NATIVE9_CONFIRMATION_V1_PROTOCOL.md)
