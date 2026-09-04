# HERA-CTGR retrospective source-development protocol

Status: locked for implementation and test before any outcome-producing HERA-CTGR run.

## Purpose and evidence boundary

This experiment measures whether the already-frozen HERA-CTGR v1 mechanisms are useful on the
same InclusiveHAR participants 1--10 that motivated them. It is a fully participant-nested,
retrospective hypothesis-generation analysis. It is neither independent validation nor evidence
for a state-of-the-art claim. InclusiveHAR participants 11--20 and every DAGHAR target remain
unloaded. A genuinely new development cohort and a separately sealed confirmation cohort remain
required.

## Fixed inputs and nesting

The five CTGR seeds, five outer participant folds, four inner folds, source windows, frozen CTGR
candidate rankings, and preserved outer RMRP/selected-expert/CTGR probabilities are reused exactly.
For each outer fold, every learned repair, calibration, and responder decision is fit using only the
eight outer-training participants. Base and gravity-expert predictions for those participants are
regenerated through their four participant-exclusive inner folds. Evaluation labels are passed only
to reporting after all evaluation probabilities and routing states are complete.

The BROAD path is the previously specified CAGE global trust-region blend, refit inside each outer
training partition. The first three entries in each already-frozen CTGR inner ranking are averaged
with equal probability weights for the selection-uncertainty ablation; HERA outcomes cannot change
their membership or weights.

## Predeclared methods

- `base_uncalibrated`: unchanged RMRP.
- `frozen_ctgr`: exact preserved selected CTGR.
- `global_trust_blend`: the fixed CAGE broad path.
- `ctgr_aggregate_calibrated`: frozen CTGR plus training-only conditional posture calibration.
- `ctgr_physics_veto`: frozen CTGR with the training-only gravity--gyro consistency veto.
- `ctgr_candidate_ensemble_top3`: equal mean of the first three frozen ranked CTGR candidates.
- `hera_ctgr_strict`: top-three mean, conditional posture calibration, then physical veto; it uses
  no evaluation-participant aggregation.
- `hera_ctgr_context_mean`: OFF/PULSE/BROAD participant routing using jackknife mean utility,
  support rejection, and the physical veto.
- `hera_ctgr_context_safe`: the same router using mean minus one jackknife standard deviation.
- `hera_ctgr_full`: safe OFF/PULSE/BROAD routing followed by calibration selected from
  leave-one-training-participant-controller predictions. Calibration is applied only to PULSE or
  BROAD rows; OFF and physically vetoed rows are restored exactly to RMRP.
- `matched_random_state_control`: a deterministic within-fold permutation of the full method's
  participant states. It is a diagnostic, not a model candidate.
- `label_informed_three_way_oracle_diagnostic`: hindsight best of base, frozen CTGR, and BROAD per
  participant. It uses evaluation labels and is only an unattainable headroom diagnostic.

The candidate ensemble belongs to the strict lane and is not silently substituted for the exact
frozen PULSE state in the context lanes. This preserves the frozen OFF/PULSE/BROAD definition and
makes each mechanism independently falsifiable.

## Selection rules

Physics uses the 0.99 outer-training quantile of the normalized gravity--gyro residual and requires
0.98 valid adjacent-sample coverage. Conditional posture calibration searches the frozen 3-by-3
grid, admits candidates within 0.005 of the best training participant macro-F1, then selects lower
NLL, proximity to identity, stronger bottom-30% performance, and deterministic numeric order.

The responder has 15 unlabelled participant aggregates, ridge penalty 25, minimum predicted utility
0.005 relative to PULSE, maximum standardized support distance 3, and PULSE fallback. Participant
macro-F1 differences form training utilities. The evaluation participant's labels, identifier value,
disability, and assistive-device metadata are unavailable to the controller.

## Interpretation

Strict-window and unlabelled participant-context results are reported separately. Because all
evaluation windows for a participant contribute to the latter's unlabelled signature, the context
lane is transductive. Seeds quantify estimator randomness, not ten new independent samples.
Advancement and breakthrough gates remain those in `hera_ctgr_v1.yaml`; failing gates stays visible.
