# HERA-CTGR v2 retrospective source-development protocol

Status: locked after the HERA-CTGR v2 implementation tag and before any v2 outcome-producing run.

## Purpose and boundary

This is one fully nested retrospective screen on the already-exhausted InclusiveHAR source
participants 1--10. It may test implementation behavior and generate hypotheses; it cannot validate
independent generalization. Participants 11--20 and DAGHAR are not loadable inputs. HERA v1 and CTGR
predictions are hash-pinned and must replay exactly.

## Participant nesting

For each of five fixed seeds and five outer participant folds, the two held participants remain
unavailable to every zero-shot fit and selection. Base, three frozen CTGR candidates, and the new
dual-frame posture expert are regenerated across the four inner participant folds. Temperature,
posture offset, dual confidence/blend parameters, rescue/harm heads, and sentinel thresholds use
only those outer-training OOF predictions.

The rescue and harm heads train only on sitting/standing disagreements and are themselves evaluated
through whole-participant exclusion during threshold selection. Evaluation labels become available
only after every zero-shot probability and route is fixed. The one-query lane is evaluated
separately; its query is selected without a label, and the subsequently read label is used only for
that participant's OFF/NORMAL/INVERTED selection. The query window is removed from both the
personalized result and its matched core comparator.

## Fixed methods

The table must contain frozen RMRP, CTGR and strict HERA v1; equal and stability-weighted top-three
CTGR; temperature-only and decision-separated cores; the global dual-frame candidate; logits-only,
physics-gated and full support-rejected rescue/harm sentinels; matched one-query rows; and
label-informed participant/window oracles. Oracle rows are unattainable diagnostics.

The released data have no authentic time, session, trial or bout order. Consequently, the
implemented bout accumulator is not evaluated and cannot be assigned a score.

## Frozen selection

All numeric grids are fixed in `hera_ctgr_v2_retrospective_v1.yaml`. Scalar temperature minimizes
training OOF NLL. The posture offset maximizes training participant macro-F1 and bottom-tail score.
The dual candidate is selected inside a 0.005 mean-F1 tolerance by bottom-tail, harm, rescue,
probability score and coverage. A sentinel threshold is eligible only if training OOF intervention
precision is at least 0.80 and harmful-change fraction at most 0.15. If none qualifies, it performs
no intervention and returns the decision-separated core exactly.

## Outcome gates

Advancement requires +0.010 mean participant macro-F1 over strict HERA v1, +0.015 bottom-30% over
CTGR, precision at least 0.80, harm fraction at most 0.15, zero mobility-membership changes, tail and
posture non-inferiority, no NLL/Brier regression, and a participant interval above zero. Breakthrough
requires +0.020 over CTGR and positive familywise mean and bottom-tail interval bounds. Failed and
unevaluable gates remain explicit.

No retrospective outcome may change the frozen method or support superiority, confirmation,
target-validation or state-of-the-art language.
