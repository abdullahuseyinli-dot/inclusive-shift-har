# HERA-CTGR v2 protocol

Status: implementation declared after the frozen HERA-CTGR v1 retrospective result and before any
v2 outcome-producing run.

## Evidence boundary

HERA-CTGR v2 is a new research candidate. It does not revise HERA-CTGR v1, CTGR, CAGE-HAR,
MPG-RMRP, their tags, or their results. Its design was motivated by repeated inspection of
InclusiveHAR participants 1--10. Those participants are exhausted hypothesis-generation evidence.
Participants 11--20 and consumed DAGHAR targets remain unavailable. Any v2 result on participants
1--10 is retrospective development only.

## Motivation from the frozen result

Strict HERA v1 reached 0.86849 versus 0.86540 CTGR, but improved only three of five seeds and failed
its advancement and breakthrough gates. Equal top-three candidate averaging was the most consistent
component. Conditional calibration improved NLL and Brier but changed hard decisions, the physics
veto had no effect on frozen CTGR in clean data, and the participant responder controller performed
no better than its random diagnostic. The three-state label-informed oracle reached only 0.88618,
leaving almost no margin above the 0.88540 breakthrough target. V2 therefore separates decisions
from calibration and adds a complementary expert rather than merely relaxing the old router.

## Fixed architecture

### Evidence-marginalized core

The first three candidates from each already-frozen CTGR inner ranking remain the candidate set.
For every window, each candidate is weighted by its Jensen--Shannon distance from their equal
consensus. A fixed temperature and positive weight floor prevent an outlying candidate from taking
over or being deleted. Labels and outcomes never determine weights.

Scalar temperature scaling is selected on participant-exclusive training predictions by NLL and
cannot change the class rank. A separately selected sitting/standing log-odds offset may change only
those two classes. Mobility probability is preserved exactly, and any proposed posture correction
that would cross the mobility boundary is rejected.

### Dual-frame posture expert

The new expert retains raw device-frame gravity coordinates and slow/DC evidence while also forming
a deterministic gravity-aligned tangent frame. It summarizes vertical and horizontal user
acceleration, rotation rate, reconstructed total acceleration, gravity-direction speed, and the
gravity--gyroscope kinematic residual. This explicitly avoids MPG-RMRP's falsified removal of
absolute posture information. Its Extra Trees posture estimator and fixed confidence/blend grid are
selected using only participant-exclusive outer-training OOF predictions.

### Rescue--harm sentinel

Two low-capacity participant-balanced logistic heads estimate the probabilities that replacing the
decision-separated core with the dual-frame candidate will rescue or harm a hard decision. Training
targets are constructed only from participant-exclusive OOF predictions. At evaluation, labels,
participant identity values, ability metadata, site, location, timestamps, and assistive-device
metadata are unavailable.

Whole-participant jackknifing supplies a lower rescue bound and upper harm bound. Intervention is
allowed only for a sitting/standing disagreement inside training feature support, with trusted
gravity evidence, sufficient lower rescue probability, bounded upper harm probability, and positive
penalized net benefit. Every other case falls back exactly to the decision-separated ensemble core.

### Separate personalization and temporal lanes

The one-query lane chooses a maximum-disagreement posture window without inspecting its label, then
uses that one label to choose OFF, NORMAL, or sitting/standing-INVERTED state. The query window is
excluded from evaluation. This is labelled personalization and cannot enter the zero-shot table.

Bout-level evidence accumulation is implemented but may be evaluated only on a new dataset with
authentic ordered session/trial/bout provenance. The released InclusiveHAR data cannot validate it.

## Required evaluation order

1. Validate the implementation using synthetic/unit tests only.
2. Commit and tag the v2 implementation.
3. Lock a retrospective protocol referencing the exact implementation commit and hashes.
4. Run the single create-only participants 1--10 retrospective screen with all ablations and
   diagnostics.
5. Preserve failures and report the outcome without changing v2 parameters.
6. Regardless of the retrospective score, use a genuinely new development cohort before selecting
   a publication candidate and a separately sealed cohort before any confirmation claim.

## Advancement and claims

Advancement requires at least +0.010 over strict HERA v1, at least +0.015 bottom-30% over CTGR,
intervention precision of at least 0.80, harmful-change fraction no greater than 0.15, exactly zero
mobility flips from posture repair, tail and posture non-inferiority, and no NLL/Brier regression. A
breakthrough additionally requires at least +0.020 over CTGR and participant-level uncertainty
bounds above zero. These are targets, not promised outcomes.

No result on the reused ten participants is independent, confirmatory, target-validated, or evidence
of state of the art.
