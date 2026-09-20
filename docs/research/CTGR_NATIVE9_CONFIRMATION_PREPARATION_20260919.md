# CTGR native-nine confirmation preparation — 2026-09-19

The independent confirmation was implemented and the preparation phase was executed.
It did not produce a performance result because no qualified fresh native-nine cohort is
present. The governing run is:

`.audit/ctgr_native9_confirmation/ctgr-native9-confirmation-prepare-20260919-002/`

The earlier `-001` preparation record is preserved. Run `-002` binds the final runner
source hash and is the current record.

## Executed checks

| Check | Result |
|---|---|
| Source-finalization artifact inventory | 155 files verified |
| Selection checkpoint replay | 45/45 |
| Final checkpoint replay | 3/3 |
| Final predictor assembly | exact replay passed |
| Frozen component bindings | B6, B9 and expert hashes passed |
| New model fits | 0 |
| Target labels accessed | no |
| Confirmation predictions created | 0 |
| Completion-manifest validation | passed, 3 files checked |

The post-hoc ten-person development differences were used only to fix planning
sensitivity. Their sample standard deviation is 0.042379. For a two-sided alpha of 0.05
and 80% paired power, the frozen +0.020 smallest worthwhile effect requires **38 complete
participants** (attained modelled power 0.808659). Effects of 0.010, 0.015 and 0.025
would require 143, 65 and 25 participants under the same assumption. This assumption may
not transport to a new cohort and does not constitute a result.

## Execution boundary

The only current acquisition blocker is the absence of:

- `data/qualified/ctgr_native9_confirmation_v1/cohort_manifest.json`;
- `data/qualified/ctgr_native9_confirmation_v1/windows_label_blind.npz`.

The retained InclusiveHAR source CSV needed to build source-only secondary controls is
available and hash-matched. Once fresh cohort signals exist, HERA-v1, Random Forest,
XGBoost and the qualified prior-style soft-router probabilities must be generated without
target labels and supplied with the required source-only provenance seal. The prediction
command fails closed if that four-method archive or its checkpoint receipts are absent.

No consumed cohort was substituted. InclusiveHAR P11-P20 remains closed. AICOS-HAR,
MotionSense, HARTH, IMU-HAR-IL and FoG-STAR remain development or external diagnostic
evidence and cannot fill this confirmation slot.

## Implemented evidence controls

The runner separates preparation, prediction and scoring into create-only directories.
Prediction rejects label-like arrays, enforces the fixed native-nine interface and roster,
and seals probabilities before custodian labels can be read. Missing native gravity is
limited to 5% and produces exact B6 bytes for all four core outputs. Scoring verifies
window alignment and the prediction hash, then reports equal-participant macro-F1,
bottom 30%, worst person, accuracy, class recall, NLL, Brier score, all participant harms,
leave-one-person sensitivity and deterministic participant-bootstrap intervals.

The primary gate is `T9 - B9`. `T9 - U9` isolates triggering; `T9 - B6` combines added
sensor information and method. Secondary controls are descriptive and cannot rescue a
failed primary gate.
