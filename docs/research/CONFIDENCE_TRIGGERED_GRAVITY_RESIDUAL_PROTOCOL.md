# Confidence-Triggered Gravity Residual protocol

Status: locked before CTGR inner selection and CTGR outer evaluation, 2026-09-04.

## Evidence boundary

CTGR is a nine-channel sensor-sufficiency experiment in InclusiveShift-HAR. It is not a replacement result for the six-channel smartphone/wearable claim. It uses six primary IMU channels plus the three recorded Core Motion gravity channels. Participant ID, disability, device, time, location, and released-block position remain forbidden model inputs.

The idea was developed after MPG-RMRP failed and after a deliberately labelled post-hoc scan of already generated source OOF probabilities suggested complementarity between RMRP and the older gravity model. Consequently, participants 1--10 are a reused development cohort for method invention. Nested selection prevents direct outer-label tuning inside this run, but it does not restore statistical independence. Participants 11--20, DAGHAR held-out results, and target predictions are forbidden. Any superiority claim requires a new sealed ability-relevant cohort.

## Architecture

The default prediction is the unchanged flat RMRP probability vector. A posture-only gravity expert is trained on sitting and standing rows within the relevant training partition. Its mobility mass always comes from RMRP; gravity may only redistribute the remaining sitting/standing mass.

The gravity branch includes:

- normalized gravity direction and its device-axis absolute and sorted coordinates;
- user acceleration and gyroscope projections parallel and perpendicular to gravity;
- reconstructed total-acceleration projections;
- gravity-direction angular speed;
- robust distribution, change, slope, and half-window summaries;
- total-acceleration GSP and combinations with the denoised RMRP representation.

CTGR triggers only when the maximum three-class RMRP probability is below a selected confidence threshold. On triggered windows it convexly blends RMRP and the gravity posture expert; all other windows remain exactly RMRP. This design makes the auxiliary sensor a conservative residual rather than a global replacement.

## Candidate map and selection

The machine-readable grid is `configs/experiments/confidence_triggered_gravity_residual_v1.yaml`. It contains the unchanged base plus every combination of four expert views, two Extra Trees posture estimators, five confidence thresholds, and three blend weights: 121 candidates total.

Selection uses seed 11 and the existing five outer participant pairs. Inside each outer training partition, all four inner folds are participant-exclusive. The largest inner mean participant macro-F1 is preferred. Candidates within 0.005 are tied and resolved by bottom-30% participant macro-F1, smaller mean trigger fraction, lower declared complexity, then candidate ID. Selection writes a create-only self-hashed freeze before any CTGR outer evaluation.

The frozen fold map is evaluated at seeds 11, 23, 47, 89, and 131. All controls share the same folds and windows. The complete matrix is aggregated before applying the gate.

## Advancement gate

CTGR must achieve all locked thresholds: mean participant macro-F1 at least 0.852903; bottom-30% at least 0.714257; mobility/sitting/standing recall at least 0.952868/0.689402/0.816910; improvement in at least three seeds and an average of three folds; NLL no more than 0.390054; and multiclass Brier no more than 0.235394.

Passing would justify testing the nine-channel hypothesis on a new cohort. It would not prove a six-channel breakthrough or independent superiority on this reused cohort. Failure is retained and stops promotion.
