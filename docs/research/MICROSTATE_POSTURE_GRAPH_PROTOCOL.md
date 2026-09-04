# Microstate Posture Graph over RMRP protocol

Status: prospectively locked source-development protocol, 2026-09-04.

This protocol belongs to the smartphone/wearable IMU **InclusiveShift-HAR** repository. It does not apply to the separate DINO/ConvNeXt image project or to the robot thesis repository. The machine-readable experiment definition is `configs/experiments/microstate_posture_graph_nested_v1.yaml`; the self-hashed lock is `results/protocol/microstate_posture_graph_v1.json`.

## Scientific boundary

InclusiveHAR participants 11--20 and the previously opened DAGHAR evaluation domains are consumed evidence. They must not select this method, a candidate, seed, threshold, ablation, or stopping rule. Development is limited to the 725 source windows from participants 1--10, using the existing five outer participant pairs and four participant-exclusive inner folds. The participant is the inference unit. Outer labels are unavailable to candidate selection, and all trainable preprocessing, codebooks, and estimators are fitted inside the relevant training partition.

The resulting evidence can establish a source-development advance only. A new zero-shot or confirmatory claim requires a genuinely new, independently sealed, ability-relevant cohort.

## Failure hypothesis

The prior RMRP family selected its denoised GSP view in all five outer folds. Residual-only features ranked last in all five folds and residual-containing concatenations were never selected. RMRP's source mean improved modestly, while participant 10 retained a sitting/standing error pattern. The next intervention therefore isolates the posture decision rather than adding another global residual concatenation.

## Primary architecture: MPG-RMRP-6

The primary input remains exactly six channels at 50 Hz and 128 samples: three-axis user acceleration and three-axis rotation rate.

The mobility head produces `q = P(mobility)` from the denoised GSP representation and a participant/class-weighted Extra Trees model with 500 trees, square-root feature sampling, and minimum leaf size one.

The posture head is fitted only on sitting and standing training windows:

1. At each timestamp, subtract a rolling spatial geometric median independently from the acceleration and gyroscope triads. The spatial median uses 24 fixed Weiszfeld updates and is equivariant to a shared proper rotation of both triads. Candidate spans are 0.20 and 0.50 seconds.
2. Form the six-vector containing detrended acceleration norm, detrended gyroscope norm, their derivative norms per second, normalized cross-sensor dot product, and normalized cross-sensor cross-product norm.
3. Fit a median/IQR scaler and participant-balanced K-means codebook on stationary timestamps from the training partition only. Candidate codebook sizes are 4, 6, and 8. K-means uses 20 initializations and a fixed candidate-derived seed.
4. Set the soft-assignment temperature to the training-only median positive within-cluster squared distance.
5. Summarize each window with state occupancies; normalized soft transition matrices at lags 1, 2, 4, 8, and 16; state persistence; hard-path mean and maximum dwell fractions; first-half minus second-half occupancy; six channel-coverage fractions; and fully-observed-timestamp coverage.
6. Fit one of three posture estimators: Extra Trees leaf one, Extra Trees leaf three, or L2 logistic regression with `C=1`. This produces `r = P(sitting | stationary)`.

The final probabilities are fixed:

- `P(mobility) = q`
- `P(sitting) = (1-q)r`
- `P(standing) = (1-q)(1-r)`

The novelty claim is restricted to the particular train-only robust microstate trajectory and multiscale posture-transition representation. Hierarchical classification, K-means codebooks, Extra Trees, and logistic regression are not claimed as individually novel.

## Pure controls

All controls are fixed before MPG outer evaluation and are reported on the same windows:

- flat RMRP: the selected denoised GSP three-class Extra Trees model;
- hierarchical RMRP: denoised GSP for both mobility and posture heads;
- hierarchical GSP: original GSP for both heads;
- RMRP mobility plus RIST posture: denoised GSP mobility with a RIST classifier trained only on stationary windows.

This separates gains from hierarchy alone, the selected denoising intervention, and a high-capacity posture-specific time-series control.

## Nested selection and fixed evaluation

There are exactly 18 MPG candidates: two detrending spans, three codebook sizes, and three posture estimators. Seed 11 is used for inner selection. For each outer fold, the candidate with the largest inner mean participant macro-F1 is preferred. Candidates within 0.005 of the largest mean are treated as tied and resolved by bottom-30% participant macro-F1, then smaller codebook, simpler estimator, declared complexity rank, and candidate ID.

Selection is a separate create-only operation. It writes a self-hashed freeze before any outer-fold model is fitted or evaluated. The five outer-fold candidate choices in that freeze are then reused without reselection at seeds 11, 23, 47, 89, and 131.

Required execution order:

1. validate code, configuration, manifests, splits, and artifacts;
2. commit and tag this protocol and implementation;
3. run inner selection and write the selection freeze;
4. evaluate the frozen selection at all five seeds;
5. aggregate the complete seed matrix and apply the advancement gate;
6. run frozen ablations and secondary lanes only under their declared gates;
7. retain every failed, partial, and completed audit directory.

## Advancement gate

Against the current RMRP source-development result, the across-seed MPG summary must satisfy all of the following:

- mean participant macro-F1 at least 0.852903;
- bottom-30% participant macro-F1 at least 0.714257;
- mobility recall at least 0.952868;
- sitting recall at least 0.689402;
- standing recall at least 0.816910;
- positive MPG-minus-flat change in at least three of five folds and three of five seeds;
- negative log-likelihood no more than 0.390054;
- multiclass Brier score no more than 0.235394.

Matching or improving the RIST worst-participant score of 0.616809 is a stretch gate. A result that improves only a lower-decile or a single participant is not a general zero-shot advancement.

## Frozen ablations

After the candidate map is frozen, ablations remove one component without retuning: occupancy only; occupancy plus lag-one transitions; full multiscale graph; no detrending; hard assignment; no half-window direction; RIST posture; original GSP mobility; and the same denoised RMRP representation for both heads. These are explanatory source-development analyses, not new selection opportunities.

## Separate secondary lanes

`MPG-RMRP-G` may use actual gravity channels only in the posture branch. It must derive the normalized gravity direction, acceleration parallel/perpendicular components, equivalent gyroscope components, and gravity-tilt statistics. It is a nine-channel sensor-sufficiency study and cannot replace the six-channel primary result.

`MPG-RMRP-M` may add an explicit provenance validity mask. Missingness must never be inferred from numeric zeros. Every mask-aware comparator receives the same mask. A new corruption suite must use new seeds and placements and include axis/modality dropout, contiguous gaps, stuck-at values, saturation, bias and linear drift, scale drift, Gaussian noise, and constrained rotations. Without timestamps, only synthetic within-window gaps may be claimed.

The Active Semantic-Gauge Sentinel is allowed only as clearly labelled personalization if zero-shot posture remains unresolved. It maintains normal/swap hypotheses, chooses a stationary query by expected information, requests a label only when a fixed Bayesian threshold is unmet, and removes queried windows from evaluation. It must be compared with no adaptation, random queries, and the existing 1+1 SAR method. It cannot be described as zero-shot.

## Evidence interpretation

Passing the source gate supports continued external validation; it does not retroactively make reused cohorts independent. Failure of the gate is retained as a useful falsification and stops promotion of the clean architecture. No desired score, model name, or publication deadline changes this rule.
