# MPG-RMRP v1 source-development result

Status: complete, prospectively evaluated, failed advancement gate, not confirmatory (2026-09-04).

## Scope

This result belongs to the smartphone/wearable InclusiveShift-HAR repository. It is not the DINO/ConvNeXt image-classification project and it is not the robot thesis repository. The experiment used only InclusiveHAR source participants 1--10 under the locked participant-exclusive nested protocol. Participants 11--20 and previously opened DAGHAR evaluation domains were not accessed.

The implementation and protocol were committed at `76e2aa81252c0945eb6ff203046bb17155d96dc0` and tagged `mpg-rmrp-protocol-v1` before inner selection or outer evaluation. The machine-readable committed result is `results/development/mpg_rmrp_v1_summary.json`; detailed create-only evidence remains under `.audit/v3/mpg-rmrp`.

## Result

All five fixed seeds completed from the same frozen fold-specific candidate map.

| Method | Mean participant macro-F1 | Bottom 30% | Worst participant | Mobility recall | Sitting recall | Standing recall | NLL | Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Flat RMRP control | **0.8395** | **0.6994** | **0.5477** | 0.9760 | **0.7162** | **0.8318** | **0.3689** | **0.2251** |
| Hierarchical RMRP control | 0.8359 | 0.6909 | 0.5402 | 0.9752 | 0.7120 | 0.8266 | 0.3714 | 0.2266 |
| Hierarchical GSP control | 0.8159 | 0.6525 | 0.4772 | **0.9822** | 0.7103 | 0.7536 | 0.4118 | 0.2565 |
| RMRP mobility + RIST posture | 0.7754 | 0.6459 | 0.5322 | 0.9783 | 0.6556 | 0.7167 | 0.4299 | 0.2752 |
| MPG-RMRP | **0.5837** | 0.5175 | 0.4993 | 0.9760 | 0.3650 | 0.5296 | 0.5462 | 0.3756 |

MPG-RMRP lost 0.2558 mean participant macro-F1 relative to flat RMRP. It won on zero of five seeds and zero of five outer folds. Only the mobility-recall gate passed; the mean, lower-tail, both posture recalls, calibration, positive-seed, positive-fold, and stretch-worst gates failed.

| Seed | Flat RMRP | MPG-RMRP | Difference |
|---:|---:|---:|---:|
| 11 | 0.8379 | 0.5878 | -0.2501 |
| 23 | 0.8412 | 0.5719 | -0.2693 |
| 47 | 0.8420 | 0.5942 | -0.2478 |
| 89 | 0.8395 | 0.5771 | -0.2624 |
| 131 | 0.8371 | 0.5875 | -0.2496 |

The failure is too large and consistent to attribute to seed variance. Flat RMRP's across-seed standard deviation was 0.0021; MPG-RMRP's was 0.0090.

## Mechanistic assessment

The mobility component was unchanged and retained 0.9760 recall. The loss is isolated to the posture branch: sitting recall fell by 0.3513 and standing recall by 0.3021. The proposed timestamp representation subtracts a rolling spatial geometric median and then keeps rotation-invariant magnitudes and cross-sensor relations. That combination deliberately removes absolute/local orientation and slow offsets. Those are nuisance factors for many activities, but this dataset's sitting-versus-standing decision evidently relies on them. A microstate-only replacement therefore discards more posture signal than its transition graph recovers.

The selected inner scores (approximately 0.592--0.654) were already much lower than RMRP, and all outer folds confirmed the warning. This also rules out the explanation that one unlucky fold or one unsuitable codebook size caused the outcome.

The architecture is retained as a falsified research result. It must not be described as an improvement or silently replaced under the v1 name.

## Next controlled work

The predeclared, no-retuning ablation runner tests occupancy only, lag-one transitions, no detrending, hard assignment, removal of half-window direction, and all pure controls. Its purpose is explanation, not rescue or candidate reselection.

The next clean six-channel invention must preserve the rich denoised GSP/RMRP posture representation and permit microstate information only as an optional residual or gated auxiliary feature. The nine-channel gravity lane remains a separate sensor-sufficiency experiment. After the primary result was sealed, a post-hoc diagnostic blend of old gravity and seed-11 RMRP probabilities reached about 0.8538 mean at one weight but did not improve the weakest participant. That scan is not valid evidence and no weight from it may be presented as preselected; it only motivates a separately locked, inner-selected gravity/posture study.

Because zero-shot posture remains unresolved, the Active Semantic-Gauge Sentinel personalization lane is now eligible. It must report labels requested, remove query windows from evaluation, and compare no adaptation, random querying, and the existing fixed 1+1 SAR method. It is labelled personalization, never zero-shot.

No further experiment on these same ten participants can supply independent confirmation. A genuinely new sealed ability-relevant cohort remains necessary for a publication-level superiority claim.

## Evidence anchors

| Artifact | Record SHA-256 | File SHA-256 |
|---|---|---|
| Inner selection freeze | `3ade58db04320a484e0c3eba6f8614b6644dbb9dcc0ebfffa0db9ff6bc497e1d` | `fcf838bc35cd6e667d0f7499a26f19ffa680faaef9fde3ff5726ab36fbcbfb17` |
| Five-seed aggregate | `65b68cbf2e08187d99f0df65012b677bd3c86f7d55952b4b9c102627cf35c7e3` | `378089f21965e6e9df538c0043c77ec59c694bf824cb941d66117079d5f3daf5` |

All selection, seed aggregate, fold record, model, and prediction hashes were replay-validated after completion.
