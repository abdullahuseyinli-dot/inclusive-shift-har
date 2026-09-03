# FuSE/ReFrame v2 paper addendum

This addendum is for the smartphone/wearable IMU InclusiveShift-HAR project, not the separate DINO/ConvNeXT activity-classification project.

## Proposed paper placement

The v1 locked target result remains the benchmark's confirmatory result. FuSE/ReFrame v2 is a post-analysis source-development extension and must be described as such.

The invented Robust Multiscale Residual Pyramid applies the deterministic Geometric Spectral Pyramid feature map to original, short-span denoised, and long-trend-residual signal views. Strict 5-by-4 participant-exclusive nested cross-validation selected the denoised Extra-Trees candidate in all five outer folds. It achieved mean/lower-decile/worst participant macro-F1 of 0.8379/0.7284/0.5494, compared with 0.8292/0.7373/0.4985 for the GSP predecessor. The mean gain (+0.0087) was below the recorded engineering threshold and was not statistically decisive; the principal positive result is improved weakest-participant and additive-noise performance.

The fixed corruption suite averaged 0.7581 after RMRP versus 0.7507 before it. RMRP improved additive noise and channel dropout, but degraded temporal-gap robustness and did not solve severe drift. These contrary findings belong in the main results rather than supplementary-only reporting.

Semantic Anchor Reconciliation is a separate sparse-labelled personalization branch. On GSP, one labelled sitting and one labelled standing anchor per participant produced 0.8654 mean, 0.7741 lower-decile, and 0.7677 worst-participant macro-F1 on the remaining windows. It is not a zero-shot result and requires independent validation.

The frozen RMRP candidate obtained an equal-domain mean participant macro-F1 of 0.6196 on held-out DAGHAR domains after a disclosed schema probe: MotionSense 0.7985, WISDM 0.6923, and KuHar 0.3679. The external result is therefore mixed and does not support a general superiority claim.

The full evidence narrative, exact comparisons, limitations, hashes, and sources are in [`../docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md`](../docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md).
