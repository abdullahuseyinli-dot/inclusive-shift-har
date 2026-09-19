# HERA compact-evidence integration v1

The compact gravity-relative branch improved denoised GSP by 0.764 percentage points in a bounded source replay. HERA-v1 strict is frozen at its matched seed-11 result. This integration tests complementarity by averaging 75% frozen HERA-v1 probabilities with 25% compact-branch probabilities. The weight is fixed before reading the integration outcome; no HERA routing, calibration, feature, or estimator parameter is retuned.

The input arrays must match exactly by labels, participant IDs, and window IDs. The result reports fixed-three-class participant macro-F1, per-class recall, calibration, every participant difference, and a 10,000-resample participant bootstrap with seed 1729. This is reused source-development evidence. A zero new HERA fit count is intentional: the integration isolates whether the learned compact evidence is complementary to the frozen HERA output before considering a larger retraining.
