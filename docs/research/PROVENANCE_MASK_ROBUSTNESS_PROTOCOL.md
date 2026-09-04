# Explicit-provenance mask robustness protocol

Status: locked before the first new-suite corruption run, 2026-09-04.

This secondary lane tests whether an explicit sensor-validity mask can recover performance under missing or invalid measurements. It replays the frozen seed-11 RMRP outer models and cannot change the clean model or select a new architecture.

Numeric zero is never interpreted as missing. Each injected case produces a Boolean `[window,time,channel]` provenance mask. Both comparators receive the same corrupted values and mask: the frozen numeric-input comparator ignores the mask, while the proposed deterministic wrapper linearly interpolates invalid spans from valid samples with edge extension. A fully absent channel receives only its outer-training-partition median. All imputation statistics are therefore training-only.

The suite uses a new seed and placement schedule and includes clean replay, one/two-axis dropout, modality dropout, contiguous gaps, stuck-at intervals, saturation, bias and linear drift, scale drift, Gaussian noise, and constrained shared rotations. Dropout, gaps, stuck-at values, and saturation carry explicit invalidity. Drift, scale, noise, and rotations are degraded but observed measurements, so their mask remains valid. Without real timestamps, gap claims are limited to synthetic within-window gaps.

Clean probabilities must replay the frozen result exactly. Corruption outcomes are diagnostic and prohibited from selecting or retraining RMRP. A robustness gain cannot be described as a clean accuracy advance or independent confirmation.
