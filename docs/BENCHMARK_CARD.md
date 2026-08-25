# InclusiveShift-HAR benchmark card

## Summary

InclusiveShift-HAR is an auditable participant-exclusive benchmark for measuring ability-associated population shift in smartphone inertial activity recognition. Models are trained and tuned using 10 InclusiveHAR v4 participants released with `disabled=0`, then evaluated once on 10 disjoint participants released with `disabled=1`. The metadata labels are used only to define cohorts; they are never inference features. The released binary label is not a direct measurement of physical ability and must not be interpreted as one.

The primary result compares participant-level recognition performance between the released source and target cohorts. The cohort flag is not a measured physical-ability variable, the groups are unpaired, and the dataset contains no clinical outcomes.

## Dataset and licence

- Dataset: InclusiveHAR v4, DOI `10.17632/r78dn3f6nc.4`.
- Dataset licence: CC BY 4.0 as stated by the versioned provider.
- Participants: 20 total; 10 source and 10 target.
- Collection: iPhone 14 Pro, vertically worn in a waist pouch, nominally 50 Hz.
- Repository policy: raw data remain immutable outside Git; Apache-2.0 applies only to repository-authored code.

The full provenance, file sizes, hashes, retrieval record, expected schema, and citation are in `manifests/datasets/inclusivehar_v4.json` and `docs/data/INCLUSIVEHAR_V4.md`.

## Primary task

- Input: `[batch, 128, 6]`.
- Channels: `motionUserAccelerationX/Y/Z` in g and `motionRotationRateX/Y/Z` in rad/s.
- Nominal duration: 2.56 seconds at 50 Hz.
- Class order: `mobility`, `sitting`, `standing`.
- Functional mapping: released `Walking` maps to `mobility`; for wheelchair users this released label denotes manual propulsion and must not be described as ordinary gait.
- Executed source-development stride: 128 after participant partitioning.
- Executed source-validation/target stride: 128.

Ramp ascent/descent are not stairs, and jogging has no UCI-HAR v1 counterpart. These labels are excluded from the functional-core primary result rather than silently remapped.

## Partition and leakage controls

Participants are assigned before window generation. Source train, source validation, and target participants are disjoint. No raw row belongs to more than one partition; windows never cross participant, released activity-label, or released contiguous-block boundaries. Normalization uses source training participants only. Calibration uses source validation only. The target cohort never participates in feature scaling, class/mapping construction, tuning, early stopping, checkpoint selection, calibration, or threshold selection.

InclusiveHAR v4 provides no timestamps or trial/session/recording identifiers. Hidden repetition boundaries within each released participant-activity block are therefore unrecoverable. The protocol is not trial-safe: the unconditional hidden-join risk bound is 100%. A documented but unverified three-repetition assumption yields a conditional upper bound of 240/3,042 windows (7.8895%). Users must retain this limitation when reusing the benchmark.

## Evaluation

Primary endpoint: mean participant-level macro-F1 on target participants, with each participant equally weighted after averaging over five seeds.

Additional endpoints include balanced accuracy, per-class recall, participant distribution, worst participant, lower decile, NLL, multiclass Brier score, secondary ECE, risk-coverage/AURC, parameter count, model size, latency, MACs/FLOPs, and peak VRAM where measured. Inference treats participants, never windows, as statistical units. Completed efficiency measurements are secondary, device-resident forward-pass profiles rather than end-to-end application latency; MAC/FLOP counts cover only declared supported operators.

The confirmatory analysis uses a 10,000-resample participant bootstrap, paired exact sign-flip tests, paired Wilcoxon tests, standardized/rank-biserial effects, and Holm correction for the predeclared MoRe-HAR comparison family.

## Locked outcome

The one-time target opening evaluated 20 frozen configurations over seeds 11, 23, 47, 89, and 131. Compact DANN achieved the numerically highest mean participant macro-F1 (0.6808; 95% participant-bootstrap CI [0.5391, 0.8093]). Compact CORAL was only 0.0000556 lower and is effectively tied at the supported precision. MoRe-HAR full achieved 0.6353 and did not improve the mean or both lower-tail endpoints against all eligible references. Its preregistered hypothesis is not supported.

The full, hash-validated table is `results/confirmatory/zero_shot_v1/model_summary_v1.md`.

## Secondary evidence status

All items below are post-confirmatory and cannot change the locked outcome:

- Few-person v1.1 completed 1,200/1,200 CUDA cells. The descriptive numerical mean leaders at k=0,1,2,4 obtained 0.6808, 0.6744, 0.7060, and 0.7593, respectively; no between-model significance tests were run. Every model's k=4 mean exceeded k=0, but only DeepConvLSTM and the static matched baseline were monotone across all three participant endpoints. None of 80 within-model paired comparisons survived Holm correction (minimum adjusted p = 0.15625).
- Disabled-cohort within-group evaluation completed 75/75 CUDA cells; compact ERM obtained mean/worst/lower-decile macro-F1 0.6067/0.2537/0.3000.
- Sensor stress completed 120 result cells. Drift and missing accelerometer-Z produced the largest target mean losses, but source n=2 versus target n=10 and uncalibrated severities prevent strong interaction claims.
- Qualified CCIL and BPD adaptations completed, but they are neither official-faithful reproductions nor locked-primary comparators.
- CUDA efficiency completed 320 profiles over 80 checkpoints with 641 contention samples. Timing validity is qualified by declared allowlisted ambient WDDM processes. FP16 autocast was slower for every model at both measured batch sizes, although it usually reduced allocated VRAM; 60 recurrent profiles used CUDA with cuDNN disabled.
- Raw/total-acceleration and SI-unit sensitivities completed; neither changes the primary signal-definition decision.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## Intended use

- Comparing source-only HAR approaches under a fixed ability-associated shift.
- Auditing participant-level failure distributions and calibration.
- Studying whether representative inclusion data improves lower-tail performance in a separate post-confirmatory protocol.
- Reproducing the evidence-gated software workflow.

## Out-of-scope use

- Medical diagnosis, rehabilitation decisions, safety-critical control, or assistive-device control.
- Claims about all disabled people, all assistive devices, or all sensing placements.
- Causal attribution of errors to disability.
- Treating the functional mobility mapping as exact locomotion semantics.
- Reporting window-level confidence intervals as though windows were independent.
- Reopening or retuning against the consumed target cohort.

## Ethical considerations

The cohort is small and heterogeneous, and released group labels compress diverse bodies, devices, contexts, and activity realizations. Aggregate comparisons can obscure individual harms; worst-participant and distributional results are therefore mandatory. Subgroup results by device/disability type are exploratory only. Avoid deficit framing: performance failures indicate a limitation of the trained system and data coverage, not of a participant's activity realization.

## Maintenance

The active split hash is `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`; the target seal is `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`. Any changed data version, ontology, split, or preprocessing interface defines a different benchmark version and requires new manifests and claims.
