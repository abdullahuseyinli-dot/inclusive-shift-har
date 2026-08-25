# Project status and evidence gates

This is a superseding status snapshot updated 2026-08-25. Earlier manifests, gates, failures, tags, and status records remain preserved and must not be relabelled.

| Stage | Status | Evidence |
|---|---|---|
| 0 - legacy preservation and audit | Passed for archive integrity and metric reconstruction | `legacy/verification_results.json`; UCI test remains development-consumed |
| 1 - literature and novelty | Passed with narrowed contribution | `docs/LITERATURE_MATRIX.md`; no verified novelty conflict; no "first" claim |
| 2 - repository and provenance | Passed | package, lockfile, manifests, CLI, tests, CI, licence metadata |
| 3 - InclusiveHAR audit and ontology | Integrity passed; trial-boundary limitation retained | `results/data_audit/inclusivehar_v4.audit.json`; timestamps/trials absent |
| 4 - released-block protocol | Conditionally passed and locked | split `ccb6c3d...`; participant-exclusive/raw-row-disjoint, not trial-safe |
| 4A - corrected UCI reproduction | Complete source-development evidence | 75/75 first-attempt CUDA cells; official UCI test and InclusiveHAR remained unopened |
| 4B - disabled within-group | Complete post-confirmatory descriptive evidence | 75/75 CUDA cells; successful aggregate `within_group_v1_1`; prior failed aggregate preserved |
| 4C - zero-shot primary | Complete locked confirmatory evidence | 20 configurations x 5 seeds; participant inference; opening 1 consumed |
| 4D - few-person inclusion | Complete post-confirmatory secondary evidence | 1,200/1,200 CUDA cells; 64 model-k summaries; 80 paired comparisons; failed aggregate attempts 1-3 preserved |
| 4E - cross-source and SSL/foundation | Not evaluated | exact-label classification blocked beyond sitting; adapted transfer not implemented; licence/interface/checkpoint gates unresolved |
| 4F - sensor reliability | Complete post-confirmatory secondary evidence | 120 stressed cells; compact ERM/CORAL; severity calibration limitation retained |
| 5-8 - models, training, and statistics | Primary and feasible secondary suites complete | locked suite, CCIL/BPD adaptations, signal sensitivities, and 320-profile CUDA efficiency aggregate |
| 9 - final evidence gate | Passed before opening | 190 tests, lint, format, mypy, manifests, splits, artifacts, clean tree, protocol tag |
| 9 - target opening | Consumed exactly once and complete | opening receipt, 100 result sidecars/arrays, immutable index, locked statistics |
| 10 - release | Private remote created; licence-evidence patch candidate in progress | failed candidates/runs `benchmark-v0.1.0`/`32799146947`, `benchmark-v0.1.1`/`32801378375`, `benchmark-v0.1.2`/`32802922698`, and `benchmark-v0.1.3`/`32811935288` preserved; intended successor is pending as `benchmark-v0.1.4`; no DOI |

## Confirmatory outcome

- Target cohort: participants 11-20; 10 participants, 807 functional-core windows.
- Frozen lineup: 20 model/ablation configurations, seeds 11, 23, 47, 89, and 131.
- Numerically highest locked-primary mean participant macro-F1: compact DANN, 0.6808438, 95% participant-bootstrap CI [0.5391426, 0.8093204]. Compact CORAL obtained 0.6807882, only 0.0000556 lower; they are effectively tied at the supported precision.
- MoRe-HAR full: mean 0.6353323, worst 0.2583943, lower decile 0.2659867.
- Strongest mean reference: compact DANN; strongest worst-participant reference: legacy joint CNN/BiLSTM (0.2942991); strongest lower decile: compact CORAL (0.3669902).
- Preregistered MoRe-HAR decision: not supported. Mean improvement and joint lower-tail improvement were both false; the source non-inferiority gate had passed.
- Candidate minus compact-DANN participant mean: -0.0455116. Holm-adjusted exact sign-flip p = 0.7207031; Holm-adjusted Wilcoxon p = 0.7558594.

Target confidence intervals are wide and participant tails are low across every model. A post-confirmatory DANN source-minus-target estimate was 0.1196 with interval [-0.0371, 0.2831], but its one-seed source-CV and five-seed final-target regimes differ. The evidence documents difficult and heterogeneous target generalization; it does not provide a controlled causal decomposition, a direct measure of physical ability, or a fairness/clinical claim.

## Completed post-confirmatory evidence

- Few-person v1.1: 1,200 validated scenario results. Descriptive numerical mean leaders were DANN at k=0 (0.6808), compact ERM at k=1 (0.6744), MoRe-HAR backbone at k=2 (0.7060), and MoRe-HAR content at k=4 (0.7593); no between-model significance tests were run. All models' k=4 means exceeded k=0, but only DeepConvLSTM and the static matched baseline were monotone across mean, worst-participant, and lower-decile endpoints. None of 80 within-model paired comparisons survived Holm correction (minimum adjusted p = 0.15625).
- Disabled within-group: compact ERM 0.6067 mean, MoRe-HAR backbone 0.5626, DeepConvLSTM 0.4001; 75/75 CUDA cells.
- Sensor stress: 120 result cells. Drift and missing accelerometer-Z had the largest target mean reductions for both compact ERM and CORAL. Source n=2 versus target n=10 and uncalibrated severities preclude strong interaction claims.
- Qualified CCIL/BPD adaptations: CCIL 0.6896 mean but slightly lower ERM-relative tails and adjusted p=0.2109; BPD 0.5710. Both are descriptive, post-confirmatory, and non-faithful adaptations.
- CUDA efficiency: 80 checkpoints, 320 profiles, and 641 contention samples. Timing status is `valid_with_declared_allowlisted_ambient_system_processes`; measurements are device-resident forward passes, not end-to-end application latency. FP16 autocast was slower than FP32 for every model at both batch sizes, while usually reducing allocated VRAM; 60 recurrent profiles used the cuDNN-disabled CUDA fallback.
- Raw/total acceleration v1.1 and SI-unit sensitivity completed. The latter demonstrated exact normalized-tensor equivalence rather than an accuracy result.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## Active immutable anchors

- Coursework ZIP SHA-256: `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94`
- Literature matrix file SHA-256: `B119EE0C05191777E6222F16E8A53D0D39F9583BC34107F714AE3FDF8AB0E92C`
- Literature registry file SHA-256: `5B424C56B322EA7AAA49E2B647D70D91EA728F44E9B56B9EEE0717B4EEED82D7`
- Dataset manifest file SHA-256: `52de5370682f13fbd9a4e9affe805b4f5ee0f28901d137743884b77471b24d29`
- Split manifest embedded SHA-256: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
- Source-window manifest embedded SHA-256: `1ad1ee3accaae5f2f93bb91ac0afa5ce583134ce1d882c3f08323b09026fa522`
- Target seal ID: `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`
- Frozen artifact-set SHA-256: `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`
- Confirmatory analysis plan SHA-256: `7b99dd5894109370397867a1ca141758c0a30b4efb2fb3d76aba23ab1ad62177`
- Opening receipt record SHA-256: `5704f65efd416e9cd16d6ed24c2735c1d52b7fcaafddb113e481498797cd182a`
- Locked target index record SHA-256: `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`
- Participant statistics record SHA-256: `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`
- Publication report record SHA-256: `3afe0ceee9f97025d1adc5f59ab3528b512a3212385ef5344028850e9cb39c66`
- Disabled within-group aggregate record SHA-256: `fbe81492df167e136c3ad14bccfdd94a4c82c12497b48c2042a50e25dc521180`
- Sensor-stress aggregate record SHA-256: `abe1aee172e9b680e2aca16848c84bf6788491d005964b3d18e597098a06df90`
- Corrected UCI grouped-reproduction record SHA-256: `c2e0191259bd2f51cb58a91e51ebc27acfa84f47e1bf654dc04cf9ecaaafa1f6`
- Few-person v1.1 statistics record SHA-256: `d14c4a071ee2460a2182fcab56ab6454be6d4cc3c6cc391445e56a368b776058`
- CUDA efficiency aggregate record SHA-256: `7c0fa71edcd0a368090df0513d6a418a35a6989b35f828734febd135f11530bb`
- Qualified CCIL/BPD aggregate record SHA-256: `c7b27e2a6d5ddf94efcd2c2064cb84aecfc70dfe3d4f38539660c3479128c180`
- Raw/total acceleration aggregate record SHA-256: `817405fd04b78a7a30ac985d84ad5009800edb78e48a69c8d246c0e4003a815a`
- SI-unit equivalence record SHA-256: `ca42d11b9b846307afc1cd2dfeda9e113c81fbdcc8bd60f1499eb22f1fb7bcaa`
- Frozen training code commit: `b4dc38fb9d5a0c17003221b61156ebc065395170`
- One-time target evidence commit: `f0d11b2`
- Active protocol tag: `protocol-v1.2.0`

## Preserved limitations and deviations

The immutable `benchmark-v0.1.0` tag points to commit
`f0a589a0bb18f60862c80f7e56eac2a33027c358`. Its first GitHub Actions run,
`32799146947`, failed before test execution on both operating systems because
the workflow combined `UV_FROZEN=1` with the mutually exclusive `uv sync
--locked` option. The immutable `benchmark-v0.1.1` tag points to commit
`4ac9b7b5471c945389348a73b5c59387c2aae069`; run `32801378375` also stopped
before tests because hosted `uv` rejects an empty `UV_FROZEN` value rather than
treating it as false. Neither tag was moved or deleted. Patch candidate
`benchmark-v0.1.2` used the explicit boolish value `false`; its Ubuntu
synthetic-validation job passed tests, lint, format, and types, but Windows
checkout converted LF evidence/config files to CRLF. Six exact-hash tests
failed and the release-security job did not run in run `32802922698`. Patch candidate
`benchmark-v0.1.3` additionally disabled Git end-of-line conversion for every
tracked path through `.gitattributes`, disabled Windows `core.autocrlf` before
checkout, and retained the explicit boolish sync override. Run `32811935288`
passed both operating-system matrices, then failed the release-security bundle
because Linux XGBoost 3.2.0 installed `nvidia-nccl-cu12==2.31.2`, whose PyPI
metadata reports `LicenseRef-NVIDIA-Proprietary`. Candidate
`benchmark-v0.1.4` records the exact Linux x86-64 wheel, dependency marker,
metadata sidecar, embedded BSD 3-Clause text, and archived NVIDIA pages whose
`nccl_2312` path renders a 2.29.2 label, and
limits the exception to a locally installed transitive runtime that is not
vendored into repository files, release assets, or container images. All four
failed tags and runs remain visible and immutable.

The split audit validates participant exclusivity, label-block containment, and raw-row disjointness. Because InclusiveHAR releases no trial/session/timestamp identifiers, its unconditional hidden-join contamination bound is 100%; a conditional three-repetition assumption gives 240/3,042 (7.8895%), but that assumption is unverified. This benchmark must not be called trial-safe or unqualified leakage-safe.

Recurrent cuDNN execution failed on this Windows/CUDA stack with process exit `0xc0000409`. Failure artifacts are preserved. Successful recurrent experiments used CUDA tensors with cuDNN disabled, not CPU neural fallback. Classical scikit-learn estimators retained their native CPU policy; XGBoost training used CUDA.

The target opening cannot be repeated. All few-person inclusion, corruption, efficiency, signal-sensitivity, predecessor-adaptation, or other follow-up is post-confirmatory and cannot alter the locked zero-shot claim. The historical 190-test pre-opening gate remains a historical fact; a separate final-release gate must validate the eventual release commit rather than rewriting that record.
