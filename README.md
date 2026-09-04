# InclusiveShift-HAR: An Auditable Participant-Exclusive Benchmark for Ability-Associated Population Shift in Smartphone Activity Recognition

InclusiveShift-HAR studies how smartphone inertial HAR systems trained on conventional participant populations transfer to people whose activities may be physically realized differently, including users of assistive devices. The repository separates legacy coursework, source-only development, and a single locked target evaluation with immutable lineage.

The original working title said "leakage-safe." InclusiveHAR v4 does not release timestamps or trial identifiers, so hidden joins inside released participant-activity blocks cannot be ruled out. The benchmark is participant-exclusive and raw-row-disjoint under the released schema, but it is not trial-safe. That limitation is part of the result, not a footnote.

## Main result

The one-time zero-shot target evaluation is complete. It used 10 held-out target participants, 807 non-overlapping released-block windows, three functional-core classes, 20 predeclared model/ablation configurations, and five frozen seeds per configuration.

| Result | Mean participant macro-F1 | Worst participant | Lower decile |
|---|---:|---:|---:|
| Compact DANN | 0.6808 | 0.2699 | 0.3559 |
| Compact CORAL | 0.6808 | 0.2741 | 0.3670 |
| MoRe-HAR full | 0.6353 | 0.2584 | 0.2660 |

Compact DANN had the numerically highest locked-primary target mean, but its lead over compact CORAL was only 0.0000556; the two are effectively tied at the precision supported by these data. The legacy joint CNN/BiLSTM had the highest worst-participant value (0.2943), and CORAL had the highest lower-decile value (0.3670). MoRe-HAR did not improve either the mean or the required lower-tail endpoints against the strongest eligible baselines. Its preregistered hypothesis is therefore **not supported**. This negative outcome is retained, and the auditable benchmark is the primary contribution.

The complete table, participant values, calibration metrics, AURC, per-class recall, comparisons, and self-hashed report are in [`results/confirmatory/zero_shot_v1/`](results/confirmatory/zero_shot_v1/). The 95% interval for compact DANN is [0.5391, 0.8093], reflecting substantial participant uncertainty. A later descriptive comparison estimated a DANN source-minus-target gap of 0.1196 with interval [-0.0371, 0.2831], but the source estimate is one-seed grouped cross-validation while the target estimate averages five final-fit seeds. The two cohorts are unpaired and were evaluated under different training regimes, so that gap is reported only as a cohort-specific descriptive comparison.

## FuSE/ReFrame v2 source-development extension

The post-analysis v2 research suite invented a deterministic Robust Multiscale Residual Pyramid (RMRP) for the same smartphone IMU task. Strict 5-by-4 participant-exclusive nested source cross-validation improved mean/worst-participant macro-F1 from 0.8292/0.4985 with the Geometric Spectral Pyramid to 0.8379/0.5494 with RMRP. The lower decile declined from 0.7373 to 0.7284, the paired mean gain was not statistically decisive, and the +0.015 recorded engineering threshold was not met. This is a useful robustness result, not a confirmed breakthrough or a replacement for the locked v1 target result.

RMRP improved the 18-corruption mean from 0.7507 to 0.7581 and was materially better under additive noise and two-channel dropout, while performing worse under temporal gaps and essentially tying under severe drift. A separate sparse-labelled semantic-anchor branch reached 0.8654 with one sitting and one standing anchor per participant, but that result is labelled personalization, not zero-shot. A frozen RMRP candidate obtained an equal-domain mean of 0.6196 on held-out DAGHAR domains after a disclosed schema probe, with large variation from MotionSense (0.7985) to KuHar (0.3679).

The number 0.7514 belongs to a v1 few-person experiment in which four target-group participants entered training; it is not the predecessor of the v2 source-only 0.8379 result. The full method, comparisons, external evaluation, failures, hashes, and publication assessment are in [`docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md`](docs/research/FUSE_REFRAME_V2_RESEARCH_REPORT.md), with a self-hashed compact ledger in [`results/development/fuse_reframe_v2/research_summary_v1.json`](results/development/fuse_reframe_v2/research_summary_v1.json).

A prospectively locked follow-on tested the nine-channel Confidence-Triggered Gravity Residual (CTGR). Across five fixed source-development seeds it obtained 0.8654 mean participant macro-F1 versus 0.8395 for matched six-channel RMRP, improved all five seeds, and passed all nine predeclared advancement checks. This is a strong gravity-sensor-sufficiency signal, not a new target result: it reuses source participants 1--10, adds three gravity channels, and was inspired by already-seen source errors. Active labelled personalization and explicit-mask robustness were also completed. See [`docs/research/MAX_RND_SECONDARY_RESULTS.md`](docs/research/MAX_RND_SECONDARY_RESULTS.md) and the self-hashed [`results/development/max_rnd_secondary_v1_summary.json`](results/development/max_rnd_secondary_v1_summary.json).

## Evidence classes

| Evidence class | Meaning | Status |
|---|---|---|
| Legacy coursework | Saved UCI-HAR notebook outputs with random window validation and repeated official-test use. | Audited; permanently `legacy_exploratory_development_consumed`. |
| Corrected reproduction / development | Subject-grouped source development, tuning, ablations, and debugging. | Complete for the InclusiveHAR source suite and the separate 75-cell UCI official-train-only reproduction; the UCI official test was not opened. |
| Locked confirmatory | Frozen source-only artifacts evaluated once on the sealed target cohort after all gates passed. | Opening 1 complete; no retry or second opening is permitted. |
| Post-confirmatory secondary | Few-person inclusion, disabled-cohort within-group evaluation, sensor stress, efficiency, signal sensitivity, and qualified CCIL/BPD adaptations. | Completed and explicitly separate from the zero-shot confirmatory claim. |
| v2 source/external development | Nested source-participant models, fixed corruptions, labelled anchors, and frozen DAGHAR held-out evaluation. | Complete as post-analysis development; no new InclusiveHAR target opening and no confirmatory breakthrough claim. |

The opening receipt is [`results/protocol/confirmatory_target_opening_1.json`](results/protocol/confirmatory_target_opening_1.json). The primary statistics record is [`participant_statistics.json`](results/confirmatory/zero_shot_v1/participant_statistics.json), and the human-readable table is [`model_summary_v1.md`](results/confirmatory/zero_shot_v1/model_summary_v1.md).

Release lineage is append-only. `benchmark-v0.1.0` remains attached to its
original candidate after Actions run `32799146947` failed before tests because
of conflicting `uv` frozen/locked options. `benchmark-v0.1.1` and run
`32801378375` are also preserved: hosted `uv` rejected its empty environment
override as a non-boolish value. `benchmark-v0.1.2` and run `32802922698` are
preserved after its Ubuntu validation job passed but Windows exposed CRLF
conversion in six byte-hash integrity tests. `benchmark-v0.1.3` and run
`32811935288` are preserved after both operating-system matrices passed and the
release-security job identified the Linux XGBoost-transitive
`nvidia-nccl-cu12==2.31.2` licence declaration. `benchmark-v0.1.4` and run
`32829208254` are preserved after Ubuntu then exposed a portability defect in
synthetic release fixtures: they inventoried only `safe@1` even though the
hardened Linux gate requires the reviewed XGBoost/NCCL pair. Windows passed
tests, lint, formatting, and mypy; the dependent release-security job was
skipped. `benchmark-v0.1.5` is preserved at commit
`eaa30d18ca60b3c123d1ccf9b095d8d78a03469d`; CI run `32836567358` passed both
synthetic-validation matrices and complete-history release security. The later
external release-bundle Gitleaks scan failed on three sanitized findings: one
actual-risk `temp_clone_token` field and two deterministic `secret_scan` hash
false positives. The bundle, raw report, and supporting evidence remain
quarantined outside Git. No release was created and no assets were uploaded;
v0.1.5 is not released. The machine-readable failure record is
[`results/release/failures/benchmark-v0.1.5-bundle-gitleaks.json`](results/release/failures/benchmark-v0.1.5-bundle-gitleaks.json).
The v0.1.6 successor removes that release-evidence defect: authenticated
repository metadata is filtered before file creation, both online capture and
offline inventory validation reject any retained `temp_clone_token` field, and
the scanner allowance is limited to the two canonical SHA-256 evidence fields
in `ci.json`. Its local gates, run `32846091138`, ready inventory, bundle
reconstruction, and authoritative scan passed. A private draft with two
validated assets was created but not published: the tracked Windows PowerShell
5 example read the UTF-8 REST response with the platform default encoding and
falsely rejected the correct body at an em dash. Strict UTF-8 decoding proves
the local and REST bodies are identical. The sanitized record is
[`benchmark-v0.1.6-draft-utf8-validation.json`](results/release/failures/benchmark-v0.1.6-draft-utf8-validation.json).
The v0.1.7 successor uses strict UTF-8 decoding for saved GitHub JSON. Neither
release patch changes an experiment, result, or target-opening record. No
earlier tag, draft, or failure is rewritten.

## Post-confirmatory findings

The few-person v1.1 curve completed all 1,200 CUDA cells. The descriptive numerical mean leader at each inclusion level was:

| Included target-group participants | Descriptive mean leader | Mean participant macro-F1 | Worst | Lower decile |
|---:|---|---:|---:|---:|
| 0 | Compact DANN | 0.6808 | 0.2699 | 0.3559 |
| 1 | Compact ERM | 0.6744 | 0.4802 | 0.5099 |
| 2 | MoRe-HAR backbone | 0.7060 | 0.2970 | 0.4836 |
| 4 | MoRe-HAR content | 0.7593 | 0.4203 | 0.5784 |

No between-model significance tests were run, so those cross-model leaders are descriptive ranks only. Every model's k=4 mean exceeded its k=0 mean, but several models declined at k=1; only DeepConvLSTM and the static matched baseline were monotone across mean, worst-participant, and lower-decile endpoints. None of the 80 within-model paired k comparisons remained significant after global Holm correction (minimum adjusted p = 0.15625). Tail leaders also differed from mean leaders. These results are promising descriptive inclusion evidence, not proof of a monotonic or population-wide benefit. The complete 64 model-k summaries and comparisons are in [`few_person_v1_1_statistics.json`](results/postconfirmatory/few_person_v1_1/statistics_attempt_004/few_person_v1_1_statistics.json); three failed aggregation attempts remain preserved.

Other completed secondary evidence includes:

- disabled-cohort within-group evaluation: 75/75 CUDA cells; compact ERM mean/worst/lower-decile macro-F1 0.6067/0.2537/0.3000;
- sensor reliability: 120 stressed result cells; linear drift and missing accelerometer-Z produced the largest target mean losses for both compact ERM and CORAL, with only two source-validation participants and uncalibrated severities;
- CUDA efficiency: 80 checkpoints and 320 profiles with 641 contention samples; batch-1 FP32 mean device-resident forward latency was 1.356 ms for compact ERM, 1.498 ms for DANN, 2.021 ms for MoRe-HAR full, 11.760 ms for DeepConvLSTM, and 19.939 ms for the legacy joint model. FP16 autocast was slower for all 16 models at both measured batch sizes, although it reduced peak allocated VRAM in 30 of 32 model-batch pairs;
- qualified post-confirmatory adaptations: CCIL mean 0.6896 but no mean-and-tail improvement and no Holm-adjusted significance; BPD mean 0.5710; neither is an official-faithful reproduction;
- raw/total-acceleration and SI-unit sensitivities: raw acceleration did not robustly improve the primary interface, while g-to-m/s² conversion was algebraically cancelled by training-only z-scoring.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## Scientific scope

The research question is:

> How reliably do smartphone HAR systems trained on conventional participant populations generalize to people whose activities are physically realized differently, including users of assistive devices?

This is an observational ability-associated population-shift benchmark. The released binary cohort metadata is not a direct measurement of physical ability. Disability and assistive-device labels are not model inputs. The released `walking` activity for wheelchair users denotes manual propulsion and is preserved as a distinct physical realization within the functional mobility concept. Ramps are not relabelled as stairs, and jogging is not silently mapped to another locomotion class.

MoRe-HAR (Motion-Realization Factorized HAR) is a compact experimental hypothesis, not an assumed novelty claim. Its full objective combines classification, source-participant supervised contrastive alignment, clean/augmented consistency, measurable realization-descriptor prediction, cross-covariance factorization, and GroupDRO. The current evidence does not support the claim that this combination improves zero-shot ability-shift performance.

## Reproducibility anchors

- InclusiveHAR v4 split hash: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
- Source-window manifest hash: `1ad1ee3accaae5f2f93bb91ac0afa5ce583134ce1d882c3f08323b09026fa522`
- Target seal ID: `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`
- Frozen artifact-set hash: `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`
- Training code commit: `b4dc38fb9d5a0c17003221b61156ebc065395170`
- Target index record hash: `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`
- Locked statistics record hash: `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`
- Publication report hash: `3afe0ceee9f97025d1adc5f59ab3528b512a3212385ef5344028850e9cb39c66`

The protocol tag is `protocol-v1.2.0`. Checkpoint and raw prediction arrays are intentionally outside normal Git history but are preserved locally and bound by the committed inventory and sidecar hashes.

## Data and protocol

The locked main sensing interface is three-axis user acceleration plus three-axis rotation rate at 50 Hz in 128-sample windows. Participants are partitioned before windowing, and the executed released-block protocol uses stride 128 in source training, source validation, and target evaluation. A stride-64 training option was considered in the initial proposal but was not used in the locked experiment. Normalization is fitted on training participants only, calibration on source validation only, and all model selection occurred before the target opening.

The functional-core result uses mobility, sitting, and standing. Inclusive-native and cross-source ontology tracks are documented separately. See:

- [`docs/data/INCLUSIVEHAR_V4.md`](docs/data/INCLUSIVEHAR_V4.md)
- [`docs/data/UCI_HAR_V1.md`](docs/data/UCI_HAR_V1.md)
- [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)
- [`docs/LOCKED_PROTOCOL.md`](docs/LOCKED_PROTOCOL.md)
- [`docs/BENCHMARK_CARD.md`](docs/BENCHMARK_CARD.md)

Raw third-party data, the coursework ZIP, secrets, and large checkpoints are excluded from Git. Dataset manifests record official/versioned retrieval sources, hashes, sizes, licences, citations, and schemas; repository code does not relicense third-party data.

## Environment and commands

The project targets CPython 3.11 and uses `uv`. Neural training and inference in the reported suite used an NVIDIA RTX PRO 3000 Blackwell Laptop GPU with PyTorch 2.12.0+cu132. Recurrent models retained CUDA tensors while disabling cuDNN after preserved Windows cuDNN crash records; neural CPU fallback was not used. Classical scikit-learn models use their native CPU implementations, and XGBoost training used CUDA.

On Linux, XGBoost 3.2.0 resolves `nvidia-nccl-cu12==2.31.2`. PyPI metadata declares `LicenseRef-NVIDIA-Proprietary`, while the wheel contains BSD 3-Clause text. The repository records both observations without replacing one with the other. The exception permits only a local, transitive runtime install: repository files, release assets, and container images do not vendor or redistribute the wheel. The exact wheel, metadata, and embedded-licence hashes are in [`docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json`](docs/release/NVIDIA_NCCL_CU12_2_31_2_REVIEW.json). That review also records that NVIDIA's `nccl_2312` archive URLs render a 2.29.2 page label, so those page bytes are supplementary rather than represented as distribution-specific 2.31.2 text.

```powershell
uv sync --extra training-cuda --group research
uv run inclusive-shift-har validate-manifests
uv run inclusive-shift-har audit-splits --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json --json
uv run inclusive-shift-har validate-artifacts --artifact-root results --require-artifacts --json
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

The package also provides deterministic split/source-window builders, source-CV aggregation, a CUDA-only UCI source-fold runner, source/final experiment modules, one-time confirmatory evaluation code, locked participant statistics, and create-only publication reporting. CI uses synthetic fixtures and never downloads the research datasets.

## Repository map

```text
src/inclusive_shift_har/   package code
configs/                   dataset, ontology, protocol, model, and experiment configs
tests/                     synthetic leakage, lineage, training, and evaluation tests
manifests/                 immutable dataset provenance
legacy/                    lawful legacy audit and reconstructed metrics
results/                   committed small evidence, hashes, gates, and tables
docs/                      audits, cards, protocol, literature, ethics, and status
paper/                     paper-ready outline and result narrative
```

## Known evidence constraints

InclusiveHAR v4 does not expose timestamps or trial/session identifiers, so the
executed split is participant-exclusive and raw-row-disjoint but cannot verify
trial-boundary safety. The locked target cohort contains ten participants, and
the DANN interval [0.5391, 0.8093] is correspondingly wide. The released cohort
flag is used to construct the source/target shift; it is not a measured physical
ability score. Released `Walking` denotes manual wheelchair propulsion for some
target participants, while ramps, stairs, and jogging remain separate concepts.
The source/target gap analysis also mixes one-seed source cross-validation with
five-seed target fits and is descriptive only. Failed runs, quarantines,
deviations, and the unsupported MoRe-HAR result remain part of the record.

## Licensing and citation

Repository-authored code is Apache-2.0 licensed. Third-party datasets, coursework artifacts, papers, checkpoints, and other external materials retain their own terms. [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json) prepare future release metadata; no DOI is claimed until a stable public release is approved.
