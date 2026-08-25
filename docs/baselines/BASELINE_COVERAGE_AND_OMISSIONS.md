# Baseline coverage, provenance, and omissions

This inventory distinguishes executed frozen configurations from requested
published baselines. It is derived from
`results/protocol/final_source_artifact_freeze_v1.json`, the one-time target
index, `configs/models/local_baselines_v1.yaml`, and
`docs/baselines/third_party_baseline_provenance.json`. It does not upgrade an
adaptation or architectural resemblance into a faithful reproduction.

## Frozen primary inventory

The primary freeze contains 100 model-seeds: 20 configuration IDs at seeds 11,
23, 47, 89, and 131. The corresponding one-time target index is complete. The
20 IDs are:

- Classical: `random-forest`, `xgboost`, `svm-rbf`, `logistic-regression`.
- Legacy reconstructions: `legacy-cnn1d`, `legacy-bilstm`,
  `legacy-joint-cnn-bilstm`.
- Local strong/DG controls: `deepconvlstm`, `compact-erm`, `compact-coral`,
  `compact-dann`, `static-dual-branch-matched`.
- MoRe-HAR ladder and modality ablations: `more-har-backbone`,
  `more-har-augmentation`, `more-har-content`, `more-har-factorized`,
  `more-har-groupdro`, `more-har-full`, `more-har-full-no-accelerometer`, and
  `more-har-full-no-gyroscope`.

These IDs are the only configurations eligible for claims about the locked
primary comparison. `more-har-groupdro` is a candidate-family ablation, not a
standalone official GroupDRO reproduction. `legacy-joint-cnn-bilstm` is one
jointly optimized two-branch model, not an ensemble of independently trained
estimators.

## CNN-HAR clarification

No faithful CNN-HAR implementation exists in this repository. CNN-HAR appears
in the WHAR Arena related-work comparison, but it has no standalone entry in the
machine-readable third-party provenance registry, no audited source revision or
software-license decision, no reconciled six-channel/128-sample adapter, and no
frozen configuration under that name.

`CompactResidualHAR` is repository-authored and was intentionally registered as
`compact_residual_*`/`compact-erm`; it is **not CNN-HAR**. It must not be cited,
plotted, or released as CNN-HAR, TinyHAR, or a faithful reproduction. A future
CNN-HAR row requires a primary architecture source,
author-linked implementation where available, pinned revision, software
license, exact input and training protocol, shape/gradient/parameter/MAC tests,
and source-only model selection before any evaluation.

## Requested baseline disposition

| Requested item | Current evidence status | Claim-safe description or blocker |
|---|---|---|
| Random Forest, XGBoost, RBF SVM, logistic regression | Executed in frozen primary suite | Comparable classical baselines; scikit-learn models use native CPU, XGBoost used CUDA |
| Coursework CNN1D, BiLSTM, joint CNN/BiLSTM | Executed in frozen primary suite | Corrected local reconstructions; old UCI test remains development-consumed |
| DeepConvLSTM | Executed in frozen primary suite | Local model in the DeepConvLSTM family, not asserted bit-for-bit official reproduction |
| CNN-HAR | Omitted | Provenance and faithful adapter not established; local compact models cannot use this name |
| TinyHAR | Omitted | Audited paper-linked repository had no software license; source reuse blocked |
| TinierHAR | Omitted | No software license plus paper/repository window and block-count conflicts |
| HARMamba | Omitted | Apache-2.0 source exists, but runtime dependencies, 9-channel/default-length mismatch, and target-selecting trainer require a new protocol-safe adapter |
| ERM, CORAL, DANN | Executed in frozen primary suite | Local controlled baselines under identical locked folds/information |
| GroupDRO | Partial | Candidate-family `more-har-groupdro` ablation executed; no separate official/general backbone reproduction claimed |
| CCIL | Not in frozen primary inventory; qualified post-confirmatory adaptation complete | No official code was located; the completed local loss adaptation is paper-derived and not a faithful reproduction |
| BPD | Not in frozen primary inventory; qualified post-confirmatory adaptation complete | Official Apache-2.0 trainer remains blocked by boundary/stride and target-selection conflicts; the completed local boundary-safe adaptation is not official-faithful |
| BenchHAR SSL | Omitted | HAR-Bench and SimMTM audited revisions had no software license; 20 Hz/120-sample interface differs from the locked 50 Hz/128-sample benchmark |
| FOCAL | Omitted | MIT upstream targets acoustic/seismic pairs; no validated smartphone-inertial adapter exists here |
| Foundation-model linear probe/adaptation/fine-tune | Omitted | No compatible, licensed, frozen adapter or equal-budget source-only experiment exists |
| Cross-source pretraining | Omitted as a result | UCI-native grouped reproduction completed separately but is not pretraining; exact all-cohort cross-source classification is scientifically blocked beyond sitting, with standing provisional and walking semantically incompatible |

## Feasibility and execution addendum — 2026-08-24

This addendum preserves the frozen-primary disposition above while recording the
later execution state at commit `ad657d3aec3e2476ce7a1e8226bb2fd3a99bd721`.
The corrected UCI-HAR source-grouped reproduction is now complete, but it is a
six-class, UCI-native source-development result. It did not refit a transferable
encoder on all source participants, replace or adapt its classification head,
or evaluate an InclusiveHAR participant. It therefore does not constitute
cross-source pretraining evidence.

The later qualified CCIL/BPD extension is also complete at
`results/postconfirmatory/ccil_bpd_v1/ccil_bpd_postconfirmatory_aggregate.json`.
CCIL obtained target mean/worst/lower-decile macro-F1 0.6896/0.2724/0.3459;
BPD obtained 0.5710/0.2471/0.2751. Both are post-confirmatory local adaptations,
not official-faithful baselines, and neither can enter or revise the frozen
primary ranking.

| Extension track | Gate status | Empirical status | Evidence-safe interpretation |
|---|---|---|---|
| Exact-label, all-cohort UCI→InclusiveHAR classification | **Blocked** | **Not run** | Sitting is the only exact shared class. Standing remains provisional, and UCI ordinary walking is not InclusiveHAR manual wheelchair propulsion. A one-class track is not a classification benchmark. |
| Adapted-label or representation-transfer pretraining | Not implemented | **Not run** | A separately declared adapted-label track is scientifically possible, but no all-source refit, encoder-transfer/head-replacement route, target-blind adaptation configuration, equal-budget selection procedure, or result artifact exists. This absence is not evidence that pretraining fails. |
| BenchHAR/SimMTM SSL comparison | **Blocked** | **Not run** | The audited HAR-Bench and SimMTM revisions have no software licence, and the BenchHAR 20 Hz/120-sample instance-normalized interface is not the locked 50 Hz/128-sample interface. No local checkpoint was available. |
| FOCAL inertial adaptation | Adapter not implemented | **Not run** | The official MIT-licensed source targets paired acoustic/seismic inputs. No independently implemented and validated smartphone-IMU adapter or local checkpoint exists. |
| Wearable-foundation-model linear probe, parameter-efficient adaptation, or full fine-tuning | **Blocked** | **Not run** | No compatible licensed local checkpoint, frozen adapter, model-specific preprocessing manifest, or equal-budget source-only configuration passed the evidence gate. |

BioBankSSL, LIMU-BERT, CRT, TS-TCC, TS2Vec, and CrossHAR remain
literature-audited candidates only; they have no integrated local implementation,
configuration, checkpoint, or result. The machine-readable form of this addendum
is `extension_status_addendum` in
`docs/baselines/third_party_baseline_provenance.json`.

Publication wording must say that these comparisons were **not evaluated**.
They are neither zero-valued rows nor negative empirical findings. Because target
opening 1 is consumed, any future target-side comparison is post-confirmatory and
cannot enter or revise the locked-primary ranking.

Full paper/repository pins and reopening criteria are in
`docs/baselines/THIRD_PARTY_BASELINE_AUDIT.md`. Missing baselines must stay visible
in limitations and tables; they are not zero-valued results and must not be
silently replaced by local proxies.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## Minimum evidence for adding a baseline

A new baseline enters a future comparison only after its code/license decision,
exact source revision, input-interface reconciliation, participant and raw-sample
split audit, source-only tuning budget, normalization/calibration lineage,
checkpoint reconstruction, five-seed CUDA execution policy, failure records,
and complete participant-level aggregation all validate. Because target opening
1 is consumed, any new comparison is post-confirmatory and cannot retroactively
join or change the locked primary ranking.
