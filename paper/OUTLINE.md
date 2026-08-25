# Paper-ready outline and evidence narrative

## Working title

**InclusiveShift-HAR: An Auditable Participant-Exclusive Benchmark for Ability-Associated Population Shift in Smartphone Activity Recognition**

## Abstract scaffold

Smartphone HAR systems are commonly developed on conventional participant cohorts, while the same activity can be physically realized differently by people with disabilities or assistive devices. We construct InclusiveShift-HAR, an evidence-gated InclusiveHAR v4 benchmark that separates source-only model development from a single locked target opening and makes participant-level lower-tail performance explicit. The released binary cohort label is not a direct measure of physical ability. The release also lacks trial identifiers and timestamps, so our protocol is participant-exclusive and raw-row-disjoint but cannot be called trial-safe. In a frozen comparison of 20 model/ablation configurations over five seeds and ten target participants, compact DANN obtained the numerically highest mean participant macro-F1 (0.6808, 95% participant-bootstrap CI [0.5391, 0.8093]), but compact CORAL was only 0.0000556 lower; worst-participant performance for DANN remained 0.2699. The proposed MoRe-HAR model obtained 0.6353 and failed its preregistered mean and lower-tail improvement rule. The benchmark, negative model result, immutable lineage, and explicit data limitation together show why participant-level evidence and honest protocol qualification matter for ability-associated HAR shift.

## 1. Research questions

- **RQ1:** What descriptive source-to-target gap and target-participant heterogeneity are observed under participant-exclusive evaluation, given that source-CV and final-target regimes differ?
- **RQ2:** Which classical, compact neural, recurrent, and domain-generalization baselines are most reliable on mean and lower-tail participant performance?
- **RQ3:** Does factorizing activity content from measurable motion realization improve zero-shot mean and worst-participant generalization without disability metadata at inference?
- **RQ4:** In a post-confirmatory analysis, how does inclusion of k = 1, 2, or 4 target-group training participants affect held-out target participants?
- **RQ5:** In a secondary post-confirmatory analysis, how do realistic sensor reliability perturbations interact with participant cohort and model family?

## 2. Hypotheses and decision rules

The locked MoRe-HAR hypothesis required improvement over every eligible reference in both (a) mean target participant macro-F1 and (b) worst-participant plus lower-decile macro-F1, while remaining within a predeclared 0.02 source non-inferiority margin. Uncertainty and multiplicity-adjusted paired tests were required; significance alone was not the support rule.

The source non-inferiority gate passed before target opening. The target mean and lower-tail improvement conditions both failed, so the hypothesis is not supported. Few-person, sensor-stress, efficiency, signal-sensitivity, and predecessor-adaptation analyses are completed post-confirmatory secondary evidence and cannot alter this decision.

## 3. Related work and novelty boundary

Use `docs/LITERATURE_MATRIX.md` and `paper/references.bib` for the pinned primary-source record. InclusiveHAR itself reports an author-described non-disabled-only training scenario, while O'Brien et al., Lonini et al., and Jamieson et al. already evaluate healthy/non-impaired-to-disabled transfer. Mennella et al. study disability and fairness, HAR-PMD provides a user-independent mobility-aid benchmark, and Age Matters evaluates demographic cross-population generalization. These works rule out “first disability HAR,” “first fairness HAR,” “first healthy-to-disabled evaluation,” and generic “first population-shift HAR” claims.

DAGHAR, HAROOD, BenchHAR, HARBench, WHAR Arena, and the Adaimi–Thomaz distribution-shift study already occupy broad smartphone/wearable DA, DG, OOD, SSL/foundation, and efficiency benchmark space. Adaimi and Thomaz also directly frame user-behaviour/skill change as a distribution shift, although their v1 reported selection rule uses held-out test-domain performance and is not a locked-target comparator. HARBench and BenchHAR are distinct projects and must not be conflated.

The method boundary is tighter still. GILE, AFFAR, BPD, and CMD-HAR already separate activity/domain-invariant information from person/domain-specific or nuisance information. ContrastSense, SICL, MultiSupConHAR, and CCIL already learn user-, subject-, domain-, or category-invariant representations; MultiSupConHAR's same-activity/different-user positives and different-activity/same-user negatives are especially close to MoRe-HAR's content objective. Therefore, neither factorization nor source-subject contrastive alignment is a defensible generic method novelty claim.

The targeted 2026-08-24 article and dataset-DOI search found no verified downstream work already providing the complete InclusiveHAR-v4 source-only-to-disabled, participant-exclusive released-block benchmark with a strong matched baseline suite, participant lower-tail/calibration evidence, and a one-time target opening. Crossref, Semantic Scholar, OpenAlex, Europe PMC, PubMed, ScienceDirect, arXiv, DataCite, and targeted IEEE/ACM/PMLR searches were negative. This is a scoped index result, not proof of novelty; an authenticated manual Google Scholar search remains required immediately before submission.

### Benchmark and model gates

- **Benchmark:** `pass_with_narrowed_contribution`. Claim only an auditable participant-exclusive, no-target-adaptation, ability-associated InclusiveHAR-v4 benchmark with explicit unrecoverable trial-boundary risk. Do not use “first,” “fair,” “clinical,” or unqualified “leakage-safe.”
- **MoRe-HAR:** `closed_not_supported` / `exploratory_negative`. Compact DANN obtained mean target-participant macro-F1 0.6808438 versus 0.6353323 for full MoRe-HAR, and the candidate failed both preregistered mean and joint lower-tail improvement conditions. Preserve the negative result; do not retune against the consumed target or present a positive model-novelty claim.
- **Predecessor adaptations:** local post-confirmatory CCIL/BPD adaptations are complete but paper-derived and non-faithful. CCIL obtained target mean 0.6896 without improving both tails or surviving adjusted paired tests; BPD obtained 0.5710. They cannot establish faithful head-to-head superiority and do not reopen the closed model gate.

## 4. Benchmark construction

Describe official versioned acquisition, hashes/licences, privacy-safe audit, participant/activity coverage, six-channel primary interface, units, stable identifiers, cache invalidation, participant-first split construction, training-only normalization, source-only calibration, and the three ontology tracks.

The primary functional-core class order is mobility, sitting, standing. Released walking for wheelchair users denotes manual propulsion, preserved as a different activity realization within a functional mobility concept. Ramp labels remain 8% ramp activities, not stairs. Cross-source exact/provisional mappings remain separate.

The central limitation is unrecoverable hidden trial boundaries: no timestamps, sessions, trial IDs, or recording IDs are released. Report the 100% unconditional hidden-join bound and the explicitly unverified conditional 7.8895% bound. Use "participant-exclusive released-block benchmark," never unqualified "leakage-safe."

## 5. Model and baselines

Baseline families include random forest, CUDA XGBoost, RBF SVM, logistic regression, legacy CNN1D/BiLSTM/joint branches, DeepConvLSTM, compact ERM/CORAL/DANN, a parameter-matched static two-branch control, and the MoRe-HAR ablation ladder. Official third-party models that could not be faithfully or licence-compatibly adapted are listed as exclusions rather than relabelled local approximations.

Describe MoRe-HAR using `docs/MODEL_CARD_MORE_HAR.md`. Emphasize that disability/device labels are not inputs and that realization descriptors are measurable signal proxies, not sensitive-attribute prediction.

## 6. Experimental protocol

- Source participants: 1-10; grouped development only.
- Frozen final source train participants: 1,2,3,4,5,6,7,9.
- Frozen source validation participants: 8,10.
- Locked target participants: 11-20.
- Seeds: 11, 23, 47, 89, 131.
- Primary unit: participant.
- Confirmatory windows: length/stride 128 within released blocks.
- Calibration: source validation only.
- Target openings: exactly one, consumed on 2026-08-24.
- Neural execution: sequential CUDA; recurrent CUDA tensors with cuDNN disabled after preserved backend failures.

## 7. Main results

The complete table is `results/confirmatory/zero_shot_v1/model_summary_v1.md`; exact CSV and participant rows are adjacent. Core rows:

| Model | Mean participant macro-F1 (95% CI) | Worst | Lower decile | NLL | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compact DANN | 0.6808 [0.5391, 0.8093] | 0.2699 | 0.3559 | 0.8813 | 0.4321 | 0.0624 | 0.1464 |
| Compact CORAL | 0.6808 [0.5377, 0.8153] | 0.2741 | 0.3670 | 0.7850 | 0.4204 | 0.0611 | 0.1475 |
| Compact ERM | 0.6777 [0.5373, 0.8065] | 0.2725 | 0.3541 | 0.8655 | 0.4320 | 0.0556 | 0.1466 |
| MoRe-HAR full | 0.6353 [0.4911, 0.7665] | 0.2584 | 0.2660 | 1.2793 | 0.5050 | 0.0946 | 0.2021 |

DANN, CORAL, and ERM are effectively close in point estimates relative to the uncertainty; do not overstate DANN's 0.0000556 numerical lead over CORAL. Lower-tail performance is poor for all methods. MoRe-HAR minus DANN is -0.0455 mean participant macro-F1; Holm-adjusted exact sign-flip p = 0.7207 and Holm-adjusted Wilcoxon p = 0.7559. The small sample does not establish equivalence or absence of a meaningful effect.

A post-confirmatory cross-cohort derivation estimated DANN source mean 0.8004 versus target mean 0.6808, a source-minus-target gap of 0.1196 with 95% interval [-0.0371, 0.2831]. The source side is one-seed participant-grouped cross-validation while the target side averages five final-fit seeds; training and evaluation regimes differ. Report this only as a descriptive gap, not a controlled population-effect estimate.

## 8. Ablations

| MoRe-HAR variant | Mean | Worst | Lower decile |
|---|---:|---:|---:|
| Backbone | 0.6482 | 0.2630 | 0.3026 |
| + augmentation | 0.5634 | 0.2563 | 0.2861 |
| + content/consistency | 0.6616 | 0.2566 | 0.2809 |
| + realization/factorization | 0.5600 | 0.2437 | 0.2832 |
| GroupDRO-only variant | 0.6435 | 0.2674 | 0.2868 |
| Full | 0.6353 | 0.2584 | 0.2660 |
| Full without accelerometer | 0.4456 | 0.1928 | 0.2070 |
| Full without gyroscope | 0.4218 | 0.1661 | 0.2063 |
| Static parameter-matched branches | 0.4528 | 0.1931 | 0.2301 |

The full objective does not exhibit monotonic gains as components are added. The content/consistency variant is the strongest MoRe-family target result. Treat this as evidence against the proposed factorization combination in the current setting, not as permission to retune after opening.

## 9. Secondary result tables

All results in this section are post-confirmatory or source-development evidence; none changes the locked zero-shot decision.

### 9.1 Corrected UCI official-train reproduction

| Model | Mean participant macro-F1 | Worst | Lower decile | 95% CI |
|---|---:|---:|---:|---:|
| Joint CNN/BiLSTM | 0.9118 | 0.7290 | 0.8631 | [0.8830, 0.9360] |
| CNN1D | 0.8852 | 0.6328 | 0.8439 | [0.8511, 0.9126] |
| BiLSTM | 0.7084 | 0.5243 | 0.6495 | [0.6828, 0.7300] |

All 75 grouped-fold/seed CUDA cells completed. The official UCI test and all InclusiveHAR material remained unopened; this is UCI-native source-development evidence, not cross-source pretraining or confirmatory evidence.

### 9.2 Disabled-cohort within-group evaluation

| Model | Mean | Worst | Lower decile |
|---|---:|---:|---:|
| Compact ERM | 0.6067 | 0.2537 | 0.3000 |
| MoRe-HAR backbone | 0.5626 | 0.2437 | 0.2580 |
| DeepConvLSTM | 0.4001 | 0.1902 | 0.1916 |

The 75-cell result is descriptive and participant-exclusive. It uses a different training regime from zero-shot evaluation, so point-estimate differences are not controlled inclusion effects.

### 9.3 Few-person inclusion curve v1.1

| k | Descriptive numerical mean leader | Mean | Worst | Lower decile | Descriptive tail leaders |
|---:|---|---:|---:|---:|---|
| 0 | Compact DANN | 0.6808 | 0.2699 | 0.3559 | worst: joint CNN/BiLSTM 0.2943; lower decile: CORAL 0.3670 |
| 1 | Compact ERM | 0.6744 | 0.4802 | 0.5099 | DANN 0.4886/0.5162 |
| 2 | MoRe-HAR backbone | 0.7060 | 0.2970 | 0.4836 | worst: static matched 0.3333; lower decile: MoRe-HAR content 0.4866 |
| 4 | MoRe-HAR content | 0.7593 | 0.4203 | 0.5784 | worst: MoRe-HAR backbone 0.4417; lower decile: joint CNN/BiLSTM 0.5986 |

All 1,200 CUDA cells validated. No between-model significance tests were run, so every cross-model leader in the table is a descriptive rank only. Every model's k=4 mean exceeded k=0, but only DeepConvLSTM and the static matched baseline were monotone across mean, worst-participant, and lower-decile endpoints. None of 80 within-model paired comparisons survived global Holm correction (minimum adjusted p = 0.15625). Full MoRe-HAR changed from 0.635332 at k=0 to 0.751408 at k=4 (+0.116075), with adjusted permutation and Wilcoxon p-values both 1.0. Treat the curve as descriptive and promising, not proven.

### 9.4 Sensor reliability

| Condition | Compact ERM target delta | Compact CORAL target delta |
|---|---:|---:|
| Gaussian noise level 1 | -0.0095 | -0.0099 |
| Contiguous 32-sample dropout | -0.0305 | -0.0382 |
| Missing accelerometer-Z | -0.2109 | -0.1953 |
| Missing gyroscope | +0.0136 | +0.0090 |
| Rate reduction factor 2 | -0.0034 | -0.0017 |
| Linear drift level 1 | -0.2495 | -0.2443 |

The aggregate covers 120 stressed cells. Source comparisons use only two validation participants versus ten target participants, and corruption severities were not empirically calibrated. The apparent cohort-dependent changes and small improvements under missing gyroscope are descriptive, not causal interaction evidence.

### 9.5 Qualified predecessor adaptations and signal sensitivities

CCIL obtained target mean/worst/lower-decile 0.6896/0.2724/0.3459 versus compact ERM 0.6777/0.2725/0.3541. Its +0.0119 mean difference did not include tail improvement, and both Holm-adjusted paired p-values were 0.2109. BPD obtained 0.5710/0.2471/0.2751. Both are local, post-confirmatory, non-faithful adaptations.

Raw/total acceleration changed compact ERM target mean from 0.6777 to 0.6166 and full MoRe-HAR from 0.6353 to 0.6308; intervals included zero. Converting g to m/s² before the same training-only z-score produced exactly equal normalized tensors and was not an independent accuracy experiment.

### 9.6 CUDA efficiency

| Model | Parameters | Batch-1 FP32 mean latency (ms) |
|---|---:|---:|
| Compact ERM | 101,955 | 1.3561 |
| Compact DANN | 112,043 | 1.4985 |
| MoRe-HAR full | 138,396 | 2.0207 |
| DeepConvLSTM | 200,867 | 11.7601 |
| Joint CNN/BiLSTM | 2,689,414 | 19.9391 |

The 320-profile aggregate validated 80 checkpoints and 641 contention samples. These are synthetic-zero, device-resident, forward-only CUDA measurements without host-to-device transfer. FP16 autocast was slower for all 16 models at both batch sizes, while reducing peak allocated VRAM in 30 of 32 model-batch comparisons; this is a memory tradeoff, not an acceleration finding. The 60 recurrent profiles used the cuDNN-disabled CUDA fallback. Timing is valid only with the declared allowlisted WDDM ambient processes, analytical MAC/FLOP counts cover only Conv1d, Linear, and LSTM modules, and no full-graph, end-to-end, or mobile latency is claimed.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## 10. Ethics

Use non-stigmatizing language and frame failures as limitations of data/model coverage. The binary released group label does not capture the diversity of disability, device use, fatigue, technique, environment, or preferences. Device/type analyses are exploratory due to very small cells. No output should guide medical care, benefits, employment, insurance, surveillance, or physical safety decisions.

## 11. Limitations

- Ten target participants and broad bootstrap intervals.
- One phone model, waist placement, and nominal sampling interface.
- Unrecoverable trial/timestamp boundaries and possible hidden joins.
- Functional mobility intentionally spans different physical realizations and is not exact gait semantics.
- Dataset order is group-aligned; order proxies are prohibited, but unobserved collection artifacts may remain.
- Source development and model-family choices are consumed evidence, not independent replication.
- A single target opening compares many frozen models; only the predeclared candidate family receives confirmatory multiplicity control.
- Calibration is source-validation based and may not transfer under shift.
- The released binary cohort label is not a direct measure of physical ability.
- Descriptive source-target gaps compare different source-CV and final-target regimes.
- Sensor-stress source evidence has two participants versus ten target participants, and severities are uncalibrated.
- Few-person and predecessor-adaptation analyses were designed or executed after the consumed target opening; none can restore confirmatory status.
- CUDA latency is device-resident forward-only evidence on one machine under a sampled contention policy.
- Cross-source transfer, SSL/foundation models, and several requested faithful third-party baselines were not evaluated.
- The released cohort label is observational, the target cohort contains ten
  participants, and the study collected neither a randomized intervention nor
  clinical outcomes; estimates describe this benchmark cohort and cannot
  identify population effects.

## 12. Reproducibility statement

The repository pins code dependencies, data identities, ontology, preprocessing, participant splits, window IDs, seeds, configurations, checkpoint selection, source-only calibrators, code commits, CUDA environment, and RNG states. CI uses synthetic fixtures and downloads no research data. Raw data/checkpoints/prediction arrays stay outside Git, while committed inventories and sidecars bind their hashes. The target receipt was written before materialization; the opening is consumed and cannot be repeated. Failures and protocol deviations remain versioned.

Primary anchors: split `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`, artifact set `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`, analysis plan `7b99dd5894109370397867a1ca141758c0a30b4efb2fb3d76aba23ab1ad62177`, target index `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`, and statistics `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`. Secondary anchors include few-person `d14c4a071ee2460a2182fcab56ab6454be6d4cc3c6cc391445e56a368b776058`, sensor stress `abe1aee172e9b680e2aca16848c84bf6788491d005964b3d18e597098a06df90`, disabled within-group `fbe81492df167e136c3ad14bccfdd94a4c82c12497b48c2042a50e25dc521180`, and efficiency `7c0fa71edcd0a368090df0513d6a418a35a6989b35f828734febd135f11530bb`.
