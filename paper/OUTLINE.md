# Paper-ready outline and evidence narrative

## Working title

**InclusiveShift-HAR: An Auditable Participant-Exclusive Benchmark for Physical-Ability Generalization in Smartphone Activity Recognition**

## Abstract scaffold

Smartphone HAR systems are commonly developed on conventional participant cohorts, while the same activity can be physically realized differently by people with disabilities or assistive devices. We construct InclusiveShift-HAR, an evidence-gated InclusiveHAR v4 benchmark that separates source-only model development from a single locked target opening and makes participant-level lower-tail performance explicit. The release lacks trial identifiers and timestamps, so our protocol is participant-exclusive and raw-row-disjoint but cannot be called trial-safe. In a frozen comparison of 20 model/ablation configurations over five seeds and ten target participants, compact DANN obtained the highest mean participant macro-F1 (0.6808, 95% participant-bootstrap CI [0.5391, 0.8093]); worst-participant performance remained 0.2699. The proposed MoRe-HAR model obtained 0.6353 and failed its preregistered mean and lower-tail improvement rule. The benchmark, negative model result, immutable lineage, and explicit data limitation together show why participant-level evidence and honest protocol qualification matter for ability-associated HAR shift.

## 1. Research questions

- **RQ1:** How large and heterogeneous is source-only-to-target ability-associated performance degradation under participant-exclusive evaluation?
- **RQ2:** Which classical, compact neural, recurrent, and domain-generalization baselines are most reliable on mean and lower-tail participant performance?
- **RQ3:** Does factorizing activity content from measurable motion realization improve zero-shot mean and worst-participant generalization without disability metadata at inference?
- **RQ4:** In a post-confirmatory analysis, how does inclusion of k = 1, 2, or 4 target-group training participants affect held-out target participants?
- **RQ5:** In a secondary post-confirmatory analysis, how do realistic sensor reliability perturbations interact with participant cohort and model family?

## 2. Hypotheses and decision rules

The locked MoRe-HAR hypothesis required improvement over every eligible reference in both (a) mean target participant macro-F1 and (b) worst-participant plus lower-decile macro-F1, while remaining within a predeclared 0.02 source non-inferiority margin. Uncertainty and multiplicity-adjusted paired tests were required; significance alone was not the support rule.

The source non-inferiority gate passed before target opening. The target mean and lower-tail improvement conditions both failed, so the hypothesis is not supported. Few-person and sensor-stress questions are post-confirmatory secondary analyses and cannot alter this decision.

## 3. Related work and novelty boundary

Use `docs/LITERATURE_MATRIX.md` and `paper/references.bib` for the pinned primary-source record. InclusiveHAR itself reports an author-described non-disabled-only training scenario, while O'Brien et al., Lonini et al., and Jamieson et al. already evaluate healthy/non-impaired-to-disabled transfer. Mennella et al. study disability and fairness, HAR-PMD provides a user-independent mobility-aid benchmark, and Age Matters evaluates demographic cross-population generalization. These works rule out “first disability HAR,” “first fairness HAR,” “first healthy-to-disabled evaluation,” and generic “first population-shift HAR” claims.

DAGHAR, HAROOD, BenchHAR, HARBench, WHAR Arena, and the Adaimi–Thomaz distribution-shift study already occupy broad smartphone/wearable DA, DG, OOD, SSL/foundation, and efficiency benchmark space. Adaimi and Thomaz also directly frame user-behaviour/skill change as a distribution shift, although their v1 reported selection rule uses held-out test-domain performance and is not a locked-target comparator. HARBench and BenchHAR are distinct projects and must not be conflated.

The method boundary is tighter still. GILE, AFFAR, BPD, and CMD-HAR already separate activity/domain-invariant information from person/domain-specific or nuisance information. ContrastSense, SICL, MultiSupConHAR, and CCIL already learn user-, subject-, domain-, or category-invariant representations; MultiSupConHAR's same-activity/different-user positives and different-activity/same-user negatives are especially close to MoRe-HAR's content objective. Therefore, neither factorization nor source-subject contrastive alignment is a defensible generic method novelty claim.

The targeted 2026-08-24 article and dataset-DOI search found no verified downstream work already providing the complete InclusiveHAR-v4 source-only-to-disabled, participant-exclusive released-block benchmark with a strong matched baseline suite, participant lower-tail/calibration evidence, and a one-time target opening. Crossref, Semantic Scholar, OpenAlex, Europe PMC, PubMed, ScienceDirect, arXiv, DataCite, and targeted IEEE/ACM/PMLR searches were negative. This is a scoped index result, not proof of novelty; an authenticated manual Google Scholar search remains required immediately before submission.

### Benchmark and model gates

- **Benchmark:** `pass_with_narrowed_contribution`. Claim only an auditable participant-exclusive, no-target-adaptation, ability-associated InclusiveHAR-v4 benchmark with explicit unrecoverable trial-boundary risk. Do not use “first,” “fair,” “clinical,” or unqualified “leakage-safe.”
- **MoRe-HAR:** `closed_not_supported` / `exploratory_negative`. Compact DANN obtained mean target-participant macro-F1 0.6808438 versus 0.6353323 for full MoRe-HAR, and the candidate failed both preregistered mean and joint lower-tail improvement conditions. Preserve the negative result; do not retune against the consumed target or present a positive model-novelty claim.
- **Predecessor adaptations:** local post-confirmatory CCIL/BPD adaptations are paper-derived and non-faithful. They cannot establish faithful head-to-head superiority and do not reopen the closed model gate.

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

DANN, CORAL, and ERM are effectively close in point estimates relative to the uncertainty; do not overstate DANN's 0.00006 numerical lead over CORAL. Lower-tail performance is poor for all methods. MoRe-HAR minus DANN is -0.0455 mean participant macro-F1; Holm-adjusted exact sign-flip p = 0.7207 and Holm-adjusted Wilcoxon p = 0.7559. The small sample does not establish equivalence or absence of a meaningful effect.

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

Populate only from validated post-confirmatory evidence; never infer missing values.

| Few-person model | k=0 | k=1 | k=2 | k=4 | Worst/lower-tail trend |
|---|---:|---:|---:|---:|---|
| Strongest compact baseline | 0.6808 | pending | pending | pending | pending |
| MoRe-HAR full | 0.6353 | pending | pending | pending | pending |

| Sensor condition | Source cohort | Target cohort | Interaction summary |
|---|---:|---:|---|
| Clean | pending harmonized analysis | locked result | pending |
| Noise/dropout/missing modality/rate/drift | pending | pending | pending |

| Model | Parameters | MACs/FLOPs | Latency | Peak VRAM | Model size |
|---|---:|---:|---:|---:|---:|
| Compact DANN | pending consolidated profile | pending | pending | pending | pending |
| MoRe-HAR full | pending consolidated profile | pending | pending | pending | pending |

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
- No causal, clinical, fairness, or population-wide conclusion.

## 12. Reproducibility statement

The repository pins code dependencies, data identities, ontology, preprocessing, participant splits, window IDs, seeds, configurations, checkpoint selection, source-only calibrators, code commits, CUDA environment, and RNG states. CI uses synthetic fixtures and downloads no research data. Raw data/checkpoints/prediction arrays stay outside Git, while committed inventories and sidecars bind their hashes. The target receipt was written before materialization; the opening is consumed and cannot be repeated. Failures and protocol deviations remain versioned.

Primary anchors: split `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`, artifact set `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`, analysis plan `7b99dd5894109370397867a1ca141758c0a30b4efb2fb3d76aba23ab1ad62177`, target index `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`, and statistics `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`.
