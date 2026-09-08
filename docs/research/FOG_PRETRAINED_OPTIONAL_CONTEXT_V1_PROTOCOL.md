# Corrected FoG pretrained optional-context protocol v1

Status: frozen prospective development protocol when bound by the content-addressed configuration. This protocol applies only to the four-arm corrected-FoG experiment specified in `.audit/research_breakthrough_review_20260908-002/EXACT_NEXT_EXPERIMENT.md`. It does not modify any historical protocol, result or gate.

## Question

Test whether a released self-supervised acceleration representation adds transferable motion information beyond current-query energy, physical history availability and an identical randomly initialized feature extractor. Preserve valid current measurements even when ten seconds of history are unavailable.

The four arms are Q, M, R and P. Q uses two current-query log-RMS features. M adds a binary physical-history mask. R adds 1,024 frozen features from a seeded random HARNET10 encoder. P is identical to R except that the encoder has the pinned released HARNET checkpoint. P is the only primary candidate. All arms use the same deterministic L2 logistic readout family.

## Evidence lane

This is adaptive development on the repeatedly consumed corrected FoGSTAR cohort. Use exactly 1,939 frozen observable candidates, 1,213 scored queries, 22 participants and the inherited five participant folds. Do not open InclusiveHAR P11-P20. Partition identity, scoring projection, labels, baseline probabilities and missing-sensor fallback are inherited byte-for-byte from the sealed L9v run.

Fit on scored, current-left-ankle-available queries from outer-training participants. Predict current-left-ankle-available observable queries for the held-out participants and report the complete scored roster. Missing-current-ankle rows retain exact B0. Rows with zero L9v stationary mass retain exact L9v. Every other current-ankle row uses the fitted motion probability, including rows with less than ten seconds of history.

## Signal and encoder

The two query features are the qualified log linear-acceleration RMS and log gyroscope RMS over the inherited 128-sample, 50-Hz current query. Fit their means and population standard deviations on the outer-training queries only.

History support is determined without annotations from participant, session, the inherited back run, the matching left-ankle finite monotonic run and timestamps. A history is the right-aligned interval ending at the inherited query end. Read native left-ankle total acceleration in g, linearly interpolate the three physical axes to 300 points at 30 Hz over ten seconds without extrapolation, and clip each value to [-3g,+3g]. The history mask must equal the sealed full-history mask exactly. Labels, activity changes, task IDs and scored eligibility cannot define support.

Use official OxWearables HARNET source commit `150550ea5d41800229c95e36f88f5bf0d2e7cf04` and `model_check_point/mtl_best.mdl`, SHA-256 `c64f9135d99e2dcdfc9ae7cc0672f2bcc438df9ceb8215665882f92cddd162a6`. Strictly load every feature-extractor parameter and buffer after removing only `module.feature_extractor.`. Reject missing, unexpected, duplicate, wrongly shaped or nonfinite feature state. The feature extractor has 10,457,408 parameters, 131 state entries and emits 1,024 values from a `3x300` input. Ignore the upstream classifier and task heads.

Freeze all parameters and BatchNorm running state and use evaluation mode under inference-only execution. Repeated inference on the same batch must be bitwise identical, and the maximum absolute difference between the first example encoded alone versus in the frozen batch of 32 must not exceed `1e-5`. For R, instantiate the identical architecture once per fold with seed `11 + fold_index`. Extract P and R only for genuine histories; never pass fabricated zero histories. Fit embedding means and population standard deviations using genuine outer-training histories only, then set missing-history embeddings to exact zero. R and P have identical 1,027 input dimensions: two query energies, one physical mask and 1,024 embeddings.

## Weighting and fitting

For each training participant i and represented original class c, assign raw weight `1/(k_i*n_ic)` and normalize all selected weights to mean one. Recompute on the common current-query support. Use the binary target `original class == mobility` without further binary or automatic class weighting.

For every arm and fold fit exactly:

`LogisticRegression(penalty='l2', C=1, solver='lbfgs', fit_intercept=True, max_iter=1000, tol=1e-6, class_weight=None, random_state=11+fold)`.

Order attempts by fold and Q/M/R/P. Maximum 20 attempts. Nonconvergence or any failure stops the controller; do not change an optimizer, feature, seed or threshold and do not retry. Save all completed states and failures. No encoder parameter is fitted. No validation or evaluation outcome selects a checkpoint, arm, hyperparameter or input adapter.

Compose motion m with frozen L9v conditional posture `q = p(sitting)/(p(sitting)+p(standing))` as `[m,(1-m)q,(1-m)(1-q)]`. Do not add epsilon. Preserve exact fallbacks defined above.

## Endpoints and claims

Primary endpoint is fixed-three-class macro-F1 within each participant followed by an equal average of all 22 people. Class order is mobility, sitting, standing; absent-class F1 is zero. Use inherited participant bootstrap and metric implementations.

Retain E2, T128, T500 and T500-P from the independently replayed motion-factorization run as hash-pinned, zero-fit contextual references alongside L9v, Lv, Bv, B0 and F3. Reproduce their stored probabilities, participant reports and identities exactly. They cannot select P or alter any fit. In particular, Q versus E2 must be reported with the explicit warning that the two methods differ in current-query support, training rows, weights and normalization.

Claim sequence:

1. P versus L9v: mean gain at least 0.015; 95% paired participant bootstrap lower endpoint above zero; at least 14 strict participant wins; bottom-seven difference at least -0.010; worst-score difference at least -0.030; no paired participant loss below -0.050; mean participant recall changes mobility at least -0.010 and sitting/standing each at least -0.020; every leave-one-person and leave-one-fold mean difference positive; complete validation.
2. P versus R: mean gain at least 0.010 and positive bootstrap lower endpoint, with the same tail, paired-person, class-recall and leave-out guards. The 14-win requirement applies only to the practical L9v comparison.
3. P versus M: the same requirements as P versus R.

Stop claim progression at the first failure while reporting every arm and all numerical contrasts. M-Q describes availability; R-M describes random nonlinear history features/capacity; P-R isolates the particular released weights; P-M measures the representation package beyond query and mask. Q versus historical E2 changes support, training rows, weights and normalization and is not a pure coverage effect.

The pre-fit label-oracle audit may only prove that the 14-win gate is structurally reachable. It must return 853 editable rows, 17 possible wins and the retained exact motion-only ceiling. It cannot select rows, thresholds, features, weights or predictions and is never a model result.

## Validation and artifacts

Before fitting, require exact hashes for source data, baseline evidence, implementation, config, this protocol, governing specification, HARNET source/checkpoint/license and dependencies. Require exact current-query replay, native support parity, label-independent signal features, strict encoder state loading, frozen-state checks, deterministic repeated extraction, batch/single equivalence, finite outputs, expected dimensions and an 8-GB GPU memory receipt.

After fitting, independently reconstruct contexts and embeddings, regenerate seeded random encoders, replay all 20 logistic states, compose probabilities and reproduce metrics and gates. Preserve per-fold indices, weights, scalers, encoder state hashes, embeddings, probabilities, convergence, runtime, participant harms, uncertainty, source provenance, failure records, worker shutdown and create-only manifests.

Do not automatically launch another seed, encoder, fine-tuning experiment, cohort or physical study. A pass remains development evidence requiring independent participant confirmation. A frozen transfer experiment and its components are prior art; novelty and population superiority are not established by this protocol.
