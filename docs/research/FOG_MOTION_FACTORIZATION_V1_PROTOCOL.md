# FoG motion factorization v1: frozen development contract

Protocol/configuration identity: `fog-motion-factorization-v1`. Status: frozen before
any fit under this protocol. This is an adaptively chosen experiment on the already
consumed corrected FoG cohort, not confirmation, population superiority, demonstrated
novelty, or a modification of historical InclusiveHAR/CTGR/HERA/FoG scientific locks.

The specification is the preserved
`.audit/research_invention_review_20260908-001/EXACT_NEXT_EXPERIMENT.md`, SHA-256
`0310bed0c618031119df8b6df6161c84f777f15b32663ae5fb7f5ce31b7a9224`.
The executable contract is `configs/experiments/fog_motion_factorization_v1.yaml`.
Its canonical semantic SHA-256 is `4fd758eae61f8e505e64c421eff298b24d3b08cbf45bc5676fffcbb22dc1ec77`.
Calculate this identity by removing only the top-level `protocol_sha256` field,
serializing the parsed YAML mapping with UTF-8 JSON, `sort_keys=True`,
`separators=(',', ':')`, `ensure_ascii=False`, `allow_nan=False`, and hashing those
bytes. The configuration separately binds the complete final bytes of this protocol.
Run records must also retain both actual file-byte hashes. This defined exclusion avoids
a circular byte-hash claim; it does not exclude any experimental parameter.

## Question, cells, and immutable comparison population

The four distinct questions are whether a dedicated movement boundary helps, whether
learned signal structure adds value beyond two energy features, whether extra actual
history helps, and whether coarse history order helps. The new cells are:

| Cell | New component | Information visible to temporal residual |
|---|---|---|
| E2 | Two-feature logistic motion head | No temporal residual |
| T128 | Frozen E2 logit plus temporal residual | Final 128 samples; earlier 372 masked |
| T500 | Same initialized residual architecture plus frozen E2 | All 500 samples |
| T500-P | Same initialized residual architecture plus frozen E2 | Three reordered past blocks; final 128 intact |

T500 is the sole prospective primary candidate. Reuse sealed L9v, Lv, Bv, B0 and F3
probabilities as references; do not refit any reference. Every successful experiment has
five E2 fits and fifteen temporal fits: twenty fits total, one E2 checkpoint shared by all
three temporal cells within each outer fold. A fit attempt begins immediately before
invoking estimator fitting or the neural optimizer loop; every attempted failure consumes
one attempt. No attempt may be silently retried. A failed dependency does not license a
substitute cell; mark unattempted cells as such and the experiment incomplete.

Use the original 22 people, five exact outer folds, 1,939 observable candidates and 1,213
scoring rows with class counts [954,74,185] in [mobility,sitting,standing] order. The complete
participant-to-fold map is in the configuration and must match sealed reference IDs and
fold arrays. All cells use the same original outer-training people and the same common
full-history, scoring-eligible training rows. Seed is 11 plus zero-based outer-fold index;
there is no added seed, inner outcome selection, extra cohort, or InclusiveHAR P11-P20
payload access. P007 is an ordinary diagnostic participant, never a special training rule.

This package retains L9v/B0 posture and availability behavior. It is not a standalone ankle
product, since original B0 fallback requires the back sensor. No comparison with a source
CTGR percentage is a matched FoG improvement contrast. Frozen CTGR B6/B9/T9/U9 evidence
and its independent-cohort substantiation track remain separate.

## Source, history reconstruction, and zero-fit barrier

Accept an explicit evidence root. Resolve every configured input path relative to it and
reject absolute/traversing paths or links resolving outside that root. Reuse, without raw
copy or download, `.audit/fog_decision_team_review_20260907-001/inputs/sensor_data.csv`:
119,629,580 bytes, SHA-256
`888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477`.
Byte-hash the actual raw object and pinned reference files before computation. The
configuration binds the retained L9v predictions, feature cache, analysis, configuration,
validation, completion and artifact manifests plus the mandatory fallback erratum.

First reproduce the existing corrected physical inference population and current-query
signals exactly. Use inherited participant/session/timestamp and sensor-specific finite
monotonic-run segmentation, inherited SI conversion, native-rate causal 0.30 Hz gravity,
and one joint offline 60-to-50 Hz polyphase resampling pass. Do not reset at annotations
or current-window edges. Exact current six-channel, gravity, ankle80, derived-nine feature,
availability and ordered-ID hashes are mandatory under their inherited hash algorithms.
Do not substitute a superficially equivalent filter, phase, dtype or resampling method.
Independently validate the current scoring projection against the retained 1,213 IDs and
class counts. The inherited annotation-independent population and label projection remain
authoritative: annotations may determine supervision/scoring admission, not the physical
candidate grid, normalization support at inference, history starts, or sensor resets.

For a current 128-sample query with exclusive end index e, require exactly `[e-500,e)`
within the same finite ankle run. Its final 128 samples must equal the retained current
query; its first 372 are actual earlier recorded samples, not concatenated cached windows.
Past activity may differ naturally; the target remains the inherited current-query label.
This explicitly extends context beyond an activity-contained supervised query without
using activity changes as boundaries or model-visible cues. Partition people before this
construction; never bridge a participant/session/gap/nonfinite reset or borrow future
grid samples. Timestamps, IDs, clinical metadata, boundaries, and labels are not signal
features. IDs only select the content-independent permutation control described below.

Full history must reproduce the frozen feasibility counts: 1,724 observable and 1,098
scored queries. The 1,817 observable/1,154 scored ankle-available population is unchanged.
All four learned cells train and intervene only on this same full-history support. The
five training row totals are [904,780,902,780,1026], and original-class counts are
[[762,19,123],[641,28,111],[753,21,128],[619,28,133],[849,32,145]]. These are requirements
derived from retained offsets, not claimed new raw-replay measurements. Any mismatch
stops before the first fit; do not repair it by selecting a different length or population.
P017 has no scored full-history query and remains scored through exact L9v fallback.

The physical preflight, an independent source/context validator, configuration and split
checks, meaningful unit tests, and a clean committed implementation are required before
fitting. Preserve any failed preflight. GPU convolution causality is model-level only:
inherited offline resampling can look ahead in raw samples, so this protocol does not
establish zero-lookahead sensor streaming or measured deployment latency.

## Weights, E2, and shared normalizers

On each actual outer-training set after common support restriction, let n_ic count rows
for participant i and original class c, and k_i count that person's represented original
classes. Raw weight is `1/(k_i*n_ic)`; divide every raw weight by their unweighted mean.
Save each row weight, participant/original-class totals and regrouped binary totals. All
four cells use those identical weights. Only the loss target changes to `y == mobility`;
there is no second binary rebalance, class weighting, balanced sampler or inverse-frequency
correction. Effective mobility/stationary weight need not be one half per class.

E2 uses exactly cached `accelerometer_norm__rms` and `gyroscope_norm__rms` from the current
query in inherited units. For each feature apply `log(max(rms,1e-8))`; use unweighted
training-query means and population standard deviations (ddof=0), replacing zero scales
with one. Fit float64 LogisticRegression with L2, C=1, lbfgs, intercept, max_iter=1000,
tol=1e-6, no class weight and no warm start. Require trained class order [0,1] where 1
means mobility. E2 probability is `predict_proba` at class 1; its logit is
`decision_function`. Retain convergence warnings/status, iterations, coefficients,
normalizer and weighted data log-loss. Also report the explicit objective
`sum(w*BCE)/sum(w) + ||coef||^2/(2*C*sum(w))`; the intercept is unpenalized.
Nonconvergence is a failed fit, not permission to extend iterations or change C.

Temporal channel normalizers use unweighted valid samples from the common training
current 128-sample queries only, pooled over queries and time with float64 population
statistics. Do not estimate them from long histories, held-out participants, available
but ineligible training rows, labels within a wearer, or the full query distribution.
Use scale one for zero variance. The identical six statistics normalize all three temporal
arms; normalized signals become float32. There is no per-wearer normalization or calibration.

## Exact residual network and training

Every temporal cell has the identical 500-by-7 input shape: six ordered linear-acceleration
XYZ/gyroscope XYZ channels plus a binary observation mask. T500/T500-P masks are one.
For T128, the first 372 mask values are zero and the six corresponding normalized signals
are set to zero; the last 128 values are unchanged. No gravity channels enter the network.

A biased 1x1 Conv1d maps 7 to 32 channels, followed immediately by masking. Six residual
blocks use dilations [1,2,4,8,16,32]. Each block applies two identical sublayer forms:
biased 32-to-32 kernel-5 causal convolution with left padding 4*d, channel-only LayerNorm
(32 channels at each time, eps=1e-5, affine), exact GELU, dropout 0.1, then observation
mask. Add the block input after the second sublayer and mask again. No activation follows
that residual addition. Masking after each sublayer prevents convolution biases and
LayerNorm affine terms from populating unobserved prefix states internally. No operation
normalizes over time, batches or participants.

Take the arithmetic mean of the last 128 outputs, followed by a biased 32-to-1 linear
head with both weight and bias initialized to zero. Other parameters use default PyTorch
constructor initialization in one fixed source order. There are 62,881 trainable
parameters, excluding the frozen E2 coefficients, and a 505-sample causal receptive field.
Verify both parameter count and effective history dependency using synthetic tests;
verify zero-state and masked-prefix perturbation invariance, not only input masking.

All three temporal arms load the same saved initial state within a fold. For each query,
cast the frozen E2 float64 logit to float32, add the float32 scalar residual, and apply
float32 sigmoid. Export this m to float64 for composition. E2 remains frozen, and E2-only
probabilities remain float64; do not claim byte equality between float32 zero-residual
probabilities and E2 float64 probabilities. The initialization and conversion difference
are explicit, not an additional fitted calibration.

Train each residual for exactly 80 epochs with AdamW: lr=1e-3, betas=(0.9,0.999),
eps=1e-8, weight_decay=1e-4, amsgrad=False, foreach=False, fused=False. At zero-based epoch
e=0..79 set `lr = 1e-5 + 0.5*(1e-3-1e-5)*(1+cos(pi*e/79))`, so the final trained epoch
uses 1e-5. Batch size is 32, retain the last incomplete batch, data-loader workers zero,
float32 without mixed precision. Minibatch loss is `sum(w*BCEWithLogits)/sum(w)` over that
batch. Report the whole-training weighted loss with its whole-training denominator
separately. Clip total gradient L2 norm to 1.0 and reject nonfinite losses/gradients.
Keep the last epoch only; no early stopping, evaluation-based checkpoint, scheduler
restart, hidden inner sweep or post-hoc calibrator is allowed.

Precompute 80 training-row permutations using NumPy `default_rng(11+fold)` and reuse
identical row orders for all temporal cells. Reset Torch CPU/CUDA RNG to 11+fold after
loading the shared initialization and before each cell's training; identical tensor
shapes and operation order preserve the dropout RNG setup. Save initial-state, epoch-order
and relevant RNG hashes. E2 consumes no neural fit and is fit once per fold, followed
in fixed order by T128, T500, T500-P. Inference uses evaluation mode and batch size 32.

The inspected dependency contract is Python 3.11.9, NumPy 2.3.5, SciPy 1.17.1,
scikit-learn 1.8.0, Torch 2.12.0+cu132, PyYAML 6.0.3 and joblib 1.5.3. Verify versions,
interpreter, complete dependency inventory, implementation import path, GPU, driver and
CUDA in a hashed environment receipt; the worktree's name does not prove its environment.
Use one controller and one CUDA GPU, one Torch/intra/inter-op and BLAS/OpenMP thread,
deterministic algorithms and cuDNN, no cuDNN benchmarking, no TF32, and
`CUBLAS_WORKSPACE_CONFIG=:4096:8`. An unsupported deterministic operation is an explicit
failure, not a silent backend change. Model/checkpoint replay allows maximum absolute
neural probability error 1e-7, with exact decisions and IDs; report measured error.
Copied references and fallbacks remain byte-exact, with no tolerance relaxation.

## Reordered-history control and output composition

For T500-P, partition the first 372 preprocessed samples into three chronological
124-sample blocks with indices 0,1,2. Compute SHA-256 of the exact UTF-8 string
`motion-history-v1|{observable_window_id}|{block_index}` for each; use the inherited
window-ID string unchanged and decimal integer block index. Sort by raw digest bytes,
then integer index. Apply that block order jointly to all six channels and mask, in
both training and evaluation; leave final 128 samples unchanged. Save the per-candidate
three-index permutation, hashes and identity count. Keep identity permutations without
rehashing. Verify multisets and current-query identity. This changes coarse history
order and joins, preserves within-block chronology, and is not complete order removal
or a unique causal test of physiology. Filtering/resampling precedes this operation.

For each eligible query, q is sealed out-of-fold
`L9v_sitting/(L9v_sitting+L9v_standing)`; output `[m,(1-m)*q,(1-m)*(1-q)]` in float64.
Only motion mass is learned; posture ratio is unchanged. Training may not consume
held-out q or outer labels/predictions for fitting or selection. Reusing q for the
corresponding held-out query is frozen baseline composition, not extra supervision.

Apply fallback priority exactly: missing ankle -> entire B0 vector; ankle available but
less than 500 samples -> entire L9v vector; full history but exactly zero L9v posture
denominator -> entire L9v vector; otherwise compose. Never insert an epsilon to invent
posture evidence. Preserve 59 scored B0 and 56 scored short-history L9v fallback rows,
and report any additional denominator-zero fallback plus actual intervention count.
Resolve B0 through verified numeric method index 3; preserve the mandatory erratum for
historical truncated human-readable IDs. F3 is a separate back control and has no
ankle-fallback equality requirement. First produce all-observable predictions in inherited
order, then project to scoring rows; never discard unsupported or nonintervened people.
Decode by NumPy argmax, first maximum in fixed class order. Although q is fixed, entering
or leaving a stationary decision can change both posture recalls; harm guards still apply.

## Metrics, paired uncertainty and resource-allocation gates

Primary score is each person's macro-F1 over all three fixed labels with zero_division=0,
then the unweighted mean of all 22 people in lexicographic participant-ID order. Absent
classes remain zero; never rescale the missing-class ceiling. Report per-person support,
confusion, precision/recall/F1, every change, mean/median/quartiles, bottom seven, worst,
pooled accuracy, descriptive present-class F1, boundary rescues/harms/ties and all
availability/history/intervention/q-zero strata. Compare the reporter against retained
L9v values before interpreting new metrics.

Class recalls average all 22 person-specific recalls, with absent classes zero. Bottom
seven difference subtracts separately sorted seven-smallest means, not worst paired
changes. Worst-person difference is min(candidate)-min(comparator); maximum paired harm
is min(candidate_i-comparator_i), and is a separate guard. NLL clips only true-class
probabilities to [1e-12,1] when taking logs; saved probabilities/decisions/Brier do not
change. NLL and summed three-class Brier average windows within each person, then people.

Report pooled and equal-person ECE with ten equal-width [0,1] bins, left-inclusive and
right-exclusive except the last includes one; confidence=max class probability,
correctness=argmax matches label, and empty bins contribute zero. Report effective
composed mobility probability AUC pooled and averaged only over people with both binary
classes. Report binary sensitivity/specificity pooled and balanced accuracy averaged over
eligible people at m>=0.5, with exact eligible denominator. This diagnostic threshold is
different from the q-dependent three-class decision boundary. Binary calibration is
binary NLL/Brier and ten-bin confidence ECE with confidence=max(m,1-m) and the same bin
rules; include raw learner m separately from effective fallback-composed m. Conditional
posture AUC uses sitting as positive on true stationary rows with nonzero denominator;
per-person AUC requires both postures and discloses its denominator. It is expected to
remain unchanged when q is unchanged. High binary AUC cannot replace failed three-class F1.

Resample 22 paired participant vectors with replacement 10,000 times using NumPy
default_rng(1729), the same draws for every contrast. Use the 2.5/97.5 percentiles with
linear interpolation. These intervals are descriptive, conditional on adaptively reused
people and overlapping training folds, not independent confirmatory familywise inference.
Also omit each person and each complete outer fold in turn and report the remaining
equal-person mean paired difference. Strict wins have delta>1e-12; harms delta<-1e-12;
other changes are ties. Unchanged P017 is a tie, not a win or reason for exclusion.

Advance the following frozen sequence only until its first failed stage while still
reporting every numerical contrast:

| Stage | Contrast | Minimum mean F1 gain |
|---|---|---:|
| 1: practical improvement | T500 minus L9v | 0.015 |
| 2: learned information beyond energies | T500 minus E2 | 0.010 |
| 3: additional actual context | T500 minus T128 | 0.010 |
| 4: coarse order and changed joins | T500 minus T500-P | 0.010 |

Every stage also requires its paired 95% interval lower endpoint>0, at least 14 strict
participant wins, bottom-seven difference>=-0.010, worst-person difference>=-0.030,
minimum paired individual difference>=-0.050, mean participant recall changes mobility
>=-0.010 and sitting/standing each>=-0.020, every leave-one-person and leave-one-fold
mean difference>0.0, and complete source/alignment/fallback/checkpoint validation.
There is no tolerance added to the magnitude or interval thresholds. These are new
prospective resource-allocation gates; the paired-individual guard and interval criterion
must not be attributed retroactively to older L9v or other historical protocols.

Report all unordered pairwise contrasts among the four new and five retained methods,
with explicitly named orientation; include E2-minus-L9v with all stage-1 practical guards.
They do not permit selecting another primary winner or restarting the claim sequence.
If E2 is best, recommend a separately frozen simple-boundary check; if T128 matches T500,
do not claim extra history was needed. Stage 2 alone does not establish a temporal-order
mechanism; additional inputs, learned representation and optimization are a package.

## Artifact lifecycle, validation and stopping

Use an explicit output path in a create-only run directory, preferably under
`.audit/fog_motion_factorization`. Reject existing output, input/output aliasing and
unsafe resolved paths; never overwrite earlier evidence. Bind the clean code commit and
source byte inventory at preflight, every fit launch and completion. Save exact command,
config/protocol/environment snapshots, raw/reference byte hashes and the L9v erratum.
Save raw-replay segmentation/context provenance, qualification and current-window parity,
observable/scoring IDs, folds, full-history masks, common training rows and weights,
normalizers, permutation arrays, initial-state and batching hashes before fitting.

Each fit has a create-only attempt receipt written before fitting and an explicit terminal
receipt, convergence or training-loss record, model hash, source/inputs/fold/seed binding,
runtime, parameter count, model bytes and memory measurements. Preserve checkpoints for
all five E2 and fifteen residual models, or identify each missing/failed/unattempted item.
Training failure/nonconvergence/nonfinite values or interruption stops further fitting,
retains completed evidence and marks the run incomplete. There is no automatic retry,
iteration extension, replacement cell, fallback masquerading as a completed new model,
extra seed, other dataset or follow-on. A clean completion requires all twenty intended
fits and all validations, not merely a promising partial score. No arbitrary new time
limit is introduced here; the fixed twenty-attempt scope remains binding.

Retain all-observable and scored probabilities for all nine methods, raw learned motion
probabilities, effective compositions/fallbacks, errors, participant reports, every
comparison/interval/gate and runtime/peak-memory/size summary. Independently restore and
replay every checkpoint without fitting, verifying shared E2 hashes, fold membership,
normalizers, parameter count and all saved outputs. Validation must not trust a stored
success flag or scorer's own metric table. Use separate raw-context validation and
meaningful synthetic leakage/mask/composition/permutation/lifecycle tests; run required
configuration checks, tests, lint, type checks, split and artifact validators before an
implementation commit or experiment launch. Generated data/checkpoints remain outside Git.

Seal artifact and completion manifests using repository self-hash conventions and actual
file-byte hashes; retain errors and interruptions as first-class evidence. Stop and verify
all task-owned workers/monitors in a final shutdown receipt on success or failure. Do not
silently edit a sealed run to repair a report; create a disclosed successor if needed.
No publication/release, remote push, tag, DOI or deposit is authorized by this protocol.

The physical pilot is required only for a later local-reference stability/reattachment
claim, not this qualified offline motion experiment. Any pretraining or posture ablation,
including a matched HARNET pretrained-versus-scratch study, requires its own later frozen
protocol and is not an automatic extension. Independent confirmation requires separately
custodied new people after final training/inference choices are frozen; consumed source,
FoG, pilot or public participants cannot be relabelled as independent confirmation.
