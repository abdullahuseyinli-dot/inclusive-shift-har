# Corrected FoG left-ankle derived-nine closure probe v1

Status: frozen before any L9v outcome-producing fit. This implements
`.audit/fog_spatial_team_review_20260907-001/NEXT_EXPERIMENT_SPECIFICATION.md`
and the 8 September 2026 team assessment. It is adaptive development on the
already consumed corrected FoG-STAR cohort, not confirmation or a novelty test.

## Question and one fitted cell

Test whether the static causal gravity deliberately omitted from Lv restores posture
information while retaining ankle dynamics. L9v concatenates the qualified left-ankle
six-channel linear-acceleration/gyroscope window with its already defined causal
three-axis gravity window and applies the existing deterministic 120-feature
derived-nine extractor.

Only L9v is fitted on the original five participant folds. Train on ankle-available,
scoring-eligible rows with the same participant-first weights as Lv. On every
ankle-unavailable evaluation candidate, copy the sealed fold-specific B0 probabilities
byte for byte. Reuse sealed Lv, Bv, B0, and F3 probabilities. L9v versus Lv changes
gravity information and representation dimension, so it is a package comparison rather
than a pure causal estimate of gravity.

## Frozen inputs and label-blind barrier

Use only the corrected FoG-STAR file with SHA-256
`888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477`,
the 1,939 existing candidate grid, 1,213 scored rows, 22-person roster, fixed five
participant folds, and the sealed spatial availability mask. This FoG roster is distinct
from the prohibited InclusiveHAR P11-P20 target; no InclusiveHAR target artifact may be
opened.

Before labels or outcome probabilities are loaded, independently rematerialize the
qualified left-ankle stream through the exact spatial path: SI conversion, finite
monotonic sensor-specific physical runs, causal 0.30 Hz gravity, one joint 60-to-50 Hz
polyphase resampling pass, and 128-sample non-overlapping candidate windows without
annotation or window resets. Require exact candidate IDs, availability, six-channel
signal, gravity, ankle80 features, source and reference hashes.

Freeze the L9v matrix and report finite rates, gravity norm scale, within-window gravity
direction stability, physical-run boundaries, and per-participant availability. These
descriptions have no outcome threshold. They may stop a corrupt measurement package,
but may not select or alter a feature, cutoff, threshold, sensor, or participant subset.

## Estimator, attempts, and resources

Use RandomForestClassifier with 500 trees, Gini criterion, `max_features="sqrt"`,
minimum leaf size two, bootstrap, no class weighting, random state 11 plus outer-fold
index, four fit workers, one prediction worker, and one BLAS/OpenMP thread. Recompute
mean-one weights `1/(m_i*n_ic)` on each actual training partition and require its row
count, participant exclusion, IDs, and weight hash to match Lv.

One controller runs folds 0 through 4 sequentially. Exactly five maximum attempts are
available, and every failed attempt counts. There is no tuning, retry, calibration,
feature selection, second sensor, second representation, or automatic follow-on. The
complete workload, including rematerialization, fitting, scoring, independent no-fit
checkpoint replay, manifests, and worker shutdown, has a hard 900-second cap.

## Reporting and advancement

Retain all 22 participants and 1,213 scored rows. Report fixed-three-class macro-F1
within participant and its equal-participant mean, pooled accuracy, NLL, Brier, class
support/precision/recall/F1, present-class sensitivity, participant confusion matrices,
quartiles, worst and bottom seven, every participant change and maximum harm,
rescues/harms/ties, available/fallback strata, 10,000 participant bootstrap resamples
at seed 1729, and leave-one-person and leave-one-complete-fold sensitivity.

L9v must independently pass against Lv, Bv, B0, and F3. Each comparison requires mean
participant F1 gain at least 0.015, at least 14/22 strict participant wins, bottom-seven
difference at least -0.01, worst-participant difference at least -0.03, mobility recall
difference at least -0.01, sitting and standing recall differences each at least -0.02,
and positive mean gain after omitting every participant and every complete fold. It must
therefore reach at least 0.6045943466990131 against Lv as well as pass every other gate.

Any failed comparison closes this fixed static ankle-gravity package on the consumed
cohort and recommends the separately planned physical-information pilot. A complete
pass freezes L9v and recommends only a separately authorized seed-stability check. It
does not authorize either follow-on, a larger architecture, publication, release, push,
tag, or deposit.

## Evidence

Use a create-only run directory. Preserve source/config/protocol snapshots, input and
reference hashes, measurement and partition preflights, feature names and hash, all five
attempt receipts/checkpoints, branch and fallback probabilities, full metrics and gates,
runtime, failures, independent checkpoint replay, manifests, and both worker-shutdown
receipts. No result may extend a cap or weaken a gate.
