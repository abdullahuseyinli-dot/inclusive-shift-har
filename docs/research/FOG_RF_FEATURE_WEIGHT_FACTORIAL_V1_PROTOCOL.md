# Corrected FoG Random Forest feature-by-weighting factorial v1

Status: frozen before the four-cell outcome on 7 September 2026.

This short development experiment resolves one specific confound left by the superseded
FoG PB-RF-D9 diagnostic: whether its apparent difference came from derived-gravity
features, participant-first weighting, or their interaction. It is not an invention
test, an external confirmation, a zero-shot transfer result, or a continuation of the
failed prompted-posture mechanism. The normative contract is
`configs/experiments/fog_rf_feature_weight_factorial_v1.yaml`.

## Corrected data contract

Use the complete 22-person FoG-STAR v3 provider roster. The existing loader must form
the participant fold plan from provider subject identifiers before session segmentation,
resampling, window creation, annotation projection, or scoring eligibility. Preserve
participants with zero eligible windows in the audit. If any roster participant has no
scored row, the complete matrix may still be recorded, but the promotion decision is
`incomplete`; that person is never silently dropped or counted as a tie.

Physical signal segments are defined only by participant, session, timestamp gaps, and
finite sensor runs. Activity, task, FoG, and severity annotations must not influence
resampling, causal gravity, candidate starts, features, folds, or prediction calls.
Activity annotations are projected after the complete observable candidate grid exists
and determine only labels and scoring eligibility. The source is streamed and verified
against the pinned size, MD5 and SHA-256; no raw local mirror is created.

The three classes, in exact probability-column order, are mobility, sitting and standing.
Six-channel means linear acceleration plus gyroscope. Derived-nine adds the existing
causal low-pass gravity estimate from total acceleration. It is not recorded native
gravity and must remain a separately named lane.

## Fixed factorial

Use `RandomForestClassifier` with 500 trees, square-root feature sampling, minimum leaf
size two, bootstrap sampling, and four fit workers. Predict with one worker so tree
probabilities are reduced in a deterministic order and checkpoint replay is byte exact.
There is one run seed, 11,
five pre-window participant folds, and exactly 20 fits. Within outer fold `k`, every cell
uses random state `11+k` so representation and weighting comparisons share estimator
randomness. No tuning, validation selection, early stopping, retry selected by score, or
old feature cache is permitted.

| Cell | Engineered feature lane | Weighting |
|---|---|---|
| F0 | six-channel | `class_weight=balanced_subsample` |
| F1 | six-channel | participant-first sample weights |
| F2 | derived-nine | `class_weight=balanced_subsample` |
| F3, primary | derived-nine | participant-first sample weights |

For participant-first weighting, for every observed outer-training participant `i` and
class `c`, let `m_i` be the number of that participant's present classes and `n_ic` its
row count. Each row receives raw weight `1/(m_i*n_ic)`, normalized to mean one over all
outer-training rows. Missing cells create no samples. Automatic class weighting is off.
The ordinary cell uses scikit-learn's `balanced_subsample` implementation and no explicit
sample weight. The weighting mechanism must be stated in every result; this is not a
pure representation comparison.

Deterministic features are computed once per lane over every observable candidate and
cached in the new run. Learned models remain inside outer folds. Every evaluation fold
is predicted over its complete observable candidate population before post-hoc scoring
indices are applied.

## Metrics and decision

The primary endpoint is fixed-three-class macro-F1 within each roster participant,
followed by an equal mean over all 22 people. Use zero for an absent class within an
otherwise scored participant. Also retain every participant, class support, bottom 30%
(the lowest seven participants), worst participant, median/quartiles, participant-macro
class recall, pooled precision/recall/F1/confusion, participant-averaged and pooled NLL
and multiclass Brier score, model size, feature dimension, and stage timing.

Select F3's strongest control from F0/F1/F2 by mean participant F1, with absolute tie
tolerance `1e-12` and priority F0, then F1, then F2. Use 10,000 paired participant
bootstrap draws from NumPy PCG64 seed 1729. Report all participant deltas, wins, harms,
ties, and all leave-one-participant-out mean differences. Report four simple-effect
factorial contrasts:
derived-nine minus six-channel under each weighting, participant-first minus ordinary
under each feature lane, plus the difference-in-differences interaction.

F3 passes only if every condition holds against the selected control: mean gain at least
0.015; bottom-30 difference at least -0.010; worst-person difference at least -0.030;
mobility recall at least -0.010; sitting and standing recalls each at least -0.020; at
least 14 of 22 people improve; every leave-one-person-out mean exceeds `1e-12`; and all
22 roster people have scored rows. These are development resource-allocation gates, not
proof of superiority.

If another cell is strongest, retain it as a conventional baseline. A failure closes
this factorial without changing weights, features, folds, gates, or seed. Do not launch
seeds 23/47, neural models, transfer, another external dataset, or the canceled
publication queue automatically.

## Evidence and execution

Run from a clean committed worktree. Before outcomes, bind code, config, protocol,
environment, dependency versions, source receipt, participant plan, data hashes,
feature names, estimator parameters and effective fold seeds. Create-only evidence must
include config/protocol snapshots, preflight, data/source audits, feature cache, all 20
trusted-local model checkpoints, full observable and scored probabilities, fold reports,
participant metrics, comparisons, gate, timings, result/failure, validation and a hash
manifest.

The validator must work without downloading the source. It verifies every artifact and
checkpoint hash, replays each checkpoint on cached held-out features, reconstructs all
scored probabilities, metrics, comparisons and the gate, and confirms the full roster,
fold and fit count. Stop all task-owned workers at completion or timeout.
