# Corrected FoG GSP temporal-order ablation v1

Status: frozen exploratory development protocol. It does not modify or reopen any
historical gate, InclusiveHAR target cohort, external publication queue, or physical
pilot.

## Question

The corrected FoG RF factorial evaluated 80/120 ordinary engineered summaries. The
successful InclusiveHAR source lineage used a richer 1,400-feature six-channel
Geometric Spectral Pyramid (GSP). This experiment asks:

1. Does raw six-channel GSP improve a participant-weighted Random Forest over the
   exact corrected FoG F1 control?
2. Does any GSP package benefit depend on retaining original within-window order?

This is a finite representation test. It is not a neural architecture search, sensor
study, clinical validation, or proof that activity information is present or absent.

## Frozen data and folds

Use the pinned FoG-STAR v3 object with SHA-256
`888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477`.
The loader must reproduce all 22 participants, the pre-window participant plan, five
outer folds, 1,939 observable candidates and the post-hoc projection to 1,213 scored
windows. Candidate extraction remains independent of activity annotations.

Continuous processing remains unchanged: provider-session/timestamp/finite segments,
causal gravity separation at native 60 Hz, offline 60-to-50 Hz resampling, and
128-sample windows. The model input contains the existing six linear-acceleration and
gyroscope channels. "Raw GSP" means that no additional denoiser, gravity lane,
normalization, clipping, or learned preprocessing is inserted.

## Cells and execution order

Run exactly A, then B, then C, across all five folds:

- A: ordinary 80-feature six-channel extractor and participant-weighted RF.
- B: existing 1,400-feature GSP on original-order six-channel windows and identical RF.
- C: identical GSP/RF after one deterministic joint sample-tuple permutation within
  every observable window.

All cells use 500 trees, `max_features=sqrt`, minimum leaf 2, bootstrap, no automatic
class weight, four fit workers, one prediction worker, and random state
`11 + outer_fold_index`. Training weights are `1/(m_i*n_ic)`, normalized to mean
one on eligible outer-training rows. IDs, fold assignments, timestamps, labels and
positions are provenance only.

A is fitted fresh. Before any fit, its 80-feature cache must exactly equal the validated
reference cache. After its five fits, its observable probability array must exactly
equal the validated F1 array. An unexplained mismatch terminates the run before B/C.

The first registered A-fold fit is the only timing benchmark and counts toward the
15-fit limit. Runtime settings are frozen before it; scores cannot alter them.

## Joint permutation

For each observable window ID and sample index 0 through 127, hash the exact UTF-8 text

`fog-gsp-order-ablation-v1|{observable_window_id}|{index:03d}`.

Sort indices by the 32-byte SHA-256 digest and then numeric index. Apply the resulting
index order jointly to all six channels, independently inside each window, after
continuous preprocessing and before all GSP construction. Apply the identical rule to
training and evaluation candidates. Save the 1,939 by 128 int16 matrix and its hash.
No other permutation seed or replicate is permitted.

The transformation must preserve every six-dimensional sample tuple exactly and must
never cross candidate boundaries. Float64 global channel means and covariance matrices
must agree within absolute tolerance 1e-10. GSP's existing float32 conversion for
invariant streams remains unchanged. The permutation intentionally alters local
subdivisions, derivatives, spectra, autocorrelation, wavelets and other order-dependent
features. It also creates artificial discontinuities. B-C therefore measures dependence
on original sample order for this whole GSP/RF package, not a specific physiological
rhythm or all temporal information. Continuous preprocessing can already encode some
history into individual sample values.

Because `max_features=sqrt` is unchanged, A and B sample roughly different numbers of
features at a split. B-A is explicitly a representation/feature-sampling package
comparison. B-C matches feature dimension and estimator settings.

## Reporting and gates

Primary endpoint: fixed-three-class macro-F1 within each participant, averaged equally
over all 22 roster participants. Preserve zero contributions for missing true classes.
Also report every participant, fixed class support, present-class sensitivity,
quartiles, median, bottom 30%, worst participant, per-class precision/recall/F1,
pooled confusion and accuracy, NLL, multiclass Brier, rescue/harm/tie events, feature
dimensions, model sizes and timings.

Use paired participant bootstrap with 10,000 resamples, seed 1729. Windows, sessions
and repeated folds are not independent inference units.

For B-A practical promotion, require every frozen condition:

- mean gain at least 0.015;
- at least 14/22 participant wins;
- bottom-30 difference at least -0.010;
- worst-participant difference at least -0.030;
- participant-macro mobility recall difference at least -0.010;
- sitting and standing recall differences each at least -0.020;
- every leave-one-participant-out mean above 1e-12;
- all 22 people scored and validation complete.

For B-C order-mechanism promotion, require mean gain at least 0.010, positive lower
endpoint of the participant-bootstrap 95% interval, at least 14/22 participant wins,
every leave-one-participant-out mean above 1e-12, all 22 people scored, and validation
complete.

These are resource-allocation gates, not confirmatory significance claims. B can pass
the practical gate while the order mechanism fails. Conversely, evidence that order
matters does not promote a practically inferior model. Failure closes this finite
GSP/RF question; it does not establish failure of every representation or temporal
model. No subset-selected promotion is allowed.

## Evidence and resource controls

Run only from a clean committed source tree. Write to one create-only directory below
`.audit/fog_gsp_order_ablation`. Bind source, code, config, protocol, reference
control, features, permutation, weights, folds, checkpoints and predictions by hashes.
Save predictions over all observable candidates before projecting scoring eligibility.
Independent validation replays all checkpoints from the cached matrices without source
download or new fitting.

The complete workload allows at most 15 fit attempts and two compute hours, including
source loading, feature extraction, fitting, prediction and technical failures. A
failure that consumes an attempt cannot be replaced beyond 15. Preserve an incomplete
failure record at the cap. One controller owns execution; no GPU, extra seed, alternate
feature, estimator, threshold, permutation, target cohort, or dataset is authorized.

After validation, stop. No architecture follow-on, physical collection, publication,
push, tag, DOI, or release operation is automatic.
