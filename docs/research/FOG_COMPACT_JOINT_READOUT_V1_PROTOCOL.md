# Corrected FoG compact joint-readout protocol v1

Status: frozen prospective adaptive-development protocol when bound by the configuration and source snapshot. It does not modify historical protocols, results, fallbacks, or gates.

## Question

Test whether a compact three-class readout can improve corrected-FoG recognition when conditional posture is allowed to change, and whether sixteen train-only coordinates from the pinned frozen pretrained HARNET add value beyond the same small physical features and a matched seeded-random encoder.

## Evidence and support

Use exactly the frozen 1,939 observable queries, 1,213 scored queries, 22 participants, five participant folds, class order mobility/sitting/standing, and seed family 11 from the completed optional-context run. Do not access InclusiveHAR P11-P20. Fit only scored, current-left-ankle rows from outer-training participants. Predict current-left-ankle rows for held-out participants.

Retain exact B0 probabilities for missing-current-ankle queries and exact L9v probabilities where L9v stationary mass is zero. The editable scored set must remain 853 rows. Do not exclude a participant or query based on labels, outcomes, history length, difficulty, or an oracle diagnostic.

## Representations

S has exactly 12 columns: the two frozen query log-RMS energy values; the annotation-blind genuine-full-history availability bit; three components of mean unit derived-gravity direction over the 128-sample current query; and centered population covariance entries xx,xy,xz,yy,yz,zz of those unit directions. Zero gravity norm maps to a zero unit vector and is reported. These gravity values are derived from the existing ankle signal and are not additional measurements.

Standardize S's eleven continuous columns with unweighted outer-training current-query mean and population deviation, replacing an exactly zero deviation by one. Preserve the availability bit as zero or one.

R16 and P16 append 16 coordinates from the frozen random or pretrained HARNET embeddings. For each fold and representation, fit embedding standardization and deterministic full-SVD PCA on genuine-full-history scored outer-training rows only. Whitening is disabled. Divide PCA coordinates by their training population deviation; do not recenter a second time. Set all final coordinates for absent histories to exact zero. Qualify all ten transforms before any supervised fit. Rank failure ends the run without classifier fitting.

## Training

Fit one multinomial three-class L2 logistic classifier for each of S, R16, and P16 in each fold: C=1, lbfgs, intercept enabled, max_iter=1000, tol=1e-6, no class_weight, seed 11+fold, float64. Weight participant i/class c by 1/(k_i*n_ic), normalized over the fold training set to mean one. No tuning, retry, extra rank, calibration, threshold, seed, nonlinear head, encoder fit, or automatic continuation is allowed.

The finite workload is ten PCA fits followed by fifteen supervised fits. Save every attempt, transform, checkpoint, coefficient, index, weight, convergence and runtime.

## Outputs

Each representation yields two outputs from the same fitted three-class probability p:

- full: use p on editable rows;
- fixed: retain p_m but compose stationary mass with frozen q_L9v.

Raw p_m must match exactly within each pair. Hard mobility decisions may differ because q changes the largest stationary probability. Binary motion metrics at threshold 0.5 must match within each pair; three-class mobility recall need not. Conditional-posture metrics must use each output's actual probabilities.

## Endpoint and gates

Primary candidate is P16-full. Primary endpoint is fixed-three-class macro-F1 per participant, equally averaged over all 22 people, with absent-class F1 zero. Use 10,000 paired participant bootstrap resamples, seed 1729.

Claim order:
1. P16-full versus L9v: gain at least 0.015, positive interval lower endpoint, at least 14 strict wins.
2. P16-full versus M: gain at least 0.010, positive interval lower endpoint.
3. P16-full versus S-full: gain at least 0.010, positive interval lower endpoint.
4. P16-full versus R16-full: gain at least 0.010, positive interval lower endpoint.

Every comparison also requires bottom-seven difference >=-0.010, worst-score difference >=-0.030, minimum paired-person difference >=-0.050, mean participant recall differences mobility >=-0.010 and sitting/standing >=-0.020, every leave-one-person and leave-one-fold mean positive, and complete validation. Only stage one has the fourteen-win rule. Stop claim advancement at the first failure while reporting all six outputs and all contrasts. No alternate winner is automatically promoted.

Report participant rows, class support/recall/confusions, pooled accuracy, NLL, Brier, ten-bin ECE, binary motion metrics, conditional posture metrics, full/current/short/missing/q-zero strata, error overlaps, rescues/harms, matched-weight train/test loss, PCA diagnostics, parameter counts, fallbacks and runtime. Stratum reports are descriptive and do not replace the whole roster endpoint.

## Validation and claims

Before fitting, bind implementation/config/protocol/specification and all frozen inputs by hash; replay identities, masks, L9v and M; prove participant separation, training-only transformations, exact zero history handling, fallbacks, label-swap input invariance, finite probabilities, class order and gate reachability. The label-informed reachability calculation may validate the design only and cannot select any model input or outcome.

After fitting, replay all transforms and checkpoints with zero additional fits. Reproduce predictions, metrics, gates and manifests; verify exact within-pair p_m, exact fallbacks, all attempt counts, complete participant export, and task-worker shutdown.

This is repeatedly consumed adaptive-development evidence. A pass may freeze a candidate for independent confirmation. It is not confirmation, a population claim, a novelty demonstration, or permission to launch more experiments automatically.

