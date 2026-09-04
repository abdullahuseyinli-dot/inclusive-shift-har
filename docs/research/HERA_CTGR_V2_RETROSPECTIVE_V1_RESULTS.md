# HERA-CTGR v2 retrospective source-development result

HERA-CTGR v2 was implemented, frozen before outcome access, and evaluated successfully. It reached
**0.86749 mean participant macro-F1**. That is **+0.11609** above the historical 0.7514 figure,
**+0.02796** above the matched RMRP base, and **+0.00209** above frozen CTGR, but **-0.00100** below
strict HERA-CTGR v1. The predeclared advancement and breakthrough gates failed. The current best
real method under this matched retrospective protocol therefore remains strict HERA-CTGR v1 at
0.86849.

## Matched result table

All non-query rows use the same 725 source windows, 10 participants, five outer participant folds,
four inner folds, and five fixed seeds. Higher F1 is better; lower NLL/Brier is better.

| Method | Role | Mean macro-F1 | Delta vs CTGR | Bottom 30% | Worst | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|---:|
| Unchanged RMRP | matched base | 0.83953 | -0.02586 | 0.69941 | 0.54767 | 0.36889 | 0.22512 |
| Frozen CTGR | matched comparator | 0.86540 | 0.00000 | 0.73990 | 0.57512 | 0.35426 | 0.21165 |
| Frozen HERA-CTGR v1 strict | prior best real | **0.86849** | **+0.00309** | 0.73933 | **0.58028** | 0.34038 | 0.20303 |
| Equal top-3 CTGR | ablation | 0.86727 | +0.00188 | 0.73846 | 0.57512 | 0.35289 | 0.21028 |
| Stability-weighted top-3 | v2 ablation | 0.86727 | +0.00188 | 0.73846 | 0.57512 | 0.35292 | 0.21031 |
| Temperature-only core | v2 ablation | 0.86727 | +0.00188 | 0.73846 | 0.57512 | **0.32721** | **0.19857** |
| Decision-separated core | v2 core | 0.86749 | +0.00209 | 0.73907 | 0.57978 | 0.32830 | 0.19912 |
| Dual-frame global candidate | v2 ablation | 0.86704 | +0.00165 | **0.74009** | 0.57978 | 0.32824 | 0.19932 |
| Rescue/harm logits only | v2 sentinel | 0.86749 | +0.00209 | 0.73907 | 0.57978 | 0.32830 | 0.19912 |
| Rescue/harm + physics | v2 sentinel | 0.86749 | +0.00209 | 0.73907 | 0.57978 | 0.32830 | 0.19912 |
| **HERA-CTGR v2 full** | **frozen v2** | **0.86749** | **+0.00209** | **0.73907** | **0.57978** | **0.32830** | **0.19912** |
| One-query matched core | query-excluded control | 0.86824 | +0.00284 | 0.74013 | 0.57978 | 0.32747 | 0.19848 |
| One-query personalization | labelled diagnostic | 0.84987 | -0.01553 | 0.68852 | 0.54918 | 0.34784 | 0.21593 |
| Participant oracle | unattainable diagnostic | 0.87973 | +0.01433 | 0.75169 | 0.58435 | 0.34641 | 0.20589 |
| Window oracle | unattainable diagnostic | 0.88831 | +0.02292 | 0.76232 | 0.59363 | 0.29110 | 0.17142 |

The 0.7514 figure is shown only as historical context. It came from a different few-person
experiment in which target-group participants entered training; it is not a protocol-matched
predecessor. The matched scientific comparisons are RMRP, CTGR, and strict HERA v1 above.

## What worked

The strongest v2 contribution was rank-preserving calibration. Every outer fold selected
temperature 0.75. It left macro-F1 unchanged but improved NLL by 0.02571 and Brier by 0.01174
relative to the stability-weighted ensemble. Against strict HERA v1, full v2 improved NLL by
0.01209 and Brier by 0.00392 despite its 0.00100 F1 loss. This is useful calibration evidence, not a
classification breakthrough.

Separating probability calibration from the sitting/standing decision offset also behaved as
intended. The rank-locked offset added 0.00022 mean F1, 0.00061 bottom-tail F1, and 0.00466 to the
worst participant over temperature alone, with zero mobility-membership changes. The exact fallback
and mobility lock therefore passed their safety contracts.

The dual-frame posture representation exposed some complementary information. Relative to the
decision core, its bottom-30% score was 0.00103 higher. Across the five seed prediction arrays, its
13 hard disagreements contained six rescues and seven harms. This explains both the oracle headroom
and why global use was not safe: the candidate lost 0.00045 mean F1 as a global replacement.

## What did not work

Stability-weighted marginalization could not exploit the top-three candidates. Mean candidate
Jensen-Shannon divergence was only 0.000257 on evaluation folds; weighted and equal averaging had
identical hard metrics, while weighting made NLL and Brier fractionally worse. These were effectively
near-duplicate experts, not a useful ensemble.

All three rescue/harm routing lanes fell back to the decision core in all 25 outer folds. Twenty-two
folds had fewer than five training responders; the remaining three had no threshold meeting the
frozen precision/harm gate. Consequently, v2 routed zero of 3,625 seed-window opportunities. The
fallback prevented harm, but intervention precision and harmful-change fraction were undefined, so
their advancement gates correctly failed rather than being treated as perfect.

One-query personalization was harmful. Only seven of 50 participant-seed instances yielded an
eligible query; the final state counts were 44 OFF, four NORMAL, and two INVERTED. Against the
query-excluded matched core, personalization lost 0.01837 mean F1, 0.05161 bottom-tail F1, and
0.03060 worst-participant F1. It must not be promoted.

The causal bout accumulator is implemented but unevaluated because the released source has no
authentic session, trial, timestamp, or bout order. Assigning it a score would invent evidence.

## Uncertainty and gates

Full v2 minus CTGR was +0.00209, with a descriptive familywise 97.5% participant-bootstrap interval
of [-0.00263, 0.00704]. Full v2 minus strict HERA v1 was -0.00100 with interval
[-0.00253, 0.00035]. V2 improved CTGR in three of five seeds and four participants, but improved
HERA v1 in only two seeds and one participant.

The +0.010 mean-over-HERA-v1 target, +0.015 bottom-tail-over-CTGR target, intervention precision,
harm fraction, positive participant interval, and +0.020 breakthrough target all failed. Mobility
preservation, posture/worst-participant non-inferiority, NLL, and Brier gates passed. Neither the
advancement gate nor the familywise breakthrough gate passed.

## Interpretation and next evidence

The participant oracle reached 0.87973 and the window oracle 0.88831, respectively +0.01224 and
+0.02082 above full v2. Thus, complementary predictions remain, but this ten-participant source does
not provide enough responder events to learn a reliable label-free router. Retuning sentinel
thresholds on these exhausted outcomes would not solve that evidence problem and would invalidate
the frozen test.

Retain the temperature calibration, rank-locked decision separation, exact fallback, mobility lock,
and dual-frame expert as an ablation. Do not promote stability weighting, the sparse meta-router,
one-query state selection, or bout performance. The next legitimate experiment needs a new ordered
multi-participant development cohort with authentic sessions and enough predeclared expert
disagreements, followed by a separately sealed confirmation cohort.

## Reproducibility and claim boundary

The method was frozen at commit `5def59c1bf1c9276a10b1fccbf7b023975d962a2`, tag
`hera-ctgr-v2-implementation-v1`. The outcome evaluator was frozen before access at commit
`227331de47de103a122d4e9a462300fc73967b42`, tag
`hera-ctgr-v2-retrospective-protocol-v1`. Its pre-outcome gate passed 737 tests plus formatting,
lint, strict typing, manifest, split, configuration, artifact, lockfile, and Git-integrity checks.

The result record self-hash is
`0196096c7ca8dfdce61a6be9dfd39f82467dc661ed9a3b5844e471a5cd5e94ec`; the prediction SHA-256 is
`5bbf25ef354f478278dd9c5093bf21df84fc88650cb9a7548d4dadce3f7863be`.

This is reused-source hypothesis generation on participants 1--10. Participants 11--20 and DAGHAR
were not loaded. It does not support independent validation, confirmation, superiority,
breakthrough, or state-of-the-art language.
