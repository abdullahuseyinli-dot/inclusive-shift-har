# CAGE-HAR v1 fully nested retrospective source result

Status: complete reused-source development result; not independent validation (2026-09-04).

## Outcome

The available nine-channel InclusiveHAR source data were sufficient to evaluate CAGE-HAR. The
fully nested run used 725 windows from participants 1--10, five fixed seeds, five outer folds and
four inner folds per outer partition. RMRP, the gravity expert and CTGR replayed their preserved
reports exactly. Participants 11--20 and DAGHAR were not loaded.

CAGE-HAR reached 0.8465 mean participant macro-F1. It improved the matched uncalibrated RMRP by
0.0070 and the decision-repaired base by 0.0050, but it remained 0.0189 below CTGR. It therefore did
not produce the intended breakthrough and failed the predeclared advancement gate. CTGR remains the
strongest matched human-data method at 0.8654.

## Five-seed results

| Method | Mean participant macro-F1 | Bottom 30% | Worst participant | NLL | Brier |
|---|---:|---:|---:|---:|---:|
| RMRP, uncalibrated | 0.8395 | 0.6994 | 0.5477 | 0.3689 | 0.2251 |
| RMRP, decision repaired | 0.8415 | 0.6970 | 0.5413 | 0.3535 | 0.2190 |
| **CAGE-HAR** | **0.8465** | **0.7246** | **0.5944** | **0.3413** | **0.2090** |
| Confidence trust gate | 0.8567 | 0.7187 | 0.5444 | 0.3442 | 0.2103 |
| Disagreement trust gate | 0.8573 | 0.7147 | **0.6002** | 0.3417 | 0.2090 |
| Global trust blend | 0.8573 | 0.7147 | **0.6002** | **0.3187** | **0.1923** |
| **Frozen CTGR** | **0.8654** | **0.7399** | 0.5751 | 0.3543 | 0.2116 |

Global and disagreement routing produced the same hard labels but different probabilities. The
global blend supplied the best NLL and Brier, while CTGR supplied the best mean and bottom-tail
macro-F1. CAGE improved its proper scores and worst participant relative to both RMRP and CTGR, but
those gains did not compensate for its lower overall and lower-tail classification score than CTGR.

| Seed | RMRP | CAGE-HAR | CTGR | CAGE minus RMRP | CAGE minus CTGR |
|---:|---:|---:|---:|---:|---:|
| 11 | 0.8379 | 0.8501 | 0.8647 | +0.0122 | -0.0146 |
| 23 | 0.8412 | 0.8472 | 0.8628 | +0.0060 | -0.0156 |
| 47 | 0.8420 | 0.8430 | 0.8664 | +0.0011 | -0.0234 |
| 89 | 0.8395 | 0.8437 | 0.8654 | +0.0043 | -0.0217 |
| 131 | 0.8371 | 0.8486 | 0.8676 | +0.0115 | -0.0189 |
| **Mean** | **0.8395** | **0.8465** | **0.8654** | **+0.0070** | **-0.0189** |

CAGE improved RMRP in all five seeds, but CTGR exceeded CAGE in all five seeds.

## Advancement gate

| Gate | Required | Observed | Result |
|---|---:|---:|---|
| Mean gain over repaired base | at least +0.0200 | +0.0050 | Fail |
| Bottom-30% change | at least 0 | +0.0276 | Pass |
| Intervention precision | at least 0.80 | 0.6944 | Fail |
| Harmful fraction of changed decisions | at most 0.15 | 0.2973 | Fail |
| Engineering score | at least 0.875 | 0.8465 | Fail |
| Stretch score | at least 0.884 | 0.8465 | Fail |
| Separate clean health-expert regression | required | unavailable | Not evaluated |
| Real corruption-curve gain | at least +0.015 | unavailable | Not evaluated |

The candidate does not advance. Passing the bottom-tail gate alone is insufficient.

## What the router actually did

Across 3,625 seed-window opportunities, CAGE routed 861 (23.75%) but changed only 37 hard labels
(1.02%). There were 25 rescues, 11 harms and one change where both predictions remained wrong. The
net gain was 14 correct windows; rescue precision among decisive changes was 0.6944.

Every hard change occurred in only four participants:

| Participant | Routed | Changed | Rescues | Harms | Net |
|---:|---:|---:|---:|---:|---:|
| 4 | 93 | 3 | 0 | 3 | -3 |
| 8 | 83 | 8 | 0 | 8 | -8 |
| 9 | 238 | 12 | 11 | 0 | +11 |
| 10 | 94 | 14 | 14 | 0 | +14 |

Participants 1, 2, 3, 5, 6 and 7 had no hard-label change. This reveals the central failure. The
same label-free window-level conditions identify genuinely useful interventions for participants 9
and 10 and harmful interventions for participants 4 and 8. The fitted linear advantage router did
not reliably infer the participant-level responder state from only eight outer-training
participants. Its conservative
jackknife bound also suppresses CTGR's useful changes for participants 2, 3 and 7.

The method is therefore simultaneously too permissive for the two adverse responder patterns and
too conservative for several positive responders. More window-level thresholding is unlikely to
solve both errors.

## Descriptive participant uncertainty

These analyses were performed after the result and are not predeclared tests. Seed-averaged CAGE
improved six of ten participants versus RMRP. Its paired mean difference was +0.0070 with a
participant-bootstrap 95% interval of [-0.0069, 0.0207], exact sign-flip p=0.3574 and Wilcoxon
p=0.3750. Versus CTGR, CAGE improved three participants and reduced seven; the mean difference was
-0.0189 with interval [-0.0411, 0.0026], sign-flip p=0.1426 and Wilcoxon p=0.1934. These values do
not establish superiority or inferiority.

## Research conclusion

CAGE-HAR is not the next winning architecture in its present form. The valuable findings are:

- fixed gravity assistance remains strong, confirming CTGR rather than replacing it;
- trust-region blending materially improves probability quality;
- CAGE can improve the weakest participant and reduce some CTGR harms;
- window-level linear advantage prediction is the wrong granularity for the observed responder
  structure.

The next invention should keep CTGR or the fixed trust blend as the probability engine and add a
participant/session-level latent responder state learned only from unlabelled aggregates, with a
safe fallback to CTGR. Development must target the p4/p8 versus p9/p10 separation and explicitly
test whether a participant-level gate recovers CTGR's gains for p2/p3/p7 without recreating the p4
and p8 harms. This is future reused-source hypothesis generation; an independent new cohort remains
mandatory for any advancement or state-of-the-art claim.

The protocol-freeze tree passed 696 repository tests, Ruff lint and formatting, mypy, manifest
validation, artifact validation, the conditional released-block split audit and dependency-lock
validation. After reporting, 45 focused result/protocol/CLI/documentation tests passed. The five
preserved RMRP, gravity-expert and CTGR reports replayed exactly.

The create-only full record is
`.audit/v4/cage-har/retrospective-source-v1-001/result.json` (SHA-256
`16714500e15aba66abd5f78e590f74834a98e924e04eea1d33e04a41cf670d5f`, record SHA-256
`1ca5123d79a712b460a690f96723890a91595330777c9608bb5ced59bbefeea6`). The preserved prediction
artifact SHA-256 is `6af02649ca4435a8402374ccdc83f68e3845d55ea8ef990bd16c0bc5eef57686`.
