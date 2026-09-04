# HERA-CTGR retrospective source-development result

HERA-CTGR v1 was implemented, frozen, and evaluated successfully. The best real variant was the
strict window lane at **0.86849 mean participant macro-F1**, a small **+0.00309** improvement over
frozen CTGR and **+0.02896** over unchanged RMRP. This is useful exploratory evidence, but it is not
a breakthrough: the predeclared gain and bottom-tail gates failed and the participant-bootstrap
interval crossed zero.

## Matched results

All rows below use the same 725 source windows, 10 participants, five outer participant folds, four
inner folds, and five seeds. Higher F1 is better; lower NLL/Brier is better.

| Method | Lane | Mean macro-F1 | Delta vs RMRP | Delta vs CTGR | Bottom 30% | Worst | NLL | Brier |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Unchanged RMRP | matched base | 0.83953 | 0.00000 | -0.02586 | 0.69941 | 0.54767 | 0.36889 | 0.22512 |
| Frozen CTGR | matched comparator | 0.86540 | +0.02586 | 0.00000 | 0.73990 | 0.57512 | 0.35426 | 0.21165 |
| Global trust blend | broad ablation | 0.85730 | +0.01776 | -0.00810 | 0.71466 | 0.60019 | 0.31873 | 0.19227 |
| CTGR + aggregate calibration | strict ablation | 0.86519 | +0.02566 | -0.00020 | 0.73773 | 0.57512 | 0.34017 | 0.20363 |
| CTGR + physics veto | strict ablation | 0.86540 | +0.02586 | 0.00000 | 0.73990 | 0.57512 | 0.35426 | 0.21165 |
| Frozen top-3 CTGR ensemble | strict ablation | 0.86727 | +0.02774 | +0.00188 | 0.73846 | 0.57512 | 0.35289 | 0.21028 |
| **HERA-CTGR strict** | **strict window** | **0.86849** | **+0.02896** | **+0.00309** | **0.73933** | **0.58028** | **0.34038** | **0.20303** |
| HERA context mean | transductive | 0.86201 | +0.02248 | -0.00339 | 0.72719 | 0.57331 | 0.34813 | 0.20894 |
| HERA context safe | transductive | 0.86592 | +0.02639 | +0.00052 | 0.73990 | 0.57512 | 0.34961 | 0.20875 |
| HERA full | transductive | 0.86516 | +0.02563 | -0.00023 | 0.73635 | 0.57512 | 0.33637 | 0.20174 |
| Matched random state | diagnostic only | 0.86652 | +0.02699 | +0.00112 | 0.73978 | 0.57512 | 0.33634 | 0.20096 |
| Label-informed three-way oracle | unattainable diagnostic | 0.88618 | +0.04665 | +0.02078 | 0.77445 | 0.60275 | 0.32891 | 0.19266 |

The historical 0.7514 few-person result is not a matched comparator: it uses a different protocol,
different inputs, and different evaluation units. It must remain in a separate table.

## What worked

The strongest new signal was frozen candidate uncertainty. Equal averaging of the first three
already-ranked CTGR candidates improved the mean in four of five seeds and reached 0.86727. Adding
conditional posture calibration and the physical veto produced the 0.86849 strict result. Strict
HERA improved eight of ten participants, including +0.01188 for participant 2, +0.01090 for
participant 8, and +0.00516 for the worst participant 10. It regressed participants 4 and 9 by
0.00567 and 0.00552 respectively.

Calibration was valuable for probability quality, not hard-label performance. On strict HERA, NLL
improved from 0.35426 to 0.34038 and Brier from 0.21165 to 0.20303. The physics veto rejected 46
strict-path windows. It rejected no original CTGR intervention, so the standalone CTGR-veto result
was exactly unchanged on this clean source set; its value remains as a fault guard to test on real
sensor degradation.

The conservative context controller was safe but weak. It selected BROAD for 292 window instances,
made four hard-label changes, rescued two, and harmed none. This yielded only +0.00052 over CTGR.
The variance-unaware controller routed much more aggressively and fell to 0.86201 with 6 rescues
and 16 harms.

## What failed

The predeclared full method did not outperform CTGR. Calibration after safe routing improved NLL and
Brier substantially, but changed the clean 2-rescue/0-harm profile into 9 rescues and 10 harms. Its
mean fell to 0.86516 and bottom-30% score to 0.73635. A matched random permutation of its state
counts scored 0.86652, so this dataset does not validate learned participant state identification.

The label-informed oracle confirms real headroom—0.88618 and +0.02078 over CTGR—but it uses held
participant labels after prediction and is not a deployable method. It repeatedly preferred OFF for
participants 4 and 8, and usually BROAD for participants 7, 9, and 10. HERA's safe controller mainly
identified participant 5 and missed those regimes. This isolates the remaining problem as responder
identification, not absence of complementary experts.

## Uncertainty and gates

For strict HERA minus CTGR, the descriptive participant bootstrap mean interval was
[-0.00095, 0.00713] and the bottom-30% interval was [-0.00552, 0.00899]. Seeds were averaged within
participant before resampling. The mean target (+0.010), bottom-tail target (+0.015), intervention
precision target (0.80), harmful-change limit (0.15), and breakthrough target (+0.020 with positive
uncertainty bounds) all failed. Worst-participant, posture-recall, NLL, Brier, and multi-participant
support checks passed.

## Implemented changes and evidence

HERA adds a gravity--gyroscope kinematic residual, a training-only physical reference and exact
fallback, fixed top-three probability marginalization, mobility-preserving posture calibration, a
15-feature unlabelled responder signature, participant-jackknife OFF/PULSE/BROAD utility models,
support-distance rejection, conservative PULSE fallback, strict/transductive separation, and
random/oracle diagnostics. The evaluator writes create-only, self-hashed result and prediction
artifacts.

The frozen method is tag `hera-ctgr-implementation-v1`; the outcome protocol is tag
`hera-ctgr-retrospective-protocol-v1`. The result record hash is
`4d296c9646c96dc5b1c6e392fefa7d7bed1c10c5534a9b4bb78eb8a7a0143bfe` and the prediction SHA-256 is
`8df7fa091abddc34c183d0f4855be8609f0db2ff0828dabdf257575b905660ad`.

## Claim boundary

This is retrospective development on the same participants 1--10 that motivated HERA. Participants
11--20 and DAGHAR were not loaded. The result supports retaining strict HERA as the next-cohort
candidate, but it does not support superiority, confirmation, breakthrough, or state-of-the-art
language. A new ability-relevant development cohort and a separately sealed confirmation cohort are
still required.
