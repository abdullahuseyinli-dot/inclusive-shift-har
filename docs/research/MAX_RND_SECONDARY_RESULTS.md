# Max R&D secondary-lane results

Status: complete source-development experiments; CTGR passed its prospective advancement gate; none of these results is an independent confirmatory target result (2026-09-04).

## Outcome

The exact plan produced one strong architectural advance, one useful personalization result, and one modest robustness improvement.

| Lane | Main outcome | Decision |
|---|---:|---|
| Confidence-Triggered Gravity Residual (CTGR) | 0.8654 mean participant macro-F1 versus 0.8395 matched RMRP | Passed all nine locked advancement checks; advance to independent validation as a nine-channel method |
| Active Semantic-Gauge Sentinel (ASGS) | 0.8569 GSP and 0.8518 RMRP with about one queried label per participant | Useful labelled personalization; never call zero-shot |
| Explicit-provenance reconstruction | 0.7459 versus 0.7426 averaged over corrupted cases | Retain as a deterministic robustness wrapper; improvement is small and corruption-specific |

The implementation was committed as `6c1f799f2610a3aa8491a134802d37ec078a0c17` and tagged `max-rnd-secondary-v1` before any outcome-producing run. The pre-run gate passed 645 tests plus formatting, lint, strict typing, manifest, split, artifact, lockfile, Git-integrity, protocol-hash, and CLI checks. Participants 11--20 and target predictions were not reopened.

## CTGR: the breakthrough signal

CTGR leaves confident RMRP predictions unchanged. On low-confidence windows, a fold-selected gravity expert may only redistribute the sitting/standing mass; RMRP retains control of the mobility probability. Selection used the four participant-exclusive inner folds of each outer training partition and was frozen before outer evaluation. The five fixed seeds then reused that fold map.

| Metric, five-seed mean | Flat RMRP | CTGR | CTGR minus RMRP |
|---|---:|---:|---:|
| Mean participant macro-F1 | 0.8395 | **0.8654** | **+0.0259** |
| Bottom-30% participant macro-F1 | 0.6994 | **0.7399** | **+0.0405** |
| Worst participant macro-F1 | 0.5477 | **0.5751** | **+0.0275** |
| Mobility recall | 0.9760 | **0.9767** | +0.0008 |
| Sitting recall | 0.7162 | **0.7735** | **+0.0573** |
| Standing recall | 0.8318 | **0.8481** | +0.0163 |
| Negative log-likelihood | 0.3689 | **0.3543** | **-0.0146** |
| Multiclass Brier score | 0.2251 | **0.2116** | **-0.0135** |

The standalone gravity posture expert scored 0.8320 mean, 0.6221 bottom-30%, and 0.5307 worst-participant macro-F1—below RMRP on all three. CTGR's result is therefore not explained by simply replacing RMRP with a stronger gravity classifier. The useful behavior is the invented residual interaction: retain the strong base and invoke complementary posture evidence on uncertain windows.

| Seed | Flat RMRP | CTGR | Difference |
|---:|---:|---:|---:|
| 11 | 0.8379 | 0.8647 | +0.0268 |
| 23 | 0.8412 | 0.8628 | +0.0216 |
| 47 | 0.8420 | 0.8664 | +0.0245 |
| 89 | 0.8395 | 0.8654 | +0.0260 |
| 131 | 0.8371 | 0.8676 | +0.0304 |

CTGR won on all five seeds and on an average of 3.4 of five outer folds. Its across-seed standard deviation was only 0.0018. The selected trigger affected roughly 12--21% of a fold's windows, so the gain did not come from replacing the base globally. Every selected expert used an Extra Trees minimum leaf size of three; three folds selected the reconstructed-total GSP, one selected the direct physics representation, and one selected dual dynamic/total GSP.

The participant-level picture is promising but not conclusive. After averaging each participant over the five seeds, CTGR improved eight of ten participants and reduced two. The mean paired gain was 0.0259; a post-run participant bootstrap interval was [-0.0005, 0.0489], the unadjusted exact sign-flip p-value was 0.0868, and the unadjusted exact Wilcoxon p-value was 0.1602. These analyses were not part of the predeclared gate and do not establish statistical superiority. They expose the remaining uncertainty from only ten reused development participants.

Most importantly, CTGR is a nine-channel result. It uses the three recorded Core Motion gravity channels in addition to the locked six-channel user-acceleration/rotation-rate interface. It therefore cannot replace the six-channel RMRP score or the original locked target result.

## Correct comparison with the older scores

| Score | Protocol | What it means |
|---:|---|---|
| 0.6808 | v1 one-time zero-shot target, compact DANN | Locked target result on participants 11--20; six-channel and independently sealed at the time |
| 0.7514 | v1 few-person k=4, full MoRe-HAR | Four target-group participants entered training; post-confirmatory |
| 0.7593 | v1 few-person k=4 numerical leader | Best k=4 model, not full MoRe-HAR; post-confirmatory |
| 0.8395 | v2 five-seed strict source CV, flat RMRP | Six-channel source-development control on participants 1--10 |
| **0.8654** | v2 five-seed strict source CV, CTGR | Nine-channel source-development result on participants 1--10 |

Numerically, CTGR is 11.40 points above the cited 0.7514 and 10.61 points above the actual k=4 leader. Those are descriptive gaps across different cohorts, training information, and sensor interfaces, not valid treatment effects. The valid matched breakthrough comparison in this experiment is CTGR 0.8654 versus RMRP 0.8395.

## ASGS: label-efficient semantic repair

ASGS selected every query without inspecting its label, read the label only after selection, and removed all queried windows from evaluation. Each matched baseline was recomputed on exactly the same remaining windows.

| Base | Strategy | Total labels | Adapted mean | Matched remaining-window baseline | Difference |
|---|---|---:|---:|---:|---:|
| GSP | ASGS, budget 1 | 10 | **0.8569** | 0.8269 | **+0.0300** |
| GSP | ASGS, budget 2/3 | 11 | **0.8565** | 0.8265 | **+0.0300** |
| GSP | Fixed SAR, 1+1 | 20 | **0.8654** | 0.8361 | **+0.0293** |
| RMRP | ASGS, budget 1/2/3 | 10 | **0.8518** | 0.8358 | **+0.0160** |
| RMRP | Fixed SAR, 1+1 | 20 | 0.8439 | 0.8439 | 0.0000 |

The active rule identified a sitting/standing semantic swap for participant 10 with one query for every base. GSP required one extra query for one participant at budgets two and three. The hash-random control made no change at budgets one or two; it found the useful swap only at budget three, using 22 GSP or 18 RMRP labels. ASGS therefore supplies a clear label-efficiency result, but its scores must always be shown with the label count and altered evaluation set.

## Explicit-provenance mask robustness

Clean replay matched frozen RMRP exactly at 0.8379032940. Across the 17 corrupted cases, deterministic mask reconstruction increased the mean from 0.7426 to 0.7459. Across the nine cases with explicitly invalid samples, it increased the mean from 0.7701 to 0.7765.

The main gains were +0.0297 for a 12.5% contiguous gap, +0.0121 for one-axis dropout, +0.0107 for modality dropout, and +0.0071 for two-axis dropout. It was nearly neutral for stuck-at and saturation cases and reduced performance by 0.0055 for a 25% contiguous gap. It cannot address bias drift, scale drift, noise, or rotation because those measurements remain marked as observed rather than missing. The wrapper is useful, but it is not a clean-model accuracy advance and should not be oversold.

## Publication decision and next gate

CTGR is now the strongest trustworthy result produced by this source-development programme under its own nine-channel protocol. It is research-worthy as a prospectively evaluated secondary sensor-sufficiency result, especially because it improves mean, lower-tail, both posture recalls, and calibration across all five seeds. The correct paper language is “passed the predeclared source-development advancement gate” or “strong breakthrough signal,” not “confirmed state of the art.”

Freeze CTGR now. Do not tune it again on participants 1--10, do not reopen participants 11--20, and do not use the consumed DAGHAR metrics for selection. The next decisive experiment is one evaluation on a genuinely new, sealed, ability-relevant cohort that records all nine required channels. It should preserve participant-level inference, include enough participants to narrow the paired interval, predeclare handling of the two negative-responder patterns, and compare six-channel RMRP, nine-channel gravity expert, and CTGR on identical windows. Runtime, model size, gravity-channel availability, and real missing-sensor episodes should also be measured before a deployment claim.

## Evidence anchors

| Artifact | Record SHA-256 | File SHA-256 |
|---|---|---|
| CTGR inner-selection freeze | `6586c9a0e61446870dd2997dda29c0030a094f327654e577ea11a5fdbb7b9a50` | `f7475bc7ec43bf02ccdfe3b8eb6ed9c6a333b51781b638a63f2d257341a37180` |
| CTGR five-seed aggregate | `1cabbd7ec48ed334bb620e0376f8b26b520fcf7443e1d26bb4e83d197c345128` | `7cd31086894c8d3d905a9de190a6da666016d38b36b5f7bc7ff39c867f6828dd` |
| ASGS result | `7cb1390559582a7aae16258cef62f6820bbbd6049b9759a402839943b2e13286` | `43a6c6919b42c15e7c77bb55a521d9e8af62c64d0404818aa27ae7e41cf6080d` |
| Provenance-mask result | `2a181d175d57a7dc6f91c0760f69d33bf5e54d069a8125db62488d79507a9268` | `628bdeecb3c191f10aeb8f0f6d45f6086d8e8e7d0ce20cb3342eb0546f7320ef` |
| Post-run paired analysis | `ea1b25f52c9ca183e692c439b845f0a16b6db27f174c86310478c4ef511c3129` | `e51f1847e4047b58d9d17b0306f26ee481c37aa2e8292b524469edb231bedae8` |

All 34 primary run/fold records initially generated by the three locked lanes, plus the post-run analysis record, passed self-hash validation. All 119 referenced files checked before reporting matched their recorded SHA-256 values.
