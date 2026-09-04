# CAGE-HAR v1 implementation and evidence status

Status: software and prospective protocol complete on 2026-09-04; real development evaluation
is waiting for a genuinely new human cohort.

## Bottom line

CAGE-HAR is implemented as the next nine-channel research candidate. The implementation includes
the counterfactual-advantage router, participant-jackknife stability bound, posture-preserving
gravity gauge, explicit sensor-health context, participant-tail and intervention-harm weighting,
KL-bounded expert intervention, optional one-query semantic gauge, physical-fault generators,
ablations, and a guarded participant-exclusive development runner.

No new human score was generated. InclusiveHAR participants 1--10 were not reused for further
invention, participants 11--20 were not opened, and consumed DAGHAR evidence was not reopened.
Consequently, CTGR remains the strongest real human-data result in this research lane. Synthetic
smoke values below establish executable contracts only and are not estimates of HAR performance.

## Human-data result ledger

| Result | Sensor interface and evidence setting | Mean participant macro-F1 | Valid matched change | Interpretation |
|---|---|---:|---:|---|
| Historical full MoRe-HAR, k=4 | Six-channel few-person post-confirmatory; four target-group participants enter training | 0.7514 | Not comparable to source-only methods | The often-cited result in the 70s; not the baseline for CTGR or CAGE-HAR |
| Historical k=4 numerical leader | Six-channel few-person post-confirmatory | 0.7593 | Not comparable to source-only methods | Best descriptive model in that different protocol |
| Matched RMRP control | Six-channel, five-seed source-development CV on participants 1--10 | 0.8395 | Reference for CTGR | Correct matched control for the CTGR experiment |
| CTGR | Nine-channel, same folds/seeds/source-development participants as matched RMRP | **0.8654** | **+0.0259 over RMRP** | Passed its nine predeclared source-development advancement checks; not independent confirmation |
| CAGE-HAR v1 | Nine-channel, genuinely new development cohort | **N/A** | **N/A** | Implementation frozen; no eligible new cohort is present |

The numerical gap from 0.7514 to 0.8654 is +0.1140, but it is not a valid treatment effect:
training information, cohorts, evidence phase, and sensor interfaces differ. The defensible matched
advance is CTGR over RMRP, +0.0259 mean participant macro-F1. A CAGE-HAR breakthrough cannot be
assessed until the prospective new-cohort run exists.

## Implemented research plan

| Work package | Status | What is now executable or enforced |
|---|---|---|
| Posture-preserving uncertain gravity gauge | Complete | Retains absolute posture evidence and partial rotation-invariant vertical/horizontal dynamics without deleting slow/DC gravity information |
| Sensor-health context | Complete | Explicit coverage, gaps, time-since-observation, flatline, saturation, drift, and high-frequency symptoms; missing gravity receives zero reliability |
| Counterfactual-advantage routing | Complete | Learns expert-minus-base true-class log-loss advantage from training labels while routing from label-free features only |
| Participant protection | Complete | Participant-balanced fitting and leave-one-participant-out jackknife lower advantage bound |
| Tail and harm protection | Complete | Fixed two-times weight for base-correct/expert-wrong events and lowest-30% training participants |
| Trust-region intervention | Complete | Reliability and advantage gates, maximum blend, KL cap, exact base fallback, and mobility-mass preservation for posture experts |
| Decision repair | Complete | Train-partition-only temperature and sitting/standing offset selection |
| Required ablations | Complete | Uncalibrated base, repaired base, global, confidence, disagreement, and full CAGE-HAR routes |
| Optional labelled semantic gauge | Complete | One query only when suspicious, soft identity-versus-swap update, queried-window exclusion contract |
| Physical-fault suite | Complete | Bias, drift, gain, colored noise, gaps, modality/axis loss, stuck-at, saturation, orientation, axis permutation, jitter, quantization, and combined faults with explicit masks |
| Baseline inventory | Complete with visible pending work | Separates six-channel, nine-channel, and modern-representation tracks; unavailable/licence-pending adapters remain declared |
| New-cohort contract and denial gate | Complete | Requires native nine channels, units/frames/timestamps/session provenance and partition-before-windowing; refuses consumed evidence |
| Real new-development training | Waiting for data | Requires a genuinely new, self-hashed development cohort |
| Independent confirmatory evaluation | Waiting for data | Requires a second sealed cohort after candidate freeze |

## Synthetic contract smoke test -- not a human result

The deterministic synthetic fixture was constructed to exercise routing, rescue, harm accounting,
trust bounds, and all six ablation outputs. It is deliberately easy and must never be included in
a human-result or publication-performance table.

| Synthetic-only check | Macro-F1 | Bottom-30% participant macro-F1 | Route/change/rescue/harm counts |
|---|---:|---:|---|
| Uncalibrated base | 0.7953 | Not a scientific endpoint | No CAGE routing |
| Decision-repaired base | 0.8841 | Not a scientific endpoint | No CAGE routing |
| Global trust blend | 0.9685 | Not a scientific endpoint | Contract ablation only |
| Confidence trust gate | 0.9685 | Not a scientific endpoint | Contract ablation only |
| Disagreement trust gate | 0.9685 | Not a scientific endpoint | Contract ablation only |
| CAGE-HAR | 0.9712 | 0.9603 | 257 / 91 / 91 / 0 |

Synthetic intervention precision was 1.0 by construction. These values validate deterministic
software behavior; they do not pass the real advancement gates, establish robustness, select the
method, or support a state-of-the-art claim.

## Validation record

| Gate | Result |
|---|---|
| Focused CAGE-HAR tests | 44 passed |
| Full repository tests on the completed CAGE source/test tree | 691 passed in 1719.04 seconds |
| Final CAGE protocol and release-documentation tests | 13 passed |
| Ruff lint | Passed |
| Ruff formatting check | Passed; 215 files already formatted |
| Mypy | Passed; 215 source files checked |
| Manifest validation | Passed; three manifests |
| Released-block split audit | Conditional released-block pass; target not accessed; documented schema limitations remain visible |
| Artifact validation | Passed; 12 referenced files across two manifests; preserved quarantine warning remains visible |
| Dependency lock check | Passed; 98 packages resolved |

## What remains before a publication performance claim

1. Collect or acquire a genuinely new development cohort with native user acceleration, rotation
   rate, gravity, timestamps, units, frames, site/device/session/trial/repetition provenance, at
   least two sessions per participant, and preserved raw evidence.
2. Run the already frozen six-ablation, participant-exclusive development protocol once. Report
   participant-level mean, bottom tail, worst participant, proper probability scores, routing
   precision/harm, per-class recall, and corruption curves with uncertainty.
3. If and only if the predeclared gates pass, freeze code, configuration, candidate map, and
   checkpoints before touching a second independent cohort.
4. Open the sealed confirmatory cohort once and keep six-channel, nine-channel, zero-shot,
   adaptation, and labelled-personalization claims separate.
5. For a deployment claim, finish the pending baseline adapters/licence audit and measure runtime,
   memory/model size, energy where feasible, gravity availability, and real missing/fault episodes.

Until those steps are complete, the accurate statement is: **CAGE-HAR is a prospectively specified,
tested implementation motivated by CTGR's negative responders; its real performance is unknown.**
