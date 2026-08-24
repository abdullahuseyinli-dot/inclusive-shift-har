# Project status and evidence gates

This is a superseding status snapshot dated 2026-08-24. Earlier manifests, gates, failures, tags, and status records remain preserved and must not be relabelled.

| Stage | Status | Evidence |
|---|---|---|
| 0 - legacy preservation and audit | Passed for archive integrity and metric reconstruction | `legacy/verification_results.json`; UCI test remains development-consumed |
| 1 - literature and novelty | Passed with narrowed contribution | `docs/LITERATURE_MATRIX.md`; no verified novelty conflict; no "first" claim |
| 2 - repository and provenance | Passed | package, lockfile, manifests, CLI, tests, CI, licence metadata |
| 3 - InclusiveHAR audit and ontology | Integrity passed; trial-boundary limitation retained | `results/data_audit/inclusivehar_v4.audit.json`; timestamps/trials absent |
| 4 - released-block protocol | Conditionally passed and locked | split `ccb6c3d...`; participant-exclusive/raw-row-disjoint, not trial-safe |
| 5-8 - models, training, and statistics | Primary suite and locked analysis complete | 20 configurations x 5 seeds; CUDA neural execution; participant inference |
| 9 - final evidence gate | Passed before opening | 190 tests, lint, format, mypy, manifests, splits, artifacts, clean tree, protocol tag |
| 9 - target opening | Consumed exactly once and complete | opening receipt, 100 result sidecars/arrays, immutable index, locked statistics |
| 10 - release | Local release preparation in progress | private GitHub push/tag follows final post-confirmatory validation; no DOI |

## Confirmatory outcome

- Target cohort: participants 11-20; 10 participants, 807 functional-core windows.
- Frozen lineup: 20 model/ablation configurations, seeds 11, 23, 47, 89, and 131.
- Highest mean participant macro-F1: compact DANN, 0.6808438, 95% participant-bootstrap CI [0.5391426, 0.8093204].
- MoRe-HAR full: mean 0.6353323, worst 0.2583943, lower decile 0.2659867.
- Strongest mean reference: compact DANN; strongest worst-participant reference: legacy joint CNN/BiLSTM (0.2942991); strongest lower decile: compact CORAL (0.3669902).
- Preregistered MoRe-HAR decision: not supported. Mean improvement and joint lower-tail improvement were both false; the source non-inferiority gate had passed.
- Candidate minus compact-DANN participant mean: -0.0455116. Holm-adjusted exact sign-flip p = 0.7207031; Holm-adjusted Wilcoxon p = 0.7558594.

Target confidence intervals are wide and participant tails are low across every model. The evidence supports a difficult ability-associated shift, not a causal explanation or a fairness/clinical claim.

## Active immutable anchors

- Coursework ZIP SHA-256: `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94`
- Literature matrix file SHA-256: `F7363E8D06E7C561ACF9BD9A47D5DD456EC5697FE3A2DCC4F60523688E30A606`
- Literature registry file SHA-256: `6DB576249EDBE530C9CBE0411728B20928D2B4A3CE852283F888743A261F8E7D`
- Dataset manifest file SHA-256: `52de5370682f13fbd9a4e9affe805b4f5ee0f28901d137743884b77471b24d29`
- Split manifest embedded SHA-256: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
- Source-window manifest embedded SHA-256: `1ad1ee3accaae5f2f93bb91ac0afa5ce583134ce1d882c3f08323b09026fa522`
- Target seal ID: `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`
- Frozen artifact-set SHA-256: `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`
- Confirmatory analysis plan SHA-256: `7b99dd5894109370397867a1ca141758c0a30b4efb2fb3d76aba23ab1ad62177`
- Opening receipt record SHA-256: `5704f65efd416e9cd16d6ed24c2735c1d52b7fcaafddb113e481498797cd182a`
- Locked target index record SHA-256: `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`
- Participant statistics record SHA-256: `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`
- Publication report record SHA-256: `3afe0ceee9f97025d1adc5f59ab3528b512a3212385ef5344028850e9cb39c66`
- Frozen training code commit: `b4dc38fb9d5a0c17003221b61156ebc065395170`
- One-time target evidence commit: `f0d11b2`
- Active protocol tag: `protocol-v1.2.0`

## Preserved limitations and deviations

The split audit validates participant exclusivity, label-block containment, and raw-row disjointness. Because InclusiveHAR releases no trial/session/timestamp identifiers, its unconditional hidden-join contamination bound is 100%; a conditional three-repetition assumption gives 240/3,042 (7.8895%), but that assumption is unverified. This benchmark must not be called trial-safe or unqualified leakage-safe.

Recurrent cuDNN execution failed on this Windows/CUDA stack with process exit `0xc0000409`. Failure artifacts are preserved. Successful recurrent experiments used CUDA tensors with cuDNN disabled, not CPU neural fallback. Classical scikit-learn estimators retained their native CPU policy; XGBoost training used CUDA.

The target opening cannot be repeated. All few-person inclusion, corruption, or other follow-up is post-confirmatory and cannot alter the locked zero-shot claim.
