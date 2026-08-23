# Project status and evidence gates

This file records workflow state. It is not a results page and does not authorize a confirmatory evaluation.

| Stage | Status | Evidence |
|---|---|---|
| 0 — legacy preservation and audit | Passed (mechanics only) | `legacy/verification_results.json`; legacy UCI test remains development-consumed |
| 1 — literature and novelty | Passed with narrowed contribution | `docs/LITERATURE_MATRIX.md`; MoRe-HAR novelty remains open/high-collision |
| 2 — repository and provenance scaffold | Passed | `results/gates/stage2_scaffold_validation.json`; synthetic tests and static gates pass |
| 3 — InclusiveHAR audit and ontology | Integrity passed; protocol quarantined | `results/data_audit/inclusivehar_v4.audit.json`; trial boundaries are unrecoverable and the split/window gate is blocked |
| 4 — protocol lock | Blocked, not started | No protocol lock, protocol-lock tag, split authorization, or final-evaluation unlock exists |
| 5–8 — models, experiments, metrics | Not started | No corrected or InclusiveHAR model results exist |
| 9 — evidence gates | Not started | No final-evaluation authorization exists |
| 10 — release | Not started | No remote repository, release, or DOI has been created |

## Locked facts so far

- Source coursework ZIP SHA-256: `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94`.
- Stage 0 verification SHA-256: `F8838A2ED86E3E2B40D72B26D79D5522A40DDF8D9F257C43A15205C6802621D0`.
- Current literature matrix SHA-256 after the UCI-license clarification: `12C587D2C9D2A1CC1E7C339634FA153DBAA8DEDA2EA0583ACE3A371405295919`.
- Current literature source registry SHA-256: `96EB2CA8664DC9F1F04D1B854CD9431FBD1D46496CD703ECABE60964B33913BC`.
- Novelty-gate wording: a candidate leakage-safe, subject/trial-exclusive, one-way ability-associated InclusiveHAR-v4 benchmark; no “first,” fairness, clinical-validity, state-of-the-art, or publishability claim.
- InclusiveHAR raw-read gate file SHA-256: `0CB9F5F3ADD587E688CB539E82257FE8A38723E6AE81CF4B13EB3028164357E3`.
- InclusiveHAR audit embedded report SHA-256: `673917D3BFD4B3F283A1AE521AE4A7D548E3FACADD2AE89CE33B2E27153500E8`; audit JSON file SHA-256: `B85E641DA9FDC3AC61B6C8440A78C2B6AEE625C160B095C8170C0B9D1A4A4204`.
- Stage 3 conclusion: artifact/schema/coverage integrity passed, but absent timestamp/trial/session/sample identifiers and group-aligned file order prohibit split/window construction under the current requirements.

Any later edit changes a file hash and must be recorded in a new validation artifact rather than silently retaining these values.
