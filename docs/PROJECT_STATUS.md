# Project status and evidence gates

This file records workflow state. It is not a results page and does not authorize a confirmatory evaluation.

| Stage | Status | Evidence |
|---|---|---|
| 0 — legacy preservation and audit | Passed (mechanics only) | `legacy/verification_results.json`; legacy UCI test remains development-consumed |
| 1 — literature and novelty | Passed with narrowed contribution | `docs/LITERATURE_MATRIX.md`; MoRe-HAR novelty remains open/high-collision |
| 2 — repository and provenance scaffold | Passed | `results/gates/stage2_scaffold_validation.json`; synthetic tests and static gates passed at that snapshot |
| 3 — InclusiveHAR audit and ontology | Integrity passed; original protocol quarantine preserved | `results/data_audit/inclusivehar_v4.audit.json`; trial boundaries remain unrecoverable |
| 4 — conditional released-block protocol | Split construction and structural audit passed conditionally | `docs/LOCKED_PROTOCOL.md`; split `ccb6c3…`; target remains sealed; no protocol tag or final-evaluation unlock |
| 5–8 — models, experiments, metrics | No InclusiveHAR experiment result | Implementation scaffolds may exist, but no target training, prediction, performance, ablation, or confirmatory result has been produced by this gate |
| 9 — final evidence gate | Closed | No final-evaluation authorization exists; target opening is forbidden |
| 10 — release | Not completed | No public release or DOI exists; release/remote status must be checked independently before any claim |

## Locked facts so far

- Source coursework ZIP SHA-256: `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94`.
- Stage 0 verification SHA-256: `F8838A2ED86E3E2B40D72B26D79D5522A40DDF8D9F257C43A15205C6802621D0`.
- Current literature matrix SHA-256 at its recorded snapshot: `12C587D2C9D2A1CC1E7C339634FA153DBAA8DEDA2EA0583ACE3A371405295919`.
- Current literature source registry SHA-256 at its recorded snapshot: `96EB2CA8664DC9F1F04D1B854CD9431FBD1D46496CD703ECABE60964B33913BC`.
- Novelty-gate wording is narrowed to a candidate leakage-safe, **participant-exclusive released-block** ability-associated InclusiveHAR-v4 benchmark. It is not trial-safe and makes no “first,” fairness, clinical-validity, state-of-the-art, or publishability claim.
- InclusiveHAR raw-read gate file SHA-256: `0CB9F5F3ADD587E688CB539E82257FE8A38723E6AE81CF4B13EB3028164357E3`.
- InclusiveHAR Stage 3 audit embedded report SHA-256: `673917D3BFD4B3F283A1AE521AE4A7D548E3FACADD2AE89CE33B2E27153500E8`; audit JSON physical SHA-256: `B85E641DA9FDC3AC61B6C8440A78C2B6AEE625C160B095C8170C0B9D1A4A4204`.
- Stage 3 conclusion remains visible: artifact/schema/coverage integrity passed, but timestamp/trial/session/sample identifiers are absent and hidden trial joins are unrecoverable.
- User-authorized deviation record canonical SHA-256: `1C0D40E1CF65CAB9371669A094CE35D6835F9BB6D5DD47C7DC04847917A44354`.
- Active locked ontology record SHA-256: `C2B4E178AFB3D285CC76745182EDB1868099EB1E607DBEBFCEE263C973A81E33`; both runnable tracks carry explicit ordered numeric schemas.
- Active split manifest embedded SHA-256: `CCB6C3D1254C1464C48E412AFB6F83E7942113299E853F89C77DF1D6CFAD131B`; physical SHA-256: `AF909C7D914B68C417B238B576945C112AA58740859A7D998B6B77219DFB1920`.
- Active split audit embedded report SHA-256: `45C49761A92B9F9F3E4CABE4E3A9DBF72B4114795948BA7104F8ED0DFF37BDD8`; status `pass_conditional_released_block` with the hidden-join, conditional-bound, and unverified-rate warnings retained.
- Source-only materialization manifest embedded SHA-256: `1AD1EE3ACCAAE5F2F93BB91AC0AFA5CE583134CE1D882C3F08323B09026FA522`; it contains 1,443 source windows and no target subject/window records.
- Target seal ID: `AECBA05FA4A0FC4E4BBC135AC30944B686BE19C0B6A2820C838E6C8B802BB29D`; target predictions/performance have not been accessed and `unlock_record` is null.
- Conditional protocol-lock record SHA-256: `71F3FDB2DE0E6E0AA8FEAFF8B8726BFBC999FD1423568251744A29AB541E9E3A`; the worktree is explicitly documented as uncommitted and no protocol tag exists.
- The v1 and v1.1 split/config/audit records remain preserved and reconstructable. Their supersession records document the nested-CV and explicit-class-schema additions without target access.

Any later edit changes a file hash and must be recorded in a new validation or supersession artifact. Existing evidence files must not be overwritten or relabelled.
