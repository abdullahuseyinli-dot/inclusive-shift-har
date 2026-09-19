# Canonical HERA-CTGR method status

**Status date:** 2026-09-19
**Scope:** source-development candidate status only
**Authority:** the locked HERA and CTGR result records named below. This document
does not reopen, replace, or reinterpret a locked target result.

## Canonical comparisons

All zero-query rows use the matched source-development contract: 725 windows,
ten source participants, five participant outer folds, four inner folds, and
five fixed seeds. Participant macro-F1 is the primary metric.

| Method | Mean participant macro-F1 | Change versus CTGR | Status |
|---|---:|---:|---|
| RMRP | 0.83953 | -0.02586 | matched base |
| CTGR | 0.86540 | 0.00000 | robust retained control |
| HERA-CTGR v1 strict | **0.86849** | **+0.00309** | highest zero-query source-development point estimate |
| HERA-CTGR v2 full | 0.86749 | +0.00209 | calibration improvement; not promoted over v1 |
| HERA compact evidence integration | 0.86461 | -0.00294 versus aligned HERA reference | closed: no complementary gain |

The exact HERA-v1 comparison reports a descriptive participant-bootstrap interval
of [-0.00095, +0.00713] for strict HERA minus CTGR. Its predeclared gain,
bottom-tail, and breakthrough gates did not pass. HERA-v1 is therefore a
candidate to validate on new people, not a confirmed successor or a
state-of-the-art claim.

CTGR has the stronger incremental source-development support: it improved the
matched RMRP package by +0.025864 F1 (2.5864 pp), with eight of ten participants
improving. Its participant bootstrap interval still crosses zero, so this also
remains development evidence rather than independent confirmation.

## Separate supervised modes

The HERA posture semantic gauge is deliberately excluded from the canonical
zero-query table. It used one labelled query per participant and removed queried
windows before scoring. Its active-hard result was 0.872558 on the 715 matched
remaining windows (+0.006110 F1); one participant improved, nine tied, and the
participant-bootstrap lower bound was zero. Random hard querying obtained a
similar +0.00565 F1. The result is an optional personalization diagnostic, not
an architecture gain over HERA.

The older few-person and historical-target studies likewise remain separate:
they changed supervision, participants, or both. Participants P11--P20 must not
be reopened for selection or tuning.

## Frozen implementation decision

Retain the following as the current source-development package:

- CTGR as the primary robust control and deployment-candidate feature path.
- HERA-v1 strict as the highest point-estimate zero-query ablation.
- HERA-v2 temperature calibration and rank-locked decision separation only as
  probability-quality and safety ablations.
- The semantic gauge only behind an explicit labelled-personalization contract.

Do not promote HERA-v2 routing, stability weighting, compact evidence
integration, gravity-invariant variants, or additional source-participant
sweeps. The source cohort has been used repeatedly for development and cannot
produce independent confirmation.

## Evidence and next valid study

- HERA-v1: `docs/research/HERA_CTGR_RETROSPECTIVE_V1_RESULTS.md`
- HERA-v2: `docs/research/HERA_CTGR_V2_RETROSPECTIVE_V1_RESULTS.md`
- Semantic gauge: `docs/research/HERA_POSTURE_GAUGE_V1_RESULTS.md`
- HERA compact integration: `.audit/hera_compact_evidence_integration/hera-compact-evidence-integration-20260916-001/result.json`
- Independent next-cohort and same-attachment design: `.audit/research_direction_20260913-001/NEXT_EXPERIMENT_SPEC.md`

The next scientific action after repository validation is a fresh,
interface-matched native-nine cohort, or the pre-specified six-wearer
same-attachment pilot if those recordings become available. Neither study may
use P11--P20 to select a successor.
