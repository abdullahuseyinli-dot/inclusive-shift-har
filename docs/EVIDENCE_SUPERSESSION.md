# Evidence supersession map

**Status date:** 2026-09-20

InclusiveShift-HAR uses append-only evidence. A later correction narrows or
replaces the interpretation of an earlier record; it does not delete, relabel,
or rewrite the historical artifact. When records conflict, use the last column
below for current interpretation.

The machine-readable companion is
[`configs/datasets/evidence_roles_20260920_v4.yaml`](../configs/datasets/evidence_roles_20260920_v4.yaml).

| Earlier record | Reason it is historical or incomplete | Current governing record |
|---|---|---|
| The former README checkpoint dated 2026-09-05 | Described the canceled external queue as pending and predates HERA, HARTH, AICOS, and final routing evidence | [Evidence index](EVIDENCE_INDEX.md) and [project status](PROJECT_STATUS.md) |
| `docs/PROJECT_STATUS.md` snapshot dated 2026-09-04 | Predates the later source methods and external diagnostics | Current version of [project status](PROJECT_STATUS.md) |
| `configs/datasets/external_har_portfolio_v1.yaml` AICOS entry labelled `unconsumed` | Acquisition-time role recorded before both provider test and development folds were evaluated; its bytes are retained for provenance | [AICOS posture review](research/AICOS_POSTURE_REVIEW_20260919.md) followed by [routing validation](research/CTGR_ROUTING_VALIDATION_20260919.md) |
| [External dataset qualification](research/EXTERNAL_GENERALIZATION_DATASET_QUALIFICATION_20260919.md) | Prospective qualification record written before AICOS consumption | AICOS is now consumed development/diagnostic evidence with conditional interface status |
| [Original AICOS zero-shot benchmark](research/AICOS_ZERO_SHOT_BENCHMARK_20260919.md) | Source models trained in g received AICOS acceleration/gravity in m/s² | Its correction banner and the [unit-corrected review](research/AICOS_POSTURE_REVIEW_20260919.md) govern all rankings |
| Eight-person corrected AICOS U9 gain | Post-hoc result on the consumed provider test suggested +4.649 points | The fixed 38-person provider-development comparison found -0.046 points and failed promotion; see [routing validation](research/CTGR_ROUTING_VALIDATION_20260919.md) |
| [Native-nine preparation](research/CTGR_NATIVE9_CONFIRMATION_PREPARATION_20260919.md), run `-002` | Its recorded runner hash predates the explicit SI-to-g conversion repair | Current code and tests contain the repair; a fresh create-only `prepare` receipt is required before claiming the present runner is bound to a new cohort |
| HERA/CTGR individual retrospective reports | Several later ablations reused the same ten source participants | [Canonical HERA/CTGR status](research/CANONICAL_HERA_CTGR_METHOD_STATUS.md) governs model selection; original reports remain the numerical sources |
| HARTH architecture proposals and intermediate runs | Later matched placement and published-protocol comparisons resolved their promotion questions | [Research-lane closure](research/RESEARCH_LANE_CLOSURE_20260919.md) |
| FoG external v1-v3 packages and pre-window participant plans | Annotation-dependent preprocessing, context-population, and split-order defects were found after structural validation | Versioned correction documents and the v4 supersession ledger under `results/research/cross_dataset_har_v4/` |
| Historical CUDA/tag publication recipe in [release evidence gate](RELEASE_EVIDENCE_GATE.md) | Binds a specific v0.1.7 candidate lineage and remains valuable release-security evidence, but does not describe the later CPU diagnostic additions | [Publication checklist](PUBLICATION_CHECKLIST.md) defines the current preparation pass; an exact release still needs candidate-bound checks |
| [`paper/OUTLINE.md`](../paper/OUTLINE.md) and [`paper/FUSE_REFRAME_V2_ADDENDUM.md`](../paper/FUSE_REFRAME_V2_ADDENDUM.md) | Historical working drafts predate the final CTGR/HERA, corrected AICOS, HARTH and routing evidence | [Paper workspace status](../paper/README.md) points to this evidence index and the current project status; no submission-ready manuscript is claimed |

## Records that are not superseded

The following are immutable within their declared scope:

- The one-time InclusiveHAR target opening and its negative MoRe-HAR decision.
- The locked ontology, participant split, target seal, and released-block risk
  statement.
- CTGR's five-seed source-development aggregate and predeclared advancement gate.
- HERA-v1 and HERA-v2 retrospective numerical records.
- Failed attempts, participant harms, interface limitations, and quarantine
  records.

A later method cannot change the historical target outcome. A later correction
to external units does not change original InclusiveHAR results. A Git or Zenodo
release must include this map or point to it prominently so readers do not quote
superseded external rankings.
