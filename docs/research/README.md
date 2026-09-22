# Experiment map

Experiments are grouped by scientific question. Start with the
[research report](../RESEARCH_REPORT.md) for methods and comparable tables.
This index is a reading guide, not a leaderboard or a new result. Frozen
protocols, failed attempts, and historical paths remain preserved.

The project-developed source-method sequence is **SpectralShape → GSP → RMRP →
CTGR → HERA**. RMRP is an earlier contribution and the six-channel control for
later gravity-based extensions. The [source-lineage audit](SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md)
separates this history from external controls, labelled personalization and
different-cohort results.

## 1. Establish the evaluation contract

| Study | Purpose and outcome | Record |
|---|---|---|
| Legacy UCI-HAR audit | Reconstruct coursework results; identify consumed test evidence and invalid comparisons with later cohorts | [Legacy audit](../LEGACY_AUDIT.md) |
| Participant-exclusive benchmark | Freeze ontology, source/target participants, endpoints, and one-time target access | [Locked protocol](../LOCKED_PROTOCOL.md), [benchmark card](../BENCHMARK_CARD.md) |
| Original target comparison | Compact DANN and CORAL effectively tied; MoRe-HAR hypothesis unsupported | [Target report](../../results/confirmatory/zero_shot_v1/publication_report_v1.json) |
| Post-confirmatory analyses | Within-group training, few-person inclusion, sensor stress, unit sensitivity, and adapted external methods answer separate secondary questions | [Results inventory](../../results/README.md) |

## 2. Improve the source model

| Family | What was learned | Record |
|---|---|---|
| HYDRA, MultiRocket, QUANT and budgeted RIST | External controls reached 70.711–77.195% participant macro-F1 on the shared seed-11 source evaluation; fixed settings differ from the project's nested selection budget | [Source-lineage audit](SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md), [FuSE/ReFrame report](FUSE_REFRAME_V2_RESEARCH_REPORT.md) |
| SpectralShape and GSP | Project feature development increased seed-11 participant macro-F1 from 78.884% to 82.916%; worst-participant performance still trailed budgeted RIST | [Source-lineage audit](SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md), [FuSE/ReFrame report](FUSE_REFRAME_V2_RESEARCH_REPORT.md) |
| RMRP | The project's denoised GSP reached 83.790%: +0.874 points over GSP; all five folds selected this view. Noise tolerance improved, temporal-gap robustness weakened, and the joint advancement gate failed | [FuSE/ReFrame report](FUSE_REFRAME_V2_RESEARCH_REPORT.md) |
| Neural controls, feature fusion and expert stacking | No retained gain; neural diagnostics using direct outer-fold early stopping are excluded from the primary comparison | [FuSE/ReFrame report](FUSE_REFRAME_V2_RESEARCH_REPORT.md) |
| CTGR | Native gravity plus confidence-triggered posture correction improved matched five-seed development over RMRP; nine advancement checks passed | [Protocol](CONFIDENCE_TRIGGERED_GRAVITY_RESIDUAL_PROTOCOL.md), [results](MAX_RND_SECONDARY_RESULTS.md) |
| CAGE-HAR | The retrospective extension did not pass advancement | [Results](CAGE_HAR_RETROSPECTIVE_V1_RESULTS.md) |
| HERA-v1 | Strict window-only variant had the highest point estimate; incremental gain and lower-tail gates failed | [Results](HERA_CTGR_RETROSPECTIVE_V1_RESULTS.md) |
| HERA-v2 | Calibration improved; rescue routing remained inactive in all outer fits; classification did not advance | [Results](HERA_CTGR_V2_RETROSPECTIVE_V1_RESULTS.md) |
| Fixed GSP/gravity/estimator ablation (Round A) | All six cells completed; flat gravity and learner substitutions failed to exceed aligned CTGR/HERA controls | [Complete table](../RESEARCH_REPORT.md#43-fixed-source-ablation), [metric audit](REPORTED_METRICS_AUDIT_20260920.md) |
| Microstate posture graph | The graph extension did not justify replacing the retained source model | [MPG results](MPG_RMRP_V1_RESULTS.md) |
| Compact evidence integration and gravity-invariant follow-up | Compact integration regressed against matched HERA; current evidence does not promote these variants | [Compact protocol](HERA_COMPACT_EVIDENCE_INTEGRATION_V1_PROTOCOL.md), [gravity protocol](GRAVITY_INVARIANT_FOLLOWUP_V1_PROTOCOL.md), [canonical status](CANONICAL_HERA_CTGR_METHOD_STATUS.md) |
| Fixed unconditional routing U9 | Source regression and participant harms; larger AICOS comparison did not reproduce the earlier small-cohort gain | [Routing validation](CTGR_ROUTING_VALIDATION_20260919.md) |

Five-seed aggregates, single-seed follow-ups, and exploratory controls are
separate comparison groups. Historical terms such as “breakthrough” in early
reports are interpreted through the current evidence and advancement gates,
not as independent confirmation or a novelty claim.

The legitimate source-development improvement from tested controls in the 70s
to project feature methods in the 80s must remain distinct from the locked
target and few-person target-inclusion results. Later CTGR/HERA scores also use
native gravity, so input information changes along that part of the sequence.

## 3. Change the information available

| Family | Information budget and interpretation | Record |
|---|---|---|
| Semantic Anchor Reconciliation and Active Semantic-Gauge Sentinel | Labelled personalization; anchor windows excluded from evaluation | [FuSE/ReFrame report](FUSE_REFRAME_V2_RESEARCH_REPORT.md), [secondary results](MAX_RND_SECONDARY_RESULTS.md) |
| HERA posture gauge | One query per participant; observed gain confined to one participant on matched remaining windows | [Gauge results](HERA_POSTURE_GAUGE_V1_RESULTS.md) |
| Provenance-mask reconstruction | Small corruption-specific robustness effect | [Secondary results](MAX_RND_SECONDARY_RESULTS.md) |
| Physical-reference and same-attachment studies | Geometric identifiability and software contracts; no prospective human pilot result | [Signal analysis](PHYSICAL_INFORMATION_SIGNAL_ANALYSIS_V1.md), [reference protocol](SAME_ATTACHMENT_DIRECTIONAL_REFERENCE_GEOMETRY_V1_PROTOCOL.md) |

## 4. Evaluate external data and sensing assumptions

| Study | Current interpretation | Record |
|---|---|---|
| External FoG and cross-dataset sequence | Retain annotation, session-grid, context-population, and split-order corrections; earlier versions are superseded | [Correction index](../EVIDENCE_SUPERSESSION.md), [v4 evidence](../../results/research/cross_dataset_har_v4/) |
| FoG feature/weight, decision-rule, spatial, motion-factorization, pretrained-context and compact-readout probes | Distinct mechanism diagnostics with their own protocols; not interchangeable with the InclusiveHAR posture endpoint | [Factorial](FOG_RF_FEATURE_WEIGHT_FACTORIAL_V1_PROTOCOL.md), [decision rule](FOG_DECISION_RULE_PROBE_V1_PROTOCOL.md), [spatial probe](FOG_SPATIAL_INFORMATION_PROBE_V1_PROTOCOL.md), [factorization](FOG_MOTION_FACTORIZATION_V1_PROTOCOL.md), [pretrained context](FOG_PRETRAINED_OPTIONAL_CONTEXT_V1_PROTOCOL.md), [compact readout](FOG_COMPACT_JOINT_READOUT_V1_PROTOCOL.md) |
| HARTH placement comparison | Thigh reached 99.723% accuracy and 97.261% participant macro-F1 over five participant folds; fusion added no measured benefit | [Metric audit](REPORTED_METRICS_AUDIT_20260920.md), [lane closure](RESEARCH_LANE_CLOSURE_20260919.md) |
| HARTH CRSP, SAGE-X, gravity hierarchy and temporal/standing gates | No reliable simultaneous posture and participant improvement supporting promotion | [CRSP](HARTH_CRSP_BACK_V1_PROTOCOL.md), [SAGE-X](HARTH_SAGE_X_V2_PROTOCOL.md), [hierarchy](HARTH_GRAVITY_HIERARCHY_V1_PROTOCOL.md), [standing gate](HARTH_STANDING_OBSERVABILITY_GATE_V1_PROTOCOL.md) |
| HARTH published-style replay | Executed XGBoost exceeded project fused RF on the merged nine-class endpoint | [Protocol](HARTH_PUBLISHED_REPLICATION_V1_PROTOCOL.md), [disposition](RESEARCH_LANE_CLOSURE_20260919.md) |
| AICOS zero-shot and unit review | Initial unit-mismatched rankings are superseded; axes/polarity remain unresolved | [Corrected review](AICOS_POSTURE_REVIEW_20260919.md) |
| AICOS fixed routing validation | 38 complete participants; U9 did not improve the mean and worsened sitting and lower-tail outcomes | [Routing report](CTGR_ROUTING_VALIDATION_20260919.md) |

## 5. Current disposition

Retain CTGR as the primary source-development control and strict HERA-v1 as
the highest point-estimate ablation. The next performance study requires a fresh
qualified cohort or new measured reference information. The implemented
[native-nine confirmation workflow](CTGR_NATIVE9_CONFIRMATION_V1_PROTOCOL.md)
has no fresh-cohort result. The six-wearer same-attachment pilot is prospective.

The [canonical status](CANONICAL_HERA_CTGR_METHOD_STATUS.md),
[research disposition](CURRENT_RESEARCH_DISPOSITION_20260919.md), and
[assessment](EXPERT_FINALIZATION_ASSESSMENT_20260920.md) explain this decision.
The [supersession map](../EVIDENCE_SUPERSESSION.md) governs interpretation of
older documentation after a correction.

```mermaid
flowchart TD
    V["Participant-exclusive protocol"] --> L["One-time target: hypothesis unsupported"]
    V --> X["Seed 11 controls: 70.711–77.195%"]
    V --> S["Project SpectralShape: 78.884%"]
    S --> G["Project GSP: 82.916%"]
    G --> R["Project denoised RMRP: 83.790%"]
    R --> B["Five-seed RMRP base: 83.953%"]
    B --> C["CTGR: 86.540%, + native gravity"]
    C --> H["HERA: +0.309 points, gates failed"]
    C --> U["Unconditional routing: rejected"]
    H --> Q["Labelled gauge: separate personalization"]
    C --> F["Freeze candidate"]
    H --> F
    E["External diagnostics"] --> A["AICOS: units matter; axes unresolved"]
    E --> T["HARTH: placement matters"]
    A --> N["Qualify fresh interface and cohort"]
    T --> P["Measure same-attachment reference information"]
    F --> N
    F --> P
```

Arrows express the research sequence or resulting decisions, not causal effects
estimated across datasets. Scores are participant-mean macro-F1. Seed-11
controls used fixed configurations; project feature methods used nested
selection. The five-seed base is a new aggregation of repeated source fits,
not a separate independent participant cohort.
