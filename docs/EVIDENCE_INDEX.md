# Current evidence index

**Status date:** 2026-09-22

**Purpose:** authoritative navigation and claim scope for the publication
candidate. This index does not modify any locked protocol or historical result.

## How to read the metrics

`Participant macro-F1` means macro-F1 is calculated separately for each person
and then averaged, so participants receive equal weight. `Pooled` metrics treat
all eligible windows or samples together. Accuracy, pooled macro-F1, and mean
participant macro-F1 are different estimands and are never substituted for one
another in this index.

Seeds measure model-fitting variation on the same people. They do not increase
the independent participant count. A comparison must state its cohort, endpoint,
channels, supervision, partitions, aggregation and selection budget. When one of
these changes, the effect cannot be attributed to architecture alone.

## A. Historical locked target opening

InclusiveHAR participants P11-P20 were opened once under the frozen six-channel
functional-core protocol. This outcome is immutable and is not a HERA/CTGR
evaluation.

| Method | Mean participant macro-F1 | Worst participant | Lower decile | Interpretation |
|---|---:|---:|---:|---|
| Compact DANN | **68.084%** | 26.99% | 35.59% | Numerically highest target mean |
| Compact CORAL | 68.079% | 27.41% | **36.70%** | Effectively tied with DANN at supported precision |
| MoRe-HAR full | 63.533% | 25.84% | 26.60% | Preregistered hypothesis not supported |

The 95% participant-bootstrap interval for Compact DANN is [53.91%, 80.93%].
Canonical evidence:

- [`results/confirmatory/zero_shot_v1/publication_report_v1.json`](../results/confirmatory/zero_shot_v1/publication_report_v1.json)
- [`results/confirmatory/zero_shot_v1/model_summary_v1.md`](../results/confirmatory/zero_shot_v1/model_summary_v1.md)
- [Benchmark card](BENCHMARK_CARD.md)

## B. Matched InclusiveHAR source development

### B.1 Earlier controls and the project's six-channel development

The following results use 725 functional-core windows, P1-P10, five outer
participant folds and seed 11. SpectralShape, GSP and RMRP are project-developed
methods. The external control configurations were fixed; project feature
pipelines used four inner participant folds for candidate selection.

| Method | Origin | Window accuracy | Mean participant macro-F1 |
|---|---|---:|---:|
| HYDRA | External control | 73.517% | 70.711% |
| MultiRocket | External control | 76.138% | 73.501% |
| MultiRocket + HYDRA | External control combination | 76.000% | 73.616% |
| QUANT | External control | 76.966% | 74.120% |
| Budgeted RIST | External control | 78.759% | 77.195% |
| SpectralShape | Project feature pipeline | 79.724% | 78.884% |
| GSP | Project geometric/spectral representation | 83.448% | 82.916% |
| RMRP: selected denoised GSP | Project denoising/view-selection development | **84.414%** | **83.790%** |

RMRP exceeded the tested RIST configuration by 6.595 points and SpectralShape by
4.906 points. This establishes observed development improvement from the 70s to
the 80s under the shared source evaluation, with unequal selection budgets.
Against RIST, seven participants improved and three declined. RMRP's worst
participant remained below RIST's: 54.936% versus 61.681%.

RMRP added 0.874 points over GSP, with five participant wins, four harms and one
tie. Its selected model was denoised GSP; no residual-containing view was
selected. Noise/dropout tolerance improved, temporal-gap robustness worsened and
the joint advancement gate failed. This modest denoising increment is one part
of the earlier representation development, not the entire contribution.

Canonical evidence: [source-lineage audit](research/SOURCE_DEVELOPMENT_LINEAGE_AUDIT_20260922.md),
[tracked aggregate record](../results/research/source_development_lineage_v1.json)
and [FuSE/ReFrame report](research/FUSE_REFRAME_V2_RESEARCH_REPORT.md).

### B.2 Five-seed development of the project's retained methods

These results use the same 725 functional-core windows from P1-P10,
five participant outer folds, four inner folds, and seeds 11, 23, 47, 89, and
131. The participants were reused during method invention, so this is development
evidence.

| Method | Channels | Window accuracy | Mean participant macro-F1 | Change versus CTGR | Evidence decision |
|---|---:|---:|---:|---:|---|
| RMRP: selected denoised GSP | 6 | 84.579% | 83.953% | -2.586 points | Project-developed six-channel predecessor |
| CTGR | 9 | 86.979% | 86.540% | reference | Retained advancement; eight of ten participants improve versus RMRP |
| Strict HERA-CTGR v1 | 9 | **87.228%** | **86.849%** | +0.309 points | Highest matched mean; incremental gates failed |

Accuracy and participant macro-F1 are each averaged over the five seeds, without
ensembling their predictions. HERA-v2 full reached 86.749% participant macro-F1;
it remains a calibration ablation and was not promoted over v1.

The matching seed-11 CTGR and HERA rows are 86.897% / 86.474% and
87.034% / 86.755% respectively (accuracy / participant macro-F1). These appear
with the earlier methods in the [development-history table](RESEARCH_REPORT.md#421-seed-11-development-history).
Do not substitute a single-seed value for the five-seed aggregate.

CTGR adds native gravity channels, so its RMRP comparison is a sensor-sufficiency
and method comparison rather than a fixed-input architecture-only effect. Its
descriptive paired participant interval versus RMRP is approximately
[-0.05, +4.89] percentage points. Strict HERA's interval versus CTGR is
[-0.095, +0.713] points. Neither establishes independent superiority.

Canonical evidence:

- [`results/development/max_rnd_secondary_v1_summary.json`](../results/development/max_rnd_secondary_v1_summary.json)
- [`results/development/hera_ctgr_retrospective_v1_summary.json`](../results/development/hera_ctgr_retrospective_v1_summary.json)
- [`results/development/hera_ctgr_v2_retrospective_v1_summary.json`](../results/development/hera_ctgr_v2_retrospective_v1_summary.json)
- [Canonical method status](research/CANONICAL_HERA_CTGR_METHOD_STATUS.md)
- [CTGR/HERA model card](MODEL_CARD_CTGR_HERA.md)
- [Recomputed accuracy and matched prediction audit](research/REPORTED_METRICS_AUDIT_20260920.md)

The [fixed seed-11 source ablation](RESEARCH_REPORT.md#43-fixed-source-ablation)
compares all six flat GSP/gravity/estimator configurations with aligned CTGR and
HERA controls. Flat nine-channel RF reached 74.576% participant macro-F1, versus
86.474% for CTGR. RF's high HARTH thigh score comes from another pipeline and
endpoint; it is not evidence that RF exceeded CTGR on InclusiveHAR.

## C. Labelled personalization

The HERA semantic gauge queries one label per participant and excludes queried
windows from evaluation. On the 715 matched remaining source windows, active-hard
gating improved mean participant macro-F1 by 0.611 points; P10 improved and nine
people tied. A random-hard query achieved a similar change. This is a labelled
personalization diagnostic, not an additional zero-query architecture gain.

Canonical report: [HERA posture gauge](research/HERA_POSTURE_GAUGE_V1_RESULTS.md).

## D. External transfer and placement diagnostics

### D.1 AICOS: source-frozen transfer

Three classes, 38 complete development participants and 40,588 shared windows;
no fitting on AICOS.

| Method | Window accuracy | Mean participant macro-F1 | Result status |
|---|---:|---:|---|
| Six-channel base B6 | **76.28%** | 67.424% | Smaller signal budget |
| Flat nine-channel Extra Trees B9 | 68.10% | 64.638% | Flat gravity control |
| Frozen CTGR/T9 | 76.27% | **68.972%** | Conditional cross-device/position diagnostic |
| U9 unconditional routing | 73.31% | 68.926% | Failed nomination; sitting recall fell 12.79 points versus CTGR |

AICOS acceleration and derived gravity were corrected from SI units to the g
interface expected by the frozen source models. The acquisition logger's full
axis/polarity convention remains unresolved, so the result cannot confirm
ability-cohort generalization or a native-nine interface.

### D.2 HARTH: binary placement diagnostic

Sitting/standing, 22 participants, **five participant-exclusive folds** and
13,715 non-overlapping five-second windows within physical and annotated label
runs. Each arm uses the same rich-feature Random Forest recipe.

| Placement | Window accuracy | Mean participant macro-F1 |
|---|---:|---:|
| Lower back | 81.291% | 59.037% |
| Right thigh | **99.723%** | **97.261%** |
| Back + thigh | 99.672% | 97.198% |

The thigh-minus-back gain is 38.224 points, with a 95% paired bootstrap interval
of [30.320, 46.353] points; 21 participants improve, none decline and one ties.
This measures posture discrimination with annotation-constrained segmentation,
not a continuous deployment result. The separate confidence-gated arm reaches
86.263% participant macro-F1 and is reported in the research report.

### D.3 HARTH: multiclass comparator replay

Twenty-two leave-one-participant-out folds; merged nine-class, pooled sample
metrics over 6,461,328 samples. All scores below were executed in this repository.

| Method | Sample accuracy | Pooled macro-F1 |
|---|---:|---:|
| Project fused rich-feature RF | 93.56% | 85.37% |
| Reference-style SVM | 93.79% | 86.02% |
| Reference-style RF | 94.06% | 86.40% |
| Reference-style XGBoost | **94.22%** | **87.82%** |

All six executed arms and both the twelve- and nine-class endpoints appear in
the [research report](RESEARCH_REPORT.md#45-external-transfer-and-placement).
Filtering and aggregation differ from the historical paper, so these are
reference-style local comparisons rather than exact reproduced paper scores.

The HARTH binary and multiclass endpoints are not comparable to each other or to
InclusiveHAR. The right-thigh result supports a placement-information lesson;
it does not establish universal superiority of the project architecture.
The HARTH models were trained within HARTH. They are distinct from source-frozen
CTGR/HERA transfer and use standard RF learning with project feature pipelines.

Canonical reports:

- [AICOS unit and posture review](research/AICOS_POSTURE_REVIEW_20260919.md)
- [Fixed routing validation](research/CTGR_ROUTING_VALIDATION_20260919.md)
- [Research-lane closure](research/RESEARCH_LANE_CLOSURE_20260919.md)
- [Audited metrics and source bindings](research/REPORTED_METRICS_AUDIT_20260920.md)

## E. Latest candidate decision

U9 changed only the consultation rule for the existing posture expert. Against
matched T9/CTGR it changed mean participant macro-F1 by -3.442 points on the
source folds and -0.046 points on AICOS development. The source non-regression,
external practical-gain, and complete interface-qualification gates all failed.
The retained decision is therefore:

`retain_original_HERA_CTGR_close_this_routing_recipe`

No threshold, sign, blend-weight, encoder, extra seed, or target-cohort search is
pending from this experiment.

## Supported publication statements

- SpectralShape, GSP and RMRP are project-developed representations and selection
  procedures built from established components. Seed-11 RMRP exceeded the tested
  budgeted RIST control by 6.595 participant macro-F1 points, with a lower worst
  participant score and a different selection budget.
- The project improved its six-channel source-development mean from SpectralShape
  78.884% to GSP 82.916% and selected denoised-GSP RMRP 83.790% under the same
  seed-11 evaluation. Subsequent gravity-based gains have a different input budget.
- CTGR passed its predeclared source-development advancement gate and improved
  matched RMRP by 2.586 participant macro-F1 points under a nine-channel protocol.
- Strict HERA-v1 has the highest matched source-development point estimate, while
  its 0.309-point increment over CTGR remains uncertain and failed promotion gates.
- The historical locked target experiment found Compact DANN and CORAL effectively
  tied and did not support the preregistered MoRe-HAR hypothesis.
- HARTH provides evidence that sensor placement materially changes sitting versus
  standing observability under the executed protocols.
- The project preserves negative results, participant harms, corrections, and
  superseded evidence as first-class outputs.

## Statements not supported

- CTGR or HERA is independently confirmed, universally generalizable, or the
  best published HAR method.
- Locked target scores, few-person target-inclusion scores, source development,
  labelled personalization and HARTH diagnostics form one interchangeable
  improvement trajectory. Only comparisons within their declared contracts are
  valid; the shared source comparison in Section B.1 is one such comparison.
- AICOS establishes ability-associated transfer or a fully qualified coordinate
  interface.
- Synthetic gyro recovers unmeasured person-specific motion.
- Lower-back posture recognition is physically impossible.
- Multiple seeds replace independent human participants.

The current scientific disposition and next valid studies are summarized in the
[expert finalization assessment](research/EXPERT_FINALIZATION_ASSESSMENT_20260920.md).
