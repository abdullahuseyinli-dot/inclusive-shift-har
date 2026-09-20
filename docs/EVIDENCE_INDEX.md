# Current evidence index

**Status date:** 2026-09-20

**Purpose:** authoritative navigation and claim scope for the publication
candidate. This index does not modify any locked protocol or historical result.

## How to read the metrics

`Participant macro-F1` means macro-F1 is calculated separately for each person
and then averaged, so participants receive equal weight. `Pooled` metrics treat
all eligible windows or samples together. Accuracy, pooled macro-F1, and mean
participant macro-F1 are different estimands and are never substituted for one
another in this index.

Seeds measure model-fitting variation on the same people. They do not increase
the independent participant count. Results are comparable only inside a row
group with the same cohort, endpoint, channels, supervision, partitions, and
aggregation.

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

These results use 725 functional-core windows from source participants P1-P10,
five participant outer folds, four inner folds, and seeds 11, 23, 47, 89, and
131. The participants were reused during method invention, so this is development
evidence.

| Method | Channels | Mean participant macro-F1 | Change versus CTGR | Evidence decision |
|---|---:|---:|---:|---|
| RMRP | 6 | 83.953% | -2.586 points | Matched six-channel base |
| CTGR | 9 | 86.540% | reference | Retained advancement; eight of ten participants improve versus RMRP |
| Strict HERA-CTGR v1 | 9 | **86.849%** | +0.309 points | Highest point estimate; incremental gates failed |
| HERA-CTGR v2 full | 9 | 86.749% | +0.209 points | Calibration ablation; not promoted over v1 |

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

## C. Labelled personalization

The HERA semantic gauge queries one label per participant and excludes queried
windows from evaluation. On the 715 matched remaining source windows, active-hard
gating improved mean participant macro-F1 by 0.611 points; P10 improved and nine
people tied. A random-hard query achieved a similar change. This is a labelled
personalization diagnostic, not an additional zero-query architecture gain.

Canonical report: [HERA posture gauge](research/HERA_POSTURE_GAUGE_V1_RESULTS.md).

## D. External transfer and placement diagnostics

| Dataset and protocol | Method | Accuracy | Macro-F1 endpoint | Result status |
|---|---|---:|---:|---|
| AICOS development, 38 complete participants | Frozen CTGR/T9 | 76.27% | 68.97% mean participant | Conditional cross-device/position diagnostic |
| AICOS development, same windows | U9 unconditional routing | 73.31% | 68.93% mean participant | Failed nomination; sitting recall fell 12.79 points |
| HARTH binary sitting/standing, right thigh | Project rich RF | not a publication headline | 97.261% mean participant | Placement/observability diagnostic |
| HARTH binary sitting/standing, lower back | Project rich RF | not a publication headline | 59.037% mean participant | Weak back-only posture observability in this protocol |
| HARTH merged 9-class | Project fused RF | 93.56% | 85.37% pooled | Competitive diagnostic, below executed published-style XGBoost |
| HARTH merged 9-class | Published-style XGBoost | 94.22% | **87.82% pooled** | Strongest executed comparator in that lane |

AICOS acceleration and derived gravity were corrected from SI units to the g
interface expected by the frozen source models. The acquisition logger's full
axis/polarity convention remains unresolved, so the result cannot confirm
ability-cohort generalization or a native-nine interface.

The HARTH binary and multiclass endpoints are not comparable to each other or to
InclusiveHAR. The right-thigh result supports a placement-information lesson;
it does not establish universal superiority of the project architecture.

Canonical reports:

- [AICOS unit and posture review](research/AICOS_POSTURE_REVIEW_20260919.md)
- [Fixed routing validation](research/CTGR_ROUTING_VALIDATION_20260919.md)
- [Research-lane closure](research/RESEARCH_LANE_CLOSURE_20260919.md)

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
- Results in the 60s, 70s, 80s, and 90s form one improvement trajectory.
- AICOS establishes ability-associated transfer or a fully qualified coordinate
  interface.
- Synthetic gyro recovers unmeasured person-specific motion.
- Lower-back posture recognition is physically impossible.
- Multiple seeds replace independent human participants.

The current scientific disposition and next valid studies are summarized in the
[expert finalization assessment](research/EXPERT_FINALIZATION_ASSESSMENT_20260920.md).
