# Expert assessment: finalize the current research phase

Assessment date: 2026-09-20. Decision: **finish the repository, documentation and
reproducibility package; do not launch another architecture search now.**

Three separate read-only reviews covered statistical evidence, architecture and
physical information, and engineering/release readiness. All recommend this
direction. This is readiness to begin finalization work, not a declaration that
the current worktree is release-ready. No new model fit, target evaluation,
publication, commit or cleanup was performed for this assessment.

## What the work has established

| Evidence | Result | Interpretation |
|---|---|---|
| Matched original source, five seeds | RMRP 83.953% to CTGR 86.540% participant macro-F1; +2.586 pp; eight of ten people improve | The strongest advancement in this development programme. It adds native gravity and selective posture correction, so the gain is not an architecture-only effect at an equal sensor budget. |
| Strict HERA versus CTGR, same five-seed source contract | HERA 86.849%; +0.309 pp; descriptive paired interval [-0.095, +0.713] pp | Highest zero-query point estimate. Incremental gain and bottom-tail gates failed; keep CTGR as the primary retained control and HERA as a candidate/ablation. |
| Labelled HERA semantic gauge | +0.611 pp on matched 715 remaining windows; one person improves, nine tie | A personalization diagnostic, not an additional zero-query architecture gain. Random hard querying is similar. |
| Latest unconditional posture routing U9 | Source -3.442 pp versus T9; AICOS development -0.046 pp | The favorable earlier eight-person result did not repeat in the broader comparison. Large individual harms and posture recall tradeoffs prevent promotion. |
| HARTH matched binary placement comparison | Back 59.037%, thigh 97.261%, fusion 97.198% participant macro-F1 | Strong evidence that placement matters in this experiment. This is not the original HERA endpoint. |
| HARTH executed merged nine-class comparison | Project fused RF 85.37%, published-style XGBoost 87.82% macro-F1 | Useful comparative evidence, without leadership or exact historical-paper replication claims. |

The original source cohort contains only ten repeatedly used people. Five seeds
do not create five independent cohorts. CTGR passed its predeclared development
advancement checks, but its descriptive participant interval for the RMRP
difference is approximately [-0.05, +4.89] pp. Neither CTGR nor HERA has thereby
established independent superiority. The 90s, 70s and 80s reported across project
history involve different tasks, participants, supervision and aggregation;
subtracting those headline scores is not a valid improvement estimate.

Canonical sources:

- [Current method status](CANONICAL_HERA_CTGR_METHOD_STATUS.md).
- [CTGR five-seed results](MAX_RND_SECONDARY_RESULTS.md).
- [HERA matched results](HERA_CTGR_RETROSPECTIVE_V1_RESULTS.md).
- [Labelled gauge results](HERA_POSTURE_GAUGE_V1_RESULTS.md).
- [Latest fixed routing validation](CTGR_ROUTING_VALIDATION_20260919.md).
- [HARTH and research-lane closure](RESEARCH_LANE_CLOSURE_20260919.md).
- HARTH comparator artifact: `.audit/harth_published_replication_20260918/run-007/REPORT.md`.

## Why another similar run has low value now

The latest routing intervention already tested a concrete explanation for posture
errors with the original experts and matched controls. It failed its prospective
gates. Earlier compact-feature blending regressed, HERA-v2 rescue routing remained
disabled in all 25 outer folds, and calibration gains did not establish better
classification. HARTH geometry, neural and decoder trials did not consistently
improve both posture classes and participant outcomes.

Repeatedly selecting methods using these same participants risks selecting cohort
peculiarities. This concern is consistent with primary research on
[selection bias from optimizing finite-sample validation estimates](https://www.jmlr.org/beta/papers/v11/cawley10a.html).
It does not mean that all architectures or temporal methods have been exhausted.
The original source release lacks authenticated session/timestamp ordering, so a
valid temporal test cannot be fabricated from file order.

The strongest counterargument is HERA's label-informed expert-selection oracle:
88.618% versus CTGR's 86.540%. This shows possible complementary information in
the experts, but the oracle uses held participant labels. It supplies no reliable
deployable rule for choosing those experts. The attempted gates have not solved
that problem. Treat it as a research question, not an attainable score claim.

AICOS also remains conditional: its acceleration/gravity unit defect was repaired,
but the logger's axis/polarity provenance remains unresolved. Native gravity and
low-pass-derived gravity have different evidence status. These limits do not
invalidate the original source result or justify declaring a model family unable
to generalize.

## Value and novelty

The project is worth preserving and making usable: it contains an implemented
gravity-residual method, matched development comparisons, participant harms,
negative interventions and explicit corrections. That supports an honest
exploratory research-software/evidence package. It does not establish a globally
novel architecture, state of the art, universal posture recognition or independent
ability-cohort generalization. A code/archive release does not require inventing
a stronger claim or obtaining another decimal of development F1.

Reference-based posture information is also established prior art. A
[hip-sensor study](https://pubmed.ncbi.nlm.nih.gov/29144567/) used walking-derived
orientation to distinguish sitting and standing; a
[reference-frame study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10346883/) describes
limitations of motion-derived forward direction during static postures. These
support testing authentic contextual information, not claiming that reference
orientation itself is new. Weak back-only performance is observed here; universal
physical impossibility has not been proved.

## Finite finalization plan

1. **Create one current evidence index.** Update `README.md`,
   `docs/PROJECT_STATUS.md` and `results/README.md` to point to it. The first two
   currently lead with 5 September and 4 September snapshots. Remove canceled
   campaigns from the current pending-work list while retaining their history.
   Each result row must declare dataset/cohort, classes, channels, seeds,
   supervision, accuracy, pooled macro-F1 versus mean participant macro-F1,
   uncertainty, participant harms, evidence status and canonical artifact.
2. **Add a supersession map without rewriting sealed evidence.** Link the original
   AICOS results to the unit correction and then to the completed U9 rejection.
   Record that AICOS is consumed: the older qualification document and portfolio
   YAML still describe it as unconsumed. Keep the current native-nine unit repair
   separate from the older preparation receipt: that receipt predates the changed
   runner. Produce a fresh no-fit preparation receipt if this interface is to be
   advertised as currently prepared. Retain original locked MoRe-HAR outcomes and
   label later HERA/CTGR development separately.
3. **Make evidence reconstruction portable.** Export reviewed aggregate tables and
   a small provenance manifest from saved predictions, with a documented command
   to reproduce the tables without training. The current runner relies on local
   `.audit/` files and another evidence worktree; a clean clone is insufficient.
   Document exactly which separately restored artifacts are required. Review
   dataset/privacy/licensing rights before distributing any predictions; do not
   automatically include raw signals or checkpoints.
4. **Integrate and validate first-party code.** At review entry the worktree had
   three modified and 27 untracked paths, including 11 `.tmp_harth_*` reference
   files. Review first-party changes in logical groups and preserve third-party
   references outside Git. Run the full applicable locked CPU/research test,
   lint, format, type, configuration, manifest, split and artifact checks on the
   candidate. The previous 27 tests cover the latest change, not the whole release.
   No new data evaluation is needed for these engineering checks.
5. **Write a CTGR/HERA method card.** Specify the preserved model bytes and source
   selection, six/nine channels, native/derived gravity, g/SI conversion, gyro
   rad/s, coordinate uncertainty, routing, calibration and fallback behavior.
   Separate zero-query inference from labelled personalization. Keep the older
   benchmark and MoRe-HAR model cards as historical records.
6. **Complete provenance and scoped release checks.** The temporary HARTH reference
   receipt does not yet establish redistribution rights. Resolve this before
   including such material. Align citation/metadata with the actual package and
   evidence. Select checks appropriate to this CPU diagnostic/software package;
   the old CUDA publication recipe and canceled campaign are not new research
   prerequisites. Publishing, remote pushes, tags and deposition remain separate
   actions after the concrete candidate is ready.

Completion means that a fresh checkout plus explicitly documented permitted
artifacts can reproduce the retained tables, the applicable software gates pass,
and no current document presents superseded, development or corrected results as
independent confirmation. Raw evidence and historical hashes must remain intact.

## When to reopen research

Reopen only for a distinct question with a new information source or independent
people. There are two separate existing options:

- **Substantiate the current model:** evaluate frozen CTGR and the native-nine B9
  control on a fresh, qualified native-nine cohort. The prepared design estimates
  38 complete participants for a two-point effect under an uncertain variance
  assumption from ten source people. This is a planning assumption, not a
  guaranteed sample size or a prerequisite for documenting existing results.
- **Test a new physical-information hypothesis:** the six-wearer, 264-bout
  same-attachment pilot asks whether measured local reference bouts improve both
  posture classes without mobility or participant harms. Start with its prescribed
  zero-fit identifiability test. Do not substitute generated gyro or unauthenticated
  file order for the missing measurements. This pilot would not confirm broad
  population generalization.

Neither qualified human input is currently supplied. Both options remain future
studies, not pending training from this assessment. P11-P20 stays closed. See
[confirmation preparation](CTGR_NATIVE9_CONFIRMATION_PREPARATION_20260919.md),
[the reference protocol](SAME_ATTACHMENT_DIRECTIONAL_REFERENCE_GEOMETRY_V1_PROTOCOL.md)
and `.audit/research_direction_20260913-001/NEXT_EXPERIMENT_SPEC.md`.

## Review record

Active worktree: `.codex_tmp/har_next_phase_20260908`, branch
`research/har-substantiation-pilot-pipeline-20260908`, HEAD
`279c53d2ace6f28546c2246c866fdcaa324301a6`.

The latest routing review's 11-file manifest was independently rechecked during
this assessment with zero hash mismatches. Prior experiment validation reported
27 passing tests and independently recomputed metrics; those tests were not
rerun as part of this read-only research review. Expert roles and the append-only
knowledge delta are recorded under
`.audit/expert_finalization_assessment_20260920-001/`.
