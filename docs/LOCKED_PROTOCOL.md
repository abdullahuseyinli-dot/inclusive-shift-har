# Conditionally locked InclusiveHAR v4 protocol

**Protocol:** `inclusivehar-released-block-v1.2`  
**Lock date:** 2026-08-23; exact lock time was not recorded  
**Evidence status:** participant-exclusive released-block protocol under an explicit user-authorized deviation  
**Target status:** sealed; no target predictions or performance have been inspected

This protocol is locked for local development subject to the residual risks below. It is **not trial-safe**. It does not erase the Stage 3 quarantine, authorize a confirmatory target opening, or claim that 2.56-second windows correspond to recovered recordings.

## Why a deviation was required

The official InclusiveHAR v4 CSV contains 396,602 data rows in 120 contiguous subject-label blocks, but no timestamp, trial, session, recording, or sample identifier. Historical dataset versions do not recover those fields. Consequently, hidden trial joins cannot be located or proven absent.

The user explicitly authorized a narrower protocol on 2026-08-23. The authorization permits participant partitioning followed by non-overlapping windows inside each released subject-label block, while requiring the remaining boundary uncertainty to stay visible. It permits the phrase **participant-exclusive released-block protocol** and forbids “trial-safe,” “trial-exclusive,” “hidden-join-free,” and equivalent claims.

Authoritative inputs are:

| Input | Canonical/content SHA-256 | Physical-file SHA-256 |
|---|---|---|
| Frozen Stage 3 audit | `673917d3bfd4b3f283a1ae521ae4a7d548e3facadd2ae89ce33b2e27153500e8` | `b85e641da9fdc3ac61b6c8440a78c2b6aee625c160b095c8170c0b9d1a4a4204` |
| Deviation authorization | `1c0d40e1cf65cab9371669a094ce35d6835f9bb6d5dd47c7dc04847917a44354` | `2986c318e4e86dcca86dc2408785e85ad8fadd0f7294133159e85a95eccdc82d` |
| Protocol configuration v1.2 | `d3425c88b0f003aa4a1a57524ceefc105d4f607c8bd2e9d859ff0056e3577220` | `639bd19a4c5da503c56e3ddc93eba77e07e35cdcb7c12e737d8d10829f2916df` |
| Ontology configuration v1.1 | `85f6d8ed911420f51c971da05539b739bad30c9e1816d376445abd18ade6d03e` | `c1d532caf873a2e580be9fa5ed897d2d408f4fc6f67434188057c5c333d340d9` |
| Preprocessing configuration v1 | `48515caf9da8e20510913baab0bcf9dd874af26e6ca415663e36be81d7848125` | `bb23327375f491708c81aa2a605b0a24e9a8274a20dee9ef66b8232cc9525145` |

## Window construction and provable guarantees

Participants are assigned to partitions before any window is extracted. For audited released block (b), let its inclusive data-row interval be ([s_b,e_b]), its length be (n_b=e_b-s_b+1), and:

\[
m_b=\left\lfloor\frac{n_b}{128}\right\rfloor,
\qquad
r_b=n_b-128m_b.
\]

The retained windows are:

\[
W_{b,k}=[s_b+128k,\ s_b+128k+127],
\qquad k=0,\ldots,m_b-1.
\]

The remaining (r_b\in[0,127]) rows are dropped. Tails are never padded, wrapped, concatenated to another block, or borrowed across participants or labels. Length and stride are both 128 in every partition. This proves:

- no participant overlap between source and target;
- no retained raw-row overlap, within or across partitions;
- no window crossing a released participant or activity boundary;
- no released block split across partitions;
- deterministic, byte-reproducible windows and split hashes.

It does **not** prove that a window stays inside one hidden trial. The nominal 2.56-second duration is conditional on the provider-declared but unverifiable 50 Hz sampling rate.

The locked construction yields:

| Quantity | Value |
|---|---:|
| Released blocks | 120 |
| Source rows | 396,602 |
| Retained windows | 3,042 |
| Retained rows | 389,376 |
| Dropped tail rows | 7,226 (1.822% of rows) |
| Source windows, IDs 1–10 | 1,443 |
| Sealed target windows, IDs 11–20 | 1,599 |

### Hidden-join contamination bound

Let (H_b) be the unknown count of hidden joins in block (b), and (C_b) the number of disjoint retained windows that cross at least one join. Because retained windows do not overlap:

\[
C_b\le\min(m_b,H_b).
\]

If—and only if—the paper’s three-repetition description is treated as exactly three contiguous hidden recordings with no other fragmentation, then (H_b\le2). Under that unverified assumption:

- aggregate bound: at most 240 of 3,042 windows, or **7.8895%**;
- source IDs 1–10: at most 120 of 1,443, or **8.3160%**;
- target IDs 11–20: at most 120 of 1,599, or **7.5047%**;
- worst released block: at most 2 of 12, or **16.67%**;
- worst participant aggregate: at most 12 of 134, or **8.9552%**.

The release does not establish the assumption. The unconditional audit-supported upper bound therefore remains **100% of windows**. A hidden join cannot create cross-partition leakage here because its complete released block belongs to one participant and partition, but it remains a signal-validity and boundary-artifact risk.

## Signals and prohibited proxies

The primary model interface is exactly, in order:

1. `motionUserAccelerationX`
2. `motionUserAccelerationY`
3. `motionUserAccelerationZ`
4. `motionRotationRateX`
5. `motionRotationRateY`
6. `motionRotationRateZ`

Identity, released group indicator, disability/device metadata, label columns, GPS/location fields, global row number, file order, released-block position, and window offset are provenance only and are forbidden model inputs. The perfect group-aligned activity block order makes row/order proxies especially dangerous.

## Locked ontology tracks

The ontology is read only from `configs/ontologies/inclusivehar_v1_1.yaml`; output dimensions may not be inferred from validation or target labels. Each runnable track has an explicit ordered numeric schema:

- Functional-core: `[mobility, sitting, standing]`, indices 0–2, schema SHA-256 `8fd23761b03ea39ebfc1fffc23430b31add0968fe6fc8522d035e7d0d9d704b4`.
- Inclusive-native: `[jogging, ramp_ascent, ramp_descent, sitting, standing, walking]`, indices 0–5, schema SHA-256 `5c4531052ffcf9fe2b2e3ea5b81a396349f68b4ebc8e56af3c1ab04551f2f197`.

| Track | Status | Locked meaning |
|---|---|---|
| Inclusive-native | Runnable, secondary | All six released labels. `Walking` preserves the released wording and must identify wheelchair realization as manual propulsion. |
| Functional-core | Runnable, primary | `Sitting→sitting`, `Standing→standing`, `Walking→mobility`. Mobility is deliberately functional and non-exact across ordinary gait and manual propulsion. |
| Cross-source-core | Blocked | Only sitting is currently exact. Standing is provisional; all-participant walking is not exact. At least two defensible exact classes must be locked before classification. |
| Ambulatory walking sensitivity | Blocked | Requires a predeclared ambulatory eligibility rule and remains exploratory. |

Ramp ascent/descent are not UCI-HAR stair labels, and jogging has no UCI-HAR v1 counterpart.

## Participant partitions and source development

Released IDs 1–10 are the conventional/source cohort. IDs 11–20 are the sealed target cohort. The source folds are derived without outcome data: lowercase SHA-256 values of `inclusive-shift-har|inclusivehar_v4|source_cv_v1|{subject_id}` are sorted and adjacent participants are paired.

| Source fold | Validation/outer-test participants | Windows |
|---|---|---:|
| 1 | 1, 4 | 294 |
| 2 | 6, 7 | 294 |
| 3 | 2, 3 | 272 |
| 4 | 5, 9 | 300 |
| 5 | 8, 10 | 283 |

For descriptive within-source performance, each pair serves once as an untouched outer test. Within each outer fold, each of the other four pairs rotates once as inner validation; the remaining six participants form inner training. Outer-test results are forbidden model-selection inputs.

For the locked source-to-target development run, the deterministic final split uses IDs 1, 2, 3, 4, 5, 6, 7, and 9 for training and IDs 8 and 10 for validation/checkpoint selection and calibration. Hyperparameters use source information only. The epoch count is the median source-fold best epoch; for a non-integral median use the lower integer, and the earliest epoch wins metric ties.

## Normalization, selection, and calibration

- Fit per-channel normalization independently for each active training fold, using exactly that fold’s training participants.
- Never fit normalization on source validation, a source outer test, any target participant, or pooled full-dataset statistics.
- Choose checkpoints using source-validation macro-F1 only.
- Fit calibration on source validation only.
- Keep tuning budgets equal across comparable models.
- Cache preprocessing only when the source artifact, preprocessing configuration, split manifest, and code-version hashes all match.

Stable window IDs are full SHA-256 values over canonical JSON binding the dataset/version, raw CSV hash, locked protocol hash, participant, released label/block, inclusive row interval, local offset, length, and stride. Every window stores `trial_id: null` and `trial_status: unrecoverable`.

## Zero-shot target seal

The primary zero-shot cohort is IDs 11–20 together. It is not a model-selection fold. Preprocessing, ontology, hyperparameters, seeds, checkpoint rule, normalization, calibration, thresholds, and reporting code must be frozen first.

Target seal ID: `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`.

The target may be opened at most once, and only after a separate `final_evaluation_unlock` record binds the split hash, protocol-lock hash, code commit, machine, reason, opening number, and successful tests/lint/types/manifests/split/artifact gates. No such unlock exists. Building and auditing the split use only the preserved descriptive block metadata and do not compute target predictions or metrics.

## Few-person inclusion curve

Few-person adaptation is separate from the sealed zero-shot endpoint and requires its own authorization after zero-shot completion. The outer evaluation pairs and nested inclusion prefixes are SHA-256-derived without performance information:

| Outer fold | Evaluation participants | Ordered inclusion candidates; use prefixes k=1,2,4 |
|---|---|---|
| 1 | 12, 18 | 13, 15, 17, 19, 16, 14, 11, 20 |
| 2 | 13, 15 | 18, 12, 17, 19, 16, 14, 11, 20 |
| 3 | 17, 19 | 18, 12, 13, 15, 16, 14, 11, 20 |
| 4 | 14, 16 | 18, 12, 13, 15, 17, 19, 11, 20 |
| 5 | 11, 20 | 18, 12, 13, 15, 17, 19, 16, 14 |

Evaluation participants never enter training, validation, normalization, calibration, or threshold selection. Hyperparameters remain fixed from source-only development, and normalization reuses the locked source-training statistics. Nonselected target participants remain unused in each scenario.

## Endpoints fixed before target opening

The primary endpoint is mean participant-level macro-F1 on the target cohort, weighting participants equally. The co-required lower-tail endpoint is worst-participant and lower-decile macro-F1. The absolute source macro-F1 non-inferiority margin is 0.02. Final model comparisons use seeds 11, 23, 47, 89, and 131, participant-clustered uncertainty, paired participant-level inference where appropriate, effect sizes, and multiple-comparison correction by model family.

The model hypothesis is supported only if mean target participant macro-F1 and lower-tail performance both improve against the strongest adequately tuned baseline without source degradation beyond the predeclared margin. Negative results remain reportable evidence.

## Machine-readable split evidence

- Split manifest: `results/protocol/splits/inclusivehar_v4_released_block_v1_2.json`
  - embedded split SHA-256: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
  - physical file SHA-256: `af909c7d914b68c417b238b576945c112aa58740859a7d998b6b77219dfb1920`
- Split audit: `results/protocol/split_audit_v1_2.json`
  - embedded report SHA-256: `45c49761a92b9f9f3e4cabe4e3a9dbf72b4114795948ba7104f8ed0dff37bdd8`
  - physical file SHA-256: `08dbba5b68514e5ff0f379710aedf21dcdaad6b6e71d264b5c59cfebb924ef6e`
  - status: `pass_conditional_released_block`; zero structural errors and three explicit residual-risk warnings
- Source-only materialization manifest: `results/protocol/source_windows/inclusivehar_v4_source_development_v1_2.json`
  - 1,443 source windows and no target subject/window records
  - embedded source-window SHA-256: `1ad1ee3accaae5f2f93bb91ac0afa5ce583134ce1d882c3f08323b09026fa522`
  - physical file SHA-256: `1d49435ad9371a9701198e15ec30a35a5c3c38ee5458f6f1980de7d4306e76e2`
- Conditional protocol lock: `results/protocol/protocol_lock_v1_2.json`
  - embedded record SHA-256: `71f3fdb2de0e6e0aa8feaff8b8726bfbc999fd1423568251744a29ab541e9e3a`
  - physical file SHA-256: `d39548f5d477114a160eda2f93f02921a347bc8ed4146001a39378bf106b0e29`
- Protocol artifact manifest: `results/protocol/artifact_manifest.json`
  - canonical SHA-256: `91189d17026f12b795504d57a2e92c740e84a50d9a82f3fac8988fd4d965a7c6`
  - physical file SHA-256: `c3ccf1cf03f8eeef1b91f59f58ee6f4be37122bef793aa1202818d0dd6836528`

Version 1 and v1.1 evidence remain preserved and reconstructable. The two supersession records state that v1.1 added explicit nested source cross-validation and v1.2 added ordered numeric class schemas. No version opened target performance.

Rebuild and audit without reading raw signal values:

```powershell
uv run inclusive-shift-har build-splits --json
uv run inclusive-shift-har audit-splits --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json --json
uv run inclusive-shift-har build-source-windows --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json --output results/protocol/source_windows/example-new-version.json --json
```

The build command is create-only and refuses to replace existing evidence. Use a new versioned destination for any authorized future revision.

## Remaining risks and claim limits

- Hidden joins are unrecoverable; the unconditional contamination bound is 100%.
- Sampling rate and 2.56-second duration are provider-declared, not timestamp-verified.
- Per-block remainder dropping is deterministic but may introduce tail-selection effects.
- Activity order is perfectly group-aligned in the released file; row/order features are prohibited.
- Activity realizations and semantics differ across participants and assistive-device use.
- Small target subgroups do not support causal disability, fairness, or clinical-validity claims.
- This protocol does not establish publishability, novelty, state of the art, or a positive MoRe-HAR result.
