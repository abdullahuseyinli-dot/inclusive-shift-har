# Physical information signal analysis v1

This module implements the zero-fit signal analysis defined by the frozen
`PHYSICAL_INFORMATION_IDENTIFIABILITY_V1_PROTOCOL.md`. It begins only after a qualified
device export has been converted into the canonical arrays below. It is not a vendor or
device adapter, does not collect data, and does not make missing human-study, device, or
collection prerequisites true.

The implementation is split between:

- `inclusive_shift_har.data.physical_information_signal`, the pure signal, projection,
  prototype, margin, energy, aggregation, uncertainty, and gate functions; and
- `inclusive_shift_har.experiments.physical_information_signal`, the create-only controller,
  incomplete/blocker recorder, artifact sealing, and independent zero-fit replay.

## Required device signal contract

The controller rejects absent, extra, ambiguous, or unsupported contract fields. The JSON
object has `record_kind: physical_information_signal_contract`, `schema_version: 1.0.0`,
`status: qualified_and_processing_frozen`, and one of these evidence statuses:

- `development_physical_measurement`
- `synthetic_fixture_not_physical_evidence`

It requires actual device ID, manufacturer, model, firmware, logger, axis convention, clock
domain, processing-freeze UTC, and SHA-256 values for the equipment receipt, placement
instruction, bench qualification, and device-specific canonicalization receipt. These hashes are provenance
bindings; the analyzer does not create the underlying qualification evidence.

For `development_physical_measurement`, `--qualification-manifest` is mandatory. Its
self-hashed JSON uses `record_kind: physical_information_signal_qualification_manifest`,
`schema_version: 1.0.0`, and `equipment` and `canonicalization` references, each containing
only `path` and `sha256`, relative to `--canonical-root`. Equipment uses the existing
`physical_information_equipment_receipt` schema and `validate_collection_records`; this
is not a second equipment schema. The controller binds device metadata, units, channel
mapping, clock, processing contract, and actual placement/bench file bytes to the signal
contract. It copies and revalidates all four qualification files for replay.

The self-hashed canonicalization receipt has
`record_kind: physical_information_canonicalization_receipt`, `status: reviewed`,
`device_id`, `equipment_receipt_sha256`, `processing_contract_sha256`, `reviewer_id`,
`adapter_code_sha256`, and `reviewed_at_utc`. The processing hash is the canonical JSON
hash of `processing_contract`. Equipment qualification must precede adapter review, and
adapter review must precede or equal processing freeze. This checks receipt provenance;
it does not execute or authenticate an external adapter. `qualification_validation.json`
leaves study prerequisites and physical-evidence qualification false. Synthetic fixture
mode does not require these receipts and can never establish physical claims.

`channel_names` maps the source export's distinct `sample_id`, `timestamp`,
`recording_break_before`, `ax`, `ay`, `az`, `gx`, `gy`, and `gz` columns. When the device
declares native gravity availability it also maps `gravity_x`, `gravity_y`, and `gravity_z`.
Canonical input retains the mapping even though the NPZ arrays use the fixed names below.

`units` declares:

- timestamp: `s` or `ms`
- acceleration and gravity: `m/s^2` or `g`
- angular velocity: `rad/s` or `deg/s`

`acceleration_semantics` is exactly `total_acceleration_including_gravity` or
`linear_acceleration_gravity_removed`. Derived gravity is valid only for total acceleration.
Provider-native gravity requires the three native channels and a null cutoff. A device may
expose native gravity while a separately frozen derived lane ignores it; the two sources are
never relabeled as each other.

`processing_contract` requires positive source and target rates, an absolute maximum gap in
seconds, `linear_interpolation_v1`, positive window and stride sample counts, and one gravity
choice:

- `provider_native_gravity` with `gravity_cutoff_hz: null`; or
- `derived_causal_lowpass` with a positive cutoff below the source and target Nyquist rates.

`frame_conditioning_limit` is null until bench uncertainty justifies a positive frozen limit.
Numerical nondegeneracy alone never fills it in.

## Canonical continuous recordings

Each canonical `.npz` contains exactly these non-object arrays:

| Array | Shape/type | Meaning |
| --- | --- | --- |
| `sample_ids` | `(N,)` integer | Unique source sample identity |
| `timestamps` | `(N,)` numeric | Source timestamps in the declared unit |
| `acceleration_xyz` | `(N,3)` numeric | Acceleration with the declared semantics/unit |
| `gyroscope_xyz` | `(N,3)` numeric | Angular velocity in the declared unit |
| `recording_break_before` | `(N,)` boolean | Explicit reset before this sample |
| `native_gravity_xyz` | `(N,3)` numeric | Required exactly when native gravity is available |

The collection manifest is a self-hashed JSON object with
`record_kind: physical_information_canonical_collection_manifest`, `schema_version: 1.0.0`,
and exactly one recording for every frozen attachment block. Each row binds `recording_id`,
`block_id`, wearer, visit, attachment, device ID, safe relative `canonical_path`, file SHA-256,
and byte size. Each row also requires `visit_started_at_utc`, finite `started_at` and
`ended_at` in common visit-clock seconds, boolean `source_clock_reset_observed`, and
nullable `clock_mapping_receipt`. The visit UTC anchor must follow processing freeze;
attachments must share that anchor and have ordered, nonoverlapping intervals. Wearer
visits cannot reverse or overlap. Canonical timestamps, converted to seconds, must lie
inside the declared interval; a second attachment cannot silently restart its clock.

An observed source-clock reset requires a reviewed mapping receipt referenced by safe
relative `path` and `sha256`. The self-hashed receipt binds
`record_kind: physical_information_clock_mapping_receipt`, `status: reviewed`,
`recording_id`, `canonical_sha256`, `clock_domain`, `reviewer_id`, `mapping_description`,
and `reviewed_at_utc`. The importer remains responsible for preserving original raw
timestamps and implementing the reviewed mapping; this controller verifies and copies
the receipt, not the external adapter transformation.

A device-specific importer must create these arrays and its qualification
receipt. This repository intentionally supplies no generic CSV guesser and no silent channel
or unit defaults.

Annotations use the already frozen nine-field annotation record: bout and recording IDs,
half-open source sample bounds, observed activity/motion, adjudication status, adjudicator,
and adjudication UTC. All 264 frozen bout IDs remain present; failures become explicit
ineligible projections rather than disappearing.

## Deterministic computation

Input recording boundaries are attachment boundaries. Within a recording, signal processing
resets only at an explicit recording break, a nonfinite timestamp/inertial/gravity row, a
nonincreasing timestamp, or a timestamp gap above the frozen maximum. An activity annotation,
bout edge, role, or later scoring decision is never a preprocessing boundary. The nominal
timestamp grid retains slots across finite-clock sensor dropout and positive timestamp gaps;
those slots and every window touching them remain explicit invalid candidates. A reset or
nonincreasing clock starts a separately identified timestamp epoch.

The derived lane applies a causal first-order gravity low-pass on each source run using each
observed time delta and resets at the run start. Every signal is then linearly interpolated to
a left-aligned target-rate grid. Provider-native total acceleration produces
`linear = total - gravity`; provider-native linear acceleration produces
`total = linear + gravity`. Derived gravity requires total acceleration and produces
`linear = total - derived_gravity`.

Full windows start at offsets `0, stride, 2*stride, ...` inside each nominal timestamp epoch.
Their analysis interval is `[start, start + window_samples/target_rate)`. Every window records
its timestamp epoch, signal run, and half-open raw source-sample support. Window gravity
direction is the normalized arithmetic mean gravity. A mean norm at or below `1e-12` is invalid.
Dynamic-acceleration and gyroscope RMS use the equations in the frozen protocol. The signal
table function accepts no annotation argument, and its array/content hash is invariant to
annotation changes.

Only after window creation does projection use each adjudicated half-open sample interval.
It adds and subtracts the fixed five-second trim and admits full grid windows inside that
interior whose raw source support is also contained in the annotated sample interval. This
prevents overlapping reset-clock epochs from entering the wrong bout. The denominator is every
such nominal grid window, including dropout/gap-invalid rows; the numerator is valid windows. A bout
requires at least three valid windows and an 80 percent valid fraction. Every bout record
contains both counts, both ID lists, the fraction, and an explicit eligibility reason.

For each reference and motion stratum, sitting and standing prototypes are normalized means
of valid support-window directions. The output retains support-window IDs, median and p90
spread, between-posture angle, inverse-absolute-sine condition, numerical degeneracy, and the
separate bench-limit frame decision. Fresh and stale margins use the exact same ordered valid
query-window IDs; a mismatch aborts rather than changing the comparison denominator.

Aggregation is window to bout, bout to condition/reference, condition/reference to wearer,
then equal wearer. A complete wearer has all 16 planned fresh posture query bouts metadata and
signal eligible. Fresh and stale wearer summaries additionally require every component of that
estimand, and stale summaries require membership in the fresh-complete wearer set. Partial
values are retained with exclusions and never averaged as if complete.
Student-t intervals use wearer as the independent unit and are suppressed below two complete
estimands. Stale results are descriptive and receive no threshold.

The feasibility gate passes only when the exact six frozen wearer IDs are complete and every
wearer's fresh mean signed margin is strictly positive in both quiet and upper-body-motion
posture queries. Passing permits only a new protocol draft. The analyzer keeps physical,
population-superiority, and architecture claims false. Synthetic fixtures never substitute
for physical evidence.

## Commands and artifacts

When no real canonical collection is available, record that fact after the code commit is
locked and the worktree is clean:

```powershell
python -m inclusive_shift_har.experiments.physical_information_signal blocker `
  --repository-root <worktree> `
  --plan <plan-package>/plan_manifest.json `
  --config <worktree>/configs/experiments/physical_information_identifiability_v1.yaml `
  --output <new-create-only-run-directory> `
  --code-commit <full-HEAD-sha>
```

This writes a self-hashed `INCOMPLETE.json`, artifact and completion manifests, validation,
and final controller receipt. It reports zero loaded outcomes, zero fits, no workers, the
exact missing external prerequisites, and `physical_identifiability_gate.status` as
`not_evaluated`. The output directory must not already exist.

After lawful collection, qualification, canonicalization, and adjudication exist:

```powershell
python -m inclusive_shift_har.experiments.physical_information_signal execute `
  --repository-root <clean-worktree> `
  --plan <plan-package>/plan_manifest.json `
  --config <worktree>/configs/experiments/physical_information_identifiability_v1.yaml `
  --contract <signal-contract.json> `
  --qualification-manifest <qualification-manifest.json> `
  --collection-manifest <canonical-collection-manifest.json> `
  --annotations <annotations.json> `
  --canonical-root <canonical-input-root> `
  --output <new-create-only-run-directory> `
  --code-commit <full-HEAD-sha>
```

The package copies hash-bound canonical inputs for replay and writes `input_manifest.json`,
snapshots, `signal_audit.json`, `signal_cache.npz`, `bout_projection.json`, `analysis.json`,
`result.json`, `runtime.json`, `worker_shutdown.json`, `artifact_manifest.json`,
`validation.json`, `completion_manifest.json`, and `controller_final_receipt.json`. Any
failure or Python interruption after output creation is retained with a sealed
`INCOMPLETE.json`, `failure_reason.json`, `failure_runtime.json`, `failure_shutdown.json`,
`failure_manifest.json`, and `failure_final_receipt.json`; existing evidence is not deleted
or overwritten. Already-written blocker/completion records remain intact, and the separate
failure reason retains the interruption. Abrupt process termination or unwritable storage
cannot guarantee a final receipt. This controller launches no workers or monitors.

The declared clean repository must be the checkout containing the executing controller
and signal module; their byte hashes are included in provenance. Canonical input copies
must match the original manifest bytes and size before analysis begins.

Independent read-only replay is:

```powershell
python -m inclusive_shift_har.experiments.physical_information_signal validate `
  --run-directory <completed-run-directory>
```

Replay reloads the copied canonical arrays, validates the frozen plan/config snapshots,
recomputes signal windows, projections, prototypes, estimands, aggregation, and gate with zero
fits, and requires exact agreement with the sealed analysis, projection, signal audit,
exposed result fields, gate/claims, and cache hashes. It revalidates annotation schemas,
qualification receipts, recording chronology, and source/copy identity. Artifact and
completion manifests require unique entries and complete file coverage; final receipt and
completion hashes are verified. A run with `INCOMPLETE.json` cannot pass signal replay.
