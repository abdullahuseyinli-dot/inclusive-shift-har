# External provider-boundary and split-timing correction

Declared 2026-09-05 at 16:19:47 UTC after the current IMU-HAR-IL classical
outcomes had been generated. This is a claim-narrowing, post-outcome correction,
not a prospective model amendment. It preserves all earlier protocols and results.
The machine-readable record is
`configs/protocols/external_har_provider_boundary_audit_v1.yaml`; evidence roles
are corrected create-only in `configs/datasets/evidence_roles_20260905_v3.yaml`.

## Repository-governance source audit

The parent workspace `AGENTS.md` names `docs/MASTER_PLAN.pdf` as authoritative,
but that file is absent from the workspace. The only matching document is
`docs/MASTER_PLAN.docx`; its contents and surrounding tree govern the separate
robot-control project that the handoff explicitly excludes. This HAR repository's
own `AGENTS.md` instead names `docs/LOCKED_PROTOCOL.md` and the hashed records
under `results/protocol/` as its scoped authorities. This correction follows both
sets of preservation rules, records the missing parent path rather than inventing
a PDF, and does not import robot Stage 1 requirements into the wearable-HAR
experiment protocol.

## Corrected finding: IMU-HAR-IL files are not physical trials

The v2 physical-grid note said that IMU-HAR-IL provider files define physical
trials. That statement is false. The official paper's data-preprocessing section
says the sensors remained active through a continuous repetition, after which the
provider split it into activity segments **using activity labels**, manually
inspected/corrected segments, and removed break periods. The official repository
documents the released participant/repetition/activity/sensor-file hierarchy and
says the raw data were segmented into 18 activities with breaks removed.

Primary sources:

- Official paper, arXiv v2: <https://arxiv.org/html/2608.07502v2>
- Official author repository at audited commit
  `7d8244db71f5f15e094664df1ebb02315769f0ba`:
  <https://github.com/mmsandhu/IMU-HAR-IL>
- CSIRO collection 74700: <https://data.csiro.au/collection/csiro:74700>

The audited official README has SHA-256
`1dfcca279400537367b4aa4bbe776a64587c5f7d72a21a1fb2ebe61e2fc81579`.
CSIRO API metadata reports collection version 1/data version 1, 83,409 files,
49,771,532,800 bytes, and CC BY-NC 4.0. A published `Body-WT.csv` contains only
Euler, acceleration, gyroscope, and activity-label columns. It contains no
timestamp, sample index, original-session offset, or continuous-stream identifier.

The repository loads each activity CSV independently. Therefore causal-gravity
state, polyphase-resampling edges, and window phase reset at an upstream
ground-truth boundary. Relabelling samples inside a released file does not drive
the numerical transform, but changing the unavailable provider segmentation
would. The public release is insufficient to repair this honestly.

All IMU-HAR-IL within-dataset scores are henceforth
`diagnostic_provider_presegmented` development evidence. IMU-to-FoG inference can
still be described as target-label-blind, but the full pipeline is not a
label-free continuous-stream transfer because its source preprocessing is
provider-label-boundary-conditioned. Artifact hashes, participant-exclusive
fitting, and reproduced metrics remain useful; none cures the semantic defect.

The smallest way to clear the blocker is an authorized provider release of the
original continuous repetition/session files with timestamps or sample offsets
and activity intervals. The repository must then resample each complete
observable segment once and project annotations afterwards. We will not fabricate
chronology by concatenating activity files.

## Participant partition timing

The repository rule and external R&D protocol require participants to be
partitioned before segmentation/windowing. Historical external loaders instead
materialized all windows first; experiment runners then derived deterministic
participant folds. A retained split audit found no participant overlap and the
recorded maps are deterministic, exhaustive, and disjoint. Learned preprocessing
and model fitting remained training-only. Nevertheless, this is a protocol-
governance defect: a post-window roster can be altered by annotation eligibility,
and an empty record set can pass the old overlap-only validator vacuously.

Replacement runs must bind a pre-window participant plan derived from observable
provider inventory. The plan must include the roster, seeds, outer and inner
assignments, and hashes. Removing or permuting annotations may change scored-window
eligibility but must not change that roster, any partition, signal transforms,
gravity, timestamps, or candidate window grid.

The corrected FoG session-grid values remain annotation-independent, and all 22
real participants happen to occur in both its observable and scored pools, so the
discovered timing defect is not evidence of numerical leakage in that retained
run. The run is still superseded for the stricter governance contract and will be
rerun from a clean correction commit. Historical results remain untouched.

## Other loaders

InclusiveHAR is intentionally unchanged: its locked deviation windows inside
released participant/activity blocks because timestamps and physical trial joins
are absent. It remains participant-exclusive released-block evidence, never
trial-safe or streaming-valid evidence. UCI-HAR and DAGHAR load provider-created
window arrays; local ontology filtering happens after those provider windows
already exist, but original continuous boundaries and provider window phase cannot
be audited locally. They therefore support only their recorded development or
consumed-replication roles, not a claim that this repository constructed an
annotation-independent continuous grid.

HAR-PMD is a scripted mobility-mode stress endpoint whose timestamped activity/
environment CSVs are provider recording units. It cannot validate continuous
free-living inference, clinical ability, or sitting-versus-standing behavior.
Native-nine-channel and six-channel results remain separate.

Sole-HARmony's dataset-level role is consumed development evidence, with two
explicitly separate diagnostic lanes. Historical camera-bout materialization
resets interpolation, windows, and temporal state at annotation boundaries, so
that lane is an oracle-boundary diagnostic only. The session-observable lane uses
session/timestamp-gap/finite-signal boundaries, projects camera annotations post
hoc, and resets temporal state only on observable events; it is a temporal
development diagnostic, not confirmation. Its offline symmetric resampling also
means it does not establish zero-lookahead deployment. Neither lane supports a
clinical or deployable claim, and their different preprocessing tuples prohibit a
numerical before/after comparison.

No finding here upgrades any dataset to prospective confirmation, supports a SOTA
claim, or makes results with different preprocessing tuples numerically comparable.
