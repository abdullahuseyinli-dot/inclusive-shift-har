# Data audit

## Current gate decision

InclusiveHAR v4 passes artifact, schema, numeric-integrity, identity, label, and participant-coverage checks. The original Stage 3 trial-safe split/windowing gate **failed and remains failed**: the released CSV has no timestamp, trial, session, recording, or raw-sample identifier, so the three repetitions reported by the article cannot be separated reliably. A later explicit deviation authorized a narrower participant-exclusive released-block protocol after recording the unconditional hidden-join risk bound. Training and the one-time target evaluation proceeded only under that superseding conditional protocol; they do not retroactively make the dataset trial-safe or erase this audit failure. See `docs/LOCKED_PROTOCOL.md` and the immutable deviation/supersession records under `results/protocol/`.

Machine-readable evidence: [`results/data_audit/inclusivehar_v4.audit.json`](../results/data_audit/inclusivehar_v4.audit.json).

Evidence status: descriptive read-only data audit, not development or confirmatory model evaluation.

## Authorized scope

The audit was authorized by `results/gates/raw_data_read_access_inclusivehar_v4.json`, which is bound to the original starter-manifest SHA-256 `e3fc4e0f21cdd580b930a1a694923a43e222957335fde8eb08960e007f4930bb`. That exact manifest is preserved at `manifests/history/inclusivehar_v4.e3fc4e0f21cdd580b930a1a694923a43e222957335fde8eb08960e007f4930bb.json`. The current manifest supersedes it with observed schema facts; it does not rewrite the authorization lineage.

No splits, windows, fitted preprocessing, models, calibration, or target evaluation were created.

The privacy-safe report can be reproduced to standard output without writing derived raw data:

```powershell
inclusive-shift-har audit-data --manifest manifests/history/inclusivehar_v4.e3fc4e0f21cdd580b930a1a694923a43e222957335fde8eb08960e007f4930bb.json --profile inclusivehar-v4 --read-only --data-root data/raw --gate-record results/gates/raw_data_read_access_inclusivehar_v4.json --json
```

The reviewed JSON under `results/data_audit/` is locked separately; a fresh CLI run has a new audit-completion timestamp and therefore a different report hash.

## Acquisition evidence

Only the two version-4 files declared by the locked manifest were acquired from their official Mendeley file URLs:

| Artifact | Bytes | SHA-256 | Result |
|---|---:|---|---|
| Sensor CSV | 148,915,541 | `0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34` | Verified |
| Sensitive participant-description DOCX | 16,997 | `b8c033dd630412103b8dd463c8a171aa4727fc7222ba6b7fdc1e2487315e0dfb` | Verified |

Both artifacts were retrieved on 2026-08-23 UTC. The DOCX downloader returned successfully at `2026-08-23T20:36:32.633474Z`. No exact CSV completion time is claimed because its downloader return failed during post-publication staging-link cleanup.

The first transfer uncovered a Windows hard-link ordering defect after the CSV had passed size/hash verification and had been published. Read-only mode was applied before removing the staging hard link, so Windows refused the unlink. The canonical path and staging path are two names for the same physical file, not two provider artifacts. Both names remain preserved. The downloader now removes the staging name before applying read-only mode, and `tests/test_acquisition.py::test_read_only_publication_removes_staging_link_before_chmod` covers the corrected order.

## CSV findings

- 33 exact columns.
- 396,602 data rows; 396,603 physical records when the header is included. The released/article count of 396,603 equals that physical-record count. Counting the header is a likely explanation, not an author-confirmed fact.
- 20 released participant IDs: 10 with `disabled=0` and 10 with `disabled=1`.
- All six released labels occur for every participant.
- No missing fields, malformed-width rows, invalid numeric tokens, non-finite numeric values, inconsistent participant group indicators, or exact full-row duplicates.
- Four repeated six-channel inertial vectors occur; these are not full-row duplicates.
- The file is strict UTF-8 without a BOM or NUL bytes, comma-delimited, CRLF-terminated, and has a trailing newline.
- The actual target column is lowercase `label`. Both `label` and the legacy spelling `Label` are explicitly excluded from model inputs.
- GPS/location fields are present. Their values and summaries were not exported and they remain excluded from model inputs.

The six primary channels are `motionUserAccelerationX/Y/Z` and `motionRotationRateX/Y/Z`. Core Motion user acceleration is retained in g and rotation rate in rad/s; units are not encoded in the CSV. Axis names are present, but sensor handedness and a device-to-body coordinate transform are not documented in the file.

UCI `body_acc_x/y/z` and `body_gyro_x/y/z` are the closest conceptual pair. The angular-rate units align, but InclusiveHAR Core Motion user acceleration and UCI's derived body acceleration use different gravity-separation pipelines. The primary bridge retains both sources in g and rad·s⁻¹ without conversion. A separately declared SI sensitivity analysis must multiply acceleration consistently by exactly `9.80665 m/s²` per g. Raw accelerometer/total-acceleration inputs remain a separate sensitivity track.

Sparse-tail counts are retained as sensitivity flags, never deletion or clipping rules: rotation rate `|value| > 10` occurs X=1, Y=29, Z=48 times; user acceleration `|value| > 5` occurs X=46, Y=0, Z=11 times.

## Boundary and sampling audit

There are exactly 120 maximal contiguous `(UserID, label)` runs: one per participant/activity cell. These are deterministic released runs, not validated trials. There is no way to locate the reported three internal repetitions.

The declared 50 Hz rate cannot be measured from the release because no clock field exists. Consequently, monotonicity, rate variation, timestamp gaps, resets, and clock discontinuities are not auditable. Durations computed as `rows / 50` are estimates only.

The article's design implies 1,080,000 rows before trimming, or 1,044,000 if 50 samples were removed from each end of every repetition. The release contains 396,602 data rows. Eleven participant/activity cells contain fewer than 2,900 rows, the nominal size of one trimmed 60-second repetition. Robust within-activity count screening also flags subject 3 sitting and subject 11 ramp ascent/descent. These are anomaly candidates, not confirmed truncations.

Official version history does not recover the missing boundaries. Version 1 contains one 148,856,120-byte CSV (`20109a3e...f0f7cb7`) with the same 33-field boundary-free schema. Versions 2 and 3 use the exact v4 CSV bytes. Version 4 adds only the participant-description DOCX.

## Order confound

The file has two activity-block orders, each perfectly aligned with a participant group:

- IDs 1–10 (`disabled=0`): ramp descent, jogging, sitting, standing, ramp ascent, walking.
- IDs 11–20 (`disabled=1`): jogging, walking, sitting, standing, ramp ascent, ramp descent.

Global row index, participant/file order, block position, sequential batch order, or any derivative must never be used as a feature or proxy. Participant-exclusive splitting remains necessary but is not sufficient to repair hidden trial boundaries.

## Sensitive metadata

The DOCX is a valid 11-member OOXML archive with 12 non-empty paragraphs and no tables. It contains 10 ordered cases whose aggregate/order patterns are consistent with the paper table's gender, height, weight, and device patterns, but it has no explicit numeric User/Subject crosswalk. The paper reports an assistive-device aggregate of one walker, one cane, two wheelchair users, and six participants with no listed device. This does **not** verify a row-to-participant join, and device use by activity remains unknown. No disability description or assistive-device free text was written to the audit report.

## Ontology consequence

All six native labels have participant coverage, but coverage does not make mappings semantically exact:

- Sitting is the current exact cross-source candidate.
- Standing is provisional because wheelchair-user realization is not explicitly documented.
- All-participant walking is a functional mobility class, not an exact UCI walking match, because released wheelchair `Walking` means manual propulsion.
- Ambulatory-only walking may be considered only as a separately documented sensitivity analysis.
- Ramp ascent/descent on an 8% incline are not UCI stair labels.
- Jogging must not be silently mapped to another locomotion label.

## Required resolution

Before split/window construction, obtain authoritative per-trial boundaries or approve an explicit protocol revision that no longer claims trial-safe windowing. The latter must preserve this deviation, justify residual leakage risk, and be locked before any model development. Until then, the data-audit gate remains `BLOCK`.
