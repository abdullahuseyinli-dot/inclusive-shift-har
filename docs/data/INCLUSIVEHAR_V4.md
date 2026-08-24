# InclusiveHAR v4 data card

## Identity and provenance

- Dataset: InclusiveHAR, version 4.
- DOI: [10.17632/r78dn3f6nc.4](https://doi.org/10.17632/r78dn3f6nc.4).
- Official repository: [Mendeley Data version 4](https://data.mendeley.com/datasets/r78dn3f6nc/4).
- Related article: [Data in Brief, DOI 10.1016/j.dib.2026.112620](https://doi.org/10.1016/j.dib.2026.112620).
- Release date: 2026-02-18.
- Licence shown by the versioned repository: CC BY 4.0.
- Repository policy: do not redistribute raw files through Git; the project licence does not relicense dataset material.

The source manifest and hashes are in `manifests/datasets/inclusivehar_v4.json`. Raw files are immutable and excluded from Git.

## Intended use in this project

InclusiveHAR is the target dataset for descriptive and, only after all protocol gates pass, participant-exclusive ability-associated population-shift evaluation. Disability status, assistive-device information, participant identity, GPS/location, labels, timestamps, and row/order proxies are never inference features.

This dataset does not establish fairness, causal disability effects, clinical validity, or generalization to all disabilities or devices.

## Participants and activities

The release contains 20 participant IDs. The CSV's binary indicator divides them into 10 `disabled=0` and 10 `disabled=1` participants. Every participant has rows for all six exact released labels:

- `Ramp ascent`
- `Ramp descent`
- `Sitting`
- `Standing`
- `Walking`
- `jogging`

Group and disability labels are descriptive released metadata, not causal attributes and not model inputs. Assistive-device and disability-type subgroup analyses would be exploratory because counts are very small.

## Sensing interface

The release has 30 numeric feature columns plus lowercase `label`, `UserID`, and `disabled`. It reports collection from a vertically worn iPhone 14 Pro in a waist pouch at 50 Hz.

The primary benchmark tensor is restricted to six fields:

1. `motionUserAccelerationX`
2. `motionUserAccelerationY`
3. `motionUserAccelerationZ`
4. `motionRotationRateX`
5. `motionRotationRateY`
6. `motionRotationRateZ`

Core Motion user acceleration is retained in g and rotation rate in rad/s. The CSV contains no unit metadata. X/Y/Z names exist, but no explicit coordinate handedness or device-to-body transform is encoded.

UCI `body_acc_x/y/z` and `body_gyro_x/y/z` are the closest conceptual bridge. InclusiveHAR Core Motion user acceleration and UCI body acceleration differ in their gravity-separation pipelines, so conceptual alignment is not algorithmic equivalence. The primary track performs no conversion when both sources remain in g and rad·s⁻¹. A separate SI sensitivity track multiplies acceleration consistently by exactly `9.80665 m/s²` per g. Raw accelerometer or total-acceleration inputs remain a separately declared sensitivity analysis and must not be silently mixed with the primary interface.

## Data quality

Observed release facts:

- 396,602 data rows and one header record.
- Exact 33-column header order.
- No missing, malformed-width, non-finite, or invalid numeric values.
- No exact full-row duplicates.
- Four duplicated primary six-channel vectors.
- Strict UTF-8, no BOM or NUL bytes, comma dialect, CRLF line endings, and a trailing newline.
- A single consistent binary group value per participant.
- All six labels available for all 20 participants.

The primary audit is machine-readable at `results/data_audit/inclusivehar_v4.audit.json`.

Sparse tails are descriptive sensitivity flags, not cleaning rules: gyro `|value| > 10` counts are X=1, Y=29, Z=48; user-acceleration `|value| > 5` counts are X=46, Y=0, Z=11.

## Privacy handling

Six location-prefixed columns exist, but their coordinate values and descriptive statistics are deliberately absent from public artifacts. They are excluded from model inputs and ordinary derived datasets.

The supplementary DOCX contains sensitive free text. Audit code retained only privacy-safe structure and aggregate observations; no descriptive text was exported. Its 10 ordered cases are aggregate/order-consistent with the paper table's gender, height, weight, and assistive-device patterns, but there is no explicit User/Subject crosswalk. The paper's aggregate is one walker, one cane, two wheelchair users, and six participants with no listed device. This does not verify a CSV join, and device use for each activity is unknown.

## Known limitations and quarantine

The released CSV has no timestamp, trial/session identifier, recording identifier, or raw-sample index. It therefore cannot support direct checks of sampling-rate variation, clock monotonicity, gaps, resets, or the reported three repetition boundaries. All 120 participant/activity cells appear once as contiguous blocks, but those blocks are not validated trials.

The two observed activity orders are perfectly group-aligned. File order, row position, block position, participant order, and deterministic sequential batching are prohibited as model features or proxies.

Observed volume is inconsistent with the reported `20 × 6 × 3 × 60 s × 50 Hz` design. Candidate short or anomalous recordings remain visible in the machine audit and must not be silently removed.

Versions 1–4 provide no boundary-bearing schema. The original Stage 3 quarantine is
preserved. A later, explicit user-authorized deviation permitted a narrower
participant-exclusive released-block protocol after documenting a 100% unconditional
hidden-join risk bound. That conditional protocol does not resolve or erase the
missing-boundary limitation and must not be described as trial-safe.

## Label-use guidance

- Inclusive-native: all six labels are covered, subject to the unresolved boundary gate.
- Functional core: sitting, standing, and mobility/walking may be studied as functional labels, with realization differences explicit.
- Cross-source exact: sitting is currently defensible.
- Cross-source provisional: standing requires clarification for wheelchair-user realization.
- Ambulatory-only extension: walking may be mapped only in a separately documented sensitivity track.
- Never equate wheelchair propulsion with ordinary gait, ramp with stairs, or jogging with another locomotion class.

## Citation and ethical scope

Users must cite the versioned dataset and related article, comply with CC BY 4.0 attribution, and review the source's participant-consent and ethical statements. Avoid stigmatizing group comparisons, causal language, or claims that the small target cohort represents disability broadly.
