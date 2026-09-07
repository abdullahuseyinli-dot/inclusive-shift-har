# FoG raw-to-cache integrity audit v1

Status: frozen before source replay on 7 September 2026. The normative machine-readable
contract is `configs/experiments/fog_raw_cache_integrity_v1.yaml`.

This is a bounded, zero-fit measurement-integrity audit of the completed corrected FoG
Random Forest factorial. It does not estimate model performance, diagnose a participant,
alter annotations, select exclusions, tune preprocessing, or reopen another dataset.

## Inputs and independence

Use only the pinned FoG-STAR v3 `sensor_data.csv` and the immutable completed reference
run named in the configuration. Verify source size, MD5, and SHA-256 before parsing any
signals. Keep the raw object in memory and do not create a local mirror. Verify the
reference artifact hashes before using its cache or audit as comparison targets.

The auditor must discover people, sessions, physical runs, candidates, and scoring rows
from raw CSV rows. Cached identities and counts are comparison targets, never the raw
discovery universe. The implementation must not call `load_fog_star`, production gravity
or resampling wrappers, the production engineered-feature extractor, or any estimator.
Using SciPy's independently invoked `resample_poly` is disclosed and does not constitute
an independent validation of SciPy itself.

## Frozen reconstruction

Read the ten provider columns declared in the configuration. Freeze the sorted roster
from `subjectID` before activity admission. Group by participant and session while
preserving source row order. Convert acceleration from g to m/s2 by multiplying by
9.80665 and gyroscope degrees/s to radians/s by multiplying by NumPy pi and dividing by
180. Segment only at participant/session changes, non-finite timestamp or sensor rows,
non-increasing timestamps, or gaps greater than `3/60` in the released numeric values.
Do not segment on task or activity.

For every physical run of at least three rows, initialize causal gravity from its first
total-acceleration sample and update at native 60 Hz with
`alpha = 1-exp(-2*pi*0.30/60)`. Subtract it to obtain linear acceleration. Concatenate
linear acceleration, gyroscope, and derived gravity and invoke one 5/6 polyphase
resampling pass over the complete run. Construct the 50 Hz timestamp grid from the
physical start, retain values before the last source timestamp plus 1/60 second, and
form the fixed non-overlapping 128-sample candidate grid. Filtering and resampling must
never restart at a label, scored window, or annotation boundary.

Compare each segment's float64 source timestamp, output timestamp, signal, gravity, and
grid hashes. Slice every candidate and cast it to float32 before independently computing
the four frozen norm RMS/STD descriptors in float64. Compare all 1,939 candidates by
feature name at the frozen absolute and relative tolerances. Also compare complete
float32 candidate signal and gravity hashes.

Only after the complete candidate grid exists, project activity annotations. Admission
requires one finite supported original provider activity code throughout both projected
samples and the covered source interval. Codes 1, 6, and 7 map to mobility only after
this check; a 1-to-6 transition is therefore mixed and excluded. Compare exact candidate
IDs, exclusion indices and reasons, eligibility mask, scoring indices, scored labels,
participant IDs, and independently generated seed-11 folds.

## Evidence and verdicts

Report all raw-discovered sessions, including sessions with no valid segment or scored
window. Per-session evidence retains raw row, omission, run, sample, tail, candidate,
scored and exclusion counts; structural checks; segment hash checks; descriptor residuals;
and unresolved provenance.

`defect_detected` requires a concrete mismatch against the frozen reconstruction.
`no_defect_detected_under_frozen_contract` means every structural and numerical check
passed within the predeclared contract. A source/network failure or missing comparison
record is `unresolved`, never silently omitted. The implementation verdict is separate
from semantic qualification: the provider's documented milliseconds versus observed
approximately 1/60 timestamp increments remains unresolved, as do clinical interpretation
and annotation adjudication. Byte agreement cannot resolve those questions.

Create-only evidence must include input snapshots, preflight, source receipt, per-session
integrity, raw/cache comparison, P7 descriptor confirmation, result, runtime, validation,
and a complete hash manifest. Compute is capped at 30 minutes with one controller and
zero model fits. Stop after the audit; do not automatically run a seed, model, dataset,
target cohort, or publication task.
