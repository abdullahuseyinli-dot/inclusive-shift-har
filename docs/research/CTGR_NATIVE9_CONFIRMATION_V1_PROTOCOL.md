# CTGR native-nine independent confirmation v1

Date frozen: 2026-09-19. Status: prospective execution protocol and executable
label-custody lane. This protocol does not turn any reused development or external
dataset into confirmation evidence.

## Question and frozen methods

The primary question is whether the finalized confidence-triggered gravity residual
(`T9`) improves participant macro-F1 over its same-input nine-channel control (`B9`)
on fresh people. `T9` versus `U9` holds the component models and blend weight fixed
and tests the confidence trigger. `T9` versus `B6` bundles the extra native-gravity
information with the method and therefore cannot isolate either one.

The exact finalized bytes are in
`.audit/ctgr_source_finalization/ctgr-source-finalization-seed11-20260908-001`:

| Object | SHA-256 |
|---|---|
| `predictor.pkl` | `623b7636222f0b52861709672395a7174b027ec09a3b63b50ab2e4969127ac3c` |
| final B6 | `8ab5b7cd9f7dbd957eb38c88c9a5999c0b7e1f5713cdd16157b842f7d89519e3` |
| final B9 | `8044e944562b57df1f10953783a58ddc12e44bb7040eb46df55a5feb5c155d8f` |
| final expert | `a34a82ee0310a32564089872c41ddf6f359188a666cdaa3001edada9e912cd28` |

HERA-v1, Random Forest, XGBoost and the closest qualified prior soft router are
secondary comparators. Their label-blind probabilities must be frozen from source-only
training under their own checkpoint receipts before the scoring archive is opened. The
probability archive requires a same-stem self-hashed JSON seal that binds every method's
contract/checkpoint hash, the archive hash, source-only training, and absence of target
label access. They cannot affect the primary gate or replace a failed primary contrast.

## Cohort and interface

Use one genuinely fresh, ability-relevant cohort with provider-recorded linear
acceleration, angular velocity and gravity at 50 Hz. Participant membership must be
fixed before windowing. Use continuous, non-overlapping 128-sample windows, never cross
participant/session/trial/discontinuity boundaries, and keep class order mobility,
sitting, standing. Units are m/s² for linear acceleration and gravity and rad/s for
angular velocity. No target normalization, fitting, calibration or selection is allowed.

At least **38 complete participants** are required, each with all three scored classes.
This number gives 80.87% two-sided paired-t power at alpha 0.05 for a 0.020 macro-F1
difference if the paired standard deviation is 0.042379. The standard deviation is from
ten reused development participants and is only a planning assumption. Sensitivity
requires 143, 65, 38, 25 and 24 complete participants for effects 0.010, 0.015, 0.020,
0.025 and the observed development mean 0.025864, respectively. Six people are therefore
not an adequate independent confirmation cohort.

InclusiveHAR P1-P10/P11-P20, MotionSense, AICOS-HAR, HARTH, IMU-HAR-IL and FoG-STAR
are consumed and excluded from this claim. A manifest must adjudicate cohort freshness,
ability relevance, provider-native gravity, pre-window participant partitioning and
interface qualification before prediction.

Provider-native gravity may be missing for at most 5% of windows. On each missing row,
all gravity-dependent outputs must equal the exact B6 probability bytes. Missingness is
reported overall and by participant. Exceeding 5%, or a participant with no native-gravity
window, makes the run incomplete.

## Sealed opening

The prediction controller receives only:

- `signals`: finite `[N,128,6]` arrays in the frozen channel order;
- `native_gravity`: `[N,128,3]` arrays;
- `native_gravity_available`: one Boolean per window;
- participant and unique window identifiers for later clustered reporting.

Any label-like array in the prediction archive is rejected. The controller writes one
create-only probability archive and a self-hashed prediction seal. A separate custodian
then supplies exactly `window_ids` and integer `labels`. Scoring checks the seal and byte
hash before reading labels. There is one opening and no post-opening tuning, refitting,
threshold change, participant replacement or alternative-winner promotion.

## Endpoints and gates

The primary endpoint is fixed-three-class macro-F1 for each participant with
`zero_division=0`, averaged equally across participants. Report each participant,
bottom 30%, worst participant, participant-mean accuracy, mobility/sitting/standing
recall, NLL, multiclass Brier score, pooled diagnostics, missingness and every harm.
Uncertainty is a deterministic 100,000-resample participant bootstrap with seed
20260919. Participants are the independent units.

Advance the independent CTGR claim only if every primary `T9 - B9` condition passes:

1. mean participant macro-F1 difference is at least +0.020;
2. the two-sided 95% paired-participant bootstrap lower endpoint is above zero;
3. bottom-30 difference is at least -0.010 and worst-score difference at least -0.030;
4. no paired participant difference is below -0.050;
5. mobility recall difference is at least -0.010 and sitting/standing each at least -0.020;
6. every leave-one-participant-out mean difference is positive;
7. native-gravity missingness and all provenance, alignment, seal and fallback checks pass.

`T9 - U9`, `T9 - B6`, HERA-v1, Random Forest, XGBoost and the prior soft router are
secondary/descriptive. No multiplicity-adjusted promotion claim is assigned to them.

## Commands

Preparation, label-blind prediction and scoring are separate create-only commands:

```powershell
python -m inclusive_shift_har.experiments.ctgr_native9_confirmation prepare `
  --config configs/experiments/ctgr_native9_confirmation_v1.yaml `
  --predictor-package .audit/ctgr_source_finalization/ctgr-source-finalization-seed11-20260908-001 `
  --cohort-directory data/qualified/ctgr_native9_confirmation_v1 `
  --output .audit/ctgr_native9_confirmation/<prepare-run-id>

python -m inclusive_shift_har.experiments.ctgr_native9_confirmation predict `
  --config configs/experiments/ctgr_native9_confirmation_v1.yaml `
  --predictor-package .audit/ctgr_source_finalization/ctgr-source-finalization-seed11-20260908-001 `
  --cohort-manifest <cohort>/cohort_manifest.json `
  --windows-archive <cohort>/windows_label_blind.npz `
  --secondary-predictions <cohort>/secondary_predictions_label_blind.npz `
  --output .audit/ctgr_native9_confirmation/<prediction-run-id>

python -m inclusive_shift_har.experiments.ctgr_native9_confirmation score `
  --config configs/experiments/ctgr_native9_confirmation_v1.yaml `
  --prediction-run .audit/ctgr_native9_confirmation/<prediction-run-id> `
  --labels-archive <custodian>/labels.npz `
  --output .audit/ctgr_native9_confirmation/<score-run-id>
```

If qualified fresh recordings do not exist, `prepare` writes an explicit `INCOMPLETE.json`
and stops without substituting a consumed dataset or creating a performance result.
