# SI acceleration-unit sensitivity runbook

Status: **executed and numerically equivalent under the training-only z-score**. The source and target normalized tensors were exactly equal in the stored float32 calculation (maximum and mean absolute differences `0.0`; predeclared tolerance `1e-6`). The underlying moment-scaling residuals were below `8.5e-15`. No model training, accuracy calculation, raw-file access, unlock call, or new target opening occurred. The create-only record SHA-256 is `ca42d11b9b846307afc1cd2dfeda9e113c81fbdcc8bd60f1499eb22f1fb7bcaa`.

This post-confirmatory preprocessing audit tests the numerical consequence of expressing InclusiveHAR user acceleration in `m/s^2` instead of `g` when the per-channel z-score is fit on the same `source_train` cache. It is not a model run, an accuracy result, or evidence of improved generalization.

The command accepts only the existing hash-pinned primary-channel cache index, the consumed opening-1 receipt/index, and the predeclared SI configuration. It exposes no raw-CSV, unlock, target-opening, model, or training argument. It reads `source_train` and `target_sealed` materialized caches, validates their hashes and ordered identities, fits both normalizers on `source_train` only, and writes one self-hashed JSON record as its final action. Target labels are loaded only because the shared cache validator checks exact opening-1 alignment; they are not used in the numerical calculation or model selection.

## Preconditions

- Create the three partition-pure primary-channel caches and retain their create-only index.
- Record the cache-index **file** SHA-256 outside that file. Do not derive the expected pin from a replacement file.
- Keep `results/protocol/confirmatory_target_opening_1.json` and `results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json` unchanged.
- Create an empty output root. Reusing the destination fails closed.

Fixed input pins:

- Split-manifest record SHA-256: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
- InclusiveHAR v4 sensor CSV SHA-256 carried in cache lineage: `0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34`
- SI sensitivity configuration file SHA-256: `20cd398b95553d0534250d46463c74657c57fb7b75a1fb0073d7952eb9b2eb1b`

## Exact command

The executed invocation used the create-only cache record `results/postconfirmatory/cache/primary_channels_opening1_v1.json` with externally pinned file SHA-256 `02a2190e90615a8b9ad4c934a3240aa00de62314de9f0be98d45916d756a00ba`. For a fresh versioned reproduction, use a new empty output directory and the following template:

```powershell
$primaryCacheRecord = "<PRIMARY_CACHE_RECORD_PATH>"
$primaryCacheRecordFileHash = "<EXTERNALLY_RECORDED_PRIMARY_CACHE_RECORD_FILE_SHA256>"
$timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
$outputRoot = "results/analysis/unit_sensitivity_v1"
New-Item -ItemType Directory -Path $outputRoot -ErrorAction Stop | Out-Null

.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.postconfirmatory_unit_sensitivity `
  --sensitivity-config configs/preprocessing/inclusivehar_primary_si_128.yaml `
  --expected-sensitivity-config-file-sha256 20cd398b95553d0534250d46463c74657c57fb7b75a1fb0073d7952eb9b2eb1b `
  --primary-cache-record $primaryCacheRecord `
  --expected-primary-cache-record-file-sha256 $primaryCacheRecordFileHash `
  --opening-receipt results/protocol/confirmatory_target_opening_1.json `
  --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --artifact-root . `
  --expected-split-manifest-sha256 ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b `
  --expected-source-artifact-sha256 0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34 `
  --output acceleration_unit_sensitivity.json `
  --output-root $outputRoot `
  --created-at-utc $timestamp
```

The expected scientific outcome is numerical equivalence up to the predeclared `1e-6` tolerance: multiplying the first three channels by the positive constant `9.80665` before fitting the same training-only z-score cancels algebraically. A tolerance failure must remain visible and must not be relabeled as equivalence.

## Focused validation

```powershell
.venv\Scripts\python.exe -m pytest -q tests/test_unit_conversion.py tests/test_unit_sensitivity.py tests/test_postconfirmatory_cache.py tests/test_postconfirmatory_unit_sensitivity.py
.venv\Scripts\python.exe -m ruff check src/inclusive_shift_har/preprocessing/units.py src/inclusive_shift_har/evaluation/unit_sensitivity.py src/inclusive_shift_har/evaluation/postconfirmatory_unit_sensitivity.py tests/test_unit_conversion.py tests/test_unit_sensitivity.py tests/test_postconfirmatory_unit_sensitivity.py
.venv\Scripts\python.exe -m ruff format --check src/inclusive_shift_har/preprocessing/units.py src/inclusive_shift_har/evaluation/unit_sensitivity.py src/inclusive_shift_har/evaluation/postconfirmatory_unit_sensitivity.py tests/test_unit_conversion.py tests/test_unit_sensitivity.py tests/test_postconfirmatory_unit_sensitivity.py
.venv\Scripts\python.exe -m mypy src/inclusive_shift_har/preprocessing/units.py src/inclusive_shift_har/evaluation/unit_sensitivity.py src/inclusive_shift_har/evaluation/postconfirmatory_unit_sensitivity.py tests/test_unit_conversion.py tests/test_unit_sensitivity.py tests/test_postconfirmatory_unit_sensitivity.py
```

The synthetic tests do not access the real InclusiveHAR dataset or invoke GPU/CPU training.
