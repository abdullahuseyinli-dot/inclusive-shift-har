# Raw/total-acceleration sensitivity runbook

Status: implemented but **not executed**. This is a post-confirmatory exploratory signal-definition sensitivity. It is not a new confirmatory opening, an exact UCI body-acceleration match, trial-safe evidence, or a causal disability analysis.

The track retrains only the compact ERM backbone and full MoRe-HAR. These architectures were selected independently of observed performance ranking. Each uses the five locked seeds and its unchanged fixed-epoch source budget. Per-channel normalization is fit only on `source_train`; scalar temperature calibration is fit only on `source_validation`. The source stage must complete and publish `source_stage_lock.json` before the consumed opening-1 target context is loaded. No target tuning, calibration, threshold selection, or refit is permitted.

## Preconditions

- Commit the new configuration, runner, aggregator, tests, and this runbook.
- Require a clean worktree and record `git rev-parse HEAD`.
- Confirm the ignored official v4 CSV is present at `data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv` with SHA-256 `0209542286e9031dc64a65abb47e1d84676088b1a77e52f738635887cb9bce34`.
- Use the CUDA-enabled locked environment. The runner checks CUDA before opening its configuration, raw data, or target context.
- Do not remove partial outputs or failure records. A failed output directory is preserved and cannot be reused.

The immutable CSV is SHA-256 checked as a whole before the source-stage lock, so the integrity pass necessarily reads every file byte. The lock's narrower guarantee is that target rows are not parsed or materialized and target context, signals, labels, metrics, calibration, and tuning are not accessed until the source checkpoints and calibrators are durably locked.

## Exact execution commands

Run from the repository root in PowerShell after the prerequisites pass:

```powershell
$executionCommit = git rev-parse HEAD
$executionTimestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")

.venv\Scripts\python.exe -m inclusive_shift_har.experiments.raw_total_acceleration `
  --config configs/experiments/raw_total_acceleration_sensitivity_v1.yaml `
  --repository-root . `
  --raw-csv "data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv" `
  --code-commit $executionCommit `
  --created-at-utc $executionTimestamp
```

Only if `results/postconfirmatory/raw_total_acceleration_v1/raw_total_acceleration_index.json` has `status: complete_create_only`, aggregate it once:

```powershell
$aggregationTimestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")

.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.raw_total_reporting `
  --index results/postconfirmatory/raw_total_acceleration_v1/raw_total_acceleration_index.json `
  --artifact-root . `
  --destination results/postconfirmatory/raw_total_acceleration_v1/raw_total_acceleration_aggregate.json `
  --csv-destination results/postconfirmatory/raw_total_acceleration_v1/raw_total_acceleration_summary.csv `
  --markdown-destination results/postconfirmatory/raw_total_acceleration_v1/raw_total_acceleration_summary.md `
  --created-at-utc $aggregationTimestamp
```

Both commands are create-only. Reusing any output destination fails closed.

## Pre-execution verification commands

```powershell
.venv\Scripts\python.exe -m ruff check src/inclusive_shift_har/experiments/raw_total_acceleration.py src/inclusive_shift_har/evaluation/raw_total_reporting.py tests/test_raw_total_acceleration.py
.venv\Scripts\python.exe -m ruff format --check src/inclusive_shift_har/experiments/raw_total_acceleration.py src/inclusive_shift_har/evaluation/raw_total_reporting.py tests/test_raw_total_acceleration.py
.venv\Scripts\python.exe -m mypy src/inclusive_shift_har/experiments/raw_total_acceleration.py src/inclusive_shift_har/evaluation/raw_total_reporting.py tests/test_raw_total_acceleration.py
.venv\Scripts\python.exe -m pytest -q tests/test_raw_total_acceleration.py
```

The synthetic tests do not open the InclusiveHAR raw file, use the GPU, or invoke a target unlock.
