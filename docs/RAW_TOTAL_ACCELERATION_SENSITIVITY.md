# Raw/total-acceleration sensitivity runbook

Status: **v1 failed and is preserved; the implementation-corrected v1.1 rerun completed on CUDA with 10/10 source and 10/10 target results and no failures**. The v1 source stage completed all ten CUDA retrains, but target materialization failed because an opening-1 prediction path stored relative to the `results` artifact namespace was incorrectly resolved relative to the repository root. No v1 target inference result was produced. The v1.1 runner uses the already hash-validated prediction path from the immutable opening index; every scientific setting is unchanged. This remains a post-confirmatory exploratory signal-definition sensitivity, not a new confirmatory opening, an exact UCI body-acceleration match, trial-safe evidence, or a causal disability analysis.

## Observed result

Using released raw/total acceleration reduced compact-ERM target mean participant macro-F1 from 0.6777 to 0.6166 (paired participant delta -0.0611; 95% bootstrap interval [-0.1500, 0.0179]), worst-participant macro-F1 from 0.2725 to 0.1684, and lower-decile macro-F1 from 0.3541 to 0.1711. For full MoRe-HAR, target mean changed from 0.6353 to 0.6308 (delta -0.0045; interval [-0.1171, 0.1027]), while worst-participant performance fell from 0.2584 to 0.1714 and the lower decile from 0.2660 to 0.2010. These exploratory results do not show a robust benefit from silently substituting raw/total acceleration for the primary user/body-acceleration interface and reinforce keeping the signal definitions separate.

The create-only v1.1 index record is `a27cb39fae30dc2041e4e1169cd5a266cd4e3482682e7897c1fb03a893b8e7c7`; the participant aggregate record is `817405fd04b78a7a30ac985d84ad5009800edb78e48a69c8d246c0e4003a815a`.

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
  --config configs/experiments/raw_total_acceleration_sensitivity_v1_1.yaml `
  --repository-root . `
  --raw-csv "data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv" `
  --code-commit $executionCommit `
  --created-at-utc $executionTimestamp
```

Only if `results/postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_index.json` has `status: complete_create_only`, aggregate it once:

```powershell
$aggregationTimestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")

.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.raw_total_reporting `
  --index results/postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_index.json `
  --artifact-root . `
  --destination results/postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_aggregate.json `
  --csv-destination results/postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_summary.csv `
  --markdown-destination results/postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_summary.md `
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
