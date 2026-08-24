# Consolidated experiment runbook

This is an operator command index, not a result or completion claim. Consult
`docs/PROJECT_STATUS.md` and the machine records before every run. All neural
runs below are CUDA-only; classical scikit-learn estimators are the documented
CPU exception and XGBoost uses CUDA. Run one process at a time, preserve failed
outputs, and never reuse a create-only destination.

The one-time zero-shot target opening is consumed. **Do not invoke, reconstruct,
or rerun target opening 1.** The locked results under
`results/confirmatory/zero_shot_v1/` are inputs to post-confirmatory analysis,
not a license to produce new clean target predictions.

## Common preflight

```powershell
uv sync --locked --extra training-cuda --group research
nvidia-smi
uv run python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
uv run inclusive-shift-har validate-manifests --json
uv run inclusive-shift-har audit-splits `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --json
uv run inclusive-shift-har validate-artifacts --artifact-root results --require-artifacts --json
uv run pytest -q
uv run ruff check src tests
uv run ruff format --check src tests
uv run mypy src tests
git status --short
$executionCommit = git rev-parse HEAD
```

Do not proceed from a dirty tree unless every unrelated change is documented in
the new run record. The historical Stage 9 gate applied to the pre-opening tree;
it does not validate later edits automatically.

## Corrected UCI official-train reproduction

Status recorded in `results/legacy_reproduction/uci_har_v1.status.json`:
`configured_not_run`. The matrix is exactly 3 models × 5 grouped folds × 5
seeds = 75 CUDA runs. It never opens the official UCI test or InclusiveHAR
target. Use the inner verified archive described in
`docs/DATA_ACQUISITION_RUNBOOK.md`.

The runner consumes and byte-pins the v1.1 YAML. Model scope, folds, seeds,
hyperparameters, mixed precision, and the model-specific recurrent cuDNN policy
are derived from that file; the CLI has no independent hyperparameter override.
It also rejects a supplied code commit that differs from the executing Git HEAD.

```powershell
$archive = "data/raw/uci_har/v1/UCI HAR Dataset.zip"
$root = "results/legacy_reproduction/uci_har_source_grouped_v1"
$experimentConfig = "configs/experiments/uci_har_corrected_reproduction_v1_1.yaml"
$experimentConfigSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $experimentConfig).Hash.ToLowerInvariant()
$folds = 1..5 | ForEach-Object { "uci_source_cv_{0:d2}" -f $_ }
$seeds = @(42, 1337, 2025, 31415, 271828)
$models = @("legacy_cnn1d_h128", "legacy_bilstm_h192", "legacy_joint_bilstm256_cnn128")
foreach ($model in $models) {
  foreach ($seed in $seeds) {
    foreach ($fold in $folds) {
      $stem = "$model--seed-$seed--$fold"
      $arguments = @(
        "run", "inclusive-shift-har", "train", "uci-source-fold",
        "--archive", $archive,
        "--dataset-manifest", "manifests/datasets/uci_har_v1.json",
        "--protocol", "results/protocol/uci_har_source_grouped_v1.json",
        "--model", $model, "--fold-id", $fold, "--seed", "$seed",
        "--code-commit", $executionCommit,
        "--repository-root", ".",
        "--experiment-config", $experimentConfig,
        "--expected-experiment-config-file-sha256", $experimentConfigSha,
        "--run-directory", "$root/runs/$stem",
        "--summary", "$root/records/$stem.json",
        "--allowed-output-root", "results"
      )
      & uv @arguments
      if ($LASTEXITCODE -ne 0) { throw "UCI run failed: $stem" }
    }
  }
}
```

Only after all 75 records and their linked arrays/checkpoints validate:

```powershell
uv run inclusive-shift-har evaluate uci-source `
  --record-directory results/legacy_reproduction/uci_har_source_grouped_v1/records `
  --protocol results/protocol/uci_har_source_grouped_v1.json `
  --experiment-config $experimentConfig `
  --expected-experiment-config-file-sha256 $experimentConfigSha `
  --output-directory results/legacy_reproduction/uci_har_source_grouped_v1 `
  --bootstrap-resamples 10000 `
  --bootstrap-seed 1729 `
  --json
```

## Hash-pinned post-confirmatory primary cache

Create this once if the SI, few-person, or stress operations require it. The
command validates the already-consumed opening receipt/index and has no unlock
argument.

```powershell
$cacheRoot = "data/cache"
$cacheDirectory = "primary_channels_opening1_v1"
$recordRoot = "results/postconfirmatory/cache"
$recordOutput = "primary_channels_opening1_v1.json"
if (Test-Path -LiteralPath "$cacheRoot/$cacheDirectory") {
  throw "Refusing to reuse $cacheRoot/$cacheDirectory"
}
if (Test-Path -LiteralPath "$recordRoot/$recordOutput") {
  throw "Refusing to overwrite $recordRoot/$recordOutput"
}
foreach ($container in @($cacheRoot, $recordRoot)) {
  if (-not (Test-Path -LiteralPath $container -PathType Container)) {
    New-Item -ItemType Directory -Path $container -ErrorAction Stop | Out-Null
  }
}
$timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")
uv run python -m inclusive_shift_har.experiments.postconfirmatory_cache `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --opening-receipt results/protocol/confirmatory_target_opening_1.json `
  --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --raw-csv "data/raw/inclusivehar/v4/InclusiveHAR_dataset_v2 (1).csv" `
  --artifact-root . `
  --cache-directory $cacheDirectory `
  --cache-root $cacheRoot `
  --record-output $recordOutput `
  --record-root $recordRoot `
  --code-commit $executionCommit `
  --created-at-utc $timestamp
Get-FileHash -Algorithm SHA256 -LiteralPath "$recordRoot/$recordOutput"
```

Record that final **file hash externally**; do not obtain an expected pin from a
replacement record later.

## Few-person inclusion curve v1.1

The v1.1 manifest is ready but no scenario result is claimed here. It contains
15 fold/k scenarios and 80 frozen neural model-seeds, hence 1,200 sequential
CUDA runs. The v1 manifest is superseded and must not be regenerated or used.
Substitute the externally recorded cache-record file hash below.

```powershell
$manifestPath = "results/protocol/few_person_inclusion_curve_v1_1.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordFileHash = "<EXTERNALLY_RECORDED_PRIMARY_CACHE_RECORD_FILE_SHA256>"
$outputRoot = "results/postconfirmatory/few_person_v1_1"
$executionCommit = (git rev-parse HEAD).Trim()
foreach ($scenario in $manifest.scenarios) {
  foreach ($model in $manifest.models) {
    $output = "$outputRoot/$($scenario.scenario_id)/$($model.model_id)/seed-$($model.seed)"
    $createdAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
    uv run inclusive-shift-har train few-person run-scenario `
      --manifest $manifestPath `
      --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
      --opening-receipt results/protocol/confirmatory_target_opening_1.json `
      --zero-shot-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
      --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
      --primary-cache-record $cacheRecord `
      --primary-cache-record-sha256 $cacheRecordFileHash `
      --artifact-root . `
      --fold-id $scenario.fold_id --k $scenario.k `
      --model-id $model.model_id --seed $model.seed `
      --code-commit $executionCommit `
      --created-at-utc $createdAt `
      --output-directory $output --output-root .
    if ($LASTEXITCODE -ne 0) { throw "Few-person run failed: $output" }
  }
}
```

Validate progress at any time. Aggregate only when `aggregation_ready` is true:

```powershell
$statisticsCreatedAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")

uv run inclusive-shift-har evaluate few-person-statistics validate-progress `
  --manifest results/protocol/few_person_inclusion_curve_v1_1.json `
  --results-root results/postconfirmatory/few_person_v1_1 `
  --artifact-root . --full-json

uv run inclusive-shift-har evaluate few-person-statistics aggregate `
  --manifest results/protocol/few_person_inclusion_curve_v1_1.json `
  --results-root results/postconfirmatory/few_person_v1_1 `
  --artifact-root . `
  --output-directory results/postconfirmatory/few_person_v1_1/statistics `
  --zero-shot-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --created-at-utc $statisticsCreatedAt
```

Each run rejects a supplied commit that is not the checkout's actual `HEAD`.
Aggregation safely creates only the final `statistics` directory beneath the
artifact root; its parent must already exist, and all exports remain create-only.

## Disabled-cohort within-group cross-subject description

This is post-confirmatory descriptive evidence, not the locked zero-shot
endpoint. The immutable manifest fixes five subject-exclusive folds, three
representative neural models, and five seeds: 75 sequential CUDA cells. The
runner parses only participant shards, fits normalization on each fold's six
training participants, freezes the fixed-last-epoch checkpoint before loading
validation shards, and freezes validation-only calibration before loading the
two evaluation participants. It has no raw-data, unlock, or target-opening
route. See `docs/DISABLED_WITHIN_GROUP_CROSS_SUBJECT.md` for the full contract.

```powershell
$withinManifest = "results/protocol/inclusivehar_disabled_within_group_v1.json"
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordFileHash = "<EXTERNALLY_RECORDED_PRIMARY_CACHE_RECORD_FILE_SHA256>"
$withinRoot = "results/postconfirmatory/within_group_v1/cells"
if (-not (Test-Path -LiteralPath $withinRoot -PathType Container)) {
  New-Item -ItemType Directory -Path $withinRoot -ErrorAction Stop | Out-Null
}
$withinFolds = 1..5 | ForEach-Object { "disabled_outer_{0:d2}" -f $_ }
$withinModels = @("compact-erm", "deepconvlstm", "more-har-backbone")
$withinSeeds = @(11, 23, 47, 89, 131)
foreach ($fold in $withinFolds) {
  foreach ($model in $withinModels) {
    foreach ($seed in $withinSeeds) {
      $timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
      uv run inclusive-shift-har train within-group `
        --manifest $withinManifest `
        --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
        --opening-receipt results/protocol/confirmatory_target_opening_1.json `
        --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
        --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
        --primary-cache-record $cacheRecord `
        --primary-cache-record-file-sha256 $cacheRecordFileHash `
        --artifact-root . --output-root $withinRoot `
        --fold-id $fold --model-id $model --seed $seed `
        --code-commit $executionCommit --created-at-utc $timestamp
      if ($LASTEXITCODE -ne 0) {
        throw "Within-group run failed: $fold / $model / $seed"
      }
    }
  }
}
```

Aggregate only after every cell is complete and no `failure.json` exists:

```powershell
$timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
uv run inclusive-shift-har evaluate within-group-statistics `
  --manifest $withinManifest `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --primary-cache-record $cacheRecord `
  --primary-cache-record-file-sha256 $cacheRecordFileHash `
  --result-root $withinRoot `
  --artifact-root . `
  --output-directory within_group_v1 `
  --output-root results/analysis `
  --code-commit $executionCommit --created-at-utc $timestamp
```

## Secondary sensor-reliability stress

This track is configured but not claimed as executed. It uses the existing
clean target predictions; it must not rerun clean target inference or tune on
target performance.

```powershell
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordFileHash = "<EXTERNALLY_RECORDED_PRIMARY_CACHE_RECORD_FILE_SHA256>"
$timestamp = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
uv run python -m inclusive_shift_har.experiments.postconfirmatory_sensor_stress `
  --stress-config configs/experiments/sensor_reliability_stress_v1.yaml `
  --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
  --opening-receipt results/protocol/confirmatory_target_opening_1.json `
  --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --primary-cache-record $cacheRecord `
  --expected-primary-cache-record-file-sha256 $cacheRecordFileHash `
  --artifact-root . `
  --output-directory results/postconfirmatory/sensor_stress_v1 `
  --output-root . `
  --created-at-utc $timestamp

uv run inclusive-shift-har evaluate sensor-stress `
  --index results/postconfirmatory/sensor_stress_v1/sensor_reliability_stress_index.json `
  --artifact-root . `
  --destination results/postconfirmatory/sensor_stress_v1/sensor_stress_aggregate.json `
  --created-at-utc $timestamp
```

Aggregate only a complete create-only index. Preserve a failed index and all
partial condition records.

## Other isolated secondary operations

- CUDA efficiency: `docs/POSTCONFIRMATORY_EFFICIENCY.md` (80 neural
  model-seeds, 320 profile cells, zero-valued synthetic inputs, no target
  signals).
- SI acceleration-unit sensitivity: `docs/SI_UNIT_CONVERSION_SENSITIVITY.md`.
- Raw/total-acceleration retraining: `docs/RAW_TOTAL_ACCELERATION_SENSITIVITY.md`.

These are separate operations with separate create-only destinations. Their
absence is not permission to relabel them complete. Cross-source pretraining,
a BenchHAR SSL/foundation comparison, and a faithful CNN-HAR integration do not
currently have executable, evidence-validated run paths; see
`docs/baselines/BASELINE_COVERAGE_AND_OMISSIONS.md`.
