# Post-confirmatory CUDA efficiency runbook

This secondary operation profiles the already-frozen neural inventory. It uses
deterministic all-zero synthetic inputs of shape `[B, 128, 6]`; it does not load
target signals, predictions, or metrics and cannot change model selection.

Run only on the recorded CUDA machine. Do not substitute CPU timing. The complete gate
is exactly 16 neural models × seeds `11, 23, 47, 89, 131` = 80 model-seeds, each at
batch sizes `1, 64` and precisions `float32, float16_autocast`, for 320 profiles.
The runner applies each checkpoint's frozen `disable_cudnn` setting and restores
the prior process setting after every profile.

The runner binds two different commits. `frozen_code_commit` remains the training
commit embedded in every immutable checkpoint. `profiler_code_commit` is the
current repository `HEAD` containing the profiling implementation; a supplied
commit that differs from `HEAD` is rejected before any consumed context is opened.

GPU contention is an executable gate, not a free-text operator assertion. The
selected-device compute-process list is sampled before the run and immediately
before and after every timed profile. The profiler process is allowed. On this
Windows machine, `dwm.exe` is the only declared ambient system process; when
`nvidia-smi` reports its protected name as `[Insufficient Permissions]`, the
runner must resolve that exact PID through Windows `Get-Process` before it can be
allowlisted. An unresolved protected process or any other compute process fails
the run and is preserved in failure lineage. The final create-only
`gpu_contention_attestation.json` contains all 641 self-hashed snapshots and a
machine-readable timing-validity classification. This is a sampled process gate,
not continuous utilization or thermal monitoring, and that limitation must remain
attached to reported timing.

From the repository root, supply a real UTC timestamp and run once:

```powershell
$profilerCommit = (git rev-parse --verify HEAD).Trim()
$createdAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")
.venv\Scripts\python.exe -m inclusive_shift_har.experiments.postconfirmatory_efficiency `
  --profile-config configs/experiments/neural_efficiency_profile_v1.yaml `
  --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
  --opening-receipt results/protocol/confirmatory_target_opening_1.json `
  --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --artifact-root . `
  --output-directory results/efficiency/postconfirmatory-v1 `
  --output-root . `
  --created-at-utc $createdAtUtc `
  --profiler-code-commit $profilerCommit
```

The command is create-only. A controlled exception returns nonzero and preserves
completed profile files plus `failure.json` and a failed index; never delete or
overwrite those records. Aggregate only an index with `status:
complete_create_only`, 80 checkpoint validations, all 320 profile cells, an exact
profiler commit, and a complete passing contention attestation:

```powershell
.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.secondary_aggregation efficiency `
  --index results/efficiency/postconfirmatory-v1/neural_efficiency_profile_index.json `
  --artifact-root . `
  --destination results/efficiency/postconfirmatory-v1/neural_efficiency_aggregate.json `
  --created-at-utc 2026-08-24T00:00:00Z `
  --aggregation-code-commit $profilerCommit
```

Reported analytical MACs/FLOPs cover only the declared `Conv1d`, `Linear`, and
`LSTM` operator subset. They are not full-graph operation counts.
