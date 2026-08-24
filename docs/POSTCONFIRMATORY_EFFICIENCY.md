# Post-confirmatory CUDA efficiency runbook

This secondary operation profiles the already-frozen neural inventory. It uses
deterministic all-zero synthetic inputs of shape `[B, 128, 6]`; it does not load
target signals, predictions, or metrics and cannot change model selection.

Run only on the recorded CUDA machine. Do not substitute CPU timing. The complete gate
is exactly 16 neural models × seeds `11, 23, 47, 89, 131` = 80 model-seeds, each at
batch sizes `1, 64` and precisions `float32, float16_autocast`, for 320 profiles.
The runner applies each checkpoint's frozen `disable_cudnn` setting and restores
the prior process setting after every profile.
It clears unused CUDA cache before every profile cell, records cell-local VRAM
measurement semantics, and requires every profile and contention snapshot to use
one shared CUDA device and software/hardware environment. Latency remains locked
to 20 warmups, 100 measured iterations, and p50/p95 reporting.

The runner binds two different commits. `frozen_code_commit` remains the training
commit embedded in every immutable checkpoint. `profiler_code_commit` is the
current repository `HEAD` containing the profiling implementation; a supplied
commit that differs from `HEAD` is rejected before any consumed context is opened.

GPU contention is an executable gate, not a free-text operator assertion. The
selected-device compute-process list is sampled before the run and immediately
before and after every timed profile. The profiler process is allowed. Read-only
preflight on this Windows WDDM machine observed `dwm.exe` and `explorer.exe` as
ambient graphics processes, so those two exact basenames are predeclared. When
`nvidia-smi` reports a protected name as `[Insufficient Permissions]`, the
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
$profileConfig = "configs/experiments/neural_efficiency_profile_v1.yaml"
$profileConfigFileSha = "179881e7771ee8e8ae2d6bdf43f68b2bc452c57b82dbf51296874b3274d83f31"
$observedProfileConfigFileSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $profileConfig).Hash.ToLowerInvariant()
if ($observedProfileConfigFileSha -ne $profileConfigFileSha) {
  throw "Efficiency profile configuration byte hash differs from the external pin"
}
.venv\Scripts\python.exe -m inclusive_shift_har.experiments.postconfirmatory_efficiency `
  --profile-config $profileConfig `
  --expected-profile-config-file-sha256 $profileConfigFileSha `
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
overwrite those records. The profile configuration must be a regular,
non-symlink file beneath `--artifact-root`; both its byte SHA-256 and normalized
semantic lock are recorded. Aggregate only an index with `status:
complete_create_only`, 80 checkpoint validations, all 320 profile cells, an exact
profiler commit, and a complete passing contention attestation:

```powershell
$aggregationCommit = (git rev-parse --verify HEAD).Trim()
$aggregationCreatedAtUtc = (Get-Date).ToUniversalTime().ToString("yyyy-MM-ddTHH:mm:ss.ffffffZ")
.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.secondary_aggregation efficiency `
  --index results/efficiency/postconfirmatory-v1/neural_efficiency_profile_index.json `
  --artifact-root . `
  --destination results/efficiency/postconfirmatory-v1/neural_efficiency_aggregate.json `
  --created-at-utc $aggregationCreatedAtUtc `
  --aggregation-code-commit $aggregationCommit
```

The aggregation commit is independently bound to its current `HEAD`; it may be
newer than the profiler commit and must not be copied from the profiling shell
variable without re-reading `HEAD`.

Reported analytical MACs/FLOPs cover only the declared `Conv1d`, `Linear`, and
`LSTM` operator subset. They are not full-graph operation counts.
