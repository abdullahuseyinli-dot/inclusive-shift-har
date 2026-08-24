# Post-confirmatory CUDA efficiency runbook

This secondary operation profiles the already-frozen neural inventory. It uses
deterministic all-zero synthetic inputs of shape `[B, 128, 6]`; it does not load
target signals, predictions, or metrics and cannot change model selection.

Run only on the recorded CUDA machine after `nvidia-smi` confirms that no other
process is competing for the GPU. Do not substitute CPU timing. The complete gate
is exactly 16 neural models × seeds `11, 23, 47, 89, 131` = 80 model-seeds, each at
batch sizes `1, 64` and precisions `float32, float16_autocast`, for 320 profiles.
The runner applies each checkpoint's frozen `disable_cudnn` setting and restores
the prior process setting after every profile.

From the repository root, supply a real UTC timestamp and run once:

```powershell
.venv\Scripts\python.exe -m inclusive_shift_har.experiments.postconfirmatory_efficiency `
  --profile-config configs/experiments/neural_efficiency_profile_v1.yaml `
  --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
  --opening-receipt results/protocol/confirmatory_target_opening_1.json `
  --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
  --artifact-root . `
  --output-directory results/efficiency/postconfirmatory-v1 `
  --output-root . `
  --created-at-utc 2026-08-24T00:00:00Z
```

The command is create-only. A controlled exception returns nonzero and preserves
completed profile files plus `failure.json` and a failed index; never delete or
overwrite those records. Aggregate only an index with `status:
complete_create_only`, 80 checkpoint validations, and all 320 profile cells:

```powershell
.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.secondary_aggregation efficiency `
  --index results/efficiency/postconfirmatory-v1/neural_efficiency_profile_index.json `
  --artifact-root . `
  --destination results/efficiency/postconfirmatory-v1/neural_efficiency_aggregate.json `
  --created-at-utc 2026-08-24T00:00:00Z
```

Reported analytical MACs/FLOPs cover only the declared `Conv1d`, `Linear`, and
`LSTM` operator subset. They are not full-graph operation counts.
