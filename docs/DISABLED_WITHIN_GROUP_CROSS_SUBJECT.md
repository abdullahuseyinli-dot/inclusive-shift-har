# Disabled-cohort within-group cross-subject evaluation

## Scientific status

This track is **post-confirmatory descriptive evidence**. It does not reopen or
replace the consumed zero-shot target evaluation, and it must never be reported
as a locked confirmatory endpoint. It asks the narrower question: how well do
models trained on some released disabled-cohort participants transfer to other
disabled-cohort participants under subject-exclusive evaluation?

The five evaluation pairs are the pairs already declared before target opening
1 for the few-person outer folds. For each fold, the next pair cyclically is the
validation pair and the remaining six participants are training-only. Every
participant from 11 through 20 is evaluated exactly once and used for validation
exactly once. Participant assignment occurs before any shard array is parsed.

The track deliberately covers three representative, classification-only neural
architectures: compact ERM, DeepConvLSTM, and the MoRe-HAR compact backbone.
Their epochs, optimizer settings, batch size, and five seeds are copied exactly
from the source-locked final inventory; source checkpoints and weights are not
reused. This is a bounded descriptive comparison, not a new tuning sweep.

For every cell:

- normalization is fit on the six fold-training participants only;
- training uses a fixed last epoch and never loads validation or evaluation
  arrays;
- the checkpoint is written before validation shards are parsed;
- scalar temperature calibration is fit on the two validation participants;
- both checkpoint and calibrator are written and hash-verified before the two
  evaluation participant shards are parsed;
- neural execution requires CUDA, with no CPU fallback; DeepConvLSTM inherits
  its recorded `disable_cudnn: true` Windows safety setting;
- the raw CSV, whole-target cache array, and opening-1 prediction arrays have no
  runner interface and are not accessed.

Cache preflight hashes every participant-shard file for integrity but does not
parse its arrays. The runner then parses only the shards authorized for the
current role, in the order training, validation, evaluation.

## Immutable protocol

The checked-in configuration and generated metadata-only manifest are:

- `configs/protocols/inclusivehar_disabled_within_group_v1.yaml`
- `results/protocol/inclusivehar_disabled_within_group_v1.json`

The manifest is create-only and binds the released-block v1.2 split, consumed
opening-1 receipt/index, final source freeze, functional-core ontology, exact
fold assignments, source-locked training configurations, and all 75 expected
cells. Rebuilding it is a validation exercise only; do not overwrite it.

## CUDA execution

Run from a committed repository state after the participant-sharded primary
cache exists. Capture its physical file hash externally:

```powershell
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordSha = (Get-FileHash -Algorithm SHA256 $cacheRecord).Hash.ToLowerInvariant()
$executionCommit = (git rev-parse HEAD).Trim()
$resultRoot = "results/postconfirmatory/within_group_v1/cells"
New-Item -ItemType Directory -Force $resultRoot | Out-Null
$folds = @(
  "disabled_outer_01", "disabled_outer_02", "disabled_outer_03",
  "disabled_outer_04", "disabled_outer_05"
)
$models = @("compact-erm", "deepconvlstm", "more-har-backbone")
$seeds = @(11, 23, 47, 89, 131)

foreach ($fold in $folds) {
  foreach ($model in $models) {
    foreach ($seed in $seeds) {
      $createdAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
      uv run inclusive-shift-har train within-group `
        --manifest results/protocol/inclusivehar_disabled_within_group_v1.json `
        --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
        --opening-receipt results/protocol/confirmatory_target_opening_1.json `
        --locked-target-index results/confirmatory/zero_shot_v1/locked_target_evaluation_index.json `
        --final-freeze-inventory results/protocol/final_source_artifact_freeze_v1.json `
        --primary-cache-record $cacheRecord `
        --primary-cache-record-file-sha256 $cacheRecordSha `
        --artifact-root . `
        --output-root $resultRoot `
        --fold-id $fold --model-id $model --seed $seed `
        --code-commit $executionCommit --created-at-utc $createdAt
      if ($LASTEXITCODE -ne 0) {
        throw "Within-group cell failed: $fold / $model / $seed"
      }
    }
  }
}
```

Each cell runs in its own process and is create-only. A controlled failure
returns nonzero and retains `failure.json` and any partial artifacts. Do not
delete, overwrite, or reuse a failed cell directory.

After all 75 cells complete, aggregate once into a new directory:

```powershell
$aggregateAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
uv run inclusive-shift-har evaluate within-group-statistics `
  --manifest results/protocol/inclusivehar_disabled_within_group_v1.json `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --primary-cache-record $cacheRecord `
  --primary-cache-record-file-sha256 $cacheRecordSha `
  --result-root $resultRoot `
  --artifact-root . `
  --output-directory within_group_v1 `
  --output-root results/analysis `
  --code-commit $executionCommit --created-at-utc $aggregateAt
```

Aggregation fails closed on missing/failed/extra cell artifacts; modified
checkpoint, calibrator, cache, or prediction hashes; participant overlap;
configuration drift; probability/logit inconsistency; or a report that does not
recompute. It reports participant-level endpoints, participant-clustered and
hierarchical bootstrap intervals, and exact participant-level paired sign-flip
tests with Holm correction. Each participant's five seed values are averaged
before the reported mean, worst-participant, and lower-decile point endpoints.
Windows are never inferential units.

## Limitations

InclusiveHAR v4 does not provide recoverable trial/timestamp boundaries for this
released-block protocol, so this track remains participant-exclusive but not
trial-safe. The cohort is only ten participants, subgroup analyses are not
powered, and within-group performance cannot establish a causal disability
effect, fairness, clinical validity, or population-wide generalization.
