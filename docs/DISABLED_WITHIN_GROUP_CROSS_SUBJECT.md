# Disabled-cohort within-group cross-subject evaluation

## Scientific status

This track is **complete post-confirmatory descriptive evidence**. All 75 CUDA
cells validated and the successful participant-level aggregate is
`results/analysis/within_group_v1_1/summary.json`, record SHA-256
`fbe81492df167e136c3ad14bccfdd94a4c82c12497b48c2042a50e25dc521180`.
It does not reopen or
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

## Completed result

| Model | Mean participant macro-F1 | Participant-cluster 95% CI | Worst participant | Lower decile |
|---|---:|---:|---:|---:|
| Compact ERM | 0.606703 | [0.470351, 0.738584] | 0.253696 | 0.299998 |
| MoRe-HAR backbone | 0.562632 | [0.426293, 0.696010] | 0.243720 | 0.257990 |
| DeepConvLSTM | 0.400068 | [0.314808, 0.477823] | 0.190200 | 0.191635 |

Relative to compact ERM, MoRe-HAR backbone had participant-mean paired
difference -0.044071 with Holm-adjusted exact sign-flip p=0.025391;
DeepConvLSTM had difference -0.206635 with adjusted p=0.003906. These tests are
descriptive and do not restore confirmatory status. Within-group training did
not outperform the locked zero-shot point estimates, but the training regimes
differ, so that observation is not a controlled inclusion-effect estimate.

The first aggregation attempt is preserved at
`results/analysis/within_group_v1/failure.json` with status
`failed_preserved_create_only` and record SHA-256
`f1bcc774db221830bd3b88ae26cd5d8471e4acd3818ae641dd49b98ffd3fe33c`.
It failed closed on checkpoint configuration/schema reconstruction. The
implementation-corrected aggregate was written create-only to `within_group_v1_1`.

## CUDA execution

Run from a committed repository state after the participant-sharded primary
cache exists. Capture its physical file hash externally:

The runner resolves `git rev-parse --verify HEAD` itself and requires it to
equal the full object ID supplied through `--code-commit` before it loads the
within-group manifest, opening receipt, target index, freeze, or cache. A stale,
mistyped, or synthetic commit therefore fails closed instead of being copied
into otherwise valid-looking cell evidence.

```powershell
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordSha = "02a2190e90615a8b9ad4c934a3240aa00de62314de9f0be98d45916d756a00ba"
$observedCacheRecordSha = (Get-FileHash -Algorithm SHA256 $cacheRecord).Hash.ToLowerInvariant()
if ($observedCacheRecordSha -ne $cacheRecordSha) {
  throw "Primary cache record differs from the external pin"
}
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

The following command documents the successful aggregate. Do not rerun it into
an existing destination. The aggregator binds its own implementation to the
current Git HEAD and separately records the single execution commit shared by
all 75 immutable cells:

```powershell
$aggregateAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ss.fffffffZ")
uv run inclusive-shift-har evaluate within-group-statistics `
  --manifest results/protocol/inclusivehar_disabled_within_group_v1.json `
  --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json `
  --primary-cache-record $cacheRecord `
  --primary-cache-record-file-sha256 $cacheRecordSha `
  --result-root $resultRoot `
  --artifact-root . `
  --output-directory within_group_v1_1 `
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
