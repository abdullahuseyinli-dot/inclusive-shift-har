# CCIL/BPD post-confirmatory extension runbook

Status: complete. The create-only CUDA run and reconstruction-based aggregation
finished on 2026-08-24 with eight source-selection fits and ten final fits, no
failed cells, and no new target opening or unlock. This extension was designed
after target opening 1 had been consumed. Every target result is therefore
descriptive post-confirmatory evidence and is ineligible for the locked primary
claim.

## Scientific identity

The extension fits two qualified adaptations on the unchanged functional-core
`[N,128,6]` cache:

- `ccil-paper-loss-compact` is the published CCIL equations 2--6 concept-mean
  loss applied to the local `compact_residual_96`. No official CCIL code was
  available, and this is not a faithful reproduction of the paper's complete
  model or training pipeline.
- `bpd-boundary-safe-compact-adaptation` uses the exact feature-producing portion
  of `compact_residual_96`, then local activity/redundant projections,
  classification/confusion, reconstruction, and a Donsker--Varadhan MINE
  objective. It is a boundary-safe protocol adaptation, not an official-faithful
  BPD reproduction. No third-party source was copied.

The provenance audit for the paper-derived choices is in
`docs/baselines/THIRD_PARTY_BASELINE_AUDIT.md`. The audited official BPD repository
was `Jie-su/BPD` at `8b2338927c118d1daa5c602d48b6ae5156dd5966`
(Apache-2.0). Its trainer is prohibited here because it can cross participant
boundaries during windowing and selects checkpoints using target performance.

Both adaptations are compared against two already-locked references without
rerunning either reference:

- `compact-erm`, the frozen compact residual ERM backbone;
- `more-har-full`, the frozen full MoRe-HAR model.

Source-reference reports are reconstructed from hash-validated frozen source
logits plus their frozen source-validation calibrators. Target-reference reports
are reconstructed from the consumed opening-1 prediction arrays. The target
arrays are not used for fitting, candidate selection, normalization, calibration,
checkpoint selection, or refitting.

## Locked design

`configs/experiments/ccil_bpd_postconfirmatory_v1.yaml` fixes:

- source training participants `1,2,3,4,5,6,7,9` (582 windows);
- source validation participants `8,10` (143 windows);
- descriptive target participants `11`--`20` (807 windows);
- three classes: mobility, sitting, standing;
- six primary user-acceleration/rotation-rate channels, 128 samples, stride 128;
- five final seeds: `11,23,47,89,131`;
- CUDA-only sequential neural execution and float16 mixed precision;
- 23 fixed epochs, matching the frozen compact-ERM training schedule;
- four source-only candidates for each adaptation, selected with seed 11 by mean
  source-participant macro-F1, worst participant, then candidate ID;
- source-train-only normalization and per-final-seed source-validation temperature
  calibration; and
- a Holm family of four target comparisons: each of the two adaptations versus
  compact ERM and versus MoRe-HAR full. Paired inference uses participant metrics
  averaged across the five seeds before model comparison.

The released InclusiveHAR files lack trial identifiers and timestamps. The cache
is participant-exclusive and released-block-boundary-safe, but it is not
trial-safe. The hidden-join unconditional risk bound remains 1.0. These results
must not be described as trial-safe, confirmatory, causal disability effects,
fairness evidence, or clinical validation.

## Evidence gates before a real run

Run from a clean committed worktree. The runner rejects an uncommitted tree and
requires the exact externally supplied Git commit and primary-cache record file
hash. Do not create a new target unlock or opening.

```powershell
# Run these commands from the repository root.

.\.venv\Scripts\python.exe -m pytest -q tests/test_paper_adaptation_extension.py tests/test_ccil_paper_adaptation.py tests/test_postconfirmatory_cache.py
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\ruff.exe format --check src tests
.\.venv\Scripts\mypy.exe --strict src tests

nvidia-smi
.\.venv\Scripts\python.exe -c "import torch; assert torch.cuda.is_available(); x=torch.ones(1,device='cuda'); assert (x+1).item()==2; print(torch.cuda.get_device_name(0), torch.version.cuda)"
```

The primary cache must already have been created by the opening-1
post-confirmatory cache procedure. It is ignored raw-derived evidence, not a Git
artifact. Pin its bytes immediately before execution:

```powershell
$cacheRecord = "results/postconfirmatory/cache/primary_channels_opening1_v1.json"
$cacheRecordSha = (Get-FileHash -Algorithm SHA256 -LiteralPath $cacheRecord).Hash.ToLowerInvariant()
$codeCommit = (git rev-parse HEAD).Trim().ToLowerInvariant()
$createdAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

if ((git status --porcelain=v1 --untracked-files=all)) { throw "Worktree must be clean" }
```

## CUDA-only execution

The command below performs, sequentially, eight candidate fits, ten final
adapter fits, source calibration/evaluation, and then one descriptive evaluation
of each final adapter checkpoint against the consumed opening-1 target cache.
There is no CPU fallback.

```powershell
.\.venv\Scripts\python.exe -m inclusive_shift_har.experiments.ccil_bpd_postconfirmatory `
  --config configs/experiments/ccil_bpd_postconfirmatory_v1.yaml `
  --repository-root . `
  --primary-cache-record $cacheRecord `
  --expected-primary-cache-record-file-sha256 $cacheRecordSha `
  --expected-code-commit $codeCommit `
  --created-at-utc $createdAt `
  --device cuda
```

The configured output directory is
`results/postconfirmatory/ccil_bpd_v1`. It must not exist before the run. All
writes are create-only. If a run fails, its partial checkpoints, records, index,
and `failures/` record must remain intact. Do not delete or overwrite them. A
retry requires a newly versioned config/output directory and a documented link
to the failed attempt.

The source selection lock is written before final five-seed fitting. The source
stage lock is written only after all final adapter source results, calibrators,
and both frozen-comparator source reference matrices validate. Only then can the
target cache be loaded. A failure during target-cache loading is marked as target
access attempted; it is not silently treated as target-blind.

## Validation and aggregation

Use the complete run index produced by the preceding command. The aggregator is
also create-only and reconstructs every report from its prediction/calibration
lineage before calculating participant statistics.

```powershell
$runRoot = "results/postconfirmatory/ccil_bpd_v1"
$aggregateAt = [DateTime]::UtcNow.ToString("yyyy-MM-ddTHH:mm:ssZ")

.\.venv\Scripts\python.exe -m inclusive_shift_har.evaluation.paper_adaptation_reporting `
  --index "$runRoot/ccil_bpd_postconfirmatory_index.json" `
  --repository-root . `
  --destination "$runRoot/ccil_bpd_postconfirmatory_aggregate.json" `
  --csv-destination "$runRoot/ccil_bpd_postconfirmatory_summary.csv" `
  --markdown-destination "$runRoot/ccil_bpd_postconfirmatory_summary.md" `
  --created-at-utc $aggregateAt
```

The JSON report records exact input/config hashes, participant-clustered
bootstrap intervals, paired sign-flip and Wilcoxon tests, effect sizes, lower-tail
metrics, calibration, and Holm-adjusted p-values for the complete four-comparison
target family. Statistical tests do not restore confirmatory status.

## Completed results

The aggregate is
`results/postconfirmatory/ccil_bpd_v1/ccil_bpd_postconfirmatory_aggregate.json`.
Its canonical `record_sha256` is
`c7b27e2a6d5ddf94efcd2c2064cb84aecfc70dfe3d4f38539660c3479128c180`.
The corresponding CSV and Markdown exports are in the same directory.

Across the ten held-out target participants, the qualified CCIL adaptation
obtained mean participant macro-F1 0.6896 (participant-clustered 95% bootstrap
interval [0.5413, 0.8254]), worst-participant 0.2724, and lower-decile 0.3459.
Compact ERM obtained 0.6777, 0.2725, and 0.3541, respectively. Thus CCIL's mean
difference was +0.0119, but its worst and lower-decile results were slightly
lower. The family-adjusted paired permutation and Wilcoxon p-values were both
0.2109. CCIL therefore does not satisfy the predeclared requirement to improve
both mean and lower-tail performance.

The boundary-safe BPD adaptation obtained mean participant macro-F1 0.5710,
worst-participant 0.2471, and lower-decile 0.2751. It underperformed compact ERM
by 0.1067 mean macro-F1 and MoRe-HAR full by 0.0644. These qualified adaptations
do not change the locked primary conclusion and do not establish an advantage
for MoRe-HAR.

## Interpretation limits

Only two source-validation participants are available for both adaptation
selection and calibration. BPD's 128-sample local temporal interface and
optimization schedule differ from its published 168-sample setting. CCIL uses a
local compact backbone rather than the paper's complete pipeline. The comparison
also mixes frozen models whose original fixed training schedules differ (MoRe-HAR
full used its already-locked schedule); neither comparator is granted additional
tuning or retraining. Report negative outcomes, failed runs, numerical failures,
and missing evidence without suppression.
