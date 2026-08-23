# Quarantined matched-baseline CV attempt

Status: **preserved provenance deviation; excluded from aggregation, model
selection, calibration, and confirmatory evaluation**.

The five `static_dual_branch_matched` source-CV jobs were launched while the
fixed-epoch infrastructure was being added. Folds 01--02 used the committed
configuration schema at `2b441daadc134385fd72903c71695d36d081fd79`; folds
03--05 imported an uncommitted schema that added
`checkpoint_selection_rule`, while their supplied `code_commit` field still
named the earlier commit. The model-selection behavior remained
`source_validation_best`, but the recorded commit is not an exact description
of the latter three executions. Combining the two configuration hashes would
also violate the requirement that every fold use an identical configuration.

The summaries and their ignored run/checkpoint artifacts are retained. A clean
five-fold rerun from one committed code state supersedes this attempt.

| Folds | Configuration SHA-256 | Exact committed lineage |
|---|---|---|
| 01--02 | `f5038d54183f07038e4e3e01d9787684099a52df990bd1d14258efff500a91d5` | Yes |
| 03--05 | `56e188fb9b966d5c85daed46ab664b4c5ba836f279c6059fe209fa714d892f6d` | No; uncommitted schema change |

This quarantine was detected before target unlock. No target subject, signal,
prediction, or metric was accessed.
