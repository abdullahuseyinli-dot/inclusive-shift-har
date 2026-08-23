# Source-only grouped cross-validation summary

Status: **source development; not confirmatory**. No target participant record, prediction, or performance value was read. Rows are sorted by configuration ID and do not constitute a model selection or freeze.

| Configuration | Mean participant macro-F1 (95% cluster bootstrap CI) | Worst | Lower decile | Mean participant balanced accuracy | NLL | Brier | Best epoch median |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compact_residual_96--8e544da70e79` | 0.7859 [0.6788, 0.8908] | 0.5555 | 0.5695 | 0.8116 | 0.5812 | 0.3284 | 23 |
| `legacy_cnn1d_h128--7d5af8ec36f4` | 0.7310 [0.6456, 0.8244] | 0.5367 | 0.5831 | 0.7545 | 0.5337 | 0.3365 | 6 |
| `logistic_regression--274ba21c48ba` | 0.6266 [0.5372, 0.7123] | 0.3708 | 0.4209 | 0.6514 | 0.9080 | 0.5105 | n/a |
| `more_har_backbone--b0a97d9f714e` | 0.7425 [0.6628, 0.8194] | 0.5200 | 0.5853 | 0.7581 | 0.6163 | 0.3776 | 25 |
| `more_har_full--fa82e7d61587` | 0.7924 [0.6898, 0.8874] | 0.5468 | 0.5602 | 0.8096 | 0.4700 | 0.2906 | 27 |
| `random_forest--92126a61caee` | 0.7035 [0.6125, 0.7978] | 0.4928 | 0.5278 | 0.7389 | 0.4733 | 0.3150 | n/a |
| `static_dual_branch_matched--56e188fb9b96` | 0.6507 [0.5836, 0.7263] | 0.4966 | 0.5489 | 0.7020 | 0.6339 | 0.4046 | 18 |
| `svm_rbf--78919913d9ed` | 0.6776 [0.5677, 0.7829] | 0.3521 | 0.5253 | 0.6937 | 0.6988 | 0.4306 | n/a |
| `xgboost--e3184e63de22` | 0.6728 [0.5829, 0.7705] | 0.4830 | 0.5283 | 0.7200 | 0.7801 | 0.4215 | n/a |

NLL and Brier are sample-weighted across disjoint grouped folds. ECE remains a secondary descriptive fold aggregate because ECE is not additive. Uncertainty resamples participants, never overlapping windows.
