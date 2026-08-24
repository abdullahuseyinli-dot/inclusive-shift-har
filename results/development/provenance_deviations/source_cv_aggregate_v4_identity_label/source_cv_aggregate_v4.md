# Source-only grouped cross-validation summary

Status: **source development; not confirmatory**. No target participant record, prediction, or performance value was read. Rows are sorted by configuration ID and do not constitute a model selection or freeze.

| Configuration | Mean participant macro-F1 (95% cluster bootstrap CI) | Worst | Lower decile | Mean participant balanced accuracy | NLL | Brier | Best epoch median |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compact_residual_96--2c62b52d708a` | 0.7859 [0.6788, 0.8908] | 0.5555 | 0.5695 | 0.8116 | 0.5812 | 0.3284 | 23 |
| `compact_residual_96--68db668cd3d1` | 0.7841 [0.6846, 0.8831] | 0.5555 | 0.5695 | 0.8120 | 0.5888 | 0.3230 | 23 |
| `deepconvlstm--9d742cc9d801` | 0.5567 [0.5252, 0.5971] | 0.4947 | 0.5078 | 0.6196 | 0.7733 | 0.4660 | 13 |
| `deepconvlstm--b34c04845d41` | 0.5946 [0.5446, 0.6435] | 0.4457 | 0.5144 | 0.6448 | 0.6401 | 0.3913 | 11 |
| `legacy_bilstm_h192--fb3a1b40113f` | 0.6615 [0.6189, 0.7079] | 0.5734 | 0.5752 | 0.6779 | 0.6715 | 0.4022 | 13 |
| `legacy_cnn1d_h128--7d5af8ec36f4` | 0.7310 [0.6456, 0.8244] | 0.5367 | 0.5831 | 0.7545 | 0.5337 | 0.3365 | 6 |
| `legacy_cnn1d_h128--8e57838b6dcd` | 0.7553 [0.6628, 0.8488] | 0.5380 | 0.5657 | 0.7904 | 0.5122 | 0.3234 | 5 |
| `legacy_joint_bilstm256_cnn128--049dc59665fd` | 0.7890 [0.7043, 0.8774] | 0.5626 | 0.6563 | 0.8040 | 0.4923 | 0.2940 | 34 |
| `logistic_regression--274ba21c48ba` | 0.6266 [0.5372, 0.7123] | 0.3708 | 0.4209 | 0.6514 | 0.9080 | 0.5105 | n/a |
| `more_har_augmentation--a51fbddd208d` | 0.7799 [0.6772, 0.8782] | 0.5362 | 0.5467 | 0.7984 | 0.5124 | 0.3207 | 18 |
| `more_har_augmentation_plus_content--a43106244ae1` | 0.7906 [0.6856, 0.8877] | 0.5189 | 0.5449 | 0.8093 | 0.4788 | 0.2936 | 34 |
| `more_har_augmentation_plus_content_plus_factorization--34ed274093cd` | 0.7346 [0.6335, 0.8408] | 0.5303 | 0.5594 | 0.7635 | 0.5282 | 0.3309 | 18 |
| `more_har_backbone--b0a97d9f714e` | 0.7425 [0.6628, 0.8194] | 0.5200 | 0.5853 | 0.7581 | 0.6163 | 0.3776 | 25 |
| `more_har_full--fa82e7d61587` | 0.7924 [0.6898, 0.8874] | 0.5468 | 0.5602 | 0.8096 | 0.4700 | 0.2906 | 27 |
| `more_har_full_zero_channels_0_1_2--6207c4fa1638` | 0.6563 [0.5954, 0.7297] | 0.5207 | 0.5751 | 0.6839 | 0.6019 | 0.3875 | 24 |
| `more_har_full_zero_channels_3_4_5--4cfb458c7d46` | 0.5838 [0.4905, 0.6916] | 0.3037 | 0.4773 | 0.6613 | 0.6902 | 0.4334 | 8 |
| `more_har_groupdro--26d4d21386e6` | 0.7847 [0.7164, 0.8500] | 0.5926 | 0.6305 | 0.7949 | 0.5483 | 0.3384 | 26 |
| `random_forest--92126a61caee` | 0.7035 [0.6125, 0.7978] | 0.4928 | 0.5278 | 0.7389 | 0.4733 | 0.3150 | n/a |
| `static_dual_branch_matched--56e188fb9b96` | 0.6507 [0.5836, 0.7263] | 0.4966 | 0.5489 | 0.7020 | 0.6339 | 0.4046 | 18 |
| `svm_rbf--78919913d9ed` | 0.6776 [0.5677, 0.7829] | 0.3521 | 0.5253 | 0.6937 | 0.6988 | 0.4306 | n/a |
| `xgboost--e3184e63de22` | 0.6728 [0.5829, 0.7705] | 0.4830 | 0.5283 | 0.7200 | 0.7801 | 0.4215 | n/a |

NLL and Brier are sample-weighted across disjoint grouped folds. ECE remains a secondary descriptive fold aggregate because ECE is not additive. Uncertainty resamples participants, never overlapping windows.
