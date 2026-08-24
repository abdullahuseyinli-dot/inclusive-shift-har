# Locked zero-shot target results

Status: **locked confirmatory target opening 1**. Participant-level macro-F1 is the primary endpoint; window metrics, calibration, and AURC are descriptive diagnostics.

| Model | Mean participant macro-F1 (95% participant bootstrap CI) | Worst | Lower decile | Balanced accuracy | NLL | Brier | ECE | AURC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `compact-dann` | 0.6808 [0.5391, 0.8093] | 0.2699 | 0.3559 | 0.7066 | 0.8813 | 0.4321 | 0.0624 | 0.1464 |
| `compact-coral` | 0.6808 [0.5377, 0.8153] | 0.2741 | 0.3670 | 0.7056 | 0.7850 | 0.4204 | 0.0611 | 0.1475 |
| `compact-erm` | 0.6777 [0.5373, 0.8065] | 0.2725 | 0.3541 | 0.7038 | 0.8655 | 0.4320 | 0.0556 | 0.1466 |
| `more-har-content` | 0.6616 [0.5116, 0.7947] | 0.2566 | 0.2809 | 0.6903 | 0.9553 | 0.4497 | 0.0727 | 0.1637 |
| `more-har-backbone` | 0.6482 [0.5069, 0.7774] | 0.2630 | 0.3026 | 0.6736 | 1.2869 | 0.5252 | 0.1288 | 0.2140 |
| `more-har-groupdro` | 0.6435 [0.5009, 0.7721] | 0.2674 | 0.2868 | 0.6701 | 1.2581 | 0.5185 | 0.1086 | 0.2136 |
| `legacy-joint-cnn-bilstm` | 0.6410 [0.5091, 0.7633] | 0.2943 | 0.3255 | 0.6625 | 1.1606 | 0.4980 | 0.0972 | 0.2218 |
| `more-har-full` | 0.6353 [0.4911, 0.7665] | 0.2584 | 0.2660 | 0.6643 | 1.2793 | 0.5050 | 0.0946 | 0.2021 |
| `xgboost` | 0.6288 [0.4654, 0.7808] | 0.2301 | 0.2795 | 0.6537 | 0.7930 | 0.4515 | 0.0921 | 0.1595 |
| `random-forest` | 0.6175 [0.4402, 0.7852] | 0.1787 | 0.2345 | 0.6429 | 0.8567 | 0.4568 | 0.1085 | 0.1512 |
| `logistic-regression` | 0.6098 [0.4456, 0.7572] | 0.2152 | 0.2602 | 0.6365 | 0.7832 | 0.4692 | 0.0648 | 0.1920 |
| `svm-rbf` | 0.5807 [0.4490, 0.7052] | 0.2769 | 0.3010 | 0.6076 | 0.9217 | 0.5421 | 0.0333 | 0.2643 |
| `more-har-augmentation` | 0.5634 [0.4445, 0.6708] | 0.2563 | 0.2861 | 0.6112 | 1.2442 | 0.5429 | 0.1256 | 0.2437 |
| `more-har-factorized` | 0.5600 [0.4408, 0.6667] | 0.2437 | 0.2832 | 0.6076 | 1.2883 | 0.5430 | 0.1125 | 0.2406 |
| `legacy-cnn1d` | 0.4849 [0.3733, 0.5842] | 0.1971 | 0.2407 | 0.5530 | 1.2164 | 0.5501 | 0.1131 | 0.2691 |
| `static-dual-branch-matched` | 0.4528 [0.3560, 0.5387] | 0.1931 | 0.2301 | 0.5292 | 1.2745 | 0.5513 | 0.1159 | 0.2648 |
| `more-har-full-no-accelerometer` | 0.4456 [0.3364, 0.5441] | 0.1928 | 0.2070 | 0.5162 | 1.2165 | 0.5754 | 0.0936 | 0.3237 |
| `legacy-bilstm` | 0.4339 [0.3409, 0.5161] | 0.2018 | 0.2055 | 0.4902 | 0.9324 | 0.5505 | 0.0831 | 0.3414 |
| `more-har-full-no-gyroscope` | 0.4218 [0.3266, 0.5049] | 0.1661 | 0.2063 | 0.5147 | 1.0104 | 0.5585 | 0.0901 | 0.2967 |
| `deepconvlstm` | 0.3966 [0.3037, 0.4786] | 0.1631 | 0.1771 | 0.5012 | 0.9707 | 0.5637 | 0.0526 | 0.3104 |

Metrics are averaged over the five predeclared seeds. The confidence interval resamples the ten target participants after seed averaging; windows are never treated as inferential replicates.
