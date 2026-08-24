# Descriptive source-target participant performance gaps

Status: **descriptive post-confirmatory analysis, not a matched confirmatory or causal comparison**. Positive gaps mean the one-seed source grouped-CV mean is higher than the five-seed target participant average.

| Model | Source mean | Target mean | Mean gap (95% independent-participant bootstrap CI) | Target worst | Source mean - target worst | Target lower decile | Source mean - target lower decile |
|---|---:|---:|---:|---:|---:|---:|---:|
| `compact-coral` | 0.7841 | 0.6808 | 0.1033 [-0.0658, 0.2792] | 0.2741 | 0.5100 | 0.3670 | 0.4171 |
| `compact-dann` | 0.8004 | 0.6808 | 0.1196 [-0.0371, 0.2831] | 0.2699 | 0.5305 | 0.3559 | 0.4446 |
| `compact-erm` | 0.7859 | 0.6777 | 0.1082 [-0.0626, 0.2840] | 0.2725 | 0.5134 | 0.3541 | 0.4318 |
| `deepconvlstm` | 0.5946 | 0.3966 | 0.1980 [0.1024, 0.3016] | 0.1631 | 0.4315 | 0.1771 | 0.4175 |
| `legacy-bilstm` | 0.6615 | 0.4339 | 0.2276 [0.1342, 0.3290] | 0.2018 | 0.4596 | 0.2055 | 0.4560 |
| `legacy-cnn1d` | 0.7553 | 0.4849 | 0.2704 [0.1310, 0.4167] | 0.1971 | 0.5582 | 0.2407 | 0.5146 |
| `legacy-joint-cnn-bilstm` | 0.7890 | 0.6410 | 0.1480 [-0.0037, 0.3081] | 0.2943 | 0.4947 | 0.3255 | 0.4635 |
| `logistic-regression` | 0.6266 | 0.6098 | 0.0169 [-0.1578, 0.2016] | 0.2152 | 0.4115 | 0.2602 | 0.3664 |
| `more-har-augmentation` | 0.7799 | 0.5634 | 0.2165 [0.0663, 0.3693] | 0.2563 | 0.5236 | 0.2861 | 0.4938 |
| `more-har-backbone` | 0.7425 | 0.6482 | 0.0942 [-0.0577, 0.2563] | 0.2630 | 0.4794 | 0.3026 | 0.4399 |
| `more-har-content` | 0.7906 | 0.6616 | 0.1289 [-0.0444, 0.3075] | 0.2566 | 0.5340 | 0.2809 | 0.5097 |
| `more-har-factorized` | 0.7346 | 0.5600 | 0.1746 [0.0257, 0.3306] | 0.2437 | 0.4909 | 0.2832 | 0.4514 |
| `more-har-full` | 0.7924 | 0.6353 | 0.1571 [-0.0117, 0.3311] | 0.2584 | 0.5340 | 0.2660 | 0.5264 |
| `more-har-full-no-accelerometer` | 0.6563 | 0.4456 | 0.2107 [0.0922, 0.3380] | 0.1928 | 0.4635 | 0.2070 | 0.4493 |
| `more-har-full-no-gyroscope` | 0.5838 | 0.4218 | 0.1619 [0.0336, 0.3018] | 0.1661 | 0.4177 | 0.2063 | 0.3775 |
| `more-har-groupdro` | 0.7847 | 0.6435 | 0.1412 [-0.0054, 0.2986] | 0.2674 | 0.5173 | 0.2868 | 0.4980 |
| `random-forest` | 0.7035 | 0.6175 | 0.0860 [-0.1071, 0.2868] | 0.1787 | 0.5248 | 0.2345 | 0.4690 |
| `static-dual-branch-matched` | 0.6507 | 0.4528 | 0.1979 [0.0880, 0.3179] | 0.1931 | 0.4576 | 0.2301 | 0.4206 |
| `svm-rbf` | 0.6776 | 0.5807 | 0.0969 [-0.0702, 0.2635] | 0.2769 | 0.4008 | 0.3010 | 0.3766 |
| `xgboost` | 0.6728 | 0.6288 | 0.0440 [-0.1363, 0.2318] | 0.2301 | 0.4428 | 0.2795 | 0.3934 |

The interval independently resamples the ten source and ten target participants 10,000 times (seed 1729). It describes participant-resampling uncertainty within these fixed records; it does not make the regimes comparable.

Limitations: source model/configuration selection is consumed; source values are from one-seed grouped cross-validation, while target values average five final-fit seeds; training and evaluation regimes differ; cohorts are unpaired; group labels support an ability-associated observational description, not a causal disability effect.
