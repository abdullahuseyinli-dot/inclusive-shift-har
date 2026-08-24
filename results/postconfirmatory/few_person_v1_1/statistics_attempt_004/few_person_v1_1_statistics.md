# Few-person inclusion curve v1.1

Status: post-confirmatory secondary evidence. Participants, not windows, are the inferential units; participant macro-F1 is averaged across five seeds first.

| Model | k | Mean macro-F1 (95% CI) | Worst | Lower decile |
|---|---:|---:|---:|---:|
| compact-coral | 0 | 0.6808 [0.5377, 0.8153] | 0.2741 | 0.3670 |
| compact-coral | 1 | 0.6350 [0.5683, 0.7018] | 0.4502 | 0.5120 |
| compact-coral | 2 | 0.6915 [0.5836, 0.7823] | 0.3228 | 0.4601 |
| compact-coral | 4 | 0.7468 [0.6438, 0.8340] | 0.3945 | 0.5276 |
| compact-dann | 0 | 0.6808 [0.5391, 0.8093] | 0.2699 | 0.3559 |
| compact-dann | 1 | 0.6706 [0.5943, 0.7497] | 0.4886 | 0.5162 |
| compact-dann | 2 | 0.6821 [0.5744, 0.7716] | 0.3071 | 0.4664 |
| compact-dann | 4 | 0.7487 [0.6473, 0.8359] | 0.4018 | 0.5290 |
| compact-erm | 0 | 0.6777 [0.5373, 0.8065] | 0.2725 | 0.3541 |
| compact-erm | 1 | 0.6744 [0.5963, 0.7548] | 0.4802 | 0.5099 |
| compact-erm | 2 | 0.6808 [0.5697, 0.7753] | 0.3098 | 0.4547 |
| compact-erm | 4 | 0.7363 [0.6398, 0.8203] | 0.4065 | 0.5298 |
| deepconvlstm | 0 | 0.3966 [0.3037, 0.4786] | 0.1631 | 0.1771 |
| deepconvlstm | 1 | 0.4078 [0.3548, 0.4519] | 0.2136 | 0.3319 |
| deepconvlstm | 2 | 0.4306 [0.3740, 0.4756] | 0.2212 | 0.3593 |
| deepconvlstm | 4 | 0.4748 [0.4288, 0.5168] | 0.3404 | 0.3847 |
| legacy-bilstm | 0 | 0.4339 [0.3409, 0.5161] | 0.2018 | 0.2055 |
| legacy-bilstm | 1 | 0.4784 [0.4238, 0.5267] | 0.3174 | 0.3504 |
| legacy-bilstm | 2 | 0.4772 [0.3947, 0.5509] | 0.2633 | 0.2646 |
| legacy-bilstm | 4 | 0.5232 [0.4557, 0.5874] | 0.3249 | 0.3887 |
| legacy-cnn1d | 0 | 0.4849 [0.3733, 0.5842] | 0.1971 | 0.2407 |
| legacy-cnn1d | 1 | 0.2320 [0.1773, 0.3025] | 0.1491 | 0.1561 |
| legacy-cnn1d | 2 | 0.3009 [0.2275, 0.3816] | 0.1682 | 0.1822 |
| legacy-cnn1d | 4 | 0.4968 [0.3700, 0.6137] | 0.1971 | 0.2114 |
| legacy-joint-cnn-bilstm | 0 | 0.6410 [0.5091, 0.7633] | 0.2943 | 0.3255 |
| legacy-joint-cnn-bilstm | 1 | 0.6365 [0.5524, 0.7129] | 0.3836 | 0.4484 |
| legacy-joint-cnn-bilstm | 2 | 0.6445 [0.5539, 0.7192] | 0.3224 | 0.4801 |
| legacy-joint-cnn-bilstm | 4 | 0.7217 [0.6265, 0.8023] | 0.3756 | 0.5986 |
| more-har-augmentation | 0 | 0.5634 [0.4445, 0.6708] | 0.2563 | 0.2861 |
| more-har-augmentation | 1 | 0.4974 [0.4375, 0.5481] | 0.3160 | 0.3375 |
| more-har-augmentation | 2 | 0.6599 [0.5552, 0.7487] | 0.2940 | 0.4706 |
| more-har-augmentation | 4 | 0.7509 [0.6502, 0.8376] | 0.4107 | 0.5803 |
| more-har-backbone | 0 | 0.6482 [0.5069, 0.7774] | 0.2630 | 0.3026 |
| more-har-backbone | 1 | 0.5922 [0.5284, 0.6442] | 0.4079 | 0.4171 |
| more-har-backbone | 2 | 0.7060 [0.5878, 0.8061] | 0.2970 | 0.4836 |
| more-har-backbone | 4 | 0.7428 [0.6513, 0.8254] | 0.4417 | 0.5860 |
| more-har-content | 0 | 0.6616 [0.5116, 0.7947] | 0.2566 | 0.2809 |
| more-har-content | 1 | 0.6590 [0.5864, 0.7212] | 0.4402 | 0.4822 |
| more-har-content | 2 | 0.6860 [0.5847, 0.7674] | 0.3271 | 0.4866 |
| more-har-content | 4 | 0.7593 [0.6593, 0.8466] | 0.4203 | 0.5784 |
| more-har-factorized | 0 | 0.5600 [0.4408, 0.6667] | 0.2437 | 0.2832 |
| more-har-factorized | 1 | 0.5073 [0.4504, 0.5553] | 0.3279 | 0.3649 |
| more-har-factorized | 2 | 0.6608 [0.5566, 0.7516] | 0.3096 | 0.4591 |
| more-har-factorized | 4 | 0.7513 [0.6502, 0.8374] | 0.4075 | 0.5670 |
| more-har-full | 0 | 0.6353 [0.4911, 0.7665] | 0.2584 | 0.2660 |
| more-har-full | 1 | 0.6161 [0.5545, 0.6686] | 0.4245 | 0.4718 |
| more-har-full | 2 | 0.6935 [0.5820, 0.7867] | 0.3130 | 0.4736 |
| more-har-full | 4 | 0.7514 [0.6579, 0.8339] | 0.4381 | 0.5914 |
| more-har-full-no-accelerometer | 0 | 0.4456 [0.3364, 0.5441] | 0.1928 | 0.2070 |
| more-har-full-no-accelerometer | 1 | 0.4390 [0.3661, 0.5022] | 0.2034 | 0.3258 |
| more-har-full-no-accelerometer | 2 | 0.4817 [0.4170, 0.5357] | 0.2823 | 0.3132 |
| more-har-full-no-accelerometer | 4 | 0.5355 [0.4503, 0.6180] | 0.3003 | 0.3353 |
| more-har-full-no-gyroscope | 0 | 0.4218 [0.3266, 0.5049] | 0.1661 | 0.2063 |
| more-har-full-no-gyroscope | 1 | 0.4547 [0.3656, 0.5353] | 0.2378 | 0.2425 |
| more-har-full-no-gyroscope | 2 | 0.4442 [0.3527, 0.5281] | 0.1723 | 0.2583 |
| more-har-full-no-gyroscope | 4 | 0.4611 [0.3656, 0.5471] | 0.2253 | 0.2450 |
| more-har-groupdro | 0 | 0.6435 [0.5009, 0.7721] | 0.2674 | 0.2868 |
| more-har-groupdro | 1 | 0.6062 [0.5409, 0.6582] | 0.4218 | 0.4261 |
| more-har-groupdro | 2 | 0.7016 [0.5846, 0.7986] | 0.2960 | 0.4834 |
| more-har-groupdro | 4 | 0.7435 [0.6472, 0.8284] | 0.4160 | 0.5807 |
| static-dual-branch-matched | 0 | 0.4528 [0.3560, 0.5387] | 0.1931 | 0.2301 |
| static-dual-branch-matched | 1 | 0.4787 [0.4070, 0.5457] | 0.2752 | 0.3300 |
| static-dual-branch-matched | 2 | 0.5700 [0.5074, 0.6213] | 0.3333 | 0.4857 |
| static-dual-branch-matched | 4 | 0.7252 [0.6119, 0.8247] | 0.3857 | 0.4971 |

Intervals are deterministic participant-cluster percentile bootstrap intervals. Paired sign-flip and Wilcoxon tests with global Holm correction are in JSON.
