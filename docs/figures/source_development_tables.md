# Source-development tables

Seed 11; 725 windows; ten source participants; five participant-exclusive outer folds.
Fixed external controls and nested-selected project methods have different selection budgets.

| Method | Channels | Window accuracy | Participant macro-F1 |
|---|---:|---:|---:|
| HYDRA | 6 | 73.517% | 70.711% |
| MultiRocket | 6 | 76.138% | 73.501% |
| MultiRocket+HYDRA | 6 | 76.000% | 73.616% |
| QUANT | 6 | 76.966% | 74.120% |
| RIST (budgeted) | 6 | 78.759% | 77.195% |
| SpectralShape | 6 | 79.724% | 78.884% |
| GSP | 6 | 83.448% | 82.916% |
| RMRP (selected denoised GSP) | 6 | 84.414% | 83.790% |
| CTGR | 9 | 86.897% | 86.474% |
| Strict HERA-v1 | 9 | 87.034% | 86.755% |

Five-seed means; same source participants and windows; seeds 11, 23, 47, 89, 131.
CTGR/HERA add native gravity to the six-channel RMRP input.

| Method | Mean window accuracy | Mean participant macro-F1 |
|---|---:|---:|
| RMRP | 84.579% | 83.953% |
| CTGR | 86.979% | 86.540% |
| Strict HERA-v1 | 87.228% | 86.849% |
