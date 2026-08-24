# Raw/total-acceleration signal-definition sensitivity

Status: **post-confirmatory exploratory analysis**. Both architectures were fixed independently of the observed ranking. This is not a new confirmatory opening, an exact UCI signal match, trial-safe evidence, or a causal disability analysis.

| Model | Raw source mean | Raw target mean | Primary target mean | Raw - primary mean delta (95% paired-participant bootstrap CI) | Raw target worst | Raw target lower decile |
|---|---:|---:|---:|---:|---:|---:|
| `compact-erm` | 0.5356 | 0.6166 | 0.6777 | -0.0611 [-0.1500, 0.0179] | 0.1684 | 0.1711 |
| `more-har-full` | 0.5430 | 0.6308 | 0.6353 | -0.0045 [-0.1171, 0.1027] | 0.1714 | 0.2010 |

Participants, not windows, are the statistical units. Participant metrics are averaged over the five fixed seeds before the paired target bootstrap (10,000 resamples; seed 1729). Source normalization and temperature calibration use only the source training and source validation partitions, respectively.

The raw/total accelerometer includes gravity and is not equivalent to UCI body acceleration. InclusiveHAR trial boundaries remain unrecoverable, so this track is not trial-safe.
