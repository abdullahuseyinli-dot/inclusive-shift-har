# Post-confirmatory CCIL/BPD paper-adaptation summary

This is descriptive evidence designed after target opening 1; it is not confirmatory.

| Cohort | Model | Mean participant macro-F1 | Worst | Lower decile | 95% CI |
|---|---|---:|---:|---:|---:|
| source_validation | compact-erm | 0.5700 | 0.5660 | 0.5668 | [0.5660, 0.5739] |
| source_validation | more-har-full | 0.6209 | 0.6113 | 0.6132 | [0.6113, 0.6305] |
| source_validation | ccil-paper-loss-compact | 0.6017 | 0.5907 | 0.5929 | [0.5907, 0.6126] |
| source_validation | bpd-boundary-safe-compact-adaptation | 0.5980 | 0.5972 | 0.5974 | [0.5972, 0.5989] |
| target_sealed | compact-erm | 0.6777 | 0.2725 | 0.3541 | [0.5373, 0.8065] |
| target_sealed | more-har-full | 0.6353 | 0.2584 | 0.2660 | [0.4911, 0.7665] |
| target_sealed | ccil-paper-loss-compact | 0.6896 | 0.2724 | 0.3459 | [0.5413, 0.8254] |
| target_sealed | bpd-boundary-safe-compact-adaptation | 0.5710 | 0.2471 | 0.2751 | [0.4414, 0.6859] |

CCIL is a paper-derived loss adaptation, not official CCIL code. The BPD row is a boundary-safe local protocol adaptation, not an official-faithful BPD reproduction.
Both adaptations are compared with locked compact ERM and locked MoRe-HAR full. Neither comparator is rerun; target references are consumed opening-1 artifacts. Holm correction covers all four adapter-versus-comparator target comparisons.
