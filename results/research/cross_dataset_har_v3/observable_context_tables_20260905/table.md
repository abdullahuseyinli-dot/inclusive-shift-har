# External results with explicit comparison groups

Evidence status: validated_development. Seeds: 11, 23, 47. Independent participants: 22.

Primary: seed-averaged participant fixed-class macro-F1. Intervals resample participants, retaining their seeds. Probability ensembles are not the primary.

## independent fixed window; 6 input channels

Different inference/channel groups are not a matched before/after comparison.

| Method | Mean F1 [95% CI] | Worst | Bottom 30% | Present-class sensitivity | Evidence status |
|---|---:|---:|---:|---:|---|
| DeepConvLSTM-6ch | 0.462206 [0.417810, 0.504585] | 0.231883 | 0.341874 | 0.543832 | validated_development |
| RMRP-DG | 0.501904 [0.447632, 0.553352] | 0.232558 | 0.347194 | 0.592421 | validated_development |
| RandomForest-6ch | 0.570950 [0.501705, 0.642152] | 0.232558 | 0.387264 | 0.662092 | validated_development |
| TinyHAR-style-6ch | 0.506208 [0.455944, 0.558883] | 0.308192 | 0.370075 | 0.589410 | validated_development |
| XGBoost-6ch | 0.565904 [0.497946, 0.635977] | 0.232558 | 0.384283 | 0.656521 | validated_development |

## independent fixed window; 9 input channels

Different inference/channel groups are not a matched before/after comparison.

| Method | Mean F1 [95% CI] | Worst | Bottom 30% | Present-class sensitivity | Evidence status |
|---|---:|---:|---:|---:|---|
| CAGE-DG | 0.528529 [0.464022, 0.595640] | 0.232558 | 0.349445 | 0.619117 | validated_development |
| CTGR-DG | 0.544901 [0.477322, 0.614344] | 0.232558 | 0.349271 | 0.635823 | validated_development |
| CTGR-DG-top3-equal | 0.545811 [0.476013, 0.618519] | 0.232558 | 0.341156 | 0.635903 | validated_development |
| HERA-DG-strict | 0.558538 [0.487251, 0.631975] | 0.232558 | 0.347492 | 0.649638 | validated_development |
| HERA-DG-v2-core | 0.547496 [0.477848, 0.617616] | 0.232558 | 0.341156 | 0.637588 | validated_development |
| HERA-DG-v2-dual | 0.545502 [0.478187, 0.613299] | 0.232558 | 0.347928 | 0.635594 | validated_development |
| HERA-DG-v2-full | 0.550177 [0.479555, 0.621339] | 0.232558 | 0.341156 | 0.640269 | validated_development |
| HERA-DG-v2-weighted | 0.544514 [0.474773, 0.617581] | 0.232558 | 0.341156 | 0.634606 | validated_development |

## noncausal participant batch; 9 input channels

Different inference/channel groups are not a matched before/after comparison.

| Method | Mean F1 [95% CI] | Worst | Bottom 30% | Present-class sensitivity | Evidence status |
|---|---:|---:|---:|---:|---|
| HERA-DG-context-safe | 0.547863 [0.478566, 0.620593] | 0.232558 | 0.349271 | 0.638785 | validated_development |
| HERA-DG-full | 0.553161 [0.482615, 0.626840] | 0.232558 | 0.348494 | 0.643306 | validated_development |


Native-nine and derived-gravity datasets are never pooled. Within a dataset, 6ch/N9/DG suffixes identify the frozen representation comparison. Neural architecture names denote repository implementations, not verified reproductions of the original papers' training recipes.

Comparison contract SHA-256: `f7ff5ede0f5fe4e761320f4e79ac05057e342795b35bd34d2a68d052d71d03bd`.

Full seed reports, per-class metrics, confusion/calibration, participant distributions and paired comparisons are in the companion JSON.
