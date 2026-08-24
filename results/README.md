# Results evidence policy

Every result belongs to exactly one evidence class:

- `legacy_exploratory_development_consumed`
- `source_development_not_confirmatory`
- `development_target_descriptive`
- `locked_confirmatory_target_opening_1`
- `post_confirmatory_secondary`
- `failed`
- `quarantined`

The primary zero-shot target evaluation is complete and immutable under `confirmatory/zero_shot_v1/`. Its receipt, 100 JSON result sidecars, locked index, participant statistics, and publication tables are committed. The larger NPZ prediction arrays remain outside normal Git history but are preserved locally and cryptographically bound by the committed sidecars/index.

No target metric may be used to alter the frozen models, preprocessing, ontology, normalization, calibration, thresholds, seeds, or hypothesis decision. Later few-person inclusion and sensor-reliability studies are explicitly post-confirmatory secondary evidence.

A run is complete only when its validator can reconstruct dataset, split, preprocessing, ontology, configuration, code/environment identity, checkpoint selection, predictions/probabilities, participant identifiers, and metrics. Failures, OOMs, backend crashes, resets, deviations, and manual interventions remain visible.
