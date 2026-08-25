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

No target metric may be used to alter the frozen models, preprocessing, ontology, normalization, calibration, thresholds, seeds, or hypothesis decision. Later few-person inclusion, disabled within-group, sensor-reliability, efficiency, signal-sensitivity, and qualified predecessor-adaptation studies are explicitly post-confirmatory secondary evidence.

## Current evidence inventory

| Evidence | Status | Canonical record |
|---|---|---|
| Locked zero-shot opening 1 | Complete and immutable | `confirmatory/zero_shot_v1/publication_report_v1.json` |
| Corrected UCI official-train grouped reproduction | Complete source development; official test unopened | `legacy_reproduction/uci_har_source_grouped_v1/uci_source_grouped_report_v1.json` |
| Disabled-cohort within-group | Complete, 75/75 CUDA cells | `analysis/within_group_v1_1/summary.json` |
| Few-person inclusion v1.1 | Complete, 1,200/1,200 CUDA cells | `postconfirmatory/few_person_v1_1/statistics_attempt_004/few_person_v1_1_statistics.json` |
| Sensor-reliability stress | Complete, 120 stressed cells | `postconfirmatory/sensor_stress_v1/sensor_stress_aggregate.json` |
| Qualified CCIL/BPD adaptations | Complete, descriptive and non-faithful | `postconfirmatory/ccil_bpd_v1/ccil_bpd_postconfirmatory_aggregate.json` |
| Raw/total acceleration sensitivity v1.1 | Complete; failed v1 preserved | `postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_aggregate.json` |
| SI-unit equivalence | Complete preprocessing-equivalence record, not an accuracy experiment | `analysis/unit_sensitivity_v1/acceleration_unit_sensitivity.json` |
| CUDA efficiency | Complete, 80 checkpoints/320 profiles/641 contention samples | `efficiency/postconfirmatory-v1-attempt-002/neural_efficiency_aggregate_attempt_002.json` |

Failed few-person aggregation attempts 1-3, the first within-group aggregation,
the first efficiency profile attempt, and the first efficiency aggregation
remain preserved beside the successful evidence. A successful later attempt
does not erase or relabel those failures. The successful few-person aggregate's
empty `preserved_failures` array means only that no scenario cell failed; it
does not describe or supersede the separate aggregation/profile failure chains.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

A run is complete only when its validator can reconstruct dataset, split, preprocessing, ontology, configuration, code/environment identity, checkpoint selection, predictions/probabilities, participant identifiers, and metrics. Failures, OOMs, backend crashes, resets, deviations, and manual interventions remain visible.
