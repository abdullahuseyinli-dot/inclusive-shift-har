# Results evidence policy

No corrected-reproduction, development, or locked-confirmatory results exist yet.

Every future run directory must declare exactly one evidence status:

- `legacy_exploratory_development_consumed`
- `development_source_only`
- `development_target_descriptive`
- `locked_confirmatory`
- `failed`
- `quarantined`

A run is incomplete unless its artifact validator can reconstruct the dataset, split, preprocessing, ontology, configuration, code/environment identity, checkpoint-selection rule, predictions, probabilities, participant identifiers, and metrics. Failures, OOMs, resets, deviations, and manual interventions remain visible.
