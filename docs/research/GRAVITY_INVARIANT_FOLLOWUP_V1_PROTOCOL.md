# Gravity-Invariant Follow-up v1

Round A showed a small benefit from denoising (A2 over A1) but a large loss when the full 285-feature native-gravity block was appended (A3/A4). This bounded source-development test keeps the A2 matrix and Extra Trees contract frozen and appends ten predeclared scalar gravity-relative groups. It excludes coordinate-axis, absolute-component, and sorted-component gravity features. The purpose is mechanism diagnosis, not a new tuned baseline.

The five existing participant-pair folds, seed 11, participant-first weights, fixed three-class macro-F1, per-person diagnostics, and 10,000-resample participant bootstrap (seed 1729) are fixed before fitting. The cached A2 predictions are aligned by labels, participant IDs, and window IDs. Only five candidate fits are performed; no target records, new seeds, or external datasets are read. Outputs are create-only and carry input hashes and a manifest.

An improvement is useful only if it is reported with the complete participant table and uncertainty. A positive exploratory result does not authorize another seed or confirmation claim automatically.
