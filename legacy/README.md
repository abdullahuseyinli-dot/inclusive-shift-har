# Legacy evidence

This directory contains small, lawful, derived audit evidence from the user-provided coursework archive. Every result is classified as **legacy exploratory/development-consumed**. The official UCI-HAR test set was evaluated repeatedly and must not be represented as fresh confirmatory evidence.

Files:

- `archive_manifest.json` — verified archive checksum plus all 180 entry paths, sizes, represented timestamps, and uncompressed-file SHA-256 values.
- `verification_results.json` — machine-readable Stage 0 acceptance checks.
- `legacy_metrics.locked.json` — exact saved `metrics.json` payloads and their source hashes, with legacy status locked in.
- `metric_reconstruction.json` — independently reconstructed accuracy, macro-F1, confusion matrices, NLL, multiclass Brier score, 15-bin top-label ECE, per-class metrics, diagnostics, and artifact hashes.
- `metric_reconstruction.csv` — compact full-precision reconstruction table.
- `notebook_cell_index.json` — notebook hash and exact zero-based index / one-based ordinal / cell-id references used by the audit.

The narrative audit is in [`docs/LEGACY_AUDIT.md`](../docs/LEGACY_AUDIT.md).

## Preservation boundary

Not stored in normal Git history:

- the approximately 160 MB source ZIP;
- extracted archive contents;
- ten large `.pt` checkpoints;
- raw or third-party datasets.

The original ZIP was not modified. Extraction used a unique, create-new-only local audit directory under `.audit/`. That local directory is preservation evidence, not a cleanup target. It should be excluded from normal Git commits without deleting it.

## Interpretation boundary

These files validate internal consistency of saved outputs. They do not supply missing UCI-HAR data, participant identifiers, raw-window provenance, a subject-exclusive split, a corrected reproduction, or a confirmatory result. See the audit for the precise leakage, repeated-test, loss-comparability, model-naming, checkpoint, and reproducibility limitations.
