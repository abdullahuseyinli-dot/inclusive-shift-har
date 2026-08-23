# Repository operating rules

- Treat `docs/LOCKED_PROTOCOL.md` and the hashed protocol records under `results/protocol/` as authoritative after the protocol gate. Do not silently change a locked ontology, split, endpoint, seed, calibration rule, or checkpoint rule.
- Keep the original coursework ZIP, `.audit/`, `data/raw/`, checkpoints, failed runs, quarantines, manifests, unlock records, and previous evidence immutable and outside destructive cleanup.
- Never present legacy, development, or repeatedly opened results as locked confirmatory evidence. Evidence status must travel with every result artifact.
- Partition participants before segmentation or windowing. Never cross participant, activity, trial/segment, or timestamp-discontinuity boundaries, and never overlap raw samples across partitions.
- Fit preprocessing and normalization on training data only. Tune and calibrate on validation data only. Do not inspect the locked target cohort for model selection.
- Do not expose disability status, assistive-device metadata, identity, timestamps, location/GPS, or labels as model features.
- Do not infer missing trials, labels, hardware, licenses, results, or acceptance outcomes. Quarantine ambiguity and record conditional/failing gates explicitly.
- Run tests, lint, type checks, manifest validation, split audit, configuration validation, and artifact validation before commits or tags.
- Do not commit third-party raw data, the source ZIP, secrets, large checkpoints, or generated caches. Repository code licensing does not relicense datasets or third-party implementations.
- Do not force-push, overwrite an existing remote, or delete preserved evidence.
