# Locked target evaluation runner

`inclusive_shift_har.evaluation.locked_target.run_locked_target_evaluation` is the
programmatic confirmatory-evaluation boundary. It validates the exact final-freeze
inventory, internal checkpoint/configuration/normalization/calibrator lineage, the
unlock record, and all create-only output destinations before target materialization.

The runner then writes exactly one canonical receipt named
`confirmatory_target_opening_1.json`. Only after that receipt exists can the supplied
materializer callback execute. Neural inference is CUDA-only; a CPU request fails
before materialization. Frozen classical checkpoints use their serialized estimator,
training-partition channel standardizer, and source-validation temperature calibrator.

Each model/seed produces a create-only NPZ plus a self-hashed JSON sidecar tagged
`locked_confirmatory_target_opening_1`. The NPZ preserves ordered window IDs,
participant IDs, labels, logits, calibrated and uncalibrated probabilities, and
predictions. The JSON stores hashes for every ordered array and a participant-level
metric report. A create-only index is published only after every frozen entry finishes.

The module intentionally has no raw-data path or automatic retry. Operational code
must inject the already-reviewed materializer and retain the receipt and any partial
artifacts if a post-opening failure occurs. CLI wiring remains gated until the final
freeze and unlock records exist; this prevents a convenience command from becoming a
second opening path.
