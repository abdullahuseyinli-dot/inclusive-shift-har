# Source aggregate v4 display-identity supersession

Status: **mathematically valid source-only development aggregate, superseded
before target unlock because one display identity was underspecified**.

Aggregate v4 validated all configuration hashes and participant-fold records,
but displayed the CORAL configuration as `compact_residual_96` rather than
including its positive CORAL coefficient in `model_variant_id`. The full
configuration and hash remained present, so metrics were not miscomputed or
merged. The three create-only exports are preserved here without alteration.

The identity derivation now emits `compact_residual_96_coral_0p01`; a new
aggregate is generated after tests and commit. No target subject, signal,
prediction, label, or performance was accessed.
