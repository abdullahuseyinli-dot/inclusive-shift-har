# Same-Attachment Reference Comparison V1 Protocol

Status: implementation qualification; prospective human-performance execution is blocked
until the physical pilot and companion diagnostic pass on verified real observations.

This protocol implements the controller specification at
`.audit/research_direction_20260913-001/NEXT_EXPERIMENT_SPEC.md`. It does not modify the
frozen physical-information pilot, reopen InclusiveHAR P11-P20, or authorize additional
seeds, datasets, placements, publication, or external-queue work.

The controller must call `validate_signal_run(..., write_receipt=False, require_final=True)`
on the completed physical package and independently verify the actual equipment,
canonicalization, human authorization/consent, processing freeze, canonical-recording and
annotation receipts. A numeric gate pass from a synthetic fixture, template, or old analysis
is insufficient. The original pilot gate and the separate pooled-reference gate must both
pass for all six fixed wearers. Historical claim fields remain unchanged.

The evaluation uses six leave-one-wearer-out folds, with all visits and attachments of one
wearer held out together. Training-only operations include all scalers, source heads,
concentration calibration and MAP priors. Four labelled held-out support bouts are allowed
calibration inputs; held-out query labels and movement strata cannot affect predictions,
preprocessing, support choice, fitting or thresholds.

The nine arms and their equations, exact weight formula, concentration optimizer, MAP
adaptation, coefficient compilation, metrics and ordered gates are normative in the governing
specification. `same_attachment_reference_comparison_v1.yaml` fixes 54 binary, six
multinomial and 12 scalar optimization objects, yielding 54 arm-fold outputs and zero encoder
fits. Identical components share serialized bytes. Every attempt consumes its assigned slot;
nonconvergence is retained as incomplete, with no automatic retry.

Known missing/stale/invalid/attachment-mismatched reference rows copy native-float64 Z
probabilities byte-for-byte. Globally invalid signal rows use the common frozen mask. Caller
metadata controls freshness; falsely fresh or mislabelled support is an explicit possible harm.

Every execution package must preserve input and source hashes, partition/support/query IDs,
fit start/completion/failure receipts, fitted transforms and coefficients, concentration
objective traces, predictions, participant values, all paired comparisons, ordered gates,
coverage and exclusions, runtimes, validation/replay, artifact manifest and shutdown receipt.
The six-wearer outcome is development evidence and cannot establish population superiority.
