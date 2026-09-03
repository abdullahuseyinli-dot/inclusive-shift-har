# Prospective v1 implementation errata

Status: documented on 3 September 2026 for prospective code only. No locked v1
artifact, checkpoint, metric, manifest, or historical training log has been
rewritten.

## MoRe-HAR total-loss log

The v1 training engine first accumulated the tensor used for optimization under
`total`, then iterated over the MoRe-HAR component dictionary, which also
contained a `total` entry. Consequently, the displayed per-epoch total was
doubled for MoRe-HAR objective runs. Backpropagation used the total tensor once,
so model weights, predictions, and reported classification metrics are
unaffected. Component logs other than `total` are unaffected.

The v2 branch skips a component called `total` after recording the optimized
total and includes a regression test. Historical logs remain evidence of what
the old program wrote and must be interpreted with this erratum.

## Augmentation coordinate system

The v1 MoRe-HAR augmentation operated on already-standardized tensors. This is
retained as part of the locked v1 method, not defended as a physical transform.
The prospective v2 pipeline performs a common proper 3-D rotation and bounded
gain, bias, drift, noise, time-warp, and channel-loss transforms in native units
before applying training-partition-only normalization.

## Window boundary and stride

The active v1.2 protocol uses non-overlapping 128-sample windows within released
subject/activity blocks. The release lacks trial, session, timestamp, and raw
sample identifiers, so it remains a participant-exclusive released-block
protocol—not a trial-safe protocol. Model work cannot repair this limitation.

## Evidence boundary

The current target opening is permanently consumed. Any FuSE/ReFrame decision
must be made from source participants and eligible external proxy datasets.
Evaluating a frozen v2 model on the old target would be post-confirmatory and
cannot establish a new breakthrough claim.
