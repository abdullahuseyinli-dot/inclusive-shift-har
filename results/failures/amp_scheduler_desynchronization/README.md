# Mixed-precision scheduler desynchronization quarantine

Status: **quarantined development evidence; excluded from aggregation, model
selection, calibration, freezing, and confirmatory evaluation**.

During the first full-model accelerometer-removal CUDA run, PyTorch reported
that the learning-rate scheduler advanced before an optimizer update. The
training loop called the scheduler once per epoch even when `GradScaler`
skipped every optimizer step in an epoch after detecting non-finite float16
gradients. This makes the schedule one epoch ahead of the actual optimizer
state and is therefore not accepted for final evidence.

The completed source-development summary was moved here without deletion. Its
checkpoint and prediction files remain preserved in the ignored run directory.
The next grouped-fold process was interrupted immediately after the warning;
its create-only run directory remains preserved even though no summary was
published. No target subject, signal, label, prediction, or performance was
accessed.

The corrected engine records optimizer updates and skipped AMP steps for every
epoch and advances the scheduler only after at least one optimizer update. All
affected ablations are rerun under a new commit and new immutable run IDs.

A checkpoint audit then compared each CUDA-float16 AdamW step counter with
`completed epochs × batches per epoch` for 121 existing source summaries. It
identified 17 summaries with exactly one skipped optimizer update: five
compact-residual grouped folds, five BiLSTM grouped folds, one GroupDRO fold,
four compact-residual tuning trials, and two BiLSTM tuning trials. To prevent
configuration/fold collisions during corrected reruns, the four unaffected
companion GroupDRO folds were quarantined with the affected fold. In total, 15
grouped-fold summaries and six source-development summaries were moved into
the subdirectories here. The older aggregate files remain immutable historical
snapshots and are superseded by a new aggregate after the reruns.
