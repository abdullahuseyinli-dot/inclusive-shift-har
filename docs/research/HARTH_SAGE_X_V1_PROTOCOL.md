# HARTH SAGE-X v1 bounded protocol

This is an exploratory standing-focused development experiment. It does not replace
the locked InclusiveHAR/HERA evidence or establish independent confirmation.

## Question

Can a lower-back temporal representation recover standing information when paired thigh
signals teach the back encoder during outer-training only, while held-out inference uses
the back stream alone?

## Fixed contract

- Official HARTH, 22 participants, lower-back and right-thigh accelerometers at 50 Hz.
- Labels 7 and 6 map to `[sitting, standing]`.
- Five participant-exclusive folds use the retained seed-11 permutation.
- Windows are 250 samples, non-overlapping, and remain inside a physical and label run.
- All normalization, model fitting, masking and selection occur within the outer training
  participants. Held-out thigh data are never used by a back-only predictor.
- One CPU seed family is used. No target dataset, synthetic gyroscope, extra seed or
  hyperparameter search is opened by this run.

## Architecture

The back tower is a four-block dilated temporal convolutional encoder with a 64-dimensional
embedding and a binary sitting/standing head. The same-sensor arm reconstructs a masked
back sequence summary. The paired arm reconstructs the paired thigh summary and aligns
the back and thigh embeddings. The final arm adds a supervised standing-focused margin
loss using only training labels. The thigh tower is discarded before held-out inference.

The exact model constants are frozen in the source module: hidden width 32, embedding 64,
kernel 5, dilations 1/2/4/8, mask probability 0.15, AdamW learning rate 1e-3,
weight decay 1e-4, 18 epochs, batch size 256, and gradient norm cap 1.0.

## Arms

| Arm | Purpose |
|---|---|
| supervised_temporal | Raw lower-back temporal classifier |
| same_sensor_ssl | Temporal classifier plus masked back reconstruction |
| paired_csmr | Paired masked thigh-summary reconstruction and embedding alignment |
| paired_contrastive | Paired arm plus standing-focused batch margin |

The retained compact and rich Random Forest results are matched controls: 55.764% and
59.037% participant macro-F1 respectively. They are loaded from the preserved run-005
artifact and are not refit.

## Metrics and gates

The primary endpoint is equal-participant binary macro-F1. Report accuracy/micro-F1,
sitting and standing precision/recall/F1, confusion matrices, NLL, Brier, ECE, every
participant, worst participant, lower decile, and paired 10,000-resample participant
bootstrap against the rich control. Standing recall and standing F1 are co-primary
diagnostics.

The prospective nomination gate is +5 percentage points macro-F1 over the rich control,
+10 points standing recall and standing F1, sitting-recall loss no greater than 2 points,
positive paired bootstrap lower bound, at least 16/22 participant wins, and no participant
loss below 5 points. A paired arm must also beat same-sensor SSL by at least 3 points to
support a cross-sensor mechanism claim. The gate is a development rule, not a guarantee.

## Provenance

The source archive is downloaded to a temporary path, SHA-256 recorded, and removed after
the run. Outputs are create-only under `.audit/harth_sage_x_20260918/`. Predictions,
fold manifests, model constants, environment information, validation and failures remain
visible. No publication, release, remote push, tag or older-dataset transfer is included.
