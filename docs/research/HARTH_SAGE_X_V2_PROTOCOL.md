# HARTH SAGE-X v2 supervised-teacher correction

This is a bounded follow-up to the completed SAGE-X v1 screen. v1 showed that the
temporal encoder can learn thigh posture (the one-fold thigh upper bound was retained
as a diagnostic) but that an unsupervised thigh branch did not transfer useful geometry
to the back. v2 therefore freezes one additional mechanism: a supervised thigh teacher
and temperature-scaled back-to-teacher logit distillation. The original v1 artifacts are
immutable and remain the primary negative screen for the weaker teacher.

Only `paired_distill` and `paired_distill_contrastive` are fitted here. Both use the
same five participant-exclusive folds, seed family, windows, weighting and held-out
back-only inference as v1. The thigh tower receives training labels and is discarded
before held-out predictions. No held-out thigh signal enters the back predictor.

The new loss adds 0.50 times supervised thigh cross-entropy and 0.25 times a temperature
2.0 KL distillation term to the v1 paired reconstruction/alignment loss. The contrastive
variant adds the v1 standing-focused margin. No parameter search or extra seed is run.

Promotion remains prospective: compare with the retained rich RF and with v1 paired CSMR;
require standing recall and F1 gains, sitting protection, participant-level uncertainty,
and no severe harms. A positive v2 result would support the teacher-supervision mechanism,
not a general state-of-the-art claim.
