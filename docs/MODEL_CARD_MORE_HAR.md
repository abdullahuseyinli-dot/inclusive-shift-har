# MoRe-HAR model card

## Model status

MoRe-HAR (Motion-Realization Factorized HAR) is a compact research hypothesis evaluated within InclusiveShift-HAR. It is not presented as state of the art, clinical technology, a fairness intervention, or proof of architectural novelty. The full model's preregistered zero-shot hypothesis was **not supported**.

## Intended task

- Input: 128 time steps by six inertial channels.
- Output: probabilities over functional mobility, sitting, and standing.
- Training cohort: source participants only for the locked zero-shot result.
- Inference metadata: no participant identity, disability status, or assistive-device status.

## Architecture

The compact architecture uses separate accelerometer and gyroscope stems, short/medium/long depthwise-separable temporal branches, residual dilated blocks, attentive statistical pooling, and content/realization latent projections. The classifier consumes only `z_content`. The realization branch predicts measurable signal descriptors and source-side transform parameters rather than sensitive labels.

The full training objective is:

```text
L = L_classification
  + lambda_content * L_supervised_contrastive
  + lambda_consistency * L_clean_augmented_consistency
  + lambda_realization * L_descriptor_or_transform_prediction
  + lambda_factor * L_cross_covariance
```

The full locked configuration also uses source-participant GroupDRO. Each component was introduced through a predeclared ablation. A parameter-matched static dual-branch control and missing-modality variants were included.

## Evaluation result

Across five frozen seeds and ten target participants, MoRe-HAR full achieved:

- mean participant macro-F1: 0.6353;
- 95% participant-bootstrap CI: [0.4911, 0.7665];
- worst participant macro-F1: 0.2584;
- lower-decile participant macro-F1: 0.2660;
- mean window balanced accuracy: 0.6643;
- NLL: 1.2793;
- multiclass Brier score: 0.5050;
- secondary ECE: 0.0946;
- AURC: 0.2021.

Compact DANN achieved a higher mean (0.6808), and the full model failed the required mean and joint lower-tail improvement rules. Candidate minus DANN mean participant macro-F1 was -0.0455; Holm-adjusted exact sign-flip p = 0.7207 and adjusted Wilcoxon p = 0.7559. Absence of significance is not evidence of equivalence, especially with only ten target participants.

The content/consistency ablation (0.6616) outperformed the full factorized model. This is evidence against claiming that the realization/factorization/GroupDRO combination produced the intended improvement in this setting. Removing either modality materially reduced performance (`no accelerometer`: 0.4456; `no gyroscope`: 0.4218), but those ablations do not identify causal sensor importance.

## Limitations

- One small public dataset and one waist-worn phone placement.
- Only three functional-core labels in the primary result.
- Hidden trial boundaries are unrecoverable in the released data.
- Target cohort size is ten; uncertainty and lower-tail instability are substantial.
- Realization descriptors are engineering proxies, not validated biomechanical constructs.
- Factorization losses do not guarantee identifiable or independent latent factors.
- Source participant labels used by GroupDRO/contrastive training may encode incidental dataset structure.
- No claim of benefit outside the locked protocol.

## Appropriate interpretation

MoRe-HAR is a documented negative/exploratory model result accompanying a stronger benchmark contribution. Future work may simplify toward the content-consistency ablation, but any such tuning requires a new development/validation cycle and cannot revise the consumed confirmatory result.
