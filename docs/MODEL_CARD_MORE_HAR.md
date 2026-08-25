# MoRe-HAR model card

## Model status

MoRe-HAR (Motion-Realization Factorized HAR) is a compact research hypothesis evaluated within InclusiveShift-HAR. The full configuration reached 0.6353 mean target-participant macro-F1 and 0.2660 lower-decile performance, below the strongest compact baselines on both endpoints. Its preregistered zero-shot hypothesis was **not supported**.

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

Compact DANN achieved a higher locked-primary mean (0.6808), with compact CORAL effectively tied only 0.0000556 lower. The full model failed the required mean and joint lower-tail improvement rules. Candidate minus DANN mean participant macro-F1 was -0.0455; Holm-adjusted exact sign-flip p = 0.7207 and adjusted Wilcoxon p = 0.7559. Absence of significance is not evidence of equivalence, especially with only ten target participants.

The content/consistency ablation (0.6616) outperformed the full factorized model. This is evidence against claiming that the realization/factorization/GroupDRO combination produced the intended improvement in this setting. Removing either modality materially reduced performance (`no accelerometer`: 0.4456; `no gyroscope`: 0.4218), but those ablations do not identify causal sensor importance.

## Post-confirmatory evidence

The few-person v1.1 analysis refit the frozen model configurations with k=1,2,4 target-group training participants under predefined held-out outer folds. Full MoRe-HAR obtained means of 0.635332, 0.616057, 0.693510, and 0.751408 at k=0,1,2,4. Its k=4-minus-k=0 mean difference was +0.116075, but both global-Holm-adjusted paired p-values were 1.0. The trajectory was not monotone because k=1 was lower than k=0. This is descriptive inclusion evidence, not support for the rejected zero-shot hypothesis.

A paper-derived CCIL loss adaptation obtained target mean/worst/lower-decile macro-F1 0.6896/0.2724/0.3459. Its mean was 0.0119 above compact ERM, but both tails were slightly lower and the family-adjusted paired permutation and Wilcoxon p-values were 0.2109. A boundary-safe local BPD adaptation obtained 0.5710/0.2471/0.2751. Neither is official code or a faithful reproduction, both were designed after the target opening, and neither establishes MoRe-HAR superiority or architectural novelty.

The CUDA efficiency aggregate reports 138,396 parameters for MoRe-HAR full and batch-1 FP32 mean latency 2.0207 ms on the recorded RTX PRO 3000 Blackwell Laptop GPU. This is a synthetic-zero, device-resident, forward-only measurement with no host-to-device transfer. Timing was valid only under the declared allowlisted ambient WDDM-process policy, and analytical MAC/FLOP counts cover only supported operators.

## Limitations

- One small public dataset and one waist-worn phone placement.
- Only three functional-core labels in the primary result.
- Hidden trial boundaries are unrecoverable in the released data.
- Target cohort size is ten; uncertainty and lower-tail instability are substantial.
- Realization descriptors are engineering proxies, not validated biomechanical constructs.
- Factorization losses do not guarantee identifiable or independent latent factors.
- The learned factorization is not a causal decomposition of activity realization.
- Source participant labels used by GroupDRO/contrastive training may encode incidental dataset structure.
- The few-person curve was post-confirmatory; none of its 80 paired comparisons survived global Holm correction.
- CCIL/BPD rows are qualified local adaptations, not official-faithful reproductions.
- Efficiency timing is machine- and measurement-scope-specific, not end-to-end deployment latency.
- External transfer remains unmeasured: the locked primary result covers one
  waist-worn smartphone dataset, three functional-core labels, and ten held-out
  target participants; no other dataset, placement, or activity ontology was
  evaluated.

## Appropriate interpretation

MoRe-HAR is a documented negative/exploratory model result accompanying a stronger benchmark contribution. Future work may simplify toward the content-consistency ablation, but any such tuning requires a new development/validation cycle and cannot revise the consumed confirmatory result.
