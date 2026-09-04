# FuSE/ReFrame v2 research report

Status: completed source-development and external held-out evaluation, 2026-09-03.

## Scope and repository identity

This report belongs to **InclusiveShift-HAR**, the smartphone/wearable inertial-sensor repository. It is not the separate DINO/ConvNeXt vision-based human-activity project and it is not the robot thesis repository. The model input here is six-channel phone IMU data: three-axis user acceleration and three-axis rotation rate.

The original InclusiveHAR target participants 11--20 were consumed once by the v1 protocol. They were not reopened for v2 model selection or evaluation. All v2 InclusiveHAR development below uses source participants 1--10 with participant-exclusive nested cross-validation. Consequently, a v2 source-development result is not a new zero-shot target result.

## Executive finding

The new work produced a credible engineering advance, but not a confirmed breakthrough.

The invented **Robust Multiscale Residual Pyramid (RMRP)** reached mean participant macro-F1 0.8379 in strict 5-by-4 nested source-participant cross-validation. Its predecessor, the **Geometric Spectral Pyramid (GSP)**, reached 0.8292. The paired mean improvement is +0.0087, below the prospectively recorded +0.015 engineering threshold; the exact paired sign-flip p-value is 0.2461 and the Wilcoxon p-value is 0.1641. The prospectively gated bottom-30% mean improved from 0.6771 to 0.6943 (+0.0171), but remained below its required +0.020 gain. The descriptive lower decile declined from 0.7373 to 0.7284, while the worst participant improved substantially from 0.4985 to 0.5494. This is promising robustness evidence, not proof of general superiority.

The often-cited 0.7514 number is from the v1 few-person k=4 setting, where four target-cohort participants enter training. It is not the previous score for this source-only nested experiment. Subtracting 0.7514 from 0.8379 would therefore create an invalid comparison.

## Protocol

- Participants are partitioned before windows.
- All preprocessing and model fitting occur inside the relevant training partition.
- Inner participant folds select candidates; outer participants are not used for selection or early stopping.
- The inferential unit is the participant, not the window.
- Primary endpoint: mean participant macro-F1.
- Prospectively gated tail endpoint: bottom-30% participant macro-F1. Lower-decile and worst-participant macro-F1 are additional descriptive endpoints.
- Ten-thousand-resample participant-cluster bootstrap intervals are reported.
- Participant ID, cohort/disability/device metadata, time, location, and released-block position are forbidden model inputs.
- No v2 target participant record, prediction, or metric was accessed.

The source benchmark contains 725 non-overlapping three-class windows from participants 1--10. The functional classes are mobility, sitting, and standing.

## Methods investigated

### Geometric Spectral Pyramid

GSP is a deterministic feature map created for short six-channel IMU windows. It combines raw-axis statistics, rotation-related scalar invariants, acceleration/gyroscope covariance geometry, a canonical principal frame, physical and normalized spectral bands, autocorrelation, Haar energy, and half/quarter temporal pyramids. Candidate estimators were selected in nested participant-exclusive folds.

### Robust Multiscale Residual Pyramid

RMRP was invented after the source-only corruption audit exposed noise and drift weaknesses. It applies the fixed GSP map to three signal views:

1. the original window;
2. a second-order Savitzky--Golay denoised view with a 0.20-second span; and
3. a residual obtained by subtracting a second-order 1.00-second trend.

The candidate set included each view and fixed concatenations. No filter parameter was learned from an outer fold or target data. All five outer folds independently selected the denoised view with participant/class-weighted Extra Trees, 500 trees, square-root feature sampling, and minimum leaf size one. The residual-only candidate ranked last in every fold, and no residual-containing concatenation was selected. The evidence therefore supports describing the selected method precisely as **denoised GSP**, while retaining RMRP as the name of the evaluated candidate family.

### Controls and rejected ideas

Every completed control is retained, including failures. Mean/lower-decile/worst participant macro-F1 were:

| Method | Mean | Lower decile | Worst | Decision |
|---|---:|---:|---:|---|
| HYDRA | 0.7071 | 0.5268 | 0.5023 | Rejected |
| MultiRocket | 0.7350 | 0.5740 | 0.5206 | Rejected |
| MultiRocket+HYDRA | 0.7362 | 0.5474 | 0.5260 | Rejected |
| QUANT | 0.7412 | 0.5517 | 0.5461 | Rejected |
| RIST | 0.7719 | 0.6649 | 0.6168 | Rejected |
| SpectralShape | 0.7888 | 0.6125 | 0.4585 | Rejected |
| SpectralShape+RIST | 0.7852 | 0.6030 | 0.4687 | Rejected |
| Cross-fitted expert stack | 0.7212 | 0.5220 | 0.5018 | Rejected |
| Gravity-augmented nine-channel GSP | 0.8090 | 0.5227 | 0.4985 | Rejected |
| DAGHAR-augmented GSP | 0.8221 | 0.6596 | 0.4440 | Rejected; every fold selected source-only |
| GSP | 0.8292 | 0.7373 | 0.4985 | Strong predecessor |
| RMRP | **0.8379** | 0.7284 | **0.5494** | Best strict source mean |

The five-seed GSP probability ensemble also failed to improve the frozen seed-11 candidate: 0.8268 versus 0.8292 mean, and 0.7085 versus 0.7373 lower decile. It was rejected. The individual seed means were 0.8292, 0.8307, 0.8170, 0.8288, and 0.8249.

Neural committee development averaged 0.6968. Direct outer-fold early-stopping numbers for DANN (0.7579) and Inception (0.7793) are optimistic, non-strict diagnostics and are excluded from the publication comparison.

## Main before-and-after results

| Strict source-development endpoint | GSP before | RMRP after | Change |
|---|---:|---:|---:|
| Mean participant macro-F1 | 0.8292 | **0.8379** | +0.0087 |
| Bottom-30% participant macro-F1 (prospective gate) | 0.6771 | **0.6943** | +0.0171 |
| Lower-decile participant macro-F1 | **0.7373** | 0.7284 | -0.0089 |
| Worst-participant macro-F1 | 0.4985 | **0.5494** | +0.0509 |
| Window accuracy, diagnostic only | 0.8345 | **0.8441** | +0.0097 |
| Mobility recall | **0.9806** | 0.9729 | -0.0078 |
| Sitting recall | 0.7051 | **0.7094** | +0.0043 |
| Standing recall | 0.8026 | **0.8369** | +0.0343 |
| 95% participant-bootstrap interval | [0.7429, 0.8924] | [0.7595, 0.8985] | Overlapping |

Participant-level RMRP macro-F1 values were: participant 1, 0.9016; 10, 0.5494; 2, 0.8961; 3, 0.9396; 4, 0.9113; 5, 0.9311; 6, 0.8270; 7, 0.7852; 8, 0.8896; and 9, 0.7483.

The improvement is directionally useful: standing recognition, the bottom-30% mean, and the weakest participant improved, while mobility remained very high. It is not statistically decisive with ten source participants, and both the mean gain (+0.0087 versus +0.015 required) and bottom-30% gain (+0.0171 versus +0.020 required) missed the pre-recorded joint engineering gate.

## Corruption robustness

The fixed diagnostic suite contains one clean case and 18 corruptions. Corruption outcomes were prohibited from selecting or retraining the evaluated candidate.

| Endpoint or representative case | GSP | RMRP | Change |
|---|---:|---:|---:|
| Mean over 18 corruptions | 0.7507 | **0.7581** | +0.0074 |
| Worst corruption | 0.5513 | **0.5673** | +0.0160 |
| Noise, severity 0.10 | 0.6588 | **0.7095** | +0.0507 |
| Noise, severity 0.30 | 0.5513 | **0.5923** | +0.0410 |
| Two-channel dropout | 0.7232 | **0.7589** | +0.0357 |
| Temporal gap, fraction 0.125 | **0.8081** | 0.7612 | -0.0469 |
| Temporal gap, fraction 0.250 | **0.7735** | 0.6989 | -0.0745 |
| Drift, severity 0.50 | **0.5692** | 0.5673 | -0.0019 |

RMRP therefore provides meaningful noise/dropout robustness, but does not solve drift and is less tolerant of contiguous temporal gaps. Publication claims must name that trade-off.

## Sparse labelled personalization

Semantic Anchor Reconciliation (SAR) is a separate labelled-personalization method, not zero-shot evaluation. For each new source participant it retains mobility and permits only a sitting/standing output-column swap when fixed, hash-selected labelled anchors provide a log Bayes factor of at least log(3). Anchor windows are removed from evaluation, and all budgets are reported without selecting the best one after seeing results.

| Base / labelled anchors per posture class | Same-remaining-window baseline | SAR | Lower decile after | Worst after |
|---|---:|---:|---:|---:|
| GSP / 1+1 | 0.8361 | **0.8654** | 0.7741 | 0.7677 |
| GSP / 3+3 | 0.8329 | **0.8656** | 0.7960 | 0.7455 |
| GSP / 5+5 | 0.8302 | **0.8642** | 0.7923 | 0.7367 |
| RMRP / 1+1 | 0.8439 | 0.8439 | 0.7193 | 0.5590 |
| RMRP / 3+3 | 0.8459 | 0.8584 | 0.7218 | 0.6810 |
| RMRP / 5+5 | 0.8417 | 0.8539 | 0.7139 | 0.6936 |

The best sparse-labelled result is the GSP/SAR branch, not RMRP. With 1+1 anchors, only participant 10 triggered a swap, which raised the worst score sharply. This is an interpretable personalization finding but requires an independent cohort and cannot replace the locked zero-shot result.

## External DAGHAR evaluation

The RMRP candidate was frozen before opening the designated evaluation domains. RealWorld-thigh, RealWorld-waist, and UCI had been used as development domains; MotionSense, KuHar, and WISDM were designated for held-out evaluation. Before that declaration, schema headers and one raw row had already been probed, so this is correctly labelled **held-out performance after schema probe**, not pristine confirmatory evidence.

The evaluation used 12,726 balanced windows, 4,242 per functional class, from 153 external participants. Acceleration was converted from the InclusiveHAR g interface to the DAGHAR m/s^2 interface with the fixed factor 9.80665; gyroscope units were rad/s in both.

| External domain | Participants | Windows | Mean participant macro-F1 | Lower decile | Worst |
|---|---:|---:|---:|---:|---:|
| MotionSense | 24 | 2,520 | 0.7985 | 0.6769 | 0.5986 |
| KuHar | 78 | 984 | 0.3679 | 0.1333 | 0.0000 |
| WISDM | 51 | 9,222 | 0.6923 | 0.5845 | 0.4145 |
| All participants, unweighted | 153 | 12,726 | 0.5436 | 0.1832 | 0.0000 |

The equal-domain mean was 0.6196 and the worst-domain mean was 0.3679. The all-participant bootstrap interval was [0.5043, 0.5816]. Class recall was 0.9995 for mobility, 0.7735 for sitting, and 0.4260 for standing. This is mixed external evidence: transfer to MotionSense is strong, WISDM is moderate, and KuHar exposes major unresolved domain/ontology shift. It does not validate a breakthrough claim.

## Interpretation of the historical scores

| Number | Actual protocol | Valid interpretation |
|---:|---|---|
| 0.6808 | v1 one-time target zero-shot, compact DANN | Best locked target mean; target participants never trained on |
| 0.6353 | v1 one-time target zero-shot, full MoRe-HAR | Unsupported v1 hypothesis |
| 0.7514 / approximately 0.7593 aggregate leader | v1 few-person k=4 | Four target-group participants enter training; post-confirmatory |
| 0.8292 | v2 strict nested source CV, GSP | Source-participant generalization only |
| 0.8379 | v2 strict nested source CV, RMRP | New best source-development mean |
| 0.8654--0.8656 | v2 GSP with labelled posture anchors | Sparse labelled personalization, not zero-shot |
| 0.6196 | v2 frozen RMRP on held-out DAGHAR domains | Equal-domain external mean; mixed transfer |

Legacy notebook scores that used random window splits, repeatedly inspected official tests, or lack recoverable participant provenance remain exploratory. They cannot be revived as publication comparators.

## Publication assessment

### What is publishable now

- A rigorous participant-exclusive benchmark with an immutable negative v1 result.
- RMRP as a clearly labelled post-analysis, source-developed robustness method.
- The contrast between a small mean gain and a much larger weakest-participant/noise gain.
- SAR as a separate interpretable sparse-labelled personalization result.
- The heterogeneous external-domain failure, including the KuHar result, as evidence that deployment robustness is unresolved.
- Complete rejected-control and failure reporting.

### Claims that are not supported

- A new zero-shot target breakthrough.
- Statistical superiority of RMRP over GSP.
- General cross-dataset superiority.
- Clinical effectiveness, causal disability effects, or universal smartwatch performance.
- A direct improvement from 0.7514 to 0.8379.

### Work still needed for a strong method paper

1. Evaluate once on a genuinely new, independently sealed ability-relevant cohort; do not reopen InclusiveHAR participants 11--20.
2. Increase the participant count enough to estimate lower-tail effects with useful precision.
3. Add a missingness-aware branch or mask/channel-health features to recover temporal-gap robustness, chosen solely on new development participants.
4. Address low-frequency drift with train-only detrending or physically justified high-pass features; current RMRP did not improve the severe-drift case.
5. Diagnose KuHar ontology, placement, sampling, and sensor-frame differences without using its reported metric to retune the already evaluated candidate.
6. Run HAR-PMD only when storage and an ontology-preserving protocol are available. The archive is approximately 7.3 GB and was not downloaded here. Its `still` label only partially identifies sitting versus standing; active crutches, walker, and manual-wheelchair modes may map to mobility, while electric/passive wheelchair modes require exclusion under the current contract.
7. Refresh the literature/novelty search immediately before submission and avoid unqualified “first” or state-of-the-art claims.

## Reproducibility anchors

Implementation commit: `c50a5e23e001763d668eb0256e29e65f69b10413`.

| Artifact | Record SHA-256 | File SHA-256 |
|---|---|---|
| GSP nested result | `db190c71a74ddf762622142348d3ba1b07dc5941129b4f394a7b8c61ca392993` | `27fbdf69998c8e617610df8e363b70c6310df783768b1ed103bf32e458949ee5` |
| RMRP canonical nested result | `bbd568f730ebe4e8df314e4399fa0ae991df99999a65890e507d78567f913c9a` | `d5fdbb4b30f91453a6511e0e0e3163c9f10db2fdeb687fac6f415c8c3c31bcee` |
| GSP corruption result | `e1c1a9bd5c99a30a355a5558b74262b52f14ee672530275386a92613661b5e13` | `ff69fee46467df20fa37fd3e6a04461d616f23ba8232f4106145442547c93021` |
| RMRP corruption result | `5bd8ba33e9edc082bf97a0dc93afe7a27c4cd9e25d2a3aa8227615ed987c3084` | `98aa38489bb58688ed0530848269bc4eda26bf82875ac92f62a828fe2d141479` |
| GSP semantic anchors | `4b9bedff7c66ca7f34bfd70c7bb50b48f9f6b4f06d5439846f1b7d4886542b6e` | `a2d8f77b1139273d8e36bcab813d6198f10c780eb195e8a5b3cbd1664050688c` |
| RMRP semantic anchors | `248d4d2aabc54495050987e9a31e16b8996136f0231662e453886e14039be139` | `bbacf8573c592e64b8e1bd9d6657fc55a74b3b210cbafe88e3a064bc918889e0` |
| DAGHAR candidate freeze | `ad90b062907d4291b002fe97eeae9aa3a2a01f0afce3045f22604cfba63d3b69` | `debb2417b253e450ce1eee9c526a8162e5e4651c5491c3506d82b30bab3bb1a5` |
| DAGHAR opening receipt | `234498aeafaed95d3d8cfd83b89addfc0d548f27512227b4d28600efa3033585` | `04d44628f03e1df231ded7caefeed8054e9c9f76f9665283bd2f6864559c89c5` |
| DAGHAR held-out result | `21f99368d68c3d1104e75483dbe00cf82e9ca806ae18992ba47aa8727b884d7f` | `e842338136f1baf7ae904ba793b2ae461a951f8cb4a086f431a5315b4831f655` |

The detailed `.audit/v2` result artifacts, external raw data, and fitted checkpoint are intentionally outside normal Git history. The committed code, configurations, dataset manifest, report, and hashes bind the claims without redistributing third-party data.

## Primary research sources

- DAGHAR data descriptor: <https://www.nature.com/articles/s41597-024-03951-4>
- DAGHAR reference code: <https://github.com/H-IAAC/DAGHAR>
- HAR-PMD archive: <https://zenodo.org/records/7939223>
- HAR-PMD data descriptor: <https://doi.org/10.1038/s41597-025-06527-y>
- OOD IMU/video representation study: <https://proceedings.mlr.press/v298/cheshmi25a.html>
- Pediatric HAR generalization study: <https://proceedings.mlr.press/v309/costantini26a.html>
- TRI-HAR preprint: <https://arxiv.org/abs/2608.15621>
- CCIL preprint: <https://arxiv.org/abs/2412.13594>
- Apple Core Motion gravity interface: <https://developer.apple.com/documentation/coremotion/cmdevicemotion/gravity>
- InclusiveHAR article: <https://pmc.ncbi.nlm.nih.gov/articles/PMC12993405/>
