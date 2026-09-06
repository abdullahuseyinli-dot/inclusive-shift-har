# External HAR literature and comparability audit — 2026-09-05

**Audit date:** 2026-09-06

**Evidence cut-off:** 2026-09-06

**Scope:** external wearable-sensor HAR datasets and recent cross-dataset or efficient HAR methods relevant to this repository

**Source rule:** primary sources only: official data records, publisher or proceedings pages, author manuscripts, and official code repositories

**Status:** publication-supporting comparison audit; it is not a result artifact and does not upgrade provisional, superseded, diagnostic, or blocked experiments

## Executive verdict

No external numerical result inspected in this audit is directly comparable with the repository's frozen external-HAR endpoint. Every inspected study differs on at least one material element of the comparison tuple: dataset bytes/version, participants, ontology, placement/channels, gravity lane, sampling/preprocessing, windowing/boundaries, participant split, supervision or target exposure, inference/reset unit, metric aggregation, or seeds/uncertainty.

Consequently:

- No state-of-the-art (SOTA) claim is valid from the currently retained comparisons.
- Published accuracy, weighted F1, balanced accuracy, or window-level macro-F1 must not be placed in a common ranking with participant-balanced fixed-class macro-F1.
- Cross-dataset methods that use unlabeled target data are not zero-shot source-only controls.
- A camera-defined bout reset is an oracle diagnostic, not a deployable temporal endpoint.
- Native gravity and derived gravity are separate sensor lanes and must never be presented as interchangeable.
- Repetitions from the same people increase observations, not independent participant count.
- Until a genuinely untouched, protocol-frozen cohort is lawfully obtained and evaluated once, this project has development, stress-test, oracle-diagnostic, and blocked/sealed evidence—not independent confirmation.

The strongest defensible wording, after corrected runs exist and validate, is **“best observed under our frozen protocol”**. “Preregistered” may be added only for a comparison whose protocol was actually timestamped and frozen before outcomes were examined. It must not be used as a synonym for a configuration that was frozen retrospectively.

## Exact comparison rule

A before/after or literature comparison is numerical only when all fields below match. A difference in any field requires separate rows and a **not comparable** label.

1. Dataset release/version and source-file hashes.
2. Participant population and participant set.
3. Label ontology and exclusion/mapping rules.
4. Sensor placement and available channels.
5. Native-sensor versus derived-gravity lane.
6. Sampling, missing-data handling, filtering, normalization, and other preprocessing.
7. Window length, stride, padding, boundary definition, and label-purity rule.
8. Participant-exclusive split and all train/validation/test reuse.
9. Supervision, personalization, unlabeled-target access, and external pretraining budget.
10. Inference unit and temporal reset rule.
11. Metric definition and aggregation unit.
12. Seeds, repeated trials, and uncertainty procedure.

“Same dataset name” is insufficient. Dataset revisions, provider-defined activity clips, reconstituted sessions, alternative placements, target-domain adaptation, and different aggregation units all change the scientific question.

## Repository endpoint used for the audit

The reference endpoint below is taken from the repository's frozen configuration and protocol documents, especially `configs/datasets/external_har_portfolio_v1.yaml`, `configs/experiments/cross_dataset_har_rnd_v1.yaml`, the composite `configs/protocols/external_har_publication_v4.yaml`, its constituent `configs/protocols/external_har_session_grid_v3.yaml` and provider-boundary/observable-context records, and `configs/protocols/sole_harmony_observable_session_temporal_v1.yaml`. A replacement publication run is reportable only if its own frozen configuration and launch receipt bind `external-har-publication-v4` and agree with every applicable constituent protocol.

| Field | Repository external-HAR contract |
|---|---|
| Core signal rate/window | 50 Hz; 128 samples (2.56 s); stride 128 (non-overlapping) unless a separately versioned endpoint says otherwise |
| Segmentation | Only participant, physical session/trial, timestamp discontinuity, and finite-data runs may define signal segments |
| Resampling and labels | Resample each complete, label-independent physical segment once; project annotations afterwards; construct one deterministic global candidate grid before homogeneous/purity eligibility |
| FoG-STAR correction | Composite `external-har-publication-v4`, retaining the signal-grid rules of `external-har-session-grid-v3` and adding observable-context, provider-boundary, and pre-window participant-partition requirements; activity, task, FoG, and severity annotations cannot affect signal resampling, gravity estimation, timestamps, or candidate starts |
| Causality | The current symmetric polyphase resampling path is offline and must not be described as zero-lookahead streaming |
| Sensor lanes | Six-channel linear acceleration plus gyroscope; nine-channel **derived-gravity** is distinct from any provider-native gravity channel |
| Splits | Participant-exclusive, five outer folds and four inner folds where applicable |
| Seeds | 11, 23, and 47 for the frozen three-seed suites |
| Primary aggregation | Mean across seeds of equal-participant, fixed-class participant macro-F1 |
| Required sensitivity/tail evidence | Present-class/eligible-participant sensitivity; per-class precision/recall/F1; participant-normalized confusion; worst participant and bottom 30%; median/quartiles; support/eligible N; rescue/harm/tie counts |
| Uncertainty | Paired participant-cluster bootstrap, 10,000 replicates, fixed bootstrap seed 20260905, for matched paired contrasts |
| Calibration | Participant-mean NLL/Brier/ECE when valid probabilities exist; pooled-window calibration is diagnostic and must be named as such |
| Temporal reset | Only inference-observable session/gap boundaries for a deployable claim; camera bouts remain an oracle upper-bound diagnostic |

## Dataset identity and acquisition matrix

`NR` means that the exact item was not reported or could not be verified in an official source as of the cut-off. An upstream MD5 is a file-integrity identifier, not proof that the scientific protocol is valid.

| Dataset | Exact release inspected | Participants/sessions | Official data/code and license | Repository role and access state |
|---|---|---|---|---|
| FoG-STAR | Zenodo **v3**, published 2025-12-06; `sensor_data.csv` 119,629,580 bytes, MD5 `952a37ab147da35e6d4e7a1e9bac44cb`; repository receipt SHA-256 `888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477` | 22 people with Parkinson's disease; four body locations; seven scripted tasks | [Official Zenodo v3 record](https://zenodo.org/records/17838806), CC BY 4.0; [official detector code](https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor/tree/2b2277eda53051142cc8198a8e0c0a3f61f2de0b), no repository software license verified | Development-consumed. Earlier FoG-derived and FoG-target results are superseded/provisional pending the label-independent resampling rerun. |
| IMU-HAR-IL | CSIRO collection **74700, data version 1**, DOI 10.25919/d7xf-n080; 83,409 files and 49,771,532,800 bytes reported; no provider aggregate content hash found | 50 adults; up to four repetitions per person | [Official CSIRO v1 record](https://data.csiro.au/collection/csiro:74700), CC BY-NC 4.0; [official version history](https://data.csiro.au/dap/ws/v2/collections/74700/versions); [official code](https://github.com/mmsandhu/IMU-HAR-IL/tree/7d8244db71f5f15e094664df1ebb02315769f0ba), MIT | Development evidence. Public release is provider-presegmented; repetitions are not independent participants. |
| HAR-PMD | Zenodo **v2**; `data_publish.zip` 7,288,182,191 bytes, MD5 `cdc7b79aaa0450d68d94f430f2a2de63` | 127 recruited, seven excluded, final N=120; 60 phone-only and 60 phone-plus-watch | [Official Zenodo v2 record](https://zenodo.org/records/7939223), CC BY 4.0; [official project code](https://github.com/HanyangHCILab/HAR-handicaps/tree/88fd8972e242776d0b148775e8385ec9a274174e), no root software license verified | Consumed pilot/stress evidence. Full 120-participant, three-seed project evaluation remains the appropriate stress endpoint; it is not clinical confirmation. |
| Sole-HARmony | Project data pin: Zenodo **v1**, published 2026-04-13, 53.4 GB. A distinct **v2**, published 2026-07-21, is 54.1 GB and changes all 13 participant archives inspected; do not relabel v1 bytes as v2. | 13 healthy adults; five consecutive days; approximately 287 hours | [Official v1 record](https://zenodo.org/records/19242395), [official v2 record](https://zenodo.org/records/21477296), both data records CC BY 4.0; [official code](https://github.com/wearable-robotic-systems-lab/Sole-Harmony/tree/3c57f159fdc90ac91e6d9ab4ce25e4bfd384d0fd), no root software license verified | A retained v3 camera-bout oracle run materialized C001-C012 and their first two ordered sessions (24 sessions; 137,997 windows), then stopped before any model result, predictions, validation, or score. This is coverage/schema/oracle-boundary diagnostic evidence only. A current-protocol observable-session evaluation is pending. |
| WearGait-PD | Synapse project `syn52540892`; project tree identifies **Version 1 (Cross-sectional)** `syn55052683` and **Version 2 (Longitudinal)** `syn74686228`; no local frozen release/receipt | Cross-sectional descriptor: 100 people with Parkinson's disease and 85 age-matched controls (N=185) | [Official Synapse project](https://www.synapse.org/Synapse:syn52540892), DOI [10.7303/syn52540892](https://doi.org/10.7303/syn52540892), CC BY 4.0; access requires a Synapse account and acceptance of applicable Terms, Privacy Policy, Code of Conduct, and data pledge | Sealed confirmation candidate. No authorized local data receipt exists; public metadata is not authorization to download or analyze controlled files. |
| Parkinson@Home | Full multimodal activity release/version/hash/license not frozen. The separately public tremor subset is DOI **10.34973/2xxa-g520, version 2**, approximately 4 GB/13 files, CC0 | Original study recruited 25 people with Parkinson's disease and 25 controls; public tremor subset reports 25 and 24, respectively | [Original study article](https://www.jmir.org/2020/10/e19068); [public tremor-subset DOI](https://doi.org/10.34973/2xxa-g520); [official tremor code](https://github.com/biomarkersParkinson/pdathome_tremor/tree/92a814024846e68a67e6fc5f6fe8e52f74fc555e), Apache-2.0 | Full activity data are sealed/request-gated. The public tremor subset is not a substitute for the full HAR corpus. |

### Version corrections that must remain visible

- **IMU-HAR-IL:** the project pins collection 74700/data version 1. The provider version service also lists later collection IDs 75538 (v2) and 76561 (v3). “Current IMU-HAR-IL” is therefore ambiguous and must not silently replace the frozen v1 bytes.
- **Sole-HARmony:** v1 and v2 are materially different data releases. Demographics matched in the inspected records, but each participant archive MD5 changed. Any cross-version comparison is invalid without a fresh receipt and rerun.
- **FoG-STAR:** the provider record supplies an MD5 and the repository receipt supplies a SHA-256. The provider documentation's timestamp-unit description and the repository's empirical timestamp audit disagree; that discrepancy must stay disclosed rather than being normalized away in prose.
- **WearGait-PD:** “v1” and “v2” identify cross-sectional and longitudinal branches in the Synapse project, not interchangeable revisions of one frozen local archive.
- **Parkinson@Home:** the openly deposited tremor subset and the full activity/video study are different evidence objects with different access scope and endpoints.

## Dataset and provider-protocol comparability

### FoG-STAR

The peer-reviewed descriptor reports 22 people with Parkinson's disease assessed off medication, with IMUs on both ankles, lower back, and most affected wrist at 60 Hz. Accelerometer and gyroscope signals accompany walking, sitting, standing, sit-to-stand, stand-to-sit, and right/left turning annotations across seven tasks; 101 freezing-of-gait episodes were annotated. See the [Scientific Data descriptor](https://www.nature.com/articles/s41597-026-06645-1) and [official data record](https://zenodo.org/records/17838806).

The provider's released detector is a FoG detector, not a standardized mobility/sitting/standing HAR benchmark. Its documented input is ankle gyroscope data in 120-sample, two-second windows at 60 Hz with 75% overlap; its endpoint and labels therefore do not match this repository's three-class fixed-grid endpoint. See the [official detector repository at the audited commit](https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor/tree/2b2277eda53051142cc8198a8e0c0a3f61f2de0b).

**Our endpoint:** a three-class development evaluation (mobility/sitting/standing), 50 Hz, 128 samples, no overlap, corrected label-independent physical-session resampling, participant-exclusive nested folds, three seeds, and equal-participant fixed-class macro-F1 with participant-cluster uncertainty.

**Verdict:** **not comparable** to the provider FoG detector or its technical validation. Sensor subset, ontology, window duration/overlap, task, split, metric, and uncertainty differ. Earlier repository scores produced before the label-independent resampling correction remain superseded/provisional even if their JSON artifacts passed structural validation.

### IMU-HAR-IL

The official author manuscript describes 50 adults, 17 activities in four domains, and up to four repetitions. Thirty Xsens DOT sensors were used: 11 body-worn and 19 object-mounted units, recording accelerometer, gyroscope, and magnetometer streams at 60 Hz. Source recordings were continuous within repetitions, but the published preparation split recordings into repetition/activity folders and included manual label correction or removal. Public Body-WT tables contain activity labels and sensor features but do not preserve a timestamp/session offset adequate to reconstruct the original continuous label-free grid. See the [official CSIRO record](https://data.csiro.au/collection/csiro:74700), [author manuscript v2](https://arxiv.org/html/2608.07502v2), and [official code](https://github.com/mmsandhu/IMU-HAR-IL/tree/7d8244db71f5f15e094664df1ebb02315769f0ba).

The provider baseline uses a reported 80/20 data holdout, one-second windows with 50% overlap, engineered time/frequency features and random forests, and average accuracy. The publication does not establish that the holdout is participant-exclusive and does not report participant-cluster confidence intervals.

**Our endpoint:** derived three-class body-placement development analysis at 50 Hz/128 samples with participant-exclusive evaluation. Because the public bytes are already activity-segmented, the repository cannot prove the stronger continuous, annotation-invariant preprocessing contract from this release. All repetitions remain clustered within the same 50 participants.

**Verdict:** **not comparable** to the provider baseline, and the repository endpoint must be labelled **provider-presegmented diagnostic/development evidence**. IMU-HAR-IL-to-FoG transfer can be target-label-blind after the FoG fix, but it is not independent confirmation because FoG-STAR was consumed during architecture development.

### HAR-PMD

The peer-reviewed descriptor reports 120 retained participants, with six activity groups: still, walking, crutches, walker, manual wheelchair, and electric wheelchair. The first 60 used a smartphone; the next 60 used a smartphone and smartwatch. Phone carriage was intentionally variable (hand, pocket, bag, and other natural placements), while the watch was on the non-dominant wrist. Recordings span indoor/outdoor scenarios and total 14,400 minutes. Nominal sampling was 60 Hz, with observed means of approximately 59.6 Hz for phones and 51.7 Hz for watches. See the [Scientific Data descriptor](https://www.nature.com/articles/s41597-025-06527-y) and [official data record](https://zenodo.org/records/7939223).

The official pipeline discards the first and last 30 seconds of each 11-minute recording, interpolates to 60 Hz, and uses non-overlapping five-second windows. Its published A/AG/AGM model families mean linear acceleration; linear acceleration plus **gyroscope**; and linear acceleration, gyroscope, plus magnetometer, respectively. AG therefore contains no gravity channel and does not establish equivalence to the repository's provider-native gravity lane. The provider benchmark reports both user-dependent random-window five-fold evaluation and user-independent leave-one-group-out five-fold evaluation; the latter uses 96/24 phone participants and 48/12 phone-plus-watch participants for train/test. Accuracy and classification reports/confusion summaries are produced, but participant-level bootstrap uncertainty is not reported. The provider record's displayed phone fold roster also has an unresolved endpoint error: fold 5 is printed as `96-120`, which contains 25 participants and overlaps fold 4 at participant 96 despite the accompanying statement that every fold contains 24 participants. The literal roster therefore cannot be treated as an internally consistent frozen split without provider clarification or a documented correction. See the [official data record](https://zenodo.org/records/7939223) and [official code](https://github.com/HanyangHCILab/HAR-handicaps/tree/88fd8972e242776d0b148775e8385ec9a274174e).

**Our endpoint:** a five-class mobility-mode stress test that excludes electric wheelchair and maps sitting/standing into provider “still,” with separate 6-channel and provider-native-gravity 9-channel lanes, 50 Hz/128 samples, participant-exclusive folds, three seeds, and participant-balanced metrics.

**Verdict:** **not comparable** to the provider benchmark. The ontology, sample rate, window, sensor lane, split, and aggregation differ. This cohort's simulated mobility-aid tasks and merged “still” class cannot validate clinical performance or a sitting-versus-standing mechanism, regardless of score.

### Sole-HARmony

The official dataset contains 13 healthy adults monitored for five consecutive days for roughly 287 hours. Bilateral instrumented insoles each contain eight force-sensing resistors and an IMU; accelerometer and gyroscope streams are used at 270 Hz, with a downward camera and activPAL on ten participants. The official hardware metadata conflict: both Zenodo v1 and v2 call each insole IMU **6-DoF**, whereas the pinned official repository calls it **9-DOF** and says the magnetometer was not used. The accessible sources agree on the analyzed tri-axial accelerometer and gyroscope channels, but this audit cannot resolve the physical IMU degree count. Labels include sitting, standing, walking, stairs down, stairs up, and undefined. The official sources also differ slightly on total hours (286.75 in the data records and 286.57 in the code README); the audit preserves rather than resolves both discrepancies. See [Zenodo v1](https://zenodo.org/records/19242395), [Zenodo v2](https://zenodo.org/records/21477296), the [Scientific Data article](https://www.nature.com/articles/s41597-026-08172-5), and the [official repository](https://github.com/wearable-robotic-systems-lab/Sole-Harmony/tree/3c57f159fdc90ac91e6d9ab4ce25e4bfd384d0fd).

The official benchmark uses three-second windows with 50% overlap, 118 handcrafted features with hierarchical XGBoost and a CNN-BiLSTM, and IMU+FSR, IMU-only, and FSR-only modalities. It reports balanced accuracy and macro-F1, including per-subject/aggregate confusion summaries, under leave-one-out evaluation of ten training participants and a separate ten-train/three-test design. The released code defaults to seed 42; participant-bootstrap intervals are not reported.

**Our current evidence:** a retained v3 right-insole, 50 Hz/128-sample, derived-gravity camera-bout oracle run materialized C001-C012 and the first two ordered sessions for each participant (24 sessions, 3,137 bouts/trials, and 137,997 windows). It was interrupted immediately after `data_audit.json`; no result, predictions, validation, or score exists. The hash-bound interruption receipt is `results/research/cross_dataset_har_v4/sole_harmony_oracle_materialization_interruption_20260906.json`. Camera-bout resets remain annotation-defined oracle boundaries. A versioned observable-session development protocol exists and uses only participant/session/gap/finite-run reset information, but it has not produced a current-protocol result and its offline symmetric resampling does not establish zero-lookahead deployment.

**Verdict:** **not comparable** to the official multimodal benchmark. Dataset version, participant subset, modality, preprocessing, window, split, reset unit, and uncertainty differ. The retained repository evidence is an interrupted **oracle diagnostic materialization, not a model result**; it supports no deployable, population-performance, or before/after claim. The dataset files are CC BY 4.0; the journal article is separately licensed by the publisher, and no root software license was verified for the code repository.

### WearGait-PD

The cross-sectional descriptor covers 100 people with Parkinson's disease and 85 age-matched controls. Thirteen Xsens IMUs were placed on the forehead, sternum, L4/L5, and bilateral wrists, thighs, shanks, ankles, and dorsal feet. Signals were acquired internally at 1000 Hz/filter 184 Hz and provided at 100 Hz with acceleration, angular velocity, magnetic field, orientation, and free acceleration. Bilateral Moticon insoles provide 16 pressure channels per insole plus 3-D accelerometer and gyroscope data at 100 Hz. The battery includes self-selected and hurried gait, turning, tandem gait, timed-up-and-go, balance, doorway, and free-walk tasks, with frame-level general and clinical events. See the [Scientific Data descriptor](https://www.nature.com/articles/s41597-026-06806-2) and [official Synapse project](https://www.synapse.org/Synapse:syn52540892).

The release does not define an identical fixed three-class, 50 Hz/128-sample benchmark. Its broad event taxonomy, multimodal placements, clinical population, and cross-sectional/longitudinal versions require an explicit mapping and frozen release before evaluation.

**Access and license:** the official project states CC BY 4.0, but file access requires account registration and acceptance of Synapse terms, privacy policy, code of conduct, and the applicable data pledge. Those agreements must be accepted by the user/data controller, not by an autonomous agent. No local authorized receipt was found in the project evidence reviewed for this audit.

**Verdict:** **blocked/sealed, not comparable**. It may be a future untouched confirmation candidate only if lawfully obtained and if method, preprocessing, comparison set, metrics, seeds, acceptance gates, and exact release/hash are frozen before first outcome inspection.

### Parkinson@Home

The original study recruited 25 people with Parkinson's disease with motor fluctuations and 25 controls for unscripted home recording, with the Parkinson group recorded in off- and on-medication states. Physilog units were worn on both wrists, both ankles, and the lower back, alongside smartwatch, smartphone, and Empatica devices. The original article describes general-behavior annotations such as standing, walking, and sitting; the later tremor paper enumerates the wider activity inventory as sitting, standing, gait, postural transitions, running or exercising, cycling, and driving a motorized vehicle. The original published endpoint centered on gait biomarkers/AUC rather than a fixed HAR benchmark. See the [official JMIR article](https://www.jmir.org/2020/10/e19068) and [npj Parkinson's Disease tremor paper](https://www.nature.com/articles/s41531-025-01056-2).

A later open object, DOI [10.34973/2xxa-g520](https://doi.org/10.34973/2xxa-g520), is a version-2 tremor subset (25 Parkinson/24 control participants, bilateral wrist accelerometer and gyroscope at 200 Hz, tremor annotations, CC0). The associated paper reports analysis of 24+24 after technical exclusions and states that access to the wider study data excluding raw video is arranged through a request process; see the [npj Parkinson's Disease paper](https://www.nature.com/articles/s41531-025-01056-2) and [official Apache-2.0 code](https://github.com/biomarkersParkinson/pdathome_tremor/tree/92a814024846e68a67e6fc5f6fe8e52f74fc555e).

**Verdict:** the full activity corpus is **blocked/sealed and not comparable** until access, exact version, license/terms, hashes, ontology, and permissible use are frozen. The public tremor subset is openly licensed but answers a different task and cannot be substituted for full-session HAR confirmation.

## Recent method audit

The table separates what was verified in publisher/paper sources from what was observed in official code. `NR` means not reported in those primary sources. None of these rows is an identical-protocol numerical baseline for this repository.

| Work | Dataset/version and split | Sensors, preprocessing, ontology | Supervision/target exposure | Metric, aggregation, uncertainty | Code/license and cost | Comparability verdict |
|---|---|---|---|---|---|---|
| **HARMamba** — IEEE Internet of Things Journal 12(3), 2025; DOI [10.1109/JIOT.2024.3463405](https://ieeexplore.ieee.org/document/10683697/) | PAMAP2, WISDM, UNIMIB-SHAR, UCI-HAR; exact source releases/hashes NR. Paper uses continuous-data 70/10/20 “mean method”; participant exclusivity is not established. | Missing values linearly interpolated, standardized, 50% overlap. Dataset-specific windows/rates: WISDM 20 Hz/200, PAMAP2 33.3 Hz/512, UNIMIB 30 Hz/151, UCI 50 Hz/128. Native ontologies differ. | Fully supervised within-dataset training; bidirectional sequence architecture. | Paper reports accuracy/precision/recall/F1. [Official code](https://github.com/dianoDouble/HARMamba/blob/3afb0dced3c66702d698af354d7ec7e753e772ad/engine.py#L214) calculates sample-weighted F1 in evaluation; default seed 0; no participant-cluster interval. | [Official code at audited commit](https://github.com/dianoDouble/HARMamba/tree/3afb0dced3c66702d698af354d7ec7e753e772ad), Apache-2.0. Reported model size about 0.2201–0.4271 M parameters and 7.94–30.60 M FLOPs depending on dataset. | **Not comparable:** different corpora, version control, splits, overlap, ontology, metric aggregation, seeds, and inference architecture. Useful only as an architecture/efficiency reference. |
| **LanHAR** — ACM IMWUT 9(4), article 230, 2025; DOI [10.1145/3770652](https://doi.org/10.1145/3770652) | HHAR, UCI-HAR, MotionSense, Shoaib, plus PAMAP2 analysis; exact upstream versions/hashes NR. Manuscript v4 assigns 80%/20% of the selected labelled source to train/validation and the entire selected target to test; the source split unit and exact participant rosters are NR. Its stage-two text says unlabeled data come from “remaining source datasets,” while the problem formulation and training overview refer more broadly to unlabeled target/all-dataset data, so selected-target exposure is ambiguous in the manuscript alone. | Resampled to 50 Hz; exact common window length/stride is NR in the inspected manuscript. Main cross-dataset mapping is walking/upstairs/downstairs/still, with sitting and standing merged; an appendix examines five classes. | The released implementation resolves the ambiguity unfavorably: stage two [loads and combines the selected target with the source](https://github.com/DASHLab/LanHAR/blob/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29/models/load_data.py#L18-L29), and its [training records include selected-target signals and generated pattern text](https://github.com/DASHLab/LanHAR/blob/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29/models/read_data.py#L80-L95). It [uses a random 20% subset of target labels for validation](https://github.com/DASHLab/LanHAR/blob/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29/models/load_data.py#L31-L49); [validation accuracy selects the saved checkpoint](https://github.com/DASHLab/LanHAR/blob/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29/models/training_stage2.py#L128-L158), while the test loader contains the full target dataset, including that validation subset. GPT-4-produced semantic descriptions and a BERT text encoder are additional external resources. | Accuracy and F1; [official testing code](https://github.com/DASHLab/LanHAR/blob/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29/models/testing.py) uses sample-weighted F1. Participant-cluster uncertainty NR. | [Official code](https://github.com/DASHLab/LanHAR/tree/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29), Apache-2.0; compute/parameter matching to our controls NR. [Author manuscript v4](https://arxiv.org/abs/2410.00003). | **Not comparable:** different datasets/ontology, released-code target exposure and label-based checkpoint selection, language-model resource, split, evaluation reuse, metric aggregation, and uncertainty. The released implementation is not a source-only, target-label-blind zero-shot control. |
| **HAR-DoReMi** — Neurocomputing 694, 2026; DOI [10.1016/j.neucom.2026.133963](https://www.sciencedirect.com/science/article/pii/S0925231226013603) | The publisher page is internally inconsistent: its Dataset section says six public datasets, while its introduction's contribution summary says four. The accessible [author preprint v1](https://arxiv.org/abs/2503.13542) details HHAR, MotionSense, Shoaib, and UCI. Exact final-version source releases/hashes and the final dataset tuple were not verified. | Preprint/code: common accelerometer+gyroscope, 20 Hz, 120 samples (6 s), non-overlap; walking/upstairs/downstairs/still with sitting and standing merged; Mahony alignment. | Preprint v1 explicitly frames the experiment as domain generalization: jointly train on multiple labelled source datasets and evaluate on an unseen selected target, without incorporating that target into pretraining. The released [`main.py`](https://github.com/ABan147/har-doremi/blob/0af2711d0bb3776bacd539fb5c44ae4c765c85c0/main.py#L77-L85) likewise accepts only a list of training domains; it does not expose a selected-target input. Target labels are necessarily used to calculate reported evaluation metrics, but no selected-target training access is reported in the preprint. | The final publisher page reports a 10.5% average accuracy gain and 30–50% of training data, whereas preprint v1 reports a different 6.51% figure. Preprint “average accuracy” is mean class recall; participant-cluster uncertainty NR; default code seed 42. | [Official author code](https://github.com/ABan147/har-doremi/tree/0af2711d0bb3776bacd539fb5c44ae4c765c85c0) states MIT in README, but no LICENSE file was verified; preprint reports approximately 1.3 M parameters and RTX 3090 use. The public repository does not by itself expose the complete final-version evaluation path. | **Not comparable:** its multi-source/pretraining budget, datasets, ontology, rate/window, aggregation, and uncertainty differ. Do not combine the publisher final headline with v1 preprint protocol details; the final exact tuple remains unverified. |
| **IMU–video OOD HAR** — MLHC 2025, PMLR 298, [official proceedings paper](https://proceedings.mlr.press/v298/cheshmi25a.html) | Pretraining on Ego4D (about 815 h paired head IMU/video) or MMEA (10 people, 32 classes, 30.4 h); downstream private Parkinson dataset has four people, about four hours, and five classes. | Downstream chest accelerometer+gyroscope, 50 Hz; median filter kernel 5, z-score normalization, 250-sample/5-second non-overlap windows; walk/turn/bend/stand/sit. | The zero-shot procedure selects labelled, class-specific video prototypes from the downstream dataset and compares target IMU embeddings with them. Few-shot settings use 10/20/50/100 labelled target windows per class and a fixed 20-window-per-class held-out set. These are target-domain video/label or sensor-label supervision budgets, not sensor-only source training. | Balanced accuracy; zero-shot F1/MRR/Recall@k. Zero-shot estimates use five repetitions of class-stratified 80% resampling; few-shot experiments use five repeated labelled-set draws. The meaning of reported ± dispersion and participant-level exclusivity/inference is NR. | [Official code](https://github.com/scheshmi/IMU-Video-OOD-HAR/tree/8bb75af52c8698e6eb3123c529f6b5f3dd8ad76e), no software license verified; paper reports two RTX 4090 GPUs. | **Not comparable:** private target, N=4, target-domain class prototypes or labels, extra video supervision, different placement/window/ontology, metric, and uncertainty. It is a useful modality-budget reference, not a matched baseline. |

### Interpretation of recent-method results

1. HARMamba is a strong compact sequence-model reference, but its weighted sample-level F1 and mixed split conventions do not answer participant-generalization under fixed-class participant macro-F1.
2. LanHAR and HAR-DoReMi must not be assigned one target-access category. LanHAR's manuscript is ambiguous and its released implementation uses selected-target signals plus selected-target labels for checkpoint selection. HAR-DoReMi preprint v1 instead declares source-only, multi-source domain generalization to an unseen target. Neither matches the repository endpoint's datasets, ontology, preprocessing, model-selection budget, or metric contract.
3. The MLHC IMU–video work adds a powerful video/pretraining resource and uses labelled, class-specific downstream video prototypes even in its “zero-shot” condition; its private target has only four participants. It is neither a sensor-only nor participant-scale matched control.
4. None of the official code audits established the repository's three-seed, equal-participant, paired-cluster-bootstrap contract.
5. Differences in public-paper and released-code metric details are material. A paper label such as “F1” does not establish macro, weighted, participant-balanced, or fixed-class aggregation.

## Compact comparison matrix against our endpoint

| Candidate comparison | Same bytes/version | Same people/split | Same ontology | Same placement/channels/gravity | Same preprocessing/window | Same supervision/reset | Same metric/uncertainty | Numerically comparable? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FoG-STAR provider detector vs corrected FoG three-class HAR | Yes for v3 data only | No | No | No | No | No | No | **No** |
| IMU-HAR-IL provider RF baseline vs project repetition suite | Yes only if collection 74700/v1 receipt matches | No/unclear | No | No | No | No | No | **No** |
| HAR-PMD provider benchmark vs project five-class stress suite | Yes only if v2 archive matches | No | No | No | No | No | No | **No** |
| Sole-HARmony official benchmark vs retained interrupted 12-participant/24-session camera-oracle materialization (no score) | Not established (project pins v1; current official record is v2) | No | No | No | No | No | No | **No** |
| WearGait-PD or full Parkinson@Home vs repository | No local authorized bytes | No | No | No | No | No | No | **Blocked; no score** |
| HARMamba | No | No | No | No | No | No | No | **No** |
| LanHAR | No | No | No | No | No | No | No | **No** |
| HAR-DoReMi | No | No | No | No | No | No | No | **No** |
| MLHC IMU–video OOD HAR | No/private target | No | No | No | No | No | No | **No** |

This table deliberately does not assign partial “comparability scores.” A single scientifically material mismatch is sufficient to prohibit a before/after or SOTA inference.

## Licensing, redistribution, and access findings

| Evidence object | Verified terms | Publication implication |
|---|---|---|
| FoG-STAR data | CC BY 4.0 | Attribute the dataset and record exact v3 file receipt. No software license was verified for the detector repository, so do not assume code redistribution rights from the data license. |
| IMU-HAR-IL data | CC BY-NC 4.0 | Non-commercial restriction must be explicit in data/model cards and Zenodo/release metadata. Repository Apache licensing does not relicense these data, derived predictions, or weights. Official code is MIT. |
| HAR-PMD data | CC BY 4.0 | Attribute dataset. No root software license was verified for official project code; do not infer one. |
| Sole-HARmony data | CC BY 4.0 for Zenodo data | Dataset terms are separate from the journal article's publisher license. No root code license was verified. Freeze v1/v2 explicitly before any redistribution or derived-artifact statement. |
| WearGait-PD | CC BY 4.0 plus Synapse account/terms/code-of-conduct/pledge workflow | An autonomous agent must not accept agreements or bypass access controls. Treat files as unavailable until the authorized user completes access and a receipt is captured. |
| Parkinson@Home public tremor subset | CC0 | Open tremor data do not authorize or substitute for request-gated full activity data. Official tremor code is Apache-2.0. |
| HARMamba / LanHAR code | Apache-2.0 | Reuse still requires attribution/notices and compatibility review; their benchmark numbers remain non-comparable. |
| HAR-DoReMi / MLHC code | No LICENSE file verified at audited commits (HAR-DoReMi README states MIT) | Do not redistribute or incorporate code on a README claim alone without confirming legally operative terms. |

Predictions and trained weights may expose participant-derived or dataset-derived information and are not automatically covered by this repository's software license. Release review must use each provider's terms and the specific derivative artifact, not only the repository's `LICENSE` file.

## What can and cannot be claimed

### Supported only after corrected artifacts validate

- Performance **under the repository's named, frozen protocol**, with the exact dataset release/hash, participant set, channels, window contract, seeds, and participant-level uncertainty stated beside the number.
- Matched within-repository contrasts only when all comparison-tuple fields match and predictions support paired participant analysis.
- Development or stress-test findings on FoG-STAR, IMU-HAR-IL, and HAR-PMD, with evidence roles named.
- Retained Sole-HARmony materialization coverage and interruption status, explicitly labelled camera-boundary oracle diagnostic with no score; an oracle-sensitivity result is not yet available.
- Negative findings, such as failure to outperform the strongest matched control or harm in the participant lower tail.

### Not supported

- “State of the art,” “SOTA,” or “outperforms published work” based on a higher number from a different split, class set, window, target-access budget, or aggregation.
- Treating the approximate 0.7514 InclusiveHAR endpoint and approximately 0.948 IMU-HAR-IL development endpoint as a before/after improvement; their tuples differ.
- Calling native gravity equivalent to derived gravity, or pooling those lanes.
- Calling multiple repetitions of the same 50 IMU-HAR-IL participants a larger independent N.
- Calling corrected FoG target-label-blind transfer independent confirmation; FoG-STAR informed development.
- Calling HAR-PMD evidence clinical validation, or using its merged “still” class to validate sitting-versus-standing discrimination.
- Calling camera-bout reset deployable.
- Reporting WearGait-PD or full Parkinson@Home results without authorized, hashed data receipts.
- Using structurally valid legacy FoG artifacts as proof of semantic validity after the label-dependent preprocessing defect was identified.

## Publication table rules

Every important table should include an evidence-status field with one of: **validated**, **provisional**, **superseded**, **diagnostic**, **blocked**, or **confirmatory**. “Validated” means the artifact and scientific protocol both pass; it does not mean independent confirmation.

For literature rows:

- preserve the authors' metric name and aggregation exactly;
- use `NR` for unreported participant splits, uncertainty, seeds, hashes, or license rather than inferring them;
- never convert accuracy, balanced accuracy, weighted F1, and fixed-class participant macro-F1 into a shared ranking;
- list target-label, unlabeled-target, personalization, and external-modality access explicitly;
- place a visible **not comparable** marker whenever the exact tuple does not match;
- do not copy a result from one version of a preprint while citing a materially changed publisher version as if the methods were identical.

For repository rows, link each claim to the exact create-only run directory, frozen configuration, source-input manifest, result self-hash, validation output, predictions, and clean `git_at_launch` receipt. A literature citation alone is not evidence for a repository result.

## Primary-source search and verification log

Searches were performed on 2026-09-05 and refreshed on 2026-09-06. Query families included the exact dataset or method name combined with `dataset`, `Zenodo`, `Scientific Data`, `DOI`, `paper`, `official code`, `GitHub`, `license`, `participants`, `sampling rate`, `window`, `split`, and `metric`. Candidate facts were retained only when an official record, publisher/proceedings page, author manuscript, or official repository supported them. Search-result snippets, commercial summaries, blogs, and citation-aggregator summaries were not used as evidence.

### Dataset sources

- FoG-STAR: [Zenodo v3](https://zenodo.org/records/17838806); [Scientific Data descriptor](https://www.nature.com/articles/s41597-026-06645-1); [official detector code, commit `2b2277e`](https://github.com/Lu1g1n0/FoG-detection-using-a-single-ankle-mounted-sensor/tree/2b2277eda53051142cc8198a8e0c0a3f61f2de0b).
- IMU-HAR-IL: [CSIRO collection 74700/v1](https://data.csiro.au/collection/csiro:74700); [provider version service](https://data.csiro.au/dap/ws/v2/collections/74700/versions); [author manuscript v2](https://arxiv.org/html/2608.07502v2); [official code, commit `7d8244d`](https://github.com/mmsandhu/IMU-HAR-IL/tree/7d8244db71f5f15e094664df1ebb02315769f0ba).
- HAR-PMD: [Zenodo v2](https://zenodo.org/records/7939223); [Scientific Data descriptor](https://www.nature.com/articles/s41597-025-06527-y); [official code, commit `88fd897`](https://github.com/HanyangHCILab/HAR-handicaps/tree/88fd8972e242776d0b148775e8385ec9a274174e).
- Sole-HARmony: [Zenodo v1](https://zenodo.org/records/19242395); [Zenodo v2](https://zenodo.org/records/21477296); [Scientific Data article](https://www.nature.com/articles/s41597-026-08172-5); [official code, commit `3c57f15`](https://github.com/wearable-robotic-systems-lab/Sole-Harmony/tree/3c57f159fdc90ac91e6d9ab4ce25e4bfd384d0fd).
- WearGait-PD: [Scientific Data descriptor](https://www.nature.com/articles/s41597-026-06806-2); [official Synapse project](https://www.synapse.org/Synapse:syn52540892); [dataset DOI](https://doi.org/10.7303/syn52540892).
- Parkinson@Home: [original JMIR study](https://www.jmir.org/2020/10/e19068); [public tremor subset DOI](https://doi.org/10.34973/2xxa-g520); [npj Parkinson's Disease tremor paper](https://www.nature.com/articles/s41531-025-01056-2); [official code, commit `92a8140`](https://github.com/biomarkersParkinson/pdathome_tremor/tree/92a814024846e68a67e6fc5f6fe8e52f74fc555e).

### Method sources

- HARMamba: [IEEE publisher page](https://ieeexplore.ieee.org/document/10683697/); [author manuscript](https://arxiv.org/abs/2403.20183); [official code, commit `3afb0dc`](https://github.com/dianoDouble/HARMamba/tree/3afb0dced3c66702d698af354d7ec7e753e772ad).
- LanHAR: [ACM DOI](https://doi.org/10.1145/3770652); [author manuscript v4](https://arxiv.org/abs/2410.00003); [official code, commit `1fe98fa`](https://github.com/DASHLab/LanHAR/tree/1fe98fa3ecf847ff185d26d37c8a0d99b28cfb29).
- HAR-DoReMi: [ScienceDirect publisher page](https://www.sciencedirect.com/science/article/pii/S0925231226013603); [author preprint v1](https://arxiv.org/abs/2503.13542); [official code, commit `0af2711`](https://github.com/ABan147/har-doremi/tree/0af2711d0bb3776bacd539fb5c44ae4c765c85c0).
- IMU–video OOD HAR: [PMLR proceedings page](https://proceedings.mlr.press/v298/cheshmi25a.html); [official code, commit `8bb75af`](https://github.com/scheshmi/IMU-Video-OOD-HAR/tree/8bb75af52c8698e6eb3123c529f6b5f3dd8ad76e).

## Final comparability conclusion

There is no verified published result in this audit whose full comparison tuple matches the repository's corrected external-HAR protocol. The literature establishes relevant datasets, architectures, cross-domain strategies, and efficiency references; it does **not** supply a valid numerical SOTA threshold for this endpoint.

Any manuscript claim must therefore remain protocol-local and evidence-role-aware. A SOTA claim would require competitive current controls rerun on the identical frozen bytes and protocol, with the same supervision budget and participant-level uncertainty, and preferably one-time success on an untouched confirmatory cohort. Those conditions are not currently met.
