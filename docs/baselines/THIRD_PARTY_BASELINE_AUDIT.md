# Third-party baseline provenance and integration audit

Status: source/provenance audit, 2026-08-23, with a later execution addendum.
At the original audit gate, no model in this document had been trained on or
evaluated against the locked InclusiveHAR target cohort. Repository inspection
used preserved read-only clones under
`.audit/baseline-source-audit-20260823/`; those clones are evidence, not vendored
project code.

Post-audit implementation addendum (2026-08-24): a source-only CCIL loss
adaptation and a locally implemented boundary-safe BPD protocol adaptation were
added behind the post-confirmatory design in
`configs/experiments/ccil_bpd_postconfirmatory_v1.yaml`. A later create-only CUDA
execution and aggregate completed. This does not change the audit's
`faithful local implementation = false` decision. The implementation copies no
BPD source, never uses the audited BPD trainer, selects and calibrates only on
source participants, and accesses target data only after a durable source-stage
lock through consumed opening-1 cache lineage. See
`docs/CCIL_BPD_POSTCONFIRMATORY.md` for the claim labels, completed results, and
run gates.

The completed post-confirmatory aggregate is
`results/postconfirmatory/ccil_bpd_v1/ccil_bpd_postconfirmatory_aggregate.json`,
record SHA-256
`c7b27e2a6d5ddf94efcd2c2064cb84aecfc70dfe3d4f38539660c3479128c180`.
The qualified CCIL adaptation obtained target mean/worst/lower-decile macro-F1
0.6896/0.2724/0.3459. Its mean was 0.0119 above compact ERM, but both tails were
slightly lower and both family-adjusted paired p-values were 0.2109. The
qualified BPD adaptation obtained 0.5710/0.2471/0.2751. These are descriptive
adaptation results, not official-faithful reproductions or additions to the
locked primary ranking.

This audit answers four separate questions for every requested baseline:

1. Is there a primary paper and an author-linked or paper-linked repository?
2. What exact repository revision and software license were observed?
3. What temporal and sensor-channel interface did the publication/code assume?
4. Can the implementation be reused without weakening the locked protocol or
   making an unsupported reproduction claim?

Paper access and a public GitHub repository do not by themselves grant permission
to copy software. `NOASSERTION` below means that no software license file or
equivalent repository grant was found at the audited revision. It does not infer
an alternative license. An arXiv manuscript license is also not a software license
for a linked repository.

## Decision summary

| Work | Primary source | Official/paper-linked code audited | Software license at revision | Published or repository input assumption most relevant here | Local decision |
|---|---|---|---|---|---|
| TinyHAR | [ISWC 2022 paper](https://doi.org/10.1145/3544794.3558467) | `teco-kit/ISWC22-HAR` at `b84b89d09f6914fe93e82cde423e294042da2741` | `NOASSERTION` | Official HAPT config: 6 channels, 50 Hz, 128 samples; no UCI entry in the original repo | Block source reuse; no local proxy may be called TinyHAR |
| TinierHAR | [UbiComp/ISWC 2025 paper](https://doi.org/10.1145/3715071.3750410) | `zhaxidele/TinierHAR` at `f2f1bbd7305689c374fb38cbb7ce853e7cefbb3f` | `NOASSERTION` | Paper: all datasets, including 9-channel UCI, use 4 s / 2 s overlap; repo UCI config instead says 50 Hz and 2.56 s | Block source reuse; architecture/count ambiguity must be resolved before any clean-room adaptation |
| HARMamba | [IEEE IoT Journal paper](https://doi.org/10.1109/JIOT.2024.3463405) | `dianoDouble/HARMamba` at `3afb0dced3c66702d698af354d7ec7e753e772ad` | Apache-2.0 | Paper UCI: 9 channels, 50 Hz, 128 samples, 50% overlap; code defaults: 9 channels and 512 samples | License-compatible, but blocked pending a protocol-safe, runtime-validated adapter |
| CCIL | [AAAI 2025 paper](https://doi.org/10.1609/aaai.v39i1.32077) | No official repository located | No software release to license | Cross-person interfaces vary by dataset; cross-dataset interface is 6 channels and 50 samples | Equations 2–6 only are implemented as a paper-derived loss, never as official CCIL code |
| BPD | [IMWUT 2022 paper](https://doi.org/10.1145/3517252) | `Jie-su/BPD` at `8b2338927c118d1daa5c602d48b6ae5156dd5966` | Apache-2.0 | Paper: 168 samples, 50% overlap, dataset-specific channels/rates; code default stride is 32 | Official trainer remains blocked; a post-audit local boundary-safe adaptation completed post-confirmatory but is not faithful |
| CMD-HAR | [arXiv:2503.21843v4](https://arxiv.org/abs/2503.21843) | No official repository located | No software release to license | Paper table: UCI is 9 channels, 50 Hz, 128 samples | Related-work comparator only; do not invent an implementation |
| BenchHAR | [arXiv:2605.08296](https://arxiv.org/abs/2605.08296) | `saiketa/HAR-Bench` at `358a377929b1b9c0a2cefc417c67f56d15d4d11c` | `NOASSERTION` | 6 s non-overlapping windows resampled to 20 Hz: 120×6 or accelerometer-only 120×3 | Block source/adapter reuse; use its findings only to predeclare SSL candidates |
| SimMTM | [NeurIPS 2023 paper](https://arxiv.org/abs/2302.00861) | `thuml/SimMTM` at `169513bef74fb676e48d98a0e30f8823793f691c` | `NOASSERTION` | Generic time-series upstream; the 120×6 inertial interface is a BenchHAR adaptation | Block code reuse; record as the BenchHAR cross-subject candidate, not a completed local baseline |
| FOCAL | [NeurIPS 2023 paper](https://arxiv.org/abs/2310.20071) | `tomoyoshki/focal` at `f6a989eef42e7590aacc0c025e5aed875c6025c9` | MIT | Official code targets paired acoustic/seismic modalities, not 6-channel smartphone IMU | Eligible for a new, clearly labeled inertial adapter; no such adapter is claimed here |
| LITEWAY | [arXiv:2608.09421v1](https://arxiv.org/abs/2608.09421) | `dominique-nshimyimana/liteway` at `982100053db3a81b10a10d225711829473ac1f3d` | `NOASSERTION` | UCI: 9 channels, 50 Hz, 128 samples, 50% overlap | Current lightweight related work; block code reuse and do not rename a local compact model |

`faithful local implementation = false` for every row at this gate. That status is
also enforced in `src/inclusive_shift_har/models/third_party_specs.py`.

## Exact repository evidence

| Local audit key | Canonical repository URL | Branch at audit | Commit timestamp | Root license evidence |
|---|---|---|---|---|
| `tinyhar` | <https://github.com/teco-kit/ISWC22-HAR> | `main` | `2022-09-15T14:27:52+02:00` | No `LICENSE`, `LICENCE`, `COPYING`, or `NOTICE` path found in the Git tree |
| `tinierhar` | <https://github.com/zhaxidele/TinierHAR> | `main` | `2025-10-12T09:33:19+02:00` | No license path found |
| `harmamba` | <https://github.com/dianoDouble/HARMamba> | `main` | `2025-03-10T22:31:54+08:00` | Root `LICENSE`, Apache-2.0; SHA-256 `B208C52B00A2DF2C8F4C3298C34407BF2BFE409968213372251C91FCC737A1A5` |
| `bpd` | <https://github.com/Jie-su/BPD> | `main` | `2025-09-20T23:13:41+08:00` | Root `LICENSE`, Apache-2.0; SHA-256 `C71D239DF91726FC519C6EB72D318EC65820627232B2F796219E87DCF35D0AB4` |
| `benchhar` | <https://github.com/saiketa/HAR-Bench> | `master` | `2026-05-23T10:34:09+08:00` | No license path found |
| `simmtm` | <https://github.com/thuml/SimMTM> | `main` | `2024-05-06T19:38:44+08:00` | No license path found |
| `focal` | <https://github.com/tomoyoshki/focal> | `main` | `2024-01-21T18:39:33-06:00` | Root `LICENSE`, MIT; SHA-256 `5A8EA32BE1A3D9C5A914E8147BF7A4AE91CDB715927987E6CC14E1A48A591AE1` |
| `liteway` | <https://github.com/dominique-nshimyimana/liteway> | `main` | `2026-08-11T13:03:46+02:00` | No license path found |

The audit clones were clean immediately after acquisition. Exact structured values
are in `third_party_baseline_provenance.json`.

## TinyHAR

The paper architecture combines per-channel temporal convolution, cross-channel
interaction/aggregation, a recurrent temporal stage, and learned temporal
aggregation. The official configuration uses `filter_num: 20` and the model
accepts `[batch, 1, time, channels]`. Its dataset configurations are heterogeneous:
HAPT is 50 Hz, 6 channels, and 2.56 s; other released settings range from 20 to
100 Hz and 3 to 77 channels. The original repository does not contain a UCI-HAR
configuration.

The repository has no software license. In addition, its shared loader forms
50%-overlap training windows and randomly partitions those window indices into
training and validation. Adjacent overlapping windows can therefore share raw
samples across development partitions. The source-independent outer test logic
does not make that validation procedure compatible with this benchmark's strict
raw-sample separation rule.

Decision: preserve the citation and architecture description, but copy no source.
Locally designed compact residual networks must retain neutral names and cannot be
reported as TinyHAR reproductions.

## TinierHAR

The paper describes residual depthwise-separable temporal convolutions, max-pool
downsampling in the first two blocks, a bidirectional GRU, learned softmax temporal
aggregation, and a linear classifier. It reports five seeds, AdamW, up to 150
epochs, learning-rate reduction after seven unimproved epochs, and early stopping
patience 15.

There are two reproducibility conflicts requiring explicit resolution:

- The paper says every dataset is segmented into 4-second windows with 2-second
  overlap and lists UCI as 9 channels at 50 Hz. The repository's UCI config says
  2.56 seconds (128 samples), not the paper's 200 samples.
- The configuration sets `nb_conv_blocks: 4`, while `models/TinierHAR.py` creates
  two initial pooled blocks and then four additional blocks. The paper's scaling
  notation describes `M` residual separable blocks besides the first two, but the
  exact result-to-code configuration still needs an end-to-end parameter/MAC check
  before making a faithful claim.

The repository also inherits the random split of overlapping source windows for
training and validation. It has no software license.

Decision: block source reuse. A future clean-room implementation would be a
"TinierHAR paper adaptation" until shapes, parameter counts, MACs, and a published
reference setting are all independently matched; it must use the local grouped,
non-overlapping validation protocol rather than the repository loader.

## HARMamba

HARMamba is the only requested compact architecture here with a permissively
licensed official repository. The paper's UCI table gives 50 Hz, 128 samples and
9 channels and describes 50% window overlap. It then partitions the continuous
data 0.7/0.1/0.2 rather than reporting a subject-exclusive UCI protocol. The table
also states 20 UCI subjects, conflicting with the dataset's 30 participants, so
its reported UCI metrics are not directly comparable to the locked benchmark.

The audited implementation is not ready to import as a faithful baseline:

- `models_mamba.py` defaults to `seq_size=512`, patch 16 and `c_in=9`, while the
  entry point constructs `HARMamba()` without resolving dataset-specific model
  arguments.
- `requirements.txt` references hard-coded local Linux wheel paths for Mamba and
  causal-conv1d and targets Python 3.9 / PyTorch 2.1.1 + CUDA 11.8. That stack is
  not the current Windows/Python 3.11/PyTorch CUDA environment.
- The training loop evaluates the test/validation loader every epoch and writes
  `best_checkpoint.pth` when its accuracy improves. It cannot be used for a sealed
  confirmatory target.
- The forward path includes NumPy squeezing of a Torch tensor and needs a device,
  gradient, and shape regression test before it can be trusted.

Decision: Apache-2.0 makes an isolated architecture adapter legally possible, but
the official trainer is prohibited. Integration remains blocked until an adapter
passes `[batch,128,6]` shape tests, CPU/CUDA forward/backward tests, parameter and
MAC reconciliation, checkpoint reconstruction, and source-only selection. Any
result would initially be labelled an "official-architecture protocol adaptation."

## CCIL

The AAAI paper and author preprint were inspected. The official article contains
no repository link, and current searches did not locate an author-owned release.
Third-party "request code" pages are not official implementations. Consequently,
there is no external source to copy and no software license to record.

The reproducible part of the paper is the concept-mean similarity regularizer:

- for penultimate feature `z` and classifier weight `W`, construct
  `M[d,c] = z[d] * W[d,c]` across all output classes;
- maintain an activity-class-specific mean concept matrix using an exponential
  moving update;
- penalize squared Frobenius distance between each example and the mean matrix of
  its activity class; and
- add the regularizer to cross-entropy with a source-selected coefficient.

The local module `CCILPaperConceptMeanLoss` implements only those published
equations. It resolves two unspecified operational details explicitly: a class is
initialized from its first source-training batch, and the current batch is folded
into the detached EMA before its loss is evaluated. Its buffers and update counts
are checkpointable. The module takes only activity labels, features, and classifier
weights—never disability, device, or participant identity.

Validation limits are material:

- this is not official CCIL code and not a reproduction of the paper's two-block
  Conv/MaxPool/BatchNorm backbone or its complete training pipeline;
- the paper uses different shapes by task (DSADS cross-person `45×1×125`, PAMAP2
  `27×1×200`, USC-HAD `6×1×200`, cross-dataset `6×1×50`) and does not run UCI in
  its cross-person experiment;
- the paper's source development split includes random sample-level 8:2
  train/validation in its described implementation, whereas local evaluation must
  remain participant-grouped and raw-sample-exclusive; and
- alpha and EMA weight remain unselected source-development hyperparameters. No
  default has been silently promoted to a locked value.

Historical gate decision: the loss was initially disabled until grouped
source-only tuning and checkpoint reconstruction passed. The later qualified
post-confirmatory route passed those local gates and completed, while every
artifact retained `paper_derived_not_official` provenance. This does not convert
the adaptation into official CCIL code or a faithful reproduction.

## BPD and CMD-HAR disentanglement predecessors

### BPD

The BPD paper factorizes activity-related and redundant behavior representations
using generators/disentanglers, reconstruction, activity classification,
independence estimation and confusion objectives. It evaluates PAMAP2, MHEALTH,
DSADS and GOTOV using 168-sample windows with 50% overlap. Because sampling rates
differ, those windows cover 1.68, 3.36, 6.72 and 2.02 seconds respectively. This is
not the same temporal interface as 128 samples at 50 Hz.

Although the official code is Apache-2.0, its end-to-end pipeline violates the
locked protocol:

- participant arrays are concatenated before the generic sliding-window dataset
  is created, so a window can cross a participant boundary;
- the paper says 50% overlap (stride 84), but `DataLoader.initialize` defaults to
  stride 32;
- the CLI has a `--win_len` argument, yet `dataset_read` does not pass it to
  `DataLoader.initialize`, so the effective code path uses 168 regardless of the
  CLI value; and
- `Solver.test()` evaluates the held-out target each epoch and saves the model
  when target macro-F1 improves.

Decision: do not run or port the official trainer. The later local safe-window,
source-only implementation completed post-confirmatory and is called a BPD
protocol adaptation, never an official-faithful reproduction. Direct comparison
with MoRe-HAR must distinguish BPD's learned activity/redundant decomposition and
MINE-based independence objective from MoRe-HAR's content/realization descriptor
hypothesis.

### CMD-HAR

CMD-HAR is a cross-modal disentanglement predecessor, not an available baseline.
The current paper version lists OPPORTUNITY, PAMAP2, WISDM, UCI-HAR, UniMiB-SHAR
and USC-HAD; its UCI table uses 9 channels, 50 Hz and 128 samples. No official code
or software license was linked in the paper or located in the audit. Some dataset
and split descriptions are not sufficiently precise to reconstruct a
participant-exclusive benchmark from the paper alone.

Decision: cite it when distinguishing cross-modal disentanglement from
motion-realization factorization. Do not infer missing implementation details or
manufacture a local CMD-HAR result.

## BenchHAR SSL candidate gate

BenchHAR is valuable evidence for candidate selection, but its data interface is a
separate benchmark: continuous signals are divided into non-overlapping 6-second
windows, resampled to 20 Hz (120 time points), and instance-normalized per window.
It tests accelerometer-only 3-channel and accelerometer-plus-gyroscope 6-channel
modalities with dataset-level five-fold generalization and separate cross-subject,
location and device analyses. This must not be silently mixed with the locked
50 Hz, 128-sample signal interface.

The paper compares BioBankSSL, LIMU-BERT, CRT, TS-TCC, TS2Vec, FOCAL, SimMTM and
CrossHAR. Its findings create two different candidate definitions:

- FOCAL has the strongest average method-level cross-dataset result reported in
  the paper's aggregate analysis, and its upstream repository is MIT licensed.
- SimMTM is the strongest reported cross-subject candidate for the combined
  accelerometer/gyroscope setting, which is more aligned with the present
  participant-shift question.

These are not interchangeable definitions of "best." The choice must be
predeclared by endpoint. For a six-channel participant-shift SSL experiment,
SimMTM is the relevance-based candidate. For broad cross-dataset mean performance,
FOCAL is the evidence-based candidate.

Neither the HAR-Bench repository nor the upstream SimMTM repository includes a
software license at the audited commits, so their code cannot be copied. FOCAL's
upstream MIT code is reusable, but it implements acoustic/seismic multimodal
learning rather than a smartphone inertial adapter. HAR-Bench's FOCAL adapter is
part of the unlicensed HAR-Bench repository and is not inherited through the
upstream MIT grant.

Decision: no BenchHAR SSL baseline is locally complete. A future FOCAL inertial
adapter is legally eligible but must be independently implemented and validated.
SimMTM remains blocked until authors provide licensing or a clean-room design is
specified from the paper and validated without copying source.

### Execution-status addendum — 2026-08-24

The preceding source audit is preserved as a time-scoped provenance record. A
later feasibility check at commit
`ad657d3aec3e2476ce7a1e8226bb2fd3a99bd721` found the following local state:

- clean, ignored audit checkouts exist for BenchHAR, SimMTM, and FOCAL at the
  revisions recorded above; they are evidence copies, not integrated package
  dependencies or validated benchmark adapters;
- no `.pt`, `.pth`, `.ckpt`, `.safetensors`, or `.onnx` checkpoint exists in any
  of those three audit checkouts;
- the main package contains no BenchHAR, SimMTM, FOCAL, BioBankSSL, LIMU-BERT,
  CRT, TS-TCC, TS2Vec, or CrossHAR model/experiment configuration or executable
  comparison route; and
- no compatible licensed wearable-foundation checkpoint, preprocessing adapter,
  frozen linear-probe route, parameter-efficient adaptation route, full-fine-tune
  route, or equal-budget source-only configuration exists locally.

The empirical status of every SSL/foundation comparison is therefore **not
run**. The combined comparison gate is **blocked** because no candidate has
cleared licence, temporal-interface, checkpoint, source-only-selection, and
artifact-lineage requirements together. BenchHAR and SimMTM are specifically
licence-blocked; their 20 Hz/120-sample interface is also not interchangeable
with the locked 50 Hz/128-sample interface. FOCAL is not licence-blocked, but its
MIT upstream is acoustic/seismic and the required smartphone-inertial adapter has
not been implemented or validated. These are documented omissions, not negative
results.

The adjacent cross-source extension has two different statuses and must not be
collapsed into one claim. Exact-label, all-cohort UCI→InclusiveHAR
classification is **blocked and not run** because sitting is the only exact
shared class; standing is provisional and ordinary UCI walking is not wheelchair
propulsion. Adapted-label or representation-transfer pretraining is **not
implemented and not run**, rather than scientifically ruled out. The completed
UCI-native grouped reproduction is source-development evidence only: there is no
all-source encoder refit, encoder-transfer/head-replacement pipeline, or
InclusiveHAR transfer result.

Claim-safe manuscript text is: “Cross-source pretraining and SSL/foundation
comparisons were not evaluated. Their absence reflects the locked exact-label
ontology and unresolved implementation, licensing, interface, and checkpoint
gates; it is not evidence of empirical failure.” Any later target-side comparison
is post-confirmatory because the locked target opening is already consumed.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

## LITEWAY current-date addition

LITEWAY v1 was submitted on 2026-08-10 and is marked accepted at UbiComp/ISWC
2026. It is highly relevant current lightweight-HAR work because it directly
compares TinyHAR and TinierHAR and replaces recurrence with a fully convolutional
stack: depthwise/residual downsampling, structured convolutional temporal
modeling, convolution-attention pooling, and a single linear classifier. It
evaluates 16 datasets, five seeds, participant-independent outer protocols, and
reports macro-F1, parameters, MACs, latency/energy evidence.

The UCI setup is 9 channels at 50 Hz with a 2.56-second/128-sample window and 50%
overlap. The official model consumes `[batch,1,time,channels]`. That is temporally
close to the benchmark but not channel-equivalent to the primary 6-channel
user-acceleration/gyroscope interface. Its repository also inherits random
training/validation splitting of overlapping source windows, so its development
split is not raw-sample-exclusive even though the outer test participants are
held out.

The paper is distributed on arXiv under CC BY-NC-ND 4.0, while the linked software
repository contains no software license at commit
`982100053db3a81b10a10d225711829473ac1f3d`. The manuscript license does not grant
permission to copy repository code.

Decision: add LITEWAY to related work and the future compact-baseline priority
list, but do not integrate its source and do not relabel a generic local model as
LITEWAY. A later author license could reopen the software gate; a future adaptation
would still need six-channel, participant-exclusive boundary-audited validation
and parameter/MAC checks.

## Integration and claim rules

The following rules are binding for experiments built from this audit:

1. Local `compact_residual_*` models are controlled baselines, not TinyHAR,
   TinierHAR, CNN-HAR, HARMamba, or LITEWAY.
2. A permissive license is necessary but not sufficient. The adapter must also
   pass shape, gradient, parameter/MAC, checkpoint, split, and source-only
   selection checks.
3. Published data loaders and trainers are never reused when they split
   overlapping windows, cross participant/trial boundaries, or select on a target.
4. Any changed interface or protocol is labelled an adaptation. Published metrics
   are context, not validation of the local adapter.
5. No missing official implementation is reverse-engineered and called faithful.
6. CCIL's local loss remains explicitly paper-derived. It is enabled only by the
   post-confirmatory extension that selects its coefficients using grouped source
   participants; it is not part of the locked primary inventory.
7. The locked InclusiveHAR target remains unopened for baseline choice,
   hyperparameter tuning, normalization, calibration, or checkpoint selection.

## Reopening criteria

| Blocked item | Evidence required to reopen |
|---|---|
| TinyHAR, TinierHAR, BenchHAR, SimMTM, LITEWAY source | An explicit software license covering the audited code/revision, or a separately reviewed clean-room implementation |
| HARMamba | Portable dependency path, verified six-channel/128-sample adapter, forward/backward tests on CPU and CUDA, parameter/MAC reconciliation, and source-only trainer |
| CCIL full reproduction | Author release with license plus architecture/config reconciliation; otherwise retain paper-derived-loss status |
| BPD | Boundary-safe per-participant/per-trial window adapter, stride/config reconciliation, target-blind training, and published-component validation |
| CMD-HAR | Official licensed implementation or a predeclared clean-room paper adaptation with all missing choices exposed |
| FOCAL HAR | Independently implemented inertial modality adapter, license notices, and source-only validation at the locked temporal interface |
