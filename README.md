# InclusiveShift-HAR

*An Auditable Participant-Exclusive Benchmark for Physical-Ability Generalization in Smartphone Activity Recognition*

InclusiveShift-HAR studies how smartphone inertial HAR systems trained on conventional participant populations transfer to people whose activities may be physically realized differently, including users of assistive devices. The repository separates legacy coursework, source-only development, and a single locked target evaluation with immutable lineage.

The original working title said "leakage-safe." InclusiveHAR v4 does not release timestamps or trial identifiers, so hidden joins inside released participant-activity blocks cannot be ruled out. The benchmark is participant-exclusive and raw-row-disjoint under the released schema, but it is not trial-safe. That limitation is part of the result, not a footnote.

## Main result

The one-time zero-shot target evaluation is complete. It used 10 held-out target participants, 807 non-overlapping released-block windows, three functional-core classes, 20 predeclared model/ablation configurations, and five frozen seeds per configuration.

| Result | Mean participant macro-F1 | Worst participant | Lower decile |
|---|---:|---:|---:|
| Compact DANN | 0.6808 | 0.2699 | 0.3559 |
| Compact CORAL | 0.6808 | 0.2741 | 0.3670 |
| MoRe-HAR full | 0.6353 | 0.2584 | 0.2660 |

Compact DANN had the highest target mean, while the legacy joint CNN/BiLSTM had the highest worst-participant value (0.2943) and CORAL had the highest lower-decile value (0.3670). MoRe-HAR did not improve either the mean or the required lower-tail endpoints against the strongest eligible baselines. Its preregistered hypothesis is therefore **not supported**. This negative outcome is retained, and the auditable benchmark is the primary contribution.

The complete table, participant values, calibration metrics, AURC, per-class recall, comparisons, and self-hashed report are in [`results/confirmatory/zero_shot_v1/`](results/confirmatory/zero_shot_v1/). The 95% interval for compact DANN is [0.5391, 0.8093], reflecting substantial participant uncertainty. These results do not establish state of the art, fairness, clinical validity, or a causal disability effect.

## Evidence classes

| Evidence class | Meaning | Status |
|---|---|---|
| Legacy coursework | Saved UCI-HAR notebook outputs with random window validation and repeated official-test use. | Audited; permanently `legacy_exploratory_development_consumed`. |
| Corrected reproduction / development | Subject-grouped source development, tuning, ablations, and debugging. | Complete for the InclusiveHAR source suite; UCI official-train-only reproduction is a separate corrected track. |
| Locked confirmatory | Frozen source-only artifacts evaluated once on the sealed target cohort after all gates passed. | Opening 1 complete; no retry or second opening is permitted. |
| Post-confirmatory secondary | Few-person inclusion, sensor stress, efficiency, and descriptive follow-up after the primary opening. | Explicitly separate from the zero-shot confirmatory claim. |

The opening receipt is [`results/protocol/confirmatory_target_opening_1.json`](results/protocol/confirmatory_target_opening_1.json). The primary statistics record is [`participant_statistics.json`](results/confirmatory/zero_shot_v1/participant_statistics.json), and the human-readable table is [`model_summary_v1.md`](results/confirmatory/zero_shot_v1/model_summary_v1.md).

## Scientific scope

The research question is:

> How reliably do smartphone HAR systems trained on conventional participant populations generalize to people whose activities are physically realized differently, including users of assistive devices?

This is an observational ability-associated population-shift benchmark. Disability and assistive-device labels are not model inputs. The released `walking` activity for wheelchair users denotes manual propulsion and is preserved as a distinct physical realization within the functional mobility concept. Ramps are not relabelled as stairs, and jogging is not silently mapped to another locomotion class.

MoRe-HAR (Motion-Realization Factorized HAR) is a compact experimental hypothesis, not an assumed novelty claim. Its full objective combines classification, source-participant supervised contrastive alignment, clean/augmented consistency, measurable realization-descriptor prediction, cross-covariance factorization, and GroupDRO. The current evidence does not support the claim that this combination improves zero-shot ability-shift performance.

## Reproducibility anchors

- InclusiveHAR v4 split hash: `ccb6c3d1254c1464c48e412afb6f83e7942113299e853f89c77df1d6cfad131b`
- Source-window manifest hash: `1ad1ee3accaae5f2f93bb91ac0afa5ce583134ce1d882c3f08323b09026fa522`
- Target seal ID: `aecba05fa4a0fc4e4bbc135ac30944b686be19c0b6a2820c838e6c8b802bb29d`
- Frozen artifact-set hash: `e759b60f32b965e7ae3e5a994d919a08553c4958f9bdcf10d7497697f685dd51`
- Training code commit: `b4dc38fb9d5a0c17003221b61156ebc065395170`
- Target index record hash: `79434d8fbc136cb55e18fa980490e5aaa94a91fb3c823837cccf6137b026b5b9`
- Locked statistics record hash: `c7f20362598922223a8d72d927fba69445fca31cb3607ef9eff0b130211f2cbd`
- Publication report hash: `3afe0ceee9f97025d1adc5f59ab3528b512a3212385ef5344028850e9cb39c66`

The protocol tag is `protocol-v1.2.0`. Checkpoint and raw prediction arrays are intentionally outside normal Git history but are preserved locally and bound by the committed inventory and sidecar hashes.

## Data and protocol

The main sensing interface is three-axis user acceleration plus three-axis rotation rate at 50 Hz in 128-sample windows. Participants are partitioned before windowing. Source training may use stride 64; confirmatory validation and target evaluation use stride 128. Normalization is fitted on training participants only, calibration on source validation only, and all model selection occurred before the target opening.

The functional-core result uses mobility, sitting, and standing. Inclusive-native and cross-source ontology tracks are documented separately. See:

- [`docs/data/INCLUSIVEHAR_V4.md`](docs/data/INCLUSIVEHAR_V4.md)
- [`docs/data/UCI_HAR_V1.md`](docs/data/UCI_HAR_V1.md)
- [`docs/DATA_AUDIT.md`](docs/DATA_AUDIT.md)
- [`docs/LOCKED_PROTOCOL.md`](docs/LOCKED_PROTOCOL.md)
- [`docs/BENCHMARK_CARD.md`](docs/BENCHMARK_CARD.md)

Raw third-party data, the coursework ZIP, secrets, and large checkpoints are excluded from Git. Dataset manifests record official/versioned retrieval sources, hashes, sizes, licences, citations, and schemas; repository code does not relicense third-party data.

## Environment and commands

The project targets CPython 3.11 and uses `uv`. Neural training and inference in the reported suite used an NVIDIA RTX PRO 3000 Blackwell Laptop GPU with PyTorch 2.12.0+cu132. Recurrent models retained CUDA tensors while disabling cuDNN after preserved Windows cuDNN crash records; neural CPU fallback was not used. Classical scikit-learn models use their native CPU implementations, and XGBoost training used CUDA.

```powershell
uv sync --extra training-cuda --group research
uv run inclusive-shift-har validate-manifests
uv run inclusive-shift-har audit-splits --split-manifest results/protocol/splits/inclusivehar_v4_released_block_v1_2.json --json
uv run inclusive-shift-har validate-artifacts --artifact-root results --require-artifacts --json
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

The package also provides deterministic split/source-window builders, source-CV aggregation, a CUDA-only UCI source-fold runner, source/final experiment modules, one-time confirmatory evaluation code, locked participant statistics, and create-only publication reporting. CI uses synthetic fixtures and never downloads the research datasets.

## Repository map

```text
src/inclusive_shift_har/   package code
configs/                   dataset, ontology, protocol, model, and experiment configs
tests/                     synthetic leakage, lineage, training, and evaluation tests
manifests/                 immutable dataset provenance
legacy/                    lawful legacy audit and reconstructed metrics
results/                   committed small evidence, hashes, gates, and tables
docs/                      audits, cards, protocol, literature, ethics, and status
paper/                     paper-ready outline and result narrative
```

## Claim limits

This repository does not claim the first disability-related HAR study, the first HAR fairness study, the first disentangled HAR architecture, state-of-the-art performance, clinical validity, safety, causal disability effects, or publication acceptance. It does not treat a Transformer/Mamba swap or generic corruption benchmark as novelty. Failed runs, backend crashes, quarantines, deviations, and the unsupported MoRe-HAR hypothesis remain visible.

## Licensing and citation

Repository-authored code is Apache-2.0 licensed. Third-party datasets, coursework artifacts, papers, checkpoints, and other external materials retain their own terms. [`CITATION.cff`](CITATION.cff) and [`.zenodo.json`](.zenodo.json) prepare future release metadata; no DOI is claimed until a stable public release is approved.
