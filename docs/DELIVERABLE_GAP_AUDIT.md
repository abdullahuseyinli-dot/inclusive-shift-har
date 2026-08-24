# Staged deliverable gap audit

Snapshot basis: repository records and worktree observed on 2026-08-24. This is a
conformance audit, not a new acceptance gate. “Complete” below means evidence is
present for the stated scope; it does not erase recorded limitations. Ongoing
uncommitted work must be reassessed before release.

| Stage | Evidence present | Material gap or limitation |
|---|---|---|
| 0 — legacy preservation/audit | Source ZIP checksum, archive inventory, notebook-cell audit, reconstructed accuracy/macro-F1/confusion/NLL/Brier/ECE, locked legacy metrics and clear development-consumed label | Large source/checkpoint material correctly remains outside normal Git; no old UCI result can become confirmatory |
| 1 — literature/novelty | Current matrix, source registry, targeted post-InclusiveHAR search, narrowed no-“first” contribution | Novelty search is scoped, not proof; refresh bibliographic indexes immediately before submission |
| 2 — package/provenance | Python package, lockfile, manifests, dataset cards, tests, CLI, synthetic CI, immutable raw-data policy, and create-only official-manifest acquisition path | Final tracked-file, secret, license, and CI evidence must be rerun on the release commit; UCI license conflict remains unresolved |
| 3 — InclusiveHAR audit/ontology | Read-only v4 audit, privacy-safe participant/activity coverage, units, six-channel allowlist, label tracks and ontology locks | Release exposes no timestamp/trial/session/raw-sample ID; hidden joins cannot be ruled out, so the benchmark is not trial-safe or unqualified leakage-safe |
| 4A — corrected UCI reproduction | Official-train adapter, grouped folds, reconstructed architectures, v1.1 CUDA matrix config and runner | 75 model/fold/seed runs and aggregate are recorded as `configured_not_run`; official test remains consumed |
| 4B — within-group | Protocol config exists | No release-stable within-group result was established by this audit snapshot; any later implementation/result needs independent lineage and participant-level validation |
| 4C — zero-shot primary | Source-only freeze, opening receipt, 100 locked target result cells, index, participant statistics and publication report | Opening is consumed exactly once and cannot be rerun; findings are observational ability-associated shift, not causal disability/fairness/clinical evidence |
| 4D — few-person | Corrected v1.1 functional-core manifest with 15 scenarios and 80 neural model-seeds | 1,200 CUDA scenario results and aggregate are not claimed complete; v1 is superseded and must remain preserved |
| 4E — cross-source pretraining | UCI provenance, adapter and semantically explicit ontology constraints | No completed cross-source-pretraining result. All-cohort exact classification is limited to sitting; standing is provisional and walking is incompatible with manual-propulsion semantics |
| 4F — sensor reliability | Deterministic physical-space corruption config, runner/aggregator and synthetic tests | Configured secondary operation is not claimed executed; severity levels are not empirically calibrated |
| 5 — baselines | Frozen classical, legacy, DeepConvLSTM-family, compact ERM/CORAL/DANN and parameter-matched controls | No faithful CNN-HAR, TinyHAR, TinierHAR, HARMamba, BenchHAR SSL/foundation result; standalone general GroupDRO, CCIL and BPD are not in the locked primary inventory. See baseline omissions audit |
| 6 — MoRe-HAR | Compact implementation, required objective ladder, modality ablations, GroupDRO variant and parameter-matched control across five seeds | Hypothesis is not supported. Direct CCIL/BPD comparison is absent from the locked primary suite; no retroactive primary addition is allowed |
| 7 — hardware/training | Observed 12 GB CUDA machine record, actual CUDA allocation, mixed precision, deterministic seeds, sequential final runs and preserved recurrent backend failures | Recurrent success required CUDA tensors with cuDNN disabled. Do not generalize timing/compatibility beyond the recorded machine |
| 8 — metrics/statistics | Participant macro-F1, tails, calibration, AURC, per-class recall, clustered bootstrap, paired tests/effect sizes/multiplicity and model sizes | CUDA latency/VRAM profile is implemented but not claimed executed; declared analytical MACs cover only Conv1d/Linear/LSTM, not every graph operator |
| 9 — evidence gates | Historical pre-opening gate, immutable split/freeze/opening/index lineage, extensive synthetic tests | All tests/lint/format/types/manifests/splits/config/artifact scans must be rerun after current changes; historical 190-test evidence does not validate this worktree |
| 10 — release | README/cards, CITATION.cff, Zenodo metadata, paper-ready outline, legacy/protocol tags | No Git remote is configured, no private repository link or benchmark tag exists, and no DOI is claimed. Final release inventory/gate and successful remote CI remain outstanding |

## Deliverable-level conclusion

The repository contains a reproducible locked primary benchmark result and an
honest negative MoRe-HAR outcome; it does **not** yet satisfy every requested
extension. The highest-value unfinished executions are the corrected UCI
reproduction, within-group evidence, few-person curve, and validated secondary
efficiency/stress/unit/signal-definition analyses. Missing licensed/faithful
third-party baselines and cross-source semantic incompatibility are scientific
or provenance blockers, not tasks that may be papered over with renamed local
models.

A full manuscript is not present; `paper/OUTLINE.md` is a paper-ready outline and
evidence narrative. A private GitHub release must wait for the new gate described
in `docs/RELEASE_EVIDENCE_GATE.md`. Post-confirmatory completions may extend the
release, but none may change the consumed zero-shot decision or be presented as
predeclared confirmatory evidence.
