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
| 4A — corrected UCI reproduction | Official-train adapter, grouped folds, reconstructed architectures, 75/75 first-attempt CUDA cells, participant-level aggregate and comparisons | Complete source-development evidence; official test remained unopened in this route and cannot become fresh confirmatory evidence |
| 4B — within-group | Five subject-exclusive folds, three models, five seeds, 75/75 validated CUDA cells and participant-level aggregate | Complete post-confirmatory descriptive evidence; prior aggregation failure remains preserved and the cohort is only ten participants |
| 4C — zero-shot primary | Source-only freeze, opening receipt, 100 locked target result cells, index, participant statistics and publication report | Opening is consumed exactly once and cannot be rerun; findings are observational ability-associated shift, not causal disability/fairness/clinical evidence |
| 4D — few-person | Corrected v1.1 functional-core manifest, 1,200/1,200 validated CUDA cells, 64 model-k summaries and 80 paired comparisons | Complete post-confirmatory secondary evidence; attempts 1-3 failed and remain preserved; no paired comparison survived global Holm correction |
| 4E — cross-source pretraining | UCI provenance, adapter and semantically explicit ontology constraints | No completed cross-source-pretraining result. All-cohort exact classification is limited to sitting; standing is provisional and walking is incompatible with manual-propulsion semantics |
| 4F — sensor reliability | Deterministic physical-space corruption config, 120 completed stressed cells, participant-level aggregate and preserved clean lineage | Complete post-confirmatory secondary evidence; source n=2 versus target n=10 and empirically uncalibrated severities prohibit strong interaction claims |
| 5 — baselines | Frozen classical, legacy, DeepConvLSTM-family, compact ERM/CORAL/DANN and parameter-matched controls; qualified CCIL/BPD post-confirmatory adaptations | No faithful CNN-HAR, TinyHAR, TinierHAR, HARMamba, BenchHAR SSL/foundation, official CCIL, or official-faithful BPD result; standalone general GroupDRO is not in the locked primary inventory |
| 6 — MoRe-HAR | Compact implementation, required objective ladder, modality ablations, GroupDRO variant and parameter-matched control across five seeds | Hypothesis is not supported. Qualified CCIL/BPD comparisons now exist only as post-confirmatory non-faithful adaptations and cannot retroactively join the primary suite |
| 7 — hardware/training | Observed 12 GB CUDA machine record, actual CUDA allocation, mixed precision, deterministic seeds, sequential final runs and preserved recurrent backend failures | Recurrent success required CUDA tensors with cuDNN disabled. Do not generalize timing/compatibility beyond the recorded machine |
| 8 — metrics/statistics | Participant macro-F1, tails, calibration, AURC, per-class recall, clustered bootstrap, paired tests/effect sizes/multiplicity, model sizes, and completed 320-profile CUDA efficiency evidence | Efficiency timing is device-resident forward-only with allowlisted WDDM ambient processes; analytical MAC/FLOP coverage includes only Conv1d/Linear/LSTM, not every graph operator |
| 8S — signal sensitivities | Completed SI-unit equivalence record and corrected raw/total-acceleration v1.1 CUDA analysis | SI conversion is a preprocessing equivalence result, not accuracy evidence; raw/total acceleration is not UCI body acceleration and did not robustly improve the primary interface |
| 9 — evidence gates | Historical pre-opening gate, immutable split/freeze/opening/index lineage, extensive synthetic tests | All tests/lint/format/types/manifests/splits/config/artifact scans must be rerun after current changes; historical 190-test evidence does not validate this worktree |
| 10 — release | README/cards, CITATION.cff, Zenodo metadata, paper-ready outline, legacy/protocol tags | No Git remote is configured, no private repository link or benchmark tag exists, and no DOI is claimed. Final release inventory/gate and successful remote CI remain outstanding |

## Deliverable-level conclusion

The repository contains a reproducible locked primary benchmark result, an
honest negative MoRe-HAR outcome, and completed feasible UCI, within-group,
few-person, sensor-stress, signal-sensitivity, qualified predecessor-adaptation,
and CUDA-efficiency extensions. Remaining scientific gaps are faithful licensed
third-party baselines, a defensible cross-source transfer experiment, and an
evidence-cleared SSL/foundation comparison. These are semantic, licensing,
interface, checkpoint, or provenance blockers, not tasks that may be papered
over with renamed local models.

Cross-source pretraining and SSL/foundation comparisons were not evaluated. Exact-label all-cohort UCI→InclusiveHAR classification is blocked because sitting is the only defensible exact shared class; standing remains provisional and ordinary UCI walking is not wheelchair propulsion. Adapted-label transfer was not implemented. BenchHAR/SimMTM, FOCAL, and foundation-model tracks did not clear the combined licensing, interface, checkpoint, adapter, and equal-budget source-only selection gates. These omissions are not zero-valued or negative empirical results.

A full manuscript is not present; `paper/OUTLINE.md` is a paper-ready outline and
evidence narrative. A private GitHub release must wait for the new gate described
in `docs/RELEASE_EVIDENCE_GATE.md`. Post-confirmatory completions may extend the
release, but none may change the consumed zero-shot decision or be presented as
predeclared confirmatory evidence.
