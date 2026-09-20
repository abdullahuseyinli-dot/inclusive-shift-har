# Results and evidence policy

Every reported value carries an evidence class. A newer method does not alter a
locked historical result, and successful validation of artifact bytes does not
by itself establish scientific comparability.

Read the [research report](../docs/RESEARCH_REPORT.md) for comparable tables and
the [experiment map](../docs/research/README.md) for the full research sequence.

## Evidence classes

| Class | Meaning |
|---|---|
| `legacy_exploratory_development_consumed` | Historical development with an already reused evaluation source |
| `source_development_not_confirmatory` | Participant-exclusive source development, ablation, or selection |
| `development_target_descriptive` | Opened external/development cohort used for diagnosis, not confirmation |
| `locked_confirmatory_target_opening_1` | Frozen one-time InclusiveHAR target result |
| `post_confirmatory_secondary` | Analysis conducted after the locked target outcome |
| `failed` | Attempt or gate that did not complete or pass |
| `quarantined` | Evidence retained but excluded from interpretation or distribution |
| `superseded` | Historical result whose current interpretation is governed by a later correction |

Participant IDs are the inference units for participant-level results. Seeds are
repeated fits on the same people and are never counted as extra participants.

## Canonical tracked evidence

| Evidence | Class and status | Canonical record |
|---|---|---|
| Locked zero-shot opening 1 | `locked_confirmatory_target_opening_1`; complete and immutable | `confirmatory/zero_shot_v1/publication_report_v1.json` |
| Locked participant statistics | Confirmatory uncertainty and paired tests | `confirmatory/zero_shot_v1/participant_statistics.json` |
| Corrected UCI official-train reproduction | Source development; official test unopened | `legacy_reproduction/uci_har_source_grouped_v1/uci_source_grouped_report_v1.json` |
| Disabled-cohort within-group | Post-confirmatory; 75/75 CUDA cells | `analysis/within_group_v1_1/summary.json` |
| Few-person inclusion v1.1 | Post-confirmatory; 1,200/1,200 cells | `postconfirmatory/few_person_v1_1/statistics_attempt_004/few_person_v1_1_statistics.json` |
| Sensor-reliability stress | Post-confirmatory; 120 result cells | `postconfirmatory/sensor_stress_v1/sensor_stress_aggregate.json` |
| Qualified CCIL/BPD adaptations | Descriptive, non-faithful adaptations | `postconfirmatory/ccil_bpd_v1/ccil_bpd_postconfirmatory_aggregate.json` |
| Raw/total acceleration sensitivity | Post-confirmatory; failed v1 preserved | `postconfirmatory/raw_total_acceleration_v1_1/raw_total_acceleration_aggregate.json` |
| SI-unit equivalence | Preprocessing equivalence, not an accuracy result | `analysis/unit_sensitivity_v1/acceleration_unit_sensitivity.json` |
| CUDA efficiency | Post-confirmatory; 320 profiles | `efficiency/postconfirmatory-v1-attempt-002/neural_efficiency_aggregate_attempt_002.json` |
| CTGR five-seed source development | Advancement gate passed; not confirmatory | `development/max_rnd_secondary_v1_summary.json` |
| CAGE-HAR retrospective | Advancement failed | `development/cage_har_retrospective_v1_summary.json` |
| HERA-CTGR v1 retrospective | Highest source point estimate; advancement failed | `development/hera_ctgr_retrospective_v1_summary.json` |
| HERA-CTGR v2 retrospective | Calibration improved; routing did not advance | `development/hera_ctgr_v2_retrospective_v1_summary.json` |
| External FoG correction chain | Development/diagnostic with supersessions | `research/cross_dataset_har_v4/` and linked correction records |
| Current publication evidence index | Interpretive index; creates no new performance result | `research/current_publication_evidence_v1.json` |

## Recent local evidence

AICOS unit correction, the 38-person fixed routing validation, HARTH placement
diagnostics, and native-nine preparation were completed after the last tracked
aggregate summaries. Their current human-readable authorities are indexed in
[`docs/EVIDENCE_INDEX.md`](../docs/EVIDENCE_INDEX.md) and
[`docs/EVIDENCE_SUPERSESSION.md`](../docs/EVIDENCE_SUPERSESSION.md).
The current machine-readable role ledger is
`configs/datasets/evidence_roles_20260920_v4.yaml`.

Large probabilities, fold models, and run manifests remain under local `.audit/`
paths named by those reports. They are intentionally excluded from Git. A future
standalone evidence asset must be assembled create-only, hash every included
file, and pass dataset/privacy/licence review before distribution.

## Preservation rules

- No target metric may alter the frozen model, preprocessing, ontology,
  normalization, calibration, thresholds, seeds, or hypothesis decision.
- InclusiveHAR target opening 1 cannot be repeated.
- Failed, partial, OOM, backend-crash, reset, quarantine, and manual-intervention
  records remain visible.
- Superseded scores remain reproducible historical evidence but cannot be used as
  current headline results.
- A complete run binds dataset, split, preprocessing, ontology, configuration,
  code/environment, checkpoint selection, predictions, participant identities,
  metrics, and evidence status.
- Raw datasets, unrestricted participant-level material, and large checkpoints
  are not normal Git artifacts.

For a reader-facing comparison, use the [current evidence index](../docs/EVIDENCE_INDEX.md).
For restoration requirements, use the [reproducibility guide](../docs/REPRODUCIBILITY.md).
