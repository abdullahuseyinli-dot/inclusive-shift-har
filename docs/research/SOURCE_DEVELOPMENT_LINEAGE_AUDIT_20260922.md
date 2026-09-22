# Source-development lineage: evidence audit

**Date:** 22 September 2026

**Status:** completed recomputation of preserved source-development predictions; no new experiment.

## Scope and finding

SpectralShape, GSP, RMRP, CTGR and HERA are project-developed feature and
decision pipelines that use established numerical methods and estimators.
RMRP is a development contribution, not an external baseline. Its selected
implementation was denoised GSP in all five outer folds; the residual view
was not selected.

The [aggregate evidence record](../../results/research/source_development_lineage_v1.json)
reconstructs ten seed-11 methods on exactly the same 725 source windows,
ten participants and five participant-exclusive outer folds. This establishes
a source-development progression from tested external controls in the 70s
to the project pipelines in the 80s. It does not establish independent
superiority, equal search effort, or worldwide novelty.

The original [v2 report](FUSE_REFRAME_V2_RESEARCH_REPORT.md) remains unchanged.
It contains additional rejected methods, corruption tests and external-domain
failures. This audit is a selected development lineage, not an exhaustive
replacement for that report or later experiment records.

## Recomputed comparison

All entries below use seed 11. Accuracy is the proportion of correctly classified
windows. The primary macro-F1 is calculated separately for each participant
over the fixed three-class set, then averaged equally across participants.
These are not pooled macro-F1 or averages over different datasets.

| Method | Origin | Channels | Accuracy | Participant macro-F1 | Worst participant macro-F1 |
|---|---|---:|---:|---:|---:|
| HYDRA | Executed published-method control | 6 | 73.517% | 70.711% | 50.229% |
| MultiRocket | Executed published-method control | 6 | 76.138% | 73.501% | 52.063% |
| MultiRocket+HYDRA | Executed published-method control | 6 | 76.000% | 73.616% | 52.603% |
| QUANT | Executed published-method control | 6 | 76.966% | 74.120% | 54.608% |
| RIST, budgeted | Executed published-method control | 6 | 78.759% | 77.195% | 61.681% |
| SpectralShape | Project feature pipeline | 6 | 79.724% | 78.884% | 45.852% |
| GSP | Project representation pipeline | 6 | 83.448% | 82.916% | 49.850% |
| RMRP, selected denoised GSP | Project denoising and representation development | 6 | 84.414% | 83.790% | 54.936% |
| CTGR | Project selective gravity correction | 9 | 86.897% | 86.474% | 61.312% |
| Strict HERA-v1 | Project CTGR extension | 9 | 87.034% | 86.755% | 61.312% |

The six-channel input contains user acceleration and rotation rate.
CTGR and HERA add three native gravity axes. Consequently, their gains over
six-channel RMRP combine additional signal information with changes in the
pipeline. The [fixed Round A comparison](../RESEARCH_REPORT.md)
examines signal, denoising and estimator effects under a separate fixed
configuration budget.

The external methods are the configurations actually executed in this project,
not scores copied from their papers or exhaustive evaluations of their
method families. They had fixed recorded configurations. SpectralShape,
GSP and RMRP used eight, seven and seven candidate configurations,
respectively, selected through four inner participant folds within each outer
fold. CTGR/HERA used nested selection and calibration with their own candidate
sets. Sharing the evaluation observations does not make these search budgets
equal.

## Paired changes and participant harms

Changes below are descriptive seed-11 comparisons; one percentage point is
0.01 in the stored F1 scale. A win or harm is the sign of each participant's
F1 difference, with absolute tolerance 1e-12.

| Change | Mean participant macro-F1 gain | Wins | Harms | Ties |
|---|---:|---:|---:|---:|
| GSP over budgeted RIST | +5.721 pp | 7 | 3 | 0 |
| RMRP over budgeted RIST | +6.595 pp | 7 | 3 | 0 |
| RMRP over GSP | +0.874 pp | 5 | 4 | 1 |
| CTGR over RMRP | +2.684 pp | 8 | 2 | 0 |
| Strict HERA-v1 over CTGR | +0.281 pp | 3 | 2 | 5 |

Mean improvement is not improvement for every participant. Budgeted RIST's
worst-participant F1 exceeds the worst-participant scores of the later methods
in this seed, despite its lower mean. RMRP improves the weakest GSP score,
but the v2 report also records reduced temporal-gap robustness. The small
HERA increment does not overturn its failed advancement gates.

No new confidence intervals or significance tests were produced in this
audit. Ten repeatedly reused source participants remain the independent units;
seeds are not additional participants. The separate
[five-seed evidence](REPORTED_METRICS_AUDIT_20260920.md) reports
RMRP 83.953%, CTGR 86.540% and strict HERA 86.849% mean participant macro-F1.
Those seed averages must not be substituted into the single-seed differences
above.

Legacy random-window scores, the v1 target few-person result near 75%, locked
target zero-shot evaluation, labelled personalization and external datasets
remain separate evaluation contracts. None supplies an interchangeable
starting point for this table.

## Verification and provenance

The export completed **1,388 checks** over **106 input-file bindings**:

- Reconstructed self-hashes for result records and checked saved prediction
  file hashes against their associated records.
- Loaded saved class probabilities with NumPy's object-pickle loading disabled;
  verified finite, nonnegative probabilities summing to one.
- Sorted unique window IDs and established exact equality of all 725 labels,
  participant identities and window identities across all ten methods.
- Checked all eight earlier methods' five fold records and fold prediction
  files, complete disjoint participant coverage, and exact agreement between
  their concatenated fold probabilities and aggregate probabilities.
- Matched the same five participant partitions to the CTGR/HERA seed-11
  records and verified recorded inner-fold counts for nested feature selection.
- Recomputed window accuracy, pooled macro-F1, class recalls, confusion matrices,
  participant macro-F1, bottom-30% and worst-participant F1.
- Matched recomputed accuracy, pooled F1, class recalls, confusion matrices,
  participant scores, primary mean and worst scores to historical records.
- Verified that the retrospective CTGR probabilities exactly replay the
  original CTGR predictions and that its base probabilities exactly replay
  the canonical RMRP predictions.

The public record contains portable relative paths, file hashes, full-precision
aggregate metrics, sorted participant scores and sorted paired differences.
It does not contain participant identifiers, window identifiers, per-window
predictions, raw signals or model checkpoints. The shared source-identity
hash binds the private aligned identity arrays without publishing those arrays.

No models were fitted. No raw dataset, external dataset or locked target
participant data were loaded. Original evidence files were read without
modification.

### Recompute metrics from retained predictions

A clone can verify the public record's self-hash and aggregate arithmetic.
Prediction-level reproduction additionally requires restoring the hash-bound
files under their recorded relative paths; those archives are intentionally
outside ordinary Git history. It does not require retraining.

The following Python example verifies each exported method directly from
its retained prediction archive. Set `evidence_root` to the directory
containing the restored `.audit` tree. NumPy and scikit-learn are included
in the documented research environment.

```python
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

evidence_root = Path("PATH_TO_RESTORED_EVIDENCE_ROOT")
record_path = Path("results/research/source_development_lineage_v1.json")
record = json.loads(record_path.read_text(encoding="utf-8"))
claimed = record.pop("record_sha256")
canonical = json.dumps(
    record, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
assert hashlib.sha256(canonical).hexdigest() == claimed
bindings = {item["path"]: item for item in record["input_files"]}
reference = None

for method in record["methods"]:
    path = evidence_root / method["prediction_path"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == (
        bindings[method["prediction_path"]]["sha256"]
    )
    with np.load(path, allow_pickle=False) as saved:
        order = np.argsort(saved["window_ids"])
        labels = saved["labels"][order]
        persons = saved["participant_ids"][order]
        windows = saved["window_ids"][order]
        predicted = saved[method["prediction_key"]][order].argmax(axis=1)
    identity = (windows, labels, persons)
    if reference is None:
        reference = identity
    assert all(np.array_equal(a, b) for a, b in zip(identity, reference))
    scores = sorted(
        f1_score(
            labels[persons == person], predicted[persons == person],
            labels=[0, 1, 2], average="macro", zero_division=0,
        )
        for person in np.unique(persons)
    )
    expected = method["metrics"]
    assert abs(float(np.mean(predicted == labels)) - expected["accuracy"]) < 1e-12
    assert abs(float(np.mean(scores)) - expected["mean_participant_macro_f1"]) < 1e-12
    assert np.allclose(
        scores, expected["participant_macro_f1_values_sorted"],
        rtol=0, atol=1e-12,
    )
    assert confusion_matrix(labels, predicted, labels=[0, 1, 2]).tolist() == (
        expected["confusion_matrix"]
    )
    print(method["display_name"], expected["accuracy"], float(np.mean(scores)))
```

The complete maintainer export and its detailed validation receipt are retained
locally at `.audit/source_lineage_export_20260922-001/export.py` and
`.audit/source_lineage_export_20260922-001/validation.json`. They are not part
of the public checkout. Its invocation was:

```powershell
.venv/Scripts/python.exe .audit/source_lineage_export_20260922-001/export.py `
  --evidence-root "PATH_TO_RESTORED_EVIDENCE_ROOT" `
  --repository . `
  --output results/research/source_development_lineage_v1.json
```

Historical source results, configurations and library versions remain bound
by their original result records. See the
[reproducibility guide](../REPRODUCIBILITY.md) for the distinction between
aggregate verification, preserved-prediction replay and full training reproduction.
