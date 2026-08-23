# Legacy Task 2 audit

> **Evidence status:** all results in this document are **legacy exploratory/development-consumed**. The official UCI-HAR test set was opened repeatedly in the coursework and is not fresh confirmatory evidence. Nothing here is a corrected reproduction, an InclusiveHAR result, or evidence for MoRe-HAR.

## Scope and reference convention

This audit covers the inertial-sensor Task 2 notebook and its saved artifacts in the user-provided archive `CS5062_Assessment2_20251031.zip`. The original ZIP was read only. It was extracted with create-new-only semantics into a uniquely named local audit directory outside normal Git evidence. The unrelated robotics and image-HAR repositories were not modified.

Notebook references below provide all three identifiers needed to avoid ambiguity:

- **JSON index**: zero-based position in `cells[]`;
- **ordinal**: one-based human cell number;
- **cell id**: stable nbformat cell identifier.

The audited notebook is `Assessment 2_Task_2/script/assign2Task2_submission.ipynb`, SHA-256 `C25D78AD1A21E485A7BD0D9DD5CC74BBAF4D3872465BE300D7BE56D8B829F636`, with 49 cells: 24 code and 25 Markdown. [The machine-readable index](../legacy/notebook_cell_index.json) locks the cited cell identities and source hashes.

## Source integrity and archive inventory

| Check | Observed result |
|---|---:|
| Expected SHA-256 | `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94` |
| Observed SHA-256 | `13DE970A22336DB695029ACF5789DEC36D237CC0FC00D9BE7D779DFC6568CA94` |
| Archive size | 163,645,730 bytes |
| ZIP entries | 180 |
| File entries | 168 |
| Directory entries | 12 |
| Uncompressed bytes | 178,250,411 |
| Compressed entry bytes | 163,583,206 |
| Duplicate entry paths | 0 |
| Unsafe rooted/parent-traversal paths | 0 |
| Extracted file size/hash mismatches | 0 |

The complete entry-level path, type, uncompressed size, compressed size, represented timestamp, SHA-256, and external attributes are in [archive_manifest.json](../legacy/archive_manifest.json). Directory hashes are `null`; every file hash is over the uncompressed bytes. All 168 extracted files were rehashed and matched the inventory, and there were no missing or unexpected extracted files.

The archive contains 10 checkpoints totalling 169,244,861 bytes, but no UCI-HAR raw data, processed signal text files, `.npz` cache, subject identifiers, split index arrays, or data manifest. The large ZIP and checkpoints are preservation evidence and must not enter normal Git history.

An audit-generation failure is retained rather than hidden: the first PowerShell pass extracted and hashed all entries correctly but its aggregate-total expression errored and wrote zero totals into a local raw manifest. That raw artifact remains under the unique `.audit` directory. The public manifest recomputes totals from the intact 180 entry records and independently revalidates every extracted file; [verification_results.json](../legacy/verification_results.json) records this lineage.

## Saved output inventory and reconstruction

Ten timestamped Task 2 run directories were found. Each has the expected checkpoint, config/environment metrics, history, classification report, raw and normalized confusion arrays, prediction CSV, probability array, and plots. Each prediction/probability pair contains 2,947 official UCI-HAR test windows and six classes.

Metrics were reconstructed without executing notebook code or loading checkpoints. For every run:

- saved `y_pred` exactly equals `argmax(test_probs)`;
- probabilities are finite and within `[0,1]`;
- probability rows sum to one within absolute tolerance `1e-6` (observed maximum errors are below `2e-7`);
- saved and reconstructed accuracy and macro-F1 agree within `1e-12`;
- saved raw confusion matrices equal reconstructed matrices exactly;
- saved normalized confusion matrices agree within `1e-12`;
- classification-report accuracy and macro-F1 agree within `1e-12`.

The derived metrics use these locked definitions:

- **NLL:** `mean(-log(clip(p_true, 1e-15, 1)))`;
- **multiclass Brier:** `mean_i sum_c (p_ic - 1[y_i=c])^2` (not divided by class count);
- **ECE:** top-label ECE, `sum_b (n_b/N) * |accuracy_b - mean_confidence_b|`, using 15 equal-width bins `[b/15,(b+1)/15)`, with confidence exactly 1 assigned to the final bin.

| Legacy run | Accuracy | Macro-F1 | NLL | Brier | ECE-15 |
|---|---:|---:|---:|---:|---:|
| `00_bilstm__seed42` | 0.901256 | 0.900937 | 0.380576 | 0.156877 | 0.050093 |
| `01_bilstm_th__seed42` | 0.921955 | 0.921127 | 0.222779 | 0.110555 | 0.031207 |
| `02_bilstm_h256_th__seed42` | 0.924330 | 0.923548 | 0.226050 | 0.110557 | 0.033864 |
| `03_bilstm_th_ls005_jit025_f003__seed42` | 0.929420 | 0.929250 | 0.214756 | 0.099923 | 0.046670 |
| `04_bilstm_th_classw__seed42` | 0.924669 | 0.924078 | 0.219078 | 0.108799 | 0.029661 |
| `05_cnn1d__seed42` | 0.918901 | 0.919453 | 0.323111 | 0.130252 | 0.035150 |
| `06_ensemble_bilstm256_cnn128__seed42` | 0.943672 | 0.942901 | 0.207038 | 0.087803 | 0.018957 |
| `06b_ensemble_bilstm512_cnn256__seed42` | 0.939939 | 0.939858 | 0.228462 | 0.097419 | 0.032530 |
| `06b_ensemble_bilstm512_cnn256__seed42__seed1337` | 0.939260 | 0.938982 | 0.201512 | 0.090531 | 0.032819 |
| `06b_ensemble_bilstm512_cnn256__seed42__seed2025` | 0.944690 | 0.944981 | 0.209914 | 0.087303 | 0.029747 |

The notebook's three-run summary for the selected `06b` configuration is also arithmetically correct: accuracy `0.9413 ± 0.0030` and macro-F1 `0.9413 ± 0.0032` using sample standard deviation. This does not repair the protocol limitations below. Full-precision values, every confusion matrix, per-class measures, ECE bin contents, deltas, and source hashes are locked in [metric_reconstruction.json](../legacy/metric_reconstruction.json); [metric_reconstruction.csv](../legacy/metric_reconstruction.csv) is a compact view. Exact original `metrics.json` payloads are preserved in [legacy_metrics.locked.json](../legacy/legacy_metrics.locked.json).

## Scientific and engineering limitations

### 1. Random window-level validation permits same-subject and raw-sample leakage

Code cell **JSON index 20 / ordinal 21 / id `d295352d`** loads official UCI-HAR training windows and, when no dedicated validation directory exists, applies `StratifiedShuffleSplit(test_size=0.10, random_state=42)` directly to window indices. It never loads `subject_train.txt`, trial identifiers, experiment identifiers, or raw-sample spans. Code cell **24 / 25 / `40aeabcf`** then treats those window subsets as train and validation loaders.

Consequences:

- validation is random-window, not subject-exclusive;
- the same participant can occur in training and validation;
- because UCI-HAR windows originate from overlapping windows, adjacent windows sharing raw samples can fall on opposite sides of the split;
- no archived subject/window provenance exists to quantify the actual overlap after the fact.

This is a **possible same-subject/overlapping-window leakage path**, not proof of the exact leaked-window count. The required subject and raw-window identifiers are absent. The unusually high saved validation accuracies (up to about 0.993) are compatible with an easier within-subject validation task but do not alone prove leakage magnitude.

The Markdown before this loader says the stratified split ensures “fair model selection”; that claim is not supported for cross-participant HAR.

### 2. The official test set was evaluated repeatedly

Code cell **32 / 33 / `b66f3efb`** reloads the selected checkpoint and calls `evaluate_full` on the test loader inside every `train(cfg)` invocation. Code cell **38 / 39 / `616f3645`** invokes `train` for eight configurations and immediately displays each test accuracy, macro-F1, and AUROC. Code cell **40 / 41 / `9fa1ede6`** invokes two more test evaluations for seeds 1337 and 2025. Code cell **42 / 43 / `1761fd71`** aggregates the selected model's three test results.

The code selects the nominal winner by validation loss rather than directly sorting test accuracy, which is a useful distinction. Nevertheless, the official test results were exposed during iterative development ten times, included in comparison tables, and used in the report narrative. They must therefore be described only as **legacy exploratory/development-consumed**, never as a fresh confirmatory test.

### 3. Validation losses used for cross-model ranking are not all comparable

Code cell **32 / 33 / `b66f3efb`** constructs one `CrossEntropyLoss` from each run's class weights and label smoothing, and uses that same run-specific criterion to calculate validation loss and checkpoint selection. Code cell **36 / 37 / `29f6ee3c`** mixes ordinary cross-entropy runs with:

- label smoothing `0.05` for run `03`; and
- inverse-frequency class weights for run `04`.

Code cell **38 / 39 / `616f3645`** then sorts every run together by `best_val_loss` and declares the first row the winner. A label-smoothed or class-weighted cross-entropy value is not on the same objective scale as ordinary unweighted cross-entropy, so this global ordering is methodologically invalid for those rows. A common selection metric or separate, predeclared objectives were needed.

The checkpoint update also requires improvement greater than `1e-4`. In the `06b` seed-42 history, epoch 37 has literal validation loss `0.0265060103`, slightly below the recorded checkpoint value `0.0265635004` at epoch 29, but not by more than `1e-4`; the saved checkpoint therefore intentionally remains epoch 29. The locked `best_val_loss` is the accepted checkpoint criterion, not always the literal history minimum.

### 4. Paths are hard-coded and data/split provenance is absent

Code cell **34 / 35 / `5ec0b08b`** hard-codes `C:\Users\user\Desktop\CS5062 Assignment\...`, creates a cache in that external dataset tree, and directs outputs there. Code cell **48 / 49 / `f2534097`** hard-codes a winner run directory again. Code cell **22 / 23 / `9b4d3227`** tries to use `__file__`, which is not generally defined in a normal notebook kernel. Every saved config points to an unarchived absolute `har_cache.npz` path.

The archive contains none of the following:

- UCI-HAR source files or cache;
- source URL, release/version, licence record, or source checksum;
- cache checksum or preprocessing-code hash;
- train/validation indices or subject identifiers;
- trial/window/raw-sample identifiers;
- deterministic split manifest.

The result artifacts can be checked internally, but the exact data tensor and split cannot be reconstructed from this archive alone.

### 5. The purported ensemble is a jointly trained two-branch model

Code cell **14 / 15 / `8fcda24e`** defines `EnsembleAvg` as the average of two submodel logits. Its preceding Markdown says this combines models “without retraining either model.” In fact, code cell **26 / 27 / `12004a09`** creates fresh BiLSTM and CNN branches inside the wrapper, and code cell **32 / 33 / `b66f3efb`** gives `model.parameters()` from both branches to a single AdamW optimizer and backpropagates one joint loss through both.

This is not an ensemble of independently trained estimators. It is more accurately a **jointly trained dual-branch logit-average network**. The original “ensemble” name is retained only as a legacy artifact label.

### 6. Checkpoints and seed metadata are only partial

There are positive elements: code cell **3 / 4 / `db8f4393`** seeds Python, NumPy, CPU Torch, and CUDA and requests deterministic CuDNN; code cell **16 / 17 / `c9c3142f`** includes a run seed in `Config`; saved metrics retain config and basic environment versions.

However, code cell **32 / 33 / `b66f3efb`** saves only:

```text
model state, config, in_feats, num_classes
```

The checkpoint omits optimizer, scheduler, AMP scaler, epoch/history, training-only normalization mean/std, label schema, data/cache hash, split hash/indices, code commit, dependency lock, and Python/NumPy/Torch/CUDA RNG states. It is neither resumable nor self-sufficient for exact reconstruction. Normalization is computed in code cell **24 / 25 / `40aeabcf`** from unavailable training tensors and never saved.

The validation split seed is fixed to 42 in cell 20 rather than controlled by the run configuration. Most configurations have one model seed; only the selected large two-branch configuration has three. Two folder labels retain `seed42` and then append the actual seed (`__seed1337` or `__seed2025`), which is potentially confusing even though the true seed is recoverable from config.

### 7. Tests, package, CLI, lockfile, and CI are missing

The complete manifest contains no test suite, package metadata, installable module, operational command-line entry point, dependency lockfile, environment lock, or CI workflow. `argparse` is imported but no usable CLI is defined. There is no automated check for schema, split integrity, normalization isolation, prediction/probability alignment, checkpoint reconstruction, or artifact completeness.

### 8. Additional implementation risks

- Code cell **24 / 25 / `40aeabcf`** derives `num_classes` using `max(y_train.max(), y_val.max(), y_test.max())`, so test labels influence model-output dimensionality. Class count should be fixed by a locked ontology or training metadata.
- Code cell **20 / 21 / `d295352d`** applies `y = y - y.min()` independently within each split. This happens to be consistent with the six saved test classes, but it could silently remap labels if a split omitted the lowest-numbered class.
- Participant-level metrics and uncertainty cannot be reconstructed: `test_predictions.csv` stores only labels and class names, not participant, trial, window, or timestamp identifiers.
- Several saved code cells have null execution counts while retaining outputs, so the notebook's saved execution state is not a reliable proof of clean top-to-bottom execution.
- The saved AUROC values are retained as legacy values. The required Stage 0 reconstruction independently checks accuracy, macro-F1, confusion matrices, NLL, Brier, and ECE; it does not elevate AUROC to confirmatory evidence.

## What the legacy evidence does and does not show

The archive does show that the ten saved prediction/probability artifacts are internally coherent with their saved classification metrics and confusion matrices. It also preserves useful starter implementations of a BiLSTM, CNN1D, and a jointly trained dual-branch model.

It does **not** establish subject-exclusive generalization, absence of overlapping-window leakage, participant-level performance, reproducibility from source data, a valid independent ensemble comparison, a fresh UCI-HAR test result, InclusiveHAR performance, ability-associated generalization, fairness, clinical validity, or publishability.

## Stage 0 acceptance gate

| Gate item | Result | Evidence |
|---|---|---|
| Source archive checksum | Pass | Expected and observed SHA-256 match |
| Complete entry inventory | Pass | 180 entries with timestamps, sizes, and file SHA-256 values |
| Safe create-new extraction | Pass | No unsafe/duplicate paths; exact extracted file set |
| Saved metric reconstruction | Pass | 10/10 accuracy, macro-F1, and confusion checks pass |
| Calibration/error metrics | Pass | NLL, multiclass Brier, and specified ECE reconstructed for 10/10 runs |
| Scientific limitations explicit | Pass | Sections 1–8 above |
| Fresh confirmatory status | **Not granted** | Official test is permanently development-consumed |

The Stage 0 mechanics gate passes. That pass authorizes preserving the audit and proceeding to an independent literature/novelty gate; it does not validate the legacy scientific claims or authorize reuse of the old UCI test as fresh evidence.
