# UCI-HAR v1 data card and provenance record

## Status and role

UCI-HAR dataset 240 is a conventional-population source dataset and legacy reproduction baseline. It is not the ability-shift target. Its official test split was opened repeatedly in the coursework, so all old and future views of that test are **legacy exploratory/development-consumed**, never fresh confirmatory evidence.

Official sources:

- [UCI catalog record](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [DOI 10.24432/C54S4K](https://doi.org/10.24432/C54S4K)
- [Official static download](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip)
- [Official Version 1.0 names/README notice](https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names)

The catalog reports 30 participants, six activities, 50 Hz sensing, 128-sample/2.56-second windows, 50% overlap, and a participant-disjoint 70/30 released train/test partition. These claims are preserved as provider statements. The processed archive contains no trial identifiers or raw-sample indices with which to reconstruct the underlying window spans.

## Retrieved artifacts

| Artifact | Bytes | SHA-256 | Status |
|---|---:|---|---|
| Official HTTPS wrapper | 61,005,872 | `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031` | Provider publishes no checksum; locally pinned after first official retrieval |
| Embedded `UCI HAR Dataset.zip` | 60,999,314 | `2045e435c955214b38145fb5fa00776c72814f01b203fec405152dac7d5bfeb0` | Create-only materialization from verified wrapper member |
| Direct `.names` notice | 6,304 | `eaee53f45825f349681a1a1b92a997a0082648f9b5bf93ddb3290caeede4b3a8` | Byte-identical to the wrapper member and inner README |

The wrapper contains exactly two members and passes a complete ZIP CRC check. The embedded processed archive has 74 entries including directories and platform metadata, expands to 282,568,739 bytes, and also passes a complete CRC check. Raw files are read-only and excluded from Git.

The first wrapper retrieval was accidentally rooted twice and published under `data/raw/data/raw/uci_har/v1/`. That path was not deleted. A create-only hard link was added at the canonical location; both names have the same NTFS file ID, hash, size, and read-only state. The incident is retained in `results/acquisition/uci_har_v1.receipt.json`.

## Conflicting licence notices

Both official notices remain visible:

1. The current UCI catalog labels the dataset CC BY 4.0.
2. The embedded Version 1.0 README requires publication acknowledgement and states that commercial use is prohibited.

The project does not assume the catalog notice silently supersedes the embedded restriction. The conflict remains `NOASSERTION`; raw redistribution is prohibited pending authoritative clarification. Apache-2.0 covers only repository-authored code.

## Verified processed schema

The adapter reads only these six released inertial matrices, in this order:

1. `body_acc_x`
2. `body_acc_y`
3. `body_acc_z`
4. `body_gyro_x`
5. `body_gyro_y`
6. `body_gyro_z`

Each released instance becomes `[128, 6]`. Labels and participants come from the row-aligned `y_<split>.txt` and `subject_<split>.txt` files and are never model features.

Read-only audit results:

| Split | Windows | Participants | Shape | Evidence status |
|---|---:|---:|---|---|
| Official train | 7,352 | 21 | `[7352, 128, 6]` | Source development |
| Official test | 2,947 | 9 | `[2947, 128, 6]` | Legacy/development-consumed audit only |

The participant sets are disjoint. All six matrices in each split agree in shape; all values are finite. Official-train class counts are Walking 1,226, Upstairs 1,073, Downstairs 986, Sitting 1,286, Standing 1,374, and Laying 1,407. Official-test counts are retained in the machine audit but are not inputs to class-count inference or model selection.

Stable IDs have the form `uci_har_v1:<released-split>:window:<one-based-row>`. These identify released window instances only. They must not be misrepresented as original raw-sample or trial IDs.

## Label tracks

The released ontology is fixed from `activity_labels.txt`; it is never derived from whichever split is loaded.

UCI-native walking, sitting, and standing map to canonical UCI names as follows:

| UCI ID | Released name | UCI-native canonical name | Cross-source status under the locked ontology |
|---:|---|---|---|
| 1 | `WALKING` | `walking` | Excluded from all-cohort exact mapping; ambulatory sensitivity only after an eligibility lock |
| 4 | `SITTING` | `sitting` | Exact |
| 5 | `STANDING` | `standing` | Provisional and blocked pending target-realization documentation |

UCI stairs are not InclusiveHAR ramps, and UCI ordinary walking is not silently equated with InclusiveHAR manual wheelchair propulsion.

## Reproducible interfaces

- Loader and audit: `inclusive_shift_har.data.uci_har`
- Source-only grouped folds: `inclusive_shift_har.protocols.uci_source`
- Dataset manifest: `manifests/datasets/uci_har_v1.json`
- Acquisition receipt: `results/acquisition/uci_har_v1.receipt.json`
- Data audit: `results/data_audit/uci_har_v1.audit.json`
- Protocol record: `results/protocol/uci_har_source_grouped_v1.json`

The test suite uses generated fixtures and performs no dataset download.
