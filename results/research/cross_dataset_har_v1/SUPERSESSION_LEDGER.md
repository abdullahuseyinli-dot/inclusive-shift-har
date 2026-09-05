# External HAR supersession ledger

This ledger is part of the evidence package. No listed artifact was deleted.

The original FoG-STAR runs and the first IMU-to-FoG transfer check are **not valid
publication metrics**. Their causal gravity filter was reset at activity-label
boundaries, so target ground truth influenced preprocessing. The corrected loader keeps
filter state continuous within each physically contiguous recording and only partitions
labels after preprocessing.

The exploratory IMU-HAR-IL runs and failed HAR-PMD run are also retained, but their
replacement runs are the only candidates for reporting. Machine-readable status,
reasons, replacements, and permitted uses are recorded in
`supersession_ledger_20260905.json`.

The first strict label-blind FoG-STAR reruns passed their evidence validators but were
completed immediately before the formatter gate. They remain preserved as replication
evidence. The `fog_star_publication_checkpoint_*_20260905` replacements bind the exact
formatted source snapshot selected for the repository checkpoint; they reproduce every
hard prediction and scientific score.

This file does not promote an in-progress replacement. A replacement becomes reportable
only after its own `result.json`, predictions, receipts, validation, and integrity hashes
exist.
