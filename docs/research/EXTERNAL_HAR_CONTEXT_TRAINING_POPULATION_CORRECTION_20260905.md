# External HAR context-training population correction

Declared 2026-09-05 at 17:56:13 UTC and last amended at 22:47:03 UTC, after the
retained v3 FoG development outcomes and before any v4 replacement outcome. This is a semantic-validity
correction, not an outcome-driven model change. Its machine-readable protocol is
`configs/protocols/external_har_observable_context_training_population_v1.yaml`;
its create-only supersession record is
`results/research/cross_dataset_har_v4/context_training_population_supersession_20260905.json`.

## Finding

The observable-context-v1 repair correctly made every outer held-out FoG
participant's complete, annotation-independent candidate stream available before
scoring eligibility was applied. The same rule was not applied inside the outer
training partition. Inner participant-held-out base and expert predictions were
created only for homogeneous, annotation-eligible windows, and the HERA responder
signature was consequently pooled over that annotation-selected subset.
The CAGE sensor-health channel scale was also fitted on annotation-eligible
outer-training windows. That scale affects CAGE context/reliability features and
the downstream HERA context-safe, full, and v2-full routes.
The final validity review found the same population error in the HERA physics
reference: its veto threshold was fitted on annotation-eligible outer-training
windows. That threshold affects the strict physics veto and the physics
reliability/trust features consumed by HERA context-safe, full, and v2-full.

This did not leak outer-test labels into fitting. It did, however, mismatch the
context population used to train the participant router with the population used
at inference. Changing annotation availability could change the training context
membership even when sensor values and observable boundaries were fixed. Artifact
hash and split validation did not detect that semantic asymmetry.

## Correction

The implementation now maintains two explicit masks:

- a supervised mask for model fitting, utility targets, calibration, candidate
  selection, and scoring; and
- an observable-candidate mask for participant-held-out prediction and context
  pooling.

Each inner model is fit only on eligible windows from its training participants,
then predicts every candidate from its held-out participants. HERA training
signatures pool all candidates from every outer-training participant. Utility
targets and calibration remain restricted to eligible labels. The outer test path
continues to predict and pool all observable candidates before post hoc scoring.
CAGE channel-scale estimation now also uses every observable candidate from the
outer-training participants; its router supervision remains eligible-only.
The HERA physics-reference threshold is likewise fitted on every observable
outer-training candidate. Its internal minimum-valid-pair-fraction filter is based
only on sensor-derived quality; labels and scoring eligibility do not select the
fit population. Physics-aware calibration, responder utility, and route selection
remain supervised-only.
A participant with candidates but no eligible training annotation remains in the
context population, while its signature is explicitly omitted from label-dependent
utility/router fitting and recorded as lacking supervision.

The result package records candidate and scored counts separately for outer and
inner folds. Validation requires those counts and participant sets to partition
the complete observable pool. Metamorphic tests alter ineligible placeholder labels
while holding candidate inputs fixed and verify unchanged candidate predictions and
selection. A separate metamorphic test changes annotation eligibility while holding
the outer participant assignment and candidate signals fixed, and verifies identical
CAGE scale, context, and reliability arrays.
It also verifies that changing annotation eligibility cannot change the fitted
physics threshold or any resulting reliability or trust output, while demonstrating
that the former annotation-selected fit would change.

The replacement parity gate may use the prior package only as an exact, still-
superseded historical witness. Its result, prediction, data-audit, launch-commit,
and source-manifest hashes are pinned in the protocol. The comparison normalizes
only the enumerated v4 audit-schema additions and key split; all pre-existing
candidate signal/gravity/identity hashes, segment timestamps and grids, admission
and exclusion records, scoring identities, and unaffected-method probabilities
must remain exact.

## Evidence disposition

The retained v3 FoG classical package remains preserved. Its `CAGE-DG`,
`HERA-DG-strict`, `HERA-DG-context-safe`, `HERA-DG-full`, and
`HERA-DG-v2-full` rows are superseded
for the corrected training-context contract. Other reasons already supersede the complete v3 package
for publication use. No before/after performance improvement may be claimed from
this correction because the inference tuple changed. Replacement values require a
clean v4 launch, full frozen seeds and controls, retained predictions, and successful
artifact and split validation.

This correction does not make participant-batch context causal or zero-lookahead,
does not restore IMU-HAR-IL continuous-stream validity, and does not support an
independent-confirmation, SOTA, clinical, or breakthrough claim.
