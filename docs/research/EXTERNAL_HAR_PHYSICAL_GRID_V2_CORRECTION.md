# External HAR physical-grid protocol correction v2

Declared 2026-09-05 at 01:56:58 UTC, before replacement scores. The authoritative
machine-readable amendment is `configs/protocols/external_har_physical_grid_v2.yaml`.
It supersedes the external v1 instruction to start windows within label runs. It
does not modify the locked InclusiveHAR protocol or reopen its target cohort.

At `bb67c84173c2a9aa8192c79874dc6e9eb5399e05`, FoG gravity estimation was continuous
over finite physical segments, but `load_fog_star` subsequently split these signals
using `_contiguous_label_runs` and called `add_uniform_trial` for each run. Polyphase
filter edges and window origins therefore depended on ground truth. The regression
`test_fog_resampling_inputs_do_not_depend_on_annotation_boundaries` failed before
repair: a label shift and one missing annotation changed six resampling calls into
eight, with identical sensor values. Seventeen initial loader tests passed after
repair; the expanded metamorphic suite is recorded in the implementation gate.

Every earlier FoG-derived or FoG-target result is superseded as protocol-invalid,
including the two publication-checkpoint packages that passed artifact validation.
Existing files and their historical validators are preserved. A new supersession
record must bind their hashes without editing the older ledger. The named transfer
replacement `imu_rep1_to_fog_trial_continuous_seed11_validation_20260905` was absent
at takeover and must not be cited as completed evidence.

The corrected path identifies finite, monotonic physical segments using only
participant/session/task provenance, timestamps and sensor validity. Gravity is
estimated on that segment, its nine channels are resampled jointly once, and the
128-sample, nonoverlapping candidate grid begins at its first sample. Annotations
are projected afterwards. A candidate is admitted only if projected annotations
and every source annotation covered by its time interval are finite, supported
and identical in the provider ontology. This retains the original strict activity
homogeneity rule even for turn/walk codes mapping to mobility. A missing annotation
or transition shorter than a downsampling interval cannot disappear from admission.
Rejected candidates do not shift later windows. Per-segment hashes cover the signal,
gravity, timestamps and candidate grid; excluded indices and unused tails remain
in the data audit.

This repairs annotation isolation. SciPy polyphase resampling uses a symmetric FIR
with observable-edge zero extension: it does not establish zero-lookahead causal
streaming. A deployable streaming implementation would need a specified buffered
latency and its own equivalence test. Window identifiers and timestamps are audit
metadata, never model features.

The pinned sensor CSV has SHA-256
`888875133fef386affddd0a5864f122d527d8d7bc588a69905cc0871bef53477` and 329,027
rows. A label-free source audit found median within-task timestamp increment
0.016666666669607366, range 0 to 5483.76666666667, and no nonpositive within-task
increments. The provider README says milliseconds, but the values and declared
60 Hz rate demonstrate seconds for this pinned file. This discrepancy is disclosed;
units are not guessed dynamically. The public HAR-PMD record calls itself v2,
despite the repository's legacy internal identifier `har_pmd_v1`.

Loader audit scope:

| Loader | Boundary/annotation finding | Permitted interpretation |
|---|---|---|
| FoG-STAR | Label-dependent resampling and window phase repaired in v2 | Corrected development only, pending clean rerun |
| IMU-HAR-IL | Provider files define physical trials; finite runs and gravity independent of sample labels. Folder-label consistency gates change participant eligibility, not signal transforms | Scripted-trial development; missing/schema exclusions must be reported |
| HAR-PMD | Physical phone CSV defines trial; timestamp/finite runs control interpolation; folder supplies trial label | Mobility stress with separate 6ch/native9ch; not posture validation |
| Sole-HARmony | Camera bouts define resampling/window/temporal reset boundaries | Explicit oracle diagnostic; deployment claim prohibited |
| InclusiveHAR | Locked provider subject-label block windows; trial/session/timestamp unavailable | Historical released-block claim only; target already consumed |
| DAGHAR | Provider prewindowed arrays; ontology selection post provider windowing | Consumed replication with upstream boundary limitations |
| UCI-HAR | Provider prewindowed signals and splits | Provider-window benchmark; no reconstructed continuous-session claim |

The versioned evidence-role ledger is
`configs/datasets/evidence_roles_20260905_v2.yaml`. Repetitions do not increase
independent participant N. FoG transfer remains development evidence even with
target-label-blind model inference. No invention tuning is justified from superseded
FoG scores. Primary replacement reporting averages participant fixed-class F1 over
the three frozen seeds; probability-ensemble scores are separately secondary. The
single primary contrast is HERA-DG-full versus XGBoost-6ch on corrected FoG; other
comparisons are descriptive. No SOTA or breakthrough claim follows from this repair.

Because preprocessing, window phase and eligibility change, historical versus v2
scores have different comparison tuples. They must be shown as non-comparable rows,
never as numerical improvement caused by a model invention.
