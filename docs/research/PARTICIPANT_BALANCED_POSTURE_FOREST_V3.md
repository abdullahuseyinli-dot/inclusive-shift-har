# Finite participant-balanced posture experiment

The effective declaration is
`configs/protocols/participant_balanced_hierarchical_posture_forest_v3.json`,
frozen at 2026-09-05T05:24:18.938717+00:00 before candidate implementation or
outcomes. Its canonical record hash is
`6f62cad560f1c66aa337421f022cd03bc8056023f1ec063c6b86a7a6f6628d5e`.
The v1 and v2 declarations remain retained. Two pre-outcome implementation
reviews corrected a proposed channel mismatch and the exact inherited forest
seed mapping; neither followed candidate performance inspection.

## Motivation and non-novelty boundary

Corrected FoG has 1,213 eligible windows, including 74 sitting windows, with
unequal and sometimes missing participant/class support. HERA-full failed the
predeclared mean-and-tail contrast against XGBoost. Random Forest was the
strongest corrected frozen control by the primary mean. The proposed test
therefore starts from that stronger base and asks whether an explicit
mobility/posture factorization and equal participant/class training mass help.

Activity hierarchies, forests, and participant balancing are not new inventions
by themselves. This is a prospective, bounded test of their combination under
this protocol, not a novelty, SOTA, clinical or independent-confirmation claim.
No additional HERA routing search is authorized by this experiment.

## Fixed design

Six configurations use 500 total trees each: the exact inherited six-channel
forest; participant/class-balanced flat six-channel and derived-nine-channel
forests; an unweighted 250+250 hierarchy; the sole participant-balanced
hierarchical candidate; and its within-training-participant shuffled-posture
negative control. Both hierarchy branches use the same six-channel features
as the primary base. The derived-nine-channel flat forest is a separate
representation diagnostic, not a matched-input before/after comparator.

Each participant receives equal total training mass, divided equally among
that participant's observed classes and then among their windows. Binary root
and stationary-specialist weights use their respective training tasks. This
is different from merely assigning equal mass to every observed participant/
class cell, which gives people with missing classes less total mass.

The classifier computes mobility probability from the root and distributes its
stationary mass between sitting and standing through the specialist. Inference
accepts only signal features. It does not accept identity, labels, annotation
boundaries, device/ability metadata, or a personalization budget.

Seeds are 11, 23 and 47, with five participant-exclusive outer folds. The exact
baseline estimator seed is `seed + outer_fold`; specialist and shuffle offsets
are +10,000 and +20,000. There is no hyperparameter search. Per-window engineered
features can be computed once because tests require exact equality to
partition-local feature extraction. Models, per-fold records and predictions
remain create-only. The reference forest's class decisions must reproduce.

## Gates and stopping

The primary contrast is PB-HPF minus the corrected RandomForest-6ch. Use the
same seed-averaged fixed-class participant macro-F1 and paired 10,000-draw
participant-cluster bootstrap (seed 20260905). Primary mean CI lower bound must
exceed zero and bottom-ceil(0.30*N) CI lower bound must be at least zero.

Only then may the fixed-sequence superiority gates against the weighted flat
six-channel forest and unweighted hierarchy support advancement. Sitting
recall must improve, mobility recall and negative log-likelihood must not
worsen, and the true specialist must outperform the shuffled control. All
failed criteria and descriptive contrasts are retained. The validator
reconstructs the advancement gate from saved seed probabilities.

If any advancement gate fails, stop expanding this candidate; do not promote
another ablation as a new winner. Only after all gates pass may one unchanged
IMU available-trial development replication be launched. FoG and IMU remain
consumed development evidence. HAR-PMD stress and Sole oracle outcomes do not
select this candidate.

Tree count is compute-budget matching, not exact parameter/runtime matching.
Report actual node counts, serialized model bytes, feature dimensions, fitting
and prediction wall time. Concurrent local work qualifies timing; these are
not portable embedded-device latency claims. No raw dataset or large model
files are committed.
