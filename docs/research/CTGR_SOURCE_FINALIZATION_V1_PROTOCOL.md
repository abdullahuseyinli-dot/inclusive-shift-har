# CTGR source-only final predictor protocol v1

This is a new source-only finalization recipe on the consumed InclusiveHAR P1--P10
development cohort. Its selection scores do not replace the historical 86.54%
nested/five-seed result and cannot establish confirmation. The historical target,
P11--P20, external datasets and confirmation outcomes are excluded.

The executable contract is `configs/experiments/ctgr_source_finalization_v1.yaml`.
The source manifest, raw artifact, dataset manifest, historical CTGR config and
Round A config are bound by existing file SHA-256 values. Only the 725 functional-core
windows are materialized. Participant partitioning precedes materialization in the
existing source contract; hidden-trial ambiguity remains unchanged.

Selection uses the original pairs (1,4), (6,7), (2,3), (5,9), (8,10). Each pair is
held out once while the other eight people train. Each fit uses random_state=11;
there are no seed searches or seed offsets in this new finalization recipe.
Each fold fits the original B6 denoised-GSP Extra Trees base and eight original
gravity experts: physics, physics_plus_rmrp, total_gsp, dual_gsp crossed with
Extra Trees leaf 3 and leaf 1. The initial implementation-task shorthand leaf
{1,2} was corrected by inspecting the locked CTGR YAML and actual estimator code.
The retained family truly uses leaf {3,1}; its order and candidate IDs are unchanged.

The original 121 probability candidates are computed algebraically from those nine
fits per fold: unchanged base plus 4 views x 2 experts x thresholds
[.45,.50,.55,.60,.65] x weights [.50,.75,1]. Probabilities use the retained CTGR
strict max-base-confidence trigger and mobility-preserving expert composition.
No auxiliary feature is identity, a clinical field, label, timestamp or location.

Aggregate per-person fixed-three-class macro-F1 over the ten unique OOF people.
Candidates within .005 of the largest mean are eligible. Resolve eligibility by
larger bottom-30% person F1, smaller mean of the five fold trigger fractions,
lower original complexity rank, then candidate ID. The original candidate ordering
function is reused. Write and hash the complete ranking, candidate grid, OOF
probabilities, IDs, labels, reports and selection freeze before any final fit.

Selection consumes at most 45 learned fits. Finalization fits B6, B9 and only the
selected expert on all ten source people at seed 11, at most three further fits.
Every attempt consumes the hard cap of 48 before the fit starts. An exception stops
the run without retry; partial checkpoints and failure receipts remain visible.

B6 and the posture experts retain the historical participant/class-cell sample
weights and `class_weight=balanced`. B9 exactly uses Round A A4's feature block order
`[native_gravity_physics, denoised_GSP]`, which is the existing physics_plus_rmrp
view. B9 uses Extra Trees 500, sqrt, leaf 1, no automatic class weighting and
mean-one weights 1/(m_i*n_ic), where m_i is the participant's present-class count.
It is not GSP extracted directly from a concatenated raw nine-channel signal.

T9 is the selected CTGR recipe. U9 shares its exact B6/expert bytes and blend weight,
with the confidence trigger forced on for valid evidence. If base_no_route wins,
T9 equals B6 exactly, no expert is fitted, and U9/trigger comparison is explicitly
not applicable. Never select an arbitrary expert to preserve a comparison table.
No final-source resubstitution score is presented as evaluation. Final model replay
on source features is solely an inference-integrity check.

Outputs are create-only: input/source/dependency receipts, snapshots, feature cache,
45 selection checkpoints at completion, OOF probabilities and reports, frozen
selection, final model bytes, replay receipts, runtime, failure/shutdown records,
and a sealed artifact manifest. Fits run sequentially with at most four estimator
threads and one BLAS thread. No subprocess experiment workers are launched.
All selection, final and validation predictions use one inference worker and one
BLAS thread. A shallow estimator view sets only prediction `n_jobs=1`; fitted
estimators and their four-worker fit parameters remain unchanged. This fixes tree
probability accumulation order before outcomes: parallel floating-point accumulation
can otherwise vary in the last bits and invalidate exact save/reload replay.
The validator verifies hashes and OOF selection arithmetic without fitting.
The controller must use a new output path; resume/retry of an existing path is rejected.

Run explicitly with `python -m inclusive_shift_har.experiments.ctgr_source_finalization
run --evidence-root ... --config ... --output-directory ... --code-commit ...`.
The independent no-fit check is the same module's `validate` command. A predictor
bundle is not an authorization to open confirmation labels or deploy a model.
