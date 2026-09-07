# Corrected FoG bounded decision-rule probe v1

Status: frozen before outcome-producing fits. This protocol implements the exact
decision in `.audit/fog_gsp_team_review_20260907-001/NEXT_EXPERIMENT_SPECIFICATION.md`.
It is exploratory development evidence on the already consumed corrected FoG cohort.

## Question

Hold the F1 six-channel 80-feature participant-weighted Random Forest and its raw
probabilities fixed. Compare ordinary argmax (D0) with a class-utility decoder (D1):

`argmax([p_mobility, m_sitting*p_sitting, m_standing*p_standing])`, where mobility is
fixed at one and sitting/standing independently take values in `{0.5, 1, 2}`.
The multiplied values are decision scores, not calibrated probabilities. Class ties
resolve mobility, sitting, standing. D0 and D1 retain identical raw probabilities,
NLL and Brier scores. Classification metrics use explicitly saved decoded labels;
confidence for an emitted label is its own raw probability.

## Data and nesting

Use only the validated corrected FoG cache: 1,939 annotation-independent observable
candidates, 1,213 scored rows, 22 people, five pre-window participant folds and fixed
classes `[mobility, sitting, standing]`. Cache and prediction hashes are frozen in the
configuration. No raw source rematerialization, target population, extra dataset,
label, identity, time/order proxy or new feature is permitted.

First fit five fresh outer F1 forests and require exact all-candidate probability
replay against both archived F1/A controls. Stop before inner fitting on any mismatch.
For outer fold `o`, rotate each other original fold `j` as inner validation. Train on
eligible rows in neither `o` nor `j`; predict every observable candidate in `j` and
select the decoder only on eligible inner-OOF rows. Thus there are 20 inner fits and
five outer fits. Existing outer-OOF predictions cannot replace inner predictions.

Each training partition recomputes mean-one weights `1/(m_i*n_ic)`. Outer seed is
`11+o`; inner seed is `1100+10*o+j`. The fixed RF has 500 trees, sqrt feature sampling,
leaf size two, bootstrap, no automatic class weight, four fit workers, one prediction
worker and one BLAS/OpenMP thread. One controller runs fits sequentially.
The execution environment is frozen to Python 3.11.9, NumPy 2.3.5 and
scikit-learn 1.8.0 in the same interpreter recorded by the validated GSP reference.
Before fitting, all effective Random Forest parameters for seeds 11 through 15 must
match the sealed GSP control records.

## Selection and outcomes

Within each outer-training population, evaluate all nine decoder rules. A candidate is
feasible when, versus identity, bottom-30 loss is at most 0.01, worst-person loss at
most 0.03, mobility-recall loss at most 0.01, and each posture-recall loss at most
0.02. Identity always qualifies. Maximize equal-participant fixed-three-class F1.
Within `1e-12`, prefer the smallest L1 norm of log2 offsets, then lexicographic sitting
and standing offsets. Select identity unless a nonidentity candidate improves more
than `1e-12`. Freeze all five policies before constructing any outer D1 label or score.

The primary mechanistic comparison is D1-D0. The required practical comparison is
D1-F3, using the aligned saved derived-nine F3 result. Each must achieve mean gain at
least 0.015, at least 14/22 strict participant wins, bottom-30 difference at least
-0.01, worst difference at least -0.03, mobility recall at least -0.01, each posture
recall at least -0.02, and all leave-one-person means above `1e-12`. Recommendation of
more compute additionally requires positive participant-weighted mean difference after
omitting every complete outer evaluation fold. Report both gates, uncertainty and all
harms even if one fails.

The fixed grid can alter decisions for 253 saved scored rows across 20 people, but this
label-free reachability fact predicts no gain. It cannot alter P007 or P017; in
particular it cannot repair P007's confidently mobility-classified posture windows.
Do not widen the grid or change the rule after results.

## Evidence and stopping

Maximum 25 fit attempts and 3,600 seconds complete compute. All attempts, failures,
partitions, weights, probabilities, hard labels, decision scores, selected-label
probabilities, selection tables, policies, checkpoints, metrics, comparisons, hashes
and timings are create-only evidence. An independent no-fit validator replays every
checkpoint and reconstructs all 45 rule rows, five selections, labels, metrics and
gates. Validation also binds the cache to both pinned controls, rechecks committed
source and protocol bytes, requires exact attempt, checkpoint and policy artifact
families, and rejects a failure record. Every partition records participant-by-class
counts. A post-fit receipt must observe no active Python worker process or thread
before the package can be completed. A failure or incomplete run stops this rule
family. No seed, GSP, router, physical pilot, external data or publication task
follows automatically.
