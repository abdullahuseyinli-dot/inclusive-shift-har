# Annotation-independent inference populations and explicit comparison groups

Protocol: `configs/protocols/external_har_observable_context_v1.yaml`, declared
2026-09-05 07:16:07 UTC before replacement outcomes. This is a validity correction,
not an additional architecture search or a new confirmatory experiment.

## Finding and retained evidence

The session-grid-v3 repair correctly isolates FoG signal resampling, gravity,
timestamps and window phase from annotations. A further audit identified a
different path: HERA-DG-context-safe and HERA-DG-full summarize probabilities and
kinematics across a participant's admitted evaluation windows. The admission
subset depended on activity annotations. Consequently context features could
change when annotations changed, despite identical physical candidate windows.

The synthetic counterexample preserves fixed candidate signals and probabilities
and changes only which annotations are missing. Two retained windows are
unchanged, but participant context differs. Its record SHA-256 is
`559d29247e225734e97624f4bb2abd13420de271e548ae1fcfb84d83b5248b89` under
`.audit/phase2_external_v3/annotation_context_counterexample_20260905`.
This demonstrates the data-flow defect; it does not claim every inactive router
changed predictions. HERA-v2's window-local rescue/harm sentinel is not the same
as these two HERA-v1 participant-context lanes.

## Implemented separation

`ObservableWindowPool` has no annotation field. The FoG accumulator retains every
finite physical-segment candidate through a signal-only interface. Annotation
projection and homogeneous admission still produce the original scoring subset.
The modelling adapter returns the full inference population, scoring indices and
a separate supervision-eligibility mask. All outer training masks intersect that
eligibility mask. Placeholder labels are not supervision, and target-transfer
containers receive no real labels. Predictions are mapped back to scoring indices
only after the complete held-out participant pool has been evaluated.

Supervised fitting and inner selection continue to use the declared eligible
source windows. Participant context at evaluation uses all supplied observable
candidates, including unsupported and unannotated FoG windows. This is noncausal
participant-batch inference, not independent-window or zero-lookahead streaming.
The existing polyphase FIR also remains offline. IMU context remains confined to
the selected scripted-trial benchmark; it is not a natural full-session claim.

A real-data parity audit found 1,939 observable candidates and the unchanged
1,213 scoring windows (726 unscored candidates). The entire previous scoring
summary and all inherited base/expert feature views at scoring positions matched
bitwise. No model was fit in that audit. Record:
`.audit/phase2_external_v3/real_fog_observable_pool_parity_01`, SHA-256
`76b0240623f7addb1f8bca29775ae73099f826e23748cd4e7ac2d50de756a094`.

## Acceptance and reporting

The validator independently identifies annotation-selected context lanes. An old
mixed package becomes `PARTIALLY_VALIDATED_METHODS`: valid window-local methods
remain available, while the two affected context methods are diagnostic rather
than annotation-independent evidence. New observable-context claims require a
complete pool audit and recorded per-fit candidate populations. Existing
validation files are preserved; revised assessments use new names.

Tables disclose each method's input channels and inference unit and separate
different groups. Six-channel controls, derived-nine-channel methods and
participant-batch context do not form an exact before/after comparison tuple.
The original HERA-full versus XGBoost numerical failure is retained; the context
budget differs and the contrast cannot establish matched-window superiority.
The repaired context comparison is descriptive, not a replacement primary test.
The PB-HPF matched-six-channel prospective failure and its stop rule are unchanged.

Replacement budgets are one FoG classical/invention suite and one transfer for
each frozen complete-case/available-trial IMU cohort. Window-local decisions must
reproduce; the predeclared 1e-12 absolute probability allowance addresses only
previously measured floating-point accumulation differences. No better backend,
score, seed or ablation is selected. Clean-source replacement completion must be
established by actual run and validation records, not by this implementation note.

## Operational evidence corrections

Transfer metadata now distinguishes annotations preloaded for scoring eligibility
from real labels supplied to fitting or inference containers. The former happens;
the latter remains prohibited. The old literal claim that labels were first read
only after predictions was broader than the implementation and is not repeated.

The first full correction gate passed 896 tests but failed two typing checks in
the new parity test. That failed gate remains preserved; a later complete gate
must cover the final source. Read-only recording, validation and table modules
no longer import model runtimes. Quality gates explicitly compare source hashes
at the start, every command launch and completion.

`external_runtime_resource_retry_v1.yaml` and `v2.yaml` preserve the failed
available-trial forest and complete-case neural replacement attempts and freeze
single, lower-concurrency technical retries. A traceback-complete failed neural
process and its verified launcher required explicit manual termination; neither
is represented as a normal native exit. All checkpoints and receipts are kept.

`external_har_verified_provider_copy_v1.yaml` permits the exact 7,288,182,191-byte
public HAR-PMD archive after whole-file MD5/SHA-256 verification. This addresses
the retained TLS-hostname acquisition failure without disabling verification.
Local reading is optional, declared, and retains per-member CRC/SHA-256 receipts;
the raw archive remains ignored and is not a Git/release artifact. No cohort,
class, sampling, window, model or seed rule changes. Old and new comparison tuples
still require explicit equivalence evidence; storage support is not permission
to pool incompatible runs or claim completed neural replacement scores.
