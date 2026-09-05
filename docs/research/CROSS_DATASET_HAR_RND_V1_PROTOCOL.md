# Cross-dataset HAR R&D v1 protocol

Status: prospective external development, declared 2026-09-04.

Amended 2026-09-05 to harden label isolation and provenance without changing a model,
threshold, endpoint, or outcome criterion. Physical-run segmentation now explicitly
ignores missing as well as changing activity annotations; held-out inference APIs do not
accept labels; and every new run records a launch-time manifest of executable source and
protocol hashes. Earlier artifacts that do not meet these stricter controls remain in the
supersession ledger. The repository's base `uv.lock` remains unchanged; the external-HAR
dependencies are supplied by a separately hashed additive overlay.

This protocol evaluates whether the CTGR, CAGE-HAR, HERA-CTGR, and HERA-CTGR-v2
mechanisms generalize beyond the exhausted InclusiveHAR development participants. It does
not alter the locked InclusiveHAR protocol and it cannot convert prior retrospective work
into independent evidence.

## Evidence lanes

The three-class core ontology is `mobility`, `sitting`, `standing`. FoG-STAR and
IMU-HAR-IL enter a **derived-gravity** lane: total acceleration is converted to SI units,
gravity is estimated independently inside each trial with a causal low-pass filter, and
linear acceleration is total acceleration minus that estimate. The resulting inventions
are named RMRP-DG, CTGR-DG, CAGE-DG, HERA-DG, and HERA-DG-v2. They are adaptations, not
unchanged evaluations of the native-nine-channel inventions.

For a trial containing multiple labels, gravity is computed over each label-independent
finite/timestamp-contiguous physical segment *before* label-homogeneous window admission.
Ground-truth label changes never reset the filter. A label-boundary reset would expose
unknown annotation timing to preprocessing and is prohibited as leakage.

HAR-PMD enters a separate **native-gravity** lane using its recorded linear acceleration,
gyroscope, and gravity vectors. Its `still` label merges sitting and standing, so it is not
a valid three-class posture endpoint. It is a participant-exclusive mobility-mode and
interface stress test. Sole-HARmony enters the derived lane and is reserved for ordered,
multi-session temporal tests. Results from these lanes and endpoints must never be pooled
into a single F1 score.

## Leakage controls

Participants are assigned to folds before any training, tuning, or normalization. Windows
are generated within a single participant, session, trial, contiguous label run, and
contiguous timestamp segment. No window may cross a boundary or a timestamp gap exceeding
three nominal sample periods. Preprocessing parameters, model parameters, probability
calibration, intervention thresholds, and router heads are fit only on the corresponding
outer-training data; inner selection uses participant-exclusive out-of-fold predictions.
The held-out participant labels are read only for final fold scoring.
Classical held-out prediction functions accept windows but no labels. Neural inference
uses a label-free dataset and loader; class-balancing weights are computed only for
training data. This API separation makes an accidental held-out-label dependency fail by
construction rather than relying on a boolean assertion after a run.

Protected attributes and identifiers are retained for grouping and audit. They are not
model features. Repetitions in IMU-HAR-IL are separate trials, not longitudinal sessions.

Before IMU-HAR-IL window generation, every requested participant/repetition/activity file
must contain all six selected sensor columns and at least one finite segment long enough
for a complete analysis window. A failure excludes that participant's complete requested
record before any of their trials are windowed. Missing files, empty sensor payloads,
schema variants, and exclusions remain explicit in the source audit.

## Fixed temporal diagnostic

The Sole-HARmony temporal diagnostic applies a causal trailing arithmetic mean separately
to each class probability and renormalizes the vector. Widths 3, 5, and 7 are all reported;
none is selected using Sole-HARmony outcomes. State resets at every camera-annotated bout
and every timestamp-contiguous segment. For each width, 100 deterministic controls shuffle
window order only within the same segment before applying the same causal calculation and
then restore predictions to their original rows. These controls diagnose whether any gain
depends on authentic chronology; they are not independent-participant hypothesis tests.
Camera-bout reset locations come from reference annotations and would not exist in a
standalone deployment. Consequently this lane is an oracle-boundary mechanism diagnostic,
not a deployable temporal classifier or evidence of method superiority. Any future
deployable version must reset only at inference-observable session or sensor-gap boundaries
and be evaluated separately.

## Execution order and claim boundary

1. Audit immutable source metadata and stream receipts.
2. Develop independently on FoG-STAR and IMU-HAR-IL.
3. Evaluate zero-shot transfer from IMU-HAR-IL to FoG-STAR without target fitting.
4. Run the HAR-PMD native-interface stress endpoint.
5. Test causal bout evidence on at least 12 Sole-HARmony participants with at least two
   sessions each and run the required within-bout shuffled-time negative control.
6. Freeze code, preprocessing, checkpoints, thresholds, and the statistical analysis.
7. Only after that freeze and legitimate controlled access, open WearGait-PD once for the
   final confirmation.

FoG-STAR, IMU-HAR-IL, HAR-PMD, and Sole-HARmony are development evidence. WearGait-PD is
currently sealed and blocked because controlled access has not been supplied. Parkinson at
Home is a future extension requiring steward approval. No result is to be fabricated for
either inaccessible source, and this protocol alone does not authorize an SOTA claim.

## Storage and provenance

The source artifacts are too large for the currently available local storage. Public files
are therefore read from their immutable HTTPS locators and processed in memory. Every run
must record the locator, declared size/checksum when the repository provides one, actual
bytes received, computed digest, selected member or file identifier, and the fact that a
full local mirror was not made. This is an explicit operational limitation, not evidence
that a download was preserved locally. Existing raw evidence is never removed to make room.
Each completed model run also records a launch-time SHA-256 manifest covering all Python
source, the two external-HAR configurations, this protocol, `pyproject.toml`, `uv.lock`,
and both external-HAR overlay requirement files.
The evidence validator checks the manifest, result self-hash, prediction archive hash,
probability normalization, participant partition disjointness, receipts, and claim flags.

The immutable base environment must be installed first. Install the additional portfolio
dependencies without synchronizing away that environment:

```powershell
uv pip install --require-hashes -r requirements/external-har-research.lock
```

The overlay is generated from `requirements/external-har-research.in`; it is not a
replacement for `uv.lock`, and `uv pip sync` must not be used with the overlay alone.

The machine-readable source registry is
`configs/datasets/external_har_portfolio_v1.yaml`; the machine-readable experiment contract
is `configs/experiments/cross_dataset_har_rnd_v1.yaml`.
