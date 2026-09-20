# External generalization dataset qualification

**Decision date:** 2026-09-19  
**Decision:** use a two-lane evaluation. AICOS-HAR is the main unconsumed,
participant-diverse **derived-gravity** benchmark. MotionSense is the closest
provider-native nine-channel interface check, but it remains a post-hoc
diagnostic because its participants and scores were already consumed by the
DAGHAR evaluation. No located public dataset qualifies as an independent,
ability-relevant, provider-native-nine confirmation cohort.

## Required interface

Frozen CTGR/HERA uses 50 Hz, 128-sample windows and the following ordered
channels:

1. user acceleration x/y/z;
2. rotation rate x/y/z; and
3. provider-native gravity x/y/z.

The endpoint is participant-exclusive three-class recognition of mobility,
sitting and standing. Provider-native and causally derived gravity remain
separate evidence lanes.

## Candidate decision table

| Dataset | Participants | Ontology | Signal match | Prior project exposure | Qualified role |
|---|---:|---|---|---|---|
| [AICOS-HAR](https://doi.org/10.5281/zenodo.19452049) | 106 | Sitting, standing, walking, running, stairs and ramps map without merging | Raw acceleration plus gyroscope; gravity and user acceleration must be causally derived | None found | Main unconsumed external **HERA-DG/CTGR-DG** generalization benchmark |
| [MotionSense](https://github.com/mmalekzadeh/motion-sense) | 24 | Exact sitting, standing, walking, jogging and stairs mapping | Exact gravity, rotation-rate and user-acceleration triplets at 50 Hz | Consumed in the frozen DAGHAR evaluation | Provider-native-nine mechanism diagnostic only |
| [Self Phone Three Positions](https://github.com/AZI350/Smartphone-IMU-Three-Positions-Dataset) | 36 reported | Exact core mapping plus posture transitions | Native user acceleration, rotation rate and gravity; observed rate is mostly 50 Hz | None found | Exploratory exact-interface replication after provider clarification; not current publication-grade evidence |
| [ExtraSensory](https://extrasensory.ucsd.edu/) | 60 | In-the-wild sitting, standing, walking and running | OS pseudo-sensors exist, but availability is intermittent and the window/placement contract differs | None found | Later ecological stress test |
| HAR-PMD v2 | 120 | `still` merges sitting and standing | Native nine phone channels | Development stress source | Mobility-mode stress only; cannot validate posture separation |

## AICOS-HAR provider and archive audit

The official Zenodo v1 record exposes `AICOS-HAR.zip`, 2,539,861,502 bytes,
MD5 `faa895e7b203d664525b0a1e74542793`, under CC BY 4.0. A range-only archive
inspection was used; the raw archive was not downloaded or modified.

The ZIP central directory reports 27,702 entries, 18,589 files and about
7.341 GB uncompressed. The archive contains 5,231 acquisitions from 106
participants, with 5,231 accelerometer files and 4,899 gyroscope files. Exact
archive-to-metadata joining also exposed a `Lift`/`Elevator` folder-name
mismatch outside the selected endpoint; the adapter must normalize and record
that discrepancy rather than silently dropping those files. Metadata assigns
every participant to exactly one provider fold: 17/17/17/17/16 participants in
folds 1--5 and 22 participants in `test`; no participant crossed those folds in
the inspected metadata.

Of 5,231 acquisitions, 4,840 use phones and 391 use devices named as wearables.
After requiring phone accelerometer plus gyroscope and mapping walking, running,
stairs and ramps to mobility:

- provider folds 1--5 contain 2,965 eligible core acquisitions from 81
  participants; 38 participants contain all three endpoint classes;
- provider `test` contains 877 eligible core acquisitions from 21 participants;
  eight participants contain all three endpoint classes; and
- all folds contain 3,842 eligible core acquisitions from 102 participants;
  46 participants contain all three endpoint classes.

Lifts and lying are excluded from the three-class endpoint. Wearable acquisitions
are excluded from the primary phone lane and may be reported only as a separate
sensor-location stress test.

### Unit qualification blocker

The archive README calls all acceleration m/s^2, but the inspected files are not
yet safe to pool under that assumption. For example,
`S93/Standing_6/AppleiPhoneSE2_Leg/Accelerometer.txt` has a sub-1 numerical
scale consistent with g rather than m/s^2, while
`S65/Walking_9/SamsungGalaxyNexus-6_Hand/Accelerometer.txt` has approximately
9.8 m/s^2 scale. The external adapter must therefore complete and record a
device/acquisition unit audit before any label score is calculated. Ambiguous
units must be quarantined; they must not be guessed from performance.

The official parser-suite repository was inspected at commit
`565e36017d952941178b1a6f11031370ac65fd0b`. It documents the standardized
m/s^2 and rad/s contract but does not remove the need to verify the values in
the released AICOS archive.

## Exact-interface exploratory dataset audit

The Self Phone Three Positions repository was inspected at commit
`348768b8034ccd7d01d696cd62cc1d673a34a186`. The release contains 646 trial
CSVs rather than the README's stated 696. Across 449,899 rows, the required
columns were present, timestamps were monotonic, labels matched their folders,
and no nonfinite row or exact duplicate file was found. File-median timing was
20 ms; the gravity-norm median was 9.8067 m/s^2. Twenty participants have all
six core activities in the `pants_pocket` folder and ten have all core
activities in each of hand, pants pocket and upper pocket.

This is a close engineering match but currently has no accompanying paper or
DOI, does not document the device or measurement units sufficiently, and its
README file count disagrees with the release. Its CC BY-NC-ND 4.0 licence also
requires a project-specific review before any transformed data are shared.
Use it only as an exploratory frozen-model replication unless the provider
clarifies those points.

## Frozen execution contract

1. Fit the already selected RMRP, CTGR and strict HERA source models once on the
   permitted InclusiveHAR source population. Seal checkpoints, feature schema,
   class order and hashes before reading any AICOS labels or scores.
2. Download and preserve the compressed AICOS archive, verify the provider MD5,
   compute SHA-256, and process entries directly from ZIP. Do not expand the
   7.341 GB archive beside the 2.54 GB raw file.
3. Use provider folds 1--5 only for signal qualification: unit manifests,
   timestamp/rate checks, missing-sensor eligibility and deterministic adapter
   contract tests. Do not use their class scores to select a model, threshold,
   feature or preprocessing option.
4. Restrict the primary lane to phone acquisitions with accelerometer and
   gyroscope. Use the existing causal 0.30 Hz gravity decomposition, resample
   jointly to 50 Hz, and create 128-sample windows inside acquisition boundaries.
   Name every method `-DG`; do not relabel it as native HERA/CTGR.
5. Map `Walking`, `Running`, `Upstairs`, `Downstairs`, `RampUp` and `RampDown`
   to mobility; retain `Sitting` and `Standing`; exclude lifts and lying.
6. Open the provider `test` fold once after all adapter checks pass. Report the
   eight complete-three-class phone participants as the primary participant
   macro-F1 set. Also report class recall and coverage over all eligible test
   participants, with missing-class status explicit.
7. Report matched RMRP-DG, CTGR-DG and strict HERA-DG; accuracy, participant
   macro-F1, bottom-30%, worst participant, sitting and standing recall, paired
   participant bootstrap intervals, acquisition/window coverage, quarantine
   counts, and device/position strata. Do not tune from these outputs.
8. Run MotionSense native-nine only as a separate diagnostic using its recorded
   gravity, rotation rate and user acceleration. Its result cannot become an
   independent confirmation because the domain has already been opened.
9. After the provider-test report is sealed, the 46 complete-three-class phone
   participants across all AICOS folds may be reported as a larger descriptive
   robustness analysis. It cannot replace the primary result because folds
   1--5 were used to qualify the adapter.

## Claim boundary

A positive AICOS result would substantiate cross-device and cross-position
generalization of the **derived-gravity** architecture. A positive MotionSense
result would substantiate native-interface portability on an already consumed
domain. Neither establishes generalization across physical abilities. The
decisive confirmation remains a fresh, sealed, ability-relevant cohort with all
nine provider-native channels.

## Storage disposition

At qualification time the C: drive had 8.70 GB free. This is sufficient for the
2.54 GB verified compressed archive plus compact caches and results, but not for
keeping both the compressed archive and its approximately 7.341 GB expansion.
Streaming ZIP processing is therefore mandatory.
