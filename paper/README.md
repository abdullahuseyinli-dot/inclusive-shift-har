# Paper workspace

The paper-ready structure and current v1 evidence narrative are in [`OUTLINE.md`](OUTLINE.md). The post-analysis source-development and external-evaluation extension is in [`FUSE_REFRAME_V2_ADDENDUM.md`](FUSE_REFRAME_V2_ADDENDUM.md). Machine-readable tables remain under `results/`; the paper workspace contains no copied raw data or large model artifacts.

Working title:

> InclusiveShift-HAR: An Auditable Participant-Exclusive Benchmark for Ability-Associated Population Shift in Smartphone Activity Recognition

The benchmark is the primary contribution. MoRe-HAR is retained as an unsupported secondary hypothesis. The released cohort label is not a direct measure of physical ability, and missing trial/timestamp identifiers prevent trial-boundary reconstruction. Completed few-person, stress, within-group, efficiency, signal-sensitivity, and qualified CCIL/BPD evidence is post-confirmatory. The paper should therefore center the participant-exclusive protocol, the ten-participant uncertainty, the effective DANN/CORAL tie, and the negative MoRe-HAR result.

FuSE/ReFrame v2 does not replace that conclusion. Its RMRP method improves strict source-development mean macro-F1 from 0.8292 to 0.8379 and worst-participant macro-F1 from 0.4985 to 0.5494, but misses the recorded improvement gate and lacks independent ability-cohort validation. Sparse labelled personalization and DAGHAR results are reported under their separate evidence classes.
