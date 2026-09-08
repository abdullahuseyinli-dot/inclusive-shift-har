# Portable retained-evidence replay v1

This zero-fit evidence check reproduces two separate development lanes: seed-11
InclusiveHAR source CTGR and corrected-FoG L9v. It does not compare their scores,
restore confirmation, train models, run checkpoint inference, or access raw data.
No raw dataset or fitted model is copied into the output.

Run from an installed source checkout:

```text
python -m inclusive_shift_har.experiments.portable_evidence_replay --evidence-root <evidence-root> --output <new-directory> --config configs/experiments/portable_evidence_replay_v1.yaml
```

The explicit evidence root can be relocated. Seven directly pinned prediction/report
files bind 19 further provenance artifacts, for 26 input files in total. CTGR's
five aggregate fold references bind each fold receipt by file hash and self-hash;
each receipt binds its `models.pkl` by byte hash. CTGR also binds the tracked
source-window and dataset manifests by byte hash. Scored window IDs and participant
IDs must match the source-window manifest. L9v's completion manifest binds all five
fitted checkpoint files, its source-code manifest and input manifest, in addition
to the three consumed prediction/report files. Duplicate and escaping paths fail.
All model bundles are hashed as opaque bytes, never deserialized or executed.
Each CTGR scored label must also match the source window's functional-core canonical
label, and every CTGR fold receipt must declare the aggregate result's source commit.

The original files remain unchanged. JSON self-hashes, class order, unique window
IDs, participant rosters, probability schemas/simplexes, scored/observable
alignment, and saved decisions must validate. Historical raw CSV hash/size
declarations are retained explicitly as indirect manifest provenance; raw bytes
are not rehashed or loaded. The recorded L9v source-code file hashes are preserved
as declarations from its sealed source manifest; current checkout files are not
silently substituted for the historical implementation. Validation does not claim
that every other historical manifest entry was checked.

Independent NumPy/confusion arithmetic computes every participant's fixed
three-class macro-F1, support, recall, NLL and Brier. Missing-class F1/recall is
zero. Participants receive equal weight; bottom 30 percent uses the ceiling of
0.3 times participant count. Pooled diagnostics are reported separately. NLL uses
a 1e-12 probability floor. Historical CTGR probability scores were pooled: the
additional equal-participant NLL/Brier is explicitly descriptive reaggregation.
Historical reports are compared at absolute tolerance 1e-12.

L9v fallback must resolve through numeric method index three to `b0` and copy its
probability bytes exactly. The preserved one-character string `b` is accepted
only together with its separately pinned erratum; provenance completeness stays
false and advancement stays failed. The erratum is attached to the replay record.

Each lane writes its own self-hashed replay or explicit failure. Independent lanes
continue if one fails. Output is create-only and includes a config snapshot,
input manifest, validation record, completion manifest and shutdown receipt.
Failure returns nonzero and retains all partial evidence. No worker or monitor
is launched. This protocol supplies correctness/reproducibility evidence only;
all historical adaptivity and scientific gates remain unchanged.
KeyboardInterrupt and SystemExit retain lifecycle/failure receipts in a finally
block before being re-raised; a completed independent lane remains preserved.
Any interrupted atomic-write partial is retained and byte-hashed as a partial file,
without falsely requiring its unfinished content to be a valid sealed JSON record.
