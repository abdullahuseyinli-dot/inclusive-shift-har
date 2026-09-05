# Session-only FoG amendment v3

Declared 2026-09-05 before replacement experiments. This addendum preserves the
earlier v2 diagnosis and protocol rather than silently rewriting them. The final
replacement protocol is `configs/protocols/external_har_session_grid_v3.yaml`.

The [provider descriptor](https://www.nature.com/articles/s41597-026-06645-1)
describes task segments as annotations; the released task code is not independent
proof of an acquisition-start event. The v2 draft retained subject/session/task
grouping from the old implementation. That is insufficient for the intended
annotation-independent contract. No v2 replacement performance experiments were
launched. The final loader ignores task annotations entirely, preserving provider
row order within participant/session and resetting only for finite-signal runs,
nonpositive timestamp increments, or timestamp gaps. The regression suite alters
and removes task annotations as well as activity labels.

This does not prove that the provider's released nominal timebase preserves every
real-world acquisition gap. The CSV is reconstructed at 60 Hz; any unrecorded
upstream cropping cannot be repaired from unavailable raw acquisition timestamps.
The claim is therefore label-independent processing of the released session
stream, not verified zero-latency deployment or annotation-independent original
data collection. Fixed global windows, homogeneous post hoc admission, and all
model/statistical choices from v2 remain unchanged. Historical scores are
superseded and non-comparable, not numerical before/after improvements.
