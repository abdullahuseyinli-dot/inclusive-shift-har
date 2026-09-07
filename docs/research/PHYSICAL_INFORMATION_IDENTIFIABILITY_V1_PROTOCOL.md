# Physical information identifiability pilot v1

Status: acquisition plan frozen; device qualification, ethics/consent, recruitment,
recording, and outcomes pending. The normative schedule and decision contract is
'configs/experiments/physical_information_identifiability_v1.yaml'.

This development characterization study asks two questions that the retained datasets
cannot answer: whether sitting/standing reference information survives separate physical
bouts and attachment/visit change, and whether non-locomotor body motion can be separated
from usual, slow, and turning mobility with the eventual qualified sensor interface. It
does not continue or reopen C3, establish population performance, or train a model.

## Fixed acquisition design

Plan six new pseudonymous wearer slots. Each wearer attends two visits, with two full
device removal/reattachment blocks per visit. This gives 24 attachment blocks. Each block
contains four 30-second support bouts first: quiet sitting, sitting with prescribed
upper-body motion, quiet standing, and standing with prescribed upper-body motion.
Seven separate query bouts follow: the same four posture conditions and 45-second usual,
slow, and turning mobility. The generated schedule rotates each order by the global block
index plus wearer index. This varies the order across wearers within every corresponding
visit/attachment cell while preserving support-before-query chronology. It provides order
variation for this bounded sample; it is not a complete carryover-balanced crossover.

There are exactly 11 bouts per block, 264 bouts overall, and 9,000 seconds (150 minutes)
of nominal recording. Transitions are retained but excluded from the fixed interior:
remove five seconds from each end of every adjudicated bout. Never search for a quieter
interval. Prompts describe intended behavior; they do not prove the observed activity.
An annotation record must separately identify the adjudicated observation and provenance.

Support and query are different physical bouts with nonoverlapping samples. A failed,
partial, withdrawn, or unresolved slot remains in the record. Do not replace a wearer or
bout because observed signals or eventual outcomes are inconvenient.

## Device and signal gate

Before collection, bind the actual device, model, firmware, logging software, placement,
axis convention, channel map, units, native gravity availability, nominal clock, measured
sampling intervals, and a bench-qualification record. Hash the placement and bench files.
Record qualification and processing-freeze UTC times. Freeze the device-specific
sampling/filter contract after bench qualification and before the first human outcome.
Until then, collection readiness is 'pending', not pass.

If the processing contract selects provider-native gravity, the receipt must state that
native gravity is available and map three distinct `gravity_x`, `gravity_y`, and
`gravity_z` channels. A derived-gravity lane cannot be relabeled as provider native.

Preserve complete raw recordings and transitions. Process each continuous observable
attachment recording before projecting bout roles or annotations. Filter/reset boundaries
are attachment changes, recording breaks, timestamp discontinuities, and nonfinite sensor
runs. Activity, support/query role, bout boundaries, and scoring eligibility do not reset
gravity, resampling, or the candidate grid. Keep native and derived gravity distinctly
named. Unknown device fields receive no default.

## Matched reference comparisons

For all 24 blocks, fresh support comes from that same block. For block 2 of each visit,
compare fresh block-2 support with earlier block-1 support on the identical block-2 query
IDs and validity mask. For both blocks in visit 2, compare fresh visit-2 support with
visit-1 support from the same attachment number on identical visit-2 queries. No
previous-visit comparison exists in visit 1, and no support may come from a future bout,
attachment, or visit.

These matched contrasts characterize reference degradation under the observed combined
attachment/time conditions. They do not by themselves isolate attachment change from
elapsed time.

## Measurements and aggregation

Use a left-aligned window grid over each continuous recording, with length and stride from
the qualified equipment contract. Never reset it at a bout boundary. Admit only full
windows inside the fixed five-second-trimmed adjudicated interval. A window is invalid if
an inertial or gravity sample is nonfinite, a timestamp gap exceeds the frozen maximum, or
the mean gravity norm is at most `1e-12`. A bout needs at least three valid windows and at
least 80 percent of all grid windows in its fixed interior. Do not impute or search for a
quieter interval.

Normalize the arithmetic mean gravity vector within each valid window. Within each motion
stratum, form sitting and standing prototypes by normalizing the arithmetic mean of the
valid support-window directions. Angular distance is
`acos(clip(dot(u, v), -1, 1))` radians. Report median and 90th-percentile support-window
angles to their prototype, the sitting/standing prototype angle, and
`1/abs(sin(angle))`; report the latter as infinity when its denominator is at most
`1e-12`.

For every posture query window, signed margin is the angle to the wrong-posture prototype
minus the angle to the correct-posture prototype. Reduce windows to the bout median.
Fresh-minus-stale is the fresh bout-median margin minus the stale bout-median margin on
the intersection of identical valid query-window IDs. Positive margin favors the correct
posture; positive fresh-minus-stale means the fresh reference performs better.

For descriptive motion overlap, compute per-window dynamic-acceleration RMS as
`sqrt(mean(sum((acceleration - gravity)^2)))` and gyroscope RMS as
`sqrt(mean(sum(gyroscope^2)))`. Reduce each to bout medians. Within each wearer and feature,
report the common-language probability that a mobility bout median exceeds an
upper-body-motion posture bout median. This description has no architecture gate.

Aggregate windows to bouts, then condition/reference scope, then wearer, and finally give
each wearer equal weight. A complete wearer has all 16 fresh posture-query bouts metadata
and signal eligible. Repeated bouts and visits do not increase independent N. Report all
wearer values, the equal-wearer mean, sample standard deviation, and a two-sided 95 percent
Student-t interval with complete-wearer count minus one degrees of freedom; suppress the
interval below two complete wearers. Report every condition, denominator, and failure.
There are no confirmatory p-values.

Prototype feasibility requires all six complete wearers to have positive wearer-mean
signed margin for both quiet and upper-body-motion posture queries. This permits drafting
a separate architecture protocol only. Stable fresh but failed stale comparisons establish
an explicit recalibration burden. Fresh instability closes the local-reference prerequisite.

Full-frame eligibility is separate. Antiparallel posture anchors may distinguish two
postures while failing to define a stable full frame. A numerical anchor angle alone is
insufficient; the conditioning limit must be justified from the qualified device's bench
uncertainty and frozen before human outcomes.

Motion-energy overlap rejects only a simple energy explanation. It does not show that raw
IMU sequences lack useful temporal information and does not automatically authorize a
boundary model. Six wearers cannot establish Parkinson, inclusive-population, or general
population superiority.

## Evidence and stopping rule

The plan generator must create exactly six wearer slots, 12 visits, 24 attachment blocks,
264 unique bout IDs, and immutable fresh/attachment-stale/visit-stale mappings. Plan
validation is separate from collection readiness and recorded-evidence validity.

In imported recording records, `visit_started_at_utc` is the UTC anchor and `started_at`
and `ended_at` are finite seconds in the declared common device clock relative to that
anchor. The clock may not silently reset between attachment blocks. If a logger resets,
the importer must preserve raw timestamps and provide a separately reviewed mapping into
this common visit clock before stale-reference chronology can pass.

An equipment receipt can establish only device-evidence integrity. It cannot establish
human-study authorization, a consent process, or participant access. Those prerequisites
remain explicit even when equipment metadata and files validate.

Raw-file integrity plus matching adjudicated labels establishes metadata eligibility only.
Signal eligibility remains pending until the frozen continuous-record processing runs and
each fixed trimmed interior passes its predeclared sample and discontinuity rules.

No collection occurs through this repository until lawful human-study authorization,
consent, participants, operator, device, placement, and bench qualification exist. No
synthetic fixture substitutes for evidence. After plan validation, stop at
'plan_valid_collection_pending'; do not train the conditional architecture automatically.
